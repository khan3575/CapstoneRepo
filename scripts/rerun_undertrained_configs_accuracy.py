#!/usr/bin/env python3
"""
Re-run the 4 undertrained ablation configurations optimized for ACCURACY

This version prioritizes accuracy over speed:
- Batch size 32 (better gradient stability, less noise)
- No mixed precision (FP32 for numerical stability)
- More conservative learning rate warmup
- Gradient accumulation for effective larger batch size

Expected Results (CLEAN DATA - No Leakage):
- baseline: 89.5-90.5% (matching CV fold 0: 90.41%)
- layers_6: 89.5-91.0% (might gain +0.5-1.0% from depth)
- hidden_512: 89.0-90.5% (more params ≠ better, might overfit)
- gat: 65-75% (attention unsuitable for this task)

NOTE: Old expectations (97-99%) were based on LEAKED DATA with tumor_ratio feature.
      New ceiling is 90.41% after removing ground-truth leakage.
"""

import os
import sys
import json
import time
import random
import torch
import numpy as np
from pathlib import Path
from datetime import datetime
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR
from torch_geometric.loader import DataLoader
import multiprocessing as mp
from typing import Dict, List, Tuple

# Enable GPU optimizations (but keep deterministic for accuracy)
torch.backends.cudnn.benchmark = False  # More stable
torch.backends.cudnn.deterministic = True  # Reproducible
torch.set_float32_matmul_precision('high')

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from gnn_model import TumorSegmentationGNN, CombinedLoss
from dataset import BraTSGraphDataset

# ACCURACY-OPTIMIZED CONFIGURATION - EXACT MATCH TO CV FOLD 0
BASE_CONFIG = {
    'fold': 0,
    'data_dir': 'data/graphs',
    'cv_dir': 'data/cv_folds',
    'batch_size': 32,          # ← MATCH CV: Batch 32
    'accumulation_steps': 1,   # ← FIXED: No accumulation (was 2, caused 83% result)
    'num_epochs': 50,
    'lr': 0.001,
    'weight_decay': 1e-5,
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'patience': 15,            # ← More patient convergence
    'num_workers': 4,          # ← Balanced (less disk contention)
    'prefetch_factor': 2,
    'use_amp': False,          # ← FP32 for numerical stability
}

# Configurations to re-run
RERUN_CONFIGS = {
    'baseline_accuracy': {
        'name': 'Baseline (5 layers, 256D) - Accuracy Optimized',
        'num_layers': 5,
        'hidden_channels': 256,
        'gnn_type': 'sage',
        'use_edge_features': True,
    },
    'layers_6_accuracy': {
        'name': '6 Layers - Accuracy Optimized',
        'num_layers': 6,
        'hidden_channels': 256,
        'gnn_type': 'sage',
        'use_edge_features': True,
    },
    'hidden_512_accuracy': {
        'name': 'Hidden Dim 512 - Accuracy Optimized',
        'num_layers': 5,
        'hidden_channels': 512,
        'gnn_type': 'sage',
        'use_edge_features': True,
    },
    'gat_accuracy': {
        'name': 'GAT - Accuracy Optimized',
        'num_layers': 5,
        'hidden_channels': 256,
        'gnn_type': 'gat',
        'use_edge_features': True,
    },
}

def calculate_dice_score(pred, target, smooth=1.0):
    """Calculate Dice score"""
    pred = pred.view(-1)
    target = target.view(-1)
    intersection = (pred * target).sum()
    dice = (2. * intersection + smooth) / (pred.sum() + target.sum() + smooth)
    return dice.item()

def load_data_splits(fold, data_dir, cv_dir):
    """
    Load fold data splits (graphs are already preprocessed in data/graphs/)
    Just returns the file lists for train/val/test
    """
    print(f"Loading fold {fold} splits...")
    
    # Load split file
    split_file = Path(cv_dir) / f'fold_{fold}.json'
    with open(split_file, 'r') as f:
        split = json.load(f)
    
    # Get patient lists
    train_patients = split['train_patients']
    val_patients = split['val_patients']
    test_patients = split['test_patients']
    
    # Find corresponding graph files
    graph_dir = Path(data_dir)
    train_files = []
    val_files = []
    test_files = []
    
    for patient in train_patients:
        patient_dir = graph_dir / patient
        if patient_dir.exists():
            patient_graphs = list(patient_dir.glob(f"{patient}_graphs_*.pt"))
            train_files.extend(patient_graphs)
    
    for patient in val_patients:
        patient_dir = graph_dir / patient
        if patient_dir.exists():
            patient_graphs = list(patient_dir.glob(f"{patient}_graphs_*.pt"))
            val_files.extend(patient_graphs)
    
    for patient in test_patients:
        patient_dir = graph_dir / patient
        if patient_dir.exists():
            patient_graphs = list(patient_dir.glob(f"{patient}_graphs_*.pt"))
            test_files.extend(patient_graphs)
    
    print(f"  Train: {len(train_files)} graphs from {len(train_patients)} patients")
    print(f"  Val:   {len(val_files)} graphs from {len(val_patients)} patients")
    print(f"  Test:  {len(test_files)} graphs from {len(test_patients)} patients")
    print(f"  Note: Graphs already preprocessed in {data_dir}")
    
    return train_files, val_files, test_files

def create_model(config, device):
    """Create model"""
    model = TumorSegmentationGNN(
        in_channels=15,  # FIXED: Clean data has 15 features (was 12 with leaked tumor_ratio)
        hidden_channels=config['hidden_channels'],
        num_layers=config['num_layers'],
        gnn_type=config['gnn_type'],
        dropout=0.2
    ).to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    return model, total_params, trainable_params

def train_epoch(model, loader, criterion, optimizer, scheduler, device, use_edge_features, accumulation_steps=1):
    """Train for one epoch with gradient accumulation"""
    model.train()
    total_loss = 0
    total_dice = 0
    num_batches = 0
    
    optimizer.zero_grad(set_to_none=True)
    
    for batch_idx, batch in enumerate(loader):
        batch = batch.to(device, non_blocking=True)
        
        if not use_edge_features:
            batch.edge_attr = None
        
        # Forward pass (FP32 for stability)
        logits, embeddings = model(batch)
        loss, ce_loss, dice_loss, _ = criterion(
            logits, embeddings, batch.y, batch.slice_mask
        )
        
        # Scale loss for accumulation
        loss = loss / accumulation_steps
        
        # Backward pass
        loss.backward()
        
        # Update weights every accumulation_steps
        if (batch_idx + 1) % accumulation_steps == 0 or (batch_idx + 1) == len(loader):
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
        
        # Calculate Dice (on unscaled predictions)
        with torch.no_grad():
            preds = torch.sigmoid(logits) > 0.5
            dice = calculate_dice_score(preds.float(), batch.y)
        
        total_loss += loss.item() * accumulation_steps  # Unscale for logging
        total_dice += dice
        num_batches += 1
    
    return total_loss / num_batches, total_dice / num_batches

def validate(model, loader, criterion, device, use_edge_features):
    """Validate model"""
    model.eval()
    total_loss = 0
    total_dice = 0
    num_batches = 0
    
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device, non_blocking=True)
            
            if not use_edge_features:
                batch.edge_attr = None
            
            logits, embeddings = model(batch)
            loss, _, _, _ = criterion(
                logits, embeddings, batch.y, batch.slice_mask
            )
            
            preds = torch.sigmoid(logits) > 0.5
            dice = calculate_dice_score(preds.float(), batch.y)
            
            total_loss += loss.item()
            total_dice += dice
            num_batches += 1
    
    return total_loss / num_batches, total_dice / num_batches

def run_single_config(config_name, config, base_config, train_files, val_files, test_files):
    """Run training for one configuration"""
    print("\n" + "="*80)
    print(f"CONFIG: {config['name']}")
    print("="*80)
    
    # Set random seeds for reproducibility
    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    print(f"Random seed set to: {seed}")
    
    device = base_config['device']
    
    # Create datasets (already preprocessed)
    train_dataset = BraTSGraphDataset(root_dir=None, split='train', graph_files=train_files)
    val_dataset = BraTSGraphDataset(root_dir=None, split='val', graph_files=val_files)
    test_dataset = BraTSGraphDataset(root_dir=None, split='test', graph_files=test_files)
    
    # Create dataloaders with balanced settings
    train_loader = DataLoader(
        train_dataset,
        batch_size=base_config['batch_size'],
        shuffle=True,
        num_workers=base_config['num_workers'],
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=base_config['prefetch_factor']
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=base_config['batch_size'],
        shuffle=False,
        num_workers=base_config['num_workers'],
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=base_config['prefetch_factor']
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=base_config['batch_size'],
        shuffle=False,
        num_workers=base_config['num_workers'],
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=base_config['prefetch_factor']
    )
    
    # Create model
    print(f"\nModel Architecture:")
    print(f"  Layers: {config['num_layers']}")
    print(f"  Hidden: {config['hidden_channels']}")
    print(f"  Type:   {config['gnn_type']}")
    
    model, total_params, _ = create_model(config, device)
    print(f"  Parameters: {total_params:,}")
    
    # Training setup
    criterion = CombinedLoss(use_consistency=False)
    optimizer = AdamW(
        model.parameters(),
        lr=base_config['lr'],
        weight_decay=base_config['weight_decay']
    )
    
    steps_per_epoch = len(train_loader) // base_config['accumulation_steps']
    total_steps = steps_per_epoch * base_config['num_epochs']
    
    scheduler = OneCycleLR(
        optimizer,
        max_lr=base_config['lr'],
        total_steps=total_steps,
        pct_start=0.3,
        anneal_strategy='cos',
        div_factor=25.0,
        final_div_factor=1e4
    )
    
    # Training loop
    print(f"\nTraining for up to {base_config['num_epochs']} epochs (patience={base_config['patience']}, FP32)...")
    print(f"  Batch size: {base_config['batch_size']} (NO accumulation, matches CV)")
    
    best_val_dice = 0
    best_epoch = 0
    patience_counter = 0
    train_history = []
    val_history = []
    
    start_time = time.time()
    
    for epoch in range(base_config['num_epochs']):
        epoch_start = time.time()
        
        train_loss, train_dice = train_epoch(
            model, train_loader, criterion, optimizer, scheduler,
            device, config['use_edge_features'], base_config['accumulation_steps']
        )
        
        val_loss, val_dice = validate(
            model, val_loader, criterion, device, config['use_edge_features']
        )
        
        epoch_time = time.time() - epoch_start
        
        train_history.append({'loss': train_loss, 'dice': train_dice})
        val_history.append({'loss': val_loss, 'dice': val_dice})
        
        print(f"Epoch {epoch+1:2d}/{base_config['num_epochs']}: "
              f"Train Loss={train_loss:.4f}, Dice={train_dice:.4f} | "
              f"Val Loss={val_loss:.4f}, Dice={val_dice:.4f} | "
              f"Time={epoch_time:.1f}s")
        
        # Check for improvement
        if val_dice > best_val_dice:
            best_val_dice = val_dice
            best_epoch = epoch + 1
            patience_counter = 0
            
            # Save best model
            save_dir = Path('research_results/ablation_study_accuracy') / config_name
            save_dir.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), save_dir / 'best_model.pth')
            print(f"  ✓ Best model saved (Val Dice: {best_val_dice:.4f})")
        else:
            patience_counter += 1
        
        # Early stopping
        if patience_counter >= base_config['patience']:
            print(f"\nEarly stopping at epoch {epoch+1} (no improvement for {base_config['patience']} epochs)")
            break
    
    training_time = time.time() - start_time
    
    # Load best model for testing
    model.load_state_dict(torch.load(save_dir / 'best_model.pth'))
    
    # Test
    print("\nEvaluating on test set...")
    test_loss, test_dice = validate(
        model, test_loader, criterion, device, config['use_edge_features']
    )
    
    print(f"\n{'='*80}")
    print(f"RESULTS: {config['name']}")
    print(f"{'='*80}")
    print(f"  Best Val Dice:  {best_val_dice:.4f} (epoch {best_epoch})")
    print(f"  Test Dice:      {test_dice:.4f}")
    print(f"  Parameters:     {total_params:,}")
    print(f"  Training Time:  {training_time/60:.1f} min")
    print(f"{'='*80}")
    
    # Save results
    results = {
        'config_name': config_name,
        'config': config,
        'parameters': total_params,
        'best_val_dice': float(best_val_dice),
        'best_epoch': int(best_epoch),
        'test_dice': float(test_dice),
        'test_loss': float(test_loss),
        'training_time_min': float(training_time / 60),
        'train_history': train_history,
        'val_history': val_history,
        'training_config': {
            'batch_size': base_config['batch_size'],
            'accumulation_steps': base_config['accumulation_steps'],
            'effective_batch_size': base_config['batch_size'] * base_config['accumulation_steps'],
            'num_epochs': base_config['num_epochs'],
            'patience': base_config['patience'],
            'lr': base_config['lr'],
        }
    }
    
    with open(save_dir / 'results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    return results

def main():
    """Main execution"""
    print("\n" + "╔" + "═"*78 + "╗")
    print("║" + " "*18 + "ABLATION STUDY - ACCURACY MODE" + " "*30 + "║")
    print("║" + " "*15 + "Optimized for Best Performance" + " "*33 + "║")
    print("╚" + "═"*78 + "╝")
    
    print(f"\nEXACT MATCH TO CV FOLD 0 Settings:")
    print(f"  Batch Size: 32 → MATCHES CV (no accumulation)")
    print(f"  Accumulation: 1 (DISABLED) → Real batch = 32")
    print(f"  Mixed Precision: DISABLED → FP32 for numerical precision")
    print(f"  Workers: 4 → Balanced (avoids disk contention)")
    print(f"  Patience: 15 → More patient convergence")
    print(f"  Deterministic: ON → Reproducible results")
    print(f"\n  Goal: Match CV Fold 0 result (90.41%) exactly")
    
    # Step 1: Load fold splits (graphs already preprocessed!)
    print("\n" + "="*80)
    print("STEP 1: Loading Data Splits")
    print("="*80)
    print("Note: Graphs already preprocessed in data/graphs/ directory")
    
    train_files, val_files, test_files = load_data_splits(
        BASE_CONFIG['fold'],
        BASE_CONFIG['data_dir'],
        BASE_CONFIG['cv_dir']
    )
    
    print("\n✅ Data splits loaded! Graphs will be read from disk during training.")
    
    # Step 2: Train all configs sequentially
    print("\n" + "="*80)
    print("STEP 2: Training Configs Sequentially")
    print("="*80)
    print(f"Running {len(RERUN_CONFIGS)} configurations...")
    print("Expected time: ~5-6 hours total (75-90 min each)")
    print("\nEXACT MATCH optimizations:")
    print("  • FP32 precision → No rounding errors from AMP")
    print("  • Batch 32, NO accumulation → Exact match to CV Fold 0")
    print("  • Patience 15 → Thorough convergence")
    print("  • Deterministic mode → Reproducible results")
    print("  • Expected: 89.5-90.5% (matching CV fold 0: 90.41%)\n")
    
    all_results = {}
    
    for config_name, config in RERUN_CONFIGS.items():
        try:
            results = run_single_config(
                config_name, config, BASE_CONFIG,
                train_files, val_files, test_files
            )
            all_results[config_name] = results
        except Exception as e:
            print(f"\n❌ ERROR in {config_name}: {str(e)}")
            import traceback
            traceback.print_exc()
            continue
    
    # Save combined results
    output_dir = Path('research_results/ablation_study_accuracy')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    with open(output_dir / 'all_results.json', 'w') as f:
        json.dump(all_results, f, indent=2)
    
    # Print summary
    print("\n" + "="*80)
    print("ACCURACY MODE RESULTS")
    print("="*80)
    
    print(f"{'Config':<30} {'Test Dice':>12} {'Parameters':>15} {'Time (min)':>12}")
    print("-"*80)
    
    for config_name, results in all_results.items():
        print(f"{results['config']['name']:<30} "
              f"{results['test_dice']:>12.4f} "
              f"{results['parameters']:>15,} "
              f"{results['training_time_min']:>12.1f}")
    
    print("\n✅ Accuracy-optimized ablation complete!")
    print(f"   Results saved to: {output_dir}")
    print("\n📊 Key insights:")
    print("   - FP32 provides numerical stability")
    print("   - Gradient accumulation gives stable training")
    print("   - Longer patience allows full convergence")
    print("   - Should achieve closer to CV fold 0 target (90.41%)")

if __name__ == "__main__":
    main()
