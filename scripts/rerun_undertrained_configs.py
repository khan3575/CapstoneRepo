#!/usr/bin/env python3
"""
Re-run the 4 undertrained ablation configurations with proper settings

Configurations to re-run:
1. baseline (5 layers, 256 hidden) - stopped at epoch 9, got 90.91%
2. layers_6 (6 layers) - stopped at epoch 12, got 92.89%
3. hidden_512 (512 hidden) - stopped at epoch 9, got 91.93%
4. gat (GAT instead of SAGE) - got 92.01% but might improve

Strategy:
- Use batch_size=32 (match CV, better GPU utilization)
- Use patience=10 (less aggressive early stopping)
- Use num_epochs=50 (match CV)
- Preprocess graphs once, share across all runs
- Run configs in parallel (4 GPUs or sequential with saved data)

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

# Enable GPU optimizations
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = False
torch.set_float32_matmul_precision('high')

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from gnn_model import TumorSegmentationGNN, CombinedLoss
from dataset import BraTSGraphDataset

# IMPROVED CONFIGURATION - Match CV settings
BASE_CONFIG = {
    'fold': 0,
    'data_dir': 'data/graphs',
    'cv_dir': 'data/cv_folds',
    'batch_size': 32,      # ← Changed from 96 to 32 (better GPU util)
    'num_epochs': 50,      # ← Changed from 25 to 50 (match CV)
    'lr': 0.001,
    'weight_decay': 1e-5,
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'patience': 10,        # ← Changed from 5 to 10 (less aggressive)
    'num_workers': 4,      # Match CV
    'prefetch_factor': 2,  # Standard
}

# Configurations to re-run
RERUN_CONFIGS = {
    'baseline_fixed': {
        'name': 'Baseline (5 layers, 256D) - Fixed Training',
        'num_layers': 5,
        'hidden_channels': 256,
        'gnn_type': 'sage',
        'use_edge_features': True,
    },
    'layers_6_fixed': {
        'name': '6 Layers - Fixed Training',
        'num_layers': 6,
        'hidden_channels': 256,
        'gnn_type': 'sage',
        'use_edge_features': True,
    },
    'hidden_512_fixed': {
        'name': 'Hidden Dim 512 - Fixed Training',
        'num_layers': 5,
        'hidden_channels': 512,
        'gnn_type': 'sage',
        'use_edge_features': True,
    },
    'gat_fixed': {
        'name': 'GAT - Fixed Training',
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
        in_channels=12,
        hidden_channels=config['hidden_channels'],
        num_layers=config['num_layers'],
        gnn_type=config['gnn_type'],
        dropout=0.2
    ).to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    return model, total_params, trainable_params

def train_epoch(model, loader, criterion, optimizer, scheduler, device, use_edge_features):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    total_dice = 0
    num_batches = 0
    
    for batch in loader:
        batch = batch.to(device, non_blocking=True)
        
        if not use_edge_features:
            batch.edge_attr = None
        
        # Forward pass (FP32 for stability)
        logits, embeddings = model(batch)
        loss, ce_loss, dice_loss, _ = criterion(
            logits, embeddings, batch.y, batch.slice_mask
        )
        
        # Backward pass
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()
        
        # Calculate Dice
        with torch.no_grad():
            preds = torch.sigmoid(logits) > 0.5
            dice = calculate_dice_score(preds.float(), batch.y)
        
        total_loss += loss.item()
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
    
    device = base_config['device']
    
    # Create datasets (already preprocessed)
    train_dataset = BraTSGraphDataset(root_dir=None, split='train', graph_files=train_files)
    val_dataset = BraTSGraphDataset(root_dir=None, split='val', graph_files=val_files)
    test_dataset = BraTSGraphDataset(root_dir=None, split='test', graph_files=test_files)
    
    # Create dataloaders with optimized settings
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
    
    steps_per_epoch = len(train_loader)
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
    print(f"\nTraining for up to {base_config['num_epochs']} epochs (patience={base_config['patience']})...")
    
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
            device, config['use_edge_features']
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
            save_dir = Path('research_results/ablation_study_clean') / config_name
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
    print("║" + " "*20 + "RE-RUN UNDERTRAINED CONFIGS" + " "*31 + "║")
    print("║" + " "*15 + "With Proper Training Settings" + " "*34 + "║")
    print("╚" + "═"*78 + "╝")
    
    print(f"\nImproved Settings (vs original ablation):")
    print(f"  Batch Size: 32 (was 96) → Better GPU utilization")
    print(f"  Patience: 10 (was 5) → Less aggressive early stopping")
    print(f"  Max Epochs: 50 (was 25) → Matches main CV training")
    print(f"  Workers: 4 (was 8) → Matches CV, reduces disk contention")
    
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
    print("Expected time: ~4-6 hours total (1-1.5 hours each)")
    print("\nKey improvement: batch_size=32 reduces disk I/O bottleneck")
    print("GPU should reach 95-100% utilization (vs 40% with batch_size=96)\n")
    
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
    output_dir = Path('research_results/ablation_study_clean')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    with open(output_dir / 'all_results.json', 'w') as f:
        json.dump(all_results, f, indent=2)
    
    # Print comparison with original
    print("\n" + "="*80)
    print("COMPARISON: Original vs Fixed Training")
    print("="*80)
    
    comparisons = {
        'baseline_fixed': ('baseline', 'Baseline (5 layers, 256)'),
        'layers_6_fixed': ('layers_6', '6 Layers'),
        'hidden_512_fixed': ('hidden_512', 'Hidden 512'),
        'gat_fixed': ('gat', 'GAT'),
    }
    
    print(f"{'Config':<25} {'Original':>12} {'Fixed':>12} {'Improvement':>15}")
    print("-"*80)
    
    for fixed_name, (orig_name, label) in comparisons.items():
        if fixed_name in all_results:
            fixed_dice = all_results[fixed_name]['test_dice']
            
            # Load original result
            orig_file = Path(f'research_results/ablation_study/{orig_name}/results.json')
            if orig_file.exists():
                with open(orig_file) as f:
                    orig_data = json.load(f)
                    orig_dice = orig_data['test_dice']
                    
                improvement = fixed_dice - orig_dice
                pct = (improvement / orig_dice) * 100
                
                print(f"{label:<25} {orig_dice:>12.4f} {fixed_dice:>12.4f} "
                      f"{improvement:>+8.4f} ({pct:>+5.1f}%)")
    
    print("\n✅ Re-run complete!")
    print(f"   Results saved to: {output_dir}")
    print("\n📊 Summary:")
    print("   - All configs trained with proper settings (batch=32, patience=10, epochs=50)")
    print("   - Smaller batch size reduces disk I/O contention")
    print("   - GPU utilization should be ~95-100% (vs ~40% in original)")
    print("   - Expected: All configs should reach 89-91% Dice (clean data)")
    print("   - Note: Old 97-99% expectations were from LEAKED data (invalid)")

if __name__ == "__main__":
    main()
