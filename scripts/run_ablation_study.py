#!/usr/bin/env python3
"""
Ablation Study for GNN Brain Tumor Segmentation

Tests the impact of different architectural choices:
1. Number of GNN layers (3, 4, 5, 6)
2. Hidden dimensions (128, 256, 512)
3. GNN type (GraphSAGE vs GAT)
4. Edge features (with vs without)

All experiments run on fold 0 for speed, with consistent hyperparameters.
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

# Enable GPU optimizations
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = False
torch.set_float32_matmul_precision('high')

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from gnn_model import TumorSegmentationGNN, CombinedLoss
from dataset import BraTSGraphDataset

def calculate_dice_score(pred, target, smooth=1.0):
    """Calculate Dice score between prediction and target"""
    pred = pred.view(-1)
    target = target.view(-1)
    intersection = (pred * target).sum()
    dice = (2. * intersection + smooth) / (pred.sum() + target.sum() + smooth)
    return dice.item()

# Configuration - OPTIMIZED FOR QUALITY (10-12 hour run)
BASE_CONFIG = {
    'fold': 0,
    'data_dir': 'data/graphs',
    'cv_dir': 'data/cv_folds',
    'batch_size': 96,  # Balanced for stability and speed
    'num_epochs': 25,  # Sufficient for convergence with early stopping
    'lr': 0.001,
    'weight_decay': 1e-5,
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'patience': 5,  # Early stopping - stops if no improvement for 5 epochs
}

# Ablation configurations - ALL 8 TESTS FOR COMPREHENSIVE STUDY
ABLATIONS = {
    'baseline': {
        'name': 'Baseline (5 layers, 256D, GraphSAGE)',
        'num_layers': 5,
        'hidden_channels': 256,
        'gnn_type': 'sage',
        'use_edge_features': True,
    },
    'layers_3': {
        'name': '3 Layers (vs 5)',
        'num_layers': 3,
        'hidden_channels': 256,
        'gnn_type': 'sage',
        'use_edge_features': True,
    },
    'layers_4': {
        'name': '4 Layers (vs 5)',
        'num_layers': 4,
        'hidden_channels': 256,
        'gnn_type': 'sage',
        'use_edge_features': True,
    },
    'layers_6': {
        'name': '6 Layers (vs 5)',
        'num_layers': 6,
        'hidden_channels': 256,
        'gnn_type': 'sage',
        'use_edge_features': True,
    },
    'hidden_128': {
        'name': 'Hidden Dim 128 (vs 256)',
        'num_layers': 5,
        'hidden_channels': 128,
        'gnn_type': 'sage',
        'use_edge_features': True,
    },
    'hidden_512': {
        'name': 'Hidden Dim 512 (vs 256)',
        'num_layers': 5,
        'hidden_channels': 512,
        'gnn_type': 'sage',
        'use_edge_features': True,
    },
    'gat': {
        'name': 'GAT (vs GraphSAGE)',
        'num_layers': 5,
        'hidden_channels': 256,
        'gnn_type': 'gat',
        'use_edge_features': True,
    },
    'no_edge_features': {
        'name': 'Without Edge Features',
        'num_layers': 5,
        'hidden_channels': 256,
        'gnn_type': 'sage',
        'use_edge_features': False,
    },
}

def load_data(fold, data_dir, cv_dir):
    """Load train/val/test data for specific fold"""
    print(f"\nLoading data for fold {fold}...")
    
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
    
    return train_files, val_files, test_files

def create_model(config, device):
    """Create GNN model with specified configuration"""
    model = TumorSegmentationGNN(
        in_channels=12,
        hidden_channels=config['hidden_channels'],
        num_layers=config['num_layers'],
        gnn_type=config['gnn_type'],
        dropout=0.2
    ).to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    return model, total_params, trainable_params

def train_epoch(model, loader, criterion, optimizer, scheduler, device, use_edge_features, scaler=None):
    """Train for one epoch with mixed precision"""
    model.train()
    total_loss = 0
    total_dice = 0
    num_batches = 0
    
    for batch_idx, batch in enumerate(loader):
        try:
            batch = batch.to(device, non_blocking=True)
        except AttributeError as e:
            print(f"ERROR at batch {batch_idx}: {e}")
            print(f"Batch type: {type(batch)}")
            print(f"Batch content: {batch}")
            raise
        
        # Remove edge features if needed
        if not use_edge_features:
            batch.edge_attr = None
        
        # Forward pass with autocast for mixed precision
        with torch.cuda.amp.autocast(enabled=(scaler is not None)):
            logits, embeddings = model(batch)
            
            # Calculate loss
            loss, ce_loss, dice_loss, _ = criterion(
                logits, embeddings, batch.y, batch.slice_mask
            )
        
        # Backward pass with gradient scaling
        optimizer.zero_grad(set_to_none=True)
        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
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
            
            # Remove edge features if needed
            if not use_edge_features:
                batch.edge_attr = None
            
            # Forward pass
            logits, embeddings = model(batch)
            
            # Calculate loss
            loss, _, _, _ = criterion(
                logits, embeddings, batch.y, batch.slice_mask
            )
            
            # Calculate Dice
            preds = torch.sigmoid(logits) > 0.5
            dice = calculate_dice_score(preds.float(), batch.y)
            
            total_loss += loss.item()
            total_dice += dice
            num_batches += 1
    
    return total_loss / num_batches, total_dice / num_batches

def run_single_ablation(ablation_name, config, base_config):
    """Run training for one ablation configuration"""
    print("\n" + "="*80)
    print(f"ABLATION: {config['name']}")
    print("="*80)
    
    device = base_config['device']
    
    # Load data
    train_files, val_files, test_files = load_data(
        base_config['fold'], 
        base_config['data_dir'],
        base_config['cv_dir']
    )
    
    # Load graphs
    print("\nLoading graphs into memory...")
    
    # Create datasets
    train_dataset = BraTSGraphDataset(root_dir=None, split='train', graph_files=train_files)
    val_dataset = BraTSGraphDataset(root_dir=None, split='val', graph_files=val_files)
    test_dataset = BraTSGraphDataset(root_dir=None, split='test', graph_files=test_files)
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=base_config['batch_size'],
        shuffle=True,
        num_workers=8,  # Increased from 4 to 8 for faster data loading
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=4  # Prefetch 4 batches per worker
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=base_config['batch_size'],
        shuffle=False,
        num_workers=8,  # Increased from 4 to 8
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=4
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=base_config['batch_size'],
        shuffle=False,
        num_workers=8,  # Increased from 4 to 8
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=4
    )
    
    # Create model
    print(f"\nCreating model...")
    print(f"  Layers: {config['num_layers']}")
    print(f"  Hidden: {config['hidden_channels']}")
    print(f"  Type:   {config['gnn_type']}")
    print(f"  Edges:  {config['use_edge_features']}")
    
    model, total_params, trainable_params = create_model(config, device)
    print(f"  Parameters: {total_params:,}")
    
    # Skip torch.compile due to dynamic graph sizes causing slow compilation
    # Rely on mixed precision (FP16) for speedup instead
    
    # Training setup
    criterion = CombinedLoss(use_consistency=False)
    optimizer = AdamW(
        model.parameters(),
        lr=base_config['lr'],
        weight_decay=base_config['weight_decay']
    )
    
    # Disable mixed precision - causing NaN loss with FP16
    scaler = None
    print("  Using FP32 (full precision) for numerical stability...")
    
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
    print(f"\nTraining for {base_config['num_epochs']} epochs...")
    
    best_val_dice = 0
    best_epoch = 0
    patience_counter = 0
    train_history = []
    val_history = []
    
    start_time = time.time()
    
    for epoch in range(base_config['num_epochs']):
        epoch_start = time.time()
        
        # Train
        train_loss, train_dice = train_epoch(
            model, train_loader, criterion, optimizer, scheduler, 
            device, config['use_edge_features'], scaler
        )
        
        # Validate
        val_loss, val_dice = validate(
            model, val_loader, criterion, device, config['use_edge_features']
        )
        
        epoch_time = time.time() - epoch_start
        
        # Save history
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
            save_dir = Path('research_results/ablation_study') / ablation_name
            save_dir.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), save_dir / 'best_model.pth')
        else:
            patience_counter += 1
        
        # Early stopping
        if patience_counter >= base_config['patience']:
            print(f"\nEarly stopping at epoch {epoch+1}")
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
        'ablation_name': ablation_name,
        'config': config,
        'parameters': total_params,
        'best_val_dice': float(best_val_dice),
        'best_epoch': int(best_epoch),
        'test_dice': float(test_dice),
        'training_time_min': float(training_time / 60),
        'train_history': train_history,
        'val_history': val_history,
    }
    
    with open(save_dir / 'results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    return results

def main():
    """Run all ablation studies"""
    print("\n" + "╔" + "═"*78 + "╗")
    print("║" + " "*25 + "ABLATION STUDY" + " "*40 + "║")
    print("║" + " "*20 + "GNN Architecture Analysis" + " "*33 + "║")
    print("╚" + "═"*78 + "╝")
    
    print(f"\nDevice: {BASE_CONFIG['device']}")
    print(f"Fold: {BASE_CONFIG['fold']}")
    print(f"Epochs: {BASE_CONFIG['num_epochs']}")
    print(f"Batch Size: {BASE_CONFIG['batch_size']}")
    print(f"\nTesting {len(ABLATIONS)} configurations...")
    
    # Run all ablations
    all_results = {}
    
    for ablation_name, config in ABLATIONS.items():
        try:
            results = run_single_ablation(ablation_name, config, BASE_CONFIG)
            all_results[ablation_name] = results
        except Exception as e:
            print(f"\n❌ ERROR in {ablation_name}: {str(e)}")
            import traceback
            traceback.print_exc()
            continue
    
    # Save combined results
    output_dir = Path('research_results/ablation_study')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    with open(output_dir / 'all_results.json', 'w') as f:
        json.dump(all_results, f, indent=2)
    
    # Print summary
    print("\n" + "="*80)
    print("ABLATION STUDY SUMMARY")
    print("="*80)
    print(f"{'Configuration':<30} {'Params':>12} {'Val Dice':>10} {'Test Dice':>10} {'Time (min)':>12}")
    print("-"*80)
    
    # Sort by test dice
    sorted_results = sorted(
        all_results.items(),
        key=lambda x: x[1]['test_dice'],
        reverse=True
    )
    
    for ablation_name, results in sorted_results:
        print(f"{results['config']['name']:<30} "
              f"{results['parameters']:>12,} "
              f"{results['best_val_dice']:>10.4f} "
              f"{results['test_dice']:>10.4f} "
              f"{results['training_time_min']:>12.1f}")
    
    print("-"*80)
    
    # Baseline comparison
    baseline_results = all_results.get('baseline', None)
    if baseline_results:
        baseline_dice = baseline_results['test_dice']
        print(f"\nBaseline Test Dice: {baseline_dice:.4f}")
        print("\nRelative Performance:")
        print(f"{'Configuration':<30} {'Δ Dice':>10} {'% Change':>10}")
        print("-"*80)
        
        for ablation_name, results in sorted_results:
            if ablation_name != 'baseline':
                delta = results['test_dice'] - baseline_dice
                pct = (delta / baseline_dice) * 100
                print(f"{results['config']['name']:<30} "
                      f"{delta:>+10.4f} "
                      f"{pct:>+9.2f}%")
    
    print("\n✅ Ablation study complete!")
    print(f"   Results saved to: {output_dir}")

if __name__ == "__main__":
    main()
