"""
Train a single cross-validation fold  — v4: BCE + DiceLoss
================================================
OPTION 2 (2026-03-14): Uses BCE + DiceLoss combined criterion.
Original BCE-only version preserved in train_cv_fold.py (binary_v3 checkpoints).
Saves to checkpoints/binary_v4/ — does NOT touch binary_v3.
Roll back by simply using binary_v3 checkpoints instead.
See FIX_LOG.md → DECISION-1 for context.
"""

import os
import sys
import json
import time
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from datetime import datetime
from torch.utils.data import DataLoader
from torch.amp import autocast, GradScaler
from torch_geometric.loader import DataLoader as GeometricDataLoader

# Import existing modules
from config import get_config
from dataset import BraTSGraphDataset, BinaryTransform
from gnn_model import TumorSegmentationGNN, DiceLoss
from cross_validation import load_fold_data


def set_seed(seed: int = 42):
    """Set all random seeds for reproducibility"""
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        # benchmark=True lets cuDNN auto-tune kernels for current input shapes
        # (~10-30% speedup on RTX 2060 Tensor Cores)
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False


def compute_metrics(predictions, labels):
    """Compute evaluation metrics"""
    predictions = predictions.cpu().numpy()
    labels = labels.cpu().numpy()
    
    # Binary predictions
    pred_binary = (predictions > 0.5).astype(int)
    
    # Confusion matrix
    tp = np.sum((pred_binary == 1) & (labels == 1))
    tn = np.sum((pred_binary == 0) & (labels == 0))
    fp = np.sum((pred_binary == 1) & (labels == 0))
    fn = np.sum((pred_binary == 0) & (labels == 1))
    
    # Metrics
    epsilon = 1e-7
    dice = (2 * tp) / (2 * tp + fp + fn + epsilon)
    accuracy = (tp + tn) / (tp + tn + fp + fn + epsilon)
    sensitivity = tp / (tp + fn + epsilon)
    specificity = tn / (tn + fp + epsilon)
    precision = tp / (tp + fp + epsilon)
    
    return {
        'dice': float(dice),
        'accuracy': float(accuracy),
        'sensitivity': float(sensitivity),
        'specificity': float(specificity),
        'precision': float(precision),
        'tp': int(tp), 'tn': int(tn), 'fp': int(fp), 'fn': int(fn)
    }


def train_epoch(model, train_loader, criterion, optimizer, scheduler, scaler, device,
                accumulation_steps=2):  # FIX-3 (2026-03-14): default matches config.yaml value
    """Train for one epoch with gradient accumulation"""
    model.train()
    total_loss = 0
    all_predictions = []
    all_labels = []
    
    optimizer.zero_grad()
    
    for batch_idx, data in enumerate(train_loader):
        data = data.to(device, non_blocking=True)  # non_blocking exploits pin_memory

        # Mixed precision training
        with autocast(device_type='cuda'):
            out, _ = model(data)  # Model returns (logits, embeddings)
            loss = criterion(out.squeeze(), data.y.float())
            loss = loss / accumulation_steps
        
        # Backward pass with gradient scaling
        scaler.scale(loss).backward()
        
        # Gradient accumulation
        if (batch_idx + 1) % accumulation_steps == 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            
            if scheduler is not None:
                scheduler.step()
        
        # Track metrics
        total_loss += loss.item() * accumulation_steps
        predictions = torch.sigmoid(out.squeeze()).detach()
        all_predictions.append(predictions)
        all_labels.append(data.y)
    
    # Compute metrics
    all_predictions = torch.cat(all_predictions)
    all_labels = torch.cat(all_labels)
    metrics = compute_metrics(all_predictions, all_labels)
    metrics['loss'] = total_loss / len(train_loader)
    
    return metrics


@torch.no_grad()
def evaluate(model, data_loader, criterion, device):
    """Evaluate model on validation/test set"""
    model.eval()
    total_loss = 0
    all_predictions = []
    all_labels = []
    
    for data in data_loader:
        data = data.to(device, non_blocking=True)

        with autocast(device_type='cuda'):
            out, _ = model(data)  # Model returns (logits, embeddings)
            loss = criterion(out.squeeze(), data.y.float())
        
        total_loss += loss.item()
        predictions = torch.sigmoid(out.squeeze())
        all_predictions.append(predictions)
        all_labels.append(data.y)
    
    # Compute metrics
    all_predictions = torch.cat(all_predictions)
    all_labels = torch.cat(all_labels)
    metrics = compute_metrics(all_predictions, all_labels)
    metrics['loss'] = total_loss / len(data_loader)
    
    return metrics


def train_fold(
    fold_idx: int,
    fold_dir: str = None,
    output_dir: str = None,
    epochs: int = None,
    batch_size: int = None,
    learning_rate: float = None,
    hidden_channels: int = None,
    num_layers: int = None,
    accumulation_steps: int = None,
    device: str = None
):
    """
    Train a single cross-validation fold

    Args:
        fold_idx: Which fold to train (0 to k-1)
        fold_dir: Directory containing fold assignments (default from config)
        output_dir: Directory to save checkpoints and results (default from config)
        epochs: Number of training epochs (default from config)
        batch_size: Batch size (default from config)
        learning_rate: Initial learning rate (default from config)
        hidden_channels: Hidden dimension size (default from config)
        num_layers: Number of GNN layers (default from config)
        accumulation_steps: Gradient accumulation steps (default from config)
        device: Device to use for training (default from config)
    """
    # Load config (lazy loading to avoid import-time crashes)
    config = get_config()

    # Load defaults from config if not provided
    fold_dir = fold_dir or config.data_cv_folds
    output_dir = output_dir or config.checkpoints_binary
    epochs = epochs if epochs is not None else config.get('model.training.max_epochs', 50)
    batch_size = batch_size if batch_size is not None else config.get('model.training.batch_size', 32)
    learning_rate = learning_rate if learning_rate is not None else config.get('model.training.learning_rate', 0.001)
    hidden_channels = hidden_channels if hidden_channels is not None else config.get('model.gnn.hidden_channels', 256)
    num_layers = num_layers if num_layers is not None else config.get('model.gnn.num_layers', 5)
    accumulation_steps = accumulation_steps if accumulation_steps is not None else config.get('model.training.accumulation_steps', 4)
    device = device or config.get('hardware.device', 'cuda')

    print("=" * 80)
    print(f"TRAINING FOLD {fold_idx}")
    print("=" * 80)
    print(f"📁 Using paths from config:")
    print(f"   Fold dir: {fold_dir}")
    print(f"   Output dir: {output_dir}")

    # Set seed for reproducibility
    seed = config.get('project.seed', 42)
    set_seed(seed)
    
    # Set device
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create output directory
    fold_output_dir = Path(output_dir) / f'fold_{fold_idx}'
    fold_output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load fold data
    print("\nLoading fold data...")
    fold_data = load_fold_data(fold_dir, fold_idx)
    
    # ⚠️ BINARY TRANSFORM (Nov 30, 2025):
    # Graphs store multi-class labels (0,1,2,4). Convert to binary on-the-fly.
    binary_transform = BinaryTransform()
    
    # Create datasets
    print("Creating datasets...")
    train_dataset = BraTSGraphDataset(
        root_dir=None,
        split='train',
        graph_files=fold_data['train_graphs'],
        transform=binary_transform  # ← Convert multi-class to binary
    )
    
    val_dataset = BraTSGraphDataset(
        root_dir=None,
        split='val',
        graph_files=fold_data['val_graphs'],
        transform=binary_transform  # ← Convert multi-class to binary
    )
    
    test_dataset = BraTSGraphDataset(
        root_dir=None,
        split='test',
        graph_files=fold_data['test_graphs'],
        transform=binary_transform  # ← Convert multi-class to binary
    )
    
    # Create dataloaders — num_workers=4 parallelises loading; data is already
    # in RAM so workers read from the preloaded dict (no disk I/O bottleneck)
    num_workers = config.get('hardware.num_workers', 4)
    loader_kwargs = dict(
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=(num_workers > 0),
        prefetch_factor=2 if num_workers > 0 else None,
    )
    train_loader = GeometricDataLoader(train_dataset, shuffle=True,  **loader_kwargs)
    val_loader   = GeometricDataLoader(val_dataset,   shuffle=False, **loader_kwargs)
    test_loader  = GeometricDataLoader(test_dataset,  shuffle=False, **loader_kwargs)
    
    print(f"\nDataset sizes:")
    print(f"  Train: {len(train_dataset)} graphs")
    print(f"  Val:   {len(val_dataset)} graphs")
    print(f"  Test:  {len(test_dataset)} graphs")
    
    # Get feature dimension from first batch
    sample_data = next(iter(train_loader))
    in_channels = sample_data.x.size(1)
    print(f"  Input features: {in_channels}")
    
    # Create model
    print("\nInitializing model...")
    model = TumorSegmentationGNN(
        in_channels=in_channels,
        hidden_channels=hidden_channels,
        num_layers=num_layers,
        dropout=0.1
    ).to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Total parameters: {total_params:,}")

    # torch.compile — fuses GNN ops into optimized kernels (~15-25% speedup)
    try:
        # 'default' mode — operator fusion without CUDA Graphs.
        # PyG has variable-size graphs per batch so 'reduce-overhead' (CUDA Graphs) fails.
        model = torch.compile(model, mode='default')
        print("  torch.compile() enabled (default mode)")
    except Exception as e:
        print(f"  torch.compile() skipped: {e}")
    
    # Loss and optimizer — v4: BCE + DiceLoss (equal weight 0.5 each)
    # Option 2 (2026-03-14): directly optimises Dice metric; more robust to class imbalance.
    # DiceLoss defined in gnn_model.py. BCE provides stable early gradients; Dice handles minority class.
    # CrossSliceConsistencyLoss intentionally excluded — model has no cross-slice architecture.
    _bce  = nn.BCEWithLogitsLoss()
    _dice = DiceLoss()
    criterion = lambda logits, targets: 0.5 * _bce(logits, targets) + 0.5 * _dice(logits, targets)
    weight_decay = config.get('model.training.weight_decay', 0.01)  # FIX-2 (2026-03-14): read from config
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    
    # Learning rate scheduler
    total_steps = len(train_loader) * epochs // accumulation_steps
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=learning_rate,
        total_steps=total_steps,
        pct_start=0.1,
        anneal_strategy='cos'
    )
    
    # Gradient scaler for mixed precision
    scaler = GradScaler(device='cuda')
    
    # Training loop
    print(f"\nStarting training for {epochs} epochs...")
    print("=" * 80)
    
    best_val_dice = 0.0
    best_checkpoint = None  # Initialize to None
    history = []
    start_time = time.time()
    
    for epoch in range(epochs):
        epoch_start = time.time()
        
        # Train
        train_metrics = train_epoch(
            model, train_loader, criterion, optimizer, scheduler, scaler,
            device, accumulation_steps
        )
        
        # Validate
        val_metrics = evaluate(model, val_loader, criterion, device)
        
        epoch_time = time.time() - epoch_start
        
        # Log progress
        print(f"Epoch {epoch+1}/{epochs} ({epoch_time:.1f}s)")
        print(f"  Train - Loss: {train_metrics['loss']:.4f}, "
              f"Dice: {train_metrics['dice']:.4f}, Acc: {train_metrics['accuracy']:.4f}")
        print(f"  Val   - Loss: {val_metrics['loss']:.4f}, "
              f"Dice: {val_metrics['dice']:.4f}, Acc: {val_metrics['accuracy']:.4f}")
        
        # Save history
        history.append({
            'epoch': epoch + 1,
            'train': train_metrics,
            'val': val_metrics,
            'time': epoch_time
        })
        
        # Save best model
        if val_metrics['dice'] > best_val_dice:
            best_val_dice = val_metrics['dice']
            best_checkpoint = {
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_dice': best_val_dice,
                'fold_idx': fold_idx
            }
            torch.save(best_checkpoint, fold_output_dir / 'best_model.pth')
            print(f"  ✓ Best model saved (Val Dice: {best_val_dice:.4f})")
    
    total_time = time.time() - start_time
    print("=" * 80)
    print(f"Training complete! Total time: {total_time/60:.1f} minutes")
    print(f"Best validation Dice: {best_val_dice:.4f}")
    
    # Evaluate on test set with best model
    print("\nEvaluating on test set...")
    if best_checkpoint is not None:
        model.load_state_dict(best_checkpoint['model_state_dict'])
    else:
        print("  ⚠️  Warning: No checkpoint saved (validation Dice never improved from 0.0)")
        # Save current model as best
        best_checkpoint = {
            'epoch': epochs,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_dice': best_val_dice,
            'fold_idx': fold_idx
        }
        torch.save(best_checkpoint, fold_output_dir / 'best_model.pth')
    
    test_metrics = evaluate(model, test_loader, criterion, device)
    
    print(f"Test Results:")
    print(f"  Dice:        {test_metrics['dice']:.4f}")
    print(f"  Accuracy:    {test_metrics['accuracy']:.4f}")
    print(f"  Sensitivity: {test_metrics['sensitivity']:.4f}")
    print(f"  Specificity: {test_metrics['specificity']:.4f}")
    print(f"  Precision:   {test_metrics['precision']:.4f}")
    
    # Save results
    results = {
        'fold_idx': fold_idx,
        'best_val_dice': best_val_dice,
        'test_metrics': test_metrics,
        'training_time': total_time,
        'history': history,
        'config': {
            'epochs': epochs,
            'batch_size': batch_size,
            'learning_rate': learning_rate,
            'hidden_channels': hidden_channels,
            'num_layers': num_layers,
            'accumulation_steps': accumulation_steps
        }
    }
    
    results_file = fold_output_dir / 'results.json'
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✅ Results saved to {results_file}")
    print("=" * 80)
    
    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Train a single CV fold (uses config.yaml for defaults)')
    parser.add_argument('--fold_idx', type=int, required=True,
                       help='Fold index to train (0 to k-1)')
    parser.add_argument('--fold_dir', type=str, default=None,
                       help='Directory containing fold assignments (default from config.yaml)')
    parser.add_argument('--output_dir', type=str, default=None,
                       help='Output directory for checkpoints (default from config.yaml)')
    parser.add_argument('--epochs', type=int, default=None,
                       help='Number of training epochs (default from config.yaml: 50)')
    parser.add_argument('--batch_size', type=int, default=None,
                       help='Batch size (default from config.yaml: 32)')
    parser.add_argument('--lr', type=float, default=None,
                       help='Learning rate (default from config.yaml: 0.001)')
    parser.add_argument('--hidden_channels', type=int, default=None,
                       help='Hidden dimension size (default from config.yaml: 256)')
    parser.add_argument('--num_layers', type=int, default=None,
                       help='Number of GNN layers (default from config.yaml: 5)')
    parser.add_argument('--accumulation_steps', type=int, default=None,
                       help='Gradient accumulation steps (default from config.yaml: 2)')  # FIX-3 (2026-03-14)
    parser.add_argument('--device', type=str, default=None,
                       help='Device to use (default from config.yaml: cuda)')

    args = parser.parse_args()

    # Train fold (config defaults will be used for None values)
    results = train_fold(
        fold_idx=args.fold_idx,
        fold_dir=args.fold_dir,
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        hidden_channels=args.hidden_channels,
        num_layers=args.num_layers,
        accumulation_steps=args.accumulation_steps,  # FIX-3 (2026-03-14)
        device=args.device
    )

    print("\n✅ Fold training complete!")
