"""
Train a single cross-validation fold
Modified from train_maxpower.py for CV training
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
from dataset import BraTSGraphDataset, BinaryTransform
from gnn_model import TumorSegmentationGNN
from cross_validation import load_fold_data


def set_seed(seed: int = 42):
    """Set all random seeds for reproducibility"""
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        # Enable deterministic algorithms for reproducibility
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


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
                accumulation_steps=4):
    """Train for one epoch with gradient accumulation"""
    model.train()
    total_loss = 0
    all_predictions = []
    all_labels = []
    
    optimizer.zero_grad()
    
    for batch_idx, data in enumerate(train_loader):
        data = data.to(device)
        
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
        data = data.to(device)
        
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
    fold_dir: str = './data/cv_folds',
    output_dir: str = './checkpoints/cv_experiments',
    epochs: int = 50,
    batch_size: int = 32,
    learning_rate: float = 0.001,
    hidden_channels: int = 256,
    num_layers: int = 5,
    accumulation_steps: int = 4,
    device: str = 'cuda'
):
    """
    Train a single cross-validation fold
    
    Args:
        fold_idx: Which fold to train (0 to k-1)
        fold_dir: Directory containing fold assignments
        output_dir: Directory to save checkpoints and results
        epochs: Number of training epochs
        batch_size: Batch size
        learning_rate: Initial learning rate
        hidden_channels: Hidden dimension size
        num_layers: Number of GNN layers
        accumulation_steps: Gradient accumulation steps
        device: Device to use for training
    """
    print("=" * 80)
    print(f"TRAINING FOLD {fold_idx}")
    print("=" * 80)
    
    # Set seed for reproducibility
    set_seed(42)
    
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
    
    # Create dataloaders
    train_loader = GeometricDataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=4, pin_memory=True
    )
    val_loader = GeometricDataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=4, pin_memory=True
    )
    test_loader = GeometricDataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
        num_workers=4, pin_memory=True
    )
    
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
    
    # Loss and optimizer
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
    
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
    
    parser = argparse.ArgumentParser(description='Train a single CV fold')
    parser.add_argument('--fold_idx', type=int, required=True,
                       help='Fold index to train (0 to k-1)')
    parser.add_argument('--fold_dir', type=str, default='./data/cv_folds',
                       help='Directory containing fold assignments')
    parser.add_argument('--output_dir', type=str, default='./checkpoints/cv_experiments',
                       help='Output directory for checkpoints')
    parser.add_argument('--epochs', type=int, default=50,
                       help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size')
    parser.add_argument('--lr', type=float, default=0.001,
                       help='Learning rate')
    parser.add_argument('--hidden_channels', type=int, default=256,
                       help='Hidden dimension size')
    parser.add_argument('--num_layers', type=int, default=5,
                       help='Number of GNN layers')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to use (cuda/cpu)')
    
    args = parser.parse_args()
    
    # Train fold
    results = train_fold(
        fold_idx=args.fold_idx,
        fold_dir=args.fold_dir,
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        hidden_channels=args.hidden_channels,
        num_layers=args.num_layers,
        device=args.device
    )
    
    print("\n✅ Fold training complete!")
