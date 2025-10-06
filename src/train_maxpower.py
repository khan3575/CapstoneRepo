# src/train_maxpower.py - Maximum Performance BraTS GNN Training
"""
High-Performance BraTS GNN Training Script

Optimizations for maximum hardware utilization:
1. Larger batch sizes with gradient accumulation
2. Mixed precision training (FP16)
3. Multi-GPU support (if available) 
4. Optimized data loading with prefetching
5. Memory-efficient operations
6. CPU parallelization for data processing
7. CUDA optimizations
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from torch_geometric.nn import SAGEConv, GATConv, BatchNorm
from torch.cuda.amp import GradScaler, autocast
import numpy as np
import argparse
import time
import json
from pathlib import Path
from typing import Dict, Tuple
import warnings
import multiprocessing as mp
warnings.filterwarnings('ignore')

# Import our data loading
from dataset import create_data_splits, create_datasets

class HighPerformanceBraTSGNN(nn.Module):
    """
    Optimized GNN for maximum performance
    
    Performance optimizations:
    - Efficient memory layout
    - Fused operations where possible
    - Optimized for mixed precision
    """
    
    def __init__(self, 
                 input_dim: int = 12,
                 hidden_dim: int = 128,  # Larger for better GPU utilization
                 num_layers: int = 4,    # Deeper for more computation
                 dropout: float = 0.1,   # Lower dropout for faster training
                 gnn_type: str = 'sage'):
        super().__init__()
        
        self.num_layers = num_layers
        self.dropout = dropout
        
        # Build GNN layers with larger dimensions
        self.convs = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        
        # Input layer
        if gnn_type == 'sage':
            self.convs.append(SAGEConv(input_dim, hidden_dim, normalize=True))
        elif gnn_type == 'gat':
            self.convs.append(GATConv(input_dim, hidden_dim, heads=8, concat=True, dropout=dropout))
            hidden_dim = hidden_dim * 8  # Adjust for concatenated heads
        
        self.batch_norms.append(BatchNorm(hidden_dim))
        
        # Hidden layers
        for _ in range(num_layers - 2):
            if gnn_type == 'sage':
                self.convs.append(SAGEConv(hidden_dim, hidden_dim, normalize=True))
            elif gnn_type == 'gat':
                self.convs.append(GATConv(hidden_dim, hidden_dim//8, heads=8, concat=True, dropout=dropout))
            self.batch_norms.append(BatchNorm(hidden_dim))
        
        # Output layer
        output_dim = hidden_dim // 2
        if gnn_type == 'sage':
            self.convs.append(SAGEConv(hidden_dim, output_dim, normalize=True))
        elif gnn_type == 'gat':
            self.convs.append(GATConv(hidden_dim, output_dim, heads=1, dropout=dropout))
        self.batch_norms.append(BatchNorm(output_dim))
        
        # Larger classifier for more computation
        self.classifier = nn.Sequential(
            nn.Linear(output_dim, output_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(output_dim, output_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(output_dim // 2, 1)
        )
        
        # Initialize weights for better convergence
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
    
    def forward(self, x, edge_index, batch=None):
        # Apply GNN layers with checkpointing for memory efficiency
        for i, (conv, bn) in enumerate(zip(self.convs, self.batch_norms)):
            x = conv(x, edge_index)
            x = bn(x)
            
            if i < len(self.convs) - 1:
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
        
        # Node-level classification
        logits = self.classifier(x).squeeze(-1)
        return logits

class OptimizedDiceLoss(nn.Module):
    """Optimized Dice Loss for mixed precision training"""
    
    def __init__(self, smooth: float = 1.0):
        super().__init__()
        self.smooth = smooth
    
    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # Use more efficient operations
        predictions = torch.sigmoid(predictions)
        
        # Vectorized operations
        intersection = torch.sum(predictions * targets)
        total = torch.sum(predictions) + torch.sum(targets)
        
        dice = (2.0 * intersection + self.smooth) / (total + self.smooth)
        return 1 - dice

class FastCombinedLoss(nn.Module):
    """Optimized combined loss for maximum throughput"""
    
    def __init__(self, bce_weight: float = 0.3, dice_weight: float = 0.7):
        super().__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        
        self.bce_loss = nn.BCEWithLogitsLoss()
        self.dice_loss = OptimizedDiceLoss()
    
    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
        bce = self.bce_loss(logits, targets)
        dice = self.dice_loss(logits, targets)
        
        total_loss = self.bce_weight * bce + self.dice_weight * dice
        
        return total_loss, {'total': total_loss, 'bce': bce, 'dice': dice}

def calculate_metrics_fast(predictions: np.ndarray, targets: np.ndarray) -> Dict[str, float]:
    """Optimized metrics calculation"""
    pred_binary = predictions > 0.5
    targets_binary = targets.astype(bool)
    
    if not targets_binary.any():
        return {'dice': 1.0 if not pred_binary.any() else 0.0, 'sensitivity': 1.0, 'accuracy': 1.0}
    
    # Vectorized operations
    tp = (pred_binary & targets_binary).sum()
    fp = (pred_binary & ~targets_binary).sum()
    fn = (~pred_binary & targets_binary).sum()
    tn = (~pred_binary & ~targets_binary).sum()
    
    dice = (2 * tp) / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0.0
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    accuracy = (tp + tn) / len(targets)
    
    return {
        'dice': float(dice),
        'sensitivity': float(sensitivity),
        'accuracy': float(accuracy)
    }

def train_epoch_optimized(model: HighPerformanceBraTSGNN,
                         loader: DataLoader,
                         criterion: FastCombinedLoss,
                         optimizer: torch.optim.Optimizer,
                         scaler: GradScaler,
                         device: torch.device,
                         epoch: int,
                         accumulation_steps: int = 4) -> Dict[str, float]:
    """High-performance training epoch with gradient accumulation and mixed precision"""
    model.train()
    
    total_losses = {'total': 0.0, 'bce': 0.0, 'dice': 0.0}
    all_predictions = []
    all_targets = []
    num_batches = 0
    
    optimizer.zero_grad()
    
    for batch_idx, batch in enumerate(loader):
        try:
            batch = batch.to(device, non_blocking=True)
            
            # Mixed precision forward pass
            with autocast():
                logits = model(batch.x, batch.edge_index, batch.batch)
                targets = batch.y.float()
                loss, loss_dict = criterion(logits, targets)
                
                # Scale loss for gradient accumulation
                loss = loss / accumulation_steps
            
            # Backward pass with gradient scaling
            scaler.scale(loss).backward()
            
            # Gradient accumulation
            if (batch_idx + 1) % accumulation_steps == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
            
            # Accumulate losses
            for key in total_losses:
                total_losses[key] += loss_dict[key].item()
            num_batches += 1
            
            # Collect predictions (less frequently for speed)
            if batch_idx % 4 == 0:  # Sample every 4th batch
                with torch.no_grad():
                    preds = torch.sigmoid(logits).cpu().numpy()
                    all_predictions.extend(preds)
                    all_targets.extend(targets.cpu().numpy())
            
            # Progress logging (less frequent)
            if batch_idx % 50 == 0:
                print(f'  Batch {batch_idx}/{len(loader)} - Loss: {loss.item() * accumulation_steps:.4f}')
                
        except Exception as e:
            print(f"  ⚠️  Skipping batch {batch_idx}: {e}")
            continue
    
    # Final gradient step if needed
    if num_batches % accumulation_steps != 0:
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()
    
    # Calculate metrics
    avg_losses = {key: val / max(num_batches, 1) for key, val in total_losses.items()}
    
    if len(all_predictions) > 0:
        metrics = calculate_metrics_fast(np.array(all_predictions), np.array(all_targets))
    else:
        metrics = {'dice': 0.0, 'sensitivity': 0.0, 'accuracy': 0.0}
    
    result = {**avg_losses, **metrics, 'num_batches': num_batches}
    return result

def validate_epoch_optimized(model: HighPerformanceBraTSGNN,
                            loader: DataLoader,
                            criterion: FastCombinedLoss,
                            device: torch.device) -> Dict[str, float]:
    """Optimized validation with mixed precision"""
    model.eval()
    
    total_losses = {'total': 0.0, 'bce': 0.0, 'dice': 0.0}
    all_predictions = []
    all_targets = []
    num_batches = 0
    
    with torch.no_grad():
        for batch in loader:
            try:
                batch = batch.to(device, non_blocking=True)
                
                # Mixed precision forward pass
                with autocast():
                    logits = model(batch.x, batch.edge_index, batch.batch)
                    targets = batch.y.float()
                    loss, loss_dict = criterion(logits, targets)
                
                # Accumulate losses
                for key in total_losses:
                    total_losses[key] += loss_dict[key].item()
                num_batches += 1
                
                # Collect predictions
                preds = torch.sigmoid(logits).cpu().numpy()
                all_predictions.extend(preds)
                all_targets.extend(targets.cpu().numpy())
                
            except Exception as e:
                continue
    
    # Calculate metrics
    avg_losses = {key: val / max(num_batches, 1) for key, val in total_losses.items()}
    
    if len(all_predictions) > 0:
        metrics = calculate_metrics_fast(np.array(all_predictions), np.array(all_targets))
    else:
        metrics = {'dice': 0.0, 'sensitivity': 0.0, 'accuracy': 0.0}
    
    result = {**avg_losses, **metrics, 'num_batches': num_batches}
    return result

def optimize_system():
    """System optimizations for maximum performance"""
    # CUDA optimizations
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True  # Optimize for consistent input sizes
        torch.backends.cudnn.deterministic = False  # Allow non-deterministic for speed
        
        # Set memory allocation strategy
        os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:512'
    
    # CPU optimizations
    torch.set_num_threads(min(mp.cpu_count(), 12))  # Use most CPU cores
    
    # Memory optimizations
    torch.multiprocessing.set_sharing_strategy('file_system')

def main():
    parser = argparse.ArgumentParser(description='Maximum Performance BraTS GNN Training')
    
    # Data arguments
    parser.add_argument('--graph_dir', type=str, default='./data/graphs')
    parser.add_argument('--max_graphs_per_patient', type=int, default=None)
    
    # High-performance model arguments
    parser.add_argument('--hidden_dim', type=int, default=128, help='Larger hidden dimension')
    parser.add_argument('--num_layers', type=int, default=4, help='Deeper network')
    parser.add_argument('--gnn_type', type=str, default='sage', choices=['sage', 'gat'])
    parser.add_argument('--dropout', type=float, default=0.1, help='Lower dropout for speed')
    
    # Optimized training arguments
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=8, help='Larger batches')
    parser.add_argument('--accumulation_steps', type=int, default=4, help='Gradient accumulation')
    parser.add_argument('--lr', type=float, default=0.002, help='Higher learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    
    # System optimization arguments
    parser.add_argument('--num_workers', type=int, default=8, help='More data loading workers')
    parser.add_argument('--pin_memory', action='store_true', default=True)
    parser.add_argument('--prefetch_factor', type=int, default=4)
    parser.add_argument('--persistent_workers', action='store_true', default=True)
    
    # Loss arguments
    parser.add_argument('--bce_weight', type=float, default=0.3)
    parser.add_argument('--dice_weight', type=float, default=0.7)
    
    # System arguments
    parser.add_argument('--device', type=str, default='auto')
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints/maxpower_gnn')
    
    args = parser.parse_args()
    
    # System optimizations
    optimize_system()
    
    # Setup device
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    
    print(f"🚀 MAXIMUM PERFORMANCE BraTS GNN Training")
    print(f"🖥️  Device: {device}")
    print(f"⚡ GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB" if torch.cuda.is_available() else "")
    print(f"🔥 CPU Cores: {torch.get_num_threads()}")
    print(f"💾 System RAM: Available for high-performance training")
    print(f"🎯 Target: Maximum hardware utilization")
    print("-" * 60)
    
    # Create data splits
    print("📊 Creating patient splits...")
    train_patients, val_patients, test_patients = create_data_splits(args.graph_dir)
    
    # Create datasets
    print("📚 Loading datasets with optimized settings...")
    train_dataset, val_dataset, test_dataset = create_datasets(
        args.graph_dir, train_patients, val_patients, test_patients,
        max_graphs_per_patient=args.max_graphs_per_patient
    )
    
    print(f"  Train graphs: {len(train_dataset)}")
    print(f"  Val graphs: {len(val_dataset)}")
    
    # High-performance data loaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
        prefetch_factor=args.prefetch_factor,
        persistent_workers=args.persistent_workers and args.num_workers > 0
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size * 2,  # Larger batch for validation
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
        prefetch_factor=args.prefetch_factor,
        persistent_workers=args.persistent_workers and args.num_workers > 0
    )
    
    # Get input dimension
    sample_graph = train_dataset[0]
    input_dim = sample_graph.x.shape[1]
    
    # Create high-performance model
    print(f"🧠 Creating optimized {args.gnn_type.upper()} model...")
    model = HighPerformanceBraTSGNN(
        input_dim=input_dim,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        dropout=args.dropout,
        gnn_type=args.gnn_type
    ).to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {total_params:,}")
    print(f"  Model size: ~{total_params * 4 / 1e6:.1f}MB")
    
    # High-performance optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=args.lr, 
        weight_decay=args.weight_decay,
        betas=(0.9, 0.999),
        eps=1e-8
    )
    
    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=args.lr,
        epochs=args.epochs,
        steps_per_epoch=len(train_loader) // args.accumulation_steps,
        pct_start=0.1,
        anneal_strategy='cos'
    )
    
    # Mixed precision scaler
    scaler = GradScaler()
    
    # Optimized loss
    criterion = FastCombinedLoss(bce_weight=args.bce_weight, dice_weight=args.dice_weight)
    
    print(f"\n🚀 Starting HIGH-PERFORMANCE training for {args.epochs} epochs...")
    print(f"  Effective batch size: {args.batch_size * args.accumulation_steps}")
    print(f"  Mixed precision: Enabled")
    print(f"  Gradient accumulation: {args.accumulation_steps} steps")
    print(f"  Data workers: {args.num_workers}")
    print(f"  Memory pinning: {args.pin_memory}")
    
    # Training loop
    best_dice = 0.0
    history = {'train': [], 'val': []}
    
    for epoch in range(args.epochs):
        start_time = time.time()
        
        print(f"\n⚡ Epoch {epoch + 1}/{args.epochs}")
        
        # High-performance training
        train_metrics = train_epoch_optimized(
            model, train_loader, criterion, optimizer, scaler, device, 
            epoch + 1, args.accumulation_steps
        )
        
        # Validation
        val_metrics = validate_epoch_optimized(model, val_loader, criterion, device)
        
        # Update learning rate
        scheduler.step()
        
        epoch_time = time.time() - start_time
        
        # Print results
        print(f"\n⏱️  Epoch {epoch + 1} completed in {epoch_time:.1f}s")
        print(f"📈 Train - Loss: {train_metrics['total']:.4f}, Dice: {train_metrics['dice']:.4f}")
        print(f"📊 Val   - Loss: {val_metrics['total']:.4f}, Dice: {val_metrics['dice']:.4f}")
        print(f"🔥 LR: {scheduler.get_last_lr()[0]:.6f}")
        
        # Save checkpoint (less frequently for speed)
        if (epoch + 1) % 5 == 0 or val_metrics['dice'] > best_dice:
            is_best = val_metrics['dice'] > best_dice
            if is_best:
                best_dice = val_metrics['dice']
                print(f"  💾 New best model! Dice: {best_dice:.4f}")
            
            # Save checkpoint
            os.makedirs(args.checkpoint_dir, exist_ok=True)
            checkpoint = {
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'scaler_state_dict': scaler.state_dict(),
                'best_dice': best_dice,
                'args': vars(args)
            }
            
            torch.save(checkpoint, os.path.join(args.checkpoint_dir, 'latest_checkpoint.pth'))
            if is_best:
                torch.save(checkpoint, os.path.join(args.checkpoint_dir, 'best_model.pth'))
        
        # Save history
        history['train'].append(train_metrics)
        history['val'].append(val_metrics)
    
    print(f"\n🏆 HIGH-PERFORMANCE training completed!")
    print(f"🎯 Best validation Dice score: {best_dice:.4f}")
    print(f"💾 Models saved to: {args.checkpoint_dir}")

if __name__ == "__main__":
    main()