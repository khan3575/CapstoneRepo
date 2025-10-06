# src/train_robust_maxpower.py - Maximum Performance with Dimension Safety
"""
Robust Maximum Performance BraTS GNN Training

Fixes:
1. Handles variable feature dimensions (12 vs 14)
2. Filters incompatible graphs
3. Maximum hardware utilization
4. Robust error handling
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from torch_geometric.nn import SAGEConv, BatchNorm
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

class RobustBraTSDataset:
    """
    Wrapper to filter graphs with consistent feature dimensions
    """
    def __init__(self, base_dataset, target_feature_dim=12):
        self.base_dataset = base_dataset
        self.target_feature_dim = target_feature_dim
        
        # Filter compatible graphs
        print(f"🔍 Filtering graphs for {target_feature_dim} features...")
        self.valid_indices = []
        
        for i in range(len(base_dataset)):
            try:
                graph = base_dataset[i]
                if graph.x.shape[1] == target_feature_dim:
                    self.valid_indices.append(i)
                else:
                    print(f"  ⚠️  Skipping graph {i}: {graph.x.shape[1]} features (expected {target_feature_dim})")
            except Exception as e:
                print(f"  ⚠️  Skipping graph {i}: {e}")
        
        print(f"✅ Filtered dataset: {len(self.valid_indices)}/{len(base_dataset)} graphs valid")
    
    def __len__(self):
        return len(self.valid_indices)
    
    def __getitem__(self, idx):
        real_idx = self.valid_indices[idx]
        return self.base_dataset[real_idx]

class MaxPowerBraTSGNN(nn.Module):
    """Ultra-optimized GNN for maximum performance"""
    
    def __init__(self, 
                 input_dim: int = 12,
                 hidden_dim: int = 256,
                 num_layers: int = 5,
                 dropout: float = 0.1):
        super().__init__()
        
        self.num_layers = num_layers
        self.dropout = dropout
        
        # Ultra-efficient SAGE layers
        self.convs = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        
        # Input layer
        self.convs.append(SAGEConv(input_dim, hidden_dim, normalize=True))
        self.batch_norms.append(BatchNorm(hidden_dim))
        
        # Hidden layers
        for _ in range(num_layers - 2):
            self.convs.append(SAGEConv(hidden_dim, hidden_dim, normalize=True))
            self.batch_norms.append(BatchNorm(hidden_dim))
        
        # Output layer
        output_dim = hidden_dim // 2
        self.convs.append(SAGEConv(hidden_dim, output_dim, normalize=True))
        self.batch_norms.append(BatchNorm(output_dim))
        
        # Efficient classifier
        self.classifier = nn.Sequential(
            nn.Linear(output_dim, output_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(output_dim // 2, 1)
        )
        
        # Initialize for fast convergence
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.kaiming_uniform_(module.weight, nonlinearity='relu')
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
    
    def forward(self, x, edge_index, batch=None):
        # Efficient forward pass
        for i, (conv, bn) in enumerate(zip(self.convs, self.batch_norms)):
            x = conv(x, edge_index)
            x = bn(x)
            
            if i < len(self.convs) - 1:
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
        
        logits = self.classifier(x).squeeze(-1)
        return logits

class UltraFastLoss(nn.Module):
    """Optimized loss for maximum throughput"""
    
    def __init__(self, bce_weight: float = 0.4, dice_weight: float = 0.6):
        super().__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.bce_loss = nn.BCEWithLogitsLoss()
    
    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
        # Fast BCE
        bce = self.bce_loss(logits, targets)
        
        # Fast Dice
        probs = torch.sigmoid(logits)
        intersection = torch.sum(probs * targets)
        total = torch.sum(probs) + torch.sum(targets)
        dice_loss = 1 - (2 * intersection + 1) / (total + 1)
        
        total_loss = self.bce_weight * bce + self.dice_weight * dice_loss
        
        return total_loss, {'total': total_loss, 'bce': bce, 'dice': dice_loss}

def ultra_fast_metrics(predictions: np.ndarray, targets: np.ndarray) -> Dict[str, float]:
    """Vectorized metrics calculation"""
    pred_binary = predictions > 0.5
    targets_binary = targets > 0.5
    
    if not targets_binary.any():
        return {'dice': 1.0 if not pred_binary.any() else 0.0, 'accuracy': 1.0}
    
    tp = (pred_binary & targets_binary).sum()
    fp = (pred_binary & ~targets_binary).sum()
    fn = (~pred_binary & targets_binary).sum()
    
    dice = (2 * tp) / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0.0
    accuracy = (pred_binary == targets_binary).mean()
    
    return {'dice': float(dice), 'accuracy': float(accuracy)}

def train_epoch_ultra(model, loader, criterion, optimizer, scaler, device, epoch, accumulation_steps):
    """Ultra-optimized training epoch"""
    model.train()
    
    total_loss = 0.0
    all_predictions = []
    all_targets = []
    num_batches = 0
    successful_batches = 0
    
    optimizer.zero_grad()
    
    for batch_idx, batch in enumerate(loader):
        try:
            batch = batch.to(device, non_blocking=True)
            
            with autocast():
                logits = model(batch.x, batch.edge_index, batch.batch)
                targets = batch.y.float()
                loss, loss_dict = criterion(logits, targets)
                loss = loss / accumulation_steps
            
            scaler.scale(loss).backward()
            
            if (batch_idx + 1) % accumulation_steps == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                successful_batches += 1
            
            total_loss += loss_dict['total'].item()
            num_batches += 1
            
            # Sample predictions for metrics (every 8th batch for speed)
            if batch_idx % 8 == 0:
                with torch.no_grad():
                    preds = torch.sigmoid(logits).cpu().numpy()
                    all_predictions.extend(preds[::4])  # Subsample for speed
                    all_targets.extend(targets.cpu().numpy()[::4])
            
            if batch_idx % 50 == 0:
                print(f'  ⚡ Batch {batch_idx}/{len(loader)} - Loss: {loss.item() * accumulation_steps:.4f} - Success: {successful_batches}')
                
        except Exception as e:
            print(f"  ⚠️  Skip batch {batch_idx}: {str(e)[:50]}...")
            continue
    
    # Final gradient step
    if num_batches % accumulation_steps != 0:
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()
    
    avg_loss = total_loss / max(num_batches, 1)
    
    if len(all_predictions) > 0:
        metrics = ultra_fast_metrics(np.array(all_predictions), np.array(all_targets))
    else:
        metrics = {'dice': 0.0, 'accuracy': 0.0}
    
    return {
        'total': avg_loss,
        'dice': metrics['dice'],
        'accuracy': metrics['accuracy'],
        'successful_batches': successful_batches,
        'total_batches': num_batches
    }

def validate_epoch_ultra(model, loader, criterion, device):
    """Ultra-fast validation"""
    model.eval()
    
    total_loss = 0.0
    all_predictions = []
    all_targets = []
    num_batches = 0
    
    with torch.no_grad():
        for batch in loader:
            try:
                batch = batch.to(device, non_blocking=True)
                
                with autocast():
                    logits = model(batch.x, batch.edge_index, batch.batch)
                    targets = batch.y.float()
                    loss, loss_dict = criterion(logits, targets)
                
                total_loss += loss_dict['total'].item()
                num_batches += 1
                
                # Collect predictions
                preds = torch.sigmoid(logits).cpu().numpy()
                all_predictions.extend(preds)
                all_targets.extend(targets.cpu().numpy())
                
            except Exception:
                continue
    
    avg_loss = total_loss / max(num_batches, 1)
    
    if len(all_predictions) > 0:
        metrics = ultra_fast_metrics(np.array(all_predictions), np.array(all_targets))
    else:
        metrics = {'dice': 0.0, 'accuracy': 0.0}
    
    return {
        'total': avg_loss,
        'dice': metrics['dice'],
        'accuracy': metrics['accuracy'],
        'num_batches': num_batches
    }

def optimize_system_ultra():
    """Maximum system optimizations"""
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
        os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:512'
    
    torch.set_num_threads(min(mp.cpu_count(), 12))
    torch.multiprocessing.set_sharing_strategy('file_system')

def main():
    parser = argparse.ArgumentParser(description='ULTRA Maximum Performance BraTS GNN')
    
    # Data
    parser.add_argument('--graph_dir', type=str, default='./data/graphs')
    parser.add_argument('--max_graphs_per_patient', type=int, default=None)
    
    # Ultra model
    parser.add_argument('--hidden_dim', type=int, default=256)
    parser.add_argument('--num_layers', type=int, default=5)
    parser.add_argument('--dropout', type=float, default=0.1)
    
    # Ultra training
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--accumulation_steps', type=int, default=32)
    parser.add_argument('--lr', type=float, default=0.003)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    
    # Ultra system
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--prefetch_factor', type=int, default=8)
    parser.add_argument('--device', type=str, default='auto')
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints/ultra_maxpower')
    
    args = parser.parse_args()
    
    # Ultra optimizations
    optimize_system_ultra()
    
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    
    print(f"🚀 ULTRA MAXIMUM PERFORMANCE BraTS GNN")
    print(f"⚡ Device: {device}")
    if torch.cuda.is_available():
        print(f"🔥 GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB")
    print(f"💪 CPU Threads: {torch.get_num_threads()}")
    print(f"🎯 Effective Batch Size: {args.batch_size * args.accumulation_steps}")
    print("=" * 60)
    
    # Create data splits
    print("📊 Creating robust data splits...")
    train_patients, val_patients, test_patients = create_data_splits(args.graph_dir)
    
    # Create base datasets
    train_dataset_base, val_dataset_base, test_dataset_base = create_datasets(
        args.graph_dir, train_patients, val_patients, test_patients,
        max_graphs_per_patient=args.max_graphs_per_patient
    )
    
    # Create robust filtered datasets
    train_dataset = RobustBraTSDataset(train_dataset_base, target_feature_dim=12)
    val_dataset = RobustBraTSDataset(val_dataset_base, target_feature_dim=12)
    
    print(f"📊 Final dataset sizes:")
    print(f"  Train: {len(train_dataset)} graphs")
    print(f"  Val: {len(val_dataset)} graphs")
    
    if len(train_dataset) == 0:
        print("❌ No valid training data!")
        return
    
    # Ultra-fast data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        prefetch_factor=args.prefetch_factor,
        persistent_workers=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size * 2,
        shuffle=False,
        num_workers=args.num_workers//2,
        pin_memory=True,
        prefetch_factor=args.prefetch_factor,
        persistent_workers=True
    )
    
    # Ultra model
    print(f"🧠 Creating ULTRA model (Hidden: {args.hidden_dim}, Layers: {args.num_layers})...")
    model = MaxPowerBraTSGNN(
        input_dim=12,  # Fixed to 12 features
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        dropout=args.dropout
    ).to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {total_params:,} (~{total_params * 4 / 1e6:.1f}MB)")
    
    # Ultra optimizer & scheduler
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
        betas=(0.9, 0.999)
    )
    
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=args.lr,
        epochs=args.epochs,
        steps_per_epoch=len(train_loader) // args.accumulation_steps,
        pct_start=0.1
    )
    
    scaler = GradScaler()
    criterion = UltraFastLoss()
    
    print(f"\n🚀 ULTRA TRAINING START!")
    print(f"  Mixed Precision: ✅")
    print(f"  Gradient Accumulation: {args.accumulation_steps}")
    print(f"  Data Workers: {args.num_workers}")
    print(f"  Prefetch Factor: {args.prefetch_factor}")
    
    # Ultra training loop
    best_dice = 0.0
    
    for epoch in range(args.epochs):
        start_time = time.time()
        
        print(f"\n⚡ EPOCH {epoch + 1}/{args.epochs}")
        
        # Ultra training
        train_metrics = train_epoch_ultra(
            model, train_loader, criterion, optimizer, scaler, device,
            epoch + 1, args.accumulation_steps
        )
        
        # Ultra validation
        val_metrics = validate_epoch_ultra(model, val_loader, criterion, device)
        
        scheduler.step()
        epoch_time = time.time() - start_time
        
        # Results
        print(f"\n⏱️  Epoch {epoch + 1} - {epoch_time:.1f}s")
        print(f"🔥 Train - Loss: {train_metrics['total']:.4f}, Dice: {train_metrics['dice']:.4f}")
        print(f"✨ Val   - Loss: {val_metrics['total']:.4f}, Dice: {val_metrics['dice']:.4f}")
        print(f"📊 Batches: {train_metrics['successful_batches']}/{train_metrics['total_batches']} successful")
        print(f"📈 LR: {scheduler.get_last_lr()[0]:.6f}")
        
        # Save best model
        if val_metrics['dice'] > best_dice:
            best_dice = val_metrics['dice']
            print(f"🏆 NEW BEST! Dice: {best_dice:.4f}")
            
            os.makedirs(args.checkpoint_dir, exist_ok=True)
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_dice': best_dice,
                'args': vars(args)
            }, os.path.join(args.checkpoint_dir, 'ultra_best_model.pth'))
    
    print(f"\n🎉 ULTRA TRAINING COMPLETE!")
    print(f"🏆 Best Dice Score: {best_dice:.4f}")
    print(f"💾 Model saved to: {args.checkpoint_dir}")

if __name__ == "__main__":
    main()