# src/train_brats_gnn.py - Clean BraTS GNN Training from Scratch
"""
BraTS GNN Training Script - Designed for Brain Tumor Segmentation

Key Design Principles:
1. Node-level tumor classification (each superpixel predicts tumor/non-tumor)
2. Handle variable graph sizes gracefully
3. Medical-focused metrics (Dice, sensitivity, specificity)
4. Robust to class imbalance (tumor regions are sparse)
5. Clear separation of concerns
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from torch_geometric.nn import SAGEConv, GATConv, global_mean_pool, BatchNorm
import numpy as np
import argparse
import time
import json
from pathlib import Path
from typing import Dict, Tuple, List
import warnings
warnings.filterwarnings('ignore')

# Import our data loading
from dataset import create_data_splits, create_datasets

class BraTSGNN(nn.Module):
    """
    Clean GNN for BraTS tumor segmentation
    
    Design choices:
    - Node-level prediction (each superpixel classified as tumor/non-tumor)
    - Multiple GNN layers for spatial context
    - Batch normalization for stable training
    - Dropout for regularization
    """
    
    def __init__(self, 
                 input_dim: int = 12,
                 hidden_dim: int = 64, 
                 num_layers: int = 3,
                 dropout: float = 0.2,
                 gnn_type: str = 'sage'):
        super().__init__()
        
        self.num_layers = num_layers
        self.dropout = dropout
        
        # Build GNN layers
        self.convs = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        
        # Input layer
        if gnn_type == 'sage':
            self.convs.append(SAGEConv(input_dim, hidden_dim))
        elif gnn_type == 'gat':
            self.convs.append(GATConv(input_dim, hidden_dim, heads=4, concat=False))
        
        self.batch_norms.append(BatchNorm(hidden_dim))
        
        # Hidden layers
        for _ in range(num_layers - 2):
            if gnn_type == 'sage':
                self.convs.append(SAGEConv(hidden_dim, hidden_dim))
            elif gnn_type == 'gat':
                self.convs.append(GATConv(hidden_dim, hidden_dim, heads=4, concat=False))
            self.batch_norms.append(BatchNorm(hidden_dim))
        
        # Output layer
        if gnn_type == 'sage':
            self.convs.append(SAGEConv(hidden_dim, hidden_dim))
        elif gnn_type == 'gat':
            self.convs.append(GATConv(hidden_dim, hidden_dim, heads=1))
        self.batch_norms.append(BatchNorm(hidden_dim))
        
        # Classifier head
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1)  # Binary classification per node
        )
        
    def forward(self, x, edge_index, batch=None):
        """
        Forward pass
        
        Args:
            x: Node features [num_nodes, input_dim]
            edge_index: Graph edges [2, num_edges]
            batch: Batch assignment for nodes [num_nodes]
            
        Returns:
            logits: Node-level predictions [num_nodes, 1]
        """
        # Apply GNN layers
        for i, (conv, bn) in enumerate(zip(self.convs, self.batch_norms)):
            x = conv(x, edge_index)
            x = bn(x)
            
            if i < len(self.convs) - 1:  # No activation after last layer
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
        
        # Node-level classification
        logits = self.classifier(x).squeeze(-1)
        
        return logits

class DiceLoss(nn.Module):
    """
    Dice Loss for medical segmentation
    Handles class imbalance better than BCE alone
    """
    
    def __init__(self, smooth: float = 1.0):
        super().__init__()
        self.smooth = smooth
    
    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # Apply sigmoid to logits
        predictions = torch.sigmoid(predictions)
        
        # Flatten
        predictions = predictions.view(-1)
        targets = targets.view(-1)
        
        # Dice coefficient
        intersection = (predictions * targets).sum()
        total = predictions.sum() + targets.sum()
        
        dice = (2.0 * intersection + self.smooth) / (total + self.smooth)
        
        return 1 - dice

class CombinedLoss(nn.Module):
    """
    Combined BCE + Dice Loss
    BCE handles individual pixel accuracy
    Dice handles region overlap (medical standard)
    """
    
    def __init__(self, bce_weight: float = 0.5, dice_weight: float = 0.5):
        super().__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        
        self.bce_loss = nn.BCEWithLogitsLoss()
        self.dice_loss = DiceLoss()
    
    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
        bce = self.bce_loss(logits, targets)
        dice = self.dice_loss(logits, targets)
        
        total_loss = self.bce_weight * bce + self.dice_weight * dice
        
        loss_dict = {
            'total': total_loss,
            'bce': bce,
            'dice': dice
        }
        
        return total_loss, loss_dict

def calculate_metrics(predictions: np.ndarray, targets: np.ndarray) -> Dict[str, float]:
    """
    Calculate medical segmentation metrics
    
    Focus on metrics relevant to tumor segmentation:
    - Dice Score (most important for medical segmentation)
    - Sensitivity (recall for tumor detection)
    - Specificity (true negative rate)
    - Precision
    - Accuracy
    """
    # Convert to binary predictions
    pred_binary = (predictions > 0.5).astype(int)
    targets_binary = targets.astype(int)
    
    # Handle edge case of no positive samples
    if targets_binary.sum() == 0:
        if pred_binary.sum() == 0:
            return {'dice': 1.0, 'sensitivity': 1.0, 'specificity': 1.0, 'precision': 1.0, 'accuracy': 1.0}
        else:
            return {'dice': 0.0, 'sensitivity': 0.0, 'specificity': 0.0, 'precision': 0.0, 'accuracy': 0.0}
    
    # Calculate confusion matrix components
    tp = ((pred_binary == 1) & (targets_binary == 1)).sum()
    tn = ((pred_binary == 0) & (targets_binary == 0)).sum()
    fp = ((pred_binary == 1) & (targets_binary == 0)).sum()
    fn = ((pred_binary == 0) & (targets_binary == 1)).sum()
    
    # Calculate metrics
    dice = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0.0
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    
    return {
        'dice': float(dice),
        'sensitivity': float(sensitivity),
        'specificity': float(specificity),
        'precision': float(precision),
        'accuracy': float(accuracy)
    }

def train_epoch(model: BraTSGNN, 
                loader: DataLoader, 
                criterion: CombinedLoss,
                optimizer: torch.optim.Optimizer,
                device: torch.device,
                epoch: int) -> Dict[str, float]:
    """Train for one epoch"""
    model.train()
    
    total_losses = {'total': 0.0, 'bce': 0.0, 'dice': 0.0}
    all_predictions = []
    all_targets = []
    num_batches = 0
    
    for batch_idx, batch in enumerate(loader):
        try:
            batch = batch.to(device)
            optimizer.zero_grad()
            
            # Forward pass
            logits = model(batch.x, batch.edge_index, batch.batch)
            targets = batch.y.float()
            
            # Calculate loss
            loss, loss_dict = criterion(logits, targets)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            # Accumulate losses
            for key in total_losses:
                total_losses[key] += loss_dict[key].item()
            num_batches += 1
            
            # Collect predictions for metrics
            with torch.no_grad():
                preds = torch.sigmoid(logits).cpu().numpy()
                all_predictions.extend(preds)
                all_targets.extend(targets.cpu().numpy())
            
            # Progress logging
            if batch_idx % 20 == 0:
                print(f'  Batch {batch_idx}/{len(loader)} - Loss: {loss.item():.4f}')
                
        except Exception as e:
            print(f"  ⚠️  Skipping batch {batch_idx}: {e}")
            continue
    
    # Calculate average losses
    avg_losses = {key: val / max(num_batches, 1) for key, val in total_losses.items()}
    
    # Calculate metrics
    if len(all_predictions) > 0:
        metrics = calculate_metrics(np.array(all_predictions), np.array(all_targets))
    else:
        metrics = {'dice': 0.0, 'sensitivity': 0.0, 'specificity': 0.0, 'precision': 0.0, 'accuracy': 0.0}
    
    # Combine losses and metrics
    result = {**avg_losses, **metrics}
    result['num_batches'] = num_batches
    
    return result

def validate_epoch(model: BraTSGNN,
                  loader: DataLoader,
                  criterion: CombinedLoss,
                  device: torch.device) -> Dict[str, float]:
    """Validate for one epoch"""
    model.eval()
    
    total_losses = {'total': 0.0, 'bce': 0.0, 'dice': 0.0}
    all_predictions = []
    all_targets = []
    num_batches = 0
    
    with torch.no_grad():
        for batch in loader:
            try:
                batch = batch.to(device)
                
                # Forward pass
                logits = model(batch.x, batch.edge_index, batch.batch)
                targets = batch.y.float()
                
                # Calculate loss
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
                print(f"  ⚠️  Skipping validation batch: {e}")
                continue
    
    # Calculate average losses
    avg_losses = {key: val / max(num_batches, 1) for key, val in total_losses.items()}
    
    # Calculate metrics
    if len(all_predictions) > 0:
        metrics = calculate_metrics(np.array(all_predictions), np.array(all_targets))
    else:
        metrics = {'dice': 0.0, 'sensitivity': 0.0, 'specificity': 0.0, 'precision': 0.0, 'accuracy': 0.0}
    
    # Combine results
    result = {**avg_losses, **metrics}
    result['num_batches'] = num_batches
    
    return result

def save_checkpoint(model: BraTSGNN,
                   optimizer: torch.optim.Optimizer,
                   epoch: int,
                   train_metrics: Dict,
                   val_metrics: Dict,
                   checkpoint_dir: str,
                   is_best: bool = False):
    """Save model checkpoint"""
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'train_metrics': train_metrics,
        'val_metrics': val_metrics,
    }
    
    # Always save latest
    latest_path = os.path.join(checkpoint_dir, 'latest_checkpoint.pth')
    torch.save(checkpoint, latest_path)
    
    # Save best model
    if is_best:
        best_path = os.path.join(checkpoint_dir, 'best_model.pth')
        torch.save(checkpoint, best_path)
        print(f"  💾 New best model saved! Dice: {val_metrics['dice']:.4f}")

def main():
    parser = argparse.ArgumentParser(description='BraTS GNN Training - Clean Implementation')
    
    # Data arguments
    parser.add_argument('--graph_dir', type=str, default='./data/graphs', 
                       help='Directory containing preprocessed graphs')
    parser.add_argument('--max_graphs_per_patient', type=int, default=None,
                       help='Limit graphs per patient (for debugging)')
    
    # Model arguments
    parser.add_argument('--hidden_dim', type=int, default=64,
                       help='Hidden dimension of GNN layers')
    parser.add_argument('--num_layers', type=int, default=3,
                       help='Number of GNN layers')
    parser.add_argument('--gnn_type', type=str, default='sage', choices=['sage', 'gat'],
                       help='Type of GNN to use')
    parser.add_argument('--dropout', type=float, default=0.2,
                       help='Dropout rate')
    
    # Training arguments
    parser.add_argument('--epochs', type=int, default=50,
                       help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=1,
                       help='Batch size (use 1 to avoid dimension mismatch)')
    parser.add_argument('--lr', type=float, default=0.001,
                       help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-5,
                       help='Weight decay for regularization')
    
    # Loss arguments
    parser.add_argument('--bce_weight', type=float, default=0.3,
                       help='Weight for BCE loss component')
    parser.add_argument('--dice_weight', type=float, default=0.7,
                       help='Weight for Dice loss component')
    
    # System arguments
    parser.add_argument('--device', type=str, default='auto',
                       help='Device to use (cuda/cpu/auto)')
    parser.add_argument('--num_workers', type=int, default=0,
                       help='Number of data loading workers')
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints/brats_gnn',
                       help='Directory to save checkpoints')
    
    args = parser.parse_args()
    
    # Setup device
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    
    print(f"🧠 BraTS GNN Training")
    print(f"🖥️  Device: {device}")
    print(f"📊 Graph directory: {args.graph_dir}")
    print(f"🎯 Goal: Node-level tumor classification")
    print("-" * 50)
    
    # Create data splits
    print("📊 Creating patient splits...")
    train_patients, val_patients, test_patients = create_data_splits(args.graph_dir)
    print(f"  Train: {len(train_patients)} patients")
    print(f"  Val: {len(val_patients)} patients") 
    print(f"  Test: {len(test_patients)} patients")
    
    # Create datasets
    print("\n📚 Loading datasets...")
    train_dataset, val_dataset, test_dataset = create_datasets(
        args.graph_dir, train_patients, val_patients, test_patients,
        max_graphs_per_patient=args.max_graphs_per_patient
    )
    print(f"  Train graphs: {len(train_dataset)}")
    print(f"  Val graphs: {len(val_dataset)}")
    print(f"  Test graphs: {len(test_dataset)}")
    
    if len(train_dataset) == 0:
        print("❌ No training data found!")
        return
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, 
                             shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size,
                           shuffle=False, num_workers=args.num_workers)
    
    # Get input dimension
    sample_graph = train_dataset[0]
    input_dim = sample_graph.x.shape[1]
    print(f"📊 Input feature dimension: {input_dim}")
    
    # Create model
    print(f"\n🧠 Creating {args.gnn_type.upper()} model...")
    model = BraTSGNN(
        input_dim=input_dim,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        dropout=args.dropout,
        gnn_type=args.gnn_type
    ).to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {total_params:,}")
    
    # Create optimizer and loss
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = CombinedLoss(bce_weight=args.bce_weight, dice_weight=args.dice_weight)
    
    print(f"\n🚀 Starting training for {args.epochs} epochs...")
    print(f"  Loss: BCE({args.bce_weight}) + Dice({args.dice_weight})")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Learning rate: {args.lr}")
    
    # Training loop
    best_dice = 0.0
    history = {'train': [], 'val': []}
    
    for epoch in range(args.epochs):
        start_time = time.time()
        
        print(f"\n📚 Epoch {epoch + 1}/{args.epochs}")
        
        # Train
        train_metrics = train_epoch(model, train_loader, criterion, optimizer, device, epoch + 1)
        
        # Validate
        val_metrics = validate_epoch(model, val_loader, criterion, device)
        
        epoch_time = time.time() - start_time
        
        # Print results
        print(f"\n⏱️  Epoch {epoch + 1} completed in {epoch_time:.1f}s")
        print(f"📈 Train - Loss: {train_metrics['total']:.4f}, Dice: {train_metrics['dice']:.4f}, "
              f"Sensitivity: {train_metrics['sensitivity']:.4f}")
        print(f"📊 Val   - Loss: {val_metrics['total']:.4f}, Dice: {val_metrics['dice']:.4f}, "
              f"Sensitivity: {val_metrics['sensitivity']:.4f}")
        
        # Save checkpoint
        is_best = val_metrics['dice'] > best_dice
        if is_best:
            best_dice = val_metrics['dice']
        
        save_checkpoint(model, optimizer, epoch + 1, train_metrics, val_metrics, 
                       args.checkpoint_dir, is_best)
        
        # Save history
        history['train'].append(train_metrics)
        history['val'].append(val_metrics)
        
        # Save training history
        history_path = os.path.join(args.checkpoint_dir, 'training_history.json')
        os.makedirs(args.checkpoint_dir, exist_ok=True)
        with open(history_path, 'w') as f:
            json.dump(history, f, indent=2, default=str)
    
    print(f"\n🎉 Training completed!")
    print(f"🏆 Best validation Dice score: {best_dice:.4f}")
    
    # Save final results
    final_results = {
        'best_dice': best_dice,
        'final_train_metrics': history['train'][-1] if history['train'] else {},
        'final_val_metrics': history['val'][-1] if history['val'] else {},
        'args': vars(args)
    }
    
    results_path = os.path.join(args.checkpoint_dir, 'final_results.json')
    with open(results_path, 'w') as f:
        json.dump(final_results, f, indent=2, default=str)
    
    print(f"💾 Results saved to {results_path}")

if __name__ == "__main__":
    main()