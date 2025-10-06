# src/train_gnn.py - Main GNN Training Script
import os
import torch
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
import argparse
import time
import json
from pathlib import Path
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score

# Import our modules
from dataset import create_data_splits, create_datasets
from train import TumorSegmentationGNN, CombinedLoss

def train_epoch(model, loader, criterion, optimizer, device, epoch):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    total_ce_loss = 0
    total_dice_loss = 0
    total_consistency_loss = 0
    num_batches = 0
    
    all_preds = []
    all_targets = []
    
    for batch_idx, batch in enumerate(loader):
        batch = batch.to(device)
        optimizer.zero_grad()
        
        # Forward pass
        logits, embeddings = model(batch)
        
        # Prepare targets and slice mask (dummy for now)
        targets = batch.y.float()  # Assuming y contains tumor labels
        slice_mask = torch.ones_like(targets)  # Dummy slice mask
        
        # Calculate loss
        loss_output = criterion(logits, embeddings, targets, slice_mask)
        if len(loss_output) == 4:
            loss, ce_loss, dice_loss, consistency_loss = loss_output
        else:
            loss, ce_loss, dice_loss = loss_output
            consistency_loss = torch.tensor(0.0)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Accumulate losses
        total_loss += loss.item()
        total_ce_loss += ce_loss.item()
        total_dice_loss += dice_loss.item()
        total_consistency_loss += consistency_loss.item()
        num_batches += 1
        
        # Collect predictions for metrics
        with torch.no_grad():
            preds = torch.sigmoid(logits) > 0.5
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
        
        # Print progress
        if batch_idx % 10 == 0:
            print(f'Epoch {epoch}, Batch {batch_idx}/{len(loader)}, Loss: {loss.item():.4f}')
    
    # Calculate metrics
    accuracy = accuracy_score(all_targets, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(all_targets, all_preds, average='binary', zero_division=0)
    
    avg_loss = total_loss / num_batches
    avg_ce_loss = total_ce_loss / num_batches
    avg_dice_loss = total_dice_loss / num_batches
    avg_consistency_loss = total_consistency_loss / num_batches
    
    return {
        'loss': avg_loss,
        'ce_loss': avg_ce_loss,
        'dice_loss': avg_dice_loss,
        'consistency_loss': avg_consistency_loss,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }

def validate_epoch(model, loader, criterion, device):
    """Validate for one epoch"""
    model.eval()
    total_loss = 0
    num_batches = 0
    
    all_preds = []
    all_targets = []
    all_probs = []
    
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            
            # Forward pass
            logits, embeddings = model(batch)
            
            # Prepare targets and slice mask
            targets = batch.y.float()
            slice_mask = torch.ones_like(targets)
            
            # Calculate loss
            loss_output = criterion(logits, embeddings, targets, slice_mask)
            loss = loss_output[0]
            
            total_loss += loss.item()
            num_batches += 1
            
            # Collect predictions
            probs = torch.sigmoid(logits)
            preds = probs > 0.5
            
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
    
    # Calculate metrics
    accuracy = accuracy_score(all_targets, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(all_targets, all_preds, average='binary', zero_division=0)
    
    # Calculate AUC if possible
    try:
        auc = roc_auc_score(all_targets, all_probs)
    except:
        auc = 0.0
    
    avg_loss = total_loss / num_batches
    
    return {
        'loss': avg_loss,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'auc': auc
    }

def save_checkpoint(model, optimizer, epoch, train_metrics, val_metrics, checkpoint_dir, is_best=False):
    """Save model checkpoint"""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'train_metrics': train_metrics,
        'val_metrics': val_metrics,
    }
    
    # Save latest checkpoint
    checkpoint_path = os.path.join(checkpoint_dir, 'latest_checkpoint.pth')
    torch.save(checkpoint, checkpoint_path)
    
    # Save best checkpoint
    if is_best:
        best_path = os.path.join(checkpoint_dir, 'best_checkpoint.pth')
        torch.save(checkpoint, best_path)
        print(f"💾 New best model saved with validation F1: {val_metrics['f1']:.4f}")

def main():
    parser = argparse.ArgumentParser(description='Train BraTS GNN')
    parser.add_argument('--graph_dir', type=str, default='./data/graphs', help='Graph directory')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--hidden_dim', type=int, default=128, help='Hidden dimension')
    parser.add_argument('--gnn_type', type=str, default='sage', choices=['sage', 'gat'], help='GNN type')
    parser.add_argument('--num_layers', type=int, default=3, help='Number of GNN layers')
    parser.add_argument('--dropout', type=float, default=0.2, help='Dropout rate')
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints', help='Checkpoint directory')
    parser.add_argument('--max_graphs_per_patient', type=int, default=None, help='Limit graphs per patient for debugging')
    parser.add_argument('--device', type=str, default='auto', help='Device (cuda/cpu/auto)')
    parser.add_argument('--num_workers', type=int, default=4, help='DataLoader workers')
    
    args = parser.parse_args()
    
    # Setup device
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    print(f"🖥️  Using device: {device}")
    
    # Create checkpoint directory
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    
    # Create data splits
    print("📊 Creating data splits...")
    train_patients, val_patients, test_patients = create_data_splits(args.graph_dir)
    
    # Create datasets
    print("📚 Creating datasets...")
    train_dataset, val_dataset, test_dataset = create_datasets(
        args.graph_dir, train_patients, val_patients, test_patients,
        max_graphs_per_patient=args.max_graphs_per_patient
    )
    
    print(f"📊 Dataset sizes: Train={len(train_dataset)}, Val={len(val_dataset)}, Test={len(test_dataset)}")
    
    if len(train_dataset) == 0:
        print("❌ No training data found!")
        return
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    
    # Get feature dimension from a sample
    sample_graph = train_dataset[0]
    feature_dim = sample_graph.x.shape[1]
    print(f"📊 Feature dimension: {feature_dim}")
    
    # Create model
    print(f"🧠 Creating {args.gnn_type.upper()} model...")
    model = TumorSegmentationGNN(
        in_channels=feature_dim,
        hidden_channels=args.hidden_dim,
        gnn_type=args.gnn_type,
        num_layers=args.num_layers,
        dropout=args.dropout
    ).to(device)
    
    # Create optimizer and criterion
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = CombinedLoss()
    
    print(f"🚀 Starting training for {args.epochs} epochs...")
    print(f"   Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Training loop
    best_val_f1 = 0.0
    train_history = []
    val_history = []
    
    for epoch in range(args.epochs):
        start_time = time.time()
        
        # Train
        train_metrics = train_epoch(model, train_loader, criterion, optimizer, device, epoch + 1)
        
        # Validate
        val_metrics = validate_epoch(model, val_loader, criterion, device)
        
        epoch_time = time.time() - start_time
        
        # Print results
        print(f"\n{'='*60}")
        print(f"Epoch {epoch + 1}/{args.epochs} ({epoch_time:.1f}s)")
        print(f"Train - Loss: {train_metrics['loss']:.4f}, Acc: {train_metrics['accuracy']:.4f}, F1: {train_metrics['f1']:.4f}")
        print(f"Val   - Loss: {val_metrics['loss']:.4f}, Acc: {val_metrics['accuracy']:.4f}, F1: {val_metrics['f1']:.4f}, AUC: {val_metrics['auc']:.4f}")
        
        # Save checkpoint
        is_best = val_metrics['f1'] > best_val_f1
        if is_best:
            best_val_f1 = val_metrics['f1']
        
        save_checkpoint(model, optimizer, epoch + 1, train_metrics, val_metrics, args.checkpoint_dir, is_best)
        
        # Save history
        train_history.append(train_metrics)
        val_history.append(val_metrics)
        
        # Save training history
        history = {'train': train_history, 'val': val_history, 'args': vars(args)}
        with open(os.path.join(args.checkpoint_dir, 'training_history.json'), 'w') as f:
            json.dump(history, f, indent=2)
    
    print(f"\n🎉 Training completed! Best validation F1: {best_val_f1:.4f}")
    
    # Final test evaluation
    print("\n🧪 Running final test evaluation...")
    test_metrics = validate_epoch(model, test_loader, criterion, device)
    print(f"Test Results - Loss: {test_metrics['loss']:.4f}, Acc: {test_metrics['accuracy']:.4f}, F1: {test_metrics['f1']:.4f}, AUC: {test_metrics['auc']:.4f}")
    
    # Save final results
    final_results = {
        'best_val_f1': best_val_f1,
        'test_metrics': test_metrics,
        'args': vars(args)
    }
    with open(os.path.join(args.checkpoint_dir, 'final_results.json'), 'w') as f:
        json.dump(final_results, f, indent=2)

if __name__ == "__main__":
    main()