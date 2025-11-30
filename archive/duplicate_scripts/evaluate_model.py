#!/usr/bin/env python3
"""
Evaluate the trained BraTS GNN model on test set
"""

import sys
import os
sys.path.append('./src')

import torch
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from dataset import create_data_splits, create_datasets
from train_maxpower import HighPerformanceBraTSGNN, calculate_metrics_fast
import numpy as np

def main():
    print('📊 EVALUATING YOUR TRAINED MODEL ON TEST SET...')
    
    # Load test data
    train_patients, val_patients, test_patients = create_data_splits('./data/graphs')
    _, _, test_dataset = create_datasets('./data/graphs', train_patients, val_patients, test_patients)
    
    print(f'🧪 Test set: {len(test_dataset)} graphs from {len(test_patients)} patients')
    
    # Load trained model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    checkpoint = torch.load('./checkpoints/maxpower_gnn/best_model.pth', map_location=device, weights_only=False)
    
    # Create model
    sample_graph = test_dataset[0]
    input_dim = sample_graph.x.shape[1]
    
    model = HighPerformanceBraTSGNN(
        input_dim=input_dim,
        hidden_dim=256,
        num_layers=5,
        dropout=0.1,
        gnn_type='sage'
    ).to(device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print('🔍 Running evaluation...')
    
    # Test evaluation
    test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False, num_workers=4)
    
    all_predictions = []
    all_targets = []
    total_graphs = 0
    
    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to(device)
            logits = model(batch.x, batch.edge_index, batch.batch)
            predictions = torch.sigmoid(logits).cpu().numpy()
            targets = batch.y.cpu().numpy()
            
            all_predictions.extend(predictions)
            all_targets.extend(targets)
            total_graphs += len(targets)
    
    # Calculate final metrics
    all_predictions = np.array(all_predictions)
    all_targets = np.array(all_targets)
    
    metrics = calculate_metrics_fast(all_predictions, all_targets)
    
    print(f'\n🎯 FINAL TEST SET PERFORMANCE:')
    print(f'  🏆 Test Dice Score: {metrics["dice"]:.4f} ({metrics["dice"]*100:.1f}%)')
    print(f'  🎯 Test Sensitivity: {metrics["sensitivity"]:.4f} ({metrics["sensitivity"]*100:.1f}%)')  
    print(f'  📊 Test Accuracy: {metrics["accuracy"]:.4f} ({metrics["accuracy"]*100:.1f}%)')
    print(f'  📈 Graphs evaluated: {total_graphs:,}')
    
    # Tumor detection analysis
    tumor_predictions = all_predictions[all_targets > 0.5]
    non_tumor_predictions = all_predictions[all_targets <= 0.5]
    
    print(f'\n🔍 DETAILED ANALYSIS:')
    print(f'  🔴 Tumor nodes: {len(tumor_predictions):,} ({len(tumor_predictions)/len(all_targets)*100:.1f}%)')
    print(f'  ⚪ Non-tumor nodes: {len(non_tumor_predictions):,} ({len(non_tumor_predictions)/len(all_targets)*100:.1f}%)')
    
    if len(tumor_predictions) > 0:
        print(f'  🎯 Avg tumor prediction: {tumor_predictions.mean():.3f}')
    if len(non_tumor_predictions) > 0:
        print(f'  ⚪ Avg non-tumor prediction: {non_tumor_predictions.mean():.3f}')
    
    print(f'\n✅ MODEL EVALUATION COMPLETE!')
    print(f'\n🎉 CONGRATULATIONS! Your BraTS GNN model achieved {metrics["dice"]*100:.1f}% Dice score!')

if __name__ == "__main__":
    main()