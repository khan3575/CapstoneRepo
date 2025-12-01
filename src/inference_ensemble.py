#!/usr/bin/env python3
"""
Ensemble Inference Script - The +1% Booster
Combines predictions from all 5 cross-validation folds to squeeze extra accuracy.

Usage:
    python src/inference_ensemble.py --output_dir research_results/ensemble

Expected improvement: +0.5-1.5% Dice over single-fold models
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import torch
import numpy as np
import json
import argparse
from tqdm import tqdm

from gnn_model import TumorSegmentationGNN
from dataset import BraTSGraphDataset, BinaryTransform
from torch_geometric.loader import DataLoader as GeometricDataLoader


def load_ensemble_models(checkpoint_dir, device='cuda'):
    """
    Load all 5 fold models for ensemble prediction.
    
    Returns:
        models: List of trained models
        fold_info: List of dicts with fold metadata
    """
    models = []
    fold_info = []
    
    checkpoint_dir = Path(checkpoint_dir)
    
    print("="*80)
    print("LOADING ENSEMBLE MODELS")
    print("="*80)
    
    for fold_idx in range(5):
        fold_dir = checkpoint_dir / f'fold_{fold_idx}'
        checkpoint_path = fold_dir / 'best_model.pth'
        results_path = fold_dir / 'results.json'
        
        if not checkpoint_path.exists():
            print(f"⚠️  Fold {fold_idx} checkpoint not found: {checkpoint_path}")
            continue
        
        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        
        # Initialize model with same architecture
        model = TumorSegmentationGNN(
            in_channels=15,
            hidden_channels=256,
            num_layers=5,
            dropout=0.1
        ).to(device)
        
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        models.append(model)
        
        # Load fold info
        if results_path.exists():
            with open(results_path) as f:
                results = json.load(f)
            fold_info.append({
                'fold': fold_idx,
                'epoch': checkpoint['epoch'],
                'val_dice': checkpoint['val_dice'],
                'test_dice': results.get('test_metrics', {}).get('dice', 0.0)
            })
            print(f"✅ Fold {fold_idx}: Val Dice={checkpoint['val_dice']:.4f}, "
                  f"Test Dice={results.get('test_metrics', {}).get('dice', 0.0):.4f}")
        else:
            fold_info.append({
                'fold': fold_idx,
                'epoch': checkpoint['epoch'],
                'val_dice': checkpoint['val_dice'],
                'test_dice': 0.0
            })
            print(f"✅ Fold {fold_idx}: Val Dice={checkpoint['val_dice']:.4f}")
    
    print(f"\n📦 Loaded {len(models)}/5 models for ensemble")
    return models, fold_info


def predict_ensemble(models, data, device='cuda', method='mean'):
    """
    Run ensemble prediction on a single graph.
    
    Args:
        models: List of trained models
        data: PyG Data object
        device: Device to use
        method: 'mean' (average logits) or 'vote' (majority vote)
    
    Returns:
        probs: Ensemble probabilities [num_nodes]
        preds: Binary predictions [num_nodes]
    """
    data = data.to(device)
    
    with torch.no_grad():
        if method == 'mean':
            # Average logits before sigmoid
            logits_sum = torch.zeros(data.x.shape[0], device=device)
            
            for model in models:
                logits, _ = model(data)  # Model returns (logits, embeddings)
                logits_sum += logits
            
            avg_logits = logits_sum / len(models)
            probs = torch.sigmoid(avg_logits)
            
        elif method == 'vote':
            # Majority vote of binary predictions
            votes = torch.zeros(data.x.shape[0], device=device)
            
            for model in models:
                logits, _ = model(data)
                preds = (torch.sigmoid(logits) > 0.5).float()
                votes += preds
            
            probs = votes / len(models)
        
        preds = (probs > 0.5).float()
    
    return probs, preds


def evaluate_ensemble(models, test_dataset, device='cuda', batch_size=1, method='mean'):
    """
    Evaluate ensemble on entire test set.
    
    Returns:
        metrics: Dict with dice, accuracy, etc.
        per_graph_results: List of per-graph Dice scores
    """
    loader = GeometricDataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    all_dice = []
    all_accuracy = []
    all_tp = 0
    all_fp = 0
    all_fn = 0
    all_tn = 0
    
    print("\n" + "="*80)
    print(f"ENSEMBLE EVALUATION ({method.upper()} method)")
    print("="*80)
    
    for data in tqdm(loader, desc="Evaluating ensemble"):
        probs, preds = predict_ensemble(models, data, device, method)
        
        # Ground truth
        y_true = (data.y > 0).float().to(device)
        
        # Dice score
        intersection = (preds * y_true).sum()
        union = preds.sum() + y_true.sum()
        dice = (2.0 * intersection + 1e-7) / (union + 1e-7)
        all_dice.append(dice.item())
        
        # Accuracy
        accuracy = (preds == y_true).float().mean()
        all_accuracy.append(accuracy.item())
        
        # Confusion matrix elements
        all_tp += ((preds == 1) & (y_true == 1)).sum().item()
        all_fp += ((preds == 1) & (y_true == 0)).sum().item()
        all_fn += ((preds == 0) & (y_true == 1)).sum().item()
        all_tn += ((preds == 0) & (y_true == 0)).sum().item()
    
    # Calculate aggregate metrics
    sensitivity = all_tp / (all_tp + all_fn) if (all_tp + all_fn) > 0 else 0.0
    specificity = all_tn / (all_tn + all_fp) if (all_tn + all_fp) > 0 else 0.0
    precision = all_tp / (all_tp + all_fp) if (all_tp + all_fp) > 0 else 0.0
    
    metrics = {
        'dice': float(np.mean(all_dice)),
        'dice_std': float(np.std(all_dice)),
        'accuracy': float(np.mean(all_accuracy)),
        'sensitivity': sensitivity,
        'specificity': specificity,
        'precision': precision,
        'num_graphs': len(all_dice)
    }
    
    return metrics, all_dice


def main():
    parser = argparse.ArgumentParser(description='Ensemble inference from 5-fold models')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints/binary_training',
                        help='Directory containing fold_X subdirectories')
    parser.add_argument('--fold_file', type=str, default='data/cv_folds/fold_0.json',
                        help='Fold file to use for test set (any fold has same test set)')
    parser.add_argument('--method', type=str, default='mean', choices=['mean', 'vote'],
                        help='Ensemble method: mean (average logits) or vote (majority)')
    parser.add_argument('--output_dir', type=str, default='research_results/ensemble',
                        help='Output directory for results')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use')
    
    args = parser.parse_args()
    
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*80)
    print("ENSEMBLE INFERENCE - THE +1% BOOSTER")
    print("="*80)
    print(f"Method: {args.method.upper()}")
    print(f"Device: {device}")
    
    # Load models
    models, fold_info = load_ensemble_models(args.checkpoint_dir, device)
    
    if len(models) == 0:
        print("\n❌ No models found! Train all 5 folds first.")
        return
    
    if len(models) < 5:
        print(f"\n⚠️  Only {len(models)}/5 folds available. Results may be suboptimal.")
        print("   For best ensemble performance, train all 5 folds.")
    
    # Load test data
    with open(args.fold_file) as f:
        fold_data = json.load(f)
    
    test_patients = fold_data['test_patients']
    test_graphs = [
        str(Path('data/graphs') / pid / f'{pid}_graphs_200.pt')
        for pid in test_patients
    ]
    
    binary_transform = BinaryTransform()
    test_dataset = BraTSGraphDataset(
        root_dir=None,
        split='test',
        graph_files=test_graphs,
        transform=binary_transform
    )
    
    print(f"\n📊 Test set: {len(test_dataset)} graphs from {len(test_patients)} patients")
    
    # Evaluate ensemble
    metrics, per_graph_dice = evaluate_ensemble(
        models, test_dataset, device, batch_size=1, method=args.method
    )
    
    # Print results
    print("\n" + "="*80)
    print("ENSEMBLE RESULTS")
    print("="*80)
    print(f"Dice:        {metrics['dice']:.4f} ± {metrics['dice_std']:.4f}")
    print(f"Accuracy:    {metrics['accuracy']:.4f}")
    print(f"Sensitivity: {metrics['sensitivity']:.4f}")
    print(f"Specificity: {metrics['specificity']:.4f}")
    print(f"Precision:   {metrics['precision']:.4f}")
    
    # Compare to individual folds
    print("\n" + "="*80)
    print("COMPARISON: ENSEMBLE vs INDIVIDUAL FOLDS")
    print("="*80)
    
    individual_scores = [f['test_dice'] for f in fold_info if f['test_dice'] > 0]
    if individual_scores:
        mean_individual = np.mean(individual_scores)
        improvement = metrics['dice'] - mean_individual
        
        print(f"Individual folds (mean): {mean_individual:.4f}")
        print(f"Ensemble:                {metrics['dice']:.4f}")
        print(f"Improvement:             {improvement:+.4f} ({improvement*100:+.2f}%)")
        
        if improvement > 0:
            print(f"\n✅ Ensemble achieved +{improvement*100:.2f}% boost!")
        else:
            print(f"\n⚠️  Ensemble did not improve over individual models")
    
    # Save results
    results = {
        'ensemble_method': args.method,
        'num_models': len(models),
        'fold_info': fold_info,
        'test_metrics': metrics,
        'per_graph_dice': per_graph_dice
    }
    
    results_file = output_dir / 'ensemble_results.json'
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✅ Results saved to: {results_file}")
    
    # Thesis statement
    print("\n" + "="*80)
    print("FOR YOUR THESIS")
    print("="*80)
    print(f"""
Ensemble Statement:
"By combining predictions from all 5 cross-validation folds using logit averaging,
we achieved an ensemble Dice score of {metrics['dice']:.4f}, representing a
{(metrics['dice'] - mean_individual)*100:+.2f}% improvement over the mean individual
fold performance ({mean_individual:.4f})."
    """)


if __name__ == "__main__":
    main()