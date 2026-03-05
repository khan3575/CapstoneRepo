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

from config import get_config
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
        
        # torch.compile() adds '_orig_mod.' prefix to state dict keys.
        # Strip it so weights load into a fresh uncompiled model.
        state_dict = checkpoint['model_state_dict']
        if any(k.startswith('_orig_mod.') for k in state_dict):
            state_dict = {k.replace('_orig_mod.', '', 1): v for k, v in state_dict.items()}
        model.load_state_dict(state_dict)
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
    # data already on device (moved by caller with non_blocking=True)
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


def evaluate_ensemble(models, test_dataset, device='cuda', batch_size=64, method='mean'):
    """
    Evaluate ensemble on entire test set.

    Returns:
        metrics: Dict with dice, accuracy, etc.
        per_graph_results: List of per-graph Dice scores
    """
    # batch_size=64: process 64 graphs at once instead of 1 — ~64x throughput gain
    loader = GeometricDataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                                 num_workers=4, pin_memory=True, persistent_workers=True)
    
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
        data = data.to(device, non_blocking=True)
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
    parser = argparse.ArgumentParser(description='Ensemble inference from 5-fold models (uses config.yaml for defaults)')
    parser.add_argument('--checkpoint_dir', type=str, default=None,
                        help='Directory containing fold_X subdirectories (default from config.yaml)')
    parser.add_argument('--test_file', type=str, default=None,
                        help='JSON file with test patients (default: data/splits/held_out_test.json)')
    parser.add_argument('--method', type=str, default='mean', choices=['mean', 'vote'],
                        help='Ensemble method: mean (average logits) or vote (majority)')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Output directory for results (default from config.yaml)')
    parser.add_argument('--device', type=str, default=None,
                        help='Device to use (default from config.yaml: cuda)')
    parser.add_argument('--demo', action='store_true',
                        help='Demo mode: run on 10 held-out patients only')

    args = parser.parse_args()

    # Load config (lazy loading to avoid import-time crashes)
    config = get_config()

    # Use config defaults for None values
    checkpoint_dir = args.checkpoint_dir or config.checkpoints_binary
    # Default to held-out test set (Phase 4 requirement — leakage-free evaluation)
    test_file = args.test_file or str(Path(config.get('paths.data.splits', 'data/splits')) / 'held_out_test.json')
    output_dir_arg = args.output_dir or config.results_ensemble
    device_str = args.device or config.get('hardware.device', 'cuda')

    device = torch.device(device_str if torch.cuda.is_available() else 'cpu')
    output_dir = Path(output_dir_arg)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("="*80)
    print("ENSEMBLE INFERENCE — HELD-OUT TEST SET EVALUATION")
    print("="*80)
    print(f"Checkpoint dir: {checkpoint_dir}")
    print(f"Test file:      {test_file}")
    print(f"Output dir:     {output_dir}")
    print(f"Method:         {args.method.upper()}")
    print(f"Device:         {device}")
    if args.demo:
        print(f"[DEMO MODE: 10 patients only]")

    # Load models
    models, fold_info = load_ensemble_models(checkpoint_dir, device)
    
    if len(models) == 0:
        print("\n❌ No models found! Train all 5 folds first.")
        return
    
    if len(models) < 5:
        print(f"\n⚠️  Only {len(models)}/5 folds available. Results may be suboptimal.")
        print("   For best ensemble performance, train all 5 folds.")
    
    # Load test patients — supports both held_out_test.json and fold_X.json formats
    with open(test_file) as f:
        test_file_data = json.load(f)

    # held_out_test.json uses 'patients'; fold_X.json uses 'test_patients'
    if 'patients' in test_file_data:
        test_patients = test_file_data['patients']
    elif 'test_patients' in test_file_data:
        test_patients = test_file_data['test_patients']
    else:
        raise ValueError(f"Cannot find patient list in {test_file}. Keys: {list(test_file_data.keys())}")

    if args.demo:
        test_patients = test_patients[:10]

    test_graphs = [
        str(Path(config.data_graphs) / pid / f'{pid}_graphs_200.pt')
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
    if individual_scores:
        print(f"""
Ensemble Statement:
"By combining predictions from all {len(models)} cross-validation fold models using
logit averaging, we achieved an ensemble Dice score of {metrics['dice']:.4f} on the
251-patient held-out test set, representing a {(metrics['dice'] - mean_individual)*100:+.2f}%
improvement over the mean individual fold performance ({mean_individual:.4f})."
        """)
    else:
        print(f"Ensemble Dice on held-out set: {metrics['dice']:.4f}")


if __name__ == "__main__":
    main()