#!/usr/bin/env python3
"""
Evaluate model performance per tumor region (WT, TC, ET) for BraTS dataset.

BraTS tumor regions:
- Whole Tumor (WT): Label 1, 2, or 4 (all tumor)
- Tumor Core (TC): Label 1 or 4 (enhancing + necrotic)
- Enhancing Tumor (ET): Label 4 only
"""

import os
import sys
import json
import argparse
import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm
import nibabel as nib

# Add src to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from torch_geometric.loader import DataLoader as GeometricDataLoader
from dataset import BraTSGraphDataset
from gnn_model import TumorSegmentationGNN


def compute_region_masks(seg_volume):
    """
    Convert BraTS segmentation to region masks.
    
    Args:
        seg_volume: numpy array with labels 0,1,2,4
        
    Returns:
        dict with WT, TC, ET binary masks
    """
    wt_mask = (seg_volume > 0).astype(np.float32)  # Label 1,2,4
    tc_mask = ((seg_volume == 1) | (seg_volume == 4)).astype(np.float32)  # Label 1,4
    et_mask = (seg_volume == 4).astype(np.float32)  # Label 4
    
    return {
        'WT': wt_mask,
        'TC': tc_mask,
        'ET': et_mask
    }


def dice_coefficient(pred, target):
    """Compute Dice coefficient between two binary masks."""
    smooth = 1e-5
    
    pred_flat = pred.flatten()
    target_flat = target.flatten()
    
    intersection = (pred_flat * target_flat).sum()
    
    dice = (2. * intersection + smooth) / (pred_flat.sum() + target_flat.sum() + smooth)
    
    return dice


def evaluate_patient_per_region(graph_dir, model, device):
    """
    Evaluate a single patient and return per-region Dice scores.
    
    Args:
        graph_dir: Path to patient graph directory
        model: Trained model
        device: torch device
        
    Returns:
        dict with Dice scores for WT, TC, ET
    """
    # Load all graph files for this patient
    graph_files = sorted(Path(graph_dir).glob("*.pt"))
    
    if len(graph_files) == 0:
        return None
    
    # Get original segmentation shape from first graph
    first_graph = torch.load(graph_files[0])
    if not hasattr(first_graph, 'original_shape'):
        # Assume standard BraTS shape
        original_shape = (155, 240, 240)
    else:
        original_shape = tuple(first_graph.original_shape)
    
    # Initialize prediction volume
    pred_volume = np.zeros(original_shape, dtype=np.float32)
    gt_volume = np.zeros(original_shape, dtype=np.uint8)
    
    # Process each slice
    model.eval()
    with torch.no_grad():
        for graph_file in graph_files:
            data = torch.load(graph_file).to(device)
            
            # Get predictions
            out, _ = model(data.unsqueeze(0))
            pred_probs = torch.sigmoid(out.squeeze()).cpu().numpy()
            
            # Get ground truth
            gt_labels = data.y.cpu().numpy()
            
            # Reconstruct slice from graph nodes
            if hasattr(data, 'slice_idx'):
                slice_idx = int(data.slice_idx)
            else:
                # Extract from filename: BraTS2021_XXXXX_slice_YYY.pt
                slice_idx = int(graph_file.stem.split('_')[-1])
            
            # Map predictions back to image space
            if hasattr(data, 'node_to_pixel'):
                # Use stored mapping
                node_to_pixel = data.node_to_pixel.numpy()
                for node_idx, (i, j) in enumerate(node_to_pixel):
                    pred_volume[slice_idx, i, j] = pred_probs[node_idx]
                    gt_volume[slice_idx, i, j] = int(gt_labels[node_idx])
            else:
                # Use superpixel mapping (stored in graph)
                if hasattr(data, 'superpixel_labels'):
                    sp_labels = data.superpixel_labels.numpy()
                    for node_idx in range(len(pred_probs)):
                        mask = (sp_labels == node_idx)
                        pred_volume[slice_idx][mask] = pred_probs[node_idx]
                        gt_volume[slice_idx][mask] = int(gt_labels[node_idx])
    
    # Threshold predictions
    pred_volume_binary = (pred_volume > 0.5).astype(np.float32)
    
    # Convert to region masks
    pred_regions = compute_region_masks(pred_volume_binary)
    gt_regions = compute_region_masks(gt_volume)
    
    # Compute Dice per region
    results = {}
    for region in ['WT', 'TC', 'ET']:
        dice = dice_coefficient(pred_regions[region], gt_regions[region])
        results[region] = float(dice)
    
    return results


def evaluate_fold_per_region(fold_idx, fold_dir, checkpoint_path, device='cuda'):
    """
    Evaluate entire fold and return per-region metrics.
    
    Args:
        fold_idx: Fold index (0-4)
        fold_dir: Directory containing fold splits
        checkpoint_path: Path to trained model checkpoint
        device: Device to use
        
    Returns:
        dict with per-region metrics
    """
    print(f"\n{'='*80}")
    print(f"Evaluating Fold {fold_idx} - Per-Region Metrics")
    print(f"{'='*80}\n")
    
    # Load test split
    test_file = Path(fold_dir) / f"fold_{fold_idx}_test.txt"
    with open(test_file, 'r') as f:
        test_graphs = [line.strip() for line in f]
    
    print(f"Test set: {len(test_graphs)} graphs")
    
    # Load model
    print(f"Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    model = TumorSegmentationGNN(
        in_channels=12,
        hidden_channels=checkpoint.get('hidden_channels', 256),
        num_layers=checkpoint.get('num_layers', 5),
        dropout=checkpoint.get('dropout', 0.1)
    ).to(device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Group graphs by patient
    patient_groups = {}
    for graph_path in test_graphs:
        # Extract patient ID from path
        patient_id = Path(graph_path).parent.name
        if patient_id not in patient_groups:
            patient_groups[patient_id] = []
        patient_groups[patient_id].append(graph_path)
    
    print(f"Evaluating {len(patient_groups)} patients...")
    
    # Evaluate each patient
    all_results = {'WT': [], 'TC': [], 'ET': []}
    
    for patient_id, graph_files in tqdm(patient_groups.items(), desc="Patients"):
        # Get patient directory
        patient_dir = Path(graph_files[0]).parent
        
        try:
            results = evaluate_patient_per_region(patient_dir, model, device)
            if results:
                for region in ['WT', 'TC', 'ET']:
                    all_results[region].append(results[region])
        except Exception as e:
            print(f"  Warning: Failed to evaluate {patient_id}: {e}")
            continue
    
    # Compute statistics
    stats = {}
    for region in ['WT', 'TC', 'ET']:
        scores = np.array(all_results[region])
        stats[region] = {
            'mean': float(np.mean(scores)),
            'std': float(np.std(scores)),
            'median': float(np.median(scores)),
            'min': float(np.min(scores)),
            'max': float(np.max(scores)),
            'n_patients': len(scores)
        }
    
    print(f"\n{'='*80}")
    print("Per-Region Results:")
    print(f"{'='*80}")
    for region in ['WT', 'TC', 'ET']:
        print(f"{region}: {stats[region]['mean']:.4f} ± {stats[region]['std']:.4f} "
              f"(n={stats[region]['n_patients']})")
    print()
    
    return stats, all_results


def main():
    parser = argparse.ArgumentParser(description='Evaluate per-region metrics')
    parser.add_argument('--fold_idx', type=int, required=True, help='Fold index (0-4)')
    parser.add_argument('--fold_dir', type=str, default='./data/cv_folds',
                        help='Directory with CV fold splits')
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints/cv_experiments',
                        help='Directory with fold checkpoints')
    parser.add_argument('--output_dir', type=str, default='./research_results/per_region',
                        help='Output directory for results')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use')
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Find checkpoint
    checkpoint_path = Path(args.checkpoint_dir) / f"fold_{args.fold_idx}" / "best_model.pth"
    if not checkpoint_path.exists():
        print(f"Error: Checkpoint not found: {checkpoint_path}")
        return
    
    # Evaluate
    stats, all_results = evaluate_fold_per_region(
        args.fold_idx,
        args.fold_dir,
        checkpoint_path,
        args.device
    )
    
    # Save results
    output_file = output_dir / f"fold_{args.fold_idx}_per_region.json"
    with open(output_file, 'w') as f:
        json.dump({
            'fold_idx': args.fold_idx,
            'statistics': stats,
            'all_scores': all_results
        }, f, indent=2)
    
    print(f"✓ Results saved to {output_file}")


if __name__ == '__main__':
    main()
