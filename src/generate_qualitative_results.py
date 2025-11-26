#!/usr/bin/env python3
"""
Generate qualitative visualizations of segmentation results.
Creates MRI overlays showing ground truth vs predictions.
"""

import os
import sys
import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm
import nibabel as nib

# Add src to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from gnn_model import TumorSegmentationGNN


def load_nifti_volume(patient_dir):
    """Load T1ce MRI and segmentation volumes."""
    patient_dir = Path(patient_dir)
    
    # Find files
    t1ce_file = list(patient_dir.glob("*_t1ce.nii.gz"))[0]
    seg_file = list(patient_dir.glob("*_seg.nii.gz"))[0]
    
    # Load volumes
    t1ce = nib.load(t1ce_file).get_fdata()
    seg = nib.load(seg_file).get_fdata()
    
    # Normalize T1ce
    t1ce = (t1ce - t1ce.min()) / (t1ce.max() - t1ce.min() + 1e-8)
    
    return t1ce, seg


def reconstruct_prediction_volume(graph_dir, model, device):
    """
    Reconstruct full 3D prediction volume from graph predictions.
    
    Args:
        graph_dir: Path to patient graph directory
        model: Trained model
        device: torch device
        
    Returns:
        3D numpy array with prediction probabilities
    """
    graph_files = sorted(Path(graph_dir).glob("*.pt"))
    
    # Get shape from first graph
    first_graph = torch.load(graph_files[0])
    if hasattr(first_graph, 'original_shape'):
        shape = tuple(first_graph.original_shape)
    else:
        shape = (155, 240, 240)
    
    pred_volume = np.zeros(shape, dtype=np.float32)
    
    model.eval()
    with torch.no_grad():
        for graph_file in graph_files:
            data = torch.load(graph_file).to(device)
            
            # Get predictions
            out, _ = model(data.unsqueeze(0))
            pred_probs = torch.sigmoid(out.squeeze()).cpu().numpy()
            
            # Get slice index
            if hasattr(data, 'slice_idx'):
                slice_idx = int(data.slice_idx)
            else:
                slice_idx = int(graph_file.stem.split('_')[-1])
            
            # Reconstruct slice
            if hasattr(data, 'superpixel_labels'):
                sp_labels = data.superpixel_labels.numpy()
                for node_idx in range(len(pred_probs)):
                    mask = (sp_labels == node_idx)
                    pred_volume[slice_idx][mask] = pred_probs[node_idx]
    
    return pred_volume


def create_overlay_visualization(mri_slice, gt_slice, pred_slice, title="", save_path=None):
    """
    Create visualization with MRI, ground truth overlay, and prediction overlay.
    
    Args:
        mri_slice: 2D MRI slice
        gt_slice: 2D ground truth segmentation
        pred_slice: 2D prediction (probabilities)
        title: Figure title
        save_path: Path to save figure
    """
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    
    # Original MRI
    axes[0].imshow(mri_slice, cmap='gray')
    axes[0].set_title('T1ce MRI')
    axes[0].axis('off')
    
    # Ground truth overlay
    axes[1].imshow(mri_slice, cmap='gray')
    gt_mask = gt_slice > 0
    axes[1].imshow(np.ma.masked_where(~gt_mask, gt_slice), 
                   cmap='Reds', alpha=0.5, vmin=0, vmax=4)
    axes[1].set_title('Ground Truth')
    axes[1].axis('off')
    
    # Prediction overlay
    axes[2].imshow(mri_slice, cmap='gray')
    pred_binary = pred_slice > 0.5
    axes[2].imshow(np.ma.masked_where(~pred_binary, pred_slice), 
                   cmap='Blues', alpha=0.5, vmin=0, vmax=1)
    axes[2].set_title('Prediction')
    axes[2].axis('off')
    
    # Comparison (GT in red, Pred in blue, overlap in purple)
    axes[3].imshow(mri_slice, cmap='gray')
    
    # Create RGB overlay
    overlay = np.zeros((*mri_slice.shape, 3))
    overlay[gt_mask, 0] = 0.7  # Red for GT
    overlay[pred_binary, 2] = 0.7  # Blue for Pred
    
    axes[3].imshow(overlay, alpha=0.5)
    axes[3].set_title('Overlay (Red=GT, Blue=Pred)')
    axes[3].axis('off')
    
    # Overall title
    dice = 2 * (gt_mask & pred_binary).sum() / (gt_mask.sum() + pred_binary.sum() + 1e-8)
    plt.suptitle(f'{title} | Dice: {dice:.4f}', fontsize=14, y=0.98)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  Saved: {save_path}")
    
    plt.close()
    
    return fig


def find_interesting_slices(seg_volume, pred_volume, n_slices=3):
    """
    Find interesting slices to visualize.
    Returns slices with good, medium, and poor Dice scores.
    
    Args:
        seg_volume: Ground truth volume
        pred_volume: Prediction volume
        n_slices: Number of slices to return
        
    Returns:
        List of (slice_idx, dice_score) tuples
    """
    slice_dice_scores = []
    
    for slice_idx in range(seg_volume.shape[0]):
        gt_slice = seg_volume[slice_idx] > 0
        pred_slice = pred_volume[slice_idx] > 0.5
        
        # Skip empty slices
        if gt_slice.sum() < 100:
            continue
        
        # Compute Dice
        intersection = (gt_slice & pred_slice).sum()
        dice = 2 * intersection / (gt_slice.sum() + pred_slice.sum() + 1e-8)
        
        slice_dice_scores.append((slice_idx, dice))
    
    if len(slice_dice_scores) == 0:
        return []
    
    # Sort by Dice
    slice_dice_scores.sort(key=lambda x: x[1])
    
    # Select diverse slices
    selected = []
    if len(slice_dice_scores) >= 3:
        # Best, worst, median
        selected.append(slice_dice_scores[-1])  # Best
        selected.append(slice_dice_scores[len(slice_dice_scores)//2])  # Median
        selected.append(slice_dice_scores[0])  # Worst
    else:
        selected = slice_dice_scores
    
    return selected[:n_slices]


def visualize_patient(patient_id, patient_data_dir, graph_dir, model, device, output_dir):
    """
    Create visualizations for a single patient.
    
    Args:
        patient_id: Patient identifier
        patient_data_dir: Path to original NIfTI files
        graph_dir: Path to graph files
        model: Trained model
        device: Device
        output_dir: Output directory for visualizations
    """
    print(f"Processing {patient_id}...")
    
    # Load original data
    try:
        mri_volume, seg_volume = load_nifti_volume(patient_data_dir)
    except Exception as e:
        print(f"  Error loading NIfTI files: {e}")
        return
    
    # Get predictions
    try:
        pred_volume = reconstruct_prediction_volume(graph_dir, model, device)
    except Exception as e:
        print(f"  Error generating predictions: {e}")
        return
    
    # Find interesting slices
    interesting_slices = find_interesting_slices(seg_volume, pred_volume, n_slices=3)
    
    if len(interesting_slices) == 0:
        print(f"  No valid slices found for {patient_id}")
        return
    
    # Create visualizations
    for idx, (slice_idx, dice) in enumerate(interesting_slices):
        quality = ['worst', 'median', 'best'][idx] if len(interesting_slices) == 3 else f'slice{idx}'
        
        save_path = output_dir / f"{patient_id}_{quality}_slice{slice_idx:03d}.png"
        
        create_overlay_visualization(
            mri_volume[slice_idx],
            seg_volume[slice_idx],
            pred_volume[slice_idx],
            title=f"{patient_id} - Slice {slice_idx} ({quality.capitalize()})",
            save_path=save_path
        )


def main():
    parser = argparse.ArgumentParser(description='Generate qualitative visualizations')
    parser.add_argument('--fold_idx', type=int, required=True, help='Fold index (0-4)')
    parser.add_argument('--fold_dir', type=str, default='./data/cv_folds',
                        help='Directory with CV fold splits')
    parser.add_argument('--data_dir', type=str, required=True,
                        help='Directory with original BraTS NIfTI files')
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints/cv_experiments',
                        help='Directory with fold checkpoints')
    parser.add_argument('--output_dir', type=str, default='./research_results/qualitative',
                        help='Output directory for visualizations')
    parser.add_argument('--n_patients', type=int, default=4,
                        help='Number of patients to visualize')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use')
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir) / f"fold_{args.fold_idx}"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load model
    checkpoint_path = Path(args.checkpoint_dir) / f"fold_{args.fold_idx}" / "best_model.pth"
    if not checkpoint_path.exists():
        print(f"Error: Checkpoint not found: {checkpoint_path}")
        return
    
    print(f"Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=args.device)
    
    model = TumorSegmentationGNN(
        in_channels=12,
        hidden_channels=checkpoint.get('hidden_channels', 256),
        num_layers=checkpoint.get('num_layers', 5),
        dropout=checkpoint.get('dropout', 0.1)
    ).to(args.device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Load test split
    test_file = Path(args.fold_dir) / f"fold_{args.fold_idx}_test.txt"
    with open(test_file, 'r') as f:
        test_graphs = [line.strip() for line in f]
    
    # Group by patient
    patient_groups = {}
    for graph_path in test_graphs:
        patient_id = Path(graph_path).parent.name
        if patient_id not in patient_groups:
            patient_groups[patient_id] = Path(graph_path).parent
    
    # Select random patients
    import random
    random.seed(42)
    selected_patients = random.sample(list(patient_groups.items()), 
                                     min(args.n_patients, len(patient_groups)))
    
    print(f"\nGenerating visualizations for {len(selected_patients)} patients...")
    print(f"Output directory: {output_dir}\n")
    
    # Process each patient
    for patient_id, graph_dir in selected_patients:
        # Find original data
        patient_data_dir = Path(args.data_dir) / patient_id
        if not patient_data_dir.exists():
            print(f"  Warning: Data not found for {patient_id}")
            continue
        
        visualize_patient(patient_id, patient_data_dir, graph_dir, 
                         model, args.device, output_dir)
    
    print(f"\n✓ Visualizations complete! Saved to {output_dir}")


if __name__ == '__main__':
    main()
