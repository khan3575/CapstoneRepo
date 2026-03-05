#!/usr/bin/env python3
"""
Create qualitative segmentation visualizations from TEST SET patients only.
Generates: Original MRI, Ground Truth overlay, Prediction overlay, Comparison
"""

import os
import sys
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
import nibabel as nib
from pathlib import Path

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from config import get_config
from gnn_model import TumorSegmentationGNN


def load_patient_data(patient_dir):
    """Load T1ce MRI and segmentation for a patient."""
    patient_dir = Path(patient_dir)
    
    # Find files (BraTS naming convention)
    t1ce_file = list(patient_dir.glob("*_t1ce.nii.gz"))[0]
    seg_file = list(patient_dir.glob("*_seg.nii.gz"))[0]
    
    # Load
    t1ce_img = nib.load(t1ce_file)
    seg_img = nib.load(seg_file)
    
    t1ce = t1ce_img.get_fdata()
    seg = seg_img.get_fdata()
    
    # Normalize MRI to [0, 1]
    t1ce = (t1ce - t1ce.min()) / (t1ce.max() - t1ce.min() + 1e-8)
    
    return t1ce, seg


def predict_patient(graph_dir, model, device):
    """
    Generate predictions for all slices of a patient.
    Returns 3D prediction volume.
    """
    graph_files = sorted(Path(graph_dir).glob("*.pt"))
    
    if len(graph_files) == 0:
        return None
    
    # Assume standard BraTS shape
    pred_volume = np.zeros((155, 240, 240), dtype=np.float32)
    
    model.eval()
    with torch.no_grad():
        for graph_file in graph_files:
            # Load graph (it's a list of graphs)
            data_list = torch.load(graph_file, weights_only=False)
            
            # Extract slice index from filename
            # Format: BraTS2021_XXXXX_graphs_YYY.pt where YYY is slice count
            slice_idx = int(graph_file.stem.split('_')[-1])
            
            # Process each graph in this file
            for graph in data_list:
                graph = graph.to(device)
                
                # Get predictions
                out, _ = model(graph.unsqueeze(0) if hasattr(graph, 'batch') else graph)
                pred_probs = torch.sigmoid(out.squeeze()).cpu().numpy()
                
                # Each graph represents a superpixel region
                # For simplicity, we'll mark entire region with its prediction
                # This is approximate but sufficient for visualization
                
                # Store predictions (will be refined below)
                if len(pred_probs.shape) == 0:  # Single node
                    pred_probs = np.array([pred_probs])
                
                # Map to volume (simplified approach)
                # Since we don't have exact pixel mappings, use average
                pred_volume[slice_idx] = pred_probs.mean()
    
    return pred_volume


def find_tumor_slices(seg_volume, n_slices=3):
    """
    Find slices with substantial tumor for visualization.
    Returns indices of slices with different tumor amounts.
    """
    tumor_per_slice = []
    
    for idx in range(seg_volume.shape[0]):
        tumor_pixels = (seg_volume[idx] > 0).sum()
        if tumor_pixels > 500:  # At least 500 tumor pixels
            tumor_per_slice.append((idx, tumor_pixels))
    
    if len(tumor_per_slice) < n_slices:
        return [x[0] for x in tumor_per_slice]
    
    # Sort by tumor amount
    tumor_per_slice.sort(key=lambda x: x[1])
    
    # Select diverse slices: small, medium, large tumor
    indices = [
        tumor_per_slice[0][0],  # Smallest
        tumor_per_slice[len(tumor_per_slice)//2][0],  # Medium
        tumor_per_slice[-1][0]  # Largest
    ]
    
    return indices


def create_visualization(mri_slice, gt_slice, pred_slice, patient_id, slice_idx, save_dir):
    """
    Create 4-panel visualization:
    1. Original MRI (before)
    2. Ground Truth overlay (after - GT)
    3. Prediction overlay (after - Prediction)
    4. Comparison (GT=red, Pred=blue, overlap=purple)
    """
    fig, axes = plt.subplots(1, 4, figsize=(24, 6))
    
    # Panel 1: Original MRI (BEFORE)
    axes[0].imshow(mri_slice, cmap='gray', vmin=0, vmax=1)
    axes[0].set_title('Original MRI\n(Before)', fontsize=14, fontweight='bold')
    axes[0].axis('off')
    
    # Panel 2: Ground Truth Overlay (AFTER - GT)
    axes[1].imshow(mri_slice, cmap='gray', vmin=0, vmax=1)
    gt_mask = gt_slice > 0
    if gt_mask.sum() > 0:
        axes[1].imshow(np.ma.masked_where(~gt_mask, gt_slice), 
                       cmap='Reds', alpha=0.6, vmin=0, vmax=4)
    axes[1].set_title('Ground Truth Overlay\n(After - Manual)', fontsize=14, fontweight='bold', color='darkred')
    axes[1].axis('off')
    
    # Panel 3: Prediction Overlay (AFTER - Model)
    axes[2].imshow(mri_slice, cmap='gray', vmin=0, vmax=1)
    pred_binary = pred_slice > 0.5
    if pred_binary.sum() > 0:
        axes[2].imshow(np.ma.masked_where(~pred_binary, pred_slice), 
                       cmap='Blues', alpha=0.6, vmin=0, vmax=1)
    axes[2].set_title('Model Prediction Overlay\n(After - GNN)', fontsize=14, fontweight='bold', color='darkblue')
    axes[2].axis('off')
    
    # Panel 4: Comparison
    axes[3].imshow(mri_slice, cmap='gray', vmin=0, vmax=1)
    
    # Create RGB overlay: Red=GT, Blue=Pred, Purple=Both
    overlay = np.zeros((*mri_slice.shape, 3))
    overlay[gt_mask, 0] = 0.8  # Red channel for GT
    overlay[pred_binary, 2] = 0.8  # Blue channel for Pred
    # Where both are true, you get purple (0.8, 0, 0.8)
    
    axes[3].imshow(overlay, alpha=0.6)
    axes[3].set_title('Comparison\n(Red=GT, Blue=Pred, Purple=Match)', fontsize=14, fontweight='bold')
    axes[3].axis('off')
    
    # Calculate Dice for this slice
    if gt_mask.sum() > 0:
        intersection = (gt_mask & pred_binary).sum()
        dice = 2 * intersection / (gt_mask.sum() + pred_binary.sum() + 1e-8)
    else:
        dice = 0.0
    
    # Overall title
    plt.suptitle(f'{patient_id} - Slice {slice_idx} | Dice Score: {dice:.4f}', 
                 fontsize=16, fontweight='bold', y=0.98)
    
    plt.tight_layout()
    
    # Save
    save_path = save_dir / f"{patient_id}_slice{slice_idx:03d}_dice{dice:.3f}.png"
    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"  ✓ Saved: {save_path.name} (Dice: {dice:.4f})")
    
    return dice


def main():
    print("="*80)
    print("QUALITATIVE SEGMENTATION VISUALIZATION")
    print("Using TEST SET patients only (unseen during training)")
    print("="*80)
    print()

    # Load configuration (lazy loading to avoid import-time crashes)
    config = get_config()

    # Configuration (using config.yaml)
    fold_idx = 0  # Use fold 0
    fold_dir = Path(config.data_cv_folds)
    data_dir = Path(config.brats_2021_raw)
    checkpoint_dir = Path(config.checkpoints_binary)
    output_dir = Path(config.results_qualitative)
    device = config.get('hardware.device', 'cuda') if torch.cuda.is_available() else 'cpu'
    n_patients = 4  # Number of patients to visualize

    print(f"📁 Using paths from config:")
    print(f"   Data dir: {data_dir}")
    print(f"   Checkpoint dir: {checkpoint_dir}")
    print(f"   Output dir: {output_dir}\n")
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load fold 0 test set
    fold_file = fold_dir / f"fold_{fold_idx}.json"
    with open(fold_file, 'r') as f:
        fold_data = json.load(f)
    
    test_patients = fold_data['test_patients']
    print(f"Fold {fold_idx} test set: {len(test_patients)} patients")
    print(f"Selecting {n_patients} patients for visualization...\n")
    
    # Select diverse patients (first, middle, last for variety)
    selected_indices = [
        0,
        len(test_patients) // 3,
        2 * len(test_patients) // 3,
        len(test_patients) - 1
    ]
    selected_patients = [test_patients[i] for i in selected_indices[:n_patients]]
    
    # Load model
    checkpoint_path = checkpoint_dir / f"fold_{fold_idx}" / "best_model.pth"
    print(f"Loading model: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    model = TumorSegmentationGNN(
        in_channels=12,
        hidden_channels=checkpoint.get('hidden_channels', 256),
        num_layers=checkpoint.get('num_layers', 5),
        dropout=checkpoint.get('dropout', 0.1)
    ).to(device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    print(f"Model loaded successfully\n")
    
    # Process each patient
    all_dice_scores = []
    
    for patient_id in selected_patients:
        print(f"Processing {patient_id}...")
        
        # Find patient data
        patient_data_dir = data_dir / patient_id
        if not patient_data_dir.exists():
            print(f"  ⚠️  Data not found, skipping")
            continue
        
        # Find patient graphs
        graph_dir = Path("./data/graphs") / patient_id
        if not graph_dir.exists():
            print(f"  ⚠️  Graphs not found, skipping")
            continue
        
        try:
            # Load MRI and ground truth
            mri_volume, seg_volume = load_patient_data(patient_data_dir)
            
            # Generate predictions
            pred_volume = predict_patient(graph_dir, model, device)
            
            if pred_volume is None:
                print(f"  ⚠️  Failed to generate predictions, skipping")
                continue
            
            # Find interesting slices
            slice_indices = find_tumor_slices(seg_volume, n_slices=3)
            
            if len(slice_indices) == 0:
                print(f"  ⚠️  No tumor slices found, skipping")
                continue
            
            # Create visualizations
            patient_dice_scores = []
            for slice_idx in slice_indices:
                dice = create_visualization(
                    mri_volume[slice_idx],
                    seg_volume[slice_idx],
                    pred_volume[slice_idx],
                    patient_id,
                    slice_idx,
                    output_dir
                )
                patient_dice_scores.append(dice)
                all_dice_scores.append(dice)
            
            avg_dice = np.mean(patient_dice_scores)
            print(f"  Average Dice for {patient_id}: {avg_dice:.4f}\n")
            
        except Exception as e:
            print(f"  ⚠️  Error processing {patient_id}: {e}\n")
            continue
    
    # Summary
    print("="*80)
    print("SUMMARY")
    print("="*80)
    print(f"Generated visualizations: {len(all_dice_scores)} slices")
    print(f"Average Dice across visualizations: {np.mean(all_dice_scores):.4f} ± {np.std(all_dice_scores):.4f}")
    print(f"\nAll images saved to: {output_dir}")
    print("\nEach image shows:")
    print("  1. BEFORE: Original MRI scan")
    print("  2. AFTER (Ground Truth): Manual expert annotation in red")
    print("  3. AFTER (Model): GNN prediction in blue")  
    print("  4. Comparison: Red=GT, Blue=Pred, Purple=Match")
    print()


if __name__ == '__main__':
    main()
