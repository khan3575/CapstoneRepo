# src/visualization.py
import os
import numpy as np
import torch
import nibabel as nib
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import argparse
import glob
from tqdm import tqdm
import time
from pathlib import Path

def create_custom_colormap():
    """Create a custom colormap for visualization."""
    # Custom colormap for visualization
    colors = [
        (0, 0, 0, 0),         # transparent for background
        (1, 0.5, 0, 0.8)      # orange with alpha for tumor
    ]
    
    return LinearSegmentedColormap.from_list('tumor_cmap', colors, N=256)

def visualize_slice(t1_slice, label_slice, pred_slice, title, cmap=None):
    """Visualize a single slice with prediction overlay."""
    # Normalize T1 for visualization
    t1_norm = (t1_slice - t1_slice.min()) / (t1_slice.max() - t1_slice.min() + 1e-8)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(8, 8))
    
    # Display T1 as background
    ax.imshow(t1_norm, cmap='gray')
    
    # Display ground truth and prediction overlays
    if label_slice is not None:
        label_mask = ax.imshow(label_slice, cmap='Reds', alpha=0.5)
    
    if pred_slice is not None:
        pred_mask = ax.imshow(pred_slice, cmap='Blues', alpha=0.5)
    
    # Add legend
    legend_elements = []
    if label_slice is not None:
        from matplotlib.patches import Patch
        legend_elements.append(Patch(facecolor='red', alpha=0.5, label='Ground Truth'))
    
    if pred_slice is not None:
        from matplotlib.patches import Patch
        legend_elements.append(Patch(facecolor='blue', alpha=0.5, label='Prediction'))
    
    if legend_elements:
        ax.legend(handles=legend_elements, loc='lower right')
    
    ax.set_title(title)
    ax.axis('off')
    
    return fig

def visualize_volume(t1_volume, label_volume=None, pred_volume=None, slices=None, output_file=None):
    """
    Visualize a 3D volume with optional ground truth and prediction overlays.
    
    Args:
        t1_volume: T1 volume data
        label_volume: Ground truth label volume (optional)
        pred_volume: Prediction volume (optional)
        slices: List of slice indices to visualize (optional)
        output_file: Output file path (optional)
    """
    # Create custom colormap
    cmap = create_custom_colormap()
    
    # Determine slices to visualize
    if slices is None:
        # Find slices with tumor in either label or prediction
        has_tumor = np.zeros(t1_volume.shape[0], dtype=bool)
        
        if label_volume is not None:
            has_tumor |= np.any(label_volume > 0.5, axis=(1, 2))
        
        if pred_volume is not None:
            has_tumor |= np.any(pred_volume > 0.5, axis=(1, 2))
        
        # If no tumor found, choose middle slices
        if not np.any(has_tumor):
            mid_slice = t1_volume.shape[0] // 2
            slices = [mid_slice - 10, mid_slice, mid_slice + 10]
            slices = [s for s in slices if 0 <= s < t1_volume.shape[0]]
        else:
            # Choose a few tumor slices evenly spaced
            tumor_slices = np.where(has_tumor)[0]
            n_slices = min(5, len(tumor_slices))
            indices = np.linspace(0, len(tumor_slices)-1, n_slices, dtype=int)
            slices = [tumor_slices[i] for i in indices]
    
    # Create figure
    fig, axes = plt.subplots(len(slices), 1, figsize=(8, 6*len(slices)))
    
    # Handle case with single slice
    if len(slices) == 1:
        axes = [axes]
    
    # Visualize each slice
    for i, slice_idx in enumerate(slices):
        # Extract slices
        t1_slice = t1_volume[slice_idx]
        label_slice = label_volume[slice_idx] if label_volume is not None else None
        pred_slice = pred_volume[slice_idx] if pred_volume is not None else None
        
        # Normalize T1 for visualization
        t1_norm = (t1_slice - t1_slice.min()) / (t1_slice.max() - t1_slice.min() + 1e-8)
        
        # Display T1 as background
        axes[i].imshow(t1_norm, cmap='gray')
        
        # Display ground truth and prediction overlays
        if label_slice is not None:
            axes[i].imshow(label_slice, cmap='Reds', alpha=0.5)
        
        if pred_slice is not None:
            axes[i].imshow(pred_slice, cmap='Blues', alpha=0.5)
        
        # Add legend
        legend_elements = []
        if label_slice is not None:
            from matplotlib.patches import Patch
            legend_elements.append(Patch(facecolor='red', alpha=0.5, label='Ground Truth'))
        
        if pred_slice is not None:
            from matplotlib.patches import Patch
            legend_elements.append(Patch(facecolor='blue', alpha=0.5, label='Prediction'))
        
        if legend_elements:
            axes[i].legend(handles=legend_elements, loc='lower right')
        
        axes[i].set_title(f'Slice {slice_idx}')
        axes[i].axis('off')
    
    plt.tight_layout()
    
    # Save figure if output file provided
    if output_file:
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        plt.close(fig)
        return None
    
    return fig

def create_error_map(label_volume, pred_volume):
    """Create an error map showing true positives, false positives, and false negatives."""
    # Binarize volumes
    label_binary = label_volume > 0.5
    pred_binary = pred_volume > 0.5
    
    # Create error map
    error_map = np.zeros(label_volume.shape, dtype=np.uint8)
    
    # True positives (both label and prediction are positive)
    error_map[np.logical_and(label_binary, pred_binary)] = 1
    
    # False positives (prediction is positive but label is negative)
    error_map[np.logical_and(~label_binary, pred_binary)] = 2
    
    # False negatives (label is positive but prediction is negative)
    error_map[np.logical_and(label_binary, ~pred_binary)] = 3
    
    return error_map

def visualize_error_map(t1_volume, error_map, slices=None, output_file=None):
    """Visualize error map with T1 background."""
    # Create colormap for error visualization
    colors = [
        (0, 0, 0, 0),          # transparent for background
        (0, 1, 0, 0.7),        # green for true positives
        (1, 0, 0, 0.7),        # red for false positives
        (1, 1, 0, 0.7)         # yellow for false negatives
    ]
    error_cmap = LinearSegmentedColormap.from_list('error_cmap', colors, N=4)
    
    # Determine slices to visualize
    if slices is None:
        # Find slices with errors
        has_error = np.any(error_map > 0, axis=(1, 2))
        
        # If no errors found, choose middle slices
        if not np.any(has_error):
            mid_slice = t1_volume.shape[0] // 2
            slices = [mid_slice - 10, mid_slice, mid_slice + 10]
            slices = [s for s in slices if 0 <= s < t1_volume.shape[0]]
        else:
            # Choose a few error slices evenly spaced
            error_slices = np.where(has_error)[0]
            n_slices = min(5, len(error_slices))
            indices = np.linspace(0, len(error_slices)-1, n_slices, dtype=int)
            slices = [error_slices[i] for i in indices]
    
    # Create figure
    fig, axes = plt.subplots(len(slices), 1, figsize=(8, 6*len(slices)))
    
    # Handle case with single slice
    if len(slices) == 1:
        axes = [axes]
    
    # Visualize each slice
    for i, slice_idx in enumerate(slices):
        # Extract slices
        t1_slice = t1_volume[slice_idx]
        error_slice = error_map[slice_idx]
        
        # Normalize T1 for visualization
        t1_norm = (t1_slice - t1_slice.min()) / (t1_slice.max() - t1_slice.min() + 1e-8)
        
        # Display T1 as background
        axes[i].imshow(t1_norm, cmap='gray')
        
        # Display error map
        axes[i].imshow(error_slice, cmap=error_cmap)
        
        # Add legend
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='green', alpha=0.7, label='True Positive'),
            Patch(facecolor='red', alpha=0.7, label='False Positive'),
            Patch(facecolor='yellow', alpha=0.7, label='False Negative')
        ]
        axes[i].legend(handles=legend_elements, loc='lower right')
        
        axes[i].set_title(f'Error Map - Slice {slice_idx}')
        axes[i].axis('off')
    
    plt.tight_layout()
    
    # Save figure if output file provided
    if output_file:
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        plt.close(fig)
        return None
    
    return fig

def main():
    parser = argparse.ArgumentParser(description='Visualize tumor segmentation results')
    parser.add_argument('--t1_dir', type=str, required=True, help='Directory with T1 volumes')
    parser.add_argument('--label_dir', type=str, help='Directory with ground truth labels')
    parser.add_argument('--pred_dir', type=str, help='Directory with prediction volumes')
    parser.add_argument('--output_dir', type=str, required=True, help='Output directory for visualizations')
    parser.add_argument('--n_slices', type=int, default=5, help='Number of slices to visualize per volume')
    parser.add_argument('--error_maps', action='store_true', help='Generate error maps')
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Find all T1 volumes
    t1_files = glob.glob(os.path.join(args.t1_dir, "*", "*T1*.nii.gz"))
    
    if not t1_files:
        print(f"No T1 files found in {args.t1_dir}")
        return
    
    print(f"Found {len(t1_files)} T1 volumes")
    
    # Process each patient
    for t1_file in tqdm(t1_files, desc="Generating visualizations"):
        # Extract patient ID
        patient_id = os.path.basename(os.path.dirname(t1_file))
        
        # Load T1 volume
        t1_nifti = nib.load(t1_file)
        t1_volume = t1_nifti.get_fdata()
        
        # Find and load label volume if available
        label_volume = None
        if args.label_dir:
            label_file = os.path.join(args.label_dir, patient_id, "whole_tumor_mask.nii.gz")
            if os.path.exists(label_file):
                label_nifti = nib.load(label_file)
                label_volume = label_nifti.get_fdata()
        
        # Find and load prediction volume if available
        pred_volume = None
        if args.pred_dir:
            pred_file = os.path.join(args.pred_dir, f"{patient_id}_prediction_volume.nii.gz")
            if os.path.exists(pred_file):
                pred_nifti = nib.load(pred_file)
                pred_volume = pred_nifti.get_fdata()
        
        # Create patient output directory
        patient_output_dir = os.path.join(args.output_dir, patient_id)
        os.makedirs(patient_output_dir, exist_ok=True)
        
        # Find slices with tumor
        tumor_slices = []
        if label_volume is not None:
            tumor_slices = np.where(np.any(label_volume > 0.5, axis=(1, 2)))[0]
        
        if len(tumor_slices) == 0 and pred_volume is not None:
            tumor_slices = np.where(np.any(pred_volume > 0.5, axis=(1, 2)))[0]
        
        if len(tumor_slices) == 0:
            # If no tumor found, use middle slices
            mid_slice = t1_volume.shape[0] // 2
            tumor_slices = [max(0, mid_slice - 10), mid_slice, min(t1_volume.shape[0] - 1, mid_slice + 10)]
        elif len(tumor_slices) > args.n_slices:
            # Select evenly spaced tumor slices
            indices = np.linspace(0, len(tumor_slices)-1, args.n_slices, dtype=int)
            tumor_slices = [tumor_slices[i] for i in indices]
        
        # Generate visualization
        vis_file = os.path.join(patient_output_dir, f"{patient_id}_visualization.png")
        visualize_volume(
            t1_volume, label_volume, pred_volume,
            slices=tumor_slices,
            output_file=vis_file
        )
        
        # Generate error maps if requested and both label and prediction are available
        if args.error_maps and label_volume is not None and pred_volume is not None:
            # Ensure volumes have the same shape
            if label_volume.shape != pred_volume.shape:
                print(f"Warning: Shape mismatch for {patient_id}")
                print(f"  Label shape: {label_volume.shape}")
                print(f"  Prediction shape: {pred_volume.shape}")
                
                # Resize to smallest shape
                min_shape = [min(l, p) for l, p in zip(label_volume.shape, pred_volume.shape)]
                label_volume = label_volume[:min_shape[0], :min_shape[1], :min_shape[2]]
                pred_volume = pred_volume[:min_shape[0], :min_shape[1], :min_shape[2]]
            
            # Create error map
            error_map = create_error_map(label_volume, pred_volume)
            
            # Visualize error map
            error_file = os.path.join(patient_output_dir, f"{patient_id}_error_map.png")
            visualize_error_map(
                t1_volume, error_map,
                slices=tumor_slices,
                output_file=error_file
            )

if __name__ == "__main__":
    start_time = time.time()
    main()
    print(f"Total visualization time: {(time.time() - start_time) / 60:.2f} minutes")