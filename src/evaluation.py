# src/evaluation.py
import os
import numpy as np
import torch
import nibabel as nib
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from skimage.metrics import structural_similarity as ssim
import matplotlib.pyplot as plt
import argparse
from pathlib import Path
import glob
from tqdm import tqdm
import time
import pandas as pd

def calculate_metrics(y_true, y_pred, threshold=0.5):
    """Calculate binary classification metrics."""
    y_pred_binary = (y_pred >= threshold).astype(int)
    
    # Calculate metrics
    metrics = {
        'accuracy': accuracy_score(y_true.flatten(), y_pred_binary.flatten()),
        'precision': precision_score(y_true.flatten(), y_pred_binary.flatten(), zero_division=0),
        'recall': recall_score(y_true.flatten(), y_pred_binary.flatten(), zero_division=0),
        'f1': f1_score(y_true.flatten(), y_pred_binary.flatten(), zero_division=0),
    }
    
    # Calculate AUC if possible
    try:
        metrics['auc'] = roc_auc_score(y_true.flatten(), y_pred.flatten())
    except:
        metrics['auc'] = 0.0
    
    # Calculate Dice coefficient
    intersection = np.sum(y_true * y_pred_binary)
    union = np.sum(y_true) + np.sum(y_pred_binary)
    metrics['dice'] = (2.0 * intersection) / (union + 1e-6)
    
    # Calculate Hausdorff distance for non-empty masks
    if np.sum(y_true) > 0 and np.sum(y_pred_binary) > 0:
        from scipy.spatial.distance import directed_hausdorff
        metrics['hausdorff'] = max(
            directed_hausdorff(np.argwhere(y_true), np.argwhere(y_pred_binary))[0],
            directed_hausdorff(np.argwhere(y_pred_binary), np.argwhere(y_true))[0]
        )
    else:
        metrics['hausdorff'] = float('inf')
    
    # Calculate SSIM
    metrics['ssim'] = ssim(y_true, y_pred_binary)
    
    return metrics

def evaluate_patient(pred_file, label_file, output_dir=None):
    """Evaluate predictions for a single patient."""
    # Load prediction volume
    pred_nifti = nib.load(pred_file)
    pred_volume = pred_nifti.get_fdata()
    
    # Load ground truth label
    label_nifti = nib.load(label_file)
    label_volume = label_nifti.get_fdata()
    
    # Make sure volumes have same shape
    if pred_volume.shape != label_volume.shape:
        print(f"Warning: Shape mismatch for {pred_file}")
        print(f"  Prediction shape: {pred_volume.shape}")
        print(f"  Label shape: {label_volume.shape}")
        
        # Resize to smallest shape
        min_shape = [min(p, l) for p, l in zip(pred_volume.shape, label_volume.shape)]
        pred_volume = pred_volume[:min_shape[0], :min_shape[1], :min_shape[2]]
        label_volume = label_volume[:min_shape[0], :min_shape[1], :min_shape[2]]
    
    # Binarize label
    label_volume = (label_volume > 0).astype(float)
    
    # Calculate metrics
    metrics = calculate_metrics(label_volume, pred_volume)
    
    # Generate visualizations if output directory is provided
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        
        # Get patient ID from filename
        patient_id = os.path.basename(pred_file).split('_prediction')[0]
        
        # Save metrics to file
        metrics_file = os.path.join(output_dir, f"{patient_id}_metrics.csv")
        pd.DataFrame([metrics]).to_csv(metrics_file, index=False)
        
        # Generate visualization
        visualize_predictions(
            label_volume, pred_volume, output_dir, patient_id
        )
    
    return metrics

def visualize_predictions(label_volume, pred_volume, output_dir, patient_id, n_slices=5):
    """Generate visualization of predictions vs ground truth."""
    # Find middle slices with tumor
    tumor_slices = np.where(np.sum(label_volume, axis=(1, 2)) > 0)[0]
    
    if len(tumor_slices) == 0:
        print(f"Warning: No tumor found in ground truth for {patient_id}")
        return
    
    # Select n_slices evenly spaced slices
    indices = np.linspace(0, len(tumor_slices)-1, n_slices, dtype=int)
    selected_slices = [tumor_slices[i] for i in indices]
    
    # Create figure
    fig, axes = plt.subplots(n_slices, 3, figsize=(15, 5*n_slices))
    
    for i, slice_idx in enumerate(selected_slices):
        # Display ground truth
        axes[i, 0].imshow(label_volume[slice_idx], cmap='hot')
        axes[i, 0].set_title(f'Ground Truth (Slice {slice_idx})')
        axes[i, 0].axis('off')
        
        # Display prediction
        axes[i, 1].imshow(pred_volume[slice_idx], cmap='hot')
        axes[i, 1].set_title(f'Prediction (Slice {slice_idx})')
        axes[i, 1].axis('off')
        
        # Display overlay
        axes[i, 2].imshow(label_volume[slice_idx], cmap='Reds', alpha=0.7)
        axes[i, 2].imshow((pred_volume[slice_idx] > 0.5).astype(float), cmap='Blues', alpha=0.5)
        axes[i, 2].set_title(f'Overlay (Slice {slice_idx})')
        axes[i, 2].axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{patient_id}_visualization.png"), dpi=150)
    plt.close(fig)

def main():
    parser = argparse.ArgumentParser(description='Evaluate tumor segmentation results')
    parser.add_argument('--predictions_dir', type=str, required=True, help='Directory with prediction volumes')
    parser.add_argument('--labels_dir', type=str, required=True, help='Directory with ground truth labels')
    parser.add_argument('--output_dir', type=str, required=True, help='Output directory for visualizations')
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Find all prediction files
    pred_files = glob.glob(os.path.join(args.predictions_dir, "*_prediction_volume.nii.gz"))
    
    if not pred_files:
        print(f"No prediction files found in {args.predictions_dir}")
        return
    
    print(f"Found {len(pred_files)} prediction files")
    
    # Evaluate each patient
    all_metrics = []
    
    for pred_file in tqdm(pred_files, desc="Evaluating patients"):
        # Get patient ID
        patient_id = os.path.basename(pred_file).split('_prediction')[0]
        
        # Find label file
        label_file = os.path.join(args.labels_dir, patient_id, "whole_tumor_mask.nii.gz")
        
        if not os.path.exists(label_file):
            print(f"Label file not found for patient {patient_id}")
            continue
        
        # Create patient output directory
        patient_output_dir = os.path.join(args.output_dir, patient_id)
        os.makedirs(patient_output_dir, exist_ok=True)
        
        # Evaluate patient
        metrics = evaluate_patient(pred_file, label_file, patient_output_dir)
        metrics['patient_id'] = patient_id
        all_metrics.append(metrics)
    
    # Calculate and save average metrics
    if all_metrics:
        df = pd.DataFrame(all_metrics)
        
        # Save all metrics
        all_metrics_file = os.path.join(args.output_dir, "all_metrics.csv")
        df.to_csv(all_metrics_file, index=False)
        
        # Calculate and save average metrics
        avg_metrics = df.drop(columns=['patient_id']).mean().to_dict()
        avg_metrics_file = os.path.join(args.output_dir, "average_metrics.csv")
        pd.DataFrame([avg_metrics]).to_csv(avg_metrics_file, index=False)
        
        # Print average metrics
        print("Average Metrics:")
        for metric, value in avg_metrics.items():
            print(f"{metric}: {value:.4f}")
    else:
        print("No patients were successfully evaluated")

if __name__ == "__main__":
    start_time = time.time()
    main()
    print(f"Total evaluation time: {(time.time() - start_time) / 60:.2f} minutes")