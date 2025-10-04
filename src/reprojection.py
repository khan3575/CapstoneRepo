# src/reprojection.py
import os
import numpy as np
import torch
import nibabel as nib
import argparse
from pathlib import Path
import glob
from tqdm import tqdm
import time

def reproject_predictions(predictions, segments, volume_shape):
    """
    Reproject predictions from graph space back to image space.
    
    Args:
        predictions: Node predictions (binary or probabilities)
        segments: Superpixel segments for each slice
        volume_shape: Shape of the output volume
    
    Returns:
        3D volume with reprojected predictions
    """
    # Initialize output volume
    output_volume = np.zeros(volume_shape, dtype=np.float32)
    
    # Track current prediction index
    pred_idx = 0
    
    # Process each slice
    for z, seg in enumerate(segments):
        # Get unique superpixel labels (excluding background 0)
        superpixel_labels = np.unique(seg)
        superpixel_labels = superpixel_labels[superpixel_labels != 0]
        
        if len(superpixel_labels) == 0:
            # No superpixels in this slice
            continue
        
        # For each superpixel, assign its prediction
        for i, sp_id in enumerate(superpixel_labels):
            if pred_idx >= len(predictions):
                # Out of predictions, break
                break
                
            # Get mask for this superpixel
            mask = (seg == sp_id)
            
            # Assign prediction to all pixels in this superpixel
            output_volume[z][mask] = predictions[pred_idx]
            
            # Increment prediction index
            pred_idx += 1
    
    return output_volume

def process_patient(patient_id, predictions_file, segments_file, original_file, output_dir):
    """Process a single patient to reproject predictions."""
    try:
        print(f"Processing patient {patient_id}")
        
        # Load predictions
        predictions = np.load(predictions_file)
        
        # Load segments
        segments = np.load(segments_file)
        
        # Load original volume to get shape and header
        orig_data = np.load(original_file)
        volume_shape = orig_data['T1'].shape
        
        # Create a NIfTI reference from the original T1
        t1_nifti = nib.Nifti1Image(orig_data['T1'], np.eye(4))
        
        # Reproject predictions
        reprojected = reproject_predictions(predictions, segments, volume_shape)
        
        # Save reprojected volume
        output_file = os.path.join(output_dir, f"{patient_id}_prediction_volume.nii.gz")
        prediction_nifti = nib.Nifti1Image(reprojected, t1_nifti.affine, t1_nifti.header)
        nib.save(prediction_nifti, output_file)
        
        print(f"Saved reprojected prediction to {output_file}")
        
        # Save probability volume if available
        if os.path.exists(os.path.join(os.path.dirname(predictions_file), f"{patient_id}_scores.npy")):
            scores = np.load(os.path.join(os.path.dirname(predictions_file), f"{patient_id}_scores.npy"))
            reprojected_probs = reproject_predictions(scores, segments, volume_shape)
            
            prob_file = os.path.join(output_dir, f"{patient_id}_probability_volume.nii.gz")
            prob_nifti = nib.Nifti1Image(reprojected_probs, t1_nifti.affine, t1_nifti.header)
            nib.save(prob_nifti, prob_file)
            
            print(f"Saved probability volume to {prob_file}")
        
        return True
    
    except Exception as e:
        print(f"Error processing {patient_id}: {str(e)}")
        return False

def main():
    parser = argparse.ArgumentParser(description='Reproject graph predictions to 3D volumes')
    parser.add_argument('--predictions_dir', type=str, required=True, help='Directory with prediction files')
    parser.add_argument('--graphs_dir', type=str, required=True, help='Directory with original graph files')
    parser.add_argument('--preprocessed_dir', type=str, required=True, help='Directory with preprocessed .npz files')
    parser.add_argument('--output_dir', type=str, required=True, help='Output directory for volumes')
    parser.add_argument('--n_superpixels', type=int, default=400, help='Number of superpixels per slice')
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Find all prediction files
    pred_files = glob.glob(os.path.join(args.predictions_dir, "*", "*_predictions.npy"))
    
    if not pred_files:
        print(f"No prediction files found in {args.predictions_dir}")
        return
    
    print(f"Found {len(pred_files)} prediction files")
    
    # Process each patient
    for pred_file in pred_files:
        patient_id = os.path.basename(pred_file).split('_predictions')[0]
        
        # Find corresponding segments file
        segments_file = os.path.join(args.graphs_dir, patient_id, f"{patient_id}_segments_{args.n_superpixels}.npy")
        
        # Find corresponding preprocessed file
        preprocessed_file = os.path.join(args.preprocessed_dir, patient_id, f"{patient_id}_preprocessed.npz")
        
        if not os.path.exists(segments_file) or not os.path.exists(preprocessed_file):
            print(f"Missing files for patient {patient_id}")
            continue
        
        # Process patient
        process_patient(patient_id, pred_file, segments_file, preprocessed_file, args.output_dir)

if __name__ == "__main__":
    start_time = time.time()
    main()
    print(f"Total reprojection time: {(time.time() - start_time) / 60:.2f} minutes")