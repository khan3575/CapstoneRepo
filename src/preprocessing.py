#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import glob
import numpy as np
import nibabel as nib
from tqdm import tqdm
import argparse
import shutil
import time
from concurrent.futures import ProcessPoolExecutor
from scipy.ndimage import binary_fill_holes
import SimpleITK as sitk
from pathlib import Path
import json

def load_slice_data(slice_path):
    """
    Load a single 2D slice from NPZ file.
    
    Args:
        slice_path: Path to slice NPZ file
    
    Returns:
        Dictionary with slice data
    """
    data = np.load(slice_path)
    
    slice_data = {
        'T1': data['T1'],
        'T1ce': data['T1ce'],
        'T2': data['T2'],
        'FLAIR': data['FLAIR'],
        'label': data['label'],
        'brain_mask': data['brain_mask'],
        'patient_id': str(data['patient_id']),
        'slice_idx': int(data['slice_idx'])
    }
    
    return slice_data

def load_patient_slices(patient_slice_dir):
    """
    Load all slices for a patient.
    
    Args:
        patient_slice_dir: Directory containing patient slices
    
    Returns:
        List of slice data dictionaries
    """
    slice_files = glob.glob(os.path.join(patient_slice_dir, "*_slice_*.npz"))
    slice_files.sort()  # Sort by filename to maintain order
    
    slices = []
    for slice_file in slice_files:
        slice_data = load_slice_data(slice_file)
        slices.append(slice_data)
    
    return slices

def get_slice_statistics(slice_dir):
    """
    Get statistics about extracted slices.
    
    Args:
        slice_dir: Directory containing extracted slices
    
    Returns:
        Dictionary with statistics
    """
    metadata_file = os.path.join(slice_dir, "overall_slice_metadata.json")
    
    if os.path.exists(metadata_file):
        with open(metadata_file, 'r') as f:
            return json.load(f)
    else:
        # Calculate statistics from files
        patient_dirs = [d for d in glob.glob(os.path.join(slice_dir, "*")) if os.path.isdir(d)]
        
        total_slices = 0
        tumor_slices = 0
        
        for patient_dir in patient_dirs:
            metadata_file = os.path.join(patient_dir, "*_slice_metadata.json")
            metadata_files = glob.glob(metadata_file)
            
            if metadata_files:
                with open(metadata_files[0], 'r') as f:
                    patient_metadata = json.load(f)
                
                total_slices += len(patient_metadata)
                tumor_slices += sum(1 for slice_info in patient_metadata if slice_info['has_tumor'])
        
        return {
            'total_patients': len(patient_dirs),
            'total_slices': total_slices,
            'tumor_slices': tumor_slices,
            'non_tumor_slices': total_slices - tumor_slices
        }

def download_brats(download_dir):
    """
    Note: BraTS dataset typically requires registration.
    This function provides instructions for manual download.
    """
    print("BraTS dataset requires registration and manual download.")
    print("Please download from: https://www.synapse.org/#!Synapse:syn51514105")
    print(f"After download, extract the files to: {download_dir}")
    print("Then run this script with --skip_download flag")
    return

def organize_files(input_dir, output_dir):
    """Organize BraTS files into a structured directory."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Find all patient directories
    patient_dirs = [d for d in glob.glob(os.path.join(input_dir, "*")) if os.path.isdir(d)]
    
    for patient_dir in tqdm(patient_dirs, desc="Organizing files"):
        patient_id = os.path.basename(patient_dir)
        patient_output_dir = os.path.join(output_dir, patient_id)
        os.makedirs(patient_output_dir, exist_ok=True)
        
        try:
            # Find MRI files - adapt these patterns to match your specific BraTS version
            t1_files = glob.glob(os.path.join(patient_dir, "*t1.nii.gz"))
            t1ce_files = glob.glob(os.path.join(patient_dir, "*t1ce.nii.gz"))
            t2_files = glob.glob(os.path.join(patient_dir, "*t2.nii.gz"))
            flair_files = glob.glob(os.path.join(patient_dir, "*flair.nii.gz"))
            seg_files = glob.glob(os.path.join(patient_dir, "*seg.nii.gz"))
            
            # Check if files were found
            if not all([t1_files, t1ce_files, t2_files, flair_files, seg_files]):
                print(f"Warning: Missing files for patient {patient_id}, trying alternative patterns...")
                # Alternative patterns for BraTS 2023
                t1_files = glob.glob(os.path.join(patient_dir, "*_t1n.nii.gz"))
                t1ce_files = glob.glob(os.path.join(patient_dir, "*_t1c.nii.gz"))
                t2_files = glob.glob(os.path.join(patient_dir, "*_t2w.nii.gz"))
                flair_files = glob.glob(os.path.join(patient_dir, "*_t2f.nii.gz"))
                seg_files = glob.glob(os.path.join(patient_dir, "*_seg.nii.gz"))
            
            # Copy files with standardized names
            shutil.copy(t1_files[0], os.path.join(patient_output_dir, "T1.nii.gz"))
            shutil.copy(t1ce_files[0], os.path.join(patient_output_dir, "T1ce.nii.gz"))
            shutil.copy(t2_files[0], os.path.join(patient_output_dir, "T2.nii.gz"))
            shutil.copy(flair_files[0], os.path.join(patient_output_dir, "FLAIR.nii.gz"))
            shutil.copy(seg_files[0], os.path.join(patient_output_dir, "segmentation.nii.gz"))
            
        except (IndexError, FileNotFoundError) as e:
            print(f"Error processing patient {patient_id}: {str(e)}")
            print(f"Files in directory: {os.listdir(patient_dir)}")
            continue
    
    print(f"Files organized in {output_dir}")
    return output_dir

def resample_volume(sitk_image, new_spacing=[1.0, 1.0, 1.0], interpolator=sitk.sitkLinear):
    """Resample a SimpleITK image to new spacing."""
    original_spacing = sitk_image.GetSpacing()
    original_size = sitk_image.GetSize()
    
    # Calculate new size
    new_size = [
        int(round(original_size[0] * (original_spacing[0] / new_spacing[0]))),
        int(round(original_size[1] * (original_spacing[1] / new_spacing[1]))),
        int(round(original_size[2] * (original_spacing[2] / new_spacing[2])))
    ]
    
    # Use the SimpleITK resample filter
    resample = sitk.ResampleImageFilter()
    resample.SetOutputSpacing(new_spacing)
    resample.SetSize(new_size)
    resample.SetOutputDirection(sitk_image.GetDirection())
    resample.SetOutputOrigin(sitk_image.GetOrigin())
    resample.SetTransform(sitk.Transform())
    resample.SetDefaultPixelValue(sitk_image.GetPixelIDValue())
    resample.SetInterpolator(interpolator)
    
    return resample.Execute(sitk_image)

def crop_pad_volume(sitk_image, target_shape=(240, 240, 155)):
    """Crop or pad a SimpleITK image to the target shape."""
    # Convert to numpy for easier manipulation
    np_img = sitk.GetArrayFromImage(sitk_image)
    current_shape = np_img.shape
    
    # SimpleITK uses (x,y,z) order, numpy uses (z,y,x) order
    target_shape_np = (target_shape[2], target_shape[1], target_shape[0])
    
    # Initialize result array
    result = np.zeros(target_shape_np, dtype=np_img.dtype)
    
    # Calculate crop/pad dimensions
    crop_pad = [(0, 0), (0, 0), (0, 0)]
    for i in range(3):
        if current_shape[i] > target_shape_np[i]:
            # Need to crop
            crop = current_shape[i] - target_shape_np[i]
            crop_pad[i] = (crop // 2, crop - (crop // 2))
        else:
            # Need to pad
            pad = target_shape_np[i] - current_shape[i]
            crop_pad[i] = (-pad // 2, -(pad - (pad // 2)))
    
    # Get the slices for cropping/padding
    src_slices = tuple(slice(max(0, -crop_pad[i][0]), 
                              min(current_shape[i], current_shape[i] - crop_pad[i][1])) 
                        for i in range(3))
    
    dst_slices = tuple(slice(max(0, crop_pad[i][0]), 
                              min(target_shape_np[i], target_shape_np[i] + crop_pad[i][1])) 
                        for i in range(3))
    
    # Perform the crop/pad operation
    result[dst_slices] = np_img[src_slices]
    
    # Convert back to SimpleITK
    result_sitk = sitk.GetImageFromArray(result)
    result_sitk.SetSpacing(sitk_image.GetSpacing())
    result_sitk.SetOrigin(sitk_image.GetOrigin())
    result_sitk.SetDirection(sitk_image.GetDirection())
    
    return result_sitk

def skull_strip(t1_sitk, t1ce_sitk, t2_sitk, flair_sitk):
    """
    Apply skull stripping using a simple threshold-based approach.
    This is a simplified version - in production, consider using HD-BET or ANTsPy.
    """
    # Create brain mask from FLAIR (usually provides good brain/non-brain contrast)
    flair_np = sitk.GetArrayFromImage(flair_sitk)
    
    # Otsu thresholding to separate brain from background
    otsu_filter = sitk.OtsuThresholdImageFilter()
    otsu_filter.SetInsideValue(0)
    otsu_filter.SetOutsideValue(1)
    mask_sitk = otsu_filter.Execute(flair_sitk)
    
    # Convert to numpy for morphological operations
    mask_np = sitk.GetArrayFromImage(mask_sitk)
    
    # Fill holes in the mask
    mask_np = binary_fill_holes(mask_np).astype(np.uint8)
    
    # Keep the largest connected component
    cc_filter = sitk.ConnectedComponentImageFilter()
    mask_cc = cc_filter.Execute(sitk.GetImageFromArray(mask_np))
    
    # Get the largest component
    lm_filter = sitk.LabelShapeStatisticsImageFilter()
    lm_filter.Execute(mask_cc)
    largest_label = 0
    largest_count = 0
    for label in range(1, cc_filter.GetObjectCount() + 1):
        size = lm_filter.GetNumberOfPixels(label)
        if size > largest_count:
            largest_count = size
            largest_label = label
    
    # Keep only the largest component
    mask_largest = sitk.GetArrayFromImage(mask_cc) == largest_label
    
    # Apply the mask to all modalities
    t1_stripped = sitk.GetArrayFromImage(t1_sitk) * mask_largest
    t1ce_stripped = sitk.GetArrayFromImage(t1ce_sitk) * mask_largest
    t2_stripped = sitk.GetArrayFromImage(t2_sitk) * mask_largest
    flair_stripped = flair_np * mask_largest
    
    # Convert back to SimpleITK
    t1_stripped_sitk = sitk.GetImageFromArray(t1_stripped)
    t1_stripped_sitk.CopyInformation(t1_sitk)
    
    t1ce_stripped_sitk = sitk.GetImageFromArray(t1ce_stripped)
    t1ce_stripped_sitk.CopyInformation(t1ce_sitk)
    
    t2_stripped_sitk = sitk.GetImageFromArray(t2_stripped)
    t2_stripped_sitk.CopyInformation(t2_sitk)
    
    flair_stripped_sitk = sitk.GetImageFromArray(flair_stripped)
    flair_stripped_sitk.CopyInformation(flair_sitk)
    
    # Create mask as SimpleITK image
    mask_sitk = sitk.GetImageFromArray(mask_largest.astype(np.uint8))
    mask_sitk.CopyInformation(t1_sitk)
    
    return t1_stripped_sitk, t1ce_stripped_sitk, t2_stripped_sitk, flair_stripped_sitk, mask_sitk

def normalize_intensity(image_sitk, mask_sitk, percentile_low=0.5, percentile_high=99.5):
    """Normalize image intensity by clipping to percentiles and z-score normalization."""
    # Convert to numpy
    image_np = sitk.GetArrayFromImage(image_sitk)
    mask_np = sitk.GetArrayFromImage(mask_sitk)
    
    # Apply mask
    masked_img = image_np * (mask_np > 0)
    
    # Get non-zero values for percentile calculation
    nonzero_values = masked_img[masked_img > 0]
    
    if len(nonzero_values) == 0:
        print("Warning: No non-zero values in masked image.")
        return image_sitk
    
    # Calculate percentiles
    p_low = np.percentile(nonzero_values, percentile_low)
    p_high = np.percentile(nonzero_values, percentile_high)
    
    # Clip intensities
    clipped = np.clip(masked_img, p_low, p_high)
    
    # Z-score normalization within the brain mask
    mean_intensity = np.mean(clipped[mask_np > 0])
    std_intensity = np.std(clipped[mask_np > 0])
    
    if std_intensity > 0:
        normalized = (clipped - mean_intensity) / std_intensity
    else:
        print("Warning: Zero standard deviation in brain region.")
        normalized = clipped - mean_intensity
    
    # Create final image (zero outside brain mask)
    final = normalized * (mask_np > 0)
    
    # Convert back to SimpleITK
    normalized_sitk = sitk.GetImageFromArray(final)
    normalized_sitk.CopyInformation(image_sitk)
    
    return normalized_sitk

def extract_2d_slices(volume_data, output_dir, patient_id, slice_range=None, skip_empty=True):
    """
    Extract 2D slices from preprocessed 3D volume data.
    
    Args:
        volume_data: Dict with volume arrays (T1, T1ce, T2, FLAIR, label, brain_mask)
        output_dir: Directory to save slices
        patient_id: Patient ID for naming
        slice_range: Tuple (start, end) for slice range, None for all slices
        skip_empty: Skip slices with no brain tissue
    
    Returns:
        List of slice information dictionaries
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Get volume dimensions
    z_dim = volume_data['T1'].shape[0]
    
    # Determine slice range
    if slice_range is None:
        start_slice, end_slice = 0, z_dim
    else:
        start_slice, end_slice = slice_range
        start_slice = max(0, start_slice)
        end_slice = min(z_dim, end_slice)
    
    slice_info_list = []
    
    for z in range(start_slice, end_slice):
        # Extract slice data
        slice_data = {
            'T1': volume_data['T1'][z],
            'T1ce': volume_data['T1ce'][z],
            'T2': volume_data['T2'][z],
            'FLAIR': volume_data['FLAIR'][z],
            'label': volume_data['label'][z],
            'brain_mask': volume_data['brain_mask'][z]
        }
        
        # Skip empty slices if requested
        if skip_empty and np.sum(slice_data['brain_mask']) == 0:
            continue
        
        # Create slice info
        slice_info = {
            'patient_id': patient_id,
            'slice_idx': z,
            'slice_data': slice_data,
            'has_tumor': np.sum(slice_data['label']) > 0,
            'brain_pixels': np.sum(slice_data['brain_mask']),
            'tumor_pixels': np.sum(slice_data['label'])
        }
        
        # Save slice as NPZ file
        slice_filename = f"{patient_id}_slice_{z:03d}.npz"
        slice_path = os.path.join(output_dir, slice_filename)
        
        np.savez_compressed(
            slice_path,
            **slice_data,
            patient_id=patient_id,
            slice_idx=z
        )
        
        slice_info['slice_path'] = slice_path
        slice_info_list.append(slice_info)
    
    return slice_info_list

def extract_slices_from_volume(preprocessed_file, output_dir, slice_range=None):
    """
    Extract 2D slices from a preprocessed volume file.
    
    Args:
        preprocessed_file: Path to preprocessed NPZ file
        output_dir: Directory to save slices
        slice_range: Tuple (start, end) for slice range
    
    Returns:
        List of slice information dictionaries
    """
    # Load preprocessed volume
    volume_data = np.load(preprocessed_file)
    
    # Get patient ID from filename
    patient_id = Path(preprocessed_file).stem.replace('_preprocessed', '')
    
    # Create patient-specific slice directory
    patient_slice_dir = os.path.join(output_dir, patient_id)
    os.makedirs(patient_slice_dir, exist_ok=True)
    
    # Extract slices
    slice_info_list = extract_2d_slices(
        volume_data, patient_slice_dir, patient_id, slice_range
    )
    
    # Save slice metadata
    metadata_file = os.path.join(patient_slice_dir, f"{patient_id}_slice_metadata.json")
    import json
    
    # Convert numpy types to Python types for JSON serialization
    metadata = []
    for slice_info in slice_info_list:
        metadata_entry = {
            'patient_id': slice_info['patient_id'],
            'slice_idx': int(slice_info['slice_idx']),
            'slice_path': slice_info['slice_path'],
            'has_tumor': bool(slice_info['has_tumor']),
            'brain_pixels': int(slice_info['brain_pixels']),
            'tumor_pixels': int(slice_info['tumor_pixels'])
        }
        metadata.append(metadata_entry)
    
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"Extracted {len(slice_info_list)} slices for patient {patient_id}")
    return slice_info_list

def preprocess_patient(patient_dir, output_dir):
    """Preprocess a single patient's data."""
    patient_id = os.path.basename(patient_dir)
    output_patient_dir = os.path.join(output_dir, patient_id)
    os.makedirs(output_patient_dir, exist_ok=True)
    
    print(f"Processing patient {patient_id}")
    
    # Load the images
    t1_path = os.path.join(patient_dir, "T1.nii.gz")
    t1ce_path = os.path.join(patient_dir, "T1ce.nii.gz")
    t2_path = os.path.join(patient_dir, "T2.nii.gz")
    flair_path = os.path.join(patient_dir, "FLAIR.nii.gz")
    seg_path = os.path.join(patient_dir, "segmentation.nii.gz")
    
    # Load as SimpleITK images
    t1_sitk = sitk.ReadImage(t1_path)
    t1ce_sitk = sitk.ReadImage(t1ce_path)
    t2_sitk = sitk.ReadImage(t2_path)
    flair_sitk = sitk.ReadImage(flair_path)
    seg_sitk = sitk.ReadImage(seg_path)
    
    # Step 1: Resample all images to 1x1x1 mm³
    print(f"  Resampling {patient_id} to 1x1x1 mm³")
    t1_resampled = resample_volume(t1_sitk)
    t1ce_resampled = resample_volume(t1ce_sitk)
    t2_resampled = resample_volume(t2_sitk)
    flair_resampled = resample_volume(flair_sitk)
    # Use nearest neighbor for segmentation to preserve labels
    seg_resampled = resample_volume(seg_sitk, interpolator=sitk.sitkNearestNeighbor)
    
    # Step 2: Crop/pad to 240x240x155
    print(f"  Cropping/padding {patient_id} to 240x240x155")
    t1_cropped = crop_pad_volume(t1_resampled)
    t1ce_cropped = crop_pad_volume(t1ce_resampled)
    t2_cropped = crop_pad_volume(t2_resampled)
    flair_cropped = crop_pad_volume(flair_resampled)
    seg_cropped = crop_pad_volume(seg_resampled)
    
    # Step 3: Skull stripping
    print(f"  Applying skull stripping for {patient_id}")
    t1_stripped, t1ce_stripped, t2_stripped, flair_stripped, brain_mask = skull_strip(
        t1_cropped, t1ce_cropped, t2_cropped, flair_cropped
    )
    
    # Step 4: Intensity normalization
    print(f"  Normalizing intensities for {patient_id}")
    t1_normalized = normalize_intensity(t1_stripped, brain_mask)
    t1ce_normalized = normalize_intensity(t1ce_stripped, brain_mask)
    t2_normalized = normalize_intensity(t2_stripped, brain_mask)
    flair_normalized = normalize_intensity(flair_stripped, brain_mask)
    
    # Create whole tumor mask (label 1, 2, or 4 in BraTS)
    seg_array = sitk.GetArrayFromImage(seg_cropped)
    whole_tumor = (seg_array > 0).astype(np.uint8)  # Any non-zero label is tumor
    
    # Convert all to numpy arrays for saving
    t1_final = sitk.GetArrayFromImage(t1_normalized)
    t1ce_final = sitk.GetArrayFromImage(t1ce_normalized)
    t2_final = sitk.GetArrayFromImage(t2_normalized)
    flair_final = sitk.GetArrayFromImage(flair_normalized)
    brain_mask_final = sitk.GetArrayFromImage(brain_mask)
    
    # Step 5: Save as NPZ
    output_file = os.path.join(output_patient_dir, f"{patient_id}_preprocessed.npz")
    np.savez_compressed(
        output_file,
        T1=t1_final,
        T1ce=t1ce_final,
        T2=t2_final,
        FLAIR=flair_final,
        label=whole_tumor,
        brain_mask=brain_mask_final
    )
    
    print(f"  Saved preprocessed data for {patient_id} to {output_file}")
    
    # Save also as nifti for visualization
    sitk.WriteImage(t1_normalized, os.path.join(output_patient_dir, "T1_preprocessed.nii.gz"))
    sitk.WriteImage(t1ce_normalized, os.path.join(output_patient_dir, "T1ce_preprocessed.nii.gz"))
    sitk.WriteImage(t2_normalized, os.path.join(output_patient_dir, "T2_preprocessed.nii.gz"))
    sitk.WriteImage(flair_normalized, os.path.join(output_patient_dir, "FLAIR_preprocessed.nii.gz"))
    
    # Save label
    label_sitk = sitk.GetImageFromArray(whole_tumor)
    label_sitk.CopyInformation(t1_normalized)
    sitk.WriteImage(label_sitk, os.path.join(output_patient_dir, "whole_tumor_mask.nii.gz"))
    
    return output_file

def process_slices_for_patients(preprocessed_dir, slice_output_dir, slice_range=None, num_workers=4):
    """
    Extract 2D slices from all preprocessed patient volumes.
    
    Args:
        preprocessed_dir: Directory containing preprocessed NPZ files
        slice_output_dir: Directory to save extracted slices
        slice_range: Tuple (start, end) for slice range
        num_workers: Number of parallel workers
    
    Returns:
        Dictionary with slice information for all patients
    """
    # Find all preprocessed files
    preprocessed_files = glob.glob(os.path.join(preprocessed_dir, "*", "*_preprocessed.npz"))
    
    if len(preprocessed_files) == 0:
        print(f"No preprocessed files found in {preprocessed_dir}")
        return {}
    
    print(f"Found {len(preprocessed_files)} preprocessed files")
    
    # Create output directory
    os.makedirs(slice_output_dir, exist_ok=True)
    
    all_slice_info = {}
    
    # Process files in parallel
    if num_workers > 1:
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            futures = [
                executor.submit(extract_slices_from_volume, file_path, slice_output_dir, slice_range)
                for file_path in preprocessed_files
            ]
            
            for i, future in enumerate(tqdm(futures, desc="Extracting slices")):
                try:
                    slice_info_list = future.result()
                    patient_id = Path(preprocessed_files[i]).stem.replace('_preprocessed', '')
                    all_slice_info[patient_id] = slice_info_list
                except Exception as e:
                    patient_id = Path(preprocessed_files[i]).stem.replace('_preprocessed', '')
                    print(f"Error processing slices for patient {patient_id}: {str(e)}")
    else:
        # Process files sequentially
        for file_path in tqdm(preprocessed_files, desc="Extracting slices"):
            try:
                slice_info_list = extract_slices_from_volume(file_path, slice_output_dir, slice_range)
                patient_id = Path(file_path).stem.replace('_preprocessed', '')
                all_slice_info[patient_id] = slice_info_list
            except Exception as e:
                patient_id = Path(file_path).stem.replace('_preprocessed', '')
                print(f"Error processing slices for patient {patient_id}: {str(e)}")
    
    # Save overall metadata
    total_slices = sum(len(slice_list) for slice_list in all_slice_info.values())
    tumor_slices = sum(
        sum(1 for slice_info in slice_list if slice_info['has_tumor'])
        for slice_list in all_slice_info.values()
    )
    
    overall_metadata = {
        'total_patients': len(all_slice_info),
        'total_slices': total_slices,
        'tumor_slices': tumor_slices,
        'non_tumor_slices': total_slices - tumor_slices,
        'slice_range': slice_range,
        'patients': list(all_slice_info.keys())
    }
    
    metadata_file = os.path.join(slice_output_dir, "overall_slice_metadata.json")
    import json
    with open(metadata_file, 'w') as f:
        json.dump(overall_metadata, f, indent=2)
    
    print(f"Slice extraction complete:")
    print(f"  Total patients: {overall_metadata['total_patients']}")
    print(f"  Total slices: {overall_metadata['total_slices']}")
    print(f"  Tumor slices: {overall_metadata['tumor_slices']}")
    print(f"  Non-tumor slices: {overall_metadata['non_tumor_slices']}")
    
    return all_slice_info

def main():
    parser = argparse.ArgumentParser(description='BraTS Dataset Preprocessing')
    parser.add_argument('--input_dir', type=str, required=True, help='Input directory with BraTS data')
    parser.add_argument('--output_dir', type=str, required=True, help='Output directory for preprocessed data')
    parser.add_argument('--skip_download', action='store_true', help='Skip download instructions')
    parser.add_argument('--skip_organize', action='store_true', help='Skip file organization')
    parser.add_argument('--extract_slices', action='store_true', help='Extract 2D slices from preprocessed volumes')
    parser.add_argument('--slice_range', type=int, nargs=2, metavar=('START', 'END'), 
                       help='Range of slices to extract (e.g., --slice_range 60 120)')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of parallel workers')
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Step 1: Download instructions
    if not args.skip_download:
        download_brats(args.input_dir)
    
    # Step 2: Organize files
    if not args.skip_organize:
        organized_dir = organize_files(args.input_dir, os.path.join(args.output_dir, "organized"))
    else:
        organized_dir = args.input_dir
    
    # Step 3: Preprocess each patient
    patient_dirs = [d for d in glob.glob(os.path.join(organized_dir, "*")) if os.path.isdir(d)]
    
    preprocessed_dir = os.path.join(args.output_dir, "preprocessed")
    os.makedirs(preprocessed_dir, exist_ok=True)
    
    # Process patients in parallel
    if args.num_workers > 1:
        with ProcessPoolExecutor(max_workers=args.num_workers) as executor:
            futures = [executor.submit(preprocess_patient, patient_dir, preprocessed_dir) 
                      for patient_dir in patient_dirs]
            
            for i, future in enumerate(tqdm(futures, desc="Preprocessing patients")):
                try:
                    future.result()
                except Exception as e:
                    print(f"Error processing patient {os.path.basename(patient_dirs[i])}: {str(e)}")
    else:
        # Process patients sequentially
        for patient_dir in tqdm(patient_dirs, desc="Preprocessing patients"):
            try:
                preprocess_patient(patient_dir, preprocessed_dir)
            except Exception as e:
                print(f"Error processing patient {os.path.basename(patient_dir)}: {str(e)}")
    
    print(f"Preprocessing complete. Preprocessed files saved to {preprocessed_dir}")
    
    # Step 4: Extract 2D slices if requested
    if args.extract_slices:
        print("Starting 2D slice extraction...")
        slice_output_dir = os.path.join(args.output_dir, "slices")
        
        slice_range = tuple(args.slice_range) if args.slice_range else None
        if slice_range:
            print(f"Extracting slices in range: {slice_range[0]} to {slice_range[1]}")
        else:
            print("Extracting all slices")
        
        all_slice_info = process_slices_for_patients(
            preprocessed_dir, slice_output_dir, slice_range, args.num_workers
        )
        
        print(f"2D slice extraction complete. Slices saved to {slice_output_dir}")

if __name__ == "__main__":
    start_time = time.time()
    main()
    print(f"Total preprocessing time: {(time.time() - start_time) / 60:.2f} minutes")