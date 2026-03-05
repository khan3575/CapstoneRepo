#!/usr/bin/env python3
"""
Visualize MRI Modalities for BraTS Dataset
Creates a side-by-side visualization of T1, T1ce, T2, FLAIR, and segmentation mask
"""

import os
import sys
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
from pathlib import Path

# Add src to path for config
sys.path.append(str(Path(__file__).parent.parent / "src"))


def find_patient_with_tumor(data_dir, num_check=10):
    """Find a patient that has visible tumor in the middle slices."""
    patients = sorted([d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))])

    print(f"Checking {num_check} patients to find one with visible tumor...")

    for i, patient_id in enumerate(patients[:num_check]):
        patient_dir = os.path.join(data_dir, patient_id)
        seg_file = os.path.join(patient_dir, f"{patient_id}_seg.nii.gz")

        if os.path.exists(seg_file):
            seg = nib.load(seg_file).get_fdata()
            mid_slice = seg.shape[2] // 2

            # Check if there's tumor in the middle region
            tumor_volume = np.sum(seg[:, :, mid_slice-10:mid_slice+10] > 0)

            if tumor_volume > 500:  # Significant tumor present
                print(f"Found patient with visible tumor: {patient_id}")
                return patient_id

    # Fallback to first patient
    print(f"Using first patient: {patients[0]}")
    return patients[0]


def load_patient_data(data_dir, patient_id):
    """Load all modalities and segmentation for a patient."""
    patient_dir = os.path.join(data_dir, patient_id)

    # Load all modalities
    t1 = nib.load(os.path.join(patient_dir, f"{patient_id}_t1.nii.gz")).get_fdata()
    t1ce = nib.load(os.path.join(patient_dir, f"{patient_id}_t1ce.nii.gz")).get_fdata()
    t2 = nib.load(os.path.join(patient_dir, f"{patient_id}_t2.nii.gz")).get_fdata()
    flair = nib.load(os.path.join(patient_dir, f"{patient_id}_flair.nii.gz")).get_fdata()
    seg = nib.load(os.path.join(patient_dir, f"{patient_id}_seg.nii.gz")).get_fdata()

    return t1, t1ce, t2, flair, seg


def find_best_slice(seg):
    """Find the slice with maximum tumor area."""
    tumor_per_slice = [np.sum(seg[:, :, i] > 0) for i in range(seg.shape[2])]
    best_slice = np.argmax(tumor_per_slice)
    print(f"Best slice for visualization: {best_slice} (tumor pixels: {tumor_per_slice[best_slice]})")
    return best_slice


def visualize_modalities(t1, t1ce, t2, flair, seg, slice_idx, patient_id, output_path):
    """Create side-by-side visualization of all modalities."""
    fig, axes = plt.subplots(1, 5, figsize=(25, 5))

    # Extract the slice
    t1_slice = t1[:, :, slice_idx]
    t1ce_slice = t1ce[:, :, slice_idx]
    t2_slice = t2[:, :, slice_idx]
    flair_slice = flair[:, :, slice_idx]
    seg_slice = seg[:, :, slice_idx]

    # Display each modality
    axes[0].imshow(t1_slice.T, cmap='gray', origin='lower')
    axes[0].set_title('T1-weighted\n(Anatomical Structure)', fontsize=14, fontweight='bold')
    axes[0].axis('off')

    axes[1].imshow(t1ce_slice.T, cmap='gray', origin='lower')
    axes[1].set_title('T1ce (Contrast-Enhanced)\n(Active Tumor Core)', fontsize=14, fontweight='bold')
    axes[1].axis('off')

    axes[2].imshow(t2_slice.T, cmap='gray', origin='lower')
    axes[2].set_title('T2-weighted\n(Fluid/Edema)', fontsize=14, fontweight='bold')
    axes[2].axis('off')

    axes[3].imshow(flair_slice.T, cmap='gray', origin='lower')
    axes[3].set_title('FLAIR\n(Edema, CSF Suppressed)', fontsize=14, fontweight='bold')
    axes[3].axis('off')

    # Overlay segmentation on FLAIR for better visualization
    axes[4].imshow(flair_slice.T, cmap='gray', origin='lower')
    axes[4].imshow(np.ma.masked_where(seg_slice == 0, seg_slice).T,
                   cmap='Reds', alpha=0.6, origin='lower', vmin=0, vmax=3)
    axes[4].set_title('Ground Truth Segmentation\n(Expert Annotation)', fontsize=14, fontweight='bold')
    axes[4].axis('off')

    plt.suptitle(f'BraTS 2021 Patient: {patient_id} (Axial Slice: {slice_idx})',
                 fontsize=16, fontweight='bold', y=0.98)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Visualization saved to: {output_path}")
    plt.close()


def main():
    # Load config
    from config import get_config
    config = get_config()

    # Get data directory
    data_dir = config.brats_2021_raw
    print(f"Data directory: {data_dir}")

    # Find a good patient
    patient_id = find_patient_with_tumor(data_dir)

    # Load patient data
    print(f"Loading data for patient: {patient_id}")
    t1, t1ce, t2, flair, seg = load_patient_data(data_dir, patient_id)
    print(f"Data shape: {t1.shape}")

    # Find best slice
    slice_idx = find_best_slice(seg)

    # Create output directory
    output_dir = Path(__file__).parent.parent / "visualizations"
    output_dir.mkdir(exist_ok=True)

    output_path = output_dir / f"mri_modalities_{patient_id}.png"

    # Create visualization
    print("Creating visualization...")
    visualize_modalities(t1, t1ce, t2, flair, seg, slice_idx, patient_id, output_path)

    print(f"\n✅ Done! Visualization saved to:")
    print(f"   {output_path}")


if __name__ == "__main__":
    main()
