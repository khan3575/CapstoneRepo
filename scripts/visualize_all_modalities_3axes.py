#!/usr/bin/env python3
"""
Complete Visualization: All 4 MRI Modalities in 3 Orthogonal Axes
Shows T1, T1ce, T2, FLAIR each in Axial, Sagittal, and Coronal views
"""

import os
import sys
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
from pathlib import Path

# Add src to path for config
sys.path.append(str(Path(__file__).parent.parent / "src"))


def find_tumor_center(seg):
    """Find the center of mass of the tumor."""
    tumor_coords = np.argwhere(seg > 0)
    if len(tumor_coords) == 0:
        # No tumor, use center of volume
        return seg.shape[0]//2, seg.shape[1]//2, seg.shape[2]//2

    center = tumor_coords.mean(axis=0).astype(int)
    print(f"Tumor center found at: x={center[0]}, y={center[1]}, z={center[2]}")
    return center[0], center[1], center[2]


def find_patient_with_tumor(data_dir, num_check=10):
    """Find a patient that has visible tumor."""
    patients = sorted([d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))])

    print(f"Checking {num_check} patients to find one with visible tumor...")

    for i, patient_id in enumerate(patients[:num_check]):
        patient_dir = os.path.join(data_dir, patient_id)
        seg_file = os.path.join(patient_dir, f"{patient_id}_seg.nii.gz")

        if os.path.exists(seg_file):
            seg = nib.load(seg_file).get_fdata()
            tumor_volume = np.sum(seg > 0)

            if tumor_volume > 1000:  # Significant tumor present
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


def visualize_all_modalities_3axes(t1, t1ce, t2, flair, seg, patient_id, output_path):
    """Create comprehensive 5x3 visualization grid."""
    # Find tumor center
    x, y, z = find_tumor_center(seg)

    # Create figure with 5 rows × 3 columns
    # Rows: T1, T1ce, T2, FLAIR, Ground Truth Segmentation
    # Columns: Axial, Sagittal, Coronal
    fig, axes = plt.subplots(5, 3, figsize=(18, 28))

    modalities = [
        (t1, 'T1-weighted'),
        (t1ce, 'T1ce (Contrast-Enhanced)'),
        (t2, 'T2-weighted'),
        (flair, 'FLAIR'),
    ]

    # First 4 rows: Show modalities WITHOUT segmentation overlay
    for row, (img, modality_name) in enumerate(modalities):
        # Axial view (horizontal slice, x-y plane)
        axial_img = img[:, :, z]
        axes[row, 0].imshow(axial_img.T, cmap='gray', origin='lower')
        axes[row, 0].axhline(y=y, color='cyan', linestyle='--', linewidth=0.8, alpha=0.7)
        axes[row, 0].axvline(x=x, color='cyan', linestyle='--', linewidth=0.8, alpha=0.7)

        if row == 0:
            axes[row, 0].set_title(f'Axial View (x-y plane)\nz={z}',
                                  fontsize=12, fontweight='bold')
        axes[row, 0].set_ylabel(modality_name, fontsize=12, fontweight='bold', rotation=0,
                                ha='right', va='center', labelpad=20)
        axes[row, 0].set_xlabel('x-axis →', fontsize=9)
        axes[row, 0].set_ylabel(modality_name + '\n\ny-axis →', fontsize=9, labelpad=10)
        axes[row, 0].tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)

        # Sagittal view (side view, y-z plane)
        sagittal_img = img[x, :, :]
        axes[row, 1].imshow(sagittal_img.T, cmap='gray', origin='lower')
        axes[row, 1].axhline(y=z, color='cyan', linestyle='--', linewidth=0.8, alpha=0.7)
        axes[row, 1].axvline(x=y, color='cyan', linestyle='--', linewidth=0.8, alpha=0.7)

        if row == 0:
            axes[row, 1].set_title(f'Sagittal View (y-z plane)\nx={x}',
                                  fontsize=12, fontweight='bold')
        axes[row, 1].set_xlabel('y-axis →', fontsize=9)
        axes[row, 1].set_ylabel('z-axis →', fontsize=9)
        axes[row, 1].tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)

        # Coronal view (front view, x-z plane)
        coronal_img = img[:, y, :]
        axes[row, 2].imshow(coronal_img.T, cmap='gray', origin='lower')
        axes[row, 2].axhline(y=z, color='cyan', linestyle='--', linewidth=0.8, alpha=0.7)
        axes[row, 2].axvline(x=x, color='cyan', linestyle='--', linewidth=0.8, alpha=0.7)

        if row == 0:
            axes[row, 2].set_title(f'Coronal View (x-z plane)\ny={y}',
                                  fontsize=12, fontweight='bold')
        axes[row, 2].set_xlabel('x-axis →', fontsize=9)
        axes[row, 2].set_ylabel('z-axis →', fontsize=9)
        axes[row, 2].tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)

    # Row 5: Ground Truth Segmentation (with overlay on FLAIR background)
    row = 4

    # Axial - segmentation
    axes[row, 0].imshow(flair[:, :, z].T, cmap='gray', origin='lower', alpha=0.7)
    axes[row, 0].imshow(np.ma.masked_where(seg[:, :, z] == 0, seg[:, :, z]).T,
                       cmap='Reds', alpha=0.8, origin='lower', vmin=0, vmax=3)
    axes[row, 0].axhline(y=y, color='cyan', linestyle='--', linewidth=0.8, alpha=0.7)
    axes[row, 0].axvline(x=x, color='cyan', linestyle='--', linewidth=0.8, alpha=0.7)
    axes[row, 0].set_ylabel('Ground Truth\nSegmentation\n\ny-axis →', fontsize=9, labelpad=10)
    axes[row, 0].set_xlabel('x-axis →', fontsize=9)
    axes[row, 0].tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)

    # Sagittal - segmentation
    axes[row, 1].imshow(flair[x, :, :].T, cmap='gray', origin='lower', alpha=0.7)
    axes[row, 1].imshow(np.ma.masked_where(seg[x, :, :] == 0, seg[x, :, :]).T,
                       cmap='Reds', alpha=0.8, origin='lower', vmin=0, vmax=3)
    axes[row, 1].axhline(y=z, color='cyan', linestyle='--', linewidth=0.8, alpha=0.7)
    axes[row, 1].axvline(x=y, color='cyan', linestyle='--', linewidth=0.8, alpha=0.7)
    axes[row, 1].set_xlabel('y-axis →', fontsize=9)
    axes[row, 1].set_ylabel('z-axis →', fontsize=9)
    axes[row, 1].tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)

    # Coronal - segmentation
    axes[row, 2].imshow(flair[:, y, :].T, cmap='gray', origin='lower', alpha=0.7)
    axes[row, 2].imshow(np.ma.masked_where(seg[:, y, :] == 0, seg[:, y, :]).T,
                       cmap='Reds', alpha=0.8, origin='lower', vmin=0, vmax=3)
    axes[row, 2].axhline(y=z, color='cyan', linestyle='--', linewidth=0.8, alpha=0.7)
    axes[row, 2].axvline(x=x, color='cyan', linestyle='--', linewidth=0.8, alpha=0.7)
    axes[row, 2].set_xlabel('x-axis →', fontsize=9)
    axes[row, 2].set_ylabel('z-axis →', fontsize=9)
    axes[row, 2].tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)

    plt.suptitle(f'BraTS 2021 - Multi-Modal MRI Visualization (3 Orthogonal Planes)\nPatient: {patient_id}',
                 fontsize=16, fontweight='bold', y=0.995)

    # Add legend at bottom
    fig.text(0.5, 0.008,
             'Cyan crosshairs: tumor center intersection | Red overlay: expert annotation (ground truth)',
             ha='center', fontsize=11, style='italic')

    plt.tight_layout(rect=[0, 0.015, 1, 0.992])
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✅ Complete visualization saved to: {output_path}")
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
    print(f"Loading all modalities for: {patient_id}")
    t1, t1ce, t2, flair, seg = load_patient_data(data_dir, patient_id)
    print(f"Data shape: {t1.shape}")

    # Create output directory
    output_dir = Path(__file__).parent.parent / "visualizations"
    output_dir.mkdir(exist_ok=True)

    output_path = output_dir / f"complete_4modalities_3axes_{patient_id}.png"

    # Create visualization
    print("Creating comprehensive 5×3 visualization...")
    visualize_all_modalities_3axes(t1, t1ce, t2, flair, seg, patient_id, output_path)

    print(f"\n✅ Done! Full visualization saved.")
    print(f"   File: {output_path}")
    print(f"\n   Layout: 5 rows (T1, T1ce, T2, FLAIR, Segmentation) × 3 columns (Axial, Sagittal, Coronal)")


if __name__ == "__main__":
    main()
