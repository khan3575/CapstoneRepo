#!/usr/bin/env python3
"""
Demo: Visualize T1 MRI in 3 orthogonal axes (Axial, Sagittal, Coronal)
Shows cross-sectional views through a tumor point
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


def visualize_3_axes(t1, seg, patient_id, output_path):
    """Create 3-axis visualization (Axial, Sagittal, Coronal)."""
    # Find tumor center
    x, y, z = find_tumor_center(seg)

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # Axial view (horizontal slice, looking down from top)
    # This is the x-y plane at depth z
    axial_img = t1[:, :, z]
    axial_seg = seg[:, :, z]

    axes[0].imshow(axial_img.T, cmap='gray', origin='lower')
    axes[0].imshow(np.ma.masked_where(axial_seg == 0, axial_seg).T,
                   cmap='Reds', alpha=0.5, origin='lower')
    axes[0].axhline(y=y, color='cyan', linestyle='--', linewidth=1, alpha=0.7)
    axes[0].axvline(x=x, color='cyan', linestyle='--', linewidth=1, alpha=0.7)
    axes[0].set_title(f'Axial View (Horizontal)\nSlice z={z}',
                      fontsize=14, fontweight='bold')
    axes[0].axis('off')

    # Sagittal view (side view, looking from the side)
    # This is the y-z plane at position x
    sagittal_img = t1[x, :, :]
    sagittal_seg = seg[x, :, :]

    axes[1].imshow(sagittal_img.T, cmap='gray', origin='lower')
    axes[1].imshow(np.ma.masked_where(sagittal_seg == 0, sagittal_seg).T,
                   cmap='Reds', alpha=0.5, origin='lower')
    axes[1].axhline(y=z, color='cyan', linestyle='--', linewidth=1, alpha=0.7)
    axes[1].axvline(x=y, color='cyan', linestyle='--', linewidth=1, alpha=0.7)
    axes[1].set_title(f'Sagittal View (Side)\nSlice x={x}',
                      fontsize=14, fontweight='bold')
    axes[1].axis('off')

    # Coronal view (front view, looking from front)
    # This is the x-z plane at position y
    coronal_img = t1[:, y, :]
    coronal_seg = seg[:, y, :]

    axes[2].imshow(coronal_img.T, cmap='gray', origin='lower')
    axes[2].imshow(np.ma.masked_where(coronal_seg == 0, coronal_seg).T,
                   cmap='Reds', alpha=0.5, origin='lower')
    axes[2].axhline(y=z, color='cyan', linestyle='--', linewidth=1, alpha=0.7)
    axes[2].axvline(x=x, color='cyan', linestyle='--', linewidth=1, alpha=0.7)
    axes[2].set_title(f'Coronal View (Front)\nSlice y={y}',
                      fontsize=14, fontweight='bold')
    axes[2].axis('off')

    plt.suptitle(f'T1-weighted MRI: 3 Orthogonal Views\nPatient: {patient_id} | Center: ({x}, {y}, {z})',
                 fontsize=16, fontweight='bold', y=0.98)

    # Add a note about the crosshairs
    fig.text(0.5, 0.02, 'Cyan crosshairs show the intersection point across all 3 views | Red overlay: tumor segmentation',
             ha='center', fontsize=11, style='italic')

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✅ 3-axis visualization saved to: {output_path}")
    plt.close()


def main():
    # Load config
    from config import get_config
    config = get_config()

    # Get data directory
    data_dir = config.brats_2021_raw
    print(f"Data directory: {data_dir}")

    # Use the same patient as before
    patient_id = "BraTS2021_00000"
    patient_dir = os.path.join(data_dir, patient_id)

    print(f"Loading T1 and segmentation for: {patient_id}")

    # Load T1 and segmentation
    t1 = nib.load(os.path.join(patient_dir, f"{patient_id}_t1.nii.gz")).get_fdata()
    seg = nib.load(os.path.join(patient_dir, f"{patient_id}_seg.nii.gz")).get_fdata()

    print(f"Data shape: {t1.shape}")

    # Create output directory
    output_dir = Path(__file__).parent.parent / "visualizations"
    output_dir.mkdir(exist_ok=True)

    output_path = output_dir / f"t1_3axes_demo_{patient_id}.png"

    # Create visualization
    print("Creating 3-axis visualization...")
    visualize_3_axes(t1, seg, patient_id, output_path)

    print(f"\n✅ Done!")


if __name__ == "__main__":
    main()
