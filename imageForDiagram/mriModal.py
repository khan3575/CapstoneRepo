import SimpleITK as sitk
import numpy as np
import matplotlib.pyplot as plt
import os

# Configuration based on your paper's best-case patient [cite: 975]
base_path = "data/organized/BraTS2021_00134"
modalities = ["T1", "T1ce", "T2", "FLAIR"]
target_slice = 87 # Representative slice from Section 5.6 [cite: 975]
output_dir = "diagram_icons"
os.makedirs(output_dir, exist_ok=True)

def create_icon(modality_name, file_path):
    # Load the NIfTI volume
    img = sitk.ReadImage(file_path)
    data = sitk.GetArrayFromImage(img)
    
    # Extract the axial slice
    slice_2d = data[target_slice, :, :]
    
    # Simple normalization for clean visual icons (0-1 range)
    # Clipping at 99th percentile to remove outliers and improve contrast
    p99 = np.percentile(slice_2d, 99.5)
    slice_norm = np.clip(slice_2d, 0, p99)
    slice_norm = slice_norm / (p99 + 1e-8)

    # Crop to brain region (removes excessive black background)
    # BraTS volumes are 240x240; cropping to the center 180x180 usually looks better
    center = 120
    side = 90
    cropped = slice_norm[center-side:center+side, center-side:center+side]

    # Save as high-quality PNG
    plt.imsave(f"{output_dir}/{modality_name}_icon.png", cropped, cmap='gray')
    print(f"Generated: {modality_name}_icon.png")

# Execute for all modalities
for m in modalities:
    # Adjust file naming convention to match your actual folders
    file_path = os.path.join(base_path, f"{m}.nii.gz")
    if os.path.exists(file_path):
        create_icon(m.upper(), file_path)
    else:
        print(f"File not found: {file_path}")
        