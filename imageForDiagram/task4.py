import SimpleITK as sitk
import numpy as np
import matplotlib.pyplot as plt
import os

# =============================
# Configuration (MATCHES Stage 2 STYLE)
# =============================
base_path = "data/organized/BraTS2021_00134"

flair_path = os.path.join(base_path, "FLAIR.nii.gz")
label_path = os.path.join(base_path, "segmentation.nii.gz")

target_slice = 87
output_dir = "diagram_stage4"
os.makedirs(output_dir, exist_ok=True)

# =============================
# Safety Check
# =============================
if not os.path.exists(flair_path):
    raise FileNotFoundError(f"FLAIR file not found: {flair_path}")

if not os.path.exists(label_path):
    raise FileNotFoundError(f"Label file not found: {label_path}")

# =============================
# 1. Load Data
# =============================
flair_volume = sitk.GetArrayFromImage(sitk.ReadImage(flair_path))
label_volume = sitk.GetArrayFromImage(sitk.ReadImage(label_path))

flair_img = flair_volume[target_slice]
label_img = label_volume[target_slice]

# =============================
# 2. Normalize Background
# =============================
p99 = np.percentile(flair_img, 99)
flair_norm = np.clip(flair_img, 0, p99)
flair_norm = flair_norm / (flair_norm.max() + 1e-8)

# =============================
# 3. Create Overlay
# =============================
plt.figure(figsize=(8, 8))
plt.imshow(flair_norm, cmap='gray')

# Mask background (0 = no tumor)
label_mask = np.ma.masked_where(label_img == 0, label_img)

# Overlay segmentation
plt.imshow(label_mask, cmap='jet', alpha=0.5)

plt.axis('off')

# =============================
# 4. Save Output
# =============================
plt.savefig(
    f"{output_dir}/final_segmentation.png",
    dpi=300,
    bbox_inches='tight',
    pad_inches=0
)

plt.close()

print(f"✅ Generated Stage 4 result in: {output_dir}")