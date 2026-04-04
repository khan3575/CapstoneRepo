import SimpleITK as sitk
import numpy as np
import matplotlib.pyplot as plt
from skimage.segmentation import slic, mark_boundaries
from skimage import graph
import os

# =============================
# Configuration
# =============================
base_path = "data/organized/BraTS2021_00134"
t1ce_path = os.path.join(base_path, "T1ce.nii.gz")
target_slice = 87
output_dir = "diagram_stage2"
os.makedirs(output_dir, exist_ok=True)

# =============================
# 1. Load MRI Slice
# =============================
img = sitk.ReadImage(t1ce_path)
volume = sitk.GetArrayFromImage(img)

# Extract 2D slice
data = volume[target_slice, :, :]

# =============================
# 2. Normalize (0–1)
# =============================
p99 = np.percentile(data, 99.5)
data_norm = np.clip(data, 0, p99) / (p99 + 1e-8)

# =============================
# 3. SLIC Superpixels
# =============================
segments = slic(
    data_norm,
    n_segments=200,
    compactness=0.1,
    sigma=0.3,
    start_label=1,
    channel_axis=None   # IMPORTANT for grayscale
)

# =============================
# 4. Save Superpixel Boundaries
# =============================
boundary_img = mark_boundaries(data_norm, segments, color=(1, 0, 0))

plt.figure(figsize=(6, 6))
plt.imshow(boundary_img)
plt.axis('off')
plt.tight_layout()

plt.savefig(
    f"{output_dir}/slic_boundaries.png",
    dpi=300,
    bbox_inches='tight',
    pad_inches=0
)
plt.close()

# =============================
# 5. Build RAG Graph
# =============================
g = graph.rag_mean_color(data_norm, segments)

# FIX: convert grayscale → RGB (required by show_rag)
data_rgb = np.stack([data_norm]*3, axis=-1)

# =============================
# 6. Visualize Graph
# =============================
fig, ax = plt.subplots(figsize=(6, 6))

graph.show_rag(
    segments,
    g,
    data_rgb,
    ax=ax,
    edge_cmap='viridis'
)

# Clean diagram look
ax.axis('off')
plt.tight_layout()

plt.savefig(
    f"{output_dir}/graph_topology.png",
    dpi=300,
    bbox_inches='tight',
    pad_inches=0,
    transparent=True
)
plt.close()

# =============================
# Done
# =============================
print(f"✅ Generated outputs in: {output_dir}")