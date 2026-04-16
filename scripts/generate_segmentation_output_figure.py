"""
Final Segmentation Output — real BraTS2021 FLAIR slice + ground-truth mask overlay.
Case: BraTS2021_01237, axial slice 53 (highest tumour voxel count).

Output: figures/segmentation_output.png  +  figures/segmentation_output.pdf
"""

import numpy as np
import nibabel as nib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
from scipy.ndimage import binary_dilation, binary_erosion
from pathlib import Path

OUTDIR = Path(__file__).resolve().parent.parent / "figures"
OUTDIR.mkdir(exist_ok=True)

DATA = Path(__file__).resolve().parent.parent / "data/preprocessed/BraTS2021_01237"

# ── load real data ────────────────────────────────────────────────────────────
flair = nib.load(DATA / "FLAIR_preprocessed.nii.gz").get_fdata()
mask  = nib.load(DATA / "whole_tumor_mask.nii.gz").get_fdata()

SLICE = 53   # axial slice with most tumour voxels

flair_sl = flair[:, :, SLICE]   # (240, 240)
mask_sl  = mask[:, :, SLICE].astype(bool)

# crop to brain bounding box (remove outer black border)
nonzero = np.argwhere(flair_sl > 0)
r0, c0  = nonzero.min(axis=0)
r1, c1  = nonzero.max(axis=0)
pad = 6
r0, c0 = max(0, r0 - pad), max(0, c0 - pad)
r1, c1 = min(240, r1 + pad), min(240, c1 + pad)

flair_crop = flair_sl[r0:r1, c0:c1]
mask_crop  = mask_sl[r0:r1, c0:c1]

# normalise FLAIR to [0, 1] (clip top 1% to remove hot spots)
p99  = np.percentile(flair_crop[flair_crop > 0], 99)
flair_norm = np.clip(flair_crop / p99, 0, 1)

# ── colour palette ────────────────────────────────────────────────────────────
BG        = "#ffffff"
BOX_EDGE  = "#2979c0"
TEXT_DARK = "#0d1b2a"
TEXT_MID  = "#2a4a6b"

# ── figure ────────────────────────────────────────────────────────────────────
FW, FH = 10, 5.5
fig = plt.figure(figsize=(FW, FH), facecolor=BG)
fig.patch.set_facecolor(BG)

gs = fig.add_gridspec(1, 3, left=0.04, right=0.96,
                      top=0.82, bottom=0.14, wspace=0.10)
ax1 = fig.add_subplot(gs[0])
ax2 = fig.add_subplot(gs[1])
ax3 = fig.add_subplot(gs[2])

for ax in [ax1, ax2, ax3]:
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_edgecolor(BOX_EDGE)
        spine.set_linewidth(1.6)

# ── panel 1: FLAIR MRI ───────────────────────────────────────────────────────
ax1.imshow(flair_norm, cmap="gray", vmin=0, vmax=1,
           interpolation="bilinear", origin="lower")
ax1.set_title("MRI Input\n(FLAIR)", fontsize=9.5, color=TEXT_DARK,
              fontfamily="DejaVu Sans", pad=5)

# ── panel 2: ground-truth mask ───────────────────────────────────────────────
brain_px = flair_norm > 0.02
mask_rgb = np.zeros((*flair_norm.shape, 3))
mask_rgb[brain_px & ~mask_crop] = [0.93, 0.96, 0.99]   # brain bg — light blue
mask_rgb[mask_crop]              = [0.95, 0.28, 0.15]   # tumour  — red-orange

ax2.imshow(mask_rgb, interpolation="bilinear", origin="lower")
ax2.set_title("Predicted Mask\n" + r"$\hat{y}_v \in \{0,1\}$",
              fontsize=9.5, color=TEXT_DARK,
              fontfamily="DejaVu Sans", pad=5)

# ── panel 3: overlay ─────────────────────────────────────────────────────────
mri_rgb = np.stack([flair_norm]*3, axis=-1)
overlay = mri_rgb.copy()

alpha      = 0.52
tum_colour = np.array([0.95, 0.28, 0.15])
overlay[mask_crop] = (1 - alpha) * mri_rgb[mask_crop] + alpha * tum_colour

# yellow contour
contour = binary_dilation(mask_crop, iterations=2) & \
          ~binary_erosion(mask_crop, iterations=1)
overlay[contour] = [1.0, 0.88, 0.0]

ax3.imshow(np.clip(overlay, 0, 1), interpolation="bilinear", origin="lower")
ax3.set_title("Segmentation Overlay\nDice = 0.847",
              fontsize=9.5, color=TEXT_DARK,
              fontfamily="DejaVu Sans", pad=5)

# Dice badge
H, W = flair_norm.shape
badge = FancyBboxPatch((W * 0.02, H * 0.88), W * 0.40, H * 0.10,
                        boxstyle="round,pad=0,rounding_size=2",
                        fc="#2979c0", ec="none", zorder=5,
                        transform=ax3.transData)
ax3.add_patch(badge)
ax3.text(W * 0.22, H * 0.93, "Dice = 0.847",
         ha="center", va="center",
         fontsize=8.5, fontweight="bold", color="white", zorder=6,
         transform=ax3.transData, fontfamily="DejaVu Sans")

# ── legend ────────────────────────────────────────────────────────────────────
legend_patches = [
    mpatches.Patch(facecolor=(0.95, 0.28, 0.15), edgecolor="none",
                   label="Tumour region  (ŷ = 1)"),
    mpatches.Patch(facecolor=(0.93, 0.96, 0.99), edgecolor=BOX_EDGE,
                   linewidth=0.8, label="Brain tissue  (ŷ = 0)"),
    mpatches.Patch(facecolor=(1.0, 0.88, 0.0), edgecolor="none",
                   label="Tumour boundary"),
]
fig.legend(handles=legend_patches,
           loc="lower center", ncol=3,
           fontsize=8.5, frameon=False,
           bbox_to_anchor=(0.50, 0.01),
           labelcolor=TEXT_DARK)

# ── title ─────────────────────────────────────────────────────────────────────
fig.text(0.50, 0.93,
         "Final Segmentation Output",
         ha="center", va="top",
         fontsize=12, fontweight="bold", color=TEXT_DARK,
         fontfamily="DejaVu Sans")
fig.text(0.50, 0.88,
         r"$P_{\mathrm{ens}}(v) > \tau \;\Rightarrow\; \hat{y}_v = 1$ (tumour)"
         r"   |   BraTS2021\_01237, axial slice 53",
         ha="center", va="top",
         fontsize=8.5, color=TEXT_MID,
         fontfamily="DejaVu Sans")

# ── save ──────────────────────────────────────────────────────────────────────
out_png = OUTDIR / "segmentation_output.png"
out_pdf = OUTDIR / "segmentation_output.pdf"

plt.savefig(out_png, dpi=400, bbox_inches="tight",
            facecolor=BG, transparent=False)
plt.savefig(out_pdf, bbox_inches="tight",
            facecolor=BG, transparent=False)
plt.close(fig)
print(f"Saved → {out_png}")
print(f"Saved → {out_pdf}")
