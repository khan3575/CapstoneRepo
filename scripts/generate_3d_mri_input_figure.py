"""
Generate a 3D-looking MRI input figure for the methodology diagram.
Three orthogonal slices (axial, coronal, sagittal) painted on the
corner faces of a virtual 3-D cube — the standard way to show MRI volumes.
Output: figures/mri_3d_input.png
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import nibabel as nib
from pathlib import Path
from PIL import Image as PILImage

# ── paths ──────────────────────────────────────────────────────────────────
BASE   = Path(__file__).resolve().parent.parent
SCAN   = BASE / "data/organized/BraTS2021_00000/T1ce.nii.gz"
SEG    = BASE / "data/organized/BraTS2021_00000/segmentation.nii.gz"
OUTDIR = BASE / "figures"
OUTDIR.mkdir(exist_ok=True)

# ── load & normalise ────────────────────────────────────────────────────────
vol = nib.load(SCAN).get_fdata().astype(np.float32)
seg = nib.load(SEG).get_fdata().astype(np.uint8)

lo, hi = np.percentile(vol[vol > 0], [1, 99])
vol = np.clip(vol, lo, hi)
vol = (vol - lo) / (hi - lo)

H, W, D = vol.shape   # 240, 240, 155

# ── crop to brain bounding box (remove black borders) ──────────────────────
brain_mask = vol > 0.02
rows = np.any(brain_mask, axis=(1, 2))
cols = np.any(brain_mask, axis=(0, 2))
deps = np.any(brain_mask, axis=(0, 1))

r0, r1 = np.where(rows)[0][[0, -1]]
c0, c1 = np.where(cols)[0][[0, -1]]
d0, d1 = np.where(deps)[0][[0, -1]]

# Add small padding
pad = 4
r0, r1 = max(r0-pad, 0), min(r1+pad, H-1)
c0, c1 = max(c0-pad, 0), min(c1+pad, W-1)
d0, d1 = max(d0-pad, 0), min(d1+pad, D-1)

vol_crop = vol[r0:r1+1, c0:c1+1, d0:d1+1]
seg_crop = seg[r0:r1+1, c0:c1+1, d0:d1+1]
CH, CW, CD = vol_crop.shape

# ── pick tumour-centred slices within cropped volume ───────────────────────
tumour_vox = np.argwhere(seg_crop > 0)
if len(tumour_vox):
    cx, cy, cz = tumour_vox.mean(axis=0).astype(int)
else:
    cx, cy, cz = CH // 2, CW // 2, CD // 2

ax_slice  = vol_crop[:, :, cz]      # axial  (H×W)
ax_seg    = seg_crop[:, :, cz]
cor_slice = vol_crop[:, cy, :]      # coronal (H×D)
sag_slice = vol_crop[cx, :, :]      # sagittal (W×D)

# ── build RGBA arrays ──────────────────────────────────────────────────────
def to_rgba(gray_2d, seg_overlay=None, brighten=1.0):
    cm = plt.get_cmap("gray")
    # Boost contrast: re-normalise non-zero pixels within slice
    sl = gray_2d.copy()
    nz = sl[sl > 0.01]
    if len(nz):
        lo_, hi_ = np.percentile(nz, [2, 98])
        sl = np.clip((sl - lo_) / max(hi_ - lo_, 1e-6), 0, 1)
    sl = np.clip(sl * brighten, 0, 1)
    rgba = cm(sl).copy()
    if seg_overlay is not None:
        for lbl, col in [(1, (1.0, 0.15, 0.15, 0.65)),
                         (2, (0.1, 0.9,  0.2,  0.55)),
                         (4, (1.0, 0.85, 0.0,  0.75))]:
            rgba[seg_overlay == lbl] = col
    return rgba

ax_rgba  = to_rgba(ax_slice,  seg_overlay=ax_seg, brighten=1.1)

# ── resize to square (pad shorter axis with zeros before resizing) ─────────
def square_pad(rgba):
    h, w = rgba.shape[:2]
    s = max(h, w)
    out = np.zeros((s, s, 4), dtype=rgba.dtype)
    r0_ = (s - h) // 2
    c0_ = (s - w) // 2
    out[r0_:r0_+h, c0_:c0_+w] = rgba
    return out

def resize_rgba(rgba, size):
    rgba = square_pad(rgba)
    img = PILImage.fromarray((rgba * 255).astype(np.uint8))
    img = img.resize((size, size), PILImage.LANCZOS)
    return np.array(img) / 255.0

N = 220   # pixels per face edge

# Orientation: row 0 of the image maps to the TOP of its face
ax_r  = resize_rgba(ax_rgba,                          N)
cor_r = resize_rgba(to_rgba(cor_slice[::-1, :], brighten=1.3), N)
sag_r = resize_rgba(to_rgba(sag_slice[::-1, :], brighten=1.3), N)

# ── 3-D cube with faces ────────────────────────────────────────────────────
fig = plt.figure(figsize=(6, 6.5), dpi=200)
ax3 = fig.add_subplot(111, projection="3d")
ax3.set_box_aspect([1, 1, 1])

def paste_face(ax, img_rgba, face):
    """
    Paint img_rgba onto a unit-cube face.
    Faces (viewer at +x,-y,+z corner with azim=-135, elev=30):
      'axial'    → z=1 top face    visible from above
      'coronal'  → y=0 front face  visible from -y direction
      'sagittal' → x=1 right face  (inside face, visible from -x direction)
    """
    N = img_rgba.shape[0]
    t = np.linspace(0.0, 1.0, N)

    if face == 'axial':
        # XY plane at z=1
        X, Y = np.meshgrid(t, t)
        Z = np.ones_like(X)
    elif face == 'coronal':
        # XZ plane at y=0: col→X, row→Z (row0=top=z=1)
        X, Z = np.meshgrid(t, t[::-1])
        Y = np.zeros_like(X)
    elif face == 'sagittal':
        # YZ plane at x=1: col→Y, row→Z (row0=top=z=1)
        Y, Z = np.meshgrid(t, t[::-1])
        X = np.ones_like(Y)

    ax.plot_surface(X, Y, Z, facecolors=img_rgba,
                    rstride=1, cstride=1,
                    shade=False, antialiased=True)

paste_face(ax3, ax_r,  'axial')
paste_face(ax3, cor_r, 'coronal')
paste_face(ax3, sag_r, 'sagittal')

# ── wireframe edges of the unit cube ───────────────────────────────────────
edges = [
    [(0,0,0),(1,0,0)], [(0,1,0),(1,1,0)],
    [(0,0,1),(1,0,1)], [(0,1,1),(1,1,1)],
    [(0,0,0),(0,1,0)], [(1,0,0),(1,1,0)],
    [(0,0,1),(0,1,1)], [(1,0,1),(1,1,1)],
    [(0,0,0),(0,0,1)], [(1,0,0),(1,0,1)],
    [(0,1,0),(0,1,1)], [(1,1,0),(1,1,1)],
]
for e in edges:
    ax3.plot([e[0][0], e[1][0]],
             [e[0][1], e[1][1]],
             [e[0][2], e[1][2]],
             color="#888888", linewidth=0.8, alpha=0.7)

# ── cosmetics ───────────────────────────────────────────────────────────────
ax3.set_xlim(0, 1); ax3.set_ylim(0, 1); ax3.set_zlim(0, 1)
ax3.set_axis_off()

# azim=-135 puts viewer at (-x,-y) quadrant → sees y=0 (coronal) and x=1 (sagittal) from inside
ax3.view_init(elev=28, azim=-135)

# Slice-plane labels (positioned outside the cube faces)
ax3.text(0.5,  1.1,  1.02, "Axial",    color="#dddddd", fontsize=8, ha="center", va="bottom", style="italic")
ax3.text(-0.1, 0.5,  -0.05, "Coronal", color="#dddddd", fontsize=8, ha="right",  va="top",    style="italic")
ax3.text(1.05, 1.05,  0.5,  "Sagittal",color="#dddddd", fontsize=8, ha="left",   va="center", style="italic")

# Tumour legend
legend_items = [
    mpatches.Patch(fc=(1.0, 0.15, 0.15), ec="white", lw=0.5, label="Necrotic Core (TC)"),
    mpatches.Patch(fc=(0.1, 0.9,  0.2),  ec="white", lw=0.5, label="Peritumoral Edema (WT)"),
    mpatches.Patch(fc=(1.0, 0.85, 0.0),  ec="white", lw=0.5, label="Enhancing Tumour (ET)"),
]
leg = ax3.legend(handles=legend_items,
                 loc="lower center",
                 bbox_to_anchor=(0.5, -0.14),
                 ncol=1, fontsize=7.5,
                 framealpha=0.15,
                 facecolor="#222222",
                 edgecolor="#555555",
                 labelcolor="white",
                 handlelength=1.2,
                 borderpad=0.6)

fig.patch.set_facecolor("#111111")
plt.subplots_adjust(left=0, right=1, top=0.96, bottom=0.1)

out = OUTDIR / "mri_3d_input.png"
fig.savefig(out, dpi=200, bbox_inches="tight",
            facecolor=fig.get_facecolor())
plt.close(fig)
print(f"Saved → {out}")
