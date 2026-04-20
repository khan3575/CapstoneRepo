"""
Generate a clean, professional side-by-side
  Raw T1CE MRI  ↔  SLIC superpixel boundary overlay
using the real BraTS2021_00209 slice already in the repo.

Style: grayscale MRI kept intact; only cyan boundary lines drawn on top;
tumour-region superpixels highlighted with a single warm accent fill.
Styled to match the dark-blue/cyan presentation theme.
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from skimage.segmentation import slic, mark_boundaries
from skimage.measure import regionprops
from PIL import Image

# ── 1. Load & crop the T1CE panel ─────────────────────────────────────────────
src = "/Users/sakibkhan/Desktop/Capstone/CapstoneRepo/paperWriting/overleaf_flat/median_BraTS2021_00209_z059.jpg"
full = Image.open(src)
W, H = full.size
panel_w = W // 4
t1ce_pil = full.crop((18, 52, panel_w - 8, H - 6))
t1ce_np  = np.array(t1ce_pil.convert("L"), dtype=float) / 255.0

# ── 2. Brain mask & SLIC ──────────────────────────────────────────────────────
brain_mask = (t1ce_np > 0.04).astype(np.uint8)
img_rgb = np.stack([t1ce_np] * 3, axis=-1)
segments = slic(img_rgb, n_segments=200, compactness=10,
                sigma=1.0, start_label=0, mask=brain_mask)
n_segs = len(np.unique(segments[brain_mask == 1]))

# ── 3. Identify tumour superpixels (high mean T1CE intensity = enhancing core) ─
TUMOUR_THRESH = 0.55          # T1CE bright regions → likely tumour
tumour_segs = set()
for region in regionprops(segments + 1):   # regionprops needs labels ≥ 1
    seg_id = region.label - 1
    mask = segments == seg_id
    if brain_mask[mask].mean() < 0.5:      # mostly outside brain — skip
        continue
    if t1ce_np[mask].mean() > TUMOUR_THRESH:
        tumour_segs.add(seg_id)

# ── 4. Build overlay: grayscale base + cyan boundaries + amber tumour fill ────
overlay = img_rgb.copy()

# Soft amber fill for tumour superpixels (semi-transparent blend)
AMBER = np.array([1.0, 0.55, 0.05])
for seg_id in tumour_segs:
    mask = segments == seg_id
    overlay[mask] = 0.45 * img_rgb[mask] + 0.55 * AMBER

# Draw cyan boundary lines on top
bounded = mark_boundaries(overlay, segments,
                           color=(0.0, 0.85, 0.95),
                           outline_color=None, mode="thick")

# ── 5. Compose figure ──────────────────────────────────────────────────────────
BG    = "#0a1628"
CYAN  = "#00d4ff"
WHITE = "#e8eaf6"
AMBER_HEX = "#ff8c14"

fig, axes = plt.subplots(1, 2, figsize=(13, 5.6),
                          gridspec_kw={"wspace": 0.04})
fig.patch.set_facecolor(BG)

panels = [
    (img_rgb, "Raw T1CE MRI Slice",     "BraTS2021_00209  ·  Slice 59"),
    (bounded,  f"SLIC Superpixel Overlay", f"~{n_segs} superpixels  ·  each region = one graph node"),
]

for ax, (im, title, sub) in zip(axes, panels):
    ax.set_facecolor(BG)
    ax.imshow(im, interpolation="bilinear", aspect="equal")
    ax.set_title(f"{title}\n{sub}", color=CYAN,
                 fontsize=12.5, fontweight="bold", pad=8, linespacing=1.55)
    ax.axis("off")

# Arrow between panels
fig.text(0.497, 0.52, "→", color=CYAN, fontsize=38,
         ha="center", va="center", fontweight="bold")

# Legend for tumour highlight
legend_patch = mpatches.Patch(facecolor=AMBER_HEX, edgecolor="white",
                               linewidth=0.6,
                               label=f"High-intensity superpixels (likely tumour, T1CE > {TUMOUR_THRESH})")
axes[1].legend(handles=[legend_patch], loc="lower right",
               fontsize=8.5, framealpha=0.25,
               facecolor="#0a1628", edgecolor=CYAN,
               labelcolor=WHITE)

# Bottom caption
fig.text(
    0.5, 0.035,
    f"SLIC  (K = 200 target)  ·  {n_segs} superpixels realised  ·  "
    "Spatial compression: 8.9 M voxels → ~3,909 nodes per patient  (2,284×)",
    color=WHITE, fontsize=9.5, ha="center", va="center", alpha=0.80,
)

# Subtle divider line
fig.add_artist(plt.Line2D([0.493, 0.493], [0.09, 0.93],
                           transform=fig.transFigure,
                           color=CYAN, linewidth=0.5, alpha=0.30))

out = "/Users/sakibkhan/Desktop/Capstone/CapstoneRepo/paperWriting/presentationDetails/slic_visualization.png"
plt.savefig(out, dpi=180, bbox_inches="tight",
            facecolor=BG, edgecolor="none")
plt.close()
print(f"Saved → {out}  ({n_segs} superpixels, {len(tumour_segs)} tumour-highlighted)")
