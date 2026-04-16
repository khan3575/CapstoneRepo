"""
Preprocessing figure – Active Slice Selection.

Layout (three zones, left → right):
  A) LEFT  – bar chart: active-voxel fraction across all 155 slices,
             threshold line τ=0.05, active slices highlighted green.
  B) MIDDLE – strip of 8 representative axial slices (T1CE) spanning the
             depth axis, labelled with their slice index.  Active slices
             get a green border, inactive get a dim red border.
  C) RIGHT  – two-column panel showing the "2-slice pairing" concept:
             consecutive active slice pairs stacked vertically, with an
             arrow showing they form one training sample.

All real data from BraTS2021_00000 T1CE.
Output: figures/preprocessing_active_slices.png
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import nibabel as nib
from pathlib import Path

# ── paths ───────────────────────────────────────────────────────────────────
BASE   = Path(__file__).resolve().parent.parent
SCAN   = BASE / "data/organized/BraTS2021_00000/T1ce.nii.gz"
SEG    = BASE / "data/organized/BraTS2021_00000/segmentation.nii.gz"
OUTDIR = BASE / "figures"
OUTDIR.mkdir(exist_ok=True)

# ── colours ─────────────────────────────────────────────────────────────────
BG      = "#0d1117"
PANEL   = "#161b22"
PANEL2  = "#1c2230"
ACTIVE  = "#2ea043"      # green
INACTIVE= "#6e2020"      # muted red
TAU_COL = "#f0883e"      # orange threshold line
SLICE_BORDER_ACTIVE   = "#3fb950"
SLICE_BORDER_INACTIVE = "#6e4040"
TEXT    = "#e6edf3"
SUBTEXT = "#8b949e"
ACCENT  = "#58a6ff"
PAIR_A  = "#1f6feb"      # blue for pair member A
PAIR_B  = "#388bfd"

# ── load & normalise ─────────────────────────────────────────────────────────
vol = nib.load(SCAN).get_fdata().astype(np.float32)
seg = nib.load(SEG).get_fdata().astype(np.uint8)

brain_mask_3d = vol > 0
nz = vol[brain_mask_3d]
mu, sd = nz.mean(), nz.std()
vol_z = (vol - mu) / (sd + 1e-8)
lo, hi = np.percentile(nz, [1, 99])
vol_n = np.clip((vol - mu) / (sd + 1e-8), (lo-mu)/sd, (hi-mu)/sd)
lo2, hi2 = vol_n[brain_mask_3d].min(), vol_n[brain_mask_3d].max()
vol_n = (vol_n - lo2) / (hi2 - lo2 + 1e-8)
vol_n = np.clip(vol_n, 0, 1)

H, W, D = vol_n.shape   # 240, 240, 155

# ── compute per-slice active fraction (τ criterion) ──────────────────────────
TAU = 0.05
total_px = H * W
active_frac = np.array([(vol_n[:, :, z] > 0.02).sum() / total_px
                         for z in range(D)])
is_active = active_frac >= TAU

n_active = is_active.sum()
active_indices = np.where(is_active)[0]

# ── tumour centre slice ───────────────────────────────────────────────────────
tumour_vox = np.argwhere(seg > 0)
_, _, cz = tumour_vox.mean(axis=0).astype(int)

# ── pick 9 representative slices (spread across depth, include tumour) ────────
# Evenly spaced across the full depth axis
sample_z = np.round(np.linspace(5, D-6, 9)).astype(int)
# Ensure tumour slice is included (replace nearest)
diffs = np.abs(sample_z - cz)
sample_z[diffs.argmin()] = cz

# ── brain-crop helper (compute from union mask) ───────────────────────────────
union_mask = np.zeros((H, W), dtype=bool)
for z in active_indices:
    union_mask |= (vol_n[:, :, z] > 0.02)
rs, cs = np.where(union_mask)
pad = 6
r0, r1 = max(rs.min()-pad, 0), min(rs.max()+pad, H-1)
c0, c1 = max(cs.min()-pad, 0), min(cs.max()+pad, W-1)
bh, bw = r1-r0+1, c1-c0+1
BS = max(bh, bw)
dr, dc = (BS-bh)//2, (BS-bw)//2

def get_slice(z):
    sl = vol_n[r0:r1+1, c0:c1+1, z]
    out = np.zeros((BS, BS), dtype=np.float32)
    out[dr:dr+bh, dc:dc+bw] = sl
    return out

def brighten(sl):
    nz2 = sl[sl > 0.01]
    if len(nz2) == 0:
        return sl
    lo_, hi_ = np.percentile(nz2, [2, 98])
    return np.clip((sl - lo_) / max(hi_-lo_, 1e-6), 0, 1)

# ── find first pair of consecutive active slices near tumour ──────────────────
# walk from cz outward to find pair (a_i, a_i+1) in active list
active_list = list(active_indices)
pair_idx = None
for i in range(len(active_list)-1):
    if abs(active_list[i] - cz) < 20:
        pair_idx = i
        break
if pair_idx is None:
    pair_idx = len(active_list)//2
z_a = active_list[pair_idx]
z_b = active_list[pair_idx + 1]

# ═══════════════════════════════════════════════════════════════════════════════
#  Figure layout
# ═══════════════════════════════════════════════════════════════════════════════
FW, FH = 16, 7.0
fig = plt.figure(figsize=(FW, FH), dpi=200, facecolor=BG)

# We use a manual axes layout for precise control
# Zone A: bar chart   [0.03, 0.10, 0.30, 0.80]
# Zone B: slice strip [0.36, 0.10, 0.40, 0.80]
# Zone C: pair panel  [0.79, 0.10, 0.18, 0.80]

# ────────────────────────────────────────────────────────────────────────────
#  ZONE A – Active-fraction bar chart
# ────────────────────────────────────────────────────────────────────────────
ax_bar = fig.add_axes([0.04, 0.13, 0.28, 0.72], facecolor=PANEL)
for sp in ax_bar.spines.values():
    sp.set_edgecolor("#30363d"); sp.set_linewidth(1.2)

# bar colours
bar_colors = [ACTIVE if is_active[z] else INACTIVE for z in range(D)]
ax_bar.bar(np.arange(D), active_frac, color=bar_colors,
           width=1.0, linewidth=0, zorder=2)

# τ threshold line
ax_bar.axhline(TAU, color=TAU_COL, lw=1.8, linestyle="--", zorder=3)
ax_bar.text(D + 1, TAU + 0.002, f"τ = {TAU}", color=TAU_COL,
            fontsize=8.5, va="bottom", ha="left", fontweight="bold")

# tumour-centre slice marker
ax_bar.axvline(cz, color="#ff6b6b", lw=1.4, linestyle=":", zorder=4)
ax_bar.text(cz + 1, 0.38, f"z={cz}\n(tumour)", color="#ff6b6b",
            fontsize=7.5, va="top", ha="left")

ax_bar.set_xlim(-1, D+2)
ax_bar.set_ylim(0, max(active_frac)*1.12)
ax_bar.set_xlabel("Axial Slice Index  z", color=SUBTEXT, fontsize=8.5, labelpad=4)
ax_bar.set_ylabel("Active Voxel Fraction", color=SUBTEXT, fontsize=8.5, labelpad=4)
ax_bar.tick_params(colors=SUBTEXT, labelsize=7.5)
for label in ax_bar.get_xticklabels() + ax_bar.get_yticklabels():
    label.set_color(SUBTEXT)
ax_bar.set_facecolor(PANEL)
ax_bar.grid(axis="y", color="#21262d", lw=0.6, zorder=0)

# legend
leg_patches = [
    mpatches.Patch(fc=ACTIVE,   ec="none", label=f"Active  (≥τ)  n={n_active}"),
    mpatches.Patch(fc=INACTIVE, ec="none", label=f"Skipped (<τ)  n={D-n_active}"),
]
ax_bar.legend(handles=leg_patches, loc="upper left",
              fontsize=7.5, framealpha=0.2,
              facecolor="#161b22", edgecolor="#30363d",
              labelcolor=TEXT, handlelength=1.0)

# Zone A title
fig.text(0.04 + 0.28/2, 0.89, "① Active-Slice Selection  (τ = 0.05)",
         ha="center", va="bottom", color=ACTIVE,
         fontsize=10.5, fontweight="bold")
fig.text(0.04 + 0.28/2, 0.86,
         "Slices where ≥5% of voxels are non-zero brain tissue",
         ha="center", va="bottom", color=SUBTEXT, fontsize=8)

# ────────────────────────────────────────────────────────────────────────────
#  ZONE B – Strip of 9 representative slices
# ────────────────────────────────────────────────────────────────────────────
N_STRIP = 9
strip_left  = 0.355
strip_width = 0.405
cell_w = strip_width / N_STRIP
cell_h = 0.60
cell_b = 0.20

fig.text(strip_left + strip_width/2, 0.89,
         "② Slice-Strip  (T₁CE · every ~17ᵗʰ axial slice)",
         ha="center", va="bottom", color=ACCENT,
         fontsize=10.5, fontweight="bold")
fig.text(strip_left + strip_width/2, 0.86,
         "Green border = active  ·  dim-red border = inactive",
         ha="center", va="bottom", color=SUBTEXT, fontsize=8)

for i, z in enumerate(sample_z):
    sl = brighten(get_slice(z))
    active_here = is_active[z]
    border_col  = SLICE_BORDER_ACTIVE if active_here else SLICE_BORDER_INACTIVE
    bg_tint     = "#0d1f10" if active_here else "#1a0d0d"

    ax_sl = fig.add_axes(
        [strip_left + i*cell_w + 0.003,
         cell_b,
         cell_w - 0.006,
         cell_h],
        facecolor=bg_tint
    )
    for sp in ax_sl.spines.values():
        lw = 2.5 if active_here else 1.2
        sp.set_edgecolor(border_col); sp.set_linewidth(lw)
    ax_sl.imshow(sl, cmap="gray", origin="lower", interpolation="bilinear",
                 vmin=0, vmax=1)
    ax_sl.set_xticks([]); ax_sl.set_yticks([])

    # Slice index label
    lbl_col = SLICE_BORDER_ACTIVE if active_here else SLICE_BORDER_INACTIVE
    ax_sl.text(0.5, -0.07, f"z={z}", transform=ax_sl.transAxes,
               ha="center", va="top", fontsize=7.5, color=lbl_col)

    # Tumour slice star
    if z == cz:
        ax_sl.text(0.5, 0.96, "★ tumour", transform=ax_sl.transAxes,
                   ha="center", va="top", fontsize=7,
                   color="#ff6b6b", fontweight="bold")

    # active badge
    if active_here:
        ax_sl.text(0.03, 0.97, "✔", transform=ax_sl.transAxes,
                   ha="left", va="top", fontsize=9,
                   color=ACTIVE, fontweight="bold")
    else:
        ax_sl.text(0.03, 0.97, "✘", transform=ax_sl.transAxes,
                   ha="left", va="top", fontsize=9,
                   color=INACTIVE)

# ────────────────────────────────────────────────────────────────────────────
#  ZONE C – 2-Slice pairing concept
# ────────────────────────────────────────────────────────────────────────────
pair_left = 0.77
pair_w    = 0.205
fig.text(pair_left + pair_w/2, 0.89,
         "③ 2-Slice Pairing",
         ha="center", va="bottom", color=TAU_COL,
         fontsize=10.5, fontweight="bold")
fig.text(pair_left + pair_w/2, 0.86,
         "Consecutive active slices → one training sample",
         ha="center", va="bottom", color=SUBTEXT, fontsize=8)

half_h = 0.295
gap    = 0.022

for k, (z_pair, z_lbl, pair_col) in enumerate([
        (z_a, f"z = {z_a}  (active)", PAIR_A),
        (z_b, f"z = {z_b}  (active)", PAIR_B),
]):
    sl = brighten(get_slice(z_pair))
    yb = cell_b + (1-k) * (half_h + gap)

    ax_p = fig.add_axes([pair_left + 0.01, yb, pair_w - 0.02, half_h],
                        facecolor="#0d1a2a")
    for sp in ax_p.spines.values():
        sp.set_edgecolor(pair_col); sp.set_linewidth(2.5)
    ax_p.imshow(sl, cmap="gray", origin="lower", interpolation="bilinear",
                vmin=0, vmax=1)
    ax_p.set_xticks([]); ax_p.set_yticks([])
    ax_p.text(0.5, 0.04, z_lbl, transform=ax_p.transAxes,
              ha="center", va="bottom", fontsize=8.0,
              color=TEXT, fontweight="bold")

# brace / arrow between pair
ax_overlay = fig.add_axes([0, 0, 1, 1], facecolor="none")
ax_overlay.set_xlim(0, 1); ax_overlay.set_ylim(0, 1)
ax_overlay.axis("off")

brace_x = pair_left + pair_w/2
y_top   = cell_b + 2*(half_h + gap) + half_h/2
y_bot   = cell_b + half_h + gap + half_h/2
ax_overlay.annotate("", xy=(brace_x, y_bot + 0.01),
                    xytext=(brace_x, y_top - 0.01),
                    arrowprops=dict(arrowstyle="-|>",
                                   color=TAU_COL, lw=2.0,
                                   mutation_scale=14))
ax_overlay.text(brace_x + 0.015,
                (y_top + y_bot) / 2,
                "1 training\nsample",
                ha="left", va="center",
                color=TAU_COL, fontsize=8.0, fontweight="bold")

# ── divider lines between zones ──────────────────────────────────────────────
for xd in [0.343, 0.762]:
    fig.add_artist(plt.Line2D([xd, xd], [0.08, 0.92],
                               transform=fig.transFigure,
                               color="#30363d", lw=1.0, linestyle="--"))

# ── global title ─────────────────────────────────────────────────────────────
fig.text(0.5, 0.96,
         "Preprocessing Pipeline  ·  Active Slice Selection & Pairing",
         ha="center", va="top",
         color=TEXT, fontsize=13, fontweight="bold")
fig.text(0.5, 0.93,
         "BraTS 2021  ·  T₁CE modality  ·  patient BraTS2021_00000",
         ha="center", va="top", color=SUBTEXT, fontsize=8.5)

# ── footer ───────────────────────────────────────────────────────────────────
fig.text(0.5, 0.03,
         f"Total slices: {D}  ·  Active (τ≥0.05): {n_active}  ·  "
         f"Z-score normalisation  ·  Tumour-centred slice: z={cz}",
         ha="center", va="bottom", color=SUBTEXT, fontsize=8)

out = OUTDIR / "preprocessing_active_slices.png"
fig.savefig(out, dpi=200, bbox_inches="tight", facecolor=BG)
plt.close(fig)
print(f"Saved → {out}")
