"""
Professional methodology diagram for:
  "Efficient Brain Tumour Segmentation using Graph Neural Networks
   on BraTS Datasets: A Superpixel-Based Approach with 5.9× Speedup"

Layout (left → right):
  [4-Modality MRI Input]  →  [Preprocessing]  →  [Superpixel Graph]
                           →  [GraphSAGE GNN]  →  [Tumour Mask Output]

All panels embed real data from BraTS2021_00000.
Output: figures/methodology_diagram.png  (300 DPI, white bg)
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.patheffects as pe
from matplotlib.patches import FancyArrowPatch
import nibabel as nib
from pathlib import Path
from PIL import Image as PILImage
from skimage.segmentation import slic, mark_boundaries
from skimage import graph as skgraph
from scipy import ndimage

# ── colour palette ──────────────────────────────────────────────────────────
C = dict(
    bg      = "#0f1117",
    panel   = "#1a1d27",
    border  = "#2e3348",
    arrow   = "#4a90d9",
    s1      = "#3a7bd5",   # blue  – input
    s2      = "#00b09b",   # teal  – preprocess
    s3      = "#f7971e",   # orange– graph
    s4      = "#c0392b",   # red   – GNN
    s5      = "#8e44ad",   # purple– output
    text    = "#e8eaf0",
    subtext = "#9099b0",
    white   = "#ffffff",
)

# ── paths ───────────────────────────────────────────────────────────────────
BASE   = Path(__file__).resolve().parent.parent
PT     = "BraTS2021_00000"
MODDIR = BASE / "data/organized" / PT
SEG_P  = MODDIR / "segmentation.nii.gz"
OUTDIR = BASE / "figures"
OUTDIR.mkdir(exist_ok=True)

MODS = [("T1",   MODDIR/"T1.nii.gz",   "#4a90d9"),
        ("T1CE", MODDIR/"T1ce.nii.gz",  "#f5a623"),
        ("T2",   MODDIR/"T2.nii.gz",    "#7ed321"),
        ("FLAIR",MODDIR/"FLAIR.nii.gz", "#bd10e0")]

# ── load & normalise ────────────────────────────────────────────────────────
def load_norm(path):
    v = nib.load(path).get_fdata().astype(np.float32)
    mask = v > 0
    if mask.sum() == 0:
        return v
    mu, sd = v[mask].mean(), v[mask].std()
    v = (v - mu) / (sd + 1e-8)
    lo, hi = np.percentile(v[mask], [1, 99])
    v = np.clip(v, lo, hi)
    v = (v - lo) / (hi - lo + 1e-8)
    return v

vols = {name: load_norm(p) for name, p, _ in MODS}
seg  = nib.load(SEG_P).get_fdata().astype(np.uint8)

H, W, D = vols["T1CE"].shape

# pick tumour-centred axial slice
tumour_vox = np.argwhere(seg > 0)
cx, cy, cz = tumour_vox.mean(axis=0).astype(int)

# ── compute shared brain bounding box (union of all modalities at cz) ────────
_union_mask = np.zeros((H, W), dtype=bool)
for _name, _, _ in MODS:
    _union_mask |= (vols[_name][:, :, cz] > 0.04)
_rs, _cs = np.where(_union_mask)
_pad = 8
_r0 = max(_rs.min()-_pad, 0); _r1 = min(_rs.max()+_pad, H-1)
_c0 = max(_cs.min()-_pad, 0); _c1 = min(_cs.max()+_pad, W-1)
_h  = _r1 - _r0 + 1; _w  = _c1 - _c0 + 1
_S  = max(_h, _w)   # square size
_dr = (_S - _h) // 2; _dc = (_S - _w) // 2

# ── helpers ─────────────────────────────────────────────────────────────────
def crop_brain(sl2d):
    """Square-crop to the shared brain bounding box (same size for all mods)."""
    crop = sl2d[_r0:_r1+1, _c0:_c1+1]
    out = np.zeros((_S, _S), dtype=np.float32)
    out[_dr:_dr+_h, _dc:_dc+_w] = crop
    return out

def panel_ax(fig, rect, title, title_color, border_color):
    """Create a panel with a coloured title bar."""
    ax = fig.add_axes(rect, facecolor=C["panel"])
    for sp in ax.spines.values():
        sp.set_edgecolor(border_color)
        sp.set_linewidth(1.8)
    ax.set_xticks([]); ax.set_yticks([])
    return ax

def add_title(fig, rect, title, color, tag=None):
    """Floating title above a panel."""
    x = rect[0] + rect[2]/2
    y = rect[1] + rect[3] + 0.012
    fig.text(x, y, title, ha="center", va="bottom",
             color=color, fontsize=9.5, fontweight="bold",
             fontfamily="DejaVu Sans")
    if tag:
        fig.text(rect[0] + 0.003, rect[1]+rect[3]+0.006,
                 tag, ha="left", va="bottom",
                 color=C["subtext"], fontsize=7.5, style="italic")

def draw_arrow(fig, x0, y0, x1, y1, color):
    """Draw a horizontal arrow between two panel centres."""
    ax = fig.add_axes([0,0,1,1], facecolor="none")
    ax.set_xlim(0,1); ax.set_ylim(0,1)
    ax.axis("off")
    ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                arrowprops=dict(arrowstyle="-|>",
                                color=color, lw=1.8,
                                mutation_scale=14))

# ════════════════════════════════════════════════════════════════════════════
#  Figure canvas
# ════════════════════════════════════════════════════════════════════════════
FW, FH = 18, 5.4   # inches
fig = plt.figure(figsize=(FW, FH), dpi=300)
fig.patch.set_facecolor(C["bg"])

# Panel geometry  [left, bottom, width, height]  (fractions of figure)
GAP   = 0.013
BW    = 0.135  # wide panel
MW    = 0.175  # modality-grid panel
GW    = 0.16   # graph panel
NW    = 0.155  # GNN panel
OW    = 0.135  # output panel
L0    = 0.025
L1    = L0 + MW + GAP + 0.035
L2    = L1 + BW + GAP + 0.035
L3    = L2 + GW + GAP + 0.035
L4    = L3 + NW + GAP + 0.035
PH    = 0.71
PB    = 0.12

panels = dict(
    inp  = [L0, PB, MW, PH],
    pre  = [L1, PB, BW, PH],
    grph = [L2, PB, GW, PH],
    gnn  = [L3, PB, NW, PH],
    out  = [L4, PB, OW, PH],
)

# ════════════════════════════════════════════════════════════════════════════
#  PANEL 1 – 4-Modality MRI Input
# ════════════════════════════════════════════════════════════════════════════
p = panels["inp"]
ax = panel_ax(fig, p, "", C["s1"], C["s1"])

r = p; sw = r[2]/2; sh = r[3]/2
for idx, (name, _, col) in enumerate(MODS):
    row, col_i = divmod(idx, 2)
    sl = crop_brain(vols[name][:, :, cz])
    sub = fig.add_axes([r[0]+col_i*sw+0.002, r[1]+(1-row)*sh-sh+0.005, sw-0.004, sh-0.008],
                       facecolor="black")
    sub.imshow(sl, cmap="gray", origin="lower", interpolation="bilinear")
    # modality label
    sub.text(0.05, 0.93, name, transform=sub.transAxes,
             color=([C["s1"],C["s3"],C["s2"],C["s5"]])[idx],
             fontsize=8.5, fontweight="bold", va="top",
             path_effects=[pe.withStroke(linewidth=2, foreground="black")])
    sub.set_xticks([]); sub.set_yticks([])
    for sp in sub.spines.values():
        sp.set_edgecolor(C["border"]); sp.set_linewidth(0.8)

add_title(fig, p, "Multi-Modal MRI Input", C["s1"], "Stage 1")
fig.text(p[0]+p[2]/2, p[1]-0.03,
         r"$\mathbf{V}\in\mathbb{R}^{4\times240\times240\times155}$",
         ha="center", color=C["subtext"], fontsize=7.5)

# ════════════════════════════════════════════════════════════════════════════
#  PANEL 2 – Preprocessing
# ════════════════════════════════════════════════════════════════════════════
p = panels["pre"]
ax_pre = panel_ax(fig, p, "", C["s2"], C["s2"])

# Show the T1CE skull-stripped slice with a "Z-score" colormap
sl_t1ce = crop_brain(vols["T1CE"][:, :, cz])
sub_pre = fig.add_axes([p[0]+0.005, p[1]+0.05, p[2]-0.01, p[3]-0.12],
                        facecolor="black")
sub_pre.imshow(sl_t1ce, cmap="gray", origin="lower", interpolation="bilinear")

# Overlay brain mask contour
mask_2d = (vols["T1CE"][:, :, cz] > 0.04).astype(float)
mh, mw = mask_2d.shape
sk = int(mh * 0.19)
mask_crop = mask_2d[sk:, :]
h2, w2 = mask_crop.shape
s2_ = max(h2,w2)
mask_sq = np.zeros((s2_,s2_))
mask_sq[(s2_-h2)//2:(s2_-h2)//2+h2, (s2_-w2)//2:(s2_-w2)//2+w2] = mask_crop
sub_pre.contour(mask_sq, levels=[0.5], colors=[C["s2"]], linewidths=0.8, alpha=0.7)

sub_pre.set_xticks([]); sub_pre.set_yticks([])
for sp in sub_pre.spines.values():
    sp.set_edgecolor(C["border"]); sp.set_linewidth(0.8)

# Bullet annotations inside panel
steps = [
    ("⊙ ", "Z-score norm.  (per modality)"),
    ("⊙ ", "Active slices  τ = 0.05"),
    ("⊙ ", "2-slice pairing → ~69 units"),
]
for i,(blt,txt) in enumerate(steps):
    y_ = p[1] + 0.042 + i*0.012
    fig.text(p[0]+0.006, y_, blt+txt, va="top",
             color=C["subtext"], fontsize=6.5,
             fontfamily="DejaVu Sans Mono")

add_title(fig, p, "Preprocessing", C["s2"], "Stage 2")
fig.text(p[0]+p[2]/2, p[1]-0.03,
         "~69 graph units / patient",
         ha="center", color=C["subtext"], fontsize=7.5)

# ════════════════════════════════════════════════════════════════════════════
#  PANEL 3 – Superpixel Graph Construction
# ════════════════════════════════════════════════════════════════════════════
p = panels["grph"]

# Build SLIC on the T1CE axial slice (use 4-ch input for better quality)
sl_4ch = np.stack([crop_brain(vols[n][:, :, cz]) for n,_,_ in MODS], axis=-1)
sl_4ch_u8 = (sl_4ch * 255).astype(np.uint8)
# SLIC on first channel (T1CE-like) with small compactness so it follows edges
sp_map = slic(sl_4ch_u8, n_segments=55, compactness=8,
              sigma=1.0, channel_axis=-1, enforce_connectivity=True)

n_sp = sp_map.max() + 1
SZ = sl_4ch.shape[0]

# Compute centroids
centroids = {}
for lbl in range(n_sp):
    m = sp_map == lbl
    if m.sum() == 0:
        continue
    rs, cs = np.where(m)
    centroids[lbl] = (cs.mean(), SZ - rs.mean())   # (x, y) with y flipped

# Build region-adjacency edges (simple 4-connectivity RAG)
edges_rag = set()
for r in range(SZ):
    for c in range(SZ-1):
        a, b = sp_map[r,c], sp_map[r,c+1]
        if a != b: edges_rag.add((min(a,b), max(a,b)))
for r in range(SZ-1):
    for c in range(SZ):
        a, b = sp_map[r,c], sp_map[r+1,c]
        if a != b: edges_rag.add((min(a,b), max(a,b)))

# Assign a mean intensity to each superpixel (for colouring)
t1ce_sl = crop_brain(vols["T1CE"][:, :, cz])
sp_mean = np.array([t1ce_sl[sp_map==l].mean() if (sp_map==l).any() else 0
                    for l in range(n_sp)])

# Colour superpixels by mean T1CE intensity
cmap_sp = plt.cm.magma
sp_rgb = cmap_sp(sp_mean / (sp_mean.max()+1e-8))[:, :3]

# Make coloured superpixel image
sp_img = np.zeros((SZ, SZ, 3))
for lbl in range(n_sp):
    mask = sp_map == lbl
    sp_img[mask] = sp_rgb[lbl]

# Boundary overlay
boundary_img = mark_boundaries(sp_img, sp_map, color=(1,1,1), mode="thin")

ax_grph = panel_ax(fig, p, "", C["s3"], C["s3"])
sub_grph = fig.add_axes([p[0]+0.005, p[1]+0.05, p[2]-0.01, p[3]-0.12],
                         facecolor="black")
sub_grph.imshow(boundary_img, origin="upper", interpolation="bilinear")

# Draw edges as light lines
np.random.seed(42)
sampled = list(edges_rag)
for (a,b) in sampled:
    if a not in centroids or b not in centroids:
        continue
    xa,ya = centroids[a]
    xb,yb = centroids[b]
    sub_grph.plot([xa,xb],[ya,yb], color="#ffffff", alpha=0.25, lw=0.5)

# Draw nodes
xs = np.array([centroids[l][0] for l in centroids])
ys = np.array([centroids[l][1] for l in centroids])
sp_means_ordered = np.array([sp_mean[l] for l in centroids])
sub_grph.scatter(xs, ys, c=sp_means_ordered, cmap="magma",
                 s=12, zorder=5, edgecolors="white", linewidths=0.4,
                 vmin=0, vmax=1)

sub_grph.set_xlim(0, SZ); sub_grph.set_ylim(0, SZ)
sub_grph.set_xticks([]); sub_grph.set_yticks([])
for sp in sub_grph.spines.values():
    sp.set_edgecolor(C["border"]); sp.set_linewidth(0.8)

# Stats annotation
fig.text(p[0]+0.005, p[1]+0.045,
         f"~46 nodes/slice  |  RAG intra + kNN inter edges\n"
         f"15-dim features per node  (4 modalities × mean+std + shape)",
         va="top", color=C["subtext"], fontsize=6.2,
         fontfamily="DejaVu Sans")

add_title(fig, p, "Superpixel Graph Construction", C["s3"], "Stage 3")
fig.text(p[0]+p[2]/2, p[1]-0.03,
         r"~92 nodes, ~180 edges per graph  ($2{,}284\times$ compression)",
         ha="center", color=C["subtext"], fontsize=7)

# ════════════════════════════════════════════════════════════════════════════
#  PANEL 4 – GraphSAGE Architecture
# ════════════════════════════════════════════════════════════════════════════
p = panels["gnn"]
ax_gnn = panel_ax(fig, p, "", C["s4"], C["s4"])
gax = fig.add_axes([p[0]+0.008, p[1]+0.045, p[2]-0.016, p[3]-0.06],
                    facecolor=C["panel"])
gax.set_xlim(0, 10); gax.set_ylim(0, 10)
gax.axis("off")

# Draw a clean vertical layer diagram
# Layers: Input(15) → 4×SAGEConv(256) → SAGEConv(64) → MLP → Logit
layer_defs = [
    ("Input\n$\\mathbf{x}\\in\\mathbb{R}^{15}$",    "#3a7bd5", 0.8),
    ("SAGEConv\nBN · ReLU · Drop",                   "#c0392b", 1.4),
    ("SAGEConv\nBN · ReLU · Drop",                   "#c0392b", 1.4),
    ("SAGEConv\nBN · ReLU · Drop",                   "#c0392b", 1.4),
    ("SAGEConv\nBN · ReLU · Drop",                   "#c0392b", 1.4),
    ("SAGEConv\n$\\mathbb{R}^{64}$",                 "#e74c3c", 1.2),
    ("MLP Head\n(64→32→1)",                          "#8e44ad", 1.2),
    ("Logit $z_v$\n$\\sigma(z_v)\\to\\hat{y}_v$",   "#27ae60", 0.8),
]
N_L = len(layer_defs)
xs_l = np.linspace(0.5, 9.5, N_L)
bh = 3.5   # box height
bw = 0.85

for i,(lbl,col,bw_) in enumerate(layer_defs):
    x = xs_l[i]
    rect = mpatches.FancyBboxPatch((x-bw_/2, 3.2), bw_, bh,
                                    boxstyle="round,pad=0.12",
                                    facecolor=col+"22", edgecolor=col,
                                    linewidth=1.2, zorder=3)
    gax.add_patch(rect)
    gax.text(x, 5.0, lbl, ha="center", va="center",
             color=C["text"], fontsize=5.5, zorder=4,
             multialignment="center")
    # channel width annotation below
    dims = ["15","256","256","256","256","64","32→1","p(tumour)"]
    gax.text(x, 2.8, dims[i], ha="center", va="top",
             color=col, fontsize=5.5, fontweight="bold")
    # arrow to next
    if i < N_L - 1:
        gax.annotate("", xy=(xs_l[i+1]-bw_/2-0.05, 5.0),
                     xytext=(x+bw_/2+0.05, 5.0),
                     arrowprops=dict(arrowstyle="-|>", color=C["arrow"],
                                     lw=0.9, mutation_scale=8))

# Layer labels
for i in range(1,6):
    gax.text(xs_l[i], 7.2, f"L{i}", ha="center", color=C["subtext"],
             fontsize=5.5, style="italic")

# 5-fold ensemble note
gax.text(5, 1.2,
         "5-fold cross-validation  ·  soft-voting ensemble  ·  threshold = 0.5",
         ha="center", color=C["subtext"], fontsize=5.8)

add_title(fig, p, "GraphSAGE Classifier", C["s4"], "Stage 4")
fig.text(p[0]+p[2]/2, p[1]-0.03,
         "439,041 params  |  5.1 MB  |  75.4 ms / patient",
         ha="center", color=C["subtext"], fontsize=7.5)

# ════════════════════════════════════════════════════════════════════════════
#  PANEL 5 – Tumour Mask Output
# ════════════════════════════════════════════════════════════════════════════
p = panels["out"]
ax_out = panel_ax(fig, p, "", C["s5"], C["s5"])

# Show T1CE with tumour region coloured
t1_bg = crop_brain(vols["T1CE"][:, :, cz])
seg_ax = seg[:, :, cz]

# Build RGBA overlay
cm_gray = plt.cm.gray
img_out = cm_gray(t1_bg).copy()
seg_sq = np.zeros(t1_bg.shape, dtype=np.uint8)

SB = t1_bg.shape[0]
# re-crop seg to match
seg_r = seg_ax
mask_b = vols["T1CE"][:, :, cz] > 0.04
rs,cs = np.where(mask_b)
if rs.size:
    r0_,r1_,c0_,c1_ = rs.min(), rs.max(), cs.min(), cs.max()
    pad2 = 8
    r0_,r1_ = max(r0_-pad2,0), min(r1_+pad2, mask_b.shape[0]-1)
    c0_,c1_ = max(c0_-pad2,0), min(c1_+pad2, mask_b.shape[1]-1)
    seg_crop_r = seg_r[r0_:r1_+1, c0_:c1_+1]
    h_, w_ = seg_crop_r.shape
    s_ = max(h_, w_)
    seg_sq_ = np.zeros((s_,s_), dtype=np.uint8)
    seg_sq_[(s_-h_)//2:(s_-h_)//2+h_, (s_-w_)//2:(s_-w_)//2+w_] = seg_crop_r
    # resize to SB × SB
    from PIL import Image as PILImage2
    seg_sq_ = np.array(PILImage2.fromarray(seg_sq_).resize((SB, SB), PILImage2.NEAREST))
    seg_sq = seg_sq_

# Overlay colours: 1=necrotic red, 2=edema green, 4=enhancing yellow
for lbl, col in [(1,(1,.15,.15,.7)), (2,(.1,.9,.2,.55)), (4,(1,.85,.0,.8))]:
    img_out[seg_sq == lbl] = col

sub_out = fig.add_axes([p[0]+0.005, p[1]+0.05, p[2]-0.01, p[3]-0.12],
                        facecolor="black")
sub_out.imshow(img_out, origin="lower", interpolation="bilinear")
sub_out.set_xticks([]); sub_out.set_yticks([])
for sp in sub_out.spines.values():
    sp.set_edgecolor(C["border"]); sp.set_linewidth(0.8)

# Legend
leg = [mpatches.Patch(fc=(1,.15,.15), ec="white", lw=.4, label="Necrotic Core"),
       mpatches.Patch(fc=(.1,.9,.2),  ec="white", lw=.4, label="Edema (WT)"),
       mpatches.Patch(fc=(1,.85,0),   ec="white", lw=.4, label="Enhancing (ET)")]
sub_out.legend(handles=leg, loc="lower right", fontsize=5.2,
               framealpha=0.35, facecolor="#111", edgecolor="#555",
               labelcolor="white", handlelength=1)

add_title(fig, p, "Tumour Mask $\\hat{M}$", C["s5"], "Stage 5")
fig.text(p[0]+p[2]/2, p[1]-0.03,
         "Dice (WT): 91.41%  |  Sens: 93.2%",
         ha="center", color=C["subtext"], fontsize=7.5)

# ════════════════════════════════════════════════════════════════════════════
#  ARROWS between panels
# ════════════════════════════════════════════════════════════════════════════
panel_list = ["inp","pre","grph","gnn","out"]
mid_y = PB + PH/2

for i in range(len(panel_list)-1):
    p0 = panels[panel_list[i]]
    p1 = panels[panel_list[i+1]]
    x0 = p0[0] + p0[2] + 0.003
    x1 = p1[0] - 0.003
    ymid = mid_y
    ax_arr = fig.add_axes([0,0,1,1], facecolor="none")
    ax_arr.set_xlim(0,1); ax_arr.set_ylim(0,1); ax_arr.axis("off")
    ax_arr.annotate("", xy=(x1, ymid), xytext=(x0, ymid),
                    arrowprops=dict(arrowstyle="-|>", color=C["arrow"],
                                    lw=2.0, mutation_scale=16))

# ════════════════════════════════════════════════════════════════════════════
#  Global title & footer
# ════════════════════════════════════════════════════════════════════════════
fig.text(0.5, 0.972,
         "Efficient Brain Tumour Segmentation via Superpixel Graph Neural Networks",
         ha="center", va="top", color=C["white"],
         fontsize=13, fontweight="bold")

fig.text(0.5, 0.944,
         "BraTS 2021  ·  GraphSAGE  ·  5-Fold CV  ·  NVIDIA RTX 2060  ·  "
         "91.41% Dice (WT)  |  5.9× faster than nnU-Net",
         ha="center", va="top", color=C["subtext"], fontsize=8)

# ════════════════════════════════════════════════════════════════════════════
out = OUTDIR / "methodology_diagram.png"
fig.savefig(out, dpi=300, bbox_inches="tight",
            facecolor=fig.get_facecolor())
plt.close(fig)
print(f"Saved → {out}")
