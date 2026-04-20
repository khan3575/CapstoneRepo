"""
Graph construction — clean, compact, presentation-ready.

Three-step horizontal flow:
  Step 1: Raw T1CE MRI slice (clean, no overlay)
  Step 2: SLIC superpixel boundaries only (no node clutter)
  Step 3: Schematic graph — two neat rows of ~10 nodes each,
          intra-slice edges as short arcs, inter-slice as straight verticals
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
from matplotlib.patches import FancyArrowPatch
from skimage.segmentation import slic, mark_boundaries
from skimage.measure import regionprops
from scipy.ndimage import gaussian_filter
from scipy.spatial import cKDTree
from PIL import Image

RNG = np.random.default_rng(42)

BG     = "#0a1628"
CYAN   = "#00d4ff"
ORANGE = "#ff8c14"
WHITE  = "#e8eaf6"
NAVY2  = "#112244"
TUMOUR_T = 0.52

# ── 1. Load T1CE slice ────────────────────────────────────────────────────────
src  = "/Users/sakibkhan/Desktop/Capstone/CapstoneRepo/paperWriting/overleaf_flat/median_BraTS2021_00209_z059.jpg"
full = Image.open(src)
pw   = full.size[0] // 4
raw  = np.array(full.crop((18, 52, pw-8, full.size[1]-6)).convert("L"), dtype=float) / 255.0
raw2 = gaussian_filter(raw, sigma=1.4)
IH, IW = raw.shape

# ── 2. SLIC ───────────────────────────────────────────────────────────────────
def run_slic(img):
    m   = (img > 0.04).astype(np.uint8)
    rgb = np.stack([img]*3, axis=-1)
    return slic(rgb, n_segments=200, compactness=10, sigma=1.0,
                start_label=0, mask=m), m

segs1, mask1 = run_slic(raw)
segs2, mask2 = run_slic(raw2)

# ── 3. Nodes ──────────────────────────────────────────────────────────────────
def get_nodes(segs, bmask, img):
    out = {}
    for r in regionprops(segs + 1):
        sid = r.label - 1
        m   = segs == sid
        if bmask[m].mean() < 0.5:
            continue
        cy, cx = r.centroid
        out[sid] = dict(cx=float(cx), cy=float(cy), intensity=float(img[m].mean()))
    return out

nodes1 = get_nodes(segs1, mask1, raw)
nodes2 = get_nodes(segs2, mask2, raw2)

# ── 4. Boundary images ────────────────────────────────────────────────────────
def bounded(img, segs):
    return mark_boundaries(np.stack([img]*3, -1), segs,
                           color=(0, 0.82, 0.92), outline_color=None, mode='thin')

img1_b = bounded(raw,  segs1)
img2_b = bounded(raw2, segs2)

# ── 5. Inter-slice KNN for schematic ─────────────────────────────────────────
def pick_representative(nodes, n=10):
    brain = {s: d for s, d in nodes.items()
             if 0.15*IW < d['cx'] < 0.85*IW and 0.15*IH < d['cy'] < 0.85*IH}
    keys = sorted(brain.keys(), key=lambda s: brain[s]['cx'])
    step = max(1, len(keys)//n)
    return [keys[i] for i in range(0, len(keys), step)][:n]

rep1 = pick_representative(nodes1, 10)
rep2 = pick_representative(nodes2, 10)

pts_r1 = np.array([[nodes1[s]['cx'], nodes1[s]['cy']] for s in rep1])
pts_r2 = np.array([[nodes2[s]['cx'], nodes2[s]['cy']] for s in rep2])
tree   = cKDTree(pts_r2)
rep_inter = []
for i, sid in enumerate(rep1):
    dists, idxs = tree.query(pts_r1[i], k=2)
    for d, j in zip(dists, idxs):
        if d <= 60:
            rep_inter.append((i, j))   # store indices, not IDs

# intra-slice RAG for representative nodes (adjacent in sorted-by-x order)
rep_rag = [(i, i+1) for i in range(len(rep1)-1)]

# ── 6. Figure ─────────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(14, 4.4), facecolor=BG)

# 5 columns: slice1 | arrow | slice2 | arrow | graph
gs = fig.add_gridspec(1, 5,
                      width_ratios=[1, 0.13, 1, 0.13, 1.3],
                      left=0.01, right=0.99,
                      top=0.84, bottom=0.14,
                      wspace=0.04)

ax1  = fig.add_subplot(gs[0])
ax_a = fig.add_subplot(gs[1])
ax2  = fig.add_subplot(gs[2])
ax_b = fig.add_subplot(gs[3])
ax3  = fig.add_subplot(gs[4])

for ax in (ax1, ax_a, ax2, ax_b, ax3):
    ax.set_facecolor(BG)
    ax.axis('off')

# ── Step 1: Raw MRI (Slice N) ─────────────────────────────────────────────────
ax1.imshow(raw, cmap='gray', aspect='equal', interpolation='bilinear',
           vmin=0, vmax=1)
ax1.set_title("Slice N  (T1CE)", color=CYAN, fontsize=10.5,
              fontweight='bold', pad=5)
ax1.text(0.04, 0.97, "Step 1", color=WHITE, fontsize=8,
         transform=ax1.transAxes, va='top', alpha=0.6)

# ── Step 2: SLIC boundaries (both slices stacked, compact) ───────────────────
GAP = 10
stack = np.zeros((IH*2 + GAP, IW, 3))
stack[:IH]          = img1_b
stack[IH+GAP:]      = img2_b
ax2.imshow(stack, aspect='auto', interpolation='bilinear')

# 20 inter-slice edges on the MRI panel
ids1  = list(nodes1.keys())
ids2  = list(nodes2.keys())
pts1  = np.array([[nodes1[i]['cx'], nodes1[i]['cy']] for i in ids1])
pts2  = np.array([[nodes2[i]['cx'], nodes2[i]['cy']] for i in ids2])
tree2 = cKDTree(pts2)
inter_mri = []
for k, sid in enumerate(ids1):
    dists, idxs = tree2.query(pts1[k], k=2)
    for d, j in zip(dists, idxs):
        if d <= 30:
            inter_mri.append((sid, ids2[j]))

# sample 20 spread across brain interior
interior = [(s1,s2) for s1,s2 in inter_mri
            if 0.2*IW < nodes1[s1]['cx'] < 0.8*IW
            and 0.25*IH < nodes1[s1]['cy'] < 0.75*IH]
sample = [interior[i] for i in RNG.choice(len(interior),
          size=min(20, len(interior)), replace=False)]

y_off = IH + GAP
for s1, s2 in sample:
    ax2.plot([nodes1[s1]['cx'], nodes2[s2]['cx']],
             [nodes1[s1]['cy'], nodes2[s2]['cy'] + y_off],
             color=ORANGE, alpha=0.65, lw=0.9, zorder=3)

# small node dots on both slices
for d in nodes1.values():
    c = ORANGE if d['intensity'] > TUMOUR_T else CYAN
    ax2.plot(d['cx'], d['cy'], 'o', color=c,
             markersize=2.2, markeredgewidth=0, alpha=0.75, zorder=4)
for d in nodes2.values():
    c = ORANGE if d['intensity'] > TUMOUR_T else CYAN
    ax2.plot(d['cx'], d['cy'] + y_off, 'o', color=c,
             markersize=2.2, markeredgewidth=0, alpha=0.75, zorder=4)

# gap divider label
ax2.axhline(y=IH + GAP/2, color=ORANGE, lw=0.6, alpha=0.5, zorder=2)
ax2.text(IW*0.5, IH + GAP/2 - 1, "Slice N+1 →",
         color=ORANGE, fontsize=6.5, ha='center', va='bottom', alpha=0.8)

ax2.set_xlim(0, IW)
ax2.set_ylim(IH*2+GAP, 0)
ax2.set_title("SLIC + Node Construction", color=CYAN,
              fontsize=10.5, fontweight='bold', pad=5)
ax2.text(0.04, 0.97, "Step 2", color=WHITE, fontsize=8,
         transform=ax2.transAxes, va='top', alpha=0.6)

# ── Arrows ────────────────────────────────────────────────────────────────────
for ax_arrow in (ax_a, ax_b):
    ax_arrow.text(0.5, 0.5, "→", color=CYAN, fontsize=22,
                  ha='center', va='center', fontweight='bold',
                  transform=ax_arrow.transAxes)

# ── Step 3: Schematic graph ───────────────────────────────────────────────────
# Two rows of 10 nodes, evenly spaced
xs = np.linspace(0.06, 0.94, 10)
Y1, Y2 = 0.72, 0.28
VPAD = 0.12   # vertical padding for labels

# intra-slice arcs (curved connections between adjacent nodes in same row)
from matplotlib.patches import Arc
for row_y, row_ns, row_nd in [(Y1, nodes1, rep1), (Y2, nodes2, rep2)]:
    for i, j in rep_rag:
        x1, x2 = xs[i], xs[j]
        mid_x  = (x1 + x2) / 2
        # draw as a small arc above (for top row) or below (for bottom row)
        sign = 1 if row_y == Y1 else -1
        t = np.linspace(0, np.pi, 30)
        arc_x = mid_x + (x2-x1)/2 * np.cos(t)
        arc_y = row_y + sign * 0.045 * np.sin(t)
        ax3.plot(arc_x, arc_y, color=WHITE, alpha=0.35, lw=0.9,
                 transform=ax3.transAxes, zorder=2)

# inter-slice edges (straight lines, mostly vertical → clean)
seen = set()
for i, j in rep_inter:
    key = (i, j)
    if key in seen:
        continue
    seen.add(key)
    ax3.plot([xs[i], xs[j]], [Y1, Y2],
             color=ORANGE, alpha=0.70, lw=1.1,
             transform=ax3.transAxes, zorder=3)

# nodes
for i, sid in enumerate(rep1):
    c = ORANGE if nodes1[sid]['intensity'] > TUMOUR_T else CYAN
    ax3.plot(xs[i], Y1, 'o', color=c, markersize=9,
             markeredgecolor='white', markeredgewidth=0.6,
             alpha=1.0, zorder=5, transform=ax3.transAxes)

for i, sid in enumerate(rep2):
    c = ORANGE if nodes2[sid]['intensity'] > TUMOUR_T else CYAN
    ax3.plot(xs[i], Y2, 'o', color=c, markersize=9,
             markeredgecolor='white', markeredgewidth=0.6,
             alpha=1.0, zorder=5, transform=ax3.transAxes)

# labels
ax3.text(0.5, Y1+VPAD, f"Slice N  ·  ~{len(nodes1)} nodes",
         color=CYAN, fontsize=9, ha='center', fontweight='bold',
         transform=ax3.transAxes)
ax3.text(0.5, Y2-VPAD, f"Slice N+1  ·  ~{len(nodes2)} nodes",
         color=CYAN, fontsize=9, ha='center', fontweight='bold',
         transform=ax3.transAxes)

# annotation arrows
ax3.annotate("intra-slice\nedge (RAG)",
             xy=(xs[2], Y1+0.046), xytext=(0.08, Y1+0.19),
             xycoords='axes fraction', textcoords='axes fraction',
             color=WHITE, fontsize=7.5, ha='center', alpha=0.75,
             arrowprops=dict(arrowstyle='->', color=WHITE, lw=0.7, alpha=0.6))

ax3.annotate("inter-slice\nedge (k-NN)",
             xy=((xs[4]+xs[4])/2, (Y1+Y2)/2),
             xytext=(0.88, 0.50),
             xycoords='axes fraction', textcoords='axes fraction',
             color=ORANGE, fontsize=7.5, ha='center', alpha=0.85,
             arrowprops=dict(arrowstyle='->', color=ORANGE, lw=0.7, alpha=0.7))

ax3.set_xlim(0, 1); ax3.set_ylim(0, 1)
ax3.set_title("Graph Structure", color=CYAN,
              fontsize=10.5, fontweight='bold', pad=5)
ax3.text(0.04, 0.97, "Step 3", color=WHITE, fontsize=8,
         transform=ax3.transAxes, va='top', alpha=0.6)

# ── Bottom caption ────────────────────────────────────────────────────────────
fig.text(0.5, 0.04,
         "~92 nodes  ·  ~180 edges per graph unit  ·  "
         "Shared boundary → intra-slice edge   ·   k-NN (k=3) → inter-slice edge",
         color=WHITE, fontsize=8.8, ha='center', alpha=0.78)

out = "/Users/sakibkhan/Desktop/Capstone/CapstoneRepo/paperWriting/presentationDetails/graph_construction_visualization.png"
plt.savefig(out, dpi=180, bbox_inches='tight', facecolor=BG, edgecolor='none')
plt.close()
print(f"Saved → {out}")
