"""
GraphSAGE Architecture Figure — slide-optimised, single-axes data-coordinate layout.

All drawing is done in data coords (x: 0..22, y: 0..10) on one axes.
No transFigure tricks → no overlap, no clipping.

Layout:
  A) Input graph          x: 0.2 – 3.0
  B) 5 GraphSAGE layers   x: 3.3 – 13.2  (vertically stacked blocks)
  C) MLP Head             x: 13.5 – 17.0 (top→bottom stack)
  D) Inference            x: 17.3 – 21.8 (top→bottom stack)

Output: figures/graphsage_architecture.png + .pdf
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from pathlib import Path

OUTDIR = Path(__file__).resolve().parent.parent / "figures"
OUTDIR.mkdir(exist_ok=True)

# ── palette ───────────────────────────────────────────────────────────────────
BG      = "#ffffff"
PANEL   = "#f4f8fd"
PANEL2  = "#f0f7f0"
BLUE    = "#2979c0"
GREEN   = "#388e3c"
ORANGE  = "#e65100"
PURPLE  = "#6a1b9a"
TEAL    = "#00796b"
RED     = "#c62828"
TEXT    = "#0d1b2a"
SUBTEXT = "#5a7a96"
WHITE   = "#ffffff"

FW, FH = 22, 10
fig, ax = plt.subplots(figsize=(FW, FH), facecolor=BG)
ax.set_facecolor(BG)
ax.set_xlim(0, FW)
ax.set_ylim(0, FH)
ax.set_aspect("equal")
ax.axis("off")

# ── helpers ───────────────────────────────────────────────────────────────────
def rbox(x, y, w, h, fc, ec, lw=2.0, radius=0.18):
    box = FancyBboxPatch((x, y), w, h,
                         boxstyle=f"round,pad=0,rounding_size={radius}",
                         fc=fc, ec=ec, lw=lw, zorder=2)
    ax.add_patch(box)

def arrow(x0, y0, x1, y1, color=BLUE, lw=2.0, ms=14):
    ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                arrowprops=dict(arrowstyle="-|>", color=color,
                                lw=lw, mutation_scale=ms))

def badge(x, y, w, h, label, fc, tc=WHITE, fs=11):
    rbox(x, y, w, h, fc, fc, lw=0, radius=0.12)
    ax.text(x + w/2, y + h/2, label,
            ha="center", va="center",
            fontsize=fs, fontweight="bold", color=tc,
            fontfamily="DejaVu Sans", zorder=4)

# ── layout constants ──────────────────────────────────────────────────────────
CY0 = 2.10   # content bottom  (extra room below for formula + stats)
CY1 = 7.80   # content top     (extra room above for section headers)
CH  = CY1 - CY0   # 5.70

# column boundaries
AX0, AX1 = 0.20, 3.00   # Input graph
BX0, BX1 = 3.30, 13.20  # GraphSAGE layers
CX0, CX1 = 13.50, 17.00 # MLP Head
DX0, DX1 = 17.30, 21.80 # Inference

# ════════════════════════════════════════════════════════════════════════════════
#  Global title
# ════════════════════════════════════════════════════════════════════════════════
ax.text(FW/2, 9.78,
        "GraphSAGE Architecture  ·  Brain Tumour Segmentation",
        ha="center", va="center",
        fontsize=17, fontweight="bold", color=TEXT, fontfamily="DejaVu Sans")
ax.text(FW/2, 9.52,
        "5-Layer Mean-Aggregation GraphSAGE  ·  "
        "Input: 15-dim superpixel features  ·  "
        "Output: per-node binary tumour label",
        ha="center", va="center",
        fontsize=11, color=SUBTEXT, fontfamily="DejaVu Sans")

# ════════════════════════════════════════════════════════════════════════════════
#  A — Input graph
# ════════════════════════════════════════════════════════════════════════════════
AW = AX1 - AX0
rbox(AX0, CY0, AW, CH, PANEL, BLUE, lw=2.0, radius=0.20)

ax.text(AX0 + AW/2, CY1 - 0.30,
        "Input Graph  G=(V,E,X)",
        ha="center", va="center",
        fontsize=12, fontweight="bold", color=TEXT, fontfamily="DejaVu Sans")
ax.text(AX0 + AW/2, CY1 - 0.70,
        "≈92 nodes · ≈180 edges",
        ha="center", va="center",
        fontsize=10, color=SUBTEXT, fontfamily="DejaVu Sans")

# mini graph diagram
cx, cy = AX0 + AW/2, (CY0 + CY1) / 2 - 0.30
sc = 0.80
mini_nodes = {
    "v":  (cx,            cy),
    "n1": (cx - 0.68*sc,  cy + 0.65*sc),
    "n2": (cx + 0.68*sc,  cy + 0.65*sc),
    "n3": (cx - 0.90*sc,  cy - 0.15*sc),
    "n4": (cx + 0.90*sc,  cy - 0.15*sc),
    "n5": (cx - 0.35*sc,  cy - 0.85*sc),
    "n6": (cx + 0.35*sc,  cy - 0.85*sc),
}
mini_edges = [("v","n1"),("v","n2"),("v","n3"),("v","n4"),("v","n5"),("v","n6")]

for a, b in mini_edges:
    x0, y0 = mini_nodes[a]; x1, y1 = mini_nodes[b]
    ax.plot([x0, x1], [y0, y1], color="#c8d8e8", lw=1.5, zorder=1)
for nb in ["n1","n2","n3","n4","n5","n6"]:
    x0, y0 = mini_nodes[nb]; x1, y1 = mini_nodes["v"]
    ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                arrowprops=dict(arrowstyle="-|>", color=ORANGE,
                                lw=1.2, mutation_scale=9, alpha=0.7))

for name, (nx, ny) in mini_nodes.items():
    if name == "v":
        ax.add_patch(plt.Circle((nx, ny), 0.25, fc="#dceeff", ec=BLUE, lw=2.5, zorder=3))
        ax.text(nx, ny, "v", ha="center", va="center",
                fontsize=11, fontweight="bold", color=TEXT, zorder=4,
                fontfamily="DejaVu Sans")
    else:
        ax.add_patch(plt.Circle((nx, ny), 0.15, fc=PANEL, ec=BLUE, lw=1.5, zorder=3))

ax.text(AX0 + AW/2, CY0 + 0.60,
        "mean aggregation from N(v)",
        ha="center", va="center", fontsize=9.5, color=ORANGE,
        style="italic", fontfamily="DejaVu Sans")
ax.text(AX0 + AW/2, CY0 + 0.22,
        r"$h_v \in \mathbb{R}^{15}$",
        ha="center", va="center", fontsize=11, color=BLUE, fontweight="bold")

ax.text(AX0, CY1 + 0.85, "A",
        fontsize=16, fontweight="bold", color=BLUE, fontfamily="DejaVu Sans")

# ════════════════════════════════════════════════════════════════════════════════
#  B — 5 GraphSAGE layer blocks (stacked bottom→top)
# ════════════════════════════════════════════════════════════════════════════════
layers = [
    ("L1", "SAGEConv  15 → 256",  "BatchNorm · ReLU · Dropout(0.1)",       True),
    ("L2", "SAGEConv  256 → 256", "BatchNorm · ReLU · Dropout(0.1)",       True),
    ("L3", "SAGEConv  256 → 256", "BatchNorm · ReLU · Dropout(0.1)",       True),
    ("L4", "SAGEConv  256 → 256", "BatchNorm · ReLU · Dropout(0.1)",       True),
    ("L5", "SAGEConv  256 → 64",  "BatchNorm only  (no ReLU / Dropout)",   False),
]

N_L   = len(layers)
BW    = BX1 - BX0
BGAP  = 0.20                            # gap between blocks
BLKH  = (CH - (N_L - 1) * BGAP) / N_L  # ≈ 1.28

for i, (lbl, line1, line2, std) in enumerate(layers):
    by = CY0 + i * (BLKH + BGAP)
    fc = PANEL  if std else PANEL2
    ec = BLUE   if std else GREEN

    rbox(BX0, by, BW, BLKH, fc, ec, lw=2.0, radius=0.18)

    # badge pill
    badge(BX0 + 0.18, by + BLKH - 0.50, 0.72, 0.38,
          lbl, BLUE if std else GREEN, fs=11)

    # operation name
    ax.text(BX0 + BW/2, by + BLKH * 0.60,
            line1,
            ha="center", va="center",
            fontsize=13, fontweight="bold", color=TEXT,
            fontfamily="DejaVu Sans")

    # normalisation sub-line
    ax.text(BX0 + BW/2, by + BLKH * 0.28,
            line2,
            ha="center", va="center",
            fontsize=11, color=SUBTEXT if std else GREEN,
            fontfamily="DejaVu Sans")

    # vertical connector to next block
    if i < N_L - 1:
        ax.plot([BX0 + BW/2] * 2,
                [by + BLKH, by + BLKH + BGAP],
                color=BLUE, lw=1.8)

# formula below B — well above the stats bar
ax.text((BX0 + BX1) / 2, CY0 - 0.30,
        r"$h_v^{(l)} = \sigma\!\left(W^{(l)} \cdot \mathrm{CONCAT}"
        r"\!\left(h_v^{(l-1)},\ \mathrm{mean}_{u \in \mathcal{N}(v)}"
        r"\ h_u^{(l-1)}\right)\right)$",
        ha="center", va="top",
        fontsize=12, color=TEXT)

# Section B header — "5-hop receptive field" merged in as subtitle, no brace
ax.text(BX0, CY1 + 0.87,
        "B  —  5-Layer GraphSAGE Encoder",
        fontsize=14, fontweight="bold", color=BLUE,
        fontfamily="DejaVu Sans")
ax.text(BX0, CY1 + 0.62,
        "5-hop receptive field",
        fontsize=10, color=SUBTEXT,
        fontfamily="DejaVu Sans")

# ════════════════════════════════════════════════════════════════════════════════
#  C — MLP Head  (stacked top → bottom so flow reads downward)
# ════════════════════════════════════════════════════════════════════════════════
mlp_steps = [
    (r"$h_v^{(5)}$  [64-dim]", TEAL,   "#e0f2f1"),
    ("Linear  64 → 32",        PURPLE, "#f3e5f5"),
    ("ReLU",                   ORANGE, "#fff3e0"),
    ("Dropout (0.1)",          RED,    "#ffebee"),
    ("Linear  32 → 1",         PURPLE, "#f3e5f5"),
    ("Logit  z_v",             BLUE,   "#dceeff"),
]

CW    = CX1 - CX0
N_C   = len(mlp_steps)
CGAP  = 0.18
CSH   = (CH - (N_C - 1) * CGAP) / N_C   # ≈ 0.94

for j, (label, color, bg) in enumerate(mlp_steps):
    # top-to-bottom: j=0 is at top (CY1 - CSH), j=N-1 at bottom (CY0)
    sy = CY1 - (j + 1) * CSH - j * CGAP
    rbox(CX0, sy, CW, CSH, bg, color, lw=2.0, radius=0.15)
    ax.text(CX0 + CW/2, sy + CSH/2, label,
            ha="center", va="center",
            fontsize=12, fontweight="bold", color=color,
            fontfamily="DejaVu Sans")
    if j < N_C - 1:
        next_sy = CY1 - (j + 2) * CSH - (j + 1) * CGAP
        arrow(CX0 + CW/2, sy, CX0 + CW/2, next_sy + CSH,
              color=color, lw=1.8, ms=11)

ax.text(CX0, CY1 + 0.85,
        "C  —  MLP Head",
        fontsize=14, fontweight="bold", color=PURPLE,
        fontfamily="DejaVu Sans")

# ════════════════════════════════════════════════════════════════════════════════
#  D — Inference  (stacked top → bottom)
# ════════════════════════════════════════════════════════════════════════════════
out_steps = [
    ("σ(z_v)  Sigmoid",     GREEN,  "#e8f5e9", "P(tumour) ∈ [0, 1]"),
    ("τ = 0.5  Threshold",  ORANGE, "#fff3e0", "binary decision"),
    (r"$\hat{y}_v \in \{0,1\}$",
                             TEAL,   "#e0f2f1", "per-node tumour label"),
    ("Expand to pixels",    GREEN,  "#f0f7f0", "superpixel sᵢ → all pixels"),
]

DW    = DX1 - DX0
N_D   = len(out_steps)
DGAP  = 0.18
DSH   = (CH - (N_D - 1) * DGAP) / N_D   # ≈ 1.60

for j, (label, color, bg, sub) in enumerate(out_steps):
    sy = CY1 - (j + 1) * DSH - j * DGAP
    lw = 1.5 if j == N_D - 1 else 2.0
    rbox(DX0, sy, DW, DSH, bg, color, lw=lw, radius=0.15)
    ax.text(DX0 + DW/2, sy + DSH * 0.65,
            label,
            ha="center", va="center",
            fontsize=12, fontweight="bold", color=color,
            fontfamily="DejaVu Sans")
    ax.text(DX0 + DW/2, sy + DSH * 0.28,
            sub,
            ha="center", va="center",
            fontsize=10, color=SUBTEXT,
            fontfamily="DejaVu Sans")
    if j < N_D - 1:
        next_sy = CY1 - (j + 2) * DSH - (j + 1) * DGAP
        arrow(DX0 + DW/2, sy, DX0 + DW/2, next_sy + DSH,
              color=color, lw=1.8, ms=11)

ax.text(DX0, CY1 + 0.85,
        "D  —  Inference",
        fontsize=14, fontweight="bold", color=GREEN,
        fontfamily="DejaVu Sans")

# ════════════════════════════════════════════════════════════════════════════════
#  Inter-section arrows
# ════════════════════════════════════════════════════════════════════════════════
mid_y = (CY0 + CY1) / 2
arrow(AX1, mid_y, BX0, mid_y, color=BLUE)
arrow(BX1, mid_y, CX0, mid_y, color=TEAL)
arrow(CX1, mid_y, DX0, mid_y, color=GREEN)

# ── vertical dividers ─────────────────────────────────────────────────────────
for xd in [(AX1 + BX0)/2, (BX1 + CX0)/2, (CX1 + DX0)/2]:
    ax.plot([xd, xd], [CY0 - 0.05, CY1 + 0.05],
            color="#c8d8e8", lw=1.2, linestyle="--")

# ── stats bar ─────────────────────────────────────────────────────────────────
stats = [
    ("Parameters",      "439,041"),
    ("Checkpoint",      "5.1 MB"),
    ("Receptive field", "5-hop"),
    ("Aggregation",     "Mean pooling"),
    ("Loss fn",         "BCEWithLogits  w⁺=9.0"),
    ("Threshold τ",     "0.5"),
    ("Inference time",  "75.4 ms / patient"),
]
bar_y = CY0 - 0.80
ax.plot([AX0, DX1], [bar_y, bar_y], color="#c8d8e8", lw=1.0)
span = DX1 - AX0
for k, (key, val) in enumerate(stats):
    xk = AX0 + (k + 0.5) / len(stats) * span
    ax.text(xk, bar_y - 0.15, key,
            ha="center", va="top",
            fontsize=9, color=SUBTEXT, fontfamily="DejaVu Sans")
    ax.text(xk, bar_y - 0.48, val,
            ha="center", va="top",
            fontsize=10, fontweight="bold", color=TEXT,
            fontfamily="DejaVu Sans")

# ── save ──────────────────────────────────────────────────────────────────────
out_png = OUTDIR / "graphsage_architecture.png"
out_pdf = OUTDIR / "graphsage_architecture.pdf"
fig.savefig(out_png, dpi=400, bbox_inches="tight", facecolor=BG)
fig.savefig(out_pdf, bbox_inches="tight", facecolor=BG)
plt.close(fig)
print(f"Saved → {out_png}")
print(f"Saved → {out_pdf}")
