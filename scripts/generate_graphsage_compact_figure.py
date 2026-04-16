"""
GraphSAGE Architecture — compact horizontal figure, RIGHT-TO-LEFT flow.

Node features start on the RIGHT; output (ŷ) lands on the LEFT.
This lets the graph-construction block (top-right) connect naturally
downward into the input of the GraphSAGE row.

Output: figures/graphsage_compact.png  +  figures/graphsage_compact.pdf
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from pathlib import Path

OUTDIR = Path(__file__).resolve().parent.parent / "figures"
OUTDIR.mkdir(exist_ok=True)

# ── colour palette ────────────────────────────────────────────────────────────
BG        = "#ffffff"
BOX_FILL  = "#e8f2fc"
BOX_FILL2 = "#e8f5e9"
BOX_EDGE  = "#2979c0"
BOX_EDGE2 = "#388e3c"
BADGE_BG  = "#2979c0"
BADGE_TXT = "#ffffff"
TEXT_DARK = "#0d1b2a"
TEXT_MID  = "#2a4a6b"
TEXT_SUB  = "#5a7a96"
ARROW_COL = "#4a7ca8"
BRACE_COL = "#2979c0"
FORM_COL  = "#1a3550"
NODE_FILL = "#ffffff"
NODE_EDGE = "#2979c0"
OUT_FILL  = "#e8f5e9"
OUT_EDGE  = "#388e3c"

FW, FH = 16, 4.6
fig, ax = plt.subplots(figsize=(FW, FH), facecolor=BG)
ax.set_xlim(0, FW)
ax.set_ylim(0, FH)
ax.set_aspect("equal")
ax.axis("off")
fig.patch.set_facecolor(BG)

# ── helpers ───────────────────────────────────────────────────────────────────
def rbox(ax, x, y, w, h, fc, ec, lw=1.8, radius=0.18, zorder=2):
    box = FancyBboxPatch((x, y), w, h,
                         boxstyle=f"round,pad=0,rounding_size={radius}",
                         fc=fc, ec=ec, lw=lw, zorder=zorder)
    ax.add_patch(box)

def arrow_lr(ax, x0, x1, y, color=ARROW_COL, lw=1.8, ms=12):
    """Arrow from x0 to x1 (works for both left and right direction)."""
    ax.annotate("", xy=(x1, y), xytext=(x0, y),
                arrowprops=dict(arrowstyle="-|>", color=color,
                                lw=lw, mutation_scale=ms))

def circle(ax, cx, cy, r, fc, ec, lw=2.0, zorder=3):
    c = plt.Circle((cx, cy), r, fc=fc, ec=ec, lw=lw, zorder=zorder)
    ax.add_patch(c)

# ── layout constants ──────────────────────────────────────────────────────────
Y_CENTER = 2.55
BOX_H    = 1.30
BOX_Y    = Y_CENTER - BOX_H / 2
LAYER_W  = 1.52
MLP_W    = 1.50
GAP      = 0.38
NODE_R   = 0.30
SIG_R    = NODE_R * 0.88

# ── x-positions  (flow: right → left) ────────────────────────────────────────
#   Node-in (right)  →  L1  →  L2  →  L3  →  L4  →  L5  →  MLP  →  Node-out  →  σ  (left)

X_NODE_IN  = 13.30          # centre of input (Node Features) circle

X_L1       = X_NODE_IN - NODE_R - GAP - LAYER_W     # 11.10  (right edge touches node)
X_L2       = X_L1 - GAP - LAYER_W                   #  9.20
X_L3       = X_L2 - GAP - LAYER_W                   #  7.30
X_L4       = X_L3 - GAP - LAYER_W                   #  5.40
X_L5       = X_L4 - GAP - LAYER_W                   #  3.50

X_MLP      = X_L5 - GAP - MLP_W                     #  1.62
X_NODE_OUT = X_MLP - GAP - 0.10                      #  centre: X_MLP - GAP - NODE_R
X_NODE_OUT_C = X_MLP - GAP - NODE_R                  #  1.14  (circle centre)
X_SIGMA_C  = X_NODE_OUT_C - 0.52                     #  0.62  (σ circle centre)

# ── right side: Node-features input circle ────────────────────────────────────
circle(ax, X_NODE_IN, Y_CENTER, NODE_R, fc=NODE_FILL, ec=NODE_EDGE)
ax.text(X_NODE_IN, Y_CENTER + 0.55,
        "Node\nfeatures\n(15-dim)",
        ha="center", va="bottom", fontsize=8.5, color=TEXT_DARK,
        fontfamily="DejaVu Sans")

# ── five SAGEConv layer blocks ────────────────────────────────────────────────
layers = [
    ("L1", "SAGEConv\n15→256",  "BatchNorm→ReLU\n→Dropout(0.1)", X_L1),
    ("L2", "SAGEConv\n256→256", "BatchNorm→ReLU\n→Dropout(0.1)", X_L2),
    ("L3", "SAGEConv\n256→256", "BatchNorm→ReLU\n→Dropout(0.1)", X_L3),
    ("L4", "SAGEConv\n256→256", "BatchNorm→ReLU\n→Dropout(0.1)", X_L4),
    ("L5", "SAGEConv\n256→64",  "BatchNorm only\n(no ReLU/Drop)", X_L5),
]

for badge, line1, line2, bx in layers:
    rbox(ax, bx, BOX_Y, LAYER_W, BOX_H, BOX_FILL, BOX_EDGE)

    # Badge pill (top-left of box)
    bw, bh = 0.34, 0.30
    rbox(ax, bx + 0.10, BOX_Y + BOX_H - bh - 0.07,
         bw, bh, BADGE_BG, BADGE_BG, lw=0, radius=0.08, zorder=3)
    ax.text(bx + 0.10 + bw/2, BOX_Y + BOX_H - bh/2 - 0.07,
            badge, ha="center", va="center",
            fontsize=9.5, fontweight="bold", color=BADGE_TXT, zorder=4,
            fontfamily="DejaVu Sans")

    ax.text(bx + LAYER_W/2, BOX_Y + BOX_H*0.54,
            line1, ha="center", va="center",
            fontsize=9.0, color=TEXT_DARK, fontweight="bold",
            fontfamily="DejaVu Sans")
    ax.text(bx + LAYER_W/2, BOX_Y + BOX_H*0.20,
            line2, ha="center", va="center",
            fontsize=8.0, color=TEXT_MID,
            fontfamily="DejaVu Sans")

# ── MLP head block ────────────────────────────────────────────────────────────
rbox(ax, X_MLP, BOX_Y, MLP_W, BOX_H, BOX_FILL2, BOX_EDGE2)

bw2, bh2 = 0.72, 0.30
rbox(ax, X_MLP + 0.10, BOX_Y + BOX_H - bh2 - 0.07,
     bw2, bh2, BOX_EDGE2, BOX_EDGE2, lw=0, radius=0.08, zorder=3)
ax.text(X_MLP + 0.10 + bw2/2, BOX_Y + BOX_H - bh2/2 - 0.07,
        "MLP head", ha="center", va="center",
        fontsize=9.5, fontweight="bold", color="white", zorder=4,
        fontfamily="DejaVu Sans")

ax.text(X_MLP + MLP_W/2, BOX_Y + BOX_H*0.56,
        "Linear 64→32\nReLU · Dropout(0.1)",
        ha="center", va="center",
        fontsize=9.0, color=TEXT_DARK, fontweight="bold",
        fontfamily="DejaVu Sans")
ax.text(X_MLP + MLP_W/2, BOX_Y + BOX_H*0.18,
        "Linear 32→1",
        ha="center", va="center",
        fontsize=8.0, color="#2e7d32",
        fontfamily="DejaVu Sans")

# ── left side: output circle ──────────────────────────────────────────────────
circle(ax, X_NODE_OUT_C, Y_CENTER, NODE_R, fc=OUT_FILL, ec=OUT_EDGE)
ax.text(X_NODE_OUT_C, Y_CENTER,
        r"$\hat{y}_v$",
        ha="center", va="center",
        fontsize=11, color="#2e7d32", fontweight="bold")

# σ circle
circle(ax, X_SIGMA_C, Y_CENTER, SIG_R,
       fc="#fff8e1", ec="#f9a825", lw=1.6)
ax.text(X_SIGMA_C, Y_CENTER + 0.02,
        r"$\sigma$",
        ha="center", va="center",
        fontsize=11, color="#e65100", fontweight="bold")
ax.text(X_SIGMA_C, Y_CENTER - 0.62,
        r"$(\tau\!=\!0.5)$",
        ha="center", va="top",
        fontsize=7.5, color="#e65100",
        fontfamily="DejaVu Sans")

# ŷ ∈ {0,1} label — to the LEFT of σ
ax.text(X_SIGMA_C - SIG_R - 0.12, Y_CENTER,
        r"$\hat{y} \in \{0,1\}$",
        ha="right", va="center",
        fontsize=9, color=TEXT_DARK,
        fontfamily="DejaVu Sans")

# ── connecting arrows  (all point LEFT) ───────────────────────────────────────
# Node-features → L1  (enter from right side of L1 box)
arrow_lr(ax, X_NODE_IN - NODE_R, X_L1 + LAYER_W, Y_CENTER)

# L1 → L2 → L3 → L4 → L5 → MLP  (exit left edge, enter right edge of next)
connections = [
    (X_L1,        X_L2 + LAYER_W),
    (X_L2,        X_L3 + LAYER_W),
    (X_L3,        X_L4 + LAYER_W),
    (X_L4,        X_L5 + LAYER_W),
    (X_L5,        X_MLP + MLP_W),
    (X_MLP,       X_NODE_OUT_C + NODE_R),
]
for x0, x1 in connections:
    arrow_lr(ax, x0, x1, Y_CENTER)

# output node → σ
arrow_lr(ax, X_NODE_OUT_C - NODE_R, X_SIGMA_C + SIG_R,
         Y_CENTER, color="#f9a825", ms=9)

# ── "5-hop receptive field" brace  (spans L1 right edge to L5 left edge) ──────
brace_y  = BOX_Y + BOX_H + 0.28
tick_y   = BOX_Y + BOX_H + 0.12
brace_x0 = X_L5          # leftmost layer left edge
brace_x1 = X_L1 + LAYER_W  # rightmost layer right edge

ax.plot([brace_x0, brace_x1], [brace_y, brace_y],
        color=BRACE_COL, lw=1.8, solid_capstyle="round")
for xb in [brace_x0, brace_x1]:
    ax.plot([xb, xb], [tick_y, brace_y],
            color=BRACE_COL, lw=1.8, solid_capstyle="round")
ax.text((brace_x0 + brace_x1)/2, brace_y + 0.08,
        "5-hop receptive field",
        ha="center", va="bottom",
        fontsize=8.5, color=BRACE_COL,
        fontfamily="DejaVu Sans")

# ── message-passing formula (below layers) ────────────────────────────────────
form_x = (brace_x0 + brace_x1) / 2
form_y = BOX_Y - 0.42
ax.text(form_x, form_y,
        r"$h_v^{(l)} = \sigma\!\left(W^{(l)} \cdot "
        r"\mathrm{CONCAT}\!\left(h_v^{(l-1)},\ "
        r"\mathrm{mean}_{u \in \mathcal{N}(v)} h_u^{(l-1)}\right)\right)$",
        ha="center", va="top",
        fontsize=10, color=FORM_COL, style="italic")

# ── param count (below output circle) ────────────────────────────────────────
ax.text(X_NODE_OUT_C, BOX_Y - 0.30,
        "439 K params",
        ha="center", va="top",
        fontsize=7.5, color=TEXT_SUB,
        fontfamily="DejaVu Sans")

# ── save ──────────────────────────────────────────────────────────────────────
out_png = OUTDIR / "graphsage_compact.png"
out_pdf = OUTDIR / "graphsage_compact.pdf"

plt.savefig(out_png, dpi=400, bbox_inches="tight",
            facecolor=BG, transparent=False)
plt.savefig(out_pdf, bbox_inches="tight",
            facecolor=BG, transparent=False)
plt.close(fig)
print(f"Saved → {out_png}")
print(f"Saved → {out_pdf}")
