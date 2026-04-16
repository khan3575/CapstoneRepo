"""
5-Fold Soft-Voting Ensemble — compact figure matching GraphSAGE compact style.

Layout (left → right):
  Input node  →  5 fold boxes (fan-out/fan-in)  →  mean node  →  P_ens  →  threshold

Output: figures/ensemble_block.png  +  figures/ensemble_block.pdf
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from pathlib import Path

OUTDIR = Path(__file__).resolve().parent.parent / "figures"
OUTDIR.mkdir(exist_ok=True)

# ── colour palette (matches graphsage_compact) ────────────────────────────────
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
NODE_FILL = "#ffffff"
NODE_EDGE = "#2979c0"
OUT_FILL  = "#e8f5e9"
OUT_EDGE  = "#388e3c"
SIG_FILL  = "#fff8e1"
SIG_EDGE  = "#f9a825"

FW, FH = 14, 6.0
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

def circle(ax, cx, cy, r, fc, ec, lw=2.0, zorder=3):
    c = plt.Circle((cx, cy), r, fc=fc, ec=ec, lw=lw, zorder=zorder)
    ax.add_patch(c)

def arrow(ax, x0, y0, x1, y1, color=ARROW_COL, lw=1.5, ms=10):
    ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                arrowprops=dict(arrowstyle="-|>", color=color,
                                lw=lw, mutation_scale=ms))

# ── layout ────────────────────────────────────────────────────────────────────
Y_MID     = FH / 2          # 3.0
FOLD_H    = 0.72
FOLD_W    = 2.00
N_FOLDS   = 5
GAP_V     = 0.20            # vertical gap between fold boxes

total_fold_height = N_FOLDS * FOLD_H + (N_FOLDS - 1) * GAP_V   # 4.40
Y_TOP     = Y_MID + total_fold_height / 2                       # 5.20
Y_BOT     = Y_MID - total_fold_height / 2                       # 0.80

# x-positions
X_IN      = 0.55            # centre of input circle
X_FOLD    = 1.55            # left edge of fold boxes
X_MEAN    = X_FOLD + FOLD_W + 0.70   # centre of mean circle
X_FORM    = X_MEAN + 0.42   # right edge of formula / output label start
X_OUT     = X_MEAN + 1.10   # centre of P_ens output
X_THRESH  = X_OUT + 1.00    # centre of threshold circle
X_FINAL   = X_THRESH + 0.90 # right label

NODE_R    = 0.30
SIG_R     = NODE_R * 0.88

# ── input node ────────────────────────────────────────────────────────────────
circle(ax, X_IN, Y_MID, NODE_R, fc=NODE_FILL, ec=NODE_EDGE)
ax.text(X_IN, Y_MID + NODE_R + 0.14,
        "Node\nembeddings\n(per fold)",
        ha="center", va="bottom", fontsize=8, color=TEXT_DARK,
        fontfamily="DejaVu Sans")

# ── five fold boxes ───────────────────────────────────────────────────────────
fold_ys = []   # centre y of each fold box
for i in range(N_FOLDS):
    by = Y_TOP - (i + 1) * FOLD_H - i * GAP_V
    fold_ys.append(by + FOLD_H / 2)

    rbox(ax, X_FOLD, by, FOLD_W, FOLD_H, BOX_FILL, BOX_EDGE)

    # badge
    bw, bh = 0.52, 0.26
    rbox(ax, X_FOLD + 0.10, by + FOLD_H - bh - 0.06,
         bw, bh, BADGE_BG, BADGE_BG, lw=0, radius=0.07, zorder=3)
    ax.text(X_FOLD + 0.10 + bw / 2, by + FOLD_H - bh / 2 - 0.06,
            f"Fold {i+1}", ha="center", va="center",
            fontsize=8.5, fontweight="bold", color=BADGE_TXT, zorder=4,
            fontfamily="DejaVu Sans")

    # formula in box
    ax.text(X_FOLD + FOLD_W / 2, by + FOLD_H * 0.38,
            r"$\sigma\!\left(z_v^{(" + str(i+1) + r")}\right)$",
            ha="center", va="center",
            fontsize=10.5, color=TEXT_DARK, fontweight="bold")

    # fan-in arrow from input to fold box
    arrow(ax, X_IN + NODE_R, Y_MID, X_FOLD, fold_ys[i])

    # fan-out arrow from fold box to mean circle
    arrow(ax, X_FOLD + FOLD_W, fold_ys[i], X_MEAN - NODE_R, Y_MID)

# ── mean aggregation node ─────────────────────────────────────────────────────
circle(ax, X_MEAN, Y_MID, NODE_R, fc=SIG_FILL, ec=SIG_EDGE, lw=2.0)
ax.text(X_MEAN, Y_MID + 0.01,
        r"$\frac{1}{5}\!\sum$",
        ha="center", va="center",
        fontsize=10, color="#e65100", fontweight="bold")

# ── output circle P_ens ───────────────────────────────────────────────────────
arrow(ax, X_MEAN + NODE_R, Y_MID, X_OUT - NODE_R, Y_MID,
      color=ARROW_COL)
circle(ax, X_OUT, Y_MID, NODE_R, fc=OUT_FILL, ec=OUT_EDGE)
ax.text(X_OUT, Y_MID,
        r"$P_{\mathrm{ens}}$",
        ha="center", va="center",
        fontsize=10, color="#2e7d32", fontweight="bold")
ax.text(X_OUT, Y_MID - NODE_R - 0.14,
        r"$P_{\mathrm{ens}}(v) = \frac{1}{5}\!\sum_{k=1}^{5}\sigma\!\left(z_v^{(k)}\right)$",
        ha="center", va="top",
        fontsize=8.5, color=TEXT_MID,
        fontfamily="DejaVu Sans")

# ── threshold circle ──────────────────────────────────────────────────────────
arrow(ax, X_OUT + NODE_R, Y_MID, X_THRESH - SIG_R, Y_MID,
      color="#f9a825", ms=9)
circle(ax, X_THRESH, Y_MID, SIG_R, fc=SIG_FILL, ec=SIG_EDGE, lw=1.6)
ax.text(X_THRESH, Y_MID + 0.02,
        r"$\tau\!=\!0.5$",
        ha="center", va="center",
        fontsize=8.5, color="#e65100", fontweight="bold")

# ── final binary output ───────────────────────────────────────────────────────
arrow(ax, X_THRESH + SIG_R, Y_MID, X_FINAL - 0.05, Y_MID,
      color=ARROW_COL, ms=9)
ax.text(X_FINAL, Y_MID,
        r"$\hat{y}_v \in \{0,1\}$",
        ha="left", va="center",
        fontsize=9.5, color=TEXT_DARK, fontweight="bold",
        fontfamily="DejaVu Sans")

# ── title / section label ─────────────────────────────────────────────────────
ax.text(FW / 2, FH - 0.18,
        "5-Fold Soft-Voting Ensemble",
        ha="center", va="top",
        fontsize=11, color=TEXT_DARK, fontweight="bold",
        fontfamily="DejaVu Sans")

# ── save ──────────────────────────────────────────────────────────────────────
out_png = OUTDIR / "ensemble_block.png"
out_pdf = OUTDIR / "ensemble_block.pdf"

plt.savefig(out_png, dpi=400, bbox_inches="tight",
            facecolor=BG, transparent=False)
plt.savefig(out_pdf, bbox_inches="tight",
            facecolor=BG, transparent=False)
plt.close(fig)
print(f"Saved → {out_png}")
print(f"Saved → {out_pdf}")
