"""
GraphSAGE Architecture — formal presentation redesign.

Four-stage left-to-right visual narrative:
  Stage 1: Superpixel graph — target node + neighbourhood + feature bars
  Stage 2: Mean aggregation — neighbour bars converging into one
  Stage 3: Node update — concatenate + linear transform → new embedding
  Stage 4: 5-layer architecture tower + binary output

Design principles:
  • Only 2 accent colours (cyan + orange) on a dark navy ground
  • Feature vectors drawn as actual bar glyphs, not text labels
  • Aggregation shown as a visual convergence funnel
  • Core GraphSAGE equation displayed once, cleanly
  • Rounded cards with consistent padding; no distracting gradients
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch

RNG = np.random.default_rng(7)

# ── Palette ───────────────────────────────────────────────────────────────────
BG       = "#0a1628"
PANEL_BG = "#0f1f38"       # slightly lighter panel fill
CYAN     = "#00d4ff"
ORANGE   = "#ff8c14"
WHITE    = "#e8eaf6"
DIM      = "#2a4060"
EDGE_COL = "#1e3a5f"

# ── Feature-bar glyph helper ─────────────────────────────────────────────────
# Draw a small "embedding vector" as N thin vertical bars next to a point.
N_BARS   = 9       # bars per glyph
BAR_W    = 0.011   # width of one bar in axes coords
BAR_GAP  = 0.003
BAR_MAXH = 0.11    # maximum bar height in axes coords

_bar_vals = {
    'target':   RNG.uniform(0.4, 1.0, N_BARS),
    'n0':       RNG.uniform(0.2, 0.9, N_BARS),
    'n1':       RNG.uniform(0.2, 0.9, N_BARS),
    'n2':       RNG.uniform(0.2, 0.9, N_BARS),
    'n3':       RNG.uniform(0.2, 0.9, N_BARS),
    'agg':      None,   # computed below as mean of neighbours
    'updated':  RNG.uniform(0.5, 1.0, N_BARS),
}
_bar_vals['agg'] = np.mean([_bar_vals['n0'], _bar_vals['n1'],
                             _bar_vals['n2'], _bar_vals['n3']], axis=0)

def draw_fbar(ax, anchor_x, anchor_y, vals, color, alpha=1.0, scale=1.0):
    """Draw a feature-bar glyph.  anchor = bottom-left of the glyph block."""
    total_w = N_BARS * (BAR_W + BAR_GAP) - BAR_GAP
    for i, v in enumerate(vals):
        bx = anchor_x + i * (BAR_W + BAR_GAP)
        bh = v * BAR_MAXH * scale
        r = mpatches.Rectangle((bx, anchor_y), BAR_W, bh,
                                facecolor=color, edgecolor='none',
                                alpha=alpha * 0.85, zorder=4,
                                transform=ax.transAxes)
        ax.add_patch(r)
    # thin bottom line under glyph
    ax.plot([anchor_x, anchor_x + total_w], [anchor_y, anchor_y],
            color=color, lw=0.6, alpha=alpha * 0.5,
            transform=ax.transAxes, zorder=3)
    return total_w   # return width for centering

def fbar_cx(anchor_x):
    """Return centre-x of a fbar starting at anchor_x."""
    return anchor_x + (N_BARS * (BAR_W + BAR_GAP) - BAR_GAP) / 2

# ── Figure ────────────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(14, 5.1), facecolor=BG)

# 7 columns: stage1 | arr | stage2 | arr | stage3 | arr | stage4
gs = fig.add_gridspec(1, 7,
                      width_ratios=[1.05, 0.07, 1.10, 0.07, 1.10, 0.07, 0.90],
                      left=0.01, right=0.99,
                      top=0.86, bottom=0.12,
                      wspace=0.0)

ax1   = fig.add_subplot(gs[0])   # graph
axA   = fig.add_subplot(gs[1])   # arrow
ax2   = fig.add_subplot(gs[2])   # aggregate
axB   = fig.add_subplot(gs[3])   # arrow
ax3   = fig.add_subplot(gs[4])   # update
axC   = fig.add_subplot(gs[5])   # arrow
ax4   = fig.add_subplot(gs[6])   # architecture

for ax in (ax1, axA, ax2, axB, ax3, axC, ax4):
    ax.set_facecolor(BG)
    ax.set_xlim(0, 1);  ax.set_ylim(0, 1)
    ax.axis('off')

# ── STAGE 1 — Superpixel Graph ────────────────────────────────────────────────
ax1.set_title("Graph Input", color=CYAN, fontsize=10.5,
              fontweight='bold', pad=5)

# Node positions (axes coords)
cx, cy = 0.50, 0.60   # target node
nbrs = [
    (0.18, 0.80),   # top-left    — healthy
    (0.78, 0.82),   # top-right   — tumour
    (0.14, 0.38),   # bottom-left — healthy
    (0.82, 0.38),   # bottom-right— healthy
]
nbr_colors  = [CYAN, ORANGE, CYAN, CYAN]
nbr_labels  = ["u₁", "u₂", "u₃", "u₄"]
nbr_barkeys = ["n0", "n1", "n2", "n3"]

# edges (draw first)
for nx, ny in nbrs:
    ax1.plot([cx, nx], [cy, ny], color=EDGE_COL, lw=1.4, zorder=1)
# one inter-neighbour edge for realism
ax1.plot([nbrs[0][0], nbrs[1][0]], [nbrs[0][1], nbrs[1][1]],
         color=DIM, lw=0.9, alpha=0.55, zorder=1)

# neighbor nodes
NODE_R = 12
for i, (nx, ny) in enumerate(nbrs):
    ax1.plot(nx, ny, 'o', color=nbr_colors[i],
             markersize=NODE_R, markeredgecolor=WHITE,
             markeredgewidth=0.7, zorder=3, transform=ax1.transAxes)
    ax1.text(nx, ny, nbr_labels[i], color=BG, fontsize=7.5,
             ha='center', va='center', zorder=4,
             fontweight='bold', transform=ax1.transAxes)

# feature bars for each neighbour (placed below the node)
fbar_anchors = {}
for i, (nx, ny) in enumerate(nbrs):
    total_w = N_BARS * (BAR_W + BAR_GAP) - BAR_GAP
    ax = ny - 0.19          # bar bottom (below node marker)
    an = nx - total_w / 2   # bar left (centred under node)
    fbar_anchors[i] = (an, ax)
    draw_fbar(ax1, an, ax, _bar_vals[nbr_barkeys[i]],
              nbr_colors[i], alpha=0.80, scale=0.85)

# target node
ax1.plot(cx, cy, 'o', color=WHITE, markersize=17,
         markeredgecolor=CYAN, markeredgewidth=2.0,
         zorder=5, transform=ax1.transAxes)
ax1.text(cx, cy, "v", color=BG, fontsize=10, ha='center', va='center',
         fontweight='bold', zorder=6, transform=ax1.transAxes)

# feature bar for target (above node)
tv_total = N_BARS * (BAR_W + BAR_GAP) - BAR_GAP
tv_an = cx - tv_total / 2
tv_ay = cy + 0.11
draw_fbar(ax1, tv_an, tv_ay, _bar_vals['target'],
          WHITE, alpha=0.90, scale=1.0)
ax1.text(cx, tv_ay + BAR_MAXH + 0.03, "hᵥ ∈ ℝ¹⁵",
         color=WHITE, fontsize=8, ha='center', va='bottom',
         transform=ax1.transAxes, alpha=0.85)

ax1.text(0.50, 0.06, "15-dim node features per superpixel",
         color=WHITE, fontsize=8, ha='center', va='bottom',
         transform=ax1.transAxes, alpha=0.65)

# ── STAGE 2 — Aggregate ───────────────────────────────────────────────────────
ax2.set_title("Mean Aggregation", color=CYAN, fontsize=10.5,
              fontweight='bold', pad=5)

# Four mini-nodes on the left, feature bars extending right
mini_ys = [0.82, 0.65, 0.48, 0.31]
mini_cols = [CYAN, ORANGE, CYAN, CYAN]
mini_lbl  = ["u₁", "u₂", "u₃", "u₄"]
mini_keys = ["n0", "n1", "n2", "n3"]

MX   = 0.13    # mini-node x
BARX = 0.23    # bar start x (right of mini-nodes)
AGG_X = 0.78   # aggregated bar x
AGG_Y = 0.52   # aggregated bar bottom
MEAN_X = 0.555 # mean-symbol x

for i, (my, mc, mk) in enumerate(zip(mini_ys, mini_cols, mini_keys)):
    # mini node
    ax2.plot(MX, my, 'o', color=mc, markersize=10,
             markeredgecolor=WHITE, markeredgewidth=0.6,
             zorder=3, transform=ax2.transAxes)
    ax2.text(MX, my, mini_lbl[i], color=BG, fontsize=6.5,
             ha='center', va='center', zorder=4,
             fontweight='bold', transform=ax2.transAxes)
    # feature bar
    draw_fbar(ax2, BARX, my - 0.05, _bar_vals[mk],
              mc, alpha=0.80, scale=0.75)
    # convergence line from end of bar to MEAN symbol
    bw = N_BARS * (BAR_W + BAR_GAP) - BAR_GAP
    ax2.annotate("", xy=(MEAN_X, 0.55), xytext=(BARX + bw + 0.02, my),
                 xycoords='axes fraction', textcoords='axes fraction',
                 arrowprops=dict(arrowstyle="-", color=DIM, lw=0.9, alpha=0.6))

# MEAN symbol (sigma box)
mean_box = FancyBboxPatch((MEAN_X - 0.065, 0.44), 0.13, 0.22,
                          boxstyle="round,pad=0.012",
                          facecolor=CYAN + "20", edgecolor=CYAN,
                          linewidth=1.4, zorder=4)
ax2.add_patch(mean_box)
ax2.text(MEAN_X, 0.57, "MEAN",   color=CYAN, fontsize=8.5,
         ha='center', va='center', fontweight='bold',
         transform=ax2.transAxes, zorder=5)
ax2.text(MEAN_X, 0.49, "pool",   color=WHITE, fontsize=7.5,
         ha='center', va='center', transform=ax2.transAxes,
         zorder=5, alpha=0.75)

# Arrow from mean box → aggregated bar
ax2.annotate("", xy=(AGG_X, AGG_Y + BAR_MAXH * 0.7),
             xytext=(MEAN_X + 0.065, 0.55),
             xycoords='axes fraction', textcoords='axes fraction',
             arrowprops=dict(arrowstyle="-|>", color=CYAN, lw=1.2, alpha=0.7,
                             mutation_scale=9))

# Aggregated bar (larger, CYAN)
draw_fbar(ax2, AGG_X, AGG_Y, _bar_vals['agg'],
          CYAN, alpha=1.0, scale=1.1)
agg_total_w = N_BARS * (BAR_W + BAR_GAP) - BAR_GAP
ax2.text(AGG_X + agg_total_w / 2, AGG_Y + BAR_MAXH * 1.1 + 0.04,
         "h̃_N(v)",
         color=CYAN, fontsize=8.5, ha='center', va='bottom',
         transform=ax2.transAxes, fontstyle='italic', alpha=0.90)

# Formula at bottom
ax2.text(0.50, 0.06,
         "h̃_N(v) = MEAN ( { hᵤ : u ∈ N(v) } )",
         color=WHITE, fontsize=8.2, ha='center', va='bottom',
         transform=ax2.transAxes, alpha=0.75, family='monospace')

# ── STAGE 3 — Update ──────────────────────────────────────────────────────────
ax3.set_title("Node Update", color=CYAN, fontsize=10.5,
              fontweight='bold', pad=5)

# Show: [h_v bar] + [h̃_N bar] → concat bracket → W box → h_v' bar
HV_X  = 0.06   # h_v bar start
HV_Y  = 0.68   # h_v bar bottom

AGN_X = HV_X   # h̃ bar start (same x, lower y)
AGN_Y = 0.42   # h̃ bar bottom

CONCAT_X = 0.46   # bracket / concat visual centre
WB_X     = 0.62   # W box left
NEWBAR_X = 0.82   # h_v' bar start

fbar_w = N_BARS * (BAR_W + BAR_GAP) - BAR_GAP

# h_v bar (white/cyan)
draw_fbar(ax3, HV_X, HV_Y, _bar_vals['target'], WHITE, alpha=0.90, scale=1.0)
ax3.text(HV_X + fbar_w / 2, HV_Y + BAR_MAXH + 0.04, "hᵥ",
         color=WHITE, fontsize=9, ha='center', va='bottom',
         transform=ax3.transAxes, fontstyle='italic')

# h̃_N bar (cyan)
draw_fbar(ax3, AGN_X, AGN_Y, _bar_vals['agg'], CYAN, alpha=0.90, scale=1.0)
ax3.text(HV_X + fbar_w / 2, AGN_Y - 0.07, "h̃_N(v)",
         color=CYAN, fontsize=8, ha='center', va='top',
         transform=ax3.transAxes, fontstyle='italic', alpha=0.88)

# Bracket showing concatenation
BRACE_X = HV_X + fbar_w + 0.03
y_top = HV_Y + BAR_MAXH
y_bot = AGN_Y
y_mid = (y_top + y_bot) / 2
for ys, ye in [(y_top, y_mid), (y_bot, y_mid)]:
    ax3.plot([BRACE_X, BRACE_X + 0.04, BRACE_X + 0.04],
             [ys, ys, ye],
             color=WHITE, lw=1.2, alpha=0.55,
             transform=ax3.transAxes, zorder=3)
ax3.plot([BRACE_X + 0.04, BRACE_X + 0.06],
         [y_mid, y_mid],
         color=WHITE, lw=1.2, alpha=0.55,
         transform=ax3.transAxes, zorder=3)
ax3.text(BRACE_X + 0.05, y_mid + 0.04, "║",
         color=WHITE, fontsize=9, ha='left', va='center',
         transform=ax3.transAxes, alpha=0.65)

# Arrow from bracket → W box
ax3.annotate("", xy=(WB_X, y_mid),
             xytext=(BRACE_X + 0.07, y_mid),
             xycoords='axes fraction', textcoords='axes fraction',
             arrowprops=dict(arrowstyle="-|>", color=WHITE, lw=1.0,
                             alpha=0.5, mutation_scale=8))

# W·[...]+b box
wb_box = FancyBboxPatch((WB_X, y_mid - 0.14), 0.17, 0.28,
                        boxstyle="round,pad=0.012",
                        facecolor=ORANGE + "22", edgecolor=ORANGE,
                        linewidth=1.5, zorder=4)
ax3.add_patch(wb_box)
ax3.text(WB_X + 0.085, y_mid + 0.04, "W · [ · ] + b",
         color=ORANGE, fontsize=8.0, ha='center', va='center',
         fontweight='bold', transform=ax3.transAxes, zorder=5)
ax3.text(WB_X + 0.085, y_mid - 0.06, "σ ( · )",
         color=WHITE, fontsize=7.5, ha='center', va='center',
         transform=ax3.transAxes, zorder=5, alpha=0.75)

# Arrow → new bar
ax3.annotate("", xy=(NEWBAR_X, y_mid),
             xytext=(WB_X + 0.17, y_mid),
             xycoords='axes fraction', textcoords='axes fraction',
             arrowprops=dict(arrowstyle="-|>", color=ORANGE, lw=1.2,
                             alpha=0.7, mutation_scale=9))

# h_v' bar (updated, CYAN + slightly brighter)
draw_fbar(ax3, NEWBAR_X, AGN_Y, _bar_vals['updated'],
          CYAN, alpha=1.0, scale=1.2)
ax3.text(NEWBAR_X + fbar_w / 2, AGN_Y + BAR_MAXH * 1.2 + 0.06,
         "hᵥ′",
         color=CYAN, fontsize=9.5, ha='center', va='bottom',
         fontweight='bold', transform=ax3.transAxes)
ax3.text(NEWBAR_X + fbar_w / 2, AGN_Y + BAR_MAXH * 1.2 + 0.14,
         "256-dim",
         color=CYAN, fontsize=7.5, ha='center', va='bottom',
         transform=ax3.transAxes, alpha=0.70)

# Core equation at bottom
ax3.text(0.50, 0.10,
         "hᵥ′ = σ( W · [ hᵥ ║ h̃_N(v) ] + b )",
         color=WHITE, fontsize=9.0, ha='center', va='bottom',
         transform=ax3.transAxes, alpha=0.85, family='monospace')
ax3.text(0.50, 0.04,
         "repeated × 5 layers",
         color=CYAN, fontsize=8, ha='center', va='bottom',
         transform=ax3.transAxes, alpha=0.70)

# ── ARROWS between stages ─────────────────────────────────────────────────────
for ax_arr in (axA, axB, axC):
    ax_arr.text(0.5, 0.52, "→", color=CYAN, fontsize=24,
                ha='center', va='center', fontweight='bold',
                transform=ax_arr.transAxes)

# ── STAGE 4 — Architecture Tower ─────────────────────────────────────────────
ax4.set_title("Architecture", color=CYAN, fontsize=10.5,
              fontweight='bold', pad=5)

layers = [
    ("Node Features",   "ℝ¹⁵",       WHITE,  0.880, True),
    ("SAGEConv  L1",    "15 → 256",   CYAN,   0.770, False),
    ("SAGEConv  L2",    "256",        CYAN,   0.665, False),
    ("SAGEConv  L3",    "256",        CYAN,   0.560, False),
    ("SAGEConv  L4",    "256",        CYAN,   0.455, False),
    ("SAGEConv  L5",    "→ 64",       CYAN,   0.350, False),
    ("Classifier",      "64 → 2",     ORANGE, 0.215, False),
]

LW, LH = 0.90, 0.090

for lname, ldim, lcolor, ly, is_input in layers:
    ls  = ':' if is_input else '-'
    lw_ = 0.6 if is_input else 1.5
    fa  = "12" if is_input else "30" if lcolor == ORANGE else "28"
    r = FancyBboxPatch((0.5 - LW/2, ly - LH/2), LW, LH,
                       boxstyle="round,pad=0.010",
                       facecolor=lcolor + fa, edgecolor=lcolor,
                       linewidth=lw_, linestyle=ls, zorder=3)
    ax4.add_patch(r)
    ax4.text(0.5 - LW/2 + 0.07, ly + 0.005, lname,
             color=lcolor, fontsize=8.0 if not is_input else 7.5,
             fontweight='bold' if not is_input else 'normal',
             va='center', zorder=4,
             alpha=0.55 if is_input else 1.0)
    ax4.text(0.5 + LW/2 - 0.04, ly, ldim,
             color=WHITE, fontsize=7.5, va='center', ha='right',
             alpha=0.50 if is_input else 0.85, zorder=4)

# Connecting arrows
for i in range(len(layers) - 1):
    yt = layers[i][3] - LH/2 - 0.004
    yb = layers[i+1][3] + LH/2 + 0.004
    ax4.annotate("", xy=(0.5, yb), xytext=(0.5, yt),
                 xycoords='axes fraction', textcoords='axes fraction',
                 arrowprops=dict(arrowstyle="-|>", color=DIM, lw=0.9,
                                 alpha=0.65, mutation_scale=7))

# Output annotation
ax4.text(0.50, 0.12,
         "439,041 parameters",
         color=WHITE, fontsize=8, ha='center', va='center',
         transform=ax4.transAxes, alpha=0.72)

# Two output class dots
for xd, cc, lbl in [(0.34, CYAN, "healthy"), (0.66, ORANGE, "tumour")]:
    ax4.plot(xd, 0.055, 'o', color=cc, markersize=9,
             markeredgecolor=WHITE, markeredgewidth=0.5,
             transform=ax4.transAxes, zorder=5)
    ax4.text(xd, 0.015, lbl, color=cc, fontsize=7, ha='center', va='bottom',
             transform=ax4.transAxes, alpha=0.80)

# ── Panel dividers ─────────────────────────────────────────────────────────────
for xf in (0.168, 0.330, 0.492, 0.654, 0.816):
    pass  # dividers handled by wspace=0

# ── Global caption ──────────────────────────────────────────────────────────
fig.text(0.50, 0.025,
         "Inductive learning — generalises to unseen graphs  ·  "
         "Mean-pool aggregation  ·  ReLU activation + Batch Normalisation per layer",
         color=WHITE, fontsize=8.5, ha='center', alpha=0.72)

fig.text(0.50, 0.91,
         "GraphSAGE — Graph Sample and Aggregate",
         color=CYAN, fontsize=13.5, ha='center', fontweight='bold')

out = ("/Users/sakibkhan/Desktop/Capstone/CapstoneRepo/"
       "paperWriting/presentationDetails/graphsage_architecture.png")
plt.savefig(out, dpi=180, bbox_inches='tight', facecolor=BG, edgecolor='none')
plt.close()
print(f"Saved → {out}")
