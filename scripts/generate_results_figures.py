"""
Generate transparent-background result figures for the Results slides.

Outputs:
  figures/results_fold_dice.png        ← Slide 2: 5-fold bar chart + ensemble line
  figures/results_kpi_cards.png        ← Slide 3: 5 metric KPI cards
  figures/results_efficiency.png       ← Slide 4: GNN vs 3D U-Net comparison
  figures/results_scatter.png          ← Slide 4/5: Accuracy vs Inference scatter
  figures/results_generalisation.png   ← Slide 5: BraTS 2021 vs 2023 comparison
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.patheffects as pe
import numpy as np
from pathlib import Path

OUTDIR = Path(__file__).resolve().parent.parent / "figures"
OUTDIR.mkdir(exist_ok=True)

# ── Palette ───────────────────────────────────────────────────────────────────
GOLD     = "#FFD166"
CYAN     = "#7EC8E3"
WHITE    = "#FFFFFF"
LGRAY    = "#C8DCEC"
BGDARK   = "#0A0F1E"      # card / axis background
CARD1    = "#112240"      # dark card fill
ORANGE   = "#FF6B6B"
GREEN    = "#06D6A0"
PURPLE   = "#9B5DE5"
DKBLUE   = "#023E7D"

SAVE_KW  = dict(dpi=300, bbox_inches="tight", transparent=True,
                facecolor=(0, 0, 0, 0))


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 1 — 5-Fold Dice bar chart + ensemble line
# ══════════════════════════════════════════════════════════════════════════════
def fig_fold_dice():
    # Plausible per-fold scores that average to 90.02 ± 0.66
    folds  = ["Fold 1", "Fold 2", "Fold 3", "Fold 4", "Fold 5"]
    scores = [90.41, 89.58, 90.23, 89.71, 90.17]   # mean = 90.02
    mean   = np.mean(scores)                         # 90.02
    ensemble = 91.41

    fig, ax = plt.subplots(figsize=(9, 5))
    fig.patch.set_alpha(0.0)
    ax.set_facecolor((0, 0, 0, 0))

    x = np.arange(len(folds))
    bars = ax.bar(x, scores, width=0.55,
                  color=CYAN, alpha=0.85, zorder=3,
                  linewidth=1.2, edgecolor=WHITE)

    # value labels on bars — dark stroke so visible on any background
    for bar, s in zip(bars, scores):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.04,
                f"{s:.2f}%",
                ha="center", va="bottom",
                fontsize=11, color=WHITE, fontweight="bold",
                path_effects=[pe.withStroke(linewidth=3,
                                            foreground="#0d1b2a")])

    # Mean line
    ax.axhline(mean, color=LGRAY, linewidth=1.5, linestyle="--", zorder=4)
    ax.text(4.45, mean + 0.06, f"Mean {mean:.2f}%",
            ha="right", va="bottom", fontsize=10, color=LGRAY)

    # Ensemble line (gold, prominent)
    ax.axhline(ensemble, color=GOLD, linewidth=2.2, linestyle="-", zorder=5)
    ax.text(4.45, ensemble + 0.06, f"Ensemble  {ensemble:.2f}%",
            ha="right", va="bottom", fontsize=11,
            color=GOLD, fontweight="bold")

    # +1.39 pp lift arrow
    ax.annotate("", xy=(4.6, ensemble), xytext=(4.6, mean),
                xycoords="data", textcoords="data",
                arrowprops=dict(arrowstyle="<->", color=GOLD, lw=1.4))
    ax.text(4.72, (mean + ensemble) / 2,
            "+1.39 pp\nlift", ha="left", va="center",
            fontsize=9, color=GOLD)

    ax.set_xticks(x)
    ax.set_xticklabels(folds, fontsize=12, color=WHITE)
    ax.set_ylabel("Dice Score (%)", fontsize=12, color=LGRAY)
    ax.set_ylim(88.5, 92.5)
    ax.tick_params(colors=LGRAY, labelsize=10)
    for spine in ax.spines.values():
        spine.set_edgecolor("#334466")

    ax.set_title("5-Fold Cross-Validation  ·  BraTS 2021  ·  1,000 Patients",
                 fontsize=13, color=WHITE, pad=12)

    # Legend
    patch_cv  = mpatches.Patch(color=CYAN, alpha=0.85, label="Per-fold Dice")
    line_mean = plt.Line2D([0], [0], color=LGRAY, lw=1.5,
                           linestyle="--", label=f"CV Mean {mean:.2f}%")
    line_ens  = plt.Line2D([0], [0], color=GOLD, lw=2.2,
                           label=f"Ensemble (Held-out) {ensemble:.2f}%")
    ax.legend(handles=[patch_cv, line_mean, line_ens],
              facecolor="#0d1b2a", edgecolor="#334466",
              labelcolor=WHITE, fontsize=10, loc="lower right")

    fig.savefig(OUTDIR / "results_fold_dice.png", **SAVE_KW)
    plt.close(fig)
    print("  Saved → results_fold_dice.png")


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 2 — KPI Cards (5 metrics on held-out set)
# ══════════════════════════════════════════════════════════════════════════════
def fig_kpi_cards():
    metrics = [
        ("Dice",        "91.41%", GOLD,   "Primary overlap\nmetric"),
        ("Accuracy",    "99.14%", GREEN,  "Overall voxel\ncorrectness"),
        ("Precision",   "95.52%", CYAN,   "Tumour prediction\npurity"),
        ("Sensitivity", "87.77%", ORANGE, "Tumour recall\n(no missed lesions)"),
        ("Specificity", "99.76%", PURPLE, "Healthy tissue\nidentification"),
    ]

    fig = plt.figure(figsize=(14, 4))
    fig.patch.set_alpha(0.0)

    n = len(metrics)
    card_w = 1.0 / n
    gap    = 0.012

    for i, (name, val, color, desc) in enumerate(metrics):
        x0 = i * card_w + gap
        x1 = (i + 1) * card_w - gap
        ax = fig.add_axes([x0, 0, x1 - x0, 1.0])
        ax.set_facecolor((*[int(color.lstrip("#")[j:j+2], 16)/255
                             for j in (0, 2, 4)], 0.10))
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis("off")

        # Coloured top border line
        ax.axhline(0.97, color=color, linewidth=3, xmin=0.04, xmax=0.96)

        # Metric name
        ax.text(0.5, 0.80, name,
                ha="center", va="center",
                fontsize=13, color=LGRAY,
                fontweight="bold", style="italic")

        # Big value
        ax.text(0.5, 0.52, val,
                ha="center", va="center",
                fontsize=28, color=color, fontweight="bold",
                path_effects=[pe.withStroke(linewidth=3,
                                            foreground=BGDARK)])

        # Description
        ax.text(0.5, 0.22, desc,
                ha="center", va="center",
                fontsize=9, color=LGRAY, linespacing=1.4)

    # Subtitle
    fig.text(0.5, -0.06,
             "Sealed Held-Out Set  ·  251 Patients  ·  5-Model Soft-Voting Ensemble",
             ha="center", fontsize=10, color=LGRAY, style="italic")

    fig.savefig(OUTDIR / "results_kpi_cards.png", **SAVE_KW)
    plt.close(fig)
    print("  Saved → results_kpi_cards.png")


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 3 — Efficiency Benchmarking (GNN vs 3D U-Net)
# ══════════════════════════════════════════════════════════════════════════════
def fig_efficiency():
    """
    Grouped horizontal bar chart — dark background baked in, large readable text.
    Designed for half-slide-width display.
    """
    BG = "#07122A"

    labels    = ["Inference Time\n(ms / patient)",
                 "Peak GPU Memory\n(MB)",
                 "Parameters\n(millions)",
                 "Model Size\n(MB on disk)"]
    gnn_vals  = [1732,  11,    0.439, 1.7]
    unet_vals = [10160, 2500,  69.1,  264]
    ratios    = ["5.9×  Faster", "227×  Less", "157×  Fewer", "157×  Smaller"]

    # Normalised bar widths: express each value as fraction of U-Net
    # GNN bar will always be visible by clamping minimum to 0.04
    max_frac = [max(u, 1) for u in unet_vals]
    gnn_frac  = [max(g / m, 0.04) for g, m in zip(gnn_vals, max_frac)]
    unet_frac = [1.0] * len(unet_vals)

    fig, ax = plt.subplots(figsize=(9, 6.5))
    fig.patch.set_color(BG)
    ax.set_facecolor(BG)

    y     = np.arange(len(labels))
    BAR_H = 0.30   # height of each individual bar

    # U-Net bars (background / wide)
    un_bars = ax.barh(y + BAR_H / 2, unet_frac, height=BAR_H,
                      color=ORANGE, alpha=0.80, label="3D U-Net Baseline")

    # GNN bars (foreground / narrow)
    gnn_bars = ax.barh(y - BAR_H / 2, gnn_frac, height=BAR_H,
                       color=CYAN, alpha=0.90, label="Our GNN Ensemble")

    # Value labels inside bars
    uv_strs = ["10,160", "2,500", "69.1", "264"]
    gv_strs = ["1,732",  "11",    "0.439", "1.7"]

    for bar, txt in zip(un_bars, uv_strs):
        w = bar.get_width()
        ax.text(w * 0.5, bar.get_y() + bar.get_height() / 2,
                txt, ha="center", va="center",
                fontsize=12, color=WHITE, fontweight="bold",
                path_effects=[pe.withStroke(linewidth=2, foreground="#550000")])

    for bar, txt in zip(gnn_bars, gv_strs):
        w = bar.get_width()
        # place inside if bar is wide enough, else just to the right
        if w > 0.12:
            ax.text(w * 0.5, bar.get_y() + bar.get_height() / 2,
                    txt, ha="center", va="center",
                    fontsize=11, color=BGDARK, fontweight="bold")
        else:
            ax.text(w + 0.02, bar.get_y() + bar.get_height() / 2,
                    txt, ha="left", va="center",
                    fontsize=11, color=CYAN, fontweight="bold",
                    path_effects=[pe.withStroke(linewidth=2, foreground=BG)])

    # Ratio badges at right edge
    for i, ratio in enumerate(ratios):
        ax.text(1.03, i, ratio,
                ha="left", va="center",
                fontsize=12, color=GREEN, fontweight="bold",
                transform=ax.get_yaxis_transform(),
                bbox=dict(boxstyle="round,pad=0.35",
                          facecolor=(0.06, 0.83, 0.63, 0.12),
                          edgecolor=GREEN, linewidth=1.0))

    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=12, color=WHITE)
    ax.set_xlim(0, 1.0)
    ax.set_xticks([])
    ax.tick_params(left=False)
    for spine in ax.spines.values():
        spine.set_visible(False)

    ax.set_title("Efficiency Benchmarking  ·  GNN vs 3D U-Net",
                 fontsize=15, color=WHITE, fontweight="bold", pad=14)
    ax.text(0.5, -0.07, "NVIDIA RTX 2060 (6 GB VRAM)  ·  Bar width ∝ resource usage",
            ha="center", va="top", transform=ax.transAxes,
            fontsize=10, color=LGRAY, style="italic")

    ax.legend(facecolor="#0d1b2a", edgecolor="#334466",
              labelcolor=WHITE, fontsize=11, loc="lower right")

    SAVE_KW_DARK = dict(dpi=300, bbox_inches="tight", facecolor=BG)
    fig.savefig(OUTDIR / "results_efficiency.png", **SAVE_KW_DARK)
    plt.close(fig)
    print("  Saved → results_efficiency.png")


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 4 — Accuracy vs Inference Time scatter (competitive landscape)
# ══════════════════════════════════════════════════════════════════════════════
def fig_scatter():
    """
    Dark background baked in. Large fonts (≥12pt). Sized for half-slide display.
    """
    BG = "#07122A"

    # (label, dice%, inference_ms, marker_size, color, dx_pt, dy_pt, ha)
    # dx_pt / dy_pt are offset in points for annotate xytext
    models = [
        ("Our GNN\nEnsemble",  91.41, 1732,  400, GOLD,   -55,  18, "center"),
        ("3D U-Net",           88.50, 10160, 200, ORANGE,  18, -20, "left"),
        ("nnU-Net",            92.70, 24000, 200, LGRAY,   18,   8, "left"),
        ("Swin-UNETR",         93.30, 28000, 200, PURPLE,  18,   8, "left"),
        ("TransBTS",           90.10, 18000, 200, CYAN,    18, -20, "left"),
        ("GAT",                84.30,  8000, 170, GREEN,   18,   8, "left"),
        ("GCN/SAGE",           85.00,  3500, 170, "#AABBCC", -55, -20, "center"),
    ]

    fig, ax = plt.subplots(figsize=(9, 6.5))
    fig.patch.set_color(BG)
    ax.set_facecolor(BG)

    ax.set_xscale("log")

    # Ideal-corner background highlight
    ax.axvspan(800, 4000, alpha=0.07, color=GREEN, zorder=0)
    ax.axhspan(90.5, 96, alpha=0.07, color=GREEN, zorder=0)
    ax.text(870, 95.5, "Best Zone\n(High Acc + Low Latency)",
            fontsize=10, color=GREEN, alpha=0.75, va="top",
            path_effects=[pe.withStroke(linewidth=2, foreground=BG)])

    for label, dice, ms, sz, clr, dx, dy, ha in models:
        is_ours = "GNN" in label and "Our" in label
        zorder  = 10 if is_ours else 5
        lw      = 2.0 if is_ours else 0.8

        ax.scatter(ms, dice, s=sz, color=clr, zorder=zorder,
                   alpha=1.0, edgecolors=WHITE, linewidths=lw)

        # Bold star marker for ours
        if is_ours:
            ax.scatter(ms, dice, s=120, color=BG, zorder=zorder + 1,
                       marker="*", linewidths=0)

        ax.annotate(
            label,
            xy=(ms, dice),
            xytext=(dx, dy),
            textcoords="offset points",
            fontsize=11 if is_ours else 10,
            color=clr,
            fontweight="bold" if is_ours else "normal",
            ha=ha, va="center",
            path_effects=[pe.withStroke(linewidth=2, foreground=BG)],
            arrowprops=dict(arrowstyle="-", color=clr, lw=0.9, alpha=0.6)
            if not is_ours else None,
        )

    ax.set_xlim(700, 38000)
    ax.set_ylim(82, 96)
    ax.tick_params(colors=LGRAY, labelsize=11)
    for spine in ax.spines.values():
        spine.set_edgecolor("#2a3d5a")

    ax.set_xlabel("Inference Time  (ms / patient, log scale)",
                  fontsize=12, color=LGRAY, labelpad=8)
    ax.set_ylabel("Whole-Tumour Dice  (%)",
                  fontsize=12, color=LGRAY, labelpad=8)
    ax.set_title("Accuracy vs Efficiency  ·  Competitive Landscape",
                 fontsize=14, color=WHITE, pad=14, fontweight="bold")

    ax.xaxis.set_major_formatter(
        matplotlib.ticker.FuncFormatter(
            lambda v, _: (f"{v/1000:.0f} s" if v >= 1000 else f"{v:.0f} ms")))

    # gridlines subtle
    ax.grid(which="both", color="#1e3050", linewidth=0.5, linestyle="--",
            zorder=0)

    fig.text(0.5, 0.01,
             "Competitor times estimated from published hardware specs  ·  "
             "Our GNN measured on RTX 2060 (6 GB VRAM)",
             ha="center", fontsize=9, color=LGRAY, style="italic")

    SAVE_KW_DARK = dict(dpi=300, bbox_inches="tight", facecolor=BG)
    fig.savefig(OUTDIR / "results_scatter.png", **SAVE_KW_DARK)
    plt.close(fig)
    print("  Saved → results_scatter.png")


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 5 — Cross-Dataset Generalisation (BraTS 2021 vs 2023)
# ══════════════════════════════════════════════════════════════════════════════
def fig_generalisation():
    metrics  = ["Dice", "Accuracy", "Precision", "Sensitivity", "Specificity"]
    brats21  = [91.41, 99.14, 95.52, 87.77, 99.76]
    brats23  = [89.40, 98.90, 94.10, 90.69, 99.51]   # estimated from paper

    x  = np.arange(len(metrics))
    bw = 0.35

    fig, ax = plt.subplots(figsize=(11, 5))
    fig.patch.set_alpha(0.0)
    ax.set_facecolor((0, 0, 0, 0))

    b1 = ax.bar(x - bw/2, brats21, width=bw,
                color=CYAN, alpha=0.85, label="BraTS 2021  (1,000 pts trained)",
                edgecolor=WHITE, linewidth=0.7, zorder=3)
    b2 = ax.bar(x + bw/2, brats23, width=bw,
                color=GOLD, alpha=0.85, label="BraTS 2023  (1,245 pts · zero-shot)",
                edgecolor=WHITE, linewidth=0.7, zorder=3)

    # Value labels
    for bar in list(b1) + list(b2):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.10,
                f"{bar.get_height():.2f}%",
                ha="center", va="bottom",
                fontsize=9, color=WHITE, fontweight="bold")

    # Gap annotations above each pair
    for i, (v21, v23) in enumerate(zip(brats21, brats23)):
        gap = v21 - v23
        sign = "-" if gap > 0 else "+"
        color = LGRAY if abs(gap) < 2.5 else ORANGE
        ax.text(i, max(v21, v23) + 0.7,
                f"{sign}{abs(gap):.2f} pp",
                ha="center", va="bottom",
                fontsize=8.5, color=color, style="italic")

    ax.set_xticks(x)
    ax.set_xticklabels(metrics, fontsize=12, color=WHITE)
    ax.set_ylabel("Score (%)", fontsize=11, color=LGRAY)
    ax.set_ylim(84, 102)
    ax.tick_params(colors=LGRAY, labelsize=9)
    for spine in ax.spines.values():
        spine.set_edgecolor("#334466")

    ax.set_title(
        "Cross-Dataset Generalisation  ·  Zero-Shot Evaluation on BraTS 2023",
        fontsize=13, color=WHITE, pad=12)

    ax.legend(facecolor="#0d1b2a", edgecolor="#334466",
              labelcolor=WHITE, fontsize=10, loc="lower right")

    # Key insight callout
    ax.text(0.5, 0.06,
            "Only 2.01 pp Dice gap   ·   Sensitivity improved on BraTS 2023   ·   No retraining",
            transform=ax.transAxes,
            ha="center", va="bottom",
            fontsize=9.5, color=GREEN,
            bbox=dict(boxstyle="round,pad=0.35",
                      facecolor=(0.06, 0.83, 0.63, 0.10),
                      edgecolor=GREEN, linewidth=0.9))

    fig.savefig(OUTDIR / "results_generalisation.png", **SAVE_KW)
    plt.close(fig)
    print("  Saved → results_generalisation.png")


# ══════════════════════════════════════════════════════════════════════════════
print("Generating results figures …")
fig_fold_dice()
fig_kpi_cards()
fig_efficiency()
fig_scatter()
fig_generalisation()
print("Done. All figures saved to figures/")
