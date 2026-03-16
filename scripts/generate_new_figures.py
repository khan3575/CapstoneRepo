"""
Generate supplementary figures from stored real data.
All data loaded from actual results files — no fabrication, no approximation.

Outputs (DPI=300, saved to research_results/figures/):
  fig_G_dice_distribution.png   — BraTS 2021 held-out per-graph Dice histogram
                                   (21,543 slices from 251 sealed patients, real data)

NOTE on BraTS 2023 per-patient histogram (fig_H):
  Only aggregate stats are stored in brats2023_evaluation/results.json.
  To generate a per-patient histogram, re-run evaluate_brats2023.py with
  per-patient tracking enabled (save list of patient-mean Dice scores to JSON).
  That is a re-inference task requiring the BraTS 2023 graph files.
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import json
import os

ROOT     = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR  = os.path.join(ROOT, "research_results", "figures")
os.makedirs(OUT_DIR, exist_ok=True)

DPI   = 300
STYLE = "seaborn-v0_8-whitegrid"
try:
    plt.style.use(STYLE)
except OSError:
    plt.style.use("seaborn-whitegrid")

GNN_BLUE    = "#2196F3"
UNET_RED    = "#F44336"
DARK        = "#212121"


# ══════════════════════════════════════════════════════════════════════════════
# Figure G — BraTS 2021 Held-Out Per-Graph (Slice-Level) Dice Distribution
# ══════════════════════════════════════════════════════════════════════════════
def fig_G_dice_distribution():
    # ── REAL DATA — loaded from stored inference results ────────────────────
    results_path = os.path.join(ROOT, "research_results", "ensemble_v2",
                                "ensemble_results.json")
    with open(results_path) as f:
        data = json.load(f)
    per_graph_dice = np.array(data["per_graph_dice"])  # 21,543 slice-level scores
    # ────────────────────────────────────────────────────────────────────────

    n_total      = len(per_graph_dice)
    n_patients   = 251        # sealed held-out set
    mean_dice    = per_graph_dice.mean()
    std_dice     = per_graph_dice.std()
    ensemble_dice = data["test_metrics"]["dice"]   # 0.9141 patient-level ensemble

    # Compute bin counts for annotation
    complete_miss = (per_graph_dice < 0.10).sum()
    partial_fail  = ((per_graph_dice >= 0.10) & (per_graph_dice < 0.60)).sum()
    high_quality  = (per_graph_dice >= 0.90).sum()
    pct_miss      = 100.0 * complete_miss / n_total
    pct_high      = 100.0 * high_quality / n_total

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5),
                             gridspec_kw={"width_ratios": [3, 1.4]})

    # ── Left: histogram ──────────────────────────────────────────────────────
    ax = axes[0]
    bins = np.linspace(0, 1, 51)   # 50 bins of width 0.02
    counts, edges, patches = ax.hist(per_graph_dice, bins=bins,
                                     color=GNN_BLUE, alpha=0.80,
                                     edgecolor="white", linewidth=0.4, zorder=3)

    # Colour complete-miss bins red
    for patch, left in zip(patches, edges[:-1]):
        if left < 0.10:
            patch.set_facecolor(UNET_RED)
            patch.set_alpha(0.85)

    # Mean & median lines
    ax.axvline(mean_dice, color=DARK, linewidth=2.0, linestyle="--", zorder=5,
               label=f"Mean = {mean_dice:.3f}  (±{std_dice:.3f} SD)")
    ax.axvline(np.median(per_graph_dice), color="#FF9800", linewidth=1.8,
               linestyle=":", zorder=5,
               label=f"Median = {np.median(per_graph_dice):.3f}")
    ax.axvline(ensemble_dice, color="#43A047", linewidth=1.8, linestyle="-.",
               zorder=5, label=f"Ensemble (patient-level) = {ensemble_dice:.4f}")

    # Annotation: complete-miss tail
    ax.annotate(
        f"Complete-miss\nslices: {complete_miss:,}\n({pct_miss:.1f}% of total)",
        xy=(0.05, counts[:3].max()), xytext=(0.18, counts[:3].max() * 1.2),
        fontsize=8.5, color=UNET_RED, ha="center",
        arrowprops=dict(arrowstyle="-|>", color=UNET_RED, lw=1.2)
    )

    # Annotation: high-quality peak
    peak_bin_left = edges[np.argmax(counts)]
    ax.annotate(
        f"High-quality slices\n(Dice ≥ 0.90): {high_quality:,}\n({pct_high:.1f}%)",
        xy=(peak_bin_left + 0.01, counts.max()),
        xytext=(0.72, counts.max() * 0.88),
        fontsize=8.5, color="#1565C0", ha="center",
        arrowprops=dict(arrowstyle="-|>", color="#1565C0", lw=1.2)
    )

    ax.set_xlabel("Dice Coefficient (per graph / 2D slice)", fontsize=12)
    ax.set_ylabel("Number of Graphs (slices)", fontsize=12)
    ax.set_title(
        f"BraTS 2021 Held-Out — Per-Graph Dice Distribution\n"
        f"({n_total:,} graphs from {n_patients} sealed patients; 5-fold soft-voting ensemble)",
        fontsize=10.5, pad=8
    )
    ax.legend(fontsize=9, framealpha=0.9, loc="upper left")
    ax.tick_params(axis="both", labelsize=10)
    ax.set_xlim(-0.02, 1.02)

    # ── Right: summary breakdown bar ─────────────────────────────────────────
    ax2 = axes[1]
    categories  = ["Complete\nmiss\n(Dice < 0.10)", "Partial\nfail\n(0.10–0.59)",
                   "Acceptable\n(0.60–0.89)", "High\nquality\n(≥ 0.90)"]
    acceptable  = ((per_graph_dice >= 0.60) & (per_graph_dice < 0.90)).sum()
    cat_counts  = [complete_miss, partial_fail, acceptable, high_quality]
    cat_pcts    = [100. * c / n_total for c in cat_counts]
    cat_colors  = [UNET_RED, "#FF7043", "#FFA726", GNN_BLUE]

    bars = ax2.bar(range(4), cat_pcts, color=cat_colors, alpha=0.85,
                   edgecolor="white", linewidth=0.8, width=0.6)
    for bar, cnt, pct in zip(bars, cat_counts, cat_pcts):
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.4,
                 f"{pct:.1f}%\n({cnt:,})",
                 ha="center", va="bottom", fontsize=8.5, fontweight="bold")

    ax2.set_xticks(range(4))
    ax2.set_xticklabels(categories, fontsize=8.5)
    ax2.set_ylim(0, max(cat_pcts) * 1.3)
    ax2.set_ylabel("Percentage of Graphs (%)", fontsize=10)
    ax2.set_title("Quality Breakdown", fontsize=10.5, pad=8)
    ax2.tick_params(axis="y", labelsize=9)

    fig.tight_layout(pad=2.5)
    path = os.path.join(OUT_DIR, "fig_G_dice_distribution.png")
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")
    print(f"  Data: {n_total:,} real slice-level Dice scores from {n_patients} patients")
    print(f"  Complete-miss slices: {complete_miss:,} ({pct_miss:.1f}%)")
    print(f"  High-quality slices:  {high_quality:,} ({pct_high:.1f}%)")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print(f"\nGenerating new supplementary figures → {OUT_DIR}\n")
    fig_G_dice_distribution()
    print("\nDone. All figures from 100% real stored data.\n")
    print("NOTE: BraTS 2023 per-patient histogram (fig_H) requires re-running")
    print("  evaluate_brats2023.py with per-patient dice tracking. See script header.")
