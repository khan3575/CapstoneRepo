#!/usr/bin/env python3
"""
Phase 6 — Figure Generation
=============================
Generates all paper-quality figures from training results.

Figures produced:
  1. training_curves_fold_X.png   — Loss & Dice curves per fold
  2. cv_dice_summary.png          — Per-fold bar chart + mean ± std
  3. roc_curves.png               — ROC curve per fold + ensemble
  4. metrics_radar.png            — Radar chart: dice/acc/sens/spec/prec
  5. ensemble_summary.png         — Ensemble vs individual fold comparison

Usage:
    # Demo: generate figures from whatever folds are currently done
    python scripts/generate_figures.py --demo

    # Full: generate all figures (run after all 5 folds + ensemble)
    python scripts/generate_figures.py

    # Specify output dir
    python scripts/generate_figures.py --output research_results/figures
"""

import sys
import json
import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')  # non-interactive backend — works without display
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))


# ── Style ─────────────────────────────────────────────────────────────────────
COLORS = ['#2196F3', '#4CAF50', '#FF9800', '#E91E63', '#9C27B0']  # fold 0-4
GRID_ALPHA = 0.3
DPI = 300
FIGSIZE_WIDE = (12, 5)
FIGSIZE_SQ   = (8, 6)


def set_paper_style():
    plt.rcParams.update({
        'font.family':      'DejaVu Sans',
        'font.size':        11,
        'axes.titlesize':   13,
        'axes.labelsize':   12,
        'xtick.labelsize':  10,
        'ytick.labelsize':  10,
        'legend.fontsize':  10,
        'figure.dpi':       DPI,
        'savefig.dpi':      DPI,
        'savefig.bbox':     'tight',
        'axes.spines.top':  False,
        'axes.spines.right':False,
    })


# ── Data loaders ──────────────────────────────────────────────────────────────

def load_fold_results(checkpoints_dir):
    """Load results.json for all available folds."""
    results = {}
    for fold_idx in range(5):
        results_path = Path(checkpoints_dir) / f'fold_{fold_idx}' / 'results.json'
        if results_path.exists():
            with open(results_path) as f:
                results[fold_idx] = json.load(f)
    return results


def load_ensemble_results(ensemble_dir):
    """Load ensemble_results.json if present."""
    path = Path(ensemble_dir) / 'ensemble_results.json'
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return None


# ── Figure 1: Training curves ─────────────────────────────────────────────────

def plot_training_curves(fold_results, output_dir):
    """Loss + Dice training curves for each completed fold."""
    saved = []
    for fold_idx, result in fold_results.items():
        history = result.get('history', [])
        if not history:
            continue

        epochs     = [h['epoch'] for h in history]
        train_loss = [h['train']['loss'] for h in history]
        val_loss   = [h['val']['loss']   for h in history]
        train_dice = [h['train']['dice'] for h in history]
        val_dice   = [h['val']['dice']   for h in history]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=FIGSIZE_WIDE)

        # Loss
        ax1.plot(epochs, train_loss, label='Train', color='#2196F3', linewidth=2)
        ax1.plot(epochs, val_loss,   label='Val',   color='#FF5722', linewidth=2, linestyle='--')
        best_epoch = result.get('best_val_dice', 0)
        best_ep_num = max(history, key=lambda h: h['val']['dice'])['epoch']
        ax1.axvline(best_ep_num, color='gray', linestyle=':', alpha=0.7, label=f'Best epoch ({best_ep_num})')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss (BCE)')
        ax1.set_title(f'Fold {fold_idx} — Loss')
        ax1.legend()
        ax1.grid(alpha=GRID_ALPHA)

        # Dice
        ax2.plot(epochs, train_dice, label='Train', color='#2196F3', linewidth=2)
        ax2.plot(epochs, val_dice,   label='Val',   color='#FF5722', linewidth=2, linestyle='--')
        ax2.axvline(best_ep_num, color='gray', linestyle=':', alpha=0.7)
        ax2.axhline(result['test_metrics']['dice'], color='green', linestyle='-.', alpha=0.8,
                    label=f"Test Dice={result['test_metrics']['dice']:.4f}")
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Dice Score')
        ax2.set_title(f'Fold {fold_idx} — Dice Score')
        ax2.set_ylim(0, 1.0)
        ax2.legend()
        ax2.grid(alpha=GRID_ALPHA)

        fig.suptitle(f'Training Curves — Fold {fold_idx}  '
                     f'(Test Dice: {result["test_metrics"]["dice"]:.4f})', fontsize=13)
        plt.tight_layout()

        out = output_dir / f'training_curves_fold_{fold_idx}.png'
        fig.savefig(out, dpi=DPI, bbox_inches='tight')
        plt.close(fig)
        saved.append(out)
        print(f"  Saved: {out.name}")

    return saved


# ── Figure 2: CV Dice summary bar chart ───────────────────────────────────────

def plot_cv_summary(fold_results, ensemble_result, output_dir):
    """Per-fold test Dice bar chart + mean ± std + ensemble line."""
    if not fold_results:
        print("  Skipping CV summary: no fold results")
        return

    fold_ids   = sorted(fold_results.keys())
    test_dices = [fold_results[i]['test_metrics']['dice'] for i in fold_ids]
    val_dices  = [fold_results[i]['best_val_dice']        for i in fold_ids]

    mean_dice = np.mean(test_dices)
    std_dice  = np.std(test_dices)

    fig, ax = plt.subplots(figsize=(9, 5))

    x = np.arange(len(fold_ids))
    bars = ax.bar(x, test_dices, color=[COLORS[i] for i in fold_ids],
                  alpha=0.85, width=0.5, edgecolor='white', linewidth=1.2)

    # Value labels on bars
    for bar, d in zip(bars, test_dices):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.003,
                f'{d:.4f}', ha='center', va='bottom', fontsize=10)

    # Mean ± std band
    ax.axhline(mean_dice, color='black', linewidth=2, linestyle='--',
               label=f'Mean = {mean_dice:.4f} ± {std_dice:.4f}')
    ax.axhspan(mean_dice - std_dice, mean_dice + std_dice, alpha=0.08, color='black')

    # Ensemble line
    if ensemble_result:
        ens_dice = ensemble_result['test_metrics']['dice']
        ax.axhline(ens_dice, color='#E91E63', linewidth=2, linestyle='-.',
                   label=f'Ensemble = {ens_dice:.4f}')

    ax.set_xticks(x)
    ax.set_xticklabels([f'Fold {i}' for i in fold_ids])
    ax.set_ylabel('Test Dice Score')
    ax.set_title('Cross-Validation Results — Test Dice per Fold')
    ax.set_ylim(max(0, min(test_dices) - 0.05), min(1.0, max(test_dices) + 0.05))
    ax.legend()
    ax.grid(axis='y', alpha=GRID_ALPHA)

    plt.tight_layout()
    out = output_dir / 'cv_dice_summary.png'
    fig.savefig(out, dpi=DPI, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {out.name}")
    return out


# ── Figure 3: ROC curves ──────────────────────────────────────────────────────

def plot_roc_curves(fold_results, output_dir):
    """ROC curves estimated from fold metrics (sensitivity/specificity points)."""
    if not fold_results:
        return

    fig, ax = plt.subplots(figsize=FIGSIZE_SQ)

    # Plot diagonal
    ax.plot([0, 1], [0, 1], 'k--', alpha=0.3, label='Random')

    # For each fold, plot the operating point (1-specificity, sensitivity)
    for fold_idx, result in sorted(fold_results.items()):
        m = result['test_metrics']
        fpr = 1 - m['specificity']
        tpr = m['sensitivity']
        ax.scatter(fpr, tpr, s=150, color=COLORS[fold_idx], zorder=5,
                   label=f"Fold {fold_idx} (Dice={m['dice']:.4f})")

    # Mean operating point
    all_fpr = [1 - fold_results[i]['test_metrics']['specificity'] for i in fold_results]
    all_tpr = [fold_results[i]['test_metrics']['sensitivity']     for i in fold_results]
    ax.scatter(np.mean(all_fpr), np.mean(all_tpr), s=250, color='black',
               marker='*', zorder=6, label=f'Mean ({np.mean(all_tpr):.4f} sens)')

    ax.set_xlabel('1 - Specificity (FPR)')
    ax.set_ylabel('Sensitivity (TPR)')
    ax.set_title('ROC Operating Points — Test Set')
    ax.set_xlim(-0.01, 0.15)
    ax.set_ylim(0.75, 1.0)
    ax.legend(loc='lower right', fontsize=9)
    ax.grid(alpha=GRID_ALPHA)

    plt.tight_layout()
    out = output_dir / 'roc_operating_points.png'
    fig.savefig(out, dpi=DPI, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {out.name}")
    return out


# ── Figure 4: Metrics radar chart ─────────────────────────────────────────────

def plot_metrics_radar(fold_results, ensemble_result, output_dir):
    """Radar chart comparing all metrics across folds and ensemble."""
    if not fold_results:
        return

    metrics_keys = ['dice', 'accuracy', 'sensitivity', 'specificity', 'precision']
    labels = ['Dice', 'Accuracy', 'Sensitivity', 'Specificity', 'Precision']
    N = len(labels)
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(7, 7), subplot_kw=dict(polar=True))

    for fold_idx, result in sorted(fold_results.items()):
        m = result['test_metrics']
        values = [m[k] for k in metrics_keys]
        values += values[:1]
        ax.plot(angles, values, linewidth=1.5, linestyle='solid',
                color=COLORS[fold_idx], alpha=0.7, label=f'Fold {fold_idx}')
        ax.fill(angles, values, color=COLORS[fold_idx], alpha=0.05)

    # Mean
    all_vals = {k: [fold_results[i]['test_metrics'][k] for i in fold_results]
                for k in metrics_keys}
    mean_vals = [np.mean(all_vals[k]) for k in metrics_keys]
    mean_vals += mean_vals[:1]
    ax.plot(angles, mean_vals, linewidth=2.5, linestyle='--', color='black',
            label=f'Mean (Dice={np.mean(all_vals["dice"]):.4f})')

    # Ensemble
    if ensemble_result:
        em = ensemble_result['test_metrics']
        ens_vals = [em[k] for k in metrics_keys]
        ens_vals += ens_vals[:1]
        ax.plot(angles, ens_vals, linewidth=2.5, linestyle='-', color='#E91E63',
                label=f'Ensemble (Dice={em["dice"]:.4f})')
        ax.fill(angles, ens_vals, color='#E91E63', alpha=0.07)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, fontsize=11)
    ax.set_ylim(0.75, 1.0)
    ax.set_yticks([0.80, 0.85, 0.90, 0.95, 1.00])
    ax.set_yticklabels(['0.80', '0.85', '0.90', '0.95', '1.00'], fontsize=8)
    ax.set_title('Performance Metrics — All Folds + Ensemble', pad=20)
    ax.legend(loc='upper right', bbox_to_anchor=(1.35, 1.1), fontsize=9)
    ax.grid(alpha=GRID_ALPHA)

    plt.tight_layout()
    out = output_dir / 'metrics_radar.png'
    fig.savefig(out, dpi=DPI, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {out.name}")
    return out


# ── Figure 5: Ensemble comparison ─────────────────────────────────────────────

def plot_ensemble_comparison(fold_results, ensemble_result, output_dir):
    """Side-by-side comparison: individual fold metrics vs ensemble."""
    if not fold_results or not ensemble_result:
        print("  Skipping ensemble comparison: need both fold results and ensemble")
        return

    metrics_keys = ['dice', 'accuracy', 'sensitivity', 'specificity', 'precision']
    labels = ['Dice', 'Accuracy', 'Sensitivity', 'Specificity', 'Precision']

    fold_ids = sorted(fold_results.keys())
    mean_metrics = {k: np.mean([fold_results[i]['test_metrics'][k] for i in fold_ids])
                    for k in metrics_keys}
    ens_metrics = ensemble_result['test_metrics']

    x = np.arange(len(labels))
    w = 0.35

    fig, ax = plt.subplots(figsize=(11, 5))
    bars1 = ax.bar(x - w/2, [mean_metrics[k] for k in metrics_keys],
                   w, label='Individual Folds (mean)', color='#2196F3', alpha=0.85)
    bars2 = ax.bar(x + w/2, [ens_metrics[k] for k in metrics_keys],
                   w, label='Ensemble (5-fold)', color='#E91E63', alpha=0.85)

    # Value labels
    for bar in bars1:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                f'{bar.get_height():.4f}', ha='center', va='bottom', fontsize=9)
    for bar in bars2:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                f'{bar.get_height():.4f}', ha='center', va='bottom', fontsize=9,
                color='#C2185B', fontweight='bold')

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel('Score')
    ax.set_title('Individual Folds vs Ensemble — Test Metrics')
    ax.set_ylim(0.75, 1.02)
    ax.legend()
    ax.grid(axis='y', alpha=GRID_ALPHA)

    plt.tight_layout()
    out = output_dir / 'ensemble_vs_individual.png'
    fig.savefig(out, dpi=DPI, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {out.name}")
    return out


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='Phase 6 — Generate paper figures')
    parser.add_argument('--checkpoints', type=str, default='checkpoints/binary_v2',
                        help='Checkpoints directory (default: checkpoints/binary_v2)')
    parser.add_argument('--ensemble', type=str, default='research_results/ensemble_v2',
                        help='Ensemble results directory')
    parser.add_argument('--output', type=str, default='research_results/figures',
                        help='Output directory for figures (default: research_results/figures)')
    parser.add_argument('--demo', action='store_true',
                        help='Demo mode: generates from whatever folds are currently done')
    args = parser.parse_args()

    project_root = Path(__file__).parent.parent
    checkpoints_dir = project_root / args.checkpoints
    ensemble_dir    = project_root / args.ensemble
    output_dir      = project_root / args.output
    output_dir.mkdir(parents=True, exist_ok=True)

    set_paper_style()

    print("=" * 60)
    print("PHASE 6 — FIGURE GENERATION")
    print("=" * 60)
    if args.demo:
        print("[DEMO MODE: using whatever folds are currently done]")
    print(f"Output: {output_dir}")
    print()

    # Load data
    fold_results    = load_fold_results(checkpoints_dir)
    ensemble_result = load_ensemble_results(ensemble_dir)

    if not fold_results:
        print("ERROR: No fold results found. Train at least one fold first.")
        print(f"  Expected: {checkpoints_dir}/fold_X/results.json")
        return

    print(f"Found {len(fold_results)}/5 fold results: folds {sorted(fold_results.keys())}")
    if ensemble_result:
        print(f"Ensemble result: Dice={ensemble_result['test_metrics']['dice']:.4f}")
    else:
        print("No ensemble result yet (run inference_ensemble.py after all folds)")
    print()

    # Print summary table
    print(f"{'Fold':>6}  {'Test Dice':>10}  {'Accuracy':>10}  {'Sensitivity':>12}  {'Specificity':>12}")
    print("-" * 60)
    dices = []
    for fold_idx in sorted(fold_results.keys()):
        m = fold_results[fold_idx]['test_metrics']
        dices.append(m['dice'])
        print(f"  {fold_idx:>4}  {m['dice']:>10.4f}  {m['accuracy']:>10.4f}  "
              f"{m['sensitivity']:>12.4f}  {m['specificity']:>12.4f}")
    print("-" * 60)
    print(f"  {'Mean':>4}  {np.mean(dices):>10.4f}  ±{np.std(dices):.4f}")
    if ensemble_result:
        em = ensemble_result['test_metrics']
        print(f"  {'Ens':>4}  {em['dice']:>10.4f}  {em['accuracy']:>10.4f}  "
              f"{em['sensitivity']:>12.4f}  {em['specificity']:>12.4f}")
    print()

    # Generate figures
    print("Generating figures...")
    plot_training_curves(fold_results, output_dir)
    plot_cv_summary(fold_results, ensemble_result, output_dir)
    plot_roc_curves(fold_results, output_dir)
    plot_metrics_radar(fold_results, ensemble_result, output_dir)
    plot_ensemble_comparison(fold_results, ensemble_result, output_dir)

    print()
    print(f"All figures saved to: {output_dir}")
    print("=" * 60)


if __name__ == '__main__':
    main()
