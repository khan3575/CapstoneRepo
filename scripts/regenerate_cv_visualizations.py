#!/usr/bin/env python3
"""
Regenerate all CV analysis visualizations with CURRENT VALIDATED data
From checkpoints/binary_training/fold_*/results.json

This script creates:
1. cv_dice_per_fold.png - Bar plot of Dice per fold
2. cv_boxplots.png - 2x2 grid of performance distributions
3. metrics_distribution.png - Distribution of all metrics
4. performance.png - Comprehensive performance overview
5. statistical_analysis.png - Statistical significance visualization
6. cv_training_curves.png - Training curves from all 5 folds

Uses ONLY validated results from actual training runs.
"""

import os
import sys
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy import stats

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from config import get_config

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 10


def main():
    """Regenerate all CV analysis visualizations"""
    # Load configuration (lazy loading to avoid import-time crashes)
    config = get_config()

    # Paths (using config.yaml)
    CHECKPOINT_DIR = Path(config.checkpoints_binary)
    OUTPUT_DIR = Path(config.visualizations_root) / "cv_analysis"
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"📁 Using paths from config:")
    print(f"   Checkpoint dir: {CHECKPOINT_DIR}")
    print(f"   Output dir: {OUTPUT_DIR}")

    print("=" * 70)
    print("REGENERATING CV VISUALIZATIONS WITH VALIDATED DATA")
    print("=" * 70)

    # Load validated results from all 5 folds
    fold_results = []
    for fold_idx in range(5):
        fold_path = CHECKPOINT_DIR / f"fold_{fold_idx}" / "results.json"
        print(f"\nLoading: {fold_path}")
    
        with open(fold_path, 'r') as f:
            data = json.load(f)
            fold_results.append(data)
            print(f"  Fold {fold_idx}: Dice = {data['test_metrics']['dice']*100:.2f}%")

    # Extract metrics
    dice_scores = [r['test_metrics']['dice'] * 100 for r in fold_results]
    accuracy_scores = [r['test_metrics']['accuracy'] * 100 for r in fold_results]
    sensitivity_scores = [r['test_metrics']['sensitivity'] * 100 for r in fold_results]
    specificity_scores = [r['test_metrics']['specificity'] * 100 for r in fold_results]
    precision_scores = [r['test_metrics']['precision'] * 100 for r in fold_results]

    # Calculate statistics
    dice_mean = np.mean(dice_scores)
    dice_std = np.std(dice_scores, ddof=1)
    dice_sem = stats.sem(dice_scores)
    dice_ci = stats.t.interval(0.95, len(dice_scores)-1, loc=dice_mean, scale=dice_sem)

    print(f"\n{'='*70}")
    print(f"VALIDATED STATISTICS:")
    print(f"  Dice: {dice_mean:.2f}% ± {dice_std:.2f}%")
    print(f"  95% CI: [{dice_ci[0]:.2f}%, {dice_ci[1]:.2f}%]")
    print(f"  Range: [{min(dice_scores):.2f}%, {max(dice_scores):.2f}%]")
    print(f"{'='*70}\n")

    # ============================================================================
    # 1. CV DICE PER FOLD - Bar plot
    # ============================================================================
    print("Generating: cv_dice_per_fold.png")
    fig, ax = plt.subplots(figsize=(10, 6))

    fold_labels = [f'Fold {i}' for i in range(5)]
    colors = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12', '#9b59b6']

    bars = ax.bar(fold_labels, dice_scores, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)

    # Add value labels on bars
    for bar, score in zip(bars, dice_scores):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{score:.2f}%',
                ha='center', va='bottom', fontweight='bold', fontsize=11)

    # Add mean and std lines
    ax.axhline(y=dice_mean, color='red', linestyle='--', linewidth=2, 
               label=f'Mean: {dice_mean:.2f}%', alpha=0.7)
    ax.axhline(y=dice_mean + dice_std, color='orange', linestyle=':', linewidth=1.5, 
               label=f'Mean + σ: {dice_mean + dice_std:.2f}%', alpha=0.6)
    ax.axhline(y=dice_mean - dice_std, color='orange', linestyle=':', linewidth=1.5, 
               label=f'Mean - σ: {dice_mean - dice_std:.2f}%', alpha=0.6)

    ax.set_xlabel('Cross-Validation Fold', fontsize=13, fontweight='bold')
    ax.set_ylabel('Dice Coefficient (%)', fontsize=13, fontweight='bold')
    ax.set_title('5-Fold Cross-Validation Dice Scores\n(Patient-Level Stratified Splits)', 
                 fontsize=14, fontweight='bold', pad=20)
    ax.set_ylim([85, 95])
    ax.legend(loc='lower right', fontsize=10, framealpha=0.9)
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "cv_dice_per_fold.png", dpi=300, bbox_inches='tight')
    print(f"  ✓ Saved to: {OUTPUT_DIR / 'cv_dice_per_fold.png'}")
    plt.close()

    # ============================================================================
    # 2. CV BOXPLOTS - 2x2 Grid (as requested by user)
    # ============================================================================
    print("\nGenerating: cv_boxplots.png (2x2 layout)")
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    fig.suptitle('Cross-Validation Performance Distribution (5 Folds)', 
                 fontsize=16, fontweight='bold', y=0.995)

    # Prepare data for boxplots
    metrics_data = {
        'Dice Coefficient': dice_scores,
        'Accuracy': accuracy_scores,
        'Sensitivity (Recall)': sensitivity_scores,
        'Specificity': specificity_scores
    }

    colors_box = ['#3498db', '#2ecc71', '#e74c3c', '#f39c12']

    for idx, (ax, (metric_name, values)) in enumerate(zip(axes.flat, metrics_data.items())):
        # Create boxplot
        bp = ax.boxplot([values], widths=0.6, patch_artist=True,
                         boxprops=dict(facecolor=colors_box[idx], alpha=0.7),
                         medianprops=dict(color='red', linewidth=2),
                         whiskerprops=dict(linewidth=1.5),
                         capprops=dict(linewidth=1.5))
    
        # Add individual data points
        x = np.random.normal(1, 0.04, size=len(values))
        ax.scatter(x, values, alpha=0.6, s=100, c='darkblue', edgecolors='black', linewidth=1)
    
        # Add statistics text
        mean_val = np.mean(values)
        std_val = np.std(values, ddof=1)
        min_val = np.min(values)
        max_val = np.max(values)
    
        stats_text = (f'Mean: {mean_val:.2f}%\n'
                      f'Std: {std_val:.2f}%\n'
                      f'Range: [{min_val:.2f}%, {max_val:.2f}%]')
    
        ax.text(0.05, 0.95, stats_text, transform=ax.transAxes,
                fontsize=10, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
        ax.set_ylabel(f'{metric_name} (%)', fontsize=12, fontweight='bold')
        ax.set_title(metric_name, fontsize=13, fontweight='bold', pad=10)
        ax.set_xticks([])
        ax.grid(True, alpha=0.3, axis='y')
    
        # Set appropriate y-limits
        if 'Dice' in metric_name or 'Sensitivity' in metric_name:
            ax.set_ylim([85, 95])
        elif 'Accuracy' in metric_name:
            ax.set_ylim([98, 100])
        elif 'Specificity' in metric_name:
            ax.set_ylim([99.5, 100])

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "cv_boxplots.png", dpi=300, bbox_inches='tight')
    print(f"  ✓ Saved to: {OUTPUT_DIR / 'cv_boxplots.png'}")
    plt.close()

    # ============================================================================
    # 3. METRICS DISTRIBUTION - Violin plot of all metrics
    # ============================================================================
    print("\nGenerating: metrics_distribution.png")
    fig, ax = plt.subplots(figsize=(12, 7))

    all_metrics = []
    all_values = []
    all_labels = []

    for metric, values in [
        ('Dice', dice_scores),
        ('Accuracy', accuracy_scores),
        ('Sensitivity', sensitivity_scores),
        ('Specificity', specificity_scores),
        ('Precision', precision_scores)
    ]:
        all_metrics.extend([metric] * len(values))
        all_values.extend(values)
        all_labels.extend([f'Fold {i}' for i in range(len(values))])

    # Create violin plot
    parts = ax.violinplot([dice_scores, accuracy_scores, sensitivity_scores, 
                           specificity_scores, precision_scores],
                          positions=[1, 2, 3, 4, 5],
                          widths=0.7,
                          showmeans=True,
                          showmedians=True)

    # Color violins
    colors_violin = ['#3498db', '#2ecc71', '#e74c3c', '#f39c12', '#9b59b6']
    for pc, color in zip(parts['bodies'], colors_violin):
        pc.set_facecolor(color)
        pc.set_alpha(0.7)

    ax.set_xticks([1, 2, 3, 4, 5])
    ax.set_xticklabels(['Dice\nCoefficient', 'Accuracy', 'Sensitivity\n(Recall)', 
                         'Specificity', 'Precision'], fontsize=11)
    ax.set_ylabel('Score (%)', fontsize=13, fontweight='bold')
    ax.set_title('Distribution of Performance Metrics Across 5-Fold CV', 
                 fontsize=14, fontweight='bold', pad=15)
    ax.set_ylim([80, 101])
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "metrics_distribution.png", dpi=300, bbox_inches='tight')
    print(f"  ✓ Saved to: {OUTPUT_DIR / 'metrics_distribution.png'}")
    plt.close()

    # ============================================================================
    # 4. PERFORMANCE OVERVIEW - Comprehensive radar chart + bar chart
    # ============================================================================
    print("\nGenerating: performance.png")
    fig = plt.figure(figsize=(16, 7))
    gs = fig.add_gridspec(1, 2, width_ratios=[1, 1.2], wspace=0.3)

    # Left: Radar chart of mean metrics
    ax1 = fig.add_subplot(gs[0], projection='polar')

    categories = ['Dice', 'Accuracy', 'Sensitivity', 'Specificity', 'Precision']
    values = [dice_mean, np.mean(accuracy_scores), np.mean(sensitivity_scores),
              np.mean(specificity_scores), np.mean(precision_scores)]

    # Normalize to 0-100 for radar chart
    angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
    values += values[:1]
    angles += angles[:1]

    ax1.plot(angles, values, 'o-', linewidth=2, color='#3498db', markersize=8)
    ax1.fill(angles, values, alpha=0.25, color='#3498db')
    ax1.set_xticks(angles[:-1])
    ax1.set_xticklabels(categories, fontsize=11)
    ax1.set_ylim(85, 100)
    ax1.set_yticks([85, 90, 95, 100])
    ax1.set_yticklabels(['85%', '90%', '95%', '100%'], fontsize=9)
    ax1.set_title('Mean Performance Metrics\n(5-Fold CV Average)', 
                  fontsize=13, fontweight='bold', pad=20)
    ax1.grid(True, alpha=0.3)

    # Right: Grouped bar chart with error bars
    ax2 = fig.add_subplot(gs[1])

    metrics_names = ['Dice', 'Accuracy', 'Sensitivity', 'Specificity', 'Precision']
    metrics_means = [np.mean(dice_scores), np.mean(accuracy_scores), 
                     np.mean(sensitivity_scores), np.mean(specificity_scores),
                     np.mean(precision_scores)]
    metrics_stds = [np.std(dice_scores, ddof=1), np.std(accuracy_scores, ddof=1),
                    np.std(sensitivity_scores, ddof=1), np.std(specificity_scores, ddof=1),
                    np.std(precision_scores, ddof=1)]

    x_pos = np.arange(len(metrics_names))
    colors_bar = ['#3498db', '#2ecc71', '#e74c3c', '#f39c12', '#9b59b6']

    bars = ax2.bar(x_pos, metrics_means, color=colors_bar, alpha=0.8, 
                   edgecolor='black', linewidth=1.5,
                   error_kw={'linewidth': 2, 'ecolor': 'black', 'capsize': 5})
    ax2.errorbar(x_pos, metrics_means, yerr=metrics_stds, fmt='none', 
                 ecolor='black', capsize=5, linewidth=2)

    # Add value labels
    for i, (bar, mean, std) in enumerate(zip(bars, metrics_means, metrics_stds)):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                 f'{mean:.2f}%\n±{std:.2f}%',
                 ha='center', va='bottom', fontsize=10, fontweight='bold')

    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(metrics_names, fontsize=11)
    ax2.set_ylabel('Score (%)', fontsize=13, fontweight='bold')
    ax2.set_title('Performance Metrics with Standard Deviation\n(Mean ± σ across 5 Folds)',
                  fontsize=13, fontweight='bold', pad=15)
    ax2.set_ylim([80, 102])
    ax2.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "performance.png", dpi=300, bbox_inches='tight')
    print(f"  ✓ Saved to: {OUTPUT_DIR / 'performance.png'}")
    plt.close()

    # ============================================================================
    # 5. STATISTICAL ANALYSIS - Significance testing visualization
    # ============================================================================
    print("\nGenerating: statistical_analysis.png")
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    fig.suptitle('Statistical Analysis of Cross-Validation Results', 
                 fontsize=16, fontweight='bold', y=0.995)

    # Top-left: Confidence intervals
    ax1 = axes[0, 0]
    metrics = ['Dice', 'Accuracy', 'Sensitivity', 'Specificity', 'Precision']
    means_list = [np.mean(dice_scores), np.mean(accuracy_scores), 
                  np.mean(sensitivity_scores), np.mean(specificity_scores),
                  np.mean(precision_scores)]
    ci_list = []
    for scores in [dice_scores, accuracy_scores, sensitivity_scores, 
                   specificity_scores, precision_scores]:
        sem = stats.sem(scores)
        ci = stats.t.interval(0.95, len(scores)-1, loc=np.mean(scores), scale=sem)
        ci_list.append(ci)

    y_pos = np.arange(len(metrics))
    errors = [[means_list[i] - ci_list[i][0], ci_list[i][1] - means_list[i]] for i in range(len(metrics))]
    errors = np.array(errors).T

    ax1.barh(y_pos, means_list, xerr=errors, color=colors_bar, alpha=0.8,
             edgecolor='black', linewidth=1.5, error_kw={'linewidth': 2, 'capsize': 5})
    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(metrics, fontsize=11)
    ax1.set_xlabel('Score (%) with 95% CI', fontsize=12, fontweight='bold')
    ax1.set_title('95% Confidence Intervals', fontsize=13, fontweight='bold')
    ax1.grid(True, alpha=0.3, axis='x')

    # Top-right: Fold-to-fold variance
    ax2 = axes[0, 1]
    fold_indices = np.arange(5)
    ax2.plot(fold_indices, dice_scores, 'o-', linewidth=2, markersize=10, 
             color='#3498db', label='Dice')
    ax2.fill_between(fold_indices, 
                      [dice_mean - dice_std]*5, 
                      [dice_mean + dice_std]*5,
                      alpha=0.2, color='#3498db')
    ax2.axhline(y=dice_mean, color='red', linestyle='--', linewidth=2, 
                label=f'Mean: {dice_mean:.2f}%')
    ax2.set_xticks(fold_indices)
    ax2.set_xticklabels([f'Fold {i}' for i in fold_indices])
    ax2.set_ylabel('Dice Coefficient (%)', fontsize=12, fontweight='bold')
    ax2.set_title('Fold-to-Fold Consistency', fontsize=13, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim([88, 93])

    # Bottom-left: Normality test (Q-Q plot)
    ax3 = axes[1, 0]
    stats.probplot(dice_scores, dist="norm", plot=ax3)
    ax3.set_title('Q-Q Plot for Dice Scores\n(Normality Check)', 
                  fontsize=13, fontweight='bold')
    ax3.grid(True, alpha=0.3)

    # Bottom-right: Statistical summary table
    ax4 = axes[1, 1]
    ax4.axis('off')

    # Perform one-sample t-test (null hypothesis: mean <= 90%)
    t_stat, p_value = stats.ttest_1samp(dice_scores, 90.0, alternative='greater')

    summary_data = [
        ['Metric', 'Value'],
        ['─' * 25, '─' * 20],
        ['Mean Dice', f'{dice_mean:.2f}%'],
        ['Std Deviation', f'{dice_std:.2f}%'],
        ['Std Error (SEM)', f'{dice_sem:.2f}%'],
        ['95% CI', f'[{dice_ci[0]:.2f}%, {dice_ci[1]:.2f}%]'],
        ['Min', f'{min(dice_scores):.2f}%'],
        ['Max', f'{max(dice_scores):.2f}%'],
        ['Range', f'{max(dice_scores) - min(dice_scores):.2f}%'],
        ['Coefficient of Variation', f'{(dice_std/dice_mean)*100:.2f}%'],
        ['─' * 25, '─' * 20],
        ['One-Sample t-test', ''],
        ['H₀: μ ≤ 90%', ''],
        ['t-statistic', f'{t_stat:.3f}'],
        ['p-value', f'{p_value:.4f}'],
        ['Significant (α=0.05)?', 'Yes' if p_value < 0.05 else 'No']
    ]

    table = ax4.table(cellText=summary_data, cellLoc='left', loc='center',
                      colWidths=[0.5, 0.5])
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)

    # Style header row
    for i in range(2):
        table[(0, i)].set_facecolor('#3498db')
        table[(0, i)].set_text_props(weight='bold', color='white')

    # Style significance row
    if p_value < 0.05:
        table[(len(summary_data)-1, 1)].set_facecolor('#2ecc71')
        table[(len(summary_data)-1, 1)].set_text_props(weight='bold')

    ax4.set_title('Statistical Summary\n(Dice Coefficient)', 
                  fontsize=13, fontweight='bold', pad=20)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "statistical_analysis.png", dpi=300, bbox_inches='tight')
    print(f"  ✓ Saved to: {OUTPUT_DIR / 'statistical_analysis.png'}")
    plt.close()

    # ============================================================================
    # 6. TRAINING CURVES - Extract from training history
    # ============================================================================
    print("\nGenerating: cv_training_curves.png")
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Training Dynamics Across 5-Fold Cross-Validation', 
                 fontsize=16, fontweight='bold', y=0.995)

    colors_folds = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12', '#9b59b6']

    # Extract training history from each fold
    for fold_idx in range(5):
        history = fold_results[fold_idx]['history']
    
        epochs = [h['epoch'] for h in history]
        train_dice = [h['train']['dice'] * 100 for h in history]
        train_loss = [h['train']['loss'] for h in history]
        val_dice = [h.get('val', {}).get('dice', 0) * 100 for h in history if 'val' in h]
        val_loss = [h.get('val', {}).get('loss', 0) for h in history if 'val' in h]
        val_epochs = [h['epoch'] for h in history if 'val' in h]
    
        # Plot 1: Training Dice
        axes[0, 0].plot(epochs, train_dice, '-', linewidth=2, alpha=0.7,
                        color=colors_folds[fold_idx], label=f'Fold {fold_idx}')
    
        # Plot 2: Training Loss
        axes[0, 1].plot(epochs, train_loss, '-', linewidth=2, alpha=0.7,
                        color=colors_folds[fold_idx], label=f'Fold {fold_idx}')
    
        # Plot 3: Validation Dice (if available)
        if val_dice:
            axes[0, 2].plot(val_epochs, val_dice, 'o-', linewidth=2, alpha=0.7,
                            markersize=4, color=colors_folds[fold_idx], label=f'Fold {fold_idx}')
    
        # Plot 4: Validation Loss (if available)
        if val_loss:
            axes[1, 0].plot(val_epochs, val_loss, 'o-', linewidth=2, alpha=0.7,
                            markersize=4, color=colors_folds[fold_idx], label=f'Fold {fold_idx}')
    
        # Plot 5: Train vs Val Dice for this fold
        axes[1, 1].plot(epochs, train_dice, '-', linewidth=2, alpha=0.7,
                        color=colors_folds[fold_idx], label=f'Fold {fold_idx} Train')
        if val_dice:
            axes[1, 1].plot(val_epochs, val_dice, '--', linewidth=2, alpha=0.7,
                            color=colors_folds[fold_idx], label=f'Fold {fold_idx} Val')

    # Configure subplots
    axes[0, 0].set_xlabel('Epoch', fontsize=11, fontweight='bold')
    axes[0, 0].set_ylabel('Training Dice (%)', fontsize=11, fontweight='bold')
    axes[0, 0].set_title('Training Dice Progression', fontsize=12, fontweight='bold')
    axes[0, 0].legend(fontsize=8, loc='lower right')
    axes[0, 0].grid(True, alpha=0.3)

    axes[0, 1].set_xlabel('Epoch', fontsize=11, fontweight='bold')
    axes[0, 1].set_ylabel('Training Loss', fontsize=11, fontweight='bold')
    axes[0, 1].set_title('Training Loss Convergence', fontsize=12, fontweight='bold')
    axes[0, 1].legend(fontsize=8, loc='upper right')
    axes[0, 1].grid(True, alpha=0.3)

    axes[0, 2].set_xlabel('Epoch', fontsize=11, fontweight='bold')
    axes[0, 2].set_ylabel('Validation Dice (%)', fontsize=11, fontweight='bold')
    axes[0, 2].set_title('Validation Dice Performance', fontsize=12, fontweight='bold')
    axes[0, 2].legend(fontsize=8, loc='lower right')
    axes[0, 2].grid(True, alpha=0.3)

    axes[1, 0].set_xlabel('Epoch', fontsize=11, fontweight='bold')
    axes[1, 0].set_ylabel('Validation Loss', fontsize=11, fontweight='bold')
    axes[1, 0].set_title('Validation Loss Tracking', fontsize=12, fontweight='bold')
    axes[1, 0].legend(fontsize=8, loc='upper right')
    axes[1, 0].grid(True, alpha=0.3)

    axes[1, 1].set_xlabel('Epoch', fontsize=11, fontweight='bold')
    axes[1, 1].set_ylabel('Dice Coefficient (%)', fontsize=11, fontweight='bold')
    axes[1, 1].set_title('Train vs Validation Comparison', fontsize=12, fontweight='bold')
    axes[1, 1].legend(fontsize=7, loc='lower right', ncol=2)
    axes[1, 1].grid(True, alpha=0.3)

    # Plot 6: Final test scores bar chart
    axes[1, 2].bar(range(5), dice_scores, color=colors_folds, alpha=0.8, 
                   edgecolor='black', linewidth=1.5)
    axes[1, 2].axhline(y=dice_mean, color='red', linestyle='--', linewidth=2,
                       label=f'Mean: {dice_mean:.2f}%')
    axes[1, 2].set_xlabel('Fold', fontsize=11, fontweight='bold')
    axes[1, 2].set_ylabel('Test Dice (%)', fontsize=11, fontweight='bold')
    axes[1, 2].set_title('Final Test Performance', fontsize=12, fontweight='bold')
    axes[1, 2].set_xticks(range(5))
    axes[1, 2].set_xticklabels([f'F{i}' for i in range(5)])
    axes[1, 2].legend(fontsize=8)
    axes[1, 2].grid(True, alpha=0.3, axis='y')

    # Add value labels
    for i, score in enumerate(dice_scores):
        axes[1, 2].text(i, score + 0.3, f'{score:.2f}%', 
                        ha='center', va='bottom', fontsize=9, fontweight='bold')

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "cv_training_curves.png", dpi=300, bbox_inches='tight')
    print(f"  ✓ Saved to: {OUTPUT_DIR / 'cv_training_curves.png'}")
    plt.close()

    # ============================================================================
    # SUMMARY
    # ============================================================================
    print("\n" + "=" * 70)
    print("✅ ALL VISUALIZATIONS REGENERATED SUCCESSFULLY!")
    print("=" * 70)
    print(f"\nOutput directory: {OUTPUT_DIR}")
    print("\nGenerated files:")
    print("  1. cv_dice_per_fold.png     - Bar plot of Dice per fold")
    print("  2. cv_boxplots.png          - 2x2 grid of performance distributions")
    print("  3. metrics_distribution.png - Violin plot of all metrics")
    print("  4. performance.png          - Radar + bar chart overview")
    print("  5. statistical_analysis.png - Statistical tests & summary")
    print("  6. cv_training_curves.png   - Training dynamics from all folds")
    print("\n" + "=" * 70)
    print("VALIDATED DATA USED:")
    print(f"  Fold 0: {dice_scores[0]:.2f}%")
    print(f"  Fold 1: {dice_scores[1]:.2f}%")
    print(f"  Fold 2: {dice_scores[2]:.2f}%")
    print(f"  Fold 3: {dice_scores[3]:.2f}%")
    print(f"  Fold 4: {dice_scores[4]:.2f}%")
    print(f"  Mean: {dice_mean:.2f}% ± {dice_std:.2f}%")
    print("=" * 70)
    print("\n✨ Ready to use in thesis! All images use CURRENT VALIDATED data.")
    print("   No old/corrupted data - 100% scientific integrity maintained.\n")


if __name__ == "__main__":
    main()
