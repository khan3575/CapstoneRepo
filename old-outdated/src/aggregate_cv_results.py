"""
Aggregate cross-validation results and perform statistical analysis
"""

import os
import json
import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats
from typing import Dict, List
import matplotlib.pyplot as plt
import seaborn as sns


def load_fold_results(cv_dir: str, k: int = 5) -> List[Dict]:
    """
    Load results from all CV folds
    
    Args:
        cv_dir: Directory containing fold results
        k: Number of folds
        
    Returns:
        List of result dictionaries, one per fold
    """
    cv_path = Path(cv_dir)
    fold_results = []
    
    print(f"Loading results from {k} folds...")
    for fold_idx in range(k):
        fold_dir = cv_path / f'fold_{fold_idx}'
        results_file = fold_dir / 'results.json'
        
        if not results_file.exists():
            print(f"  ⚠️  Warning: Results not found for fold {fold_idx}")
            continue
        
        with open(results_file, 'r') as f:
            results = json.load(f)
            fold_results.append(results)
            print(f"  ✓ Fold {fold_idx}: Test Dice = {results['test_metrics']['dice']:.4f}")
    
    if len(fold_results) == 0:
        raise ValueError(f"No fold results found in {cv_dir}")
    
    print(f"\nLoaded {len(fold_results)} fold results\n")
    return fold_results


def compute_statistics(values: List[float]) -> Dict:
    """
    Compute comprehensive statistics for a metric
    
    Args:
        values: List of metric values across folds
        
    Returns:
        Dictionary with mean, std, CI, etc.
    """
    values = np.array(values)
    n = len(values)
    
    # Basic statistics
    mean = np.mean(values)
    std = np.std(values, ddof=1)  # Sample std
    sem = stats.sem(values)  # Standard error of mean
    
    # Confidence interval (95%)
    ci_95 = stats.t.interval(0.95, n-1, loc=mean, scale=sem)
    
    # Additional statistics
    median = np.median(values)
    min_val = np.min(values)
    max_val = np.max(values)
    
    return {
        'mean': float(mean),
        'std': float(std),
        'sem': float(sem),
        'median': float(median),
        'min': float(min_val),
        'max': float(max_val),
        'ci_95_lower': float(ci_95[0]),
        'ci_95_upper': float(ci_95[1]),
        'n': n
    }


def aggregate_metrics(fold_results: List[Dict]) -> Dict:
    """
    Aggregate metrics across all folds
    
    Args:
        fold_results: List of fold result dictionaries
        
    Returns:
        Dictionary with aggregated statistics
    """
    # Extract metrics from each fold
    metrics_by_fold = {
        'dice': [],
        'accuracy': [],
        'sensitivity': [],
        'specificity': [],
        'precision': [],
        'val_dice': []
    }
    
    for fold_result in fold_results:
        test_metrics = fold_result['test_metrics']
        metrics_by_fold['dice'].append(test_metrics['dice'])
        metrics_by_fold['accuracy'].append(test_metrics['accuracy'])
        metrics_by_fold['sensitivity'].append(test_metrics['sensitivity'])
        metrics_by_fold['specificity'].append(test_metrics['specificity'])
        metrics_by_fold['precision'].append(test_metrics['precision'])
        metrics_by_fold['val_dice'].append(fold_result['best_val_dice'])
    
    # Compute statistics for each metric
    aggregated = {}
    for metric_name, values in metrics_by_fold.items():
        aggregated[metric_name] = compute_statistics(values)
    
    return aggregated


def create_results_table(aggregated: Dict) -> pd.DataFrame:
    """
    Create publication-ready results table
    
    Args:
        aggregated: Aggregated statistics dictionary
        
    Returns:
        Pandas DataFrame with formatted results
    """
    metrics = ['dice', 'accuracy', 'sensitivity', 'specificity', 'precision']
    
    rows = []
    for metric in metrics:
        stats = aggregated[metric]
        rows.append({
            'Metric': metric.capitalize(),
            'Mean': f"{stats['mean']:.4f}",
            'Std': f"{stats['std']:.4f}",
            'Median': f"{stats['median']:.4f}",
            'Min': f"{stats['min']:.4f}",
            'Max': f"{stats['max']:.4f}",
            '95% CI': f"[{stats['ci_95_lower']:.4f}, {stats['ci_95_upper']:.4f}]"
        })
    
    df = pd.DataFrame(rows)
    return df


def plot_cv_results(fold_results: List[Dict], aggregated: Dict, output_dir: str):
    """
    Create visualization plots for CV results
    
    Args:
        fold_results: List of fold results
        aggregated: Aggregated statistics
        output_dir: Directory to save plots
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Set style
    sns.set_style("whitegrid")
    plt.rcParams['figure.figsize'] = (12, 8)
    
    # Plot 1: Box plots for all metrics
    metrics = ['dice', 'accuracy', 'sensitivity', 'specificity', 'precision']
    metric_data = {metric: [] for metric in metrics}
    
    for fold_result in fold_results:
        for metric in metrics:
            metric_data[metric].append(fold_result['test_metrics'][metric])
    
    fig, axes = plt.subplots(1, 5, figsize=(20, 4))
    for idx, metric in enumerate(metrics):
        axes[idx].boxplot(metric_data[metric])
        axes[idx].set_title(f'{metric.capitalize()}', fontsize=12, fontweight='bold')
        axes[idx].set_ylabel('Score')
        axes[idx].set_ylim([0.9, 1.0])
        axes[idx].grid(True, alpha=0.3)
        
        # Add mean line
        mean_val = aggregated[metric]['mean']
        axes[idx].axhline(y=mean_val, color='r', linestyle='--', 
                         label=f'Mean: {mean_val:.4f}')
        axes[idx].legend()
    
    plt.tight_layout()
    plt.savefig(output_path / 'cv_boxplots.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved boxplot to {output_path / 'cv_boxplots.png'}")
    
    # Plot 2: Dice scores across folds
    fig, ax = plt.subplots(figsize=(10, 6))
    fold_indices = [r['fold_idx'] for r in fold_results]
    dice_scores = [r['test_metrics']['dice'] for r in fold_results]
    
    ax.bar(fold_indices, dice_scores, alpha=0.7, color='steelblue', edgecolor='black')
    ax.axhline(y=aggregated['dice']['mean'], color='red', linestyle='--', 
               linewidth=2, label=f"Mean: {aggregated['dice']['mean']:.4f}")
    ax.fill_between([-0.5, len(fold_results)-0.5], 
                     aggregated['dice']['ci_95_lower'], 
                     aggregated['dice']['ci_95_upper'],
                     alpha=0.2, color='red', label='95% CI')
    
    ax.set_xlabel('Fold Index', fontsize=12, fontweight='bold')
    ax.set_ylabel('Dice Score', fontsize=12, fontweight='bold')
    ax.set_title('Dice Scores Across Cross-Validation Folds', 
                fontsize=14, fontweight='bold')
    ax.set_xticks(fold_indices)
    ax.set_ylim([0.90, 1.0])
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path / 'cv_dice_per_fold.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved dice plot to {output_path / 'cv_dice_per_fold.png'}")
    
    # Plot 3: Training curves (if available)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    for fold_result in fold_results:
        fold_idx = fold_result['fold_idx']
        history = fold_result.get('history', [])
        
        if len(history) > 0:
            epochs = [h['epoch'] for h in history]
            train_dice = [h['train']['dice'] for h in history]
            val_dice = [h['val']['dice'] for h in history]
            
            axes[0].plot(epochs, train_dice, alpha=0.5, label=f'Fold {fold_idx}')
            axes[1].plot(epochs, val_dice, alpha=0.5, label=f'Fold {fold_idx}')
    
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Dice Score')
    axes[0].set_title('Training Dice Scores')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Dice Score')
    axes[1].set_title('Validation Dice Scores')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path / 'cv_training_curves.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved training curves to {output_path / 'cv_training_curves.png'}")


def perform_statistical_tests(fold_results: List[Dict]) -> Dict:
    """
    Perform statistical significance tests
    
    Args:
        fold_results: List of fold results
        
    Returns:
        Dictionary with statistical test results
    """
    dice_scores = [r['test_metrics']['dice'] for r in fold_results]
    
    # One-sample t-test (test if mean > 0.95)
    t_stat, p_value = stats.ttest_1samp(dice_scores, 0.95)
    
    # Normality test
    shapiro_stat, shapiro_p = stats.shapiro(dice_scores)
    
    return {
        'one_sample_t_test': {
            'null_hypothesis': 'mean_dice <= 0.95',
            't_statistic': float(t_stat),
            'p_value': float(p_value),
            'significant': p_value < 0.05 and t_stat > 0
        },
        'normality_test': {
            'test': 'Shapiro-Wilk',
            'statistic': float(shapiro_stat),
            'p_value': float(shapiro_p),
            'is_normal': shapiro_p > 0.05
        }
    }


def generate_report(
    fold_results: List[Dict],
    aggregated: Dict,
    statistical_tests: Dict,
    output_dir: str
):
    """
    Generate comprehensive markdown report
    
    Args:
        fold_results: List of fold results
        aggregated: Aggregated statistics
        statistical_tests: Statistical test results
        output_dir: Directory to save report
    """
    output_path = Path(output_dir)
    report_file = output_path / 'cv_report.md'
    
    with open(report_file, 'w') as f:
        f.write("# Cross-Validation Results Report\n\n")
        f.write(f"**Generated:** {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write(f"**Number of Folds:** {len(fold_results)}\n\n")
        
        # Overall results
        f.write("## Overall Performance\n\n")
        f.write("### Test Set Metrics (Mean ± Std)\n\n")
        f.write("| Metric | Mean | Std | 95% CI | Min | Max |\n")
        f.write("|--------|------|-----|--------|-----|-----|\n")
        
        for metric in ['dice', 'accuracy', 'sensitivity', 'specificity', 'precision']:
            stats = aggregated[metric]
            f.write(f"| {metric.capitalize()} | "
                   f"{stats['mean']:.4f} | "
                   f"{stats['std']:.4f} | "
                   f"[{stats['ci_95_lower']:.4f}, {stats['ci_95_upper']:.4f}] | "
                   f"{stats['min']:.4f} | "
                   f"{stats['max']:.4f} |\n")
        
        # Per-fold results
        f.write("\n## Per-Fold Results\n\n")
        f.write("| Fold | Dice | Accuracy | Sensitivity | Specificity | Precision |\n")
        f.write("|------|------|----------|-------------|-------------|----------|\n")
        
        for fold_result in fold_results:
            fold_idx = fold_result['fold_idx']
            tm = fold_result['test_metrics']
            f.write(f"| {fold_idx} | "
                   f"{tm['dice']:.4f} | "
                   f"{tm['accuracy']:.4f} | "
                   f"{tm['sensitivity']:.4f} | "
                   f"{tm['specificity']:.4f} | "
                   f"{tm['precision']:.4f} |\n")
        
        # Statistical tests
        f.write("\n## Statistical Analysis\n\n")
        f.write("### One-Sample t-test\n")
        f.write(f"- **Null Hypothesis:** Mean Dice ≤ 0.95\n")
        f.write(f"- **t-statistic:** {statistical_tests['one_sample_t_test']['t_statistic']:.4f}\n")
        f.write(f"- **p-value:** {statistical_tests['one_sample_t_test']['p_value']:.6f}\n")
        
        if statistical_tests['one_sample_t_test']['significant']:
            f.write(f"- **Result:** ✅ Significant (p < 0.05). Mean Dice is significantly > 0.95\n")
        else:
            f.write(f"- **Result:** Not significant\n")
        
        f.write("\n### Normality Test (Shapiro-Wilk)\n")
        f.write(f"- **Statistic:** {statistical_tests['normality_test']['statistic']:.4f}\n")
        f.write(f"- **p-value:** {statistical_tests['normality_test']['p_value']:.6f}\n")
        
        if statistical_tests['normality_test']['is_normal']:
            f.write(f"- **Result:** Data is normally distributed (p > 0.05)\n")
        else:
            f.write(f"- **Result:** Data may not be normally distributed\n")
        
        # Training time
        f.write("\n## Training Time Analysis\n\n")
        total_times = [r['training_time'] for r in fold_results]
        f.write(f"- **Total training time:** {sum(total_times)/3600:.2f} hours\n")
        f.write(f"- **Average per fold:** {np.mean(total_times)/60:.1f} minutes\n")
        f.write(f"- **Std per fold:** {np.std(total_times)/60:.1f} minutes\n")
        
        # Conclusion
        f.write("\n## Conclusion\n\n")
        dice_mean = aggregated['dice']['mean']
        dice_std = aggregated['dice']['std']
        f.write(f"The model achieved a mean Dice score of **{dice_mean:.4f} ± {dice_std:.4f}** "
               f"across {len(fold_results)}-fold cross-validation. ")
        
        if statistical_tests['one_sample_t_test']['significant']:
            f.write("This performance is statistically significantly better than 0.95 (p < 0.05), "
                   "demonstrating strong and consistent segmentation performance.\n")
        
        f.write("\n## Visualizations\n\n")
        f.write("- Box plots: `cv_boxplots.png`\n")
        f.write("- Dice per fold: `cv_dice_per_fold.png`\n")
        f.write("- Training curves: `cv_training_curves.png`\n")
    
    print(f"✓ Saved report to {report_file}")


def aggregate_cross_validation_results(
    cv_dir: str,
    output_dir: str = None,
    k: int = 5
):
    """
    Main function to aggregate and analyze CV results
    
    Args:
        cv_dir: Directory containing fold results
        output_dir: Directory to save aggregated results (default: cv_dir/aggregated)
        k: Number of folds
    """
    if output_dir is None:
        output_dir = str(Path(cv_dir) / 'aggregated')
    
    print("=" * 80)
    print("CROSS-VALIDATION RESULTS AGGREGATION")
    print("=" * 80)
    
    # Load fold results
    fold_results = load_fold_results(cv_dir, k)
    
    # Aggregate metrics
    print("Computing statistics...")
    aggregated = aggregate_metrics(fold_results)
    
    # Statistical tests
    print("Performing statistical tests...")
    statistical_tests = perform_statistical_tests(fold_results)
    
    # Create visualizations
    print("Creating visualizations...")
    plot_cv_results(fold_results, aggregated, output_dir)
    
    # Generate report
    print("Generating report...")
    generate_report(fold_results, aggregated, statistical_tests, output_dir)
    
    # Save aggregated results as JSON
    results_json = Path(output_dir) / 'aggregated_results.json'
    with open(results_json, 'w') as f:
        json.dump({
            'aggregated_metrics': aggregated,
            'statistical_tests': statistical_tests,
            'fold_results': fold_results
        }, f, indent=2)
    print(f"✓ Saved JSON results to {results_json}")
    
    # Print summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"\nTest Set Performance ({k}-fold CV):")
    print(f"  Dice Score:  {aggregated['dice']['mean']:.4f} ± {aggregated['dice']['std']:.4f}")
    print(f"  95% CI:      [{aggregated['dice']['ci_95_lower']:.4f}, {aggregated['dice']['ci_95_upper']:.4f}]")
    print(f"  Range:       [{aggregated['dice']['min']:.4f}, {aggregated['dice']['max']:.4f}]")
    print(f"\nResults saved to: {output_dir}")
    print("=" * 80)
    
    return aggregated


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Aggregate CV results')
    parser.add_argument('--cv_dir', type=str,
                       default='./checkpoints/cv_experiments',
                       help='Directory containing fold results')
    parser.add_argument('--output_dir', type=str, default=None,
                       help='Output directory (default: cv_dir/aggregated)')
    parser.add_argument('--k', type=int, default=5,
                       help='Number of folds')
    
    args = parser.parse_args()
    
    # Aggregate results
    aggregated = aggregate_cross_validation_results(
        cv_dir=args.cv_dir,
        output_dir=args.output_dir,
        k=args.k
    )
    
    print("\n✅ Cross-validation analysis complete!")
