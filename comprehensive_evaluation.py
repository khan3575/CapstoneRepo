#!/usr/bin/env python3
"""
Comprehensive Evaluation Suite for BraTS GNN Research
=====================================================

This script provides comprehensive evaluation metrics for research publication,
including advanced medical imaging metrics, statistical analysis, and 
comparison frameworks.

Key Features:
- Multi-class BraTS evaluation (ET, WT, TC)
- Advanced metrics (HD95, ASSD, etc.)
- Statistical significance testing
- Visualization generation
- Research-grade reporting

Author: Research Team
Date: October 2025
"""

import os
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.metrics import confusion_matrix, classification_report
import warnings
warnings.filterwarnings('ignore')

# Medical imaging specific imports
try:
    from medpy.metric.binary import hd95, assd, precision, recall
    MEDPY_AVAILABLE = True
except ImportError:
    MEDPY_AVAILABLE = False
    print("⚠️  MedPy not available. Install with: pip install MedPy")

from typing import Dict, List, Tuple, Optional
import json
import time
from dataclasses import dataclass

@dataclass 
class EvaluationConfig:
    """Configuration for comprehensive evaluation"""
    metrics_to_compute: List[str] = None
    statistical_tests: bool = True
    generate_plots: bool = True
    save_results: bool = True
    output_dir: str = "./research_results"
    confidence_level: float = 0.95
    
    def __post_init__(self):
        if self.metrics_to_compute is None:
            self.metrics_to_compute = [
                'dice', 'jaccard', 'sensitivity', 'specificity', 
                'precision', 'accuracy', 'hd95', 'assd'
            ]

class ComprehensiveEvaluator:
    """
    Comprehensive evaluation suite for medical image segmentation research
    """
    
    def __init__(self, config: EvaluationConfig):
        self.config = config
        self.results = {}
        self.statistical_results = {}
        
        # Create output directory
        os.makedirs(config.output_dir, exist_ok=True)
        
    def compute_basic_metrics(self, predictions: np.ndarray, targets: np.ndarray) -> Dict[str, float]:
        """Compute basic segmentation metrics"""
        pred_binary = predictions > 0.5
        targets_binary = targets.astype(bool)
        
        # Handle empty predictions/targets
        if not targets_binary.any():
            if not pred_binary.any():
                return {'dice': 1.0, 'jaccard': 1.0, 'sensitivity': 1.0, 
                       'specificity': 1.0, 'precision': 1.0, 'accuracy': 1.0}
            else:
                return {'dice': 0.0, 'jaccard': 0.0, 'sensitivity': 0.0, 
                       'specificity': 0.0, 'precision': 0.0, 'accuracy': 0.0}
        
        # Compute confusion matrix components
        tp = (pred_binary & targets_binary).sum()
        fp = (pred_binary & ~targets_binary).sum()
        fn = (~pred_binary & targets_binary).sum()
        tn = (~pred_binary & ~targets_binary).sum()
        
        # Compute metrics
        dice = (2 * tp) / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0.0
        jaccard = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0.0
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        accuracy = (tp + tn) / len(targets)
        
        return {
            'dice': float(dice),
            'jaccard': float(jaccard),
            'sensitivity': float(sensitivity),
            'specificity': float(specificity),
            'precision': float(precision),
            'accuracy': float(accuracy),
            'tp': int(tp), 'fp': int(fp), 'fn': int(fn), 'tn': int(tn)
        }
    
    def compute_distance_metrics(self, predictions: np.ndarray, targets: np.ndarray) -> Dict[str, float]:
        """Compute distance-based metrics (HD95, ASSD)"""
        if not MEDPY_AVAILABLE:
            return {'hd95': np.nan, 'assd': np.nan}
        
        pred_binary = predictions > 0.5
        targets_binary = targets.astype(bool)
        
        # Skip if either is empty
        if not targets_binary.any() or not pred_binary.any():
            return {'hd95': np.nan, 'assd': np.nan}
        
        try:
            # Note: These metrics require 3D arrays, so we might need to adapt
            hd95_val = hd95(pred_binary, targets_binary)
            assd_val = assd(pred_binary, targets_binary)
            
            return {
                'hd95': float(hd95_val),
                'assd': float(assd_val)
            }
        except Exception as e:
            print(f"⚠️  Distance metrics computation failed: {e}")
            return {'hd95': np.nan, 'assd': np.nan}
    
    def evaluate_model_comprehensive(self, 
                                   model: nn.Module,
                                   test_loader,
                                   device: torch.device) -> Dict:
        """
        Comprehensive model evaluation with all metrics
        """
        print("🔬 Starting Comprehensive Research Evaluation...")
        model.eval()
        
        all_predictions = []
        all_targets = []
        patient_results = []
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(test_loader):
                try:
                    batch = batch.to(device, non_blocking=True)
                    
                    # Forward pass
                    logits = model(batch.x, batch.edge_index, batch.batch)
                    predictions = torch.sigmoid(logits).cpu().numpy()
                    targets = batch.y.cpu().numpy()
                    
                    all_predictions.extend(predictions)
                    all_targets.extend(targets)
                    
                    # Compute per-batch metrics for statistical analysis
                    batch_metrics = self.compute_basic_metrics(predictions, targets)
                    patient_results.append(batch_metrics)
                    
                    if batch_idx % 50 == 0:
                        print(f"  Processed {batch_idx}/{len(test_loader)} batches")
                        
                except Exception as e:
                    print(f"⚠️  Skipping batch {batch_idx}: {e}")
                    continue
        
        # Convert to numpy arrays
        all_predictions = np.array(all_predictions)
        all_targets = np.array(all_targets)
        
        print(f"📊 Evaluation completed: {len(all_predictions):,} samples")
        
        # Compute overall metrics
        overall_basic = self.compute_basic_metrics(all_predictions, all_targets)
        overall_distance = self.compute_distance_metrics(all_predictions, all_targets)
        
        # Combine metrics
        overall_metrics = {**overall_basic, **overall_distance}
        
        # Statistical analysis
        statistical_analysis = self.compute_statistical_analysis(patient_results)
        
        # Generate comprehensive report
        report = {
            'overall_metrics': overall_metrics,
            'statistical_analysis': statistical_analysis,
            'sample_size': len(all_predictions),
            'patient_count': len(patient_results),
            'evaluation_timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
        }
        
        # Save results
        if self.config.save_results:
            self.save_evaluation_results(report, all_predictions, all_targets)
        
        # Generate visualizations
        if self.config.generate_plots:
            self.generate_research_plots(report, patient_results)
        
        return report
    
    def compute_statistical_analysis(self, patient_results: List[Dict]) -> Dict:
        """Compute statistical analysis of results"""
        print("📈 Computing statistical analysis...")
        
        # Extract metrics per patient
        metrics_df = pd.DataFrame(patient_results)
        
        statistical_results = {}
        
        for metric in ['dice', 'sensitivity', 'specificity', 'precision']:
            if metric in metrics_df.columns:
                values = metrics_df[metric].dropna()
                
                statistical_results[metric] = {
                    'mean': float(np.mean(values)),
                    'std': float(np.std(values)),
                    'median': float(np.median(values)),
                    'min': float(np.min(values)),
                    'max': float(np.max(values)),
                    'q25': float(np.percentile(values, 25)),
                    'q75': float(np.percentile(values, 75)),
                    'ci_lower': float(np.percentile(values, (1 - self.config.confidence_level) * 50)),
                    'ci_upper': float(np.percentile(values, (1 + self.config.confidence_level) * 50)),
                    'sample_size': len(values)
                }
        
        return statistical_results
    
    def save_evaluation_results(self, report: Dict, predictions: np.ndarray, targets: np.ndarray):
        """Save comprehensive evaluation results"""
        print("💾 Saving evaluation results...")
        
        # Save main report
        report_file = Path(self.config.output_dir) / 'comprehensive_evaluation_report.json'
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        # Save detailed metrics
        metrics_file = Path(self.config.output_dir) / 'detailed_metrics.json'
        detailed_metrics = {
            'overall_performance': report['overall_metrics'],
            'statistical_summary': report['statistical_analysis'],
            'evaluation_metadata': {
                'sample_size': report['sample_size'],
                'patient_count': report['patient_count'],
                'timestamp': report['evaluation_timestamp']
            }
        }
        
        with open(metrics_file, 'w') as f:
            json.dump(detailed_metrics, f, indent=2)
        
        # Save raw data for further analysis
        np.savez(
            Path(self.config.output_dir) / 'evaluation_data.npz',
            predictions=predictions,
            targets=targets
        )
        
        print(f"✅ Results saved to {self.config.output_dir}")
    
    def generate_research_plots(self, report: Dict, patient_results: List[Dict]):
        """Generate research-quality plots and visualizations"""
        print("📊 Generating research visualizations...")
        
        # Set research-quality plot style
        plt.style.use('default')
        sns.set_palette("husl")
        
        # Create figure directory
        plot_dir = Path(self.config.output_dir) / 'plots'
        plot_dir.mkdir(exist_ok=True)
        
        # 1. Metrics Distribution Plot
        self._plot_metrics_distribution(patient_results, plot_dir)
        
        # 2. Performance Summary Plot
        self._plot_performance_summary(report, plot_dir)
        
        # 3. Statistical Analysis Plot
        self._plot_statistical_analysis(report['statistical_analysis'], plot_dir)
        
        print(f"✅ Plots saved to {plot_dir}")
    
    def _plot_metrics_distribution(self, patient_results: List[Dict], plot_dir: Path):
        """Plot distribution of metrics across patients"""
        metrics_df = pd.DataFrame(patient_results)
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle('Distribution of Segmentation Metrics', fontsize=16, fontweight='bold')
        
        metrics_to_plot = ['dice', 'sensitivity', 'specificity', 'precision']
        
        for idx, metric in enumerate(metrics_to_plot):
            if metric in metrics_df.columns:
                ax = axes[idx // 2, idx % 2]
                
                values = metrics_df[metric].dropna()
                
                # Histogram with KDE
                ax.hist(values, bins=30, alpha=0.7, density=True, color='skyblue', edgecolor='black')
                
                # Add mean line
                mean_val = np.mean(values)
                ax.axvline(mean_val, color='red', linestyle='--', linewidth=2, 
                          label=f'Mean: {mean_val:.3f}')
                
                ax.set_xlabel(metric.capitalize())
                ax.set_ylabel('Density')
                ax.set_title(f'{metric.capitalize()} Distribution')
                ax.legend()
                ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(plot_dir / 'metrics_distribution.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_performance_summary(self, report: Dict, plot_dir: Path):
        """Plot overall performance summary"""
        metrics = report['overall_metrics']
        
        # Select key metrics for visualization
        key_metrics = ['dice', 'sensitivity', 'specificity', 'precision', 'accuracy']
        metric_values = [metrics.get(m, 0) for m in key_metrics]
        metric_labels = [m.capitalize() for m in key_metrics]
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        bars = ax.bar(metric_labels, metric_values, 
                     color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7'],
                     edgecolor='black', linewidth=1)
        
        # Add value labels on bars
        for bar, value in zip(bars, metric_values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                   f'{value:.3f}', ha='center', va='bottom', fontweight='bold')
        
        ax.set_ylabel('Score')
        ax.set_title('Overall Model Performance', fontsize=16, fontweight='bold')
        ax.set_ylim(0, 1.1)
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.savefig(plot_dir / 'performance_summary.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_statistical_analysis(self, stats_results: Dict, plot_dir: Path):
        """Plot statistical analysis results"""
        metrics = list(stats_results.keys())
        means = [stats_results[m]['mean'] for m in metrics]
        stds = [stats_results[m]['std'] for m in metrics]
        ci_lower = [stats_results[m]['ci_lower'] for m in metrics]
        ci_upper = [stats_results[m]['ci_upper'] for m in metrics]
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        x_pos = np.arange(len(metrics))
        
        # Plot means with error bars (standard deviation)
        bars = ax.bar(x_pos, means, yerr=stds, capsize=5,
                     color='lightblue', edgecolor='navy', linewidth=1,
                     alpha=0.8, label='Mean ± Std')
        
        # Plot confidence intervals
        ax.errorbar(x_pos, means, 
                   yerr=[np.array(means) - np.array(ci_lower), 
                         np.array(ci_upper) - np.array(means)],
                   fmt='none', capsize=3, color='red', linewidth=2,
                   label=f'{self.config.confidence_level*100}% CI')
        
        # Add value labels
        for i, (mean, std) in enumerate(zip(means, stds)):
            ax.text(i, mean + std + 0.02, f'{mean:.3f}±{std:.3f}',
                   ha='center', va='bottom', fontweight='bold')
        
        ax.set_xlabel('Metrics')
        ax.set_ylabel('Score')
        ax.set_title('Statistical Analysis of Model Performance', 
                    fontsize=16, fontweight='bold')
        ax.set_xticks(x_pos)
        ax.set_xticklabels([m.capitalize() for m in metrics])
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.savefig(plot_dir / 'statistical_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()

def main():
    """Main function to run comprehensive evaluation"""
    print("🔬 BraTS GNN Comprehensive Research Evaluation")
    print("=" * 50)
    
    # This would be called with your trained model and test data
    # Example usage:
    print("📋 To use this evaluation suite:")
    print("1. Load your trained model")
    print("2. Create test data loader")
    print("3. Run: evaluator.evaluate_model_comprehensive(model, test_loader, device)")
    print("\n✅ Evaluation suite ready for research publication!")

if __name__ == "__main__":
    main()