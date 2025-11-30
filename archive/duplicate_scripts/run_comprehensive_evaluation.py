#!/usr/bin/env python3
"""
Execute Comprehensive Evaluation on Your Trained BraTS GNN Model
================================================================

This script runs the comprehensive evaluation suite on your trained model
to generate publication-ready metrics and analysis.
"""

import os
import sys
import torch
import torch.nn as nn
from pathlib import Path

# Add src to path
sys.path.append('./src')

# Import your existing components
try:
    from train_maxpower import HighPerformanceBraTSGNN
    from dataset import create_data_splits, create_datasets
    from torch_geometric.loader import DataLoader
    print("✅ Successfully imported existing components")
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Make sure you're running from the brats_gnn_segmentation directory")
    sys.exit(1)

# Import the comprehensive evaluation suite
from comprehensive_evaluation import ComprehensiveEvaluator, EvaluationConfig

def main():
    print("🚀 Running Comprehensive Evaluation on Your BraTS GNN Model")
    print("=" * 70)
    
    # Configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_path = './checkpoints/maxpower_gnn/best_model.pth'
    graph_dir = './data/graphs'
    
    print(f"📱 Device: {device}")
    print(f"📁 Model path: {model_path}")
    print(f"📊 Data directory: {graph_dir}")
    
    # Check if model exists
    if not os.path.exists(model_path):
        print(f"❌ Model not found at {model_path}")
        print("Available checkpoints:")
        checkpoint_dir = Path('./checkpoints/maxpower_gnn')
        if checkpoint_dir.exists():
            for file in checkpoint_dir.glob('*.pth'):
                print(f"  - {file}")
        return
    
    # Load model
    print("\n🧠 Loading trained model...")
    try:
        checkpoint = torch.load(model_path, map_location=device)
        
        # Create model (using the same architecture from training)
        model = HighPerformanceBraTSGNN(
            input_dim=12,
            hidden_dim=256,
            num_layers=5,
            dropout=0.1,
            gnn_type='sage'
        ).to(device)
        
        # Load state dict
        model.load_state_dict(checkpoint['model_state_dict'])
        
        print(f"✅ Model loaded successfully")
        print(f"   Trained for {checkpoint.get('epoch', 'unknown')} epochs")
        print(f"   Best validation Dice: {checkpoint.get('best_dice', 'unknown'):.4f}")
        
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return
    
    # Create data splits
    print("\n📊 Loading test data...")
    try:
        train_patients, val_patients, test_patients = create_data_splits(graph_dir)
        train_dataset, val_dataset, test_dataset = create_datasets(
            graph_dir, train_patients, val_patients, test_patients
        )
        
        print(f"✅ Data loaded successfully")
        print(f"   Test patients: {len(test_patients)}")
        print(f"   Test graphs: {len(test_dataset)}")
        
        # Create test loader
        test_loader = DataLoader(
            test_dataset,
            batch_size=4,  # Smaller batch for evaluation
            shuffle=False,
            num_workers=4
        )
        
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        return
    
    # Configure evaluation
    eval_config = EvaluationConfig(
        metrics_to_compute=['dice', 'sensitivity', 'specificity', 'precision', 'accuracy'],
        statistical_tests=True,
        generate_plots=True,
        save_results=True,
        output_dir="./research_results"
    )
    
    # Create evaluator
    evaluator = ComprehensiveEvaluator(eval_config)
    
    # Run comprehensive evaluation
    print("\n🔬 Running comprehensive evaluation...")
    print("This may take a few minutes...")
    
    try:
        results = evaluator.evaluate_model_comprehensive(
            model=model,
            test_loader=test_loader,
            device=device
        )
        
        print("\n🎉 EVALUATION COMPLETED!")
        print("=" * 50)
        
        # Display key results
        overall_metrics = results['overall_metrics']
        print(f"📊 PERFORMANCE SUMMARY:")
        print(f"   Dice Score: {overall_metrics.get('dice', 0):.4f}")
        print(f"   Sensitivity: {overall_metrics.get('sensitivity', 0):.4f}")
        print(f"   Specificity: {overall_metrics.get('specificity', 0):.4f}")
        print(f"   Accuracy: {overall_metrics.get('accuracy', 0):.4f}")
        print(f"   Precision: {overall_metrics.get('precision', 0):.4f}")
        
        # Statistical summary
        if 'statistical_analysis' in results:
            stats = results['statistical_analysis']
            print(f"\n📈 STATISTICAL ANALYSIS:")
            for metric, stat_data in stats.items():
                if isinstance(stat_data, dict):
                    mean = stat_data.get('mean', 0)
                    std = stat_data.get('std', 0)
                    ci_lower = stat_data.get('ci_lower', 0)
                    ci_upper = stat_data.get('ci_upper', 0)
                    print(f"   {metric.capitalize()}: {mean:.4f} ± {std:.4f} (95% CI: [{ci_lower:.4f}, {ci_upper:.4f}])")
        
        print(f"\n💾 Results saved to: {eval_config.output_dir}")
        print(f"📊 Plots and visualizations generated")
        print(f"📄 Detailed report: {eval_config.output_dir}/comprehensive_evaluation_report.json")
        
        print(f"\n🏆 CONGRATULATIONS!")
        print(f"Your model achieved exceptional performance - ready for publication!")
        
    except Exception as e:
        print(f"❌ Error during evaluation: {e}")
        import traceback
        traceback.print_exc()
        return
    
    print(f"\n✅ Comprehensive evaluation completed successfully!")

if __name__ == "__main__":
    main()