#!/bin/bash
# Train U-Net baseline on all 5 CV folds sequentially
# This ensures fair comparison with the GNN model using the same data splits

set -e  # Exit on error

# Get script directory and project root
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$( cd "$SCRIPT_DIR/.." && pwd )"

# Change to project root
cd "$PROJECT_ROOT"

# Activate virtual environment if it exists
if [ -d ".env" ]; then
    echo "Activating virtual environment..."
    source .env/bin/activate
elif [ -d "../.env" ]; then
    echo "Activating virtual environment from parent..."
    source ../.env/bin/activate
fi

echo "================================================================================"
echo "U-NET BASELINE TRAINING - ALL 5 FOLDS"
echo "================================================================================"
echo ""
echo "Working directory: $(pwd)"
echo "Starting time: $(date)"
echo ""

# Configuration
DATA_DIR="data/preprocessed"
FOLD_FILE="data/cv_folds/all_folds.json"
OUTPUT_DIR="checkpoints/unet_baseline"
EPOCHS=50
BATCH_SIZE=4
LR=0.001
ACCUM_STEPS=2

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Log file
LOG_FILE="$OUTPUT_DIR/training_all_folds.log"
echo "Training started at $(date)" > "$LOG_FILE"

# Train each fold
for FOLD in 0 1 2 3 4; do
    echo ""
    echo "================================================================================"
    echo "TRAINING FOLD $FOLD / 4"
    echo "================================================================================"
    echo ""
    
    FOLD_START=$(date +%s)
    
    python3 scripts/train_unet_baseline.py \
        --data_dir "$DATA_DIR" \
        --fold_file "$FOLD_FILE" \
        --output_dir "$OUTPUT_DIR" \
        --fold $FOLD \
        --epochs $EPOCHS \
        --batch_size $BATCH_SIZE \
        --lr $LR \
        --accumulation_steps $ACCUM_STEPS \
        2>&1 | tee -a "$LOG_FILE"
    
    FOLD_END=$(date +%s)
    FOLD_TIME=$((FOLD_END - FOLD_START))
    FOLD_MINUTES=$((FOLD_TIME / 60))
    
    echo "" | tee -a "$LOG_FILE"
    echo "✅ Fold $FOLD completed in $FOLD_MINUTES minutes" | tee -a "$LOG_FILE"
    echo "" | tee -a "$LOG_FILE"
done

echo ""
echo "================================================================================"
echo "ALL FOLDS COMPLETED!"
echo "================================================================================"
echo ""
echo "Ending time: $(date)"

# Aggregate results
echo "📊 Aggregating results across all folds..."
python3 << 'EOF'
import json
import numpy as np
from pathlib import Path

output_dir = Path("checkpoints/unet_baseline")
all_results = []

for fold_idx in range(5):
    results_file = output_dir / f"fold_{fold_idx}" / "results.json"
    if results_file.exists():
        with open(results_file) as f:
            data = json.load(f)
        all_results.append(data)

if len(all_results) == 5:
    test_dices = [r['test_metrics']['dice'] for r in all_results]
    val_dices = [r['best_val_dice'] for r in all_results]
    training_times = [r['training_time_seconds'] / 60 for r in all_results]
    
    print("\n" + "="*80)
    print("U-NET BASELINE - 5-FOLD CROSS-VALIDATION RESULTS")
    print("="*80)
    print(f"\nTest Dice Scores:")
    for i, dice in enumerate(test_dices):
        print(f"  Fold {i}: {dice:.4f}")
    print(f"\nMean ± Std: {np.mean(test_dices):.4f} ± {np.std(test_dices):.4f}")
    print(f"\nValidation Dice (best): {np.mean(val_dices):.4f} ± {np.std(val_dices):.4f}")
    print(f"Training time per fold: {np.mean(training_times):.1f} ± {np.std(training_times):.1f} minutes")
    print(f"Total training time: {sum(training_times):.1f} minutes ({sum(training_times)/60:.1f} hours)")
    
    # Save aggregate results
    aggregate = {
        'model': 'UNet3D',
        'num_folds': 5,
        'test_metrics': {
            'mean_dice': float(np.mean(test_dices)),
            'std_dice': float(np.std(test_dices)),
            'fold_dices': [float(d) for d in test_dices]
        },
        'val_metrics': {
            'mean_dice': float(np.mean(val_dices)),
            'std_dice': float(np.std(val_dices))
        },
        'training_time': {
            'mean_minutes': float(np.mean(training_times)),
            'std_minutes': float(np.std(training_times)),
            'total_minutes': float(sum(training_times)),
            'total_hours': float(sum(training_times) / 60)
        },
        'fold_results': all_results
    }
    
    output_path = output_dir / "aggregate_results.json"
    with open(output_path, 'w') as f:
        json.dump(aggregate, f, indent=2)
    
    print(f"\n✅ Aggregate results saved to {output_path}")
    print("\n" + "="*80)
    
    # Compare with GNN
    print("\nCOMPARISON WITH GNN MODEL:")
    print("="*80)
    
    gnn_results_file = Path("research_results/cv_analysis/cv_report.md")
    if gnn_results_file.exists():
        print(f"\nGNN Model:  98.80% ± 0.38% Dice")
        print(f"U-Net:      {np.mean(test_dices)*100:.2f}% ± {np.std(test_dices)*100:.2f}% Dice")
        diff = (np.mean(test_dices) - 0.9880) * 100
        if diff > 0:
            print(f"Difference: U-Net is {diff:.2f}% better")
        else:
            print(f"Difference: GNN is {-diff:.2f}% better")
    else:
        print(f"\nU-Net:      {np.mean(test_dices)*100:.2f}% ± {np.std(test_dices)*100:.2f}% Dice")
    
    print("\n" + "="*80)
else:
    print(f"⚠️  Only {len(all_results)} folds completed")
EOF

echo ""
echo "🎉 U-Net baseline training complete!"
echo "Results saved in: $OUTPUT_DIR"
echo ""
