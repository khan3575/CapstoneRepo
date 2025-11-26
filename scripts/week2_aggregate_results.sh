#!/bin/bash
# Week 2: Aggregate Cross-Validation Results
# Run this after all folds are trained

set -e  # Exit on error

# Activate virtual environment
source /mnt/bigdata/capstone/.env/bin/activate

echo "================================================================================"
echo "AGGREGATING CROSS-VALIDATION RESULTS"
echo "================================================================================"
echo ""

# Configuration
CV_DIR="./checkpoints/cv_experiments"
OUTPUT_DIR="$CV_DIR/aggregated"
K_FOLDS=5

# Check if all folds exist
missing_folds=0
for fold in {0..4}; do
    if [ ! -f "$CV_DIR/fold_$fold/results.json" ]; then
        echo "❌ Warning: Fold $fold results not found"
        missing_folds=$((missing_folds + 1))
    fi
done

if [ $missing_folds -gt 0 ]; then
    echo ""
    echo "❌ Error: $missing_folds fold(s) missing. Please train all folds first."
    exit 1
fi

echo "All $K_FOLDS folds found. Aggregating results..."
echo ""

# Aggregate results
python3 src/aggregate_cv_results.py \
    --cv_dir "$CV_DIR" \
    --output_dir "$OUTPUT_DIR" \
    --k $K_FOLDS

echo ""
echo "================================================================================"
echo "AGGREGATION COMPLETE!"
echo "================================================================================"
echo ""
echo "Generated files:"
echo "  ✓ $OUTPUT_DIR/aggregated_results.json"
echo "  ✓ $OUTPUT_DIR/cv_report.md"
echo "  ✓ $OUTPUT_DIR/cv_boxplots.png"
echo "  ✓ $OUTPUT_DIR/cv_dice_per_fold.png"
echo "  ✓ $OUTPUT_DIR/cv_training_curves.png"
echo ""
echo "Review the results:"
echo "  cat $OUTPUT_DIR/cv_report.md"
echo ""
