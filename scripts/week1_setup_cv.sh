#!/bin/bash
# Week 1 Execution Script: Cross-Validation Setup
# This script sets up CV folds and trains Fold 0

set -e  # Exit on error

# Activate virtual environment
source /mnt/bigdata/capstone/.env/bin/activate

echo "================================================================================"
echo "WEEK 1: CROSS-VALIDATION INFRASTRUCTURE"
echo "================================================================================"
echo ""

# Configuration
GRAPHS_DIR="./data/graphs"
CV_FOLDS_DIR="./data/cv_folds"
OUTPUT_DIR="./checkpoints/cv_experiments"
K_FOLDS=5
EPOCHS=50

# Step 1: Create CV Folds
echo "Step 1: Creating cross-validation folds..."
echo "--------------------------------------------------------------------------------"
python3 src/cross_validation.py \
    --graphs_dir "$GRAPHS_DIR" \
    --output_dir "$CV_FOLDS_DIR" \
    --k $K_FOLDS \
    --seed 42 \
    --val_ratio 0.1

echo ""
echo "✅ Fold assignments created!"
echo ""

# Step 2: Train Fold 0 (estimate time for remaining folds)
echo "Step 2: Training Fold 0 (time estimation)..."
echo "--------------------------------------------------------------------------------"
python3 src/train_cv_fold.py \
    --fold_idx 0 \
    --fold_dir "$CV_FOLDS_DIR" \
    --output_dir "$OUTPUT_DIR" \
    --epochs $EPOCHS \
    --batch_size 32 \
    --lr 0.001 \
    --hidden_channels 256 \
    --num_layers 5 \
    --device cuda

echo ""
echo "✅ Fold 0 training complete!"
echo ""

# Display results
echo "================================================================================"
echo "WEEK 1 COMPLETE!"
echo "================================================================================"
echo ""
echo "Created:"
echo "  ✓ $K_FOLDS cross-validation folds"
echo "  ✓ Fold 0 trained model"
echo "  ✓ Fold 0 results and metrics"
echo ""
echo "Next Steps:"
echo "  1. Review fold 0 results: $OUTPUT_DIR/fold_0/results.json"
echo "  2. If satisfied, run week2_train_all_folds.sh to train remaining folds"
echo "  3. Training time estimate per fold: Check fold_0/results.json"
echo ""
echo "To train all folds overnight, run:"
echo "  ./scripts/week2_train_all_folds.sh"
echo ""
echo "================================================================================"
