#!/bin/bash

# Script to train CV folds 1, 2, 3, and 4 sequentially
# Each fold takes ~4-5 hours, total ~20 hours

set -e  # Exit on any error

# Activate virtual environment
source /mnt/bigdata/capstone/.env/bin/activate

# Base directory
BASE_DIR="/mnt/bigdata/capstone/brats_gnn_segmentation"
cd "$BASE_DIR"

# Training parameters
FOLD_DIR="./data/cv_folds"
OUTPUT_DIR="./checkpoints/cv_experiments"
EPOCHS=50
BATCH_SIZE=32
LR=0.001
HIDDEN_CHANNELS=256
NUM_LAYERS=5
DEVICE="cuda"

echo "=========================================="
echo "Sequential CV Fold Training (1-4)"
echo "=========================================="
echo "Start time: $(date)"
echo ""

# Train Fold 1
echo "=========================================="
echo "Starting Fold 1 Training..."
echo "=========================================="
python3 src/train_cv_fold.py \
    --fold_idx 1 \
    --fold_dir "$FOLD_DIR" \
    --output_dir "$OUTPUT_DIR" \
    --epochs "$EPOCHS" \
    --batch_size "$BATCH_SIZE" \
    --lr "$LR" \
    --hidden_channels "$HIDDEN_CHANNELS" \
    --num_layers "$NUM_LAYERS" \
    --device "$DEVICE" \
    2>&1 | tee cv_fold1_training.log

echo ""
echo "✓ Fold 1 completed at $(date)"
echo ""

# Train Fold 2
echo "=========================================="
echo "Starting Fold 2 Training..."
echo "=========================================="
python3 src/train_cv_fold.py \
    --fold_idx 2 \
    --fold_dir "$FOLD_DIR" \
    --output_dir "$OUTPUT_DIR" \
    --epochs "$EPOCHS" \
    --batch_size "$BATCH_SIZE" \
    --lr "$LR" \
    --hidden_channels "$HIDDEN_CHANNELS" \
    --num_layers "$NUM_LAYERS" \
    --device "$DEVICE" \
    2>&1 | tee cv_fold2_training.log

echo ""
echo "✓ Fold 2 completed at $(date)"
echo ""

# Train Fold 3
echo "=========================================="
echo "Starting Fold 3 Training..."
echo "=========================================="
python3 src/train_cv_fold.py \
    --fold_idx 3 \
    --fold_dir "$FOLD_DIR" \
    --output_dir "$OUTPUT_DIR" \
    --epochs "$EPOCHS" \
    --batch_size "$BATCH_SIZE" \
    --lr "$LR" \
    --hidden_channels "$HIDDEN_CHANNELS" \
    --num_layers "$NUM_LAYERS" \
    --device "$DEVICE" \
    2>&1 | tee cv_fold3_training.log

echo ""
echo "✓ Fold 3 completed at $(date)"
echo ""

# Train Fold 4
echo "=========================================="
echo "Starting Fold 4 Training..."
echo "=========================================="
python3 src/train_cv_fold.py \
    --fold_idx 4 \
    --fold_dir "$FOLD_DIR" \
    --output_dir "$OUTPUT_DIR" \
    --epochs "$EPOCHS" \
    --batch_size "$BATCH_SIZE" \
    --lr "$LR" \
    --hidden_channels "$HIDDEN_CHANNELS" \
    --num_layers "$NUM_LAYERS" \
    --device "$DEVICE" \
    2>&1 | tee cv_fold4_training.log

echo ""
echo "✓ Fold 4 completed at $(date)"
echo ""

echo "=========================================="
echo "ALL FOLDS COMPLETED!"
echo "=========================================="
echo "End time: $(date)"
echo ""
echo "Next step: Run aggregation script"
echo "  python3 src/aggregate_cv_results.py \\"
echo "    --fold_dir ./data/cv_folds \\"
echo "    --results_dir ./checkpoints/cv_experiments \\"
echo "    --output_dir ./research_results/cv_analysis"
echo ""
