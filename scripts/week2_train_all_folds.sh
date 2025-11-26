#!/bin/bash
# Week 2: Train All Remaining CV Folds
# Run this after Week 1 to train folds 1-4

set -e  # Exit on error

# Activate virtual environment
source /mnt/bigdata/capstone/.env/bin/activate

echo "================================================================================"
echo "WEEK 2: TRAINING ALL CROSS-VALIDATION FOLDS"
echo "================================================================================"
echo ""

# Configuration
CV_FOLDS_DIR="./data/cv_folds"
OUTPUT_DIR="./checkpoints/cv_experiments"
K_FOLDS=5
EPOCHS=50

# Check if fold 0 exists
if [ ! -f "$OUTPUT_DIR/fold_0/results.json" ]; then
    echo "❌ Error: Fold 0 not found. Please run week1_setup_cv.sh first."
    exit 1
fi

echo "Training folds 1-4 (fold 0 already complete)"
echo "This will take approximately 6-8 hours on RTX 2060"
echo ""
read -p "Continue? (y/n) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Cancelled."
    exit 1
fi

# Train folds 1-4
for fold in {1..4}; do
    echo ""
    echo "================================================================================"
    echo "Training Fold $fold / 4"
    echo "================================================================================"
    
    python3 src/train_cv_fold.py \
        --fold_idx $fold \
        --fold_dir "$CV_FOLDS_DIR" \
        --output_dir "$OUTPUT_DIR" \
        --epochs $EPOCHS \
        --batch_size 32 \
        --lr 0.001 \
        --hidden_channels 256 \
        --num_layers 5 \
        --device cuda
    
    echo "✅ Fold $fold complete!"
done

echo ""
echo "================================================================================"
echo "ALL FOLDS TRAINED!"
echo "================================================================================"
echo ""
echo "Next Step: Aggregate results"
echo "  python3 src/aggregate_cv_results.py --cv_dir $OUTPUT_DIR"
echo ""
