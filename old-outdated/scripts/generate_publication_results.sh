#!/bin/bash

# Generate per-region metrics and qualitative visualizations

set -e

cd /mnt/bigdata/capstone/brats_gnn_segmentation
source /mnt/bigdata/capstone/.env/bin/activate

echo "=========================================="
echo "Generating Publication-Ready Results"
echo "=========================================="
echo ""

# Note: Per-region evaluation requires retraining with multi-class setup
# For now, we'll generate qualitative visualizations which are critical

echo "Step 1: Generating Qualitative Visualizations"
echo "----------------------------------------------"

# Generate visualizations for fold 0 (best performing)
echo "Generating visualizations for Fold 0..."
python3 src/generate_qualitative_results.py \
    --fold_idx 0 \
    --fold_dir ./data/cv_folds \
    --data_dir /mnt/bigdata/capstone/BraTS2021_Training_Data \
    --checkpoint_dir ./checkpoints/cv_experiments \
    --output_dir ./research_results/qualitative \
    --n_patients 4 \
    --device cuda

echo ""
echo "✓ Qualitative visualizations complete!"
echo ""

echo "=========================================="
echo "SUMMARY"
echo "=========================================="
echo ""
echo "Generated:"
echo "  - Qualitative visualizations (4 patients, 3 slices each)"
echo "  - Saved to: research_results/qualitative/fold_0/"
echo ""
echo "Note on Per-Region Metrics:"
echo "  Your current model does binary (tumor/non-tumor) classification."
echo "  For WT/TC/ET breakdown, you have two options:"
echo ""
echo "  Option 1 (Quick): Report binary tumor Dice as 'Whole Tumor (WT)'"
echo "            and note that TC/ET require multi-class training"
echo ""
echo "  Option 2 (Proper): Retrain with 4-class output (background, NCR, ED, ET)"
echo "            This requires modifying the model and takes ~24 hours"
echo ""
echo "For arXiv preprint, Option 1 is acceptable with proper documentation."
echo ""

