#!/bin/bash
# Train all 5 folds sequentially — unattended overnight run
# Usage: bash train_all_folds.sh
# Logs saved to: logs/training/fold_X.log

set -e  # Stop if any fold crashes

source /mnt/bigdata/capstone/.env/bin/activate
cd /mnt/bigdata/capstone/brats_gnn_segmentation

mkdir -p logs/training

echo "========================================"
echo "Starting 5-fold training: $(date)"
echo "Logs: logs/training/fold_X.log"
echo "========================================"

for FOLD in 0 1 2 3 4; do
    echo ""
    echo "--- FOLD $FOLD starting: $(date) ---"
    python src/train_cv_fold.py --fold_idx $FOLD 2>&1 | tee logs/training/fold_${FOLD}.log
    echo "--- FOLD $FOLD finished: $(date) ---"
done

echo ""
echo "========================================"
echo "ALL FOLDS COMPLETE: $(date)"
echo "========================================"

# Print final Dice from each fold's results.json
echo ""
echo "Results summary:"
for FOLD in 0 1 2 3 4; do
    RESULT=checkpoints/binary_v2/fold_${FOLD}/results.json
    if [ -f "$RESULT" ]; then
        DICE=$(python -c "import json; d=json.load(open('$RESULT')); print(f\"Fold $FOLD: {d['test_metrics']['dice']:.4f}\")")
        echo "  $DICE"
    fi
done
