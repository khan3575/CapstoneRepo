#!/usr/bin/env bash
# =============================================================================
# train_all_folds_v5.sh  —  Retrain 5 folds: BCE+Dice + pos_weight=16 + flip aug
# =============================================================================
# Builds on v4. Two additions over v4:
#   1. pos_weight=16 in BCE (tumor nodes = 5.8% of all nodes, ratio ~16x)
#   2. RandomFlipTransform on train set (p=0.5 horizontal flip of norm_x)
#
# SAFE TO RUN: saves to checkpoints/binary_v5/ — does NOT touch binary_v3 or v4.
# Roll back: just use binary_v3 or binary_v4 checkpoints instead.
# See NEXT_STEPS.md -> Branch A for context.
#
# Saves to: checkpoints/binary_v5/fold_X/
# After this completes, run:
#   python src/inference_ensemble.py --checkpoint_dir checkpoints/binary_v5
# =============================================================================

set -e

source /mnt/bigdata/capstone/.env/bin/activate
cd /mnt/bigdata/capstone/brats_gnn_segmentation

OUTPUT_DIR="checkpoints/binary_v5"
BATCH_SIZE=24
EPOCHS=50
ACCUMULATION=1

mkdir -p logs/training_v5

echo "========================================"
echo "v5 Training — BCE+Dice+pos_weight+flip, batch_size=$BATCH_SIZE"
echo "Output: $OUTPUT_DIR"
echo "Started: $(date)"
echo "========================================"

for FOLD in 0 1 2 3 4; do
    # Skip if already done
    RESULT="$OUTPUT_DIR/fold_${FOLD}/results.json"
    if [ -f "$RESULT" ]; then
        DICE=$(python3 -c "import json; d=json.load(open('$RESULT')); print(f\"{d['test_metrics']['dice']:.4f}\")" 2>/dev/null || echo "?")
        echo "[SKIP] Fold $FOLD already done (test_dice=$DICE)"
        continue
    fi

    echo ""
    echo "--- Fold $FOLD starting: $(date) ---"
    python src/train_cv_fold_v5.py \
        --fold_idx $FOLD \
        --output_dir "$OUTPUT_DIR" \
        --batch_size $BATCH_SIZE \
        --epochs $EPOCHS \
        --accumulation_steps $ACCUMULATION \
        2>&1 | tee logs/training_v5/fold_${FOLD}.log
    echo "--- Fold $FOLD finished: $(date) ---"
done

echo ""
echo "========================================"
echo "ALL FOLDS COMPLETE: $(date)"
echo "========================================"

echo ""
echo "Results summary (v5 — BCE+Dice+pos_weight+flip, batch_24):"
python3 -c "
import json
from pathlib import Path

output_dir = Path('$OUTPUT_DIR')
dices = []
for fold in range(5):
    rfile = output_dir / f'fold_{fold}' / 'results.json'
    if rfile.exists():
        d = json.load(open(rfile))
        test = d['test_metrics']['dice']
        val  = d['best_val_dice']
        dices.append(test)
        print(f'  Fold {fold}: val={val:.4f}  test={test:.4f}')
    else:
        print(f'  Fold {fold}: NOT DONE')

if dices:
    import numpy as np
    print(f'  Mean: {np.mean(dices):.4f} +/- {np.std(dices):.4f}')
    print()
    print('Next step:')
    print('  python src/inference_ensemble.py --checkpoint_dir checkpoints/binary_v5')
"
