#!/usr/bin/env bash
# =============================================================================
# train_all_folds_v4.sh  —  Retrain 5 folds with BCE + DiceLoss (Option 2)
# =============================================================================
# Key change from v3: uses train_cv_fold_v4.py which combines BCE + DiceLoss
# (equal weight 0.5 each). Directly optimises Dice metric; robust to class imbalance.
#
# SAFE TO RUN: saves to checkpoints/binary_v4/ — does NOT touch binary_v3.
# Roll back: just use binary_v3 checkpoints instead.
# See FIX_LOG.md → DECISION-1 for context.
#
# Saves to: checkpoints/binary_v4/fold_X/
# After this completes, run:
#   python src/inference_ensemble.py --checkpoint_dir checkpoints/binary_v4
# =============================================================================

set -e

source /mnt/bigdata/capstone/.env/bin/activate
cd /mnt/bigdata/capstone/brats_gnn_segmentation

OUTPUT_DIR="checkpoints/binary_v4"
BATCH_SIZE=24
EPOCHS=50
ACCUMULATION=1

mkdir -p logs/training_v4

echo "========================================"
echo "v4 Training — BCE + DiceLoss, batch_size=$BATCH_SIZE"
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
    python src/train_cv_fold_v4.py \
        --fold_idx $FOLD \
        --output_dir "$OUTPUT_DIR" \
        --batch_size $BATCH_SIZE \
        --epochs $EPOCHS \
        --accumulation_steps $ACCUMULATION \
        2>&1 | tee logs/training_v4/fold_${FOLD}.log
    echo "--- Fold $FOLD finished: $(date) ---"
done

echo ""
echo "========================================"
echo "ALL FOLDS COMPLETE: $(date)"
echo "========================================"

echo ""
echo "Results summary (v4 — BCE+Dice, batch_24):"
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
    print('  python src/inference_ensemble.py --checkpoint_dir $OUTPUT_DIR')
"
