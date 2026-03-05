#!/usr/bin/env bash
# =============================================================================
# train_all_folds_v3.sh  —  Retrain 5 folds with best ablation config
# =============================================================================
# Key change from v2: batch_size=24 (ablation showed +0.65% over batch=32)
# Saves to: checkpoints/binary_v3/fold_X/
# After this completes, run:
#   python src/inference_ensemble.py --checkpoint_dir checkpoints/binary_v3
# =============================================================================

set -e

source /mnt/bigdata/capstone/.env/bin/activate
cd /mnt/bigdata/capstone/brats_gnn_segmentation

OUTPUT_DIR="checkpoints/binary_v3"
BATCH_SIZE=24
EPOCHS=50
ACCUMULATION=1   # No accumulation needed — batch_24 trains cleanly solo

mkdir -p logs/training_v3

echo "========================================"
echo "v3 Training — batch_size=$BATCH_SIZE"
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
    python src/train_cv_fold.py \
        --fold_idx $FOLD \
        --output_dir "$OUTPUT_DIR" \
        --batch_size $BATCH_SIZE \
        --epochs $EPOCHS \
        --accumulation_steps $ACCUMULATION \
        2>&1 | tee logs/training_v3/fold_${FOLD}.log
    echo "--- Fold $FOLD finished: $(date) ---"
done

echo ""
echo "========================================"
echo "ALL FOLDS COMPLETE: $(date)"
echo "========================================"

echo ""
echo "Results summary (v3 — batch_24):"
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
