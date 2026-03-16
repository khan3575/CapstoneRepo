#!/usr/bin/env bash
# =============================================================================
# search_pos_weight.sh — Overnight pos_weight grid search (fold 0 only)
# =============================================================================
# Tests pos_weight values: 4, 6, 8, 10, 12
# Each run: fold 0, 50 epochs, BCE+Dice+flip (same as v5 but lower pos_weight)
# Total time: ~5 x 8 min = ~40 min
#
# After this completes, check results:
#   python scripts/search_pos_weight_summary.py
#
# Then retrain full 5 folds with the best pos_weight:
#   bash scripts/train_all_folds_v6.sh  (update POS_WEIGHT in that script first)
# =============================================================================

set -e

source /mnt/bigdata/capstone/.env/bin/activate
cd /mnt/bigdata/capstone/brats_gnn_segmentation

SEARCH_DIR="checkpoints/pos_weight_search"
mkdir -p logs/pos_weight_search "$SEARCH_DIR"

echo "========================================"
echo "pos_weight grid search — fold 0 only"
echo "Testing: 4 6 8 10 12"
echo "Started: $(date)"
echo "========================================"

for PW in 4 6 8 10 12; do
    RESULT="$SEARCH_DIR/pw_${PW}/fold_0/results.json"
    if [ -f "$RESULT" ]; then
        DICE=$(python3 -c "import json; d=json.load(open('$RESULT')); print(f\"{d['test_metrics']['dice']*100:.2f}%\")" 2>/dev/null)
        echo "[SKIP] pos_weight=$PW already done (test_dice=$DICE)"
        continue
    fi

    echo ""
    echo "--- pos_weight=$PW starting: $(date) ---"
    python src/train_cv_fold_v5.py \
        --fold_idx 0 \
        --output_dir "$SEARCH_DIR/pw_${PW}" \
        --batch_size 24 \
        --epochs 50 \
        --accumulation_steps 1 \
        --pos_weight $PW \
        2>&1 | tee logs/pos_weight_search/pw_${PW}_fold0.log
    echo "--- pos_weight=$PW done: $(date) ---"
done

echo ""
echo "========================================"
echo "SEARCH COMPLETE: $(date)"
echo "========================================"
echo ""

# Print summary
python3 -c "
import json, numpy as np
from pathlib import Path

print('pos_weight search results (fold 0):')
print('-'*55)
print(f'  {\"pos_weight\":<12} {\"Dice\":>8} {\"Sensitivity\":>13} {\"Precision\":>11}')
print('-'*55)

results = []
for pw in [4, 6, 8, 10, 12]:
    f = Path(f'checkpoints/pos_weight_search/pw_{pw}/fold_0/results.json')
    if f.exists():
        m = json.load(open(f))['test_metrics']
        results.append((pw, m['dice']))
        print(f'  pw={pw:<9} {m[\"dice\"]*100:>7.2f}%  {m[\"sensitivity\"]*100:>11.2f}%  {m[\"precision\"]*100:>9.2f}%')

# Reference
for ver, path in [('v3(BCE)', 'binary_v3'), ('v4(BCE+Dice)', 'binary_v4')]:
    f = Path(f'checkpoints/{path}/fold_0/results.json')
    if f.exists():
        m = json.load(open(f))['test_metrics']
        print(f'  {ver:<12} {m[\"dice\"]*100:>7.2f}%  {m[\"sensitivity\"]*100:>11.2f}%  {m[\"precision\"]*100:>9.2f}%  [reference]')

if results:
    best_pw, best_dice = max(results, key=lambda x: x[1])
    print()
    print(f'  BEST: pos_weight={best_pw}  Dice={best_dice*100:.2f}%')
    print()
    print(f'  Next: bash scripts/train_all_folds_v6.sh  (set POS_WEIGHT={best_pw})')
"
