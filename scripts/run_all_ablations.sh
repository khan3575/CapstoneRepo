#!/usr/bin/env bash
# =============================================================================
# run_all_ablations.sh  —  Run ALL ablation variants sequentially (safe restart)
# =============================================================================
# - Skips variants already completed (30 epochs, valid val_dice in results.json)
# - Cleans up partial checkpoint dirs before re-running (no stale state)
# - Does NOT abort on individual variant failure (continues to next)
# - Prints a summary table at the end
#
# SKIPPED (require pre-built graphs that don't exist):
#   superpixels_50/80/100/150/200, slices_100/150, window_1/window_5
#
# Usage:
#   bash scripts/run_all_ablations.sh              # skip completed variants
#   bash scripts/run_all_ablations.sh --force      # re-run everything
# =============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
ABLATION_SCRIPT="$SCRIPT_DIR/train_ablation.py"
CHECKPOINT_DIR="$PROJECT_DIR/checkpoints/ablation"

FORCE=0
if [[ "${1}" == "--force" ]]; then
    FORCE=1
fi

cd "$PROJECT_DIR"
source /mnt/bigdata/capstone/.env/bin/activate 2>/dev/null || true

# Variants to run (architecture + batch variants; superpixel/slice/window need pre-built graphs)
VARIANTS=(
    baseline
    layers_3
    layers_4
    deeper_network
    hidden_128
    wider_network
    gcn_architecture
    gat_architecture
    gin_architecture
    graph_transformer
    batch_16
    batch_24
    batch_48
    batch_64
)

# =============================================================================
# Helper: check if a variant is cleanly completed
# =============================================================================
is_done() {
    local variant=$1
    local results_file="$CHECKPOINT_DIR/$variant/results.json"
    [[ -f "$results_file" ]] || return 1

    python3 - "$results_file" <<'EOF'
import sys, json
try:
    d = json.load(open(sys.argv[1]))
    epochs = d.get("epochs_trained", 0)
    val    = d.get("best_val_dice", 0.0)
    # Done = 30 epochs and meaningful val_dice
    ok = (int(epochs) == 30 and float(val) > 0.1)
    sys.exit(0 if ok else 1)
except Exception:
    sys.exit(1)
EOF
}

# =============================================================================
# Dry-run summary
# =============================================================================
echo "============================================================"
echo "  ABLATION STUDY — ALL VARIANTS"
echo "============================================================"
echo "Project:    $PROJECT_DIR"
echo "Force:      $FORCE"
echo ""
echo "Plan:"
for variant in "${VARIANTS[@]}"; do
    if [[ $FORCE -eq 0 ]] && is_done "$variant"; then
        echo "  [SKIP] $variant  (already complete)"
    elif [[ -d "$CHECKPOINT_DIR/$variant" ]] && ! is_done "$variant"; then
        echo "  [CLEAN+RUN] $variant  (partial/failed — will clean up first)"
    else
        echo "  [RUN] $variant"
    fi
done
echo ""

# =============================================================================
# Main loop
# =============================================================================
FAILED=()
START_TIME=$(date +%s)

for variant in "${VARIANTS[@]}"; do
    # Skip if already done (unless --force)
    if [[ $FORCE -eq 0 ]] && is_done "$variant"; then
        results_file="$CHECKPOINT_DIR/$variant/results.json"
        val=$(python3 -c "import json; d=json.load(open('$results_file')); print(f\"{d.get('best_val_dice',0):.4f}\")" 2>/dev/null || echo "?")
        echo "[SKIP] $variant  (val_dice=$val)"
        continue
    fi

    # Clean up any partial/stale checkpoint dir before re-running
    if [[ -d "$CHECKPOINT_DIR/$variant" ]]; then
        echo "[CLEAN] Removing partial checkpoint: $CHECKPOINT_DIR/$variant"
        rm -rf "$CHECKPOINT_DIR/$variant"
    fi

    echo ""
    echo "------------------------------------------------------------"
    echo "[RUN] $variant  $(date '+%H:%M:%S')"
    echo "------------------------------------------------------------"

    if python3 "$ABLATION_SCRIPT" --variant "$variant"; then
        echo "[OK] $variant  $(date '+%H:%M:%S')"
    else
        echo "[FAIL] $variant exited with error — continuing to next variant"
        FAILED+=("$variant")
    fi
done

END_TIME=$(date +%s)
ELAPSED=$(( (END_TIME - START_TIME) / 60 ))

# =============================================================================
# Summary table
# =============================================================================
echo ""
echo "============================================================"
echo "  ABLATION RESULTS SUMMARY"
echo "============================================================"

python3 - <<'EOF'
import json
from pathlib import Path

checkpoint_dir = Path("checkpoints/ablation")
variants = [
    "baseline", "layers_3", "layers_4", "deeper_network",
    "hidden_128", "wider_network",
    "gcn_architecture", "gat_architecture", "gin_architecture", "graph_transformer",
    "batch_16", "batch_24", "batch_48", "batch_64"
]

rows = []
for v in variants:
    rfile = checkpoint_dir / v / "results.json"
    if rfile.exists():
        try:
            d = json.load(open(rfile))
            val  = d.get("best_val_dice", 0.0)
            tm   = d.get("test_metrics", {})
            test = tm.get("dice", d.get("test_dice", 0.0))
            ep   = d.get("epochs_trained", "?")
            rows.append((v, val, test, ep, True))
        except Exception as e:
            rows.append((v, 0.0, 0.0, "ERR", False))
    else:
        rows.append((v, 0.0, 0.0, "N/A", False))

# Sort by test dice descending
rows.sort(key=lambda r: r[2] if r[4] else -1, reverse=True)

print(f"{'Variant':<22} {'Val Dice':>10} {'Test Dice':>10} {'Epochs':>8}")
print(f"{'-'*22} {'-'*10} {'-'*10} {'-'*8}")
for v, val, test, ep, ok in rows:
    if ok and float(val) > 0.01:
        print(f"{v:<22} {val:>10.4f} {test:>10.4f} {ep:>8}")
    else:
        print(f"{v:<22} {'N/A':>10} {'N/A':>10} {str(ep):>8}")
EOF

echo ""
if [[ ${#FAILED[@]} -gt 0 ]]; then
    echo "Failed variants: ${FAILED[*]}"
fi
echo "Total time: ${ELAPSED} min"
echo "============================================================"
