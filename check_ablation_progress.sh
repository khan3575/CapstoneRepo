#!/bin/bash
# Quick status check for ablation study

echo "╔══════════════════════════════════════════════════════════════════════╗"
echo "║                   ABLATION STUDY PROGRESS REPORT                     ║"
echo "╚══════════════════════════════════════════════════════════════════════╝"
echo ""

# Check if running
if pgrep -f "run_ablation_study.py" > /dev/null; then
    PID=$(pgrep -f "run_ablation_study.py" | head -1)
    echo "✅ Status: RUNNING (PID: $PID)"
    
    # Runtime
    RUNTIME=$(ps -p $PID -o etime= 2>/dev/null | tr -d ' ')
    echo "⏱️  Runtime: $RUNTIME"
    echo ""
else
    echo "❌ Status: NOT RUNNING"
    echo ""
    echo "Last 10 lines of log:"
    tail -10 /mnt/bigdata/capstone/brats_gnn_segmentation/ablation_study.log
    exit 1
fi

# GPU Status
echo "🎮 GPU Status:"
nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw --format=csv,noheader | \
    awk -F', ' '{printf "   Utilization: %s | Memory: %s / %s | Temp: %s | Power: %s\n", $1, $2, $3, $4, $5}'
echo ""

# Completed configs
echo "📊 Completed Configurations:"
COMPLETED=$(ls -d research_results/ablation_study/*/ 2>/dev/null | wc -l)
echo "   $COMPLETED / 8 configurations completed"

if [ $COMPLETED -gt 0 ]; then
    echo ""
    echo "   Completed:"
    for dir in research_results/ablation_study/*/; do
        if [ -f "$dir/results.json" ]; then
            CONFIG=$(basename "$dir")
            DICE=$(grep '"test_dice"' "$dir/results.json" | awk '{print $2}' | tr -d ',')
            echo "      ✓ $CONFIG: Dice = $DICE"
        fi
    done
fi

# Current progress
echo ""
echo "📈 Current Training Progress:"
LAST_EPOCH=$(grep -E "Epoch [0-9]+/25:" ablation_study.log | tail -1)
if [ -n "$LAST_EPOCH" ]; then
    echo "   $LAST_EPOCH"
else
    echo "   Still in first epoch..."
fi

# Recent activity (last 3 lines)
echo ""
echo "🔍 Recent Activity (last 3 lines):"
tail -3 ablation_study.log | sed 's/^/   /'

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "To monitor in real-time: tail -f ablation_study.log"
echo "Expected completion: ~10-12 hours from start"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
