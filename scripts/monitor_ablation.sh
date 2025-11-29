#!/bin/bash
# Monitor ablation study progress

LOG_FILE="ablation_study.log"

echo "=========================================="
echo "ABLATION STUDY PROGRESS MONITOR"
echo "=========================================="
echo ""

# Check if process is running
if pgrep -f "run_ablation_study.py" > /dev/null; then
    echo "✅ Ablation study is RUNNING"
    PID=$(pgrep -f "run_ablation_study.py")
    echo "   Process ID: $PID"
else
    echo "❌ Ablation study is NOT running"
fi

echo ""
echo "Log file size: $(du -h $LOG_FILE 2>/dev/null | cut -f1 || echo '0')"
echo ""

# Count completed configurations
COMPLETED=$(grep -c "RESULTS:" $LOG_FILE 2>/dev/null || echo "0")
echo "Completed configurations: $COMPLETED / 8"
echo ""

# Show current configuration
echo "=========================================="
echo "CURRENT PROGRESS:"
echo "=========================================="
tail -50 $LOG_FILE 2>/dev/null | grep -A 5 "ABLATION:\|Epoch\|RESULTS:" | tail -20

echo ""
echo "=========================================="
echo "To monitor in real-time, run:"
echo "  tail -f $LOG_FILE"
echo "=========================================="
