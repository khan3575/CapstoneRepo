#!/bin/bash

# Launch Ablation Study Re-run on CLEAN DATA
# Expected Results: 89-91% Dice (not 97-99%)

echo "========================================"
echo "ABLATION STUDY RE-RUN: CLEAN DATA"
echo "========================================"
echo ""
echo "⚠️  CRITICAL CONTEXT:"
echo "   - OLD results (97-99%) came from LEAKED data (tumor_ratio feature)"
echo "   - NEW ceiling is 90.41% (Fold 0 clean data result)"
echo "   - Expected range: 89-91% for all configs"
echo ""
echo "Configurations to test:"
echo "   1. Baseline (5L, 256D) - Expected: 89.5-90.5%"
echo "   2. 6 Layers           - Expected: 89.5-91.0%"
echo "   3. Hidden 512         - Expected: 89.0-90.5%"
echo "   4. GAT                - Expected: 65-75% (unsuitable)"
echo ""
echo "Settings:"
echo "   - Batch size: 32 (95% GPU util)"
echo "   - Patience: 10"
echo "   - Max epochs: 50"
echo "   - Estimated time: 4-6 hours"
echo ""

read -p "Start re-training now? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]
then
    echo "Launching training..."
    nohup python3 scripts/rerun_undertrained_configs.py > retrain_clean.log 2>&1 &
    PID=$!
    echo $PID > retrain_clean.pid
    
    echo ""
    echo "✅ Training started!"
    echo "   Process ID: $PID"
    echo "   Log file: retrain_clean.log"
    echo "   PID file: retrain_clean.pid"
    echo ""
    echo "Monitor progress:"
    echo "   tail -f retrain_clean.log"
    echo "   nvidia-smi  # Check GPU utilization"
    echo ""
    echo "Check results (after 4-6 hours):"
    echo "   ls research_results/ablation_study_clean/*/results.json"
    echo ""
    echo "Expected output folder: research_results/ablation_study_clean/"
    echo "   (Old contaminated results in: ablation_study_fixed/)"
else
    echo "Cancelled. Run manually with:"
    echo "   nohup python3 scripts/rerun_undertrained_configs.py > retrain_clean.log 2>&1 &"
fi
