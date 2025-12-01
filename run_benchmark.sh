#!/bin/bash

# Activate virtual environment
source /mnt/bigdata/capstone/.env/bin/activate

# Run speed benchmark script
echo "=========================================="
echo "Running Speed Benchmark: GNN vs U-Net"
echo "=========================================="

python3 scripts/benchmark_speed.py \
    --num_patients 50 \
    --fold 0 \
    --device cuda \
    --output research_results/speed_benchmark

echo ""
echo "✅ Benchmark complete!"
echo "📁 Check research_results/speed_benchmark/benchmark_results.json"
