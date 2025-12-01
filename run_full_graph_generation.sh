#!/bin/bash
# Full Graph Generation Script
# Processes all 1,251 patients with 15 features (no leakage)

set -e  # Exit on error

echo "=========================================================================="
echo "FULL GRAPH GENERATION - Phase 2 (Fixed, No Leakage)"
echo "=========================================================================="
echo ""
echo "Configuration:"
echo "  - Patients: 1,251"
echo "  - Features: 15 per node (no ground-truth leakage)"
echo "  - Superpixels: 200 per slice"
echo "  - Workers: 12 parallel processes"
echo "  - Memory: 24GB total (2GB per worker)"
echo "  - Expected time: 8-12 hours"
echo ""

# Activate virtual environment
source /mnt/bigdata/capstone/.env/bin/activate

# Run graph construction
python3 src/graph_construction.py \
  --input_dir data/preprocessed \
  --output_dir data/graphs \
  --superpixels 200 \
  --max_memory_gb 24.0 \
  --num_workers 12

echo ""
echo "=========================================================================="
echo "GRAPH GENERATION COMPLETE!"
echo "=========================================================================="
