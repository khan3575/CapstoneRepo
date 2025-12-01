#!/bin/bash

# Activate virtual environment
source /mnt/bigdata/capstone/.env/bin/activate

# Run visualization script
echo "=========================================="
echo "Generating Qualitative Visualizations"
echo "=========================================="

python3 scripts/visualize_qualitative.py \
    --fold 0 \
    --num_patients 10 \
    --output_dir visualizations/qualitative \
    --device cuda

echo ""
echo "✅ Visualization complete!"
echo "📁 Check visualizations/qualitative/fold_0/ for images"
echo ""
echo "To view images:"
echo "  ls -lh visualizations/qualitative/fold_0/*/slice_*.png"
echo ""
echo "Total images created:"
find visualizations/qualitative/fold_0/ -name "*.png" 2>/dev/null | wc -l
