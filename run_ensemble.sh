#!/bin/bash

# Activate virtual environment
source /mnt/bigdata/capstone/.env/bin/activate

# Run ensemble inference (THE +1% BOOSTER)
echo "=========================================="
echo "Ensemble Inference - Combining 5 Folds"
echo "=========================================="

python3 src/inference_ensemble.py \
    --checkpoint_dir checkpoints/binary_training \
    --fold_file data/cv_folds/fold_0.json \
    --method mean \
    --output_dir research_results/ensemble \
    --device cuda

echo ""
echo "✅ Ensemble inference complete!"
echo "📁 Results saved to research_results/ensemble/ensemble_results.json"
