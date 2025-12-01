#!/bin/bash
# 5-Fold Cross-Validation Training
# Expected: 2-3 days for all 5 folds

source /mnt/bigdata/capstone/.env/bin/activate

echo "=========================================================================="
echo "BINARY TUMOR SEGMENTATION - 5-Fold Cross-Validation"
echo "=========================================================================="
echo ""
echo "Configuration:"
echo "  - Graphs: data/graphs/ (1,251 patients, 15 features, NO LEAKAGE)"
echo "  - Folds: 5 (training each fold independently)"
echo "  - Expected Dice: 85-92% (realistic, defensible)"
echo "  - Expected time: 2-3 days"
echo ""
echo "Starting training..."
echo ""

# Run training for all folds
for fold in 0 1 2 3 4; do
    echo "=========================================================================="
    echo "Training Fold $fold/4"
    echo "=========================================================================="
    
    python3 src/train_cv_fold.py \
        --fold_idx $fold \
        --fold_dir data/cv_folds \
        --output_dir checkpoints/binary_training \
        --epochs 50 \
        --batch_size 32 \
        --lr 0.001 \
        --hidden_channels 256 \
        --num_layers 5 \
        --device cuda
    
    if [ $? -ne 0 ]; then
        echo "❌ ERROR: Fold $fold failed!"
        exit 1
    fi
    
    echo "✅ Fold $fold complete!"
    echo ""
done

echo "=========================================================================="
echo "ALL FOLDS COMPLETE!"
echo "=========================================================================="
