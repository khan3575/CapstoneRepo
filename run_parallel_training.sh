#!/bin/bash

# Optimized Parallel Training Script
# Trains Folds 1-4 simultaneously with resource partitioning
# RTX 2060 (6GB) + 16 CPU cores + 31GB RAM

source /mnt/bigdata/capstone/.env/bin/activate

echo "=========================================="
echo "PARALLEL TRAINING: Folds 1-4"
echo "=========================================="
echo "Hardware: RTX 2060 (6GB), 16 cores, 31GB RAM"
echo "Strategy: 4 folds in parallel with resource partitioning"
echo ""

# Create log directory
mkdir -p logs

# Resource allocation per fold:
# - GPU: Shared (PyTorch handles this automatically)
# - CPU: 3 cores per fold (4 × 3 = 12 cores, leave 4 for system)
# - Batch size: Reduced to 16 (from 32) to fit 4 models in 6GB VRAM

# Check which folds are already complete
FOLD1_DONE=false
FOLD2_DONE=false
FOLD3_DONE=false
FOLD4_DONE=false

if [ -f "checkpoints/binary_training/fold_1/best_model.pth" ]; then
    echo "✅ Fold 1 already complete - SKIPPING"
    FOLD1_DONE=true
    FOLD1_PID=0
fi

if [ -f "checkpoints/binary_training/fold_2/best_model.pth" ]; then
    echo "✅ Fold 2 already complete - SKIPPING"
    FOLD2_DONE=true
    FOLD2_PID=0
fi

if [ -f "checkpoints/binary_training/fold_3/best_model.pth" ]; then
    echo "✅ Fold 3 already complete - SKIPPING"
    FOLD3_DONE=true
    FOLD3_PID=0
fi

if [ -f "checkpoints/binary_training/fold_4/best_model.pth" ]; then
    echo "✅ Fold 4 already complete - SKIPPING"
    FOLD4_DONE=true
    FOLD4_PID=0
fi

echo ""

# Start Fold 1 if not done
if [ "$FOLD1_DONE" = false ]; then
    echo "Starting Fold 1..."
    CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=3 taskset -c 0-2 \
        python3 src/train_cv_fold.py \
        --fold_idx 1 \
        --epochs 50 \
        --batch_size 16 \
        --lr 0.001 \
        --hidden_channels 256 \
        --num_layers 5 \
        > logs/train_binary_fold1.log 2>&1 &
    FOLD1_PID=$!
    echo "  Fold 1 PID: $FOLD1_PID (CPUs 0-2)"
    sleep 5
fi

# Start Fold 2 if not done
if [ "$FOLD2_DONE" = false ]; then
    echo "Starting Fold 2..."
    CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=3 taskset -c 3-5 \
        python3 src/train_cv_fold.py \
        --fold_idx 2 \
        --epochs 50 \
        --batch_size 16 \
        --lr 0.001 \
        --hidden_channels 256 \
        --num_layers 5 \
        > logs/train_binary_fold2.log 2>&1 &
    FOLD2_PID=$!
    echo "  Fold 2 PID: $FOLD2_PID (CPUs 3-5)"
    sleep 5
fi

# Start Fold 3 if not done
if [ "$FOLD3_DONE" = false ]; then
    echo "Starting Fold 3..."
    CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=3 taskset -c 6-8 \
        python3 src/train_cv_fold.py \
        --fold_idx 3 \
        --epochs 50 \
        --batch_size 16 \
        --lr 0.001 \
        --hidden_channels 256 \
        --num_layers 5 \
        > logs/train_binary_fold3.log 2>&1 &
    FOLD3_PID=$!
    echo "  Fold 3 PID: $FOLD3_PID (CPUs 6-8)"
    sleep 5
fi

# Start Fold 4 if not done
if [ "$FOLD4_DONE" = false ]; then
    echo "Starting Fold 4..."
    CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=3 taskset -c 9-11 \
        python3 src/train_cv_fold.py \
        --fold_idx 4 \
        --epochs 50 \
        --batch_size 16 \
        --lr 0.001 \
        --hidden_channels 256 \
        --num_layers 5 \
        > logs/train_binary_fold4.log 2>&1 &
    FOLD4_PID=$!
    echo "  Fold 4 PID: $FOLD4_PID (CPUs 9-11)"
    sleep 5
fi

echo ""
echo "=========================================="
echo "All 4 folds started!"
echo "=========================================="
echo "PIDs: $FOLD1_PID, $FOLD2_PID, $FOLD3_PID, $FOLD4_PID"
echo ""
echo "Monitor progress:"
echo "  watch -n 10 'nvidia-smi; echo; tail -5 logs/train_binary_fold*.log'"
echo ""
echo "Check specific fold:"
echo "  tail -f logs/train_binary_fold1.log"
echo ""
echo "Expected completion: ~6-8 hours (vs 20 hours sequential)"
echo ""

# Wait for all folds to complete
echo "Waiting for incomplete folds to finish..."

if [ "$FOLD1_DONE" = false ] && [ $FOLD1_PID -gt 0 ]; then
    wait $FOLD1_PID
    echo "✅ Fold 1 complete"
fi

if [ "$FOLD2_DONE" = false ] && [ $FOLD2_PID -gt 0 ]; then
    wait $FOLD2_PID
    echo "✅ Fold 2 complete"
fi

if [ "$FOLD3_DONE" = false ] && [ $FOLD3_PID -gt 0 ]; then
    wait $FOLD3_PID
    echo "✅ Fold 3 complete"
fi

if [ "$FOLD4_DONE" = false ] && [ $FOLD4_PID -gt 0 ]; then
    wait $FOLD4_PID
    echo "✅ Fold 4 complete"
fi

echo ""
echo "=========================================="
echo "ALL TRAINING COMPLETE!"
echo "=========================================="
echo "Time to run ensemble: ./run_ensemble.sh"
