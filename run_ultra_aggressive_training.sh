#!/bin/bash

# ULTRA-AGGRESSIVE Parallel Training
# Maximum resource utilization (use with caution)
# May cause system slowdown but fastest completion

source /mnt/bigdata/capstone/.env/bin/activate

echo "=========================================="
echo "⚡ ULTRA-AGGRESSIVE TRAINING ⚡"
echo "=========================================="
echo "WARNING: This will max out your GPU and CPU"
echo "System may be slow during training"
echo "Press Ctrl+C within 5 seconds to cancel..."
sleep 5

mkdir -p logs

# Check which folds are already complete
FOLD1_DONE=false
FOLD2_DONE=false
FOLD3_DONE=false
FOLD4_DONE=false

echo "Checking for completed folds..."
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
echo "Starting ULTRA-AGGRESSIVE parallel training..."
echo ""

# Fold 1
if [ "$FOLD1_DONE" = false ]; then
    CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=4 \
        python3 src/train_cv_fold.py \
        --fold_idx 1 \
        --epochs 50 \
        --batch_size 20 \
        --lr 0.001 \
        --hidden_channels 256 \
        --num_layers 5 \
         \
        --num_workers 4 \
        > logs/train_binary_fold1_ultra.log 2>&1 &
    FOLD1_PID=$!
    echo "Fold 1 started (PID: $FOLD1_PID)"
    sleep 3
fi

# Fold 2
if [ "$FOLD2_DONE" = false ]; then
    CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=4 \
        python3 src/train_cv_fold.py \
        --fold_idx 2 \
        --epochs 50 \
        --batch_size 20 \
        --lr 0.001 \
        --hidden_channels 256 \
        --num_layers 5 \
         \
        --num_workers 4 \
        > logs/train_binary_fold2_ultra.log 2>&1 &
    FOLD2_PID=$!
    echo "Fold 2 started (PID: $FOLD2_PID)"
    sleep 3
fi

# Fold 3
if [ "$FOLD3_DONE" = false ]; then
    CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=4 \
        python3 src/train_cv_fold.py \
        --fold_idx 3 \
        --epochs 50 \
        --batch_size 20 \
        --lr 0.001 \
        --hidden_channels 256 \
        --num_layers 5 \
         \
        --num_workers 4 \
        > logs/train_binary_fold3_ultra.log 2>&1 &
    FOLD3_PID=$!
    echo "Fold 3 started (PID: $FOLD3_PID)"
    sleep 3
fi

# Fold 4
if [ "$FOLD4_DONE" = false ]; then
    CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=4 \
        python3 src/train_cv_fold.py \
        --fold_idx 4 \
        --epochs 50 \
        --batch_size 20 \
        --lr 0.001 \
        --hidden_channels 256 \
        --num_layers 5 \
         \
        --num_workers 4 \
        > logs/train_binary_fold4_ultra.log 2>&1 &
    FOLD4_PID=$!
    echo "Fold 4 started (PID: $FOLD4_PID)"
    sleep 3
fi

echo ""
echo "=========================================="
echo "🔥 ALL 4 FOLDS RUNNING AT MAX SPEED 🔥"
echo "=========================================="
echo ""
echo "Monitor GPU:"
echo "  watch -n 5 nvidia-smi"
echo ""
echo "Monitor training:"
echo "  tail -f logs/train_binary_fold1_ultra.log"
echo ""
echo "Expected completion: ~5-7 hours (fastest possible)"
echo ""

# Create monitoring script
cat > monitor_training.sh << 'EOF'
#!/bin/bash
while true; do
    clear
    echo "=========================================="
    echo "Training Progress Monitor"
    echo "$(date)"
    echo "=========================================="
    echo ""
    
    # GPU Status
    nvidia-smi --query-gpu=utilization.gpu,utilization.memory,memory.used,memory.total --format=csv,noheader,nounits | \
        awk '{printf "GPU: %d%% utilization, Memory: %dMB / %dMB (%d%%)\n", $1, $3, $4, $2}'
    
    echo ""
    echo "Fold Progress:"
    echo "--------------"
    
    # Check each fold
    for fold in 1 2 3 4; do
        log_file="logs/train_binary_fold${fold}_ultra.log"
        if [ -f "$log_file" ]; then
            # Get last epoch line
            last_line=$(grep "Epoch" "$log_file" | tail -1)
            if [ ! -z "$last_line" ]; then
                echo "Fold $fold: $last_line"
            else
                echo "Fold $fold: Starting..."
            fi
        else
            echo "Fold $fold: Not started"
        fi
    done
    
    echo ""
    echo "Press Ctrl+C to exit monitoring"
    sleep 10
done
EOF
chmod +x monitor_training.sh

echo "Live monitoring available:"
echo "  ./monitor_training.sh"
echo ""

# Wait for completion
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
echo "✅ ALL TRAINING COMPLETE!"
echo "=========================================="
