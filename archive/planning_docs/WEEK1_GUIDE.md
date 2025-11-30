# 📅 WEEK 1 EXECUTION GUIDE

**Date Started:** November 24, 2025  
**Goal:** Set up cross-validation infrastructure and train first fold  
**Estimated Time:** 2-3 hours (mostly automated)

---

## 🎯 What We're Doing This Week

1. **Create 5-fold cross-validation splits** (patient-level, no data leakage)
2. **Train Fold 0** (estimate time for remaining folds)
3. **Verify everything works** before running overnight training

---

## ✅ Prerequisites Checklist

- [ ] Graph files exist in `./data/graphs/`
- [ ] CUDA is available (`nvidia-smi` works)
- [ ] Python environment activated
- [ ] All dependencies installed

### Quick Verification
```bash
# Check GPU
nvidia-smi

# Check graphs
ls -lh data/graphs/ | head

# Check Python
python3 --version  # Should be 3.8+
```

---

## 🚀 STEP-BY-STEP EXECUTION

### Step 0: Verify Current Status (5 minutes)

```bash
cd /mnt/bigdata/capstone/brats_gnn_segmentation

# Check graph files
echo "Counting graph files..."
find data/graphs -name "*.pt" | wc -l

# Check patients
echo "Counting patients..."
ls data/graphs | grep BraTS2021 | wc -l

# Expected: ~107,000 graphs from ~1,251 patients
```

### Step 1: Run Week 1 Script (2-3 hours)

**Option A: Automatic (Recommended)**
```bash
# Run complete Week 1 workflow
./scripts/week1_setup_cv.sh
```

**Option B: Manual Step-by-Step**

If you want to see each step:

```bash
# 1. Create CV folds (30 seconds)
python src/cross_validation.py \
    --graphs_dir ./data/graphs \
    --output_dir ./data/cv_folds \
    --k 5 \
    --seed 42 \
    --val_ratio 0.1

# 2. Train Fold 0 (1.5-2 hours)
python src/train_cv_fold.py \
    --fold_idx 0 \
    --fold_dir ./data/cv_folds \
    --output_dir ./checkpoints/cv_experiments \
    --epochs 50 \
    --batch_size 32 \
    --lr 0.001 \
    --hidden_channels 256 \
    --num_layers 5 \
    --device cuda
```

### Step 2: Monitor Training (while running)

Open a new terminal and monitor:

```bash
# Watch GPU usage
watch -n 1 nvidia-smi

# Or tail the output
tail -f nohup.out  # If running in background
```

### Step 3: Verify Results (5 minutes)

After training completes:

```bash
# Check fold assignments
cat data/cv_folds/fold_0.json | head -50

# Check training results
cat checkpoints/cv_experiments/fold_0/results.json

# Expected output:
# - test_metrics.dice: ~0.98
# - training_time: ~5400 seconds (1.5 hours)
```

---

## 📊 Expected Outcomes

### Fold Assignments Created
```
data/cv_folds/
├── all_folds.json         # Complete fold information
├── fold_0.json            # Fold 0 assignments
├── fold_1.json            # Fold 1 assignments
├── fold_2.json            # Fold 2 assignments
├── fold_3.json            # Fold 3 assignments
└── fold_4.json            # Fold 4 assignments
```

**Each fold should have:**
- ~900 train patients (~77,000 graphs)
- ~100 val patients (~8,500 graphs)
- ~250 test patients (~21,000 graphs)

### Fold 0 Training Complete
```
checkpoints/cv_experiments/fold_0/
├── best_model.pth         # Best model checkpoint
└── results.json           # Training results
```

**Expected metrics:**
- Test Dice: ~0.975-0.990
- Training time: ~1.5-2 hours
- Best validation Dice: ~0.980-0.995

---

## 🔍 Troubleshooting

### Issue 1: "No graph files found"
```bash
# Check if graphs exist
ls -R data/graphs/ | grep "\.pt$" | wc -l

# If zero, you need to run graph construction first
python src/graph_construction.py --input_dir ../BraTS2021_Training_Data --output_dir ./data/graphs
```

### Issue 2: "CUDA out of memory"
```bash
# Reduce batch size
python src/train_cv_fold.py --fold_idx 0 --batch_size 16  # Instead of 32
```

### Issue 3: "Dataset class doesn't accept graph_files"
```bash
# The dataset.py has been updated - make sure you're using the modified version
grep "graph_files" src/dataset.py  # Should show the parameter
```

### Issue 4: Training very slow
```bash
# Check GPU utilization
nvidia-smi

# If <50% utilization, increase batch size or accumulation steps
# If 100% utilization, all good - just wait
```

---

## 📈 Success Criteria

**Week 1 is complete when:**
- ✅ 5 fold assignments created (all_folds.json exists)
- ✅ Fold 0 trained successfully (best_model.pth exists)
- ✅ Test Dice > 0.97 (check results.json)
- ✅ No errors in the output

---

## 🎯 What's Next?

### Immediate Next Step (Week 2)
Once Fold 0 is complete and verified:

```bash
# Train remaining folds (run overnight)
./scripts/week2_train_all_folds.sh

# This will:
# - Train folds 1, 2, 3, 4
# - Take ~6-8 hours total
# - Can run overnight
```

### Running Overnight
```bash
# Option 1: Use nohup
nohup ./scripts/week2_train_all_folds.sh > week2_output.log 2>&1 &

# Option 2: Use screen
screen -S cv_training
./scripts/week2_train_all_folds.sh
# Ctrl+A, D to detach
# screen -r cv_training to reattach

# Option 3: Use tmux
tmux new -s cv_training
./scripts/week2_train_all_folds.sh
# Ctrl+B, D to detach
# tmux attach -t cv_training to reattach
```

---

## 📝 Daily Progress Log

### Day 1: ________ (Date)
- [ ] Verified prerequisites
- [ ] Created CV folds
- [ ] Started Fold 0 training
- **Time started:** _______
- **Notes:** _______________

### Day 2: ________ (Date)
- [ ] Fold 0 training complete
- [ ] Verified results
- [ ] Started remaining folds (overnight)
- **Fold 0 Dice:** _______
- **Training time:** _______

---

## 💡 Tips

1. **Check Fold 0 carefully** - If it fails, don't train remaining folds
2. **Monitor first 10 epochs** - Make sure Dice is increasing
3. **Save output logs** - Use `tee` to save and display: `./script.sh | tee output.log`
4. **Use tmux/screen** - Don't rely on keeping terminal open
5. **Check disk space** - Each fold checkpoint is ~60MB

---

## 🆘 Need Help?

If anything goes wrong:

1. **Check the error message** - Usually tells you what's wrong
2. **Verify file paths** - Use absolute paths if relative paths fail
3. **Check GPU memory** - `nvidia-smi` should show usage
4. **Review this guide** - Troubleshooting section covers common issues

---

## 📊 Quick Reference Commands

```bash
# Check progress
ls checkpoints/cv_experiments/

# View results
cat checkpoints/cv_experiments/fold_0/results.json | grep dice

# Monitor GPU
watch -n 1 nvidia-smi

# Kill training if needed
pkill -f train_cv_fold

# Resume training (if interrupted)
python src/train_cv_fold.py --fold_idx 0 --resume  # (if resume is implemented)
```

---

## ✅ Week 1 Completion Checklist

Before moving to Week 2:

- [ ] Fold assignments created and verified
- [ ] Fold 0 trained successfully
- [ ] Test Dice > 0.97
- [ ] results.json file exists and is valid JSON
- [ ] Understand training time per fold
- [ ] Ready to run overnight training
- [ ] Saved output logs for reference
- [ ] GPU works properly

**If all boxes checked: Ready for Week 2! 🚀**

---

**Last Updated:** November 24, 2025  
**Status:** Week 1 - Ready to Execute
