# ✅ WEEK 1 EXECUTION CHECKLIST

**Date:** November 24, 2025  
**Executor:** _____________  
**Status:** 🚧 IN PROGRESS

---

## PRE-EXECUTION (5 minutes)

- [ ] Verified Python 3.12.3 installed
- [ ] Confirmed GPU working (`nvidia-smi`)
- [ ] Checked graph files exist (1,251 files)
- [ ] Scripts are executable (`chmod +x scripts/*.sh`)
- [ ] Read START_HERE.md
- [ ] Understand what will happen

---

## EXECUTION (2-3 hours)

### Part 1: Start Training
- [ ] Opened terminal in project directory
- [ ] Executed: `./scripts/week1_setup_cv.sh 2>&1 | tee week1_execution.log`
- [ ] Saw "Creating cross-validation folds..." message
- [ ] Confirmed 5 folds created successfully

**Fold Creation Output:**
```
Expected:
  Fold 0: 250 patients, ~21000 graphs
  Fold 1: 250 patients, ~21000 graphs
  ...
  
Actual (fill in):
  Fold 0: ___ patients, ____ graphs
  Fold 1: ___ patients, ____ graphs
  Fold 2: ___ patients, ____ graphs
  Fold 3: ___ patients, ____ graphs
  Fold 4: ___ patients, ____ graphs
```

### Part 2: Monitor Training
- [ ] Opened second terminal
- [ ] Ran: `watch -n 1 nvidia-smi`
- [ ] Confirmed GPU utilization >80%
- [ ] Saw training progressing (epochs incrementing)

**Training Progress Notes:**
```
Start time: ________
Estimated end time: ________ (start + 1.5 hours)
GPU utilization: ____%
Memory usage: ____/6GB
```

### Part 3: Wait for Completion
- [ ] Training reached Epoch 50/50
- [ ] Saw "Training complete!" message
- [ ] Saw test results printed
- [ ] No errors in output

---

## POST-EXECUTION (10 minutes)

### Verify Results
- [ ] Checked fold assignments created:
  ```bash
  ls data/cv_folds/
  # Expected: all_folds.json, fold_0.json, ..., fold_4.json
  ```

- [ ] Checked model saved:
  ```bash
  ls -lh checkpoints/cv_experiments/fold_0/
  # Expected: best_model.pth (~60MB), results.json
  ```

- [ ] Read results file:
  ```bash
  cat checkpoints/cv_experiments/fold_0/results.json
  ```

**RESULTS (fill in):**
```json
test_metrics: {
  "dice": ___________,
  "accuracy": ___________,
  "sensitivity": ___________,
  "specificity": ___________
}
training_time: ___________ seconds
```

### Success Criteria
- [ ] Test Dice > 0.97 ✓ / ✗
- [ ] Training time < 10,000 seconds ✓ / ✗
- [ ] No errors in execution ✓ / ✗
- [ ] All files created ✓ / ✗

---

## DECISION POINT

### If All Success Criteria Met: ✅ PROCEED TO WEEK 2
- [ ] Updated PROGRESS_TRACKER.md (Week 1 complete)
- [ ] Saved execution log
- [ ] Planned Week 2 execution time
- [ ] Ready to train remaining folds

**Next Command:**
```bash
# Run overnight or when ready
./scripts/week2_train_all_folds.sh 2>&1 | tee week2_execution.log
```

### If Any Criteria Failed: ⚠️ DEBUG FIRST
**Issue:** _______________________________________

**Error Message:** ____________________________

**Action Taken:** ____________________________

**Resolution:** __________________________________

---

## WEEK 1 SUMMARY

**Completion Date:** ___________  
**Total Time Spent:** ___________ hours  
**Final Dice Score:** ___________  
**Training Time:** ___________ minutes  
**Issues Encountered:** ___________ (0 = none)

**Overall Status:** 
- [ ] ✅ COMPLETE - Ready for Week 2
- [ ] ⚠️ PARTIAL - Needs debugging
- [ ] ❌ FAILED - Need help

---

## WEEK 2 PLANNING

**Scheduled Start Date:** ___________  
**Expected Completion:** ___________ (start + 8 hours)  
**Execution Method:**
- [ ] Run overnight
- [ ] Run during day with monitoring
- [ ] Run in screen/tmux session

**Command to Execute:**
```bash
# Option 1: Foreground with logging
./scripts/week2_train_all_folds.sh 2>&1 | tee week2_execution.log

# Option 2: Background (overnight)
nohup ./scripts/week2_train_all_folds.sh > week2_output.log 2>&1 &

# Option 3: Screen session
screen -S cv_training
./scripts/week2_train_all_folds.sh
# Ctrl+A, D to detach
```

---

## NOTES / OBSERVATIONS

**Positive:**
- ___________________________________________
- ___________________________________________

**Negative:**
- ___________________________________________
- ___________________________________________

**Questions:**
- ___________________________________________
- ___________________________________________

**Lessons Learned:**
- ___________________________________________
- ___________________________________________

---

## SIGN-OFF

**Completed by:** ___________  
**Date:** ___________  
**Time:** ___________  
**Signature:** ___________

**Ready for Week 2:** YES / NO

**If YES:** Execute Week 2 script  
**If NO:** Review issues and debug

---

**Next Review:** After Week 2 completion
