# Cleanup & U-Net Retraining Plan
**Created:** 2026-03-31
**Status:** PHASE 1 COMPLETE, PHASE 2 NEXT

---

## PHASE 1: Repo Cleanup (Remove AI-Confusing Files)

### Move to `archive/` (stale docs, old trackers, audit artifacts):
- [x] `FIX_LOG.md` — audit fix log from March 14, done
- [x] `FORENSIC_AUDIT_2.md` — forensic audit from March 15, done
- [x] `NEXT_STEPS.md` — old next-steps, outdated
- [x] `TASK_TRACKER.md` — old task tracker, outdated
- [x] `PROJECT_CONTEXT.md` — old AI context doc, outdated
- [x] `PROJECT_INTEGRITY_REPORT.md` — old audit report, done
- [x] `PIPELINE_DOCUMENTATION.md` — old pipeline doc (info is in paper now)
- [x] `SUPERVISOR_PRESENTATION_TABLES.md` — presentation artifact
- [x] `results.doc` — old results doc
- [x] `result tables.pdf` — old results PDF
- [x] `texput.log` — LaTeX junk
- [x] `verify_timing.py` — one-off verification script
- [x] `train_all_folds.sh` — old top-level training script
- [x] `deprecated/` — already marked deprecated
- [x] `old-outdated/` — already marked old
- [x] `logs/PHASE1_VERIFICATION_REPORT.md` — old audit artifact
- [x] `logs/ablation_final_v3.log` — old log
- [x] `logs/resource_plan_full.json` — old resource plan
- [x] `logs/pos_weight_search/` — old search logs
- [x] `logs/training/` — old training logs (v3 is current)
- [x] `logs/training_v4/` — superseded by v5 or v3
- [x] `logs/training_v5/` — superseded
- [x] `checkpoints/binary_v4/` — superseded by binary_v3 (final model)
- [x] `checkpoints/binary_v5/` — superseded
- [x] `checkpoints/pos_weight_search/` — old search
- [x] `src/train_cv_fold_v4.py` — old version
- [x] `src/train_cv_fold_v5.py` — old version
- [x] `scripts/train_all_folds_v4.sh` — old version
- [x] `scripts/train_all_folds_v5.sh` — old version

### Keep (active/needed):
- `config.yaml` — current config
- `src/` (core files: config.py, dataset.py, gnn_model.py, graph_construction.py, etc.)
- `src/train_cv_fold.py` — current training script
- `scripts/train_unet_baseline.py` — NEEDS UPDATE for retraining
- `scripts/train_all_folds_v3.sh` — current fold runner
- `scripts/benchmark_two_scenarios.py` — needed for benchmarking
- `scripts/evaluate_brats2023.py` — needed
- `scripts/train_ablation.py` — needed
- `checkpoints/binary_v3/` — FINAL GNN model (5 folds)
- `checkpoints/ablation/` — ablation results
- `logs/training_v3/` — current training logs
- `research_results/` — final results (figures, benchmarks, etc.)
- `paperWriting/` — the paper
- `data/` — datasets (don't touch)
- `README.md` — keep but could update later
- `requirements.txt` / `requirements-minimal.txt` — keep
- `.gitignore` — keep

### Add to .gitignore:
- `archive/`

---

## PHASE 2: U-Net Retraining

### Config: base_channels=56, num_levels=4 → 69.1M params
### Verified: fits on RTX 2060 at 2.0GB peak VRAM with AMP

### Changes to train_unet_baseline.py:
1. Update default `base_channels=56`, `num_levels=4`
2. Add AMP (autocast + GradScaler)
3. Add data augmentation (random flips, intensity jitter)
4. Add sliding-window inference for evaluation
5. Add positive class weight to BCE component
6. Keep: BCE + Dice loss, AdamW, OneCycleLR, 50 epochs, early stopping

### Training:
- [ ] Run 5-fold training (est. 30-40 hours total)
- [ ] Evaluate on held-out 251 patients
- [ ] Run BraTS 2023 evaluation
- [ ] Run benchmarks (speed, memory, params)

---

## PHASE 3: Paper Updates (After Retraining)

- [ ] Update all U-Net numbers (Dice, params confirmed, speed, memory)
- [ ] Update comparison tables
- [ ] Regenerate figures
- [ ] Add Wilcoxon test results
- [ ] Report median Dice alongside mean
- [ ] Reframe as efficiency paper
- [ ] Cut verbose sections (~20%)
- [ ] Fix Tumour/Tumor inconsistency
