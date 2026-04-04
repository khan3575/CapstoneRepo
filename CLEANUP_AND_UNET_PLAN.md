# Cleanup & U-Net Retraining Plan
**Created:** 2026-03-31
**Status:** PHASE 2 SCRIPT READY — NEEDS TO RUN TRAINING

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

### Changes to train_unet_baseline.py: DONE (2026-03-31)
1. [x] base_channels=56, num_levels=4 → 69.1M params
2. [x] AMP (autocast + GradScaler) → 2.1GB peak training VRAM
3. [x] Data augmentation (random flips + intensity jitter)
4. [x] Sliding-window inference (overlap=0.5) for patient-level eval
5. [x] pos_weight=9.0 on BCE component
6. [x] InstanceNorm3d + LeakyReLU (more standard for medical)
7. [x] Per-patient metrics (Dice, accuracy, sensitivity, specificity, precision)
8. [x] Smoke tested: training loop, sliding-window, all verified

### To run training:
```bash
source /mnt/bigdata/capstone/.env/bin/activate
cd /mnt/bigdata/capstone/brats_gnn_segmentation

# All 5 folds (est. 30-40 hours):
nohup python3 scripts/train_unet_baseline.py > logs/unet_training.log 2>&1 &

# Or one fold at a time:
python3 scripts/train_unet_baseline.py --fold 0
python3 scripts/train_unet_baseline.py --fold 1
# ... etc
```

### After training completes:
- [ ] Check results: cat checkpoints/unet_baseline/aggregate_results.json
- [ ] Evaluate on held-out 251 patients (need script update)
- [ ] Run BraTS 2023 evaluation
- [ ] Run benchmarks (speed, memory, params)

---

## PHASE 3: Paper Updates (After Retraining)

- [x] Update all U-Net numbers (Dice 87.84%, params 69.1M) in LaTeX and README — DONE 2026-04-01
- [ ] Update comparison tables
- [ ] Regenerate figures
- [ ] Add Wilcoxon test results
- [ ] Report median Dice alongside mean
- [ ] Reframe as efficiency paper
- [ ] Cut verbose sections (~20%)
- [ ] Fix Tumour/Tumor inconsistency

### Table 10 & 11 Integrity Issues (review after evaluations)
- [x] **Table 11 mixes measured + published data** — FIXED: replaced with 4-column table (GNN Single | GNN Ensemble | Our 3D U-Net | nnU-Net published) with Source row.
- [x] **Cherry-picked numbers in Table 11** — FIXED: ensemble Dice paired with ensemble params (2.2M), single Dice with 439K.
- [x] **"70× fewer params" claim is invalid** — FIXED: removed. Replaced with 157× (measured) and separate nnU-Net comparison.
- [x] **Add "Source" clarity** — FIXED: Source row added to Table 11, Source column added to Table 10.
- [x] **Audit Table 10 (SOTA comparison)** — FIXED: added Source column, clarified provenance for all rows.
- [x] **Add narrative explaining U-Net underperformance** — already existed in paper. Added inference vs training VRAM clarification paragraph.
- [x] **Inference vs training VRAM distinction** — FIXED: added paragraph clarifying U-Net fits for inference (2,500 MB) but training is constrained (batch 4), and nnU-Net can't train on 6GB at all.

### Ablation 512-dim Issue
- [x] **512-dim "disqualify" claim was false** — 1.71M params fits easily on 6GB. Removed false VRAM disqualification.
- [x] **Reframed as future work** — 512-dim deferred to future investigation under matched training budgets.
- [x] **Added to Future Work section** — 512-dim exploration as Fourth item in future work.
