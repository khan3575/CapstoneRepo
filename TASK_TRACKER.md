# Project Task Tracker
**Goal:** Fix all known issues in the 2D binary GNN project → clean results → paper-ready
**Scope:** Binary segmentation only (no multi-class). BraTS 2021 + BraTS 2023.
**Last updated:** March 2026

---

## ⚠️ Why Order Matters

Everything downstream depends on the data split and trained models.
Do NOT skip ahead — each phase feeds the next.

```
Data Split → Retrain → Ablation → Ensemble → Timing → Figures → BraTS2023 → Paper
    │            │          │          │          │        │          │         │
    └────────────┴──────────┴──────────┴──────────┴────────┴──────────┴─────────┘
         Fix this first or everything above it needs to be redone
```

---

## PHASE 1 — Data Split ✅ COMPLETE

### Task 1.1 — Create proper held-out test split ✅
- [x] Write `scripts/create_held_out_split.py`
  - Stratifies by tumour volume quartile
  - 251 patients → `data/splits/held_out_test.json` (sealed, never used in training)
  - 1,000 patients → `data/splits/cv_pool.json` (all CV runs use this only)
- [x] Run script, verify patient counts (251 held-out, 1,000 cv_pool)
- [x] Verified zero overlap between held-out and cv_pool
- **Output:** `data/splits/held_out_test.json`, `data/splits/cv_pool.json` ✅

### Task 1.2 — Regenerate fold files from cv_pool only ✅
- [x] Write `scripts/create_cv_folds_v2.py` (reads cv_pool, not all 1,251)
- [x] Generated `data/cv_folds_v2/fold_0.json` … `fold_4.json`
- [x] Verified: no held-out patient in any fold
- [x] Updated `config.yaml`: cv_folds → `data/cv_folds_v2/`, checkpoints_binary → `checkpoints/binary_v2/`, ensemble → `research_results/ensemble_v2/`
- **Output:** `data/cv_folds_v2/` (5 folds: 720 train / 80 val / 200 test each) ✅

---

## PHASE 2 — Retrain All 5 Folds
> Uses the new cv_pool folds. Runs overnight (~25 hours GPU compute).
> **Run in your own terminal** (CUDA unavailable in Claude Code terminal).

### Task 2.1 — Confirm training script is ready
- [x] `config.yaml` now points to `data/cv_folds_v2/` and `checkpoints/binary_v2/`
- [x] Verified `data/graphs/` has graphs for all 1,000 cv_pool patients (1000/1000 ✓) and all 251 held-out patients (251/251 ✓)
- [x] Output will go to `checkpoints/binary_v2/`

### Task 2.2 — Launch training (all 5 folds)
Run sequentially in your terminal (each fold ~5h, total ~25h):
```bash
source /mnt/bigdata/capstone/.env/bin/activate
cd /mnt/bigdata/capstone/brats_gnn_segmentation
python src/train_cv_fold.py --fold 0
python src/train_cv_fold.py --fold 1
python src/train_cv_fold.py --fold 2
python src/train_cv_fold.py --fold 3
python src/train_cv_fold.py --fold 4
```
- [x] Fold 0 — best epoch 26, val Dice 90.00%, test Dice 88.72%
- [x] Fold 1 — best epoch 30, val Dice 89.74%, test Dice 90.48%
- [x] Fold 2 — best epoch 40, val Dice 88.79%, test Dice 90.31%
- [x] Fold 3 — best epoch 32, val Dice 88.12%, test Dice 90.13%
- [x] Fold 4 — best epoch 27, val Dice 90.35%, test Dice 90.47%
- **Actual time:** ~7.3 min/fold (with torch.compile + batch_size=64 optimizations)
- **Actual Dice:** 89.88% ± 0.72% (CV mean)

### Task 2.3 — Record per-fold results ✅
- [x] Collect Dice, Accuracy, Sensitivity, Specificity, Precision for each fold
- [x] Update results table in paper

---

## PHASE 3 — Ablation Study (with speed fixes)
> Run AFTER Phase 2 so ablation uses the same data split as main results.
> With speed fixes: ~35–40 min per variant instead of 8 hours.

### Task 3.1 — Apply ablation speed fixes ✅
- [x] Created `scripts/train_ablation.py` wrapper (fold 0 only, 30 epochs, cudnn.benchmark=True)

### Task 3.2 — Run all ablation experiments
Each run: ~35–40 min on GPU

**Architecture variants:**
- [x] Baseline (5 layers, 256 dim) — test Dice 84.03% (`research_results/ablation_study_accuracy/baseline_accuracy/`)
- [x] Deeper: 6 layers, 256 dim — test Dice 84.00% (`layers_6_accuracy/`)
- [~] Wider: 5 layers, 512 dim — model saved, results.json missing (training interrupted)
- [ ] GAT: replace GraphSAGE with GAT, 5 layers, 256 dim

**Batch size variants:**
- [ ] Batch size 16
- [ ] Batch size 24
- [ ] Batch size 32 (baseline)
- [ ] Batch size 48
- [ ] Batch size 64

**Superpixel count variants (justifies graph construction choice):**
- [ ] 50 superpixels/slice
- [ ] 80 superpixels/slice
- [ ] 100 superpixels/slice (current default is adaptive ~144)
- [ ] 150 superpixels/slice
- [ ] 200 superpixels/slice

**Slice count variants (justifies 200 slices choice):**
- [ ] 100 slices/patient
- [ ] 150 slices/patient
- [ ] 200 slices/patient (current default)

**Inter-slice window size variants (justifies 3-slice window):**
- [ ] 1 slice (no inter-slice edges)
- [ ] 3 slices (current default)
- [ ] 5 slices

### Task 3.3 — Record all ablation results
- [ ] Fill complete ablation table (all real numbers, no blanks, no estimates)

---

## PHASE 4 — Ensemble Evaluation (Clean)
> Run AFTER Phase 2. Uses the 251-patient held-out set.

### Task 4.1 — Fix `src/inference_ensemble.py` ✅
- [x] Changed default test set to `held_out_test.json`
- [x] Loads all 5 models from `checkpoints/binary_v2/`
- [x] Evaluated ensemble on 251 held-out patients
- [x] Results saved to `research_results/ensemble_v2/ensemble_results.json`

### Task 4.2 — Run and record ensemble results ✅
- [x] **Ensemble Dice: 91.41%**, Accuracy: 99.14%, Sensitivity: 87.77%, Specificity: 99.76%, Precision: 95.52%
- [x] CV mean: 89.88% ± 0.72% → ensemble improves by +1.53%
- [x] Statistical test: one-sample t-test, t=-4.21, **p=0.0136** (significant, p<0.05). Ensemble strictly beats all 5 individual models. Results: `research_results/ensemble_v2/statistical_test.json`
- **Actual result: 91.41% Dice on 251 held-out patients (21,543 graphs)**

---

## PHASE 5 — Timing Benchmarks
> Run AFTER Phase 2 (needs trained models).
> **Run in your own terminal** (CUDA unavailable in Claude Code terminal).

### Task 5.1 — GPU timing for Scenario 1 (inference only) ✅
- [x] Run: `python scripts/benchmark_two_scenarios.py --num_patients 50 --fold 0 --device cuda`
- [x] **Scenario 1: 84ms mean, 81ms median** (47 patients, GPU)

### Task 5.2 — Update benchmark script with fast-slic for Scenario 2 ✅
- [x] `src/graph_construction.py` updated with fast_slic (8.5× speedup end-to-end)
- [x] Reran end-to-end benchmark: 13.3s → **1.57s** (8.5× faster)
- [x] Recorded updated timing (47 patients, RTX 2060)

### Task 5.3 — Record both numbers for paper ✅
| Scenario | Result |
|---|---|
| Inference only (GPU, pre-built graph) | **74ms** mean, 73ms median |
| End-to-end with fast-slic (graph construction + GNN) | **1.57s** mean (graph: 1.56s, GNN: 6ms) |
| End-to-end old (skimage SLIC) | 13.3s — superseded |
| Model size | 5.1 MB |
| Peak GPU memory | 11 MB |

---

## PHASE 6 — Figure Regeneration
> Run AFTER Phase 4 (needs final metrics). High-quality figures for paper.

### Task 6.1 — Regenerate all metric plots ✅
- [x] Training curves (loss, Dice) for all 5 folds — `training_curves_fold_X.png`
- [x] Per-fold Dice bar chart + mean ± std + ensemble line — `cv_dice_summary.png`
- [x] ROC operating points per fold + mean — `roc_operating_points.png`
- [x] Metrics radar chart (all 5 metrics, folds + ensemble) — `metrics_radar.png`
- [x] Ensemble vs individual comparison bar chart — `ensemble_vs_individual.png`
- All saved at 300 DPI in `research_results/figures/`

### Task 6.2 — Qualitative examples
- [ ] From held-out ensemble results, select: 3 worst cases, 1 median, 1 best
- [ ] For each: original MRI (4 modalities), ground truth, prediction, overlay
- [ ] Save to `research_results/failure_cases/`
- [ ] Write 1-paragraph failure analysis for Section 6.4

---

## PHASE 7 — BraTS 2023 Cross-Dataset Validation
> Run AFTER Phase 2. No retraining needed — zero-shot evaluation.

### Task 7.1 — Get BraTS 2023 data
- [ ] Create Synapse account at synapse.org if not already done
- [ ] Download BraTS 2023 Glioma Task (~100GB)
  - URL: https://www.synapse.org/#!Synapse:syn51514105
- [ ] Store at `/mnt/bigdata/brats2023/`

### Task 7.2 — Preprocess BraTS 2023
- [ ] Run existing preprocessing pipeline on BraTS 2023 patients
  (same skull-strip + normalise + slice selection pipeline — no changes needed)
- [ ] Fix label scheme: BraTS 2023 uses label 3 instead of label 4 for ET
  - In preprocessing: remap label 3 → label 4, then binary transform works as-is
- [ ] Build graphs: run graph construction on all BraTS 2023 patients

### Task 7.3 — Zero-shot inference
- [ ] Write `scripts/evaluate_brats2023.py`
  - Load ensemble from Phase 2 (checkpoints/binary_v2/)
  - Run inference on all BraTS 2023 patients
  - Report Dice, Sensitivity, Specificity on BraTS 2023
- [ ] Record generalisation gap: BraTS 2021 score − BraTS 2023 score

### Task 7.4 — Add to paper
- [ ] Add cross-dataset validation table to Section 6
- [ ] Expected: ~2–5% Dice drop (normal, honest to report)

---

## PHASE 8 — Paper Rewrite
> Final phase. Do this LAST when all numbers are clean and final.

### Task 8.1 — Update all result numbers
- [ ] Table 1 (CV results): use Phase 2 numbers
- [ ] Ensemble result: use Phase 4 clean number
- [ ] Efficiency table: use Phase 5 two-scenario timing
- [ ] Ablation table: use Phase 3 complete numbers
- [ ] Add BraTS 2023 table from Phase 7
- [ ] Add failure case figure from Phase 6

### Task 8.2 — Fix paper text
- [ ] Abstract: replace 92.92% with real clean ensemble number from Phase 4
- [ ] Section 6.3: replace "6.9× faster" with two-scenario table
- [ ] Section 6.4: add failure case analysis paragraph
- [ ] Section 2 (Related Work): add citations — GNN-SEG, BiTr-UNet, SVGFormer
- [ ] Section 4: add sentence — "We report binary whole-tumour Dice; SOTA methods
      report multi-class results. Direct comparison is therefore indicative, not strict."
- [ ] Address the one-sample t-test result (p=0.1385) explicitly or remove it

### Task 8.3 — Rebuild PDF
- [ ] Run: `cd paperWriting/Template_TextOnly && pdflatex main.tex` (or latexmk)
- [ ] Verify all images render correctly
- [ ] Check page count and figure placement

---

## Status Legend
- [ ] Not started
- [~] In progress
- [x] Done
- [!] Blocked — needs something else first

---

## Quick Reference — Key Commands

```bash
# Activate environment
source /mnt/bigdata/capstone/.env/bin/activate
cd /mnt/bigdata/capstone/brats_gnn_segmentation

# Phase 1 (DONE — scripts already run):
#   data/splits/held_out_test.json  →  251 held-out patients
#   data/splits/cv_pool.json        →  1,000 cv_pool patients
#   data/cv_folds_v2/fold_X.json    →  720 train / 80 val / 200 test per fold

# Phase 2 — Train a fold (run in your terminal, not Claude Code)
python src/train_cv_fold.py --fold_idx 0   # repeats for folds 1-4
# Output: checkpoints/binary_v2/fold_X/best_model.pth

# Phase 3 — Run ablation variant (fold 0 only, 30 epochs)
python scripts/train_ablation.py --variant deeper_network --fold 0 --epochs 30

# Phase 4 — Run ensemble on held-out set (after Phase 2)
python src/inference_ensemble.py --test_file data/splits/held_out_test.json

# Phase 5 — Run timing benchmark (MUST be in normal terminal for CUDA)
python scripts/benchmark_two_scenarios.py --num_patients 50 --fold 0 --device cuda

# Phase 8 — Rebuild PDF (LaTeX)
cd paperWriting/Template_TextOnly && latexmk -pdf main.tex
```

---

## Known Issues Log

| Issue | Status | Notes |
|---|---|---|
| CUDA not accessible from Claude Code terminal | Known | Run GPU scripts in your own terminal |
| Ensemble 92.92% has data leakage | Will be fixed in Phase 4 | Do not use this number in paper until then |
| Ablation table has blank rows | Will be fixed in Phase 3 | |
| Timing inconsistency (12.7ms vs 1.47s vs 1484ms) | Will be fixed in Phase 5 | |
| Related work missing SVGFormer, BiTr-UNet | Will be fixed in Phase 8 | |
| cudnn.deterministic=True causes 2-3× training slowdown | Fixed in Phase 3 ablation only | Do NOT change in main training (affects reproducibility) |
