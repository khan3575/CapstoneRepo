# PROJECT CONTEXT — Source of Truth
**Last updated:** 2026-04-01 | **Author:** Sakib Khan

---

## 1. Current Objective

Publish a journal paper on **efficient brain tumour segmentation using Graph Neural Networks**.
The GNN pipeline is complete and validated. The U-Net baseline has been **retrained at proper scale (69.1M params)**.
Remaining work: update the paper with new U-Net results, run held-out/BraTS-2023 evaluations for U-Net, regenerate comparison figures, and submit.

**Target journals:** Computers in Biology and Medicine *or* Biomedical Signal Processing and Control.
**Framing:** Efficiency-first (not accuracy-competitive). The GNN is 157x smaller and 5.9x faster.

---

## 2. Active Tech Stack

| Component | Version | Notes |
|---|---|---|
| Python | 3.12.3 | System Python on Ubuntu |
| PyTorch | 2.8.0+cu128 | CUDA 12.8 |
| PyTorch Geometric | 2.6.1 | GraphSAGE via `SAGEConv` |
| NumPy | 2.3.3 | |
| SciPy | 1.16.2 | |
| scikit-learn | 1.7.2 | |
| NiBabel | 5.3.2 | NIfTI I/O |
| GPU | RTX 2060 6GB | AMP training for both models |
| Virtual env | `/mnt/bigdata/capstone/.env/` | Activate before any Python work |
| LaTeX | Overleaf (flat export) | `paperWriting/overleaf_flat/` |
| OS | Ubuntu, kernel 6.17.0-19 | |

---

## 3. Architecture Map

```
Raw BraTS 2021 NIfTI (1,251 patients, 4 MRI modalities)
        |
        v
[preprocessing.py] --> data/preprocessed/
        |
        v
[graph_construction.py] --> data/graphs/
  (SLIC superpixels, 200/slice, 15-dim node features, RAG edges)
        |
        v
[train_cv_fold.py]  --> 5-fold CV on 1,000-patient pool
  (GraphSAGE 5L/256D, BCE loss, AMP, batch 24, accum 2)
        |
        v
[inference_ensemble.py] --> Soft-voting ensemble on 251 held-out
        |
        v
Research results: figures, benchmarks, ablation, BraTS-2023 eval
```

**U-Net baseline** (`scripts/train_unet_baseline.py`): Standard 3D U-Net, 69.1M params (base=56, levels=4), BCE+Dice loss, sliding-window inference, same CV splits.

---

## 4. The "State of Truth" Ledger

### IMPLEMENTED (Stable)

| Item | Status | Key File(s) |
|---|---|---|
| GNN 5-fold CV (binary_v3) | FINAL | `checkpoints/binary_v3/` [Ref: config.yaml] |
| Ensemble inference | FINAL | `research_results/ensemble_v2/` |
| BraTS 2023 zero-shot eval | FINAL | `research_results/brats2023_evaluation/` |
| Speed/memory benchmark | FINAL | `research_results/timing_benchmark/` |
| Ablation study | FINAL | `research_results/ablation_study_accuracy/`, `checkpoints/ablation/` |
| Failure case analysis | FINAL | `research_results/failure_cases/` |
| 7 paper figures (fig_A-G) | FINAL | `research_results/figures/` |
| Repo cleanup (Phase 1) | DONE | Stale files moved to `archive/` [Ref: CLEANUP_AND_UNET_PLAN.md] |
| U-Net 5-fold CV training | DONE | `checkpoints/unet_baseline/aggregate_results.json` |

### IN-PROGRESS

| Item | Status | Notes |
|---|---|---|
| U-Net held-out (251 pts) evaluation | NOT STARTED | Need ensemble or best-fold inference script |
| U-Net BraTS 2023 eval | NOT STARTED | |
| U-Net speed/memory benchmark | NOT STARTED | |
| Paper: update U-Net numbers | NOT STARTED | Old paper claimed 68M/~87.5% — now 69.1M/87.84% confirmed |
| Paper: comparison tables & figures | NOT STARTED | Must regenerate with real U-Net numbers |
| Paper: Wilcoxon signed-rank test | NOT STARTED | GNN vs U-Net per-patient Dice |
| Paper: editorial fixes | NOT STARTED | Tumour/Tumor consistency, cut ~20% verbosity |

### ROADMAP (Immediate Next Steps)

1. **Run U-Net held-out evaluation** — ensemble or single-best-fold on 251 sealed patients
2. **Run U-Net speed + memory benchmark** — inference time and peak VRAM
3. **Run Wilcoxon test** — statistical comparison GNN vs U-Net (per-patient Dice available in both aggregate JSONs)
4. **Update paper** — tables, figures, claims, abstract numbers
5. **Reframe efficiency narrative** — GNN is ~157x smaller, 5.9x faster, competitive Dice
6. **Submit to journal**

---

## 5. Final Numbers

### GNN (binary_v3) — CONFIRMED [Ref: ensemble_v2/ensemble_results.json, timing_benchmark/two_scenario_results.json]

| Metric | Value |
|---|---|
| CV Dice (5-fold) | 90.02% +/- 0.66% |
| Ensemble Dice (251 held-out) | **91.41%** |
| Accuracy | 99.14% |
| Sensitivity | 87.77% |
| Specificity | 99.76% |
| Precision | 95.52% |
| BraTS 2023 zero-shot | 89.40% (gap 2.01pp) |
| Parameters | 439,041 |
| Model size | 5.1 MB |
| Peak GPU memory (inference) | 11 MB |
| Inference (pre-built graph) | 75.4 ms |
| Inference (end-to-end) | 1,730 ms |

### U-Net Baseline (69.1M params) — NEWLY TRAINED [Ref: checkpoints/unet_baseline/aggregate_results.json]

| Metric | Value |
|---|---|
| CV Dice (5-fold) | 87.84% +/- 2.38% |
| Fold 0 / 1 / 2 / 3 / 4 | 88.93 / 86.67 / 90.14 / 83.74 / 89.73 |
| Parameters | 69,146,113 |
| Training time (5 folds) | 21.5 hours |
| Loss | BCE(pos_weight=9) + Dice |
| Held-out Dice | **PENDING** |
| BraTS 2023 eval | **PENDING** |
| Speed benchmark | **PENDING** |

### Comparison Summary (CV Dice)

| Model | Params | CV Dice | Ratio |
|---|---|---|---|
| GNN (binary_v3) | 439K | 90.02% | 1x |
| U-Net (3D) | 69.1M | 87.84% | 157x larger |
| **GNN advantage** | **157x smaller** | **+2.18pp** | |

---

## 6. Decision Log

- **Final GNN model = binary_v3 (BCE only):** v4 (BCE+Dice) improved CV by +0.23% but dropped ensemble from 91.41% to 91.29%. [Ref: project_state memory, 2026-03-15]
- **U-Net retrained at 69.1M params:** Original run was 1.4M (base=16, levels=3) — a config error that inflated claims. Fixed to base=56, levels=4. [Ref: CLEANUP_AND_UNET_PLAN.md, 2026-03-31]
- **Paper framing = efficiency-first:** GNN beats U-Net on Dice (90.02% vs 87.84% CV) and is 157x smaller — but we lead with efficiency, not SOTA claims. [Ref: project_journal_retraining memory]
- **Multi-class and 3D supervoxels deferred:** Scoped out to a follow-up paper to keep this submission focused. [Ref: project_journal_retraining memory]
- **15 node features (leakage-free):** Removed `tumor_ratio` (ground-truth leak) during forensic audit. Folds regenerated as `cv_folds_v2`. [Ref: README.md, FIX_LOG.md (archived)]
- **Non-deterministic training:** `cudnn.benchmark=True` for speed; seed 42 used only for data splits. [Ref: config.yaml, README.md]
- **Held-out set sealed:** 251 patients from `data/splits/held_out_test.json`, never used in training or fold selection. [Ref: config.yaml]

---

## 7. Avoid List

Do NOT:

1. **Touch `checkpoints/binary_v3/`** — this is the final GNN model. Read-only.
2. **Retrain the GNN** — all 5 folds + ensemble are final and audited.
3. **Use the old U-Net numbers** (68M params, ~87.5% Dice from README) — these were from the 1.4M misconfigured run. Use the new 69.1M/87.84% numbers.
4. **Reference `archive/PROJECT_CONTEXT.md`** — that is the old, outdated context file. THIS file is the current one.
5. **Reference any file in `archive/`** — all archived files are stale (old trackers, audits, deprecated code).
6. **Suggest multi-class segmentation** — explicitly deferred to follow-up paper.
7. **Suggest 3D supervoxels** — deferred to follow-up paper.
8. **Add `tumor_ratio` back to features** — it was a ground-truth leak, removed permanently.
9. **Change CV folds** — `data/cv_folds_v2/` is the final, leakage-free split. Do not regenerate.
10. **Mock databases or test data in evaluations** — user requires real data for all benchmarks.
11. **Give time estimates** — user has noted inaccuracy in past estimates.
12. **Add trailing summaries** — user prefers terse responses.

---

## 8. Key Paths Quick Reference

```
Project root:     /mnt/bigdata/capstone/brats_gnn_segmentation/
Python env:       /mnt/bigdata/capstone/.env/
Config:           config.yaml
GNN train:        src/train_cv_fold.py
GNN model:        src/gnn_model.py
GNN checkpoints:  checkpoints/binary_v3/fold_{0..4}/
U-Net train:      scripts/train_unet_baseline.py
U-Net checkpoints:checkpoints/unet_baseline/fold_{0..4}/
Ensemble:         src/inference_ensemble.py
CV folds:         data/cv_folds_v2/
Held-out split:   data/splits/held_out_test.json
Figures:          research_results/figures/fig_{A..G}_*.png
Paper (LaTeX):    paperWriting/overleaf_flat/main_bubt_paper_no_images.tex
Active plan:      CLEANUP_AND_UNET_PLAN.md
```

---

*This file supersedes all previous context documents. The archived `archive/PROJECT_CONTEXT.md` is obsolete.*
