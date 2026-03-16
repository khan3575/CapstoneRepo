# Next Steps
**Updated:** 2026-03-15
**Final model: binary_v3 (BCE only) — all experiments concluded**

---

## ✅ EVERYTHING CLOSED — Final Numbers

All forensic audit fixes done. All experiments done. All results verified with binary_v3.

| Result | Value | Source |
|---|---|---|
| CV Dice | 90.02% ± 0.66% | `checkpoints/binary_v3/fold_{0-4}/results.json` |
| Ensemble Dice (held-out 251 pts) | **91.41%** | `research_results/ensemble_v2/ensemble_results.json` |
| Accuracy | 99.14% | same |
| Sensitivity | 87.77% | same |
| Specificity | 99.76% | same |
| Precision | 95.52% | same |
| BraTS 2023 zero-shot Dice | **89.40%** | `research_results/brats2023_evaluation/results.json` |
| Generalisation gap | **2.01%** | same |
| Inference-only speed | **75.4 ms** | `research_results/timing_benchmark/two_scenario_results.json` |
| End-to-end speed | **~1,730 ms** | same (JSON mean=1732ms) |
| Peak GPU memory | **11 MB** | same |
| Model size | **5.1 MB** | same |
| Parameters | **439,041** | `src/gnn_model.py` |
| Loss function | **BCEWithLogitsLoss** | `src/train_cv_fold.py` line 284 |

---

## ✅ ALL EXPERIMENTS AND FIGURES DONE — 2026-03-15

### Ensemble (RERUN-5)
Re-run with binary_v3 after Audit 2 found ensemble_results.json was contaminated with v4 data.
**Restored result: 91.41%** — all 5 fold val_dice values now match binary_v3 exactly.

### All 7 figures final (DPI=300) — `research_results/figures/`
| Figure | Description | Key number |
|---|---|---|
| fig_A | 5-fold CV bar chart | 90.02%, ensemble 91.41% |
| fig_B | Efficiency bubble | 5.9× speedup vs U-Net |
| fig_C | BraTS 2021 vs 2023 generalisation | gap 2.01pp |
| fig_D | Speed & memory | 5.9×, 227× less GPU memory |
| fig_E | Ensemble lift | +1.39% |
| fig_F | Ablation study | GraphSAGE 5L Pareto-optimal |
| fig_G | Per-graph Dice histogram | 81.2% high-quality, 5.0% miss |

All numbers sourced from verified JSONs (no hardcoded stale values remain).

---

---

## 🟢 THESIS WRITING
LaTeX: `paperWriting/final_paper/`

Key things to state correctly:
- Loss function: **BCEWithLogitsLoss** (NOT CombinedLoss — that was defined but never used)
- Batch size: **24** (not 32)
- Effective batch: **48** (accumulation_steps=2)
- CV fold sizes: **720 train / 80 val / 200 test** per fold
- Folds directory: `data/cv_folds_v2/` (not cv_folds/)
- Determinism: **non-deterministic** (cudnn.benchmark=True, deterministic=False)

---

## Experiment Archive (for reference / thesis methodology section)

| Version | CV Dice | Ensemble | Notes |
|---|---|---|---|
| binary_v3 | 90.02% | **91.41%** | ✅ FINAL |
| binary_v4 (BCE+Dice) | 90.26% | 91.29% | better CV, worse held-out |
| binary_v5 (pw=16+flip) | 88.90% | — | pos_weight too aggressive |
| pos_weight search (4–12) | ≤88.98% fold-0 | — | none beat v4 |

---
*Maintained by Claude Sonnet 4.6 · See FIX_LOG.md for full audit trail*
