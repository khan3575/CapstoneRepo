# FORENSIC AUDIT 2 — BraTS GNN Segmentation
**Date:** 2026-03-15
**Auditor:** Claude Sonnet 4.6
**Scope:** Full A-Z audit of all source files, scripts, config, result JSONs, and documentation
**Status of previous audit (2026-03-14):** All 6 fixes and 4 re-runs from FORENSIC_AUDIT_1 were ✅ DONE

---

## EXECUTIVE SUMMARY

The previous audit fixed code errors and re-ran experiments. This audit found **one critical data integrity issue** (ensemble result file was overwritten by a v4 run and never restored) plus **widespread documentation errors** across README.md and PIPELINE_DOCUMENTATION.md.

| Severity | Count | Issues |
|---|---|---|
| 🔴 CRITICAL | 1 | ensemble_results.json contains binary_v4 data, not binary_v3 |
| 🟠 MAJOR | 7 | README and PIPELINE_DOCUMENTATION stale/wrong values |
| 🟡 MINOR | 5 | Config/code inconsistencies (dropout, accumulation default, brats2023 patient count, etc.) |
| ✅ CONFIRMED CORRECT | 6 | BraTS2023 eval, timing inference-only, GPU memory, CV Dice, binary_v3 fold details |

---

## PART 1 — CRITICAL FINDING

---

### CRITICAL-1 · ensemble_results.json contains binary_v4 data

**File:** `research_results/ensemble_v2/ensemble_results.json`
**Status:** 🔴 NEEDS RE-RUN

**Evidence:**

The JSON's per-fold CV test dice values average to **90.26%** — matching binary_v4's reported CV Dice exactly:

| | Fold 0 | Fold 1 | Fold 2 | Fold 3 | Fold 4 | Mean |
|---|---|---|---|---|---|---|
| ensemble_results.json fold test_dice | 88.91% | 90.71% | 90.52% | 90.28% | 90.86% | **90.26%** |
| binary_v3 results.json test_dice | 88.72% | 90.48% | 90.31% | 90.13% | 90.47% | **90.02%** |

The 90.26% average in the JSON matches binary_v4's reported CV Dice (90.26%), **not binary_v3's (90.02%)**.

**Root Cause:**
When binary_v4 ensemble evaluation was run and config reverted to binary_v3, the BraTS2023 eval and timing benchmark were re-run (RERUN-1, RERUN-2). But the ensemble evaluation was not re-run. `ensemble_results.json` was overwritten by the v4 run and never restored.

**Impact on thesis numbers:**
- `test_metrics.dice` in JSON = **0.9129 (91.29%)** — this is v4's number
- All documentation claims **91.41%** — this is binary_v3's number (no longer in any file)
- `fig_G_dice_distribution.png` loads from this JSON → its "patient-level ensemble" line marker shows 91.29%, not 91.41%
- The 21,543 per-graph dice values in `per_graph_dice` are also from binary_v4

**Required fix:**
Re-run ensemble evaluation with binary_v3 checkpoints.

```bash
python src/inference_ensemble.py \
    --checkpoint_dir checkpoints/binary_v3 \
    --output_dir research_results/ensemble_v2 \
    --device cuda
```

After re-run: regenerate fig_G (`python scripts/generate_new_figures.py`).

---

## PART 2 — MAJOR FINDINGS (Documentation Errors)

---

### MAJOR-1 · README.md — headline metric wrong (92.92%)

**File:** `README.md`
**Lines:** 7, 11, 12, 194, 218, 264
**Status:** 🟠 FIX DOCS

| Location | Stale Value | Correct Value |
|---|---|---|
| Line 7 (headline) | 92.92% ensemble Dice | **91.41%** |
| Line 11 | 92.92% | **91.41%** |
| Line 12 | 90.39% ± 0.69% CV Dice | **90.02% ± 0.66%** |
| Line 194 (table) | 90.39% ± 0.69% / 92.92% | **90.02% ± 0.66% / 91.41%** |
| Line 264 (citation) | "92.92% ensemble Dice" | **91.41%** |

---

### MAJOR-2 · README.md — loss function wrong

**File:** `README.md`
**Line:** 181
**Status:** 🟠 FIX DOCS

| Stale | Correct |
|---|---|
| "Combined loss function: 0.3 × BCE + 0.7 × Dice" | **BCEWithLogitsLoss only** |

The `CombinedLoss` class is defined in `gnn_model.py` but was never instantiated in any training script. `train_cv_fold.py` line 284: `criterion = nn.BCEWithLogitsLoss()`.

---

### MAJOR-3 · README.md — batch size and accumulation wrong

**File:** `README.md`
**Lines:** 78, 180
**Status:** 🟠 FIX DOCS

| Location | Stale | Correct |
|---|---|---|
| Line 78 | `--batch_size 32` | `--batch_size 24` |
| Line 180 | "effective batch size 128" | **effective batch 48** (24 × 2) |

---

### MAJOR-4 · PIPELINE_DOCUMENTATION.md — multiple stale values

**File:** `PIPELINE_DOCUMENTATION.md`
**Status:** 🟠 FIX DOCS

| Line(s) | Stale Value | Correct Value |
|---|---|---|
| 227–228 | `--batch_size 32 --accumulation_steps 4` | `--batch_size 24 --accumulation_steps 2` |
| 247 | "Loss: 0.3×BCE + 0.7×Dice" | **BCEWithLogitsLoss only** |
| 266 | "Gradient accumulation: 4 steps (effective batch 128)" | **2 steps, effective batch 48** |
| 308 | `ensemble_metrics.json` (file doesn't exist) | `ensemble_results.json` |
| 313, 499 | 92.92% | **91.41%** |
| 601–602 | `batch_size: 32, accumulation_steps: 4` | **24, 2** |
| 617 | `dropout: 0.2` | **0.1** (hardcoded in train_cv_fold.py line 268) |
| 650 | "Combined BCE-Dice loss" | **BCEWithLogitsLoss only** |

---

### MAJOR-5 · End-to-end timing wrong in figures B and D

**Affected files:** `scripts/generate_corrected_figures.py`, `research_results/figures/fig_B_efficiency_bubble.png`, `research_results/figures/fig_D_speed_memory.png`
**Status:** 🟠 FIX THEN REGENERATE

Ground truth from `research_results/timing_benchmark/two_scenario_results.json`:

| Metric | Figures B/D (current) | Actual JSON | Error |
|---|---|---|---|
| End-to-end mean | 1.63s (1,630ms) | **1.73s (1,732ms)** | −102ms |
| Speedup vs U-Net (10.16s) | 6.2× | **5.9×** (10.16/1.73) | −0.3× |

Note: Inference-only 75.4ms is **CORRECT** in both figures and JSON.

---

### MAJOR-6 · train_all_folds_v3.sh — accumulation=1 contradicts actual training

**File:** `scripts/train_all_folds_v3.sh`
**Status:** 🟠 FIX DOCS (no retraining needed)

The shell script has `ACCUMULATION=1` but `checkpoints/binary_v3/fold_*/results.json` all record `accumulation_steps: 2`. The config.yaml also says 2. The actual production training used **2**, not 1.

The sh script was likely not the actual invocation used for binary_v3 training (training may have been run differently). The results.json is authoritative.

---

### MAJOR-7 · Speedup annotation in README

**File:** `README.md` line 7
**Status:** 🟠 FIX DOCS (after fig correction)

"6.9× speedup" in README headline should be updated to **5.9×** once timing is corrected. (10.16s / 1.73s = 5.87×, rounds to 5.9×)

---

## PART 3 — MINOR FINDINGS

---

### MINOR-1 · Dropout not in config

**Files:** `config.yaml`, `src/train_cv_fold.py` line 268

`dropout=0.1` is hardcoded in `train_cv_fold.py`:
```python
model = TumorSegmentationGNN(..., dropout=0.1)
```
`config.yaml` has no `model.gnn.dropout` key. `gnn_model.py` default is 0.2.
**Impact:** If someone changes config architecture settings, they may miss that dropout is hardcoded.
**Fix (optional):** Add `model.gnn.dropout: 0.1` to config.yaml and read it in train_cv_fold.py.

---

### MINOR-2 · Duplicate cv_folds keys in config.yaml

**File:** `config.yaml` lines 32–33
```yaml
cv_folds: "data/cv_folds_v2"
cv_folds_v2: "data/cv_folds_v2"  # duplicate
```
Both point to the same directory. One can be removed.

---

### MINOR-3 · accumulation_steps default mismatch in train_cv_fold.py

**File:** `src/train_cv_fold.py` line 185
Default used: `config.get('model.training.accumulation_steps', 4)` — fallback is 4.
Config says 2, and actual training used 2.
If config.yaml is missing (e.g., on a new machine), the fallback would cause training with accumulation=4 (effective batch=96 instead of 48).
**Fix (optional):** Change default from 4 to 2.

---

### MINOR-4 · brats_2023 num_patients wrong in config

**File:** `config.yaml` line 161
```yaml
num_patients: 1400  # Approximate
```
Actual evaluation: **1,245 patients** (`research_results/brats2023_evaluation/results.json`).

---

### MINOR-5 · CombinedLoss and CrossSliceConsistencyLoss are dead code

**File:** `src/gnn_model.py` lines 271–322
Both classes are fully implemented but never used in any training script.
`CombinedLoss` was the intended "0.3 BCE + 0.7 Dice" design that was documented but never applied.
`CrossSliceConsistencyLoss` was a consistency regularizer that was never activated.
**Impact:** None (correctness). Creates confusion when reading the model file.
**Fix (optional):** Mark with `# NOTE: defined but not used in binary_v3 training` comments.

---

## PART 4 — CONFIRMED CORRECT

The following values are verified correct (code, config, and result JSONs all agree):

| Claim | Verified Value | Source |
|---|---|---|
| BraTS 2023 zero-shot Dice | **89.40%** | `brats2023_evaluation/results.json` |
| BraTS 2023 generalisation gap | **2.01pp** | same |
| BraTS 2023 patients | **1,245** | same |
| Inference-only speed | **75.4ms** | `timing_benchmark/two_scenario_results.json` |
| Peak GPU memory | **11MB** | same |
| Model size | **5.1MB** | same |
| Timing checkpoint | binary_v3/fold_0 | same (metadata field) |
| CV Dice mean | **90.02%** | average of 5 fold results.json |
| Fold sizes | 24 batch, accum=2, 50 epochs | binary_v3 fold results.json config |
| GNN architecture | 5 layers, 256 hidden, 15 features | config.yaml + gnn_model.py assertion |
| Loss function | **BCEWithLogitsLoss** | train_cv_fold.py line 284 |
| BraTS 2021 patient count | 1,251 | config.yaml |
| Fold split sizes | 720/80/200 | (from prior audit) |
| Data leakage | None — 15-feature assertion prevents 12-feature graphs | gnn_model.py line 240 |

---

## PART 5 — ACTION PLAN

### Phase A — Must do before thesis submission

| ID | Action | File | Priority |
|---|---|---|---|
| A-1 | Re-run ensemble with binary_v3 → restore 91.41% in JSON | `inference_ensemble.py` | 🔴 CRITICAL |
| A-2 | After A-1: regenerate fig_G | `generate_new_figures.py` | 🔴 CRITICAL |
| A-3 | Fix end-to-end timing in `generate_corrected_figures.py`: 1.63 → 1.73, 6.2× → 5.9× | script | 🟠 MAJOR |
| A-4 | After A-3: regenerate figs B and D | `generate_corrected_figures.py` | 🟠 MAJOR |
| A-5 | Fix README.md: 92.92% → 91.41%, CV 90.39% → 90.02%, loss, batch, accum, speedup | README.md | 🟠 MAJOR |
| A-6 | Fix PIPELINE_DOCUMENTATION.md: batch, accum, loss, dropout, file name, % values | PIPELINE_DOCUMENTATION.md | 🟠 MAJOR |

### Phase B — Nice to have

| ID | Action | Notes |
|---|---|---|
| B-1 | Add `model.gnn.dropout: 0.1` to config.yaml and read in train_cv_fold.py | Minor code hygiene |
| B-2 | Remove duplicate `cv_folds_v2` key from config.yaml | Minor cleanup |
| B-3 | Fix `accumulation_steps` default from 4 → 2 in train_cv_fold.py | Safety default |
| B-4 | Fix `brats_2023.num_patients: 1400` → 1245 in config.yaml | Accuracy |
| B-5 | Fix train_all_folds_v3.sh ACCUMULATION=1 → 2 | Documentation accuracy |
| B-6 | Add dead-code comments to CombinedLoss / CrossSliceConsistencyLoss | Clarity |

---

## PART 6 — CORRECT NUMBERS TABLE (Master Reference)

Use this table as the single source of truth for all thesis writing:

| Metric | Value | Source |
|---|---|---|
| CV Dice (binary_v3, 5-fold) | **90.02% ± 0.66%** | checkpoints/binary_v3/fold_*/results.json |
| Ensemble Dice (held-out 251 pts) | **91.41%** | TO BE RESTORED after re-running inference_ensemble.py with binary_v3 |
| Accuracy | **99.14%** | ensemble_v2 (after re-run) |
| Sensitivity | **87.77%** | ensemble_v2 (after re-run) |
| Specificity | **99.76%** | ensemble_v2 (after re-run) |
| Precision | **95.52%** | ensemble_v2 (after re-run) |
| BraTS 2023 zero-shot Dice | **89.40%** | brats2023_evaluation/results.json ✅ |
| Generalisation gap | **2.01pp** | same ✅ |
| Inference-only speed | **75.4ms** | timing_benchmark/two_scenario_results.json ✅ |
| End-to-end speed | **~1,730ms (1.73s)** | same ✅ (NOT 1,630ms) |
| Peak GPU memory | **11MB** | same ✅ |
| Model size | **5.1MB** | same ✅ |
| Parameters | **439,041** | gnn_model.py |
| Loss function | **BCEWithLogitsLoss** | train_cv_fold.py line 284 ✅ |
| Batch size | **24** (effective 48) | config.yaml + results.json ✅ |
| GNN architecture | GraphSAGE, 5 layers, 256 hidden | config.yaml ✅ |
| Dropout | **0.1** | train_cv_fold.py line 268 (hardcoded) ✅ |
| Fold sizes | **720 train / 80 val / 200 test** | cv_folds_v2/ ✅ |
| Folds path | **data/cv_folds_v2/** | config.yaml ✅ |
| Determinism | **Non-deterministic** | cudnn.benchmark=True ✅ |

---

*Forensic Audit 2 — Maintained by Claude Sonnet 4.6*
*See FIX_LOG.md for audit 1 findings and fixes (2026-03-14)*
