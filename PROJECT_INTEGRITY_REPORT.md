# PROJECT INTEGRITY REPORT
## BraTS GNN Segmentation — Full Forensic Audit
**Date:** 2026-03-14
**Method:** Full source read of `src/`, `scripts/`, `config.yaml`, `research_results/`, `old-outdated/`, `deprecated/`, all logs and all JSON result files.
**Status:** 9 findings — 4 CRITICAL, 3 MEDIUM, 2 INFO

---

## PART 1 — COMPONENT STATUS

### [VALID] — Use As-Is for Paper

| File / Folder | Last Modified | Why Valid |
|---|---|---|
| `src/config.py` | Feb 9 | Singleton, lazy-load, no hardcoded paths |
| `src/gnn_model.py` | Mar 6 01:00 | 15-feature assertion, architecture correct |
| `src/graph_construction.py` | Mar 7 00:31 | Reads all params from config, uses 15-feature schema |
| `src/preprocessing.py` | Nov 30 | Stable BraTS 2021 preprocessor |
| `src/inference_ensemble.py` | Mar 5 22:23 | Handles `_orig_mod.` prefix, correct |
| `src/dataset.py` | Mar 5 17:33 | Correct (one ghost line noted in FINDING-4) |
| `scripts/train_all_folds_v3.sh` | Mar 6 01:32 | **Authoritative training entry point** |
| `scripts/train_ablation.py` | Mar 6 01:00 | Correct 15-feature ablation wrapper |
| `scripts/benchmark_two_scenarios.py` | Mar 5 22:23 | Valid benchmark — caveat in FINDING-7 |
| `scripts/preprocess_brats2023.py` | Mar 6 16:54 | Valid BraTS 2023 preprocessor (hardcoded path, acceptable) |
| `scripts/evaluate_brats2023.py` | Mar 6 16:55 | Valid script — caveat in FINDING-6 |
| `scripts/generate_corrected_figures.py` | Mar 13 09:22 | Latest figure script |
| `checkpoints/binary_v3/` | Mar 6 | **Production checkpoints — cite these** |
| `research_results/ensemble_v2/` | Mar 6 | Generated from `binary_v3` — valid |
| `research_results/figures/fig_*.png` | Mar 13 | Latest figures — valid |
| `research_results/failure_cases/` | Mar 7 | Valid qualitative results |
| `research_results/brats2023_evaluation/` | Mar 7 | Valid result — see FINDING-6 for caveat |
| `data/splits/held_out_test.json` | — | Sealed 251-patient test set |
| `data/cv_folds_v2/fold_*.json` | — | Leakage-free 5-fold splits |

---

### [DEPRECATED] — Move to `old-outdated/` Immediately

| File | Reason |
|---|---|
| `train_all_folds.sh` (root) | Result-reading section points to deleted `binary_v2` checkpoints (line 34). Misleading entry point. |
| `src/cross_validation.py` | Staged for git deletion (`git status: D`). Physically on disk but path references old `data/cv_folds/`. Import will silently use wrong fold data. |
| `src/evaluation.py` | Sep 29 2025. NIfTI-based evaluator. Incompatible with current GNN pipeline. |
| `src/visualization.py` | Sep 29 2025. NIfTI-based visualizer. Orphaned. |
| `src/aggregate_cv_results.py` | Nov 24 2025. Hardcoded `./checkpoints/cv_experiments/`. Never called by any active script. |
| `src/evaluate_per_region.py` | Nov 26 2025. Hardcoded `in_channels=12`. Will crash on current 15-feature graphs. |
| `src/generate_qualitative_results.py` | Nov 26 2025. Hardcoded old checkpoint paths. Orphaned. |
| `PIPELINE_DOCUMENTATION.md` | Feb 9. Claims 92.92% Dice, batch=32, `data/cv_folds/`. All three are wrong. |

---

### [CONTAMINATED] — Results That Need Cautious Handling

| Result Folder | Timestamp | Issue | Action |
|---|---|---|---|
| `research_results/ablation_study_accuracy/baseline_accuracy/` | **Dec 4 2025** | Run by old script with different config: 50 epochs (not 30), full dataset (not fold-0 only), `patience=15` (not 10). **Not directly comparable to Mar 7 ablation results.** | Re-run with `scripts/train_ablation.py --variant baseline` |
| `research_results/ablation_study_accuracy/layers_6_accuracy/` | **Dec 4 2025** | Same issue as above. Different training regime. | Re-run with `scripts/train_ablation.py --variant deeper_network` |
| `research_results/timing_benchmark/two_scenario_results.json` | Mar 5 | Checkpoint used was `checkpoints/binary_v2/fold_0` (confirmed in `metadata.checkpoint` field). Production model is `binary_v3`. | Re-run after confirming `binary_v3` is the benchmark target |
| `research_results/brats2023_evaluation/results.json` | Mar 7 | Generated with `evaluate_brats2023.py` which hardcodes `checkpoints/binary_v2`. Production model is now `binary_v3`. Result is technically valid but from the prior model generation. | Re-run or document checkpoint version used |

---

## PART 2 — THE INCONSISTENCY LOG

---

### FINDING-1 🔴 CRITICAL: Loss function mismatch — `CombinedLoss` defined, `BCEWithLogitsLoss` used

**Where:**

```python
# src/gnn_model.py — defines (never called by training):
class CombinedLoss(nn.Module):
    # BCE + Dice + CrossSliceConsistency

# src/train_cv_fold.py — line 284 (what actually ran):
criterion = nn.BCEWithLogitsLoss()   # ← BCE only
```

**Impact:** The `CrossSliceConsistencyLoss` and `DiceLoss` components defined in `gnn_model.py` are **never used** during training. If your thesis framework section describes "a combined loss of BCE, Dice and cross-slice consistency," that description does not match the code that produced the results. All checkpoints (`binary_v3`) were trained with BCE-only.

**What to do:** Either (a) update the thesis to say "BCE loss" only, or (b) add `CombinedLoss` to `train_cv_fold.py` and retrain — be aware this will change your Dice numbers.

---

### FINDING-2 🔴 CRITICAL: `weight_decay` hardcoded at `0.01` in training script — config says `0.00001`

**Where:**

```python
# src/train_cv_fold.py — line 285 (hardcoded):
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=learning_rate,
    weight_decay=0.01    # ← HARDCODED, never reads from config
)
```

```yaml
# config.yaml:
weight_decay: 0.00001   # ← 1000× different from what was actually used
```

**Impact:** The regularisation actually applied during all `binary_v3` training is `0.01`. The config value `0.00001` is a documentation lie. Any claim in the thesis that weight decay was `1e-5` is factually wrong. The effective value is `1e-2`.

---

### FINDING-3 🔴 CRITICAL: `evaluate_brats2023.py` hardcodes `binary_v2` checkpoints — production is `binary_v3`

**Where:**

```python
# scripts/evaluate_brats2023.py — line 38:
CHECKPOINT_DIR = "checkpoints/binary_v2"   # ← hardcoded old checkpoint
```

**Confirmed by the result file itself:**

```json
// research_results/brats2023_evaluation/results.json → metadata:
"checkpoint_dir": "/mnt/bigdata/capstone/.../checkpoints/binary_v2"
```

**Impact:** The BraTS 2023 result (89.21%) was generated with `binary_v2` ensemble, not `binary_v3`. The difference between binary_v2 and binary_v3 is batch size (32 vs 24) and minor optimisation settings. The Dice number itself is likely close, but it is not produced by the production model. If the thesis states "our model (binary_v3) achieves 89.21% on BraTS 2023," that is technically incorrect.

---

### FINDING-4 🔴 CRITICAL: `dataset.py` dummy graph has 12 features — model requires 15

**Where:**

```python
# src/dataset.py — line 152 (error fallback path):
dummy_graph = Data(
    x=torch.zeros(1, 12),   # ← 12 features!
    edge_index=torch.zeros(2, 0, dtype=torch.long)
)
```

**The model assertion in `gnn_model.py` line 240:**

```python
assert x.shape[1] == 15, "Expected 15 input features..."
```

**Impact:** If a graph file fails to load during training or inference, the fallback dummy graph with 12 features will be returned and will trigger the model's assertion error, crashing the run instead of gracefully skipping the bad file. The intent of the dummy graph (silent skip) is defeated. This has not crashed production runs because all graph files are valid, but it is a latent defect.

---

### FINDING-5 🟡 MEDIUM: `accumulation_steps` config vs. script vs. code default are three different values

Three different values exist simultaneously:

| Location | Value | Notes |
|---|---|---|
| `config.yaml` | `2` | Documented "production value" |
| `scripts/train_all_folds_v3.sh` | `1` | **Actual value passed to training** |
| `src/train_cv_fold.py` line 73 (function default) | `4` | Never reached, but confusing |
| `src/train_cv_fold.py` line 185 (config fallback default) | `4` | Used if config key missing |

**True effective batch size for all `binary_v3` results:** `24 × 1 = 24` — not 48 as stated in config comments and memory notes.

---

### FINDING-6 🟡 MEDIUM: Ablation table mixes two incomparable training regimes

| Variant | Timestamp | Script Used | Epochs | Fold | Patience |
|---|---|---|---|---|---|
| `baseline_accuracy` | Dec 4 2025 | Old `rerun_undertrained_configs_accuracy.py` | **50** | **All folds, full data** | **15** |
| `layers_6_accuracy` | Dec 4 2025 | Same old script | **50** | **All folds, full data** | **15** |
| `gat_architecture` | Mar 7 2026 | `scripts/train_ablation.py` | **30** | **Fold 0 only** | **10** |
| `wider_network` | Mar 7 2026 | `scripts/train_ablation.py` | **30** | **Fold 0 only** | **10** |

These four rows cannot be placed in the same table without a clear note that the Dec 4 entries used a different protocol. The Dec 4 baseline (84.03%) is ~6 points lower than the Mar 7 fold-0 baseline (which would be ~88.7%), making the "5 layers is optimal" conclusion look much stronger than it actually is when comparing the old entries to the new ones.

---

### FINDING-7 🟡 MEDIUM: Timing benchmark used `binary_v2`, not `binary_v3`

**Confirmed from `two_scenario_results.json` metadata:**

```json
"checkpoint": ".../checkpoints/binary_v2/fold_0/best_model.pth"
```

The benchmark was run on Mar 5 — one day **before** the `binary_v3` training (Mar 6). The model weights differ slightly between v2 and v3. Inference time is determined by architecture (identical), so the timing numbers are valid. GPU memory is also architecture-determined. This is a documentation issue, not a performance issue, but the benchmark should state which checkpoint was used.

---

### FINDING-8 ℹ️ INFO: `PIPELINE_DOCUMENTATION.md` has stale performance numbers and wrong parameters

**Filed Feb 9 — never updated after binary_v3 training.**

| Claim in Doc | Actual Value | Source |
|---|---|---|
| "92.92% Ensemble Dice" | **91.41%** | `ensemble_v2/ensemble_results.json` |
| "90.39% ± 0.69% CV" | **90.02% ± 0.74%** | `binary_v3` fold results |
| "batch 32 best" | **batch 24 used** | `train_all_folds_v3.sh` |
| "seed=42, deterministic mode" | **deterministic=False** | `train_cv_fold.py` line 37 |
| Fold counts: "Train 900 / Val 100 / Test 251" | **Train 720 / Val 80 / Test 200** | `cv_folds_v2/fold_0.json`, training log |
| References `src/cross_validation.py` | **Deleted from git** | `git status` |
| References `data/cv_folds/` | **Correct path is `data/cv_folds_v2/`** | `config.yaml` |

---

### FINDING-9 ℹ️ INFO: `src/train_cv_fold.py` has no `--accumulation_steps` CLI argument

```python
# src/train_cv_fold.py argparse (lines 411–444):
# Arguments defined: --fold_idx, --fold_dir, --output_dir, --epochs,
#                    --batch_size, --lr, --hidden_channels, --num_layers, --device
# MISSING: --accumulation_steps
```

The `train_fold()` function accepts `accumulation_steps` as a parameter but the CLI does not expose it. `train_all_folds_v3.sh` passes `--accumulation_steps 1` which is silently **ignored** — the script calls `train_fold()` without that argument because argparse never parsed it. This means `accumulation_steps` always falls through to the config/fallback value.

Since v3.sh passes `--accumulation_steps 1` but argparse ignores it, the actual value used comes from `config.get('model.training.accumulation_steps', 4)` = **2** (from config.yaml). So the true effective batch for binary_v3 is **24 × 2 = 48**, not 24.

> **Note:** This contradicts FINDING-5 and requires verification by checking the saved `results.json` which records the `accumulation_steps` used.

---

## PART 3 — DATA LINEAGE: SCRIPT ACTIVITY MAP

### Active Pipeline (files that are called by a main orchestrator):

```
ENTRY POINT
    scripts/train_all_folds_v3.sh
        └── python src/train_cv_fold.py --fold_idx N
                ├── from config import get_config          [src/config.py] ✅
                ├── from dataset import BraTSGraphDataset  [src/dataset.py] ✅
                ├── from gnn_model import TumorSegmentationGNN  [src/gnn_model.py] ✅
                └── from cross_validation import load_fold_data
                        └── src/cross_validation.py ← GIT DELETED, disk-only ⚠️

    python src/inference_ensemble.py
        ├── from config import get_config   ✅
        ├── from gnn_model import ...       ✅
        └── from dataset import ...         ✅

    bash scripts/run_all_ablations.sh
        └── python scripts/train_ablation.py --variant X
                └── [calls train_fold() from src/train_cv_fold.py internally] ✅

    python scripts/benchmark_two_scenarios.py
        ├── from gnn_model import ...       ✅
        ├── from dataset import ...         ✅
        └── from graph_construction import GraphBuilder  ✅

    python scripts/evaluate_brats2023.py
        ├── from gnn_model import ...       ✅
        ├── from dataset import ...         ✅
        └── CHECKPOINT_DIR = "checkpoints/binary_v2"  ← WRONG ⚠️

    python scripts/generate_corrected_figures.py
        └── [reads hardcoded values from embedded dict — no src imports] ⚠️
```

### Orphaned Scripts (exist but are called by nothing):

| File | Why Orphaned |
|---|---|
| `train_all_folds.sh` (root) | Duplicate of v3.sh; result-reader points to deleted checkpoints |
| `src/evaluation.py` | NIfTI-based, no caller |
| `src/visualization.py` | NIfTI-based, no caller |
| `src/aggregate_cv_results.py` | Old hardcoded paths, no caller |
| `src/evaluate_per_region.py` | 12-feature assumption, no caller |
| `src/generate_qualitative_results.py` | Old checkpoint paths, no caller |
| `scripts/paranoid_audit.py` | One-off tool, never called by pipeline |
| `scripts/train_unet_baseline.py` | Nov 26. Exists but no .sh script calls it |
| `verify_timing.py` (root) | Mar 12. Ad-hoc verification script, not in pipeline |
| `scripts/compute_ttest.py` | Mar 12. Ad-hoc stats, not in pipeline |
| `scripts/compute_dimensionality_reduction.py` | Mar 12. Ad-hoc analysis |
| `scripts/generate_new_figures.py` | Mar 13. Older figure script, superseded by `generate_corrected_figures.py` |

---

## PART 4 — PARAMETER EXTRACTION: THE THREE-WAY COMPARISON

| Parameter | `config.yaml` | `train_cv_fold.py` (actual code) | `PIPELINE_DOCUMENTATION.md` | Match? |
|---|---|---|---|---|
| Architecture | `graphsage` | reads from config ✅ | GraphSAGE ✅ | ✅ |
| Hidden channels | `256` | reads from config ✅ | 256 ✅ | ✅ |
| GNN layers | `5` | reads from config ✅ | 5 ✅ | ✅ |
| Input features | `15` | reads from first batch ✅ | Not stated | ✅ |
| Dropout | `0.1` | hardcoded `0.1` in model call (line 268) | Not stated | ✅ |
| Batch size | `24` | reads from config ✅ | **32** ❌ | ❌ |
| Learning rate | `0.001` | reads from config ✅ | 0.001 ✅ | ✅ |
| **Weight decay** | `0.00001` | **hardcoded `0.01`** ❌ | Not stated | ❌ |
| **Accumulation steps** | `2` | **v3.sh passes arg but CLI doesn't exist → config value used** | Not stated | ⚠️ |
| Max epochs | `50` | reads from config ✅ | 50 ✅ | ✅ |
| Early stopping | `10` | hardcoded `patience` in training loop (not read from config!) | Not stated | ⚠️ |
| Loss function | Not in config | **BCEWithLogitsLoss only** ❌ | "BCE + Dice" ❌ | ❌ |
| Deterministic | Not in config | `deterministic=False` | **"deterministic mode"** ❌ | ❌ |
| n_folds | `5` | reads from config ✅ | 5 ✅ | ✅ |
| Fold split sizes | 720/80/200 | reads from cv_folds_v2 ✅ | **900/100/251** ❌ | ❌ |
| Graph n_superpixels | `200` | reads from config ✅ | Not stated | ✅ |
| Graph knn_k | `3` | reads from config ✅ | Not stated | ✅ |

---

## PART 5 — BraTS 2021 vs BraTS 2023 SEPARATION CHECK

### Are the two flows strictly separated?

| Stage | BraTS 2021 | BraTS 2023 | Separated? |
|---|---|---|---|
| Preprocessing script | `src/preprocessing.py` | `scripts/preprocess_brats2023.py` | ✅ Separate files |
| Preprocessed data dir | `data/preprocessed/` | `data/preprocessed_brats2023/` | ✅ Separate dirs |
| Graph data dir | `data/graphs/` | `data/graphs_brats2023/` | ✅ Separate dirs |
| Label remapping | 0/1/2/4 native | Label 3→4 remapped in preprocess script | ✅ Handled |
| Training | `src/train_cv_fold.py` | ❌ **Not trained on 2023** (zero-shot) | ✅ No contamination |
| Evaluation | `src/inference_ensemble.py` | `scripts/evaluate_brats2023.py` | ✅ Separate |
| BinaryTransform | Same class used for both | Both call `(y > 0).float()` | ✅ Compatible |
| Checkpoint used | `binary_v3` (production) | `binary_v2` (hardcoded in eval script) | ⚠️ Inconsistency |
| Results dir | `research_results/ensemble_v2/` | `research_results/brats2023_evaluation/` | ✅ Separate |

**Verdict:** The two pipelines are cleanly separated with no data collisions. The only issue is the `evaluate_brats2023.py` checkpoint reference pointing to `binary_v2` instead of `binary_v3`.

---

## PART 6 — THE TRUTH: 1-PAGE TECHNICAL SPECIFICATION

**As the code actually exists and ran:**

```
════════════════════════════════════════════════════════════════════
PROJECT: BraTS GNN Segmentation
VERIFIED: 2026-03-14 via full source + log + JSON forensic audit
════════════════════════════════════════════════════════════════════

DATASET
  BraTS 2021: 1,251 patients, 4 modalities (T1, T1ce, T2, FLAIR)
  Split: 251 held-out (sealed) | 1,000 CV pool
  CV: 5 folds from cv_folds_v2/ — 720 train / 80 val / 200 test
  Stratification: patient-level by tumour volume quartile

PREPROCESSING (src/preprocessing.py)
  Normalization: Z-score per modality per patient
  Target size: 240×240×155, isotropic 1.0mm spacing
  Slice extraction: 200 tumour-priority slices/patient
  Min brain pixels: 1,000

GRAPH CONSTRUCTION (src/graph_construction.py)
  Backend: fast_slic (fallback: skimage SLIC)
  Superpixels: 200 per 2D slice (n_superpixels=200)
  Node features: 15-dim (morphological×5, T1ce stats×4, multimodal×4, cross-slice×2)
  Intra-slice edges: SLIC boundary adjacency
  Inter-slice edges: IoU threshold=0.1 + KNN k=3
  SLIC params: sigma=0.3, compactness=0.1, max_iter=30

MODEL (src/gnn_model.py → TumorSegmentationGNN)
  GNN type: GraphSAGE (SAGEConv)
  Layers: 5
  Hidden dim: 256
  Input features: 15 (enforced by assertion)
  Output: node-level binary logit (1 per superpixel)
  Classifier: Linear(64→32) + ReLU + Dropout(0.1) + Linear(32→1)
  Total parameters: 439,041

TRAINING (src/train_cv_fold.py + scripts/train_all_folds_v3.sh)
  Loss: BCEWithLogitsLoss [NOT CombinedLoss — see FINDING-1]
  Optimizer: AdamW, lr=0.001, weight_decay=0.01 [NOT 0.00001 — see FINDING-2]
  Scheduler: OneCycleLR, cosine annealing, pct_start=0.1
  Batch size: 24 (from config via v3.sh)
  Accumulation steps: 2 (from config.yaml — CLI arg missing, see FINDING-9)
  → Effective batch: 24 × 2 = 48
  Epochs: 50 max, early stopping patience=10
  AMP: enabled (GradScaler, autocast)
  torch.compile: mode='default'
  cuDNN: benchmark=True, deterministic=False
  Seed: 42 (for data splits only — training is non-deterministic)
  Checkpoints: checkpoints/binary_v3/fold_{0..4}/best_model.pth

RESULTS (all from binary_v3 unless noted)
  5-fold CV Dice:     Fold 0: 88.72% | 1: 90.48% | 2: 90.31% | 3: 90.13% | 4: 90.47%
  CV Mean:            90.02% ± 0.74% (sample std)
  Ensemble (held-out 251 pts): 91.41% Dice
  Additional metrics: Acc 99.14%, Sens 87.77%, Spec 99.76%, Prec 95.52%
  Inference time:     74ms (scenario 1), 1,570ms end-to-end (scenario 2) [binary_v2]
  Peak GPU memory:    11.06 MB
  Model size:         5.07 MB
  Params vs U-Net:    439K vs 68M → 155× fewer

BraTS 2023 ZERO-SHOT (from binary_v2 — see FINDING-3)
  Patients: 1,245 | Graphs: 1,662
  Dice: 89.21% ± 11.14% | Generalisation gap: 2.20%

════════════════════════════════════════════════════════════════════
```

---

## PART 7 — PRIORITISED FIX LIST

Fix these in order. Each fix is independent except where noted.

| # | Severity | Finding | Fix | Affects Paper? |
|---|---|---|---|---|
| 1 | 🔴 | FINDING-1: Loss function | Decide: keep BCE-only or add CombinedLoss and retrain. Update thesis accordingly. | Yes — framework section |
| 2 | 🔴 | FINDING-2: weight_decay | Fix `train_cv_fold.py` line 285 to read from config. Update config.yaml to `weight_decay: 0.01`. | Yes — hyperparameter table |
| 3 | 🔴 | FINDING-3: BraTS 2023 checkpoint | Update `evaluate_brats2023.py` line 38 to `checkpoints/binary_v3`. Re-run evaluation. | Yes — generalisation result |
| 4 | 🔴 | FINDING-4: Dummy graph 12 features | Fix `dataset.py` line 152: change `12` to `15`. | No — latent defect only |
| 5 | 🟡 | FINDING-5: accumulation_steps | Add `--accumulation_steps` to argparse in `train_cv_fold.py`. Update config.yaml comment. | Yes — hyperparameter table |
| 6 | 🟡 | FINDING-6: Ablation table | Re-run `baseline` and `deeper_network` (layers_6) with `scripts/train_ablation.py`. | Yes — ablation table |
| 7 | 🟡 | FINDING-7: Timing benchmark checkpoint | Re-run `scripts/benchmark_two_scenarios.py` with `binary_v3/fold_0`. | Minor |
| 8 | ℹ️ | FINDING-8: PIPELINE_DOCUMENTATION.md | Update all stale numbers and references. | No — internal doc |
| 9 | ℹ️ | FINDING-9: CLI missing arg | Add `--accumulation_steps` to argparse (done as part of Fix 5). | No — usability |

---

*End of Report. Generated: 2026-03-14.*
