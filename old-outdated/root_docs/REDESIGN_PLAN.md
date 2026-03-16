# BraTS GNN Redesign — Full Project Plan
**Version 2.0 | From 2D Binary → 3D Multi-class**
**Date:** March 2026

---

## The One-Line Goal

Build a 3D hierarchical GNN for multi-class brain tumor segmentation on BraTS 2021/2023 that is **~40× smaller than Swin-UNETR**, runs **end-to-end in under 1 second**, and achieves **WT ≥91%, TC ≥82%, ET ≥73% Dice** — with a clean, leakage-free evaluation protocol.

---

## What Changes vs. What Stays

### KEEP (reuse directly)
| Component | File | Status |
|---|---|---|
| Preprocessing pipeline | `src/dataset.py` | Keep skull-strip, normalise, slice selection logic |
| Training loop structure | `src/train_cv_fold.py` | Keep fold loop, early stopping, logging |
| Config system | `src/config.py` + `config.yaml` | Keep lazy loading pattern |
| Benchmarking scripts | `scripts/benchmark_speed.py` | Extend with end-to-end timing |
| Visualisation scripts | `scripts/` | Keep all, extend for multi-class |

### CHANGE (core redesign)
| Component | File | What changes |
|---|---|---|
| Graph construction | `src/graph_construction.py` | 2D SLIC → 3D SLIC supervoxels |
| Node features | `src/graph_construction.py` | 15 features → 40 features |
| GNN model | `src/gnn_model.py` | Flat GraphSAGE → Hierarchical GNN |
| Output heads | `src/gnn_model.py` | 1 binary → 3 binary heads (WT/TC/ET) |
| Loss function | `src/train_cv_fold.py` | BCE → Weighted multi-task BCE + Dice |
| Data split | `src/cross_validation.py` | Add 20% held-out test set before CV |
| Ensemble eval | `src/inference_ensemble.py` | Fix leakage — evaluate on held-out only |
| Labels | `src/dataset.py` | Binary → 3-class labels |

### ADD (new files)
| File | Purpose |
|---|---|
| `src/graph_construction_3d.py` | 3D SLIC supervoxel graph builder |
| `src/hierarchical_gnn.py` | Hierarchical GNN with graph pooling |
| `src/losses.py` | Combined BCE + Dice loss, per-class weighting |
| `src/progressive_trainer.py` | Phase-based training (WT → TC → ET) |
| `scripts/evaluate_brats2023.py` | Zero-shot cross-dataset evaluation |
| `scripts/benchmark_endtoend.py` | Full pipeline timing (raw NIfTI → mask) |

---

## Phase-by-Phase Implementation Plan

---

### PHASE 0 — Data Split (Day 1, non-negotiable first step)
**Goal:** Create the held-out test set before any training happens.

```
1,251 patients
├── 250 patients → held_out_test.json   ← sealed, never touched until final eval
└── 1,001 patients → cv_pool.json       ← all CV runs on this only
```

**Steps:**
1. Stratify by tumour volume quartile to ensure test set is representative
2. Save patient IDs to `data/splits/held_out_test.json`
3. Save remaining to `data/splits/cv_pool.json`
4. **Never** use held_out_test patients during any training or ablation

**Files to create/modify:**
- New script: `scripts/create_data_split.py`
- Output: `data/splits/held_out_test.json`, `data/splits/cv_pool.json`

**Time: 2 hours**

---

### PHASE 1 — 3D Supervoxel Graph Construction (Days 2–5)
**Goal:** Replace 2D SLIC slices with 3D SLIC supervoxels.

**Key design decisions:**
- **~15,000–20,000 supervoxels per patient** (vs ~10,000 nodes currently)
- **Compactness parameter**: 0.1 (favours boundary adherence over shape regularity)
- **3D adjacency**: two supervoxels are connected if they share a face/edge in 3D space
- **Pre-compute and cache** all graphs to disk (same pattern as current pipeline)

**Node feature vector (40 dimensions):**
```
Intensity per modality × 4 modalities:
  [mean, std, min, max, median, skewness] × 4 = 24 features

Gradient magnitude per modality:
  [mean_gradient] × 4 = 4 features

Spatial position:
  [x_norm, y_norm, z_norm] = 3 features

Geometry:
  [volume_norm, elongation] = 2 features

Enhancement ratio:
  [T1ce_mean / (T1_mean + 1e-6)] = 1 feature  ← key for ET detection

Tumour prior:
  [distance_to_brain_centre_norm] = 1 feature  ← tumours cluster centrally

Total: 35 features (extend to 40 with texture if time permits)
```

**Node label assignment (multi-class):**
```python
# For each supervoxel, majority vote of constituent voxels:
# Label 0 = background    (BraTS label 0)
# Label 1 = whole tumour  (BraTS labels 1+2+4)
# Label 2 = tumour core   (BraTS labels 1+4)
# Label 3 = enhancing     (BraTS label 4)
# → 3 binary heads: WT=(label>=1), TC=(label>=2), ET=(label==3)
```

**Files:**
- `src/graph_construction_3d.py` — new file, ~300 lines
- `config.yaml` — add `data_graphs_3d` path

**Time: 3 days** (1 day coding, 2 days running preprocessing for 1,251 patients)

---

### PHASE 2 — Hierarchical GNN Model (Days 6–9)
**Goal:** Replace flat GraphSAGE with a 2-level hierarchical model.

**Architecture:**
```
Input: 3D supervoxel graph (~17,500 nodes, 35 features each)
       │
   ┌───▼────────────────────────────────────────┐
   │  ENCODER — Level 1 (Fine Graph)            │
   │  GraphSAGE × 3 layers, 256 hidden          │
   │  Output: node embeddings [N × 256]         │
   └───────────────────┬────────────────────────┘
                       │  DiffPool (cluster N → N/10)
   ┌───────────────────▼────────────────────────┐
   │  ENCODER — Level 2 (Coarse Graph)          │
   │  GraphSAGE × 2 layers, 256 hidden          │
   │  Output: cluster embeddings [N/10 × 256]   │
   └───────────────────┬────────────────────────┘
                       │  Unpool + skip connection from Level 1
   ┌───────────────────▼────────────────────────┐
   │  DECODER — Back to Level 1                 │
   │  GraphSAGE × 2 layers, 128 hidden          │
   │  Output: refined node embeddings [N × 128] │
   └───────────────────┬────────────────────────┘
                       │
        ┌──────────────┼──────────────┐
        ▼              ▼              ▼
   Head: WT       Head: TC       Head: ET
   Linear(128,1)  Linear(128,1)  Linear(128,1)
   P(whole tumor) P(tumor core)  P(enhancing)
```

**Why this structure:**
- Level 1 captures local voxel-level features (important for ET boundaries)
- Level 2 captures regional context (important for TC shape understanding)
- Skip connection preserves fine detail lost during pooling (same principle as U-Net)
- 3 separate heads allow independent loss weighting per sub-region

**Expected parameters: ~1.5–2M** (still ~30–40× fewer than Swin-UNETR)

**Files:**
- `src/hierarchical_gnn.py` — new file, ~250 lines
- `src/losses.py` — new file, ~80 lines

**Loss function:**
```python
# Per-head weighted BCE + Dice loss
L_WT = BCE(p_wt, y_wt, pos_weight=3.0)   + dice_loss(p_wt, y_wt)
L_TC = BCE(p_tc, y_tc, pos_weight=8.0)   + dice_loss(p_tc, y_tc)
L_ET = BCE(p_et, y_et, pos_weight=25.0)  + dice_loss(p_et, y_et)

L_total = L_WT + L_TC + (3.0 × L_ET)    # ET weighted 3× overall
```

**Time: 3 days**

---

### PHASE 3 — Progressive Training (Days 10–14)
**Goal:** Train WT first, then add TC, then ET. Prevents ET from being crushed by gradient imbalance at the start.

**Training schedule:**
```
Phase A  (epochs 1–20):   Only L_WT active
         → Model learns "where is the tumour?"
         → Early stopping patience = 5

Phase B  (epochs 21–40):  L_WT + L_TC active
         → Model learns "what is the core?"
         → Initialise from Phase A checkpoint
         → Early stopping patience = 5

Phase C  (epochs 41–60):  L_WT + L_TC + L_ET active
         → Model learns "where is enhancement?"
         → Initialise from Phase B checkpoint
         → Early stopping patience = 10 (ET harder to converge)
```

**Cross-validation:**
- Same 5-fold structure on the 1,001-patient CV pool
- Each fold: ~5–7 hours compute (longer than before due to 3D graphs)
- Total training: ~35 hours for all 5 folds

**Files:**
- `src/progressive_trainer.py` — new file, ~200 lines
- `src/train_cv_fold.py` — modify to support progressive training

**Time: 5 days** (1 day coding, 4 days training compute)

---

### PHASE 4 — Ablation Study (Days 15–18)
**Goal:** Justify every architectural choice with evidence.

| Experiment | Variable | Options to test |
|---|---|---|
| Supervoxel count | granularity | 8K, 15K (optimal), 25K nodes |
| Feature set | node features | 15 (current), 25, 35, 40 |
| Hierarchy depth | levels | flat (1-level), 2-level (proposed), 3-level |
| Loss weighting | ET weight | 1×, 2×, 3× (proposed), 5× |
| Batch size | batch | 16, 32 (proposed), 48, 64 |
| Progressive training | schedule | all-at-once vs phased (proposed) |

Each experiment = 1 training run (~6 hours). Run in parallel if multiple GPUs available.

**Time: 3 days** (1 day setup, 2 days compute)

---

### PHASE 5 — Ensemble + Final Evaluation (Days 19–21)
**Goal:** Clean ensemble evaluation on the 250-patient held-out set.

**Procedure:**
1. Load all 5 fold models (trained on cv_pool only)
2. Run inference on all 250 held-out patients
3. Average logits across 5 models (soft voting, per head)
4. Apply threshold 0.5 per head
5. Report WT/TC/ET Dice, sensitivity, specificity, precision
6. Statistical tests: paired t-test vs U-Net and nnU-Net on same held-out set

**Ensemble is valid because:** none of the 5 models ever saw the held-out patients.

**Time: 2 days**

---

### PHASE 6 — BraTS 2023 Cross-Dataset Validation (Days 22–25)
**Goal:** Show the model generalises to unseen data distribution without retraining.

**What BraTS 2023 has:**
- Same 4 modalities (T1, T1ce, T2, FLAIR)
- Different label scheme: label 3 instead of label 4 for ET (1-line fix)
- Different institutions, scanners, acquisition protocols

**Procedure:**
1. Download BraTS 2023 glioma task (~100GB, requires Synapse account)
2. Run same preprocessing pipeline (skull-strip, normalise)
3. Run same 3D SLIC graph construction
4. Run ensemble inference — no retraining
5. Report generalisation gap: BraTS 2021 → BraTS 2023 performance drop

**Expected:** ~2–4% drop in Dice — this is normal and honest to report.

**Files:**
- `scripts/evaluate_brats2023.py` — new script, ~150 lines

**Time: 3 days** (mostly data download and preprocessing)

---

### PHASE 7 — Timing Benchmarks (Day 26)
**Goal:** Honest two-scenario timing report.

**Scenario 1 — Preprocessed (inference only):**
```
Input:  pre-built 3D supervoxel graph
Output: WT/TC/ET mask
Time:   X ms (expected: 20–50ms)
```

**Scenario 2 — Raw MRI (end-to-end):**
```
Input:  raw NIfTI files (T1, T1ce, T2, FLAIR)
Steps:  skull-strip → normalise → SLIC3D → graph build → inference → reconstruct mask
Time:   X seconds (expected: 0.5–2s)
Compare to U-Net end-to-end: ~3.5s → still faster
```

Both numbers go in the paper. No cherry-picking.

**Files:**
- `scripts/benchmark_endtoend.py` — new script, ~100 lines

**Time: 1 day**

---

### PHASE 8 — Paper Writing (Days 27–35)
**Goal:** Update Paper_Draft.md with clean numbers and submit-ready text.

**Sections to rewrite:**
- Abstract — new numbers, multi-class results
- Section 4 (Methodology) — 3D supervoxels, hierarchical GNN, progressive training
- Section 6 (Results) — WT/TC/ET tables, clean ensemble, BraTS 2023
- Section 7 (Ablation) — complete table with real numbers
- Update all efficiency figures with two-scenario timing
- Add failure case analysis (find 2–3 patients where ET was missed, explain why)

**Time: 1 week**

---

## Full Timeline

```
Week 1:  Phase 0 + Phase 1   → Data split + 3D graph construction
Week 2:  Phase 2 + Phase 3   → Model + training (compute running overnight)
Week 3:  Phase 4             → Ablation experiments
Week 4:  Phase 5 + Phase 6   → Final evaluation + BraTS 2023
Week 5:  Phase 7 + Phase 8   → Timing + paper writing
Week 6:  Phase 8 continued   → Revisions, figures, PDF
```

**Total: 6 weeks**

---

## Expected Final Results

| Metric | V1 (current) | V2 (redesign target) | Heavyweight SOTA |
|---|---|---|---|
| WT Dice | 90.38% | 91–92% | 93% |
| TC Dice | — | 82–85% | 87% |
| ET Dice | — | 73–78% | 82% |
| Parameters | 0.44M | ~1.5M | 31–62M |
| Inference (only) | 12.7ms | ~30ms | ~95–180ms |
| End-to-end | unknown | ~1s | ~3.5–5s |
| Ensemble valid? | No (leakage) | Yes (held-out) | — |
| Multi-dataset | No | BraTS 2021 + 2023 | — |

---

## Key Research Claims (for the paper)

1. **Efficiency**: 20–40× fewer parameters than all compared methods, with <2% WT Dice gap vs best SOTA
2. **Speed**: End-to-end inference under 1 second (vs 3.5–5s for CNN/transformer baselines)
3. **Edge deployment**: 3–6 MB total model, deployable on devices with 2.1 GB GPU
4. **Generalisation**: Validated on two independent datasets (BraTS 2021 + 2023)
5. **Novel finding**: Batch size sensitivity in graph-based segmentation (extend ablation)
6. **Clean evaluation**: Proper held-out test set — all ensemble numbers leakage-free

---

## Can Claude shorten the timeline?

### Honest breakdown:

| Task | Without Claude | With Claude | Saving |
|---|---|---|---|
| Writing boilerplate code (data loaders, training loops, config) | 3–4 days | 4–6 hours | ~3 days |
| Debugging shape/dimension errors in new graph code | 1–2 days | 2–4 hours | ~1.5 days |
| Writing the 3D SLIC graph construction script | 1 day | 1–2 hours | ~6 hours |
| Writing the hierarchical GNN architecture | 1 day | 2–3 hours | ~6 hours |
| Writing loss functions + progressive trainer | 1 day | 1–2 hours | ~6 hours |
| Writing evaluation + benchmarking scripts | 1 day | 1–2 hours | ~6 hours |
| Paper writing (drafting sections, tables, captions) | 1 week | 1–2 days | ~5 days |
| Ablation experiment setup | 1 day | 2–3 hours | ~6 hours |
| **Total coding + writing** | **~18 days** | **~5 days** | **~13 days** |

### What Claude CANNOT shorten:
| Task | Time | Why |
|---|---|---|
| Training compute (5 folds × ~6 hours) | ~35 hours | GPU-bound, cannot be parallelised easily |
| BraTS 2023 data download | 1–2 days | Network-bound |
| 3D graph preprocessing (1,251 patients) | 1–2 days | CPU-bound |
| Your review + validation of results | ~1 week | Requires your judgement |

### Realistic timeline with Claude:

```
Without Claude: 6 weeks
With Claude:    3.5–4 weeks
```

The ~2 week saving comes entirely from code generation and paper writing. The compute time is fixed — GPUs train at the same speed regardless.

### How to use Claude most effectively:

- **Give Claude one task at a time** with clear input/output specs
- **Always validate the code** before running on the full dataset — run on 5 patients first
- **Use Claude for paper writing** — give it your result numbers, it writes the narrative
- **Do NOT blindly trust** numeric claims Claude makes about expected results — those are estimates, your actual trained numbers are the truth
- **Use Claude to debug** — paste error messages directly, it finds root causes fast

---

## Files to Create (in order)

```
Week 1:
  scripts/create_data_split.py
  src/graph_construction_3d.py
  config.yaml  (add new paths)

Week 2:
  src/hierarchical_gnn.py
  src/losses.py
  src/progressive_trainer.py
  src/train_cv_fold.py  (modified)

Week 3:
  scripts/run_ablation.py

Week 4:
  src/inference_ensemble.py  (fixed)
  scripts/evaluate_brats2023.py

Week 5:
  scripts/benchmark_endtoend.py
  paperWriting/Paper_Draft.md  (full rewrite)
```

---

*This plan assumes 1 GPU available for training. If multiple GPUs are available, folds can be parallelised and Week 2–3 compute time halves.*
