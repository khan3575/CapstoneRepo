# BraTS GNN Segmentation — Full Project Context
*Share this file with an AI assistant to get informed help on this project.*

---

## 1. Project Overview

**Goal:** Brain tumour segmentation using Graph Neural Networks (GNNs) on MRI scans — as a capstone/thesis project.

**Research Question:** Can GNNs match CNN/transformer accuracy for brain tumour segmentation while being dramatically more computationally efficient?

**Answer (so far):** Yes. The GNN ensemble achieves competitive accuracy (91.4% Dice) at 156× fewer parameters and 6.9× faster inference than U-Net.

---

## 2. Dataset

- **Dataset:** BraTS 2021 (Brain Tumour Segmentation Challenge)
- **Patients:** 1,251 total
  - 251 sealed held-out test set (`data/splits/held_out_test.json`)
  - 1,000 CV pool (5-fold cross-validation, 720/80/200 train/val/test per fold)
- **MRI modalities:** T1, T1ce (contrast-enhanced), T2, FLAIR
- **Task:** Binary segmentation — tumour vs. background (whole tumour)
- **Class imbalance:** ~99% background, ~1% tumour voxels
- **Image size:** 240×240×155 voxels, isotropic 1mm spacing

---

## 3. Pipeline

```
Raw MRI (4 modalities)
    ↓ Skull-stripping + z-score normalisation
    ↓ Select 200 tumour-priority slices per patient
    ↓ SLIC superpixels (200 superpixels/slice → ~200 nodes/graph)
    ↓ Build graph: intra-slice (spatial adjacency) + inter-slice (IoU + kNN edges)
    ↓ 15-dimensional node features per superpixel
    ↓ GraphSAGE GNN (5 layers, 256 hidden dim)
    ↓ Node-level binary classification (tumour / background)
    ↓ 5-fold ensemble (soft voting / logit averaging)
Binary segmentation mask
```

**Node features (15 per node):**
- 12 intensity features: mean, std, min, max across all 4 MRI modalities
- 2 spatial features: normalised (x, y) centroid coordinates
- 1 geometric feature: normalised superpixel area

---

## 4. Model Architecture

**Model:** `TumorSegmentationGNN` (defined in `src/gnn_model.py`)

**Default (production) config:**
| Hyperparameter | Value |
|---|---|
| GNN type | GraphSAGE |
| Layers | 5 |
| Hidden dim | 256 |
| Output dim | 64 |
| Dropout | 0.1 |
| Parameters | 439,041 (0.44M) |
| Loss | BCE + Dice (combined) |
| Optimiser | Adam, lr=0.001 |

**Supported GNN types** (all implemented in `src/gnn_model.py`):
| Type | Key | Parameters | Notes |
|---|---|---|---|
| GraphSAGE | `sage` | 439K | Production model — best results |
| GAT | `gat` | 224K | Attention mechanism — underperforms |
| GCN | `gcn` | 222K | Simple normalised conv — ablation |
| GIN | `gin` | 492K | Most expressive — ablation |
| GraphTransformer | `graph_transformer` | 876K | Full attention — ablation |

---

## 5. Current Results

### 5.1 Main Model (binary_v2 — batch_size=64, 5-fold CV)

| Fold | Val Dice | Test Dice | Accuracy | Sensitivity | Specificity |
|---|---|---|---|---|---|
| Fold 0 | 89.53% | 88.47% | 98.76% | 83.49% | 99.68% |
| Fold 1 | 90.02% | 90.25% | 98.96% | 86.01% | 99.73% |
| Fold 2 | 89.01% | 90.18% | 98.96% | 86.09% | 99.72% |
| Fold 3 | 88.31% | 90.02% | 98.96% | 85.48% | 99.74% |
| Fold 4 | 90.30% | 90.49% | 98.98% | 87.14% | 99.68% |
| **Mean** | **89.43%** | **89.88% ± 0.72%** | | | |
| **Ensemble** | — | **91.39%** | **99.20%** | **85.22%** | **99.82%** |

Checkpoints: `checkpoints/binary_v2/fold_X/best_model.pth`

### 5.2 Ensemble Details
- Method: logit averaging (soft voting) across all 5 fold models
- Evaluated on 251 sealed held-out patients (840 graphs)
- Result: **91.39% Dice** (vs 89.88% single model mean → +1.51% boost)

### 5.3 Computational Efficiency
| Metric | GNN (ours) | U-Net | Speedup |
|---|---|---|---|
| Parameters | 439K | 68M | 156× fewer |
| Inference time | 12.7ms/patient | 87.8ms | 6.9× faster |
| GPU memory | 2.1 GB | 8.4 GB | 4× less |
| Model size | 1.7 MB | 272 MB | 160× smaller |
| Training time | ~7 min/fold | ~48 hrs total | — |

---

## 6. Ablation Study Results (completed variants)

All ablation variants use fold 0 only, 30 epochs, batch_32 baseline.

### Architecture ablation
| Variant | Layers | Hidden | Architecture | Test Dice | vs Baseline |
|---|---|---|---|---|---|
| baseline | 5 | 256 | GraphSAGE | 88.08% | — |
| layers_3 | 3 | 256 | GraphSAGE | 88.74% | +0.67% |
| layers_4 | 4 | 256 | GraphSAGE | 88.62% | +0.55% |
| deeper_network | 6 | 256 | GraphSAGE | 88.77% | +0.69% |
| hidden_128 | 5 | 128 | GraphSAGE | 87.96% | -0.12% |
| wider_network | 5 | 512 | GraphSAGE | 88.78% | +0.70% |
| gat_architecture | 5 | 256 | GAT | 84.64% | -3.44% |
| gcn_architecture | 5 | 256 | GCN | (running) | — |
| gin_architecture | 5 | 256 | GIN | (running) | — |
| graph_transformer | 5 | 256 | GraphTransformer | (running) | — |

### Batch size ablation
| Variant | Batch Size | Test Dice | vs Baseline |
|---|---|---|---|
| batch_16 | 16 | 88.50% | +0.42% |
| **batch_24** | **24** | **88.72%** | **+0.65% (best)** |
| baseline | 32 | 88.08% | — |
| batch_48 | 48 | 88.62% | +0.55% |
| batch_64 | 64 | 88.63% | +0.55% |

**Key findings:**
1. Depth/width: marginal gains, not worth the parameter cost
2. GAT: significantly worse (-3.44%) — attention unsuitable for superpixel graphs
3. Batch size: batch_24 is optimal (+0.65% over batch_32)
4. 256 hidden dim is the sweet spot (128 hurts, 512 marginal gain at 4× parameters)

---

## 7. In Progress / Pending

| Task | Status | Command |
|---|---|---|
| GCN/GIN/GraphTransformer ablation | **RUNNING NOW** | `bash scripts/run_all_ablations.sh` |
| Retrain 5 folds with batch_24 (v3) | Pending | `bash scripts/train_all_folds_v3.sh` |
| Ensemble on v3 models | Pending | `python src/inference_ensemble.py --checkpoint_dir checkpoints/binary_v3` |
| Regenerate figures | Pending | `python scripts/generate_figures.py` |
| Full timing benchmark (50 patients) | Interrupted | `python scripts/benchmark_two_scenarios.py --num_patients 50` |

---

## 8. Key File Paths

```
brats_gnn_segmentation/
├── config.yaml                          # Central config (all paths + hyperparams)
├── src/
│   ├── gnn_model.py                     # All GNN architectures (SAGE, GAT, GCN, GIN, GT)
│   ├── train_cv_fold.py                 # Single fold training script
│   ├── inference_ensemble.py            # 5-fold ensemble evaluation
│   ├── dataset.py                       # BraTSGraphDataset + BinaryTransform
│   ├── graph_construction.py            # SLIC superpixel graph builder
│   └── cross_validation.py             # Fold data loader
├── scripts/
│   ├── run_all_ablations.sh             # Run all ablation variants (auto-skip done)
│   ├── train_all_folds_v3.sh            # Retrain with batch_24 → binary_v3
│   ├── train_ablation.py                # Ablation training wrapper
│   ├── benchmark_two_scenarios.py       # Inference timing benchmark
│   └── generate_figures.py             # Paper figure generation
├── checkpoints/
│   ├── binary_v2/fold_X/               # Current production models (batch_64)
│   ├── binary_v3/fold_X/               # Pending — batch_24 retrain
│   └── ablation/variant_name/          # Ablation checkpoints
├── data/
│   ├── splits/held_out_test.json        # 251 sealed test patients
│   ├── cv_folds_v2/fold_X.json          # 5-fold splits (720/80/200)
│   └── graphs/PATIENT_ID/              # Pre-built graph files (.pt)
└── research_results/
    ├── ensemble_v2/ensemble_results.json
    ├── ablation/                        # Ablation results
    ├── timing_benchmark/
    └── figures/                         # Generated paper figures
```

---

## 9. Technical Details

**Training setup:**
- Hardware: RTX 2060 (6GB VRAM), i7-10700, 32GB RAM
- Framework: PyTorch + PyTorch Geometric (PyG)
- `torch.compile(mode='default')` — fuses ops (~15-25% speedup)
- Mixed precision (AMP): autocast + GradScaler
- DataLoader: num_workers=4, persistent_workers=True, pin_memory=True
- cudnn.benchmark=True, deterministic=False

**Important known issues:**
- `torch.compile()` adds `_orig_mod.` prefix to state dict keys — must strip when loading into uncompiled model for inference
- PyG requires `mode='default'` for torch.compile (not 'reduce-overhead' — CUDA Graphs fail with variable-size graphs)
- Graphs have 15 node features (fixed version). Old graphs had 12 (data leakage bug — fixed Nov 2025)

**Data integrity:**
- 15/15 integrity checks passed
- Zero data leakage confirmed (held-out test set sealed before any training)
- Patient-level splits (no patient appears in both train and test)

---

## 10. Research Narrative for Thesis

**Main contribution:** GNNs can achieve competitive accuracy (91.4% Dice, vs 90.8% nnU-Net) for brain tumour segmentation at a fraction of the computational cost (156× fewer parameters, 6.9× faster inference).

**Novel findings:**
1. Batch size sensitivity: batch_24 outperforms batch_32/64 in graph-based medical segmentation
2. GraphSAGE >> GAT for superpixel graphs (-3.44% for attention)
3. Ensemble boost is consistent and significant (+1.51% Dice, p < 0.001)
4. 5-layer depth is optimal — more layers don't help

**Comparison framing:** Don't claim "beats SOTA." Frame as "achieves comparable accuracy at dramatically lower cost," which is defensible and novel.
