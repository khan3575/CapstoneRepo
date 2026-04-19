# Comprehensive Analysis: Efficient Brain Tumour Segmentation using Graph Neural Networks on BraTS Datasets

**Team:** Sakib Khan, Rifa Sanjida, Kishor Kumar Das, Md. Mahamudul Hasan, Md. Minhajur Rahman
**Supervisor:** Mr. Shamim Ahmed, BUBT
**Sources:** `main_bubt_paper.tex` (authoritative) + `final Presentation.pdf` (27 slides)

---

## 1. Problem Statement

Brain tumour segmentation from multi-modal MRI is critical for surgical planning and treatment monitoring. Existing deep learning methods (CNNs, U-Net variants) achieve high accuracy but demand large GPU memory and long inference times, making them impractical for resource-constrained clinical settings. This work proposes a lightweight, graph-based alternative.

---

## 2. Research Objectives (4 Total)

1. Develop a GNN-based segmentation pipeline using superpixel graphs on multi-modal BraTS MRI.
2. Achieve competitive Dice scores compared to CNN/U-Net baselines.
3. Dramatically reduce computational requirements (memory, inference time, parameter count).
4. Demonstrate cross-dataset generalisation (BraTS 2021 → BraTS 2023, zero-shot).

---

## 3. Dataset

| Split | Details |
|-------|---------|
| Primary training/eval | BraTS 2021 — 1,000 patients (5-fold CV pool) |
| Sealed held-out | BraTS 2021 — 251 patients (never touched during training or fold selection) |
| Zero-shot generalisation | BraTS 2023 (different acquisition, same label schema) |

**Modalities used:** T1, T1CE, T2, FLAIR (4-channel input)
**Label:** Whole tumour binary mask (tumour vs. healthy)
**Class imbalance at superpixel level:** ~19:1 (healthy : tumour)

---

## 4. Methodology

### 4.1 Graph Construction Pipeline

**Step 1 — SLIC Superpixels**
- Applied on T1CE channel (highest tumour contrast)
- Parameters: K=200 target, compactness=0.1, sigma=0.3
- Realised superpixels per slice: ~46
- 2,284× spatial compression: 8,928,000 voxels → ~3,909 nodes per patient

**Step 2 — Paired-Slice Graph**
- Unit: 2 consecutive axial slices
- ~92 nodes and ~180 edges per graph unit

**Step 3 — Edge Types**
- Intra-slice: Region Adjacency Graph (RAG), 4-connectivity boundary adjacency
- Inter-slice: kNN (k=3), IoU > 0.1, centroid distance < 10mm

**Step 4 — Node Features (15-dimensional)**

| Group | Features |
|-------|---------|
| Intensity (8) | mean + std for T1, T1CE, T2, FLAIR |
| Spatial (4) | area, normalised area, y-centroid, x-centroid |
| Morphological (3) | perimeter, compactness, intensity range |

### 4.2 GNN Architecture — GraphSAGE

| Hyperparameter | Value |
|----------------|-------|
| Layers | 5 |
| Hidden channels | 256 |
| Output dimension | 64 |
| Total parameters | 439,041 |
| Aggregation | Mean pooling |
| Type | Inductive (can generalise to unseen graphs) |

**Why GraphSAGE?** Inductive framework — works on graphs not seen during training, enabling zero-shot cross-dataset generalisation.

### 4.3 Training Protocol

| Setting | Value |
|---------|-------|
| Optimizer | AdamW (LR=1e-3, weight decay=0.01) |
| Scheduler | OneCycleLR with 30% warmup |
| Batch size | 24 (gradient accumulation ×2 → effective batch 48) |
| Precision | AMP FP16 |
| Max epochs | 50 |
| Early stopping patience | 10 (never triggered) |
| Loss function | BCEWithLogitsLoss, positive class weight w⁺=9.0 |
| Dice smoothing ε | 10⁻⁷ |

**Hardware:** Intel i7-10700, 32GB RAM, NVIDIA RTX 2060 (6GB VRAM)
**Software:** Python 3.12, PyTorch 2.8.0 (CUDA 12.8), PyTorch Geometric 2.6.1, scikit-image 0.25.2, NumPy 2.3.3

### 4.4 Ensemble Inference

- 5-fold patient-level cross-validation: 720 train / 80 val / 200 test per fold
- Ensemble: average of 5 sigmoid probabilities (soft voting)
- Decision threshold τ_d = 0.5

---

## 5. Experimental Results

### 5.1 Cross-Validation (BraTS 2021, 1,000 patients)

| Fold | Dice Score |
|------|-----------|
| Fold 0 | 88.72% |
| Fold 1 | 90.48% |
| Fold 2 | 90.31% |
| Fold 3 | 90.13% |
| Fold 4 | 90.47% |
| **Mean** | **90.02% ± 0.66%** |

### 5.2 Ensemble on Sealed Held-Out Set (251 patients)

| Metric | Score |
|--------|-------|
| Dice | **91.41%** |
| Accuracy | 99.14% |
| Precision | 95.52% |
| Sensitivity | 87.77% |
| Specificity | 99.76% |

### 5.3 Efficiency Comparison (GNN vs. U-Net, same hardware)

| Metric | GNN (ours) | U-Net | Improvement |
|--------|-----------|-------|-------------|
| Dice Score | 91.41% | 87.84% | +3.57pp |
| End-to-end inference | 1,732 ms | 10,160 ms | 5.9× faster |
| GNN inference only | 75.4 ms | — | — |
| Memory (peak) | 11 MB | 2,500 MB | 227× less |
| Parameters | 439,041 | 69,146,113 | 157× fewer |
| Storage | 1.7 MB | 264 MB | 157× smaller |

### 5.4 Zero-Shot Generalisation (BraTS 2023, no retraining)

| Metric | BraTS 2021 | BraTS 2023 | Change |
|--------|-----------|-----------|--------|
| Dice | 91.41% | 89.40% | −1.01pp |
| Accuracy | 99.14% | 98.85% | −0.29pp |
| Sensitivity | 87.77% | 90.69% | **+2.92pp** |
| Specificity | 99.76% | 99.45% | −0.31pp |
| Precision | 95.52% | 92.46% | −3.06pp |

Key insight: Sensitivity *improved* on BraTS 2023, meaning the model detects tumour even better on unseen acquisition protocols. The modest Dice drop (−1.01pp) with no retraining confirms strong generalisation.

### 5.5 Ablation Study (Fold 0)

| Variant | Dice | Parameters |
|---------|------|-----------|
| **Ours (5L, 256-dim)** | **84.03%** | **439K** |
| 6 Layers | 84.00% | 571K |
| 512-dim hidden | 88.78% | 1,710K |
| GAT (instead of SAGE) | 85.03% | 1,184K |

Note: 512-dim achieves higher Dice but at 4× parameter cost. Our design is the optimal efficiency-accuracy trade-off.

---

## 6. Failure Analysis

- **Slice-level failure rate:** ~5% of slices
- **Patient-level complete failures (Dice ≈ 0):** 3 patients
  - BraTS2021_01405
  - BraTS2021_01366
  - BraTS2021_01407
- **Root cause:** Absent or very faint T1CE enhancement — these tumours are non-enhancing, so SLIC superpixels on T1CE do not capture tumour boundaries, and all 15 node features lose discriminative power.

---

## 7. Limitations (4 Total)

1. Superpixel graph depends heavily on T1CE quality — fails for non-enhancing tumours.
2. Binary segmentation only — does not distinguish tumour sub-regions (GD-enhancing, necrotic, oedema).
3. Fixed graph topology per inference — no dynamic graph refinement during prediction.
4. Trained exclusively on adult glioma — unknown performance on paediatric or rare tumour types.

---

## 8. Future Work (5 Directions)

1. Multi-class segmentation (WT, TC, ET sub-regions per BraTS hierarchy).
2. Dynamic graph construction that refines superpixels based on prediction confidence.
3. Extend to paediatric BraTS datasets.
4. Integrate radiomics or clinical metadata as additional node features.
5. Real-time clinical deployment on edge hardware (e.g., Raspberry Pi or Jetson).

---

## 9. Conclusions (4 Points)

1. GraphSAGE on superpixel graphs achieves 91.41% Dice on BraTS 2021 — surpassing U-Net baseline (87.84%) by +3.57pp.
2. Computational cost reduced by 5.9× inference, 227× memory, 157× parameters compared to U-Net.
3. Zero-shot transfer to BraTS 2023 yields 89.40% Dice with improved sensitivity (+2.92pp), confirming cross-dataset generalisation.
4. The system is viable for deployment on consumer-grade GPU hardware (6GB VRAM), making it accessible for resource-constrained clinical settings.

---

## 10. Comparison with SOTA

- **SOTA CNNs/transformers** achieve ~92–94% Dice on BraTS but require 16–32GB VRAM, 100M+ parameters.
- **Our GNN** achieves 91.41% at 6GB VRAM and 439K parameters — near-SOTA accuracy at a fraction of the compute.
- The efficiency advantage is the primary scientific contribution, not the raw Dice number.

---

*Analysis compiled from: `paperWriting/overleaf_flat/main_bubt_paper.tex` (2,584 lines) and `diagram/final Presentation.pdf` (27 slides)*
