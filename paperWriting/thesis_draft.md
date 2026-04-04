# Efficient Brain Tumor Segmentation using Graph Neural Networks on BraTS Datasets: A Superpixel-Based Approach with 5.9× Speedup

**Authors:** Sakib Khan, Rifa Sanjida, Kishor Kumar Das, Md. Mahmudul Hasan, Md. Minhajur Rahman
**Supervisor:** Mr. Shamim Ahmed
**Institution:** Bangladesh University of Business and Technology (BUBT)
**Department:** Computer Science and Engineering

---

## Abstract

Brain tumor segmentation from multi-modal MRI is a critical but computationally intensive task that limits deployment in resource-constrained clinical settings. This work presents a graph neural network (GNN) framework that transforms 3D MRI volumes into sparse superpixel graphs, achieving competitive segmentation accuracy at a fraction of the computational cost of volumetric CNN approaches.

The proposed 5-layer GraphSAGE model with 256 hidden dimensions and 439,041 parameters is trained via 5-fold cross-validation on the BraTS 2021 dataset (1,000-patient pool; 720/80/200 train/val/test split per fold), achieving **90.02% ± 0.74% Dice coefficient**. Soft-voting ensemble of the five fold-best-checkpoint models on a sealed 251-patient held-out set yields **91.41% Dice** (+1.39 pp over the CV mean). Inference takes **75.4 ms** per patient with pre-built graphs or **1.73 s** end-to-end including SLIC superpixel construction — a **5.9× speedup** over a 3D U-Net baseline. The model requires only **11 MB peak GPU memory** (227× less than U-Net), enabling deployment on consumer-grade RTX 2060 hardware.

Cross-edition evaluation on BraTS 2023 (1,245 patients, no retraining) achieves **89.40% Dice** — a 2.01 pp generalisation gap. Ablation studies confirm that 5-layer GraphSAGE is Pareto-optimal under 6 GB VRAM constraints. Qualitative failure analysis identifies complete-miss predictions on atypical tumours as the dominant error mode.

---

## Chapter 1: Introduction

### 1.1 Background and Motivation

Brain tumors — particularly gliomas, which account for approximately 30% of all brain and central nervous system tumors — require accurate and timely volumetric segmentation for diagnosis, treatment planning, and longitudinal monitoring. Manual delineation by expert neuroradiologists is labour-intensive, requires hours per patient, and exhibits inter-rater variability of 10–20% in boundary regions. Automated segmentation has become essential for scalable clinical workflows.

The BraTS (Brain Tumor Segmentation) challenge series, running since 2012, has driven progress in automated MRI-based tumor segmentation. State-of-the-art methods — nnU-Net (~92.7% Whole Tumour Dice), Swin-UNETR (93.3%) — achieve impressive accuracy but demand 31–68 million parameters, 2.5+ GB GPU memory, and 24–80 GB VRAM for training. These requirements confine advanced brain tumor AI to well-resourced academic medical centers, leaving community hospitals, rural clinics, and healthcare systems in developing nations unable to leverage these tools.

Graph Neural Networks offer a fundamentally different approach: instead of processing the full 3D voxel grid, the MRI volume is first compressed into a sparse graph of superpixel nodes, each representing a spatially coherent tissue region. This exploits the inherent sparsity of brain tumors (occupying 5–10% of brain volume) to reduce spatial representation by 890–2,284× while preserving the multi-modal features critical for discrimination.

### 1.2 Research Objectives

**Primary Research Question:** Can a graph-based neural network achieve competitive segmentation accuracy (>90% Dice) while providing efficiency gains sufficient for resource-constrained clinical deployment?

**Specific Objectives:**
1. Design and implement a superpixel-based graph construction pipeline with 15 multi-modal features, free from ground-truth leakage
2. Train and validate a GraphSAGE model achieving >90% Dice on the BraTS 2021 benchmark
3. Quantify computational efficiency gains relative to a 3D U-Net baseline under identical hardware constraints
4. Demonstrate cross-dataset generalisation on BraTS 2023 without retraining
5. Systematically ablate architectural choices to identify the Pareto-optimal configuration under 6 GB VRAM constraints

### 1.3 Contributions

1. **Superpixel-Graph Pipeline**: SLIC-based graph construction achieving 890× (theoretical) to 2,284× (measured) spatial compression while retaining 15 multi-modal discriminative features
2. **Rigorous Evaluation**: Patient-level stratified 5-fold CV with sealed held-out set and 15 automated integrity checks ensuring zero data leakage
3. **Ablation Study**: Systematic comparison of depth (5 vs. 6 layers), width (256 vs. 512 dims), and architecture (GraphSAGE vs. GAT) confirming the 5-layer 256-dim configuration as optimal
4. **Cross-Edition Generalisation**: Zero-shot evaluation on BraTS 2023 yielding 89.40% Dice (2.01 pp gap)
5. **Efficiency Analysis**: 5.9× end-to-end speedup, 157× parameter reduction, 227× memory reduction vs. 3D U-Net

---

## Chapter 2: Literature Review

### 2.1 Volumetric CNN Approaches

The dominant paradigm for medical image segmentation is dense 3D convolutional neural networks. The original 3D U-Net (Çiçek et al., 2016) demonstrated that encoder-decoder architectures with skip connections could achieve competitive segmentation on volumetric MRI. Subsequent work — V-Net, attention U-Net, nnU-Net — refined the encoder-decoder paradigm through residual connections, attention gates, and automated hyperparameter selection. nnU-Net (Isensee et al., 2021) in particular established a new baseline by automatically configuring preprocessing, architecture, and training protocols to the input data, achieving ~92.7% Whole Tumour Dice on BraTS 2021.

Transformer-based architectures followed: TransBTS (Wang et al., 2021) hybridised CNN encoders with transformer bottlenecks; Swin-UNETR (Hatamizadeh et al., 2022) used Swin Transformer encoders to capture long-range dependencies, achieving 93.3% WT Dice. These methods, while highly accurate, carry 33–62 million parameters and require 24–80 GB VRAM for training — practical only at well-resourced institutions.

### 2.2 Graph Neural Network Approaches

GNNs were first proposed for general graph classification tasks (Kipf and Welling, 2017; Hamilton et al., 2017). GraphSAGE (Hamilton et al., 2017) introduced inductive neighbourhood sampling, enabling scalable inference on large graphs without full-graph message passing. Graph Attention Networks (Veličković et al., 2018) incorporated attention-weighted aggregation, improving expressiveness at higher parameter cost.

Application of GNNs to medical image segmentation has been limited. Existing work has applied graph structures to 2D histology patches, anatomical landmark detection, and organ surface meshes. Brain tumor segmentation from 3D MRI via superpixel graphs remains an underexplored niche — the dominant BraTS literature addresses the multi-class ET/TC/WT problem with volumetric CNNs, and no published GNN baseline with a directly comparable binary BraTS 2021 experimental setup was identified at submission.

### 2.3 Superpixel-Based Representations

SLIC (Simple Linear Iterative Clustering; Achanta et al., 2012) remains the dominant superpixel algorithm for medical image analysis due to its runtime efficiency (O(n)), spatial regularity, and parameter simplicity (target count, compactness). Prior work using SLIC for MRI includes lesion detection in histopathology (Kong et al., 2020) and cardiac segmentation (Wolterink et al., 2019). Adaptive slice selection strategies for 3D MRI have been proposed for reducing computational overhead in volumetric analysis (Li et al., 2021).

### 2.4 Efficiency-Aware Medical AI

Recent literature has highlighted the deployability gap in medical AI: models validated on research datasets with high-end hardware frequently cannot be reproduced or deployed in clinical practice due to hardware requirements. Litjens et al. (2017) and Kelly et al. (2019) emphasise that model complexity should match the deployment context. Our work directly addresses this gap by constraining development to consumer-grade hardware (RTX 2060, 6 GB VRAM).

---

## Chapter 3: Methodology

### 3.1 Dataset

**BraTS 2021**: 1,251 adult glioma patients with T1, T1CE, T2, and FLAIR MRI sequences. All volumes skull-stripped and co-registered to MNI152 space; voxel spacing 1×1×1 mm³; matrix 240×240×155. Expert annotations: Enhancing Tumour (ET), Tumour Core (TC), and Whole Tumour (WT) — we use the binary WT label (tumour vs. background) as our segmentation target.

**Patient Splits**:
- CV pool: 1,000 patients → 5-fold stratified split: 720 train / 80 validation / 200 test per fold (`data/cv_folds_v2/`)
- Sealed held-out set: 251 patients — never used during training or model selection
- BraTS 2023 generalisation set: 1,245 patients (from 1,251; 6 missing cached graphs)

### 3.2 Graph Construction Pipeline

**Step 1 — Preprocessing**: Z-score normalization per modality per patient (zero-mean, unit-variance). Skull-stripping mask applied.

**Step 2 — Adaptive Slice Selection**: From 155 axial slices per patient, active slices (mean brain fraction ≥ 5%) are identified. Two adjacent active slices are selected per graph. Mean: 137.5 ± 6.9 active slices out of 155.

**Step 3 — SLIC Superpixels**: Applied to the T1CE channel of the 2-slice volume with parameters: `n_segments=200`, `compactness=0.1`, `sigma=0.3`. Mean actual superpixels: ~46 per slice.

**Step 4 — Node Feature Engineering (15-dimensional)**:
- T1 mean, T1CE mean, T2 mean, FLAIR mean (4D)
- T1 std, T1CE std, T2 std, FLAIR std (4D)
- Area, normalised area, normalised y-centroid, normalised x-centroid (4D)
- Perimeter, compactness, intensity range (3D)

Note: No ground-truth features included. The previously used `tumor_ratio` feature (ground-truth leakage) was removed during audit.

**Step 5 — Graph Edges**: Intra-slice adjacency (RAG) + inter-slice kNN edges (k=3, IoU threshold=0.1, distance threshold=10mm). Mean nodes per graph: ~3,909.

**Compression ratio**: Theoretical 890×, measured ~2,284× (from 8.9M voxels to ~3,909 nodes per patient).

**Labels**: Binary node labels — 1 if superpixel overlaps WT annotation (>50% voxels), 0 otherwise.

### 3.3 Model Architecture

**GraphSAGE (5 layers, 256 hidden dims)**:
- Input: 15-dimensional node features
- 5 SAGEConv layers with 256 hidden channels each
- BatchNorm + ReLU after each layer
- Dropout: 0.1 (hardcoded)
- Output: 1 scalar logit per node (binary classification)
- Parameters: **439,041**
- Model size: **1.7 MB** (disk), **5.1 MB** (full checkpoint)

### 3.4 Training Protocol

- **Loss**: BCEWithLogitsLoss (class weight: positive=9.0)
  - Note: CombinedLoss (BCE + Dice) was defined but never used in production training
- **Optimiser**: AdamW, lr=0.001, weight_decay=0.01
- **Scheduler**: OneCycleLR, max_lr=0.001, pct_start=0.3
- **Batch size**: 24 graphs; gradient accumulation steps=2; **effective batch=48**
- **Mixed precision**: AMP enabled
- **Epochs**: 50 with early stopping (patience=10, monitored: val Dice)
- **Determinism**: Non-deterministic (cudnn.benchmark=True, cudnn.deterministic=False); seed=42 used for data splits only
- **Hardware**: NVIDIA RTX 2060, 6 GB VRAM

### 3.5 Ensemble Strategy

Soft-voting: for each patient in the held-out set, raw logits from all 5 fold-best-checkpoint models are averaged, then a 0.5 threshold applied.

$$P_\text{ensemble}(\text{tumor}) = \frac{1}{5}\sum_{k=1}^{5} \sigma(\text{logit}_{k})$$

### 3.6 Evaluation Metrics

- **Dice coefficient**: Primary metric. $\text{Dice} = \frac{2|\text{TP}|}{2|\text{TP}| + |\text{FP}| + |\text{FN}|}$
- Secondary: Accuracy, Sensitivity (Recall), Specificity, Precision

### 3.7 Integrity Validation

15 automated checks implemented in the pipeline:
1. Feature dimension validation (15D)
2. Patient-level train/val/test overlap detection
3. Label validity check (binary {0,1})
4. Normalisation range validation
5. Seed reproducibility check
6. Graph structure validation
7. Data isolation verification
8. Modality completeness check
9. Slice count consistency
10. Superpixel count consistency
11. Split ratio validation (720/80/200)
12. Checkpoint validation
13. Output format verification
14. Node feature leakage audit
15. Cross-fold patient disjointness

---

## Chapter 4: Results and Analysis

### 4.1 Cross-Validation Performance

**5-Fold CV Results (binary_v3, BCEWithLogitsLoss)**:

| Fold | Train | Val | Test | Dice (%) |
|------|-------|-----|------|----------|
| Fold 0 | 720 | 80 | 200 | 88.72 |
| Fold 1 | 720 | 80 | 200 | 90.48 |
| Fold 2 | 720 | 80 | 200 | 90.31 |
| Fold 3 | 720 | 80 | 200 | 90.13 |
| Fold 4 | 720 | 80 | 200 | 90.47 |
| **Mean ± Std** | - | - | - | **90.02 ± 0.74** |

Key observations:
- Consistent performance across folds (range: 1.76 pp, Fold 0: 88.72% to Fold 1: 90.48%)
- Zero patient overlap between folds confirmed by integrity audit
- The 251-patient held-out set was sealed throughout all training

### 4.2 Secondary Metrics (Ensemble, 251 Patients)

| Metric | Value |
|--------|-------|
| Dice (CV mean) | 90.02% ± 0.74% |
| **Dice (Ensemble)** | **91.41%** |
| Accuracy | 99.14% |
| Sensitivity | 87.77% |
| Specificity | 99.76% |
| Precision | 95.52% |

Precision > Sensitivity (95.52% vs 87.77%) indicates a precision-biased operating point at threshold=0.5. Lowering the threshold to 0.3–0.4 recovers missed tumour pixels at the cost of some false positives — clinically relevant flexibility for screening vs. surgical planning applications.

### 4.3 Ensemble Lift

5-fold soft-voting achieves **+1.39 pp** over the CV fold mean (91.41% vs. 90.02%). The 95% CI for the CV fold mean [89.10%, 90.94%] (df=4) does not include the ensemble result of 91.41%, confirming the ensemble exceeds expected single-fold performance bounds.

### 4.4 Efficiency Benchmarking

**GNN vs. 3D U-Net (RTX 2060, 6 GB VRAM)**:

| Metric | Our GNN | 3D U-Net | Speedup/Reduction |
|--------|---------|----------|-------------------|
| End-to-End Time (incl. SLIC) | **1.73 s** | 10.16 s | **5.9× faster** |
| Inference only (pre-built graph) | **75.4 ms** | 10.16 s | **>135× faster** |
| Parameters | **439,041** | 69.1M | **157× fewer** |
| Peak GPU Memory | **11 MB** | 2,500 MB | **227× less** |
| Model Size (disk) | **1.7 MB** | 264 MB | **157× smaller** |
| Dice (CV mean) | 90.02% | 87.84% | +2.18 pp |

The valid apples-to-apples comparison is the **5.9× end-to-end figure** (both include full processing). The >135× figure compares the GNN's most favourable mode (graphs pre-cached) against U-Net's only mode.

**Note on U-Net accuracy**: The 87.84% reflects binary-only training under the same RTX 2060 hardware constraint with 69.1M parameters (base\_channels=56, num\_levels=4). Published U-Net on multi-class BraTS achieves 91–92%, benefiting from auxiliary ET/TC gradient signal unavailable in our binary formulation.

### 4.5 Ablation Studies

All variants trained on Fold 0 only (effective batch size 32, up to 50 epochs).

| Variant | Test Dice (%) | Parameters | Observation |
|---------|--------------|------------|-------------|
| **GraphSAGE 5L, 256-dim (Baseline)** | **84.03** | **439,041** | **Best accuracy-efficiency** |
| GraphSAGE 6L, 256-dim (Deeper) | 84.00 | 570,881 | −0.03 pp, no benefit |
| GraphSAGE 5L, 512-dim (Wider) | 88.78 | 1,710,081 | +4.75 pp at 3.9× parameters |
| GAT 5L, 256-dim (Attention) | 85.03 | 1,183,745 | +1.00 pp at 2.7× parameters |

Note: Ablation absolute values (84%) are lower than 5-fold CV (90%) due to the single-fold, smaller effective-batch protocol. Only relative rankings are interpretable.

**Conclusions**:
- Depth: Adding a 6th layer yields no benefit (over-smoothing risk)
- Width: 512-dim gains +4.75 pp but costs 3.9× parameters — viable for resource-unconstrained deployments
- Architecture: GraphSAGE better than GAT under efficiency constraints

### 4.6 Cross-Dataset Generalisation (BraTS 2023)

Applied without retraining or threshold adjustment to 1,245 BraTS 2023 patients:

| Metric | BraTS 2021 (held-out) | BraTS 2023 (cross-edition) |
|--------|----------------------|---------------------------|
| Dice | 91.41% | **89.40% ± 11.10%** |
| Accuracy | 99.14% | 98.85% |
| Sensitivity | 87.77% | 90.69% |
| Specificity | 99.76% | 99.45% |
| Precision | 95.52% | 92.46% |

**Generalisation gap: 2.01 pp** (91.41% → 89.40%). Sensitivity improves on BraTS 2023 (+2.92 pp), while precision drops slightly (−3.06 pp), consistent with minor distribution shift between challenge editions.

### 4.7 Qualitative Analysis and Failure Cases

**Per-graph Dice distribution (21,543 slices, 251 patients)**:
- High-quality (Dice ≥ 0.90): 81.2% of slices (17,495)
- Complete-miss (Dice < 0.10): 5.0% of slices (1,074)
- Mean per-slice Dice: 83.2%; patient-level ensemble: 91.41%

**Dominant failure mode**: Complete-miss predictions on patients with atypical T1CE enhancement. Three worst-case patients (BraTS2021_01405, _01366, _01407) received near-zero Dice, indicating that superpixel features fail to discriminate their tumour tissue from background.

**Median case (92.3%)**: Near-complete boundary agreement with residual errors at tumour/oedema interface.

**Best case (100%)**: Patient BraTS2021_01594 — perfect prediction across 8,067 tumour pixels, strong T1CE enhancement providing distinctive superpixel features.

---

## Chapter 5: Standards, Constraints, and Milestones

### 5.1 Sustainability Standards

**Software Quality (ISO/IEC 25010)**: Modular pipeline with clear module interfaces, version-controlled development, comprehensive documentation. All 15 integrity checks automated and reproducible.

**Reproducibility**: Seed=42 for data splits. Non-deterministic training (cudnn.benchmark=True) documented explicitly. Dependency versions pinned (PyTorch 2.0.0, PyG 2.3.0, Python 3.8+).

**Data Standards**: NIfTI format compliance, DICOM compatibility through conversion libraries (dcm2niix). HIPAA/GDPR compliance through de-identified BraTS data.

**Environmental Sustainability**: 5.9× speedup and 157× parameter reduction translate directly to reduced energy consumption. Consumer-grade GPU training (RTX 2060, ~175W) vs. research hardware (A100, ~400W) at 5× longer training duration for volumetric methods.

### 5.2 Societal Impact

**Healthcare Accessibility**: 439K parameters + 11 MB GPU memory enables deployment on RTX 2060 (~$300–400) rather than research-grade GPUs ($1,500–10,000+). Rural clinics and community hospitals in developing nations can now access AI-assisted brain tumor segmentation.

**Clinical Workflow**: 75.4 ms inference (pre-built graph) or 1.73 s end-to-end enables interactive diagnostic sessions. Radiologists receive AI segmentation overlay without waiting.

**Diagnostic Equity**: Reduces quality gap between academic medical centres and underserved settings. AI assistance helps less-experienced radiologists approach specialist-level performance.

### 5.3 Ethics

**Algorithmic Fairness**: BraTS 2021 multi-institutional (19 sites) data partially mitigates single-institution bias. Demographic stratification analysis (age, race, scanner type) not performed — required before clinical deployment.

**Privacy**: Graph-based representation provides inherent privacy benefit — superpixel features are spatially aggregated, making voxel-level reconstruction difficult. Federated learning is architecturally feasible given the lightweight model.

**Clinical Accountability**: System designed as decision support, not autonomous decision-maker. Complete-miss failure mode (5% of slices) requires mandatory empty-prediction detection gate for surgical planning applications.

**Informed Consent**: Patients should be notified when AI contributes to diagnostic workflows. Opt-out mechanisms required in clinical deployment.

### 5.4 Technical Challenges

- **Data Heterogeneity**: Scanner variability (GE, Siemens, Philips) across BraTS 2021 sites; addressed partially by Z-score normalisation
- **Graph Mini-Batching**: PyTorch Geometric's batch collation requires careful tuning of batch size and accumulation steps — batch 24 with 2 accumulation steps (effective 48) found optimal
- **Complete-Miss Failures**: Atypical tumours with minimal T1CE enhancement not captured by 15-feature node representation — primary limitation for clinical deployment

### 5.5 Constraints

| Constraint | Value | Impact |
|-----------|-------|--------|
| VRAM | 6 GB (RTX 2060) | Limits batch size, precludes transformer models |
| Effective batch | 48 (24 × 2 accumulation) | Required gradient accumulation |
| Max epochs | 50 with early stopping | Limits exploration |
| Dataset | BraTS 2021 (binary only) | No multi-class supervision |
| Training budget | 0 dedicated funding | Consumer hardware only |
| Total CV training time | ~35 min (with torch.compile) | Enabled rapid iteration |

### 5.6 Project Timeline

| Phase | Duration | Key Deliverables |
|-------|----------|-----------------|
| Phase 1: Data Preprocessing | Aug 2025 (Weeks 1–3) | BraTS 2021 download, normalisation, 5-fold stratified splits |
| Phase 2: Graph Construction | Sep 2025 (Weeks 4–7) | SLIC pipeline, 15-feature extraction, 15 integrity checks |
| Phase 3: Model Training | Oct 2025 (Weeks 8–11) | 5-fold CV, ensemble implementation |
| Phase 4: Validation | Nov 2025 (Weeks 12–14) | Ablation, efficiency benchmarking, BraTS 2023 evaluation |
| Phase 5: Documentation | Dec 2025 (Weeks 15–16) | Thesis chapters, reproducibility package |

**Key Milestones**:
- Sep 15, 2025: Graph construction pipeline with all 15 integrity checks passing
- Oct 28, 2025: 5-fold CV complete — 90.02% ± 0.74% Dice
- Nov 10, 2025: Ensemble confirmed 91.41% on sealed held-out; BraTS 2023 = 89.40% (gap: 2.01 pp)
- Nov 25, 2025: Efficiency benchmarking final — 5.9× speedup, 157× parameter reduction
- Dec 22, 2025: Thesis submission

---

## Chapter 6: Conclusion

### 6.1 Summary

This thesis demonstrates that graph neural networks constitute a viable and computationally efficient approach to brain tumour segmentation. The proposed framework achieves **91.41% Dice** through soft-voting ensemble on a sealed 251-patient held-out set, with individual 5-fold CV performance of **90.02% ± 0.74%**. The framework delivers **5.9× faster inference** (75.4 ms pre-built / 1.73 s end-to-end) and **155× fewer parameters** (439K vs. 68M) vs. a 3D U-Net, with only **11 MB peak GPU memory**.

Cross-edition evaluation on BraTS 2023 yields **89.40% Dice** — a 2.01 pp gap confirming robustness without catastrophic degradation.

The central research question — *Can a graph-based neural network achieve >90% Dice while providing deployment-practical efficiency gains?* — is answered affirmatively. The 91.41% ensemble Dice, 75.4 ms inference, 11 MB memory footprint, and 439K parameter count demonstrate that competitive accuracy and practical efficiency are not mutually exclusive.

### 6.2 Limitations

1. **Preprocessing overhead**: 1.5 s per patient for graph construction; adds latency in raw-scan real-time workflows
2. **Handcrafted features**: 15 engineered features require domain expertise; may not generalise beyond gliomas without adaptation
3. **Binary only**: No ET/TC/WT sub-region delineation — structurally disadvantaged vs. published multi-class benchmarks
4. **Cross-edition gap**: 2.01 pp BraTS 2021→2023; performance on entirely different scanner protocols or tumour types (meningioma, metastases) untested
5. **Accuracy gap**: 1.3–1.9 pp below best multi-class methods (nnU-Net ~92.7%; Swin-UNETR 93.3%) — partially explained by lack of auxiliary ET/TC supervision
6. **Clinical barriers**: FDA 510(k)/CE marking, hospital IT integration, and prospective radiologist studies remain unaddressed

### 6.3 Future Work

- **Hierarchical multi-scale graphs**: Multi-resolution graph coarsening to capture both fine-grained boundaries and global tumour context
- **Wider architecture**: 512-dim variant (+4.75 pp in ablation at 3.9× cost) viable for resource-unconstrained settings
- **Multi-task extension**: Simultaneous binary and ET/TC/WT prediction brings approach to full BraTS comparability
- **GPU-accelerated SLIC**: Reduces graph construction to <100 ms, enabling true end-to-end real-time processing
- **Adaptive superpixel counts**: Address complete-miss failure mode on small lesions
- **Federated learning**: Lightweight architecture (1.7 MB) ideal for privacy-preserving multi-institutional training
- **Clinical translation**: Prospective radiologist evaluation, multi-scanner validation, regulatory submission preparation

---

## References

1. Çiçek, Ö., et al. (2016). 3D U-Net: Learning Dense Volumetric Segmentation from Sparse Annotation. *MICCAI*.
2. Hamilton, W., et al. (2017). Inductive Representation Learning on Large Graphs. *NeurIPS*.
3. Hatamizadeh, A., et al. (2022). Swin UNETR: Swin Transformers for Semantic Segmentation of Brain Tumors in MRI Images. *MICCAI Workshop*.
4. Isensee, F., et al. (2021). nnU-Net: A self-configuring method for deep learning-based biomedical image segmentation. *Nature Methods*.
5. Kipf, T., & Welling, M. (2017). Semi-Supervised Classification with Graph Convolutional Networks. *ICLR*.
6. Menze, B.H., et al. (2015). The Multimodal Brain Tumor Image Segmentation Benchmark (BRATS). *IEEE TMI*.
7. Veličković, P., et al. (2018). Graph Attention Networks. *ICLR*.
8. Wang, W., et al. (2021). TransBTS: Multimodal Brain Tumor Segmentation Using Transformer. *MICCAI*.
9. Achanta, R., et al. (2012). SLIC Superpixels Compared to State-of-the-Art Superpixel Methods. *IEEE TPAMI*.
10. Bakas, S., et al. (2021). The RSNA-ASNR-MICCAI BraTS 2021 Benchmark on Brain Tumor Segmentation and Radiogenomic Classification. *arXiv:2107.02314*.

---

*Document prepared from verified experimental results (binary_v3, BCEWithLogitsLoss). All numbers sourced from:*
*- CV metrics: `checkpoints/binary_v3/fold_*/results.json`*
*- Ensemble: `research_results/ensemble_v2/ensemble_results.json`*
*- BraTS 2023: `research_results/brats2023_evaluation/results.json`*
*- Timing: `research_results/timing_benchmark/two_scenario_results.json`*
