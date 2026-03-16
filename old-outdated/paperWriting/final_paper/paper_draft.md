# Graph Neural Networks for Efficient Brain Tumour Segmentation: A Superpixel-Based Approach on BraTS 2021

---

## Abstract

Accurate brain tumour segmentation is essential for diagnosis and treatment planning, yet current deep learning methods demand substantial computational resources that hinder clinical deployment. This paper proposes a graph neural network (GNN) framework for efficient binary brain tumour segmentation on the BraTS 2021 dataset (1,251 patients). Rather than processing entire 3D MRI volumes, patient scans are converted into graph representations using SLIC superpixels applied across all 155 axial slices per patient (with up to 200 superpixels per slice, adaptively reduced; average ~46 per active slice), with each node encoding 15 handcrafted multi-modal features. This representation exploits inherent tumour sparsity to achieve at least 890× spatial dimensionality reduction (conservative lower bound; measured compression ~2,284× per patient) while preserving discriminative spatial and intensity information across four MRI modalities (T1, T1CE, T2, FLAIR).

The framework is evaluated through rigorous 5-fold cross-validation on a 1,000-patient pool with patient-level stratified splits (720 train / 80 val / 200 fold-test per fold), and a completely sealed 251-patient held-out test set used exclusively for final evaluation. The 5-layer GraphSAGE model achieves **90.02% ± 0.74% Dice coefficient** across folds. Soft-voting ensemble aggregation of all five models on the held-out set yields **91.41% Dice**, a statistically significant improvement over individual fold models (one-sample _t_-test, _t_ = −4.19, _p_ = 0.014, 95% CI [89.10%, 90.94%], _df_ = 4). Inference time is **74 ms per patient** (GPU, pre-built graph) or **1.47 s end-to-end** including fast superpixel construction, representing a **6.9× speedup** over a 3D U-Net baseline. The model requires only **439K parameters** (155× fewer than our benchmarked 68M U-Net) and **11 MB peak GPU memory**.

To assess cross-dataset generalisation, the trained ensemble is applied zero-shot to BraTS 2023 (1,245 patients), achieving **89.21% Dice** — a 2.20% generalisation gap consistent with expected domain shift. Ablation studies over network depth (5 vs. 6 layers), hidden dimensionality (256 vs. 512), and GNN architecture (GraphSAGE vs. GAT) validate the chosen design. Qualitative failure case analysis identifies complete-miss failures on atypical tumours as the primary error mode.

These results demonstrate that graph neural networks deliver a compelling accuracy-efficiency trade-off for medical image segmentation, enabling deployment on consumer-grade hardware (NVIDIA RTX 2060, 6 GB VRAM) in resource-constrained clinical settings where volumetric models are not feasible.

**Keywords:** Brain Tumour Segmentation · Graph Neural Networks · Medical Image Analysis · BraTS 2021 · BraTS 2023 · Computational Efficiency · GraphSAGE · Cross-Dataset Generalisation

---

# Chapter 1: Introduction

## 1.1 Background

Accurate brain tumour segmentation from MRI scans is essential for diagnosis, treatment planning, and longitudinal monitoring of therapeutic response. The BraTS (Brain Tumour Segmentation) challenge, established in 2012, has become the de facto benchmark for evaluating automated segmentation algorithms on multi-parametric MRI data. Binary segmentation — detecting tumour versus non-tumour tissue — is a fundamental prerequisite for screening workflows, rapid clinical assessment, and initial diagnostics, forming the basis upon which multi-class sub-region delineation builds.

Most current methods rely on volumetric CNNs, particularly 3D U-Net architectures [Çiçek et al., 2016]. These approaches achieve strong performance but impose substantial computational demands: tens of millions of parameters and 8+ GB of GPU memory. These requirements create significant barriers to deployment in resource-limited clinical environments, on edge devices, and in settings where rapid inference is essential. There is a clear need for approaches that maintain competitive accuracy while substantially reducing computational cost.

Graph Neural Networks (GNNs) offer a compelling alternative paradigm. Rather than processing entire 3D volumes, GNNs represent images as graphs — superpixels become nodes and spatial adjacency relationships become edges — capturing local texture and global structure while operating on substantially smaller data representations. This is a natural fit for brain tumour imaging: tumours typically occupy only 5–10% of total brain volume, creating inherent sparsity that graph-based representations can exploit directly.

## 1.2 Problem Statement

Automated brain tumour segmentation is clinically important, yet current deep learning methods face a persistent efficiency-accuracy trade-off. Volumetric CNNs achieve high accuracy but require substantial computational resources — 31M–68M+ parameters, 8+ GB of GPU memory, and slow inference times — rendering them impractical for most real-world clinical settings. Lightweight alternatives typically sacrifice too much accuracy to be clinically useful.

The central research question is:

> _Can a graph-based neural network achieve competitive segmentation accuracy (>90% Dice) while remaining efficient enough for deployment in resource-constrained clinical environments?_

Addressing this requires: (1) converting 3D MRI volumes into meaningful graph representations without losing diagnostically relevant features; (2) designing a compact yet sufficiently expressive architecture; (3) ensuring zero data leakage through rigorous patient-level validation protocols; and (4) systematically quantifying efficiency gains against established baselines.

## 1.3 Research Objectives

The primary objectives are:

- Develop a robust graph construction pipeline converting 3D MRI volumes into graph representations via SLIC superpixels applied to all 155 axial slices per patient
- Design and validate a GraphSAGE architecture for binary brain tumour segmentation, balancing expressiveness with parameter efficiency (<1M parameters)
- Conduct rigorous 5-fold cross-validation with patient-level stratified splits and 15 automated integrity checks ensuring zero data leakage
- Benchmark inference speed, GPU memory, and model size against a 3D U-Net baseline under identical hardware conditions
- Perform systematic ablation studies validating architectural choices across depth, hidden dimensionality, and GNN architecture type
- Evaluate zero-shot cross-dataset generalisation on BraTS 2023 (1,245 patients)
- Demonstrate deployment feasibility on consumer-grade hardware (RTX 2060, 6 GB VRAM)

## 1.4 Motivations

**Clinical Deployment Challenges:** Current deep learning models for brain tumour segmentation impose hardware requirements, long inference times, and integration complexity that create substantial operational barriers for clinical deployment.

**Resource-Constrained Settings:** Many healthcare facilities, particularly in developing regions, lack access to high-end GPUs. This creates a diagnostic equity problem in which AI-assisted analysis is available only to well-funded institutions.

**Real-Time Applications:** Emerging clinical scenarios — intra-operative guidance, emergency screening, telemedicine — require near-instant results. Existing volumetric methods struggle to meet these latency requirements.

**Structural Promise of Graphs:** Tumours are spatially sparse, occupying only 5–10% of total brain volume. Graph representations exploiting this sparsity enable meaningful efficiency gains relative to dense volumetric processing.

**Data Integrity Concerns:** Recent literature has identified concerning instances of data leakage in medical imaging studies. This motivated comprehensive auditing procedures, with all 15 integrity checks passing.

## 1.5 Significance and Contributions

This work makes the following specific contributions:

1. **Novel Graph Construction Pipeline:** A superpixel-based graph generation method applying SLIC across all 155 axial slices per patient (~138 active after empty-slice filtering), extracting up to 200 superpixels per slice (average ~46 adaptively) with 15 engineered node features (intensity statistics, spatial, shape/texture) — zero ground-truth leakage.

2. **Optimised GNN Architecture:** A 5-layer GraphSAGE model with 256 hidden dimensions (439,041 parameters), validated via ablation showing no benefit from additional depth (6 layers: 84.00% vs. 5 layers: 84.03%).

3. **Strong Performance with Efficiency:** 90.02% ± 0.74% Dice on 5-fold CV and 91.41% ensemble on the sealed 251-patient held-out set, while achieving 6.9× faster inference (74 ms pre-built graph; 1.47 s end-to-end) and 155× fewer parameters (439K vs. 68M) compared to a 3D U-Net baseline.

4. **Comprehensive Validation Framework:** Patient-level stratified 5-fold cross-validation with zero data leakage, verified through 15 automated integrity checks covering feature dimensions, patient overlap, label validity, normalisation, seed consistency, and output format.

5. **Ablation Study:** Systematic evaluation across depth (5 vs. 6 layers), width (256 vs. 512 hidden dimensions), and architecture (GraphSAGE vs. GAT), confirming 5-layer 256-dim GraphSAGE as the Pareto-optimal design under efficiency constraints.

6. **Zero-Shot Cross-Dataset Generalisation:** Applied to BraTS 2023 (1,245 patients) without retraining, achieving 89.21% Dice — a 2.20% generalisation gap consistent with expected domain shift.

## 1.6 Thesis Organisation

**Chapter 2** surveys the evolution of medical image segmentation from classical methods through CNNs and transformer-based architectures, reviews GNN foundations and superpixel methods, and identifies the efficiency-accuracy gap motivating this work.

**Chapter 3** details the proposed methodology: BraTS 2021 dataset characteristics, the 8-phase pipeline from preprocessing to ensemble, GraphSAGE architecture design, training protocol, and the 15-check integrity validation framework.

**Chapter 4** presents comprehensive experimental results: 5-fold CV performance, ensemble results, efficiency benchmarking, ablation studies, zero-shot BraTS 2023 evaluation, and qualitative failure case analysis.

**Chapter 5** examines sustainability standards, societal impacts, ethical considerations, technical and operational constraints, and the 16-week project timeline.

**Chapter 6** synthesises key findings, limitations, and directions for future work.

---

# Chapter 2: Background Study and Literature Review

Brain tumour segmentation has advanced substantially over the past two decades, driven by progress in both medical imaging technology and deep learning methodology. This chapter surveys the evolution from classical image processing through CNNs to emerging graph-based approaches, identifying the persistent efficiency challenge that motivates the GNN framework proposed in this thesis.

## 2.1 Evolution of Medical Image Segmentation

### 2.1.1 Classical Approaches

Early brain tumour segmentation relied on traditional techniques including intensity thresholding, region growing, and atlas-based registration. These methods required extensive manual parameter tuning and struggled to accommodate high inter-patient variability. The BraTS Challenge, established in 2012 [Menze et al., 2015], standardised evaluation protocols and provided large-scale annotated datasets, substantially accelerating innovation.

### 2.1.2 The CNN Revolution: U-Net and Its Variants

Fully convolutional networks fundamentally transformed medical image segmentation. Ronneberger et al. [2015] introduced **U-Net**, establishing the symmetric encoder-decoder architecture with skip connections that preserves spatial information during upsampling. U-Net became the standard baseline for medical segmentation, substantially outperforming classical methods while requiring comparatively little training data.

**Çiçek et al. [2016]** extended U-Net to 3D medical imaging through volumetric convolutions — enabling direct processing of multi-slice MRI volumes. However, 3D U-Net's computational requirements scale cubically with volume dimensions, necessitating 8 GB+ of VRAM and limiting deployment on resource-constrained hardware.

**Isensee et al. [2021]** introduced **nnU-Net**, a self-configuring framework that automatically optimises preprocessing, architecture, and training hyperparameters, achieving state-of-the-art results across multiple segmentation tasks. However, automatic configuration requires extensive computational resources, and the resulting models remain substantial (approximately 31M parameters for the standard BraTS 3D full-resolution configuration).

### 2.1.3 Transformer-Based Architectures

Recent works have explored vision transformers for medical segmentation. **Hatamizadeh et al. [2022]** proposed **Swin-UNETR**, combining Swin Transformer encoders with CNN decoders, achieving 93.3% WT Dice on BraTS 2021 (62M parameters). **TransBTS** [Wang et al., 2021] similarly integrates transformer self-attention with 3D convolutions (~33M parameters, evaluated on BraTS 2020 achieving ~90.1% WT Dice). While these hybrid architectures push accuracy boundaries, they require substantially larger computational budgets (33–62M parameters and 8–24 GB VRAM), exacerbating deployment challenges for resource-constrained clinical settings.

## 2.2 Graph Neural Networks in Medical Imaging

### 2.2.1 Fundamentals of Graph Representation Learning

Graph neural networks extend deep learning to non-Euclidean data structures, learning representations through iterative message passing between connected nodes [Hamilton et al., 2017; Kipf & Welling, 2017].

**Hamilton et al. [2017]** introduced **GraphSAGE** (Graph Sample and Aggregate), which learns node embeddings by sampling and aggregating features from local neighbourhoods. The inductive learning capability of GraphSAGE — learning a generalised aggregation function rather than node-specific embeddings — enables generalisation to unseen graph structures, making it particularly suitable for medical imaging where each patient represents a structurally distinct graph.

**Kipf & Welling [2017]** introduced **GCN** (Graph Convolutional Networks), which propagate features using a spectral-domain convolution approximation. While effective for transductive settings, GCNs require the full graph to be present during training, limiting scalability.

### 2.2.2 Superpixel Segmentation

Superpixel algorithms group perceptually similar pixels into compact regions, providing mid-level representations that preserve object boundaries while reducing dimensionality. **Achanta et al. [2012]** developed **SLIC** (Simple Linear Iterative Clustering), a computationally efficient method performing k-means clustering in combined colour-spatial space. SLIC generates compact superpixels with controllable size (via the `n_segments` parameter), running in linear time in pixel count — making it ideal for large-scale medical image preprocessing.

### 2.2.3 Graph-Based Medical Segmentation: Existing Work

**Hybrid CNN-GNN Approaches:** Most existing work combines CNNs for feature extraction with GNNs for spatial reasoning. These hybrid methods retain the computational overhead of volumetric convolutions while adding graph processing costs, yielding limited efficiency gains relative to pure-CNN baselines.

**Region Adjacency Graphs:** Some methods construct graphs where nodes represent anatomical structures and edges encode spatial relationships. While interpretable, these approaches require prior segmentation of regions, introducing preprocessing dependencies.

**Point Cloud Representations:** Voxel-to-point sampling combined with PointNet-style architectures has been investigated, but point clouds discard spatial structure information that graphs can preserve through explicit edge connectivity.

## 2.3 Research Gap and Motivation

### 2.3.1 The Efficiency-Accuracy Dilemma

Despite impressive accuracy, modern deep learning methods face persistent deployment barriers:

**Computational Requirements:** State-of-the-art models include nnU-Net (~31M parameters for BraTS configuration), Swin-UNETR (62M parameters), TransBTS (~33M parameters), and standard 3D U-Net implementations (~68M parameters), all requiring 8–24+ GB of GPU memory. This limits deployment to high-end research hardware unavailable in rural clinics, mobile diagnostic units, and healthcare systems in developing regions.

**Inference Latency:** Processing a 240×240×155 MRI volume through deep 3D CNNs requires 10+ seconds per patient in constrained hardware settings (as measured in our benchmarks), which is problematic for real-time clinical workflows.

**Energy Consumption:** Large models consume substantial power during inference. As medical AI scales to millions of patients globally, the environmental impact becomes significant, motivating architectures that balance accuracy with energy efficiency.

### 2.3.2 Unexplored GNN Potential

**Pure GNN Approaches Absent:** Most medical imaging GNN research employs hybrid CNN-GNN designs. No prior work has rigorously benchmarked a pure, lightweight GNN architecture — without a CNN backbone — for efficient brain tumour segmentation with systematic ablation and cross-dataset validation.

**Superpixel-GNN Integration Underexplored:** Combining SLIC-based preprocessing (enabling ≥890× spatial dimensionality reduction; measured ~2,284× per patient) with inductive graph learning via GraphSAGE is an underexplored avenue that may achieve competitive accuracy with dramatic efficiency improvements.

**Reproducibility Gap:** Medical AI research has faced scrutiny over reproducibility. Establishing rigorous validation standards — patient-level splits, comprehensive integrity checks, sealed test sets — is critical for the field's maturity.

## 2.4 Positioning This Work

This thesis addresses the identified gap through three key contributions:

1. **Pure GNN Architecture:** A lightweight GraphSAGE model (439K parameters) without convolutional components achieves 91.41% Dice — approaching state-of-the-art transformers while offering 155× fewer parameters than our benchmarked U-Net baseline (68M), ~70× fewer than nnU-Net (31M), ~75× fewer than TransBTS (33M), and ~141× fewer than Swin-UNETR (62M).

2. **Rigorous Validation Framework:** Patient-level stratified cross-validation with 15 comprehensive integrity checks ensures zero data leakage and reproducibility.

3. **Clinical Deployment Focus:** Near-real-time performance (74 ms pre-built graph; 1.47 s end-to-end) on consumer-grade hardware (RTX 2060, 6 GB VRAM) demonstrates practical viability for resource-constrained settings.

---

# Chapter 3: Research Methodology

This chapter presents the methodology developed to build and validate the graph-based neural network for brain tumour segmentation. The research follows a systematic pipeline comprising data preprocessing, graph construction, model architecture design, training, and rigorous validation — each stage designed to ensure reproducibility, eliminate data leakage, and enable fair comparison with baseline methods.

## 3.1 Proposed Framework

The framework progresses from raw multi-parametric MRI input through eight interconnected phases, transforming raw data into validated tumour segmentation predictions. Figure 1 provides a unified visual overview of the four core stages of the pipeline.

![Figure 1: The Proposed Superpixel-Based GNN Framework for Brain Tumour Segmentation. **Stage 1 — Multi-Modal MRI Preprocessing:** All 155 axial slices of the four co-registered MRI modalities (T1, T1CE, T2, FLAIR) are extracted and Z-score normalised per modality per patient; empty slices (brain mask sum = 0) are discarded, retaining ~138 active slices. **Stage 2 — Superpixel Graph Construction:** The SLIC algorithm (compactness = 0.1, σ = 0.3) segments each active slice into adaptive superpixels (target ≤ 200 per slice; measured average ~46). Each superpixel becomes a graph node with 15 engineered features spanning multi-modal intensity statistics (8), spatial and area descriptors (4), and shape/texture features (3). Spatially adjacent superpixels are connected by undirected edges, forming a planar graph (~3,909 ± 821 nodes per patient; compression ≥ 890×). **Stage 3 — GraphSAGE Message Passing and Node Classification:** A 5-layer GraphSAGE encoder (256 hidden dimensions, 439,041 parameters) performs iterative neighbourhood aggregation over 5 hops, learning discriminative node embeddings. A sigmoid binary classifier assigns each node a tumour probability. **Stage 4 — Volumetric Mask Reconstruction:** Per-superpixel predictions are mapped back to the original pixel grid (majority-vote propagation) and stacked across all 155 axial slices to reconstruct the final 3D binary tumour segmentation mask.](proposed_framework_diagram.png)

**Phase 1 — Data Preprocessing:** Standardises the BraTS 2021 dataset through skull-stripping verification, Z-score normalisation per modality per patient, and patient-level stratified 5-fold split generation.

**Phase 2 — Graph Construction:** For each patient, all 155 axial slices are processed; empty slices (brain mask sum = 0) are filtered, retaining ~138 active slices on average. The SLIC algorithm segments each active slice into superpixels (adaptive target: up to 200 per slice; actual average ~46 per active slice due to adaptive sizing). Each superpixel becomes a graph node with 15 engineered features. Edges connect spatially adjacent superpixels.

**Phase 3 — Feature Engineering:** Extracts 15 domain-specific features without ground-truth leakage (see Section 3.3.2 for full specification).

**Phase 4 — Model Training:** 5-layer GraphSAGE (256 hidden dim); AdamW optimizer; binary cross-entropy loss with class weighting; effective batch size 64 (batch 32 with 2 gradient accumulation steps); 50 epochs per fold; deterministic seed 42.

**Phase 5 — Cross-Validation:** Patient-level stratified 5-fold splits ensuring no patient appears in both training and test sets.

**Phase 6 — Ensemble:** Soft-voting (averaging sigmoid outputs from all 5 fold models) evaluated on the sealed 251-patient held-out set.

**Phase 7 — Efficiency Benchmarking:** Compares inference speed, GPU memory, and model size against a 3D U-Net baseline under identical RTX 2060 / 6 GB VRAM conditions.

**Phase 8 — Integrity Validation:** 15 automated auditing checks (detailed in Section 3.1.1 below).

### 3.1.1 The 15 Integrity Checks

1. Feature dimension verification (15 features per node)
2. Patient leakage detection across all splits
3. Label distribution analysis across folds
4. Normalisation consistency (Z-score per modality)
5. Seed reproducibility (deterministic output verification)
6. Graph node count validation (~3,900 nodes per patient; measured 3,909 ± 821)
7. Edge connectivity verification (planar graph structure)
8. Cross-fold data isolation audit (zero shared patients)
9. Ground-truth label integrity (binary mask validity)
10. MRI modality completeness (all 4 modalities present)
11. Slice count validation (155 axial slices per BraTS patient; ~138 active after filtering)
12. Superpixel boundary consistency
13. Split ratio verification (720/80/200 per fold from the 1,000-patient CV pool; 251 sealed held-out)
14. Model checkpoint consistency
15. Prediction output format verification

All 15 checks passed.

## 3.2 Dataset Analysis

### 3.2.1 BraTS 2021 Dataset

The BraTS 2021 training set contains **1,251 multi-institutional glioma cases** from 19 international institutions. Each case includes four co-registered MRI modalities (T1, T1CE, T2, FLAIR), pre-processed by BraTS organisers to 1 mm³ isotropic resolution with skull-stripping applied. Volume dimensions are 240×240×155 voxels.

Expert neuroradiologists annotated three nested tumour sub-regions (ET, TC, WT). For this work we focus on binary segmentation, combining all tumour regions into a single positive class.

**Table 1: BraTS 2021 Dataset Statistics and Cross-Validation Split**

| Characteristic | Value |
|---|---|
| Total Patients | 1,251 |
| MRI Modalities | 4 (T1, T1CE, T2, FLAIR) |
| Image Resolution | 240 × 240 × 155 |
| Voxel Spacing | 1 mm³ (isotropic) |
| CV Pool (train+val+fold-test) | 1,000 (80% of total) |
| Training Patients (per fold) | 720 (72% of CV pool) |
| Validation Patients (per fold) | 80 (8% of CV pool) |
| Fold-Test Patients (per fold) | 200 (20% of CV pool; non-overlapping) |
| Sealed Held-Out Set | 251 (20% of total; never seen during CV) |
| Average Tumour Volume | 5–10% of brain |
| Cross-Validation Folds | 5 (patient-level stratified) |
| Total Graph Nodes (per patient) | ~3,900 (measured: 3,909 ± 821) |
| Node Features | 15 (intensity + spatial + shape) |

![Figure 1: Qualitative segmentation example — median case BraTS2021_00209 (Dice 92.3%). Four panels: T1CE MRI, ground-truth overlay (red), GNN prediction (blue), comparison (purple = true positive, red = false negative, blue = false positive)](../../research_results/failure_cases/median_BraTS2021_00209_z059.png)

### 3.2.2 Preprocessing Pipeline

- **Z-score normalisation:** applied independently per modality per patient: x_norm = (x − μ) / σ
- **Slice selection:** All 155 axial slices extracted per patient; empty slices (brain mask sum = 0) filtered, retaining on average ~138 active slices
- **Quality control:** Automated checks for missing files, corrupted images, inconsistent dimensions, and label validity

## 3.3 Algorithm / Model Analysis

### 3.3.1 Graph Construction Algorithm

**Step 1 — Slice Extraction:** Process all 155 axial slices of the BraTS 3D volume (240×240×155). Empty slices are filtered, retaining ~138 active slices per patient.

**Step 2 — Superpixel Generation:** Apply SLIC to each active slice with: `n_segments` adaptively set to brain_pixels / 200 (target: up to 200 per slice); `compactness = 0.1` (favouring image-content-following boundaries over spatial uniformity); Gaussian smoothing σ = 0.3 prior to clustering. The adaptive configuration yields an average of **~46 superpixels per active slice**, resulting in roughly **3,900 total nodes per patient** (measured: mean 3,909 ± 821, range 2,347–6,041).

**Step 3 — Node Feature Extraction:** 15 features computed per superpixel (Section 3.3.2).

**Step 4 — Edge Construction:** Two nodes are connected if they share a boundary in image space. This creates a planar graph with average degree 4–6.

**Step 5 — Node Label Assignment:** Binary label assigned by majority voting: if >50% of a superpixel's voxels are annotated as tumour in ground truth, the node is labelled positive.

**Dimensionality Reduction:**

Each 3D MRI volume spans 240×240×155 = 8,928,000 voxels per modality. The theoretical compression ratio using the conservative assumption f_brain = 0.30 is:

> R_theory = (8,928,000 × 0.30) / (200 × 15) = 2,678,400 / 3,000 ≈ **890×**

Measured across 50 patients: f_brain = 0.154 ± 0.017 (skull-stripped BraTS volumes are more compact), yielding actual spatial compression:

> R_actual = 8,928,000 / 3,909 ≈ **2,284×**

The 890× figure used throughout this work is therefore a conservative lower bound. All claims cite the conservative 890× figure.

### 3.3.2 Feature Engineering

**15 features per superpixel node, in three categories:**

**1. Multi-Modal Intensity Statistics (8 features):** For each of the four MRI modalities (T1, T1CE, T2, FLAIR): mean intensity and standard deviation over all voxels within the superpixel. Gliomas exhibit characteristic patterns: hypo-intense in T1, hyper-intense in T2/FLAIR, rim enhancement in T1CE.

**2. Spatial and Area Features (4 features):** Superpixel area (absolute pixel count), normalised area (fraction of slice area), normalised centroid y-coordinate, normalised centroid x-coordinate.

**3. Shape and Texture Features (3 features):** Perimeter (boundary pixel count via morphological erosion), compactness (4πA / P²), intensity range (peak-to-peak signal across all four modalities).

**Critical design decision:** No ground-truth label information is included in the feature set. Inclusion of such information (e.g., a tumour_ratio feature computed from ground-truth labels) constitutes data leakage and invalidates downstream evaluation. Following removal of an early draft tumour_ratio feature and retraining, the validated performance of 90.02% CV mean was obtained.

### 3.3.3 GraphSAGE Architecture

We employ GraphSAGE [Hamilton et al., 2017] for its inductive learning capability — learning an aggregation function applicable to unseen graph structures — making it well-suited for settings where each patient graph is structurally distinct.

**Core update rule (mean aggregator):**

```
h_N(v)^(k) = mean_{u ∈ N(v)} h_u^(k)
h_v^(k+1) = σ( W · [h_v^(k) || h_N(v)^(k)] )
```

where h_v^(k) is node v's representation at layer k, N(v) is its neighbourhood, W is a learnable weight matrix, and σ is ReLU.

**Architecture specification:**

| Component | Configuration |
|---|---|
| Input Layer | 15 node features |
| Hidden Layers | 5 × GraphSAGE, 256 hidden dimensions each |
| Activation | ReLU after each layer |
| Normalisation | Batch normalisation |
| Output Layer | Single neuron + sigmoid (binary) |
| **Total Parameters** | **439,041** |

The 5-layer depth provides a 5-hop receptive field, enabling each node to aggregate information from a wide spatial context. Ablation studies confirm that 6 layers provide no additional benefit.

### 3.3.4 Training Protocol

**Loss:** Binary Cross-Entropy with Logits (BCEWithLogitsLoss); class imbalance addressed through Dice coefficient as primary evaluation metric.

**Optimiser:** AdamW; learning rate 0.001; β₁ = 0.9, β₂ = 0.999; weight decay 0.01.

**Learning Rate Schedule:** OneCycleLR with cosine annealing; warm-up comprises 10% of total training steps.

**Batch Configuration:** Effective batch size 64, implemented as batch size 32 with 2 gradient accumulation steps (to fit within 6 GB VRAM while maintaining stable gradients).

**Duration:** 50 epochs per fold; ~5 hours per fold on RTX 2060.

**Reproducibility:** Seed 42 across Python, NumPy, PyTorch, and CUDA; `CUBLAS_WORKSPACE_CONFIG=:4096:8`; `CUDNN_DETERMINISTIC=True`.

### 3.3.5 Ensemble Strategy

Soft-voting aggregates sigmoid probabilities from all 5 fold models:

> P_ensemble(tumour) = (1/5) × Σ P_fold_k(tumour)

Final binary prediction: tumour if P_ensemble > 0.5. Evaluated on the sealed 251-patient held-out set, this yields a **+1.39% absolute Dice improvement** over the CV mean (90.02% → 91.41%).

### 3.3.6 Evaluation Metrics

**Primary:** Dice Similarity Coefficient (DSC) = 2|A ∩ B| / (|A| + |B|)

**Secondary:** Sensitivity (recall), specificity, precision, inference time (ms), peak GPU memory (MB), model parameters.

## 3.4 Implementation Details

The pipeline is implemented in Python using PyTorch 2.0.0, PyTorch Geometric 2.3.0, SimpleITK 2.2.1, nibabel 5.1.0, and scikit-image 0.21.0. Preprocessed data is stored as compressed `.npz` files; graphs are structured as PyTorch Geometric `Data` objects (node features: N×15; edge indices: 2×E; labels: N×1). All experiments were conducted on an NVIDIA RTX 2060 (6 GB VRAM), demonstrating accessibility on consumer-grade hardware.

---

# Chapter 4: Experimental Results and Analysis

This chapter presents comprehensive experimental results demonstrating the performance, efficiency, and generalisation of the proposed graph-based segmentation framework.

## 4.1 Environment Setup

**Hardware:** NVIDIA RTX 2060 (6 GB VRAM); 32 GB RAM; SSD storage.

**Software:** Python 3.12; PyTorch 2.0.0 (CUDA 11.8); PyTorch Geometric 2.3.0; SimpleITK 2.2.1; scikit-image 0.21.0.

**Reproducibility:** Seed 42 (Python, NumPy, PyTorch, CUDA); `CUBLAS_WORKSPACE_CONFIG=:4096:8`; FP32 throughout.

This consumer-grade GPU configuration (RTX 2060, 6 GB VRAM) contrasts with the 24 GB+ VRAM typically required by state-of-the-art volumetric models and directly demonstrates the accessibility of the proposed approach.

## 4.2 Cross-Validation Performance

**Table 2: 5-Fold Cross-Validation Results on BraTS 2021 (720/80/200 splits from 1,000-patient CV pool)**

| Fold | Train | Val | Test | Dice (%) |
|---|---|---|---|---|
| Fold 0 | 720 | 80 | 200 | 88.72 |
| Fold 1 | 720 | 80 | 200 | 90.48 |
| Fold 2 | 720 | 80 | 200 | 90.31 |
| Fold 3 | 720 | 80 | 200 | 90.13 |
| Fold 4 | 720 | 80 | 200 | 90.47 |
| **Mean ± Std** | — | — | — | **90.02 ± 0.74** |

![Figure 2: 5-fold cross-validation performance. Bars show per-fold Test Dice (88.72%–90.48%). Dashed line = CV mean 90.02% ± 0.74%; dash-dot line = ensemble 91.41% on sealed 251-patient held-out set. Grey band = ±1 SD.](../../research_results/figures/fig_A_cv_performance.png)

**Key observations:**

- **High average performance:** 90.02% mean Dice surpasses typical clinical acceptability thresholds (85–88%)
- **Low variance:** 0.74% standard deviation indicates robust generalisation, not overfitting to specific patient subsets
- **Consistent range:** All folds within 88–91%; 1.76% spread between best (Fold 1) and worst (Fold 0)
- **Patient-level integrity:** Zero patient overlap between splits, verified through all 15 integrity checks; the 251-patient held-out set was sealed throughout all CV training

### 4.2.1 Secondary Metrics (Ensemble on 251-Patient Held-Out Set)

**Table 3: Comprehensive Performance Metrics**

| Metric | Value (%) |
|---|---|
| Dice Coefficient (CV mean) | 90.02 ± 0.74 |
| **Dice Coefficient (Ensemble)** | **91.41** |
| Accuracy | 99.14 |
| Sensitivity (Recall) | 87.77 |
| Specificity | 99.76 |
| Precision | 95.52 |

**Clinical interpretation:** High precision (95.52%) and specificity (99.76%) indicate the model rarely generates false positives — clinically appropriate for a system designed to flag suspicious regions for radiologist review. The slightly conservative sensitivity (87.77%) reflects a precision-biased operating point at threshold 0.5; lowering the threshold to 0.3–0.4 recovers missed tumour regions with minimal specificity cost for screening applications.

## 4.3 Ensemble Performance

**Table 4: Single Model vs. Ensemble on Sealed 251-Patient Held-Out Set**

| Method | Dice (%) | Improvement |
|---|---|---|
| Best Single Model (Fold 1) | 90.48 | — |
| CV Mean (average single model) | 90.02 ± 0.74 | — |
| **Ensemble (5 Models)** | **91.41** | **+1.39%** |

A one-sample _t_-test confirms the improvement is statistically significant: _t_ = −4.19, _p_ = 0.014, 95% CI [89.10%, 90.94%], _df_ = 4.

![Figure 6: Per-metric comparison — individual fold mean (blue) vs. 5-fold soft-voting ensemble (pink). Ensemble evaluated on the sealed 251-patient held-out set. Bidirectional arrow on Dice bars shows the statistically significant +1.39% lift (p=0.014).](../../research_results/figures/fig_E_ensemble_vs_individual.png)

**Analysis:** Each fold trains on a different 720-patient subset of the 1,000-patient CV pool, learning complementary patterns and making independent errors. Soft-voting averages out individual model biases (variance reduction) and improves robustness on edge cases and ambiguous tumour boundaries.

## 4.4 Efficiency Benchmarking

### 4.4.1 Baseline Comparison

We compare against a **3D U-Net baseline** (68M parameters) [Çiçek et al., 2016] implemented and benchmarked under identical RTX 2060 / 6 GB VRAM conditions. This baseline represents a standard volumetric CNN architecture. For reference, published SOTA models occupy a similar or smaller parameter scale: Swin-UNETR (62M), nnU-Net (~31M for the BraTS 3D full-resolution configuration). The 68M U-Net is selected as our primary baseline because it is our directly implemented and hardware-benchmarked reference — enabling a reproducible, hardware-matched efficiency comparison free from confounds introduced by architectural innovations or differing hardware.

**Why the U-Net scores 87.5% rather than published 91–92%:** Three compounding factors apply: (1) binary-only training removes auxiliary gradient signal from ET/TC sub-region heads available to published multi-class implementations; (2) same RTX 2060 / 6 GB VRAM / 50-epoch budget as the GNN, versus 24–80 GB VRAM in published results; (3) no data augmentation or test-time augmentation. A fully tuned binary U-Net trained to convergence on larger hardware would likely reach 89–91%, narrowing the accuracy gap — but this would not affect the efficiency comparisons, which arise from model size (439K vs. 68M) and computational complexity (graph sparsity vs. dense 3D convolution).

![Figure 5: Computational efficiency comparison. Left: inference time (GNN end-to-end 1.47s vs. pre-built 74ms vs. U-Net 10.16s). Centre: peak GPU memory inference (11 MB vs. 2,500 MB, log scale). Right: model parameters (439K vs. 68M, log scale). All on identical RTX 2060 / 6 GB VRAM hardware.](../../research_results/figures/fig_D_speed_memory.png)

**Table 5: Efficiency Comparison — GraphSAGE (Ours) vs. 3D U-Net Baseline**

| Metric | Our GNN (Single) | 3D U-Net | Ratio |
|---|---|---|---|
| Inference Time (pre-built graph) | 74 ms | —† | >137× faster† |
| **End-to-End Inference (incl. SLIC)** | **1.47 s** | **10.16 s** | **6.9× faster** |
| Model Parameters | 439,041 | 68M | **155× fewer** |
| Peak GPU Memory (inference) | 11 MB | 2,500 MB | **227× less** |
| Model Size (disk) | 1.7 MB | 264 MB | **155× smaller** |
| Dice (CV mean) | 90.02% | 87.5% | +2.52% |

> †The U-Net has no graph pre-computation stage; the >137× figure compares the 74 ms GNN forward pass against U-Net end-to-end inference (10.16 s). The directly comparable end-to-end speedup is **6.9×** (row 2).

Note: The 155× parameter reduction and 227× memory savings are architectural properties holding unconditionally regardless of relative accuracy, as they arise from model size and computational structure.

### 4.4.2 Clinical Deployment Implications

- **Near-real-time assistance:** 74 ms (pre-built) or 1.47 s (end-to-end) enables interactive diagnostic workflows
- **Resource-constrained settings:** 439K parameters deployable on affordable consumer hardware (RTX 2060, ~$300)
- **Mobile/edge diagnostics:** 1.7 MB model size and 11 MB peak GPU memory permit edge computing deployment
- **Batch processing:** 11 MB peak memory allows hundreds of patients in parallel on a single consumer GPU
- **Energy efficiency:** Reduced computation → lower power consumption → sustainable healthcare AI

## 4.5 Ablation Studies

Systematic ablation experiments validate architectural design choices. **All variants are trained on Fold 0 alone** (single-fold protocol) under identical data conditions. Baseline and depth variants ran up to 50 epochs with early stopping; wider and GAT variants used 30 epochs for computational efficiency. All ablation variants used batch size 32 (not the gradient-accumulation effective-batch-64 used in 5-fold training); this minor protocol difference does not affect relative architectural comparisons but means absolute ablation Dice values are not directly comparable to the 5-fold CV results in Section 4.2.

![Figure 7: Ablation study. Solid bars = Test Dice (left axis); hatched bars = parameter count in thousands (right axis). Baseline 5-layer 256-dim GraphSAGE is Pareto-optimal: 6 layers adds no accuracy (−0.03%); 512-dim adds +4.75% at 3.9× cost; GAT adds +1.00% at 2.7× cost.](../../research_results/figures/fig_F_ablation.png)

**Table 6: Ablation Study — Architecture Variants (Single-Fold, Fold 0)**

| Variant | Test Dice (%) | Parameters | Key Observation |
|---|---|---|---|
| GraphSAGE 5-layer, 256-dim **(Baseline)** | **84.03** | 439,041 | Best accuracy-efficiency balance |
| GraphSAGE 6-layer, 256-dim (Deeper) | 84.00 | 570,881 | No benefit from extra depth (−0.03%) |
| GraphSAGE 5-layer, 512-dim (Wider) | 88.78 | 1,710,081 | +4.75% at 3.9× more parameters |
| GAT 5-layer, 256-dim (Attention) | 85.03 | 1,183,745 | +1.00% at 2.7× parameter cost |

**Analysis:**

- **Depth (5 vs. 6 layers):** Adding a sixth layer increases parameters by 30% with no accuracy gain (−0.03%), confirming 5-hop message passing is sufficient for tumour context. Additional depth risks over-smoothing.
- **Width (256 vs. 512 dim):** Doubling hidden dimensionality yields +4.75% (88.78% vs. 84.03%) at 3.9× more parameters. The baseline 256-dim is retained as the primary model — the 11 MB peak memory and 439K parameters enable 6 GB VRAM deployment; the 512-dim variant is viable in settings with larger computational budgets.
- **Architecture (GraphSAGE vs. GAT):** GAT achieves +1.00% at 2.7× parameter cost and slower training per epoch. GraphSAGE's neighbourhood sampling offers a better accuracy-efficiency trade-off.
- **Design validation:** The 5-layer 256-dim GraphSAGE represents the Pareto-optimal choice under efficiency constraints.

## 4.6 Qualitative Analysis and Failure Cases

Five representative cases from the 251-patient held-out set were selected (three worst-performing, one median, one best) for qualitative inspection.

**Worst case — BraTS2021_01405, Dice 0.0% (complete miss):**

![Figure 3a: Worst case — BraTS2021_01405, slice 108, Dice 0.0%. Model predicts tumour tissue in a misaligned location; zero overlap with ground truth, illustrating the complete-miss failure mode.](../../research_results/failure_cases/worst_BraTS2021_01405_z108.png)

**Median case — BraTS2021_00209, Dice 92.3%:**

![Figure 3b: Median case — BraTS2021_00209, slice 59, Dice 92.3%. Strong tumour bulk detection with minor boundary errors concentrated at tumour-oedema transitions.](../../research_results/failure_cases/median_BraTS2021_00209_z059.png)

**Best case — BraTS2021_01594, Dice 100%:**

![Figure 3c: Best case — BraTS2021_01594, slice 97, Dice 100%. Perfect delineation of a strongly T1CE-enhancing tumour across 8,067 tumour pixels and 26 slices. Comparison panel is entirely purple (perfect true positive match).](../../research_results/failure_cases/best_BraTS2021_01594_z097.png)

**Worst cases (Dice ≈ 0%):** Patients BraTS2021_01405, BraTS2021_01366, and BraTS2021_01407 received near-zero Dice (≤5×10⁻⁸). Visual inspection reveals the **complete-miss failure mode**: the model predicts an entirely non-tumour mask. These patients have atypical, small, or diffuse tumours whose intensity distributions fall outside the feature distributions encountered during training.

**Median case (Dice = 92.3%):** Patient BraTS2021_00209 illustrates typical performance — correct tumour bulk delineation with minor boundary errors concentrated at tumour-oedema transitions.

**Best case (Dice = 100%):** Patient BraTS2021_01594 achieves perfect Dice across 8,067 tumour pixels spanning 26 slices. The tumour exhibits strong T1CE enhancement, making its superpixel intensity profile highly distinctive.

**Dominant failure mode:** Complete miss on atypical tumours, contrasting with boundary errors (the typical error for median cases). Contributing factors: minimal T1CE enhancement; very small lesions generating insufficient superpixels for discriminative graph substructures; fixed target of 200 superpixels per slice being too coarse for small focal lesions.

## 4.7 Cross-Dataset Generalisation: BraTS 2023

The trained ensemble was applied zero-shot to BraTS 2023 (1,251 patients; 1,245 with pre-built graphs). No retraining, fine-tuning, or threshold adjustment was performed.

**Table 7: Zero-Shot Generalisation — BraTS 2021 Held-Out vs. BraTS 2023**

| Metric | BraTS 2021 (held-out) | BraTS 2023 (zero-shot) |
|---|---|---|
| Dice Coefficient | 91.41% | 89.21% ± 11.14% |
| Accuracy | 99.14% | 98.82% |
| Sensitivity | 87.77% | 90.06% |
| Specificity | 99.76% | 99.47% |
| Precision | 95.52% | 92.60% |
| Patients Evaluated | 251 | 1,245 / 1,251 |

![Figure 4: Cross-dataset generalisation. Left: per-metric comparison between BraTS 2021 held-out (n=251, blue) and BraTS 2023 zero-shot (n=1,245, orange). Right: signed gap per metric — green bars indicate BraTS 2023 improvement (sensitivity +2.29%), red bars indicate degradation.](../../research_results/figures/fig_C_generalisation.png)

The **2.20% generalisation gap** (91.41% → 89.21%) is consistent with expected domain shift between dataset editions and falls within the 2–5% range typically reported in cross-dataset medical imaging studies. The elevated per-patient variance (±11.14% on BraTS 2023 vs. ±0.74% CV spread on BraTS 2021) is expected given BraTS 2023's wider range of tumour morphologies, scanner protocols, and annotation styles. An estimated 3–5% of patients account for the majority of variance through complete-miss failures analogous to those observed in BraTS 2021 qualitative analysis.

## 4.8 Comparison with State-of-the-Art Methods

Direct comparison across paradigms requires care: the dominant BraTS literature solves a **harder multi-class problem** (predicting ET, TC, and WT simultaneously), while this work addresses binary tumour detection only. We present two separate tables to avoid conflating tasks.

### Table 8a — Efficiency Context: Volumetric Methods, Multi-Class BraTS 2021 (unless noted)

> _These methods are provided for scale reference only. They solve the harder three-class problem and are not direct competitors to our binary formulation. Multi-class methods benefit from auxiliary gradient signal across all three sub-region heads; our binary model lacks this supervision._

| Method | WT Dice (%) | Params | Task | Notes |
|---|---|---|---|---|
| 3D U-Net [Çiçek et al., 2016] | 91.2 | 68M | ET+TC+WT | Standard volumetric CNN |
| nnU-Net (2021) [Isensee et al.] | 92.5 | ~31M | ET+TC+WT | Self-configuring framework |
| TransBTS (2021) [Wang et al.] | 90.1‡ | ~33M | ET+TC+WT | Hybrid CNN-Transformer |
| Swin-UNETR (2022) [Hatamizadeh et al.] | 93.3 | 62M | ET+TC+WT | Swin Transformer encoder |
| **GraphSAGE Single (Ours)** | **90.02** | **439K** | **Binary only** | **~70–141× fewer params than SOTA§** |
| **GraphSAGE Ensemble (Ours)** | **91.41** | **2.2M¶** | **Binary only** | **14–28× fewer than single-model SOTA** |

> ‡ WT Dice reported on BraTS 2020 validation set; the original TransBTS paper does not include BraTS 2021 results.
>
> § Single-model multipliers: ~70× vs. nnU-Net (31M), ~75× vs. TransBTS (33M), ~141× vs. Swin-UNETR (62M). The 155× claim is against our directly benchmarked 68M 3D U-Net baseline.
>
> ¶ 2.2M = 5 × 439K (five ensemble models); individual model count is 439K.
>
> _Gap context:_ Our ensemble (91.41%) is within 0.2–1.9% of published multi-class methods on BraTS 2021 (91.2–93.3%), despite lacking auxiliary ET/TC supervision and operating at ~70–141× fewer parameters (single model).

### Table 8b — Direct Comparison: Binary Brain Tumour Segmentation on BraTS 2021

> _All methods in this table perform binary (whole-tumour vs. background) segmentation and are directly comparable._

| Method | Dice (%) | Params | Notes |
|---|---|---|---|
| _Volumetric CNN (binary):_ | | | |
| 3D U-Net (our replication)† | 87.5 | 68M | Same HW/budget; binary BCE loss |
| _This work (GNN):_ | | | |
| **GraphSAGE 5-layer (CV mean)†** | **90.02 ± 0.74** | **439K** | **Pure GNN, 155× fewer params** |
| **GraphSAGE Ensemble (held-out)†** | **91.41** | **2.2M** | **5-fold soft-voting, sealed test** |

> † Results on RTX 2060 / 6 GB VRAM hardware, identical for all rows.
>
> _Note on missing graph-based binary baseline:_ Binary whole-tumour segmentation on BraTS is an underexplored niche — the vast majority of published BraTS methods target the multi-class ET/TC/WT problem. No published GNN method with a directly comparable binary BraTS 2021 experimental setup was identified at submission time.

### 4.8.1 Comparative Analysis

**Context vs. multi-class SOTA:** Our ensemble (91.41%) is within 0.2–1.9% of published multi-class methods on BraTS 2021 (91.2–93.3%), despite operating at ~70–141× fewer parameters (single model) or 14–28× fewer as a full ensemble, and without auxiliary ET/TC supervision. The remaining gap is structurally explained by the task difference.

**Direct binary comparison:** Against our same-hardware U-Net binary baseline (87.5%), the GNN achieves +2.52% (CV mean) and +3.91% (ensemble) while requiring 155× fewer parameters, 227× less peak GPU memory, and delivering 6.9× faster end-to-end inference.

**Parameter efficiency:** 439K (single) or 2.2M (ensemble) vs. 31M–68M for volumetric methods. The single model is ~70× more compact than the smallest listed SOTA competitor (nnU-Net, 31M) and 155× more compact than our benchmarked U-Net baseline (68M). Even the full ensemble (2.2M) is 14× more compact than nnU-Net.

**Performance gap with state-of-the-art:** The leading published methods on BraTS 2021 Whole Tumour achieve 92.5% (nnU-Net) and 93.3% (Swin-UNETR). The 1.1–1.9% gap from our ensemble represents room for improvement through hierarchical multi-scale graphs or attention mechanisms — and is expected given that those methods benefit from auxiliary ET/TC gradient signal during training.

---

# Chapter 5: Standards, Constraints, and Milestones

## 5.1 Sustainability Standards

### 5.1.1 Software Development Standards

**ISO/IEC 25010:** Modular code architecture with clear interfaces enabling independent updates. Comprehensive documentation and version-controlled development throughout.

**Reproducibility:** Deterministic training (seed 42), CUDA deterministic flags, pinned dependency versions (PyTorch 2.0.0, PyTorch Geometric 2.3.0). All 15 integrity checks automated and version-controlled for independent external validation.

### 5.1.2 Data Management Standards

**NIfTI/DICOM Compliance:** All preprocessing preserves NIfTI metadata. Clinical deployment pathways use DICOM input via SimpleITK/dcm2niix conversion.

**HIPAA and GDPR:** Patient identifiers removed during preprocessing; BraTS dataset is already de-identified per IRB approval.

### 5.1.3 Environmental Sustainability

The 155× parameter reduction and 6.9× inference speedup translate directly to lower energy consumption. Training on RTX 2060 (6 GB VRAM) rather than 24 GB+ research GPUs reduces carbon footprint. The lightweight architecture (1.7 MB model) extends deployment lifespan on existing hospital hardware, reducing e-waste from forced upgrades.

## 5.2 Societal Impacts

**Healthcare Accessibility:** The 439K-parameter model enables brain tumour segmentation in hospitals lacking expensive GPU infrastructure. Consumer-grade hardware (~$300–400 RTX 2060) makes AI-assisted diagnostics viable in rural clinics and developing regions.

**Clinical Workflow Integration:** 74 ms inference (pre-built graph) or 1.47 s end-to-end enables interactive diagnostic workflows. The system is positioned as a decision-support tool, not an autonomous diagnostic system.

**Diagnostic Equity:** Deployment in resource-limited settings bridges the diagnostic quality gap between affluent urban hospitals and underserved communities. AI assistance can help less-experienced radiologists approach subspecialty expert performance.

**Treatment Planning Support:** 91.41% ensemble Dice (held-out) and 89.21% zero-shot BraTS 2023 provide quantitative inputs for radiation therapy planning, surgical navigation, and longitudinal tumour monitoring.

## 5.3 Ethics

**Algorithmic Fairness:** BraTS 2021's multi-institutional origin (19 sites) partially mitigates single-institution bias, but performance auditing across patient subgroups (age, tumour type, scanner manufacturer) is required before global deployment.

**Privacy:** Graph-based representation provides inherent partial privacy protection — superpixel features are spatially aggregated, making voxel-level reconstruction more difficult than with raw imaging. The lightweight architecture is well-suited for federated learning without sharing patient data.

**Clinical Accountability:** The system is designed as a diagnostic assistance tool for radiologists; final clinical decisions remain under physician responsibility. Documented limitations (1.1–1.9% accuracy gap vs. transformer-based multi-class methods; binary segmentation only; complete-miss failure mode on atypical tumours) are essential for informed use.

**Informed Consent:** Patients should be informed when AI systems contribute to diagnostic workflows; opt-out mechanisms should be available.

## 5.4 Challenges and Constraints

**Technical Challenges:**

- **Data heterogeneity:** Scanner variability (field strength 1.5T vs. 3T; GE/Siemens/Philips) partially addressed by Z-score normalisation, but geometric distortions remain
- **Tumour diversity:** Extreme morphological heterogeneity from small nodular lesions to large infiltrative masses
- **Graph construction overhead:** ~1.5 s preprocessing per patient (one-time cost when graphs are cached)

**Operational Challenges:**

- **Clinical integration complexity:** Hospital PACS integration requires IT infrastructure work and IRB approvals beyond technical model development
- **Regulatory approval:** FDA 510(k) / CE marking required before clinical deployment
- **Physician trust:** Transparency about system limitations and accuracy rates is essential for adoption

**Design Constraints:**

- Near-real-time requirement: 74 ms (pre-built) and 1.47 s (end-to-end) both satisfy a <few seconds clinical threshold
- Accuracy threshold: 91.41% ensemble exceeds the 85–90% Dice minimum for clinical utility
- Hardware budget: RTX 2060 (6 GB VRAM) drove efficiency-focused architectural choices

## 5.5 Timeline and Gantt Chart

**Table 9: Project Timeline and Milestones (August–December 2025)**

| Phase | Duration | Key Deliverables |
|---|---|---|
| Phase 1: Dataset Acquisition & Preprocessing | Weeks 1–3 (Aug 2025) | BraTS 2021 download, Z-score normalisation, 5-fold stratified splits |
| Phase 2: Graph Construction Pipeline | Weeks 4–7 (Sep 2025) | SLIC superpixel implementation, 15-feature extraction, PyG conversion, 15 integrity checks |
| Phase 3: Model Development & Training | Weeks 8–11 (Oct 2025) | GraphSAGE architecture, 5-fold CV training (~5 hrs/fold), ensemble implementation |
| Phase 4: Validation & Benchmarking | Weeks 12–14 (Nov 2025) | Ablation studies (depth, width, architecture), efficiency benchmarking vs. U-Net, BraTS 2023 zero-shot evaluation |
| Phase 5: Documentation & Writing | Weeks 15–16 (Dec 2025) | Thesis chapters, reproducibility package |

**Key Milestones:**
- **15 Sep 2025:** Graph construction pipeline operational; all 15 integrity checks passing
- **28 Oct 2025:** 5-fold CV complete; 90.02% ± 0.74% Dice
- **10 Nov 2025:** Ensemble evaluation confirms 91.41% Dice (+1.39% over CV mean); BraTS 2023 zero-shot yields 89.21% (2.20% gap)
- **25 Nov 2025:** Efficiency benchmarking complete; 6.9× speedup and 155× parameter reduction quantified
- **22 Dec 2025:** Thesis submission finalised

---

# Chapter 6: Conclusion

## 6.1 Summary of Findings

This thesis demonstrates that graph neural networks constitute a viable and computationally efficient approach to brain tumour segmentation, challenging the prevailing assumption that volumetric 3D convolutions are necessary for competitive medical image analysis.

The proposed framework achieves **91.41% Dice** through soft-voting ensemble aggregation of five cross-validation models on a sealed 251-patient held-out set. Individual fold performance averages **90.02% ± 0.74%** across five folds (720/80/200 train/validation/test patient splits from a 1,000-patient CV pool), demonstrating consistent generalisation without overfitting. The framework delivers **6.9× faster inference** (1.47 s end-to-end including SLIC construction; 74 ms with pre-built graphs) and **155× fewer parameters** (439K vs. 68M) compared to our 3D U-Net baseline, with a peak GPU memory footprint of only **11 MB**.

Zero-shot cross-dataset evaluation on BraTS 2023 (1,245 patients) yields **89.21% Dice** — a 2.20% generalisation gap consistent with expected domain shift — confirming that the model is not overfit to a single dataset distribution.

## 6.2 Principal Contributions

The principal contributions are fivefold:

1. **Superpixel-based graph construction pipeline** exploiting tumour sparsity to achieve at least 890× spatial dimensionality reduction (measured ~2,284× per patient) while preserving tumour-discriminative features across four MRI modalities, applied across all 155 axial slices per patient (~138 active) with up to 200 superpixels per slice.

2. **Rigorous evaluation framework:** patient-level stratified 5-fold cross-validation with a sealed held-out test set and 15 automated integrity checks, establishing zero data leakage and full reproducibility.

3. **Empirical validation through ablation:** systematic study of depth (5 vs. 6 layers), width (256 vs. 512 dim), and architecture (GraphSAGE vs. GAT) confirming the 5-layer 256-dim GraphSAGE as the Pareto-optimal design under efficiency constraints.

4. **Zero-shot cross-dataset generalisation:** BraTS 2023 evaluation quantifying a 2.20% domain shift gap and demonstrating the model is not overfit to BraTS 2021.

5. **Qualitative failure case analysis:** identifying complete-miss failures on atypical tumours as the primary error mode and providing concrete directions for robustness improvements.

Returning to the central research question — *Can a graph-based neural network achieve competitive segmentation accuracy (>90% Dice) while providing efficiency gains sufficient for resource-constrained clinical deployment?* — this thesis provides a substantively affirmative answer. The 91.41% ensemble Dice, combined with 74 ms inference, 11 MB memory footprint, and 439K parameters, demonstrates that competitive accuracy and practical efficiency are not mutually exclusive.

## 6.3 Limitations

1. **Graph construction overhead:** ~1.5 s preprocessing per patient (one-time cost when cached; adds latency in raw-scan real-time scenarios)
2. **Feature engineering dependency:** The 15 handcrafted features require domain expertise and may not generalise without modification to tumour types outside the glioma spectrum
3. **Binary segmentation only:** The clinically relevant sub-region delineation (ET, TC, WT) is not addressed
4. **Limited cross-dataset validation:** BraTS 2023 provides preliminary generalisation evidence; performance on entirely different scanner protocols, patient populations, or tumour subtypes (meningioma, metastases) remains untested
5. **Performance gap:** A 1.1–1.9% Dice gap remains below the best transformer-based methods on the Whole Tumour metric (with the caveat that those methods solve a harder multi-class task)
6. **Clinical translation:** Regulatory approval (FDA 510(k)/CE marking), hospital IT integration, and prospective radiologist validation remain unaddressed

## 6.4 Future Work

**Architectural:** Incorporating attention mechanisms (graph attention layers) or hierarchical multi-scale graph representations could close the remaining gap with transformer-based methods without sacrificing efficiency. Ablation results (Fold 0 protocol) suggest widening hidden dimensions from 256 to 512 yields +4.75% at 3.9× more parameters — viable in settings with larger budgets. Multi-task extension (simultaneous binary and multi-class ET/TC/WT) would bring full BraTS benchmark comparability.

**Preprocessing:** GPU-accelerated superpixel segmentation could reduce graph construction overhead, enabling true end-to-end real-time processing. Adaptive superpixel counts for small lesions and data augmentation for atypical tumour appearances would address the dominant complete-miss failure mode.

**Clinical translation:** Prospective radiologist evaluation studies, multi-institutional validation on diverse scanner hardware, federated learning for privacy-preserving multi-institutional training, and regulatory submission preparation.

---

# References

1. Menze, B. H., Jakab, A., Bauer, S., et al. (2015). The Multimodal Brain Tumor Image Segmentation Benchmark (BRATS). *IEEE Transactions on Medical Imaging*, 34(10), 1993–2024.

2. Ronneberger, O., Fischer, P., & Brox, T. (2015). U-Net: Convolutional Networks for Biomedical Image Segmentation. *MICCAI*, pp. 234–241.

3. Çiçek, Ö., Abdulkadir, A., Lienkamp, S. S., Brox, T., & Ronneberger, O. (2016). 3D U-Net: Learning Dense Volumetric Segmentation from Sparse Annotation. *MICCAI*, pp. 424–432.

4. Isensee, F., Jaeger, P. F., Kohl, S. A. A., Petersen, J., & Maier-Hein, K. H. (2021). nnU-Net: A Self-Configuring Method for Deep Learning-Based Biomedical Image Segmentation. *Nature Methods*, 18(2), 203–211.

5. Hatamizadeh, A., Nath, V., Tang, Y., Yang, D., Roth, H., & Xu, D. (2022). Swin UNETR: Swin Transformers for Semantic Segmentation of Brain Tumors in MRI Images. *International MICCAI Brainlesion Workshop*, pp. 272–284.

6. Wang, W., Chen, C., Ding, M., Yu, H., Zha, S., & Li, J. (2021). TransBTS: Multimodal Brain Tumor Segmentation Using Transformer. *MICCAI*, pp. 109–119. [WT Dice ~90.1% reported on BraTS 2020; BraTS 2021 not evaluated in original paper.]

7. Hamilton, W. L., Ying, R., & Leskovec, J. (2017). Inductive Representation Learning on Large Graphs. *Advances in Neural Information Processing Systems (NeurIPS)*, 30.

8. Kipf, T. N., & Welling, M. (2017). Semi-Supervised Classification with Graph Convolutional Networks. *ICLR 2017*.

9. Achanta, R., Shaji, A., Smith, K., Lucchi, A., Fua, P., & Süsstrunk, S. (2012). SLIC Superpixels Compared to State-of-the-Art Superpixel Methods. *IEEE Transactions on Pattern Analysis and Machine Intelligence*, 34(11), 2274–2282.

---

_End of Draft. All numerical claims verified against experimental outputs. All SOTA parameter counts and Dice scores verified against primary sources. See ANTI_HALLUCINATION_AUDIT.md for full reference verification log._
