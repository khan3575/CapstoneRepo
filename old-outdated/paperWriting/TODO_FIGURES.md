# TODO: Figures to Create for Thesis

## Status: December 22, 2025

### ✅ COMPLETED (Images Added)
1. **CV Performance Visualization** - Added to Chapter 4
   - `cv_dice_per_fold.png` - Bar plot of fold performance (Figure after Table 4.1)
   - `cv_boxplots.png` - Boxplot distribution (Figure after cv_dice)
   
2. **Qualitative Segmentation Examples** - Added to Chapter 4
   - `BraTS2021_00501_slice149.png` - Example 1 (before Key Findings section)
   - `BraTS2021_00491_slice086.png` - Example 2
   - `BraTS2021_00559_slice105.png` - Example 3

---

## 🔴 CRITICAL: Figures You MUST Create (Supervisor's Red Flags)

### 1. Figure ?? - Pipeline Architecture (Chapter 3, Page ~15)
**Location:** Section 3.1 "Proposed Framework"
**Current text:** "Figure ?? illustrates the complete pipeline architecture."

**What to create:**
A horizontal flowchart showing the 8-phase pipeline:

```
Raw MRI    →    Preprocessing    →    Graph         →    Feature       →    GNN Training
(4 modalities)   (skull-strip,         Construction       Engineering        (5-layer GraphSAGE,
T1/T1CE/          normalize,            (SLIC superpixels  (15 features:      256 hidden dims,
T2/FLAIR)         200 slices)           80-100/slice,      intensity stats,   BCE loss,
                                        ~10K nodes)         spatial, geometric) batch 32)
                                        
    ↓                    ↓                    ↓
    
Cross-Validation  →  Ensemble        →  Benchmarking    →  Validation
(5-fold stratified,   (soft voting,       (vs U-Net:         (15 integrity
patient-level,        92.92% Dice)        6.9× speedup)      checks)
90.39±0.69% Dice)
```

**Tool:** PowerPoint, draw.io, or Lucidchart
**Save as:** `image/pipeline_architecture.png`

**LaTeX insertion point (Chapter 3, after "Proposed Framework" intro):**
```latex
\begin{figure}[htbp]
\centering
\includegraphics[width=\textwidth]{image/pipeline_architecture.png}
\caption{Complete 8-Phase Pipeline Architecture. The framework progresses from raw multi-parametric MRI input through preprocessing, graph construction with superpixel-based dimensionality reduction (890× compression), feature engineering (15 domain-specific features), GraphSAGE training, patient-level cross-validation, ensemble aggregation, efficiency benchmarking, and comprehensive validation. Each phase incorporates integrity checks to prevent data leakage.}
\label{fig:framework}
\end{figure}
```

---

### 2. Algorithm ?? - Graph Construction Pseudocode (Chapter 3, Page ~20)
**Location:** Section 3.3.1 "Graph Construction Algorithm"
**Current text:** "Algorithm ?? details the procedure."

**Option A: Create Algorithm Box (Recommended)**

Use LaTeX algorithm2e package:

```latex
\begin{algorithm}[htbp]
\caption{Superpixel-Based Graph Construction}
\label{alg:graph_construction}
\SetAlgoLined
\KwIn{3D MRI volume $V \in \mathbb{R}^{240 \times 240 \times 155 \times 4}$ (4 modalities), Ground truth mask $M$}
\KwOut{Graph $G = (V_{nodes}, E_{edges}, X_{features}, Y_{labels})$}

\textbf{Step 1: Slice Extraction}\;
Extract 200 axial slices: $S = \{s_i | i \in [20, 220]\}$ from volume $V$\;

\textbf{Step 2: Superpixel Generation}\;
\For{each slice $s_i \in S$}{
    Apply SLIC($s_i$, n\_segments=80-100, compactness=10) $\rightarrow$ superpixels $SP_i$\;
}

\textbf{Step 3: Feature Extraction}\;
\For{each superpixel $sp_j \in SP_i$}{
    Extract 15 features: intensity stats (mean, std, min, max, median) × 4 modalities + spatial (x, y) + geometric (area)\;
    $X_j \leftarrow$ [feature vector]\;
}

\textbf{Step 4: Edge Construction}\;
Build adjacency matrix: Connect superpixels $sp_j, sp_k$ if they share boundary\;
$E \leftarrow \{(j, k) | sp_j \text{ adjacent to } sp_k\}$\;

\textbf{Step 5: Label Assignment}\;
\For{each superpixel $sp_j$}{
    $Y_j \leftarrow 1$ if $>$50\% of pixels in $sp_j$ are tumor (from $M$), else $Y_j \leftarrow 0$\;
}

\Return Graph $G = (V_{nodes}, E, X, Y)$ with $\sim$10,000 nodes, 890× compression\;
\end{algorithm}
```

**Add to main.tex preamble:**
```latex
\usepackage[ruled,vlined]{algorithm2e}
```

**Option B: Simple Text Replacement (If Algorithm Package Issues)**

Change the text in chapter3.tex from:
> "Algorithm ?? details the procedure."

To:
> "The graph construction procedure consists of five steps detailed below:"

---

### 3. Figure (Optional but Recommended) - GraphSAGE Architecture Diagram
**Location:** Section 3.3.2 "GraphSAGE Architecture" (Chapter 3)
**Currently:** No figure, just equations

**What to create:**
A vertical diagram showing the 5-layer GraphSAGE architecture:

```
Input Layer (15 features)
    ↓ [Message Passing Layer 1: Aggregate → Concat → Transform]
Hidden Layer 1 (256 dims)
    ↓ [Message Passing Layer 2]
Hidden Layer 2 (256 dims)
    ↓ [Message Passing Layer 3]
Hidden Layer 3 (256 dims)
    ↓ [Message Passing Layer 4]
Hidden Layer 4 (256 dims)
    ↓ [Message Passing Layer 5]
Hidden Layer 5 (256 dims)
    ↓ [Output Layer: Linear → Sigmoid]
Output (1 dim: tumor probability)

Total Parameters: 439,041
Receptive Field: 5-hop neighborhood
```

**Tool:** PowerPoint or draw.io
**Save as:** `image/graphsage_architecture.png`

**LaTeX insertion (Chapter 3, after GraphSAGE equations):**
```latex
\begin{figure}[htbp]
\centering
\includegraphics[width=0.6\textwidth]{image/graphsage_architecture.png}
\caption{5-Layer GraphSAGE Architecture. Each layer performs message passing (aggregate neighbors → concatenate with self → linear transform → ReLU activation). The 5-layer depth provides 5-hop receptive field sufficient for tumor context aggregation, as validated through ablation studies showing no benefit from 6 layers.}
\label{fig:graphsage_arch}
\end{figure}
```

---

## 📋 CHECKLIST: Final Figure Insertion

Before submitting thesis:

- [ ] **Create Pipeline Architecture diagram** (Figure ??, page 15, Chapter 3)
  - Use PowerPoint/draw.io
  - Show all 8 phases with arrows
  - Save as `image/pipeline_architecture.png`
  - Insert LaTeX code with \includegraphics and proper caption
  - Update \ref{fig:framework} to work

- [ ] **Fix Algorithm ?? reference** (page 20, Chapter 3)
  - Either: Add algorithm2e package + create algorithm box
  - Or: Change text to remove "Algorithm ??" reference

- [ ] **Optional: Create GraphSAGE architecture diagram**
  - Visual representation of 5 layers
  - Save as `image/graphsage_architecture.png`
  - Insert in Section 3.3.2

- [ ] **Regenerate List of Figures** (page 8)
  - After adding images, recompile LaTeX 2-3 times
  - Check that List of Figures auto-populates
  - Should show:
    - Figure 4.1: CV Dice Scores (page X)
    - Figure 4.2: CV Boxplots (page Y)
    - Figure 4.3: Qualitative Example 1 (page Z)
    - Figure 4.4: Qualitative Example 2
    - Figure 4.5: Qualitative Example 3
    - Figure 3.1: Pipeline Architecture (if added)
    - Figure 3.2: GraphSAGE Architecture (if added)

- [ ] **Final compilation check**
  ```bash
  cd paperWriting/Template
  pdflatex main.tex
  biber main
  pdflatex main.tex
  pdflatex main.tex  # Third time to update List of Figures
  ```

---

## 🎯 Supervisor's Expectation

> "Spend 2 hours creating 3 images (Pipeline, Architecture, Results Overlay), insert them to fix the ?? errors, and you are ready to print."

**Status:** 
- ✅ Results overlays added (3 qualitative examples)
- ✅ CV performance plots added (2 figures)
- 🔴 Pipeline diagram CRITICAL - MUST CREATE
- 🔴 Algorithm/Architecture diagram - MUST FIX
- 🔴 List of Figures will auto-generate after recompilation

**Time estimate:** 1-2 hours for pipeline + architecture diagrams in PowerPoint/draw.io

---

## 📝 Tools Recommended

1. **draw.io** (https://app.diagrams.net) - Free, browser-based
   - Best for: Pipeline flowchart, architecture diagrams
   - Export as PNG (300 DPI recommended)

2. **PowerPoint** - If you have Microsoft Office
   - Best for: Quick block diagrams
   - Use shapes: Rectangle, Arrow, Text box
   - Save as PNG (high resolution)

3. **Lucidchart** - Alternative online tool
   - Professional templates available
   - Free tier sufficient for simple diagrams

---

## 💡 Quick Pipeline Diagram Tutorial (PowerPoint)

1. Open PowerPoint, blank slide
2. Insert → Shapes → Rectangle (for each phase)
3. Insert → Shapes → Arrow (between rectangles)
4. Label each rectangle: "Preprocessing", "Graph Construction", etc.
5. Add small text below each box with key details
6. Use consistent colors: Blue for input, Green for processing, Red for output
7. File → Save As → PNG (or Export → PNG)
8. Copy PNG to `paperWriting/Template/image/`
9. Add LaTeX \includegraphics code

**Estimated time:** 30 minutes for pipeline, 20 minutes for architecture

Good luck! 🎓
