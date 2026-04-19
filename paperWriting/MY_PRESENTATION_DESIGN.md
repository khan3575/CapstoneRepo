# My Redesigned Presentation: GNN Brain Tumour Segmentation
*How I would present this paper from scratch — independent judgment on structure, narrative, and emphasis*

---

## Core Narrative Strategy

The existing slides lead with *methodology*, then arrive at efficiency gains late (Slide 19).
This buries the lede.

**The real story is:** Everyone assumes deep learning = big GPU. We broke that assumption.
Our tiny GNN (439K parameters, 6GB GPU) *outperforms* a full U-Net (69M parameters, 2,500MB memory).
That is the hook. Everything else — graphs, superpixels, SAGE — is the *explanation* of how we did it.

**Narrative arc:**
```
THE PROBLEM (clinical stakes)
  → THE TRAP (CNNs work but are too heavy)
    → THE INSIGHT (brains have structure; exploit it as a graph)
      → THE MACHINE (how the graph gets built and learned)
        → THE PROOF (numbers: it works AND it's 5.9× faster)
          → THE GENERALIZATION (it holds on unseen data)
            → THE HONEST ACCOUNTING (failures, limits)
              → THE VISION (where this leads)
```

---

## Slide Count: 22 Slides

Lean deck. Every slide earns its place.

---

## Slide-by-Slide Design

---

### SLIDE 1 — Title
**Layout:** Dark background. Single large headline. Minimal.

```
EFFICIENT BRAIN TUMOUR SEGMENTATION
USING GRAPH NEURAL NETWORKS

BraTS 2021 · BraTS 2023
BUBT · 2025

Team: Sakib Khan, Rifa Sanjida, Kishor Kumar Das,
      Md. Mahamudul Hasan, Md. Minhajur Rahman
Supervisor: Mr. Shamim Ahmed
```

**No logo soup. No decorative borders. Clean.**

---

### SLIDE 2 — The Human Stake
**Layout:** One powerful MRI image (real BraTS slice). One stat. One sentence.

```
[Full-bleed axial MRI slice showing tumour]

330,000 new brain tumour diagnoses per year globally.
Accurate segmentation determines whether surgery succeeds.
```

*Why:* Ground the committee in clinical reality before any algorithm.
The research matters because people die. Say that first.

---

### SLIDE 3 — What Segmentation Actually Looks Like
**Layout:** 2×2 grid — raw T1, T1CE, T2, FLAIR. Then a side panel showing the ground-truth mask overlaid.

```
Four MRI Modalities → One Binary Mask

T1  | T1CE         [mask overlay on T1CE]
T2  | FLAIR
```

**One annotation:** "We want to automate this — reliably, quickly, on any hospital's hardware."

*Why:* Most committee members are not medical imaging specialists. 
Show them what the input/output actually is before explaining the method.

---

### SLIDE 4 — The Problem with Current Best Practice
**Layout:** Two-column table. Left = CNN methods. Right = their cost.

```
STATE OF THE ART          WHAT IT COSTS
─────────────────────────────────────────
Swin-UNETR    93.3%  →   62M params, 16–32GB VRAM
nnU-Net       92.7%  →   31M params, requires cluster
TransBTS      90.1%  →   33M params, 16GB VRAM
3D U-Net      ~91%   →   68M params, 2,500MB peak memory

Most hospitals in developing countries:
consumer-grade GPU, 6–8GB VRAM, 1 machine.
```

**Bold bottom line:** "High accuracy exists. Accessibility does not."

*Why:* This is the gap our work fills. Make the committee feel the gap.

---

### SLIDE 5 — Our Claim (Before We Show Evidence)
**Layout:** Single bold statement. No bullets.

```
We built a model with 439,000 parameters
that outperforms a 69-million-parameter U-Net
on the same hardware.

91.41% Dice.   5.9× faster.   227× less memory.
```

*Why:* State the claim early. The rest of the talk is the proof.
This is how good talks work — thesis first, evidence second.

---

### SLIDE 6 — The Key Insight: Brain Has Structure
**Layout:** Annotated MRI cross-section with region boundaries drawn.

```
[MRI slice with SLIC superpixel boundaries overlaid]

Pixels are not the right unit of analysis.
Anatomical regions are.

Tumour boundaries follow tissue transitions.
Graph edges can model tissue adjacency.
```

**Annotation arrows:** "~46 regions per slice, not 145,600 pixels"

*Why:* This is the conceptual leap of the paper — from grid to graph.
Make the audience *see* why the graph representation makes sense
before explaining how it's built.

---

### SLIDE 7 — Graph Construction (The Recipe)
**Layout:** Four-step horizontal pipeline with icons.

```
Step 1: SLIC Superpixels
Applied to T1CE channel (highest tumour contrast)
K=200 target → ~46 superpixels per slice
2,284× compression: 8.9M voxels → 3,909 nodes/patient

Step 2: Paired-Slice Graph  
Unit: 2 consecutive axial slices
~92 nodes, ~180 edges per graph unit

Step 3: Two Edge Types
• Intra-slice: RAG adjacency (tissue borders)
• Inter-slice: kNN (k=3), IoU > 0.1, distance < 10mm

Step 4: 15-Dimensional Node Features
Intensity (8): mean+std for T1, T1CE, T2, FLAIR
Spatial  (4): area, normalised area, centroid x/y
Morphology (3): perimeter, compactness, intensity range
```

*Why:* This is the most technical slide. Use a pipeline visual, not a wall of text.
Let the audience absorb the compression number: **2,284×**. That's the secret weapon.

---

### SLIDE 8 — Why GraphSAGE (Not GCN, Not GAT)
**Layout:** Side-by-side: GCN limitation vs GraphSAGE solution. One comparison table.

```
GCN / GAT                    GraphSAGE
─────────────────────────────────────────────
Transductive:                Inductive:
retrains for new graphs      works on unseen graphs

Cannot generalise            Zero-shot transfer
to BraTS 2023                to BraTS 2023

Fixed adjacency matrix       Learns neighbourhood
needed at train time         sampling function

                  ↓ Our choice ↓

5 layers · 256 hidden dim · 64 output dim
Mean aggregation · 439,041 parameters
```

*Why:* GraphSAGE being inductive is *the architectural reason* we can do zero-shot generalisation.
This slide makes that causal link explicit. Audiences often miss it.

---

### SLIDE 9 — Training Setup (Compact)
**Layout:** Two-column card. Left = protocol. Right = hardware.

```
PROTOCOL                     HARDWARE
────────────────────         ─────────────────────
5-fold CV                    Intel i7-10700
1,000 BraTS 2021 patients    32GB RAM
720 train / 80 val /         NVIDIA RTX 2060
200 test per fold            6GB VRAM only

AdamW · LR 1e-3              251 patients
OneCycleLR · 30% warmup      sealed held-out set
Batch 24 · AMP FP16          (never used in training)
BCEWithLogits · w+ = 9.0
(19:1 class imbalance)
```

*Why:* Single slide. Not a dissertation chapter. 
Highlight the 6GB VRAM constraint — everything was built to fit there.

---

### SLIDE 10 — Cross-Validation Results
**Layout:** Bar chart (fold performance) + summary box.

```
[Bar chart: 5 folds with heights ~88–90%]

Fold 0: 88.72%
Fold 1: 90.48%
Fold 2: 90.31%
Fold 3: 90.13%
Fold 4: 90.47%

Mean: 90.02% ± 0.66%

Low variance across folds → stable, reproducible model.
```

*Why:* Variance matters as much as mean. Low std = trustworthy training.
Show both explicitly.

---

### SLIDE 11 — Held-Out Test: The Sealed Envelope Result
**Layout:** Large metric display. Use visual weight to signal importance.

```
 SEALED HELD-OUT SET
 251 Patients — Never Touched During Training

 ┌────────────────────────────────────┐
 │  Dice        91.41%                │
 │  Accuracy    99.14%                │
 │  Precision   95.52%                │
 │  Sensitivity 87.77%                │  
 │  Specificity 99.76%                │
 └────────────────────────────────────┘

Higher than 5-fold mean (90.02%) — no overfitting.
```

**Note:** "Sealed = we only ran inference once. No tuning on this set."

*Why:* The sealed held-out result is the most credible number in the paper.
Make the audience understand *why it matters* — it's an honest test.

---

### SLIDE 12 — The Efficiency Comparison (Core Contribution)
**Layout:** Head-to-head comparison table with visual magnitude indicators.

```
                   GNN (Ours)     U-Net        Advantage
────────────────────────────────────────────────────────
Dice Score         91.41%         87.84%       +3.57 pp  ▲
Inference Time     1,732 ms       10,160 ms    5.9×  faster
Peak Memory        11 MB          2,500 MB     227×  less  ◀◀◀
Parameters         439K           69.1M        157×  fewer
Storage            1.7 MB         264 MB       157×  smaller
```

*Why:* This slide IS the paper's contribution in table form.
Use visual cues (arrows, bold) so the audience immediately knows
"less is better" for all non-Dice rows.

**Key talking point:** "We don't just match U-Net. We beat it. On the same machine. With 157× fewer parameters."

---

### SLIDE 13 — Ablation Study: Why This Architecture
**Layout:** Small table + one-sentence takeaway.

```
Architecture Variant    Dice     Parameters   Verdict
─────────────────────────────────────────────────────
Ours (5L, 256-dim)      84.03%   439K         ✓ Optimal
6 Layers                84.00%   571K         Same Dice, heavier
512-dim hidden          88.78%   1,710K       Better but 4× bigger
GAT (attention)         85.03%   1,184K       Worse Dice, much heavier

Note: 84.03% is Fold 0 only; ensemble reaches 91.41%.
```

**Takeaway:** "Adding depth or width beyond our design buys little accuracy per parameter. We found the knee of the curve."

*Why:* Ablation validates that the architectural choices weren't arbitrary.
The committee will ask "why not bigger?" — this answers it preemptively.

---

### SLIDE 14 — Zero-Shot Generalisation
**Layout:** Side-by-side metric comparison. Highlight the Sensitivity improvement.

```
                  BraTS 2021     BraTS 2023     Δ
                  (trained on)   (zero-shot)
──────────────────────────────────────────────────
Dice              91.41%         89.40%         −1.01 pp
Accuracy          99.14%         98.85%         −0.29 pp
Sensitivity       87.77%    →    90.69%         +2.92 pp  ▲ IMPROVED
Specificity       99.76%         99.45%         −0.31 pp
Precision         95.52%         92.46%         −3.06 pp

No retraining. Different acquisition protocol.
```

**Talking point:** "Sensitivity — the ability to find tumour — actually *improved* on data the model never saw.
That is the inductive property of GraphSAGE in action."

*Why:* The sensitivity improvement is the most surprising result in the paper.
Lead with it, not the Dice drop.

---

### SLIDE 15 — Compared to State of the Art
**Layout:** Scatter plot — x-axis = parameters (log scale), y-axis = Dice score. Our model is a labelled dot.

```
[Scatter plot]

Swin-UNETR  93.3%  ●   62M params
nnU-Net     92.7%  ●   31M params
3D U-Net    91.0%  ●   68M params  
OURS        91.4%  ★   439K params   ← optimal zone
TransBTS    90.1%  ●   33M params
Patel GNN   84.3%  ●   10.2M params
```

**Takeaway:** "Near-SOTA accuracy. Orders of magnitude smaller. The only model in the accessible zone."

*Why:* A scatter plot makes the Pareto frontier immediately visible.
We are not the most accurate — we're the most efficient near the top. That's the story.

---

### SLIDE 16 — Where the Model Fails
**Layout:** Three MRI thumbnails. Failure cases. Honest.

```
3 Complete Failures (Dice ≈ 0)
Patient BraTS2021_01405, _01366, _01407

[MRI slice showing faint/absent T1CE enhancement]

Root cause: Non-enhancing tumours.
SLIC superpixels built on T1CE — if enhancement is absent,
tumour has no visible boundary.
All 15 node features lose discriminative power.

~5% slice-level failure rate overall.
```

*Why:* Showing failures honestly is more credible than hiding them.
The committee respects self-awareness. Frame as "known failure mode with known cause."

---

### SLIDE 17 — Limitations (4 Honest Points)
**Layout:** Numbered list. Short, direct.

```
1. T1CE dependency
   Fails for non-enhancing tumours. Poor contrast = poor superpixels.

2. Binary segmentation only
   We detect tumour vs. healthy. We do NOT distinguish:
   GD-enhancing core / necrotic tissue / oedema.

3. Fixed graph topology
   Graph structure is set at inference time. No dynamic refinement.

4. Adult glioma only
   Trained on adult glioma cases. Paediatric or rare tumour types: unknown.
```

*Why:* Four real limitations means we read the paper carefully.
No hedging language ("might", "could potentially"). Direct statements only.

---

### SLIDE 18 — What Comes Next
**Layout:** Roadmap with three priority tiers.

```
NEAR-TERM (extends this work directly)
  • Multi-class segmentation: WT, TC, ET sub-regions
  • Replace T1CE-only SLIC with multi-modal boundary detection

MEDIUM-TERM (new capability)
  • Dynamic graph refinement based on prediction confidence
  • Paediatric BraTS extension
  • Radiomics / clinical metadata as additional node features

LONG-TERM (deployment)
  • Real-time clinical inference on edge hardware
    (Raspberry Pi, Jetson Nano)
  • 6GB VRAM today → sub-watt embedded hardware tomorrow
```

*Why:* Future work should feel like a roadmap, not an apology.
Tier it so the committee sees near-term credibility and long-term vision.

---

### SLIDE 19 — Summary: Four Findings
**Layout:** Four large numbered points. No tables.

```
1  91.41% Dice on BraTS 2021 sealed held-out set
   (+3.57 pp above our U-Net baseline)

2  5.9× faster inference  ·  227× less memory
   on identical hardware (RTX 2060, 6GB)

3  89.40% Dice on BraTS 2023 — zero-shot, no retraining
   Sensitivity improved +2.92 pp on unseen acquisition protocol

4  Deployable on consumer hardware
   439K parameters  ·  1.7MB storage  ·  6GB VRAM
```

*Why:* Match the 4 objectives stated in the introduction.
Committee members who zoned out will re-engage. Keep it parallel.

---

### SLIDE 20 — Closing Statement
**Layout:** One paragraph. Centred. Not a bullet list.

```
The prevailing assumption is that high-accuracy medical AI
requires high-end infrastructure.

This work challenges that assumption.
A graph neural network with 439,000 parameters,
trained on a 6GB consumer GPU,
achieves near state-of-the-art segmentation
and outperforms models 157× its size.

Efficient AI is not a compromise.
It is a design choice.
```

*Why:* End on conviction, not a table.
The closing message should stick — summarise the philosophy, not just the numbers.

---

### SLIDE 21 — Acknowledgements
**Layout:** Clean, brief.

```
Supervisor:    Mr. Shamim Ahmed, BUBT
Dataset:       BraTS 2021 / BraTS 2023 Challenge Organisers
               (RSNA-ASNR-MICCAI)
Framework:     PyTorch, PyTorch Geometric, scikit-image
Institution:   Bangladesh University of Business and Technology
```

---

### SLIDE 22 — Q&A (Stays On Screen)
**Layout:** Our key numbers on screen while answering questions.

```
QUICK REFERENCE — KEY NUMBERS

Dice: 91.41% (held-out) · 90.02% ± 0.66% (5-fold)
Zero-shot BraTS 2023: 89.40% Dice
Inference: 1,732ms total · 75.4ms GNN-only
Memory: 11MB peak (our) vs 2,500MB (U-Net)
Parameters: 439,041 · Storage: 1.7MB
Compression: 2,284× (voxels to superpixel nodes)
Hardware: RTX 2060 6GB · Python 3.12 · PyTorch 2.8.0
Failure rate: ~5% slice-level · 3 patient-level zeros
```

*Why:* Anticipate the hardest questions. Numbers on screen = no stumbling.

---

## Presentation Delivery Notes

### Pacing
- Slides 1–5: 3 minutes (hook, problem, claim)
- Slides 6–9: 5 minutes (methodology)
- Slides 10–15: 6 minutes (results)
- Slides 16–17: 2 minutes (failures/limits — say these confidently, not defensively)
- Slides 18–20: 2 minutes (future, close)
- Q&A: 7–10 minutes

**Total: ~20 minutes + Q&A**

### Key Talking Points to Memorise
1. **"2,284× compression"** — this is why the GNN is fast. Fewer nodes = faster everything.
2. **"Inductive learning"** — GraphSAGE learns a sampling function, not a fixed matrix. That's why it generalises zero-shot.
3. **"Sensitivity improved on BraTS 2023"** — the surprising result that validates generalisation.
4. **"91.41% vs 87.84%"** — we don't just match U-Net, we beat it.
5. **"6GB VRAM"** — repeat this often. It anchors the accessibility narrative.

### Anticipated Hard Questions

| Question | Answer |
|----------|--------|
| Why not sub-region segmentation? | Binary first as proof-of-concept. Multi-class is direct next step. Architecture handles it — change output head only. |
| Why SLIC only on T1CE? | T1CE has highest tumour contrast. Multi-modal SLIC is future work. |
| How does 84% (ablation Fold 0) reconcile with 91.41%? | Ablation is single fold, no ensemble. Ensemble of 5 folds = 91.41%. |
| Is 75ms inference realistic in clinic? | That's GNN inference only. Total pipeline (preprocessing) = 1,732ms. Still 5.9× faster than U-Net. |
| How were the 3 failure patients identified? | Post-hoc analysis: Dice ≈ 0, manual inspection showed absent T1CE enhancement. Consistent pattern. |
| Why 19:1 class weight = 9.0, not 19? | w+ = sqrt(ratio) heuristic, not the ratio itself. Prevents overconfident positives while still correcting imbalance. |

---

## What I Changed vs. The Existing Slides

| Existing Deck | My Design | Why |
|---------------|-----------|-----|
| 27 slides | 22 slides | Tighter. No slide should exist without a job. |
| Methodology first, efficiency late | Claim on Slide 5, methodology after | Lead with the finding; explain how second. |
| BraTS 2023 buried (Slide 22) | BraTS 2023 gets its own prominent slide (14) | Zero-shot generalisation is a top-3 result. |
| Failure cases absent | Slide 16 dedicated to failures | Credibility comes from honesty. |
| Future work = vague bullets | Tiered roadmap | Shows thought, not padding. |
| Closing = summary table | Closing = conviction statement | Tables don't inspire. A thesis statement does. |
| Q&A = blank slide | Q&A = key numbers reference | Keep numbers visible while you answer. |
