# Response to Reviewers

**Manuscript:** Graph Neural Networks for Efficient Brain Tumour Segmentation:
A Resource-Constrained Alternative to Volumetric Deep Learning

**Revision Type:** Major Revision

---

We thank the reviewers for their thorough and constructive feedback. All reviewer
comments have been addressed. Substantive changes are highlighted below. We are
confident that the revised manuscript is factually accurate and that all SOTA
comparisons have been independently verified against the original published papers.

---

## Reviewer 1 — Factual Accuracy of SOTA Comparisons

### R1.1 — TransBTS parameter count

**Comment:** *"The manuscript states TransBTS has 71M parameters. This appears
inconsistent with the published paper."*

**Response:** The reviewer is correct. We had propagated an inaccurate parameter
figure from a secondary source. We re-verified against the TransBTS paper
(Wang et al., MICCAI 2021) and its publicly released code. The correct parameter
count for the standard TransBTS configuration is **approximately 33M**, not 71M.
The manuscript has been updated in Table 6a (Chapter 4), the Literature Review
(Chapter 2), and all derived comparison ratios throughout.

**Change:** All instances of "71M" for TransBTS replaced with "~33M".

---

### R1.2 — TransBTS evaluation dataset

**Comment:** *"The paper claims TransBTS achieves 93.2% Dice on BraTS 2021. However,
the TransBTS paper was published at MICCAI 2021 and the BraTS 2021 dataset was
released later that year. Please verify the evaluation dataset."*

**Response:** The reviewer has identified a critical factual error. The original
TransBTS paper (Wang et al., MICCAI 2021) evaluated exclusively on **BraTS 2020**,
not BraTS 2021. The reported whole-tumour Dice is approximately **90.1% on BraTS 2020**.
There is no TransBTS result on BraTS 2021 in the original publication. The manuscript
has been corrected: the table now shows "~90.1%‡" with a footnote reading
"‡ Evaluated on BraTS 2020; BraTS 2021 result not available in original publication."
The performance gap analysis has been updated accordingly.

**Change:** Table 6a TransBTS row: 93.2% (BraTS 2021) → ~90.1%‡ (BraTS 2020).
Footnote added. Performance gap recalculated.

---

### R1.3 — Swin-UNETR parameter count

**Comment:** *"The 92M parameter count listed for Swin-UNETR seems very high for
a Swin Transformer backbone. Please clarify the source."*

**Response:** The reviewer is correct to question this. We traced the 92M figure
to the **UNETR** paper (Hatamizadeh et al., WACV 2022), which is the predecessor
architecture using a plain ViT encoder, not the Swin Transformer. Swin-UNETR
(Hatamizadeh et al., CVPR 2022) uses a hierarchical Swin Transformer encoder and
has **approximately 62M parameters** in the standard configuration used for BraTS
evaluation. All occurrences of "92M" referring to Swin-UNETR have been corrected.
Chapter 2 also now clearly distinguishes UNETR (ViT backbone, 92M) from
Swin-UNETR (Swin Transformer backbone, 62M).

**Change:** Swin-UNETR parameters: 92M → 62M throughout manuscript.

---

### R1.4 — Swin-UNETR Dice score

**Comment:** *"The reported 93.8% whole-tumour Dice for Swin-UNETR — please provide
the exact table and page number from the source paper."*

**Response:** After re-reading the Swin-UNETR paper, we could not find a table
reporting exactly 93.8% WT Dice on BraTS 2021 under identical evaluation conditions.
The consistently reportable figure from the original paper is **93.3% WT Dice** on
BraTS 2021. We have conservatively corrected the table to 93.3% and updated all
performance gap comparisons (previously stated as "1.6–2.6 pp", now corrected to
"1.1–1.9 pp").

**Change:** Swin-UNETR BraTS 2021 Dice: 93.8% → 93.3%.

---

## Reviewer 2 — Methodology and Fairness of Comparison

### R2.1 — nnU-Net parameter count

**Comment:** *"The claim that nnU-Net has 80M+ parameters requires a citation.
nnU-Net is a framework, not a fixed architecture, and parameter counts vary by
configuration."*

**Response:** The reviewer makes an important distinction. nnU-Net (Isensee et al.,
Nature Methods 2021) is indeed a self-configuring framework. The 80M figure was
an overestimate. For the standard 3D full-resolution configuration on BraTS 2021,
which uses a standard 3D U-Net encoder–decoder, the parameter count is
**approximately 31M**. This is consistent with published analyses of nnU-Net on
BraTS tasks. The manuscript now states "approximately 31M parameters for the
standard BraTS 3D full-resolution configuration" with a qualifier that the count
is task-dependent.

**Change:** nnU-Net parameters: "80M+" → "~31M" (BraTS 3D full-resolution config).

---

### R2.2 — 3D U-Net citation

**Comment:** *"The baseline '3D U-Net' is cited as Ronneberger et al. (2015).
Ronneberger et al. is the original 2D U-Net. For volumetric/3D extension please
cite Çiçek et al. (2016)."*

**Response:** The reviewer is entirely correct. The 2D U-Net by Ronneberger et al.
(MICCAI 2015) should not be cited as the baseline for our 3D volumetric model.
The correct citation is **Çiçek et al., "3D U-Net: Learning Dense Volumetric
Segmentation from Sparse Annotation," MICCAI 2016**. The BibTeX key has been
corrected from `ronneberger2015unet` to `cicek20163d`, and the in-text citation
updated accordingly.

**Change:** 3D U-Net citation: Ronneberger et al. (2015) → Çiçek et al. (2016).

---

### R2.3 — Table 6a conflates single-model and ensemble results

**Comment:** *"Table 6a shows a single 'Our Model' row with 2.2M parameters but
the 91.41% Dice result. The 2.2M figure is the ensemble (5 × 439K). The single-model
result should be reported separately."*

**Response:** This is a valid clarity concern. The table now has **two rows** for
our approach:

| Method | Dice | Params | vs. SOTA |
|--------|------|--------|----------|
| GNN Single (GraphSAGE, 1 model)§ | 90.02% ± 0.74% | 439K | ~70–141× fewer |
| GNN Ensemble (5 models, soft-voting)¶ | **91.41%** | 2.2M | 14–28× fewer |

Footnotes clarify: §CV mean over 5 folds evaluated on per-fold test sets;
¶Evaluated on sealed 251-patient held-out set (no exposure during training).

**Change:** Table 6a "Our" row split into single-model and ensemble rows.
All parameter comparison ratios updated accordingly.

---

## Reviewer 3 — Presentation and Clarity

### R3.1 — Incorrect comparison narrative in baseline justification

**Comment:** *"The text claims 'nnU-Net (80M) and Swin-UNETR (92M) would show
even greater parameter disparity'. After correcting parameters, these models
(31M and 62M) are at similar scale to your 68M U-Net baseline. The narrative
no longer holds."*

**Response:** Agreed. After correcting all SOTA parameter counts, the claim is
factually incorrect — the SOTA models range from 31M to 62M, comparable to our
68M U-Net baseline. The paragraph has been rewritten. The justification for
choosing the 3D U-Net as the baseline now correctly states that it was chosen
because it was **directly benchmarked on identical hardware** (RTX 2060, 6 GB VRAM)
under the same binary segmentation protocol, making it the only valid direct
efficiency comparator. We do not claim our baseline represents the largest SOTA
model.

**Change:** Baseline justification paragraph rewritten; incorrect "greater
disparity" claim removed.

---

### R3.2 — Performance gap range inconsistency

**Comment:** *"The stated performance gap of '1.6–2.6 percentage points' vs. SOTA
does not match the individual numbers in Table 6a after corrections."*

**Response:** After correcting all SOTA Dice values (nnU-Net 92.5%, Swin-UNETR
93.3%, TransBTS ~90.1% on BraTS 2020), the correct performance gap for our
ensemble (91.41%) vs. SOTA methods evaluated on BraTS 2021 is:
- vs. nnU-Net (92.5%): **1.09 pp**
- vs. Swin-UNETR (93.3%): **1.89 pp**

Range corrected to **1.1–1.9 percentage points** throughout the manuscript.

**Change:** Performance gap: "1.6–2.6 pp" → "1.1–1.9 pp".

---

### R3.3 — Architecture label for Swin-UNETR in Table 6a

**Comment:** *"Table 6a labels Swin-UNETR as 'Pure vision transformer'. Swin-UNETR
uses a hierarchical Swin Transformer, not a plain ViT. Please correct the
category label."*

**Response:** Agreed. The label has been updated to **"Swin Transformer encoder
(hybrid CNN decoder)"** which accurately reflects the Swin-UNETR architecture:
a Swin Transformer encoder with skip connections to a CNN decoder. This also
distinguishes it from UNETR, which uses a plain ViT encoder.

**Change:** Table 6a Swin-UNETR label: "Pure vision transformer" →
"Swin Transformer encoder".

---

## Summary of All Changes

| # | Location | Before | After |
|---|----------|--------|-------|
| 1 | Table 6a, Ch4 | TransBTS: 71M, 93.2% (BraTS 2021) | ~33M, ~90.1%‡ (BraTS 2020) |
| 2 | Table 6a footnote | No BraTS 2020 note | Added ‡ footnote |
| 3 | Table 6a, Ch4 | Swin-UNETR: 93.8%, 92M | 93.3%, 62M |
| 4 | Table 6a, Ch4 | nnU-Net: 80M+ | ~31M |
| 5 | Ch4 citation | ronneberger2015unet | cicek20163d |
| 6 | Table 6a | Single "Our" row (2.2M) | Two rows: single (439K) + ensemble (2.2M) |
| 7 | Ch4 baseline text | "80M and 92M show greater disparity" | Rewritten (removed incorrect claim) |
| 8 | Ch4 performance gap | 1.6–2.6 pp | 1.1–1.9 pp |
| 9 | Ch2, Ch4 | Parameter range "68M–92M" | "31M–68M" |
| 10 | Table 6a label | "Pure vision transformer" | "Swin Transformer encoder" |

All corrections are consistent with the primary sources (papers, GitHub repositories,
and official model cards). We thank the reviewers for the rigour of their scrutiny —
the manuscript is measurably more accurate as a result.

---

*Submitted by the authors.*
