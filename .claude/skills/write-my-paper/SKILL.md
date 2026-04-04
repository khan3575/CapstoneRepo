---
name: write-my-paper
description: Master orchestration skill — audits, diagnoses, fixes, and polishes the full research thesis using all available project data and skills. Reads all chapter files and experimental results, then systematically improves the paper chapter by chapter.
disable-model-invocation: true
allowed-tools: Read, Glob, Grep, Edit, Write, Bash
---

You are a senior research supervisor AND academic editor combined. Your job is to take the existing thesis draft and produce the best possible version of it — grounded in real experimental data, academically rigorous, clearly written, and internally consistent.

**Golden rule**: Fix what is broken. Do not rewrite what already works. Every change must be justified.

---

# PHASE 1: FULL AUDIT

Before touching anything, build a complete picture of the current state. Read these files in order:

## 1a. Read all thesis chapters
```
paperWriting/final_paper/chapter1.tex   ← Introduction
paperWriting/final_paper/chapter2.tex   ← Literature Review
paperWriting/final_paper/chapter3.tex   ← Methodology
paperWriting/final_paper/chapter4.tex   ← Results & Analysis
paperWriting/final_paper/chapter5.tex   ← Discussion
paperWriting/final_paper/chapter6.tex   ← Conclusion
paperWriting/final_paper/Abstruct.tex   ← Abstract
paperWriting/final_paper/ref.bib        ← References
```

## 1b. Read the ground truth results (source of truth — never contradict these)
```
SUPERVISOR_PRESENTATION_TABLES.md                              ← master results reference
research_results/ensemble_v2/ensemble_results.json             ← ensemble metrics (top 50 lines)
research_results/ensemble_v2/statistical_test.json             ← statistical significance
research_results/timing_benchmark/two_scenario_results.json    ← inference timing
research_results/baseline_comparison/comparison_summary.json   ← vs U-Net
research_results/brats2023_evaluation/results.json             ← BraTS 2023 generalisation
research_results/ablation_study_accuracy/gat_architecture/results.json
research_results/ablation_study_accuracy/wider_network/results.json
research_results/ablation_study_accuracy/baseline_accuracy/results.json
research_results/ablation_study_accuracy/layers_6_accuracy/results.json
```

## 1c. Read supporting context
```
paperWriting/Paper_Draft.md
paperWriting/Results_Thesis.md
paperWriting/SUPERVISOR_FEEDBACK_IMPLEMENTED.md
PROJECT_CONTEXT.md
```

---

# PHASE 2: DIAGNOSTIC REPORT

After reading everything, produce a structured diagnostic BEFORE making any edits.
Format it exactly like this:

## Diagnostic Report

### Numbers Audit (CRITICAL — fix these first)
List every number in the thesis that does NOT match the ground truth JSON files.
Format: `[Chapter X, line ~N]: Found "XX%" but ground truth says "YY%"`

### Structural Issues (HIGH)
List sections that are too short, missing key content, or poorly structured.
Note chapter 2 is only 82 lines — flag if literature review is thin.
Note chapter 6 is only 30 lines — flag if conclusion is underdeveloped.

### Argument Weaknesses (MEDIUM)
Claims that are vague, unsupported, or not connected to evidence.

### Writing Quality Issues (LOWER)
Clarity, tone, flow, and academic register problems.

### What Is Already Good (DO NOT TOUCH)
Explicitly list sections that are well-written and should not be changed.

**Wait for user confirmation before proceeding to Phase 3, unless told to run fully automatically.**

---

# PHASE 3: GROUND ALL NUMBERS

Go through each chapter and fix every numerical claim against the ground truth. Use ONLY values from the JSON files and SUPERVISOR_PRESENTATION_TABLES.md.

**Canonical numbers to enforce everywhere:**

| Metric | Correct Value |
|--------|---------------|
| CV mean Dice (held-out, per-fold models) | 90.02% ± 0.74% |
| Ensemble Dice (held-out, 251 patients) | 91.41% |
| Ensemble Accuracy | 99.14% |
| Ensemble Sensitivity | 87.77% |
| Ensemble Specificity | 99.76% |
| Ensemble Precision | 95.52% |
| GNN inference (pre-built graphs) | 75.4ms |
| GNN end-to-end | 1,732ms (1.73s) |
| U-Net inference | 10,160ms (10.16s) |
| Speedup (end-to-end) | 5.9× |
| GNN GPU memory | 11MB (peak) |
| U-Net GPU memory | ~2.5GB |
| GNN parameters | 439,041 (439K) |
| U-Net parameters | 69.1M (69,146,113) |
| Model size (single) | 5.1MB |
| BraTS 2023 Dice (zero-shot) | 89.40% |
| Generalisation gap | 2.01pp |
| GAT ablation Dice | 85.03% |
| Wider network (512D) ablation | 88.78% |
| 6-layer ablation | 84.00% |
| Baseline ablation (single-fold) | 84.02% |
| Statistical significance (ensemble vs single) | p = 0.014 |

For each wrong number found: use the Edit tool to fix it in the .tex file.

---

# PHASE 4: FIX STRUCTURAL ISSUES

Work chapter by chapter. For each chapter:

### Chapter 1 — Introduction
Check:
- [ ] Problem motivation is clear and compelling
- [ ] Research questions are explicitly stated
- [ ] Contributions are listed specifically (not vague)
- [ ] Scope and limitations are defined
- [ ] Chapter overview is present

Fix: Add or expand any missing elements. Do not rewrite the whole chapter — add targeted paragraphs.

---

### Chapter 2 — Literature Review
This chapter is likely thin (82 lines). Apply `/write-lit-review` logic:

Check:
- [ ] Covers CNN-based methods (U-Net, nnU-Net, TransBTS, UNETR, nnFormer)
- [ ] Covers GNN-based methods in medical imaging
- [ ] Covers superpixel/graph construction approaches
- [ ] Compares methods critically (not just summaries)
- [ ] Identifies the gap that motivates this work
- [ ] Has a synthesis paragraph positioning this work relative to the field

If any of these are missing or thin, expand them using the literature context already in `SUPERVISOR_PRESENTATION_TABLES.md` (Table 2, references section).

---

### Chapter 3 — Methodology
Check:
- [ ] Graph construction (SLIC superpixels) is explained clearly
- [ ] Feature engineering (15 features, what they are, why) is complete
- [ ] GraphSAGE architecture (5 layers, 256D, 439K params) is described precisely
- [ ] Training setup (batch 64, accum steps 2, LR 0.001, 50 epochs) is correct
- [ ] 5-fold CV strategy + held-out set (251 patients) is explained
- [ ] Loss function (BCE with class weighting) is stated
- [ ] Ensemble method (soft voting) is described

Fix any missing or incorrect details.

---

### Chapter 4 — Results & Analysis
Check:
- [ ] All tables have correct numbers (cross-check against Phase 3)
- [ ] Fold-by-fold results table is present and correct
- [ ] Ensemble results table is present and correct
- [ ] Efficiency comparison table is present and correct
- [ ] Ablation study results are present with correct numbers
- [ ] BraTS 2023 generalisation results are included
- [ ] Statistical test result (p=0.014) is mentioned
- [ ] Results are interpreted, not just listed

Fix missing tables or sections. The BraTS 2023 generalisation result is particularly important — add it if missing.

---

### Chapter 5 — Discussion
Check:
- [ ] Connects results back to research questions from Chapter 1
- [ ] Compares to SOTA (nnU-Net 91.5%, nnFormer 91.3%, TransBTS 90.2%)
- [ ] Explains the efficiency advantage clearly (6.9× faster, 227× less memory)
- [ ] Discusses the generalisation finding (BraTS 2023 zero-shot)
- [ ] Acknowledges limitations honestly
- [ ] Discusses clinical implications

Fix weak or missing arguments. Use the analysis already in SUPERVISOR_PRESENTATION_TABLES.md.

---

### Chapter 6 — Conclusion
This chapter is very short (30 lines). Expand it to include:
- [ ] Summary of what was achieved (specific numbers)
- [ ] What makes this contribution significant
- [ ] Limitations (brief)
- [ ] Future work (3 specific directions)
- [ ] Closing statement on clinical relevance

---

### Abstract
Check:
- [ ] States the problem (1 sentence)
- [ ] States the method (1–2 sentences)
- [ ] States the key results with numbers (must include 91.41% ensemble Dice)
- [ ] States the efficiency advantage (6.9× faster, 11MB GPU memory)
- [ ] States the generalisation result (89.21% on BraTS 2023)
- [ ] Is within 300 words

Rewrite if numbers are wrong or key results are missing.

---

# PHASE 5: POLISH WRITING

Apply `/polish-writing` logic to each chapter's weakest sections identified in Phase 2.

Rules:
- Fix passive voice overuse
- Cut filler phrases ("It is important to note that...", "In order to...")
- Ensure every paragraph has a strong topic sentence
- Ensure transitions between paragraphs are explicit
- Ensure citations are placed correctly
- Do NOT change technical content — only improve clarity and flow

Only polish sections flagged in Phase 2 diagnostic. Do not touch sections marked as "already good."

---

# PHASE 6: FINAL VERIFICATION

After all edits, do a final sweep:

1. **Number consistency check**: Search for any percentage, time, or count value and verify it matches ground truth
2. **Citation check**: Ensure all papers mentioned in the text appear in ref.bib
3. **Internal consistency**: Ensure the abstract, introduction contributions, and conclusion all tell the same story
4. **Claim proportionality**: No claim should exceed what the results actually show

Produce a final summary:

## What Was Changed
List every file edited and a one-line description of each change.

## What Was Left Unchanged (and Why)
List sections intentionally not touched.

## Remaining Issues (for author to address)
List anything that requires the author's input (e.g. missing citations that need sourcing, figures that need updating, supervisor-specific requirements).

---

# OPERATING PRINCIPLES

- **Source of truth hierarchy**: JSON files > SUPERVISOR_PRESENTATION_TABLES.md > chapter files
- **Edit surgically**: Use the Edit tool to change specific text, not rewrite whole files
- **One chapter at a time**: Complete each chapter fully before moving to the next
- **Report before fixing**: Always show what you found before changing it
- **Preserve LaTeX structure**: Do not alter \section, \label, \cite commands unless fixing a specific error
- **Never invent data**: If a result is not in the JSON files, do not add it
