# Claude Skills Reference

A quick-reference guide for all custom skills in this project.
Invoke any skill with `/skill-name` in Claude Code.

---

## 1. `/explain-paper`
**Purpose**: Understand a dense research paper quickly.

**Prompt used**:
> Act as a senior academic mentor. Explain this research paper in simple but precise language. Break it into: research problem, methodology, key findings, limitations, and future research opportunities.

**How to use**:
```
/explain-paper
[paste paper text or abstract]

/explain-paper "Attention Is All You Need"

/explain-paper path/to/paper.pdf
```

---

## 2. `/find-research-gaps`
**Purpose**: Identify unresolved gaps in a body of literature. Great for PhD topics.

**Prompt used**:
> Analyse this literature review and identify unresolved research gaps. Classify them as methodological, theoretical, and empirical gaps.

**How to use**:
```
/find-research-gaps
[paste literature review or notes]

/find-research-gaps "graph neural networks in medical imaging"
```

---

## 3. `/write-lit-review`
**Purpose**: Turn bullet notes into publication-ready literature review prose.

**Prompt used**:
> Convert the following bullet notes into a structured literature review. Compare and contrast studies, highlight methodological differences, and synthesize key findings.

**How to use**:
```
/write-lit-review
[paste bullet notes or paper summaries]
```
> Tip: Run `/explain-paper` on several papers first, then feed the summaries here.

---

## 4. `/compare-studies`
**Purpose**: Side-by-side comparison of multiple studies. Excellent for systematic reviews.

**Prompt used**:
> Compare these studies in terms of research design, data sources, analytical methods, and conclusions. Present the differences in a table and explain what each study contributes.

**How to use**:
```
/compare-studies
[paste summaries of 2 or more papers]
```
> Tip: Pairs well with `/extract-study-data` — extract data from each paper first, then compare.

---

## 5. `/explain-method`
**Purpose**: Deep explanation of any statistical, modelling, or experimental method from scratch.

**Prompt used**:
> Explain this methodology as if teaching a graduate student encountering it for the first time. Include step-by-step logic of the method and practical examples.

**How to use**:
```
/explain-method GraphSAGE

/explain-method 5-fold cross-validation

/explain-method
[paste a methods section from a paper]
```

---

## 6. `/polish-writing`
**Purpose**: Improve clarity, argument strength, and academic tone without changing meaning or citations.

**Prompt used**:
> Rewrite this paragraph to improve clarity, argument strength, and academic tone while preserving the original meaning and citations.

**How to use**:
```
/polish-writing
[paste any paragraph or section]
```
> Works on: thesis chapters, paper drafts, abstracts, discussion sections, grant applications.

---

## 7. `/propose-research-questions`
**Purpose**: Generate 10 strong thesis-level research questions from a literature summary.

**Prompt used**:
> Based on the following literature summary, propose 10 strong research questions suitable for a Master's or PhD thesis.

**How to use**:
```
/propose-research-questions
[paste literature summary or notes]

/propose-research-questions "brain tumour segmentation using GNNs"
```
> Tip: Run `/find-research-gaps` first and feed the gaps output directly into this skill.

---

## 8. `/review-manuscript`
**Purpose**: Pre-submission peer review — covers originality, methodology, clarity, and contribution.

**Prompt used**:
> Act as a journal reviewer. Critically evaluate this manuscript for originality, methodology, clarity, and contribution to the field. Provide constructive feedback.

**How to use**:
```
/review-manuscript
[paste full paper or any sections]
```
> Works even on just abstract + introduction for early-stage feedback.

---

## 9. `/extract-study-data`
**Purpose**: Extract variables, sample sizes, methods, and findings into structured tables. For systematic reviews.

**Prompt used**:
> Extract all key variables, sample sizes, study locations, and major findings from this paper and present them in a structured table.

**How to use**:
```
/extract-study-data
[paste full paper text]
```
> Tip: Run on multiple papers in sequence, then feed all tables into `/compare-studies`.

---

## 10. `/design-followup-study`
**Purpose**: Design a new study from existing findings — hypothesis, methodology, contribution, limitations.

**Prompt used**:
> Using the findings of this paper, design a follow-up study including hypothesis, methodology, expected contribution, and possible limitations.

**How to use**:
```
/design-followup-study
[paste paper findings or summary]
```

---

## Recommended Workflows

### Full Literature Review Pipeline
```
/explain-paper       ← understand each paper
/extract-study-data  ← pull structured data
/compare-studies     ← side-by-side comparison
/find-research-gaps  ← identify what's missing
/write-lit-review    ← synthesise into prose
```

### PhD Proposal Pipeline
```
/find-research-gaps          ← identify the gap
/propose-research-questions  ← generate thesis questions
/design-followup-study       ← turn the best question into a study design
```

### Paper Submission Pipeline
```
/review-manuscript   ← catch issues before submission
/fix-from-review     ← implement the reviewer's fixes into the actual LaTeX files
/polish-writing      ← sharpen each section
```

### Full Thesis Improvement (one command)
```
/write-my-paper      ← runs all 6 phases automatically
```

---

## 12. `/fix-from-review`
**Purpose**: Parse a peer review and implement every actionable fix directly into the thesis LaTeX files. Handles number corrections, wording, framing, table restructuring, and claims. Marks items that require new data as pending.

**Prompt used**:
> Paste a peer review (from /review-manuscript or a real reviewer). Claude will triage each concern as FIXABLE / STRUCTURAL / REQUIRES DATA / JUDGMENT CALL, show you a fix plan, then execute all FIXABLE and STRUCTURAL items in the actual .tex files.

**How to use**:
```
/fix-from-review
[paste the full review text]
```

**What it does**:
1. **TRIAGE** — Classifies every concern in the review by type
2. **PLAN** — Lists every change with file + line before touching anything
3. **EXECUTE** — Makes all FIXABLE and STRUCTURAL edits in the LaTeX files
4. **SUMMARISE** — Final table of what was fixed, what is pending data, what was a judgment call

> Pairs perfectly with `/review-manuscript` → `/fix-from-review` workflow.

---

## 11. `/write-my-paper` ⭐ MASTER SKILL
**Purpose**: Fully audits, diagnoses, fixes, and polishes the entire thesis using all project data and all other skills. Runs a 6-phase pipeline — no arguments needed.

**Prompt used**:
> Using all available project data and skills, completely audit and improve my research thesis — fix wrong numbers, strengthen weak sections, enhance the literature review, and polish the writing throughout.

**How to use**:
```
/write-my-paper
```

**6 phases it runs**:
1. **AUDIT** — Reads all 6 chapter `.tex` files + all `research_results/*.json` files
2. **DIAGNOSE** — Reports every issue (wrong numbers, thin sections, weak arguments) before editing
3. **GROUND** — Fixes all numbers against JSON source of truth
4. **FIX** — Repairs structural issues chapter by chapter (ch1→ch2→...→abstract)
5. **POLISH** — Improves writing quality on flagged sections only (skips good sections)
6. **VERIFY** — Final consistency sweep + full change report

> Pauses after Phase 2 Diagnostic for your confirmation before editing.
> To run without pausing: type `/write-my-paper` then say "run all phases automatically".

---

*Skills live in `.claude/skills/` — project-scoped, only active in this repo.*
