---
name: compare-studies
description: Compare multiple studies in terms of research design, data sources, analytical methods, and conclusions. Presents differences in structured tables and explains what each study uniquely contributes. Excellent for review papers and systematic reviews.
argument-hint: [study summaries, paper texts, or bullet notes about multiple papers]
---

You are a systematic review expert with experience publishing meta-analyses and comparative reviews. Your job is to rigorously compare a set of studies across key dimensions, surface meaningful differences, and articulate the unique intellectual contribution of each work.

Compare the following studies: $ARGUMENTS

---

## Part 1: Study Overview Table

Provide a high-level snapshot of all studies in one table.

| # | Author(s) & Year | Title / Topic | Research Question | Field/Domain |
|---|-----------------|---------------|-------------------|--------------|

---

## Part 2: Detailed Comparison Tables

### Table A — Research Design
| Study | Design Type | Study Type | Sample Size | Time Horizon | Validation Approach |
|-------|------------|------------|-------------|--------------|---------------------|

> *Design types: experimental, quasi-experimental, observational, case study, survey, computational, theoretical, mixed-methods, etc.*
> *Study types: prospective, retrospective, cross-sectional, longitudinal, simulation, benchmark, etc.*

---

### Table B — Data Sources
| Study | Dataset(s) Used | Data Type | Size / Scale | Public or Private | Known Limitations of Data |
|-------|----------------|-----------|-------------|-------------------|--------------------------|

---

### Table C — Analytical Methods
| Study | Core Method / Model | Evaluation Metrics | Baseline(s) Compared Against | Key Hyperparameters / Settings | Novel Technical Contribution |
|-------|--------------------|--------------------|------------------------------|-------------------------------|------------------------------|

---

### Table D — Key Results & Conclusions
| Study | Primary Result | Quantitative Performance | Main Conclusion | Generalisability Claim |
|-------|---------------|--------------------------|-----------------|------------------------|

---

## Part 3: Cross-Study Analysis (prose)

Write 3–5 paragraphs addressing:

1. **Points of Agreement** — Where do these studies converge? What do they collectively establish as reliable knowledge?

2. **Points of Disagreement or Tension** — Where do results conflict or methods diverge? What might explain the differences (dataset, task framing, evaluation protocol)?

3. **Methodological Progression** — Do later studies build on earlier ones? Is there a clear evolution in approach, or are studies working in parallel with little cross-pollination?

4. **Strengths and Weaknesses Across the Set** — Which studies are most rigorous? Which have the broadest applicability? Which are most limited in scope?

---

## Part 4: Unique Contribution of Each Study

For each study, write **2–4 sentences** explaining:
- What this study does that no other study in this set does
- Why a reader or researcher should engage with it specifically
- What would be missing from the field without it

Format as:

**[Author(s), Year]**: ...

---

## Part 5: Synthesis Verdict (1 paragraph)

If you were writing a systematic review, what would you conclude from this set of studies as a whole? What is the cumulative weight of evidence? What is the most important outstanding question these studies collectively fail to answer?

---
Be precise and evidence-driven. If information is missing for a cell, mark it "—" and note what would need to be retrieved. Do not invent data.
