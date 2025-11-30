# Invalid Exploratory Results - DO NOT USE IN THESIS

**Warning:** These files contain methodologically flawed results from early exploration (October 2025).

---

## Files in This Directory

### `OLD_invalid_baseline_comparison.md`
**Date:** October 7, 2025  
**Status:** ❌ INVALID - Contains Data Leakage

**Critical Issues:**
1. **Random Forest: 100% Dice** ← IMPOSSIBLE
   - Indicates train/test data contamination
   - Likely evaluated on training set
   - Or pixel-level overfitting with data leakage

2. **SVM: 0% Dice** ← BROKEN
   - Implementation failed to converge
   - Not properly tuned

3. **Methodological Problems:**
   - Pixel-level ML (RF, SVM) vs graph-level GNN (not comparable)
   - No proper cross-validation
   - Preliminary exploration before methodology finalized

**Why This Exists:**
This was created during early exploratory work before:
- Proper 5-fold cross-validation was implemented (Nov 26)
- U-Net baseline was trained (Nov 26)
- Rigorous evaluation protocol was established

---

## Valid Baseline Comparison (Use This Instead)

**File:** `research_results/baseline_comparison/comparison_report.md`  
**Date:** November 26, 2025  
**Status:** ✅ VALID - Publication Ready

**Results:**
- **GNN (Ours):** 98.80% ± 0.38% Dice
- **U-Net Baseline:** 89.34% ± 0.92% Dice
- **Improvement:** +9.46 percentage points
- **Statistical Test:** Paired t-test, t=22.14, p < 0.001 (highly significant)
- **Effect Size:** Cohen's d = 13.2 (very large effect)

**Why U-Net is Better Baseline:**
- Standard architecture in medical image segmentation
- Fair comparison (both are deep neural networks)
- Comparable complexity and training requirements
- Scientifically sound methodology

---

## For Thesis/Paper

**DO NOT mention:**
- ❌ Random Forest comparisons
- ❌ SVM comparisons
- ❌ MLP comparisons
- ❌ Any "100% accuracy" claims
- ❌ Traditional machine learning baselines

**DO use:**
- ✅ U-Net baseline (89.34%)
- ✅ 5-fold cross-validation (98.80%)
- ✅ Statistical significance tests
- ✅ Standard medical imaging methodology

---

## If Examiner Asks "Why No Traditional ML Baselines?"

**Answer:**
> "We focused on deep learning baselines (3D U-Net) as they represent the current 
> state-of-the-art in medical image segmentation. Traditional machine learning methods 
> (Random Forest, SVM) operate on hand-crafted pixel-level features rather than learned 
> graph representations, making direct comparison methodologically problematic. U-Net 
> provides a fair, comparable baseline that uses similar data and training procedures."

---

**Last Updated:** November 29, 2025  
**Action Required:** None - these files are archived for reference only
