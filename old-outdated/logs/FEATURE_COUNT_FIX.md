# Feature Count Fix: 14 → 15 Features

## Issue Discovered (Nov 30, 2025 - 16:30)

**Problem:** After implementing the ground-truth leakage fix, graph generation produced **14 features** instead of the expected **15 features**.

### Root Cause Analysis

1. **Original leaked graphs (12 features):**
   - 4D: Intensity means (T1, T1ce, T2, FLAIR)
   - 4D: Intensity stds (T1, T1ce, T2, FLAIR)
   - 3D: area (raw), norm_y, norm_x
   - **1D: tumor_ratio (LEAKED from ground-truth)**
   - Total: 4 + 4 + 3 + 1 = **12 features**

2. **First fix attempt (14 features - INCORRECT):**
   - Removed: tumor_ratio (1 feature)
   - Added: perimeter, compactness, intensity_range (3 features)
   - Result: 12 - 1 + 3 = **14 features** ❌
   - Missing: norm_area (normalized area [0, 1])

3. **Final corrected version (15 features - CORRECT):**
   - 4D: Intensity means (T1, T1ce, T2, FLAIR)
   - 4D: Intensity stds (T1, T1ce, T2, FLAIR)
   - **4D: area (raw), norm_area, norm_y, norm_x** (added norm_area)
   - 3D: perimeter, compactness, intensity_range (NEW)
   - Total: 4 + 4 + 4 + 3 = **15 features** ✅

## Code Changes

### File: `src/graph_construction.py`

**Lines 236-240 (ADDED):**
```python
# Compute shape features
area = np.sum(mask)
norm_area = area / (h * w)  # Normalized area [0, 1]  ← NEW!
```

**Lines 269-278 (UPDATED):**
```python
# NEW FEATURE VECTOR (15D) - NO GROUND-TRUTH LEAKAGE
feature = [
    # Intensity means (4D)
    t1_mean, t1ce_mean, t2_mean, flair_mean,
    # Intensity stds (4D)
    t1_std, t1ce_std, t2_std, flair_std,
    # Spatial features (4D) - added norm_area
    area, norm_area, norm_y, norm_x,  ← CHANGED: was 3D, now 4D
    # Shape/texture features (3D) - NEW
    perimeter, compactness, intensity_range
]
```

### File: `src/gnn_model.py`

**Lines 126-131 (UPDATED):**
```python
# ⚠️ SAFETY CHECK: Prevent loading old leaked graphs (Nov 30, 2025)
# Old graphs had 12 features (with tumor_ratio leakage)
# New graphs have 15 features (no leakage, enhanced features)
assert x.shape[1] == 15, \  ← CHANGED: from `!= 12` to `== 15`
    f"❌ CRITICAL: Expected 15 input features (fixed version), but got {x.shape[1]} features! " \
    f"Old leaked graphs had 12 features. Please regenerate graphs with updated graph_construction.py"
```

## Verification Results

### Test 1: Single Patient Graph Generation
```
Patient: BraTS2021_00731
Result: ✅ Generated 59 graphs
Features per node: 15 (expected: 15)
Status: 🎉 SUCCESS: 15 features confirmed!
```

### Feature Breakdown Verification
| Category | Features | Count | Names |
|----------|----------|-------|-------|
| Intensity means | 4 | 4 | t1_mean, t1ce_mean, t2_mean, flair_mean |
| Intensity stds | 4 | 4 | t1_std, t1ce_std, t2_std, flair_std |
| **Spatial** | **4** | **4** | **area, norm_area, norm_y, norm_x** |
| Shape/texture | 3 | 3 | perimeter, compactness, intensity_range |
| **TOTAL** | **15** | **15** | **✅ CORRECT** |

## Next Steps

1. **Mini-Test Execution (In Progress):**
   - Running: `bash scripts/phase2_mini_test.sh`
   - Status: Background process started
   - Log file: `logs/mini_test_run.log`
   - Expected duration: ~30 minutes
   - Expected Dice: 0.1-0.3 initially (LOW = proof of no leakage)

2. **Full Graph Regeneration (Pending mini-test success):**
   - Duration: 8-12 hours (overnight)
   - Patients: 1,251 total
   - Resources: 95% (full power mode recommended)
   - Command: `bash scripts/generate_all_graphs.sh`

3. **Binary Training (After regeneration):**
   - Duration: 2-3 days
   - Method: 5-fold cross-validation
   - Expected Dice: 85-92% (realistic, defensible)
   - vs U-Net: +3-5% improvement (honest comparison)

## Impact on Model Architecture

**No changes needed!** 

- Model config: `hidden_dim=64, num_layers=3`
- Input layer: Automatically adapts to feature dimension
- The GNN uses node embeddings, not raw feature count
- Safety assertion ensures 15 features are present

## Comparison: Old vs New Features

| Feature # | Old (12D - LEAKED) | New (15D - CLEAN) |
|-----------|-------------------|-------------------|
| 1-4 | Intensity means | ✅ Same (intensity means) |
| 5-8 | Intensity stds | ✅ Same (intensity stds) |
| 9 | area (raw) | ✅ Same (area raw) |
| 10 | norm_y | ⚠️ **NEW: norm_area** |
| 11 | norm_x | ✅ norm_y (shifted) |
| 12 | **tumor_ratio (LEAKED)** | ✅ norm_x (shifted) |
| 13 | N/A | ✅ **NEW: perimeter** |
| 14 | N/A | ✅ **NEW: compactness** |
| 15 | N/A | ✅ **NEW: intensity_range** |

**Key Changes:**
- ❌ Removed: tumor_ratio (ground-truth leakage)
- ✅ Added: norm_area (legitimate spatial feature)
- ✅ Added: perimeter (boundary length)
- ✅ Added: compactness (circularity: 4πA/P²)
- ✅ Added: intensity_range (texture contrast: peak-to-peak)

## Expected Outcomes After Fix

### Old Results (INVALID - With Leakage):
- Training Dice: 99.58% (too high, model cheating)
- Validation Dice: 98.80% ± 0.38%
- vs U-Net: +9.46% (unrealistic improvement)
- Model behavior: Learned to threshold feature #12

### New Results (VALID - No Leakage):
- Training Dice: **85-92%** (realistic, defensible)
- Validation Dice: **85-92%** ± reasonable variance
- vs U-Net: **+3-5%** (honest, competitive improvement)
- Model behavior: Learns actual tumor patterns from shape/intensity

## Validation Checklist

- [x] Feature count correct (15 not 12 or 14)
- [x] No ground-truth information in features
- [x] All features computable from input MRI only
- [x] Safety assertion updated (== 15, not != 12)
- [x] Old graphs backed up (graphs_backup_20251130_153644/)
- [ ] Mini-test completes successfully (in progress)
- [ ] Training loss decreases gradually (not instantly)
- [ ] Dice starts LOW and climbs (not 0.99 immediately)
- [ ] Full regeneration successful
- [ ] Final results realistic and thesis-worthy

## File Status

```
✅ Fixed: src/graph_construction.py (15 features)
✅ Fixed: src/gnn_model.py (assertion == 15)
✅ Verified: Single patient test (15 features confirmed)
⏳ Running: Mini-test (5 patients, 5 epochs)
📦 Backup: data/graphs_backup_20251130_153644/ (1.2GB)
📝 Documented: This file + PHASE2_FUTURE_PROOF_IMPLEMENTATION.md
```

## Timeline

- **16:00** - Discovered 14 features (should be 15)
- **16:15** - Identified missing norm_area
- **16:20** - Fixed code, added norm_area computation
- **16:25** - Verified 15 features on test patient
- **16:30** - Started mini-test in background
- **17:00** - Expected mini-test completion (30 min)
- **18:00** - Decision point: proceed to full regeneration?

---
**Status:** ✅ Feature count fix complete, mini-test running.
**Next:** Wait for mini-test results before full regeneration.
