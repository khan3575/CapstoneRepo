# Code Changes Needed — Post Paper Review

These are discrepancies found between the paper (now corrected) and the actual code.
They do not affect results — the paper now accurately describes what the code does.
These are optional improvements to make the code better match best practices.

---

## 1. Class Imbalance — No Handling in Loss Function

**File:** `src/train_cv_fold.py` line 274

**Current:**
```python
criterion = nn.BCEWithLogitsLoss()
```

**Issue:** Tumour nodes are ~10% of all nodes. No weighting is applied.
The paper now honestly says Dice metric handles this implicitly, which is true but
adding `pos_weight` would directly help the model during training.

**Suggested fix:**
```python
# Approximate pos_weight: non-tumour / tumour ratio ≈ 9.0
pos_weight = torch.tensor([9.0]).to(device)
criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
```

**Priority:** Medium — may improve single-fold Dice by 1–2%

---

## 2. Early Stopping — Defined in Config but Never Implemented

**File:** `src/train_cv_fold.py` (training loop ~line 290–340)
**Config:** `config.yaml` → `early_stopping_patience: 10`

**Current:** Code runs for all 50 epochs regardless of validation performance.
The config field `early_stopping_patience` is read into config but never used.

**Suggested fix:** Add early stopping counter after the best-model save block:
```python
patience = config.get('model.training.early_stopping_patience', 10)
epochs_no_improve = 0

# Inside training loop, after saving best model:
if val_metrics['dice'] > best_val_dice:
    best_val_dice = val_metrics['dice']
    epochs_no_improve = 0
    # ... save checkpoint ...
else:
    epochs_no_improve += 1
    if epochs_no_improve >= patience:
        print(f"Early stopping at epoch {epoch+1}")
        break
```

**Priority:** Low — OneCycleLR already provides implicit regularisation.
Either implement it or remove `early_stopping_patience` from config.yaml to avoid confusion.

---

## 3. Unused Config Field — Clean Up config.yaml

**File:** `config.yaml`

**Current:**
```yaml
model:
  training:
    max_epochs: 50
    early_stopping_patience: 10   # ← defined but never read by code
```

**Suggested fix:** Either implement early stopping (see #2 above) or remove this field.

**Priority:** Low — cosmetic, but misleading to anyone reading the config.

---

## 4. End-to-End Benchmark Missing — Inference Time is Incomplete

**File:** `scripts/benchmark_speed.py`

**Current:** Only measures GNN neural network forward pass time (~12.7ms).
Does NOT measure graph construction time (~2–3s per patient).

**Issue:** This makes the 6.9× speedup claim apply only to the inference phase,
not the full pipeline. The paper now clarifies this, but the benchmark script
should also record total pipeline time for completeness.

**Suggested fix:** Add a `benchmark_full_pipeline()` function that measures:
1. Graph construction time (SLIC + feature extraction)
2. GNN inference time
3. Total = 1 + 2

Then report both: `inference_only_ms` and `total_pipeline_ms`.

**Priority:** Medium — important for honest comparison if this work is extended.

---

## 5. SoTA Numbers in Paper — Must Be Manually Verified

**File:** `paperWriting/Template_TextOnly/chapter4.tex` (Table 4 / tab:sota)

**Current numbers in paper:**
| Method | Dice (%) |
|--------|----------|
| 3D U-Net | 91.2 |
| nnU-Net (2021) | 92.5 |
| Swin-UNETR (2022) | 93.8 |
| TransBTS (2022) | 93.2 |

**Action required:** Before final thesis submission, manually verify each number
against the original paper's Table (Whole Tumour Dice on BraTS 2021).

- Swin-UNETR: Hatamizadeh et al., MICCAI Brainlesion Workshop 2022
- TransBTS: Wang et al., MICCAI 2021 (originally BraTS 2019 — check if BraTS 2021 numbers exist)
- nnU-Net: Isensee et al., Nature Methods 2021

If the exact BraTS 2021 WT Dice numbers are not available for a method, either
cite a different metric clearly, or remove that row from the table.

**Priority:** HIGH — wrong numbers here are a critical academic integrity issue.

---

## 6. Long-Term: Multi-Class Extension (for Publication)

**Files:** `src/train_cv_fold.py`, `src/dataset.py`, `src/graph_construction.py`

**Current:** Binary labels only (tumour / non-tumour).

**For publication at any BraTS venue:** Need to predict ET, TC, WT sub-regions.

**Approach:**
- Change output layer from 1 neuron → 3 neurons (one per sub-region, sigmoid each)
- Change labels from binary mask → 3-channel mask (ET, TC=ET+necrotic, WT=TC+edema)
- Change loss from BCEWithLogitsLoss → multi-label BCE or Dice loss per channel
- Report WT, TC, ET Dice separately (standard BraTS evaluation)

**Priority:** HIGH if targeting journal/conference publication. Not needed for thesis.

---

## Summary Table

| # | File | Change Type | Priority |
|---|------|------------|----------|
| 1 | `src/train_cv_fold.py` | Add `pos_weight` to BCEWithLogitsLoss | Medium |
| 2 | `src/train_cv_fold.py` | Implement early stopping using config value | Low |
| 3 | `config.yaml` | Remove unused `early_stopping_patience` OR implement #2 | Low |
| 4 | `scripts/benchmark_speed.py` | Add full pipeline timing (preprocessing + inference) | Medium |
| 5 | `paperWriting/chapter4.tex` | Manually verify SoTA Dice numbers from original papers | **HIGH** |
| 6 | `src/` (multiple files) | Multi-class ET/TC/WT extension | HIGH (for publication) |
