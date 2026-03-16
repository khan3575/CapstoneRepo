# 📁 PREPROCESSING FILES - COMPLETE OVERVIEW

**Date:** January 29, 2026  
**Project:** BraTS GNN Segmentation

---

## 🎯 MAIN PREPROCESSING FILE

### **PRIMARY: `src/preprocessing.py`** (759 lines)
**Location:** `/mnt/bigdata/capstone/brats_gnn_segmentation/src/preprocessing.py`

**Responsibility:** All MRI volume preprocessing operations

**Key Functions:**

| Function | Purpose | Input | Output |
|----------|---------|-------|--------|
| `organize_files()` | Rename/organize raw BraTS files to standard structure | Raw BraTS folders | Organized `T1.nii.gz`, `T1ce.nii.gz`, etc. |
| `resample_volume()` | Resample MRI to 1mm³ isotropic resolution | SimpleITK image | Resampled image |
| `crop_pad_volume()` | Crop or pad to target 240×240×155 shape | Image + target_shape | Standardized volume |
| `load_slice_data()` | Load individual 2D slice from NPZ | NPZ file path | Slice dictionary (4 modalities + mask + label) |
| `load_patient_slices()` | Load all slices for one patient | Patient directory | List of slice dictionaries |
| `get_slice_statistics()` | Get statistics about extracted slices | Slice directory | Metadata JSON |

**What preprocessing.py Does:**

```
Step 1: Organize Files
  └─ Takes raw BraTS structure
  └─ Standardizes filenames (T1.nii.gz, T1ce.nii.gz, T2.nii.gz, FLAIR.nii.gz)

Step 2: Load Raw Volumes
  └─ Uses nibabel to read 4 MRI modalities
  └─ Loads segmentation mask

Step 3: Resample
  └─ Uses SimpleITK to resample to 1mm³ isotropic
  └─ Calculates new dimensions based on spacing

Step 4: Crop/Pad
  └─ Standardizes all volumes to 240 × 240 × 155
  └─ Crops oversized volumes, pads undersized ones

Step 5: Intensity Normalization (Z-Score)
  └─ Calculates mean/std from brain tissue only (mask > 0)
  └─ Formula: (x - mean) / (std + 1e-8)
  └─ Resets background to 0

Step 6: Extract Slices
  └─ Selects tumor-priority slices (~200 per patient)
  └─ Saves as NPZ (compressed NumPy format)
  └─ Each slice: 4 modalities + brain mask + label

Step 7: Save Output
  └─ Compressed .nii.gz files (if saving full volumes)
  └─ or NPZ slices (if saving for graph construction)
```

**Command to Run:**
```bash
python3 src/preprocessing.py \
    --input_dir /path/to/raw/BraTS2021 \
    --output_dir data/preprocessed \
    --num_workers 8
```

**Output:**
```
data/preprocessed/
├── BraTS2021_00000/
│   ├── BraTS2021_00000_t1.nii.gz       (preprocessed T1)
│   ├── BraTS2021_00000_t1ce.nii.gz     (preprocessed T1ce)
│   ├── BraTS2021_00000_t2.nii.gz       (preprocessed T2)
│   ├── BraTS2021_00000_flair.nii.gz    (preprocessed FLAIR)
│   └── BraTS2021_00000_seg.nii.gz      (segmentation mask)
└── ... (1,250 more patients)
```

---

## 🔗 RELATED FILES

### **SECONDARY: `src/dataset.py`** (340 lines)
**Location:** `/mnt/bigdata/capstone/brats_gnn_segmentation/src/dataset.py`

**Responsibility:** Load preprocessed data for training

**Key Classes:**

```python
class BinaryTransform:
    """Convert multi-class labels to binary (any tumor = 1)"""
    - Input: y in {0, 1, 2, 4}
    - Output: y in {0, 1}
    - Used during data loading, not preprocessing

class BraTSGraphDataset(Dataset):
    """Load graph data for training/validation/testing"""
    - Loads preprocessed graphs (.pt files)
    - Can load from explicit file list (for CV)
    - Returns Data objects (PyG format)
    - Supports transforms (e.g., BinaryTransform)
```

**Connection to Preprocessing:**
```
preprocessing.py creates preprocessed volumes
    ↓
graph_construction.py converts volumes to graphs
    ↓
dataset.py loads graphs for training
```

---

### **SUPPORTING: `src/graph_construction.py`** (774 lines)
**Location:** `/mnt/bigdata/capstone/brats_gnn_segmentation/src/graph_construction.py`

**Responsibility:** Convert preprocessed MRI to graph representations

**But NOTE:** This is **NOT preprocessing**—it's the **next step** after preprocessing

**Process:**
```
Preprocessed MRI Volume (from preprocessing.py)
    ↓
1. Select tumor-priority slices (~200)
2. Extract superpixels from each slice (SLIC algorithm)
3. Compute 15D features per superpixel node
4. Build edges (intra-slice + inter-slice)
5. Save as graph (.pt file)
    ↓
Graph Data (nodes, edges, features, labels)
```

**NOT part of preprocessing:** Graph construction is separate downstream step

---

## 🔄 DATA FLOW

```
┌─────────────────────────────────────────────────────────────┐
│ RAW DATA (BraTS 2021 Dataset)                              │
│ • 1,251 patients                                            │
│ • 4 MRI modalities per patient (T1, T1ce, T2, FLAIR)      │
│ • 1 segmentation mask per patient                          │
│ • High resolution, variable scaling                        │
└──────────────────────────┬──────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│ 📄 PREPROCESSING (src/preprocessing.py)                    │
│ ✓ Organize files → standardized names                      │
│ ✓ Resample → 1mm³ isotropic                                │
│ ✓ Crop/Pad → 240×240×155                                   │
│ ✓ Z-score normalize → standardized intensities            │
│ ✓ Extract tumor-priority slices → 200 slices per patient  │
│ ✓ Save → NPZ or .nii.gz format                             │
└──────────────────────────┬──────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│ PREPROCESSED DATA (data/preprocessed/)                      │
│ • 1,251 clean MRI volumes                                  │
│ • Normalized intensities                                   │
│ • Standardized dimensions                                  │
│ • Ready for feature extraction                             │
└──────────────────────────┬──────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│ 📊 GRAPH CONSTRUCTION (src/graph_construction.py)          │
│ ✓ Extract superpixels (SLIC)                               │
│ ✓ Compute node features (15D)                              │
│ ✓ Build edges (spatial + KNN)                              │
│ ✓ Create graph objects (PyG)                               │
│ ✓ Save → .pt files                                         │
└──────────────────────────┬──────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│ GRAPH DATA (data/graphs/)                                   │
│ • 1,251 graph representations                              │
│ • Nodes = superpixels                                      │
│ • Features = 15D per node                                  │
│ • Ready for GNN training                                   │
└──────────────────────────┬──────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│ 📥 DATA LOADING (src/dataset.py)                           │
│ ✓ Load graph files                                         │
│ ✓ Apply transforms (e.g., binary labels)                  │
│ ✓ Create PyTorch DataLoader batches                        │
│ ✓ Return to training loop                                  │
└──────────────────────────┬──────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│ 🧠 MODEL TRAINING (src/train_cv_fold.py)                  │
│ • GraphSAGE network processes nodes                        │
│ • Message passing on edges                                 │
│ • Binary classification (tumor or not)                     │
└─────────────────────────────────────────────────────────────┘
```

---

## 📋 PREPROCESSING PIPELINE (Step-by-Step)

### **Phase 1: File Organization** (preprocessing.py)
```python
def organize_files(input_dir, output_dir):
    """
    Input:  Raw BraTS with variable naming (t1.nii.gz, T1.nii.gz, etc.)
    Output: Standardized structure with T1.nii.gz, T1ce.nii.gz, etc.
    """
    # Find all patient directories
    # For each patient:
    #   - Find T1, T1ce, T2, FLAIR, segmentation files
    #   - Copy with standard names
    #   - Handle alternative naming patterns (BraTS 2021 vs 2023)
```

### **Phase 2: Resample to Isotropic** (preprocessing.py)
```python
def resample_volume(sitk_image, new_spacing=[1.0, 1.0, 1.0]):
    """
    Input:  Variable resolution MRI (e.g., 1×1×2mm)
    Output: Isotropic 1×1×1mm resolution
    Method: SimpleITK ResampleImageFilter with linear interpolation
    """
    # Calculate new dimensions
    # Apply resample filter
    # Return resampled image
```

### **Phase 3: Crop/Pad to Standard Size** (preprocessing.py)
```python
def crop_pad_volume(sitk_image, target_shape=(240, 240, 155)):
    """
    Input:  Variable size volumes
    Output: All volumes exactly 240×240×155
    Method: Crop or pad with zeros
    """
    # Compare current shape to target
    # Crop if too large
    # Pad with zeros if too small
```

### **Phase 4: Z-Score Normalization** (preprocessing.py)
```python
# Inside preprocessing function
mean = volume[brain_mask].mean()
std = volume[brain_mask].std()
volume_normalized = (volume - mean) / (std + 1e-8)
volume_normalized[~brain_mask] = 0  # Reset background to 0
```

**Key Points:**
- Calculate mean/std from brain pixels only (mask > 0)
- Prevents background zeros from distorting statistics
- Normalizes to roughly -1 to +4 range
- Applied to each modality independently

### **Phase 5: Extract Slices** (preprocessing.py)
```python
# Select tumor-priority slices (~200 per patient)
# For each slice:
#   - Extract 4 modalities as separate arrays
#   - Create brain mask
#   - Assign labels (0=background, 1=tumor)
#   - Save as NPZ (compressed NumPy)
```

**Output Format (NPZ):**
```python
{
    'T1': array(240, 240),          # T1 modality slice
    'T1ce': array(240, 240),        # T1ce modality slice
    'T2': array(240, 240),          # T2 modality slice
    'FLAIR': array(240, 240),       # FLAIR modality slice
    'brain_mask': array(240, 240),  # Binary brain mask
    'label': array(240, 240),       # 0/1 segmentation
    'patient_id': 'BraTS2021_00000',
    'slice_idx': 50                 # Which slice in volume
}
```

---

## 🔍 KEY PREPROCESSING DETAILS

### **What BraTS Already Provides**
```
✅ Co-registration (all 4 modalities aligned)
✅ Skull stripping (background = 0)
✅ Resampling (approximately 1mm isotropic)
```

### **What WE Add**
```
✅ Additional z-score normalization (more aggressive)
✅ Strict 240×240×155 standardization
✅ Tumor-priority slice extraction
✅ NPZ format for efficient loading
```

### **Why Each Step Matters**

| Step | Why | Impact |
|------|-----|--------|
| **File Organization** | Standardize naming across BraTS versions | Prevents file-not-found errors |
| **Resample** | Clinical consistency (1mm³ standard) | Ensures comparable spatial resolution |
| **Crop/Pad** | All volumes same size for batching | Enables parallel processing |
| **Z-Score Normalize** | Standardize intensities across scanners | Makes features comparable |
| **Extract Slices** | Reduce memory for graph extraction | ~200 slices × 200 superpixels = manageable |

---

## 📊 PREPROCESSING OUTPUT

### **For Full Volumes (if saved):**
```
data/preprocessed/
├── BraTS2021_00000/
│   ├── BraTS2021_00000_t1.nii.gz (240×240×155, normalized)
│   ├── BraTS2021_00000_t1ce.nii.gz
│   ├── BraTS2021_00000_t2.nii.gz
│   ├── BraTS2021_00000_flair.nii.gz
│   └── BraTS2021_00000_seg.nii.gz
└── ... 1,250 more patients
```

### **For Extracted Slices (if saved):**
```
data/slices/
├── BraTS2021_00000/
│   ├── BraTS2021_00000_slice_000.npz
│   ├── BraTS2021_00000_slice_001.npz
│   ...
│   └── BraTS2021_00000_slice_199.npz
└── ... 1,250 more patients

Total: ~250,000 slice NPZ files
Size: ~50GB (compressed)
```

---

## 🎯 FOR PERSON 3 (DEFENSE PREPARATION)

### **What to Say on Slide 16 (1-2 minutes):**
```
"BraTS data comes pre-processed by organizers. We applied 
additional z-score normalization to standardize intensities 
across different scanners.

The 5 steps:
1. Load 4 MRI modalities (T1, T1ce, T2, FLAIR)
2. Co-register to same space
3. Skull strip (remove non-brain)
4. Z-score normalize: (x - mean) / std
5. Resample to 1mm³ isotropic

Result: 1,251 clean preprocessed patients
Time: ~15 minutes with 8 parallel workers"
```

### **If Asked Technical Questions:**

**Q: "What's z-score normalization?"**
> "Subtract mean, divide by std dev. Formula: (x - μ) / σ.
> Compresses variable intensities (0-3000) into standard range."

**Q: "Why calculate on brain mask only?"**
> "Background is already 0. Using brain tissue only gives 
> relevant statistics. Including background would distort results."

**Q: "Why 1mm³ isotropic?"**
> "Clinical standard. Makes voxels equal in all directions.
> Improves feature extraction consistency."

**Q: "How long does this take?"**
> "About 15 minutes for all 1,251 patients with 8 workers.
> One-time preprocessing cost."

---

## ✅ SUMMARY

| File | Responsibility | Output |
|------|-----------------|--------|
| **src/preprocessing.py** | All preprocessing operations | Clean, normalized MRI volumes |
| **src/dataset.py** | Load preprocessed data | PyTorch batches for training |
| **src/graph_construction.py** | Convert to graphs | Graph representations |

**Key takeaway:** Preprocessing is the foundation. Good preprocessing → good features → good model.

---

*Last updated: January 29, 2026*
