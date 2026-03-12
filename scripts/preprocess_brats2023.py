#!/usr/bin/env python3
"""
BraTS 2023 Preprocessing Script — Phase 7 (Task 7.2)

Same pipeline as BraTS 2021 (resample → crop/pad → skull-strip → z-score normalize).
Key differences from BraTS 2021:
  - File naming: *-t1n.nii.gz, *-t1c.nii.gz, *-t2w.nii.gz, *-t2f.nii.gz, *-seg.nii.gz
  - Label scheme: BraTS 2023 uses label 3 for ET — remapped to 4 so binary transform works

Output format (identical to BraTS 2021):
  data/preprocessed_brats2023/<patient_id>/<patient_id>_preprocessed.npz
  Keys: T1, T1ce, T2, FLAIR, label (multi-class, 0/1/2/4), brain_mask

Usage:
    python scripts/preprocess_brats2023.py --workers 12
    python scripts/preprocess_brats2023.py --workers 12 --demo  # 5 patients only
"""

import os
import sys
import glob
import gc
import time
import argparse
import traceback
import numpy as np
from pathlib import Path
from multiprocessing import Pool
from tqdm import tqdm
from scipy.ndimage import binary_fill_holes

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    import nibabel as nib
    _NIB_AVAILABLE = True
except ImportError:
    _NIB_AVAILABLE = False

try:
    import SimpleITK as sitk
    _SITK_AVAILABLE = True
except ImportError:
    _SITK_AVAILABLE = False


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BRATS2023_RAW = "/mnt/bigdata/capstone/ASNR-MICCAI-BraTS2023-GLI-Challenge-TrainingData"
OUTPUT_BASE = "data/preprocessed_brats2023"


# ---------------------------------------------------------------------------
# Core processing functions (mirror src/preprocessing.py)
# ---------------------------------------------------------------------------

def _resample(sitk_img, new_spacing=(1.0, 1.0, 1.0), interpolator=None):
    if interpolator is None:
        interpolator = sitk.sitkLinear
    orig_spacing = sitk_img.GetSpacing()
    orig_size = sitk_img.GetSize()
    new_size = [
        int(round(orig_size[i] * orig_spacing[i] / new_spacing[i]))
        for i in range(3)
    ]
    resample = sitk.ResampleImageFilter()
    resample.SetOutputSpacing(new_spacing)
    resample.SetSize(new_size)
    resample.SetOutputDirection(sitk_img.GetDirection())
    resample.SetOutputOrigin(sitk_img.GetOrigin())
    resample.SetTransform(sitk.Transform())
    resample.SetDefaultPixelValue(sitk_img.GetPixelIDValue())
    resample.SetInterpolator(interpolator)
    return resample.Execute(sitk_img)


def _crop_pad(sitk_img, target=(240, 240, 155)):
    """Crop or pad to target shape (x, y, z) → numpy (z, y, x)."""
    arr = sitk.GetArrayFromImage(sitk_img)          # (z, y, x)
    target_np = (target[2], target[1], target[0])   # (z, y, x)
    result = np.zeros(target_np, dtype=arr.dtype)
    slices_src = []
    slices_dst = []
    for i in range(3):
        s, t = arr.shape[i], target_np[i]
        if s >= t:
            start = (s - t) // 2
            slices_src.append(slice(start, start + t))
            slices_dst.append(slice(0, t))
        else:
            start = (t - s) // 2
            slices_src.append(slice(0, s))
            slices_dst.append(slice(start, start + s))
    result[tuple(slices_dst)] = arr[tuple(slices_src)]
    out = sitk.GetImageFromArray(result)
    out.SetSpacing(sitk_img.GetSpacing())
    out.SetOrigin(sitk_img.GetOrigin())
    out.SetDirection(sitk_img.GetDirection())
    return out


def _skull_strip(t1, t1ce, t2, flair):
    """Otsu brain mask from FLAIR, morphological fill, largest component."""
    otsu = sitk.OtsuThresholdImageFilter()
    otsu.SetInsideValue(0)
    otsu.SetOutsideValue(1)
    mask = otsu.Execute(flair)

    mask_np = sitk.GetArrayFromImage(mask)
    mask_np = binary_fill_holes(mask_np).astype(np.uint8)

    cc = sitk.ConnectedComponentImageFilter()
    mask_cc = cc.Execute(sitk.GetImageFromArray(mask_np))
    lm = sitk.LabelShapeStatisticsImageFilter()
    lm.Execute(mask_cc)

    best_label, best_count = 0, 0
    for lab in range(1, cc.GetObjectCount() + 1):
        n = lm.GetNumberOfPixels(lab)
        if n > best_count:
            best_count, best_label = n, lab
    mask_np = (sitk.GetArrayFromImage(mask_cc) == best_label).astype(np.uint8)

    def _apply(img):
        arr = sitk.GetArrayFromImage(img) * mask_np
        out = sitk.GetImageFromArray(arr)
        out.CopyInformation(img)
        return out

    mask_sitk = sitk.GetImageFromArray(mask_np)
    mask_sitk.CopyInformation(t1)
    return _apply(t1), _apply(t1ce), _apply(t2), _apply(flair), mask_sitk


def _normalize(img_sitk, mask_sitk, plo=0.5, phi=99.5):
    """Percentile clip + z-score within brain mask."""
    arr = sitk.GetArrayFromImage(img_sitk).astype(np.float32)
    mask = sitk.GetArrayFromImage(mask_sitk).astype(bool)
    brain_vals = arr[mask]
    if brain_vals.size == 0:
        return img_sitk
    p_lo = np.percentile(brain_vals, plo)
    p_hi = np.percentile(brain_vals, phi)
    clipped = np.clip(arr, p_lo, p_hi) * mask
    vals = clipped[mask]
    mu, sigma = vals.mean(), vals.std()
    norm = (clipped - mu) / (sigma if sigma > 0 else 1.0) * mask
    out = sitk.GetImageFromArray(norm)
    out.CopyInformation(img_sitk)
    return out


# ---------------------------------------------------------------------------
# Per-patient worker
# ---------------------------------------------------------------------------

def process_one_patient(args):
    patient_dir, output_base = args
    patient_id = Path(patient_dir).name
    output_dir = Path(output_base) / patient_id
    out_file = output_dir / f"{patient_id}_preprocessed.npz"

    if out_file.exists():
        return patient_id, True, "skipped (already done)"

    try:
        output_dir.mkdir(parents=True, exist_ok=True)

        # --- find files (BraTS 2023 naming) ---
        def _find(suffix):
            matches = glob.glob(os.path.join(patient_dir, f"*{suffix}"))
            if not matches:
                raise FileNotFoundError(f"No file matching *{suffix} in {patient_dir}")
            return matches[0]

        t1_path    = _find("-t1n.nii.gz")
        t1ce_path  = _find("-t1c.nii.gz")
        t2_path    = _find("-t2w.nii.gz")
        flair_path = _find("-t2f.nii.gz")
        seg_path   = _find("-seg.nii.gz")

        # --- load ---
        t1_s    = sitk.ReadImage(t1_path)
        t1ce_s  = sitk.ReadImage(t1ce_path)
        t2_s    = sitk.ReadImage(t2_path)
        flair_s = sitk.ReadImage(flair_path)
        seg_s   = sitk.ReadImage(seg_path)

        # --- resample to 1mm isotropic ---
        t1_s    = _resample(t1_s)
        t1ce_s  = _resample(t1ce_s)
        t2_s    = _resample(t2_s)
        flair_s = _resample(flair_s)
        seg_s   = _resample(seg_s, interpolator=sitk.sitkNearestNeighbor)

        # --- crop/pad to 240x240x155 ---
        t1_s    = _crop_pad(t1_s)
        t1ce_s  = _crop_pad(t1ce_s)
        t2_s    = _crop_pad(t2_s)
        flair_s = _crop_pad(flair_s)
        seg_s   = _crop_pad(seg_s)

        # --- skull strip ---
        t1_s, t1ce_s, t2_s, flair_s, mask_s = _skull_strip(t1_s, t1ce_s, t2_s, flair_s)

        # --- normalize ---
        t1_s    = _normalize(t1_s,    mask_s)
        t1ce_s  = _normalize(t1ce_s,  mask_s)
        t2_s    = _normalize(t2_s,    mask_s)
        flair_s = _normalize(flair_s, mask_s)

        # --- label: remap 3→4 (BraTS 2023 ET uses 3, BraTS 2021 uses 4) ---
        seg_arr = sitk.GetArrayFromImage(seg_s).astype(np.uint8)
        seg_arr[seg_arr == 3] = 4   # normalise to BraTS 2021 convention

        # --- to numpy ---
        T1       = sitk.GetArrayFromImage(t1_s).astype(np.float32)
        T1ce     = sitk.GetArrayFromImage(t1ce_s).astype(np.float32)
        T2       = sitk.GetArrayFromImage(t2_s).astype(np.float32)
        FLAIR    = sitk.GetArrayFromImage(flair_s).astype(np.float32)
        brain_mask = sitk.GetArrayFromImage(mask_s).astype(np.float32)

        # --- save ---
        np.savez_compressed(
            str(out_file),
            T1=T1, T1ce=T1ce, T2=T2, FLAIR=FLAIR,
            label=seg_arr,
            brain_mask=brain_mask
        )

        # cleanup
        del T1, T1ce, T2, FLAIR, brain_mask, seg_arr
        gc.collect()

        return patient_id, True, "ok"

    except Exception as e:
        tb = traceback.format_exc()
        return patient_id, False, f"ERROR: {e}\n{tb}"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Preprocess BraTS 2023 dataset")
    parser.add_argument("--raw_dir",   default=BRATS2023_RAW,  help="BraTS 2023 raw data dir")
    parser.add_argument("--output_dir", default=OUTPUT_BASE,   help="Output preprocessed dir")
    parser.add_argument("--workers",   type=int, default=12,   help="Number of parallel workers")
    parser.add_argument("--demo",      action="store_true",    help="Process 5 patients only")
    args = parser.parse_args()

    if not _SITK_AVAILABLE:
        print("ERROR: SimpleITK not available. Install with: pip install SimpleITK")
        sys.exit(1)

    # Collect patient dirs
    patient_dirs = sorted([
        d for d in glob.glob(os.path.join(args.raw_dir, "BraTS-GLI-*"))
        if os.path.isdir(d)
    ])

    if not patient_dirs:
        print(f"ERROR: No patient directories found in {args.raw_dir}")
        sys.exit(1)

    if args.demo:
        patient_dirs = patient_dirs[:5]
        print(f"DEMO MODE: processing {len(patient_dirs)} patients")
    else:
        print(f"Found {len(patient_dirs)} patient directories")

    # Check how many already done
    already_done = sum(
        1 for d in patient_dirs
        if (Path(args.output_dir) / Path(d).name /
            f"{Path(d).name}_preprocessed.npz").exists()
    )
    print(f"Already preprocessed: {already_done}/{len(patient_dirs)}")
    print(f"To process: {len(patient_dirs) - already_done}")
    print(f"Workers: {args.workers}")
    print(f"Output: {args.output_dir}/")
    print()

    os.makedirs(args.output_dir, exist_ok=True)

    work_args = [(d, args.output_dir) for d in patient_dirs]

    t0 = time.time()

    if args.workers == 1:
        results = []
        for a in tqdm(work_args, desc="Preprocessing"):
            results.append(process_one_patient(a))
    else:
        with Pool(processes=args.workers) as pool:
            results = list(tqdm(
                pool.imap_unordered(process_one_patient, work_args),
                total=len(work_args),
                desc="Preprocessing BraTS 2023"
            ))

    elapsed = time.time() - t0
    success = [r for r in results if r[1]]
    skipped = [r for r in results if r[1] and r[2] == "skipped (already done)"]
    failed  = [r for r in results if not r[1]]

    print(f"\n{'='*60}")
    print(f"Done in {elapsed/60:.1f} min")
    print(f"  Success: {len(success)}")
    print(f"  Skipped (already done): {len(skipped)}")
    print(f"  Failed:  {len(failed)}")
    if failed:
        print("\nFailed patients:")
        for pid, _, msg in failed:
            print(f"  {pid}: {msg[:200]}")

    # Final count
    total_npz = len(list(Path(args.output_dir).glob("*/*_preprocessed.npz")))
    print(f"\nTotal NPZ files in output: {total_npz}")
    print(f"Output dir: {Path(args.output_dir).resolve()}")


if __name__ == "__main__":
    main()
