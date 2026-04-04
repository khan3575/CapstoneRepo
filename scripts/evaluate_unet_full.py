#!/usr/bin/env python3
"""
U-Net Full Evaluation Pipeline
================================
Runs all 4 pending evaluations:

  1. Held-out evaluation (251 patients, 5-fold ensemble)
  2. BraTS 2023 zero-shot evaluation
  3. Speed/memory benchmark
  4. Wilcoxon signed-rank test (GNN vs U-Net)

Usage:
    python scripts/evaluate_unet_full.py                    # all tasks
    python scripts/evaluate_unet_full.py --task held_out    # single task
    python scripts/evaluate_unet_full.py --task brats2023
    python scripts/evaluate_unet_full.py --task benchmark
    python scripts/evaluate_unet_full.py --task wilcoxon
"""

import os
import sys
import json
import time
import gc
import argparse
import numpy as np
from pathlib import Path
from scipy import stats

import torch
import torch.nn as nn
import torch.nn.functional as F
import nibabel as nib

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# ============================================================
#  Paths
# ============================================================
PROJECT_ROOT = Path(__file__).parent.parent
UNET_CKPT_DIR = PROJECT_ROOT / "checkpoints" / "unet_baseline"
GNN_CKPT_DIR = PROJECT_ROOT / "checkpoints" / "binary_v3"
DATA_DIR = PROJECT_ROOT / "data" / "preprocessed"
BRATS2023_DIR = PROJECT_ROOT / "data" / "preprocessed_brats2023"
HELD_OUT_FILE = PROJECT_ROOT / "data" / "splits" / "held_out_test.json"
GNN_ENSEMBLE_FILE = PROJECT_ROOT / "research_results" / "ensemble_v2" / "ensemble_results.json"
OUTPUT_DIR = PROJECT_ROOT / "research_results" / "unet_evaluation"


# ============================================================
#  U-Net Model (same as train_unet_baseline.py)
# ============================================================

class UNet3D(nn.Module):
    def __init__(self, in_channels=4, base_channels=56, num_levels=4):
        super().__init__()
        self.num_levels = num_levels
        self.enc_blocks = nn.ModuleList()
        self.pool = nn.MaxPool3d(2)
        channels = base_channels
        for i in range(num_levels):
            in_ch = in_channels if i == 0 else channels // 2
            self.enc_blocks.append(self._conv_block(in_ch, channels))
            channels *= 2
        self.bottleneck = self._conv_block(channels // 2, channels)
        self.dec_blocks = nn.ModuleList()
        self.upconvs = nn.ModuleList()
        for i in range(num_levels):
            self.upconvs.append(nn.ConvTranspose3d(channels, channels // 2, kernel_size=2, stride=2))
            self.dec_blocks.append(self._conv_block(channels, channels // 2))
            channels //= 2
        self.out_conv = nn.Conv3d(base_channels, 1, kernel_size=1)

    def _conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.InstanceNorm3d(out_channels, affine=True),
            nn.LeakyReLU(0.01, inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.InstanceNorm3d(out_channels, affine=True),
            nn.LeakyReLU(0.01, inplace=True),
        )

    def forward(self, x):
        skip_connections = []
        for enc_block in self.enc_blocks:
            x = enc_block(x)
            skip_connections.append(x)
            x = self.pool(x)
        x = self.bottleneck(x)
        for upconv, dec_block in zip(self.upconvs, self.dec_blocks):
            x = upconv(x)
            skip = skip_connections.pop()
            if x.shape != skip.shape:
                x = F.interpolate(x, size=skip.shape[2:], mode='trilinear', align_corners=False)
            x = torch.cat([x, skip], dim=1)
            x = dec_block(x)
        return self.out_conv(x)


# ============================================================
#  Sliding-window inference
# ============================================================

def sliding_window_inference(model, volume_tensor, patch_size=(96, 96, 96),
                             overlap=0.5, device='cuda'):
    model.eval()
    C, H, W, D = volume_tensor.shape
    ph, pw, pd = patch_size
    sh = max(1, int(ph * (1 - overlap)))
    sw = max(1, int(pw * (1 - overlap)))
    sd = max(1, int(pd * (1 - overlap)))

    pad_h = max(0, ph - H)
    pad_w = max(0, pw - W)
    pad_d = max(0, pd - D)
    if pad_h > 0 or pad_w > 0 or pad_d > 0:
        volume_tensor = F.pad(volume_tensor, (0, pad_d, 0, pad_w, 0, pad_h))
        _, H, W, D = volume_tensor.shape

    output_sum = torch.zeros(1, H, W, D, device=device)
    count_map = torch.zeros(1, H, W, D, device=device)

    starts_h = list(range(0, H - ph + 1, sh))
    starts_w = list(range(0, W - pw + 1, sw))
    starts_d = list(range(0, D - pd + 1, sd))
    if starts_h[-1] + ph < H: starts_h.append(H - ph)
    if starts_w[-1] + pw < W: starts_w.append(W - pw)
    if starts_d[-1] + pd < D: starts_d.append(D - pd)

    with torch.no_grad(), torch.amp.autocast('cuda'):
        for h0 in starts_h:
            for w0 in starts_w:
                for d0 in starts_d:
                    patch = volume_tensor[:, h0:h0+ph, w0:w0+pw, d0:d0+pd]
                    pred = model(patch.unsqueeze(0).to(device))
                    output_sum[0, h0:h0+ph, w0:w0+pw, d0:d0+pd] += pred[0, 0]
                    count_map[0, h0:h0+ph, w0:w0+pw, d0:d0+pd] += 1.0

    output_avg = output_sum / count_map.clamp(min=1)
    orig_H = H - pad_h if pad_h > 0 else H
    orig_W = W - pad_w if pad_w > 0 else W
    orig_D = D - pad_d if pad_d > 0 else D
    return output_avg[:, :orig_H, :orig_W, :orig_D]


# ============================================================
#  Load helpers
# ============================================================

def load_unet_ensemble(device):
    """Load all 5 U-Net fold models."""
    models = []
    for fold_idx in range(5):
        ckpt_path = UNET_CKPT_DIR / f"fold_{fold_idx}" / "best_model.pt"
        if not ckpt_path.exists():
            print(f"  WARNING: fold {fold_idx} not found: {ckpt_path}")
            continue
        ckpt = torch.load(str(ckpt_path), map_location=device, weights_only=False)
        model = UNet3D(in_channels=4, base_channels=56, num_levels=4).to(device)
        model.load_state_dict(ckpt['model_state_dict'])
        model.eval()
        models.append(model)
        print(f"  Fold {fold_idx}: loaded (val_dice={ckpt.get('val_dice', 'N/A')})")
    print(f"  Loaded {len(models)}/5 U-Net models")
    return models


def load_patient_nifti(patient_dir):
    """Load a BraTS 2021 patient from NIfTI files."""
    patient_dir = Path(patient_dir)
    modalities = []
    for mod in ['FLAIR', 'T1', 'T1ce', 'T2']:
        img = nib.load(str(patient_dir / f"{mod}_preprocessed.nii.gz")).get_fdata()
        modalities.append(img)
    volume = torch.from_numpy(np.stack(modalities, axis=0).astype(np.float32))
    seg = nib.load(str(patient_dir / "whole_tumor_mask.nii.gz")).get_fdata()
    gt_mask = torch.from_numpy((seg > 0.5).astype(np.float32))
    return volume, gt_mask


def load_patient_npz(npz_path):
    """Load a BraTS 2023 patient from npz file."""
    data = np.load(str(npz_path))
    # Stack modalities: FLAIR, T1, T1ce, T2
    volume = np.stack([data['FLAIR'], data['T1'], data['T1ce'], data['T2']], axis=0).astype(np.float32)
    label = data['label'].astype(np.float32)
    # BraTS 2023: whole tumour = any label > 0
    gt_mask = (label > 0).astype(np.float32)
    return torch.from_numpy(volume), torch.from_numpy(gt_mask)


def compute_metrics(pred_mask, gt_mask):
    """Compute Dice, accuracy, sensitivity, specificity, precision."""
    pred = pred_mask.flatten().float()
    gt = gt_mask.flatten().float()
    tp = (pred * gt).sum()
    fp = (pred * (1 - gt)).sum()
    fn = ((1 - pred) * gt).sum()
    tn = ((1 - pred) * (1 - gt)).sum()
    dice = (2 * tp / (2 * tp + fp + fn)).item() if (2 * tp + fp + fn) > 0 else 1.0
    accuracy = ((tp + tn) / (tp + tn + fp + fn)).item()
    sensitivity = (tp / (tp + fn)).item() if (tp + fn) > 0 else 0.0
    specificity = (tn / (tn + fp)).item() if (tn + fp) > 0 else 0.0
    precision_val = (tp / (tp + fp)).item() if (tp + fp) > 0 else 0.0
    return {
        'dice': dice, 'accuracy': accuracy, 'sensitivity': sensitivity,
        'specificity': specificity, 'precision': precision_val,
    }


def ensemble_predict_patient(models, volume, device, patch_size=(96, 96, 96)):
    """Run ensemble sliding-window inference on a single patient volume."""
    logit_sum = None
    for model in models:
        logits = sliding_window_inference(model, volume, patch_size, overlap=0.5, device=device)
        if logit_sum is None:
            logit_sum = logits
        else:
            logit_sum += logits
    avg_logits = logit_sum / len(models)
    pred_mask = (torch.sigmoid(avg_logits[0].cpu()) > 0.5).float()
    return pred_mask


# ============================================================
#  Task 1: Held-out evaluation
# ============================================================

def run_held_out_evaluation(device):
    print("\n" + "=" * 80)
    print("TASK 1: U-Net Held-Out Evaluation (251 patients, ensemble)")
    print("=" * 80)

    with open(HELD_OUT_FILE) as f:
        held_out = json.load(f)
    patient_ids = held_out['patients']
    print(f"  Patients: {len(patient_ids)}")

    models = load_unet_ensemble(device)
    if len(models) == 0:
        print("  ERROR: No models loaded!")
        return None

    all_metrics = []
    per_patient_dice = {}

    for i, pid in enumerate(patient_ids):
        patient_dir = DATA_DIR / pid
        try:
            volume, gt_mask = load_patient_nifti(patient_dir)
            pred_mask = ensemble_predict_patient(models, volume, device)
            metrics = compute_metrics(pred_mask, gt_mask)
            metrics['patient_id'] = pid
            all_metrics.append(metrics)
            per_patient_dice[pid] = metrics['dice']
            if (i + 1) % 10 == 0 or (i + 1) == len(patient_ids):
                print(f"  [{i+1}/{len(patient_ids)}] {pid}: Dice={metrics['dice']:.4f}")
        except Exception as e:
            print(f"  ERROR {pid}: {e}")
            all_metrics.append({
                'patient_id': pid, 'dice': 0.0, 'accuracy': 0.0,
                'sensitivity': 0.0, 'specificity': 0.0, 'precision': 0.0,
            })
            per_patient_dice[pid] = 0.0

    dices = [m['dice'] for m in all_metrics]
    results = {
        'task': 'held_out_evaluation',
        'num_patients': len(patient_ids),
        'num_models': len(models),
        'ensemble_method': 'mean_logits',
        'metrics': {
            'mean_dice': float(np.mean(dices)),
            'std_dice': float(np.std(dices)),
            'median_dice': float(np.median(dices)),
            'mean_accuracy': float(np.mean([m['accuracy'] for m in all_metrics])),
            'mean_sensitivity': float(np.mean([m['sensitivity'] for m in all_metrics])),
            'mean_specificity': float(np.mean([m['specificity'] for m in all_metrics])),
            'mean_precision': float(np.mean([m['precision'] for m in all_metrics])),
        },
        'per_patient_dice': per_patient_dice,
    }

    print(f"\n  RESULTS:")
    print(f"    Mean Dice:   {results['metrics']['mean_dice']:.4f} +/- {results['metrics']['std_dice']:.4f}")
    print(f"    Median Dice: {results['metrics']['median_dice']:.4f}")
    print(f"    Sensitivity: {results['metrics']['mean_sensitivity']:.4f}")
    print(f"    Specificity: {results['metrics']['mean_specificity']:.4f}")
    print(f"    Precision:   {results['metrics']['mean_precision']:.4f}")

    out_path = OUTPUT_DIR / "held_out_results.json"
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"  Saved: {out_path}")

    # Free GPU memory
    del models
    torch.cuda.empty_cache()
    gc.collect()

    return results


# ============================================================
#  Task 2: BraTS 2023 zero-shot evaluation
# ============================================================

def run_brats2023_evaluation(device):
    print("\n" + "=" * 80)
    print("TASK 2: U-Net BraTS 2023 Zero-Shot Evaluation")
    print("=" * 80)

    if not BRATS2023_DIR.exists():
        print(f"  ERROR: BraTS 2023 data not found at {BRATS2023_DIR}")
        return None

    patient_dirs = sorted(BRATS2023_DIR.iterdir())
    patient_dirs = [d for d in patient_dirs if d.is_dir()]
    print(f"  Patients: {len(patient_dirs)}")

    models = load_unet_ensemble(device)
    if len(models) == 0:
        print("  ERROR: No models loaded!")
        return None

    all_metrics = []
    per_patient_dice = {}

    for i, pdir in enumerate(patient_dirs):
        pid = pdir.name
        npz_files = list(pdir.glob("*_preprocessed.npz"))
        if not npz_files:
            print(f"  SKIP {pid}: no npz file")
            continue
        try:
            volume, gt_mask = load_patient_npz(npz_files[0])
            pred_mask = ensemble_predict_patient(models, volume, device)
            metrics = compute_metrics(pred_mask, gt_mask)
            metrics['patient_id'] = pid
            all_metrics.append(metrics)
            per_patient_dice[pid] = metrics['dice']
            if (i + 1) % 50 == 0 or (i + 1) == len(patient_dirs):
                print(f"  [{i+1}/{len(patient_dirs)}] {pid}: Dice={metrics['dice']:.4f}")
        except Exception as e:
            print(f"  ERROR {pid}: {e}")
            all_metrics.append({
                'patient_id': pid, 'dice': 0.0, 'accuracy': 0.0,
                'sensitivity': 0.0, 'specificity': 0.0, 'precision': 0.0,
            })
            per_patient_dice[pid] = 0.0

    dices = [m['dice'] for m in all_metrics]

    # GNN BraTS 2023 reference
    gnn_brats2023_file = PROJECT_ROOT / "research_results" / "brats2023_evaluation" / "brats2023_results.json"
    gnn_dice = None
    if gnn_brats2023_file.exists():
        with open(gnn_brats2023_file) as f:
            gnn_data = json.load(f)
        gnn_dice = gnn_data.get('metrics', {}).get('dice', None)

    results = {
        'task': 'brats2023_zero_shot',
        'num_patients': len(all_metrics),
        'num_models': len(models),
        'ensemble_method': 'mean_logits',
        'metrics': {
            'mean_dice': float(np.mean(dices)),
            'std_dice': float(np.std(dices)),
            'median_dice': float(np.median(dices)),
            'mean_accuracy': float(np.mean([m['accuracy'] for m in all_metrics])),
            'mean_sensitivity': float(np.mean([m['sensitivity'] for m in all_metrics])),
            'mean_specificity': float(np.mean([m['specificity'] for m in all_metrics])),
            'mean_precision': float(np.mean([m['precision'] for m in all_metrics])),
        },
        'per_patient_dice': per_patient_dice,
        'gnn_brats2023_dice': gnn_dice,
    }

    print(f"\n  RESULTS:")
    print(f"    Mean Dice:   {results['metrics']['mean_dice']:.4f} +/- {results['metrics']['std_dice']:.4f}")
    print(f"    Median Dice: {results['metrics']['median_dice']:.4f}")
    if gnn_dice:
        gap = results['metrics']['mean_dice'] - gnn_dice
        print(f"    GNN BraTS2023 Dice: {gnn_dice:.4f} (gap: {gap:+.4f})")

    out_path = OUTPUT_DIR / "brats2023_results.json"
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"  Saved: {out_path}")

    del models
    torch.cuda.empty_cache()
    gc.collect()

    return results


# ============================================================
#  Task 3: Speed/memory benchmark
# ============================================================

def run_benchmark(device):
    print("\n" + "=" * 80)
    print("TASK 3: U-Net Speed/Memory Benchmark")
    print("=" * 80)

    # Load a single model (not ensemble — benchmark per-model cost)
    ckpt_path = UNET_CKPT_DIR / "fold_0" / "best_model.pt"
    ckpt = torch.load(str(ckpt_path), map_location=device, weights_only=False)
    model = UNet3D(in_channels=4, base_channels=56, num_levels=4).to(device)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()

    num_params = sum(p.numel() for p in model.parameters())
    model_size_mb = sum(p.nelement() * p.element_size() for p in model.parameters()) / (1024 * 1024)
    print(f"  Parameters: {num_params:,}")
    print(f"  Model size: {model_size_mb:.1f} MB")

    # Pick a held-out patient for benchmarking
    with open(HELD_OUT_FILE) as f:
        held_out = json.load(f)
    test_pid = held_out['patients'][0]
    patient_dir = DATA_DIR / test_pid
    volume, gt_mask = load_patient_nifti(patient_dir)
    print(f"  Test patient: {test_pid}, volume shape: {tuple(volume.shape)}")

    # Warmup
    print("  Warming up...")
    torch.cuda.reset_peak_memory_stats(device)
    for _ in range(2):
        _ = sliding_window_inference(model, volume, device=device)
        torch.cuda.synchronize()

    # Benchmark: single model inference
    print("  Benchmarking single-model inference...")
    torch.cuda.reset_peak_memory_stats(device)
    times = []
    num_runs = 5
    for run in range(num_runs):
        torch.cuda.synchronize()
        start = time.perf_counter()
        _ = sliding_window_inference(model, volume, device=device)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        times.append(elapsed)
        print(f"    Run {run+1}: {elapsed:.3f}s")

    peak_mem_bytes = torch.cuda.max_memory_allocated(device)
    peak_mem_mb = peak_mem_bytes / (1024 * 1024)

    # Benchmark: ensemble inference (5 models)
    print("  Benchmarking ensemble inference (5 models)...")
    models = load_unet_ensemble(device)
    torch.cuda.reset_peak_memory_stats(device)
    ensemble_times = []
    for run in range(3):
        torch.cuda.synchronize()
        start = time.perf_counter()
        _ = ensemble_predict_patient(models, volume, device)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        ensemble_times.append(elapsed)
        print(f"    Ensemble run {run+1}: {elapsed:.3f}s")

    ensemble_peak_mem = torch.cuda.max_memory_allocated(device) / (1024 * 1024)

    results = {
        'task': 'speed_memory_benchmark',
        'model': {
            'name': 'UNet3D',
            'base_channels': 56,
            'num_levels': 4,
            'parameters': num_params,
            'model_size_mb': round(model_size_mb, 1),
        },
        'test_volume': {
            'patient_id': test_pid,
            'shape': list(volume.shape),
        },
        'single_model': {
            'mean_inference_seconds': float(np.mean(times)),
            'std_inference_seconds': float(np.std(times)),
            'min_inference_seconds': float(np.min(times)),
            'peak_gpu_memory_mb': round(peak_mem_mb, 1),
            'num_runs': num_runs,
        },
        'ensemble_5_models': {
            'mean_inference_seconds': float(np.mean(ensemble_times)),
            'std_inference_seconds': float(np.std(ensemble_times)),
            'peak_gpu_memory_mb': round(ensemble_peak_mem, 1),
            'num_runs': len(ensemble_times),
        },
        'device': str(device),
        'gpu_name': torch.cuda.get_device_name(device) if device.type == 'cuda' else 'N/A',
    }

    print(f"\n  RESULTS:")
    print(f"    Single model: {results['single_model']['mean_inference_seconds']:.3f}s "
          f"(peak {results['single_model']['peak_gpu_memory_mb']:.0f} MB)")
    print(f"    Ensemble (5): {results['ensemble_5_models']['mean_inference_seconds']:.3f}s "
          f"(peak {results['ensemble_5_models']['peak_gpu_memory_mb']:.0f} MB)")

    out_path = OUTPUT_DIR / "benchmark_results.json"
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"  Saved: {out_path}")

    del models, model
    torch.cuda.empty_cache()
    gc.collect()

    return results


# ============================================================
#  Task 4: Wilcoxon signed-rank test
# ============================================================

def run_wilcoxon_test():
    print("\n" + "=" * 80)
    print("TASK 4: Wilcoxon Signed-Rank Test (GNN vs U-Net)")
    print("=" * 80)

    # Load U-Net held-out per-patient Dice
    unet_results_file = OUTPUT_DIR / "held_out_results.json"
    if not unet_results_file.exists():
        print("  ERROR: Run held-out evaluation first (Task 1)")
        return None

    with open(unet_results_file) as f:
        unet_data = json.load(f)
    unet_per_patient = unet_data['per_patient_dice']

    # Load GNN per-patient Dice from CV folds (since ensemble doesn't have per-patient)
    # Actually, we need to compare on the SAME patients.
    # Option A: Use CV fold data (both models tested on same 1000 patients)
    # Option B: Compute GNN held-out per-patient Dice
    #
    # For CV comparison: U-Net has per_patient_dice in each fold.
    # GNN does NOT have per-patient dice in fold results.
    # So we use the held-out set. We need GNN per-patient dice on held-out.
    # The GNN ensemble ran on held-out but only has per-graph (per-slice) dice.
    #
    # Best approach: compute GNN per-patient Dice from held-out graphs.
    # This requires loading GNN models + graph data. Let's do it.

    print("  Computing GNN per-patient Dice on held-out set...")
    gnn_per_patient = compute_gnn_held_out_per_patient()

    if gnn_per_patient is None:
        print("  ERROR: Could not compute GNN per-patient Dice")
        return None

    # Match patients
    common_patients = sorted(set(unet_per_patient.keys()) & set(gnn_per_patient.keys()))
    print(f"  Common patients: {len(common_patients)}")

    unet_dices = np.array([unet_per_patient[p] for p in common_patients])
    gnn_dices = np.array([gnn_per_patient[p] for p in common_patients])

    # Wilcoxon signed-rank test
    stat, p_value = stats.wilcoxon(gnn_dices, unet_dices, alternative='two-sided')

    # Also test one-sided: GNN > U-Net
    stat_greater, p_greater = stats.wilcoxon(gnn_dices, unet_dices, alternative='greater')

    results = {
        'task': 'wilcoxon_signed_rank_test',
        'num_patients': len(common_patients),
        'gnn_mean_dice': float(np.mean(gnn_dices)),
        'gnn_std_dice': float(np.std(gnn_dices)),
        'gnn_median_dice': float(np.median(gnn_dices)),
        'unet_mean_dice': float(np.mean(unet_dices)),
        'unet_std_dice': float(np.std(unet_dices)),
        'unet_median_dice': float(np.median(unet_dices)),
        'two_sided': {
            'statistic': float(stat),
            'p_value': float(p_value),
            'significant_005': bool(p_value < 0.05),
            'significant_001': bool(p_value < 0.01),
        },
        'one_sided_gnn_greater': {
            'statistic': float(stat_greater),
            'p_value': float(p_greater),
            'significant_005': bool(p_greater < 0.05),
        },
        'per_patient': {p: {'gnn': float(gnn_dices[i]), 'unet': float(unet_dices[i])}
                        for i, p in enumerate(common_patients)},
    }

    print(f"\n  RESULTS:")
    print(f"    GNN  held-out Dice: {results['gnn_mean_dice']:.4f} +/- {results['gnn_std_dice']:.4f}")
    print(f"    U-Net held-out Dice: {results['unet_mean_dice']:.4f} +/- {results['unet_std_dice']:.4f}")
    print(f"    Difference: {results['gnn_mean_dice'] - results['unet_mean_dice']:+.4f}")
    print(f"    Wilcoxon (two-sided): W={stat:.1f}, p={p_value:.6f}")
    print(f"    Wilcoxon (GNN > U-Net): p={p_greater:.6f}")
    if p_value < 0.001:
        print(f"    Significant at p < 0.001")
    elif p_value < 0.01:
        print(f"    Significant at p < 0.01")
    elif p_value < 0.05:
        print(f"    Significant at p < 0.05")
    else:
        print(f"    NOT significant at p < 0.05")

    out_path = OUTPUT_DIR / "wilcoxon_results.json"
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"  Saved: {out_path}")

    return results


def compute_gnn_held_out_per_patient():
    """Compute GNN ensemble per-patient Dice on the held-out set."""
    try:
        from gnn_model import TumorSegmentationGNN
        from dataset import BraTSGraphDataset, BinaryTransform
        from torch_geometric.loader import DataLoader as GeometricDataLoader
    except ImportError as e:
        print(f"  Import error: {e}")
        return None

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load GNN ensemble
    models = []
    for fold_idx in range(5):
        ckpt_path = GNN_CKPT_DIR / f"fold_{fold_idx}" / "best_model.pth"
        if not ckpt_path.exists():
            print(f"  WARNING: GNN fold {fold_idx} not found")
            continue
        ckpt = torch.load(str(ckpt_path), map_location=device, weights_only=False)
        model = TumorSegmentationGNN(
            in_channels=15, hidden_channels=256, num_layers=5, dropout=0.1
        ).to(device)
        state_dict = ckpt['model_state_dict']
        if any(k.startswith('_orig_mod.') for k in state_dict):
            state_dict = {k.replace('_orig_mod.', '', 1): v for k, v in state_dict.items()}
        model.load_state_dict(state_dict)
        model.eval()
        models.append(model)
    print(f"  Loaded {len(models)}/5 GNN models")

    # Load held-out patient list
    with open(HELD_OUT_FILE) as f:
        held_out = json.load(f)
    patient_ids = held_out['patients']

    graphs_dir = PROJECT_ROOT / "data" / "graphs"
    binary_transform = BinaryTransform()
    per_patient_dice = {}

    for i, pid in enumerate(patient_ids):
        graph_file = graphs_dir / pid / f"{pid}_graphs_200.pt"
        if not graph_file.exists():
            per_patient_dice[pid] = 0.0
            continue

        try:
            dataset = BraTSGraphDataset(
                root_dir=None, split='test',
                graph_files=[str(graph_file)],
                transform=binary_transform
            )
            loader = GeometricDataLoader(dataset, batch_size=64, shuffle=False)

            total_tp, total_fp, total_fn = 0, 0, 0

            with torch.no_grad():
                for data in loader:
                    data = data.to(device, non_blocking=True)
                    logits_sum = torch.zeros(data.x.shape[0], device=device)
                    for model in models:
                        logits, _ = model(data)
                        logits_sum += logits
                    avg_logits = logits_sum / len(models)
                    preds = (torch.sigmoid(avg_logits) > 0.5).float()
                    y_true = (data.y > 0).float()

                    total_tp += (preds * y_true).sum().item()
                    total_fp += (preds * (1 - y_true)).sum().item()
                    total_fn += ((1 - preds) * y_true).sum().item()

            dice = (2 * total_tp) / (2 * total_tp + total_fp + total_fn) if (2 * total_tp + total_fp + total_fn) > 0 else 1.0
            per_patient_dice[pid] = dice

            if (i + 1) % 25 == 0:
                print(f"    GNN [{i+1}/{len(patient_ids)}] {pid}: Dice={dice:.4f}")

        except Exception as e:
            print(f"    GNN ERROR {pid}: {e}")
            per_patient_dice[pid] = 0.0

    # Save GNN per-patient results for future use
    gnn_out = OUTPUT_DIR / "gnn_held_out_per_patient.json"
    with open(gnn_out, 'w') as f:
        json.dump(per_patient_dice, f, indent=2)
    print(f"  Saved GNN per-patient Dice: {gnn_out}")

    # Cleanup
    del models
    torch.cuda.empty_cache()
    gc.collect()

    return per_patient_dice


# ============================================================
#  Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(description='U-Net Full Evaluation Pipeline')
    parser.add_argument('--task', type=str, default='all',
                        choices=['all', 'held_out', 'brats2023', 'benchmark', 'wilcoxon'],
                        help='Which evaluation to run')
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(device)}")
        print(f"VRAM: {torch.cuda.get_device_properties(device).total_memory / 1024**3:.1f} GB")

    if args.task in ('all', 'held_out'):
        run_held_out_evaluation(device)

    if args.task in ('all', 'brats2023'):
        run_brats2023_evaluation(device)

    if args.task in ('all', 'benchmark'):
        run_benchmark(device)

    if args.task in ('all', 'wilcoxon'):
        run_wilcoxon_test()

    print("\n" + "=" * 80)
    print("ALL DONE")
    print("=" * 80)


if __name__ == "__main__":
    main()
