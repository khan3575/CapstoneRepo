#!/usr/bin/env python3
"""
Two-Scenario Timing Benchmark
==============================
Measures and reports both honest timing scenarios for the paper:

  Scenario 1 — Inference Only (preprocessed):
    Input:  pre-built graph (.pt file)
    Steps:  load graph → GNN forward pass → threshold
    Use:    deployment setting where graphs are precomputed offline

  Scenario 2 — End-to-End (from preprocessed MRI):
    Input:  preprocessed .npz (skull-stripped, normalised MRI)
    Steps:  SLIC superpixel construction → graph build → GNN forward pass → threshold
    Use:    realistic clinical use — new patient arrives with preprocessed scan

Both numbers are reported transparently in the paper to avoid cherry-picking.

Usage:
    python scripts/benchmark_two_scenarios.py --num_patients 50 --fold 0
    python scripts/benchmark_two_scenarios.py --num_patients 50 --fold 0 --device cpu
"""

import sys
import time
import json
import gc
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

import torch
import numpy as np
from tqdm import tqdm
from torch_geometric.loader import DataLoader as GeometricDataLoader

from gnn_model import TumorSegmentationGNN
from dataset import BraTSGraphDataset, BinaryTransform
from graph_construction import GraphBuilder, Config as GraphConfig
from config import get_config


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def gpu_sync(device):
    if device.type == 'cuda':
        torch.cuda.synchronize()


def load_model(checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model = TumorSegmentationGNN(
        in_channels=15,
        hidden_channels=256,
        num_layers=5,
        dropout=0.1
    ).to(device)
    # torch.compile() adds '_orig_mod.' prefix — strip it for loading into uncompiled model
    state_dict = checkpoint['model_state_dict']
    if any(k.startswith('_orig_mod.') for k in state_dict):
        state_dict = {k.replace('_orig_mod.', '', 1): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict)
    model.eval()
    return model


def model_size_mb(checkpoint_path):
    return Path(checkpoint_path).stat().st_size / (1024 ** 2)


# ─────────────────────────────────────────────────────────────────────────────
# Scenario 1 — Inference Only (pre-built graph)
# ─────────────────────────────────────────────────────────────────────────────

def benchmark_scenario1(model, graph_files, device, warmup=3):
    """
    Measures GNN forward pass time only.
    Input is a pre-built .pt graph file (graph construction already done offline).
    """
    binary_transform = BinaryTransform()
    times_ms = []

    # Warm up GPU with first few patients (not recorded)
    warmup_files = graph_files[:warmup]
    for gf in warmup_files:
        if not Path(gf).exists():
            continue
        ds = BraTSGraphDataset(root_dir=None, split='test',
                               graph_files=[gf], transform=binary_transform)
        loader = GeometricDataLoader(ds, batch_size=1, shuffle=False)
        with torch.no_grad():
            for data in loader:
                data = data.to(device, non_blocking=True)
                _ = model(data)
        gpu_sync(device)

    # Timed runs — batch_size=32 loads multiple slice-graphs per patient at once
    for gf in tqdm(graph_files[warmup:], desc='  Scenario 1 — inference only'):
        if not Path(gf).exists():
            continue

        ds = BraTSGraphDataset(root_dir=None, split='test',
                               graph_files=[gf], transform=binary_transform)
        loader = GeometricDataLoader(ds, batch_size=32, shuffle=False,
                                     num_workers=2, pin_memory=True)

        gpu_sync(device)
        t0 = time.perf_counter()

        with torch.no_grad():
            for data in loader:
                data = data.to(device, non_blocking=True)
                logits, _ = model(data)
                preds = (torch.sigmoid(logits) > 0.5).float()

        gpu_sync(device)
        elapsed_ms = (time.perf_counter() - t0) * 1000
        times_ms.append(elapsed_ms)

        del ds, loader
        gc.collect()
        if device.type == 'cuda':
            torch.cuda.empty_cache()

    return times_ms


# ─────────────────────────────────────────────────────────────────────────────
# Scenario 2 — End-to-End (from preprocessed .npz)
# ─────────────────────────────────────────────────────────────────────────────

def benchmark_scenario2(model, patient_ids, preprocessed_dir, device, warmup=3):
    """
    Measures full pipeline: SLIC superpixel construction + graph build + inference.
    Input is a preprocessed .npz (skull-stripped, z-score normalised MRI).
    This is the realistic 'new patient arrives' scenario.
    """
    graph_config = GraphConfig(n_superpixels=200, max_memory_gb=3.0)
    builder = GraphBuilder(graph_config)
    binary_transform = BinaryTransform()

    times_construction_ms = []
    times_inference_ms = []
    times_total_ms = []
    errors = 0

    for i, pid in enumerate(tqdm(patient_ids, desc='  Scenario 2 — end-to-end')):
        npz_path = Path(preprocessed_dir) / pid / f'{pid}_preprocessed.npz'
        if not npz_path.exists():
            errors += 1
            continue

        try:
            volume_data = np.load(npz_path)

            # ── Step A: SLIC graph construction ──────────────────────────────
            gpu_sync(device)
            t_start = time.perf_counter()

            graphs, _ = builder.process_volume({
                'T1':    volume_data['T1'],
                'T1ce':  volume_data['T1ce'],
                'T2':    volume_data['T2'],
                'FLAIR': volume_data['FLAIR'],
                'label': volume_data['label'],
                'brain_mask': volume_data.get(
                    'brain_mask',
                    (volume_data['T1'] > 0).astype(np.float32)
                )
            })

            t_after_construction = time.perf_counter()

            # ── Step B: GNN inference on all slice graphs ─────────────────────
            # Convert to PyG format and run inference
            from torch_geometric.data import Data, Batch
            pyg_graphs = []
            for g in graphs:
                pyg_g = Data(
                    x=torch.tensor(g.x, dtype=torch.float32),
                    edge_index=torch.tensor(g.edge_index, dtype=torch.long),
                    y=torch.tensor(g.y, dtype=torch.long) if hasattr(g, 'y') else None
                )
                pyg_graphs.append(pyg_g)

            if len(pyg_graphs) == 0:
                errors += 1
                continue

            gpu_sync(device)
            t_inference_start = time.perf_counter()

            batch = Batch.from_data_list(pyg_graphs).to(device)
            with torch.no_grad():
                logits, _ = model(batch)
                preds = (torch.sigmoid(logits) > 0.5).float()

            gpu_sync(device)
            t_end = time.perf_counter()

            # ── Record times ─────────────────────────────────────────────────
            construction_ms = (t_after_construction - t_start) * 1000
            inference_ms    = (t_end - t_inference_start) * 1000
            total_ms        = (t_end - t_start) * 1000

            # Skip warmup patients from recording
            if i >= warmup:
                times_construction_ms.append(construction_ms)
                times_inference_ms.append(inference_ms)
                times_total_ms.append(total_ms)

            del volume_data, graphs, pyg_graphs, batch
            gc.collect()
            if device.type == 'cuda':
                torch.cuda.empty_cache()

        except Exception as e:
            errors += 1
            if errors <= 3:
                print(f'\n  Warning: error on {pid}: {e}')

    if errors > 0:
        print(f'\n  Skipped {errors} patients due to errors.')

    return times_construction_ms, times_inference_ms, times_total_ms


# ─────────────────────────────────────────────────────────────────────────────
# Memory measurement
# ─────────────────────────────────────────────────────────────────────────────

def measure_peak_memory(model, graph_file, device):
    if device.type != 'cuda':
        return {'peak_mb': 0, 'allocated_mb': 0}

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    binary_transform = BinaryTransform()
    ds = BraTSGraphDataset(root_dir=None, split='test',
                           graph_files=[graph_file], transform=binary_transform)
    loader = GeometricDataLoader(ds, batch_size=1, shuffle=False)

    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            _ = model(data)

    return {
        'peak_mb':      torch.cuda.max_memory_allocated() / 1024**2,
        'allocated_mb': torch.cuda.memory_allocated() / 1024**2
    }


# ─────────────────────────────────────────────────────────────────────────────
# Summary printer
# ─────────────────────────────────────────────────────────────────────────────

def stats(times_ms):
    arr = np.array(times_ms)
    return {
        'mean_ms':   float(np.mean(arr)),
        'std_ms':    float(np.std(arr)),
        'median_ms': float(np.median(arr)),
        'min_ms':    float(np.min(arr)),
        'max_ms':    float(np.max(arr)),
        'n':         len(arr)
    }


def print_results(s1_ms, s2_construction_ms, s2_inference_ms, s2_total_ms, memory, model_mb):
    s1 = stats(s1_ms)
    s2c = stats(s2_construction_ms)
    s2i = stats(s2_inference_ms)
    s2t = stats(s2_total_ms)

    w = 70
    print('\n' + '=' * w)
    print('TIMING RESULTS — Two-Scenario Benchmark')
    print('=' * w)

    print(f'\n{"Scenario 1 — Inference Only (pre-built graph)":}')
    print(f'  Input:   pre-built .pt graph file')
    print(f'  Mean:    {s1["mean_ms"]:.1f} ms  ±  {s1["std_ms"]:.1f} ms')
    print(f'  Median:  {s1["median_ms"]:.1f} ms')
    print(f'  Range:   [{s1["min_ms"]:.1f}, {s1["max_ms"]:.1f}] ms')
    print(f'  Patients: {s1["n"]}')

    print(f'\n{"Scenario 2 — End-to-End (from preprocessed .npz)":}')
    print(f'  Input:   preprocessed .npz (skull-stripped, normalised MRI)')
    print(f'  ├─ SLIC graph construction: {s2c["mean_ms"]:.0f} ms  ±  {s2c["std_ms"]:.0f} ms')
    print(f'  ├─ GNN inference:           {s2i["mean_ms"]:.1f} ms  ±  {s2i["std_ms"]:.1f} ms')
    print(f'  └─ Total:                   {s2t["mean_ms"]:.0f} ms  ±  {s2t["std_ms"]:.0f} ms  ({s2t["mean_ms"]/1000:.2f}s)')
    print(f'  Patients: {s2t["n"]}')

    print(f'\n{"Model & Memory":}')
    print(f'  Model file size:  {model_mb:.1f} MB')
    print(f'  Peak GPU memory:  {memory["peak_mb"]:.0f} MB  (inference only)')

    print(f'\n{"Paper Table Entry":}')
    print(f'  {"":40s} {"GNN (ours)":>12}')
    print(f'  {"Inference time (preprocessed graphs)":40s} {s1["mean_ms"]:.0f} ms')
    print(f'  {"End-to-end time (from preprocessed MRI)":40s} {s2t["mean_ms"]/1000:.2f} s')
    print(f'  {"Model size":40s} {model_mb:.1f} MB')
    print(f'  {"Peak GPU memory (inference)":40s} {memory["peak_mb"]:.0f} MB')
    print('=' * w)

    return {
        'scenario1_inference_only': s1,
        'scenario2_graph_construction': s2c,
        'scenario2_gnn_inference': s2i,
        'scenario2_end_to_end': s2t,
        'model_size_mb': model_mb,
        'memory': memory
    }


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='Two-scenario timing benchmark')
    parser.add_argument('--num_patients', type=int, default=50,
                        help='Number of patients to benchmark (default: 50)')
    parser.add_argument('--fold', type=int, default=0,
                        help='Which fold model to use (default: 0)')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device: cuda or cpu (default: cuda)')
    parser.add_argument('--output', type=str,
                        default='research_results/timing_benchmark',
                        help='Output directory for JSON results')
    parser.add_argument('--demo', action='store_true',
                        help='Demo mode: 8 patients, warmup=2 — verify no errors before full run')
    args = parser.parse_args()

    if args.demo:
        args.num_patients = 8
        print("[DEMO MODE: 8 patients]")

    config = get_config()
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

    print('=' * 70)
    print('Two-Scenario Timing Benchmark')
    print('=' * 70)
    print(f'Device:   {device}')
    if device.type == 'cuda':
        print(f'GPU:      {torch.cuda.get_device_name(0)}')
    print(f'Patients: {args.num_patients}')
    print(f'Fold:     {args.fold}')

    # ── Load model ───────────────────────────────────────────────────────────
    checkpoint_path = Path(config.checkpoints_binary) / f'fold_{args.fold}' / 'best_model.pth'
    if not checkpoint_path.exists():
        print(f'\nERROR: Checkpoint not found: {checkpoint_path}')
        print('Train fold models first with: python src/train_cv_fold.py')
        return

    print(f'\nLoading model from: {checkpoint_path}')
    model = load_model(checkpoint_path, device)
    model_mb = model_size_mb(checkpoint_path)
    print(f'Model size: {model_mb:.1f} MB')
    print(f'Parameters: {sum(p.numel() for p in model.parameters()):,}')

    # ── Load patient list (use test patients from the specified fold) ─────────
    fold_file = Path(config.data_cv_folds) / f'fold_{args.fold}.json'
    with open(fold_file) as f:
        fold_data = json.load(f)

    # Use test patients so we benchmark on held-out patients
    all_test_patients = fold_data['test_patients']
    patients = all_test_patients[:args.num_patients]
    print(f'Using {len(patients)} test patients from fold {args.fold}')

    graph_files = [
        str(Path(config.data_graphs) / pid / f'{pid}_graphs_200.pt')
        for pid in patients
    ]

    # Filter to files that exist
    valid = [(p, gf) for p, gf in zip(patients, graph_files) if Path(gf).exists()]
    patients_valid = [p for p, _ in valid]
    graph_files_valid = [gf for _, gf in valid]
    print(f'Valid graph files found: {len(graph_files_valid)}/{len(patients)}')

    # ── Measure peak memory ──────────────────────────────────────────────────
    print('\nMeasuring peak GPU memory...')
    memory = measure_peak_memory(model, graph_files_valid[0], device)
    print(f'Peak GPU memory: {memory["peak_mb"]:.0f} MB')

    # ── Scenario 1 ───────────────────────────────────────────────────────────
    print('\nRunning Scenario 1 — Inference Only...')
    s1_times = benchmark_scenario1(model, graph_files_valid, device)

    # ── Scenario 2 ───────────────────────────────────────────────────────────
    print('\nRunning Scenario 2 — End-to-End from preprocessed .npz...')
    s2c_times, s2i_times, s2t_times = benchmark_scenario2(
        model, patients_valid, config.data_preprocessed, device
    )

    # ── Print and save results ───────────────────────────────────────────────
    results = print_results(s1_times, s2c_times, s2i_times, s2t_times, memory, model_mb)

    results['metadata'] = {
        'device': str(device),
        'gpu': torch.cuda.get_device_name(0) if device.type == 'cuda' else 'CPU',
        'fold': args.fold,
        'num_patients': len(s1_times),
        'checkpoint': str(checkpoint_path)
    }

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    out_file = output_dir / 'two_scenario_results.json'
    with open(out_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f'\nResults saved to: {out_file}')


if __name__ == '__main__':
    main()
