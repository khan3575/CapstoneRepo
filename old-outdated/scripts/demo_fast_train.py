"""
Demo Fast Training Script - Verify optimizations before full CV run
Runs fold 0 with a small subset (10 train / 5 val patients) for 3 epochs.

Fixes applied vs original train_cv_fold.py:
  1. num_workers=4 (was 0) — parallelizes data loading, big speedup
  2. cudnn.benchmark=True (was False) — lets cuDNN find fastest kernels for RTX 2060
  3. cudnn.deterministic=False (was True) — removes reproducibility overhead
  4. persistent_workers=True — workers stay alive between epochs
  5. prefetch_factor=2 — workers preload next batch while GPU trains
  6. batch_size=64 (was 32) — better GPU utilization on 6GB VRAM w/ 439K param model
  7. torch.compile(model) — PyTorch 2.8 fuses GNN ops, ~15-25% speedup
  8. accumulation_steps=2 (was 4) — matches increased batch size
"""

import os
import sys
import json
import time
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from torch.amp import autocast, GradScaler
from torch_geometric.loader import DataLoader as GeometricDataLoader

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from config import get_config
from dataset import BraTSGraphDataset, BinaryTransform
from gnn_model import TumorSegmentationGNN


# ─── Hyperparameters ─────────────────────────────────────────────────────────
DEMO_TRAIN_PATIENTS = 10   # patients in demo train set  (~700 graphs)
DEMO_VAL_PATIENTS   = 5    # patients in demo val set    (~300 graphs)
DEMO_EPOCHS         = 3
BATCH_SIZE          = 64   # up from 32 — saturates GPU better
ACCUMULATION_STEPS  = 2    # down from 4 — matches larger batch
NUM_WORKERS         = 4    # parallel data loading workers
LEARNING_RATE       = 0.001
HIDDEN_CHANNELS     = 256
NUM_LAYERS          = 5
# ─────────────────────────────────────────────────────────────────────────────


def set_seed_fast(seed: int = 42):
    """Seed for reproducibility WITHOUT sacrificing cuDNN speed."""
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # KEY FIX: benchmark=True lets cuDNN profile & pick fastest kernels
    # KEY FIX: deterministic=False removes the ~10-20% reproducibility overhead
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False


def compute_metrics(predictions, labels):
    predictions = predictions.cpu().numpy()
    labels = labels.cpu().numpy()
    pred_binary = (predictions > 0.5).astype(int)
    tp = np.sum((pred_binary == 1) & (labels == 1))
    tn = np.sum((pred_binary == 0) & (labels == 0))
    fp = np.sum((pred_binary == 1) & (labels == 0))
    fn = np.sum((pred_binary == 0) & (labels == 1))
    epsilon = 1e-7
    dice = (2 * tp) / (2 * tp + fp + fn + epsilon)
    accuracy = (tp + tn) / (tp + tn + fp + fn + epsilon)
    return {'dice': float(dice), 'accuracy': float(accuracy)}


def train_epoch(model, train_loader, criterion, optimizer, scheduler, scaler, device):
    model.train()
    total_loss = 0
    all_preds, all_labels = [], []
    optimizer.zero_grad()

    for batch_idx, data in enumerate(train_loader):
        data = data.to(device, non_blocking=True)  # non_blocking for pin_memory speed

        with autocast(device_type='cuda'):
            out, _ = model(data)
            loss = criterion(out.squeeze(), data.y.float()) / ACCUMULATION_STEPS

        scaler.scale(loss).backward()

        if (batch_idx + 1) % ACCUMULATION_STEPS == 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            if scheduler is not None:
                scheduler.step()

        total_loss += loss.item() * ACCUMULATION_STEPS
        all_preds.append(torch.sigmoid(out.squeeze()).detach())
        all_labels.append(data.y)

    metrics = compute_metrics(torch.cat(all_preds), torch.cat(all_labels))
    metrics['loss'] = total_loss / len(train_loader)
    return metrics


@torch.no_grad()
def evaluate(model, data_loader, criterion, device):
    model.eval()
    total_loss = 0
    all_preds, all_labels = [], []

    for data in data_loader:
        data = data.to(device, non_blocking=True)
        with autocast(device_type='cuda'):
            out, _ = model(data)
            loss = criterion(out.squeeze(), data.y.float())
        total_loss += loss.item()
        all_preds.append(torch.sigmoid(out.squeeze()))
        all_labels.append(data.y)

    metrics = compute_metrics(torch.cat(all_preds), torch.cat(all_labels))
    metrics['loss'] = total_loss / len(data_loader)
    return metrics


def main():
    print("=" * 70)
    print("DEMO FAST TRAIN — fold 0 subset, 3 epochs")
    print("=" * 70)
    print(f"System: RTX 2060 | i7-10700 (16 cores) | 32GB RAM")
    print(f"Config: batch={BATCH_SIZE}, workers={NUM_WORKERS}, "
          f"accum={ACCUMULATION_STEPS}, epochs={DEMO_EPOCHS}")
    print()

    set_seed_fast(42)
    config = get_config()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    print()

    # ── Load fold 0 JSON ──────────────────────────────────────────────────────
    fold_path = Path(config.data_cv_folds) / 'fold_0.json'
    print(f"Loading fold data from: {fold_path}")
    with open(fold_path) as f:
        fold_data = json.load(f)

    # Subset: first N patients only
    def subset_graphs(graph_files, n_patients):
        """Return graph files for the first n_patients only."""
        from collections import OrderedDict
        patient_map = OrderedDict()
        for gf in sorted(graph_files):
            pid = Path(gf).parent.name
            if pid not in patient_map:
                patient_map[pid] = []
            patient_map[pid].append(gf)
            if len(patient_map) >= n_patients:
                break
        return [gf for files in patient_map.values() for gf in files]

    train_graphs = subset_graphs(fold_data['train_graphs'], DEMO_TRAIN_PATIENTS)
    val_graphs   = subset_graphs(fold_data['val_graphs'],   DEMO_VAL_PATIENTS)

    print(f"Demo subset: {len(train_graphs)} train graph files / "
          f"{len(val_graphs)} val graph files")

    # ── Build datasets ────────────────────────────────────────────────────────
    binary_transform = BinaryTransform()
    t0 = time.time()

    train_dataset = BraTSGraphDataset(
        root_dir=None, split='train',
        graph_files=train_graphs,
        transform=binary_transform
    )
    val_dataset = BraTSGraphDataset(
        root_dir=None, split='val',
        graph_files=val_graphs,
        transform=binary_transform
    )
    load_time = time.time() - t0
    print(f"Data loaded into RAM in {load_time:.1f}s")
    print(f"  Train: {len(train_dataset)} graphs")
    print(f"  Val:   {len(val_dataset)} graphs")
    print()

    # ── DataLoaders with optimized settings ───────────────────────────────────
    loader_kwargs = dict(
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        persistent_workers=(NUM_WORKERS > 0),
        prefetch_factor=2 if NUM_WORKERS > 0 else None,
    )
    train_loader = GeometricDataLoader(train_dataset, shuffle=True,  **loader_kwargs)
    val_loader   = GeometricDataLoader(val_dataset,   shuffle=False, **loader_kwargs)

    # ── Model ─────────────────────────────────────────────────────────────────
    sample = next(iter(train_loader))
    in_channels = sample.x.size(1)

    model = TumorSegmentationGNN(
        in_channels=in_channels,
        hidden_channels=HIDDEN_CHANNELS,
        num_layers=NUM_LAYERS,
        dropout=0.1
    ).to(device)

    # torch.compile — fuses ops, reduces Python overhead (~15-25% faster)
    try:
        # 'default' mode — operator fusion without CUDA Graphs.
        # 'reduce-overhead' (CUDA Graphs) fails with PyG because graph sizes
        # are dynamic (variable nodes/edges per batch).
        model = torch.compile(model, mode='default')
        print("torch.compile() enabled (default mode)")
    except Exception as e:
        print(f"torch.compile() not available: {e}")

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {total_params:,}")
    print()

    # ── Optimizer / Scheduler / Scaler ────────────────────────────────────────
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.01)
    total_steps = len(train_loader) * DEMO_EPOCHS // ACCUMULATION_STEPS
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=LEARNING_RATE, total_steps=total_steps,
        pct_start=0.1, anneal_strategy='cos'
    )
    scaler = GradScaler(device='cuda')

    # ── Training loop ─────────────────────────────────────────────────────────
    print(f"Starting {DEMO_EPOCHS}-epoch demo...")
    print("-" * 70)

    epoch_times = []
    overall_start = time.time()

    for epoch in range(DEMO_EPOCHS):
        t_ep = time.time()
        train_m = train_epoch(model, train_loader, criterion, optimizer, scheduler, scaler, device)
        val_m   = evaluate(model, val_loader, criterion, device)
        ep_time = time.time() - t_ep
        epoch_times.append(ep_time)

        print(f"Epoch {epoch+1}/{DEMO_EPOCHS}  ({ep_time:.1f}s)")
        print(f"  Train  loss={train_m['loss']:.4f}  dice={train_m['dice']:.4f}  acc={train_m['accuracy']:.4f}")
        print(f"  Val    loss={val_m['loss']:.4f}  dice={val_m['dice']:.4f}  acc={val_m['accuracy']:.4f}")

    total_time = time.time() - overall_start
    avg_epoch  = np.mean(epoch_times)

    print("-" * 70)
    print(f"\nDemo complete!")
    print(f"  Total time:        {total_time:.1f}s")
    print(f"  Avg epoch time:    {avg_epoch:.1f}s")
    print()

    # ── Extrapolate to full CV ────────────────────────────────────────────────
    # Scale factor: full fold has 61K train graphs vs demo ~700-800
    full_train_graphs = len(fold_data['train_graphs'])
    demo_train_graphs = len(train_dataset)
    scale_factor = full_train_graphs / demo_train_graphs
    est_epoch_full = avg_epoch * scale_factor
    est_fold_full  = est_epoch_full * 50          # 50 epochs
    est_cv_full    = est_fold_full * 5            # 5 folds

    print(f"  Estimated times for FULL training:")
    print(f"    1 epoch  (full fold)  ~{est_epoch_full/60:.1f} min")
    print(f"    1 fold   (50 epochs)  ~{est_fold_full/3600:.1f} hours")
    print(f"    All 5 folds           ~{est_cv_full/3600:.1f} hours")
    print()
    print("If these estimates look good, run the full training with:")
    print("  bash train_all_folds.sh")
    print("=" * 70)


if __name__ == "__main__":
    main()
