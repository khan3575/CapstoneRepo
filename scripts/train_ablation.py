#!/usr/bin/env python3
"""
Phase 3 — Ablation Study Training Wrapper
==========================================
Runs individual ablation experiments faster than full training by:
  - Using fold 0 only (not all 5 folds)
  - 30 epochs max (not 50)
  - cudnn.benchmark=True, deterministic=False (2-3x faster)
  - Saves to checkpoints/ablation/<variant_name>/

Expected time: ~35-40 min per variant on RTX 2060.

Usage:
    python scripts/train_ablation.py --variant baseline
    python scripts/train_ablation.py --variant deeper_network
    python scripts/train_ablation.py --variant wider_network
    python scripts/train_ablation.py --variant gat_architecture
    python scripts/train_ablation.py --variant batch_16
    python scripts/train_ablation.py --variant batch_24
    python scripts/train_ablation.py --variant batch_48
    python scripts/train_ablation.py --variant batch_64
    python scripts/train_ablation.py --variant superpixels_50
    python scripts/train_ablation.py --variant superpixels_80
    python scripts/train_ablation.py --variant superpixels_100
    python scripts/train_ablation.py --variant superpixels_150
    python scripts/train_ablation.py --variant superpixels_200
    python scripts/train_ablation.py --variant slices_100
    python scripts/train_ablation.py --variant slices_150
    python scripts/train_ablation.py --variant slices_200
    python scripts/train_ablation.py --variant window_1
    python scripts/train_ablation.py --variant window_3
    python scripts/train_ablation.py --variant window_5

Run all variants sequentially:
    for v in baseline deeper_network wider_network gat_architecture \\
              batch_16 batch_24 batch_48 batch_64 \\
              superpixels_50 superpixels_80 superpixels_100 superpixels_150 superpixels_200 \\
              slices_100 slices_150 slices_200 \\
              window_1 window_3 window_5; do
        python scripts/train_ablation.py --variant $v
    done
"""

import sys
import json
import argparse
import torch
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))


# =============================================================================
# Variant Definitions
# =============================================================================

VARIANTS = {
    # ── Architecture variants ──────────────────────────────────────────────────
    'baseline': {
        'description': 'Baseline: 5 layers, 256 dim, batch 32, GraphSAGE',
        'hidden_channels': 256,
        'num_layers': 5,
        'batch_size': 32,
        'architecture': 'graphsage',
        'graph_variant': None,  # Uses pre-built 200-superpixel, 200-slice graphs
    },
    'layers_3': {
        'description': '3 layers, 256 dim, batch 32, GraphSAGE',
        'hidden_channels': 256,
        'num_layers': 3,
        'batch_size': 32,
        'architecture': 'graphsage',
        'graph_variant': None,
    },
    'layers_4': {
        'description': '4 layers, 256 dim, batch 32, GraphSAGE',
        'hidden_channels': 256,
        'num_layers': 4,
        'batch_size': 32,
        'architecture': 'graphsage',
        'graph_variant': None,
    },
    'deeper_network': {
        'description': '6 layers, 256 dim, batch 32, GraphSAGE',
        'hidden_channels': 256,
        'num_layers': 6,
        'batch_size': 32,
        'architecture': 'graphsage',
        'graph_variant': None,
    },
    'hidden_128': {
        'description': '5 layers, 128 dim, batch 32, GraphSAGE',
        'hidden_channels': 128,
        'num_layers': 5,
        'batch_size': 32,
        'architecture': 'graphsage',
        'graph_variant': None,
    },
    'wider_network': {
        'description': '5 layers, 512 dim, batch 32, GraphSAGE',
        'hidden_channels': 512,
        'num_layers': 5,
        'batch_size': 32,
        'architecture': 'graphsage',
        'graph_variant': None,
    },
    'gat_architecture': {
        'description': '5 layers, 256 dim, batch 32, GAT (replace GraphSAGE)',
        'hidden_channels': 256,
        'num_layers': 5,
        'batch_size': 32,
        'architecture': 'gat',
        'graph_variant': None,
    },
    'gcn_architecture': {
        'description': '5 layers, 256 dim, batch 32, GCN (simple normalized conv)',
        'hidden_channels': 256,
        'num_layers': 5,
        'batch_size': 32,
        'architecture': 'gcn',
        'graph_variant': None,
    },
    'gin_architecture': {
        'description': '5 layers, 256 dim, batch 32, GIN (most expressive)',
        'hidden_channels': 256,
        'num_layers': 5,
        'batch_size': 32,
        'architecture': 'gin',
        'graph_variant': None,
    },
    'graph_transformer': {
        'description': '5 layers, 256 dim, batch 32, Graph Transformer (full attention)',
        'hidden_channels': 256,
        'num_layers': 5,
        'batch_size': 32,
        'architecture': 'graph_transformer',
        'graph_variant': None,
    },

    # ── Batch size variants ───────────────────────────────────────────────────
    'batch_16': {
        'description': '5 layers, 256 dim, batch 16, GraphSAGE',
        'hidden_channels': 256,
        'num_layers': 5,
        'batch_size': 16,
        'architecture': 'graphsage',
        'graph_variant': None,
    },
    'batch_24': {
        'description': '5 layers, 256 dim, batch 24, GraphSAGE',
        'hidden_channels': 256,
        'num_layers': 5,
        'batch_size': 24,
        'architecture': 'graphsage',
        'graph_variant': None,
    },
    'batch_48': {
        'description': '5 layers, 256 dim, batch 48, GraphSAGE',
        'hidden_channels': 256,
        'num_layers': 5,
        'batch_size': 48,
        'architecture': 'graphsage',
        'graph_variant': None,
    },
    'batch_64': {
        'description': '5 layers, 256 dim, batch 64, GraphSAGE',
        'hidden_channels': 256,
        'num_layers': 5,
        'batch_size': 64,
        'architecture': 'graphsage',
        'graph_variant': None,
    },

    # ── Superpixel count variants ─────────────────────────────────────────────
    # NOTE: These require separately pre-built graphs with different superpixel counts.
    # Graph files must exist at: data/graphs_sp{N}/PATIENT_ID/PATIENT_ID_graphs_{N}.pt
    # Build them with: python src/graph_construction.py --n_superpixels N
    'superpixels_50': {
        'description': '50 superpixels/slice (smaller graphs)',
        'hidden_channels': 256,
        'num_layers': 5,
        'batch_size': 32,
        'architecture': 'graphsage',
        'graph_variant': 'sp50',  # Expects data/graphs_sp50/
        'n_superpixels': 50,
    },
    'superpixels_80': {
        'description': '80 superpixels/slice',
        'hidden_channels': 256,
        'num_layers': 5,
        'batch_size': 32,
        'architecture': 'graphsage',
        'graph_variant': 'sp80',
        'n_superpixels': 80,
    },
    'superpixels_100': {
        'description': '100 superpixels/slice',
        'hidden_channels': 256,
        'num_layers': 5,
        'batch_size': 32,
        'architecture': 'graphsage',
        'graph_variant': 'sp100',
        'n_superpixels': 100,
    },
    'superpixels_150': {
        'description': '150 superpixels/slice',
        'hidden_channels': 256,
        'num_layers': 5,
        'batch_size': 32,
        'architecture': 'graphsage',
        'graph_variant': 'sp150',
        'n_superpixels': 150,
    },
    'superpixels_200': {
        'description': '200 superpixels/slice (current default, fixed not adaptive)',
        'hidden_channels': 256,
        'num_layers': 5,
        'batch_size': 32,
        'architecture': 'graphsage',
        'graph_variant': 'sp200',
        'n_superpixels': 200,
    },

    # ── Slice count variants ──────────────────────────────────────────────────
    # NOTE: These require separately pre-built graphs with different slice counts.
    # Graph files must exist at: data/graphs_slices{N}/...
    'slices_100': {
        'description': '100 slices/patient',
        'hidden_channels': 256,
        'num_layers': 5,
        'batch_size': 32,
        'architecture': 'graphsage',
        'graph_variant': 'slices100',
        'n_slices': 100,
    },
    'slices_150': {
        'description': '150 slices/patient',
        'hidden_channels': 256,
        'num_layers': 5,
        'batch_size': 32,
        'architecture': 'graphsage',
        'graph_variant': 'slices150',
        'n_slices': 150,
    },
    'slices_200': {
        'description': '200 slices/patient (current default)',
        'hidden_channels': 256,
        'num_layers': 5,
        'batch_size': 32,
        'architecture': 'graphsage',
        'graph_variant': None,  # Uses existing graphs
    },

    # ── Inter-slice window size variants ──────────────────────────────────────
    # NOTE: These require separately pre-built graphs with different window sizes.
    'window_1': {
        'description': '1-slice window (no inter-slice edges)',
        'hidden_channels': 256,
        'num_layers': 5,
        'batch_size': 32,
        'architecture': 'graphsage',
        'graph_variant': 'win1',
        'window_size': 1,
    },
    'window_3': {
        'description': '3-slice window (current default)',
        'hidden_channels': 256,
        'num_layers': 5,
        'batch_size': 32,
        'architecture': 'graphsage',
        'graph_variant': None,  # Uses existing graphs
    },
    'window_5': {
        'description': '5-slice window (wider context)',
        'hidden_channels': 256,
        'num_layers': 5,
        'batch_size': 32,
        'architecture': 'graphsage',
        'graph_variant': 'win5',
        'window_size': 5,
    },
}


def ablation_set_seed(seed: int = 42):
    """
    Seed for ablation — uses cudnn.benchmark=True (NOT deterministic).
    This is 2-3x faster than the main training seed, acceptable for ablation.
    """
    import random
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = False   # ← faster
        torch.backends.cudnn.benchmark = True        # ← faster


def get_graph_files_for_variant(variant_config, cv_pool_patients, project_root):
    """
    Return list of graph file paths for the given variant.
    Falls back to standard data/graphs/ if no graph_variant specified.
    """
    graph_variant = variant_config.get('graph_variant')

    if graph_variant is None:
        # Use standard pre-built graphs
        graphs_dir = project_root / 'data' / 'graphs'
        graph_files = [
            str(graphs_dir / pid / f'{pid}_graphs_200.pt')
            for pid in cv_pool_patients
        ]
    else:
        # Use variant-specific graphs
        graphs_dir = project_root / 'data' / f'graphs_{graph_variant}'
        if not graphs_dir.exists():
            print(f"\nERROR: Graph directory not found: {graphs_dir}")
            print(f"You need to pre-build graphs for this variant first:")
            if 'n_superpixels' in variant_config:
                n = variant_config['n_superpixels']
                print(f"  python src/graph_construction.py --n_superpixels {n} --output data/graphs_{graph_variant}/")
            elif 'n_slices' in variant_config:
                n = variant_config['n_slices']
                print(f"  python src/graph_construction.py --n_slices {n} --output data/graphs_{graph_variant}/")
            elif 'window_size' in variant_config:
                n = variant_config['window_size']
                print(f"  python src/graph_construction.py --window_size {n} --output data/graphs_{graph_variant}/")
            return None

        # Find the actual .pt file naming convention
        sample_pid = cv_pool_patients[0]
        candidate_files = list((graphs_dir / sample_pid).glob('*.pt')) if (graphs_dir / sample_pid).exists() else []
        if not candidate_files:
            print(f"\nERROR: No .pt files found in {graphs_dir / sample_pid}")
            return None
        pt_suffix = candidate_files[0].name.replace(sample_pid, '')

        graph_files = [
            str(graphs_dir / pid / f'{pid}{pt_suffix}')
            for pid in cv_pool_patients
        ]

    # Filter to existing files only
    existing = [f for f in graph_files if Path(f).exists()]
    if len(existing) < len(graph_files):
        print(f"  Warning: {len(graph_files) - len(existing)} graph files missing "
              f"({len(existing)}/{len(graph_files)} found)")

    return existing


def main():
    parser = argparse.ArgumentParser(description='Phase 3 ablation study runner')
    parser.add_argument('--variant', type=str, required=True,
                        choices=list(VARIANTS.keys()),
                        help='Which ablation variant to run')
    parser.add_argument('--fold', type=int, default=0,
                        help='Fold to use (default: 0 — ablation uses fold 0 only)')
    parser.add_argument('--epochs', type=int, default=30,
                        help='Max training epochs (default: 30 for speed)')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device: cuda or cpu (default: cuda)')
    parser.add_argument('--output_root', type=str, default='checkpoints/ablation',
                        help='Root for ablation checkpoints (default: checkpoints/ablation)')
    parser.add_argument('--list', action='store_true',
                        help='List all available variants and exit')
    parser.add_argument('--demo', action='store_true',
                        help='Demo mode: 3 epochs, 10 train patients — verify no errors before full run')
    args = parser.parse_args()

    if args.list:
        print("Available ablation variants:")
        for name, cfg in VARIANTS.items():
            print(f"  {name:25s}  {cfg['description']}")
        return

    project_root = Path(__file__).parent.parent
    sys.path.insert(0, str(project_root / 'src'))

    from config import get_config
    from gnn_model import TumorSegmentationGNN
    from dataset import BraTSGraphDataset, BinaryTransform
    from cross_validation import load_fold_data

    config = get_config()
    variant_config = VARIANTS[args.variant]

    if args.demo:
        args.epochs = 3
        print("*** DEMO MODE: 3 epochs, 10 train patients ***")

    print("=" * 70)
    print(f"ABLATION: {args.variant}")
    print(f"  {variant_config['description']}")
    print(f"  Fold: {args.fold}, Epochs: {args.epochs}")
    if args.demo:
        print(f"  [DEMO MODE]")
    print("=" * 70)

    # Apply faster seed (cudnn.benchmark=True)
    ablation_set_seed(config.get('project.seed', 42))

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Load fold 0 data (ablation always uses fold 0 for consistency)
    fold_data = load_fold_data(config.data_cv_folds, args.fold)
    print(f"\nFold {args.fold}: {len(fold_data['train_patients'])} train, "
          f"{len(fold_data['val_patients'])} val, "
          f"{len(fold_data['test_patients'])} test patients")

    # Determine graph files
    graph_variant = variant_config.get('graph_variant')
    if graph_variant is None:
        train_graphs = fold_data['train_graphs']
        val_graphs   = fold_data['val_graphs']
        test_graphs  = fold_data['test_graphs']

        # Demo mode: subset to first 10 train patients, 5 val/test
        if args.demo:
            from collections import OrderedDict
            def _subset(graph_files, n_patients):
                pm = OrderedDict()
                for gf in sorted(graph_files):
                    from pathlib import Path as _P
                    pid = _P(gf).parent.name
                    if pid not in pm:
                        pm[pid] = []
                    pm[pid].append(gf)
                    if len(pm) >= n_patients:
                        break
                return [f for files in pm.values() for f in files]
            train_graphs = _subset(train_graphs, 10)
            val_graphs   = _subset(val_graphs, 5)
            test_graphs  = _subset(test_graphs, 5)
    else:
        # Need variant-specific graphs
        graphs_dir = project_root / 'data' / f'graphs_{graph_variant}'
        if not graphs_dir.exists():
            print(f"\nERROR: Graph directory not found: {graphs_dir}")
            print(f"Pre-build graphs for this variant first.")
            return

        def get_graphs(patients):
            files = []
            for pid in patients:
                pt_files = list((graphs_dir / pid).glob('*.pt')) if (graphs_dir / pid).exists() else []
                files.extend([str(f) for f in pt_files])
            return files

        train_graphs = get_graphs(fold_data['train_patients'])
        val_graphs = get_graphs(fold_data['val_patients'])
        test_graphs = get_graphs(fold_data['test_patients'])
        print(f"Using variant graphs: {graphs_dir}")
        print(f"  Train graphs: {len(train_graphs)}, Val: {len(val_graphs)}, Test: {len(test_graphs)}")

    # Create datasets
    binary_transform = BinaryTransform()

    from torch_geometric.loader import DataLoader as GeometricDataLoader

    train_dataset = BraTSGraphDataset(root_dir=None, split='train',
                                       graph_files=train_graphs, transform=binary_transform)
    val_dataset = BraTSGraphDataset(root_dir=None, split='val',
                                     graph_files=val_graphs, transform=binary_transform)
    test_dataset = BraTSGraphDataset(root_dir=None, split='test',
                                      graph_files=test_graphs, transform=binary_transform)

    batch_size = variant_config['batch_size']
    loader_kwargs = dict(
        batch_size=batch_size, num_workers=4, pin_memory=True,
        persistent_workers=True, prefetch_factor=2,
    )
    train_loader = GeometricDataLoader(train_dataset, shuffle=True,  **loader_kwargs)
    val_loader   = GeometricDataLoader(val_dataset,   shuffle=False, **loader_kwargs)
    test_loader  = GeometricDataLoader(test_dataset,  shuffle=False, **loader_kwargs)

    print(f"\nDatasets: train={len(train_dataset)}, val={len(val_dataset)}, test={len(test_dataset)}")

    # Create model
    architecture = variant_config.get('architecture', 'graphsage')
    hidden_channels = variant_config['hidden_channels']
    num_layers = variant_config['num_layers']

    sample = next(iter(train_loader))
    in_channels = sample.x.size(1)

    # gnn_type: 'sage' for GraphSAGE, 'gat' for GAT
    gnn_type = 'gat' if architecture == 'gat' else 'sage'
    model = TumorSegmentationGNN(
        in_channels=in_channels,
        hidden_channels=hidden_channels,
        num_layers=num_layers,
        dropout=0.1,
        gnn_type=gnn_type,
    ).to(device)

    # torch.compile — operator fusion, ~15-25% speedup (default mode for PyG dynamic graphs)
    try:
        model = torch.compile(model, mode='default')
        print(f"  torch.compile() enabled")
    except Exception as e:
        print(f"  torch.compile() skipped: {e}")

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model: {architecture}, {num_layers} layers, {hidden_channels} hidden, {total_params:,} params")

    # Output directory
    output_dir = project_root / args.output_root / args.variant
    output_dir.mkdir(parents=True, exist_ok=True)

    # Train using the same loop as train_cv_fold.py
    import time
    import numpy as np
    import torch.nn as nn
    from torch.amp import autocast, GradScaler

    criterion = nn.BCEWithLogitsLoss()
    lr = config.get('model.training.learning_rate', 0.001)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)

    accumulation_steps = config.get('model.training.accumulation_steps', 4)
    total_steps = len(train_loader) * args.epochs // accumulation_steps
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=lr, total_steps=total_steps, pct_start=0.1, anneal_strategy='cos'
    )
    scaler = GradScaler(device='cuda')

    best_val_dice = 0.0
    best_checkpoint = None
    history = []
    start_time = time.time()

    print(f"\nTraining for {args.epochs} epochs (ablation mode: cudnn.benchmark=True)...")

    for epoch in range(args.epochs):
        # Train
        model.train()
        total_loss = 0
        all_preds, all_labels = [], []
        optimizer.zero_grad()

        for batch_idx, data in enumerate(train_loader):
            data = data.to(device, non_blocking=True)
            with autocast(device_type='cuda'):
                out, _ = model(data)
                loss = criterion(out.squeeze(), data.y.float()) / accumulation_steps
            scaler.scale(loss).backward()
            if (batch_idx + 1) % accumulation_steps == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                scheduler.step()
            total_loss += loss.item() * accumulation_steps
            all_preds.append(torch.sigmoid(out.squeeze()).detach().cpu())
            all_labels.append(data.y.cpu())

        all_preds = torch.cat(all_preds).numpy()
        all_labels = torch.cat(all_labels).numpy()
        pred_bin = (all_preds > 0.5).astype(int)
        tp = ((pred_bin == 1) & (all_labels == 1)).sum()
        fp = ((pred_bin == 1) & (all_labels == 0)).sum()
        fn = ((pred_bin == 0) & (all_labels == 1)).sum()
        train_dice = (2 * tp) / (2 * tp + fp + fn + 1e-7)
        train_loss = total_loss / len(train_loader)

        # Validate
        model.eval()
        val_preds, val_labels = [], []
        with torch.no_grad():
            for data in val_loader:
                data = data.to(device, non_blocking=True)
                with autocast(device_type='cuda'):
                    out, _ = model(data)
                val_preds.append(torch.sigmoid(out.squeeze()).cpu())
                val_labels.append(data.y.cpu())

        val_preds = torch.cat(val_preds).numpy()
        val_labels = torch.cat(val_labels).numpy()
        pred_bin = (val_preds > 0.5).astype(int)
        tp = ((pred_bin == 1) & (val_labels == 1)).sum()
        fp = ((pred_bin == 1) & (val_labels == 0)).sum()
        fn = ((pred_bin == 0) & (val_labels == 1)).sum()
        val_dice = float((2 * tp) / (2 * tp + fp + fn + 1e-7))

        elapsed = time.time() - start_time
        print(f"Epoch [{epoch+1:2d}/{args.epochs}] "
              f"loss={train_loss:.4f} train_dice={train_dice:.4f} "
              f"val_dice={val_dice:.4f} [{elapsed/60:.1f}min]")

        history.append({'epoch': epoch + 1, 'train_loss': train_loss,
                        'train_dice': float(train_dice), 'val_dice': val_dice})

        if val_dice > best_val_dice:
            best_val_dice = val_dice
            best_checkpoint = {
                'epoch': epoch + 1,
                'model_state_dict': {k: v.clone() for k, v in model.state_dict().items()},
                'val_dice': val_dice,
                'variant': args.variant,
                'variant_config': variant_config,
            }
            torch.save(best_checkpoint, output_dir / 'best_model.pth')

    # Test evaluation
    if best_checkpoint is None:
        # Val dice never improved (can happen with tiny demo subsets); use final state
        best_checkpoint = {
            'model_state_dict': {k: v.clone() for k, v in model.state_dict().items()},
            'val_dice': 0.0, 'variant': args.variant, 'variant_config': variant_config,
        }
        torch.save(best_checkpoint, output_dir / 'best_model.pth')
    model.load_state_dict(best_checkpoint['model_state_dict'])
    model.eval()
    test_preds, test_labels = [], []
    with torch.no_grad():
        for data in test_loader:
            data = data.to(device, non_blocking=True)
            with autocast(device_type='cuda'):
                out, _ = model(data)
            test_preds.append(torch.sigmoid(out.squeeze()).cpu())
            test_labels.append(data.y.cpu())

    test_preds = torch.cat(test_preds).numpy()
    test_labels = torch.cat(test_labels).numpy()
    pred_bin = (test_preds > 0.5).astype(int)
    tp = ((pred_bin == 1) & (test_labels == 1)).sum()
    tn = ((pred_bin == 0) & (test_labels == 0)).sum()
    fp = ((pred_bin == 1) & (test_labels == 0)).sum()
    fn = ((pred_bin == 0) & (test_labels == 1)).sum()
    eps = 1e-7
    test_metrics = {
        'dice':        float((2 * tp) / (2 * tp + fp + fn + eps)),
        'accuracy':    float((tp + tn) / (tp + tn + fp + fn + eps)),
        'sensitivity': float(tp / (tp + fn + eps)),
        'specificity': float(tn / (tn + fp + eps)),
        'precision':   float(tp / (tp + fp + eps)),
    }

    total_time_min = (time.time() - start_time) / 60
    print(f"\nTest results for {args.variant}:")
    for k, v in test_metrics.items():
        print(f"  {k:12s}: {v:.4f}")
    print(f"\nBest val Dice: {best_val_dice:.4f}")
    print(f"Total time: {total_time_min:.1f} min")

    # Save results
    results = {
        'variant': args.variant,
        'variant_config': {k: v for k, v in variant_config.items() if k != 'graph_variant'},
        'fold': args.fold,
        'epochs_trained': args.epochs,
        'best_val_dice': best_val_dice,
        'test_metrics': test_metrics,
        'training_history': history,
        'total_time_min': total_time_min,
    }

    import json
    with open(output_dir / 'results.json', 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nSaved to: {output_dir}")
    print(f"  best_model.pth")
    print(f"  results.json")


if __name__ == '__main__':
    main()
