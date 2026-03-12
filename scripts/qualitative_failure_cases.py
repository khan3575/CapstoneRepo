#!/usr/bin/env python3
"""
Phase 6.2: Qualitative failure case visualization.

Selects worst (3), median (1), best (1) patients from the 251-patient held-out set,
then generates 4-panel figures: T1ce MRI | GT overlay | Prediction overlay | Comparison.

Pixel-level predictions are reconstructed by re-running SLIC on the selected slice
and mapping per-superpixel GNN outputs back to pixels.

Output: research_results/failure_cases/
Run:    python scripts/qualitative_failure_cases.py --device cuda
"""

import os
import sys
import json
import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
from pathlib import Path
from torch_geometric.data import Data

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))
from config import get_config
from gnn_model import TumorSegmentationGNN
from graph_construction import GraphBuilder
from graph_construction import Config as GraphConfig


# ──────────────────────────────────────────────────────────────────────────────
# Model loading
# ──────────────────────────────────────────────────────────────────────────────

def load_ensemble(config, device):
    ckpt_dir = Path(config.checkpoints_binary)
    models = []
    for fold in range(5):
        ckpt_path = ckpt_dir / f'fold_{fold}' / 'best_model.pth'
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        model = TumorSegmentationGNN(
            in_channels=ckpt.get('in_channels', 15),
            hidden_channels=ckpt.get('hidden_channels', 256),
            num_layers=ckpt.get('num_layers', 5),
            dropout=ckpt.get('dropout', 0.1),
        ).to(device)
        state_dict = ckpt['model_state_dict']
        if any(k.startswith('_orig_mod.') for k in state_dict):
            state_dict = {k.replace('_orig_mod.', '', 1): v for k, v in state_dict.items()}
        model.load_state_dict(state_dict)
        model.eval()
        models.append(model)
        print(f'  Fold {fold}: loaded (val Dice={ckpt["val_dice"]:.4f})')
    return models


# ──────────────────────────────────────────────────────────────────────────────
# Per-patient Dice using pre-built graphs (fast — no SLIC)
# ──────────────────────────────────────────────────────────────────────────────

def compute_patient_dice(models, patient_id, graph_dir, device):
    """Return ensemble Dice for one patient using pre-built graphs."""
    pt_file = graph_dir / patient_id / f'{patient_id}_graphs_200.pt'
    if not pt_file.exists():
        return None
    graphs = torch.load(pt_file, weights_only=False)
    if not graphs:
        return None

    all_preds, all_labels = [], []
    with torch.no_grad():
        for g in graphs:
            g = g.to(device)
            logits = torch.stack([m(g)[0] for m in models]).mean(0)
            preds = (torch.sigmoid(logits) > 0.5).float()
            labels = (g.y > 0).float()
            all_preds.append(preds.cpu())
            all_labels.append(labels.cpu())

    p = torch.cat(all_preds)
    l = torch.cat(all_labels)
    intersection = (p * l).sum()
    union = p.sum() + l.sum()
    if union == 0:
        return 1.0
    return float((2 * intersection + 1e-7) / (union + 1e-7))


# ──────────────────────────────────────────────────────────────────────────────
# Pixel-level reconstruction for a single slice (re-runs SLIC)
# ──────────────────────────────────────────────────────────────────────────────

def predict_slice_pixels(models, slice_data, device):
    """
    Re-run SLIC on one slice, build a single-slice graph, run ensemble inference,
    then map per-superpixel predictions back to a pixel mask.

    Returns: (segments_2d, pred_prob_map)  both shape (H, W)
    """
    graph_config = GraphConfig()  # use defaults: 200 superpixels
    gb = GraphBuilder(graph_config)
    sp = gb.slice_processor

    segments = sp.compute_superpixels(slice_data)
    unique_labels = np.unique(segments)
    unique_labels = unique_labels[unique_labels > 0]

    if len(unique_labels) == 0:
        return segments, np.zeros(segments.shape, dtype=np.float32)

    features, _, masks, _ = sp.compute_features(slice_data, segments)
    if len(features) == 0:
        return segments, np.zeros(segments.shape, dtype=np.float32)

    # Build edge index from adjacency
    edge_list = gb.edge_builder.build_adjacency_edges(segments)
    if edge_list:
        edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
    else:
        edge_index = torch.zeros((2, 0), dtype=torch.long)

    labels = gb._compute_labels_from_masks(masks, slice_data)
    x = torch.tensor(features, dtype=torch.float32)
    y = torch.tensor(labels, dtype=torch.float32)
    g = Data(x=x, edge_index=edge_index, y=y).to(device)

    with torch.no_grad():
        logits = torch.stack([m(g)[0] for m in models]).mean(0)
        probs = torch.sigmoid(logits).cpu().numpy()

    # Map superpixel probs → pixel mask
    pred_pixel = np.zeros(segments.shape, dtype=np.float32)
    for i, seg_id in enumerate(unique_labels):
        if i < len(probs):
            pred_pixel[segments == seg_id] = probs[i]

    return segments, pred_pixel


# ──────────────────────────────────────────────────────────────────────────────
# Figure generation
# ──────────────────────────────────────────────────────────────────────────────

def make_figure(t1ce_slice, gt_slice, pred_prob, patient_id, slice_z,
                patient_dice_score, save_path, label):
    """4-panel figure: T1ce | GT overlay | Pred overlay | Comparison."""
    # Normalise T1ce to [0,1]
    vmin, vmax = t1ce_slice.min(), t1ce_slice.max()
    mri = (t1ce_slice - vmin) / (vmax - vmin + 1e-8)

    gt_mask = gt_slice > 0
    pred_mask = pred_prob > 0.5

    fig, axes = plt.subplots(1, 4, figsize=(22, 5.5))
    fig.patch.set_facecolor('black')
    for ax in axes:
        ax.set_facecolor('black')

    # 1 — Original MRI
    axes[0].imshow(mri, cmap='gray', vmin=0, vmax=1)
    axes[0].set_title('T1ce MRI', color='white', fontsize=13, fontweight='bold')
    axes[0].axis('off')

    # 2 — Ground truth overlay
    axes[1].imshow(mri, cmap='gray', vmin=0, vmax=1)
    if gt_mask.any():
        axes[1].imshow(np.ma.masked_where(~gt_mask, gt_mask.astype(float)),
                       cmap='Reds', alpha=0.55, vmin=0, vmax=1)
    axes[1].set_title('Ground Truth', color='tomato', fontsize=13, fontweight='bold')
    axes[1].axis('off')

    # 3 — Prediction overlay
    axes[2].imshow(mri, cmap='gray', vmin=0, vmax=1)
    if pred_mask.any():
        axes[2].imshow(np.ma.masked_where(~pred_mask, pred_prob),
                       cmap='Blues', alpha=0.55, vmin=0, vmax=1)
    axes[2].set_title('GNN Prediction', color='cornflowerblue', fontsize=13, fontweight='bold')
    axes[2].axis('off')

    # 4 — Comparison (GT=red, Pred=blue, overlap=purple)
    axes[3].imshow(mri, cmap='gray', vmin=0, vmax=1)
    overlay = np.zeros((*mri.shape, 3))
    overlay[gt_mask, 0] = 0.85
    overlay[pred_mask, 2] = 0.85
    axes[3].imshow(overlay, alpha=0.6)
    axes[3].set_title('Comparison\n(Red=GT  Blue=Pred  Purple=Match)',
                      color='white', fontsize=11, fontweight='bold')
    axes[3].axis('off')

    dice_pct = patient_dice_score * 100
    fig.suptitle(
        f'[{label.upper()}]  {patient_id}  —  Slice {slice_z}  |  Patient Dice: {dice_pct:.1f}%',
        color='white', fontsize=14, fontweight='bold', y=1.01
    )
    plt.tight_layout(pad=0.5)
    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='black')
    plt.close()


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='Phase 6.2: qualitative failure cases')
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--demo', action='store_true', help='Run on 20 patients only (for quick testing)')
    args = parser.parse_args()

    config = get_config()
    device = torch.device(args.device)

    graph_dir = Path(config.data_graphs)
    preprocessed_dir = Path('data/preprocessed')
    output_dir = Path('research_results/failure_cases')
    output_dir.mkdir(parents=True, exist_ok=True)

    print('=' * 70)
    print('PHASE 6.2 — QUALITATIVE FAILURE CASES')
    print('=' * 70)
    print(f'Device: {device}')

    # Load ensemble
    print('\nLoading ensemble models...')
    models = load_ensemble(config, device)
    print(f'Loaded {len(models)}/5 models\n')

    # Load held-out patient list
    with open('data/splits/held_out_test.json') as f:
        held_out = json.load(f)
    patients = held_out['patients']
    if args.demo:
        patients = patients[:20]
        print(f'[DEMO MODE: {len(patients)} patients]\n')

    # ── Step 1: compute per-patient Dice ──────────────────────────────────────
    print(f'Computing per-patient Dice for {len(patients)} held-out patients...')
    patient_scores = {}
    for i, pid in enumerate(patients):
        dice = compute_patient_dice(models, pid, graph_dir, device)
        if dice is not None:
            patient_scores[pid] = dice
        if (i + 1) % 50 == 0:
            print(f'  {i+1}/{len(patients)} done')

    print(f'  Evaluated {len(patient_scores)} patients\n')

    # ── Step 2: select cases ──────────────────────────────────────────────────
    ranked = sorted(patient_scores.items(), key=lambda x: x[1])
    n = len(ranked)

    selected = []  # (patient_id, dice, label)
    selected.append((ranked[0][0],   ranked[0][1],   'worst'))
    selected.append((ranked[1][0],   ranked[1][1],   'worst'))
    selected.append((ranked[2][0],   ranked[2][1],   'worst'))
    selected.append((ranked[n//2][0], ranked[n//2][1], 'median'))
    selected.append((ranked[-1][0],  ranked[-1][1],  'best'))

    print('Selected cases:')
    for pid, dice, lbl in selected:
        print(f'  [{lbl:6s}]  {pid}  Dice={dice:.4f}')
    print()

    # ── Step 3: generate figures ──────────────────────────────────────────────
    summary = []
    for pid, dice, lbl in selected:
        print(f'Generating figure for {pid} ({lbl}, Dice={dice:.4f})...')

        npz_path = preprocessed_dir / pid / f'{pid}_preprocessed.npz'
        if not npz_path.exists():
            print(f'  WARNING: preprocessed NPZ not found, skipping')
            continue

        data_np = np.load(npz_path)
        label_vol = data_np['label']
        t1ce_vol = data_np['T1ce']

        # Find slice with most tumor pixels
        tumor_counts = [(z, int((label_vol[z] > 0).sum())) for z in range(label_vol.shape[0])]
        tumor_counts = [(z, c) for z, c in tumor_counts if c > 200]
        if not tumor_counts:
            print(f'  WARNING: no tumor slices found, skipping')
            continue

        # Pick the slice with the most tumor (most visually informative)
        best_z = max(tumor_counts, key=lambda x: x[1])[0]

        slice_data = {k: data_np[k][best_z] for k in ['T1', 'T1ce', 'T2', 'FLAIR', 'label', 'brain_mask']}

        try:
            segments, pred_prob = predict_slice_pixels(models, slice_data, device)
        except Exception as e:
            print(f'  WARNING: inference failed ({e}), skipping')
            continue

        save_path = output_dir / f'{lbl}_{pid}_z{best_z:03d}.png'
        make_figure(
            t1ce_vol[best_z], label_vol[best_z], pred_prob,
            pid, best_z, dice, save_path, lbl
        )
        print(f'  Saved: {save_path.name}')
        summary.append({'patient': pid, 'label': lbl, 'dice': dice,
                        'slice': best_z, 'figure': str(save_path)})

    # ── Step 4: save summary ──────────────────────────────────────────────────
    summary_path = output_dir / 'summary.json'
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)

    print()
    print('=' * 70)
    print('DONE')
    print('=' * 70)
    print(f'Figures: {output_dir}')
    print(f'Summary: {summary_path}')
    print()
    print('Failure analysis (for Section 6.4):')
    worst = [s for s in summary if s['label'] == 'worst']
    if worst:
        worst_dice = np.mean([s['dice'] for s in worst])
        print(f'  Mean worst-case Dice: {worst_dice:.4f} ({worst_dice*100:.1f}%)')
    best = [s for s in summary if s['label'] == 'best']
    if best:
        print(f'  Best-case Dice:       {best[0]["dice"]:.4f} ({best[0]["dice"]*100:.1f}%)')
    median = [s for s in summary if s['label'] == 'median']
    if median:
        print(f'  Median-case Dice:     {median[0]["dice"]:.4f} ({median[0]["dice"]*100:.1f}%)')


if __name__ == '__main__':
    main()
