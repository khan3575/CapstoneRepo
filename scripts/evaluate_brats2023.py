#!/usr/bin/env python3
"""
BraTS 2023 Zero-Shot Evaluation — Phase 7 (Task 7.4)

Loads the 5-fold ensemble trained on BraTS 2021 and evaluates it on BraTS 2023
without any retraining (zero-shot cross-dataset generalisation).

Prerequisites:
  1. All 5 folds trained: checkpoints/binary_v2/fold_{0..4}/best_model.pth
  2. BraTS 2023 preprocessed: data/preprocessed_brats2023/
  3. BraTS 2023 graphs built:  data/graphs_brats2023/

Usage:
    python scripts/evaluate_brats2023.py --device cuda
    python scripts/evaluate_brats2023.py --device cuda --demo  # 10 patients
"""

import sys
import json
import argparse
import numpy as np
import torch
from pathlib import Path
from tqdm import tqdm

# project root on path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from gnn_model import TumorSegmentationGNN
from dataset import BraTSGraphDataset, BinaryTransform
from torch_geometric.loader import DataLoader as GeometricDataLoader


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
GRAPHS_DIR    = "data/graphs_brats2023"
CHECKPOINT_DIR = "checkpoints/binary_v3"  # FIX-5 (2026-03-14): was binary_v2 — production model is binary_v3
OUTPUT_DIR    = "research_results/brats2023_evaluation"

BRATS2021_ENSEMBLE_DICE = 0.9141   # Reference: Phase 4 held-out result


# ---------------------------------------------------------------------------
# Model loading (same as inference_ensemble.py)
# ---------------------------------------------------------------------------

def load_ensemble_models(checkpoint_dir, device):
    models = []
    print("="*70)
    print("LOADING ENSEMBLE MODELS (trained on BraTS 2021)")
    print("="*70)
    for fold_idx in range(5):
        ckpt_path = Path(checkpoint_dir) / f"fold_{fold_idx}" / "best_model.pth"
        if not ckpt_path.exists():
            print(f"  WARNING: fold {fold_idx} not found: {ckpt_path}")
            continue
        ckpt = torch.load(str(ckpt_path), map_location=device, weights_only=False)
        model = TumorSegmentationGNN(
            in_channels=15, hidden_channels=256, num_layers=5, dropout=0.1
        ).to(device)
        sd = ckpt["model_state_dict"]
        # strip torch.compile prefix if present
        if any(k.startswith("_orig_mod.") for k in sd):
            sd = {k.replace("_orig_mod.", "", 1): v for k, v in sd.items()}
        model.load_state_dict(sd)
        model.eval()
        models.append(model)
        val_dice = ckpt.get("val_dice", float("nan"))
        print(f"  Fold {fold_idx}: loaded  (val Dice={val_dice:.4f})")
    print(f"\nLoaded {len(models)}/5 models")
    return models


# ---------------------------------------------------------------------------
# Ensemble prediction (batch-aware)
# ---------------------------------------------------------------------------

@torch.no_grad()
def predict_batch(models, batch, device):
    """Average logits across ensemble, return probs and binary preds."""
    logits_sum = torch.zeros(batch.x.shape[0], device=device)
    for model in models:
        logits, _ = model(batch)
        logits_sum += logits
    probs = torch.sigmoid(logits_sum / len(models))
    preds = (probs > 0.5).float()
    return probs, preds


# ---------------------------------------------------------------------------
# Evaluation loop
# ---------------------------------------------------------------------------

def evaluate(models, graph_files, device, batch_size=64):
    transform = BinaryTransform()
    dataset = BraTSGraphDataset(
        root_dir=None,
        split="test",
        graph_files=graph_files,
        transform=transform
    )
    loader = GeometricDataLoader(
        dataset, batch_size=batch_size, shuffle=False,
        num_workers=4, pin_memory=True, persistent_workers=True
    )

    print(f"\nEvaluating on {len(dataset)} graphs from {len(graph_files)} patients ...")
    print("="*70)

    all_dice = []
    total_tp = total_fp = total_fn = total_tn = 0

    for batch in tqdm(loader, desc="Inference"):
        batch = batch.to(device, non_blocking=True)
        _, preds = predict_batch(models, batch, device)
        y_true = (batch.y > 0).float()

        inter = (preds * y_true).sum().item()
        union = preds.sum().item() + y_true.sum().item()
        dice  = (2.0 * inter + 1e-7) / (union + 1e-7)
        all_dice.append(dice)

        total_tp += ((preds == 1) & (y_true == 1)).sum().item()
        total_fp += ((preds == 1) & (y_true == 0)).sum().item()
        total_fn += ((preds == 0) & (y_true == 1)).sum().item()
        total_tn += ((preds == 0) & (y_true == 0)).sum().item()

    sensitivity = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    specificity = total_tn / (total_tn + total_fp) if (total_tn + total_fp) > 0 else 0.0
    precision   = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    accuracy    = (total_tp + total_tn) / (total_tp + total_fp + total_fn + total_tn + 1e-9)

    return {
        "dice":        float(np.mean(all_dice)),
        "dice_std":    float(np.std(all_dice)),
        "accuracy":    float(accuracy),
        "sensitivity": float(sensitivity),
        "specificity": float(specificity),
        "precision":   float(precision),
        "num_graphs":  len(all_dice),
        "num_patients": len(graph_files),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="BraTS 2023 zero-shot ensemble evaluation")
    parser.add_argument("--graphs_dir",     default=GRAPHS_DIR,     help="BraTS 2023 graph dir")
    parser.add_argument("--checkpoint_dir", default=CHECKPOINT_DIR, help="Ensemble checkpoint dir")
    parser.add_argument("--output_dir",     default=OUTPUT_DIR,     help="Results output dir")
    parser.add_argument("--device",         default="cuda",         help="cuda or cpu")
    parser.add_argument("--batch_size",     type=int, default=64,   help="Graphs per batch")
    parser.add_argument("--demo",           action="store_true",    help="10 patients only")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if args.demo:
        print("[DEMO MODE: 10 patients]")

    # --- Collect graph files ---
    graphs_dir = Path(args.graphs_dir)
    if not graphs_dir.exists():
        print(f"ERROR: graphs dir not found: {graphs_dir}")
        print("Run graph construction first:")
        print("  python src/graph_construction.py \\")
        print("    --input_dir data/preprocessed_brats2023 \\")
        print("    --output_dir data/graphs_brats2023 \\")
        print("    --num_workers 12")
        sys.exit(1)

    graph_files = sorted(graphs_dir.glob("*/*_graphs_200.pt"))
    graph_files = [str(g) for g in graph_files]

    if not graph_files:
        print(f"ERROR: No *_graphs_200.pt files found in {graphs_dir}")
        sys.exit(1)

    if args.demo:
        graph_files = graph_files[:10]

    print(f"\nFound {len(graph_files)} patient graph files")

    # --- Load models ---
    models = load_ensemble_models(args.checkpoint_dir, device)
    if not models:
        print("ERROR: No models loaded. Train all 5 folds first.")
        sys.exit(1)

    # --- Evaluate ---
    metrics = evaluate(models, graph_files, device, args.batch_size)

    # --- Report ---
    gap = BRATS2021_ENSEMBLE_DICE - metrics["dice"]
    print("\n" + "="*70)
    print("BRATS 2023 ZERO-SHOT EVALUATION RESULTS")
    print("="*70)
    print(f"  Patients evaluated:  {metrics['num_patients']}")
    print(f"  Graphs evaluated:    {metrics['num_graphs']}")
    print(f"  Dice:                {metrics['dice']*100:.2f}% +/- {metrics['dice_std']*100:.2f}%")
    print(f"  Accuracy:            {metrics['accuracy']*100:.2f}%")
    print(f"  Sensitivity:         {metrics['sensitivity']*100:.2f}%")
    print(f"  Specificity:         {metrics['specificity']*100:.2f}%")
    print(f"  Precision:           {metrics['precision']*100:.2f}%")
    print()
    print("GENERALISATION GAP:")
    print(f"  BraTS 2021 (held-out ensemble): {BRATS2021_ENSEMBLE_DICE*100:.2f}%")
    print(f"  BraTS 2023 (zero-shot):         {metrics['dice']*100:.2f}%")
    print(f"  Gap:                            {gap*100:.2f}%  ({'expected 2-5%' if 0.01 < gap < 0.07 else 'outside expected range'})")
    print("="*70)

    # --- Save ---
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results = {
        "brats2023_metrics": metrics,
        "brats2021_reference": {
            "ensemble_dice": BRATS2021_ENSEMBLE_DICE,
            "note": "Phase 4 held-out set (251 patients, no data leakage)"
        },
        "generalisation_gap": float(gap),
        "metadata": {
            "device": str(device),
            "demo_mode": args.demo,
            "graphs_dir": str(graphs_dir.resolve()),
            "checkpoint_dir": str(Path(args.checkpoint_dir).resolve()),
        }
    }

    out_file = output_dir / "results.json"
    with open(out_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {out_file}")


if __name__ == "__main__":
    main()
