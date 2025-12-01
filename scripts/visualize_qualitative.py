#!/usr/bin/env python3
"""
Qualitative Visualization Script
Creates side-by-side comparisons of GNN predictions vs ground truth.

Output: RGB images showing:
  - Left: Raw FLAIR MRI
  - Middle: Ground Truth (Green overlay)
  - Right: GNN Prediction with overlay
    * Yellow: Correct prediction (TP)
    * Red: False Positive
    * Blue: False Negative
    * Black: True Negative

Usage:
    python scripts/visualize_qualitative.py --fold 0 --num_patients 10
"""

import sys
from pathlib import Path

# Add src directory to path
script_dir = Path(__file__).parent
project_root = script_dir.parent
sys.path.insert(0, str(project_root / 'src'))

import torch
import numpy as np
import matplotlib.pyplot as plt
import argparse
import json
from tqdm import tqdm

from gnn_model import TumorSegmentationGNN
from dataset import BraTSGraphDataset, BinaryTransform
from torch_geometric.loader import DataLoader as GeometricDataLoader


def load_model(checkpoint_path, device='cuda'):
    """Load trained model from checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    # Get model config from checkpoint
    in_channels = 15  # Fixed after leakage fix
    hidden_channels = 256
    num_layers = 5
    
    model = TumorSegmentationGNN(
        in_channels=in_channels,
        hidden_channels=hidden_channels,
        num_layers=num_layers,
        dropout=0.1
    ).to(device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print(f"✅ Loaded model from epoch {checkpoint['epoch']}")
    print(f"   Validation Dice: {checkpoint['val_dice']:.4f}")
    
    return model


def predict_patient_graphs(model, graph_files, device='cuda'):
    """
    Run inference on all graphs for a patient.
    
    Returns:
        predictions: List of predicted labels per graph
    """
    binary_transform = BinaryTransform()
    
    dataset = BraTSGraphDataset(
        root_dir=None,
        split='test',
        graph_files=graph_files,
        transform=binary_transform
    )
    
    loader = GeometricDataLoader(dataset, batch_size=1, shuffle=False)
    
    predictions = []
    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            output, _ = model(data)  # Model returns (logits, embeddings)
            pred = (torch.sigmoid(output) > 0.5).float()
            predictions.append(pred.cpu().numpy())
    
    return predictions


def reconstruct_volume_from_graphs(predictions, segments, original_shape):
    """
    Reconstruct full 3D volume from graph predictions.
    
    Args:
        predictions: List of prediction arrays per graph (from 2-slice pairs)
        segments: List of segment arrays per slice
        original_shape: (D, H, W) shape of original volume
    
    Returns:
        volume: 3D binary prediction volume
    """
    D, H, W = original_shape
    volume = np.zeros((D, H, W), dtype=np.float32)
    count = np.zeros((D, H, W), dtype=np.int32)
    
    # Each graph corresponds to a 2-slice pair
    # predictions[i] has node predictions for graph i (slices i and i+1)
    
    for graph_idx, pred in enumerate(predictions):
        # This graph covers slices graph_idx and graph_idx+1
        slice_idx = graph_idx
        
        if slice_idx >= len(segments):
            break
        
        # Get segments for this slice
        seg = segments[slice_idx]
        
        # pred is [num_nodes, 1], flatten it
        pred_flat = pred.flatten()
        
        # Map predictions back to pixels
        for node_idx in range(len(pred_flat)):
            if node_idx >= seg.max() + 1:
                break
            mask = (seg == node_idx)
            volume[slice_idx][mask] += pred_flat[node_idx]
            count[slice_idx][mask] += 1
    
    # Average predictions where multiple graphs overlap
    volume = np.divide(volume, count, where=count > 0)
    volume = (volume > 0.5).astype(np.uint8)
    
    return volume


def create_comparison_figure(flair_slice, gt_slice, pred_slice, slice_idx, patient_id, save_path):
    """
    Create side-by-side comparison figure.
    
    Args:
        flair_slice: 2D FLAIR image (H, W)
        gt_slice: 2D ground truth mask (H, W)
        pred_slice: 2D prediction mask (H, W)
        slice_idx: Slice index for title
        patient_id: Patient ID for title
        save_path: Path to save figure
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Normalize FLAIR for display
    flair_norm = flair_slice.copy()
    if flair_norm.max() > flair_norm.min():
        flair_norm = (flair_norm - flair_norm.min()) / (flair_norm.max() - flair_norm.min())
    
    # Left: Raw FLAIR
    axes[0].imshow(flair_norm, cmap='gray')
    axes[0].set_title('FLAIR MRI', fontsize=14)
    axes[0].axis('off')
    
    # Middle: Ground Truth overlay (Green)
    axes[1].imshow(flair_norm, cmap='gray')
    gt_overlay = np.zeros((*gt_slice.shape, 3))
    gt_overlay[gt_slice > 0] = [0, 1, 0]  # Green
    axes[1].imshow(gt_overlay, alpha=0.4)
    axes[1].set_title('Ground Truth', fontsize=14)
    axes[1].axis('off')
    
    # Right: Prediction with error visualization
    axes[2].imshow(flair_norm, cmap='gray')
    
    # Create colored overlay
    pred_overlay = np.zeros((*pred_slice.shape, 3))
    
    # True Positive (TP): Yellow
    tp_mask = (gt_slice > 0) & (pred_slice > 0)
    pred_overlay[tp_mask] = [1, 1, 0]
    
    # False Positive (FP): Red
    fp_mask = (gt_slice == 0) & (pred_slice > 0)
    pred_overlay[fp_mask] = [1, 0, 0]
    
    # False Negative (FN): Blue
    fn_mask = (gt_slice > 0) & (pred_slice == 0)
    pred_overlay[fn_mask] = [0, 0, 1]
    
    axes[2].imshow(pred_overlay, alpha=0.5)
    axes[2].set_title('GNN Prediction\n(Yellow=TP, Red=FP, Blue=FN)', fontsize=14)
    axes[2].axis('off')
    
    # Calculate metrics for this slice
    tp = np.sum(tp_mask)
    fp = np.sum(fp_mask)
    fn = np.sum(fn_mask)
    
    dice = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0.0
    
    fig.suptitle(f'{patient_id} - Slice {slice_idx} | Dice: {dice:.4f}', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    return dice


def visualize_patient(patient_id, model, fold_data, output_dir, device='cuda'):
    """
    Generate visualizations for a single patient.
    
    Returns:
        mean_dice: Average Dice score across visualized slices
    """
    print(f"\n📊 Processing {patient_id}...")
    
    # Load preprocessed volume for FLAIR and ground truth
    preproc_path = Path('data/preprocessed') / patient_id / f'{patient_id}_preprocessed.npz'
    
    if not preproc_path.exists():
        print(f"  ❌ Preprocessed data not found: {preproc_path}")
        return 0.0
    
    volume_data = np.load(preproc_path)
    flair = volume_data['FLAIR']
    gt_label = volume_data['label']
    
    # Load graphs and segments
    graph_path = Path('data/graphs') / patient_id / f'{patient_id}_graphs_200.pt'
    segment_path = Path('data/graphs') / patient_id / f'{patient_id}_segments_200.npy'
    
    if not graph_path.exists() or not segment_path.exists():
        print(f"  ❌ Graph data not found")
        return 0.0
    
    graphs = torch.load(graph_path, map_location='cpu', weights_only=False)
    segments = np.load(segment_path)
    
    # Run prediction
    print(f"  🔮 Running inference on {len(graphs)} graphs...")
    predictions = predict_patient_graphs(model, [str(graph_path)], device)
    
    # Reconstruct volume
    print(f"  🔄 Reconstructing volume...")
    pred_volume = reconstruct_volume_from_graphs(predictions, segments, flair.shape)
    
    # Select slices with tumor for visualization
    tumor_slices = [i for i in range(gt_label.shape[0]) if np.sum(gt_label[i]) > 100]
    
    if len(tumor_slices) == 0:
        print(f"  ⚠️  No tumor slices found")
        return 0.0
    
    # Visualize up to 5 slices
    selected_slices = tumor_slices[::max(1, len(tumor_slices)//5)][:5]
    
    patient_output_dir = output_dir / patient_id
    patient_output_dir.mkdir(parents=True, exist_ok=True)
    
    dice_scores = []
    for slice_idx in selected_slices:
        save_path = patient_output_dir / f'slice_{slice_idx:03d}.png'
        
        dice = create_comparison_figure(
            flair[slice_idx],
            gt_label[slice_idx],
            pred_volume[slice_idx],
            slice_idx,
            patient_id,
            save_path
        )
        dice_scores.append(dice)
        print(f"  ✓ Slice {slice_idx}: Dice = {dice:.4f}")
    
    mean_dice = np.mean(dice_scores)
    print(f"  📈 Mean Dice: {mean_dice:.4f}")
    
    return mean_dice


def main():
    parser = argparse.ArgumentParser(description='Visualize GNN predictions')
    parser.add_argument('--fold', type=int, default=0, help='Which fold to visualize')
    parser.add_argument('--num_patients', type=int, default=10, help='Number of patients to visualize')
    parser.add_argument('--output_dir', type=str, default='visualizations/qualitative',
                        help='Output directory for visualizations')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use')
    
    args = parser.parse_args()
    
    print("="*80)
    print("QUALITATIVE VISUALIZATION - GNN vs Ground Truth")
    print("="*80)
    
    # Setup paths
    checkpoint_dir = Path('checkpoints/binary_training') / f'fold_{args.fold}'
    checkpoint_path = checkpoint_dir / 'best_model.pth'
    
    if not checkpoint_path.exists():
        print(f"❌ Checkpoint not found: {checkpoint_path}")
        return
    
    # Load fold data
    fold_file = Path('data/cv_folds') / f'fold_{args.fold}.json'
    with open(fold_file) as f:
        fold_data = json.load(f)
    
    # Load model
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"\n📦 Loading model from {checkpoint_path}")
    model = load_model(checkpoint_path, device)
    
    # Create output directory
    output_dir = Path(args.output_dir) / f'fold_{args.fold}'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Select patients (from test set for unbiased visualization)
    test_patients = fold_data['test_patients'][:args.num_patients]
    
    print(f"\n🎨 Generating visualizations for {len(test_patients)} patients...")
    print(f"   Output: {output_dir}")
    
    # Process each patient
    dice_scores = []
    for patient_id in tqdm(test_patients, desc="Visualizing patients"):
        mean_dice = visualize_patient(patient_id, model, fold_data, output_dir, device)
        if mean_dice > 0:
            dice_scores.append(mean_dice)
    
    # Summary
    print("\n" + "="*80)
    print("VISUALIZATION COMPLETE")
    print("="*80)
    print(f"✅ Generated visualizations for {len(dice_scores)} patients")
    print(f"📊 Average Dice: {np.mean(dice_scores):.4f} ± {np.std(dice_scores):.4f}")
    print(f"📁 Images saved to: {output_dir}")
    print("\n💡 Next steps:")
    print("   1. Review the images to qualitatively assess segmentation quality")
    print("   2. Select best examples for thesis figures")
    print("   3. Run on other folds to ensure consistency")


if __name__ == "__main__":
    main()
