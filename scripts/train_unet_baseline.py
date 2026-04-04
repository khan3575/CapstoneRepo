#!/usr/bin/env python3
"""
U-Net Baseline for BraTS Tumor Segmentation
Trains a 3D U-Net (69.1M params) on the same CV splits as the GNN model.

Config: base_channels=56, num_levels=4, patch_size=96³, AMP training.
Uses BCE+Dice loss, AdamW, OneCycleLR, data augmentation, sliding-window inference.
"""

import os
import sys
import json
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import nibabel as nib
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR
from typing import Dict, List, Tuple
import argparse

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


# ============================================================
#  Model
# ============================================================

class UNet3D(nn.Module):
    """3D U-Net for binary tumor segmentation."""
    def __init__(self, in_channels=4, base_channels=56, num_levels=4):
        super().__init__()
        self.num_levels = num_levels

        # Encoder
        self.enc_blocks = nn.ModuleList()
        self.pool = nn.MaxPool3d(2)
        channels = base_channels
        for i in range(num_levels):
            in_ch = in_channels if i == 0 else channels // 2
            self.enc_blocks.append(self._conv_block(in_ch, channels))
            channels *= 2

        # Bottleneck
        self.bottleneck = self._conv_block(channels // 2, channels)

        # Decoder
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
#  Dataset with augmentation
# ============================================================

class BraTSVolumeDataset(Dataset):
    def __init__(self, patient_ids: List[str], data_dir: str,
                 patch_size: Tuple[int, int, int] = (96, 96, 96),
                 augment: bool = False, fg_crop_prob: float = 0.67):
        self.patient_ids = patient_ids
        self.data_dir = Path(data_dir)
        self.patch_size = patch_size
        self.augment = augment
        self.fg_crop_prob = fg_crop_prob
        print(f"  Volume dataset: {len(patient_ids)} patients "
              f"(augment={augment}, fg_crop={fg_crop_prob:.0%})")

    def __len__(self):
        return len(self.patient_ids)

    def __getitem__(self, idx):
        patient_id = self.patient_ids[idx]
        patient_dir = self.data_dir / patient_id

        max_retries = 3
        for attempt in range(max_retries):
            try:
                modalities = []
                for mod in ['FLAIR', 'T1', 'T1ce', 'T2']:
                    mod_path = patient_dir / f"{mod}_preprocessed.nii.gz"
                    img = nib.load(str(mod_path)).get_fdata()
                    modalities.append(img)

                volume = np.stack(modalities, axis=0).astype(np.float32)

                seg_path = patient_dir / "whole_tumor_mask.nii.gz"
                seg = nib.load(str(seg_path)).get_fdata().astype(np.float32)
                seg = seg[np.newaxis, ...]

                volume, seg = self._smart_crop(volume, seg)

                if self.augment:
                    volume, seg = self._augment(volume, seg)

                return torch.from_numpy(volume.copy()), torch.from_numpy(seg.copy())

            except Exception as e:
                if attempt < max_retries - 1:
                    time.sleep(0.1)
                    continue
                else:
                    print(f"  Failed to load {patient_id}: {e}")
                    ps = self.patch_size
                    return (torch.zeros(4, *ps), torch.zeros(1, *ps))

    def _smart_crop(self, volume, seg):
        """Foreground-biased cropping: fg_crop_prob of crops centered on tumor."""
        _, h, w, d = volume.shape
        ph, pw, pd = self.patch_size

        # Pad if needed
        if h < ph or w < pw or d < pd:
            pad_h, pad_w, pad_d = max(0, ph - h), max(0, pw - w), max(0, pd - d)
            volume = np.pad(volume, ((0, 0), (0, pad_h), (0, pad_w), (0, pad_d)), mode='constant')
            seg = np.pad(seg, ((0, 0), (0, pad_h), (0, pad_w), (0, pad_d)), mode='constant')
            _, h, w, d = volume.shape

        # Try foreground-centered crop
        if np.random.rand() < self.fg_crop_prob:
            fg_coords = np.argwhere(seg[0] > 0.5)
            if len(fg_coords) > 0:
                # Pick a random foreground voxel as crop center
                center = fg_coords[np.random.randint(len(fg_coords))]
                sh = np.clip(center[0] - ph // 2, 0, max(h - ph, 0))
                sw = np.clip(center[1] - pw // 2, 0, max(w - pw, 0))
                sd = np.clip(center[2] - pd // 2, 0, max(d - pd, 0))

                return (volume[:, sh:sh+ph, sw:sw+pw, sd:sd+pd],
                        seg[:, sh:sh+ph, sw:sw+pw, sd:sd+pd])

        # Fallback: random crop
        sh = np.random.randint(0, max(h - ph, 0) + 1)
        sw = np.random.randint(0, max(w - pw, 0) + 1)
        sd = np.random.randint(0, max(d - pd, 0) + 1)

        return volume[:, sh:sh+ph, sw:sw+pw, sd:sd+pd], seg[:, sh:sh+ph, sw:sw+pw, sd:sd+pd]

    def _augment(self, volume, seg):
        # Random flips along each spatial axis
        for axis in [1, 2, 3]:
            if np.random.rand() > 0.5:
                volume = np.flip(volume, axis=axis)
                seg = np.flip(seg, axis=axis)

        # Random intensity scaling per modality
        for c in range(volume.shape[0]):
            if np.random.rand() > 0.5:
                scale = np.random.uniform(0.9, 1.1)
                shift = np.random.uniform(-0.1, 0.1)
                volume[c] = volume[c] * scale + shift

        return volume, seg


# ============================================================
#  Sliding-window inference
# ============================================================

def sliding_window_inference(model, volume_tensor, patch_size=(96, 96, 96),
                             overlap=0.5, device='cuda'):
    """
    Run inference with overlapping patches and average predictions.
    volume_tensor: (4, H, W, D) single patient volume.
    Returns: (1, H, W, D) logit map.
    """
    model.eval()
    C, H, W, D = volume_tensor.shape
    ph, pw, pd = patch_size
    sh = max(1, int(ph * (1 - overlap)))
    sw = max(1, int(pw * (1 - overlap)))
    sd = max(1, int(pd * (1 - overlap)))

    # Pad volume if smaller than patch
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
                    pred = model(patch.unsqueeze(0).to(device))  # (1,1,ph,pw,pd)
                    output_sum[0, h0:h0+ph, w0:w0+pw, d0:d0+pd] += pred[0, 0]
                    count_map[0, h0:h0+ph, w0:w0+pw, d0:d0+pd] += 1.0

    output_avg = output_sum / count_map.clamp(min=1)

    # Remove padding
    orig_H = H - pad_h if pad_h > 0 else H
    orig_W = W - pad_w if pad_w > 0 else W
    orig_D = D - pad_d if pad_d > 0 else D
    return output_avg[:, :orig_H, :orig_W, :orig_D]


# ============================================================
#  Metrics
# ============================================================

def dice_score(pred, target, threshold=0.5):
    pred = (torch.sigmoid(pred) > threshold).float()
    intersection = (pred * target).sum()
    union = pred.sum() + target.sum()
    if union == 0:
        return 1.0 if intersection == 0 else 0.0
    return (2.0 * intersection / union).item()


def compute_patient_metrics(pred_mask, gt_mask):
    """Compute Dice, accuracy, sensitivity, specificity, precision on binary masks."""
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
    precision = (tp / (tp + fp)).item() if (tp + fp) > 0 else 0.0

    return {
        'dice': dice, 'accuracy': accuracy, 'sensitivity': sensitivity,
        'specificity': specificity, 'precision': precision,
    }


# ============================================================
#  Training
# ============================================================

def train_epoch(model, dataloader, optimizer, scheduler, scaler, device,
                accumulation_steps=4, pos_weight=9.0):
    model.train()
    total_loss = 0
    total_dice = 0

    bce_pos_weight = torch.tensor([pos_weight], device=device)
    optimizer.zero_grad()

    for batch_idx, (volumes, masks) in enumerate(dataloader):
        volumes = volumes.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)

        with torch.amp.autocast('cuda'):
            outputs = model(volumes)

            # Weighted BCE + Dice loss
            bce_loss = F.binary_cross_entropy_with_logits(
                outputs, masks, pos_weight=bce_pos_weight
            )
            pred_prob = torch.sigmoid(outputs)
            intersection = (pred_prob * masks).sum()
            union_val = pred_prob.sum() + masks.sum()
            dice_loss = 1 - (2 * intersection + 1) / (union_val + 1)

            loss = bce_loss + dice_loss

        scaler.scale(loss / accumulation_steps).backward()

        if (batch_idx + 1) % accumulation_steps == 0:
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            optimizer.zero_grad()

        total_loss += loss.item()
        with torch.no_grad():
            total_dice += dice_score(outputs, masks)

    n = len(dataloader)
    return total_loss / n, total_dice / n


def validate_patches(model, dataloader, device, pos_weight=9.0):
    """Fast patch-based validation during training."""
    model.eval()
    total_loss = 0
    total_dice = 0

    bce_pos_weight = torch.tensor([pos_weight], device=device)

    with torch.no_grad(), torch.amp.autocast('cuda'):
        for volumes, masks in dataloader:
            volumes = volumes.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)

            outputs = model(volumes)
            bce_loss = F.binary_cross_entropy_with_logits(
                outputs, masks, pos_weight=bce_pos_weight
            )
            pred_prob = torch.sigmoid(outputs)
            intersection = (pred_prob * masks).sum()
            union_val = pred_prob.sum() + masks.sum()
            dice_loss = 1 - (2 * intersection + 1) / (union_val + 1)
            loss = bce_loss + dice_loss

            total_loss += loss.item()
            total_dice += dice_score(outputs, masks)

    n = len(dataloader)
    return total_loss / n, total_dice / n


def evaluate_patients_sliding_window(model, patient_ids, data_dir, device,
                                     patch_size=(96, 96, 96)):
    """Full patient-level evaluation with sliding-window inference."""
    model.eval()
    data_dir = Path(data_dir)
    patient_metrics = []

    for pid in patient_ids:
        patient_dir = data_dir / pid
        try:
            modalities = []
            for mod in ['FLAIR', 'T1', 'T1ce', 'T2']:
                img = nib.load(str(patient_dir / f"{mod}_preprocessed.nii.gz")).get_fdata()
                modalities.append(img)
            volume = torch.from_numpy(np.stack(modalities, axis=0).astype(np.float32))

            seg = nib.load(str(patient_dir / "whole_tumor_mask.nii.gz")).get_fdata()
            gt_mask = torch.from_numpy((seg > 0.5).astype(np.float32))

            logits = sliding_window_inference(model, volume, patch_size, overlap=0.5, device=device)
            pred_mask = (torch.sigmoid(logits[0].cpu()) > 0.5).float()

            metrics = compute_patient_metrics(pred_mask, gt_mask)
            metrics['patient_id'] = pid
            patient_metrics.append(metrics)

        except Exception as e:
            print(f"  Error evaluating {pid}: {e}")
            patient_metrics.append({
                'patient_id': pid, 'dice': 0.0, 'accuracy': 0.0,
                'sensitivity': 0.0, 'specificity': 0.0, 'precision': 0.0,
            })

    return patient_metrics


# ============================================================
#  Training loop per fold
# ============================================================

def train_fold(fold_idx: int, fold_data: Dict, args):
    print(f"\n{'='*80}")
    print(f"Training U-Net Baseline - Fold {fold_idx}")
    print(f"  base_channels={args.base_channels}, num_levels={args.num_levels}")
    print(f"{'='*80}\n")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Datasets
    train_dataset = BraTSVolumeDataset(
        fold_data['train_patients'], args.data_dir,
        patch_size=(96, 96, 96), augment=True
    )
    val_dataset = BraTSVolumeDataset(
        fold_data['val_patients'], args.data_dir,
        patch_size=(96, 96, 96), augment=False
    )

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=4, pin_memory=True, persistent_workers=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=4, pin_memory=True, persistent_workers=True
    )

    # Model
    model = UNet3D(in_channels=4, base_channels=args.base_channels,
                   num_levels=args.num_levels).to(device)

    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {num_params:,}")

    # Optimizer, scheduler, scaler
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)

    total_steps = (len(train_loader) * args.epochs) // args.accumulation_steps
    scheduler = OneCycleLR(
        optimizer, max_lr=args.lr, total_steps=max(total_steps, 1),
        pct_start=0.3, anneal_strategy='cos'
    )
    scaler = torch.amp.GradScaler('cuda')

    # Training
    best_val_dice = 0
    best_epoch = 0
    patience_counter = 0
    history = []
    checkpoint_dir = Path(args.output_dir) / f"fold_{fold_idx}"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    start_time = time.time()

    for epoch in range(args.epochs):
        epoch_start = time.time()

        train_loss, train_dice = train_epoch(
            model, train_loader, optimizer, scheduler, scaler, device,
            args.accumulation_steps, pos_weight=args.pos_weight
        )

        val_loss, val_dice = validate_patches(
            model, val_loader, device, pos_weight=args.pos_weight
        )

        epoch_time = time.time() - epoch_start

        history.append({
            'epoch': epoch + 1,
            'train': {'loss': train_loss, 'dice': train_dice},
            'val': {'loss': val_loss, 'dice': val_dice},
            'time': epoch_time
        })

        print(f"Epoch {epoch+1:3d}/{args.epochs} | "
              f"Train: loss={train_loss:.4f} dice={train_dice:.4f} | "
              f"Val: loss={val_loss:.4f} dice={val_dice:.4f} | "
              f"Time: {epoch_time:.1f}s | Best: {best_val_dice:.4f} (ep{best_epoch})")

        if val_dice > best_val_dice:
            best_val_dice = val_dice
            best_epoch = epoch + 1
            patience_counter = 0
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'val_dice': val_dice,
                'num_params': num_params,
                'base_channels': args.base_channels,
                'num_levels': args.num_levels,
            }, checkpoint_dir / "best_model.pt")
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"\n  Early stopping at epoch {epoch+1}")
                break

    training_time = time.time() - start_time
    print(f"\nTraining completed in {training_time/3600:.2f} hours")

    # Full patient-level evaluation with sliding window
    print(f"\n  Evaluating test set ({len(fold_data['test_patients'])} patients) "
          f"with sliding-window inference...")
    checkpoint = torch.load(checkpoint_dir / "best_model.pt", weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])

    test_metrics = evaluate_patients_sliding_window(
        model, fold_data['test_patients'], args.data_dir, device
    )

    test_dices = [m['dice'] for m in test_metrics]
    mean_dice = np.mean(test_dices)
    mean_acc = np.mean([m['accuracy'] for m in test_metrics])
    mean_sens = np.mean([m['sensitivity'] for m in test_metrics])
    mean_spec = np.mean([m['specificity'] for m in test_metrics])
    mean_prec = np.mean([m['precision'] for m in test_metrics])

    results = {
        'fold': fold_idx,
        'config': {
            'base_channels': args.base_channels,
            'num_levels': args.num_levels,
            'num_parameters': num_params,
            'patch_size': '96x96x96',
            'epochs_trained': len(history),
            'max_epochs': args.epochs,
            'batch_size': args.batch_size,
            'accumulation_steps': args.accumulation_steps,
            'effective_batch': args.batch_size * args.accumulation_steps,
            'learning_rate': args.lr,
            'pos_weight': args.pos_weight,
            'loss': 'BCE(pos_weight) + Dice',
            'amp': True,
            'augmentation': 'random_flip + intensity_jitter',
            'eval_method': 'sliding_window_overlap_0.5',
        },
        'best_val_dice': best_val_dice,
        'best_epoch': best_epoch,
        'test_metrics': {
            'mean_dice': float(mean_dice),
            'std_dice': float(np.std(test_dices)),
            'median_dice': float(np.median(test_dices)),
            'mean_accuracy': float(mean_acc),
            'mean_sensitivity': float(mean_sens),
            'mean_specificity': float(mean_spec),
            'mean_precision': float(mean_prec),
        },
        'per_patient_dice': {m['patient_id']: m['dice'] for m in test_metrics},
        'training_time_seconds': training_time,
        'history': history,
    }

    with open(checkpoint_dir / "results.json", 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n  Fold {fold_idx} complete!")
    print(f"   Best Val Dice: {best_val_dice:.4f} (epoch {best_epoch})")
    print(f"   Test Dice (sliding-window): {mean_dice:.4f} +/- {np.std(test_dices):.4f}")
    print(f"   Median Test Dice: {np.median(test_dices):.4f}")
    print(f"   Sensitivity: {mean_sens:.4f} | Specificity: {mean_spec:.4f}")
    print(f"   Precision: {mean_prec:.4f} | Accuracy: {mean_acc:.4f}")
    print(f"   Training time: {training_time/3600:.2f} hours")

    return results


# ============================================================
#  Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(description='Train 3D U-Net baseline')
    parser.add_argument('--data_dir', type=str, default='data/preprocessed')
    parser.add_argument('--fold_file', type=str, default='data/cv_folds_v2/all_folds.json')
    parser.add_argument('--output_dir', type=str, default='checkpoints/unet_baseline')
    parser.add_argument('--fold', type=int, default=None, help='Specific fold (None=all)')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--accumulation_steps', type=int, default=4)
    parser.add_argument('--patience', type=int, default=20)
    parser.add_argument('--pos_weight', type=float, default=9.0)
    parser.add_argument('--base_channels', type=int, default=56)
    parser.add_argument('--num_levels', type=int, default=4)
    args = parser.parse_args()

    print(f"U-Net Baseline Training")
    print(f"  Config: base_channels={args.base_channels}, num_levels={args.num_levels}")
    print(f"  Loss: BCE(pos_weight={args.pos_weight}) + Dice")
    print(f"  AMP: enabled | Augmentation: flip + intensity")
    print(f"  Eval: sliding-window (overlap=0.5)")
    print()

    with open(args.fold_file) as f:
        fold_json = json.load(f)

    all_folds = fold_json['folds']
    print(f"  Loaded {len(all_folds)} folds from {args.fold_file}\n")

    if args.fold is not None:
        fold_data = all_folds[f'fold_{args.fold}']
        train_fold(args.fold, fold_data, args)
    else:
        all_results = []
        for fold_idx in range(len(all_folds)):
            fold_data = all_folds[f'fold_{fold_idx}']
            results = train_fold(fold_idx, fold_data, args)
            all_results.append(results)

        # Aggregate
        test_dices = [r['test_metrics']['mean_dice'] for r in all_results]
        print(f"\n{'='*80}")
        print(f"U-NET BASELINE - CROSS-VALIDATION RESULTS")
        print(f"{'='*80}")
        print(f"Model: base_channels={args.base_channels}, num_levels={args.num_levels}")
        print(f"Parameters: {all_results[0]['config']['num_parameters']:,}")
        print(f"Mean Test Dice: {np.mean(test_dices):.4f} +/- {np.std(test_dices):.4f}")
        for i, d in enumerate(test_dices):
            print(f"  Fold {i}: {d:.4f}")

        total_time = sum(r['training_time_seconds'] for r in all_results)
        print(f"Total training time: {total_time/3600:.2f} hours")

        aggregate = {
            'model': 'UNet3D',
            'base_channels': args.base_channels,
            'num_levels': args.num_levels,
            'num_parameters': all_results[0]['config']['num_parameters'],
            'num_folds': len(all_results),
            'mean_test_dice': float(np.mean(test_dices)),
            'std_test_dice': float(np.std(test_dices)),
            'per_fold_dice': test_dices,
            'total_training_hours': total_time / 3600,
            'fold_results': all_results,
        }

        output_path = Path(args.output_dir) / "aggregate_results.json"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(aggregate, f, indent=2)

        print(f"\n  Results saved to {output_path}")


if __name__ == "__main__":
    main()
