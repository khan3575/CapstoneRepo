#!/usr/bin/env python3
"""
U-Net Baseline for BraTS Tumor Segmentation
Trains a standard 3D U-Net on the same CV splits as the GNN model for fair comparison
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

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

class UNet3D(nn.Module):
    """
    Simple 3D U-Net for tumor segmentation
    Binary segmentation: tumor vs background
    """
    def __init__(self, in_channels=4, base_channels=16, num_levels=3):
        super().__init__()
        
        self.num_levels = num_levels
        
        # Encoder (downsampling)
        self.enc_blocks = nn.ModuleList()
        self.pool = nn.MaxPool3d(2)
        
        channels = base_channels
        for i in range(num_levels):
            in_ch = in_channels if i == 0 else channels // 2
            self.enc_blocks.append(self._conv_block(in_ch, channels))
            channels *= 2
        
        # Bottleneck
        self.bottleneck = self._conv_block(channels // 2, channels)
        
        # Decoder (upsampling)
        self.dec_blocks = nn.ModuleList()
        self.upconvs = nn.ModuleList()
        
        for i in range(num_levels):
            self.upconvs.append(nn.ConvTranspose3d(channels, channels // 2, kernel_size=2, stride=2))
            self.dec_blocks.append(self._conv_block(channels, channels // 2))
            channels //= 2
        
        # Final output
        self.out_conv = nn.Conv3d(base_channels, 1, kernel_size=1)
    
    def _conv_block(self, in_channels, out_channels):
        """Double convolution block with batch norm and ReLU"""
        return nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        # Encoder
        skip_connections = []
        for i, enc_block in enumerate(self.enc_blocks):
            x = enc_block(x)
            skip_connections.append(x)
            x = self.pool(x)
        
        # Bottleneck
        x = self.bottleneck(x)
        
        # Decoder
        for i, (upconv, dec_block) in enumerate(zip(self.upconvs, self.dec_blocks)):
            x = upconv(x)
            skip = skip_connections[-(i+1)]
            
            # Handle size mismatch
            if x.shape != skip.shape:
                x = F.interpolate(x, size=skip.shape[2:], mode='trilinear', align_corners=False)
            
            x = torch.cat([x, skip], dim=1)
            x = dec_block(x)
        
        # Output
        x = self.out_conv(x)
        return x


class BraTSVolumeDataset(Dataset):
    """
    Dataset for loading full 3D MRI volumes (not graphs)
    Loads from preprocessed data directory
    """
    def __init__(self, patient_ids: List[str], data_dir: str, patch_size: Tuple[int, int, int] = (128, 128, 128)):
        self.patient_ids = patient_ids
        self.data_dir = Path(data_dir)
        self.patch_size = patch_size
        
        print(f"📊 Volume dataset: {len(patient_ids)} patients")
    
    def __len__(self):
        return len(self.patient_ids)
    
    def __getitem__(self, idx):
        patient_id = self.patient_ids[idx]
        patient_dir = self.data_dir / patient_id
        
        max_retries = 3
        for attempt in range(max_retries):
            try:
                # Load 4 modalities (using actual preprocessed filenames)
                modalities = []
                for mod in ['FLAIR', 'T1', 'T1ce', 'T2']:
                    mod_path = patient_dir / f"{mod}_preprocessed.nii.gz"
                    if not mod_path.exists():
                        raise FileNotFoundError(f"Missing {mod_path}")
                    
                    img = nib.load(str(mod_path)).get_fdata()
                    modalities.append(img)
                
                # Stack modalities: (4, H, W, D)
                volume = np.stack(modalities, axis=0).astype(np.float32)
                
                # Load segmentation mask
                seg_path = patient_dir / "whole_tumor_mask.nii.gz"
                if not seg_path.exists():
                    raise FileNotFoundError(f"Missing {seg_path}")
                
                seg = nib.load(str(seg_path)).get_fdata().astype(np.float32)
                
                # Mask is already binary (0 or 1)
                # Add batch dimension for seg
                seg = seg[np.newaxis, ...]  # (1, H, W, D)
                
                # Random crop to patch_size (simple augmentation)
                volume, seg = self._random_crop(volume, seg)
                
                return torch.from_numpy(volume), torch.from_numpy(seg)
                
            except Exception as e:
                if attempt < max_retries - 1:
                    import time
                    time.sleep(0.1)  # Brief delay before retry
                    continue
                else:
                    # After retries, return a zero sample (will have low dice, but won't crash)
                    print(f"⚠️  Failed to load {patient_id} after {max_retries} attempts: {e}")
                    volume = np.zeros((4, 96, 96, 96), dtype=np.float32)
                    seg = np.zeros((1, 96, 96, 96), dtype=np.float32)
                    return torch.from_numpy(volume), torch.from_numpy(seg)
    
    def _random_crop(self, volume, seg):
        """Random crop to patch size"""
        _, h, w, d = volume.shape
        ph, pw, pd = self.patch_size
        
        # If volume smaller than patch, pad
        if h < ph or w < pw or d < pd:
            pad_h = max(0, ph - h)
            pad_w = max(0, pw - w)
            pad_d = max(0, pd - d)
            
            volume = np.pad(volume, ((0, 0), (0, pad_h), (0, pad_w), (0, pad_d)), mode='constant')
            seg = np.pad(seg, ((0, 0), (0, pad_h), (0, pad_w), (0, pad_d)), mode='constant')
            
            h, w, d = volume.shape[1:]
        
        # Random crop
        start_h = np.random.randint(0, h - ph + 1) if h > ph else 0
        start_w = np.random.randint(0, w - pw + 1) if w > pw else 0
        start_d = np.random.randint(0, d - pd + 1) if d > pd else 0
        
        volume = volume[:, start_h:start_h+ph, start_w:start_w+pw, start_d:start_d+pd]
        seg = seg[:, start_h:start_h+ph, start_w:start_w+pw, start_d:start_d+pd]
        
        return volume, seg


def dice_score(pred, target, threshold=0.5):
    """Calculate Dice score"""
    pred = (torch.sigmoid(pred) > threshold).float()
    intersection = (pred * target).sum()
    union = pred.sum() + target.sum()
    
    if union == 0:
        return 1.0 if intersection == 0 else 0.0
    
    return (2.0 * intersection / union).item()


def train_epoch(model, dataloader, optimizer, scheduler, device, accumulation_steps=4):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    total_dice = 0
    
    optimizer.zero_grad()
    
    for batch_idx, (volumes, masks) in enumerate(dataloader):
        volumes = volumes.to(device)
        masks = masks.to(device)
        
        # Forward
        outputs = model(volumes)
        
        # BCE + Dice loss
        bce_loss = F.binary_cross_entropy_with_logits(outputs, masks)
        
        # Dice loss component
        pred_prob = torch.sigmoid(outputs)
        intersection = (pred_prob * masks).sum()
        union = pred_prob.sum() + masks.sum()
        dice_loss = 1 - (2 * intersection + 1) / (union + 1)
        
        loss = bce_loss + dice_loss
        
        # Backward with gradient accumulation
        (loss / accumulation_steps).backward()
        
        if (batch_idx + 1) % accumulation_steps == 0:
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
        
        total_loss += loss.item()
        total_dice += dice_score(outputs, masks)
    
    return total_loss / len(dataloader), total_dice / len(dataloader)


def validate(model, dataloader, device):
    """Validate model"""
    model.eval()
    total_loss = 0
    total_dice = 0
    
    with torch.no_grad():
        for volumes, masks in dataloader:
            volumes = volumes.to(device)
            masks = masks.to(device)
            
            outputs = model(volumes)
            
            # Loss
            bce_loss = F.binary_cross_entropy_with_logits(outputs, masks)
            pred_prob = torch.sigmoid(outputs)
            intersection = (pred_prob * masks).sum()
            union = pred_prob.sum() + masks.sum()
            dice_loss = 1 - (2 * intersection + 1) / (union + 1)
            loss = bce_loss + dice_loss
            
            total_loss += loss.item()
            total_dice += dice_score(outputs, masks)
    
    return total_loss / len(dataloader), total_dice / len(dataloader)


def train_fold(fold_idx: int, fold_data: Dict, args):
    """Train U-Net on one CV fold"""
    print(f"\n{'='*80}")
    print(f"Training U-Net Baseline - Fold {fold_idx}")
    print(f"{'='*80}\n")
    
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # Create datasets
    train_dataset = BraTSVolumeDataset(
        fold_data['train_patients'],
        args.data_dir,
        patch_size=(96, 96, 96)
    )
    
    val_dataset = BraTSVolumeDataset(
        fold_data['val_patients'],
        args.data_dir,
        patch_size=(96, 96, 96)
    )
    
    test_dataset = BraTSVolumeDataset(
        fold_data['test_patients'],
        args.data_dir,
        patch_size=(96, 96, 96)
    )
    
    # Dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=8,
        pin_memory=True,
        prefetch_factor=3,
        persistent_workers=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=8,
        pin_memory=True,
        prefetch_factor=3,
        persistent_workers=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    # Model
    model = UNet3D(in_channels=4, base_channels=16, num_levels=3).to(device)
    
    # Count parameters
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {num_params:,}")
    
    # Optimizer and scheduler
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
    
    total_steps = len(train_loader) * args.epochs // args.accumulation_steps
    scheduler = OneCycleLR(
        optimizer,
        max_lr=args.lr,
        total_steps=total_steps,
        pct_start=0.3,
        anneal_strategy='cos'
    )
    
    # Training loop
    best_val_dice = 0
    best_epoch = 0
    patience = 10
    patience_counter = 0
    
    history = []
    start_time = time.time()
    
    for epoch in range(args.epochs):
        epoch_start = time.time()
        
        # Train
        train_loss, train_dice = train_epoch(
            model, train_loader, optimizer, scheduler, device, args.accumulation_steps
        )
        
        # Validate
        val_loss, val_dice = validate(model, val_loader, device)
        
        epoch_time = time.time() - epoch_start
        
        # Log
        history.append({
            'epoch': epoch + 1,
            'train': {'loss': train_loss, 'dice': train_dice},
            'val': {'loss': val_loss, 'dice': val_dice},
            'time': epoch_time
        })
        
        print(f"Epoch {epoch+1}/{args.epochs} | "
              f"Train: loss={train_loss:.4f} dice={train_dice:.4f} | "
              f"Val: loss={val_loss:.4f} dice={val_dice:.4f} | "
              f"Time: {epoch_time:.1f}s")
        
        # Early stopping
        if val_dice > best_val_dice:
            best_val_dice = val_dice
            best_epoch = epoch + 1
            patience_counter = 0
            
            # Save checkpoint
            checkpoint_dir = Path(args.output_dir) / f"fold_{fold_idx}"
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_dice': val_dice,
            }, checkpoint_dir / "best_model.pt")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"\n⏹️  Early stopping at epoch {epoch+1}")
                break
    
    training_time = time.time() - start_time
    
    # Test evaluation
    print(f"\n📊 Evaluating on test set...")
    checkpoint = torch.load(checkpoint_dir / "best_model.pt")
    model.load_state_dict(checkpoint['model_state_dict'])
    
    test_loss, test_dice = validate(model, test_loader, device)
    
    # Save results
    results = {
        'fold': fold_idx,
        'config': {
            'epochs': args.epochs,
            'batch_size': args.batch_size,
            'learning_rate': args.lr,
            'base_channels': 16,
            'num_levels': 3,
            'patch_size': '96x96x96',
            'accumulation_steps': args.accumulation_steps,
            'model': 'UNet3D_Lightweight',
            'num_parameters': num_params
        },
        'best_val_dice': best_val_dice,
        'best_epoch': best_epoch,
        'test_metrics': {
            'dice': test_dice,
            'loss': test_loss
        },
        'training_time_seconds': training_time,
        'history': history
    }
    
    with open(checkpoint_dir / "results.json", 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✅ Fold {fold_idx} completed!")
    print(f"   Best Val Dice: {best_val_dice:.4f} (epoch {best_epoch})")
    print(f"   Test Dice: {test_dice:.4f}")
    print(f"   Training time: {training_time/60:.1f} minutes")
    
    return results


def main():
    parser = argparse.ArgumentParser(description='Train U-Net baseline')
    parser.add_argument('--data_dir', type=str, default='data/preprocessed',
                        help='Directory with preprocessed volumes')
    parser.add_argument('--fold_file', type=str, default='data/cv_folds/all_folds.json',
                        help='Path to CV fold file')
    parser.add_argument('--output_dir', type=str, default='checkpoints/unet_baseline',
                        help='Output directory')
    parser.add_argument('--fold', type=int, default=None,
                        help='Specific fold to train (None = all folds)')
    parser.add_argument('--epochs', type=int, default=50,
                        help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=1,
                        help='Batch size (small due to 3D volumes)')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate')
    parser.add_argument('--accumulation_steps', type=int, default=4,
                        help='Gradient accumulation steps')
    
    args = parser.parse_args()
    
    # Load CV folds
    with open(args.fold_file) as f:
        fold_json = json.load(f)
    
    all_folds = fold_json['folds']
    print(f"📁 Loaded {len(all_folds)} folds from {args.fold_file}")
    
    # Train specified fold(s)
    if args.fold is not None:
        fold_data = all_folds[f'fold_{args.fold}']
        results = train_fold(args.fold, fold_data, args)
    else:
        # Train all folds
        all_results = []
        for fold_idx in range(len(all_folds)):
            fold_data = all_folds[f'fold_{fold_idx}']
            results = train_fold(fold_idx, fold_data, args)
            all_results.append(results)
        
        # Aggregate results
        test_dices = [r['test_metrics']['dice'] for r in all_results]
        print(f"\n{'='*80}")
        print(f"U-NET BASELINE - CROSS-VALIDATION RESULTS")
        print(f"{'='*80}")
        print(f"Mean Test Dice: {np.mean(test_dices):.4f} ± {np.std(test_dices):.4f}")
        print(f"Folds: {test_dices}")
        
        # Save aggregate
        aggregate = {
            'model': 'UNet3D',
            'num_folds': len(all_results),
            'mean_test_dice': float(np.mean(test_dices)),
            'std_test_dice': float(np.std(test_dices)),
            'fold_results': all_results
        }
        
        output_path = Path(args.output_dir) / "aggregate_results.json"
        with open(output_path, 'w') as f:
            json.dump(aggregate, f, indent=2)
        
        print(f"\n✅ Results saved to {output_path}")


if __name__ == "__main__":
    main()
