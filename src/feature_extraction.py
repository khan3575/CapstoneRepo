import os
import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
from torch_geometric.data import Data
from skimage.transform import resize
from tqdm import tqdm
import argparse
from concurrent.futures import ProcessPoolExecutor
import glob
import time
import traceback

class FeatureExtractor:
    def __init__(self, backbone="resnet18", output_dim=128, device=None):
        """
        Initialize the CNN feature extractor.
        
        Args:
            backbone: CNN backbone model ('resnet18', 'resnet34', etc.)
            output_dim: Dimension of output feature vector
            device: Device to run the model on
        """
        self.output_dim = output_dim
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load pretrained model
        if backbone == "resnet18":
            self.model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
            self.feature_dim = self.model.fc.in_features
        elif backbone == "resnet34":
            self.model = models.resnet34(weights=models.ResNet34_Weights.IMAGENET1K_V1)
            self.feature_dim = self.model.fc.in_features
        elif backbone == "resnet50":
            self.model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
            self.feature_dim = self.model.fc.in_features
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")
        
        # Modify the first convolutional layer to accept 4 channels instead of 3
        # Save the pretrained weights for the first 3 channels
        first_conv_weights = self.model.conv1.weight.data.clone()
        
        # Create a new conv layer with 4 input channels
        self.model.conv1 = nn.Conv2d(4, 64, kernel_size=7, stride=2, padding=3, bias=False)
        
        # Initialize the new conv layer with the pretrained weights for the first 3 channels
        # and the average of the pretrained weights for the 4th channel
        with torch.no_grad():
            self.model.conv1.weight[:, :3, :, :] = first_conv_weights
            self.model.conv1.weight[:, 3:, :, :] = torch.mean(first_conv_weights, dim=1, keepdim=True)
        
        # Replace the final FC layer with a new one
        self.model.fc = nn.Linear(self.feature_dim, self.output_dim)
        
        # Move model to device
        self.model.to(self.device)
        
        # Set model to evaluation mode
        self.model.eval()
    
    def extract_patch(self, volume, mask, patch_size=32):
        """
        Extract a patch centered on the superpixel.
        
        Args:
            volume: Multi-modal volume data (H, W, C)
            mask: Superpixel mask (H, W)
            patch_size: Size of the patch to extract
        
        Returns:
            Patch tensor (C, patch_size, patch_size)
        """
        # Find bounding box of the mask
        y_indices, x_indices = np.where(mask)
        if len(y_indices) == 0 or len(x_indices) == 0:
            # Empty mask, return zero patch
            return torch.zeros(volume.shape[-1], patch_size, patch_size)
        
        y_min, y_max = np.min(y_indices), np.max(y_indices)
        x_min, x_max = np.min(x_indices), np.max(x_indices)
        
        # Calculate center
        center_y = (y_min + y_max) // 2
        center_x = (x_min + x_max) // 2
        
        # Calculate patch boundaries
        half_size = patch_size // 2
        y_start = max(0, center_y - half_size)
        y_end = min(volume.shape[0], center_y + half_size)
        x_start = max(0, center_x - half_size)
        x_end = min(volume.shape[1], center_x + half_size)
        
        # Extract patch
        patch = volume[y_start:y_end, x_start:x_end]
        
        # Pad if necessary
        pad_y_before = max(0, half_size - center_y)
        pad_y_after = max(0, center_y + half_size - volume.shape[0])
        pad_x_before = max(0, half_size - center_x)
        pad_x_after = max(0, center_x + half_size - volume.shape[1])
        
        if pad_y_before > 0 or pad_y_after > 0 or pad_x_before > 0 or pad_x_after > 0:
            patch = np.pad(
                patch,
                ((pad_y_before, pad_y_after), (pad_x_before, pad_x_after), (0, 0)),
                mode='constant'
            )
        
        # Resize to patch_size x patch_size if necessary
        if patch.shape[0] != patch_size or patch.shape[1] != patch_size:
            patch = resize(patch, (patch_size, patch_size, volume.shape[-1]), 
                          order=1, preserve_range=True)
        
        # Convert to torch tensor (C, H, W)
        patch_tensor = torch.tensor(patch.transpose(2, 0, 1), dtype=torch.float)
        
        return patch_tensor
    
    def extract_features(self, slice_data, segments, batch_size=8):
        """
        Extract CNN features for all superpixels in a slice.
        
        Args:
            slice_data: Dict with multi-modal slice data (T1, T1ce, T2, FLAIR)
            segments: Superpixel segmentation for the slice
            batch_size: Batch size for feature extraction
        
        Returns:
            List of feature vectors for each superpixel
        """
        # Get unique superpixel labels (excluding background 0)
        superpixel_labels = np.unique(segments)
        superpixel_labels = superpixel_labels[superpixel_labels != 0]
        
        if len(superpixel_labels) == 0:
            # No superpixels in this slice
            return []
        
        # Create multi-channel slice volume
        multichannel_slice = np.stack([
            slice_data["T1"],
            slice_data["T1ce"],
            slice_data["T2"],
            slice_data["FLAIR"]
        ], axis=-1)
        
        # Extract patches for all superpixels
        patches = []
        for sp_id in superpixel_labels:
            mask = (segments == sp_id)
            patch = self.extract_patch(multichannel_slice, mask)
            patches.append(patch)
        
        # Process patches in batches
        all_features = []
        
        with torch.no_grad():
            for i in range(0, len(patches), batch_size):
                batch_patches = patches[i:i+batch_size]
                batch_tensor = torch.stack(batch_patches).to(self.device)
                
                # Normalize patches
                batch_tensor = (batch_tensor - batch_tensor.mean(dim=(2, 3), keepdim=True)) / (
                    batch_tensor.std(dim=(2, 3), keepdim=True) + 1e-6)
                
                # Extract features
                features = self.model(batch_tensor)
                all_features.append(features.cpu())
        
        # Concatenate all features
        if all_features:
            return torch.cat(all_features, dim=0).numpy()
        else:
            return []
    
    def process_graphs(self, graphs, volume_data, segments):
        """
        Process all graphs for a volume to extract CNN features.
        
        Args:
            graphs: List of PyG Data objects
            volume_data: Dict with volume data
            segments: List of superpixel segmentations
        
        Returns:
            List of updated PyG Data objects with CNN features
        """
        processed_graphs = []
        
        for graph in tqdm(graphs, desc="Extracting CNN features"):
            # Get slice indices
            slice_indices = graph.slice_indices.numpy()
            z1, z2 = slice_indices[0], slice_indices[1]
            
            # Get slice data - use UPPERCASE keys
            slice_data1 = {
                "T1": volume_data["T1"][z1],
                "T1ce": volume_data["T1ce"][z1],
                "T2": volume_data["T2"][z1],
                "FLAIR": volume_data["FLAIR"][z1],
            }
            
            slice_data2 = {
                "T1": volume_data["T1"][z2],
                "T1ce": volume_data["T1ce"][z2],
                "T2": volume_data["T2"][z2],
                "FLAIR": volume_data["FLAIR"][z2],
            }
            
            # Get slice segments
            segments1 = segments[z1]
            segments2 = segments[z2]
            
            # Extract CNN features
            features1 = self.extract_features(slice_data1, segments1)
            features2 = self.extract_features(slice_data2, segments2)
            
            if len(features1) == 0 or len(features2) == 0:
                # Skip empty slices
                continue
            
            # Combine features
            cnn_features = np.vstack([features1, features2])
            
            # Get original features
            orig_features = graph.x.numpy()
            
            # Ensure dimensions match
            if len(cnn_features) != len(orig_features):
                print(f"Warning: Feature dimensions mismatch. CNN: {len(cnn_features)}, Original: {len(orig_features)}")
                # Use the minimum length
                min_len = min(len(cnn_features), len(orig_features))
                cnn_features = cnn_features[:min_len]
                orig_features = orig_features[:min_len]
                
                # Update graph.y as well
                graph.y = graph.y[:min_len]
                
                # Update slice_mask
                graph.slice_mask = graph.slice_mask[:min_len]
            
            # Concatenate original features with CNN features
            combined_features = np.hstack([orig_features, cnn_features])
            
            # Update graph features
            graph.x = torch.tensor(combined_features, dtype=torch.float)
            
            processed_graphs.append(graph)
        
        return processed_graphs

def process_patient(graph_file, segments_file, npz_file, output_dir, backbone="resnet18", output_dim=128):
    """Process a single patient to extract CNN features."""
    try:
        patient_id = os.path.basename(graph_file).split('_graphs_')[0]
        n_superpixels = int(os.path.basename(graph_file).split('_graphs_')[1].split('.pt')[0])
        
        print(f"Processing patient {patient_id} with {n_superpixels} superpixels")
        
        # Load preprocessed data
        volume_data = np.load(npz_file)
        
        # Print keys in NPZ file for debugging
        print(f"Available keys in NPZ file: {list(volume_data.keys())}")
        
        # Load segments
        segments = np.load(segments_file)
        
        # Add PyTorch Geometric classes to safe globals for loading
        import torch.serialization
        from torch_geometric.data.data import Data
        
        # Add safe globals for PyG classes
        torch.serialization.add_safe_globals([
            Data,
            'torch_geometric.data.data.Data',
            'torch_geometric.data.data.DataEdgeAttr'
        ])
        
        # Load graphs with weights_only=False for backward compatibility
        graphs = torch.load(graph_file, weights_only=False)
        
        # Initialize feature extractor
        feature_extractor = FeatureExtractor(backbone=backbone, output_dim=output_dim)
        
        # Extract CNN features
        processed_graphs = feature_extractor.process_graphs(graphs, volume_data, segments)
        
        # Save processed graphs
        output_file = os.path.join(output_dir, f'{patient_id}_graphs_cnn_{n_superpixels}.pt')
        torch.save(processed_graphs, output_file)
        
        print(f"Saved processed graphs for {patient_id} with CNN features: {len(processed_graphs)} graphs")
        
        return patient_id, len(processed_graphs)
    
    except Exception as e:
        print(f"Error processing {graph_file}: {str(e)}")
        traceback.print_exc()  # Print the full stack trace
        return os.path.basename(graph_file), 0

def main():
    parser = argparse.ArgumentParser(description='Extract CNN features for superpixel graphs')
    parser.add_argument('--graph_dir', type=str, required=True, help='Directory with graph files')
    parser.add_argument('--preprocessed_dir', type=str, required=True, help='Directory with preprocessed .npz files')
    parser.add_argument('--output_dir', type=str, required=True, help='Output directory for processed graphs')
    parser.add_argument('--backbone', type=str, default='resnet18', choices=['resnet18', 'resnet34', 'resnet50'],
                        help='CNN backbone model')
    parser.add_argument('--output_dim', type=int, default=128, help='Output feature dimension')
    parser.add_argument('--num_workers', type=int, default=1, help='Number of parallel workers')
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Find all graph files
    graph_files = glob.glob(os.path.join(args.graph_dir, "*/*_graphs_*.pt"))
    
    if not graph_files:
        print(f"No graph files found in {args.graph_dir}")
        return
    
    print(f"Found {len(graph_files)} graph files")
    
    # Process each patient
    if args.num_workers > 1 and len(graph_files) > 1:
        with ProcessPoolExecutor(max_workers=args.num_workers) as executor:
            futures = []
            
            for graph_file in graph_files:
                patient_id = os.path.basename(graph_file).split('_graphs_')[0]
                n_superpixels = os.path.basename(graph_file).split('_graphs_')[1].split('.pt')[0]
                
                # Find corresponding segments file
                segments_file = os.path.join(os.path.dirname(graph_file), f'{patient_id}_segments_{n_superpixels}.npy')
                
                # Find corresponding preprocessed file
                preprocessed_dir = os.path.join(args.preprocessed_dir, patient_id)
                npz_file = os.path.join(preprocessed_dir, f'{patient_id}_preprocessed.npz')
                
                if not os.path.exists(segments_file) or not os.path.exists(npz_file):
                    print(f"Warning: Missing files for patient {patient_id}")
                    continue
                
                futures.append(
                    executor.submit(
                        process_patient,
                        graph_file,
                        segments_file,
                        npz_file,
                        args.output_dir,
                        args.backbone,
                        args.output_dim
                    )
                )
            
            results = []
            for future in tqdm(futures, desc="Extracting CNN features"):
                results.append(future.result())
    else:
        results = []
        for graph_file in tqdm(graph_files, desc="Extracting CNN features"):
            patient_id = os.path.basename(graph_file).split('_graphs_')[0]
            n_superpixels = os.path.basename(graph_file).split('_graphs_')[1].split('.pt')[0]
            
            # Find corresponding segments file
            segments_file = os.path.join(os.path.dirname(graph_file), f'{patient_id}_segments_{n_superpixels}.npy')
            
            # Find corresponding preprocessed file
            preprocessed_dir = os.path.join(args.preprocessed_dir, patient_id)
            npz_file = os.path.join(preprocessed_dir, f'{patient_id}_preprocessed.npz')
            
            if not os.path.exists(segments_file) or not os.path.exists(npz_file):
                print(f"Warning: Missing files for patient {patient_id}")
                continue
            
            results.append(process_patient(
                graph_file,
                segments_file,
                npz_file,
                args.output_dir,
                args.backbone,
                args.output_dim
            ))
    
    # Report results
    successful = [r for r in results if r[1] > 0]
    print(f"Successfully processed {len(successful)} patients")
    print(f"Failed to process {len(results) - len(successful)} patients")

if __name__ == "__main__":
    start_time = time.time()
    main()
    print(f"Total feature extraction time: {(time.time() - start_time) / 60:.2f} minutes")