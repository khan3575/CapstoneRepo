# src/graph_construction.py
import os
import numpy as np
import torch
from torch_geometric.data import Data
from skimage.segmentation import slic
from skimage.measure import regionprops, label
from skimage.util import img_as_float
from scipy.ndimage import center_of_mass, zoom
from tqdm import tqdm
import matplotlib.pyplot as plt
from pathlib import Path
import argparse
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import networkx as nx
import time
import glob
import traceback
import gc
import psutil

# Global memory tracking
def get_memory_usage():
    """Get current memory usage in MB."""
    try:
        import psutil
        process = psutil.Process(os.getpid())
        return process.memory_info().rss / (1024 * 1024)
    except ImportError:
        return 0.0  # Return 0 if psutil not available

def downsample_volume(volume, factor=2):
    """Downsample volume by a factor."""
    # Skip if already small
    if volume.shape[1] <= 128 or volume.shape[2] <= 128:
        return volume
    return zoom(volume, (1, 1/factor, 1/factor), order=1)

class SuperpixelGraphBuilder:
    def __init__(self, n_superpixels=400, sigma=0.5, compactness=0.1, 
                 max_iter=50, inter_slice_threshold=5.0, iou_threshold=0.3,
                 sparse_edges=True, sparse_factor=0.8, process_every_n=1):
        """
        Initialize the superpixel graph builder with optimized parameters.
        
        Args:
            n_superpixels: Number of superpixels per slice
            sigma: Width of Gaussian smoothing kernel for SLIC (reduced for speed)
            compactness: Compactness parameter for SLIC
            max_iter: Maximum SLIC iterations
            inter_slice_threshold: Distance threshold for inter-slice connections
            iou_threshold: IoU threshold for inter-slice connections
            sparse_edges: Whether to create sparse graphs
            sparse_factor: Percentage of edges to keep (0.8 = 80%)
            process_every_n: Process every Nth slice for speed
        """
        self.n_superpixels = n_superpixels
        self.sigma = sigma
        self.compactness = compactness
        self.max_iter = max_iter
        self.inter_slice_threshold = inter_slice_threshold
        self.iou_threshold = iou_threshold
        self.sparse_edges = sparse_edges
        self.sparse_factor = sparse_factor
        self.process_every_n = process_every_n
    
    def _compute_superpixels(self, slice_data):
        """Compute superpixels for a single multi-modal slice - optimized."""
        # Combine the 4 modalities into a multi-channel image
        multichannel_slice = np.stack([
            img_as_float(slice_data["T1"]),
            img_as_float(slice_data["T1ce"]),
            img_as_float(slice_data["T2"]),
            img_as_float(slice_data["FLAIR"])
        ], axis=-1)
        
        # Create a mask for brain regions
        mask = slice_data["brain_mask"] > 0
        
        # Compute SLIC superpixels only within the brain mask
        segments = np.zeros(mask.shape, dtype=np.int32)
        if np.any(mask):  # Only compute if mask has non-zero elements
            # In newer versions of scikit-image, 'multichannel' parameter was replaced with 'channel_axis'
            try:
                segments[mask] = slic(
                    multichannel_slice[mask].reshape(-1, 1, 4),  # Reshape for slic
                    n_segments=self.n_superpixels,
                    sigma=self.sigma,
                    compactness=self.compactness,
                    channel_axis=-1,  # Use channel_axis for newer scikit-image versions
                    max_iter=self.max_iter,
                    start_label=1,
                ).flatten()
            except TypeError:
                # Fall back to older parameter name if needed
                segments[mask] = slic(
                    multichannel_slice[mask].reshape(-1, 1, 4),
                    n_segments=self.n_superpixels,
                    sigma=self.sigma,
                    compactness=self.compactness,
                    multichannel=True,  # For older scikit-image versions
                    max_iter=self.max_iter,
                    start_label=1,
                ).flatten()
            
            # Relabel segments to ensure consecutive numbering
            unique_labels = np.unique(segments)
            if len(unique_labels) > 1 and unique_labels[0] == 0:
                unique_labels = unique_labels[1:]
            
            if len(unique_labels) > 0:
                mapping = {old: new+1 for new, old in enumerate(unique_labels)}
                mapped_segments = np.zeros_like(segments)
                for old, new in mapping.items():
                    mapped_segments[segments == old] = new
                segments = mapped_segments
        
        return segments
    
    def _compute_node_features(self, slice_data, segments):
        """Compute basic node features for each superpixel - optimized."""
        # Get unique superpixel labels (excluding background 0)
        superpixel_labels = np.unique(segments)
        superpixel_labels = superpixel_labels[superpixel_labels != 0]
        
        if len(superpixel_labels) == 0:
            # No superpixels in this slice
            return [], [], []
        
        # Initialize features list, coordinates list, and superpixel masks list
        features = []
        centroids = []
        superpixel_masks = []
        
        # For each superpixel, compute features
        for sp_id in superpixel_labels:
            # Create mask for this superpixel
            sp_mask = (segments == sp_id)
            superpixel_masks.append(sp_mask)
            
            # Calculate centroid
            props = regionprops(sp_mask.astype(int))
            if len(props) == 0:
                continue  # Skip if empty region
                
            centroid = props[0].centroid
            centroids.append(centroid)
            
            # Calculate mean intensity for each modality
            t1_mean = np.mean(slice_data["T1"][sp_mask])
            t1ce_mean = np.mean(slice_data["T1ce"][sp_mask])
            t2_mean = np.mean(slice_data["T2"][sp_mask])
            flair_mean = np.mean(slice_data["FLAIR"][sp_mask])
            
            # Calculate standard deviation for each modality
            t1_std = np.std(slice_data["T1"][sp_mask])
            t1ce_std = np.std(slice_data["T1ce"][sp_mask])
            t2_std = np.std(slice_data["T2"][sp_mask])
            flair_std = np.std(slice_data["FLAIR"][sp_mask])
            
            # Calculate ratio of tumor pixels (from ground truth label)
            tumor_ratio = np.mean(slice_data["label"][sp_mask])
            
            # Calculate shape properties - only essential ones
            area = props[0].area
            perimeter = props[0].perimeter if hasattr(props[0], 'perimeter') else 0
            
            # Skip eccentricity calculation for speed
            eccentricity = 0.0
            
            # Create feature vector
            feature = [
                t1_mean, t1ce_mean, t2_mean, flair_mean,
                t1_std, t1ce_std, t2_std, flair_std,
                area, perimeter, eccentricity,
                centroid[0] / slice_data["T1"].shape[0],  # Normalized y-coordinate
                centroid[1] / slice_data["T1"].shape[1],  # Normalized x-coordinate
                tumor_ratio
            ]
            features.append(feature)
        
        return features, centroids, superpixel_masks
    
    def _build_intra_slice_edges(self, segments):
        """Build adjacency graph for superpixels within a slice - with sparsity."""
        # Create a graph to represent superpixel adjacency
        G = nx.Graph()
        
        # Get unique superpixel labels (excluding background 0)
        superpixel_labels = np.unique(segments)
        superpixel_labels = superpixel_labels[superpixel_labels != 0]
        
        if len(superpixel_labels) == 0:
            # No superpixels in this slice
            return [], {}
        
        # Add nodes to the graph
        for sp_id in superpixel_labels:
            G.add_node(sp_id)
        
        # Find adjacent superpixels - with sparsity
        h, w = segments.shape
        for y in range(h-1):
            for x in range(w-1):
                current = segments[y, x]
                if current == 0:  # Skip background
                    continue
                
                # Check right neighbor - with sparsity
                right = segments[y, x+1]
                if right != 0 and right != current:
                    if not self.sparse_edges or np.random.random() < self.sparse_factor:
                        G.add_edge(current, right)
                
                # Check bottom neighbor - with sparsity
                bottom = segments[y+1, x]
                if bottom != 0 and bottom != current:
                    if not self.sparse_edges or np.random.random() < self.sparse_factor:
                        G.add_edge(current, bottom)
        
        # Convert to edge list with 0-indexed nodes
        edges = []
        label_to_idx = {label: i for i, label in enumerate(superpixel_labels)}
        
        for u, v in G.edges():
            try:
                edges.append((label_to_idx[u], label_to_idx[v]))
                edges.append((label_to_idx[v], label_to_idx[u]))  # Add both directions for undirected graph
            except KeyError:
                # Skip edges with labels that aren't in label_to_idx
                pass
        
        return edges, label_to_idx
    
    def _build_inter_slice_edges(self, curr_centroids, next_centroids, 
                               curr_masks, next_masks,
                               curr_label_to_idx, next_label_to_idx):
        """Build edges between superpixels in adjacent slices - optimized."""
        inter_slice_edges = []
        
        # If either slice has no superpixels, return empty list
        if len(curr_centroids) == 0 or len(next_centroids) == 0:
            return inter_slice_edges
        
        # Create reverse mappings for safer lookup
        curr_idx_to_label = {}
        for label, idx in curr_label_to_idx.items():
            curr_idx_to_label[idx] = label
            
        next_idx_to_label = {}
        for label, idx in next_label_to_idx.items():
            next_idx_to_label[idx] = label
        
        # Use a sparse sampling approach for inter-slice edges
        # Sample a subset of current centroids for efficiency
        if len(curr_centroids) > 100:
            sampled_indices = np.random.choice(
                range(len(curr_centroids)), 
                size=min(len(curr_centroids), 100),  # Limit to at most 100 samples
                replace=False
            )
        else:
            sampled_indices = list(range(len(curr_centroids)))
        
        # For each sampled superpixel in current slice
        for i in sampled_indices:
            if i not in curr_idx_to_label:
                continue
                
            curr_centroid = curr_centroids[i]
            curr_mask = curr_masks[i]
            curr_y, curr_x = curr_centroid
            curr_label = curr_idx_to_label[i]
            
            # Sample a subset of next centroids for efficiency
            if len(next_centroids) > 100:
                next_sampled_indices = np.random.choice(
                    range(len(next_centroids)), 
                    size=min(len(next_centroids), 100),  # Limit to at most 100 samples
                    replace=False
                )
            else:
                next_sampled_indices = list(range(len(next_centroids)))
            
            # For each sampled superpixel in next slice
            for j in next_sampled_indices:
                if j not in next_idx_to_label:
                    continue
                    
                next_centroid = next_centroids[j]
                next_mask = next_masks[j]
                next_y, next_x = next_centroid
                next_label = next_idx_to_label[j]
                
                # Calculate Euclidean distance between centroids
                distance = np.sqrt((curr_y - next_y)**2 + (curr_x - next_x)**2)
                
                # Only calculate IoU if distance is below threshold - saves computation
                if distance < self.inter_slice_threshold:
                    # Apply sparsity - randomly skip some edges
                    if self.sparse_edges and np.random.random() > self.sparse_factor:
                        continue
                        
                    # Calculate IoU between masks
                    intersection = np.logical_and(curr_mask, next_mask).sum()
                    union = np.logical_or(curr_mask, next_mask).sum()
                    iou = intersection / union if union > 0 else 0
                    
                    # Connect if distance is below threshold or IoU is above threshold
                    if iou > self.iou_threshold:
                        try:
                            inter_slice_edges.append((
                                curr_label_to_idx[curr_label], 
                                next_label_to_idx[next_label] + len(curr_label_to_idx)
                            ))
                            
                            # For bidirectional graph
                            inter_slice_edges.append((
                                next_label_to_idx[next_label] + len(curr_label_to_idx),
                                curr_label_to_idx[curr_label]
                            ))
                        except KeyError:
                            # Skip if there's a mismatch in labels
                            pass
        
        return inter_slice_edges
    
    def build_graphs(self, volume_data, save_dir=None, visualize=False):
        """
        Build graphs for all slices in the volume - optimized.
        
        Args:
            volume_data: Dict with keys T1, T1ce, T2, FLAIR, label, brain_mask
            save_dir: Directory to save visualizations (optional)
            visualize: Whether to generate visualizations
        
        Returns:
            List of torch_geometric.data.Data objects for each slice or slice pair
        """
        # Extract volume dimensions
        n_slices = volume_data["T1"].shape[0]
        
        # Initialize lists to store results
        all_graphs = []
        all_segments = []
        all_features = []
        all_centroids = []
        all_superpixel_masks = []
        all_label_to_idx = []
        
        # Select slices to process
        slices_to_process = []
        for z in range(n_slices):
            # Process slices with tumor or every Nth slice (based on process_every_n)
            if np.any(volume_data["label"][z] > 0) or z % self.process_every_n == 0:
                slices_to_process.append(z)
        
        # Process each selected slice
        for z in tqdm(slices_to_process, desc="Processing slices"):
            if z >= n_slices:
                continue
                
            # Create slice data dictionary
            slice_data = {
                "T1": volume_data["T1"][z],
                "T1ce": volume_data["T1ce"][z],
                "T2": volume_data["T2"][z],
                "FLAIR": volume_data["FLAIR"][z],
                "label": volume_data["label"][z],
                "brain_mask": volume_data["brain_mask"][z]
            }
            
            # Compute superpixels
            segments = self._compute_superpixels(slice_data)
            all_segments.append(segments)
            
            # Compute node features
            features, centroids, sp_masks = self._compute_node_features(slice_data, segments)
            all_features.append(features)
            all_centroids.append(centroids)
            all_superpixel_masks.append(sp_masks)
            
            # Build intra-slice edges
            intra_edges, label_to_idx = self._build_intra_slice_edges(segments)
            all_label_to_idx.append(label_to_idx)
            
            # Visualize if requested - but only for slices with tumors to save time
            if visualize and save_dir and np.any(slice_data["label"] > 0):
                self._visualize_superpixels(slice_data, segments, z, save_dir)
            
            # Free memory after each slice
            for key in slice_data:
                slice_data[key] = None
            del slice_data
        
        # Build inter-slice edges and create graph objects
        processed_slices = slices_to_process
        for i in tqdm(range(len(processed_slices) - 1), desc="Building inter-slice connections"):
            z = processed_slices[i]
            next_z = processed_slices[i+1]
            
            # Skip if slices are too far apart
            if next_z - z > 3:  # Skip if slices are more than 3 apart
                continue
                
            # Use direct indices instead of searching
            z_idx = i
            next_z_idx = i + 1
            
            if (z_idx >= len(all_features) or next_z_idx >= len(all_features) or
                len(all_features[z_idx]) == 0 or len(all_features[next_z_idx]) == 0):
                continue
            
            # Build inter-slice edges
            inter_edges = self._build_inter_slice_edges(
                all_centroids[z_idx], all_centroids[next_z_idx],
                all_superpixel_masks[z_idx], all_superpixel_masks[next_z_idx],
                all_label_to_idx[z_idx], all_label_to_idx[next_z_idx]
            )
            
            # Combine intra-slice edges
            intra_edges_z = []
            try:
                for u, v in self._build_intra_slice_edges(all_segments[z_idx])[0]:
                    intra_edges_z.append((u, v))
            except (ValueError, IndexError):
                pass  # Empty edges list or index error
                
            intra_edges_z_plus_1 = []
            try:
                for u, v in self._build_intra_slice_edges(all_segments[next_z_idx])[0]:
                    intra_edges_z_plus_1.append((u + len(all_features[z_idx]), v + len(all_features[z_idx])))
            except (ValueError, IndexError):
                pass  # Empty edges list or index error
            
            # Combine all edges
            all_edges = intra_edges_z + intra_edges_z_plus_1 + inter_edges
            
            # Skip if no edges (can happen with small or noisy slices)
            if not all_edges:
                continue
            
            # Combine node features
            all_nodes_features = all_features[z_idx] + all_features[next_z_idx]
            
            # Skip if no features
            if not all_nodes_features:
                continue
            
            # Extract node labels (tumor ratio is the last feature)
            node_labels = np.array([f[-1] > 0.5 for f in all_nodes_features], dtype=np.float32)
            
            # Create a PyTorch Geometric Data object
            try:
                edge_index = torch.tensor(all_edges, dtype=torch.long).t().contiguous()
                x = torch.tensor(all_nodes_features, dtype=torch.float)
                y = torch.tensor(node_labels, dtype=torch.float)
                
                # Create mask for nodes to identify which slice they belong to
                slice_mask = torch.zeros(len(all_nodes_features), dtype=torch.bool)
                slice_mask[:len(all_features[z_idx])] = True
                
                # Store slice indices for reference
                slice_indices = torch.tensor([z, next_z], dtype=torch.long)
                
                graph_data = Data(
                    x=x,
                    edge_index=edge_index,
                    y=y,
                    slice_mask=slice_mask,
                    slice_indices=slice_indices
                )
                
                all_graphs.append(graph_data)
            except RuntimeError as e:
                print(f"Error creating graph for slices {z} and {next_z}: {str(e)}")
            
            # Force memory cleanup
            if i % 10 == 0:
                gc.collect()
        
        # Final cleanup
        all_centroids.clear()
        all_superpixel_masks.clear()
        all_label_to_idx.clear()
        gc.collect()
        
        return all_graphs, all_segments
    
    def _visualize_superpixels(self, slice_data, segments, slice_idx, save_dir):
        """Generate visualization of superpixels for a slice - optimized."""
        os.makedirs(save_dir, exist_ok=True)
        
        try:
            # Create smaller figure with lower dpi for memory efficiency
            fig, axs = plt.subplots(2, 3, figsize=(10, 6), dpi=80)
            
            # Display the original T1ce image
            axs[0, 0].imshow(slice_data["T1ce"], cmap='gray')
            axs[0, 0].set_title('T1ce')
            axs[0, 0].axis('off')
            
            # Display the FLAIR image
            axs[0, 1].imshow(slice_data["FLAIR"], cmap='gray')
            axs[0, 1].set_title('FLAIR')
            axs[0, 1].axis('off')
            
            # Display the label
            axs[0, 2].imshow(slice_data["label"], cmap='hot')
            axs[0, 2].set_title('Tumor Mask')
            axs[0, 2].axis('off')
            
            # Display the brain mask
            axs[1, 0].imshow(slice_data["brain_mask"], cmap='gray')
            axs[1, 0].set_title('Brain Mask')
            axs[1, 0].axis('off')
            
            # Display the superpixels
            from skimage.color import label2rgb
            overlay = label2rgb(segments, slice_data["T1ce"], kind='avg', bg_label=0)
            axs[1, 1].imshow(overlay)
            axs[1, 1].set_title(f'Superpixels (n={self.n_superpixels})')
            axs[1, 1].axis('off')
            
            # Display superpixel boundaries
            from skimage.segmentation import mark_boundaries
            boundaries = mark_boundaries(slice_data["T1ce"], segments)
            axs[1, 2].imshow(boundaries)
            axs[1, 2].set_title('Superpixel Boundaries')
            axs[1, 2].axis('off')
            
            # Save the figure
            plt.tight_layout()
            plt.savefig(os.path.join(save_dir, f'superpixels_slice_{slice_idx}.png'), dpi=80)
            plt.close(fig)
            
            # Explicitly clear matplotlib's memory
            plt.clf()
            plt.close('all')
            
        except Exception as e:
            print(f"Error visualizing slice {slice_idx}: {str(e)}")

def process_in_batches(patient_files, args):
    """Process patients in small batches to manage memory usage."""
    batch_size = args.batch_size if args.batch_size > 0 else max(1, min(4, args.num_workers // 3))
    total_patients = len(patient_files)
    batches = [patient_files[i:i+batch_size] for i in range(0, total_patients, batch_size)]
    
    print(f"Processing {len(batches)} batches with {batch_size} patients per batch")
    all_results = []
    
    for i, batch in enumerate(batches):
        memory_usage = get_memory_usage()
        if memory_usage > 0:
            print(f"Batch {i+1}/{len(batches)} ({len(batch)} patients) - Memory: {memory_usage:.1f} MB")
        else:
            print(f"Batch {i+1}/{len(batches)} ({len(batch)} patients)")
        
        # Process batch with parallelization
        if args.num_workers > 1 and not args.low_memory:
            with ProcessPoolExecutor(max_workers=min(args.num_workers, len(batch))) as executor:
                futures = [
                    executor.submit(
                        process_patient, 
                        patient_file, 
                        args.output_dir,
                        args.superpixels,
                        args.visualize,
                        args.downsample_factor,
                        args.process_every_n,
                        args.sparse_edges
                    ) 
                    for patient_file in batch
                ]
                
                batch_results = []
                for future in tqdm(futures, total=len(batch), desc="Building graphs"):
                    batch_results.append(future.result())
        else:
            # Sequential processing
            batch_results = []
            for patient_file in tqdm(batch, desc="Building graphs"):
                batch_results.append(process_patient(
                    patient_file, 
                    args.output_dir,
                    args.superpixels,
                    args.visualize,
                    args.downsample_factor,
                    args.process_every_n,
                    args.sparse_edges
                ))
        
        all_results.extend(batch_results)
        
        # Force garbage collection between batches
        gc.collect()
        
    return all_results

def process_patient(patient_file, output_dir, n_superpixels=400, visualize=False, 
                    downsample_factor=1, process_every_n=1, sparse_edges=False):
    """Process a single patient's data to build graphs - optimized."""
    start_time = time.time()
    
    try:
        # Load preprocessed data (fallback from memory mapping if it fails)
        try:
            data = np.load(patient_file, mmap_mode='r')  # Try memory mapping first
        except (ValueError, OSError):
            data = np.load(patient_file)  # Fallback to normal loading
        patient_id = os.path.basename(patient_file).split('_preprocessed.npz')[0]
        
        print(f"Data keys for {patient_id}: {list(data.keys())}")
        
        # Create output directory
        patient_output_dir = os.path.join(output_dir, patient_id)
        os.makedirs(patient_output_dir, exist_ok=True)
        
        # Create visualization directory if needed
        vis_dir = os.path.join(patient_output_dir, 'visualizations') if visualize else None
        if vis_dir:
            os.makedirs(vis_dir, exist_ok=True)
        
        # Handle different possible key names in the data
        volume_data = {}
        # Check for different possible key names
        for key in ['T1', 't1']:
            if key in data:
                volume_data['T1'] = data[key]
                break
        for key in ['T1ce', 't1ce']:
            if key in data:
                volume_data['T1ce'] = data[key]
                break
        for key in ['T2', 't2']:
            if key in data:
                volume_data['T2'] = data[key]
                break
        for key in ['FLAIR', 'flair']:
            if key in data:
                volume_data['FLAIR'] = data[key]
                break
        for key in ['label', 'seg']:
            if key in data:
                volume_data['label'] = data[key]
                break
        
        # Create brain mask if not present
        if 'brain_mask' in data:
            volume_data['brain_mask'] = data['brain_mask']
        else:
            # Create a simple brain mask from the T1 image (non-zero values)
            volume_data['brain_mask'] = (volume_data['T1'] > 0).astype(np.float32)
            print("Created brain mask from T1 data")
        
        # Check if all required keys are present
        required_keys = ['T1', 'T1ce', 'T2', 'FLAIR', 'label', 'brain_mask']
        missing_keys = [key for key in required_keys if key not in volume_data]
        if missing_keys:
            print(f"Warning: Missing keys in data: {missing_keys}")
            return patient_id, 0
        
        # Downsample data if requested
        if downsample_factor > 1:
            for key in volume_data:
                if key != 'brain_mask':  # Don't downsample mask
                    volume_data[key] = downsample_volume(volume_data[key], downsample_factor)
                    print(f"Downsampled {key} to {volume_data[key].shape}")
        
        # Initialize graph builder with optimized parameters
        graph_builder = SuperpixelGraphBuilder(
            n_superpixels=n_superpixels,
            sigma=0.5,          # Reduced for speed
            compactness=0.1,
            max_iter=50,        # Limited iterations
            inter_slice_threshold=5.0,
            iou_threshold=0.2,  # Reduced threshold for more connections
            sparse_edges=sparse_edges,
            sparse_factor=0.8,  # Keep 80% of edges
            process_every_n=process_every_n
        )
        
        # Build graphs
        graphs, segments = graph_builder.build_graphs(
            volume_data=volume_data,
            save_dir=vis_dir,
            visualize=visualize
        )
        
        # Save graphs if any were created
        if len(graphs) > 0:
            graph_file = os.path.join(patient_output_dir, f'{patient_id}_graphs_{n_superpixels}.pt')
            torch.save(graphs, graph_file)
            
            # Save segments for later use in reprojection
            segments_file = os.path.join(patient_output_dir, f'{patient_id}_segments_{n_superpixels}.npy')
            np.save(segments_file, segments)
            
            processing_time = (time.time() - start_time) / 60
            print(f"Processed patient {patient_id}: created {len(graphs)} graphs in {processing_time:.2f} minutes")
        else:
            print(f"Warning: No graphs were created for patient {patient_id}")
        
        # Clean up memory
        for key in list(volume_data.keys()):
            volume_data[key] = None
        volume_data.clear()
        data = None
        gc.collect()
        
        return patient_id, len(graphs)
    
    except Exception as e:
        print(f"Error processing {patient_file}: {str(e)}")
        print(traceback.format_exc())  # Print the full stack trace
        return os.path.basename(patient_file), 0

def main():
    parser = argparse.ArgumentParser(description='Build superpixel graphs from preprocessed BraTS data')
    parser.add_argument('--input_dir', type=str, required=True, help='Directory with preprocessed .npz files')
    parser.add_argument('--output_dir', type=str, required=True, help='Output directory for graphs')
    parser.add_argument('--superpixels', type=int, default=400, choices=[100, 200, 300, 400, 600, 800], 
                        help='Number of superpixels per slice')
    parser.add_argument('--visualize', action='store_true', help='Generate superpixel visualizations')
    parser.add_argument('--num_workers', type=int, default=1, help='Number of parallel workers')
    parser.add_argument('--low_memory', action='store_true', help='Use sequential processing to prevent memory issues')
    parser.add_argument('--batch_size', type=int, default=0, help='Patients to process per batch (0=auto)')
    parser.add_argument('--downsample_factor', type=int, default=1, choices=[1, 2, 4], 
                       help='Factor to downsample images (1=no downsampling)')
    parser.add_argument('--process_every_n', type=int, default=1, choices=[1, 2, 3, 5], 
                       help='Process every Nth slice (1=all slices)')
    parser.add_argument('--sparse_edges', action='store_true', 
                       help='Create sparse graphs with fewer edges')
    args = parser.parse_args()
    
    # Print initial memory usage
    initial_memory = get_memory_usage()
    if initial_memory > 0:
        print(f"Initial memory usage: {initial_memory:.1f} MB")
    else:
        print("Memory tracking disabled (psutil not available)")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Get all preprocessed patient files
    patient_files = glob.glob(os.path.join(args.input_dir, "*/*_preprocessed.npz"))
    
    print(f"Found {len(patient_files)} preprocessed patient files")
    
    # Process patients in batches to manage memory
    results = process_in_batches(patient_files, args)
    
    # Report results
    successful = [r for r in results if r[1] > 0]
    print(f"Successfully processed {len(successful)} patients")
    print(f"Failed to process {len(results) - len(successful)} patients")
    
    # Calculate total graphs
    total_graphs = sum(r[1] for r in results)
    print(f"Total graphs created: {total_graphs}")
    
    # Final memory usage
    final_memory = get_memory_usage()
    if final_memory > 0:
        print(f"Final memory usage: {final_memory:.1f} MB")

if __name__ == "__main__":
    start_time = time.time()
    main()
    print(f"Total graph building time: {(time.time() - start_time) / 60:.2f} minutes")