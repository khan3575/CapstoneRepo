# src/graph_construction.py
import os
import numpy as np
import torch
from torch_geometric.data import Data
from skimage.segmentation import slic
from skimage.measure import regionprops, label
from skimage.util import img_as_float
from scipy.ndimage import center_of_mass
from tqdm import tqdm
import matplotlib.pyplot as plt
from pathlib import Path
import argparse
from concurrent.futures import ProcessPoolExecutor
import networkx as nx
import time
import glob
import traceback

class SuperpixelGraphBuilder:
    def __init__(self, n_superpixels=400, sigma=1, compactness=0.1, 
                 inter_slice_threshold=5.0, iou_threshold=0.3):
        """
        Initialize the superpixel graph builder.
        
        Args:
            n_superpixels: Number of superpixels per slice
            sigma: Width of Gaussian smoothing kernel for SLIC
            compactness: Compactness parameter for SLIC
            inter_slice_threshold: Distance threshold for inter-slice connections
            iou_threshold: IoU threshold for inter-slice connections
        """
        self.n_superpixels = n_superpixels
        self.sigma = sigma
        self.compactness = compactness
        self.inter_slice_threshold = inter_slice_threshold
        self.iou_threshold = iou_threshold
    
    def _compute_superpixels(self, slice_data):
        """Compute superpixels for a single multi-modal slice."""
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
        """
        Compute basic node features for each superpixel.
        
        These are initial features before CNN extraction.
        """
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
            
            # Calculate shape properties
            area = props[0].area
            perimeter = props[0].perimeter
            
            # Some properties might be missing in older versions
            try:
                eccentricity = props[0].eccentricity
            except AttributeError:
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
        """Build adjacency graph for superpixels within a slice."""
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
        
        # Find adjacent superpixels
        h, w = segments.shape
        for y in range(h-1):
            for x in range(w-1):
                current = segments[y, x]
                if current == 0:  # Skip background
                    continue
                
                # Check right neighbor
                right = segments[y, x+1]
                if right != 0 and right != current:
                    G.add_edge(current, right)
                
                # Check bottom neighbor
                bottom = segments[y+1, x]
                if bottom != 0 and bottom != current:
                    G.add_edge(current, bottom)
        
        # Convert to edge list with 0-indexed nodes
        edges = []
        label_to_idx = {label: i for i, label in enumerate(superpixel_labels)}
        
        for u, v in G.edges():
            try:
                edges.append((label_to_idx[u], label_to_idx[v]))
                edges.append((label_to_idx[v], label_to_idx[u]))  # Add both directions for undirected graph
            except KeyError:
                # Skip edges with labels that aren't in label_to_idx (shouldn't happen, but just in case)
                pass
        
        return edges, label_to_idx
    
    def _build_inter_slice_edges(self, curr_centroids, next_centroids, 
                               curr_masks, next_masks,
                               curr_label_to_idx, next_label_to_idx):
        """Build edges between superpixels in adjacent slices."""
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
        
        # For each superpixel in current slice
        for i in range(len(curr_centroids)):
            if i >= len(curr_centroids) or i not in curr_idx_to_label:
                continue
                
            curr_centroid = curr_centroids[i]
            curr_mask = curr_masks[i]
            curr_y, curr_x = curr_centroid
            curr_label = curr_idx_to_label[i]
            
            # For each superpixel in next slice
            for j in range(len(next_centroids)):
                if j >= len(next_centroids) or j not in next_idx_to_label:
                    continue
                    
                next_centroid = next_centroids[j]
                next_mask = next_masks[j]
                next_y, next_x = next_centroid
                next_label = next_idx_to_label[j]
                
                # Calculate Euclidean distance between centroids
                distance = np.sqrt((curr_y - next_y)**2 + (curr_x - next_x)**2)
                
                # Calculate IoU between masks
                intersection = np.logical_and(curr_mask, next_mask).sum()
                union = np.logical_or(curr_mask, next_mask).sum()
                iou = intersection / union if union > 0 else 0
                
                # Connect if distance is below threshold or IoU is above threshold
                if distance < self.inter_slice_threshold or iou > self.iou_threshold:
                    try:
                        inter_slice_edges.append((
                            curr_label_to_idx[curr_label], 
                            next_label_to_idx[next_label] + len(curr_label_to_idx)
                        ))
                        
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
        Build graphs for all slices in the volume.
        
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
        
        # Process each slice
        for z in tqdm(range(n_slices), desc="Processing slices"):
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
            
            # Visualize if requested
            if visualize and save_dir:
                self._visualize_superpixels(slice_data, segments, z, save_dir)
        
        # Build inter-slice edges and create graph objects
        for z in tqdm(range(n_slices - 1), desc="Building inter-slice connections"):
            # Skip if either slice has no superpixels
            if len(all_features[z]) == 0 or len(all_features[z+1]) == 0:
                continue
            
            # Build inter-slice edges
            inter_edges = self._build_inter_slice_edges(
                all_centroids[z], all_centroids[z+1],
                all_superpixel_masks[z], all_superpixel_masks[z+1],
                all_label_to_idx[z], all_label_to_idx[z+1]
            )
            
            # Combine intra-slice edges
            intra_edges_z = []
            try:
                for u, v in self._build_intra_slice_edges(all_segments[z])[0]:
                    intra_edges_z.append((u, v))
            except ValueError:
                pass  # Empty edges list
                
            intra_edges_z_plus_1 = []
            try:
                for u, v in self._build_intra_slice_edges(all_segments[z+1])[0]:
                    intra_edges_z_plus_1.append((u + len(all_features[z]), v + len(all_features[z])))
            except ValueError:
                pass  # Empty edges list
            
            # Combine all edges
            all_edges = intra_edges_z + intra_edges_z_plus_1 + inter_edges
            
            # Skip if no edges (can happen with small or noisy slices)
            if not all_edges:
                continue
            
            # Combine node features
            all_nodes_features = all_features[z] + all_features[z+1]
            
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
                slice_mask[:len(all_features[z])] = True
                
                # Store slice indices for reference
                slice_indices = torch.tensor([z, z+1], dtype=torch.long)
                
                graph_data = Data(
                    x=x,
                    edge_index=edge_index,
                    y=y,
                    slice_mask=slice_mask,
                    slice_indices=slice_indices
                )
                
                all_graphs.append(graph_data)
            except RuntimeError as e:
                print(f"Error creating graph for slices {z} and {z+1}: {str(e)}")
        
        return all_graphs, all_segments
    
    def _visualize_superpixels(self, slice_data, segments, slice_idx, save_dir):
        """Generate visualization of superpixels for a slice."""
        os.makedirs(save_dir, exist_ok=True)
        
        try:
            # Create figure with subplots
            fig, axs = plt.subplots(2, 3, figsize=(15, 10))
            
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
            plt.savefig(os.path.join(save_dir, f'superpixels_slice_{slice_idx}.png'), dpi=150)
            plt.close(fig)
        except Exception as e:
            print(f"Error visualizing slice {slice_idx}: {str(e)}")

def process_patient(patient_file, output_dir, n_superpixels=400, visualize=False):
    """Process a single patient's data to build graphs."""
    try:
        # Load preprocessed data
        data = np.load(patient_file)
        patient_id = os.path.basename(patient_file).split('_preprocessed.npz')[0]
        
        print(f"Data keys: {list(data.keys())}")
        
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
        
        # Initialize graph builder
        graph_builder = SuperpixelGraphBuilder(n_superpixels=n_superpixels)
        
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
            
            print(f"Processed patient {patient_id}: created {len(graphs)} graphs")
        else:
            print(f"Warning: No graphs were created for patient {patient_id}")
        
        return patient_id, len(graphs)
    
    except Exception as e:
        print(f"Error processing {patient_file}: {str(e)}")
        print(traceback.format_exc())  # Print the full stack trace
        return os.path.basename(patient_file), 0

def main():
    parser = argparse.ArgumentParser(description='Build superpixel graphs from preprocessed BraTS data')
    parser.add_argument('--input_dir', type=str, required=True, help='Directory with preprocessed .npz files')
    parser.add_argument('--output_dir', type=str, required=True, help='Output directory for graphs')
    parser.add_argument('--superpixels', type=int, default=400, choices=[200, 400, 800], 
                        help='Number of superpixels per slice')
    parser.add_argument('--visualize', action='store_true', help='Generate superpixel visualizations')
    parser.add_argument('--num_workers', type=int, default=1, help='Number of parallel workers')
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Get all preprocessed patient files
    patient_files = glob.glob(os.path.join(args.input_dir, "*/*_preprocessed.npz"))
    
    print(f"Found {len(patient_files)} preprocessed patient files")
    
    # Process patients
    if args.num_workers > 1 and len(patient_files) > 1:
        with ProcessPoolExecutor(max_workers=args.num_workers) as executor:
            futures = [
                executor.submit(
                    process_patient, 
                    patient_file, 
                    args.output_dir,
                    args.superpixels,
                    args.visualize
                ) 
                for patient_file in patient_files
            ]
            
            results = []
            for future in tqdm(futures, total=len(patient_files), desc="Building graphs"):
                results.append(future.result())
    else:
        results = []
        for patient_file in tqdm(patient_files, desc="Building graphs"):
            results.append(process_patient(
                patient_file, 
                args.output_dir,
                args.superpixels,
                args.visualize
            ))
    
    # Report results
    successful = [r for r in results if r[1] > 0]
    print(f"Successfully processed {len(successful)} patients")
    print(f"Failed to process {len(results) - len(successful)} patients")
    
    # Calculate total graphs
    total_graphs = sum(r[1] for r in results)
    print(f"Total graphs created: {total_graphs}")

if __name__ == "__main__":
    start_time = time.time()
    main()
    print(f"Total graph building time: {(time.time() - start_time) / 60:.2f} minutes")