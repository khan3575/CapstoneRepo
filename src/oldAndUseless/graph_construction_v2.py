# src/graph_construction_v2.py - Clean rewrite with optimal design
import os
import numpy as np
import torch
from torch_geometric.data import Data
from skimage.segmentation import slic
from skimage.measure import regionprops
from skimage.util import img_as_float
from tqdm import tqdm
import argparse
import glob
import time
import gc
from pathlib import Path
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional, Generator
import warnings
warnings.filterwarnings('ignore')

@dataclass
class Config:
    """Configuration for graph construction."""
    n_superpixels: int = 200
    sigma: float = 0.3
    compactness: float = 0.1
    max_iter: int = 30
    inter_slice_threshold: float = 10.0
    iou_threshold: float = 0.1
    max_memory_gb: float = 4.0
    min_brain_pixels: int = 1000
    tumor_slice_priority: bool = True

class MemoryManager:
    """Simple memory management."""
    
    def __init__(self, max_memory_gb: float = 4.0):
        self.max_memory_bytes = max_memory_gb * 1024**3
        
    def get_usage_mb(self) -> float:
        """Get current memory usage in MB."""
        try:
            import psutil
            return psutil.Process().memory_info().rss / (1024**2)
        except ImportError:
            return 0.0
    
    def check_memory(self) -> bool:
        """Check if memory usage is within limits."""
        try:
            import psutil
            return psutil.virtual_memory().available > self.max_memory_bytes
        except ImportError:
            return True
    
    def cleanup(self):
        """Force garbage collection."""
        gc.collect()

class SliceProcessor:
    """Processes individual slices into superpixels and features."""
    
    def __init__(self, config: Config):
        self.config = config
    
    def is_valid_slice(self, slice_data: Dict[str, np.ndarray]) -> bool:
        """Check if slice has enough brain content."""
        brain_pixels = np.sum(slice_data['brain_mask'] > 0)
        return brain_pixels >= self.config.min_brain_pixels
    
    def compute_superpixels(self, slice_data: Dict[str, np.ndarray]) -> np.ndarray:
        """Compute SLIC superpixels efficiently."""
        # Stack modalities
        multichannel = np.stack([
            img_as_float(slice_data["T1"]),
            img_as_float(slice_data["T1ce"]),
            img_as_float(slice_data["T2"]),
            img_as_float(slice_data["FLAIR"])
        ], axis=-1)
        
        # Brain mask
        mask = slice_data["brain_mask"] > 0
        
        # Initialize segments
        segments = np.zeros(mask.shape, dtype=np.int32)
        
        if not np.any(mask):
            return segments
        
        # Compute SLIC
        try:
            segments_masked = slic(
                multichannel,
                n_segments=self.config.n_superpixels,
                sigma=self.config.sigma,
                compactness=self.config.compactness,
                max_iter=self.config.max_iter,
                mask=mask,
                start_label=1,
                channel_axis=-1
            )
            segments[mask] = segments_masked[mask]
        except Exception:
            # Fallback for older scikit-image
            try:
                segments_masked = slic(
                    multichannel,
                    n_segments=self.config.n_superpixels,
                    sigma=self.config.sigma,
                    compactness=self.config.compactness,
                    max_iter=self.config.max_iter,
                    mask=mask,
                    start_label=1,
                    multichannel=True
                )
                segments[mask] = segments_masked[mask]
            except Exception:
                print("Warning: SLIC failed, returning empty segments")
                return segments
        
        return segments
    
    def compute_features(self, slice_data: Dict[str, np.ndarray], 
                        segments: np.ndarray) -> Tuple[np.ndarray, np.ndarray, List[np.ndarray]]:
        """Compute node features vectorized."""
        unique_labels = np.unique(segments)
        unique_labels = unique_labels[unique_labels > 0]
        
        if len(unique_labels) == 0:
            return np.array([]), np.array([]), []
        
        features = []
        centroids = []
        masks = []
        
        h, w = segments.shape
        
        for label in unique_labels:
            mask = (segments == label)
            masks.append(mask)
            
            # Compute centroid
            y_coords, x_coords = np.where(mask)
            if len(y_coords) == 0:
                continue
                
            centroid_y = np.mean(y_coords)
            centroid_x = np.mean(x_coords)
            centroids.append([centroid_y, centroid_x])
            
            # Compute intensity features
            t1_mean = np.mean(slice_data["T1"][mask])
            t1ce_mean = np.mean(slice_data["T1ce"][mask])
            t2_mean = np.mean(slice_data["T2"][mask])
            flair_mean = np.mean(slice_data["FLAIR"][mask])
            
            t1_std = np.std(slice_data["T1"][mask])
            t1ce_std = np.std(slice_data["T1ce"][mask])
            t2_std = np.std(slice_data["T2"][mask])
            flair_std = np.std(slice_data["FLAIR"][mask])
            
            # Compute shape features
            area = np.sum(mask)
            tumor_ratio = np.mean(slice_data["label"][mask])
            
            # Normalized coordinates
            norm_y = centroid_y / h
            norm_x = centroid_x / w
            
            feature = [
                t1_mean, t1ce_mean, t2_mean, flair_mean,
                t1_std, t1ce_std, t2_std, flair_std,
                area, norm_y, norm_x, tumor_ratio
            ]
            features.append(feature)
        
        return np.array(features), np.array(centroids), masks

class EdgeBuilder:
    """Builds graph edges efficiently."""
    
    def __init__(self, config: Config):
        self.config = config
    
    def build_adjacency_edges(self, segments: np.ndarray) -> List[Tuple[int, int]]:
        """Build intra-slice adjacency edges efficiently."""
        unique_labels = np.unique(segments)
        unique_labels = unique_labels[unique_labels > 0]
        
        if len(unique_labels) == 0:
            return []
        
        # Create label to index mapping
        label_to_idx = {label: i for i, label in enumerate(unique_labels)}
        
        # Find boundaries efficiently
        edges = set()
        h, w = segments.shape
        
        # Check horizontal neighbors
        for y in range(h):
            for x in range(w-1):
                left, right = segments[y, x], segments[y, x+1]
                if left > 0 and right > 0 and left != right:
                    idx1, idx2 = label_to_idx[left], label_to_idx[right]
                    edges.add((min(idx1, idx2), max(idx1, idx2)))
        
        # Check vertical neighbors
        for y in range(h-1):
            for x in range(w):
                top, bottom = segments[y, x], segments[y+1, x]
                if top > 0 and bottom > 0 and top != bottom:
                    idx1, idx2 = label_to_idx[top], label_to_idx[bottom]
                    edges.add((min(idx1, idx2), max(idx1, idx2)))
        
        # Convert to bidirectional edge list
        edge_list = []
        for i, j in edges:
            edge_list.extend([(i, j), (j, i)])
        
        return edge_list
    
    def build_inter_slice_edges(self, centroids1: np.ndarray, centroids2: np.ndarray,
                               masks1: List[np.ndarray], masks2: List[np.ndarray],
                               offset: int) -> List[Tuple[int, int]]:
        """Build inter-slice edges with smart sampling."""
        if len(centroids1) == 0 or len(centroids2) == 0:
            return []
        
        edges = []
        
        # Smart sampling for large numbers of superpixels
        n1, n2 = len(centroids1), len(centroids2)
        max_comparisons = 2000  # Limit comparisons
        
        if n1 * n2 > max_comparisons:
            # Sample superpixels based on tumor content and distance
            indices1 = self._smart_sample(centroids1, masks1, min(n1, 50))
            indices2 = self._smart_sample(centroids2, masks2, min(n2, 50))
        else:
            indices1 = list(range(n1))
            indices2 = list(range(n2))
        
        for i in indices1:
            c1 = centroids1[i]
            for j in indices2:
                c2 = centroids2[j]
                
                # Distance check
                distance = np.sqrt(np.sum((c1 - c2)**2))
                if distance < self.config.inter_slice_threshold:
                    # Quick IoU approximation
                    mask1, mask2 = masks1[i], masks2[j]
                    if self._quick_overlap_check(mask1, mask2):
                        edges.extend([(i, j + offset), (j + offset, i)])
        
        return edges
    
    def _smart_sample(self, centroids: np.ndarray, masks: List[np.ndarray], 
                     n_samples: int) -> List[int]:
        """Sample superpixels intelligently."""
        n = len(centroids)
        if n <= n_samples:
            return list(range(n))
        
        # Prioritize larger superpixels and those near center
        scores = []
        h, w = masks[0].shape
        center_y, center_x = h // 2, w // 2
        
        for i, (centroid, mask) in enumerate(zip(centroids, masks)):
            area = np.sum(mask)
            dist_to_center = np.sqrt((centroid[0] - center_y)**2 + (centroid[1] - center_x)**2)
            score = area / (1 + dist_to_center * 0.01)  # Larger and more central = higher score
            scores.append(score)
        
        # Select top scoring superpixels
        top_indices = np.argsort(scores)[-n_samples:]
        return top_indices.tolist()
    
    def _quick_overlap_check(self, mask1: np.ndarray, mask2: np.ndarray) -> bool:
        """Quick overlap check using bounding boxes."""
        # Get bounding boxes
        y1, x1 = np.where(mask1)
        y2, x2 = np.where(mask2)
        
        if len(y1) == 0 or len(y2) == 0:
            return False
        
        # Check bounding box overlap
        min1_y, max1_y = np.min(y1), np.max(y1)
        min1_x, max1_x = np.min(x1), np.max(x1)
        min2_y, max2_y = np.min(y2), np.max(y2)
        min2_x, max2_x = np.min(x2), np.max(x2)
        
        # Check overlap
        overlap_y = max1_y >= min2_y and max2_y >= min1_y
        overlap_x = max1_x >= min2_x and max2_x >= min1_x
        
        return overlap_y and overlap_x

class GraphBuilder:
    """Main graph construction pipeline."""
    
    def __init__(self, config: Config):
        self.config = config
        self.memory_manager = MemoryManager(config.max_memory_gb)
        self.slice_processor = SliceProcessor(config)
        self.edge_builder = EdgeBuilder(config)
    
    def select_slices(self, volume_data: Dict[str, np.ndarray]) -> List[int]:
        """Select slices to process intelligently."""
        n_slices = volume_data["T1"].shape[0]
        selected = []
        
        # Always include tumor slices
        for z in range(n_slices):
            if np.any(volume_data["label"][z] > 0):
                selected.append(z)
        
        # Add representative non-tumor slices
        tumor_set = set(selected)
        non_tumor_candidates = [z for z in range(n_slices) if z not in tumor_set]
        
        # Sample every 3rd non-tumor slice with brain content
        for z in non_tumor_candidates[::3]:
            slice_data = {key: volume_data[key][z] for key in volume_data}
            if self.slice_processor.is_valid_slice(slice_data):
                selected.append(z)
        
        return sorted(selected)
    
    def process_volume(self, volume_data: Dict[str, np.ndarray]) -> Tuple[List[Data], List[np.ndarray]]:
        """Process entire volume into graphs."""
        selected_slices = self.select_slices(volume_data)
        
        if len(selected_slices) < 2:
            print("Warning: Not enough slices to create graphs")
            return [], []
        
        print(f"Processing {len(selected_slices)} selected slices...")
        
        # Process slices
        slice_results = []
        
        for z in tqdm(selected_slices, desc="Processing slices"):
            slice_data = {key: volume_data[key][z] for key in volume_data}
            
            if not self.slice_processor.is_valid_slice(slice_data):
                slice_results.append(None)
                continue
            
            # Compute superpixels and features
            segments = self.slice_processor.compute_superpixels(slice_data)
            features, centroids, masks = self.slice_processor.compute_features(slice_data, segments)
            
            if len(features) == 0:
                slice_results.append(None)
                continue
            
            slice_results.append({
                'features': features,
                'centroids': centroids,
                'masks': masks,
                'segments': segments,
                'slice_idx': z
            })
            
            # Memory management
            if not self.memory_manager.check_memory():
                self.memory_manager.cleanup()
        
        # Build graphs from consecutive slice pairs
        graphs = []
        all_segments = []
        
        for i in tqdm(range(len(slice_results) - 1), desc="Building graphs"):
            if slice_results[i] is None or slice_results[i+1] is None:
                continue
            
            slice1, slice2 = slice_results[i], slice_results[i+1]
            
            # Skip if slices are too far apart
            if slice2['slice_idx'] - slice1['slice_idx'] > 5:
                continue
            
            graph = self._build_slice_pair_graph(slice1, slice2)
            if graph is not None:
                graphs.append(graph)
            
            # Store segments for the first slice (store last slice separately after loop)
            if i == 0:
                all_segments.append(slice1['segments'])
        
        # Add the last slice segments
        if slice_results and slice_results[-1] is not None:
            all_segments.append(slice_results[-1]['segments'])
        
        return graphs, all_segments
    
    def _build_slice_pair_graph(self, slice1: Dict, slice2: Dict) -> Optional[Data]:
        """Build graph from two consecutive slices."""
        try:
            # Combine features
            features1, features2 = slice1['features'], slice2['features']
            n1, n2 = len(features1), len(features2)
            
            if n1 == 0 or n2 == 0:
                return None
            
            all_features = np.vstack([features1, features2])
            
            # Build edges
            edges1 = self.edge_builder.build_adjacency_edges(slice1['segments'])
            edges2 = self.edge_builder.build_adjacency_edges(slice2['segments'])
            
            # Offset edges for second slice
            edges2_offset = [(u + n1, v + n1) for u, v in edges2]
            
            # Inter-slice edges
            inter_edges = self.edge_builder.build_inter_slice_edges(
                slice1['centroids'], slice2['centroids'],
                slice1['masks'], slice2['masks'],
                n1
            )
            
            # Combine all edges
            all_edges = edges1 + edges2_offset + inter_edges
            
            if not all_edges:
                return None
            
            # Create tensors
            edge_index = torch.tensor(all_edges, dtype=torch.long).t().contiguous()
            x = torch.tensor(all_features, dtype=torch.float32)
            
            # Labels (tumor ratio > 0.5)
            y = torch.tensor(all_features[:, -1] > 0.5, dtype=torch.float32)
            
            # Slice mask
            slice_mask = torch.zeros(len(all_features), dtype=torch.bool)
            slice_mask[:n1] = True
            
            # Slice indices
            slice_indices = torch.tensor([slice1['slice_idx'], slice2['slice_idx']], dtype=torch.long)
            
            return Data(
                x=x,
                edge_index=edge_index,
                y=y,
                slice_mask=slice_mask,
                slice_indices=slice_indices
            )
            
        except Exception as e:
            print(f"Error building graph: {e}")
            return None

def process_patient_file(patient_file: str, output_dir: str, config: Config) -> Tuple[str, int]:
    """Process a single patient file."""
    start_time = time.time()
    
    try:
        # Load data
        data = np.load(patient_file)
        patient_id = Path(patient_file).stem.replace('_preprocessed', '')
        
        # Create output directory
        patient_output_dir = Path(output_dir) / patient_id
        patient_output_dir.mkdir(parents=True, exist_ok=True)
        
        # Prepare volume data
        volume_data = {}
        for key in ['T1', 'T1ce', 'T2', 'FLAIR', 'label', 'brain_mask']:
            if key in data:
                volume_data[key] = data[key]
            elif key == 'brain_mask' and 'T1' in data:
                volume_data[key] = (data['T1'] > 0).astype(np.float32)
            else:
                print(f"Warning: Missing key {key} for patient {patient_id}")
                return patient_id, 0
        
        # Build graphs
        graph_builder = GraphBuilder(config)
        graphs, segments = graph_builder.process_volume(volume_data)
        
        if len(graphs) == 0:
            print(f"Warning: No graphs created for patient {patient_id}")
            return patient_id, 0
        
        # Save results
        graph_file = patient_output_dir / f'{patient_id}_graphs_{config.n_superpixels}.pt'
        segments_file = patient_output_dir / f'{patient_id}_segments_{config.n_superpixels}.npy'
        
        torch.save(graphs, graph_file)
        np.save(segments_file, segments)
        
        processing_time = (time.time() - start_time) / 60
        print(f"✓ {patient_id}: {len(graphs)} graphs in {processing_time:.1f}min")
        
        return patient_id, len(graphs)
        
    except Exception as e:
        print(f"✗ Error processing {patient_file}: {e}")
        return Path(patient_file).stem, 0
    
    finally:
        # Cleanup
        gc.collect()

def main():
    parser = argparse.ArgumentParser(description='Clean Graph Construction for BraTS')
    parser.add_argument('--input_dir', type=str, required=True, help='Input directory')
    parser.add_argument('--output_dir', type=str, required=True, help='Output directory')
    parser.add_argument('--superpixels', type=int, default=200, help='Number of superpixels')
    parser.add_argument('--max_memory_gb', type=float, default=4.0, help='Max memory usage in GB')
    parser.add_argument('--num_workers', type=int, default=1, help='Number of workers (sequential only for now)')
    
    args = parser.parse_args()
    
    # Configuration
    config = Config(
        n_superpixels=args.superpixels,
        max_memory_gb=args.max_memory_gb
    )
    
    # Find patient files
    pattern = os.path.join(args.input_dir, "*", "*_preprocessed.npz")
    patient_files = glob.glob(pattern)
    
    if not patient_files:
        print(f"No patient files found in {args.input_dir}")
        return
    
    print(f"Found {len(patient_files)} patient files")
    print(f"Configuration: {config.n_superpixels} superpixels, {config.max_memory_gb}GB memory limit")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Process files
    results = []
    
    for patient_file in tqdm(patient_files, desc="Processing patients"):
        result = process_patient_file(patient_file, args.output_dir, config)
        results.append(result)
        
        # Memory cleanup between patients
        gc.collect()
    
    # Report results
    successful = [r for r in results if r[1] > 0]
    total_graphs = sum(r[1] for r in results)
    
    print("\n" + "="*50)
    print(f"✓ Successfully processed: {len(successful)}/{len(results)} patients")
    print(f"✓ Total graphs created: {total_graphs}")
    print(f"✓ Failed patients: {len(results) - len(successful)}")

if __name__ == "__main__":
    start_time = time.time()
    main()
    total_time = (time.time() - start_time) / 60
    print(f"✓ Total processing time: {total_time:.1f} minutes")