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
from multiprocessing import Pool
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
        """Compute SLIC superpixels with robust fallbacks."""
        # Brain mask
        mask = slice_data["brain_mask"] > 0
        brain_pixels = np.sum(mask)
        
        # Initialize segments
        segments = np.zeros(mask.shape, dtype=np.int32)
        
        if brain_pixels < 100:  # Not enough pixels
            print(f"⚠️  Too few brain pixels ({brain_pixels}), using grid fallback")
            return self._create_grid_superpixels(mask, target_segments=min(20, self.config.n_superpixels))
        
        # Stack modalities (normalize to 0-1 range)
        def safe_normalize(arr):
            arr_min, arr_max = arr.min(), arr.max()
            if arr_max > arr_min:
                return (arr - arr_min) / (arr_max - arr_min)
            else:
                return np.zeros_like(arr)
        
        multichannel = np.stack([
            safe_normalize(slice_data["T1"]),
            safe_normalize(slice_data["T1ce"]),
            safe_normalize(slice_data["T2"]),
            safe_normalize(slice_data["FLAIR"])
        ], axis=-1)
        
        # Adaptive parameters based on brain size
        n_segments_adj = min(self.config.n_superpixels, max(10, brain_pixels // 200))
        
        # Smart SLIC with automatic version detection
        try:
            # Check scikit-image version and use appropriate parameters
            import skimage
            version = skimage.__version__
            print(f"🔍 Using scikit-image version: {version}")
            
            # Try modern API first (max_num_iter)
            try:
                if hasattr(slic, '__code__') and 'channel_axis' in slic.__code__.co_varnames:
                    # Modern version with channel_axis
                    segments_full = slic(
                        multichannel,
                        n_segments=n_segments_adj,
                        sigma=1.0,
                        compactness=0.1,
                        max_num_iter=20,
                        start_label=1,
                        channel_axis=-1
                    )
                    print("✅ Used modern SLIC API (channel_axis + max_num_iter)")
                else:
                    # Modern version without channel_axis
                    segments_full = slic(
                        multichannel,
                        n_segments=n_segments_adj,
                        sigma=1.0,
                        compactness=0.1,
                        max_num_iter=20,
                        start_label=1,
                        multichannel=True
                    )
                    print("✅ Used modern SLIC API (max_num_iter)")
            except Exception as modern_e:
                print(f"⚠️  Modern SLIC API failed ({modern_e}), trying legacy API...")
                
                # Fallback to older API (max_iter)
                if hasattr(slic, '__code__') and 'channel_axis' in slic.__code__.co_varnames:
                    # Legacy version with channel_axis
                    segments_full = slic(
                        multichannel,
                        n_segments=n_segments_adj,
                        sigma=1.0,
                        compactness=0.1,
                        max_iter=20,
                        start_label=1,
                        channel_axis=-1
                    )
                    print("✅ Used legacy SLIC API (channel_axis + max_iter)")
                else:
                    # Legacy version
                    segments_full = slic(
                        multichannel,
                        n_segments=n_segments_adj,
                        sigma=1.0,
                        compactness=0.1,
                        max_iter=20,
                        start_label=1,
                        multichannel=True
                    )
                    print("✅ Used legacy SLIC API (max_iter)")
            
            # Apply mask
            segments[mask] = segments_full[mask]
            unique_count = len(np.unique(segments)) - 1
            print(f"✅ SLIC success: {unique_count} superpixels created")
            return segments
            
        except Exception as e:
            print(f"⚠️  All SLIC methods failed ({e}), using simple grid fallback")
            return self._create_grid_superpixels(mask, target_segments=min(50, self.config.n_superpixels))
    
    def _create_grid_superpixels(self, mask: np.ndarray, target_segments: int = 50) -> np.ndarray:
        """Create simple grid-based superpixels as fallback."""
        h, w = mask.shape
        segments = np.zeros((h, w), dtype=np.int32)
        
        # Calculate grid size
        grid_size = int(np.sqrt(target_segments))
        if grid_size < 2:
            grid_size = 2
        
        step_h = max(1, h // grid_size)
        step_w = max(1, w // grid_size)
        
        segment_id = 1
        for i in range(0, h, step_h):
            for j in range(0, w, step_w):
                # Only assign to brain regions
                region_mask = mask[i:i+step_h, j:j+step_w]
                if np.any(region_mask):
                    segments[i:i+step_h, j:j+step_w][region_mask] = segment_id
                    segment_id += 1
        
        print(f"🔧 Created {segment_id-1} grid-based superpixels")
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
            
            # Fix BraTS label handling: Convert multi-class (0,1,2,4) to binary (0,1)
            tumor_binary = slice_data["label"][mask] > 0  # Any non-zero label is tumor
            tumor_ratio = np.mean(tumor_binary.astype(float))  # Ratio of tumor pixels in superpixel
            
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
        tumor_slices = []
        
        # Always include tumor slices
        for z in range(n_slices):
            if np.any(volume_data["label"][z] > 0):
                selected.append(z)
                tumor_slices.append(z)
        
        print(f"🎯 Found {len(tumor_slices)} tumor slices: {tumor_slices[:10]}{'...' if len(tumor_slices) > 10 else ''}")
        
        # Add representative non-tumor slices
        tumor_set = set(selected)
        non_tumor_candidates = [z for z in range(n_slices) if z not in tumor_set]
        
        added_non_tumor = []
        # Sample every 3rd non-tumor slice with brain content
        for z in non_tumor_candidates[::3]:
            slice_data = {key: volume_data[key][z] for key in volume_data}
            if self.slice_processor.is_valid_slice(slice_data):
                selected.append(z)
                added_non_tumor.append(z)
        
        print(f"🧠 Added {len(added_non_tumor)} non-tumor slices: {added_non_tumor[:10]}{'...' if len(added_non_tumor) > 10 else ''}")
        
        return sorted(selected)
    
    def process_volume(self, volume_data: Dict[str, np.ndarray]) -> Tuple[List[Data], List[np.ndarray]]:
        """Process entire volume into graphs."""
        n_total_slices = volume_data["T1"].shape[0]
        selected_slices = self.select_slices(volume_data)
        
        print(f"📊 Volume info: {n_total_slices} total slices, {len(selected_slices)} selected")
        
        if len(selected_slices) < 2:
            print("❌ Warning: Not enough slices to create graphs")
            return [], []
        
        print(f"🔄 Processing {len(selected_slices)} selected slices...")
        
        # Process slices
        slice_results = []
        valid_slice_count = 0
        
        for z in tqdm(selected_slices, desc="Processing slices"):
            slice_data = {key: volume_data[key][z] for key in volume_data}
            
            if not self.slice_processor.is_valid_slice(slice_data):
                print(f"⚠️  Slice {z}: Invalid (insufficient brain content)")
                slice_results.append(None)
                continue
            
            # Compute superpixels and features
            segments = self.slice_processor.compute_superpixels(slice_data)
            unique_segments = len(np.unique(segments)) - 1  # subtract background
            
            if unique_segments == 0:
                print(f"⚠️  Slice {z}: No superpixels created")
                slice_results.append(None)
                continue
            
            features, centroids, masks = self.slice_processor.compute_features(slice_data, segments)
            
            if len(features) == 0:
                print(f"⚠️  Slice {z}: No features extracted")
                slice_results.append(None)
                continue
            
            print(f"✅ Slice {z}: {unique_segments} superpixels, {len(features)} features")
            slice_results.append({
                'features': features,
                'centroids': centroids,
                'masks': masks,
                'segments': segments,
                'slice_idx': z
            })
            valid_slice_count += 1
            
            # Memory management
            if not self.memory_manager.check_memory():
                self.memory_manager.cleanup()
        
        print(f"📈 Valid slices processed: {valid_slice_count}/{len(selected_slices)}")
        
        # Build graphs from consecutive slice pairs
        graphs = []
        all_segments = []
        graph_attempts = 0
        successful_graphs = 0
        
        for i in tqdm(range(len(slice_results) - 1), desc="Building graphs"):
            if slice_results[i] is None or slice_results[i+1] is None:
                continue
            
            slice1, slice2 = slice_results[i], slice_results[i+1]
            
            # Skip if slices are too far apart
            if slice2['slice_idx'] - slice1['slice_idx'] > 5:
                print(f"⚠️  Skipping slices {slice1['slice_idx']}-{slice2['slice_idx']}: too far apart")
                continue
            
            graph_attempts += 1
            graph = self._build_slice_pair_graph(slice1, slice2)
            if graph is not None:
                graphs.append(graph)
                successful_graphs += 1
                print(f"✅ Graph {successful_graphs}: slices {slice1['slice_idx']}-{slice2['slice_idx']}")
            else:
                print(f"❌ Failed to create graph for slices {slice1['slice_idx']}-{slice2['slice_idx']}")
            
            # Store segments for the first slice (store last slice separately after loop)
            if i == 0:
                all_segments.append(slice1['segments'])
        
        # Add the last slice segments
        if slice_results and slice_results[-1] is not None:
            all_segments.append(slice_results[-1]['segments'])
        
        print(f"📊 Graph creation: {successful_graphs}/{graph_attempts} successful")
        return graphs, all_segments
    
    def _build_slice_pair_graph(self, slice1: Dict, slice2: Dict) -> Optional[Data]:
        """Build graph from two consecutive slices."""
        try:
            # Combine features
            features1, features2 = slice1['features'], slice2['features']
            n1, n2 = len(features1), len(features2)
            
            if n1 == 0 or n2 == 0:
                print(f"🚫 No features: slice1={n1}, slice2={n2}")
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
            
            print(f"🔗 Edges: intra1={len(edges1)}, intra2={len(edges2_offset)}, inter={len(inter_edges)}, total={len(all_edges)}")
            
            if not all_edges:
                print("🚫 No edges created")
                return None
            
            # Create tensors
            edge_index = torch.tensor(all_edges, dtype=torch.long).t().contiguous()
            x = torch.tensor(all_features, dtype=torch.float32)
            
            # Labels: Superpixel is tumor if >10% of its pixels are tumor
            # (More sensitive threshold for medical segmentation)
            y = torch.tensor(all_features[:, -1] > 0.1, dtype=torch.float32)
            tumor_nodes = torch.sum(y).item()
            
            # Slice mask
            slice_mask = torch.zeros(len(all_features), dtype=torch.bool)
            slice_mask[:n1] = True
            
            # Slice indices
            slice_indices = torch.tensor([slice1['slice_idx'], slice2['slice_idx']], dtype=torch.long)
            
            print(f"📊 Graph: {len(all_features)} nodes ({tumor_nodes} tumor), {len(all_edges)} edges")
            
            return Data(
                x=x,
                edge_index=edge_index,
                y=y,
                slice_mask=slice_mask,
                slice_indices=slice_indices
            )
            
        except Exception as e:
            print(f"❌ Error building graph: {e}")
            import traceback
            traceback.print_exc()
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

def process_patient_wrapper(args_tuple):
    """Wrapper for multiprocessing"""
    patient_file, output_dir, config = args_tuple
    return process_patient_file(patient_file, output_dir, config)

def main():
    parser = argparse.ArgumentParser(description='Clean Graph Construction for BraTS')
    parser.add_argument('--input_dir', type=str, required=True, help='Input directory')
    parser.add_argument('--output_dir', type=str, required=True, help='Output directory')
    parser.add_argument('--superpixels', type=int, default=200, help='Number of superpixels')
    parser.add_argument('--max_memory_gb', type=float, default=24.0, help='Max memory usage in GB')
    parser.add_argument('--num_workers', type=int, default=12, help='Number of parallel workers')
    
    args = parser.parse_args()
    
    # Configuration - adjust memory per worker
    memory_per_worker = args.max_memory_gb / args.num_workers
    config = Config(
        n_superpixels=args.superpixels,
        max_memory_gb=memory_per_worker
    )
    
    # Find patient files
    pattern = os.path.join(args.input_dir, "*", "*_preprocessed.npz")
    patient_files = glob.glob(pattern)
    
    if not patient_files:
        print(f"No patient files found in {args.input_dir}")
        return
    
    print(f"Found {len(patient_files)} patient files")
    print(f"Configuration: {config.n_superpixels} superpixels, {args.max_memory_gb}GB total memory")
    print(f"Workers: {args.num_workers} parallel processes, {memory_per_worker:.1f}GB per worker")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Process files
    if args.num_workers == 1:
        # Sequential processing
        results = []
        for patient_file in tqdm(patient_files, desc="Processing patients"):
            result = process_patient_file(patient_file, args.output_dir, config)
            results.append(result)
            gc.collect()
    else:
        # Parallel processing
        print(f"🚀 Starting parallel processing with {args.num_workers} workers...")
        
        # Prepare arguments for multiprocessing
        process_args = [(pf, args.output_dir, config) for pf in patient_files]
        
        with Pool(processes=args.num_workers) as pool:
            results = list(tqdm(
                pool.imap(process_patient_wrapper, process_args),
                total=len(patient_files),
                desc="Processing patients"
            ))
    
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