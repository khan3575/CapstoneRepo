# src/graph_construction_v2.py - Clean rewrite with optimal design
import os
import numpy as np
import torch
from torch_geometric.data import Data
from skimage.segmentation import slic
from skimage.measure import regionprops
from skimage.util import img_as_float

# Fast SLIC backend (16x faster than skimage)
try:
    from fast_slic import Slic as FastSlic
    _FAST_SLIC_AVAILABLE = True
except ImportError:
    _FAST_SLIC_AVAILABLE = False
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
    
    def _to_uint8_3ch(self, multichannel: np.ndarray) -> np.ndarray:
        """Convert 4-channel float [0,1] to 3-channel uint8 for fast_slic."""
        # Use T1ce (1), T2 (2), FLAIR (3) — most informative for tumor
        out = np.zeros((multichannel.shape[0], multichannel.shape[1], 3), dtype=np.uint8)
        for dst, src in enumerate([1, 2, 3]):
            ch = multichannel[:, :, src]
            mn, mx = ch.min(), ch.max()
            if mx > mn:
                out[:, :, dst] = ((ch - mn) / (mx - mn) * 255).astype(np.uint8)
        return out

    def compute_superpixels(self, slice_data: Dict[str, np.ndarray]) -> np.ndarray:
        """Compute SLIC superpixels — uses fast_slic (16x faster) with skimage fallback."""
        mask = slice_data["brain_mask"] > 0
        brain_pixels = np.sum(mask)

        segments = np.zeros(mask.shape, dtype=np.int32)

        if brain_pixels < 100:
            return self._create_grid_superpixels(mask, target_segments=min(20, self.config.n_superpixels))

        def safe_normalize(arr):
            arr_min, arr_max = arr.min(), arr.max()
            if arr_max > arr_min:
                return (arr - arr_min) / (arr_max - arr_min)
            return np.zeros_like(arr)

        multichannel = np.stack([
            safe_normalize(slice_data["T1"]),
            safe_normalize(slice_data["T1ce"]),
            safe_normalize(slice_data["T2"]),
            safe_normalize(slice_data["FLAIR"])
        ], axis=-1)

        n_segments_adj = min(self.config.n_superpixels, max(10, brain_pixels // 200))

        # --- fast_slic path (primary) ---
        if _FAST_SLIC_AVAILABLE:
            try:
                img_uint8 = self._to_uint8_3ch(multichannel)
                fast_slic_obj = FastSlic(
                    num_components=n_segments_adj,
                    compactness=10,
                    convert_to_lab=False,
                    num_threads=1  # prevent over-subscription in multiprocessing
                )
                segments_full = fast_slic_obj.iterate(img_uint8).astype(np.int32) + 1  # 1-indexed
                segments[mask] = segments_full[mask]
                unique_count = len(np.unique(segments)) - 1
                return segments
            except Exception:
                pass  # fall through to skimage

        # --- skimage fallback ---
        try:
            try:
                if hasattr(slic, '__code__') and 'channel_axis' in slic.__code__.co_varnames:
                    segments_full = slic(multichannel, n_segments=n_segments_adj, sigma=1.0,
                                        compactness=0.1, max_num_iter=20, start_label=1, channel_axis=-1)
                else:
                    segments_full = slic(multichannel, n_segments=n_segments_adj, sigma=1.0,
                                        compactness=0.1, max_num_iter=20, start_label=1, multichannel=True)
            except Exception:
                if hasattr(slic, '__code__') and 'channel_axis' in slic.__code__.co_varnames:
                    segments_full = slic(multichannel, n_segments=n_segments_adj, sigma=1.0,
                                        compactness=0.1, max_iter=20, start_label=1, channel_axis=-1)
                else:
                    segments_full = slic(multichannel, n_segments=n_segments_adj, sigma=1.0,
                                        compactness=0.1, max_iter=20, start_label=1, multichannel=True)
            segments[mask] = segments_full[mask]
            unique_count = len(np.unique(segments)) - 1
            return segments
        except Exception:
            return self._create_grid_superpixels(mask, target_segments=min(50, self.config.n_superpixels))
    
    def _create_grid_superpixels(self, mask: np.ndarray, target_segments: int = 50) -> np.ndarray:
        """Create simple grid-based superpixels as fallback."""
        h, w = mask.shape
        segments = np.zeros((h, w), dtype=np.int32)
        grid_size = max(2, int(np.sqrt(target_segments)))
        step_h = max(1, h // grid_size)
        step_w = max(1, w // grid_size)
        segment_id = 1
        for i in range(0, h, step_h):
            for j in range(0, w, step_w):
                region_mask = mask[i:i+step_h, j:j+step_w]
                if np.any(region_mask):
                    segments[i:i+step_h, j:j+step_w][region_mask] = segment_id
                    segment_id += 1
        return segments
    
    def compute_features(self, slice_data: Dict[str, np.ndarray],
                        segments: np.ndarray) -> Tuple[np.ndarray, np.ndarray, List[np.ndarray], np.ndarray]:
        """
        Compute node features — fully vectorized with bincount scatter ops.
        No per-superpixel Python loops; all aggregations done at once.
        Returns (features, centroids, masks, bboxes).
        """
        unique_labels = np.unique(segments)
        unique_labels = unique_labels[unique_labels > 0]

        if len(unique_labels) == 0:
            return np.array([]), np.array([]), [], np.empty((0, 4), dtype=np.int32)

        h, w = segments.shape
        n_labels = len(unique_labels)
        max_label = int(unique_labels[-1]) + 1  # sorted, so last is max

        seg_flat = segments.ravel()
        active   = seg_flat > 0
        seg_act  = seg_flat[active].astype(np.int64)

        # --- Coordinate grids ---
        y_grid = np.repeat(np.arange(h, dtype=np.float64), w)
        x_grid = np.tile(np.arange(w, dtype=np.float64), h)
        y_act = y_grid[active]
        x_act = x_grid[active]

        # --- Area and centroids via bincount ---
        area_all = np.bincount(seg_act, minlength=max_label)
        cnt = area_all[unique_labels].astype(np.float64)  # (n_labels,)

        cy_all = np.bincount(seg_act, weights=y_act, minlength=max_label)
        cx_all = np.bincount(seg_act, weights=x_act, minlength=max_label)
        centroid_y = cy_all[unique_labels] / cnt
        centroid_x = cx_all[unique_labels] / cnt

        # --- Intensity means via bincount ---
        def _mean_std(mod_flat):
            s   = np.bincount(seg_act, weights=mod_flat,          minlength=max_label)[unique_labels] / cnt
            s2  = np.bincount(seg_act, weights=mod_flat ** 2,     minlength=max_label)[unique_labels] / cnt
            std = np.sqrt(np.maximum(s2 - s ** 2, 0.0))
            return s, std

        T1_f    = slice_data["T1"].ravel()[active]
        T1ce_f  = slice_data["T1ce"].ravel()[active]
        T2_f    = slice_data["T2"].ravel()[active]
        FLAIR_f = slice_data["FLAIR"].ravel()[active]

        t1_mean,   t1_std   = _mean_std(T1_f)
        t1ce_mean, t1ce_std = _mean_std(T1ce_f)
        t2_mean,   t2_std   = _mean_std(T2_f)
        fl_mean,   fl_std   = _mean_std(FLAIR_f)

        # --- Sort once by label (shared for intensity range + bounding boxes) ---
        sort_idx = np.argsort(seg_act, kind='stable')
        sorted_labels = seg_act[sort_idx]
        cuts = np.searchsorted(sorted_labels, unique_labels)

        # --- Intensity range: max/min across all 4 modalities per label ---
        # Stack modalities, take per-pixel max/min, then reduceat per label
        all_mods = np.stack([T1_f, T1ce_f, T2_f, FLAIR_f], axis=1)  # (n, 4)
        pix_max = all_mods.max(axis=1)[sort_idx]
        pix_min = all_mods.min(axis=1)[sort_idx]
        intensity_range = np.maximum(0.0,
            np.maximum.reduceat(pix_max, cuts) - np.minimum.reduceat(pix_min, cuts))

        # --- Bounding boxes: min/max of y/x per label via reduceat ---
        y_act_i = y_act.astype(np.int32)
        x_act_i = x_act.astype(np.int32)
        y_sorted = y_act_i[sort_idx]
        x_sorted = x_act_i[sort_idx]
        min_y = np.minimum.reduceat(y_sorted, cuts)
        max_y = np.maximum.reduceat(y_sorted, cuts)
        min_x = np.minimum.reduceat(x_sorted, cuts)
        max_x = np.maximum.reduceat(x_sorted, cuts)
        bboxes = np.stack([min_y, max_y, min_x, max_x], axis=1)

        # --- Perimeter via boundary map (single vectorized pass) ---
        s = segments
        is_bnd = (
            (s != np.roll(s,  1, axis=0)) | (s != np.roll(s, -1, axis=0)) |
            (s != np.roll(s,  1, axis=1)) | (s != np.roll(s, -1, axis=1))
        ) & (s > 0)
        peri_all = np.bincount(s[is_bnd].ravel(), minlength=max_label)
        perimeter = np.maximum(1.0, peri_all[unique_labels].astype(np.float64))

        # --- Derived shape features ---
        area       = cnt
        norm_area  = area / (h * w)
        norm_y     = centroid_y / h
        norm_x     = centroid_x / w
        compactness = np.clip((4 * np.pi * area) / (perimeter ** 2), 0.0, 1.0)

        # --- Assemble feature matrix (n_labels, 15) ---
        features = np.column_stack([
            t1_mean, t1ce_mean, t2_mean, fl_mean,
            t1_std,  t1ce_std,  t2_std,  fl_std,
            area, norm_area, norm_y, norm_x,
            perimeter, compactness, intensity_range
        ])

        centroids = np.column_stack([centroid_y, centroid_x])

        # Masks still needed for _compute_labels_from_masks
        masks = [(segments == lab) for lab in unique_labels]

        return features, centroids, masks, bboxes

class EdgeBuilder:
    """Builds graph edges efficiently."""
    
    def __init__(self, config: Config):
        self.config = config
    
    def build_adjacency_edges(self, segments: np.ndarray) -> List[Tuple[int, int]]:
        """Build intra-slice adjacency edges — fully vectorized with numpy."""
        unique_labels = np.unique(segments)
        unique_labels = unique_labels[unique_labels > 0]

        if len(unique_labels) == 0:
            return []

        # Map label values to compact 0-based indices
        max_label = int(segments.max()) + 1
        label_to_idx = np.full(max_label, -1, dtype=np.int32)
        for i, lab in enumerate(unique_labels):
            label_to_idx[lab] = i

        # Vectorized horizontal neighbors: compare each pixel with its right neighbor
        left  = segments[:, :-1]   # (H, W-1)
        right = segments[:, 1:]    # (H, W-1)
        h_mask = (left > 0) & (right > 0) & (left != right)
        h_l, h_r = left[h_mask], right[h_mask]

        # Vectorized vertical neighbors: compare each pixel with its bottom neighbor
        top = segments[:-1, :]     # (H-1, W)
        bot = segments[1:, :]      # (H-1, W)
        v_mask = (top > 0) & (bot > 0) & (top != bot)
        v_t, v_b = top[v_mask], bot[v_mask]

        if h_l.size == 0 and v_t.size == 0:
            return []

        # Combine all boundary pairs and convert to compact indices
        all_a = np.concatenate([h_l, v_t])
        all_b = np.concatenate([h_r, v_b])
        idx_a = label_to_idx[all_a]
        idx_b = label_to_idx[all_b]

        # Sort each pair so (i,j) with i<j, then deduplicate
        pairs = np.stack([np.minimum(idx_a, idx_b),
                          np.maximum(idx_a, idx_b)], axis=1)
        pairs = np.unique(pairs, axis=0)

        # Emit bidirectional edges
        edge_list = []
        for i, j in pairs:
            edge_list.append((int(i), int(j)))
            edge_list.append((int(j), int(i)))

        return edge_list
    
    def build_inter_slice_edges(self, centroids1: np.ndarray, centroids2: np.ndarray,
                               bboxes1: np.ndarray, bboxes2: np.ndarray,
                               offset: int) -> List[Tuple[int, int]]:
        """Build inter-slice edges — fully vectorized centroid distance + bbox overlap."""
        if len(centroids1) == 0 or len(centroids2) == 0:
            return []

        c1 = np.asarray(centroids1)  # (n1, 2)
        c2 = np.asarray(centroids2)  # (n2, 2)

        # Vectorized pairwise centroid distance matrix (n1, n2)
        diff = c1[:, None, :] - c2[None, :, :]  # (n1, n2, 2)
        dists = np.sqrt((diff ** 2).sum(axis=2))  # (n1, n2)

        # Candidate pairs within distance threshold
        ci, cj = np.where(dists < self.config.inter_slice_threshold)
        if len(ci) == 0:
            return []

        # Vectorized bbox overlap check for all candidate pairs at once
        b1 = bboxes1[ci]  # (K, 4) — [min_y, max_y, min_x, max_x]
        b2 = bboxes2[cj]  # (K, 4)
        overlap_y = (b1[:, 1] >= b2[:, 0]) & (b2[:, 1] >= b1[:, 0])
        overlap_x = (b1[:, 3] >= b2[:, 2]) & (b2[:, 3] >= b1[:, 2])
        valid = overlap_y & overlap_x

        edges = []
        for i, j in zip(ci[valid].tolist(), cj[valid].tolist()):
            edges.append((i, j + offset))
            edges.append((j + offset, i))
        return edges

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

        # Add representative non-tumor slices (every 3rd with enough brain content)
        tumor_set = set(selected)
        non_tumor_candidates = [z for z in range(n_slices) if z not in tumor_set]
        for z in non_tumor_candidates[::3]:
            slice_data = {key: volume_data[key][z] for key in volume_data}
            if self.slice_processor.is_valid_slice(slice_data):
                selected.append(z)

        return sorted(selected)
    
    def process_volume(self, volume_data: Dict[str, np.ndarray]) -> Tuple[List[Data], List[np.ndarray]]:
        """Process entire volume into graphs."""
        selected_slices = self.select_slices(volume_data)

        if len(selected_slices) < 2:
            return [], []

        # Process slices
        slice_results = []
        for z in selected_slices:
            slice_data = {key: volume_data[key][z] for key in volume_data}

            if not self.slice_processor.is_valid_slice(slice_data):
                slice_results.append(None)
                continue

            segments = self.slice_processor.compute_superpixels(slice_data)
            if len(np.unique(segments)) <= 1:
                slice_results.append(None)
                continue

            features, centroids, masks, bboxes = self.slice_processor.compute_features(slice_data, segments)
            if len(features) == 0:
                slice_results.append(None)
                continue

            slice_results.append({
                'features': features,
                'centroids': centroids,
                'masks': masks,
                'bboxes': bboxes,
                'segments': segments,
                'slice_idx': z,
                'slice_data': slice_data
            })

            if not self.memory_manager.check_memory():
                self.memory_manager.cleanup()

        # Build graphs from consecutive slice pairs
        graphs = []
        all_segments = []

        for i in range(len(slice_results) - 1):
            if slice_results[i] is None or slice_results[i+1] is None:
                continue

            slice1, slice2 = slice_results[i], slice_results[i+1]
            if slice2['slice_idx'] - slice1['slice_idx'] > 5:
                continue

            graph = self._build_slice_pair_graph(slice1, slice2)
            if graph is not None:
                graphs.append(graph)
                if i == 0:
                    all_segments.append(slice1['segments'])

        if slice_results and slice_results[-1] is not None:
            all_segments.append(slice_results[-1]['segments'])

        return graphs, all_segments
    
    def _compute_labels_from_masks(self, masks: List[np.ndarray], slice_data: Dict[str, np.ndarray]) -> np.ndarray:
        """
        Compute majority-class GT label per superpixel — fully vectorized.
        BraTS labels: 0=background, 1=NCR/NET, 2=Edema, 4=Enhancing Tumor.
        Stores MULTI-CLASS labels for flexibility (binary: threshold y > 0).
        """
        if not masks:
            return np.array([], dtype=np.float32)

        # Build a segment-id array aligned with masks list (1-indexed, compact)
        # Recover segment map from masks
        segments = np.zeros(masks[0].shape, dtype=np.int32)
        for k, mask in enumerate(masks):
            segments[mask] = k + 1  # 1-indexed compact labels

        seg_flat = segments.ravel()
        lbl_flat = slice_data["label"].ravel().astype(np.int32)
        n_segs   = len(masks)

        # Count each BraTS label class per segment
        # BraTS labels: 0, 1, 2, 4  — map 4 → index 3 for compact storage
        BRATS_VALUES = np.array([0, 1, 2, 4], dtype=np.int32)
        # (4, n_segs) count matrix
        counts = np.zeros((4, n_segs + 1), dtype=np.int64)
        for ki, bl in enumerate(BRATS_VALUES):
            match = (lbl_flat == bl).astype(np.float64)
            counts[ki] = np.bincount(seg_flat, weights=match, minlength=n_segs + 1)

        # Majority class per segment (indices 1..n_segs)
        majority_idx = counts[:, 1:].argmax(axis=0)  # (n_segs,)
        out = BRATS_VALUES[majority_idx].astype(np.float32)
        return out
    
    def _build_slice_pair_graph(self, slice1: Dict, slice2: Dict) -> Optional[Data]:
        """Build graph from two consecutive slices."""
        try:
            features1, features2 = slice1['features'], slice2['features']
            n1, n2 = len(features1), len(features2)
            if n1 == 0 or n2 == 0:
                return None

            all_features = np.vstack([features1, features2])

            # Compute labels from ACTUAL ground-truth
            labels1 = self._compute_labels_from_masks(slice1['masks'], slice1['slice_data'])
            labels2 = self._compute_labels_from_masks(slice2['masks'], slice2['slice_data'])
            all_labels = np.concatenate([labels1, labels2])

            # Build edges
            edges1 = self.edge_builder.build_adjacency_edges(slice1['segments'])
            edges2 = self.edge_builder.build_adjacency_edges(slice2['segments'])
            edges2_offset = [(u + n1, v + n1) for u, v in edges2]
            inter_edges = self.edge_builder.build_inter_slice_edges(
                slice1['centroids'], slice2['centroids'],
                slice1['bboxes'], slice2['bboxes'],
                n1
            )

            all_edges = edges1 + edges2_offset + inter_edges
            if not all_edges:
                return None

            edge_index = torch.tensor(all_edges, dtype=torch.long).t().contiguous()
            x = torch.tensor(all_features, dtype=torch.float32)
            # Store multi-class labels (0,1,2,4); binary transform converts on-the-fly
            y = torch.tensor(all_labels, dtype=torch.float32)
            slice_mask = torch.zeros(len(all_features), dtype=torch.bool)
            slice_mask[:n1] = True
            slice_indices = torch.tensor([slice1['slice_idx'], slice2['slice_idx']], dtype=torch.long)

            return Data(
                x=x,
                edge_index=edge_index,
                y=y,
                slice_mask=slice_mask,
                slice_indices=slice_indices
            )

        except Exception as e:
            return None

def process_patient_file(patient_file: str, output_dir: str, config: Config) -> Tuple[str, int]:
    """Process a single patient file."""
    start_time = time.time()

    try:
        patient_id = Path(patient_file).stem.replace('_preprocessed', '')
        patient_output_dir = Path(output_dir) / patient_id

        # Skip if already done (check before loading NPZ)
        graph_file = patient_output_dir / f'{patient_id}_graphs_{config.n_superpixels}.pt'
        if graph_file.exists():
            return patient_id, -1  # -1 = skipped

        patient_output_dir.mkdir(parents=True, exist_ok=True)

        # Load data
        data = np.load(patient_file)

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
        
        processing_time = time.time() - start_time
        print(f"✓ {patient_id}: {len(graphs)} graphs in {processing_time:.0f}s")
        
        return patient_id, len(graphs)
        
    except Exception as e:
        print(f"✗ Error processing {patient_file}: {e}")
        return Path(patient_file).stem, 0
    
    finally:
        # Cleanup
        gc.collect()

def process_patient_wrapper(args_tuple):
    """Wrapper for multiprocessing — limits BLAS threads to avoid over-subscription."""
    try:
        from threadpoolctl import threadpool_limits
        threadpool_limits(limits=1)
    except ImportError:
        pass
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
                pool.imap_unordered(process_patient_wrapper, process_args),
                total=len(patient_files),
                desc="Processing patients"
            ))
    
    # Report results
    skipped    = [r for r in results if r[1] == -1]
    successful = [r for r in results if r[1] > 0]
    failed     = [r for r in results if r[1] == 0]
    total_graphs = sum(r[1] for r in results if r[1] > 0)

    print("\n" + "="*50)
    print(f"✓ Skipped (already done): {len(skipped)}/{len(results)} patients")
    print(f"✓ Newly processed:        {len(successful)}/{len(results)} patients")
    print(f"✓ Total graphs created:   {total_graphs}")
    print(f"✓ Failed:                 {len(failed)}")

if __name__ == "__main__":
    start_time = time.time()
    main()
    total_time = (time.time() - start_time) / 60
    print(f"✓ Total processing time: {total_time:.1f} minutes")