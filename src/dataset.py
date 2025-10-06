# src/dataset.py - BraTS Graph Dataset Loader
import os
import glob
import torch
import numpy as np
from torch.utils.data import Dataset
from torch_geometric.data import Data
from pathlib import Path
import random
from typing import List, Tuple, Dict, Optional

class BraTSGraphDataset(Dataset):
    """
    Custom Dataset for loading BraTS graph data
    """
    
    def __init__(self, 
                 graph_dir: str,
                 patient_ids: Optional[List[str]] = None,
                 transform=None,
                 max_graphs_per_patient: Optional[int] = None):
        """
        Args:
            graph_dir: Directory containing graph files
            patient_ids: List of patient IDs to include (if None, use all)
            transform: Optional transform to apply
            max_graphs_per_patient: Limit graphs per patient (for debugging)
        """
        self.graph_dir = graph_dir
        self.transform = transform
        self.max_graphs_per_patient = max_graphs_per_patient
        
        # Find all graph files and expand to individual graphs
        self.graph_entries = self._find_graph_files(patient_ids)
        
        unique_patients = len(set([self._get_patient_id(entry[0]) for entry in self.graph_entries]))
        print(f"📊 Dataset initialized: {len(self.graph_entries)} graphs from {unique_patients} patients")
    
    def _find_graph_files(self, patient_ids: Optional[List[str]] = None) -> List[Tuple[str, int]]:
        """Find all graph files and expand to individual graphs"""
        if patient_ids is None:
            # Use all patients
            pattern = os.path.join(self.graph_dir, "BraTS2021_*", "*_graphs_*.pt")
            all_files = glob.glob(pattern)
        else:
            # Filter by specific patient IDs
            all_files = []
            for patient_id in patient_ids:
                pattern = os.path.join(self.graph_dir, patient_id, f"{patient_id}_graphs_*.pt")
                all_files.extend(glob.glob(pattern))
        
        # Expand to individual graphs: (file_path, graph_index)
        graph_entries = []
        for graph_file in sorted(all_files):
            try:
                # Load file to count graphs
                graphs = torch.load(graph_file, map_location='cpu', weights_only=False)
                if isinstance(graphs, list):
                    num_graphs = len(graphs)
                else:
                    num_graphs = 1
                
                # Add entry for each graph in the file
                for i in range(num_graphs):
                    graph_entries.append((graph_file, i))
                    
            except Exception as e:
                print(f"⚠️ Warning: Could not load {graph_file}: {e}")
                continue
        
        return graph_entries
    
    def _get_patient_id(self, graph_file: str) -> str:
        """Extract patient ID from graph file path"""
        return Path(graph_file).parent.name
    
    def __len__(self) -> int:
        return len(self.graph_entries)
    
    def __getitem__(self, idx: int) -> Data:
        """Load and return a specific graph"""
        graph_file, graph_idx = self.graph_entries[idx]
        
        try:
            # Load the graph with weights_only=False for PyTorch 2.6+
            graphs = torch.load(graph_file, map_location='cpu', weights_only=False)
            
            # Handle different formats
            if isinstance(graphs, list):
                # Multiple graphs in file - get the specific one
                graph = graphs[graph_idx]
            else:
                # Single graph (graph_idx should be 0)
                graph = graphs
            
            # Ensure it's a PyTorch Geometric Data object
            if not isinstance(graph, Data):
                raise ValueError(f"Expected torch_geometric.data.Data, got {type(graph)}")
            
            # Add metadata
            graph.patient_id = self._get_patient_id(graph_file)
            graph.graph_idx = graph_idx
            graph.graph_file = graph_file
            
            # Apply transform if specified
            if self.transform is not None:
                graph = self.transform(graph)
            
            return graph
            
        except Exception as e:
            print(f"⚠️  Error loading {graph_file}: {e}")
            # Return a dummy graph with patient info to avoid crashing
            dummy_graph = Data(x=torch.zeros(1, 12), edge_index=torch.zeros(2, 0, dtype=torch.long))
            dummy_graph.patient_id = self._get_patient_id(graph_file)
            dummy_graph.graph_file = graph_file
            return dummy_graph

def create_data_splits(graph_dir: str, 
                      train_ratio: float = 0.7,
                      val_ratio: float = 0.15,
                      test_ratio: float = 0.15,
                      random_seed: int = 42) -> Tuple[List[str], List[str], List[str]]:
    """
    Create train/validation/test splits at patient level
    
    Args:
        graph_dir: Directory containing graph files
        train_ratio: Ratio for training set
        val_ratio: Ratio for validation set  
        test_ratio: Ratio for test set
        random_seed: Random seed for reproducibility
    
    Returns:
        Tuple of (train_patient_ids, val_patient_ids, test_patient_ids)
    """
    # Set random seed
    random.seed(random_seed)
    
    # Find all patient directories
    patient_dirs = glob.glob(os.path.join(graph_dir, "BraTS2021_*"))
    patient_ids = [Path(d).name for d in patient_dirs]
    
    # Shuffle patients
    random.shuffle(patient_ids)
    
    # Calculate split sizes
    n_total = len(patient_ids)
    n_train = int(n_total * train_ratio)
    n_val = int(n_total * val_ratio)
    
    # Create splits
    train_patients = patient_ids[:n_train]
    val_patients = patient_ids[n_train:n_train + n_val]
    test_patients = patient_ids[n_train + n_val:]
    
    print(f"📊 Data splits created:")
    print(f"   Training: {len(train_patients)} patients ({len(train_patients)/n_total*100:.1f}%)")
    print(f"   Validation: {len(val_patients)} patients ({len(val_patients)/n_total*100:.1f}%)")
    print(f"   Test: {len(test_patients)} patients ({len(test_patients)/n_total*100:.1f}%)")
    
    return train_patients, val_patients, test_patients

def create_datasets(graph_dir: str,
                   train_patients: List[str],
                   val_patients: List[str], 
                   test_patients: List[str],
                   max_graphs_per_patient: Optional[int] = None) -> Tuple[BraTSGraphDataset, BraTSGraphDataset, BraTSGraphDataset]:
    """
    Create train, validation, and test datasets
    
    Args:
        graph_dir: Directory containing graph files
        train_patients: List of training patient IDs
        val_patients: List of validation patient IDs
        test_patients: List of test patient IDs
        max_graphs_per_patient: Limit graphs per patient (for debugging)
    
    Returns:
        Tuple of (train_dataset, val_dataset, test_dataset)
    """
    train_dataset = BraTSGraphDataset(
        graph_dir=graph_dir,
        patient_ids=train_patients,
        max_graphs_per_patient=max_graphs_per_patient
    )
    
    val_dataset = BraTSGraphDataset(
        graph_dir=graph_dir,
        patient_ids=val_patients,
        max_graphs_per_patient=max_graphs_per_patient
    )
    
    test_dataset = BraTSGraphDataset(
        graph_dir=graph_dir,
        patient_ids=test_patients,
        max_graphs_per_patient=max_graphs_per_patient
    )
    
    return train_dataset, val_dataset, test_dataset

def analyze_dataset(dataset: BraTSGraphDataset, name: str = "Dataset") -> Dict:
    """
    Analyze dataset statistics
    
    Args:
        dataset: BraTSGraphDataset to analyze
        name: Name for logging
    
    Returns:
        Dictionary of statistics
    """
    print(f"\n🔍 Analyzing {name}...")
    
    if len(dataset) == 0:
        print("   ⚠️  Empty dataset!")
        return {}
    
    # Sample a few graphs to get statistics
    sample_size = min(100, len(dataset))
    sample_indices = random.sample(range(len(dataset)), sample_size)
    
    node_counts = []
    edge_counts = []
    feature_dims = []
    tumor_ratios = []
    
    for i in sample_indices:
        try:
            graph = dataset[i]
            node_counts.append(graph.x.shape[0])
            edge_counts.append(graph.edge_index.shape[1])
            feature_dims.append(graph.x.shape[1])
            
            # Calculate tumor ratio if y exists
            if hasattr(graph, 'y') and graph.y is not None:
                tumor_ratios.append(float(torch.mean(graph.y)))
            
        except Exception as e:
            print(f"   ⚠️  Error analyzing sample {i}: {e}")
    
    # Calculate statistics
    stats = {
        'total_graphs': len(dataset),
        'avg_nodes': np.mean(node_counts) if node_counts else 0,
        'avg_edges': np.mean(edge_counts) if edge_counts else 0,
        'feature_dim': feature_dims[0] if feature_dims else 0,
        'avg_tumor_ratio': np.mean(tumor_ratios) if tumor_ratios else 0,
    }
    
    print(f"   📊 Total graphs: {stats['total_graphs']}")
    print(f"   📊 Avg nodes per graph: {stats['avg_nodes']:.1f}")
    print(f"   📊 Avg edges per graph: {stats['avg_edges']:.1f}")
    print(f"   📊 Feature dimension: {stats['feature_dim']}")
    if tumor_ratios:
        print(f"   📊 Avg tumor ratio: {stats['avg_tumor_ratio']:.3f}")
    
    return stats

# Example usage and testing
if __name__ == "__main__":
    # Test the dataset
    graph_dir = "./data/graphs"
    
    # Create splits
    train_patients, val_patients, test_patients = create_data_splits(graph_dir)
    
    # Create datasets (limit for testing)
    train_dataset, val_dataset, test_dataset = create_datasets(
        graph_dir, train_patients, val_patients, test_patients,
        max_graphs_per_patient=5  # Limit for testing
    )
    
    # Analyze datasets
    analyze_dataset(train_dataset, "Training Set")
    analyze_dataset(val_dataset, "Validation Set") 
    analyze_dataset(test_dataset, "Test Set")
    
    # Test loading a sample
    if len(train_dataset) > 0:
        print(f"\n🔍 Testing sample loading...")
        sample = train_dataset[0]
        print(f"   Sample shape: {sample.x.shape}")
        print(f"   Edge shape: {sample.edge_index.shape}")
        print(f"   Patient ID: {sample.patient_id}")