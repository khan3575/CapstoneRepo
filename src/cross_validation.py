"""
Cross-Validation Infrastructure for BraTS GNN
Creates stratified patient-level k-fold splits ensuring no data leakage
"""

import os
import json
import random
import numpy as np
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple
import torch


def set_seed(seed: int = 42):
    """Set random seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_patient_graphs(graphs_dir: str) -> Dict[str, List[str]]:
    """
    Group graph files by patient ID
    
    Args:
        graphs_dir: Directory containing .pt graph files
        
    Returns:
        Dictionary mapping patient_id -> list of graph file paths
    """
    graphs_path = Path(graphs_dir)
    patient_graphs = defaultdict(list)
    
    # Find all .pt files
    graph_files = list(graphs_path.rglob("*.pt"))
    print(f"Found {len(graph_files)} total graph files")
    
    for graph_file in graph_files:
        # Extract patient ID from filename
        # Expected format: BraTS2021_XXXXX_*_graph.pt or similar
        filename = graph_file.name
        
        # Try to extract patient ID (BraTS2021_XXXXX)
        if "BraTS2021" in filename:
            parts = filename.split("_")
            if len(parts) >= 2:
                patient_id = f"{parts[0]}_{parts[1]}"  # e.g., BraTS2021_00000
                patient_graphs[patient_id].append(str(graph_file))
    
    print(f"Found {len(patient_graphs)} unique patients")
    return dict(patient_graphs)


def stratified_split_patients(
    patient_graphs: Dict[str, List[str]], 
    k: int = 5, 
    seed: int = 42
) -> List[List[str]]:
    """
    Create stratified k-fold split of patients
    
    Args:
        patient_graphs: Dictionary mapping patient_id -> graph files
        k: Number of folds
        seed: Random seed
        
    Returns:
        List of k folds, each containing list of patient IDs
    """
    set_seed(seed)
    
    # Get all patient IDs
    patient_ids = list(patient_graphs.keys())
    n_patients = len(patient_ids)
    
    # Shuffle patients
    random.shuffle(patient_ids)
    
    # Split into k folds (as equal as possible)
    folds = [[] for _ in range(k)]
    for idx, patient_id in enumerate(patient_ids):
        fold_idx = idx % k
        folds[fold_idx].append(patient_id)
    
    # Print fold statistics
    print(f"\nCreated {k} folds:")
    for i, fold in enumerate(folds):
        n_graphs = sum(len(patient_graphs[pid]) for pid in fold)
        print(f"  Fold {i}: {len(fold)} patients, {n_graphs} graphs")
    
    return folds


def create_cv_folds(
    graphs_dir: str,
    output_dir: str,
    k: int = 5,
    seed: int = 42,
    val_ratio: float = 0.1
):
    """
    Create k-fold cross-validation splits and save to disk
    
    For each fold:
    - Test set: fold i patients
    - Train set: folds != i patients (minus validation)
    - Val set: val_ratio of train set patients
    
    Args:
        graphs_dir: Directory containing graph files
        output_dir: Directory to save fold assignments
        k: Number of folds
        seed: Random seed
        val_ratio: Ratio of training data to use for validation
    """
    set_seed(seed)
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Get patient graphs
    print("=" * 80)
    print("CREATING CROSS-VALIDATION FOLDS")
    print("=" * 80)
    patient_graphs = get_patient_graphs(graphs_dir)
    
    if len(patient_graphs) == 0:
        raise ValueError(f"No patients found in {graphs_dir}")
    
    # Create k-fold split
    folds = stratified_split_patients(patient_graphs, k, seed)
    
    # CRITICAL: In proper k-fold CV, for fold i:
    # - Test: fold[i] patients ONLY
    # - Train+Val: ALL remaining patients (fold[0:i] + fold[i+1:k])
    # This is CORRECT because each patient appears in exactly 1 test set across all folds
    
    # For each fold, create train/val/test splits
    fold_assignments = {}
    
    for fold_idx in range(k):
        print(f"\n{'=' * 80}")
        print(f"FOLD {fold_idx}")
        print(f"{'=' * 80}")
        
        # Test set: ONLY current fold
        test_patients = folds[fold_idx]
        
        # Train+Val candidates: ALL other folds
        # This is CORRECT for k-fold CV: we train on k-1 folds, test on 1 fold
        remaining_patients = []
        for i in range(k):
            if i != fold_idx:
                remaining_patients.extend(folds[i])
        
        # Now split remaining into train/val
        # This creates the validation set from the k-1 training folds
        n_val = int(len(remaining_patients) * val_ratio)
        
        # Use deterministic split with fold-specific seed for reproducibility
        fold_seed = seed + fold_idx
        random.seed(fold_seed)
        random.shuffle(remaining_patients)
        
        val_patients = remaining_patients[:n_val]
        train_patients = remaining_patients[n_val:]
        
        # Reset seed to main seed
        random.seed(seed)
        
        # Get graph file paths for each set
        train_graphs = []
        val_graphs = []
        test_graphs = []
        
        for pid in train_patients:
            train_graphs.extend(patient_graphs[pid])
        for pid in val_patients:
            val_graphs.extend(patient_graphs[pid])
        for pid in test_patients:
            test_graphs.extend(patient_graphs[pid])
        
        # Store fold assignment
        fold_data = {
            'fold_idx': fold_idx,
            'train_patients': train_patients,
            'val_patients': val_patients,
            'test_patients': test_patients,
            'train_graphs': train_graphs,
            'val_graphs': val_graphs,
            'test_graphs': test_graphs,
            'stats': {
                'n_train_patients': len(train_patients),
                'n_val_patients': len(val_patients),
                'n_test_patients': len(test_patients),
                'n_train_graphs': len(train_graphs),
                'n_val_graphs': len(val_graphs),
                'n_test_graphs': len(test_graphs)
            }
        }
        
        fold_assignments[f'fold_{fold_idx}'] = fold_data
        
        # Print statistics
        print(f"Train: {len(train_patients)} patients, {len(train_graphs)} graphs")
        print(f"Val:   {len(val_patients)} patients, {len(val_graphs)} graphs")
        print(f"Test:  {len(test_patients)} patients, {len(test_graphs)} graphs")
        
        # Save individual fold file
        fold_file = output_path / f'fold_{fold_idx}.json'
        with open(fold_file, 'w') as f:
            json.dump(fold_data, f, indent=2)
        print(f"Saved fold assignment to {fold_file}")
    
    # Save complete fold assignments
    complete_file = output_path / 'all_folds.json'
    with open(complete_file, 'w') as f:
        json.dump({
            'k': k,
            'seed': seed,
            'val_ratio': val_ratio,
            'total_patients': len(patient_graphs),
            'total_graphs': sum(len(graphs) for graphs in patient_graphs.values()),
            'folds': fold_assignments
        }, f, indent=2)
    
    print(f"\n{'=' * 80}")
    print(f"COMPLETE! Saved all fold assignments to {complete_file}")
    print(f"{'=' * 80}\n")
    
    return fold_assignments


def load_fold_data(fold_dir: str, fold_idx: int) -> Dict:
    """
    Load fold assignment from disk
    
    Args:
        fold_dir: Directory containing fold JSON files
        fold_idx: Which fold to load (0 to k-1)
        
    Returns:
        Dictionary with train/val/test splits
    """
    fold_file = Path(fold_dir) / f'fold_{fold_idx}.json'
    
    if not fold_file.exists():
        raise FileNotFoundError(f"Fold file not found: {fold_file}")
    
    with open(fold_file, 'r') as f:
        fold_data = json.load(f)
    
    return fold_data


def get_fold_dataloaders(
    fold_dir: str,
    fold_idx: int,
    batch_size: int = 32,
    num_workers: int = 4,
    dataset_class=None
):
    """
    Create PyTorch DataLoaders for a specific fold
    
    Args:
        fold_dir: Directory containing fold assignments
        fold_idx: Which fold to use
        batch_size: Batch size for dataloaders
        num_workers: Number of worker processes
        dataset_class: Dataset class to use (must accept graph_files argument)
        
    Returns:
        train_loader, val_loader, test_loader
    """
    from torch.utils.data import DataLoader
    
    # Load fold data
    fold_data = load_fold_data(fold_dir, fold_idx)
    
    if dataset_class is None:
        # Import default dataset
        from dataset import BraTSGraphDataset
        dataset_class = BraTSGraphDataset
    
    # Create datasets
    train_dataset = dataset_class(
        root_dir=None,  # Not used when graph_files provided
        split='train',
        graph_files=fold_data['train_graphs']
    )
    
    val_dataset = dataset_class(
        root_dir=None,
        split='val',
        graph_files=fold_data['val_graphs']
    )
    
    test_dataset = dataset_class(
        root_dir=None,
        split='test',
        graph_files=fold_data['test_graphs']
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    print(f"Created DataLoaders for fold {fold_idx}:")
    print(f"  Train: {len(train_dataset)} graphs")
    print(f"  Val:   {len(val_dataset)} graphs")
    print(f"  Test:  {len(test_dataset)} graphs")
    
    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Create cross-validation folds')
    parser.add_argument('--graphs_dir', type=str, 
                       default='./data/graphs',
                       help='Directory containing graph files')
    parser.add_argument('--output_dir', type=str,
                       default='./data/cv_folds',
                       help='Directory to save fold assignments')
    parser.add_argument('--k', type=int, default=5,
                       help='Number of folds')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    parser.add_argument('--val_ratio', type=float, default=0.1,
                       help='Ratio of training data for validation')
    
    args = parser.parse_args()
    
    # Create folds
    fold_assignments = create_cv_folds(
        graphs_dir=args.graphs_dir,
        output_dir=args.output_dir,
        k=args.k,
        seed=args.seed,
        val_ratio=args.val_ratio
    )
    
    print("\n✅ Cross-validation folds created successfully!")
    print(f"   Fold assignments saved to: {args.output_dir}")
    print(f"   Ready to train {args.k} folds")
