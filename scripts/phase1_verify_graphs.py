#!/usr/bin/env python3
"""
Phase 1: Verify graph feature dimensions and check for GT leakage
Read-only verification - does NOT modify any files
"""

import torch
import numpy as np
import random
import glob
from pathlib import Path
import json
from datetime import datetime

def verify_graphs():
    """Check 5 random graph files for feature dimensions"""
    
    # Find all graph files
    graph_files = glob.glob('/mnt/bigdata/capstone/brats_gnn_segmentation/data/graphs/*/*.pt')
    
    if len(graph_files) == 0:
        print("❌ ERROR: No graph files found!")
        return None
    
    print(f"✅ Found {len(graph_files)} total graph files")
    
    # Sample 5 random files
    sample_files = random.sample(graph_files, min(5, len(graph_files)))
    
    results = {
        'timestamp': datetime.now().isoformat(),
        'total_graph_files': len(graph_files),
        'sampled_files': [],
        'feature_dimensions': [],
        'consistent_dims': True,
        'sample_features': []
    }
    
    print("\n" + "="*80)
    print("PHASE 1 VERIFICATION: Graph Feature Dimensions")
    print("="*80)
    
    for i, graph_file in enumerate(sample_files, 1):
        patient_id = Path(graph_file).parent.name
        print(f"\n[{i}/5] Loading: {patient_id}")
        print(f"      Path: {graph_file}")
        
        try:
            # Load graph list (using weights_only=False for PyTorch 2.6+)
            graphs = torch.load(graph_file, weights_only=False)
            
            if len(graphs) == 0:
                print(f"      ⚠️  WARNING: Empty graph list")
                continue
            
            # Check first graph
            graph = graphs[0]
            
            # Get feature dimensions
            num_nodes = graph.x.shape[0]
            num_features = graph.x.shape[1]
            num_edges = graph.edge_index.shape[1]
            
            print(f"      ✅ Loaded successfully")
            print(f"      📊 Graphs in file: {len(graphs)}")
            print(f"      📊 First graph - Nodes: {num_nodes}, Features: {num_features}, Edges: {num_edges}")
            
            # Get first 3 feature vectors as sample
            sample_feats = graph.x[:3].numpy()
            print(f"      📋 First 3 node feature vectors (shape: {sample_feats.shape}):")
            for j, feat in enumerate(sample_feats):
                print(f"          Node {j}: [{', '.join([f'{v:.4f}' for v in feat[:6]])}..., {feat[-1]:.4f}]")
                print(f"                   (showing first 6 and LAST feature)")
            
            # Check labels
            if hasattr(graph, 'y'):
                num_tumor = torch.sum(graph.y > 0).item()
                print(f"      🎯 Tumor nodes: {num_tumor}/{num_nodes} ({100*num_tumor/num_nodes:.1f}%)")
            
            # Store results
            results['sampled_files'].append({
                'patient_id': patient_id,
                'path': graph_file,
                'num_graphs': len(graphs),
                'num_nodes': num_nodes,
                'num_features': num_features,
                'num_edges': num_edges,
                'sample_last_feature': float(sample_feats[0, -1])
            })
            
            results['feature_dimensions'].append(num_features)
            results['sample_features'].append(sample_feats.tolist())
            
        except Exception as e:
            print(f"      ❌ ERROR loading graph: {e}")
            import traceback
            traceback.print_exc()
    
    # Check consistency
    if len(results['feature_dimensions']) > 0:
        unique_dims = set(results['feature_dimensions'])
        results['consistent_dims'] = len(unique_dims) == 1
        results['detected_feature_dim'] = list(unique_dims)
        
        print("\n" + "="*80)
        print("SUMMARY:")
        print("="*80)
        print(f"✅ Successfully loaded: {len(results['sampled_files'])}/5 files")
        print(f"📊 Feature dimensions found: {unique_dims}")
        
        if len(unique_dims) == 1:
            dim = list(unique_dims)[0]
            print(f"✅ CONSISTENT: All graphs have {dim} features")
            
            if dim == 12:
                print(f"\n⚠️  CRITICAL: 12 features detected!")
                print(f"    This suggests ground-truth 'tumor_ratio' is included as feature #12")
                print(f"    Feature #12 values from samples: {[r['sample_last_feature'] for r in results['sampled_files']]}")
                print(f"    These are likely tumor ratios (0.0-1.0 range)")
            elif dim == 11:
                print(f"✅ GOOD: 11 features (no GT leakage)")
            else:
                print(f"⚠️  UNEXPECTED: {dim} features (expected 11 or 12)")
        else:
            print(f"❌ INCONSISTENT: Multiple feature dimensions found!")
            results['consistent_dims'] = False
    
    # Save results
    output_file = '/mnt/bigdata/capstone/brats_gnn_segmentation/logs/phase1_graph_verification.json'
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✅ Results saved to: {output_file}")
    print("="*80)
    
    return results

if __name__ == '__main__':
    verify_graphs()
