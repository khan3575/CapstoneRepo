#!/usr/bin/env python3
"""
Check graphs that contain tumor nodes to verify last feature (tumor_ratio)
"""

import torch
import numpy as np
import glob
from pathlib import Path

def check_tumor_graphs():
    """Find graphs with tumor and check last feature"""
    
    graph_files = glob.glob('/mnt/bigdata/capstone/brats_gnn_segmentation/data/graphs/*/*.pt')
    
    print("="*80)
    print("Searching for graphs with TUMOR nodes to verify feature #12 (tumor_ratio)")
    print("="*80)
    
    tumor_found = 0
    checked = 0
    
    for graph_file in graph_files[:50]:  # Check first 50 files
        try:
            graphs = torch.load(graph_file, weights_only=False)
            
            for graph in graphs:
                if hasattr(graph, 'y'):
                    num_tumor = torch.sum(graph.y > 0).item()
                    
                    if num_tumor > 0:
                        # Found tumor! Check feature values
                        tumor_mask = graph.y > 0
                        last_features = graph.x[tumor_mask, -1].numpy()
                        
                        patient_id = Path(graph_file).parent.name
                        print(f"\n✅ Found {num_tumor} tumor nodes in {patient_id}")
                        print(f"   Feature #12 (LAST) values for tumor nodes:")
                        print(f"   Min: {last_features.min():.4f}, Max: {last_features.max():.4f}")
                        print(f"   Mean: {last_features.mean():.4f}, Median: {np.median(last_features):.4f}")
                        print(f"   Sample values: {last_features[:5]}")
                        
                        tumor_found += 1
                        
                        if tumor_found >= 10:
                            print("\n" + "="*80)
                            print(f"✅ Verified {tumor_found} graphs with tumor")
                            print("="*80)
                            return
            
            checked += 1
            
        except Exception as e:
            continue
    
    print(f"\n⚠️  Checked {checked} files, found {tumor_found} with tumor")

if __name__ == '__main__':
    check_tumor_graphs()
