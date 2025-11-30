#!/usr/bin/env python3
"""
Fix inconsistent graph feature dimensions by regenerating problematic patients.
"""

import torch
import glob
import os
import sys
import subprocess
from collections import defaultdict
from pathlib import Path

def find_problematic_patients():
    """Find patients with inconsistent feature dimensions."""
    print("🔍 Scanning all graph files for dimension inconsistencies...")
    
    patient_features = defaultdict(list)
    graph_files = glob.glob('./data/graphs/*/BraTS2021_*_graphs_*.pt')
    
    problematic_patients = set()
    
    for graph_file in graph_files:
        try:
            graphs = torch.load(graph_file, weights_only=False)
            patient_id = graph_file.split('/')[-2]  # Extract BraTS2021_XXXXX
            
            if isinstance(graphs, list):
                for graph in graphs:
                    if hasattr(graph, 'x') and graph.x.shape[1] != 12:
                        problematic_patients.add(patient_id)
                        break
            else:
                if hasattr(graphs, 'x') and graphs.x.shape[1] != 12:
                    problematic_patients.add(patient_id)
        except Exception as e:
            print(f'❌ Error loading {graph_file}: {e}')
            continue
    
    return list(problematic_patients)

def backup_patient_graphs(patient_id):
    """Backup existing graphs before regeneration."""
    patient_dir = f"./data/graphs/{patient_id}"
    backup_dir = f"./data/graphs/{patient_id}_backup"
    
    if os.path.exists(patient_dir):
        print(f"📦 Backing up {patient_id}...")
        if os.path.exists(backup_dir):
            subprocess.run(['rm', '-rf', backup_dir], check=True)
        subprocess.run(['mv', patient_dir, backup_dir], check=True)
        return True
    return False

def regenerate_patient_graphs(patient_id):
    """Regenerate graphs for a specific patient."""
    print(f"🔄 Regenerating graphs for {patient_id}...")
    
    # Find original data path
    data_paths = [
        f"./BraTS2021_Training_Data/{patient_id}",
        f"./{patient_id}"  # In case it's in root
    ]
    
    data_path = None
    for path in data_paths:
        if os.path.exists(path):
            data_path = path
            break
    
    if not data_path:
        print(f"❌ Could not find data for {patient_id}")
        return False
    
    # Run graph construction for this specific patient
    cmd = [
        'python3', 'src/graph_construction.py',
        '--single_patient', data_path,
        '--output_dir', './data/graphs'
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode == 0:
        print(f"✅ Successfully regenerated {patient_id}")
        return True
    else:
        print(f"❌ Failed to regenerate {patient_id}")
        print(f"Error: {result.stderr}")
        return False

def main():
    """Main execution."""
    print("🎯 Starting graph consistency fix...")
    
    # Find problematic patients
    problematic_patients = find_problematic_patients()
    
    if not problematic_patients:
        print("✅ No inconsistent patients found!")
        return
    
    print(f"\n🎯 Found {len(problematic_patients)} patients with inconsistent features:")
    for patient in problematic_patients:
        print(f"  - {patient}")
    
    # Ask for confirmation
    response = input(f"\n❓ Regenerate graphs for these {len(problematic_patients)} patients? (y/N): ")
    if response.lower() != 'y':
        print("❌ Aborted by user")
        return
    
    # Process each patient
    success_count = 0
    for patient_id in problematic_patients:
        try:
            # Backup existing graphs
            if backup_patient_graphs(patient_id):
                # Regenerate graphs
                if regenerate_patient_graphs(patient_id):
                    success_count += 1
                else:
                    print(f"⚠️  Failed to regenerate {patient_id}, restoring backup...")
                    subprocess.run(['rm', '-rf', f'./data/graphs/{patient_id}'], check=True)
                    subprocess.run(['mv', f'./data/graphs/{patient_id}_backup', f'./data/graphs/{patient_id}'], check=True)
            else:
                print(f"⚠️  No existing graphs found for {patient_id}")
        except Exception as e:
            print(f"❌ Error processing {patient_id}: {e}")
    
    print(f"\n🎯 Regeneration complete: {success_count}/{len(problematic_patients)} successful")
    
    # Verify fix
    print("\n🔍 Verifying fix...")
    remaining_problems = find_problematic_patients()
    if not remaining_problems:
        print("✅ All inconsistencies fixed!")
    else:
        print(f"⚠️  {len(remaining_problems)} patients still have issues: {remaining_problems}")

if __name__ == "__main__":
    main()
