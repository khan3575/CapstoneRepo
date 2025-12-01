#!/bin/bash
# Phase 2 Mini-Test: Verify fixed graph construction works
# Tests on 5 patients, 5 epochs with resource limits (75% CPU/GPU)

set -e  # Exit on error

echo "=============================================================================="
echo "PHASE 2 MINI-TEST: Verify Ground-Truth Leakage Fix"
echo "=============================================================================="
echo ""
echo "📋 Test Configuration:"
echo "   - Patients: 5 (sampled from training set)"
echo "   - Epochs: 5 (quick sanity check)"
echo "   - Expected behavior: Training loss decreases, Dice starts LOW (~0.1-0.3)"
echo "   - Resource limit: 75% CPU (12/16 cores), 75% GPU (4608/6144 MB)"
echo ""

# Get base directory
BASE_DIR="/mnt/bigdata/capstone/brats_gnn_segmentation"
cd "$BASE_DIR"

# Activate virtual environment
source /mnt/bigdata/capstone/.env/bin/activate

# Create output directories
mkdir -p data/graphs_fixed_mini_test
mkdir -p logs/mini_test
mkdir -p checkpoints/mini_test

# Step 1: Select 5 random patients
echo "=============================================================================="
echo "STEP 1: Selecting 5 random test patients..."
echo "=============================================================================="

# Get first 5 patients from fold 0
python3 -c "
import json
with open('data/cv_folds/fold_0.json') as f:
    fold = json.load(f)
patients = fold['train_patients'][:5]
print('Selected patients:', patients)
with open('data/mini_test_patients.txt', 'w') as f:
    for p in patients:
        f.write(p + '\n')
"

PATIENTS_FILE="data/mini_test_patients.txt"
echo "✅ Patient list saved to: $PATIENTS_FILE"
cat "$PATIENTS_FILE"
echo ""

# Step 2: Generate graphs with FIXED code (no leakage)
echo "=============================================================================="
echo "STEP 2: Generating graphs with FIXED graph_construction.py (15 features)..."
echo "=============================================================================="
echo "⚙️  Using CONSERVATIVE resource limits (75% CPU/GPU)"
echo ""

# Run graph generation with resource limits
taskset -c 0-11 nice -n 10 python3 -c "
import sys
sys.path.insert(0, 'src')
from graph_construction import GraphBuilder, Config
from pathlib import Path
import numpy as np

# Read patient list
with open('$PATIENTS_FILE') as f:
    patients = [line.strip() for line in f]

print(f'🔄 Processing {len(patients)} patients...')

config = Config(
    n_superpixels=200,
    max_memory_gb=3.0  # Conservative for mini-test
)

builder = GraphBuilder(config)

for patient_id in patients:
    print(f'\n📊 Processing {patient_id}...')
    
    # Load preprocessed data
    data_path = Path('data/preprocessed') / patient_id / f'{patient_id}_preprocessed.npz'
    if not data_path.exists():
        print(f'⚠️  Not found: {data_path}')
        continue
    
    data = np.load(data_path)
    
    # Prepare volume data
    volume_data = {
        'T1': data['T1'],
        'T1ce': data['T1ce'],
        'T2': data['T2'],
        'FLAIR': data['FLAIR'],
        'label': data['label'],
        'brain_mask': data.get('brain_mask', (data['T1'] > 0).astype(np.float32))
    }
    
    # Build graphs
    graphs, segments = builder.process_volume(volume_data)
    
    if len(graphs) > 0:
        # Save
        import torch
        output_dir = Path('data/graphs_fixed_mini_test') / patient_id
        output_dir.mkdir(parents=True, exist_ok=True)
        
        torch.save(graphs, output_dir / f'{patient_id}_graphs_200.pt')
        np.save(output_dir / f'{patient_id}_segments_200.npy', segments)
        
        print(f'✅ Saved {len(graphs)} graphs for {patient_id}')
        print(f'   Features per node: {graphs[0].x.shape[1]}')
        print(f'   Expected: 15 features (NO LEAKAGE)')
        print(f'   Labels stored: Multi-class (0,1,2,4) for future-proofing')
        print(f'   Label range: {graphs[0].y.min():.0f} to {graphs[0].y.max():.0f}')
        
        # Verify feature count
        assert graphs[0].x.shape[1] == 15, f'ERROR: Expected 15 features, got {graphs[0].x.shape[1]}'
        
        # Verify labels are multi-class (not binary)
        unique_labels = torch.unique(graphs[0].y)
        print(f'   Unique labels in graphs: {unique_labels.tolist()}')
    else:
        print(f'⚠️  No graphs generated for {patient_id}')

print('\n✅ Graph generation complete!')
"

echo ""
echo "✅ Step 2 complete!"
echo ""

# Step 3: Quick training test (5 epochs)
echo "=============================================================================="
echo "STEP 3: Mini-training test (5 epochs)..."
echo "=============================================================================="
echo "🎯 Success criteria:"
echo "   - Model loads without 'leaked graph' assertion error"
echo "   - Training loss decreases"
echo "   - Dice starts LOW (0.1-0.3), NOT high (0.99)"
echo ""

# Create mini config
cat > configs/mini_test_config.yaml << 'EOF'
# Mini-test configuration
model:
  hidden_channels: 128
  gnn_out_channels: 64
  num_layers: 3
  dropout: 0.1
  gnn_type: 'sage'

training:
  epochs: 5
  batch_size: 16  # Smaller for mini-test
  learning_rate: 0.001
  weight_decay: 0.00001
  patience: 10

data:
  graph_dir: 'data/graphs_fixed_mini_test'
  num_workers: 3
EOF

echo "📝 Mini-test config created"
echo ""

# Run training with resource limits
echo "🚀 Starting training..."
taskset -c 0-11 nice -n 10 python3 -c "
import torch
torch.cuda.set_per_process_memory_fraction(0.75, device=0)
torch.set_num_threads(6)

import sys
sys.path.insert(0, 'src')
from train_cv_fold import train_fold
import json
from pathlib import Path

# Load fold data
with open('data/cv_folds/fold_0.json') as f:
    fold = json.load(f)

# Override with mini-test patients
with open('$PATIENTS_FILE') as f:
    mini_patients = [line.strip() for line in f]

fold['train_patients'] = mini_patients[:4]  # 4 for training
fold['val_patients'] = mini_patients[4:5]   # 1 for validation
fold['test_patients'] = mini_patients[4:5]  # 1 for testing

# Add graph file paths
from pathlib import Path
graph_dir = Path('data/graphs_fixed_mini_test')
fold['train_graphs'] = [str(graph_dir / p / f'{p}_graphs_200.pt') for p in fold['train_patients']]
fold['val_graphs'] = [str(graph_dir / p / f'{p}_graphs_200.pt') for p in fold['val_patients']]
fold['test_graphs'] = [str(graph_dir / p / f'{p}_graphs_200.pt') for p in fold['test_patients']]

# Mini-test arguments
class Args:
    fold_file = 'data/cv_folds/fold_0.json'
    fold = 0
    graph_dir = 'data/graphs_fixed_mini_test'
    output_dir = 'checkpoints/mini_test'
    epochs = 5
    batch_size = 16
    lr = 0.001
    hidden_channels = 128
    gnn_out_channels = 64
    num_layers = 3
    dropout = 0.1
    gnn_type = 'sage'
    accumulation_steps = 1
    num_workers = 3

args = Args()

print('🔧 Mini-test configuration:')
print(f'   Train patients: {len(fold[\"train_patients\"])}')
print(f'   Val patients: {len(fold[\"val_patients\"])}')
print(f'   Epochs: {args.epochs}')
print(f'   Batch size: {args.batch_size}')
print()

# Save modified fold
Path('checkpoints/mini_test').mkdir(parents=True, exist_ok=True)
fold_dir = Path('checkpoints/mini_test')
fold_file = fold_dir / 'fold_0.json'  # Must match expected filename format
with open(fold_file, 'w') as f:
    json.dump(fold, f)  # Save fold directly, not wrapped in {'fold_0': fold}

print('🚀 Starting mini-training...')
print('⚠️  If you see assertion error about 12 features, old graphs are being used!')
print()

# This will fail if old graphs (12 features) are loaded
try:
    results = train_fold(
        fold_idx=0,
        fold_dir=str(fold_dir),
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        hidden_channels=args.hidden_channels,
        num_layers=args.num_layers,
        accumulation_steps=args.accumulation_steps,
        device='cuda'
    )
    
    print()
    print('=' * 80)
    print('✅ MINI-TEST PASSED!')
    print('=' * 80)
    print(f'Final validation Dice: {results[\"best_val_dice\"]:.4f}')
    print(f'Expected range: 0.10-0.40 (should be LOW, not 0.99!)')
    print()
    
    if results['best_val_dice'] > 0.90:
        print('⚠️  WARNING: Dice > 0.90 is suspiciously high for 5 epochs!')
        print('    This might indicate remaining leakage.')
    elif results['best_val_dice'] < 0.05:
        print('⚠️  WARNING: Dice < 0.05 is very low.')
        print('    Model might not be learning. Check loss convergence.')
    else:
        print('✅ Dice is in expected range for clean data!')
    
    print()
    print('📊 Training history:')
    for epoch in results['history'][-3:]:
        print(f'   Epoch {epoch[\"epoch\"]}: train_dice={epoch[\"train\"][\"dice\"]:.4f}, val_dice={epoch[\"val\"][\"dice\"]:.4f}')
    
except AssertionError as e:
    print()
    print('=' * 80)
    print('❌ MINI-TEST FAILED!')
    print('=' * 80)
    print(f'Error: {e}')
    print()
    print('This means old leaked graphs (12 features) are still being loaded.')
    print('Solution: Make sure graphs are generated with FIXED graph_construction.py')
    sys.exit(1)

except Exception as e:
    print()
    print('=' * 80)
    print('❌ MINI-TEST ERROR!')
    print('=' * 80)
    print(f'Error: {e}')
    import traceback
    traceback.print_exc()
    sys.exit(1)
" 2>&1 | tee "logs/mini_test/mini_test_$(date +%Y%m%d_%H%M%S).log"

echo ""
echo "=============================================================================="
echo "PHASE 2 MINI-TEST COMPLETE"
echo "=============================================================================="
echo ""
echo "📁 Results saved to:"
echo "   - Graphs: data/graphs_fixed_mini_test/"
echo "   - Checkpoints: checkpoints/mini_test/"
echo "   - Logs: logs/mini_test/"
echo ""
echo "✅ If mini-test passed, proceed to full graph regeneration"
echo "❌ If mini-test failed, check error messages above"
