#!/usr/bin/env python3
"""
Phase 1: Check model architecture input dimensions
Read-only verification - does NOT modify any files
"""

import sys
import re
from pathlib import Path

def check_model_architecture():
    """Check model input dimensions from gnn_model.py"""
    
    model_file = '/mnt/bigdata/capstone/brats_gnn_segmentation/src/gnn_model.py'
    
    print("="*80)
    print("PHASE 1 VERIFICATION: Model Input Dimensions")
    print("="*80)
    print(f"\nReading: {model_file}\n")
    
    try:
        with open(model_file, 'r') as f:
            content = f.read()
        
        # Search for input dimension patterns
        patterns = [
            r'in_features\s*=\s*(\d+)',
            r'input_dim\s*=\s*(\d+)',
            r'in_channels\s*=\s*(\d+)',
            r'num_features\s*=\s*(\d+)',
            r'__init__.*in_dim\s*=\s*(\d+)',
            r'self\.conv1.*\(\s*(\d+)\s*,',
        ]
        
        found_dims = []
        
        for pattern in patterns:
            matches = re.finditer(pattern, content, re.IGNORECASE)
            for match in matches:
                dim = int(match.group(1))
                line_num = content[:match.start()].count('\n') + 1
                context_start = max(0, match.start() - 100)
                context_end = min(len(content), match.end() + 100)
                context = content[context_start:context_end]
                
                found_dims.append({
                    'dimension': dim,
                    'line': line_num,
                    'pattern': pattern,
                    'match': match.group(0),
                    'context': context
                })
        
        print(f"✅ Found {len(found_dims)} potential input dimension references:\n")
        
        for i, dim_info in enumerate(found_dims, 1):
            print(f"[{i}] Line {dim_info['line']}: {dim_info['match']}")
            print(f"    Dimension: {dim_info['dimension']}")
            print(f"    Context: ...{dim_info['context'].strip()[:100]}...")
            print()
        
        # Check for explicit feature count
        if '12' in [str(d['dimension']) for d in found_dims]:
            print("⚠️  CRITICAL: Model expects 12 input features!")
            print("    This confirms GT leakage if graphs have 12 features")
        elif '11' in [str(d['dimension']) for d in found_dims]:
            print("✅ Model expects 11 input features (no GT)")
        
        # Also search for comments about features
        print("\n" + "="*80)
        print("Searching for feature descriptions in comments:")
        print("="*80 + "\n")
        
        feature_comments = re.finditer(r'#.*feature.*', content, re.IGNORECASE)
        for match in feature_comments:
            line_num = content[:match.start()].count('\n') + 1
            print(f"Line {line_num}: {match.group(0).strip()}")
        
    except Exception as e:
        print(f"❌ ERROR reading model file: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "="*80)

if __name__ == '__main__':
    check_model_architecture()
