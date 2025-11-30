#!/usr/bin/env python3
"""
Installation Verification Script
===============================

This script tests that all required dependencies are properly installed
and the BraTS GNN system is ready to run.
"""

import sys
import importlib
from pathlib import Path

def test_import(module_name, display_name=None, optional=False):
    """Test if a module can be imported"""
    if display_name is None:
        display_name = module_name
    
    try:
        module = importlib.import_module(module_name)
        version = getattr(module, '__version__', 'unknown')
        print(f"✅ {display_name}: {version}")
        return True
    except ImportError as e:
        if optional:
            print(f"⚠️  {display_name}: Not installed (optional)")
        else:
            print(f"❌ {display_name}: Not installed - {e}")
        return not optional

def test_pytorch_setup():
    """Test PyTorch and CUDA setup"""
    try:
        import torch
        print(f"✅ PyTorch: {torch.__version__}")
        
        # Test CUDA
        if torch.cuda.is_available():
            print(f"✅ CUDA: Available (version {torch.version.cuda})")
            print(f"   GPU count: {torch.cuda.device_count()}")
            for i in range(torch.cuda.device_count()):
                gpu_name = torch.cuda.get_device_name(i)
                gpu_memory = torch.cuda.get_device_properties(i).total_memory / (1024**3)
                print(f"   GPU {i}: {gpu_name} ({gpu_memory:.1f}GB)")
        else:
            print("⚠️  CUDA: Not available (CPU-only mode)")
        
        return True
    except ImportError:
        print("❌ PyTorch: Not installed")
        return False

def test_torch_geometric():
    """Test PyTorch Geometric setup"""
    try:
        import torch_geometric
        print(f"✅ PyTorch Geometric: {torch_geometric.__version__}")
        
        # Test basic functionality
        from torch_geometric.data import Data
        from torch_geometric.nn import SAGEConv
        print("✅ PyTorch Geometric components: Working")
        
        return True
    except ImportError as e:
        print(f"❌ PyTorch Geometric: Not installed - {e}")
        return False

def test_project_structure():
    """Test if project structure is correct"""
    required_dirs = [
        'src',
        'checkpoints',
        'data',
        'research_results'
    ]
    
    required_files = [
        'src/train_maxpower.py',
        'src/graph_construction.py',
        'src/dataset.py',
        'comprehensive_evaluation.py',
        'baseline_comparison.py',
        'ablation_study.py'
    ]
    
    all_good = True
    
    print("\n📁 Project Structure:")
    for dir_name in required_dirs:
        if Path(dir_name).exists():
            print(f"✅ {dir_name}/")
        else:
            print(f"❌ {dir_name}/ - Missing")
            Path(dir_name).mkdir(exist_ok=True)
            print(f"   Created {dir_name}/")
    
    for file_name in required_files:
        if Path(file_name).exists():
            print(f"✅ {file_name}")
        else:
            print(f"❌ {file_name} - Missing")
            all_good = False
    
    return all_good

def main():
    print("🔍 BraTS GNN Installation Verification")
    print("=" * 50)
    
    # Python version check
    python_version = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    print(f"🐍 Python: {python_version}")
    
    if sys.version_info < (3, 8):
        print("❌ Python 3.8+ required")
        return False
    
    print("\n📦 Core Dependencies:")
    
    # Test core dependencies
    core_deps = [
        ('numpy', 'NumPy'),
        ('scipy', 'SciPy'),
        ('pandas', 'Pandas'),
        ('sklearn', 'scikit-learn'),
        ('matplotlib', 'Matplotlib'),
        ('seaborn', 'Seaborn'),
        ('tqdm', 'tqdm')
    ]
    
    all_good = True
    for module, name in core_deps:
        if not test_import(module, name):
            all_good = False
    
    print("\n🏥 Medical Imaging Dependencies:")
    
    # Test medical imaging dependencies
    medical_deps = [
        ('nibabel', 'NiBabel'),
        ('SimpleITK', 'SimpleITK'),
        ('skimage', 'scikit-image')
    ]
    
    for module, name in medical_deps:
        if not test_import(module, name):
            all_good = False
    
    # Test optional medical dependency
    test_import('medpy', 'MedPy', optional=True)
    
    print("\n🔥 Deep Learning Dependencies:")
    
    # Test PyTorch
    if not test_pytorch_setup():
        all_good = False
    
    # Test PyTorch Geometric
    if not test_torch_geometric():
        all_good = False
    
    print(f"\n📁 Project Structure:")
    if not test_project_structure():
        all_good = False
    
    print(f"\n🧪 Quick Functionality Test:")
    try:
        # Test basic imports from the project
        sys.path.append('src')
        
        # Test if we can import our modules
        test_modules = [
            'dataset',
            'graph_construction'
        ]
        
        for module in test_modules:
            try:
                importlib.import_module(module)
                print(f"✅ src.{module}: Importable")
            except ImportError as e:
                print(f"⚠️  src.{module}: Import issues - {e}")
    
    except Exception as e:
        print(f"⚠️  Project modules: Some import issues - {e}")
    
    print("\n" + "=" * 50)
    
    if all_good:
        print("🎉 INSTALLATION VERIFICATION SUCCESSFUL!")
        print("\n✅ All required dependencies are installed and working")
        print("✅ Project structure is complete")
        print("✅ System is ready for BraTS GNN training and evaluation")
        print("\n🚀 Next steps:")
        print("1. Prepare your BraTS data in ./data/raw/")
        print("2. Run: python src/preprocessing.py")
        print("3. Run: python src/graph_construction.py")
        print("4. Run: python src/train_maxpower.py")
    else:
        print("❌ INSTALLATION VERIFICATION FAILED!")
        print("\n📋 Issues found:")
        print("- Some required dependencies are missing")
        print("- Run: pip install -r requirements.txt")
        print("- Or use the automated installer: ./install.sh")
        return False
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)