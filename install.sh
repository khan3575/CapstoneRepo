#!/bin/bash
# BraTS GNN Segmentation - Easy Installation Script
# This script sets up the complete environment for running the BraTS GNN system

echo "🚀 BraTS GNN Segmentation - Installation Script"
echo "================================================"

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Check Python version
echo "🐍 Checking Python installation..."
if command_exists python3; then
    PYTHON_VERSION=$(python3 --version 2>&1 | awk '{print $2}')
    echo "✅ Python $PYTHON_VERSION found"
    
    # Check if Python version is 3.8 or higher
    PYTHON_MAJOR=$(echo $PYTHON_VERSION | cut -d. -f1)
    PYTHON_MINOR=$(echo $PYTHON_VERSION | cut -d. -f2)
    
    if [ "$PYTHON_MAJOR" -eq 3 ] && [ "$PYTHON_MINOR" -ge 8 ]; then
        echo "✅ Python version is compatible (3.8+)"
    else
        echo "❌ Python 3.8+ required. Current version: $PYTHON_VERSION"
        exit 1
    fi
else
    echo "❌ Python3 not found. Please install Python 3.8 or higher."
    exit 1
fi

# Check pip
echo "📦 Checking pip installation..."
if command_exists pip3; then
    echo "✅ pip3 found"
    PIP_CMD="pip3"
elif command_exists pip; then
    echo "✅ pip found"
    PIP_CMD="pip"
else
    echo "❌ pip not found. Please install pip."
    exit 1
fi

# Create virtual environment (recommended)
echo "🔧 Setting up virtual environment..."
read -p "Do you want to create a virtual environment? (recommended) [y/N]: " create_venv

if [[ $create_venv =~ ^[Yy]$ ]]; then
    if [ ! -d "venv" ]; then
        echo "Creating virtual environment..."
        python3 -m venv venv
    fi
    
    echo "Activating virtual environment..."
    source venv/bin/activate
    PIP_CMD="pip"
    echo "✅ Virtual environment activated"
else
    echo "⚠️  Installing globally (not recommended)"
fi

# Upgrade pip
echo "📦 Upgrading pip..."
$PIP_CMD install --upgrade pip

# Install PyTorch first (with CUDA support if available)
echo "🔥 Installing PyTorch..."
if command_exists nvidia-smi; then
    echo "🎮 NVIDIA GPU detected, installing PyTorch with CUDA support..."
    $PIP_CMD install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
else
    echo "💻 No NVIDIA GPU detected, installing CPU-only PyTorch..."
    $PIP_CMD install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
fi

# Install PyTorch Geometric
echo "🌐 Installing PyTorch Geometric..."
$PIP_CMD install torch-geometric

# Install remaining requirements
echo "📚 Installing all other dependencies..."
$PIP_CMD install -r requirements.txt

# Verify installation
echo "🔍 Verifying installation..."
python3 -c "
import torch
import torch_geometric
import numpy as np
import nibabel as nib
import SimpleITK as sitk
import sklearn
import matplotlib.pyplot as plt
import pandas as pd
import scipy
import tqdm

print('✅ All core dependencies successfully installed!')
print(f'PyTorch version: {torch.__version__}')
print(f'PyTorch Geometric version: {torch_geometric.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA version: {torch.version.cuda}')
    print(f'GPU count: {torch.cuda.device_count()}')
"

if [ $? -eq 0 ]; then
    echo ""
    echo "🎉 INSTALLATION COMPLETED SUCCESSFULLY!"
    echo "================================================"
    echo ""
    echo "📋 Next Steps:"
    echo "1. Edit config.yaml to set your BraTS dataset paths"
    echo "2. Run preprocessing: python src/preprocessing.py (paths from config.yaml)"
    echo "3. Run graph construction: python src/graph_construction.py (paths from config.yaml)"
    echo "4. Create CV folds: python src/cross_validation.py"
    echo "5. Train models: python src/train_cv_fold.py --fold_idx 0 (repeat for folds 0-4)"
    echo "6. Run ensemble: python src/inference_ensemble.py"
    echo ""
    echo "📖 For detailed instructions, see README.md and PIPELINE_DOCUMENTATION.md"
    echo "⚙️  All paths are configured in config.yaml - edit this file to adapt to your system"
    echo ""
    if [[ $create_venv =~ ^[Yy]$ ]]; then
        echo "💡 To activate the environment in future sessions:"
        echo "   source venv/bin/activate"
    fi
else
    echo "❌ Installation verification failed. Please check the error messages above."
    exit 1
fi