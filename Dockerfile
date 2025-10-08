# BraTS GNN Segmentation - Docker Configuration
# Build: docker build -t brats_gnn .
# Run: docker run --gpus all -v $(pwd)/data:/app/data brats_gnn

FROM pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    git \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first (for better Docker caching)
COPY requirements.txt requirements-minimal.txt ./

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY src/ ./src/
COPY *.py ./
COPY *.md ./
COPY *.sh ./

# Create necessary directories
RUN mkdir -p data/raw data/preprocessed data/graphs checkpoints research_results

# Make scripts executable
RUN chmod +x *.sh

# Set environment variables
ENV PYTHONPATH=/app/src:/app
ENV CUDA_VISIBLE_DEVICES=0

# Default command
CMD ["python", "test_installation.py"]