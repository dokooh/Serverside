#!/bin/bash
# Kaggle setup script for the model benchmarking suite

echo "Setting up Hugging Face Model Benchmark Suite for Kaggle..."

# Create directories
mkdir -p models results logs

# Install requirements
echo "Installing Python dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

# Set environment variables for better performance
export TOKENIZERS_PARALLELISM=false
export HF_HUB_DISABLE_SYMLINKS_WARNING=1
export TRANSFORMERS_OFFLINE=0

# Check system capabilities
echo "Checking system capabilities..."
python -c "
import torch
import sys
print(f'Python version: {sys.version}')
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA version: {torch.version.cuda}')
    print(f'GPU count: {torch.cuda.device_count()}')
    for i in range(torch.cuda.device_count()):
        print(f'GPU {i}: {torch.cuda.get_device_name(i)}')
        print(f'  Memory: {torch.cuda.get_device_properties(i).total_memory / 1e9:.1f} GB')
"

echo "Setup complete! You can now run:"
echo "  python main_orchestrator.py"

# Optional: Run a quick system check
echo "Running quick system resource check..."
python -c "
from resource_monitor import quick_resource_check
import json
print(json.dumps(quick_resource_check(), indent=2))
"

echo "Setup finished successfully!"