#!/bin/bash
# One-shot setup script for musica-inn
# Run on login node: bash setup_musica_inn.sh
set -euo pipefail

echo "=== Setting up TurboGNN environment on musica-inn ==="

# 1. Install Miniconda if not present
if [ ! -d "$HOME/miniconda3" ]; then
    echo "Installing Miniconda..."
    wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh
    bash /tmp/miniconda.sh -b -p "$HOME/miniconda3"
    rm /tmp/miniconda.sh
    echo "Miniconda installed."
else
    echo "Miniconda already installed."
fi

# Initialize conda
source "$HOME/miniconda3/etc/profile.d/conda.sh"

# 2. Create turbognn environment
if conda env list | grep -q "turbognn"; then
    echo "turbognn env already exists."
else
    echo "Creating turbognn conda environment..."
    conda create -y -n turbognn python=3.10
fi

conda activate turbognn

# 3. Install PyTorch with CUDA 12.1
echo "Installing PyTorch..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# 4. Install PyG (torch-geometric)
echo "Installing PyG..."
pip install torch-geometric

# 5. Install other dependencies
echo "Installing scanpy, pandas, networkx, scipy..."
pip install scanpy pandas networkx scipy matplotlib

# 6. Create directory structure
mkdir -p ~/TurboGNN/data/processed
mkdir -p ~/TurboGNN/data/scperturb
mkdir -p ~/TurboGNN_Benchmark/results_benchmark/hvg200
mkdir -p ~/TurboGNN_Benchmark/results_benchmark/hvg500
mkdir -p ~/TurboGNN_Benchmark/results_benchmark/hvg1000
mkdir -p ~/TurboGNN_Benchmark/logs

echo ""
echo "=== Setup complete ==="
echo "Conda env: turbognn"
echo "Python: $(python --version)"
echo "PyTorch: $(python -c 'import torch; print(torch.__version__)')"
echo "CUDA available: $(python -c 'import torch; print(torch.cuda.is_available())')"
echo ""
echo "Next: transfer data files to ~/TurboGNN/data/"
