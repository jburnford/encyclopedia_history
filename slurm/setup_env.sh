#!/bin/bash
#SBATCH --account=def-jic823
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=0:30:00
#SBATCH --job-name=setup_embed_env

# Setup virtual environment for embedding tests
# Run once: sbatch slurm/setup_env.sh

module load python/3.11 cuda/12.6 arrow

VENV_DIR=~/projects/def-jic823/embedding_test_venv

# Create venv if it doesn't exist or is empty
if [ ! -f "$VENV_DIR/bin/activate" ]; then
    echo "Creating virtual environment..."
    python -m venv $VENV_DIR
fi

source $VENV_DIR/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install PyTorch with CUDA support
pip install torch --index-url https://download.pytorch.org/whl/cu121

# Install other dependencies
pip install transformers>=4.40 sentence-transformers>=2.6
pip install numpy pandas tqdm python-dotenv voyageai

# Verify installation
python -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}')"
python -c "from sentence_transformers import SentenceTransformer; print('sentence-transformers OK')"

echo "Environment setup complete!"
echo "Virtual environment: $VENV_DIR"
