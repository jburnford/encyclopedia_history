#!/bin/bash
#SBATCH --account=def-jic823
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=1:00:00
#SBATCH --job-name=download_nv_embed

# Download NV-Embed-v2 model to HuggingFace cache
# Run once: sbatch slurm/download_model.sh

module load python/3.11 arrow

source ~/projects/def-jic823/embedding_test_venv/bin/activate

# Set HuggingFace cache to project space (shared, persistent)
export HF_HOME=~/projects/def-jic823/models/huggingface_cache
export TRANSFORMERS_CACHE=$HF_HOME
export HF_HUB_CACHE=$HF_HOME

# Load HF token from .env (copy .env to Nibi separately)
cd ~/projects/def-jic823/encyclopedia_history
if [ -f .env ]; then
    export $(grep -v '^#' .env | xargs)
fi

mkdir -p $HF_HOME

echo "Downloading NV-Embed-v2 to $HF_HOME..."
echo "Model size: ~14GB"

# Download model using Python
python << 'EOF'
from sentence_transformers import SentenceTransformer
import os

print(f"HF_HOME: {os.environ.get('HF_HOME')}")

print("Loading model (this will download if not cached)...")
model = SentenceTransformer('nvidia/NV-Embed-v2', trust_remote_code=True)
print("Model loaded successfully!")

# Quick test
test_text = "This is a test sentence."
embedding = model.encode([test_text])
print(f"Test embedding shape: {embedding.shape}")
print("Download complete!")
EOF

echo "Model download finished!"
ls -lh $HF_HOME/
