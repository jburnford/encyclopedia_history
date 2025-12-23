#!/bin/bash
#SBATCH --account=def-jic823
#SBATCH --gres=gpu:nvidia_h100_80gb_hbm3_3g.40gb:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=2:00:00
#SBATCH --job-name=nv_embed_test

# Run NV-Embed-v2 embedding test on encyclopedia articles
# Usage: sbatch slurm/run_nv_embed_test.sh

module load python/3.11 cuda/12.6 arrow

source ~/projects/def-jic823/embedding_test_venv/bin/activate

# Set HuggingFace cache
export HF_HOME=~/projects/def-jic823/models/huggingface_cache
export TRANSFORMERS_CACHE=$HF_HOME
export HF_HUB_CACHE=$HF_HOME

# Go to project directory
cd ~/projects/def-jic823/encyclopedia_history

# Load tokens from .env (copy .env to Nibi separately: scp .env nibi:~/projects/def-jic823/encyclopedia_history/)
if [ -f .env ]; then
    export $(grep -v '^#' .env | xargs)
fi

echo "=========================================="
echo "NV-Embed-v2 Encyclopedia Test"
echo "=========================================="
echo "Date: $(date)"
echo "Node: $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo ""

# Run test with instruction
echo "=== Running WITH historical instruction ==="
python scripts/test_nv_embed.py \
    --data data/articles_1815_clean.jsonl \
    --output results/nv_embed_with_instruction.json \
    --test-type all \
    --batch-size 16

# Run test without instruction (for comparison)
echo ""
echo "=== Running WITHOUT instruction ==="
python scripts/test_nv_embed.py \
    --data data/articles_1815_clean.jsonl \
    --output results/nv_embed_no_instruction.json \
    --test-type all \
    --no-instruction \
    --batch-size 16

echo ""
echo "=========================================="
echo "Test complete!"
echo "Results saved to results/"
echo "=========================================="
