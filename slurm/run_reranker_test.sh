#!/bin/bash
#SBATCH --account=def-jic823
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=2:00:00
#SBATCH --job-name=bge_reranker_test

# Test BGE-Reranker-v2.5-gemma2-lightweight on encyclopedia articles
# Two-stage retrieval: NV-Embed-v2 → BGE-Reranker

module load python/3.11 cuda/12.6 arrow

source ~/projects/def-jic823/embedding_test_venv/bin/activate

# Set HuggingFace cache
export HF_HOME=~/projects/def-jic823/models/huggingface_cache
export TRANSFORMERS_CACHE=$HF_HOME
export HF_HUB_CACHE=$HF_HOME

cd ~/projects/def-jic823/encyclopedia_history

# Load tokens from .env
if [ -f .env ]; then
    export $(grep -v '^#' .env | xargs)
fi

echo "==========================================="
echo "BGE-Reranker Two-Stage Retrieval Test"
echo "==========================================="
echo "Date: $(date)"
echo "Node: $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo ""

# Install FlagEmbedding if not already installed
pip install --quiet FlagEmbedding

# Check GPU memory
nvidia-smi

echo ""
echo "=== Running FULL-POWER reranker test ==="
echo "Using all 42 layers of gemma2-9b (SOTA accuracy)"
echo ""

# Full power - all layers, maximum accuracy
# NV-Embed-v2 retrieval → BGE-Reranker-v2.5 (full) reranking
python scripts/test_bge_reranker.py \
    --data data/articles_1815_clean.jsonl \
    --output results/reranker_full_power.json \
    --test-type all \
    --batch-size 2 \
    --retrieve-k 50

echo ""
echo "=== Running LIGHTWEIGHT comparison (8 layers) ==="

# Lightweight comparison - see accuracy vs speed tradeoff
python scripts/test_bge_reranker.py \
    --data data/articles_1815_clean.jsonl \
    --output results/reranker_lightweight.json \
    --test-type all \
    --batch-size 2 \
    --retrieve-k 50 \
    --lightweight \
    --cutoff-layers 8

echo ""
echo "==========================================="
echo "Reranker test complete!"
echo "Results saved to results/"
echo "==========================================="
