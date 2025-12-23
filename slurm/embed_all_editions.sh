#!/bin/bash
#SBATCH --account=def-jic823
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=4:00:00
#SBATCH --job-name=embed_encyclopedias

# Embed all encyclopedia editions with NV-Embed-v2
#
# Prerequisites:
# 1. Copy chunk files to Nibi:
#    scp /home/jic823/1815EncyclopediaBritannicaNLS/output/chunks_*.jsonl \
#        nibi:~/projects/def-jic823/encyclopedia_history/data/
#
# 2. Submit job:
#    cd ~/projects/def-jic823/encyclopedia_history && sbatch slurm/embed_all_editions.sh
#
# Estimates:
# - 1778 edition: ~5,600 chunks → ~45 minutes
# - 1823 edition: ~12,000 chunks → ~90 minutes
# - Total: ~2.5 hours (4 hour limit for safety)

module load python/3.11 cuda/12.6 arrow

source ~/projects/def-jic823/embedding_test_venv/bin/activate

# Set HuggingFace cache (model already downloaded)
export HF_HOME=~/projects/def-jic823/models/huggingface_cache
export TRANSFORMERS_CACHE=$HF_HOME
export HF_HUB_CACHE=$HF_HOME

# Go to project directory
cd ~/projects/def-jic823/encyclopedia_history

echo "=========================================="
echo "Encyclopedia Britannica Embedding Pipeline"
echo "=========================================="
echo "Date: $(date)"
echo "Node: $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader)"
echo ""

# Create embeddings directory if needed
mkdir -p embeddings

# Check for data files
echo "Checking data files..."
if [ ! -f "data/chunks_1778.jsonl" ]; then
    echo "ERROR: data/chunks_1778.jsonl not found!"
    echo "Run: scp /home/jic823/1815EncyclopediaBritannicaNLS/output/chunks_*.jsonl nibi:~/projects/def-jic823/encyclopedia_history/data/"
    exit 1
fi

if [ ! -f "data/chunks_1823.jsonl" ]; then
    echo "WARNING: data/chunks_1823.jsonl not found (optional)"
fi

echo ""
echo "=== Embedding 1778 Edition (2nd) ==="
echo "Input: data/chunks_1778.jsonl"
echo "Output: embeddings/embeddings_1778.json"
echo ""

START_1778=$(date +%s)
python scripts/embed_chunks_nv.py \
    --input data/chunks_1778.jsonl \
    --output embeddings/embeddings_1778.json \
    --batch-size 2

if [ $? -eq 0 ]; then
    END_1778=$(date +%s)
    ELAPSED_1778=$((END_1778 - START_1778))
    echo "1778 edition completed in ${ELAPSED_1778}s"
else
    echo "ERROR: 1778 embedding failed!"
    exit 1
fi

echo ""

# Only embed 1823 if file exists
if [ -f "data/chunks_1823.jsonl" ]; then
    echo "=== Embedding 1823 Edition (6th) ==="
    echo "Input: data/chunks_1823.jsonl"
    echo "Output: embeddings/embeddings_1823.json"
    echo ""

    START_1823=$(date +%s)
    python scripts/embed_chunks_nv.py \
        --input data/chunks_1823.jsonl \
        --output embeddings/embeddings_1823.json \
        --batch-size 2

    if [ $? -eq 0 ]; then
        END_1823=$(date +%s)
        ELAPSED_1823=$((END_1823 - START_1823))
        echo "1823 edition completed in ${ELAPSED_1823}s"
    else
        echo "ERROR: 1823 embedding failed!"
        exit 1
    fi
else
    echo "Skipping 1823 edition (file not found)"
fi

echo ""
echo "=========================================="
echo "EMBEDDING PIPELINE COMPLETE"
echo "=========================================="
echo "End time: $(date)"
echo ""
echo "Output files:"
ls -lh embeddings/embeddings_*.json 2>/dev/null || echo "  (no embedding files found)"
echo ""
echo "Next steps:"
echo "1. Copy embeddings to local:"
echo "   scp nibi:~/projects/def-jic823/encyclopedia_history/embeddings/*.json ./embeddings/"
echo ""
echo "2. Load to Neo4j:"
echo "   python scripts/load_neo4j.py --chunks data/chunks_1778.jsonl --embeddings embeddings/embeddings_1778.json"
echo "=========================================="
