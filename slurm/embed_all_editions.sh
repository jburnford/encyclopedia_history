#!/bin/bash
#SBATCH --account=def-jic823
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=8:00:00
#SBATCH --job-name=embed_encyclopedias

# Embed all encyclopedia editions with NV-Embed-v2
#
# This script automatically processes all chunks_*.jsonl files in data/
#
# Prerequisites:
# 1. Copy chunk files to Nibi:
#    scp /home/jic823/1815EncyclopediaBritannicaNLS/output/chunks_*.jsonl \
#        nibi:~/projects/def-jic823/encyclopedia_history/data/
#
# 2. Submit job:
#    cd ~/projects/def-jic823/encyclopedia_history && sbatch slurm/embed_all_editions.sh
#
# Timing estimates (~2 chunks/sec on H100):
# - 1771 (1st): ~3,000 chunks → ~25 minutes
# - 1778 (2nd): ~5,600 chunks → ~45 minutes
# - 1823 (6th): ~12,000 chunks → ~100 minutes
# - 1842 (7th): ~15,000 chunks → ~125 minutes (estimate)

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

# Find all chunk files
CHUNK_FILES=$(ls data/chunks_*.jsonl 2>/dev/null | sort)

if [ -z "$CHUNK_FILES" ]; then
    echo "ERROR: No chunk files found in data/"
    echo "Run: scp chunks_*.jsonl nibi:~/projects/def-jic823/encyclopedia_history/data/"
    exit 1
fi

echo "Found chunk files:"
for f in $CHUNK_FILES; do
    wc -l "$f" | awk '{print "  " $2 ": " $1 " chunks"}'
done
echo ""

# Track overall progress
TOTAL_START=$(date +%s)
SUCCESS_COUNT=0
FAIL_COUNT=0

# Process each edition
for CHUNK_FILE in $CHUNK_FILES; do
    # Extract year from filename (chunks_1778.jsonl -> 1778)
    YEAR=$(basename "$CHUNK_FILE" | sed 's/chunks_\([0-9]*\)\.jsonl/\1/')
    OUTPUT_FILE="embeddings/embeddings_${YEAR}.json"

    echo "=========================================="
    echo "=== Embedding $YEAR Edition ==="
    echo "=========================================="
    echo "Input:  $CHUNK_FILE"
    echo "Output: $OUTPUT_FILE"
    echo ""

    # Skip if already processed
    if [ -f "$OUTPUT_FILE" ]; then
        echo "SKIP: $OUTPUT_FILE already exists"
        echo ""
        SUCCESS_COUNT=$((SUCCESS_COUNT + 1))
        continue
    fi

    START_TIME=$(date +%s)
    python scripts/embed_chunks_nv.py \
        --input "$CHUNK_FILE" \
        --output "$OUTPUT_FILE" \
        --batch-size 2

    if [ $? -eq 0 ]; then
        END_TIME=$(date +%s)
        ELAPSED=$((END_TIME - START_TIME))
        echo ""
        echo "$YEAR edition completed in ${ELAPSED}s"
        SUCCESS_COUNT=$((SUCCESS_COUNT + 1))
    else
        echo ""
        echo "ERROR: $YEAR embedding failed!"
        FAIL_COUNT=$((FAIL_COUNT + 1))
    fi
    echo ""
done

TOTAL_END=$(date +%s)
TOTAL_ELAPSED=$((TOTAL_END - TOTAL_START))

echo "=========================================="
echo "EMBEDDING PIPELINE COMPLETE"
echo "=========================================="
echo "End time: $(date)"
echo "Total time: ${TOTAL_ELAPSED}s ($((TOTAL_ELAPSED / 60)) minutes)"
echo "Success: $SUCCESS_COUNT, Failed: $FAIL_COUNT"
echo ""
echo "Output files:"
ls -lh embeddings/embeddings_*.json 2>/dev/null || echo "  (no embedding files found)"
echo ""
echo "Next steps:"
echo "1. Copy embeddings to local:"
echo "   scp nibi:~/projects/def-jic823/encyclopedia_history/embeddings/*.json ./embeddings/"
echo ""
echo "2. Load to Neo4j (for each edition):"
echo "   python scripts/load_neo4j.py --chunks data/chunks_YEAR.jsonl --embeddings embeddings/embeddings_YEAR.json"
echo "=========================================="
