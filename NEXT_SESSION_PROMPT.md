# Prompt for Next Session: Encyclopedia Britannica GraphRAG Pipeline

Copy this prompt to start a new Claude session:

---

## Context

I'm building a GraphRAG system for historical Encyclopaedia Britannica editions. The plan is documented in `/home/jic823/encyclopedia_history/GRAPHRAG_PLAN.md`.

## What's Done

1. **Parsing complete**: Two editions parsed into chunks
   - 1778 (2nd edition): 5,642 chunks in `/home/jic823/1815EncyclopediaBritannicaNLS/output/chunks_1778.jsonl`
   - 1823 (6th edition): ~12,000 chunks in `/home/jic823/1815EncyclopediaBritannicaNLS/output/chunks_1823.jsonl`

2. **Embedding model tested**: NV-Embed-v2 beat voyage-3 by 7.5% MRR
   - Results: 0.961 MRR, 100% Recall@5
   - Perfect on semantic drift queries (PHLOGISTON, BROADCAST, ATOM)
   - Requires H100 80GB, batch_size=2
   - Test results in `/home/jic823/encyclopedia_history/COMPARISON_REPORT.md`

3. **Infrastructure ready**:
   - Nibi cluster access with H100 GPUs
   - Neo4j at `bolt://206.12.90.118:7687` (user: neo4j, pass in .env)
   - Virtual env on Nibi: `~/projects/def-jic823/embedding_test_venv/`
   - NV-Embed-v2 already downloaded to Nibi HF cache

## What Needs to Be Done

1. **Create Nibi embedding script** (`scripts/embed_chunks_nv.py`):
   - Load chunks from JSONL
   - Embed with NV-Embed-v2 (batch_size=2)
   - Save embeddings + chunk_ids to JSON
   - Handle both editions

2. **Create SLURM job** (`slurm/embed_all_editions.sh`):
   - H100 80GB, 4 hours
   - Embed both 1778 and 1823 editions
   - Save to `embeddings/` directory

3. **Create Neo4j loader** (`scripts/load_neo4j.py`):
   - Read chunks JSONL + embeddings JSON
   - Create EB_Edition, EB_Article, EB_Chunk nodes
   - NV-Embed-v2 produces 4096-dim embeddings (not 1024 like voyage-3)
   - Create vector index for similarity search

4. **Test the pipeline** end-to-end

## Key Technical Details

- **NV-Embed-v2 embedding dimension**: 4096 (not 1024)
- **transformers version**: Must use 4.46.0 (newer versions break NV-Embed-v2)
- **Batch size**: 2 (larger causes OOM on H100)
- **No instruction prompt needed**: Plain embedding works best

## Repository

- GitHub: `git@github.com:jburnford/encyclopedia_history.git`
- Local: `/home/jic823/encyclopedia_history/`
- Nibi: `~/projects/def-jic823/encyclopedia_history/`

## Files to Read First

1. `/home/jic823/encyclopedia_history/GRAPHRAG_PLAN.md` - Full architecture plan
2. `/home/jic823/encyclopedia_history/STATUS.md` - Test results and history
3. `/home/jic823/encyclopedia_history/scripts/test_nv_embed.py` - Reference for how to use NV-Embed-v2

## Task

Create the scripts listed above to:
1. Embed both editions with NV-Embed-v2 on Nibi
2. Load the embeddings into Neo4j with proper schema
3. Enable vector similarity search

Start by reading the GRAPHRAG_PLAN.md file, then create the embedding script for Nibi.

---
