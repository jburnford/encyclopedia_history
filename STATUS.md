# NV-Embed-v2 Testing Status

**Date**: December 23, 2025
**Project**: Encyclopedia History Embedding Comparison
**Status**: ✅ COMPLETED

---

## Executive Summary

**NV-Embed-v2 significantly outperforms voyage-3** for historical encyclopedia retrieval:

| Model | MRR | Recall@5 | Recall@10 |
|-------|-----|----------|-----------|
| voyage-3 (baseline) | 0.894 | 0.644 | - |
| **NV-Embed-v2 (no instruction)** | **0.961** | **1.000** | **1.000** |
| NV-Embed-v2 (with instruction) | 0.934 | 1.000 | 1.000 |

**Key Finding**: The historical instruction prompt slightly *reduced* performance (0.934 vs 0.961). The model's pretrained knowledge was sufficient without explicit guidance.

**Semantic Drift**: Perfect 1.000 MRR on all semantic drift queries (PHLOGISTON, BROADCAST, ATOM, ENTHUSIASM, PHYSICS).

---

## Objective

Compare **NV-Embed-v2** (instruction-tuned, #1 MTEB) against **voyage-3** (API) for historical encyclopedia retrieval, testing whether instruction-tuning helps with semantic drift in 1815 text.

---

## Final Results

### Overall Performance

| Model | MRR | Recall@5 | Recall@10 | Improvement |
|-------|-----|----------|-----------|-------------|
| voyage-3 | 0.894 | 0.644 | - | baseline |
| NV-Embed-v2 (no instruction) | 0.961 | 1.000 | 1.000 | **+7.5% MRR** |
| NV-Embed-v2 (with instruction) | 0.934 | 1.000 | 1.000 | +4.5% MRR |

### By Query Type

| Query Type | NV-Embed-v2 (no inst.) | NV-Embed-v2 (with inst.) |
|------------|------------------------|--------------------------|
| Leather/Tanning | 0.950 | 0.900 |
| Semantic Drift | 1.000 | 1.000 |
| Long Articles | 0.938 | 0.938 |

### Semantic Drift Queries (Perfect Performance)

| Query | Expected | Rank | MRR |
|-------|----------|------|-----|
| "combustion burning fire theory" | PHLOGISTON | 1 | 1.000 |
| "scattering seeds sowing field" | BROADCAST | 1 | 1.000 |
| "indivisible matter particle philosophy" | ATOM | 1 | 1.000 |
| "religious fervor divine inspiration" | ENTHUSIASM | 1 | 1.000 |
| "natural philosophy motion mechanics" | PHYSICS | 1 | 1.000 |

---

## Previous Results: voyage-3 vs voyage-context-3

| Model | MRR | Recall@5 | Finding |
|-------|-----|----------|---------|
| voyage-3 | 0.894-1.000 | 0.644-0.964 | Excellent baseline |
| voyage-context-3 | 0.909-1.000 | 0.508-0.893 | No improvement |

**Conclusion**: Context feature unnecessary for encyclopedia content (chunks are self-contained).

---

## Why NV-Embed-v2?

- **#1 on MTEB** (72.31 overall, 62.65 retrieval)
- **Instruction-tuned**: Can provide historical context prompt
- **CC-BY-NC-4.0**: Non-commercial use (our project qualifies)

### Historical Instruction Prompt (Tested)

```
Given a query about 18th-century knowledge from the 1815 Encyclopaedia Britannica,
retrieve relevant passages. Note that historical terminology may differ from modern
usage - for example, 'phlogiston' was the accepted theory of combustion, 'physics'
often meant medicine, and 'broadcast' meant scattering seeds.
```

**Result**: Instruction did not improve results; model's pretrained knowledge sufficient.

---

## Infrastructure Setup

### Repository
- **GitHub**: `git@github.com:jburnford/encyclopedia_history.git`
- **Local**: `/home/jic823/encyclopedia_history/`
- **Nibi**: `~/projects/def-jic823/encyclopedia_history/`

### Nibi Environment
- **venv**: `~/projects/def-jic823/embedding_test_venv/`
- **Model cache**: `~/projects/def-jic823/models/huggingface_cache/` (15GB)
- **Python**: 3.11, CUDA 12.6
- **transformers**: 4.46.0 (downgraded for NV-Embed-v2 compatibility)

### Data
- **File**: `data/articles_1815_clean.jsonl` (94MB, 18,172 articles)
- **Test subset**: 27 articles, 760 chunks

---

## Issues Encountered & Resolved

### 1. transformers Version Incompatibility
**Error**: `AttributeError: 'DynamicCache' object has no attribute 'get_usable_length'`

**Cause**: transformers 4.57.3 broke NV-Embed-v2 API

**Fix**: Downgrade to transformers 4.46.0
```bash
pip install transformers==4.46.0 sentence-transformers==2.7.0
```

### 2. Missing `datasets` Package
**Error**: `ImportError: This modeling file requires the following packages: datasets`

**Fix**: Load arrow module before pip install (Compute Canada requirement)
```bash
module load arrow
pip install datasets
```

### 3. CUDA Out of Memory

| GPU Slice | VRAM | Result |
|-----------|------|--------|
| 2g.20gb | 20GB | OOM on model load |
| 3g.40gb | 40GB | OOM during inference (batch_size=16) |
| h100 (full) | 80GB | ✅ Success (batch_size=2) |

**Solution**: Request full H100 + reduce batch_size to 2

---

## Job History

| Job ID | GPU | Batch Size | Status | Notes |
|--------|-----|------------|--------|-------|
| 6178185 | CPU | - | ✅ Completed | venv setup |
| 6178332 | CPU | - | ❌ Failed | Model download (missing datasets) |
| 6180928 | CPU | - | ❌ Failed | Model download (transformers error) |
| 6184104 | 2g.20gb | 16 | ❌ OOM | 20GB insufficient |
| 6184721 | 3g.40gb | 16 | ❌ OOM | 40GB insufficient with batch=16 |
| **6189921** | **h100** | **2** | **✅ Completed** | Full 80GB, batch=2 |

---

## Test Queries

### Leather/Tanning Domain (10 queries)
- "vegetable tanning oak bark process"
- "hide preparation lime soaking"
- "morocco red leather Turkey dyeing"

### Semantic Drift (5 queries)
- "combustion burning fire theory" → expects PHLOGISTON ✅
- "scattering seeds sowing field" → expects BROADCAST ✅
- "indivisible matter particle philosophy" → expects ATOM ✅
- "religious fervor divine inspiration" → expects ENTHUSIASM ✅
- "natural philosophy motion mechanics" → expects PHYSICS ✅

### Long Articles (4 queries)
- "vitrification of materials" → expects GLASS
- "moveable type letters" → expects PRINTING

---

## Conclusions & Recommendations

### 1. Use NV-Embed-v2 for Production
- **+7.5% MRR improvement** over voyage-3
- **100% Recall@5** vs 64.4% for voyage-3
- Perfect semantic drift handling

### 2. Skip Instruction Prompts
- Instruction prompt slightly hurt performance
- Model's pretrained historical knowledge sufficient
- Simpler deployment without custom prompts

### 3. Hardware Requirements
- **Minimum**: 80GB VRAM (full H100)
- **Batch size**: 2 (conservative for large chunks)
- **Alternative**: stella_en_1.5B_v5 for smaller GPUs

### 4. Cost Comparison
| Model | Type | Cost/1M tokens |
|-------|------|----------------|
| voyage-3 | API | ~$0.06 |
| NV-Embed-v2 | Self-hosted | H100 compute only |

For large-scale processing (18K+ articles), self-hosted is more cost-effective.

---

## Files

```
encyclopedia_history/
├── README.md                    # Project overview
├── STATUS.md                    # This file
├── requirements.txt             # Python dependencies
├── .env                         # API keys (not in git)
├── .gitignore
├── data/
│   └── articles_1815_clean.jsonl
├── scripts/
│   ├── chunk_articles.py        # Chunking utilities
│   └── test_nv_embed.py         # Main test script
├── slurm/
│   ├── setup_env.sh             # venv setup
│   ├── download_model.sh        # Model download
│   └── run_nv_embed_test.sh     # Test job
└── results/
    ├── nv_embed_with_instruction.json
    └── nv_embed_no_instruction.json
```

---

## Commands Reference

```bash
# Check job status
ssh nibi "squeue -u jic823"

# View job output
ssh nibi "tail -50 ~/projects/def-jic823/encyclopedia_history/slurm-JOBID.out"

# Resubmit test
ssh nibi "cd ~/projects/def-jic823/encyclopedia_history && sbatch slurm/run_nv_embed_test.sh"

# Check GPU memory
ssh nibi "nvidia-smi"
```

---

## Phase 2: Two-Stage Retrieval with Reranking

### Target Architecture

```
Query → [NV-Embed-v2] → Top-50 → [BGE-Reranker-v2.5] → Top-5 → [GPT-OSS-120B]
         Fast Retrieval         Precise Reranking        Generation
         (~0.5ms/query)         (~50ms/query)
```

### Models

| Stage | Model | Parameters | VRAM | Purpose |
|-------|-------|------------|------|---------|
| Retrieval | NV-Embed-v2 | 7B | ~40GB | Fast bi-encoder, MTEB #1 |
| Reranking | BGE-Reranker-v2.5-gemma2 | 9B | ~40GB | Cross-encoder precision |
| Generation | GPT-OSS-120B | 120B | - | Already on Nibi |

### Why Two-Stage?

1. **Bi-encoder (NV-Embed-v2)**: Embeds query and documents separately. Fast but less precise.
2. **Cross-encoder (BGE-Reranker)**: Looks at query+document pairs together. Slower but more accurate.

For semantic drift queries, the reranker can catch subtle contextual clues the bi-encoder misses.

### Test Script

```bash
# Submit reranker test (full power, H100 80GB)
ssh nibi "cd ~/projects/def-jic823/encyclopedia_history && sbatch slurm/run_reranker_test.sh"
```

### Expected Results

| Configuration | MRR | Notes |
|---------------|-----|-------|
| NV-Embed-v2 only | 0.961 | Current baseline |
| + BGE-Reranker (full) | >0.97? | Hypothesis: reranking helps |
| + BGE-Reranker (8 layers) | ~0.96? | Speed vs accuracy tradeoff |

---

## Phase 3: Neo4j Knowledge Graph + RAG

### Architecture

```
encyclopedia_history/
├── Neo4j Graph Database
│   ├── EB_Article nodes (headword, edition_year)
│   ├── EB_Chunk nodes (text, embedding, section)
│   ├── Vector index for similarity search
│   └── Future: NER entities + relationships
│
├── Retrieval Pipeline
│   ├── voyage-3 embeddings (stored in Neo4j)
│   ├── Vector similarity search
│   └── BGE-Reranker for precision
│
└── Generation
    └── GPT-OSS-120B (on Nibi)
```

### Files Created

- `scripts/embed_and_load_neo4j.py` - Embedding + Neo4j loader
- `scripts/search_encyclopedia.py` - Interactive search
- `scripts/test_bge_reranker.py` - Reranker evaluation
- `slurm/run_reranker_test.sh` - SLURM job for reranker

### Neo4j Connection

```
URI: bolt://206.12.90.118:7687
User: neo4j
Pass: hl;kn258*vcA7492
```
