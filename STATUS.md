# NV-Embed-v2 Testing Status

**Date**: December 23, 2025
**Project**: Encyclopedia History Embedding Comparison

---

## Objective

Compare **NV-Embed-v2** (instruction-tuned, #1 MTEB) against **voyage-3** (API) for historical encyclopedia retrieval, testing whether instruction-tuning helps with semantic drift in 1815 text.

---

## Previous Results: voyage-3 vs voyage-context-3

| Model | MRR | Recall@5 | Finding |
|-------|-----|----------|---------|
| voyage-3 | 0.894-1.000 | 0.644-0.964 | Excellent baseline |
| voyage-context-3 | 0.909-1.000 | 0.508-0.893 | No improvement |

**Conclusion**: Context feature unnecessary for encyclopedia content (chunks are self-contained).

---

## Current Test: NV-Embed-v2

### Why NV-Embed-v2?

- **#1 on MTEB** (72.31 overall, 62.65 retrieval)
- **Instruction-tuned**: Can provide historical context prompt
- **CC-BY-NC-4.0**: Non-commercial use (our project qualifies)

### Historical Instruction Prompt

```
Given a query about 18th-century knowledge from the 1815 Encyclopaedia Britannica,
retrieve relevant passages. Note that historical terminology may differ from modern
usage - for example, 'phlogiston' was the accepted theory of combustion, 'physics'
often meant medicine, and 'broadcast' meant scattering seeds.
```

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

## Issues Encountered

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
| h100 (full) | 80GB | Testing now (batch_size=2) |

**Current fix**: Request full H100 + reduce batch_size to 2

---

## Job History

| Job ID | GPU | Batch Size | Status | Notes |
|--------|-----|------------|--------|-------|
| 6178185 | CPU | - | Completed | venv setup |
| 6178332 | CPU | - | Failed | Model download (missing datasets) |
| 6180928 | CPU | - | Failed | Model download (transformers error) |
| 6184104 | 2g.20gb | 16 | OOM | 20GB insufficient |
| 6184721 | 3g.40gb | 16 | OOM | 40GB insufficient with batch=16 |
| **6189921** | **h100** | **2** | **Running** | Full 80GB, batch=2 |

---

## Test Queries

### Leather/Tanning Domain (10 queries)
- "vegetable tanning oak bark process"
- "hide preparation lime soaking"
- "morocco red leather Turkey dyeing"

### Semantic Drift (5 queries)
- "combustion burning fire theory" → expects PHLOGISTON
- "scattering seeds sowing field" → expects BROADCAST
- "indivisible matter particle philosophy" → expects ATOM

### Long Articles (4 queries)
- "vitrification of materials" → expects GLASS
- "moveable type letters" → expects PRINTING

---

## Expected Results

| Metric | voyage-3 | NV-Embed-v2 Target |
|--------|----------|-------------------|
| MRR (overall) | 0.894 | > 0.90 |
| MRR (semantic drift) | TBD | > 0.80 |
| Recall@5 | 0.644 | > 0.70 |

**Hypothesis**: Instruction-tuning should help NV-Embed-v2 outperform voyage-3 on semantic drift queries.

---

## Next Steps

1. **Wait for job 6189921** to complete
2. **Analyze results** - compare with/without instruction
3. **If OOM persists**: Try stella_en_1.5B_v5 (MIT license, 1.5B params, ~6GB)
4. **Document findings** in final report

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
    └── .gitkeep
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
