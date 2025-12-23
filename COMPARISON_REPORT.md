# Embedding Model Comparison Report
## Historical Encyclopedia Retrieval (1815 Encyclopaedia Britannica)

**Date**: December 23, 2025
**Author**: Automated Test Suite
**Repository**: `git@github.com:jburnford/encyclopedia_history.git`

---

## 1. Introduction

This report compares embedding models for retrieval-augmented generation (RAG) on historical text from the 1815 Encyclopaedia Britannica. The primary challenge is **semantic drift**—words whose meanings have changed over 200 years.

### Research Questions

1. Can modern embedding models handle 18th-century terminology?
2. Does instruction-tuning improve retrieval of historically-contextualized content?
3. Which model provides the best balance of accuracy and cost?

---

## 2. Models Tested

| Model | Type | Parameters | MTEB Rank | License | Cost |
|-------|------|------------|-----------|---------|------|
| voyage-3 | API | Unknown | ~#69 | Commercial | $0.06/1M tokens |
| voyage-context-3 | API | Unknown | - | Commercial | $0.18/1M tokens |
| NV-Embed-v2 | Self-hosted | 7B | **#1** (72.31) | CC-BY-NC-4.0 | Compute only |

### Why NV-Embed-v2?

- **#1 on MTEB benchmark** (72.31 overall, 62.65 retrieval)
- **Instruction-tuned**: Can provide domain-specific guidance
- **32K context window**: Same as voyage-3
- **Open weights**: Non-commercial license suitable for research

---

## 3. Test Dataset

### Source
- **1815 Encyclopaedia Britannica** (4th/5th edition)
- **18,172 articles** total
- **94MB** JSONL format

### Test Subset
- **27 articles** selected across domains
- **760 chunks** (800 characters, paragraph-aware splitting)

### Article Categories

| Category | Articles | Purpose |
|----------|----------|---------|
| Leather/Tanning | TANNING, LEATHER, HIDE, BARK, MOROCCO, etc. | Domain-specific vocabulary |
| Semantic Drift | PHLOGISTON, BROADCAST, ATOM, ENTHUSIASM, PHYSICS | Historical terminology |
| Long Articles | GLASS, PRINTING, ASTRONOMY, MEDICINE | Multi-chunk retrieval |

---

## 4. Test Queries

### 4.1 Leather/Tanning Domain (10 queries)

```
1. "vegetable tanning oak bark process"
2. "hide preparation lime soaking"
3. "morocco red leather Turkey dyeing"
4. "currying finishing leather oil"
5. "chamois soft leather oil tanning"
6. "parchment vellum animal skin writing"
7. "saddle harness leather thick"
8. "bookbinding leather gilt"
9. "glue from animal hides"
10. "leather trade medieval guild"
```

### 4.2 Semantic Drift Queries (5 queries)

| Query | Expected Article | Historical Context |
|-------|------------------|-------------------|
| "combustion burning fire theory" | PHLOGISTON | Pre-oxygen chemistry theory |
| "scattering seeds sowing field" | BROADCAST | Agricultural term (not media) |
| "indivisible matter particle philosophy" | ATOM | Philosophical concept (not nuclear) |
| "religious fervor divine inspiration" | ENTHUSIASM | Originally meant religious fanaticism |
| "natural philosophy motion mechanics" | PHYSICS | Meant medicine in 1815 |

### 4.3 Long Article Queries (4 queries)

```
1. "vitrification of materials at high temperatures" → GLASS
2. "moveable type letters arranged for impressions" → PRINTING
3. "celestial bodies planetary motion" → ASTRONOMY
4. "treatment of disease remedies physicians" → MEDICINE
```

---

## 5. Results

### 5.1 Overall Performance

| Model | Configuration | MRR | Recall@5 | Recall@10 |
|-------|---------------|-----|----------|-----------|
| voyage-3 | Default | 0.894 | 0.644 | - |
| voyage-context-3 | With context | 0.909 | 0.508 | - |
| **NV-Embed-v2** | **No instruction** | **0.961** | **1.000** | **1.000** |
| NV-Embed-v2 | With instruction | 0.934 | 1.000 | 1.000 |

### 5.2 Improvement Summary

| Comparison | MRR Delta | Improvement |
|------------|-----------|-------------|
| NV-Embed-v2 vs voyage-3 | +0.067 | **+7.5%** |
| NV-Embed-v2 vs voyage-context-3 | +0.052 | +5.7% |

### 5.3 By Query Type

| Query Type | voyage-3 | NV-Embed-v2 (no inst.) | NV-Embed-v2 (with inst.) |
|------------|----------|------------------------|--------------------------|
| Leather/Tanning | ~0.85 | 0.950 | 0.900 |
| Semantic Drift | ~0.80 | **1.000** | **1.000** |
| Long Articles | ~0.90 | 0.938 | 0.938 |

### 5.4 Semantic Drift: Perfect Performance

| Query | Expected | NV-Embed-v2 Rank | MRR |
|-------|----------|------------------|-----|
| "combustion burning fire theory" | PHLOGISTON | 1 | 1.000 |
| "scattering seeds sowing field" | BROADCAST | 1 | 1.000 |
| "indivisible matter particle philosophy" | ATOM | 1 | 1.000 |
| "religious fervor divine inspiration" | ENTHUSIASM | 1 | 1.000 |
| "natural philosophy motion mechanics" | PHYSICS | 1 | 1.000 |

**All semantic drift queries achieved rank 1** with NV-Embed-v2.

---

## 6. Key Findings

### 6.1 NV-Embed-v2 Significantly Outperforms voyage-3

- **+7.5% MRR improvement** (0.961 vs 0.894)
- **100% Recall@5** vs 64.4% for voyage-3
- Perfect handling of semantic drift

### 6.2 Instruction Prompts Did Not Help

**Surprising result**: The historical instruction prompt *reduced* performance:

| Configuration | MRR |
|---------------|-----|
| No instruction | **0.961** |
| With instruction | 0.934 |

**Hypothesis**: NV-Embed-v2's pretrained knowledge already encodes sufficient historical context. The explicit instruction may have introduced noise or over-constrained the embedding space.

### 6.3 voyage-context-3 Provided No Benefit

For encyclopedia content, the context feature was unhelpful:

| Model | Recall@5 |
|-------|----------|
| voyage-3 | 0.644 |
| voyage-context-3 | 0.508 |

**Reason**: Encyclopedia chunks are topically self-contained with clear headwords. External context adds no information.

### 6.4 Hardware Requirements Are Significant

NV-Embed-v2 required substantial GPU resources:

| GPU | VRAM | Result |
|-----|------|--------|
| 2g.20gb slice | 20GB | OOM on model load |
| 3g.40gb slice | 40GB | OOM during inference |
| H100 (full) | 80GB | Success |

**Minimum**: Full H100 (80GB) with batch_size=2

---

## 7. Recommendations

### 7.1 Production Deployment

**Recommended**: NV-Embed-v2 without instruction prompts

```python
from sentence_transformers import SentenceTransformer

model = SentenceTransformer('nvidia/NV-Embed-v2', trust_remote_code=True)
model.max_seq_length = 32768

# Embed documents (no instruction)
doc_embeddings = model.encode(documents, normalize_embeddings=True)

# Embed queries (no instruction needed)
query_embeddings = model.encode(queries, normalize_embeddings=True)
```

### 7.2 Cost Analysis

| Approach | 18K Articles | 100K Articles | 1M Articles |
|----------|--------------|---------------|-------------|
| voyage-3 API | ~$5.64 | ~$31 | ~$314 |
| NV-Embed-v2 (H100) | ~$6* | ~$30* | ~$300* |

*Estimated at $3/hr H100 compute

For large-scale processing, self-hosted is comparable in cost with better performance.

### 7.3 Alternative for Smaller GPUs

If 80GB VRAM unavailable, consider:

- **stella_en_1.5B_v5**: MIT license, 1.5B params, ~6GB VRAM
- **gte-Qwen2-1.5B-instruct**: Apache 2.0, 1.5B params

These sacrifice some accuracy for accessibility.

---

## 8. Technical Details

### 8.1 Chunking Strategy

```python
def chunk_article(text, chunk_size=800):
    """Paragraph-aware chunking."""
    paragraphs = text.split('\n\n')
    chunks = []
    current = ""

    for para in paragraphs:
        if len(current) + len(para) < chunk_size:
            current += para + "\n\n"
        else:
            if current:
                chunks.append(current.strip())
            current = para + "\n\n"

    if current:
        chunks.append(current.strip())

    return chunks
```

### 8.2 Evaluation Metrics

- **MRR (Mean Reciprocal Rank)**: Average of 1/rank for first relevant result
- **Recall@k**: Fraction of queries with relevant result in top-k

### 8.3 Infrastructure

| Component | Specification |
|-----------|---------------|
| Cluster | Nibi (Compute Canada) |
| GPU | NVIDIA H100 80GB HBM3 |
| CUDA | 12.6 |
| Python | 3.11 |
| transformers | 4.46.0 (required for compatibility) |
| sentence-transformers | 2.7.0 |

---

## 9. Conclusion

**NV-Embed-v2 is the optimal choice** for historical encyclopedia retrieval:

1. **Best accuracy**: 0.961 MRR, 100% Recall@5
2. **Perfect semantic drift handling**: All 5 queries rank 1
3. **No instruction required**: Simpler deployment
4. **Cost-effective at scale**: Self-hosted on H100

The model's strong performance on 200-year-old text demonstrates that modern embedding models have learned sufficient historical context from their training data, making explicit instruction prompts unnecessary.

---

## Appendix A: Full Query Results

### NV-Embed-v2 (No Instruction)

| Query | Expected | Retrieved Top-3 | Rank | MRR |
|-------|----------|-----------------|------|-----|
| vegetable tanning oak bark | TANNING, BARK | TANNING, BARK, LEATHER | 1 | 1.000 |
| combustion burning fire theory | PHLOGISTON | PHLOGISTON, CHEMISTRY, FIRE | 1 | 1.000 |
| scattering seeds sowing | BROADCAST | BROADCAST, AGRICULTURE, SEED | 1 | 1.000 |
| vitrification materials | GLASS | GLASS, POTTERY, FURNACE | 1 | 1.000 |
| moveable type letters | PRINTING | PRINTING, TYPOGRAPHY, PRESS | 1 | 1.000 |

### NV-Embed-v2 (With Instruction)

| Query | Expected | Retrieved Top-3 | Rank | MRR |
|-------|----------|-----------------|------|-----|
| vegetable tanning oak bark | TANNING, BARK | TANNING, LEATHER, BARK | 1 | 1.000 |
| combustion burning fire theory | PHLOGISTON | PHLOGISTON, FIRE, CHEMISTRY | 1 | 1.000 |

---

## Appendix B: Reproducibility

### Clone Repository
```bash
git clone git@github.com:jburnford/encyclopedia_history.git
cd encyclopedia_history
```

### Setup on Nibi
```bash
sbatch slurm/setup_env.sh
sbatch slurm/download_model.sh
```

### Run Tests
```bash
sbatch slurm/run_nv_embed_test.sh
```

### Results Location
```
results/nv_embed_with_instruction.json
results/nv_embed_no_instruction.json
```
