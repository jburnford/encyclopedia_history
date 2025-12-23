# Encyclopedia Britannica GraphRAG Pipeline

## Overview

Build a Graph-RAG system for historical Encyclopaedia Britannica editions using:
- **NV-Embed-v2** for embeddings (MTEB #1, tested +7.5% over voyage-3)
- **Neo4j** for graph storage with vector indexes
- **BGE-Reranker-v2.5** for query-time reranking (optional)
- **GPT-OSS-120B** for generation (already on Nibi)

## Architecture

```
┌──────────────────────────────────────────────────────────────────────┐
│                         INDEX TIME (Nibi H100)                        │
├──────────────────────────────────────────────────────────────────────┤
│                                                                      │
│   chunks_1778.jsonl ──┐                                              │
│   chunks_1823.jsonl ──┼──► [NV-Embed-v2] ──► embeddings_*.json       │
│   chunks_XXXX.jsonl ──┘        7B, H100                              │
│                                                                      │
└──────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌──────────────────────────────────────────────────────────────────────┐
│                         LOAD TO NEO4J (Local)                         │
├──────────────────────────────────────────────────────────────────────┤
│                                                                      │
│   embeddings_*.json ──► [Neo4j Loader] ──► Neo4j Graph Database      │
│                                                                      │
│   Nodes:                                                             │
│   - EB_Edition (year, title)                                         │
│   - EB_Article (headword, edition_year)                              │
│   - EB_Chunk (text, embedding, section_title)                        │
│                                                                      │
│   Relationships:                                                     │
│   - (Edition)-[:CONTAINS]->(Article)                                 │
│   - (Article)-[:HAS_CHUNK]->(Chunk)                                  │
│                                                                      │
│   Indexes:                                                           │
│   - Vector index on EB_Chunk.embedding (cosine, 4096 dims)           │
│                                                                      │
└──────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌──────────────────────────────────────────────────────────────────────┐
│                         QUERY TIME                                    │
├──────────────────────────────────────────────────────────────────────┤
│                                                                      │
│   Query ──► [NV-Embed-v2] ──► Vector Search ──► Top-50 candidates    │
│                                    │                                 │
│                                    ▼                                 │
│                          [BGE-Reranker-v2.5] ──► Top-5 final         │
│                                    │                                 │
│                                    ▼                                 │
│                            [GPT-OSS-120B] ──► Response               │
│                                                                      │
└──────────────────────────────────────────────────────────────────────┘
```

## Data

### Editions Parsed

| Edition | Year | Sections | Chunks | Size | Status |
|---------|------|----------|--------|------|--------|
| 2nd | 1778 | 2,827 | 5,642 | 12 MB | ✅ Ready |
| 6th | 1823 | ~5,200 | ~12,000 | 27 MB | ✅ Ready |

### Data Location

**Local (WSL):**
```
/home/jic823/1815EncyclopediaBritannicaNLS/output/
├── chunks_1778.jsonl      # 5,642 chunks
├── chunks_1823.jsonl      # ~12,000 chunks
├── sections_1778.jsonl    # Section metadata
└── sections_1823.jsonl    # Section metadata
```

**Nibi:**
```
~/projects/def-jic823/encyclopedia_history/
├── data/                  # Copy chunks here
├── embeddings/            # NV-Embed-v2 output
└── scripts/               # Processing scripts
```

### Chunk Format

```json
{
  "text": "CAABA, or CAABAM, properly signifies...",
  "index": 0,
  "parent_headword": "CAABA",
  "edition_year": 1778,
  "char_start": 0,
  "char_end": 529,
  "section_title": "Historical Significance",
  "section_index": 0
}
```

## NV-Embed-v2 Test Results

| Model | MRR | Recall@5 | Notes |
|-------|-----|----------|-------|
| voyage-3 | 0.894 | 0.644 | Baseline |
| **NV-Embed-v2** | **0.961** | **1.000** | +7.5% MRR |

- **Semantic drift queries**: Perfect 1.000 MRR (PHLOGISTON, BROADCAST, ATOM, etc.)
- **Instruction prompt**: Not needed (slightly hurt performance)
- **Hardware**: Requires H100 80GB, batch_size=2

## Pipeline Steps

### Step 1: Copy Data to Nibi

```bash
# From local WSL
scp /home/jic823/1815EncyclopediaBritannicaNLS/output/chunks_*.jsonl \
    nibi:~/projects/def-jic823/encyclopedia_history/data/
```

### Step 2: Create Embedding Script for Nibi

**Script:** `scripts/embed_chunks_nv.py`

- Load chunks from JSONL
- Embed with NV-Embed-v2 (batch_size=2)
- Save embeddings to JSON with chunk IDs
- Output: `embeddings/embeddings_1778.json`, `embeddings/embeddings_1823.json`

### Step 3: SLURM Job for Embedding

**Script:** `slurm/embed_all_editions.sh`

```bash
#SBATCH --gres=gpu:h100:1
#SBATCH --mem=64G
#SBATCH --time=4:00:00

# Embed 1778 edition (~5,600 chunks)
python scripts/embed_chunks_nv.py \
    --input data/chunks_1778.jsonl \
    --output embeddings/embeddings_1778.json

# Embed 1823 edition (~12,000 chunks)
python scripts/embed_chunks_nv.py \
    --input data/chunks_1823.jsonl \
    --output embeddings/embeddings_1823.json
```

### Step 4: Copy Embeddings Back to Local

```bash
# From local WSL
scp nibi:~/projects/def-jic823/encyclopedia_history/embeddings/*.json \
    /home/jic823/1815EncyclopediaBritannicaNLS/embeddings/
```

### Step 5: Load to Neo4j

**Script:** `scripts/load_neo4j.py`

- Read chunks JSONL + embeddings JSON
- Create EB_Edition, EB_Article, EB_Chunk nodes
- Create relationships
- Create vector index

**Neo4j Connection:**
```
URI: bolt://206.12.90.118:7687
User: neo4j
Password: hl;kn258*vcA7492
```

### Step 6: Test Vector Search

```cypher
// Find similar chunks
CALL db.index.vector.queryNodes('eb_chunk_embedding', 5, $query_embedding)
YIELD node, score
RETURN node.headword, node.section_title, score
```

## Neo4j Schema

### Nodes

```cypher
// Edition
CREATE (e:EB_Edition {
    year: 1778,
    title: "Encyclopaedia Britannica, 2nd Edition"
})

// Article
CREATE (a:EB_Article {
    headword: "CAABA",
    edition_year: 1778,
    chunk_count: 3
})

// Chunk (with embedding)
CREATE (c:EB_Chunk {
    chunk_id: "CAABA_1778_0",
    text: "...",
    headword: "CAABA",
    edition_year: 1778,
    chunk_index: 0,
    section_title: "Historical Significance",
    section_index: 0,
    char_start: 0,
    char_end: 529,
    embedding: [...]  // 4096-dim vector
})
```

### Indexes

```cypher
// Constraints
CREATE CONSTRAINT eb_article_id FOR (a:EB_Article)
    REQUIRE (a.headword, a.edition_year) IS UNIQUE;

CREATE CONSTRAINT eb_chunk_id FOR (c:EB_Chunk)
    REQUIRE c.chunk_id IS UNIQUE;

// Vector index (NV-Embed-v2 produces 4096-dim embeddings)
CREATE VECTOR INDEX eb_chunk_embedding FOR (c:EB_Chunk) ON c.embedding
    OPTIONS {indexConfig: {
        `vector.dimensions`: 4096,
        `vector.similarity_function`: 'cosine'
    }};
```

## Future: NER & Relationships

After embeddings are loaded, extract:

1. **Named Entities**
   - People (authors, historical figures)
   - Places (countries, cities, regions)
   - Concepts (scientific terms, philosophical ideas)
   - Dates/Time periods

2. **Relationships**
   - Cross-references between articles
   - Concept hierarchies
   - Geographic relationships
   - Temporal relationships

3. **Cross-Edition Links**
   - Same article across editions
   - Concept evolution over time

## Files to Create

| File | Location | Purpose |
|------|----------|---------|
| `scripts/embed_chunks_nv.py` | Nibi | NV-Embed-v2 embedding script |
| `slurm/embed_all_editions.sh` | Nibi | SLURM job for embedding |
| `scripts/load_neo4j.py` | Local | Load embeddings to Neo4j |
| `scripts/search_neo4j.py` | Local | Vector search utilities |

## Commands Summary

```bash
# 1. Copy data to Nibi
scp output/chunks_*.jsonl nibi:~/projects/def-jic823/encyclopedia_history/data/

# 2. Run embedding job on Nibi
ssh nibi "cd ~/projects/def-jic823/encyclopedia_history && sbatch slurm/embed_all_editions.sh"

# 3. Check job status
ssh nibi "squeue -u jic823"

# 4. Copy embeddings back
scp nibi:~/projects/def-jic823/encyclopedia_history/embeddings/*.json ./embeddings/

# 5. Load to Neo4j (local)
python scripts/load_neo4j.py --chunks output/chunks_1778.jsonl --embeddings embeddings/embeddings_1778.json
```

## References

- **NV-Embed-v2**: https://huggingface.co/nvidia/NV-Embed-v2
- **BGE-Reranker-v2.5**: https://huggingface.co/BAAI/bge-reranker-v2.5-gemma2-lightweight
- **Neo4j Vector Index**: https://neo4j.com/docs/cypher-manual/current/indexes/semantic-indexes/vector-indexes/
