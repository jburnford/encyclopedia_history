# Encyclopedia Parser - Development Status

**Last Updated:** December 23, 2025

---

## Overview

New modular parser for extracting articles from OLMoCR output (MD and JSONL formats).

**Location:** `/home/jic823/1815EncyclopediaBritannicaNLS/encyclopedia_parser/`

---

## Completed (Phase 1: Article Extraction)

### Module Structure
```
encyclopedia_parser/
├── __init__.py              # Main exports (v0.2.0)
├── models.py                # Pydantic models (Article, TextChunk, CrossReference, EditionConfig)
├── patterns.py              # Consolidated regex patterns for headword/reference detection
├── classifiers.py           # Article type classification (dictionary/treatise/bio/geo/xref)
├── chunkers.py              # Semantic chunking for treatises (Phase 2) ✅ NEW
├── extractors/
│   ├── __init__.py
│   ├── base.py              # Abstract base extractor class
│   ├── md_extractor.py      # Markdown file parser (main one for OLMoCR output)
│   └── jsonl_extractor.py   # JSONL file parser (uses md_extractor internally)
└── validators/
    ├── __init__.py
    └── rules.py             # Rule-based validation
```

### Test Results (2nd Edition, Vol 1)
| Metric | Value |
|--------|-------|
| Articles extracted | 2,471 |
| Dictionary entries | 1,723 |
| Geographical entries | 416 |
| Biographical entries | 138 |
| Cross-references | 113 |
| Treatises | 81 |
| With cross-refs extracted | 679 |
| With coordinates | 223 |

### Usage
```python
from encyclopedia_parser import parse_markdown_file, classify_articles
from encyclopedia_parser.models import get_edition_config

# Parse a volume
articles = parse_markdown_file(
    "ocr_results/britannica_pipeline_batch/britannica_nls_144850370.md",
    edition_year=1778,
    volume=1
)

# Classify articles
classified = classify_articles(articles, get_edition_config(1778))

# Filter by type
treatises = [a for a in classified if a.article_type == "treatise"]
geo_entries = [a for a in classified if a.article_type == "geographical"]
```

### Key Features
- **Multi-pattern detection:** Bold headwords (`**ABACUS**`), plain caps (`ABACUS,`), standalone, parenthetical
- **Front matter skipping:** Automatically detects where real content starts
- **Multi-sense handling:** Same headword with multiple meanings tracked with `sense` field
- **Confidence scoring:** Based on pattern type and validation rules
- **Edition configs:** Pre-configured for editions 1704-1860 with major treatise lists

### Known Limitations
1. Single-letter entries (A, B, C) with inline sub-meanings not fully captured
2. Some edge cases need LLM validation (Phase 1.3 - not implemented yet)

---

## Completed: Phase 2 - Treatise Chunking

**Status:** ✅ COMPLETE (December 23, 2025)

### Implementation

Created `encyclopedia_parser/chunkers.py` with:

1. **`TreatiseChunker` class** - Full-featured chunker with:
   - Lazy initialization of SemanticChunker (saves resources)
   - Configurable breakpoint threshold (percentile, standard_deviation, interquartile)
   - Section title detection from chunk content
   - Character position tracking for all chunks

2. **`chunk_treatise(article)` function** - Main entry point:
   - Short articles (<5000 chars) return single chunk
   - Only treatises get semantic chunking (cost-effective)
   - Falls back gracefully on errors

3. **`chunk_articles(articles)` function** - Batch processing with stats

4. **`estimate_chunking_cost(articles)` function** - Dry-run cost estimation

5. **`ChunkingStats` dataclass** - Tracks:
   - Articles processed/chunked
   - Total chunks and characters
   - Embedding API calls
   - Estimated tokens and USD cost

### Usage

```python
from encyclopedia_parser import (
    parse_markdown_file, classify_articles,
    chunk_treatise, chunk_articles, estimate_chunking_cost
)
from encyclopedia_parser.models import get_edition_config

# Parse and classify
articles = parse_markdown_file(
    "ocr_results/britannica_pipeline_batch/britannica_nls_144850370.md",
    edition_year=1778
)
classified = classify_articles(articles, get_edition_config(1778))

# Get treatises
treatises = [a for a in classified if a.article_type == "treatise"]

# Estimate cost before chunking (no API calls)
cost_estimate = estimate_chunking_cost(treatises)
print(f"Estimated cost: ${cost_estimate['estimated_cost_usd']:.4f}")

# Chunk a single treatise
chunks = chunk_treatise(treatises[0])
print(f"{treatises[0].headword}: {len(chunks)} chunks")

# Batch chunk with cost tracking
all_chunks, stats = chunk_articles(treatises)
print(stats)
```

### Dependencies Required
```bash
pip install langchain langchain-experimental langchain-openai openai
```

### Environment Variables
```bash
export OPENAI_API_KEY="sk-..."  # Required for semantic chunking
```

### Cost Model
- Uses `text-embedding-3-small` model ($0.02 per 1M tokens)
- Estimated 4 characters per token
- 2nd edition vol 1: ~81 treatises, estimated <$0.02 for full volume

---

## Phase 3 - Cross-Reference Extraction (Pending)

Extract "See X" references to build article relationship graph.

Already partially implemented in `patterns.py`:
- `extract_cross_references()` function exists
- `CrossReference` model defined
- Needs: deduplication, normalization, graph output

---

## Phase 4 - LLM Validation (Pending)

Use LLM to validate edge cases flagged with low confidence.

---

## OCR Data Locations

### Downloaded Locally
```
/home/jic823/1815EncyclopediaBritannicaNLS/ocr_results/
├── britannica_pipeline_batch/   # 82 JSONL + 82 MD + 82 metadata
│   └── (2nd, 6th, 7th, 8th editions + supplements)
├── 1797_britannica_3rd/         # 18 MD + 13 JSONL
└── (other editions in separate dirs)
```

### On Nibi Cluster
- `~/projects/def-jic823/britannica_pipeline/01_downloaded/`
- `~/projects/def-jic823/olmocr/encyclopedias/`

---

## Edition Coverage

| Edition | Year | Vols | OCR Status | Parse Status |
|---------|------|------|------------|--------------|
| Lexicon Technicum | 1704 | 1 | Complete | Not tested |
| Chambers | 1728 | 2 | Complete | Not tested |
| Coetlogon | 1745 | 2 | Complete | Not tested |
| Britannica 1st | 1771 | 3 | Complete | Not tested |
| Britannica 2nd | 1778 | 10 | Complete | **Tested** |
| Britannica 3rd | 1797 | 18 | Complete | Not tested |
| Britannica 4th | 1810 | 20 | Complete | Not tested |
| Britannica 5th | 1815 | 20 | Complete | Not tested |
| Britannica 6th | 1823 | 20 | Complete | Not tested |
| Britannica 7th | 1842 | 21 | Complete | Not tested |
| Britannica 8th | 1860 | 21 | Complete | Not tested |
