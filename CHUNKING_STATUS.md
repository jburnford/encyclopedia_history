# Encyclopedia Parser: Chunking & Section Extraction Status

**Date:** December 23, 2025
**Module Version:** 0.3.1

---

## Overview

Built a comprehensive text processing pipeline for encyclopedia articles that creates a hierarchical structure suitable for both knowledge graphs and RAG-based retrieval.

```
Article → Sections → Chunks → Embeddings → Vector DB
                ↓
           Neo4j Knowledge Graph
```

---

## What's Implemented

### 1. Semantic Chunking (`chunkers.py`) ✅

**Pipeline:** `SemanticChunker(threshold=90) → merge(<300) → split(>4000, overlap=200)`

| Component | Purpose |
|-----------|---------|
| `TreatiseChunker` | LangChain SemanticChunker wrapper |
| `chunk_treatise(article)` | Chunk single article |
| `chunk_articles(articles)` | Batch processing with stats |
| `estimate_chunking_cost()` | Dry-run cost estimation |
| `ChunkingStats` | Tracks tokens, API calls, USD cost |

**Test Results (1815 Edition):**
| Article | Input | Chunks | Range | Average |
|---------|-------|--------|-------|---------|
| ANATOMY | 176K | 100 | 317-3,986 | 1,788 |
| HERALDRY | 174K | 214 | 300-3,963 | 1,364 |
| SCOTLAND | 160K | 147 | 326-3,999 | 1,955 |

**Key insight:** ~93 chunks per 100K chars, predictable scaling.

### 2. Section Extraction (`sections.py`) ✅

Three extraction strategies:

| Method | When Used | Cost |
|--------|-----------|------|
| **Explicit** | Articles with §, PART, SECT, CHAP markers | Free |
| **LLM Multi-Sample** | Long articles (>30K) without markers | ~$0.001/article |
| **Fallback** | When LLM unavailable | Free |

**Test Results:**
| Article | Sections | Method | Size Range |
|---------|----------|--------|------------|
| ANATOMY | 11 | explicit | 1K-45K |
| SCOTLAND | 11 | llm | 1K-31K |
| HERALDRY | 5 | explicit | 0.7K-99K (with intro) |

**SCOTLAND sections (LLM-detected):**
- Geography and Boundaries (1K)
- Topography and Counties (13K)
- Flora and Fauna (1K)
- Natural Resources and Minerals (13K)
- Early History and Roman Invasion (31K)
- The Reign of Kenneth II (29K)
- The Usurpation of Macbeth (14K)
- The Reign of Malcolm II (15K)
- The Reign of David I (31K)
- The Claim of Edward I (6K)
- Succession Crisis (2K)

### 3. Section-Aware Chunking (`chunkers.py`) ✅ NEW

**Main function:** `chunk_article_with_sections(article)`

| Component | Purpose |
|-----------|---------|
| `chunk_article_with_sections()` | Extract sections + chunk within each section |
| `chunk_section()` | Chunk a single section (recursive for oversized) |
| `print_article_structure()` | Tree visualization of Article→Section→Chunk hierarchy |

**Features:**
- **Intro sections:** Automatically creates "Introduction" section for content before first explicit marker
- **Oversized section splitting:** Sections >30K chars are split at paragraph boundaries, then recursively chunked
- **Section metadata:** Every chunk has `section_title` and `section_index` fields
- **Sequential chunk indexing:** Chunks are indexed sequentially across all sections

**Example output:**
```
Article: HERALDRY (174,098 chars)
├── Section 0: Introduction (34,775 chars)
│   ├── Chunk 0 (2,045 chars)
│   └── ... 16 more chunks
├── Section 1: CHAP. III. Of the Charges. (689 chars)
│   └── Chunk 17 (689 chars)
├── Section 2: SECT. I. Of Honourable Ordinaries. (99,775 chars)
│   ├── Chunk 18 (2,036 chars)
│   └── ... 48 more chunks
├── Section 3: SECT. I. Of Crowns. (21,910 chars)
│   ├── Chunk 67 (2,191 chars)
│   └── ... 9 more chunks
└── Section 4: CHAP V. Of the Rules or Laws of Heraldry. (16,544 chars)
    ├── Chunk 77 (2,068 chars)
    └── ... 7 more chunks
```

### 4. Data Models (`models.py`) ✅

```python
class Article(BaseModel):
    headword: str
    text: str
    article_type: ArticleType  # dictionary, treatise, biographical, geographical, cross_reference
    edition_year: int
    # ... location, metadata fields

class Section(BaseModel):
    title: str
    level: int  # 1=PART, 2=SECT/§, 3=subsection
    parent_headword: str
    char_start: int
    char_end: int
    text: str
    extraction_method: Literal["explicit", "llm", "fallback"]

class TextChunk(BaseModel):
    text: str
    index: int
    parent_headword: str
    section_title: Optional[str]
    section_index: Optional[int]
    char_start: int
    char_end: int
```

---

## Issues Resolved ✅

### 1. Explicit Section Gaps - FIXED
HERALDRY now gets an "Introduction" section (34K chars) for content before the first CHAP marker.

### 2. Large Sections Not Split - FIXED
Sections >30K chars are now recursively split at paragraph boundaries before semantic chunking.

### 3. Chunking Not Section-Aware - FIXED
New `chunk_article_with_sections()` function chunks within section boundaries, preserving metadata.

---

## Target Knowledge Graph Schema

```
(Article {headword: "SCOTLAND", edition: 1815})
    │
    ├──[:HAS_SECTION]──> (Section {title: "Geography", index: 0})
    │                        ├──[:HAS_CHUNK]──> (Chunk {text: "...", index: 0})
    │                        └──[:HAS_CHUNK]──> (Chunk {text: "...", index: 1})
    │
    ├──[:HAS_SECTION]──> (Section {title: "Early History", index: 4})
    │                        ├──[:HAS_CHUNK]──> (Chunk {text: "...", index: 0})
    │                        └──[:HAS_CHUNK]──> ...
    └── ...
```

---

## Files Created/Modified

| File | Status | Purpose |
|------|--------|---------|
| `encyclopedia_parser/models.py` | Modified | Added Section model |
| `encyclopedia_parser/chunkers.py` | Created | Semantic chunking with merge/split |
| `encyclopedia_parser/sections.py` | Created | Section extraction (explicit + LLM) |
| `encyclopedia_parser/__init__.py` | Modified | Exports v0.3.0 |

---

## Dependencies

```bash
pip install langchain langchain-experimental langchain-openai openai pydantic
```

**Environment:**
```bash
export OPENAI_API_KEY="sk-..."
```

---

## Usage

```python
from encyclopedia_parser import (
    Article, Section, TextChunk,
    extract_sections, get_section_stats,
    chunk_treatise, chunk_articles,
)

# Load article
article = Article(headword="SCOTLAND", text="...", edition_year=1815, edition_name="Britannica 5th")

# Extract sections
sections = extract_sections(article, use_llm=True)
print(get_section_stats(sections))

# Chunk (currently whole article - needs section integration)
chunks = chunk_treatise(article)
```

---

## Cost Estimates

| Operation | Cost |
|-----------|------|
| Section extraction (LLM) | ~$0.001 per article |
| Semantic chunking | ~$0.008 per 100K chars |
| Final embeddings | ~$0.004 per 100K chars |

**Full 1815 edition (~1800 treatises):** ~$0.50 total

---

## Next Steps

1. **Fix explicit section gaps** - Add intro section for pre-marker content
2. **Integrate sections with chunking** - Chunk within sections, preserve hierarchy
3. **Split oversized sections** - Recursive splitting for sections >30K chars
4. **Build Neo4j loader** - Create Article → Section → Chunk graph structure
5. **Test retrieval quality** - Compare section-aware vs flat chunking for RAG

---

## Testing Commands

```bash
# Test section extraction
OPENAI_API_KEY="..." python3 -c "
from encyclopedia_parser import Article, extract_sections
import json

with open('articles_1815_clean.jsonl') as f:
    for line in f:
        a = json.loads(line)
        if a['headword'] == 'SCOTLAND':
            article = Article(headword=a['headword'], text=a['text'], edition_year=1815, edition_name='Britannica 5th')
            sections = extract_sections(article, use_llm=True)
            for s in sections:
                print(f'{s.title}: {len(s.text):,} chars')
            break
"
```
