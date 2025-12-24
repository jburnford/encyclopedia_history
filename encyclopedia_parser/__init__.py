"""
Encyclopedia Parser - Extract and classify articles from historical encyclopedias.

This module provides tools for parsing OCR'd encyclopedia text (MD and JSONL formats)
into structured articles with type classification, cross-reference extraction,
and semantic chunking for RAG applications.

Usage:
    from encyclopedia_parser import parse_markdown_file, parse_jsonl_file
    from encyclopedia_parser.models import Article, TextChunk

    articles = parse_markdown_file("path/to/volume.md", edition_year=1778)
    for article in articles:
        print(f"{article.headword}: {article.article_type}")

Chunking (Phase 2):
    from encyclopedia_parser import chunk_treatise, chunk_articles, classify_articles

    # Parse and classify
    articles = parse_markdown_file("volume.md", edition_year=1778)
    classified = classify_articles(articles)

    # Chunk a single treatise
    treatise = [a for a in classified if a.article_type == "treatise"][0]
    chunks = chunk_treatise(treatise)

    # Batch chunk with cost tracking
    chunks, stats = chunk_articles(classified)
    print(stats)  # Shows estimated cost

Section-Aware Chunking (Phase 2.5):
    from encyclopedia_parser import chunk_article_with_sections, print_article_structure

    # Chunk with section awareness
    sections, chunks = chunk_article_with_sections(article)

    # Visualize structure
    print(print_article_structure(article, sections, chunks))

    # Output:
    # Article: SCOTLAND (160,234 chars)
    # ├── Section 0: Geography (1,234 chars)
    # │   └── Chunk 0 (1,234 chars)
    # ├── Section 1: Early History (31,456 chars)
    # │   ├── Chunk 1 (2,345 chars)
    # │   └── ... 12 more chunks
"""

from .models import Article, TextChunk, CrossReference, EditionConfig, Section, ArticleType
from .extractors.md_extractor import MarkdownExtractor, parse_markdown_file
from .extractors.jsonl_extractor import JsonlExtractor, parse_jsonl_file
from .classifiers import classify_article, classify_articles, ArticleClassifier
from .chunkers import (
    chunk_treatise,
    chunk_articles,
    estimate_chunking_cost,
    TreatiseChunker,
    ChunkingStats,
    # Section-aware chunking (Phase 2.5)
    chunk_article_with_sections,
    chunk_section,
    print_article_structure,
)
from .sections import (
    extract_sections,
    extract_explicit_sections,
    extract_sections_with_llm,
    get_section_stats,
)
from .patterns import PATTERNS

__version__ = "0.3.1"
__all__ = [
    # Models
    "Article",
    "ArticleType",
    "TextChunk",
    "CrossReference",
    "EditionConfig",
    "Section",
    # Extractors
    "MarkdownExtractor",
    "JsonlExtractor",
    "parse_markdown_file",
    "parse_jsonl_file",
    # Classifiers
    "classify_article",
    "classify_articles",
    "ArticleClassifier",
    # Sections (Phase 2.5)
    "extract_sections",
    "extract_explicit_sections",
    "extract_sections_with_llm",
    "get_section_stats",
    # Chunkers (Phase 2)
    "chunk_treatise",
    "chunk_articles",
    "estimate_chunking_cost",
    "TreatiseChunker",
    "ChunkingStats",
    # Section-aware chunking (Phase 2.5)
    "chunk_article_with_sections",
    "chunk_section",
    "print_article_structure",
    # Patterns
    "PATTERNS",
]
