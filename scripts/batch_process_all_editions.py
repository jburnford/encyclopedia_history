#!/usr/bin/env python3
"""
Batch process all Britannica editions with section-aware chunking.

This script:
1. Loads articles from existing JSONL files (1771, 1797, 1815)
2. Parses articles from OCR markdown files (1778, 1823, 1842, 1860)
3. Extracts sections and chunks each treatise article
4. Saves results to JSONL files

Usage:
    export OPENAI_API_KEY="sk-..."
    python3 batch_process_all_editions.py

Output:
    output/sections_YYYY.jsonl - Extracted sections per edition
    output/chunks_YYYY.jsonl - Semantic chunks per edition
    output/processing_stats.json - Statistics summary
"""

import json
import logging
import os
import sys
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Optional

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent))

from encyclopedia_parser import (
    Article, ArticleType, Section, TextChunk,
    parse_markdown_file, classify_articles,
)
from encyclopedia_parser.models import get_edition_config
from encyclopedia_parser.chunkers import chunk_article_with_sections
from encyclopedia_parser.sections import get_section_stats


@dataclass
class EditionStats:
    """Statistics for processing an edition."""
    edition_year: int
    edition_name: str
    total_articles: int = 0
    treatises_found: int = 0
    treatises_processed: int = 0
    total_sections: int = 0
    total_chunks: int = 0
    total_chars: int = 0
    processing_time_seconds: float = 0
    errors: list = None

    def __post_init__(self):
        if self.errors is None:
            self.errors = []


# Edition configurations
EDITIONS = {
    1771: {
        "name": "Britannica 1st",
        "jsonl": "articles_1771.jsonl",
        "ocr_dir": None,
    },
    1778: {
        "name": "Britannica 2nd",
        "jsonl": None,
        "ocr_dir": "ocr_results/britannica_pipeline_batch",
        "nls_pattern": "Second edition",
    },
    1797: {
        "name": "Britannica 3rd",
        "jsonl": "articles_1797.jsonl",
        "ocr_dir": None,
    },
    1815: {
        "name": "Britannica 5th",
        "jsonl": "articles_1815_clean.jsonl",
        "ocr_dir": None,
    },
    1823: {
        "name": "Britannica 6th",
        "jsonl": None,
        "ocr_dir": "ocr_results/britannica_pipeline_batch",
        "nls_pattern": "Sixth edition",
    },
    1842: {
        "name": "Britannica 7th",
        "jsonl": None,
        "ocr_dir": "ocr_results/britannica_pipeline_batch",
        "nls_pattern": "Seventh edition",
    },
    1860: {
        "name": "Britannica 8th",
        "jsonl": None,
        "ocr_dir": "ocr_results/britannica_pipeline_batch",
        "nls_pattern": "Eighth edition",
    },
}


def load_articles_from_jsonl(jsonl_path: str, edition_year: int, edition_name: str) -> list[Article]:
    """Load articles from a JSONL file and classify them."""
    articles = []
    with open(jsonl_path) as f:
        for line in f:
            data = json.loads(line)
            article = Article(
                headword=data['headword'],
                text=data['text'],
                edition_year=edition_year,
                edition_name=edition_name,
                article_type=data.get('article_type', 'dictionary'),
            )
            articles.append(article)

    # Classify articles if they don't have types
    if articles and articles[0].article_type == 'dictionary':
        logger.info(f"Classifying {len(articles)} articles from {jsonl_path}...")
        config = get_edition_config(edition_year)
        articles = classify_articles(articles, config)

    return articles


def load_articles_from_ocr(ocr_dir: str, nls_pattern: str, edition_year: int) -> list[Article]:
    """Load and parse articles from OCR markdown files."""
    ocr_path = Path(ocr_dir)
    all_articles = []

    # Find matching metadata files
    for meta_file in ocr_path.glob("*.meta.json"):
        with open(meta_file) as f:
            meta = json.load(f)

        title = meta.get("title", "")
        if nls_pattern not in title:
            continue

        # Find corresponding markdown file
        md_file = meta_file.with_suffix("").with_suffix(".md")
        if not md_file.exists():
            logger.warning(f"Markdown file not found: {md_file}")
            continue

        logger.info(f"Parsing: {title[:60]}...")

        try:
            articles = parse_markdown_file(str(md_file), edition_year=edition_year)
            config = get_edition_config(edition_year)
            articles = classify_articles(articles, config)
            all_articles.extend(articles)
        except Exception as e:
            logger.error(f"Error parsing {md_file}: {e}")

    return all_articles


def process_edition(
    edition_year: int,
    config: dict,
    output_dir: Path,
    openai_api_key: Optional[str] = None,
) -> EditionStats:
    """Process a single edition with section-aware chunking."""
    import time
    start_time = time.time()

    stats = EditionStats(
        edition_year=edition_year,
        edition_name=config["name"],
    )

    # Load articles
    logger.info(f"\n{'='*60}")
    logger.info(f"Processing {config['name']} ({edition_year})")
    logger.info(f"{'='*60}")

    if config.get("jsonl"):
        jsonl_path = Path(__file__).parent / config["jsonl"]
        if jsonl_path.exists():
            articles = load_articles_from_jsonl(str(jsonl_path), edition_year, config["name"])
            logger.info(f"Loaded {len(articles)} articles from {config['jsonl']}")
        else:
            logger.warning(f"JSONL file not found: {jsonl_path}")
            return stats
    else:
        ocr_dir = Path(__file__).parent / config["ocr_dir"]
        articles = load_articles_from_ocr(str(ocr_dir), config["nls_pattern"], edition_year)
        logger.info(f"Parsed {len(articles)} articles from OCR files")

    stats.total_articles = len(articles)

    # Filter to treatises
    treatises = [a for a in articles if a.article_type in (ArticleType.TREATISE, "treatise")]
    stats.treatises_found = len(treatises)
    logger.info(f"Found {len(treatises)} treatise articles")

    if not treatises:
        logger.warning("No treatises found, skipping")
        return stats

    # Process each treatise
    all_sections = []
    all_chunks = []

    for i, article in enumerate(treatises):
        if len(article.text) < 5000:
            continue  # Skip short articles

        logger.info(f"  [{i+1}/{len(treatises)}] {article.headword} ({len(article.text):,} chars)")

        try:
            sections, chunks = chunk_article_with_sections(
                article,
                openai_api_key=openai_api_key,
                use_llm_sections=True,
                max_section_size=30000,
            )

            # Convert to dicts for JSON serialization
            for section in sections:
                section_dict = {
                    "title": section.title,
                    "level": section.level,
                    "index": section.index,
                    "parent_headword": section.parent_headword,
                    "edition_year": section.edition_year,
                    "char_start": section.char_start,
                    "char_end": section.char_end,
                    "text_length": len(section.text),
                    "extraction_method": section.extraction_method,
                }
                all_sections.append(section_dict)

            for chunk in chunks:
                chunk_dict = {
                    "text": chunk.text,
                    "index": chunk.index,
                    "parent_headword": chunk.parent_headword,
                    "edition_year": chunk.edition_year,
                    "char_start": chunk.char_start,
                    "char_end": chunk.char_end,
                    "section_title": chunk.section_title,
                    "section_index": chunk.section_index,
                }
                all_chunks.append(chunk_dict)

            stats.treatises_processed += 1
            stats.total_sections += len(sections)
            stats.total_chunks += len(chunks)
            stats.total_chars += len(article.text)

        except Exception as e:
            logger.error(f"    Error processing {article.headword}: {e}")
            stats.errors.append({"headword": article.headword, "error": str(e)})

    # Save results
    sections_file = output_dir / f"sections_{edition_year}.jsonl"
    chunks_file = output_dir / f"chunks_{edition_year}.jsonl"

    with open(sections_file, 'w') as f:
        for section in all_sections:
            f.write(json.dumps(section) + '\n')

    with open(chunks_file, 'w') as f:
        for chunk in all_chunks:
            f.write(json.dumps(chunk) + '\n')

    stats.processing_time_seconds = time.time() - start_time

    logger.info(f"Saved {len(all_sections)} sections to {sections_file.name}")
    logger.info(f"Saved {len(all_chunks)} chunks to {chunks_file.name}")
    logger.info(f"Processing time: {stats.processing_time_seconds:.1f}s")

    return stats


def main():
    # Check for API key
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        logger.error("OPENAI_API_KEY not set. Please set it and re-run.")
        sys.exit(1)

    # Create output directory
    output_dir = Path(__file__).parent / "output"
    output_dir.mkdir(exist_ok=True)

    # Process each edition
    all_stats = []

    for edition_year, config in EDITIONS.items():
        try:
            stats = process_edition(edition_year, config, output_dir, api_key)
            all_stats.append(asdict(stats))
        except Exception as e:
            logger.error(f"Failed to process {config['name']}: {e}")
            all_stats.append({
                "edition_year": edition_year,
                "edition_name": config["name"],
                "error": str(e),
            })

    # Save summary stats
    stats_file = output_dir / "processing_stats.json"
    with open(stats_file, 'w') as f:
        json.dump({
            "timestamp": datetime.now().isoformat(),
            "editions": all_stats,
        }, f, indent=2)

    # Print summary
    print("\n" + "="*60)
    print("PROCESSING SUMMARY")
    print("="*60)

    total_sections = 0
    total_chunks = 0

    for stats in all_stats:
        if "error" in stats and stats.get("total_sections", 0) == 0:
            print(f"  {stats['edition_name']}: ERROR - {stats.get('error', 'unknown')}")
        else:
            sections = stats.get('total_sections', 0)
            chunks = stats.get('total_chunks', 0)
            total_sections += sections
            total_chunks += chunks
            print(f"  {stats['edition_name']}: {stats.get('treatises_processed', 0)} treatises, "
                  f"{sections} sections, {chunks} chunks")

    print(f"\nTOTAL: {total_sections} sections, {total_chunks} chunks")
    print(f"Results saved to: {output_dir}")


if __name__ == "__main__":
    main()
