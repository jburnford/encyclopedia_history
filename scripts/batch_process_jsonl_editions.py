#!/usr/bin/env python3
"""
Process JSONL editions (1771, 1797, 1815) with classification and section-aware chunking.
"""

import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

sys.path.insert(0, str(Path(__file__).parent))

from encyclopedia_parser import (
    Article, ArticleType, classify_articles,
)
from encyclopedia_parser.models import get_edition_config
from encyclopedia_parser.chunkers import chunk_article_with_sections


EDITIONS = {
    1771: {
        "name": "Britannica 1st",
        "jsonl": "articles_1771.jsonl",
    },
    1797: {
        "name": "Britannica 3rd",
        "jsonl": "articles_1797.jsonl",
    },
    1815: {
        "name": "Britannica 5th",
        "jsonl": "articles_1815_clean.jsonl",
    },
}


def load_and_classify_articles(jsonl_path: str, edition_year: int, edition_name: str):
    """Load articles from JSONL and classify them."""
    articles = []
    with open(jsonl_path) as f:
        for line in f:
            data = json.loads(line)
            article = Article(
                headword=data['headword'],
                text=data['text'],
                edition_year=edition_year,
                edition_name=edition_name,
                article_type='dictionary',
            )
            articles.append(article)

    logger.info(f"Loaded {len(articles)} articles, classifying...")
    config = get_edition_config(edition_year)
    articles = classify_articles(articles, config)

    return articles


def process_edition(edition_year: int, config: dict, output_dir: Path, api_key: str):
    """Process a single edition."""
    import time
    start_time = time.time()

    logger.info(f"\n{'='*60}")
    logger.info(f"Processing {config['name']} ({edition_year})")
    logger.info(f"{'='*60}")

    jsonl_path = Path(__file__).parent / config["jsonl"]
    if not jsonl_path.exists():
        logger.error(f"File not found: {jsonl_path}")
        return

    articles = load_and_classify_articles(str(jsonl_path), edition_year, config["name"])

    # Filter to treatises
    treatises = [a for a in articles if a.article_type in (ArticleType.TREATISE, "treatise")]
    logger.info(f"Found {len(treatises)} treatise articles")

    # Filter by length
    treatises = [t for t in treatises if len(t.text) >= 5000]
    logger.info(f"Processing {len(treatises)} treatises (>5K chars)")

    all_sections = []
    all_chunks = []
    errors = []

    for i, article in enumerate(treatises):
        logger.info(f"  [{i+1}/{len(treatises)}] {article.headword} ({len(article.text):,} chars)")

        try:
            sections, chunks = chunk_article_with_sections(
                article,
                openai_api_key=api_key,
                use_llm_sections=True,
                max_section_size=30000,
            )

            for section in sections:
                all_sections.append({
                    "title": section.title,
                    "level": section.level,
                    "index": section.index,
                    "parent_headword": section.parent_headword,
                    "edition_year": section.edition_year,
                    "char_start": section.char_start,
                    "char_end": section.char_end,
                    "text_length": len(section.text),
                    "extraction_method": section.extraction_method,
                })

            for chunk in chunks:
                all_chunks.append({
                    "text": chunk.text,
                    "index": chunk.index,
                    "parent_headword": chunk.parent_headword,
                    "edition_year": chunk.edition_year,
                    "char_start": chunk.char_start,
                    "char_end": chunk.char_end,
                    "section_title": chunk.section_title,
                    "section_index": chunk.section_index,
                })

        except Exception as e:
            logger.error(f"    Error: {e}")
            errors.append({"headword": article.headword, "error": str(e)})

    # Save results
    sections_file = output_dir / f"sections_{edition_year}.jsonl"
    chunks_file = output_dir / f"chunks_{edition_year}.jsonl"

    with open(sections_file, 'w') as f:
        for s in all_sections:
            f.write(json.dumps(s) + '\n')

    with open(chunks_file, 'w') as f:
        for c in all_chunks:
            f.write(json.dumps(c) + '\n')

    elapsed = time.time() - start_time
    logger.info(f"Saved {len(all_sections)} sections, {len(all_chunks)} chunks")
    logger.info(f"Time: {elapsed:.1f}s, Errors: {len(errors)}")

    return {
        "edition_year": edition_year,
        "sections": len(all_sections),
        "chunks": len(all_chunks),
        "errors": len(errors),
        "time": elapsed,
    }


def main():
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        logger.error("OPENAI_API_KEY not set")
        sys.exit(1)

    output_dir = Path(__file__).parent / "output"
    output_dir.mkdir(exist_ok=True)

    results = []
    for edition_year, config in EDITIONS.items():
        try:
            result = process_edition(edition_year, config, output_dir, api_key)
            if result:
                results.append(result)
        except Exception as e:
            logger.error(f"Failed {config['name']}: {e}")

    print("\n" + "="*60)
    print("JSONL EDITIONS COMPLETE")
    print("="*60)
    for r in results:
        print(f"  {r['edition_year']}: {r['sections']} sections, {r['chunks']} chunks")


if __name__ == "__main__":
    main()
