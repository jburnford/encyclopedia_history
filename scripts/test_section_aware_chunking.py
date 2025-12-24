#!/usr/bin/env python3
"""
Test section-aware chunking on HERALDRY, SCOTLAND, and ANATOMY articles.

Tests:
1. Intro section creation for pre-marker content (HERALDRY)
2. LLM section detection (SCOTLAND)
3. Explicit section markers (ANATOMY)
4. Oversized section splitting
5. Section-aware chunk boundaries

Usage:
    OPENAI_API_KEY="sk-..." python3 test_section_aware_chunking.py
"""

import json
import logging
import os
import sys

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from encyclopedia_parser import Article, ArticleType
from encyclopedia_parser.chunkers import chunk_article_with_sections, print_article_structure
from encyclopedia_parser.sections import extract_explicit_sections, get_section_stats


def load_test_articles(jsonl_path: str, headwords: list[str]) -> dict[str, Article]:
    """Load specific articles from JSONL file."""
    articles = {}
    with open(jsonl_path) as f:
        for line in f:
            data = json.loads(line)
            if data['headword'] in headwords:
                article = Article(
                    headword=data['headword'],
                    text=data['text'],
                    edition_year=1815,
                    edition_name="Britannica 5th",
                    article_type=ArticleType.TREATISE,
                )
                articles[data['headword']] = article
                logger.info(f"Loaded {data['headword']}: {len(data['text']):,} chars")
    return articles


def test_intro_section(article: Article) -> bool:
    """Test that intro sections are created for pre-marker content."""
    print(f"\n{'='*60}")
    print(f"TEST: Intro Section Creation - {article.headword}")
    print(f"{'='*60}")

    sections = extract_explicit_sections(article)

    if not sections:
        print(f"  No explicit sections found for {article.headword}")
        return False

    print(f"  Found {len(sections)} sections:")
    for s in sections:
        print(f"    [{s.index}] {s.title}: {len(s.text):,} chars (pos {s.char_start:,}-{s.char_end:,})")

    # Check if first section starts at 0 (intro section was created)
    if sections[0].char_start == 0:
        print(f"\n  ✓ Intro section created: '{sections[0].title}'")
        return True
    else:
        gap = sections[0].char_start
        print(f"\n  ✗ Gap before first section: {gap:,} chars not captured")
        return False


def test_section_aware_chunking(article: Article, use_llm: bool = False) -> bool:
    """Test section-aware chunking."""
    print(f"\n{'='*60}")
    print(f"TEST: Section-Aware Chunking - {article.headword}")
    print(f"{'='*60}")

    sections, chunks = chunk_article_with_sections(
        article,
        use_llm_sections=use_llm,
        max_section_size=30000
    )

    if not sections:
        print(f"  No sections extracted for {article.headword}")
        return False

    # Print structure
    print("\n" + print_article_structure(article, sections, chunks))

    # Verify all chunks have section metadata
    chunks_without_section = [c for c in chunks if c.section_index is None]
    if chunks_without_section:
        print(f"\n  ✗ {len(chunks_without_section)} chunks without section metadata")
        return False

    print(f"\n  ✓ All {len(chunks)} chunks have section metadata")

    # Verify chunks cover the article
    total_chunk_chars = sum(len(c.text) for c in chunks)
    coverage = total_chunk_chars / len(article.text) * 100
    print(f"  Coverage: {coverage:.1f}% ({total_chunk_chars:,} / {len(article.text):,} chars)")

    # Stats
    stats = get_section_stats(sections)
    print(f"\n  Section stats: {stats}")

    return True


def test_oversized_section_splitting(article: Article) -> bool:
    """Test that oversized sections are split properly."""
    print(f"\n{'='*60}")
    print(f"TEST: Oversized Section Splitting - {article.headword}")
    print(f"{'='*60}")

    sections = extract_explicit_sections(article)

    oversized = [s for s in sections if len(s.text) > 30000]
    print(f"  Found {len(oversized)} oversized sections (>30K chars)")
    for s in oversized:
        print(f"    '{s.title}': {len(s.text):,} chars")

    if not oversized:
        print("  No oversized sections to test")
        return True

    # Run section-aware chunking with splitting
    sections, chunks = chunk_article_with_sections(
        article,
        use_llm_sections=False,
        max_section_size=30000
    )

    # Check that oversized sections got multiple chunks
    for s in sections:
        if len(s.text) > 30000:
            section_chunks = [c for c in chunks if c.section_index == s.index]
            print(f"  Section '{s.title}' ({len(s.text):,} chars) -> {len(section_chunks)} chunks")
            if len(section_chunks) < 2:
                print(f"    ✗ Expected multiple chunks for oversized section")
                return False

    print(f"\n  ✓ All oversized sections properly split")
    return True


def main():
    # Check for OpenAI API key (needed for some tests)
    has_api_key = bool(os.environ.get("OPENAI_API_KEY"))
    if not has_api_key:
        print("WARNING: OPENAI_API_KEY not set. LLM-based tests will be skipped.")
        print("Set OPENAI_API_KEY to test LLM section detection and semantic chunking.\n")

    # Load test articles
    jsonl_path = "articles_1815_clean.jsonl"
    if not os.path.exists(jsonl_path):
        print(f"ERROR: Test data file not found: {jsonl_path}")
        sys.exit(1)

    test_headwords = ["HERALDRY", "SCOTLAND", "ANATOMY"]
    articles = load_test_articles(jsonl_path, test_headwords)

    if len(articles) < len(test_headwords):
        missing = set(test_headwords) - set(articles.keys())
        print(f"WARNING: Missing articles: {missing}")

    results = {}

    # Test 1: Intro section creation (HERALDRY has 34K chars before first marker)
    if "HERALDRY" in articles:
        results["intro_section"] = test_intro_section(articles["HERALDRY"])

    # Test 2: Oversized section splitting (HERALDRY has 99K section)
    if "HERALDRY" in articles:
        results["oversized_splitting"] = test_oversized_section_splitting(articles["HERALDRY"])

    # Test 3: Explicit section chunking (ANATOMY)
    if "ANATOMY" in articles and has_api_key:
        results["explicit_chunking"] = test_section_aware_chunking(
            articles["ANATOMY"], use_llm=False
        )

    # Test 4: LLM section detection + chunking (SCOTLAND)
    if "SCOTLAND" in articles and has_api_key:
        results["llm_chunking"] = test_section_aware_chunking(
            articles["SCOTLAND"], use_llm=True
        )

    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    for test_name, passed in results.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"  {test_name}: {status}")

    all_passed = all(results.values())
    print(f"\nOverall: {'All tests passed!' if all_passed else 'Some tests failed.'}")

    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
