#!/usr/bin/env python3
"""
Chunking utilities for encyclopedia articles.
Reused from voyage-context testing with improvements.
"""

import json
from dataclasses import dataclass
from typing import List, Dict, Iterator
from pathlib import Path


@dataclass
class Article:
    """Represents an encyclopedia article."""
    headword: str
    text: str
    article_id: str = ""

    def __post_init__(self):
        if not self.article_id:
            # Create unique ID from headword + text hash
            self.article_id = f"{self.headword}_{hash(self.text[:100]) % 10000}"


@dataclass
class Chunk:
    """Represents a chunk of an article."""
    headword: str
    article_id: str
    chunk_id: int
    text: str

    @property
    def id(self) -> str:
        return f"{self.article_id}_chunk_{self.chunk_id}"


def load_articles(jsonl_path: str, headwords: List[str] = None) -> List[Article]:
    """
    Load articles from JSONL file.

    Args:
        jsonl_path: Path to articles JSONL file
        headwords: Optional list of headwords to filter (case-insensitive)

    Returns:
        List of Article objects
    """
    articles = []
    headwords_upper = {h.upper() for h in headwords} if headwords else None

    with open(jsonl_path) as f:
        for line in f:
            data = json.loads(line)
            hw = data.get('headword', '')

            if headwords_upper is None or hw.upper() in headwords_upper:
                articles.append(Article(
                    headword=hw,
                    text=data.get('text', '')
                ))

    return articles


def chunk_article(article: Article, chunk_size: int = 800, overlap: int = 100) -> List[Chunk]:
    """
    Split article into chunks using paragraph-aware splitting.

    Args:
        article: Article to chunk
        chunk_size: Target chunk size in characters
        overlap: Overlap between chunks (not implemented yet)

    Returns:
        List of Chunk objects
    """
    text = article.text
    paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]

    chunks = []
    current_text = ''
    chunk_id = 0

    for para in paragraphs:
        # If adding this paragraph exceeds chunk size, save current and start new
        if len(current_text) + len(para) + 2 > chunk_size and current_text:
            chunks.append(Chunk(
                headword=article.headword,
                article_id=article.article_id,
                chunk_id=chunk_id,
                text=current_text.strip()
            ))
            chunk_id += 1
            current_text = para
        else:
            current_text = (current_text + '\n\n' + para) if current_text else para

    # Don't forget the last chunk
    if current_text:
        chunks.append(Chunk(
            headword=article.headword,
            article_id=article.article_id,
            chunk_id=chunk_id,
            text=current_text.strip()
        ))

    return chunks


def chunk_articles(articles: List[Article], chunk_size: int = 800) -> List[Chunk]:
    """Chunk multiple articles."""
    all_chunks = []
    for article in articles:
        all_chunks.extend(chunk_article(article, chunk_size))
    return all_chunks


def load_and_chunk(
    jsonl_path: str,
    headwords: List[str] = None,
    chunk_size: int = 800,
    max_articles: int = None
) -> tuple[List[Article], List[Chunk]]:
    """
    Convenience function to load and chunk articles in one call.

    Returns:
        Tuple of (articles, chunks)
    """
    articles = load_articles(jsonl_path, headwords)

    if max_articles:
        articles = articles[:max_articles]

    chunks = chunk_articles(articles, chunk_size)

    return articles, chunks


# Test queries for evaluation
LEATHER_QUERIES = [
    ("vegetable tanning oak bark process pit", ["TANNING", "BARK", "OAK"]),
    ("hide preparation lime soaking unhairing", ["HIDE", "TANNING", "LEATHER"]),
    ("currying finishing leather oil grease", ["LEATHER", "TANNING"]),
    ("morocco red leather Turkey dyeing goatskin", ["LEATHER", "MOROCCO"]),
    ("tanning liquor ooze tan yard", ["TANNING", "TAN", "BARK"]),
    ("raw hides salted green dried", ["HIDE", "LEATHER", "TANNING"]),
    ("sole leather butts backs thick", ["TANNING", "LEATHER"]),
    ("chamois soft white leather oil tawed", ["LEATHER"]),
    ("parchment vellum animal skin writing", ["PARCHMENT", "LEATHER"]),
    ("fuller cloth wool process", ["FULLER"]),
]

SEMANTIC_DRIFT_QUERIES = [
    # PHLOGISTON - combustion theory (replaced by oxygen)
    ("combustion burning fire theory inflammable", ["PHLOGISTON"]),
    # BROADCAST - agricultural term (now media)
    ("scattering seeds sowing field hand", ["BROADCAST"]),
    # ATOM - philosophical indivisible (now nuclear)
    ("indivisible matter particle philosophy smallest", ["ATOM"]),
    # ENTHUSIASM - religious fanaticism (now positive excitement)
    ("religious zeal fanaticism sect inspired", ["ENTHUSIASM"]),
    # PHYSICS - natural philosophy/medicine (now different)
    ("physic medicine healing art body", ["PHYSICS"]),
]

LONG_ARTICLE_QUERIES = [
    ("vitrification of materials at high temperatures", ["GLASS"]),
    ("moveable type letters arranged for impressions on paper", ["PRINTING"]),
    ("the French improvements to this art and their methods", ["GLASS", "PRINTING"]),
    ("experiments and observations as the foundation of knowledge", ["EXPERIMENTAL PHILOSOPHY"]),
]

# Target articles for testing
LEATHER_ARTICLES = [
    'TANNING', 'LEATHER', 'HIDE', 'BARK', 'FULLER', 'TAN',
    'MOROCCO', 'OAK', 'ALUM', 'PARCHMENT', 'CHAMOIS', 'CURRIER',
    'TANNER', 'VELLUM', 'CORDWAINER'
]

SEMANTIC_DRIFT_ARTICLES = [
    'PHLOGISTON', 'BROADCAST', 'ATOM', 'ENTHUSIASM', 'PHYSICS',
    'ENGINE', 'ELECTRICITY'
]

LONG_ARTICLES = [
    'GLASS', 'PRINTING', 'EXPERIMENTAL PHILOSOPHY'
]


if __name__ == '__main__':
    # Quick test
    import sys
    if len(sys.argv) > 1:
        path = sys.argv[1]
        articles, chunks = load_and_chunk(path, LEATHER_ARTICLES[:5], max_articles=5)
        print(f"Loaded {len(articles)} articles, {len(chunks)} chunks")
        for a in articles:
            print(f"  {a.headword}: {len(a.text)} chars")
