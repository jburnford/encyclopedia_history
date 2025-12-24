"""
Semantic chunking for encyclopedia articles.

Provides chunking functionality for long treatise articles to enable
vector embedding and RAG-based retrieval.

Uses LangChain's SemanticChunker for intelligent text splitting based
on embedding similarity, ensuring semantically coherent chunks.
"""

import logging
import os
import re
from dataclasses import dataclass, field
from typing import Optional

from .models import Article, TextChunk, ArticleType, Section

logger = logging.getLogger(__name__)


@dataclass
class ChunkingStats:
    """Statistics and cost tracking for chunking operations."""

    articles_processed: int = 0
    articles_chunked: int = 0  # Articles that were actually split
    total_chunks: int = 0
    total_characters: int = 0
    embedding_calls: int = 0
    estimated_tokens: int = 0

    # OpenAI embedding pricing (per 1M tokens) as of Dec 2025
    # text-embedding-3-small: $0.02 per 1M tokens
    EMBEDDING_COST_PER_MILLION: float = 0.02

    @property
    def estimated_cost_usd(self) -> float:
        """Estimated cost in USD based on token usage."""
        return (self.estimated_tokens / 1_000_000) * self.EMBEDDING_COST_PER_MILLION

    def __str__(self) -> str:
        return (
            f"Chunking Stats:\n"
            f"  Articles processed: {self.articles_processed}\n"
            f"  Articles chunked: {self.articles_chunked}\n"
            f"  Total chunks: {self.total_chunks}\n"
            f"  Total characters: {self.total_characters:,}\n"
            f"  Estimated tokens: {self.estimated_tokens:,}\n"
            f"  Embedding API calls: {self.embedding_calls}\n"
            f"  Estimated cost: ${self.estimated_cost_usd:.4f}"
        )


class TreatiseChunker:
    """
    Chunks long treatise articles using semantic similarity.

    Uses LangChain's SemanticChunker which splits text at points where
    the embedding similarity between adjacent sentences drops below a
    threshold, creating semantically coherent chunks.
    """

    # Minimum article length to trigger semantic chunking
    MIN_CHUNK_LENGTH = 5000

    # Target chunk size (soft limit)
    TARGET_CHUNK_SIZE = 1500

    def __init__(
        self,
        openai_api_key: Optional[str] = None,
        breakpoint_threshold_type: str = "percentile",
        breakpoint_threshold_amount: float = 95,
        model: str = "text-embedding-3-small",
    ):
        """
        Initialize the chunker.

        Args:
            openai_api_key: OpenAI API key. If None, uses OPENAI_API_KEY env var.
            breakpoint_threshold_type: How to calculate breakpoints.
                Options: "percentile", "standard_deviation", "interquartile"
            breakpoint_threshold_amount: Threshold value for breakpoints.
                For percentile, 95 means split at top 5% dissimilarity.
            model: OpenAI embedding model to use.
        """
        self.api_key = openai_api_key or os.environ.get("OPENAI_API_KEY")
        self.breakpoint_threshold_type = breakpoint_threshold_type
        self.breakpoint_threshold_amount = breakpoint_threshold_amount
        self.model = model

        self.stats = ChunkingStats()
        self._chunker = None  # Lazy initialization

    def _get_chunker(self):
        """Lazily initialize the SemanticChunker."""
        if self._chunker is None:
            try:
                from langchain_experimental.text_splitter import SemanticChunker
                from langchain_openai import OpenAIEmbeddings
            except ImportError as e:
                raise ImportError(
                    "SemanticChunker requires langchain-experimental and langchain-openai. "
                    "Install with: pip install langchain langchain-experimental langchain-openai"
                ) from e

            if not self.api_key:
                raise ValueError(
                    "OpenAI API key required. Set OPENAI_API_KEY env var or pass to constructor."
                )

            embeddings = OpenAIEmbeddings(
                model=self.model,
                openai_api_key=self.api_key,
            )

            self._chunker = SemanticChunker(
                embeddings=embeddings,
                breakpoint_threshold_type=self.breakpoint_threshold_type,
                breakpoint_threshold_amount=self.breakpoint_threshold_amount,
            )

        return self._chunker

    def chunk_article(self, article: Article) -> list[TextChunk]:
        """
        Chunk an article into TextChunks.

        Short articles (< MIN_CHUNK_LENGTH) return a single chunk.
        Treatises are split using semantic similarity.

        Args:
            article: The article to chunk

        Returns:
            List of TextChunk objects
        """
        self.stats.articles_processed += 1
        text = article.text

        # Short articles get single chunk
        if len(text) < self.MIN_CHUNK_LENGTH:
            return [self._make_single_chunk(article)]

        # Only semantic chunk treatises
        if article.article_type not in (ArticleType.TREATISE, "treatise"):
            return [self._make_single_chunk(article)]

        # Semantic chunking for treatises
        try:
            chunks = self._semantic_chunk(article)
            self.stats.articles_chunked += 1
            return chunks
        except Exception as e:
            logger.warning(
                f"Semantic chunking failed for {article.headword}: {e}. "
                f"Falling back to single chunk."
            )
            return [self._make_single_chunk(article)]

    def _make_single_chunk(self, article: Article) -> TextChunk:
        """Create a single chunk containing the entire article."""
        self.stats.total_chunks += 1
        self.stats.total_characters += len(article.text)

        return TextChunk(
            text=article.text,
            index=0,
            parent_headword=article.headword,
            edition_year=article.edition_year,
            char_start=0,
            char_end=len(article.text),
            section_title=None,
        )

    def _semantic_chunk(self, article: Article) -> list[TextChunk]:
        """
        Perform semantic chunking on a treatise article.

        Uses OpenAI embeddings to find natural breakpoints in the text.
        """
        text = article.text
        chunker = self._get_chunker()

        # Estimate tokens for cost tracking (rough estimate: 4 chars per token)
        estimated_tokens = len(text) // 4
        self.stats.estimated_tokens += estimated_tokens
        self.stats.embedding_calls += 1

        logger.info(
            f"Semantic chunking {article.headword} "
            f"({len(text):,} chars, ~{estimated_tokens:,} tokens)"
        )

        # Get chunks from LangChain
        chunk_texts = chunker.split_text(text)

        # Build TextChunk objects with position tracking
        chunks = []
        current_pos = 0

        for i, chunk_text in enumerate(chunk_texts):
            # Find the chunk's position in original text
            # Account for potential whitespace variations
            chunk_start = text.find(chunk_text.strip()[:100], current_pos)
            if chunk_start == -1:
                # Fallback: use approximate position
                chunk_start = current_pos

            chunk_end = chunk_start + len(chunk_text)
            current_pos = chunk_end

            # Try to detect section title from chunk start
            section_title = self._detect_section_title(chunk_text)

            chunk = TextChunk(
                text=chunk_text,
                index=i,
                parent_headword=article.headword,
                edition_year=article.edition_year,
                char_start=chunk_start,
                char_end=chunk_end,
                section_title=section_title,
            )
            chunks.append(chunk)

            self.stats.total_chunks += 1
            self.stats.total_characters += len(chunk_text)

        logger.info(f"  Created {len(chunks)} chunks for {article.headword}")

        return chunks

    def _detect_section_title(self, text: str) -> Optional[str]:
        """
        Attempt to detect a section title at the start of chunk text.

        Looks for patterns like:
            PART I.
            SECT. II.
            CHAPTER III.
            Section 1. Of the...
        """
        # Check first 200 chars for section headers
        sample = text[:200]

        patterns = [
            r'^(PART\s+[IVXLCDM]+\.?)',
            r'^(SECT(?:ION)?\.?\s+[IVXLCDM\d]+\.?)',
            r'^(CHAPTER\s+[IVXLCDM\d]+\.?)',
            r'^(Section\s+\d+\.?\s*[^\n]{0,50})',
            r'^([A-Z][a-z]+\s+[IVXLCDM]+\.)',  # Division I.
        ]

        for pattern in patterns:
            match = re.match(pattern, sample.strip(), re.IGNORECASE)
            if match:
                return match.group(1).strip()

        return None

    def chunk_articles(self, articles: list[Article]) -> list[TextChunk]:
        """
        Chunk multiple articles.

        Args:
            articles: List of articles to chunk

        Returns:
            Flat list of all TextChunks from all articles
        """
        all_chunks = []
        for article in articles:
            chunks = self.chunk_article(article)
            all_chunks.extend(chunks)
        return all_chunks

    def get_stats(self) -> ChunkingStats:
        """Get current chunking statistics."""
        return self.stats

    def reset_stats(self):
        """Reset statistics counters."""
        self.stats = ChunkingStats()


# =============================================================================
# Section-Aware Chunking
# =============================================================================

def chunk_section(
    section: Section,
    chunker: "TreatiseChunker",
    max_section_size: int = 30000,
) -> list[TextChunk]:
    """
    Chunk a single section, recursively splitting if oversized.

    Args:
        section: The section to chunk
        chunker: TreatiseChunker instance to use
        max_section_size: Maximum section size before recursive splitting

    Returns:
        List of TextChunks from this section
    """
    text = section.text

    # Small sections get a single chunk
    if len(text) < chunker.MIN_CHUNK_LENGTH:
        chunk = TextChunk(
            text=text,
            index=0,
            parent_headword=section.parent_headword,
            edition_year=section.edition_year,
            char_start=section.char_start,
            char_end=section.char_end,
            section_title=section.title,
            section_index=section.index,
        )
        chunker.stats.total_chunks += 1
        chunker.stats.total_characters += len(text)
        return [chunk]

    # Oversized sections: split into sub-sections by paragraph boundaries, then chunk each
    if len(text) > max_section_size:
        logger.info(
            f"Section '{section.title}' is oversized ({len(text):,} chars), "
            f"splitting at paragraph boundaries"
        )
        return _split_oversized_section(section, chunker, max_section_size)

    # Normal-sized sections: semantic chunking
    return _semantic_chunk_section(section, chunker)


def _split_oversized_section(
    section: Section,
    chunker: "TreatiseChunker",
    max_section_size: int,
) -> list[TextChunk]:
    """
    Split an oversized section at paragraph boundaries, then chunk each part.
    """
    text = section.text
    target_size = max_section_size // 2  # Aim for half the max

    # Find paragraph boundaries
    paragraphs = re.split(r'\n\s*\n', text)

    # Group paragraphs into roughly equal parts
    parts = []
    current_part = []
    current_size = 0

    for para in paragraphs:
        para_len = len(para) + 2  # Account for paragraph separator
        if current_size + para_len > target_size and current_part:
            parts.append('\n\n'.join(current_part))
            current_part = [para]
            current_size = para_len
        else:
            current_part.append(para)
            current_size += para_len

    if current_part:
        parts.append('\n\n'.join(current_part))

    # Create TextChunks from each part, recursively chunking if needed
    all_chunks = []
    current_offset = section.char_start

    for i, part_text in enumerate(parts):
        part_end = current_offset + len(part_text)

        # Create a pseudo-section for recursive processing
        sub_section = Section(
            title=f"{section.title} (Part {i+1})",
            level=section.level + 1,
            index=section.index,
            parent_headword=section.parent_headword,
            edition_year=section.edition_year,
            char_start=current_offset,
            char_end=part_end,
            text=part_text,
            extraction_method=section.extraction_method,
        )

        # Recursively chunk this part
        part_chunks = chunk_section(sub_section, chunker, max_section_size)

        # Update chunks to reference original section
        for chunk in part_chunks:
            chunk.section_title = section.title
            chunk.section_index = section.index

        all_chunks.extend(part_chunks)
        current_offset = part_end + 2  # Account for paragraph separator

    return all_chunks


def _semantic_chunk_section(
    section: Section,
    chunker: "TreatiseChunker",
) -> list[TextChunk]:
    """
    Apply semantic chunking to a section.
    """
    text = section.text
    langchain_chunker = chunker._get_chunker()

    # Track tokens for cost estimation
    estimated_tokens = len(text) // 4
    chunker.stats.estimated_tokens += estimated_tokens
    chunker.stats.embedding_calls += 1

    logger.debug(
        f"Semantic chunking section '{section.title}' "
        f"({len(text):,} chars, ~{estimated_tokens:,} tokens)"
    )

    # Get chunks from LangChain
    try:
        chunk_texts = langchain_chunker.split_text(text)
    except Exception as e:
        logger.warning(f"Semantic chunking failed for section '{section.title}': {e}")
        # Fallback: return single chunk
        chunk = TextChunk(
            text=text,
            index=0,
            parent_headword=section.parent_headword,
            edition_year=section.edition_year,
            char_start=section.char_start,
            char_end=section.char_end,
            section_title=section.title,
            section_index=section.index,
        )
        chunker.stats.total_chunks += 1
        chunker.stats.total_characters += len(text)
        return [chunk]

    # Build TextChunk objects with position tracking
    chunks = []
    current_pos = 0

    for i, chunk_text in enumerate(chunk_texts):
        # Find chunk position in section text
        chunk_start_in_section = text.find(chunk_text.strip()[:100], current_pos)
        if chunk_start_in_section == -1:
            chunk_start_in_section = current_pos

        chunk_end_in_section = chunk_start_in_section + len(chunk_text)
        current_pos = chunk_end_in_section

        # Convert to article-level positions
        chunk_start = section.char_start + chunk_start_in_section
        chunk_end = section.char_start + chunk_end_in_section

        chunk = TextChunk(
            text=chunk_text,
            index=i,
            parent_headword=section.parent_headword,
            edition_year=section.edition_year,
            char_start=chunk_start,
            char_end=chunk_end,
            section_title=section.title,
            section_index=section.index,
        )
        chunks.append(chunk)

        chunker.stats.total_chunks += 1
        chunker.stats.total_characters += len(chunk_text)

    return chunks


def chunk_article_with_sections(
    article: Article,
    openai_api_key: Optional[str] = None,
    use_llm_sections: bool = True,
    max_section_size: int = 30000,
) -> tuple[list[Section], list[TextChunk]]:
    """
    Extract sections and chunk an article with section-aware boundaries.

    This is the main entry point for hierarchical chunking. It:
    1. Extracts sections (explicit markers or LLM-based)
    2. Chunks each section individually
    3. Recursively splits oversized sections
    4. Preserves section metadata in each chunk

    Args:
        article: The article to process
        openai_api_key: OpenAI API key (or set OPENAI_API_KEY env var)
        use_llm_sections: Whether to use LLM for section detection
        max_section_size: Maximum section size before recursive splitting

    Returns:
        Tuple of (list of Sections, list of TextChunks)

    Example:
        >>> from encyclopedia_parser import Article
        >>> from encyclopedia_parser.chunkers import chunk_article_with_sections
        >>>
        >>> article = Article(headword="SCOTLAND", text="...", edition_year=1815)
        >>> sections, chunks = chunk_article_with_sections(article)
        >>>
        >>> for section in sections:
        ...     section_chunks = [c for c in chunks if c.section_index == section.index]
        ...     print(f"{section.title}: {len(section_chunks)} chunks")
    """
    # Import here to avoid circular imports
    from .sections import extract_sections

    # Create chunker instance
    chunker = TreatiseChunker(openai_api_key=openai_api_key)

    # Extract sections
    sections = extract_sections(article, use_llm=use_llm_sections, openai_api_key=openai_api_key)

    if not sections:
        # No sections - fall back to whole-article chunking
        logger.info(f"No sections found for {article.headword}, using whole-article chunking")
        chunks = chunker.chunk_article(article)
        return [], chunks

    logger.info(
        f"Processing {article.headword}: {len(sections)} sections, "
        f"{len(article.text):,} total chars"
    )

    # Chunk each section
    all_chunks = []
    for section in sections:
        section_chunks = chunk_section(section, chunker, max_section_size)

        # Re-index chunks to be sequential across article
        for chunk in section_chunks:
            chunk.index = len(all_chunks)
            all_chunks.append(chunk)

        logger.debug(
            f"  Section '{section.title}': {len(section.text):,} chars -> "
            f"{len(section_chunks)} chunks"
        )

    chunker.stats.articles_processed += 1
    chunker.stats.articles_chunked += 1

    logger.info(
        f"Created {len(all_chunks)} chunks from {len(sections)} sections "
        f"for {article.headword}"
    )

    return sections, all_chunks


def print_article_structure(
    article: Article,
    sections: list[Section],
    chunks: list[TextChunk],
) -> str:
    """
    Print a tree-style visualization of article structure.

    Example output:
        Article: SCOTLAND (160,234 chars)
        ├── Section 0: Geography (1,234 chars)
        │   └── Chunk 0 (1,234 chars)
        ├── Section 1: Early History (31,456 chars)
        │   ├── Chunk 1 (2,345 chars)
        │   ├── Chunk 2 (3,456 chars)
        │   └── ... 12 more chunks
    """
    lines = [f"Article: {article.headword} ({len(article.text):,} chars)"]

    for i, section in enumerate(sections):
        is_last_section = (i == len(sections) - 1)
        prefix = "└──" if is_last_section else "├──"
        lines.append(f"{prefix} Section {section.index}: {section.title} ({len(section.text):,} chars)")

        # Find chunks for this section
        section_chunks = [c for c in chunks if c.section_index == section.index]
        child_prefix = "    " if is_last_section else "│   "

        if len(section_chunks) <= 5:
            for j, chunk in enumerate(section_chunks):
                is_last_chunk = (j == len(section_chunks) - 1)
                chunk_marker = "└──" if is_last_chunk else "├──"
                lines.append(f"{child_prefix}{chunk_marker} Chunk {chunk.index} ({len(chunk.text):,} chars)")
        else:
            # Show first 3, then summary
            for j, chunk in enumerate(section_chunks[:3]):
                lines.append(f"{child_prefix}├── Chunk {chunk.index} ({len(chunk.text):,} chars)")
            lines.append(f"{child_prefix}└── ... {len(section_chunks) - 3} more chunks")

    return "\n".join(lines)


# =============================================================================
# Convenience functions
# =============================================================================

def chunk_treatise(
    article: Article,
    openai_api_key: Optional[str] = None,
) -> list[TextChunk]:
    """
    Chunk a single article into TextChunks.

    This is the main entry point for chunking. Short articles return
    a single chunk; treatises are semantically chunked.

    Args:
        article: The article to chunk
        openai_api_key: OpenAI API key (or set OPENAI_API_KEY env var)

    Returns:
        List of TextChunk objects

    Example:
        >>> from encyclopedia_parser import parse_markdown_file, classify_articles
        >>> from encyclopedia_parser.chunkers import chunk_treatise
        >>>
        >>> articles = parse_markdown_file("volume.md", 1778)
        >>> classified = classify_articles(articles)
        >>>
        >>> for article in classified:
        ...     if article.article_type == "treatise":
        ...         chunks = chunk_treatise(article)
        ...         print(f"{article.headword}: {len(chunks)} chunks")
    """
    chunker = TreatiseChunker(openai_api_key=openai_api_key)
    return chunker.chunk_article(article)


def chunk_articles(
    articles: list[Article],
    openai_api_key: Optional[str] = None,
    treatises_only: bool = True,
) -> tuple[list[TextChunk], ChunkingStats]:
    """
    Chunk multiple articles with cost tracking.

    Args:
        articles: List of articles to chunk
        openai_api_key: OpenAI API key (or set OPENAI_API_KEY env var)
        treatises_only: If True, only chunk treatise articles (recommended
            for cost efficiency). Other articles get single chunks.

    Returns:
        Tuple of (list of TextChunks, ChunkingStats)

    Example:
        >>> chunks, stats = chunk_articles(articles)
        >>> print(stats)
        Chunking Stats:
          Articles processed: 81
          Articles chunked: 45
          Total chunks: 312
          Estimated cost: $0.0142
    """
    chunker = TreatiseChunker(openai_api_key=openai_api_key)

    if treatises_only:
        # Filter to treatises for semantic chunking, but still create
        # single chunks for others if needed for completeness
        articles_to_process = articles
    else:
        articles_to_process = articles

    chunks = chunker.chunk_articles(articles_to_process)

    return chunks, chunker.get_stats()


def estimate_chunking_cost(
    articles: list[Article],
    chars_per_token: int = 4,
    cost_per_million_tokens: float = 0.02,
) -> dict:
    """
    Estimate the cost of chunking articles without making API calls.

    Args:
        articles: List of articles (should be classified)
        chars_per_token: Estimated characters per token (default 4)
        cost_per_million_tokens: Cost in USD per 1M tokens

    Returns:
        Dictionary with cost estimates
    """
    treatises = [a for a in articles
                 if a.article_type in (ArticleType.TREATISE, "treatise")
                 and len(a.text) >= TreatiseChunker.MIN_CHUNK_LENGTH]

    total_chars = sum(len(a.text) for a in treatises)
    estimated_tokens = total_chars // chars_per_token
    estimated_cost = (estimated_tokens / 1_000_000) * cost_per_million_tokens

    return {
        "treatise_count": len(treatises),
        "total_characters": total_chars,
        "estimated_tokens": estimated_tokens,
        "estimated_cost_usd": estimated_cost,
        "cost_per_million_tokens": cost_per_million_tokens,
    }
