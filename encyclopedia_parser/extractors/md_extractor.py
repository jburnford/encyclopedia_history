"""
Markdown file extractor for OLMoCR output.

Handles the .md files produced by OLMoCR which use bold formatting (**HEADWORD**)
for article entries rather than markdown headers.
"""

from pathlib import Path
from typing import Iterator, Optional
import logging

from .base import BaseExtractor
from ..models import Article, EditionConfig
from ..patterns import (
    find_all_headwords,
    find_first_article,
    is_front_matter,
    is_structural_header,
    FRONT_MATTER_INDICATORS,
)

logger = logging.getLogger(__name__)


class MarkdownExtractor(BaseExtractor):
    """
    Extract encyclopedia articles from OLMoCR markdown files.

    OLMoCR produces markdown with:
    - Bold headwords: **ABACUS**, **MORAL PHILOSOPHY**,
    - Sub-entries: **Abacus Pythagoricus**,
    - Cross-references: See ASTRONOMY
    - No markdown headers (#) for article boundaries
    """

    def __init__(
        self,
        edition_year: int,
        edition_config: Optional[EditionConfig] = None,
        volume: Optional[int] = None,
        skip_front_matter: bool = True,
        min_article_length: int = 10,
    ):
        """
        Initialize the markdown extractor.

        Args:
            edition_year: Publication year
            edition_config: Optional custom config
            volume: Volume number
            skip_front_matter: Whether to skip title pages, prefaces, etc.
            min_article_length: Minimum article text length to include
        """
        super().__init__(edition_year, edition_config, volume)
        self.skip_front_matter = skip_front_matter
        self.min_article_length = min_article_length

    def extract_from_file(self, file_path: Path) -> Iterator[Article]:
        """
        Extract articles from a markdown file.

        Args:
            file_path: Path to the .md file

        Yields:
            Article objects
        """
        file_path = Path(file_path)

        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        # Try to detect volume from filename
        if self.volume is None:
            self.volume = self._detect_volume_from_filename(file_path.name)

        logger.info(f"Extracting from {file_path.name} (volume {self.volume})")

        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read()

        yield from self.extract_from_text(text)

    def extract_from_text(self, text: str) -> Iterator[Article]:
        """
        Extract articles from markdown text content.

        Args:
            text: The markdown text

        Yields:
            Article objects
        """
        # Find where actual content starts (skip front matter)
        if self.skip_front_matter:
            start_pos = find_first_article(text)
            if start_pos > 0:
                logger.info(f"Skipping {start_pos} chars of front matter")
                text = text[start_pos:]

        # Find all headword positions
        headwords = find_all_headwords(text)

        if not headwords:
            logger.warning("No headwords found in text")
            return

        logger.info(f"Found {len(headwords)} potential headwords")

        # Track headword counts for multi-sense entries
        headword_counts: dict[str, int] = {}

        # Extract articles between headword positions
        for i, (headword, pos, pattern_name) in enumerate(headwords):
            # Validate headword
            is_valid, error_reason = self._validate_headword(headword)

            # Determine end position (start of next article or end of text)
            if i + 1 < len(headwords):
                end_pos = headwords[i + 1][1]
            else:
                end_pos = len(text)

            # Extract article text
            article_text = text[pos:end_pos].strip()

            # Skip very short articles (likely errors)
            if len(article_text) < self.min_article_length:
                if is_valid:
                    self.warnings.append(f"Skipping short article: {headword}")
                continue

            # Track sense number for duplicate headwords
            normalized_headword = headword.upper().strip()
            headword_counts[normalized_headword] = headword_counts.get(normalized_headword, 0) + 1
            sense = headword_counts[normalized_headword]

            # Calculate confidence based on pattern and validation
            confidence = self._calculate_confidence(
                headword, article_text, pattern_name, is_valid
            )

            # Create article
            article = self._create_article(
                headword=headword,
                text=article_text,
                sense=sense,
                char_start=pos,
                char_end=end_pos,
                confidence=confidence,
                potential_error=not is_valid,
                error_reason=error_reason,
            )

            self.articles_extracted += 1
            yield article

        logger.info(f"Extracted {self.articles_extracted} articles")

    def _calculate_confidence(
        self,
        headword: str,
        text: str,
        pattern_name: str,
        is_valid: bool,
    ) -> float:
        """
        Calculate extraction confidence score.

        Args:
            headword: The extracted headword
            text: Article text
            pattern_name: Name of the pattern that matched
            is_valid: Whether headword passed validation

        Returns:
            Confidence score from 0.0 to 1.0
        """
        confidence = 1.0

        # Pattern-based adjustments
        pattern_confidence = {
            "bold_comma": 1.0,      # Most reliable
            "bold_period": 0.95,
            "bold_subentry": 0.85,  # Sub-entries might be false positives
            "caps_comma": 0.9,
            "caps_period": 0.85,
            "standalone": 0.7,      # Less reliable
            "parenthetical": 0.8,
        }
        confidence *= pattern_confidence.get(pattern_name, 0.7)

        # Validation-based adjustments
        if not is_valid:
            confidence *= 0.5

        # Length-based adjustments
        if len(text) < 50:
            confidence *= 0.8  # Very short articles are suspicious
        elif len(text) > 5000:
            confidence *= 1.0  # Long articles are usually correct

        # Known treatise boost
        if headword.upper() in self.edition_config.major_treatises:
            confidence = min(1.0, confidence * 1.2)

        return round(confidence, 3)


def parse_markdown_file(
    file_path: str | Path,
    edition_year: int,
    volume: Optional[int] = None,
    edition_config: Optional[EditionConfig] = None,
) -> list[Article]:
    """
    Convenience function to parse a markdown file.

    Args:
        file_path: Path to the markdown file
        edition_year: Publication year of the edition
        volume: Optional volume number
        edition_config: Optional custom configuration

    Returns:
        List of extracted Article objects
    """
    extractor = MarkdownExtractor(
        edition_year=edition_year,
        edition_config=edition_config,
        volume=volume,
    )
    return list(extractor.extract_from_file(Path(file_path)))


def parse_markdown_directory(
    directory: str | Path,
    edition_year: int,
    pattern: str = "*.md",
    edition_config: Optional[EditionConfig] = None,
) -> Iterator[Article]:
    """
    Parse all markdown files in a directory.

    Args:
        directory: Directory containing markdown files
        edition_year: Publication year
        pattern: Glob pattern for files (default: *.md)
        edition_config: Optional custom configuration

    Yields:
        Article objects from all files
    """
    directory = Path(directory)

    if not directory.is_dir():
        raise NotADirectoryError(f"Not a directory: {directory}")

    md_files = sorted(directory.glob(pattern))
    logger.info(f"Found {len(md_files)} markdown files in {directory}")

    for md_file in md_files:
        extractor = MarkdownExtractor(
            edition_year=edition_year,
            edition_config=edition_config,
        )
        try:
            yield from extractor.extract_from_file(md_file)
        except Exception as e:
            logger.error(f"Error processing {md_file}: {e}")
            continue
