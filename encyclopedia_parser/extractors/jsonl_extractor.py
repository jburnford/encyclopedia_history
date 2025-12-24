"""
JSONL file extractor for OLMoCR output.

Handles the .jsonl files produced by OLMoCR which contain the full document
text along with page mapping metadata.
"""

import json
from pathlib import Path
from typing import Iterator, Optional
import logging

from .base import BaseExtractor
from .md_extractor import MarkdownExtractor
from ..models import Article, EditionConfig

logger = logging.getLogger(__name__)


class JsonlExtractor(BaseExtractor):
    """
    Extract encyclopedia articles from OLMoCR JSONL files.

    JSONL files contain:
    - "text": Full OCR'd text content
    - "attributes.pdf_page_numbers": [[start_char, end_char, page_num], ...]
    - "metadata": Source file info, token counts, etc.
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
        Initialize the JSONL extractor.

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

        # Page mapping from JSONL metadata
        self._page_map: list[tuple[int, int, int]] = []  # (start_char, end_char, page_num)

    def extract_from_file(self, file_path: Path) -> Iterator[Article]:
        """
        Extract articles from a JSONL file.

        Args:
            file_path: Path to the .jsonl file

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

        # Read and parse JSONL (each line is a complete JSON document)
        with open(file_path, "r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue

                try:
                    doc = json.loads(line)
                except json.JSONDecodeError as e:
                    logger.error(f"JSON parse error on line {line_num}: {e}")
                    self.errors_encountered += 1
                    continue

                # Extract text content
                text = doc.get("text", "")
                if not text:
                    logger.warning(f"Empty text on line {line_num}")
                    continue

                # Extract page mapping if available
                attributes = doc.get("attributes", {})
                self._page_map = attributes.get("pdf_page_numbers", [])

                # Extract articles from text
                yield from self.extract_from_text(text)

    def extract_from_text(self, text: str) -> Iterator[Article]:
        """
        Extract articles from text content.

        Uses MarkdownExtractor internally since JSONL text is in markdown format.

        Args:
            text: The text content

        Yields:
            Article objects with page numbers populated
        """
        # Create a MarkdownExtractor to do the actual parsing
        md_extractor = MarkdownExtractor(
            edition_year=self.edition_year,
            edition_config=self.edition_config,
            volume=self.volume,
            skip_front_matter=self.skip_front_matter,
            min_article_length=self.min_article_length,
        )

        # Extract articles
        for article in md_extractor.extract_from_text(text):
            # Add page numbers from JSONL metadata
            if article.char_start is not None and self._page_map:
                article.start_page = self._get_page_for_char(article.char_start)
            if article.char_end is not None and self._page_map:
                article.end_page = self._get_page_for_char(article.char_end - 1)

            self.articles_extracted += 1
            yield article

        # Copy statistics from markdown extractor
        self.warnings.extend(md_extractor.warnings)

    def _get_page_for_char(self, char_pos: int) -> Optional[int]:
        """
        Look up the page number for a character position.

        Args:
            char_pos: Character position in the text

        Returns:
            Page number or None if not found
        """
        for start_char, end_char, page_num in self._page_map:
            if start_char <= char_pos <= end_char:
                return page_num
        return None


def parse_jsonl_file(
    file_path: str | Path,
    edition_year: int,
    volume: Optional[int] = None,
    edition_config: Optional[EditionConfig] = None,
) -> list[Article]:
    """
    Convenience function to parse a JSONL file.

    Args:
        file_path: Path to the JSONL file
        edition_year: Publication year of the edition
        volume: Optional volume number
        edition_config: Optional custom configuration

    Returns:
        List of extracted Article objects
    """
    extractor = JsonlExtractor(
        edition_year=edition_year,
        edition_config=edition_config,
        volume=volume,
    )
    return list(extractor.extract_from_file(Path(file_path)))


def parse_jsonl_directory(
    directory: str | Path,
    edition_year: int,
    pattern: str = "*.jsonl",
    edition_config: Optional[EditionConfig] = None,
) -> Iterator[Article]:
    """
    Parse all JSONL files in a directory.

    Args:
        directory: Directory containing JSONL files
        edition_year: Publication year
        pattern: Glob pattern for files (default: *.jsonl)
        edition_config: Optional custom configuration

    Yields:
        Article objects from all files
    """
    directory = Path(directory)

    if not directory.is_dir():
        raise NotADirectoryError(f"Not a directory: {directory}")

    jsonl_files = sorted(directory.glob(pattern))
    logger.info(f"Found {len(jsonl_files)} JSONL files in {directory}")

    for jsonl_file in jsonl_files:
        extractor = JsonlExtractor(
            edition_year=edition_year,
            edition_config=edition_config,
        )
        try:
            yield from extractor.extract_from_file(jsonl_file)
        except Exception as e:
            logger.error(f"Error processing {jsonl_file}: {e}")
            continue
