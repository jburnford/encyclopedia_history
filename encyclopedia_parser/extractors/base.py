"""
Abstract base class for encyclopedia extractors.
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Iterator, Optional

from ..models import Article, EditionConfig, get_edition_config


class BaseExtractor(ABC):
    """
    Abstract base class for extracting articles from OCR output files.

    Subclasses implement specific file format handling (markdown, JSONL, etc.)
    """

    def __init__(
        self,
        edition_year: int,
        edition_config: Optional[EditionConfig] = None,
        volume: Optional[int] = None,
    ):
        """
        Initialize the extractor.

        Args:
            edition_year: Publication year of the edition
            edition_config: Optional custom configuration, otherwise loaded from defaults
            volume: Optional volume number for multi-volume works
        """
        self.edition_year = edition_year
        self.edition_config = edition_config or get_edition_config(edition_year)
        self.volume = volume

        # Statistics
        self.articles_extracted = 0
        self.errors_encountered = 0
        self.warnings = []

    @abstractmethod
    def extract_from_file(self, file_path: Path) -> Iterator[Article]:
        """
        Extract articles from a file.

        Args:
            file_path: Path to the OCR output file

        Yields:
            Article objects extracted from the file
        """
        pass

    @abstractmethod
    def extract_from_text(self, text: str) -> Iterator[Article]:
        """
        Extract articles from raw text content.

        Args:
            text: The OCR text content

        Yields:
            Article objects extracted from the text
        """
        pass

    def _create_article(
        self,
        headword: str,
        text: str,
        sense: int = 1,
        char_start: Optional[int] = None,
        char_end: Optional[int] = None,
        start_page: Optional[int] = None,
        end_page: Optional[int] = None,
        confidence: float = 1.0,
        potential_error: bool = False,
        error_reason: Optional[str] = None,
    ) -> Article:
        """
        Create an Article instance with common fields populated.

        Args:
            headword: The article headword
            text: Full article text
            sense: Sense number for multi-entry headwords
            char_start: Character offset in source
            char_end: End character offset
            start_page: Starting page number
            end_page: Ending page number
            confidence: Extraction confidence (0.0-1.0)
            potential_error: Whether this might be a false positive
            error_reason: Explanation if potential_error is True

        Returns:
            Article instance
        """
        return Article(
            headword=headword.upper().strip(),
            sense=sense,
            edition_year=self.edition_year,
            edition_name=self.edition_config.name,
            volume=self.volume,
            char_start=char_start,
            char_end=char_end,
            start_page=start_page,
            end_page=end_page,
            text=text.strip(),
            confidence=confidence,
            potential_error=potential_error,
            error_reason=error_reason,
        )

    def _validate_headword(self, headword: str) -> tuple[bool, Optional[str]]:
        """
        Validate that a headword is legitimate.

        Args:
            headword: The candidate headword

        Returns:
            (is_valid, error_reason) tuple
        """
        from ..patterns import is_structural_header

        # Skip structural headers
        if is_structural_header(headword):
            return False, "Structural header detected"

        # Skip very short headwords (except single letters like A, B)
        if len(headword) < 2 and not headword.isalpha():
            return False, "Headword too short"

        # Skip if it's just numbers
        if headword.replace("-", "").replace("'", "").isdigit():
            return False, "Numeric headword"

        # Validate against volume letter range if configured
        if self.volume and self.edition_config.volume_ranges:
            if self.volume in self.edition_config.volume_ranges:
                start_letter, end_letter = self.edition_config.volume_ranges[self.volume]
                first_letter = headword[0].upper()
                if not (start_letter <= first_letter <= end_letter):
                    return False, f"Headword '{headword}' outside volume range {start_letter}-{end_letter}"

        return True, None

    def _detect_volume_from_filename(self, filename: str) -> Optional[int]:
        """
        Try to detect volume number from filename.

        Args:
            filename: The filename to parse

        Returns:
            Volume number if detected, otherwise None
        """
        import re

        # Pattern: Volume 1, Vol. 1, Vol 1, v1, etc.
        patterns = [
            r'Volume\s*(\d+)',
            r'Vol\.?\s*(\d+)',
            r'[Vv](\d+)',
            r'_(\d+)(?:_|\.)',  # underscore delimited
        ]

        for pattern in patterns:
            match = re.search(pattern, filename, re.IGNORECASE)
            if match:
                return int(match.group(1))

        return None

    def get_statistics(self) -> dict:
        """
        Get extraction statistics.

        Returns:
            Dictionary with extraction stats
        """
        return {
            "articles_extracted": self.articles_extracted,
            "errors_encountered": self.errors_encountered,
            "warnings_count": len(self.warnings),
            "warnings": self.warnings[:10],  # First 10 warnings
            "edition_year": self.edition_year,
            "edition_name": self.edition_config.name,
            "volume": self.volume,
        }
