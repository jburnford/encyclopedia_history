"""
Rule-based validation for extracted articles.
"""

from typing import Optional
import logging

from ..models import Article

logger = logging.getLogger(__name__)


class RuleBasedValidator:
    """
    Validates extracted articles using rule-based heuristics.

    Flags potential errors and low-confidence extractions.
    """

    def __init__(
        self,
        min_length: int = 20,
        max_headword_length: int = 50,
        require_text_after_headword: bool = True,
    ):
        """
        Initialize the validator.

        Args:
            min_length: Minimum article text length
            max_headword_length: Maximum headword length (longer = suspicious)
            require_text_after_headword: Whether article must have text after headword
        """
        self.min_length = min_length
        self.max_headword_length = max_headword_length
        self.require_text_after_headword = require_text_after_headword

        # Suspicious patterns that might indicate running headers
        self.suspicious_patterns = [
            "PLATE",
            "VOLUME",
            "VOL",
            "CHAPTER",
            "PART",
            "PAGE",
        ]

    def validate(self, article: Article) -> tuple[bool, Optional[str]]:
        """
        Validate an article.

        Args:
            article: The article to validate

        Returns:
            (is_valid, error_reason) tuple
        """
        headword = article.headword
        text = article.text

        # Check headword length
        if len(headword) > self.max_headword_length:
            return False, f"Headword too long ({len(headword)} chars)"

        # Check for suspicious patterns
        for pattern in self.suspicious_patterns:
            if headword.startswith(pattern):
                return False, f"Suspicious pattern: {pattern}"

        # Check minimum length
        if len(text) < self.min_length:
            return False, f"Text too short ({len(text)} chars)"

        # Check that text contains more than just the headword
        if self.require_text_after_headword:
            # Remove headword from beginning of text
            text_without_headword = text.replace(headword, "", 1).strip()
            # Remove common punctuation
            text_without_headword = text_without_headword.lstrip(",.:;")
            if len(text_without_headword) < 10:
                return False, "No content after headword"

        # Check for pure Roman numerals (likely page/section markers)
        if self._is_roman_numeral(headword):
            return False, "Roman numeral headword"

        return True, None

    def _is_roman_numeral(self, text: str) -> bool:
        """Check if text is a pure Roman numeral."""
        # Common Roman numerals, but exclude real words
        if text in ["I", "II", "III", "IV", "V", "VI", "VII", "VIII", "IX", "X",
                    "XI", "XII", "XIII", "XIV", "XV", "XVI", "XVII", "XVIII", "XIX", "XX",
                    "XXI", "XXX", "XL", "L", "LX", "LXX", "LXXX", "XC", "C", "CC", "CCC",
                    "CD", "D", "DC", "DCC", "DCCC", "CM", "M", "MM", "MMM"]:
            return True
        return False

    def validate_batch(self, articles: list[Article]) -> tuple[list[Article], list[Article]]:
        """
        Validate a batch of articles.

        Args:
            articles: List of articles to validate

        Returns:
            (valid_articles, invalid_articles) tuple
        """
        valid = []
        invalid = []

        for article in articles:
            is_valid, error_reason = self.validate(article)
            if is_valid:
                valid.append(article)
            else:
                article.potential_error = True
                article.error_reason = error_reason
                invalid.append(article)

        logger.info(f"Validation: {len(valid)} valid, {len(invalid)} invalid")
        return valid, invalid
