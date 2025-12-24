"""
Article type classification and metadata extraction.

Classifies articles as: dictionary, treatise, biographical, geographical, cross_reference
"""

import re
from typing import Optional
import logging

from .models import Article, ArticleType, EditionConfig
from .patterns import (
    FULL_CROSS_REF,
    extract_coordinates,
    extract_dates,
    extract_cross_references,
)

logger = logging.getLogger(__name__)


class ArticleClassifier:
    """
    Classifies encyclopedia articles by type and extracts relevant metadata.
    """

    # Thresholds for classification
    TREATISE_LENGTH_THRESHOLD = 5000  # Characters
    SHORT_CROSS_REF_THRESHOLD = 100   # Max length for "See X" only entries

    def __init__(self, edition_config: Optional[EditionConfig] = None):
        """
        Initialize the classifier.

        Args:
            edition_config: Optional edition configuration for treatise list
        """
        self.edition_config = edition_config
        self.major_treatises = set(
            edition_config.major_treatises if edition_config else []
        )

    def classify(self, article: Article) -> Article:
        """
        Classify an article and extract relevant metadata.

        Modifies the article in place and returns it.

        Args:
            article: The article to classify

        Returns:
            The article with article_type and metadata populated
        """
        text = article.text
        headword = article.headword

        # Extract metadata first (needed for classification)
        article.cross_references = [ref for ref, _ in extract_cross_references(text)]
        article.coordinates = extract_coordinates(text)
        article.person_dates = extract_dates(text)

        # Classify in priority order
        if self._is_cross_reference(article):
            article.article_type = ArticleType.CROSS_REFERENCE
            article.is_cross_reference = True

        elif self._is_geographical(article):
            article.article_type = ArticleType.GEOGRAPHICAL

        elif self._is_biographical(article):
            article.article_type = ArticleType.BIOGRAPHICAL

        elif self._is_treatise(article):
            article.article_type = ArticleType.TREATISE

        else:
            article.article_type = ArticleType.DICTIONARY

        return article

    def _is_cross_reference(self, article: Article) -> bool:
        """
        Check if article is a cross-reference entry.

        Cross-references are short entries that just redirect to another article.
        Examples:
            "See ASTRONOMY"
            "**ABRIDGEMENT.** See Abstract."
        """
        text = article.text.strip()

        # Check if entire text is a "See X" reference
        if FULL_CROSS_REF.match(text):
            return True

        # Short text with "See" is likely a cross-reference
        if len(text) < self.SHORT_CROSS_REF_THRESHOLD:
            if re.search(r'\bSee\s+[A-Z]', text, re.IGNORECASE):
                return True

        return False

    def _is_geographical(self, article: Article) -> bool:
        """
        Check if article is a geographical entry.

        Geographical entries describe places and typically contain coordinates.
        Examples:
            "MONTREAL, a city in Canada... E. Long. 73.35. N. Lat. 45.30."
        """
        # Has coordinates
        if article.coordinates is not None:
            return True

        # Contains coordinate patterns but extraction failed
        text = article.text
        if re.search(r'[EW]\.\s*Long\.', text, re.IGNORECASE):
            return True
        if re.search(r'[NS]\.\s*Lat\.', text, re.IGNORECASE):
            return True

        # Common geographical indicators
        geo_indicators = [
            r'\b(?:city|town|village|river|mountain|island|county|district)\s+(?:of|in)\b',
            r'\b(?:lies|situated|bounded|borders)\b',
            r'\bmiles?\s+(?:north|south|east|west|from)\b',
            r'\bpopulation\s+(?:of\s+)?\d',
        ]

        text_lower = text.lower()
        indicator_count = sum(
            1 for pattern in geo_indicators
            if re.search(pattern, text_lower)
        )

        # Multiple indicators suggest geographical entry
        return indicator_count >= 2

    def _is_biographical(self, article: Article) -> bool:
        """
        Check if article is a biographical entry.

        Biographical entries describe people with birth/death dates.
        Examples:
            "NEWTON (Sir Isaac), born 1642, died 1727..."
        """
        # Has extracted dates
        if article.person_dates is not None:
            return True

        text = article.text

        # Common biographical patterns
        bio_patterns = [
            r'\bborn\s+(?:in\s+)?(?:about\s+)?\d{4}',
            r'\bdied\s+(?:in\s+)?(?:about\s+)?\d{4}',
            r'\(\d{4}\s*[-–—]\s*\d{4}\)',  # (1642-1727)
            r'\bflourished\s+(?:about\s+)?\d{4}',
            r'\b(?:he|she)\s+was\s+(?:a\s+)?(?:celebrated|famous|eminent|distinguished)',
            r'\b(?:son|daughter)\s+of\b',
        ]

        for pattern in bio_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                return True

        return False

    def _is_treatise(self, article: Article) -> bool:
        """
        Check if article is a treatise (long scholarly article).

        Treatises are major articles covering entire subjects like ASTRONOMY,
        CHEMISTRY, MEDICINE, etc.
        """
        headword = article.headword.upper()

        # Known major treatise
        if headword in self.major_treatises:
            return True

        # Long articles are likely treatises
        if len(article.text) > self.TREATISE_LENGTH_THRESHOLD:
            return True

        # Contains section structure
        section_patterns = [
            r'\bPART\s+[IVXLCDM]+\b',
            r'\bSECT(?:ION)?\.?\s+[IVXLCDM\d]+\b',
            r'\bCHAPTER\s+[IVXLCDM\d]+\b',
        ]

        for pattern in section_patterns:
            if re.search(pattern, article.text[:2000]):  # Check beginning
                return True

        return False


def classify_article(
    article: Article,
    edition_config: Optional[EditionConfig] = None,
) -> Article:
    """
    Convenience function to classify a single article.

    Args:
        article: The article to classify
        edition_config: Optional edition configuration

    Returns:
        The classified article
    """
    classifier = ArticleClassifier(edition_config)
    return classifier.classify(article)


def classify_articles(
    articles: list[Article],
    edition_config: Optional[EditionConfig] = None,
) -> list[Article]:
    """
    Classify a list of articles.

    Args:
        articles: List of articles to classify
        edition_config: Optional edition configuration

    Returns:
        List of classified articles
    """
    classifier = ArticleClassifier(edition_config)
    return [classifier.classify(article) for article in articles]


def get_classification_stats(articles: list[Article]) -> dict:
    """
    Get classification statistics for a list of articles.

    Args:
        articles: List of classified articles

    Returns:
        Dictionary with counts by article type
    """
    stats = {
        "total": len(articles),
        "by_type": {},
        "with_cross_references": 0,
        "with_coordinates": 0,
        "with_dates": 0,
        "average_length": 0,
        "treatise_count": 0,
    }

    type_counts = {}
    total_length = 0

    for article in articles:
        # Count by type
        article_type = article.article_type
        if isinstance(article_type, ArticleType):
            article_type = article_type.value
        type_counts[article_type] = type_counts.get(article_type, 0) + 1

        # Count metadata
        if article.cross_references:
            stats["with_cross_references"] += 1
        if article.coordinates:
            stats["with_coordinates"] += 1
        if article.person_dates:
            stats["with_dates"] += 1

        total_length += len(article.text)

    stats["by_type"] = type_counts
    stats["average_length"] = total_length // len(articles) if articles else 0
    stats["treatise_count"] = type_counts.get("treatise", 0)

    return stats
