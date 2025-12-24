"""
Consolidated regex patterns for encyclopedia article extraction.

These patterns detect article headwords in various formats produced by OLMoCR.
"""

import re
from dataclasses import dataclass
from typing import Pattern


@dataclass
class ArticlePattern:
    """A named pattern with priority and description."""
    name: str
    pattern: Pattern
    priority: int  # Lower = higher priority
    description: str


# =============================================================================
# HEADWORD DETECTION PATTERNS
# =============================================================================

# Bold headword followed by comma (most common in OLMoCR output)
# Matches: **ABACUS**, **MORAL PHILOSOPHY**,
BOLD_HEADWORD_COMMA = re.compile(
    r'^\*\*([A-Z][A-Z\'\-]+(?:\s+[A-Z][A-Z\'\-]+)*)\*\*,\s*',
    re.MULTILINE
)

# Bold headword followed by period (treatise start)
# Matches: **ASTRONOMY**.
BOLD_HEADWORD_PERIOD = re.compile(
    r'^\*\*([A-Z][A-Z\'\-]+(?:\s+[A-Z][A-Z\'\-]+)*)\*\*\.\s*$',
    re.MULTILINE
)

# Bold sub-entry (title case, not all caps)
# Matches: **Abacus Pythagoricus**, **Chinese Abacus**
BOLD_SUBENTRY = re.compile(
    r'^\*\*([A-Z][a-z][A-Za-z\'\-\s]+)\*\*[,.]',
    re.MULTILINE
)

# Plain ALL-CAPS headword with comma (no bold)
# Matches: ABACUS, the name of...
CAPS_HEADWORD_COMMA = re.compile(
    r'^([A-Z][A-Z\'\-]+(?:\s+[A-Z][A-Z\'\-]+)*),\s+',
    re.MULTILINE
)

# Plain ALL-CAPS headword with period (treatise, no bold)
# Matches: ASTRONOMY.
CAPS_HEADWORD_PERIOD = re.compile(
    r'^([A-Z][A-Z\'\-]+(?:\s+[A-Z][A-Z\'\-]+)*)\.\s*$',
    re.MULTILINE
)

# Headword alone on line followed by double newline
# Matches: CHEMISTRY\n\nChemistry is...
HEADWORD_STANDALONE = re.compile(
    r'^([A-Z][A-Z\'\-]+(?:\s+[A-Z][A-Z\'\-]+)*)\s*$\n\n(?=[A-Z])',
    re.MULTILINE
)

# Headword with parenthetical
# Matches: **MOON, (Luna,)**, BANK (financial),
HEADWORD_PARENTHETICAL = re.compile(
    r'^\*?\*?([A-Z][A-Z\'\-]+(?:\s+[A-Z][A-Z\'\-]+)*)\s*[,]?\s*\([^)]+\)\*?\*?[,.]',
    re.MULTILINE
)


# =============================================================================
# CROSS-REFERENCE PATTERNS
# =============================================================================

# Simple "See X" reference
# Matches: See ASTRONOMY, See CHEMISTRY
CROSS_REF_SEE = re.compile(
    r'\bSee\s+([A-Z][A-Z\'\-]+(?:\s+[A-Z][A-Z\'\-]+)*)',
    re.IGNORECASE
)

# "See the article X" reference
CROSS_REF_ARTICLE = re.compile(
    r'\bSee\s+(?:the\s+)?article\s+([A-Z][A-Z\'\-]+)',
    re.IGNORECASE
)

# Bold cross-reference: See **Astronomy**
CROSS_REF_BOLD = re.compile(
    r'\bSee\s+\*\*([A-Za-z][A-Za-z\'\-\s]+)\*\*',
    re.IGNORECASE
)

# "under X" reference
CROSS_REF_UNDER = re.compile(
    r'\bunder\s+(?:the\s+(?:article|head|word)\s+)?([A-Z][A-Z\'\-]+)',
    re.IGNORECASE
)

# Full cross-reference article detection (entire text is a reference)
FULL_CROSS_REF = re.compile(
    r'^See\s+[A-Z]',
    re.IGNORECASE
)


# =============================================================================
# GEOGRAPHICAL PATTERNS
# =============================================================================

# Longitude/Latitude patterns
COORDINATES_LONG_LAT = re.compile(
    r'[EW]\.\s*Long\.\s*([\d\.]+).*?[NS]\.\s*Lat\.\s*([\d\.]+)',
    re.IGNORECASE | re.DOTALL
)

# Alternative coordinate format
COORDINATES_LAT_LONG = re.compile(
    r'Lat\.\s*([\d\.]+)\s*[NS]?.*?Long\.\s*([\d\.]+)\s*[EW]?',
    re.IGNORECASE | re.DOTALL
)


# =============================================================================
# BIOGRAPHICAL PATTERNS
# =============================================================================

# Birth-death dates in parentheses: (1723-1790)
DATES_PARENS = re.compile(
    r'\((\d{4})\s*[-–—]\s*(\d{4})\)'
)

# "born XXXX" pattern
DATE_BORN = re.compile(
    r'\b(?:born|b\.)\s+(?:in\s+)?(\d{4})',
    re.IGNORECASE
)

# "died XXXX" pattern
DATE_DIED = re.compile(
    r'\b(?:died|d\.)\s+(?:in\s+)?(\d{4})',
    re.IGNORECASE
)

# Century references for older figures
DATE_CENTURY = re.compile(
    r'\b(\d{1,2})(?:st|nd|rd|th)\s+century',
    re.IGNORECASE
)


# =============================================================================
# STRUCTURAL PATTERNS (for filtering false positives)
# =============================================================================

# Running headers / structural elements to skip
STRUCTURAL_HEADERS = re.compile(
    r'^(?:CHAPTER|PART|SECT(?:ION)?|PLATE|FIG(?:URE)?|TABLE|VOL(?:UME)?|BOOK)\s*[IVXLCDM\d]+',
    re.IGNORECASE | re.MULTILINE
)

# Page numbers
PAGE_NUMBER = re.compile(
    r'^\d+\s*$',
    re.MULTILINE
)

# Roman numerals (but not real words like VI=6)
ROMAN_NUMERAL = re.compile(
    r'^[IVXLCDM]+\s*$',
    re.MULTILINE
)


# =============================================================================
# FRONT MATTER DETECTION
# =============================================================================

# Patterns indicating front matter (title page, preface, etc.)
FRONT_MATTER_INDICATORS = [
    re.compile(r'ENCYCLOP[AÆ]DIA\s+BRITANNICA', re.IGNORECASE),
    re.compile(r'DICTIONARY\s+OF\s+ARTS', re.IGNORECASE),
    re.compile(r'PREFACE', re.IGNORECASE),
    re.compile(r'DEDICATION', re.IGNORECASE),
    re.compile(r'TO\s+THE\s+KING', re.IGNORECASE),
    re.compile(r'ADVERTISEMENT', re.IGNORECASE),
    re.compile(r'TABLE\s+OF\s+CONTENTS', re.IGNORECASE),
    re.compile(r'COPPERPLATES', re.IGNORECASE),
    re.compile(r'EDINBURGH.*Printed', re.IGNORECASE | re.DOTALL),
]

# First real article indicators (single letter entries like "A", "B")
ALPHABET_ENTRY = re.compile(
    r'^([A-Z])\s*$\n\n',
    re.MULTILINE
)


# =============================================================================
# CONSOLIDATED PATTERN LIST
# =============================================================================

PATTERNS = {
    "headwords": [
        ArticlePattern("bold_comma", BOLD_HEADWORD_COMMA, 1, "Bold ALL-CAPS with comma"),
        ArticlePattern("bold_period", BOLD_HEADWORD_PERIOD, 2, "Bold ALL-CAPS with period"),
        ArticlePattern("bold_subentry", BOLD_SUBENTRY, 3, "Bold title-case sub-entry"),
        ArticlePattern("caps_comma", CAPS_HEADWORD_COMMA, 4, "Plain ALL-CAPS with comma"),
        ArticlePattern("caps_period", CAPS_HEADWORD_PERIOD, 5, "Plain ALL-CAPS with period"),
        ArticlePattern("standalone", HEADWORD_STANDALONE, 6, "Headword alone on line"),
        ArticlePattern("parenthetical", HEADWORD_PARENTHETICAL, 7, "Headword with parenthetical"),
    ],
    "cross_references": [
        ArticlePattern("see", CROSS_REF_SEE, 1, "See X reference"),
        ArticlePattern("article", CROSS_REF_ARTICLE, 2, "See the article X"),
        ArticlePattern("bold", CROSS_REF_BOLD, 3, "See **X** bold reference"),
        ArticlePattern("under", CROSS_REF_UNDER, 4, "under X reference"),
    ],
    "coordinates": [
        ArticlePattern("long_lat", COORDINATES_LONG_LAT, 1, "E. Long. / N. Lat."),
        ArticlePattern("lat_long", COORDINATES_LAT_LONG, 2, "Lat. / Long."),
    ],
    "dates": [
        ArticlePattern("parens", DATES_PARENS, 1, "(1723-1790) format"),
        ArticlePattern("born", DATE_BORN, 2, "born XXXX"),
        ArticlePattern("died", DATE_DIED, 3, "died XXXX"),
        ArticlePattern("century", DATE_CENTURY, 4, "Nth century"),
    ],
    "structural": [
        ArticlePattern("headers", STRUCTURAL_HEADERS, 1, "Chapter/Section headers"),
        ArticlePattern("page_num", PAGE_NUMBER, 2, "Page numbers"),
        ArticlePattern("roman", ROMAN_NUMERAL, 3, "Roman numerals"),
    ],
}


def find_all_headwords(text: str) -> list[tuple[str, int, str]]:
    """
    Find all potential article headwords in text.

    Returns:
        List of (headword, position, pattern_name) tuples, sorted by position.
    """
    matches = []

    for pattern_info in PATTERNS["headwords"]:
        for match in pattern_info.pattern.finditer(text):
            headword = match.group(1).strip()
            matches.append((headword, match.start(), pattern_info.name))

    # Sort by position, deduplicate overlapping matches
    matches.sort(key=lambda x: x[1])

    # Remove duplicates within 20 characters
    deduped = []
    last_pos = -100
    for headword, pos, pattern_name in matches:
        if pos - last_pos > 20:
            deduped.append((headword, pos, pattern_name))
            last_pos = pos

    return deduped


def extract_cross_references(text: str) -> list[tuple[str, str]]:
    """
    Extract all cross-references from article text.

    Returns:
        List of (target_headword, reference_type) tuples.
    """
    refs = []

    for pattern_info in PATTERNS["cross_references"]:
        for match in pattern_info.pattern.finditer(text):
            target = match.group(1).strip().upper()
            refs.append((target, pattern_info.name))

    # Deduplicate
    return list(set(refs))


def extract_coordinates(text: str) -> tuple[float, float] | None:
    """
    Extract longitude/latitude coordinates from geographical entry text.

    Returns:
        (longitude, latitude) tuple or None if not found.
    """
    for pattern_info in PATTERNS["coordinates"]:
        match = pattern_info.pattern.search(text)
        if match:
            try:
                if pattern_info.name == "long_lat":
                    long_val = float(match.group(1))
                    lat_val = float(match.group(2))
                else:
                    lat_val = float(match.group(1))
                    long_val = float(match.group(2))
                return (long_val, lat_val)
            except ValueError:
                continue
    return None


def extract_dates(text: str) -> str | None:
    """
    Extract biographical dates from text.

    Returns:
        Date string like "1723-1790" or "born 1650" or None.
    """
    # Try parenthetical dates first
    match = DATES_PARENS.search(text[:500])  # Only check beginning
    if match:
        return f"{match.group(1)}-{match.group(2)}"

    # Try born/died
    born = DATE_BORN.search(text[:500])
    died = DATE_DIED.search(text[:500])

    if born and died:
        return f"{born.group(1)}-{died.group(1)}"
    elif born:
        return f"born {born.group(1)}"
    elif died:
        return f"died {died.group(1)}"

    return None


def is_structural_header(headword: str) -> bool:
    """Check if a headword is actually a structural header (CHAPTER I, etc.)."""
    return bool(STRUCTURAL_HEADERS.match(headword))


def is_front_matter(text: str) -> bool:
    """Check if text appears to be front matter (title page, preface, etc.)."""
    for pattern in FRONT_MATTER_INDICATORS:
        if pattern.search(text[:2000]):  # Only check beginning
            return True
    return False


def find_first_article(text: str) -> int:
    """
    Find the position of the first real article in the text.

    Skips front matter and finds where actual encyclopedia content begins.

    Returns:
        Character position of first article, or 0 if not determinable.
    """
    # Look for single-letter entry (A, B, etc.) which marks start of content
    match = ALPHABET_ENTRY.search(text)
    if match:
        return match.start()

    # Otherwise find first bold headword after front matter region
    headwords = find_all_headwords(text)
    for headword, pos, pattern in headwords:
        # Skip if in first 1000 chars (likely front matter)
        if pos < 1000:
            continue
        # Skip structural headers
        if is_structural_header(headword):
            continue
        # Skip very short headwords that might be false positives
        if len(headword) < 2:
            continue
        return pos

    return 0
