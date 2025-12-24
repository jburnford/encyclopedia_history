"""
Section extraction for encyclopedia articles.

Extracts hierarchical sections from treatise articles using:
1. Explicit markers (§, PART, SECT, CHAP) - free, rule-based
2. LLM-based detection for articles without explicit structure - uses OpenAI

This creates the middle layer of the knowledge graph:
    Article → Section → Chunk
"""

import logging
import os
import re
from typing import Optional

from .models import Article, Section

logger = logging.getLogger(__name__)


# =============================================================================
# Explicit Section Extraction (Free, Rule-Based)
# =============================================================================

# Patterns for explicit section markers (must be at start of line or after newline)
SECTION_PATTERNS = [
    # Level 1: Major divisions
    (r'(?:^|\n)(PART\s+[IVXLCDM]+\.?\s*[^\n]*)', 1, "PART"),

    # Level 2: Sections and chapters
    (r'(?:^|\n)(§\s*\d+\.?\s*[^\n.]{3,80}?)(?:\n|\.(?:\s|$))', 2, "§"),
    (r'(?:^|\n)(SECT\.?\s*[IVXLCDM\d]+\.?\s*[^\n]*)', 2, "SECT"),
    (r'(?:^|\n)(CHAP\.?\s*[IVXLCDM\d]+\.?\s*[^\n]*)', 2, "CHAP"),
    (r'(?:^|\n)(Chapter\s+[IVXLCDM\d]+\.?\s*[^\n]*)', 2, "Chapter"),

    # Level 2: Numbered sections in various formats
    (r'(?:^|\n)(Division\s+[IVXLCDM\d]+\.?\s*[^\n]*)', 2, "Division"),
    (r'(?:^|\n)(Article\s+[IVXLCDM\d]+\.?\s*[^\n]*)', 2, "Article"),
]


def extract_explicit_sections(
    article: Article,
    min_section_length: int = 500,
    min_intro_length: int = 1000,
) -> list[Section]:
    """
    Extract sections using explicit markers (§, PART, SECT, CHAP, etc.).

    Args:
        article: The article to extract sections from
        min_section_length: Minimum chars for a valid section (filters noise)
        min_intro_length: Minimum chars before first marker to create intro section

    Returns:
        List of Section objects, or empty list if no explicit structure found
    """
    text = article.text

    # Find all section markers with their positions
    markers = []

    for pattern, level, marker_type in SECTION_PATTERNS:
        for match in re.finditer(pattern, text):
            title = match.group(1).strip()
            start_pos = match.start(1)

            # Skip if this looks like it's inside a sentence (lowercase follows)
            context_after = text[match.end():match.end()+20]
            if context_after and context_after[0].islower():
                continue

            # Clean up the title
            title = re.sub(r'\s+', ' ', title)
            title = title[:100]  # Cap length

            markers.append({
                'title': title,
                'level': level,
                'type': marker_type,
                'start': start_pos,
            })

    if not markers:
        return []

    # Sort by position
    markers.sort(key=lambda x: x['start'])

    # Remove duplicates (same position)
    seen_positions = set()
    unique_markers = []
    for m in markers:
        if m['start'] not in seen_positions:
            seen_positions.add(m['start'])
            unique_markers.append(m)
    markers = unique_markers

    # Create sections
    sections = []

    # Check for intro section (content before first marker)
    first_marker_start = markers[0]['start'] if markers else len(text)
    if first_marker_start >= min_intro_length:
        intro_text = text[:first_marker_start]
        sections.append(Section(
            title="Introduction",
            level=1,
            index=0,
            parent_headword=article.headword,
            edition_year=article.edition_year,
            char_start=0,
            char_end=first_marker_start,
            text=intro_text,
            extraction_method="explicit",
        ))
        logger.info(
            f"Added intro section for {article.headword}: {len(intro_text):,} chars before first marker"
        )

    for i, marker in enumerate(markers):
        # End position is start of next marker, or end of text
        end_pos = markers[i + 1]['start'] if i + 1 < len(markers) else len(text)

        section_text = text[marker['start']:end_pos]

        # Skip very short sections (likely false positives)
        if len(section_text) < min_section_length:
            continue

        section = Section(
            title=marker['title'],
            level=marker['level'],
            index=len(sections),
            parent_headword=article.headword,
            edition_year=article.edition_year,
            char_start=marker['start'],
            char_end=end_pos,
            text=section_text,
            extraction_method="explicit",
        )
        sections.append(section)

    logger.info(
        f"Extracted {len(sections)} explicit sections from {article.headword}"
    )

    return sections


# =============================================================================
# LLM-Based Section Extraction (For articles without explicit markers)
# =============================================================================

SECTION_EXTRACTION_PROMPT_SIMPLE = """Analyze this encyclopedia article and identify its major thematic sections.

Article: {headword}
Text (first 8000 chars):
---
{text_sample}
---

Identify 3-8 major thematic sections in this article. For each section, provide:
1. A short descriptive title (2-5 words)
2. The approximate starting text (first 10-15 words of that section)

Output as JSON array:
[
  {{"title": "Section Title", "starts_with": "first words of section..."}},
  ...
]

Rules:
- Sections should be substantial (at least 1000+ chars each)
- Titles should be descriptive (e.g., "History", "Geography", "Trade and Commerce")
- Don't create sections for very short content
- If the article has no clear structure, return fewer sections

JSON output:"""


SECTION_EXTRACTION_PROMPT_MULTI = """Analyze these samples from an encyclopedia article about {headword} to identify its major thematic sections.

{samples_text}

Based on these samples from different parts of the article, identify 8-15 major sections.
For each section, provide a title and a distinctive marker phrase (exact text that appears at the section start).

IMPORTANT:
- Identify where topics CHANGE (geography → statistics → early history → medieval history, etc.)
- Make sure sections are spread throughout the article
- Each section should be 5,000-20,000 chars ideally

Output as JSON array:
[
  {{"title": "Section Name", "marker": "exact phrase from article start"}},
  ...
]

JSON:"""


async def extract_sections_with_llm_async(
    article: Article,
    openai_api_key: Optional[str] = None,
    model: str = "gpt-4o-mini",
) -> list[Section]:
    """
    Extract sections using LLM for articles without explicit markers.

    Async version for batch processing.
    """
    try:
        from openai import AsyncOpenAI
    except ImportError:
        raise ImportError("LLM section extraction requires openai package")

    api_key = openai_api_key or os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OpenAI API key required")

    client = AsyncOpenAI(api_key=api_key)

    # Prepare prompt with text sample
    text_sample = article.text[:8000]
    prompt = SECTION_EXTRACTION_PROMPT.format(
        headword=article.headword,
        text_sample=text_sample,
    )

    response = await client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
        max_tokens=1000,
    )

    return _parse_llm_sections(article, response.choices[0].message.content)


def extract_sections_with_llm(
    article: Article,
    openai_api_key: Optional[str] = None,
    model: str = "gpt-4o-mini",
    multi_sample_threshold: int = 30000,
) -> list[Section]:
    """
    Extract sections using LLM for articles without explicit markers.

    For longer articles (>30K chars), uses multi-sample approach to detect
    sections throughout the entire article.

    Sync version.
    """
    try:
        from openai import OpenAI
    except ImportError:
        raise ImportError("LLM section extraction requires openai package")

    api_key = openai_api_key or os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OpenAI API key required")

    client = OpenAI(api_key=api_key)
    text = article.text

    # Use multi-sample for longer articles
    if len(text) > multi_sample_threshold:
        return _extract_sections_multi_sample(article, client, model)

    # Simple approach for shorter articles
    text_sample = text[:8000]
    prompt = SECTION_EXTRACTION_PROMPT_SIMPLE.format(
        headword=article.headword,
        text_sample=text_sample,
    )

    logger.info(f"Using LLM to extract sections from {article.headword}")

    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
        max_tokens=1000,
    )

    return _parse_llm_sections(article, response.choices[0].message.content)


def _extract_sections_multi_sample(
    article: Article,
    client,
    model: str,
) -> list[Section]:
    """
    Multi-sample approach for longer articles.

    Samples text at regular intervals throughout the article to detect
    section transitions across the entire content.
    """
    text = article.text
    sample_size = 3000
    sample_interval = 15000

    # Generate sample positions
    positions = list(range(0, len(text), sample_interval))
    if len(text) - positions[-1] > 5000:
        positions.append(len(text) - sample_size)

    logger.info(
        f"Multi-sample LLM extraction for {article.headword}: "
        f"{len(positions)} samples"
    )

    # Build samples text
    samples_text = ""
    for i, pos in enumerate(positions):
        sample = text[pos:pos + sample_size]
        pct = 100 * pos // len(text)
        samples_text += f"\n--- SAMPLE {i+1} (position {pos:,}, ~{pct}% through article) ---\n"
        samples_text += sample + "\n"

    prompt = SECTION_EXTRACTION_PROMPT_MULTI.format(
        headword=article.headword,
        samples_text=samples_text,
    )

    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
        max_tokens=2000,
    )

    return _parse_llm_sections_multi(article, response.choices[0].message.content)


def _parse_llm_sections(article: Article, llm_response: str) -> list[Section]:
    """Parse LLM response into Section objects (simple format with starts_with)."""
    import json

    # Extract JSON from response
    try:
        json_match = re.search(r'\[.*\]', llm_response, re.DOTALL)
        if json_match:
            section_data = json.loads(json_match.group())
        else:
            logger.warning(f"No JSON found in LLM response for {article.headword}")
            return []
    except json.JSONDecodeError as e:
        logger.warning(f"Failed to parse LLM response for {article.headword}: {e}")
        return []

    text = article.text
    sections = []

    for i, item in enumerate(section_data):
        title = item.get('title', f'Section {i+1}')
        starts_with = item.get('starts_with', '')

        # Find the section start position
        if starts_with:
            start_pos = text.find(starts_with[:50])
            if start_pos == -1:
                start_pos = _fuzzy_find(text, starts_with[:50])
        else:
            start_pos = -1

        if start_pos == -1:
            logger.debug(f"Could not locate section '{title}' in {article.headword}")
            continue

        sections.append({'title': title, 'start': start_pos})

    return _build_sections_from_positions(article, sections, "llm")


def _parse_llm_sections_multi(article: Article, llm_response: str) -> list[Section]:
    """Parse LLM response into Section objects (multi-sample format with marker)."""
    import json

    # Extract JSON from response
    try:
        json_match = re.search(r'\[.*\]', llm_response, re.DOTALL)
        if json_match:
            section_data = json.loads(json_match.group())
        else:
            logger.warning(f"No JSON found in LLM response for {article.headword}")
            return []
    except json.JSONDecodeError as e:
        logger.warning(f"Failed to parse LLM response for {article.headword}: {e}")
        return []

    text = article.text
    sections = []

    for i, item in enumerate(section_data):
        title = item.get('title', f'Section {i+1}')
        marker = item.get('marker', '')

        # Find marker position with fallback to partial match
        start_pos = -1
        if marker:
            start_pos = text.find(marker)
            if start_pos == -1:
                # Try progressively shorter matches
                for length in [60, 40, 25, 15]:
                    if len(marker) >= length:
                        start_pos = text.find(marker[:length])
                        if start_pos >= 0:
                            break

        if start_pos == -1:
            logger.debug(f"Could not locate section '{title}' in {article.headword}")
            continue

        sections.append({'title': title, 'start': start_pos})

    return _build_sections_from_positions(article, sections, "llm")


def _build_sections_from_positions(
    article: Article,
    sections: list[dict],
    method: str,
) -> list[Section]:
    """Build Section objects from a list of {title, start} dicts."""
    if not sections:
        return []

    text = article.text

    # Sort by position and remove duplicates
    sections.sort(key=lambda x: x['start'])
    unique = []
    seen_pos = set()
    for sec in sections:
        # Skip if too close to previous (within 500 chars)
        if any(abs(sec['start'] - p) < 500 for p in seen_pos):
            continue
        seen_pos.add(sec['start'])
        unique.append(sec)
    sections = unique

    # Build Section objects
    result = []
    for i, sec in enumerate(sections):
        end_pos = sections[i + 1]['start'] if i + 1 < len(sections) else len(text)

        section = Section(
            title=sec['title'],
            level=2,
            index=i,
            parent_headword=article.headword,
            edition_year=article.edition_year,
            char_start=sec['start'],
            char_end=end_pos,
            text=text[sec['start']:end_pos],
            extraction_method=method,
        )
        result.append(section)

    logger.info(f"Built {len(result)} sections from {article.headword} ({method})")

    return result


def _fuzzy_find(text: str, query: str, threshold: float = 0.8) -> int:
    """Find approximate location of query in text."""
    # Simple approach: search for longest matching substring
    query_words = query.split()[:5]  # First 5 words

    for num_words in range(len(query_words), 0, -1):
        search_str = ' '.join(query_words[:num_words])
        pos = text.find(search_str)
        if pos != -1:
            return pos

    return -1


# =============================================================================
# Fallback Section Extraction (When explicit and LLM both unavailable)
# =============================================================================

def extract_sections_by_length(
    article: Article,
    target_section_size: int = 10000,
    min_sections: int = 2,
    max_sections: int = 10,
) -> list[Section]:
    """
    Fallback: split article into roughly equal sections by paragraph boundaries.

    Used when:
    - No explicit markers found
    - LLM not available or too expensive
    """
    text = article.text

    # Calculate target number of sections
    num_sections = max(min_sections, min(max_sections, len(text) // target_section_size))

    # Find paragraph boundaries
    paragraphs = re.split(r'\n\s*\n', text)

    if len(paragraphs) < num_sections:
        # Not enough paragraphs, just split evenly
        chunk_size = len(text) // num_sections
        sections = []
        for i in range(num_sections):
            start = i * chunk_size
            end = (i + 1) * chunk_size if i < num_sections - 1 else len(text)
            sections.append(Section(
                title=f"Part {i + 1}",
                level=2,
                index=i,
                parent_headword=article.headword,
                edition_year=article.edition_year,
                char_start=start,
                char_end=end,
                text=text[start:end],
                extraction_method="fallback",
            ))
        return sections

    # Group paragraphs into sections
    target_para_per_section = len(paragraphs) // num_sections

    sections = []
    current_start = 0
    current_pos = 0
    para_count = 0

    for para in paragraphs:
        para_count += 1
        current_pos += len(para) + 2  # +2 for \n\n

        if para_count >= target_para_per_section and len(sections) < num_sections - 1:
            sections.append(Section(
                title=f"Part {len(sections) + 1}",
                level=2,
                index=len(sections),
                parent_headword=article.headword,
                edition_year=article.edition_year,
                char_start=current_start,
                char_end=current_pos,
                text=text[current_start:current_pos],
                extraction_method="fallback",
            ))
            current_start = current_pos
            para_count = 0

    # Add final section
    if current_start < len(text):
        sections.append(Section(
            title=f"Part {len(sections) + 1}",
            level=2,
            index=len(sections),
            parent_headword=article.headword,
            edition_year=article.edition_year,
            char_start=current_start,
            char_end=len(text),
            text=text[current_start:],
            extraction_method="fallback",
        ))

    logger.info(f"Fallback extracted {len(sections)} sections from {article.headword}")

    return sections


# =============================================================================
# Main Entry Point
# =============================================================================

def extract_sections(
    article: Article,
    use_llm: bool = True,
    openai_api_key: Optional[str] = None,
    min_article_length: int = 5000,
) -> list[Section]:
    """
    Extract sections from an article using the best available method.

    Strategy:
    1. Try explicit markers first (free)
    2. If no explicit sections and use_llm=True, use LLM
    3. Fall back to length-based splitting

    Args:
        article: The article to extract sections from
        use_llm: Whether to use LLM for articles without explicit markers
        openai_api_key: OpenAI API key (or use OPENAI_API_KEY env var)
        min_article_length: Don't extract sections from short articles

    Returns:
        List of Section objects
    """
    # Short articles don't need sections
    if len(article.text) < min_article_length:
        return [Section(
            title=article.headword,
            level=1,
            index=0,
            parent_headword=article.headword,
            edition_year=article.edition_year,
            char_start=0,
            char_end=len(article.text),
            text=article.text,
            extraction_method="fallback",
        )]

    # Try explicit extraction first
    sections = extract_explicit_sections(article)

    if len(sections) >= 2:
        return sections

    # Try LLM if enabled
    if use_llm:
        try:
            sections = extract_sections_with_llm(article, openai_api_key)
            if len(sections) >= 2:
                return sections
        except Exception as e:
            logger.warning(f"LLM section extraction failed for {article.headword}: {e}")

    # Fall back to length-based
    return extract_sections_by_length(article)


def get_section_stats(sections: list[Section]) -> dict:
    """Get statistics about extracted sections."""
    if not sections:
        return {"count": 0}

    sizes = [len(s.text) for s in sections]
    methods = {}
    for s in sections:
        methods[s.extraction_method] = methods.get(s.extraction_method, 0) + 1

    return {
        "count": len(sections),
        "total_chars": sum(sizes),
        "min_size": min(sizes),
        "max_size": max(sizes),
        "avg_size": sum(sizes) // len(sizes),
        "by_method": methods,
        "levels": {s.level: sum(1 for x in sections if x.level == s.level) for s in sections},
    }
