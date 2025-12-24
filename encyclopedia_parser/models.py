"""
Pydantic models for encyclopedia article parsing.
"""

from typing import Literal, Optional
from pydantic import BaseModel, Field
from enum import Enum


class ArticleType(str, Enum):
    """Classification of encyclopedia article types."""
    DICTIONARY = "dictionary"       # Short definition entries
    TREATISE = "treatise"          # Long scholarly articles (e.g., ASTRONOMY, CHEMISTRY)
    BIOGRAPHICAL = "biographical"   # Person entries with dates
    GEOGRAPHICAL = "geographical"   # Place entries with coordinates
    CROSS_REFERENCE = "cross_reference"  # "See X" entries


class Article(BaseModel):
    """Represents a single encyclopedia article."""

    # Core identification
    headword: str = Field(description="The main entry word (normalized to uppercase)")
    sense: int = Field(default=1, description="Sense number for multi-entry headwords")

    # Classification
    article_type: ArticleType = Field(default=ArticleType.DICTIONARY)
    confidence: float = Field(default=1.0, ge=0.0, le=1.0, description="Extraction confidence score")

    # Source information
    edition_year: int = Field(description="Publication year of the edition")
    edition_name: str = Field(description="Human-readable edition name")
    volume: Optional[int] = Field(default=None, description="Volume number if applicable")

    # Location in source
    start_page: Optional[int] = Field(default=None)
    end_page: Optional[int] = Field(default=None)
    char_start: Optional[int] = Field(default=None, description="Character offset in source file")
    char_end: Optional[int] = Field(default=None)

    # Content
    text: str = Field(description="Full article text")

    # Flags
    is_cross_reference: bool = Field(default=False)
    potential_error: bool = Field(default=False)
    error_reason: Optional[str] = Field(default=None)

    # Extracted metadata (populated by classifiers)
    cross_references: list[str] = Field(default_factory=list, description="Referenced headwords")
    coordinates: Optional[tuple[float, float]] = Field(default=None, description="(longitude, latitude) for geographical entries")
    person_dates: Optional[str] = Field(default=None, description="Birth-death dates for biographical entries")

    class Config:
        use_enum_values = True


class Section(BaseModel):
    """A structural section within an encyclopedia article."""

    title: str = Field(description="Section title/heading")
    level: int = Field(default=1, ge=1, le=3, description="Hierarchy level: 1=PART, 2=SECT/§, 3=subsection")
    index: int = Field(description="Section index within article")

    # Parent article info
    parent_headword: str = Field(description="Headword of parent article")
    edition_year: int

    # Location within article
    char_start: int = Field(description="Character offset within article text")
    char_end: int

    # Content
    text: str = Field(description="Full section text")

    # Extraction metadata
    extraction_method: Literal["explicit", "llm", "fallback"] = Field(
        default="explicit",
        description="How this section was identified"
    )


class TextChunk(BaseModel):
    """A semantic chunk of text from a treatise article, suitable for embedding."""

    text: str = Field(description="The chunk text content")
    index: int = Field(description="Chunk index within the parent article")
    parent_headword: str = Field(description="Headword of the parent article")
    edition_year: int

    # Location within article
    char_start: int = Field(description="Character offset within article text")
    char_end: int

    # Section context (for knowledge graph hierarchy)
    section_title: Optional[str] = Field(default=None, description="Title of parent section")
    section_index: Optional[int] = Field(default=None, description="Index of parent section")


class CrossReference(BaseModel):
    """Represents a cross-reference between articles."""

    source_headword: str = Field(description="Article containing the reference")
    target_headword: str = Field(description="Referenced article headword")
    reference_type: Literal["see", "see_also", "under", "related"] = Field(default="see")
    context: str = Field(description="Surrounding text for debugging")
    source_edition_year: int


class EditionConfig(BaseModel):
    """Configuration for a specific encyclopedia edition."""

    year: int
    name: str
    volumes: int

    # Volume letter ranges (volume_num -> (start_letter, end_letter))
    volume_ranges: dict[int, tuple[str, str]] = Field(default_factory=dict)

    # Known major treatises that shouldn't be split
    major_treatises: list[str] = Field(default_factory=list)

    # Front matter patterns to skip
    front_matter_keywords: list[str] = Field(
        default_factory=lambda: [
            "PREFACE", "DEDICATION", "CONTENTS", "ADVERTISEMENT",
            "DICTIONARY OF ARTS", "ENCYCLOPÆDIA BRITANNICA"
        ]
    )


# Pre-configured editions
EDITION_CONFIGS = {
    1704: EditionConfig(
        year=1704,
        name="Lexicon Technicum",
        volumes=1,
        major_treatises=[]
    ),
    1728: EditionConfig(
        year=1728,
        name="Chambers Cyclopaedia",
        volumes=2,
        major_treatises=[]
    ),
    1771: EditionConfig(
        year=1771,
        name="Britannica 1st",
        volumes=3,
        volume_ranges={
            1: ("A", "B"),
            2: ("C", "L"),
            3: ("M", "Z")
        },
        major_treatises=[
            "AGRICULTURE", "ALGEBRA", "ANATOMY", "ARCHITECTURE", "ARITHMETIC",
            "ASTRONOMY", "BOTANY", "BREWING", "CHEMISTRY", "CHRONOLOGY",
            "COMMERCE", "COSMOGRAPHY", "CRITICISM", "DISTILLING", "DYEING",
            "ELECTRICITY", "ETHICS", "FARRIERY", "FLUXIONS", "FORTIFICATION",
            "GEOGRAPHY", "GEOMETRY", "GRAMMAR", "HERALDRY", "HISTORY",
            "HYDRAULICS", "HYDROSTATICS", "LAW", "LOGIC", "MAGNETISM",
            "MECHANICS", "MEDICINE", "METAPHYSICS", "MIDWIFERY", "MUSIC",
            "NAVIGATION", "OPTICS", "PAINTING", "PERSPECTIVE", "PHARMACY",
            "PHILOSOPHY", "PHYSIOLOGY", "PNEUMATICS", "POETRY", "RHETORIC",
            "SCULPTURE", "SURGERY", "THEOLOGY", "TRIGONOMETRY", "WAR"
        ]
    ),
    1778: EditionConfig(
        year=1778,
        name="Britannica 2nd",
        volumes=10,
        major_treatises=[
            "AGRICULTURE", "ALGEBRA", "ANATOMY", "ARCHITECTURE", "ARITHMETIC",
            "ASTRONOMY", "BOTANY", "CHEMISTRY", "CHRONOLOGY", "COMMERCE",
            "ELECTRICITY", "GEOMETRY", "GRAMMAR", "HISTORY", "LAW",
            "LOGIC", "MECHANICS", "MEDICINE", "MUSIC", "NAVIGATION",
            "OPTICS", "PAINTING", "PHILOSOPHY", "PHYSIOLOGY", "POETRY",
            "SURGERY", "THEOLOGY"
        ]
    ),
    1797: EditionConfig(
        year=1797,
        name="Britannica 3rd",
        volumes=18,
        major_treatises=[
            "AGRICULTURE", "ALGEBRA", "ANATOMY", "ARCHITECTURE", "ARITHMETIC",
            "ASTRONOMY", "BOTANY", "CHEMISTRY", "CHRONOLOGY", "COMMERCE",
            "ELECTRICITY", "GEOMETRY", "GRAMMAR", "HISTORY", "LAW",
            "LOGIC", "MECHANICS", "MEDICINE", "MUSIC", "NAVIGATION",
            "OPTICS", "PAINTING", "PHILOSOPHY", "PHYSIOLOGY", "POETRY",
            "SURGERY", "THEOLOGY"
        ]
    ),
    1810: EditionConfig(
        year=1810,
        name="Britannica 4th",
        volumes=20,
        major_treatises=[]  # TODO: Add known treatises
    ),
    1815: EditionConfig(
        year=1815,
        name="Britannica 5th",
        volumes=20,
        volume_ranges={
            1: ("A", "ANA"), 2: ("ANA", "BAC"), 3: ("BAC", "BUR"),
            4: ("BUR", "CHI"), 5: ("CHI", "CRI"), 6: ("CRI", "ECO"),
            7: ("ECO", "FEU"), 8: ("FEU", "GOR"), 9: ("GOR", "HYD"),
            10: ("HYD", "LEP"), 11: ("LEP", "MED"), 12: ("MED", "MUS"),
            13: ("MUS", "PEN"), 14: ("PEN", "PRI"), 15: ("PRI", "SAL"),
            16: ("SAL", "SHI"), 17: ("SHI", "STE"), 18: ("STE", "TUR"),
            19: ("TUR", "WAT"), 20: ("WAT", "ZYM")
        },
        major_treatises=[
            "AGRICULTURE", "ALGEBRA", "ANATOMY", "ARCHITECTURE", "ARITHMETIC",
            "ASTRONOMY", "BOTANY", "BREWING", "CHEMISTRY", "CHRONOLOGY",
            "COMMERCE", "CRYSTALLIZATION", "DISTILLING", "DYEING",
            "ELECTRICITY", "ETHICS", "FARRIERY", "FLUXIONS", "FORTIFICATION",
            "GEOGRAPHY", "GEOLOGY", "GEOMETRY", "GRAMMAR", "HERALDRY",
            "HISTORY", "HYDRAULICS", "HYDROSTATICS", "LAW", "LOGIC",
            "MAGNETISM", "MECHANICS", "MEDICINE", "METAPHYSICS", "MIDWIFERY",
            "MINERALOGY", "MUSIC", "NAVIGATION", "OPTICS", "PAINTING",
            "PERSPECTIVE", "PHARMACY", "PHILOSOPHY", "PHYSIOLOGY", "PNEUMATICS",
            "POETRY", "RHETORIC", "SCULPTURE", "SURGERY", "THEOLOGY",
            "TRIGONOMETRY", "WAR"
        ]
    ),
    1823: EditionConfig(
        year=1823,
        name="Britannica 6th",
        volumes=20,
        major_treatises=[]
    ),
    1842: EditionConfig(
        year=1842,
        name="Britannica 7th",
        volumes=21,
        major_treatises=[]
    ),
    1860: EditionConfig(
        year=1860,
        name="Britannica 8th",
        volumes=21,
        major_treatises=[]
    ),
}


def get_edition_config(year: int) -> EditionConfig:
    """Get configuration for a specific edition year."""
    if year not in EDITION_CONFIGS:
        # Return a generic config
        return EditionConfig(
            year=year,
            name=f"Edition {year}",
            volumes=1
        )
    return EDITION_CONFIGS[year]
