"""
Data structures for the classification layer.

Pure data — no logic. These are the interchange types that
flow between modules.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum


class SignalSource(Enum):
    """Where a signal was detected."""
    FOLDER_TYPE = "folder_type"
    FOLDER_CATEGORY = "folder_category"
    FOLDER_ENTITY = "folder_entity"
    FILENAME_TOKEN = "filename_token"
    FILENAME_ABBREVIATION = "filename_abbreviation"
    FILENAME_PATTERN = "filename_pattern"
    EXTENSION_HINT = "extension_hint"


class MatchMethod(Enum):
    """How the match was made."""
    TOKEN = "token"
    ABBREVIATION = "abbreviation"
    PATTERN = "pattern"
    FUZZY_TOKEN = "fuzzy_token"


@dataclass(frozen=True)
class Signal:
    """A single classification signal detected from a segment or filename."""
    source: SignalSource
    label: str              # The matched type or category name
    match_term: str         # The term/pattern that triggered the match
    match_method: MatchMethod
    base_weight: float      # Weight from signal_weights config
    depth: int = 0          # Folder depth where found (0-indexed, -1 for filename)
    text: str = ""          # The original text that was scanned
    origin_layer: int = 0   # 0=base, 1=industry, 2=client


@dataclass
class TypeDefinition:
    """A document type from the config dictionary."""
    name: str
    tokens: list[str]
    abbreviations: list[str]
    compiled_patterns: list[re.Pattern]
    belongs_to: str
    extensions: list[str] = field(default_factory=list)
    origin_layer: int = 0  # 0=base, 1=industry, 2=client


@dataclass
class CategoryDefinition:
    """A category from the config dictionary."""
    name: str
    signals: list[str]
    aliases: list[str]
    scheme_signal: bool = False


@dataclass
class EntityDefinition:
    """An entity type from the config dictionary.

    Same matching fields as TypeDefinition (tokens, abbreviations,
    patterns) but answers a different question: "what real-world
    thing is this document about?" rather than "what kind of
    document is this?"
    """
    name: str
    tokens: list[str]
    abbreviations: list[str]
    compiled_patterns: list[re.Pattern]
    abbreviation_map: dict[str, str] = field(default_factory=dict)


@dataclass
class SchemeExclusions:
    """Folder names/patterns that should never be detected as schemes.

    Loaded from the ``scheme_exclusions`` section of the industry or
    client dictionary YAML.  Used by scheme_detector to filter false
    positives.
    """
    names: set[str]                          # exact names (lower-cased)
    substrings: list[str]                    # lower-cased substrings
    compiled_patterns: list[re.Pattern]      # pre-compiled regexes
    container_keywords: list[str] = field(default_factory=list)


@dataclass
class ClassificationConfig:
    """Merged, ready-to-use classification config."""
    types: dict[str, TypeDefinition]
    categories: dict[str, CategoryDefinition]
    weights: dict
    entities: dict[str, EntityDefinition] = field(default_factory=dict)
    scheme_exclusions: SchemeExclusions = field(
        default_factory=lambda: SchemeExclusions(set(), [], [])
    )
    place_names: set[str] = field(default_factory=set)
    fuzzy_index: dict[str, str] = field(default_factory=dict)
    version: str = "1.0"


@dataclass
class ClassificationResult:
    """The output of classifying a single file."""
    inferred_type: str
    type_confidence: float
    inferred_category: str
    category_confidence: float
    overall_confidence: float
    confidence_band: str          # "High", "Medium", or "Low"
    readiness_status: str         # "Ready" (High band) or "Not Ready"
    reasoning_trace: list[str]
    secondary_type: str = ""               # Runner-up type (if any)
    secondary_type_confidence: float = 0.0
    entities: dict[str, str] = field(default_factory=dict)
    entity_confidences: dict[str, float] = field(default_factory=dict)
    entity_depths: dict[str, int] = field(default_factory=dict)
