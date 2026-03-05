"""
Pre-computed fuzzy matching index.

Builds an edit-distance-1 variant dictionary at config load time so
that classification-time lookups remain O(1). Only applied to type
tokens — not categories or entities.

Safety guards:
  - Words shorter than ``min_token_length`` (default 5) are excluded.
  - Variants that are themselves canonical dictionary words are excluded
    (prevents mapping real words to other real words).
"""

from __future__ import annotations

import logging
from string import ascii_lowercase

from classification.models import ClassificationConfig

logger = logging.getLogger(__name__)


def build_fuzzy_index(config: ClassificationConfig) -> dict[str, str]:
    """Build a variant → canonical word mapping for fuzzy type matching.

    Scans all type token phrases, splits into individual words, and
    generates edit-distance-1 variants for words meeting the length
    threshold.  Returns a dict suitable for O(1) lookup.

    Args:
        config: Fully loaded classification config (types + weights).

    Returns:
        Mapping of ``{variant_word: canonical_word}`` for all valid
        variants.  Empty dict if fuzzy is disabled or has no eligible
        words.
    """
    fuzzy_cfg = config.weights.get("fuzzy", {})
    min_length = fuzzy_cfg.get("min_token_length", 5)

    # Collect all canonical words from type token phrases.
    canonical_words: set[str] = set()
    for type_def in config.types.values():
        for phrase in type_def.tokens:
            for word in phrase.split():
                canonical_words.add(word)

    # Also include category signals so we don't remap real category words.
    for cat_def in config.categories.values():
        for signal in cat_def.signals:
            for word in signal.split():
                canonical_words.add(word)

    # Build variant index — only for words >= min_length.
    index: dict[str, str] = {}
    eligible = {w for w in canonical_words if len(w) >= min_length}

    for word in eligible:
        for variant in _edit_distance_1_variants(word):
            # Skip if variant is a real dictionary word.
            if variant in canonical_words:
                continue
            # First canonical mapping wins (deterministic via set order,
            # but collisions across different canonical words are rare
            # at edit distance 1 with length >= 5).
            if variant not in index:
                index[variant] = word

    logger.info(
        "Fuzzy index built: %d canonical words (>=%d chars), %d variants",
        len(eligible),
        min_length,
        len(index),
    )
    return index


def _edit_distance_1_variants(word: str) -> set[str]:
    """Generate all strings within edit distance 1 of *word*.

    Produces variants via four operations:
      - Deletions:      remove one character
      - Insertions:     insert one a-z character at each position
      - Substitutions:  replace one character with a-z
      - Transpositions: swap two adjacent characters

    Args:
        word: Lowercase input word.

    Returns:
        Set of variant strings (excludes the original word itself).
    """
    variants: set[str] = set()
    n = len(word)

    # Deletions
    for i in range(n):
        variants.add(word[:i] + word[i + 1:])

    # Insertions
    for i in range(n + 1):
        for c in ascii_lowercase:
            variants.add(word[:i] + c + word[i:])

    # Substitutions
    for i in range(n):
        for c in ascii_lowercase:
            if c != word[i]:
                variants.add(word[:i] + c + word[i + 1:])

    # Transpositions
    for i in range(n - 1):
        if word[i] != word[i + 1]:
            variants.add(word[:i] + word[i + 1] + word[i] + word[i + 2:])

    variants.discard(word)
    return variants
