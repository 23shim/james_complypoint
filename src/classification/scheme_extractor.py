"""
Scheme name extraction for clustering.

Extracts the core location identity from scheme folder names by:
  1. Stripping phase indicators, dates, and admin noise
  2. Finding address triggers (street suffixes) and capturing
     up to 2 adjacent words
  3. Finding exact place name matches (from the place_names dict)
  4. Combining both signals into a clean, normalised string

Runs AFTER scheme detection and hierarchy enforcement, BEFORE
entity clustering.  The extracted scheme name replaces the raw
folder name for clustering similarity so that "Phase 2 Liberty
Park Hoo Road" and "Liberty Park Phase 6" produce similar
extracted values and cluster together naturally.

Every scheme is guaranteed to have at least one address trigger
or place name (enforced by scheme detection's location gate).
"""

from __future__ import annotations

import re

from classification.address_extractor import (
    EDGE_PUNCT,
    WHITESPACE_RE,
    post_process,
    purge_dates,
)
from classification.models import EntityDefinition

# ---------------------------------------------------------------------------
# Noise patterns — strip universal metadata before extraction.
# Mirrors the patterns in entity_cluster.strip_scheme_noise but applied
# to original-cased text (all patterns are case-insensitive).
# ---------------------------------------------------------------------------

_MONTHS = (
    r"jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?"
    r"|jul(?:y)?|aug(?:ust)?|sep(?:t(?:ember)?)?|oct(?:ober)?"
    r"|nov(?:ember)?|dec(?:ember)?"
)
_ORDINALS = (
    r"one|two|three|four|five|six|seven|eight|nine|ten"
)

# Phase indicators: "Phase 1", "Ph.2", "Phase 4 & 5", "PHASE TWO"
_PHASE_RE = re.compile(
    rf"\b(?:phase|ph\.?)\s*(?:\d+(?:\s*&\s*\d+)*|{_ORDINALS})\b",
    re.IGNORECASE,
)
# "New" admin annotations: "New Dec 20", "New file - ...", "New scheme"
_NEW_ANNOTATION_RE = re.compile(
    rf"\bnew\s+(?:file\b[\s-]*|scheme\b|(?:{_MONTHS})(?:\s+\d{{2,4}})?)",
    re.IGNORECASE,
)
# Standalone 4-digit year (19xx/20xx)
_STANDALONE_YEAR_RE = re.compile(r"\b(?:19|20)\d{2}\b")
# Orphaned separators left after noise removal
_ORPHAN_SEPARATOR_RE = re.compile(r"\s+-\s+(?=-|\s|$)|\s+-$|^-\s+")
# Collapse whitespace
_COLLAPSE_WS_RE = re.compile(r"\s+")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def extract_scheme(
    raw_value: str,
    address_entity_def: EntityDefinition,
    place_names: set[str],
) -> str:
    """Extract the location identity from a scheme folder name.

    Args:
        raw_value: The raw scheme folder name
            (e.g. "Phase 2 Liberty Park Hoo Road").
        address_entity_def: The address EntityDefinition
            (tokens, abbreviations, abbreviation_map) — used to
            find street-suffix triggers.
        place_names: Set of known place names (lowercase, >= 4 chars).

    Returns:
        Extracted location identity string (lowercase, punctuation
        stripped), or empty string if no location signal found.
    """
    if not raw_value or not raw_value.strip():
        return ""

    text = raw_value.strip()

    # Step 0: Strip noise (phases, dates, annotations)
    text = _strip_noise(text)
    if not text:
        return ""

    words = text.split()
    if not words:
        return ""

    words_clean = [w.strip(EDGE_PUNCT).lower() for w in words]

    # Track which word indices to include in the output.
    include: set[int] = set()
    # Track abbreviation expansions (index → expanded form).
    expansions: dict[int, str] = {}

    # Step 1: Find place name matches FIRST — these take precedence
    # over abbreviation expansion (e.g. "St" in "St Albans" is a
    # place name, not "street").
    place_indices: set[int] = set()
    _find_place_names(words_clean, place_names, place_indices)
    include.update(place_indices)

    # Step 2: Find address triggers and capture context.
    # Words that are part of a place name match are not expanded.
    _find_address_triggers(
        words, words_clean, address_entity_def, include, expansions,
        place_indices,
    )

    if not include:
        return ""

    # Build result from included word indices, in order.
    result_words = []
    for i in sorted(include):
        if i in expansions:
            result_words.append(expansions[i])
        else:
            result_words.append(words[i])

    return post_process(" ".join(result_words))


# ---------------------------------------------------------------------------
# Step 0: Noise stripping
# ---------------------------------------------------------------------------

def _strip_noise(text: str) -> str:
    """Remove phase indicators, dates, and admin annotations.

    Applied before location extraction so that "Phase 2",
    "2020-2021", "New Dec 20" etc. don't pollute the output.
    """
    # Purge date patterns (written dates, dotted, 6-digit codes, year ranges)
    text = purge_dates(text)

    # Strip scheme-specific noise
    text = _NEW_ANNOTATION_RE.sub("", text)
    text = _PHASE_RE.sub("", text)
    text = _STANDALONE_YEAR_RE.sub("", text)

    # Collapse whitespace BEFORE orphan separator removal so that
    # leading/trailing dashes are at the string boundary.
    text = _COLLAPSE_WS_RE.sub(" ", text).strip()
    text = _ORPHAN_SEPARATOR_RE.sub("", text)
    text = _COLLAPSE_WS_RE.sub(" ", text)
    return text.strip()


# ---------------------------------------------------------------------------
# Step 1: Address trigger detection
# ---------------------------------------------------------------------------

def _find_address_triggers(
    words: list[str],
    words_clean: list[str],
    entity_def: EntityDefinition,
    include: set[int],
    expansions: dict[int, str],
    place_indices: set[int] | None = None,
) -> None:
    """Find address triggers and capture up to 2 preceding + 1 following word.

    Scans ALL triggers (not just the first), since scheme names can
    contain multiple street references (e.g. "Park Road" has both
    "park" and "road" as triggers).

    Words that are part of a place name match (in ``place_indices``)
    are NOT expanded — "St" in "St Albans" stays as "st", not "street".

    Mirrors address_extractor's Rule B (preceding) and Rule D (following).
    """
    trigger_tokens = set(entity_def.tokens)
    trigger_abbrs = set(entity_def.abbreviations)
    abbr_map = entity_def.abbreviation_map
    _place_indices = place_indices or set()

    for i, wc in enumerate(words_clean):
        is_token = wc in trigger_tokens
        is_abbr = wc in trigger_abbrs

        if not is_token and not is_abbr:
            continue

        include.add(i)

        # Expand abbreviation (rd → road, st → street, etc.)
        # but NOT if this word is part of a place name match
        # (e.g. "St" in "St Albans" is Saint, not Street).
        if is_abbr and i not in _place_indices:
            expanded = abbr_map.get(wc, "")
            if expanded:
                expansions[i] = expanded

        # Up to 2 preceding alphabetic words (like address Rule B).
        # Stop at numbers, linking words, or non-alphabetic tokens.
        count = 0
        for j in range(i - 1, max(i - 3, -1), -1):
            jwc = words_clean[j]
            if jwc.isalpha() and count < 2:
                include.add(j)
                count += 1
            else:
                break

        # Conditional following word (like address Rule D).
        # Skip if starts with bracket/digit, preserving the same
        # guards as the address extractor.
        if i + 1 < len(words):
            next_word = words[i + 1]
            if (
                next_word
                and next_word[0] not in "([{-"
                and not next_word[0].isdigit()
            ):
                include.add(i + 1)


# ---------------------------------------------------------------------------
# Step 2: Place name detection
# ---------------------------------------------------------------------------

def _find_place_names(
    words_clean: list[str],
    place_names: set[str],
    include: set[int],
) -> None:
    """Find exact place name matches in the tokenised scheme name.

    Checks individual tokens and consecutive bigrams against the
    place_names set.  Matches are taken exactly — the place name
    IS the location identity signal.
    """
    if not place_names:
        return

    # Individual tokens
    for i, wc in enumerate(words_clean):
        if wc in place_names:
            include.add(i)

    # Bigrams (e.g. "st albans", "kings lynn", "milton keynes")
    for i in range(len(words_clean) - 1):
        bigram = words_clean[i] + " " + words_clean[i + 1]
        if bigram in place_names:
            include.add(i)
            include.add(i + 1)
