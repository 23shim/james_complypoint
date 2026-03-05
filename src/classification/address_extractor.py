"""
Address extraction from noisy folder/file name segments.

Uses an "Anchor-Trigger" method to extract clean, standardised
address substrings from raw entity values like folder names.
The trigger is a known street-suffix token/abbreviation; the
anchor is a preceding house number or street name words.

Pipeline:
  0. Purge dates (6-digit codes, dotted/slashed, written dates)
  1. Extract postcodes (structurally unambiguous)
  2. Anchor-Trigger extraction (number anchor + preceding words +
     trigger token + conditional following word)
  3. Post-process (lowercase, strip punctuation, expand trigger
     abbreviation only)

Runs AFTER address validation and BEFORE entity clustering.
The extracted address is used for clustering similarity instead
of the raw folder segment name.
"""

from __future__ import annotations

import re

from classification.models import EntityDefinition

# ---------------------------------------------------------------------------
# Compiled regex patterns — built once at import time.
# ---------------------------------------------------------------------------

# Month names for date detection (full and abbreviated).
_MONTHS_FULL = (
    "January|February|March|April|May|June|"
    "July|August|September|October|November|December"
)
_MONTHS_ABBR = "Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec"
_MONTHS_ALL = f"{_MONTHS_FULL}|{_MONTHS_ABBR}"

# Date patterns — applied in order (written dates before 6-digit codes).
_DATE_PATTERNS: list[re.Pattern] = [
    # "10 July 24", "3 January 2021"
    re.compile(
        rf"\b\d{{1,2}}\s+(?:{_MONTHS_ALL})\s+\d{{2,4}}\b", re.IGNORECASE,
    ),
    # "Aug 2022", "January 2021"
    re.compile(
        rf"\b(?:{_MONTHS_ALL})\s+\d{{4}}\b", re.IGNORECASE,
    ),
    # "(2015-2016)" — parenthesised year range
    re.compile(r"\(\d{4}[-\u2013]\d{4}\)"),
    # "2015-2016" — bare year range
    re.compile(r"\b\d{4}[-\u2013]\d{4}\b"),
    # "18.10.13", "20/05/22"
    re.compile(r"\b\d{1,2}[./]\d{1,2}[./]\d{2,4}\b"),
    # 6-digit date codes: "160219", "240120"
    re.compile(r"\b\d{6}\b"),
]

# UK postcode: SW1A 1AA, TN13 1AB, DA1 2AB, ME10 3ZZ etc.
_UK_POSTCODE_RE = re.compile(
    r"\b([A-Z]{1,2}\d{1,2}[A-Z]?\s*\d[A-Z]{2})\b",
    re.IGNORECASE,
)

# Collapse runs of whitespace.
WHITESPACE_RE = re.compile(r"\s+")

# Linking words that the greedy number anchor can swallow.
LINKING_WORDS = {"to", "and", "&"}

# Characters to strip from word edges during comparison.
EDGE_PUNCT = ".,;:'\"!?#()[]{}/"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def extract_address(
    raw_value: str,
    entity_def: EntityDefinition,
) -> str:
    """Extract a clean, standardised address from a raw segment name.

    Args:
        raw_value: The raw folder segment name
            (e.g. "plots 67-70 Hawkes Way").
        entity_def: The address EntityDefinition
            (tokens, abbreviations, patterns, abbreviation_map).

    Returns:
        Extracted and standardised address string (lowercase,
        trigger abbreviation expanded), or empty string if no
        address signal found.
    """
    if not raw_value or not raw_value.strip():
        return ""

    text = raw_value.strip()

    # Step 0: Purge dates
    text = purge_dates(text)
    if not text:
        return ""

    # Step 1: Extract postcode (set aside)
    postcode = ""
    postcode_match = _UK_POSTCODE_RE.search(text)
    if postcode_match:
        postcode = postcode_match.group(1).upper()
        text = text[:postcode_match.start()] + text[postcode_match.end():]
        text = WHITESPACE_RE.sub(" ", text).strip()

    # Step 2: Anchor-Trigger extraction
    extracted = _anchor_trigger_extract(text, entity_def)

    # Step 3: Post-process
    if extracted:
        result = post_process(extracted)
    elif postcode:
        result = postcode.lower()
    else:
        return ""

    # Append postcode if both street and postcode found
    if postcode and extracted:
        result = f"{result} {postcode.lower()}"

    return result


# ---------------------------------------------------------------------------
# Step 0: Date purging
# ---------------------------------------------------------------------------

def purge_dates(text: str) -> str:
    """Remove all date patterns from the string.

    Applied before any address logic so that trailing dates
    (e.g. "160219", "18.10.13", "10 July 24") don't interfere
    with the number anchor or pollute similarity matching.
    """
    for pattern in _DATE_PATTERNS:
        text = pattern.sub("", text)
    return WHITESPACE_RE.sub(" ", text).strip()


# ---------------------------------------------------------------------------
# Step 2: Anchor-Trigger extraction
# ---------------------------------------------------------------------------

def _anchor_trigger_extract(
    text: str,
    entity_def: EntityDefinition,
) -> str:
    """Core Anchor-Trigger extraction.

    Finds the trigger token/abbreviation, then:
      - Rule B: capture 1-2 preceding alphabetic words
      - Rule A: greedily scan backward for number anchor
      - Rule C: capture the trigger (expand if abbreviation)
      - Rule D: conditionally capture the following word
    """
    words = text.split()
    if not words:
        return ""

    # Clean versions for matching (stripped of edge punctuation, lowered).
    words_clean = [w.strip(EDGE_PUNCT).lower() for w in words]

    # Build trigger sets from config.
    trigger_tokens = set(entity_def.tokens)
    trigger_abbrs = set(entity_def.abbreviations)
    abbr_map = entity_def.abbreviation_map

    # Scan left-to-right for the first trigger.
    # Tokens take priority over abbreviations: do a full pass for
    # tokens first so that "St Mary's Close" matches "close" (token)
    # rather than "st" (abbreviation for "street").
    trigger_idx = -1
    trigger_is_abbr = False

    for i, wc in enumerate(words_clean):
        if wc in trigger_tokens:
            trigger_idx = i
            trigger_is_abbr = False
            break

    if trigger_idx < 0:
        for i, wc in enumerate(words_clean):
            if wc in trigger_abbrs:
                trigger_idx = i
                trigger_is_abbr = True
                break

    if trigger_idx < 0:
        return ""

    # --- Rule B: Preceding words (1-2 alphabetic words before trigger) ---
    preceding: list[str] = []
    b_start = trigger_idx  # will move backward

    for j in range(trigger_idx - 1, max(trigger_idx - 3, -1), -1):
        wc = words_clean[j]
        if is_numeric_ish(wc) or wc in LINKING_WORDS:
            break  # hit the number zone — stop collecting preceding words
        if wc.isalpha():
            preceding.insert(0, words[j])
            b_start = j
            if len(preceding) >= 2:
                break
        else:
            break

    # --- Rule A: Greedy number anchor (scan backward from before Rule B) ---
    anchor: list[str] = []
    for j in range(b_start - 1, -1, -1):
        wc = words_clean[j]
        if is_numeric_ish(wc):
            anchor.insert(0, words[j])
        elif wc in LINKING_WORDS:
            # Only capture linking word if we already have a numeric word
            # adjacent (closer to the trigger).
            anchor.insert(0, words[j])
        elif len(wc) == 1 and wc.isalpha():
            # Single letter (A, B) — capture only if the next word toward
            # the trigger is numeric.
            if anchor and is_numeric_ish(words_clean[j + 1]):
                anchor.insert(0, words[j])
            else:
                break
        else:
            break

    # Strip trailing linking words from anchor (e.g., if anchor ends
    # with "to" but no number follows it within the anchor).
    while anchor and anchor[-1].strip(EDGE_PUNCT).lower() in LINKING_WORDS:
        anchor.pop()

    # --- Rule C: Trigger token (expanded if abbreviation) ---
    trigger_word = words[trigger_idx]
    if trigger_is_abbr:
        expanded = abbr_map.get(words_clean[trigger_idx], "")
        if expanded:
            trigger_word = expanded

    # --- Rule D: Conditional following word ---
    # Check the RAW word for leading delimiters (not the stripped
    # version) so that "(Vent-Axia" is correctly blocked.
    following: list[str] = []
    if trigger_idx + 1 < len(words):
        next_word = words[trigger_idx + 1]
        if (
            next_word
            and next_word[0] not in "([{-"
            and not next_word[0].isdigit()
        ):
            following.append(next_word)

    # Assemble parts: anchor + preceding + trigger + following.
    parts = anchor + preceding + [trigger_word] + following
    return " ".join(parts)


# ---------------------------------------------------------------------------
# Step 3: Post-processing
# ---------------------------------------------------------------------------

def post_process(extracted: str) -> str:
    """Strip edge punctuation, lowercase, collapse whitespace.

    Strips punctuation from each individual word so that internal
    commas/semicolons (e.g. "Road," → "road") are also removed.
    """
    text = extracted.lower()
    words = text.split()
    words = [w.strip(EDGE_PUNCT) for w in words]
    words = [w for w in words if w]  # drop empties
    return " ".join(words)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def is_numeric_ish(word: str) -> bool:
    """Check if a word is 'numeric-ish' for greedy anchor purposes.

    Returns True if the word contains at least one digit.
    Covers: "44", "44B", "67-70", "1-4", etc.
    """
    return any(c.isdigit() for c in word)
