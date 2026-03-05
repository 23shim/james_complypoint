"""
Plot extraction from noisy folder/file name segments.

Uses a "Keyword-Forward" method to extract clean, standardised
plot reference substrings from raw entity values like folder names.
The trigger keyword comes BEFORE the number (Plot 12, Block A)
— the mirror of addresses where the trigger comes AFTER.

Pipeline:
  0. Purge dates (reused from address_extractor)
  1. Find trigger keyword (from entity tokens + abbreviations +
     keywords derived from compiled patterns)
  2. Greedy forward capture (numbers, single letters, linking words)
  3. Descending-number heuristic (trim false ranges like "66 - 42")
  4. Pattern fallback (if no keyword trigger found)
  5. Post-process (lowercase, strip punctuation, expand abbreviation)

Runs AFTER plot validation and BEFORE entity clustering.
The extracted plot ref is used for clustering similarity instead
of the raw folder segment name.
"""

from __future__ import annotations

import re

from classification.address_extractor import (
    EDGE_PUNCT,
    LINKING_WORDS,
    WHITESPACE_RE,
    is_numeric_ish,
    post_process,
    purge_dates,
)
from classification.models import EntityDefinition


def extract_plot(
    raw_value: str,
    entity_def: EntityDefinition,
    address_entity_def: EntityDefinition | None = None,
) -> str:
    """Extract a clean, standardised plot reference from a raw segment name.

    Args:
        raw_value: The raw folder segment name
            (e.g. "Plot 16-30 Colemans Close ashford,").
        entity_def: The plot EntityDefinition
            (tokens, abbreviations, patterns, abbreviation_map).
        address_entity_def: Optional address EntityDefinition used for
            address-aware range trimming.  When provided, street-suffix
            tokens are checked after the captured range to detect false
            ranges where the second number is actually a house number.

    Returns:
        Extracted and standardised plot reference string (lowercase,
        trigger abbreviation expanded), or empty string if no
        plot signal found.
    """
    if not raw_value or not raw_value.strip():
        return ""

    text = raw_value.strip()

    # Step 0: Purge dates
    text = purge_dates(text)
    if not text:
        return ""

    # Build address trigger token set for address-aware trimming
    address_tokens: set[str] = set()
    if address_entity_def:
        for t in address_entity_def.tokens:
            address_tokens.add(t.lower())

    # Step 1-3: Keyword-forward extraction
    extracted = _keyword_forward_extract(text, entity_def, address_tokens)

    if extracted:
        return post_process(extracted)

    # Step 4: Pattern fallback — no keyword trigger found, try compiled patterns
    fallback = _pattern_fallback(text, entity_def)
    if fallback:
        return post_process(fallback)

    return ""


# ---------------------------------------------------------------------------
# Trigger keyword resolution
# ---------------------------------------------------------------------------

def _build_trigger_set(entity_def: EntityDefinition) -> set[str]:
    """Build the set of trigger keywords from tokens + abbreviations + patterns.

    Derives additional keywords from compiled pattern strings by extracting
    the literal prefix (e.g. "block" from r"\\bblock[-_ ]?...").
    """
    triggers: set[str] = set()

    # Direct tokens and abbreviations
    for t in entity_def.tokens:
        triggers.add(t.lower())
    for a in entity_def.abbreviations:
        triggers.add(a.lower())

    # Derive keywords from compiled patterns
    for pat in entity_def.compiled_patterns:
        m = re.match(r"\\b(\w+)", pat.pattern)
        if m:
            triggers.add(m.group(1).lower())

    return triggers


# ---------------------------------------------------------------------------
# Step 1-3: Keyword-Forward extraction
# ---------------------------------------------------------------------------

def _keyword_forward_extract(
    text: str,
    entity_def: EntityDefinition,
    address_tokens: set[str] | None = None,
) -> str:
    """Core Keyword-Forward extraction.

    Finds the first trigger keyword, then greedily captures forward:
    numbers, single letters, linking words (&, to, and), standalone "-".
    Stops at multi-letter alphabetic words, parenthetical noise, or
    another trigger keyword.

    Applies descending-number heuristic to trim false ranges, then
    address-aware trimming to catch cases like "Plot 10 - 12 Pippin Court"
    where the number after the dash is a house number, not a range end.
    """
    words = text.split()
    if not words:
        return ""

    # Clean versions for matching (stripped of edge punctuation, lowered).
    words_clean = [w.strip(EDGE_PUNCT).lower() for w in words]

    trigger_set = _build_trigger_set(entity_def)
    abbr_map = entity_def.abbreviation_map

    # Find first trigger keyword (check plurals by stripping trailing 's')
    trigger_idx = -1
    for i, wc in enumerate(words_clean):
        if wc in trigger_set:
            trigger_idx = i
            break
        # Check plural form
        if wc.endswith("s") and len(wc) > 1 and wc[:-1] in trigger_set:
            trigger_idx = i
            break

    if trigger_idx < 0:
        return ""

    # The trigger word (expanded if abbreviation)
    trigger_clean = words_clean[trigger_idx]
    # For plural forms, use the raw word as-is (preserves "Plots", "Blocks")
    trigger_word = words[trigger_idx]

    # Expand abbreviation if applicable (use singular form for lookup)
    singular = trigger_clean[:-1] if (
        trigger_clean.endswith("s")
        and len(trigger_clean) > 1
        and trigger_clean not in trigger_set
        and trigger_clean[:-1] in trigger_set
    ) else trigger_clean

    expanded = abbr_map.get(singular, "")
    if expanded:
        # Preserve plural if the original was plural
        if trigger_clean.endswith("s") and trigger_clean != singular:
            trigger_word = expanded + "s"
        else:
            trigger_word = expanded

    # --- Greedy forward capture ---
    captured: list[str] = []
    stop_idx = len(words)  # default: all words after trigger consumed
    for i in range(trigger_idx + 1, len(words)):
        raw_word = words[i]
        wc = words_clean[i]

        # Stop if word starts with "("
        if raw_word and raw_word[0] in "([{":
            stop_idx = i
            break

        # Stop if word is another trigger keyword
        wc_singular = wc[:-1] if (
            wc.endswith("s") and len(wc) > 1 and wc[:-1] in trigger_set
        ) else wc
        if wc in trigger_set or wc_singular in trigger_set:
            stop_idx = i
            break

        # Capture if word contains digits (after stripping edge punctuation)
        if is_numeric_ish(wc):
            captured.append(raw_word)
            continue

        # Capture standalone "-" as range connector
        if wc == "-" or raw_word.strip() == "-":
            captured.append(raw_word)
            continue

        # Capture single letter (A-Z) — for Block A, Block B
        if len(wc) == 1 and wc.isalpha():
            captured.append(raw_word)
            continue

        # Capture linking words: &, to, and
        if wc in LINKING_WORDS:
            captured.append(raw_word)
            continue

        # Multi-letter alphabetic word (not linking) — stop
        stop_idx = i
        break

    # Strip trailing linking words and standalone dashes from capture
    while captured:
        tail = captured[-1].strip(EDGE_PUNCT).lower()
        if tail in LINKING_WORDS or tail == "-" or captured[-1].strip() == "-":
            captured.pop()
        else:
            break

    # Apply descending-number heuristic
    captured = _apply_descending_heuristic(captured)

    # Apply address-aware trim (catches ascending false ranges
    # like "Plot 10 - 12 Pippin Court" where 12 is a house number)
    if address_tokens and captured:
        remaining = words[stop_idx:]
        captured = _apply_address_aware_trim(
            captured, remaining, address_tokens,
        )

    # A trigger keyword without any captured identifier is not a plot reference
    # (e.g. "Apartment floor finishes" — just the word "apartment" in context).
    if not captured:
        return ""

    # Assemble: trigger + captured
    parts = [trigger_word] + captured
    return " ".join(parts)


# ---------------------------------------------------------------------------
# Descending-number heuristic
# ---------------------------------------------------------------------------

def _apply_descending_heuristic(captured: list[str]) -> list[str]:
    """Trim false ranges where first number > second after a standalone '-'.

    Scans for the pattern: ... number(s) - number(s) ...
    If the last number before '-' is greater than the first number after '-',
    trim from the '-' onward (it's a separator, not a range).

    Examples:
        ["66", "-", "42"] → ["66"]  (66 > 42 → trim)
        ["1", "-", "5"] → ["1", "-", "5"]  (1 < 5 → keep)
        ["76", "&", "77", "-", "28"] → ["76", "&", "77"]  (77 > 28 → trim)
    """
    if not captured:
        return captured

    # Find standalone "-" positions
    dash_indices = []
    for i, word in enumerate(captured):
        cleaned = word.strip(EDGE_PUNCT).lower()
        if cleaned == "-" or word.strip() == "-":
            dash_indices.append(i)

    for dash_idx in dash_indices:
        # Find last number before the dash
        last_num_before = None
        for j in range(dash_idx - 1, -1, -1):
            num = _extract_leading_number(captured[j])
            if num is not None:
                last_num_before = num
                break

        # Find first number after the dash
        first_num_after = None
        for j in range(dash_idx + 1, len(captured)):
            num = _extract_leading_number(captured[j])
            if num is not None:
                first_num_after = num
                break

        if (
            last_num_before is not None
            and first_num_after is not None
            and last_num_before > first_num_after
        ):
            # Trim from the dash onward
            return captured[:dash_idx]

    return captured


def _extract_leading_number(word: str) -> int | None:
    """Extract the leading numeric portion of a word, or None."""
    cleaned = word.strip(EDGE_PUNCT)
    m = re.match(r"(\d+)", cleaned)
    if m:
        return int(m.group(1))
    return None


# ---------------------------------------------------------------------------
# Address-aware range trimming
# ---------------------------------------------------------------------------

# Maximum gap between two numbers in a hyphenated token (e.g. "10-103")
# before we consider the second number a house number rather than a
# range end.  Standalone dashes ("10 - 12") are always trimmed when an
# address suffix follows because the spaced dash is almost always a
# separator in housing folder names.
_ADDRESS_RANGE_GAP_THRESHOLD = 30


def _apply_address_aware_trim(
    captured: list[str],
    remaining_words: list[str],
    address_tokens: set[str],
) -> list[str]:
    """Trim false ranges where the number after '-' is a house number.

    When the words following the captured portion contain a street suffix
    (road, court, lane, etc.), a trailing range is likely a separator
    between plot number and address rather than a genuine plot range.

    Rules:
      - Standalone dash  (``10 - 12``): trim from the last dash onward.
        The spaced dash in housing folder names is overwhelmingly a
        separator (e.g. "Plot 10 - 12 Pippin Court" = Plot 10 *at*
        12 Pippin Court).
      - Hyphenated number (``10-103``): split only when the numeric gap
        exceeds ``_ADDRESS_RANGE_GAP_THRESHOLD`` (default 30) so that
        genuine compact ranges like "Plot 16-30 Colemans Close" survive.

    Examples:
        captured=["10", "-", "12"], remaining=["Pippin", "Court"]
            → ["10"]  (standalone dash, "court" is address suffix)
        captured=["10-103"], remaining=["Manston", "Road"]
            → ["10"]  (gap 93 > 30, "road" is address suffix)
        captured=["16-30"], remaining=["Colemans", "Close"]
            → ["16-30"]  (gap 14 ≤ 30, kept as genuine range)
        captured=["10", "-", "12"], remaining=["Something"]
            → ["10", "-", "12"]  (no address suffix → unchanged)
    """
    if not captured or not remaining_words:
        return captured

    # Check if any of the first 3 remaining words is a street suffix.
    # Strip dashes as well as standard edge punctuation — folder names
    # often use dashes as separators (e.g. "Road- Callie").
    _strip_chars = EDGE_PUNCT + "-\u2013\u2014"
    remaining_clean = [
        w.strip(_strip_chars).lower() for w in remaining_words[:3]
    ]
    if not any(wc in address_tokens for wc in remaining_clean):
        return captured

    # Case 1: Standalone dash in captured → trim from last dash
    for i in range(len(captured) - 1, -1, -1):
        cleaned = captured[i].strip(EDGE_PUNCT).lower()
        if cleaned == "-" or captured[i].strip() == "-":
            trimmed = captured[:i]
            # Strip any trailing linking words left over
            while trimmed:
                tail = trimmed[-1].strip(EDGE_PUNCT).lower()
                if tail in LINKING_WORDS:
                    trimmed.pop()
                else:
                    break
            return trimmed

    # Case 2: Last captured token is a hyphenated number (e.g. "10-103")
    last_clean = captured[-1].strip(EDGE_PUNCT)
    if "-" in last_clean:
        parts = last_clean.split("-", 1)
        first_str, second_str = parts[0].strip(), parts[1].strip()
        first_num = _extract_leading_number(first_str)
        second_num = _extract_leading_number(second_str)
        if (
            first_num is not None
            and second_num is not None
            and second_num - first_num > _ADDRESS_RANGE_GAP_THRESHOLD
        ):
            # Replace the last token with just the first part
            return captured[:-1] + [first_str]

    return captured


# ---------------------------------------------------------------------------
# Step 4: Pattern fallback
# ---------------------------------------------------------------------------

def _pattern_fallback(
    text: str,
    entity_def: EntityDefinition,
) -> str:
    """Try compiled patterns when no keyword trigger was found.

    Extracts just the matched portion (e.g. "P239" from "P239-Lv0").
    """
    text_lower = text.lower()
    for pat in entity_def.compiled_patterns:
        m = pat.search(text_lower)
        if m:
            return m.group(0)
    return ""
