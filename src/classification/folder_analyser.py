"""
Tier 1 — Folder path analysis.

Scans each segment in the path for category, type, and entity signals.
Returns all signals with depth metadata so the scorer can
apply deepest-wins logic for categories and entities.
"""

from __future__ import annotations

from classification.matcher import (
    match_category_signals,
    match_entity_signals,
    match_type_signals,
)
from classification.models import ClassificationConfig, Signal, SignalSource


def analyse_folders(
    segments: list[str],
    config: ClassificationConfig,
) -> list[Signal]:
    """Analyse folder segments for classification signals.

    Iterates each segment and checks against category, type, and
    entity dictionaries. Returns all signals found — the scorer
    decides which to use.

    Entity matching uses the type/category match result to determine
    whether the segment is organisational. On organisational folders,
    only pattern entity matches are allowed (prevents false positives
    like "Street Lighting" → address).

    Args:
        segments: Ordered list of folder names (from path_parser).
        config: Merged classification config.

    Returns:
        List of Signals with depth metadata.
    """
    signals: list[Signal] = []

    for depth, segment in enumerate(segments):
        # Check for category signals (Finance, Legal, Construction, etc.)
        cat_signals = match_category_signals(segment, config, depth=depth)
        signals.extend(cat_signals)

        # Check for type signals (folder named "Invoices", "Snagging", etc.)
        type_signals = match_type_signals(
            segment, config, source=SignalSource.FOLDER_TYPE, depth=depth
        )
        signals.extend(type_signals)

        # Check for entity signals (Plot 12, Lenham Road, etc.)
        # Organisational if this segment matched a type or category —
        # restricts entity matching to pattern-only (no bare tokens).
        is_organisational = bool(cat_signals or type_signals)
        entity_signals = match_entity_signals(
            segment, config, depth=depth, is_organisational=is_organisational,
        )
        signals.extend(entity_signals)

    return signals
