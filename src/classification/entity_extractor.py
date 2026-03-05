"""
Entity extraction from folder segments.

Thin wrapper around matcher.match_entity_signals() for backward
compatibility.  The primary entity matching now flows through
folder_analyser → scorer as part of the unified signal pipeline.

This module is retained for:
- scheme_detector.py (calls extract_entities_with_confidence directly)
- scripts that import from here
- find_all_entity_matches() for raw_* columns
"""

from __future__ import annotations

from dataclasses import dataclass

from classification.matcher import match_entity_signals
from classification.models import ClassificationConfig


@dataclass(frozen=True)
class EntityMatch:
    """Result of matching an entity in a segment."""
    value: str           # The matched segment text (e.g. "Plot 12")
    confidence: float    # Base confidence from match method
    depth: int           # Folder depth where found


def extract_entities(
    segments: list[str],
    config: ClassificationConfig,
) -> dict[str, str]:
    """Extract entities from folder segments (deepest wins).

    Args:
        segments: Ordered list of folder names (from path_parser).
        config: Merged classification config (must have entities loaded).

    Returns:
        Dict mapping entity type name to the matched segment value.
    """
    results = extract_entities_with_confidence(segments, config)
    return {name: match.value for name, match in results.items()}


def extract_entities_with_confidence(
    segments: list[str],
    config: ClassificationConfig,
) -> dict[str, EntityMatch]:
    """Extract entities with confidence scores (deepest wins).

    Delegates matching to matcher.match_entity_signals() and applies
    deepest-wins resolution.

    Returns:
        Dict mapping entity type name to EntityMatch.
    """
    if not config.entities:
        return {}

    matches: dict[str, EntityMatch] = {}

    for depth, segment in enumerate(segments):
        # Determine if this is an organisational folder by checking
        # for type/category matches (same logic as folder_analyser).
        from classification.matcher import (
            match_category_signals,
            match_type_signals,
        )
        from classification.models import SignalSource

        is_organisational = bool(
            match_type_signals(
                segment, config,
                source=SignalSource.FOLDER_TYPE, depth=0,
            )
            or match_category_signals(segment, config, depth=0)
        )

        entity_signals = match_entity_signals(
            segment, config, depth=depth,
            is_organisational=is_organisational,
        )

        for signal in entity_signals:
            entity_name = signal.label
            # Deepest wins
            if entity_name not in matches or depth > matches[entity_name].depth:
                matches[entity_name] = EntityMatch(
                    value=signal.text,
                    confidence=signal.base_weight,
                    depth=depth,
                )

    return matches


def find_all_entity_matches(
    segments: list[str],
    entity_name: str,
    config: ClassificationConfig,
) -> list[EntityMatch]:
    """Find ALL segments matching a specific entity type.

    Returns every matching segment with its confidence and depth,
    ordered deepest-first.

    Args:
        segments: Ordered folder segments from path_parser.
        entity_name: The entity type to search for (e.g. "plot").
        config: Merged classification config.

    Returns:
        List of EntityMatch for every matching segment, ordered
        deepest-first.
    """
    if not config.entities or entity_name not in config.entities:
        return []

    matches: list[EntityMatch] = []

    for depth, segment in enumerate(segments):
        from classification.matcher import (
            match_category_signals,
            match_type_signals,
        )
        from classification.models import SignalSource

        is_organisational = bool(
            match_type_signals(
                segment, config,
                source=SignalSource.FOLDER_TYPE, depth=0,
            )
            or match_category_signals(segment, config, depth=0)
        )

        entity_signals = match_entity_signals(
            segment, config, depth=depth,
            is_organisational=is_organisational,
        )

        for signal in entity_signals:
            if signal.label == entity_name:
                matches.append(EntityMatch(
                    value=signal.text,
                    confidence=signal.base_weight,
                    depth=depth,
                ))

    # Sort deepest-first
    matches.sort(key=lambda m: -m.depth)
    return matches
