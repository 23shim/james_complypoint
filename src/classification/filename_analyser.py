"""
Tier 2 — Filename analysis.

Scans the filename stem for document type signals using token,
abbreviation, and pattern matching. Also checks extension hints.
"""

from __future__ import annotations

from classification.matcher import match_extension_hint, match_type_signals
from classification.models import ClassificationConfig, Signal, SignalSource


def analyse_filename(
    filename_stem: str,
    extension: str,
    config: ClassificationConfig,
) -> list[Signal]:
    """Analyse a filename for document type signals.

    Args:
        filename_stem: Filename without extension.
        extension: Lowercase extension without dot.
        config: Merged classification config.

    Returns:
        List of type Signals from filename + extension.
    """
    signals: list[Signal] = []

    # Match filename stem against type definitions
    if filename_stem:
        type_signals = match_type_signals(
            filename_stem,
            config,
            source=SignalSource.FILENAME_TOKEN,
            depth=-1,
        )
        signals.extend(type_signals)

    # Check extension hints (low weight, additive)
    ext_signals = match_extension_hint(extension, config)
    signals.extend(ext_signals)

    return signals
