"""
Signal matching against dictionaries.

Pure functions: text in, signals out. No state, no side effects.
Each function checks one matching strategy (tokens, abbreviations,
patterns, fuzzy) and returns any signals found.
"""

from __future__ import annotations

from classification.models import (
    CategoryDefinition,
    ClassificationConfig,
    EntityDefinition,
    MatchMethod,
    Signal,
    SignalSource,
    TypeDefinition,
)
from classification.tokeniser import all_ngrams, tokenise


def match_type_signals(
    text: str,
    config: ClassificationConfig,
    source: SignalSource,
    depth: int = -1,
) -> list[Signal]:
    """Match text against all type definitions.

    Runs token, abbreviation, pattern, and fuzzy matching for each type.
    Returns all signals found (there may be multiple types matched).

    Args:
        text: Raw text to scan (folder segment or filename stem).
        config: Merged classification config.
        source: Signal source (FOLDER_TYPE or FILENAME_*).
        depth: Folder depth (0-indexed), -1 for filename.

    Returns:
        List of Signals for any types matched.
    """
    tokens = tokenise(text)
    if not tokens:
        return []

    phrases = all_ngrams(tokens)
    weights = config.weights.get("signal_weights", {})
    fuzzy_cfg = config.weights.get("fuzzy", {})
    signals = []

    # Build canonical token set for fuzzy matching (O(n), done once per text).
    # Maps each text token to its canonical dictionary form via the fuzzy index,
    # so later checks are simple set membership.
    token_set = set(tokens)
    canonical_tokens = set(tokens)
    if config.fuzzy_index:
        for tok in tokens:
            canonical = config.fuzzy_index.get(tok)
            if canonical is not None:
                canonical_tokens.add(canonical)

    for type_def in config.types.values():
        signal = _match_single_type(
            text, tokens, phrases, type_def, source, depth, weights,
            fuzzy_cfg=fuzzy_cfg,
            token_set=token_set,
            canonical_tokens=canonical_tokens,
        )
        if signal:
            signals.append(signal)

    return signals


def match_category_signals(
    text: str,
    config: ClassificationConfig,
    depth: int = 0,
) -> list[Signal]:
    """Match text against all category definitions.

    Checks both multi-word signals and single-token aliases.

    Args:
        text: Raw text to scan (folder segment).
        config: Merged classification config.
        depth: Folder depth where this segment sits.

    Returns:
        List of category Signals found.
    """
    tokens = tokenise(text)
    if not tokens:
        return []

    phrases = all_ngrams(tokens)
    weights = config.weights.get("signal_weights", {})
    base_weight = weights.get("folder_category", 0.30)
    signals = []

    for cat_def in config.categories.values():
        signal = _match_single_category(
            text, tokens, phrases, cat_def, depth, base_weight
        )
        if signal:
            signals.append(signal)

    return signals


def match_extension_hint(
    extension: str,
    config: ClassificationConfig,
) -> list[Signal]:
    """Check if a file extension hints at a type.

    Args:
        extension: Lowercase extension without dot (e.g. "msg").
        config: Merged classification config.

    Returns:
        List of extension hint Signals (usually 0 or 1).
    """
    if not extension:
        return []

    weights = config.weights.get("signal_weights", {})
    base_weight = weights.get("extension_hint", 0.15)
    signals = []

    for type_def in config.types.values():
        if extension in type_def.extensions:
            signals.append(Signal(
                source=SignalSource.EXTENSION_HINT,
                label=type_def.name,
                match_term=f".{extension}",
                match_method=MatchMethod.TOKEN,
                base_weight=base_weight,
                depth=-1,
                text=extension,
                origin_layer=type_def.origin_layer,
            ))

    return signals


def _match_single_type(
    text: str,
    tokens: list[str],
    phrases: list[str],
    type_def: TypeDefinition,
    source: SignalSource,
    depth: int,
    weights: dict,
    *,
    fuzzy_cfg: dict | None = None,
    token_set: set[str] | None = None,
    canonical_tokens: set[str] | None = None,
) -> Signal | None:
    """Try to match a single type definition against text.

    Priority: token match > pattern match > abbreviation match > fuzzy match.
    Returns the highest-priority match found, or None.
    """
    # 1. Token match (exact multi-word or single-word)
    for token in type_def.tokens:
        if token in phrases:
            signal_source = source
            if source == SignalSource.FOLDER_TYPE:
                weight_key = "folder_type"
            else:
                weight_key = "filename_token"
            return Signal(
                source=signal_source,
                label=type_def.name,
                match_term=token,
                match_method=MatchMethod.TOKEN,
                base_weight=weights.get(weight_key, 0.60),
                depth=depth,
                text=text,
                origin_layer=type_def.origin_layer,
            )

    # 2. Pattern match (regex)
    for pattern in type_def.compiled_patterns:
        if pattern.search(text):
            if source == SignalSource.FOLDER_TYPE:
                weight_key = "folder_type"
            else:
                weight_key = "filename_pattern"
            return Signal(
                source=source,
                label=type_def.name,
                match_term=pattern.pattern,
                match_method=MatchMethod.PATTERN,
                base_weight=weights.get(weight_key, 0.60),
                depth=depth,
                text=text,
                origin_layer=type_def.origin_layer,
            )

    # 3. Abbreviation match (whole token only)
    for abbr in type_def.abbreviations:
        if abbr in tokens:
            if source == SignalSource.FOLDER_TYPE:
                weight_key = "folder_type"
            else:
                weight_key = "filename_abbreviation"
            return Signal(
                source=source,
                label=type_def.name,
                match_term=abbr,
                match_method=MatchMethod.ABBREVIATION,
                base_weight=weights.get(weight_key, 0.50),
                depth=depth,
                text=text,
                origin_layer=type_def.origin_layer,
            )

    # 4. Fuzzy match (pre-computed edit-distance-1 variants)
    if canonical_tokens and token_set and fuzzy_cfg:
        penalty = fuzzy_cfg.get("fuzzy_penalty", 0.15)
        for token_phrase in type_def.tokens:
            words = token_phrase.split()
            # All component words must appear in canonical_tokens,
            # but at least one must be a fuzzy substitution (not in
            # the original token_set) — otherwise exact match above
            # would have already caught it.
            if all(w in canonical_tokens for w in words) and not all(
                w in token_set for w in words
            ):
                if source == SignalSource.FOLDER_TYPE:
                    weight_key = "folder_type"
                else:
                    weight_key = "filename_token"
                base = weights.get(weight_key, 0.60)
                return Signal(
                    source=source,
                    label=type_def.name,
                    match_term=token_phrase,
                    match_method=MatchMethod.FUZZY_TOKEN,
                    base_weight=round(base * (1 - penalty), 4),
                    depth=depth,
                    text=text,
                    origin_layer=type_def.origin_layer,
                )

    return None


def _match_single_category(
    text: str,
    tokens: list[str],
    phrases: list[str],
    cat_def: CategoryDefinition,
    depth: int,
    base_weight: float,
) -> Signal | None:
    """Try to match a single category definition against text."""
    # Check multi-word signals first (more specific)
    for signal_term in cat_def.signals:
        if signal_term in phrases:
            return Signal(
                source=SignalSource.FOLDER_CATEGORY,
                label=cat_def.name,
                match_term=signal_term,
                match_method=MatchMethod.TOKEN,
                base_weight=base_weight,
                depth=depth,
                text=text,
            )

    # Check aliases (single-token abbreviations)
    for alias in cat_def.aliases:
        if alias in tokens:
            return Signal(
                source=SignalSource.FOLDER_CATEGORY,
                label=cat_def.name,
                match_term=alias,
                match_method=MatchMethod.ABBREVIATION,
                base_weight=base_weight,
                depth=depth,
                text=text,
            )

    return None


# Entity extraction defaults — overridden by weights.yaml "entity_extraction" section.
_ENTITY_DEFAULTS = {
    "token_confidence": 0.80,
    "pattern_confidence": 0.65,
    "abbreviation_confidence": 0.50,
}


def match_entity_signals(
    text: str,
    config: ClassificationConfig,
    depth: int = 0,
    is_organisational: bool = False,
) -> list[Signal]:
    """Match text against all entity definitions.

    Same token > pattern > abbreviation priority as type matching.
    When ``is_organisational`` is True (segment also matched a type or
    category), only pattern matches are allowed — bare token/abbreviation
    matches would be false positives (e.g. "Street Lighting" → address).

    Args:
        text: Raw text to scan (folder segment).
        config: Merged classification config.
        depth: Folder depth where this segment sits.
        is_organisational: True if segment matched a type or category.

    Returns:
        List of entity Signals found.
    """
    if not config.entities:
        return []

    tokens = tokenise(text)
    if not tokens:
        return []

    phrases = all_ngrams(tokens)
    ew_overrides = config.weights.get("entity_extraction", {})
    ew = {k: ew_overrides.get(k, v) for k, v in _ENTITY_DEFAULTS.items()}
    signals = []

    for entity_def in config.entities.values():
        signal = _match_single_entity(
            text, tokens, phrases, entity_def, depth, ew, is_organisational,
        )
        if signal:
            signals.append(signal)

    return signals


def _match_single_entity(
    text: str,
    tokens: list[str],
    phrases: list[str],
    entity_def: EntityDefinition,
    depth: int,
    ew: dict,
    is_organisational: bool,
) -> Signal | None:
    """Try to match a single entity definition against text.

    Same priority as types: token > pattern > abbreviation.
    On organisational folders only pattern matches are allowed.
    """
    if is_organisational:
        for pattern in entity_def.compiled_patterns:
            if pattern.search(text):
                return Signal(
                    source=SignalSource.FOLDER_ENTITY,
                    label=entity_def.name,
                    match_term=pattern.pattern,
                    match_method=MatchMethod.PATTERN,
                    base_weight=ew["pattern_confidence"],
                    depth=depth,
                    text=text,
                )
        return None

    # 1. Token match (exact multi-word or single-word)
    for token in entity_def.tokens:
        if token in phrases:
            return Signal(
                source=SignalSource.FOLDER_ENTITY,
                label=entity_def.name,
                match_term=token,
                match_method=MatchMethod.TOKEN,
                base_weight=ew["token_confidence"],
                depth=depth,
                text=text,
            )

    # 2. Pattern match (regex)
    for pattern in entity_def.compiled_patterns:
        if pattern.search(text):
            return Signal(
                source=SignalSource.FOLDER_ENTITY,
                label=entity_def.name,
                match_term=pattern.pattern,
                match_method=MatchMethod.PATTERN,
                base_weight=ew["pattern_confidence"],
                depth=depth,
                text=text,
            )

    # 3. Abbreviation match (whole token only)
    for abbr in entity_def.abbreviations:
        if abbr in tokens:
            return Signal(
                source=SignalSource.FOLDER_ENTITY,
                label=entity_def.name,
                match_term=abbr,
                match_method=MatchMethod.ABBREVIATION,
                base_weight=ew["abbreviation_confidence"],
                depth=depth,
                text=text,
            )

    return None
