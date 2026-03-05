"""
Confidence scoring and conflict resolution.

Takes raw signals from folder + filename analysis and produces
a final classification with confidence scores and reasoning.

Scoring philosophy:
  - Base weights directly represent confidence (no secondary discounts).
  - Filename signals are trusted most (the name IS the document).
  - Folder signals are moderate (describes contents broadly).
  - Modifiers are small nudges for structural agreement/disagreement.
  - Overall confidence = type confidence.
    Category is informational — it doesn't gatekeep readiness.
"""

from __future__ import annotations

from classification.models import (
    ClassificationResult,
    Signal,
    SignalSource,
)

# Signal sources that indicate a type from a folder
_FOLDER_TYPE_SOURCES = {SignalSource.FOLDER_TYPE}

# Entity signals from folder segments
_ENTITY_SOURCES = {SignalSource.FOLDER_ENTITY}

# Signal sources that indicate a type from a filename
_FILENAME_TYPE_SOURCES = {
    SignalSource.FILENAME_TOKEN,
    SignalSource.FILENAME_ABBREVIATION,
    SignalSource.FILENAME_PATTERN,
}

# All type sources (excluding extension hints which are separate)
_TYPE_SOURCES = _FOLDER_TYPE_SOURCES | _FILENAME_TYPE_SOURCES


def score(
    signals: list[Signal],
    weights: dict,
    type_domains: dict[str, str] | None = None,
) -> ClassificationResult:
    """Score classification signals and resolve conflicts.

    Args:
        signals: All signals from folder + filename analysis.
        weights: Weights config dict (from weights.yaml).
        type_domains: Optional mapping of type name -> domain (e.g. "Sales",
            "universal"). Used for category-domain reinforcement/conflict.

    Returns:
        ClassificationResult with type, category, confidence, and reasoning.
    """
    if type_domains is None:
        type_domains = {}
    reasoning: list[str] = []
    reinf = weights.get("reinforcement", {})
    conflict = weights.get("conflict", {})
    thresholds = weights.get("thresholds", {})
    readiness = weights.get("readiness", {})

    # --- Separate signals by kind ---
    category_signals = [
        s for s in signals if s.source == SignalSource.FOLDER_CATEGORY
    ]
    type_signals = [s for s in signals if s.source in _TYPE_SOURCES]
    extension_signals = [
        s for s in signals if s.source == SignalSource.EXTENSION_HINT
    ]
    entity_signals = [s for s in signals if s.source in _ENTITY_SOURCES]

    # Log what was found
    for s in signals:
        source_label = _source_label(s)
        reasoning.append(
            f"{source_label} '{s.text}' -> {s.label} "
            f"({s.match_method.value} match: '{s.match_term}', w={s.base_weight:.2f})"
        )

    # --- Resolve category (deepest wins) ---
    inferred_category, category_confidence = _resolve_category(category_signals)
    if inferred_category != "Unknown":
        reasoning.append(
            f"category: {inferred_category} (deepest folder signal)"
        )

    # --- Resolve type ---
    folder_types = [s for s in type_signals if s.source in _FOLDER_TYPE_SOURCES]
    filename_types = [s for s in type_signals if s.source in _FILENAME_TYPE_SOURCES]

    generic_types = set(weights.get("generic_types", []))
    generic_discount = weights.get("generic_suppression_discount", 0.85)
    inferred_type, type_confidence = _resolve_type(
        folder_types, filename_types, extension_signals, reasoning,
        generic_types, generic_discount,
    )

    # --- Resolve secondary type (runner-up) ---
    secondary_type, secondary_type_confidence = _resolve_secondary_type(
        inferred_type, folder_types, filename_types, extension_signals,
        generic_types,
    )

    # --- Apply reinforcement/conflict modifiers ---
    if inferred_type != "Unknown":
        type_confidence = _apply_modifiers(
            inferred_type,
            type_confidence,
            inferred_category,
            folder_types,
            filename_types,
            type_domains,
            reinf,
            conflict,
            weights,
            reasoning,
        )

    # --- Clamp ---
    type_confidence = max(0.0, min(1.0, type_confidence))
    category_confidence = max(0.0, min(1.0, category_confidence))

    # --- Apply minimum threshold ---
    min_conf = thresholds.get("minimum_confidence", 0.15)
    if inferred_type != "Unknown" and type_confidence < min_conf:
        reasoning.append(
            f"type confidence {type_confidence:.2f} < minimum {min_conf} -> Unknown"
        )
        inferred_type = "Unknown"
        type_confidence = 0.0

    # --- Overall confidence = type confidence ---
    # Category is informational only — it doesn't affect readiness.
    overall = type_confidence

    # --- Ambiguity detection ---
    if (
        inferred_type != "Unknown"
        and secondary_type
        and secondary_type_confidence > 0
    ):
        gap = type_confidence - secondary_type_confidence
        if gap < 0.15:
            reasoning.append(
                f"ambiguous: runner-up '{secondary_type}' "
                f"({secondary_type_confidence:.2f}) is close "
                f"(gap {gap:.2f})"
            )

    # --- Confidence band & readiness ---
    high_threshold = readiness.get("high_threshold", 0.60)
    medium_threshold = readiness.get("medium_threshold", 0.35)
    unknown_status = readiness.get("unknown_status", "Not Ready")

    if inferred_type == "Unknown":
        confidence_band = "Low"
        readiness_status = unknown_status
    elif overall >= high_threshold:
        confidence_band = "High"
        readiness_status = "Ready"
    elif overall >= medium_threshold:
        confidence_band = "Medium"
        readiness_status = "Review"
    else:
        confidence_band = "Low"
        readiness_status = "Not Ready"

    # --- Resolve entities (deepest wins, same as categories) ---
    entities, entity_confidences, entity_depths = _resolve_entities(
        entity_signals,
    )
    for ename, evalue in entities.items():
        reasoning.append(
            f"entity: {ename}='{evalue}' "
            f"(depth={entity_depths[ename]}, conf={entity_confidences[ename]:.2f})"
        )

    reasoning.append(
        f"final: type={inferred_type}({type_confidence:.2f}), "
        f"category={inferred_category}({category_confidence:.2f}), "
        f"overall={overall:.2f}, band={confidence_band}, status={readiness_status}"
    )

    return ClassificationResult(
        inferred_type=inferred_type,
        type_confidence=round(type_confidence, 3),
        inferred_category=inferred_category,
        category_confidence=round(category_confidence, 3),
        overall_confidence=round(overall, 3),
        confidence_band=confidence_band,
        readiness_status=readiness_status,
        reasoning_trace=reasoning,
        secondary_type=secondary_type,
        secondary_type_confidence=round(secondary_type_confidence, 3),
        entities=entities,
        entity_confidences=entity_confidences,
        entity_depths=entity_depths,
    )


def _resolve_category(
    category_signals: list[Signal],
) -> tuple[str, float]:
    """Pick the best category — deepest folder signal wins."""
    if not category_signals:
        return "Unknown", 0.0

    # Sort by depth descending, then weight descending
    best = max(category_signals, key=lambda s: (s.depth, s.base_weight))
    return best.label, best.base_weight


def _resolve_type(
    folder_types: list[Signal],
    filename_types: list[Signal],
    extension_signals: list[Signal],
    reasoning: list[str],
    generic_types: set[str] | None = None,
    generic_discount: float = 0.85,
) -> tuple[str, float]:
    """Pick the best type. Filename > folder > extension-only.

    Generic types (Photo, Report) are low-information — they describe
    the file medium or broad function, not its specific purpose. When
    the deepest type is generic but a specific (non-generic) ancestor
    exists, the ancestor is promoted with a confidence discount.

    When ONLY generic types exist in the path, the deepest generic wins
    normally — no suppression occurs.

    No secondary discounts are applied to folder signals — the lower
    base weight (0.50 vs 0.85) already captures the uncertainty.
    """
    if generic_types is None:
        generic_types = set()

    if filename_types:
        best = max(filename_types, key=lambda s: (s.base_weight, s.origin_layer))

        # If the filename match is for a generic type, check whether a
        # specific folder context should override it.
        if best.label in generic_types and folder_types:
            non_generic = [s for s in folder_types if s.label not in generic_types]
            if non_generic:
                specific = max(non_generic, key=lambda s: (s.depth, s.origin_layer, s.base_weight))
                discounted = specific.base_weight * generic_discount
                reasoning.append(
                    f"type: {specific.label} (generic filename type '{best.label}' "
                    f"overridden by folder context at depth {specific.depth}, "
                    f"discounted {specific.base_weight:.2f} -> {discounted:.2f})"
                )
                return specific.label, discounted

        # Normal: filename type is PRIMARY
        reasoning.append(f"type: {best.label} (filename signal, primary)")
        return best.label, best.base_weight

    if folder_types:
        best = max(folder_types, key=lambda s: (s.depth, s.origin_layer, s.base_weight))

        # If the deepest folder type is generic, try to promote a
        # specific ancestor. When only generics exist, the deepest
        # generic wins — no suppression.
        if best.label in generic_types:
            non_generic = [s for s in folder_types if s.label not in generic_types]
            if non_generic:
                specific = max(non_generic, key=lambda s: (s.depth, s.origin_layer, s.base_weight))
                discounted = specific.base_weight * generic_discount
                reasoning.append(
                    f"type: {specific.label} (generic '{best.label}' at depth "
                    f"{best.depth} suppressed, promoted specific ancestor at depth "
                    f"{specific.depth}, discounted {specific.base_weight:.2f} -> "
                    f"{discounted:.2f})"
                )
                return specific.label, discounted

        # Folder type — no additional discount (base weight is already moderate)
        reasoning.append(f"type: {best.label} (folder signal only)")
        return best.label, best.base_weight

    if extension_signals:
        # Extension hint only — very low confidence
        best = max(extension_signals, key=lambda s: (s.base_weight, s.origin_layer))
        reasoning.append(f"type: {best.label} (extension hint only)")
        return best.label, best.base_weight

    return "Unknown", 0.0


def _resolve_secondary_type(
    primary_type: str,
    folder_types: list[Signal],
    filename_types: list[Signal],
    extension_signals: list[Signal],
    generic_types: set[str] | None = None,
) -> tuple[str, float]:
    """Find the runner-up type candidate (different from primary).

    Collects all unique type candidates from all signal sources,
    picks the strongest one that isn't the primary type.
    """
    if generic_types is None:
        generic_types = set()

    # Gather all candidates with their best weight
    candidates: dict[str, float] = {}
    for s in filename_types:
        if s.label != primary_type:
            candidates[s.label] = max(candidates.get(s.label, 0.0), s.base_weight)
    for s in folder_types:
        if s.label != primary_type:
            candidates[s.label] = max(candidates.get(s.label, 0.0), s.base_weight)
    for s in extension_signals:
        if s.label != primary_type:
            candidates[s.label] = max(candidates.get(s.label, 0.0), s.base_weight)

    if not candidates:
        return "", 0.0

    best_label = max(candidates, key=candidates.get)
    return best_label, candidates[best_label]


def _apply_modifiers(
    inferred_type: str,
    type_confidence: float,
    inferred_category: str,
    folder_types: list[Signal],
    filename_types: list[Signal],
    type_domains: dict[str, str],
    reinf: dict,
    conflict: dict,
    weights: dict,
    reasoning: list[str],
) -> float:
    """Apply reinforcement and conflict modifiers to type confidence.

    Two simple modifier groups:
      1. Folder-filename agreement / parent-child / disagreement
      2. Category-domain reinforcement (boost only — no mismatch penalty)
    """
    # 1. Folder + filename type agreement / parent-child / conflict
    folder_type_names = {s.label for s in folder_types}
    filename_type_names = {s.label for s in filename_types}
    parent_child = weights.get("parent_child", {})
    parent_child_boost_val = weights.get("parent_child_boost", 0.05)

    if folder_type_names and filename_type_names:
        if inferred_type in folder_type_names:
            boost = reinf.get("type_agreement_boost", 0.10)
            type_confidence += boost
            reasoning.append(f"boost: folder+filename agree on {inferred_type} (+{boost})")
        elif _is_parent_child(folder_type_names, inferred_type, parent_child):
            parent_name = _find_parent(folder_type_names, inferred_type, parent_child)
            type_confidence += parent_child_boost_val
            reasoning.append(
                f"boost: {inferred_type} is valid child of folder "
                f"type {parent_name} (+{parent_child_boost_val})"
            )
        else:
            penalty = conflict.get("type_disagreement_penalty", 0.10)
            type_confidence -= penalty
            reasoning.append(
                f"penalty: folder says {folder_type_names}, "
                f"filename says {filename_type_names} (-{penalty})"
            )

    # 2. Category-domain reinforcement (boost only — no mismatch penalty).
    # A document mis-filed in the wrong category folder is still that document.
    if inferred_category != "Unknown" and inferred_type in type_domains:
        domain = type_domains[inferred_type]
        if domain != "universal" and domain == inferred_category:
            boost = reinf.get("category_domain_boost", 0.05)
            type_confidence += boost
            reasoning.append(
                f"boost: {inferred_type} domain '{domain}' "
                f"matches category '{inferred_category}' (+{boost})"
            )

    return type_confidence


def _is_parent_child(
    folder_type_names: set[str],
    child_type: str,
    parent_child: dict[str, list[str]],
) -> bool:
    """Check if any folder type is a valid parent of the child type."""
    for folder_type in folder_type_names:
        children = parent_child.get(folder_type, [])
        if child_type in children:
            return True
    return False


def _find_parent(
    folder_type_names: set[str],
    child_type: str,
    parent_child: dict[str, list[str]],
) -> str:
    """Return the name of the parent folder type."""
    for folder_type in folder_type_names:
        children = parent_child.get(folder_type, [])
        if child_type in children:
            return folder_type
    return "Unknown"


def _resolve_entities(
    entity_signals: list[Signal],
) -> tuple[dict[str, str], dict[str, float], dict[str, int]]:
    """Pick the deepest entity per entity type.

    Same logic as _resolve_category — deepest signal wins.
    Returns three dicts keyed by entity name: values, confidences, depths.
    """
    entities: dict[str, str] = {}
    confidences: dict[str, float] = {}
    depths: dict[str, int] = {}

    for signal in entity_signals:
        name = signal.label
        if name not in entities or signal.depth > depths[name]:
            entities[name] = signal.text
            confidences[name] = signal.base_weight
            depths[name] = signal.depth

    return entities, confidences, depths


def _source_label(signal: Signal) -> str:
    """Human-readable label for the signal source."""
    if signal.source == SignalSource.FOLDER_CATEGORY:
        return f"folder[{signal.depth}]"
    if signal.source == SignalSource.FOLDER_TYPE:
        return f"folder[{signal.depth}]"
    if signal.source == SignalSource.FOLDER_ENTITY:
        return f"entity[{signal.depth}]"
    if signal.source == SignalSource.EXTENSION_HINT:
        return "extension"
    return "filename"
