"""
Config loading and merging.

Loads base.yaml, merges the industry overlay, optionally merges
a client override, and compiles regex patterns once.

Loading order: base.yaml <- industry/{name}.yaml <- client (optional)
Industry ADDS to base — extends signal lists, adds new types/categories.
"""

import csv
import logging
import re
from pathlib import Path

import yaml

from classification.models import (
    CategoryDefinition,
    ClassificationConfig,
    EntityDefinition,
    SchemeExclusions,
    TypeDefinition,
)

logger = logging.getLogger(__name__)


def load_config(
    config_dir: str | Path,
    industry: str,
    client_path: str | Path | None = None,
) -> ClassificationConfig:
    """Load and merge classification config from YAML files.

    Args:
        config_dir: Root config directory (contains classification/, dictionaries/).
        industry: Industry name — loads dictionaries/industry/{industry}.yaml.
        client_path: Optional path to a client-specific override YAML.

    Returns:
        ClassificationConfig with merged types, categories, and compiled patterns.
    """
    config_dir = Path(config_dir)

    # --- Load base ---
    base_path = config_dir / "classification" / "base.yaml"
    base = _load_yaml(base_path, required=True)

    # --- Load industry overlay ---
    industry_path = config_dir / "dictionaries" / "industry" / f"{industry}.yaml"
    industry_data = _load_yaml(industry_path, required=False)

    # --- Load optional client override ---
    client_data = None
    if client_path:
        client_data = _load_yaml(Path(client_path), required=False)

    # --- Merge ---
    merged_categories = _merge_categories(
        base.get("categories", {}),
        industry_data.get("categories", {}) if industry_data else {},
        client_data.get("categories", {}) if client_data else {},
    )

    merged_types = _merge_types(
        base.get("types", {}),
        industry_data.get("types", {}) if industry_data else {},
        client_data.get("types", {}) if client_data else {},
    )

    merged_entities = _merge_entities(
        base.get("entities", {}),
        industry_data.get("entities", {}) if industry_data else {},
        client_data.get("entities", {}) if client_data else {},
    )

    # --- Merge scheme exclusions ---
    scheme_exclusions = _merge_scheme_exclusions(
        base.get("scheme_exclusions", {}),
        industry_data.get("scheme_exclusions", {}) if industry_data else {},
        client_data.get("scheme_exclusions", {}) if client_data else {},
    )

    # --- Load weights ---
    weights_path = config_dir / "classification" / "weights.yaml"
    weights = _load_yaml(weights_path, required=True)

    # --- Load place names for scheme detection ---
    place_min_length = weights.get("place_names", {}).get("min_length", 4)
    place_names = _load_place_names(config_dir, min_length=place_min_length)

    logger.info(
        f"Classification config loaded: "
        f"{len(merged_types)} types, {len(merged_categories)} categories, "
        f"{len(merged_entities)} entities, "
        f"{len(scheme_exclusions.names)} scheme exclusion names "
        f"(base + {industry}"
        f"{' + client' if client_data else ''})"
    )

    return ClassificationConfig(
        types=merged_types,
        categories=merged_categories,
        weights=weights,
        entities=merged_entities,
        scheme_exclusions=scheme_exclusions,
        place_names=place_names,
    )


def _load_yaml(path: Path, required: bool = True) -> dict | None:
    """Load a YAML file, returning None if optional and missing."""
    if not path.exists():
        if required:
            raise FileNotFoundError(f"Required config file not found: {path}")
        logger.debug(f"Optional config not found (skipped): {path}")
        return None

    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f)

    if data is None:
        return {}

    logger.debug(f"Loaded config: {path}")
    return data


def _merge_categories(
    base: dict, industry: dict, client: dict
) -> dict[str, CategoryDefinition]:
    """Merge category definitions. Industry/client EXTEND signal lists."""
    merged = {}

    for layer in [base, industry, client]:
        for name, defn in layer.items():
            if name in merged:
                existing = merged[name]
                new_signals = [s.lower() for s in defn.get("signals", [])]
                new_aliases = [a.lower() for a in defn.get("aliases", [])]
                existing.signals = _unique_list(existing.signals + new_signals)
                existing.aliases = _unique_list(existing.aliases + new_aliases)
                if defn.get("scheme_signal"):
                    existing.scheme_signal = True
            else:
                merged[name] = CategoryDefinition(
                    name=name,
                    signals=[s.lower() for s in defn.get("signals", [])],
                    aliases=[a.lower() for a in defn.get("aliases", [])],
                    scheme_signal=bool(defn.get("scheme_signal", False)),
                )

    return merged


def _merge_types(
    base: dict, industry: dict, client: dict
) -> dict[str, TypeDefinition]:
    """Merge type definitions. Industry/client ADD new types or extend existing."""
    merged = {}

    for layer_index, layer in enumerate([base, industry, client]):
        for name, defn in layer.items():
            if name in merged:
                # Extend existing type with additional tokens/patterns
                # Keep the original origin_layer (first definition wins)
                existing = merged[name]
                existing.tokens = _unique_list(
                    existing.tokens + [t.lower() for t in defn.get("tokens", [])]
                )
                existing.abbreviations = _unique_list(
                    existing.abbreviations
                    + [a.lower() for a in defn.get("abbreviations", [])]
                )
                new_patterns = _compile_patterns(defn.get("patterns", []))
                existing.compiled_patterns.extend(new_patterns)
                if defn.get("extensions"):
                    existing.extensions = _unique_list(
                        existing.extensions
                        + [e.lower() for e in defn["extensions"]]
                    )
            else:
                merged[name] = TypeDefinition(
                    name=name,
                    tokens=[t.lower() for t in defn.get("tokens", [])],
                    abbreviations=[a.lower() for a in defn.get("abbreviations", [])],
                    compiled_patterns=_compile_patterns(defn.get("patterns", [])),
                    belongs_to=defn.get("belongs_to", defn.get("domain", "universal")),
                    extensions=[e.lower() for e in defn.get("extensions", [])],
                    origin_layer=layer_index,
                )

    return merged


def _merge_entities(
    base: dict, industry: dict, client: dict
) -> dict[str, EntityDefinition]:
    """Merge entity definitions. Industry/client ADD or EXTEND entities."""
    merged = {}

    for layer in [base, industry, client]:
        for name, defn in layer.items():
            if name in merged:
                existing = merged[name]
                existing.tokens = _unique_list(
                    existing.tokens + [t.lower() for t in defn.get("tokens", [])]
                )
                existing.abbreviations = _unique_list(
                    existing.abbreviations
                    + [a.lower() for a in defn.get("abbreviations", [])]
                )
                new_patterns = _compile_patterns(defn.get("patterns", []))
                existing.compiled_patterns.extend(new_patterns)
                # Merge abbreviation_map: later layers override per key
                raw_map = defn.get("abbreviation_map", {})
                existing.abbreviation_map.update(
                    {k.lower(): v.lower() for k, v in raw_map.items()}
                )
            else:
                raw_map = defn.get("abbreviation_map", {})
                merged[name] = EntityDefinition(
                    name=name,
                    tokens=[t.lower() for t in defn.get("tokens", [])],
                    abbreviations=[a.lower() for a in defn.get("abbreviations", [])],
                    compiled_patterns=_compile_patterns(defn.get("patterns", [])),
                    abbreviation_map={
                        k.lower(): v.lower() for k, v in raw_map.items()
                    },
                )

    return merged


def _load_place_names(config_dir: Path, min_length: int = 4) -> set[str]:
    """Load place names for scheme detection.

    Loads from two sources in ``dictionaries/place_names/``:

    1. ``uk_places.txt`` — one lowercased name per line (curated list).
    2. ``IPN_GB_2019.csv`` — ONS Index of Place Names; extracts the
       ``place18nm`` column to get the full ~99k GB locality dataset.

    Results are combined so the final set is as comprehensive as possible.
    Gracefully skips any file that is missing (falls back to the other).
    Returns an empty set only if both files are absent (the place-name
    signal is then silently skipped during scheme scoring).
    """
    names: set[str] = set()
    place_dir = config_dir / "dictionaries" / "place_names"

    # ── uk_places.txt ─────────────────────────────────────────────────
    txt_path = place_dir / "uk_places.txt"
    if txt_path.exists():
        with open(txt_path, encoding="utf-8") as f:
            for line in f:
                name = line.strip().lower()
                if name and len(name) >= min_length:
                    names.add(name)
        logger.debug("uk_places.txt: %s entries", f"{len(names):,}")
    else:
        logger.debug("uk_places.txt not found (skipped)")

    # ── IPN_GB_2019.csv ───────────────────────────────────────────────
    ipn_path = place_dir / "IPN_GB_2019.csv"
    if ipn_path.exists():
        before = len(names)
        try:
            with open(ipn_path, encoding="utf-8", errors="replace") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    name = row.get("place18nm", "").strip().lower()
                    if name and len(name) >= min_length:
                        names.add(name)
            added = len(names) - before
            logger.debug("IPN_GB_2019.csv: +%s new entries", f"{added:,}")
        except Exception as exc:
            logger.warning("Could not read IPN_GB_2019.csv: %s", exc)
    else:
        logger.debug("IPN_GB_2019.csv not found (skipped)")

    if names:
        logger.info(
            "Loaded %s place names (%s sources)",
            f"{len(names):,}",
            sum([txt_path.exists(), ipn_path.exists()]),
        )
    else:
        logger.debug("No place names loaded — place-name signal disabled")

    return names


def _merge_scheme_exclusions(
    base: dict, industry: dict, client: dict,
) -> SchemeExclusions:
    """Merge scheme exclusion lists from all layers."""
    names: set[str] = set()
    substrings: list[str] = []
    patterns: list[str] = []
    container_keywords: list[str] = []

    for layer in [base, industry, client]:
        if not layer:
            continue
        for n in layer.get("names", []):
            names.add(n.lower().strip())
        for s in layer.get("substrings", []):
            val = s.lower().strip()
            if val not in substrings:
                substrings.append(val)
        for p in layer.get("patterns", []):
            if p not in patterns:
                patterns.append(p)
        for kw in layer.get("container_keywords", []):
            val = kw.lower().strip()
            if val not in container_keywords:
                container_keywords.append(val)

    return SchemeExclusions(
        names=names,
        substrings=substrings,
        compiled_patterns=_compile_patterns(patterns),
        container_keywords=container_keywords,
    )


def _compile_patterns(patterns: list[str]) -> list[re.Pattern]:
    """Compile regex patterns, logging warnings for invalid ones."""
    compiled = []
    for pat in patterns:
        try:
            compiled.append(re.compile(pat, re.IGNORECASE))
        except re.error as e:
            logger.warning(f"Invalid regex pattern '{pat}': {e}")
    return compiled


def _unique_list(items: list) -> list:
    """Deduplicate while preserving order."""
    seen = set()
    result = []
    for item in items:
        if item not in seen:
            seen.add(item)
            result.append(item)
    return result
