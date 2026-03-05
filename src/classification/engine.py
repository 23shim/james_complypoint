"""
Classification engine — DataFrame-level orchestration.

Takes a DataFrame with path segments, filename_stem, and extension
columns, runs folder + filename analysis, scores each row, and
appends classification columns.
"""

from __future__ import annotations

import logging
from collections import defaultdict

import pandas as pd

from classification.address_extractor import extract_address
from classification.config_loader import load_config
from classification.plot_extractor import extract_plot
from classification.scheme_extractor import extract_scheme
from classification.entity_cluster import cluster_entities
from classification.folder_analyser import analyse_folders
from classification.filename_analyser import analyse_filename
from classification.models import (
    ClassificationConfig,
    ClassificationResult,
    SignalSource,
)
import dataclasses
import json as _json_mod

from classification.scheme_detector import detect_and_assign_schemes
from classification.scorer import score

logger = logging.getLogger(__name__)


def enforce_entity_hierarchy(df: pd.DataFrame) -> pd.DataFrame:
    """Enforce the Scheme > Plot / Address hierarchy.

    Two rules:

    Rule 1 — Scheme beats plot/address at same depth.
      A folder identified as a scheme should not also count as a plot or
      address.  If entity_{type}_depth <= scheme_depth, clear that entity.
      Entities deeper than the scheme are genuine sub-entities and kept.

    Rule 2 — Plot must not sit below an address (no plot within an address).
      The valid relationships are Scheme→Plot and Scheme→Address.  A plot
      entity found deeper in the path than an address entity means the
      address is acting as a location/zone context, not a unit identifier.
      In that case the address is a false positive — clear it.  (If the
      address IS the scheme name, Rule 1 already clears it first.)

    Must run AFTER detect_and_assign_schemes() and BEFORE cluster_entities().
    """
    if df.empty or "entity_scheme" not in df.columns:
        return df

    has_scheme = (df["entity_scheme"] != "") & (df["entity_scheme_depth"] >= 0)

    # Track whether suppressed entities matched the scheme name itself
    df["entity_address_is_scheme_name"] = False
    df["entity_plot_is_scheme_name"] = False

    # ---- Rule 1: suppress plot/address at or above scheme level ----

    for entity_type in ("plot", "address"):
        val_col = f"entity_{entity_type}"
        conf_col = f"entity_{entity_type}_confidence"
        depth_col = f"entity_{entity_type}_depth"

        if val_col not in df.columns or depth_col not in df.columns:
            continue

        has_entity = df[val_col] != ""
        at_or_above_scheme = df[depth_col] <= df["entity_scheme_depth"]
        suppress_mask = has_scheme & has_entity & at_or_above_scheme

        count = suppress_mask.sum()
        if count > 0:
            # Mark rows where the entity value matches the scheme name
            is_scheme_name = suppress_mask & (df[val_col] == df["entity_scheme"])
            df.loc[is_scheme_name, f"entity_{entity_type}_is_scheme_name"] = True

            df.loc[suppress_mask, val_col] = ""
            df.loc[suppress_mask, conf_col] = 0.0
            df.loc[suppress_mask, depth_col] = -1
            logger.info(
                f"  Hierarchy enforcement: suppressed {count:,} "
                f"{entity_type} entities at/above scheme level"
            )

    # ---- Rule 2: no plot within an address ----
    # Both columns must be present and have valid depths.

    if (
        "entity_plot" in df.columns
        and "entity_address" in df.columns
        and "entity_plot_depth" in df.columns
        and "entity_address_depth" in df.columns
    ):
        has_plot = (df["entity_plot"] != "") & (df["entity_plot_depth"] >= 0)
        has_addr = (df["entity_address"] != "") & (df["entity_address_depth"] >= 0)
        # Plot deeper than address → address is a location context, not a unit id
        plot_below_addr = df["entity_plot_depth"] > df["entity_address_depth"]
        suppress_addr = has_plot & has_addr & plot_below_addr

        count = suppress_addr.sum()
        if count > 0:
            df.loc[suppress_addr, "entity_address"] = ""
            df.loc[suppress_addr, "entity_address_confidence"] = 0.0
            df.loc[suppress_addr, "entity_address_depth"] = -1
            logger.info(
                f"  Hierarchy enforcement: suppressed {count:,} address entities "
                f"sitting above a plot (no-plot-within-address rule)"
            )

    return df



def _extract_addresses(
    df: pd.DataFrame,
    config: ClassificationConfig,
) -> pd.DataFrame:
    """Extract clean address strings from raw entity values.

    Creates ``entity_address_extracted`` containing the cleaned address
    substring for the validated address.  Also creates
    ``extracted_addresss`` transforming all raw match values for
    clustering.

    Must run AFTER enforce_entity_hierarchy(), BEFORE cluster_entities().
    """
    if "address" not in config.entities:
        return df

    entity_def = config.entities["address"]

    # Extract from validated entity_address column
    if "entity_address" in df.columns:
        df["entity_address_extracted"] = df["entity_address"].apply(
            lambda v: extract_address(v, entity_def)
            if isinstance(v, str) and v.strip() else ""
        )
    else:
        df["entity_address_extracted"] = ""

    # Extract from raw_addresss column (all matches, for clustering pool)
    raw_col = "raw_addresss"
    extracted_col = "extracted_addresss"
    if raw_col in df.columns:
        def _extract_raw_list(raw_list):
            if not isinstance(raw_list, list):
                return []
            result = []
            for match in raw_list:
                if isinstance(match, dict):
                    v = match.get("value", "")
                    extracted = extract_address(v, entity_def) if v else ""
                    result.append({**match, "extracted": extracted})
                elif isinstance(match, str):
                    extracted = extract_address(match, entity_def)
                    result.append({"value": match, "extracted": extracted})
            return result

        df[extracted_col] = df[raw_col].apply(_extract_raw_list)

    count = (df["entity_address_extracted"] != "").sum()
    logger.info(f"  Address extraction: {count:,} addresses extracted")

    return df


def _extract_plots(
    df: pd.DataFrame,
    config: ClassificationConfig,
) -> pd.DataFrame:
    """Extract clean plot references from raw entity values.

    Creates ``entity_plot_extracted`` containing the cleaned plot
    reference for the validated plot.  Also creates
    ``extracted_plots`` transforming all raw match values for
    clustering.

    Must run AFTER enforce_entity_hierarchy(), BEFORE cluster_entities().
    """
    if "plot" not in config.entities:
        return df

    entity_def = config.entities["plot"]
    address_entity_def = config.entities.get("address")

    # Extract from validated entity_plot column
    if "entity_plot" in df.columns:
        df["entity_plot_extracted"] = df["entity_plot"].apply(
            lambda v: extract_plot(v, entity_def, address_entity_def)
            if isinstance(v, str) and v.strip() else ""
        )
    else:
        df["entity_plot_extracted"] = ""

    # Extract from raw_plots column (all matches, for clustering pool)
    raw_col = "raw_plots"
    extracted_col = "extracted_plots"
    if raw_col in df.columns:
        def _extract_raw_list(raw_list):
            if not isinstance(raw_list, list):
                return []
            result = []
            for match in raw_list:
                if isinstance(match, dict):
                    v = match.get("value", "")
                    extracted = (
                        extract_plot(v, entity_def, address_entity_def)
                        if v else ""
                    )
                    result.append({**match, "extracted": extracted})
                elif isinstance(match, str):
                    extracted = extract_plot(
                        match, entity_def, address_entity_def,
                    )
                    result.append({"value": match, "extracted": extracted})
            return result

        df[extracted_col] = df[raw_col].apply(_extract_raw_list)

    count = (df["entity_plot_extracted"] != "").sum()
    logger.info(f"  Plot extraction: {count:,} plots extracted")

    return df


def _extract_schemes(
    df: pd.DataFrame,
    config: ClassificationConfig,
) -> pd.DataFrame:
    """Extract location identity from scheme folder names.

    Creates ``entity_scheme_extracted`` containing the cleaned
    location identity for the validated scheme.  Used by
    clustering to compare scheme names by their core location
    rather than raw folder names with phase/date noise.

    Must run AFTER detect_and_assign_schemes() and
    enforce_entity_hierarchy(), BEFORE cluster_entities().
    """
    if "address" not in config.entities:
        if "entity_scheme" in df.columns:
            df["entity_scheme_extracted"] = ""
        return df

    address_def = config.entities["address"]
    place_names = config.place_names

    if "entity_scheme" in df.columns:
        df["entity_scheme_extracted"] = df["entity_scheme"].apply(
            lambda v: extract_scheme(v, address_def, place_names)
            if isinstance(v, str) and v.strip() else ""
        )
    else:
        df["entity_scheme_extracted"] = ""

    count = (df["entity_scheme_extracted"] != "").sum()
    logger.info(f"  Scheme extraction: {count:,} schemes extracted")

    return df


def classify(
    df: pd.DataFrame,
    config: ClassificationConfig,
    root_depth: int = -1,
    skip_clustering: bool = False,
    candidates_path: str | None = None,
) -> pd.DataFrame:
    """Classify every row and append result columns.

    Entity matching is integrated into the signal pipeline: folder_analyser
    produces entity signals alongside type/category signals, and scorer
    resolves entities using deepest-wins (same as categories).

    Args:
        df: DataFrame with columns: segments, filename_stem, extension.
        config: Loaded ClassificationConfig.
        root_depth: Number of residual container folder levels above the
            first meaningful segment (normally 0 when root_prefix is fully
            stripped). Passed through to scheme detection so depth penalties
            are applied relative to the estate's actual root level.

    Returns:
        DataFrame with classification, entity, and secondary type columns.
    """
    # Build type_domains lookup once
    type_domains = {
        name: td.belongs_to for name, td in config.types.items()
    }
    weights = config.weights

    results: list[ClassificationResult] = []
    raw_entity_all: list[dict[str, list[dict]]] = []

    # Collect per-folder category signals for scheme detection.
    # Maps folder path → set of category names detected at that segment.
    folder_categories: dict[str, set[str]] = defaultdict(set)

    for idx, row in df.iterrows():
        segments = row.get("segments") or []
        stem = row.get("filename_stem") or ""
        ext = row.get("extension") or ""

        # Tier 1: folder signals (type + category + entity)
        signals = analyse_folders(segments, config)

        # Capture per-segment category signals for scheme detection.
        for signal in signals:
            if signal.source == SignalSource.FOLDER_CATEGORY and signal.depth >= 0:
                seg_path = "\\".join(segments[:signal.depth + 1])
                folder_categories[seg_path].add(signal.label)

        # Tier 2: filename signals
        signals.extend(analyse_filename(stem, ext, config))

        # Score and resolve (type + category + entities)
        result = score(signals, weights, type_domains)
        results.append(result)

        # Raw all-matches — every segment that passes entity matching,
        # ordered deepest-first. Derived from the entity signals already
        # in the signal list (no separate matching call needed).
        raw_all: dict[str, list[dict]] = {}
        entity_signals = [
            s for s in signals if s.source == SignalSource.FOLDER_ENTITY
        ]
        for entity_name in config.entities:
            raw_all[entity_name] = sorted(
                [
                    {"value": s.text, "confidence": s.base_weight, "depth": s.depth}
                    for s in entity_signals if s.label == entity_name
                ],
                key=lambda m: -m["depth"],
            )
        raw_entity_all.append(raw_all)

    # Unpack classification results into columns
    df["inferred_type"] = [r.inferred_type for r in results]
    df["type_confidence"] = [r.type_confidence for r in results]
    df["secondary_type"] = [r.secondary_type for r in results]
    df["secondary_type_confidence"] = [r.secondary_type_confidence for r in results]
    df["inferred_category"] = [r.inferred_category for r in results]
    df["category_confidence"] = [r.category_confidence for r in results]
    df["overall_confidence"] = [r.overall_confidence for r in results]
    df["confidence_band"] = [r.confidence_band for r in results]
    df["readiness_status"] = [r.readiness_status for r in results]
    df["reasoning_trace"] = [r.reasoning_trace for r in results]

    # Unpack entity results from ClassificationResult
    entity_names = list(config.entities.keys())
    for entity_name in entity_names:
        val_col = f"entity_{entity_name}"
        conf_col = f"entity_{entity_name}_confidence"
        depth_col = f"entity_{entity_name}_depth"
        df[val_col] = [r.entities.get(entity_name, "") for r in results]
        df[conf_col] = [r.entity_confidences.get(entity_name, 0.0) for r in results]
        df[depth_col] = [r.entity_depths.get(entity_name, -1) for r in results]

    if entity_names:
        for entity_name in entity_names:
            col = f"entity_{entity_name}"
            found = (df[col] != "").sum()
            logger.info(f"  entity_{entity_name}: {found:,} matched")

    # Raw all-matches columns — lists of every matching segment per entity type.
    for entity_name in entity_names:
        col = f"raw_{entity_name}s"
        df[col] = [r.get(entity_name, []) for r in raw_entity_all]

    # Scheme detection — runs after initial classification.
    df, all_candidates = detect_and_assign_schemes(
        df, config, folder_categories=folder_categories, root_depth=root_depth,
    )

    # Persist ALL scored scheme candidates (chosen + excluded) so report
    # scripts can show why each folder was or wasn't selected as a scheme.
    if candidates_path and all_candidates:
        try:
            sidecar = {}
            for path, (match, status, unique_ents) in all_candidates.items():
                entry = dataclasses.asdict(match)
                entry["candidate_status"] = status
                if unique_ents:
                    entry["unique_entities"] = unique_ents
                sidecar[path] = entry
            with open(candidates_path, "w", encoding="utf-8") as _f:
                _json_mod.dump(sidecar, _f)
            chosen = sum(1 for _, (_, s, _) in all_candidates.items() if s == "chosen")
            logger.info(
                f"  Scheme candidates saved: {candidates_path} "
                f"({len(sidecar):,} total, {chosen:,} chosen)"
            )
        except Exception as _e:
            logger.warning(f"  Could not save scheme candidates: {_e}")

    # Hierarchy enforcement — suppresses plot/address at/above scheme level and
    # enforces no-plot-within-address rule.
    df = enforce_entity_hierarchy(df)

    # Address extraction — extract clean address substrings from raw segment
    # names for improved clustering.  Runs after hierarchy enforcement so
    # those steps operate on the original raw segment values.
    df = _extract_addresses(df, config)

    # Plot extraction — extract clean plot references from raw segment names
    # for improved clustering.  Same timing as address extraction.
    df = _extract_plots(df, config)

    # Scheme extraction — extract location identity from scheme folder names
    # for improved clustering.  Strips phases/dates, captures address triggers
    # and place names.
    df = _extract_schemes(df, config)

    if not skip_clustering:
        # Entity clustering — estate-wide post-processing
        df = cluster_entities(df, config)

    # Log distribution
    type_counts = df["readiness_status"].value_counts()
    for status, count in type_counts.items():
        logger.info(f"  {status}: {count:,}")

    return df
