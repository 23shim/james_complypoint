"""
Entity clustering via character-trigram similarity.

Groups variant spellings of the same real-world entity
(e.g. "Mills Crescent", "Mills Cres", "mills crescent")
under a single canonical form with a shared cluster ID.

This is a post-processing step that runs AFTER entity extraction
and scheme detection.  For each entity type (plot, address, scheme),
it collects all unique values estate-wide, normalises them, computes
pairwise character-trigram Jaccard similarity, and single-linkage
clusters values above a threshold.

No config-driven abbreviation maps — normalisation is purely
mechanical (lowercase, strip punctuation, collapse whitespace).
The n-gram similarity handles abbreviations and typos naturally.

Plot clustering is context-aware: two plot values (e.g. "plot 7"
from scheme A and "plot 7" from scheme B) are only merged if they
share the same scheme_cluster_id or address_cluster_id.  Plots with
no scheme or address context are isolated into singleton clusters so
they never contaminate scheme-linked plots.

Processing order inside cluster_entities: scheme first (no
dependencies), then address (scheme context available), then plot
(scheme + address context available).
"""

from __future__ import annotations

import logging
import re
from collections import Counter

import pandas as pd

from classification.models import ClassificationConfig
from classification.similarity import (
    UnionFind,
    char_trigrams,
    jaccard_similarity,
    overlap_coefficient,
)

logger = logging.getLogger(__name__)

# Characters to strip from entity values during normalisation.
# Keep hyphens and ampersands (appear in real names: "Health & Safety",
# "Lenham-on-Sea").
_STRIP_CHARS_RE = re.compile(r"[()[\]{},;:'\"!?#@*/\\]")
_COLLAPSE_WHITESPACE_RE = re.compile(r"\s+")

# Entity clustering defaults — overridden by weights.yaml "entity_clustering" section.
_CLUSTER_DEFAULTS = {
    "similarity_threshold": 0.55,
}


def _load_cluster_weights(weights: dict) -> dict:
    """Load entity clustering weights from config, falling back to defaults."""
    overrides = weights.get("entity_clustering", {})
    return {k: overrides.get(k, v) for k, v in _CLUSTER_DEFAULTS.items()}

# Prefix on no-context plot keys — never merged across folders.
_NO_CONTEXT_PREFIX = "__nocontext__"

# ---- Scheme noise patterns ----
# Universal metadata that appears in housing development folder names
# across all estates.  Stripped before the second-pass similarity check
# so the comparison focuses on the core scheme identity (location/name).

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
# Date ranges: "2015-2016", "2020-2023"
_DATE_RANGE_RE = re.compile(r"\b\d{4}\s*-\s*\d{4}\b")
# Day-month-year: "11 May 21", "3 February 2024"
_DAY_MONTH_YEAR_RE = re.compile(
    rf"\b\d{{1,2}}\s+(?:{_MONTHS})\s+\d{{2,4}}\b", re.IGNORECASE,
)
# Month-year: "Feb 2020", "December 21", "July 22"
_MONTH_YEAR_RE = re.compile(
    rf"\b(?:{_MONTHS})\s+\d{{2,4}}\b", re.IGNORECASE,
)
# Standalone 4-digit year (19xx/20xx) at word boundary
_STANDALONE_YEAR_RE = re.compile(r"\b(?:19|20)\d{2}\b")
# "New" admin annotations: "New Dec 20", "New file - 11 May 21", "New scheme"
_NEW_ANNOTATION_RE = re.compile(
    rf"\bnew\s+(?:file\b[\s-]*|scheme\b|(?:{_MONTHS})(?:\s+\d{{2,4}})?)",
    re.IGNORECASE,
)
# Orphaned separators left after noise removal (e.g. " - " between stripped tokens)
_ORPHAN_SEPARATOR_RE = re.compile(r"\s+-\s+(?=-|\s|$)|\s+-$|^-\s+")


# ---------------------------------------------------------------------------
# Standard entity collection and clustering (address, scheme, others)
# ---------------------------------------------------------------------------

def _collect_folder_values(
    df: pd.DataFrame, entity_type: str,
) -> Counter:
    """Collect unique folder-level entity values for clustering.

    Scans the raw matches column (e.g. raw_addresss) across all rows,
    deduplicates by folder path (segments[:depth+1]), and returns
    value → number_of_unique_folders.

    For entity types with extracted values (address, plot), uses
    extracted values from the ``extracted_{type}s`` column when
    available.  This gives the clustering pool cleaner values
    (e.g. "67-70 hawkes way" instead of "plots 67-70 Hawkes Way").

    Folder-centric: if 500 files sit under a folder called "Ash Close",
    that's ONE folder contributing "Ash Close" to the count.  The
    clustering pool is built from unique folders, not files.
    """
    raw_col = f"raw_{entity_type}s"
    extracted_col = f"extracted_{entity_type}s"
    use_extracted = extracted_col in df.columns

    if raw_col not in df.columns:
        return Counter()

    # folder_path → cluster_value (one entry per unique folder)
    seen_folders: dict[str, str] = {}

    for idx in df.index:
        segs = df.at[idx, "segments"]
        if not isinstance(segs, list):
            continue

        raw_list = df.at[idx, raw_col]
        if not isinstance(raw_list, list):
            continue

        # Pre-fetch extracted list for entity types with extraction.
        extracted_list = None
        if use_extracted:
            ext = df.at[idx, extracted_col]
            if isinstance(ext, list):
                extracted_list = ext

        for i, match in enumerate(raw_list):
            if isinstance(match, dict):
                v = match.get("value", "")
                depth = int(match.get("depth", -1))
            elif isinstance(match, str):
                v = match
                depth = -1
            else:
                continue

            if not (v and isinstance(v, str) and v.strip()):
                continue

            # Use extracted value when available.
            cluster_value = v
            if extracted_list and i < len(extracted_list):
                ext_match = extracted_list[i]
                if isinstance(ext_match, dict):
                    ext_v = ext_match.get("extracted", "")
                    if ext_v:
                        cluster_value = ext_v

            # Build folder path from segments up to match depth
            if 0 <= depth < len(segs):
                folder_path = "\\".join(segs[:depth + 1])
            else:
                folder_path = f"__nodepth__{v}"

            # One entry per unique folder path
            if folder_path not in seen_folders:
                seen_folders[folder_path] = cluster_value

    # Count unique folders per value
    return Counter(seen_folders.values())


def _backfill_from_raw(
    df: pd.DataFrame,
    entity_type: str,
    lookup: dict[str, tuple[str, str]],
) -> int:
    """Backfill cluster assignment for rows with empty entity but raw matches.

    For files that have no validated entity_{type} value (suppressed by
    validation or hierarchy enforcement) but DO have raw matches, assign
    the cluster_id and canonical from the highest-confidence raw match.

    For entity types with extraction (address, plot), also checks
    extracted values against the lookup (since the clustering pool
    may be built from extracted values rather than raw segment names).

    Returns the number of rows backfilled.
    """
    value_col = f"entity_{entity_type}"
    conf_col = f"entity_{entity_type}_confidence"
    depth_col = f"entity_{entity_type}_depth"
    cluster_col = f"entity_{entity_type}_cluster_id"
    canonical_col = f"entity_{entity_type}_canonical"
    raw_col = f"raw_{entity_type}s"
    extracted_col = f"extracted_{entity_type}s"
    use_extracted = extracted_col in df.columns

    if raw_col not in df.columns:
        return 0

    backfilled = 0

    for idx in df.index:
        # Skip rows that already have a cluster assignment
        if df.at[idx, cluster_col]:
            continue

        raw_list = df.at[idx, raw_col]
        if not isinstance(raw_list, list) or not raw_list:
            continue

        # Pre-fetch extracted list for entity types with extraction.
        extracted_list = None
        if use_extracted:
            ext = df.at[idx, extracted_col]
            if isinstance(ext, list):
                extracted_list = ext

        # Find best raw match that exists in the cluster lookup
        best_value = None
        best_lookup_key = None
        best_conf = -1.0
        best_depth = -1

        for i, match in enumerate(raw_list):
            if isinstance(match, dict):
                v = match.get("value", "")
                conf = float(match.get("confidence", 0))
                depth = int(match.get("depth", -1))
            elif isinstance(match, str):
                v = match
                conf = 0.0
                depth = -1
            else:
                continue

            # Try extracted value first, then raw value.
            ext_v = ""
            if extracted_list and i < len(extracted_list):
                ext_match = extracted_list[i]
                if isinstance(ext_match, dict):
                    ext_v = ext_match.get("extracted", "")

            lookup_key = ext_v if (ext_v and ext_v in lookup) else v

            if lookup_key in lookup and conf > best_conf:
                best_value = v
                best_lookup_key = lookup_key
                best_conf = conf
                best_depth = depth

        if best_value is not None and best_lookup_key is not None:
            cid, canonical = lookup[best_lookup_key]
            df.at[idx, value_col] = best_value
            df.at[idx, conf_col] = round(best_conf, 3)
            df.at[idx, depth_col] = best_depth
            df.at[idx, cluster_col] = cid
            df.at[idx, canonical_col] = canonical
            backfilled += 1

    return backfilled


# ---------------------------------------------------------------------------
# Plot-specific context-aware clustering
# ---------------------------------------------------------------------------

def _build_plot_folder_context(df: pd.DataFrame) -> dict[str, str]:
    """Map each plot-matched folder path to a context key.

    Called after scheme and address clustering so that
    entity_scheme_cluster_id and entity_address_cluster_id are
    already populated on the DataFrame.

    Context key priority (first non-empty wins):
      1. entity_scheme_cluster_id of any row whose path includes
         this plot folder — ensures plots in scheme A never merge
         with identically-named plots in scheme B.
      2. entity_address_cluster_id, when the address folder is a
         genuine ancestor of the plot folder (address_depth < plot_depth).
         Handles cases where no scheme was detected but a street
         address provides the grouping context.
      3. A unique per-folder key (``__nocontext__{folder_path}``) that
         guarantees the plot is isolated into its own singleton cluster
         and never merged with any other plot.

    Args:
        df: DataFrame with entity columns and scheme/address cluster IDs
            already populated.

    Returns:
        Dict mapping plot_folder_path → context_key string.
    """
    folder_to_context: dict[str, str] = {}

    raw_col = "raw_plots"
    scheme_cluster_col = "entity_scheme_cluster_id"
    address_cluster_col = "entity_address_cluster_id"
    address_depth_col = "entity_address_depth"

    if raw_col not in df.columns:
        return folder_to_context

    for idx in df.index:
        segs = df.at[idx, "segments"]
        if not isinstance(segs, list) or not segs:
            continue

        raw_list = df.at[idx, raw_col]
        if not isinstance(raw_list, list):
            continue

        # Row-level scheme context.
        scheme_ctx = ""
        if scheme_cluster_col in df.columns:
            v = df.at[idx, scheme_cluster_col]
            if isinstance(v, str) and v:
                scheme_ctx = v

        # Row-level address context (only valid when address is above plot).
        addr_ctx = ""
        addr_depth = -1
        if address_cluster_col in df.columns and address_depth_col in df.columns:
            v = df.at[idx, address_cluster_col]
            if isinstance(v, str) and v:
                addr_ctx = v
            d = df.at[idx, address_depth_col]
            try:
                addr_depth = int(d)
            except (TypeError, ValueError):
                addr_depth = -1

        for match in raw_list:
            if isinstance(match, dict):
                plot_depth = int(match.get("depth", -1))
            else:
                continue

            if plot_depth < 0 or plot_depth >= len(segs):
                continue

            folder_path = "\\".join(segs[:plot_depth + 1])

            # First row to reach this folder wins — all rows under the
            # same folder share the same ancestor scheme, so the result
            # is stable across rows.
            if folder_path in folder_to_context:
                continue

            if scheme_ctx:
                context = scheme_ctx
            elif addr_ctx and 0 <= addr_depth < plot_depth:
                # Address is a genuine ancestor of the plot folder.
                context = f"addr:{addr_ctx}"
            else:
                # No grouping context — isolate this folder.
                context = f"{_NO_CONTEXT_PREFIX}{folder_path}"

            folder_to_context[folder_path] = context

    return folder_to_context


def _collect_plot_context_pool(
    df: pd.DataFrame,
    folder_to_context: dict[str, str],
) -> Counter:
    """Build plot clustering pool as Counter[(context_key, extracted_value)].

    Like _collect_folder_values but namespaces each value by its
    context key so that "plot 7" under scheme A and "plot 7" under
    scheme B are separate items and can never be merged.

    Uses ALL raw plot matches (from raw_plots / extracted_plots),
    including candidates that were suppressed by the plot validator.
    This mirrors the address clustering approach: every folder segment
    that passed any string/token/pattern match contributes to the pool.

    Each unique plot folder path contributes exactly one entry.

    Args:
        df: DataFrame with raw_plots and extracted_plots columns.
        folder_to_context: Mapping from plot folder path to context key
            (built by _build_plot_folder_context).

    Returns:
        Counter mapping (context_key, extracted_value) → unique folder count.
    """
    raw_col = "raw_plots"
    extracted_col = "extracted_plots"
    use_extracted = extracted_col in df.columns

    if raw_col not in df.columns:
        return Counter()

    # folder_path → (context, cluster_value)
    seen_folders: dict[str, tuple[str, str]] = {}

    for idx in df.index:
        segs = df.at[idx, "segments"]
        if not isinstance(segs, list):
            continue

        raw_list = df.at[idx, raw_col]
        if not isinstance(raw_list, list):
            continue

        extracted_list = None
        if use_extracted:
            ext = df.at[idx, extracted_col]
            if isinstance(ext, list):
                extracted_list = ext

        for i, match in enumerate(raw_list):
            if isinstance(match, dict):
                v = match.get("value", "")
                depth = int(match.get("depth", -1))
            elif isinstance(match, str):
                v = match
                depth = -1
            else:
                continue

            if not (v and isinstance(v, str) and v.strip()):
                continue

            # Prefer the extracted (clean) value when available.
            cluster_value = v
            if extracted_list and i < len(extracted_list):
                ext_match = extracted_list[i]
                if isinstance(ext_match, dict):
                    ext_v = ext_match.get("extracted", "")
                    if ext_v:
                        cluster_value = ext_v

            if 0 <= depth < len(segs):
                folder_path = "\\".join(segs[:depth + 1])
            else:
                folder_path = f"__nodepth__{v}"

            if folder_path not in seen_folders:
                context = folder_to_context.get(
                    folder_path, f"{_NO_CONTEXT_PREFIX}{folder_path}",
                )
                seen_folders[folder_path] = (context, cluster_value)

    return Counter(seen_folders.values())


def _cluster_plot_context_values(
    pool: Counter,
    similarity_threshold: float,
) -> list[tuple[tuple[str, str], set[tuple[str, str]]]]:
    """Cluster (context_key, value) pairs for plot entities.

    Two pairs can only merge if:
      1. They share the same context_key (same scheme or address cluster).
      2. Their extracted plot values meet the Jaccard similarity threshold.

    Pairs whose context key starts with ``__nocontext__`` are never
    merged — each becomes its own singleton cluster.  This prevents
    "plot 7" with no scheme context from polluting scheme-linked
    "plot 7" clusters.

    Args:
        pool: Counter[(context_key, extracted_value)] from
            _collect_plot_context_pool.
        similarity_threshold: Jaccard threshold on normalised values.

    Returns:
        List of (canonical_pair, set_of_member_pairs), sorted by
        cluster size descending.
    """
    unique_pairs = list(pool.keys())
    n = len(unique_pairs)

    if n == 0:
        return []

    # Pre-compute normalised forms and trigrams on the VALUE portion only.
    normalised = [normalise_entity_value(val) for _, val in unique_pairs]
    trigrams = [char_trigrams(nv) for nv in normalised]

    uf = UnionFind(n)

    for i in range(n):
        ctx_i = unique_pairs[i][0]
        if ctx_i.startswith(_NO_CONTEXT_PREFIX):
            continue  # Never merge no-context plots

        for j in range(i + 1, n):
            ctx_j = unique_pairs[j][0]
            if ctx_j.startswith(_NO_CONTEXT_PREFIX):
                continue

            # Context must match exactly — plots in different scheme or
            # address contexts are never merged regardless of similarity.
            if ctx_i != ctx_j:
                continue

            sim = jaccard_similarity(trigrams[i], trigrams[j])
            if sim >= similarity_threshold:
                uf.union(i, j)

    index_clusters = uf.clusters()

    result = []
    for members_indices in index_clusters:
        members = {unique_pairs[i] for i in members_indices}
        # Canonical: most frequent pair; tie-break by value length.
        canonical = max(
            members,
            key=lambda p: (pool.get(p, 0), len(p[1])),
        )
        result.append((canonical, members))

    result.sort(key=lambda x: (-len(x[1]), str(x[0])))
    return result


def _assign_plot_clusters(
    df: pd.DataFrame,
    context_lookup: dict[tuple[str, str], tuple[str, str]],
    folder_to_context: dict[str, str],
    context_col: str = "",
) -> int:
    """Assign plot cluster IDs to rows with a validated entity_plot.

    Uses the plot folder path to determine context, then looks up
    (context_key, entity_plot_extracted) in context_lookup.

    If context_col is provided, also writes the context key to that
    column so report scripts can identify the linked scheme/address.

    Returns the number of rows assigned.
    """
    cluster_col = "entity_plot_cluster_id"
    canonical_col = "entity_plot_canonical"
    extracted_col = "entity_plot_extracted"
    depth_col = "entity_plot_depth"

    if extracted_col not in df.columns:
        return 0

    assigned = 0

    for idx in df.index:
        val = df.at[idx, extracted_col]
        if not isinstance(val, str) or not val.strip():
            continue

        depth = df.at[idx, depth_col] if depth_col in df.columns else -1
        segs = df.at[idx, "segments"]

        if not isinstance(segs, list) or not isinstance(depth, (int, float)):
            continue

        depth = int(depth)
        if depth < 0 or depth >= len(segs):
            continue

        folder_path = "\\".join(segs[:depth + 1])
        context = folder_to_context.get(
            folder_path, f"{_NO_CONTEXT_PREFIX}{folder_path}",
        )
        key = (context, val)

        cid, canonical = context_lookup.get(key, ("", ""))
        if cid:
            df.at[idx, cluster_col] = cid
            df.at[idx, canonical_col] = canonical
            if context_col:
                df.at[idx, context_col] = context
            assigned += 1

    return assigned


def _backfill_plot_clusters(
    df: pd.DataFrame,
    context_lookup: dict[tuple[str, str], tuple[str, str]],
    folder_to_context: dict[str, str],
    context_col: str = "",
) -> int:
    """Backfill plot cluster for rows with empty cluster but raw matches.

    For rows where entity_plot_cluster_id is still empty, scans
    raw_plots (and extracted_plots) for any match that resolves to a
    known (context, value) key.  Picks the highest-confidence match.

    If context_col is provided, also writes the context key to that
    column so report scripts can identify the linked scheme/address.

    Returns the number of rows backfilled.
    """
    value_col = "entity_plot"
    conf_col = "entity_plot_confidence"
    depth_col = "entity_plot_depth"
    cluster_col = "entity_plot_cluster_id"
    canonical_col = "entity_plot_canonical"
    raw_col = "raw_plots"
    extracted_col = "extracted_plots"
    use_extracted = extracted_col in df.columns

    if raw_col not in df.columns:
        return 0

    backfilled = 0

    for idx in df.index:
        if df.at[idx, cluster_col]:
            continue  # Already assigned

        raw_list = df.at[idx, raw_col]
        if not isinstance(raw_list, list) or not raw_list:
            continue

        segs = df.at[idx, "segments"]
        if not isinstance(segs, list):
            continue

        extracted_list = None
        if use_extracted:
            ext = df.at[idx, extracted_col]
            if isinstance(ext, list):
                extracted_list = ext

        best_value = None
        best_cid = ""
        best_canonical = ""
        best_context = ""
        best_conf = -1.0
        best_depth = -1

        for i, match in enumerate(raw_list):
            if isinstance(match, dict):
                v = match.get("value", "")
                conf = float(match.get("confidence", 0))
                depth = int(match.get("depth", -1))
            elif isinstance(match, str):
                v = match
                conf = 0.0
                depth = -1
            else:
                continue

            ext_v = ""
            if extracted_list and i < len(extracted_list):
                ext_match = extracted_list[i]
                if isinstance(ext_match, dict):
                    ext_v = ext_match.get("extracted", "")

            lookup_val = ext_v if ext_v else v

            if 0 <= depth < len(segs):
                folder_path = "\\".join(segs[:depth + 1])
            else:
                folder_path = f"__nodepth__{v}"

            context = folder_to_context.get(
                folder_path, f"{_NO_CONTEXT_PREFIX}{folder_path}",
            )
            key = (context, lookup_val)

            if key in context_lookup and conf > best_conf:
                best_value = v
                best_cid, best_canonical = context_lookup[key]
                best_conf = conf
                best_depth = depth
                best_context = context

        if best_cid:
            df.at[idx, value_col] = best_value
            df.at[idx, conf_col] = round(best_conf, 3)
            df.at[idx, depth_col] = best_depth
            df.at[idx, cluster_col] = best_cid
            df.at[idx, canonical_col] = best_canonical
            if context_col:
                df.at[idx, context_col] = best_context
            backfilled += 1

    return backfilled


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def cluster_entities(
    df: pd.DataFrame,
    config: ClassificationConfig,
    similarity_threshold: float | None = None,
) -> pd.DataFrame:
    """Cluster entity values estate-wide and assign canonical forms.

    Processing order is fixed so that context IDs are available when
    needed:
      1. scheme  — no dependencies; produces entity_scheme_cluster_id
      2. address — uses scheme context if desired in future; produces
                   entity_address_cluster_id
      3. plot    — context-aware: requires scheme_cluster_id and
                   address_cluster_id to disambiguate plot numbers that
                   repeat across different developments
      4. others  — any remaining entity types in config order

    For scheme, address, and other types: standard folder-centric
    character-trigram clustering (same as before).

    For plot: context-namespaced clustering.  The pool is built from ALL
    raw plot matches (including validator-suppressed candidates, mirroring
    the address approach).  Two plot values only merge if they share the
    same context key (scheme_cluster_id or address_cluster_id) AND meet
    the similarity threshold.  Plots without any scheme or address context
    are isolated into singleton clusters.

    Adds columns per entity type:
      - entity_{type}_cluster_id: str (e.g. "addr_001", "plot_0002")
      - entity_{type}_canonical: str (most frequent variant)

    Args:
        df: DataFrame with entity columns already populated.
        config: Classification config (used to get entity type names).
        similarity_threshold: Jaccard threshold for clustering.

    Returns:
        DataFrame with cluster columns added.
    """
    # Resolve similarity threshold from config, parameter override, or default.
    if similarity_threshold is None:
        cw = _load_cluster_weights(config.weights)
        similarity_threshold = cw["similarity_threshold"]

    logger.info("Starting entity clustering...")

    # Determine processing order.
    # Scheme first (no dependencies), then address, then plot, then others.
    entity_types = list(config.entities.keys())
    if "scheme" not in entity_types:
        entity_types = ["scheme"] + entity_types
    else:
        entity_types = ["scheme"] + [e for e in entity_types if e != "scheme"]

    # Ensure address is processed before plot so address_cluster_id is
    # available when building plot context.
    if "address" in entity_types and "plot" in entity_types:
        a_idx = entity_types.index("address")
        p_idx = entity_types.index("plot")
        if p_idx < a_idx:
            entity_types.pop(a_idx)
            entity_types.insert(p_idx, "address")

    df = df.copy()

    for entity_type in entity_types:

        # ------------------------------------------------------------------ #
        #  Plot: context-aware clustering                                     #
        # ------------------------------------------------------------------ #

        if entity_type == "plot":
            cluster_col = "entity_plot_cluster_id"
            canonical_col = "entity_plot_canonical"
            context_col = "entity_plot_cluster_context"
            df[cluster_col] = ""
            df[canonical_col] = ""
            df[context_col] = ""

            if "entity_plot" not in df.columns:
                logger.info("  plot: no entity_plot column — skipping")
                continue

            # Step 1: Build folder → context mapping using already-computed
            # scheme and address cluster IDs.
            folder_to_context = _build_plot_folder_context(df)

            # Step 2: Build (context, value) pool from ALL raw plot matches.
            pool = _collect_plot_context_pool(df, folder_to_context)

            if not pool:
                logger.info("  plot: no values to cluster")
                continue

            # Step 3: Cluster with context constraint.
            clusters = _cluster_plot_context_values(pool, similarity_threshold)

            # Step 4: Build (context, value) → (cluster_id, canonical) lookup.
            context_lookup: dict[tuple[str, str], tuple[str, str]] = {}
            for idx, (canonical_pair, members) in enumerate(clusters):
                cluster_id = f"plot_{idx:04d}"
                canonical_val = canonical_pair[1]
                for pair in members:
                    context_lookup[pair] = (cluster_id, canonical_val)

            # Step 5: Assign clusters to rows with validated entity_plot.
            assigned = _assign_plot_clusters(
                df, context_lookup, folder_to_context, context_col,
            )

            # Step 6: Backfill rows with no validated entity_plot but raw matches.
            backfilled = _backfill_plot_clusters(
                df, context_lookup, folder_to_context, context_col,
            )

            n_context_keys = len({ctx for ctx, _ in pool.keys()
                                   if not ctx.startswith(_NO_CONTEXT_PREFIX)})
            n_no_context = len({fp for fp, ctx in folder_to_context.items()
                                 if ctx.startswith(_NO_CONTEXT_PREFIX)})
            n_merged = len(pool) - len(clusters)
            logger.info(
                f"  plot: {len(pool):,} unique context-value pairs "
                f"({n_context_keys:,} scheme/address contexts, "
                f"{n_no_context:,} isolated no-context folders) "
                f"-> {len(clusters):,} clusters "
                f"({n_merged:,} merged, threshold={similarity_threshold})"
            )
            if backfilled > 0:
                logger.info(
                    f"    backfilled {backfilled:,} rows from raw matches"
                )

            # Log top multi-member clusters
            multi_member = [
                (canonical_pair, members)
                for canonical_pair, members in clusters
                if len(members) > 1
            ]
            for canonical_pair, members in sorted(
                multi_member, key=lambda x: -len(x[1])
            )[:5]:
                canonical_val = canonical_pair[1]
                variants = ", ".join(
                    sorted({v for _, v in members} - {canonical_val})
                )
                logger.info(
                    f"    \"{canonical_val}\" ({len(members)} variants: {variants})"
                )

            continue

        # ------------------------------------------------------------------ #
        #  All other entity types: standard clustering                        #
        # ------------------------------------------------------------------ #

        value_col = f"entity_{entity_type}"
        cluster_col = f"entity_{entity_type}_cluster_id"
        canonical_col = f"entity_{entity_type}_canonical"

        if value_col not in df.columns:
            df[cluster_col] = ""
            df[canonical_col] = ""
            logger.info(f"  {entity_type}: no values to cluster")
            continue

        # Collect unique non-empty values from the validated column
        # (file-level counts — used as fallback when no raw column exists).
        # For entity types with extraction (address, plot), use extracted
        # values for a cleaner clustering pool.
        extracted_col = f"entity_{entity_type}_extracted"
        use_extracted = extracted_col in df.columns
        count_col = extracted_col if use_extracted else value_col

        validated_counts = Counter(
            v for v in df[count_col] if isinstance(v, str) and v.strip()
        )

        # Collect from raw matches, deduplicated at the folder level.
        # Each unique folder path contributes one address value to the pool.
        # This is folder-centric: 500 files under "Ash Close" = 1 folder.
        folder_counts = _collect_folder_values(df, entity_type)

        # Merge: folder counts broaden the pool; validated counts fill
        # in for entity types without raw columns (e.g. scheme).
        all_values: set[str] = set(validated_counts.keys()) | set(folder_counts.keys())
        merged_counts: Counter = Counter()
        for v in all_values:
            merged_counts[v] = max(
                validated_counts.get(v, 0), folder_counts.get(v, 0),
            )

        unique_values = list(merged_counts.keys())

        if not unique_values:
            df[cluster_col] = ""
            df[canonical_col] = ""
            logger.info(f"  {entity_type}: no values to cluster")
            continue

        n_validated = len(validated_counts)
        n_from_folders = len(folder_counts)
        n_raw_only = len(all_values) - n_validated

        # Cluster unique values
        clusters = _cluster_values(
            unique_values, merged_counts, similarity_threshold,
            entity_type=entity_type,
        )

        # Build lookup: raw value -> (cluster_id, canonical)
        prefix = _entity_prefix(entity_type)
        lookup: dict[str, tuple[str, str]] = {}
        for idx, (canonical, members) in enumerate(clusters):
            cluster_id = f"{prefix}_{idx:04d}"
            for member in members:
                lookup[member] = (cluster_id, canonical)

        # Map validated entity values to clusters.
        # For entity types with extraction, use extracted values as
        # lookup keys (since the clustering pool was built from them).
        map_col = count_col  # extracted_col for addresses, value_col otherwise
        df[cluster_col] = df[map_col].map(
            lambda v, lk=lookup: lk.get(v, ("", ""))[0]
            if isinstance(v, str) and v.strip() else ""
        )
        df[canonical_col] = df[map_col].map(
            lambda v, lk=lookup: lk.get(v, ("", ""))[1]
            if isinstance(v, str) and v.strip() else ""
        )

        # Backfill empty rows from raw matches
        backfilled = _backfill_from_raw(df, entity_type, lookup)

        n_merged = len(unique_values) - len(clusters)
        logger.info(
            f"  {entity_type}: {len(unique_values):,} unique values "
            f"({n_validated:,} validated + {n_raw_only:,} from other folders) "
            f"across {n_from_folders:,} unique folders "
            f"-> {len(clusters):,} clusters "
            f"({n_merged:,} merged, threshold={similarity_threshold})"
        )
        if backfilled > 0:
            logger.info(
                f"    backfilled {backfilled:,} rows from raw matches"
            )

        # Log top clusters by member count
        multi_member = [
            (canonical, members)
            for canonical, members in clusters
            if len(members) > 1
        ]
        for canonical, members in sorted(
            multi_member, key=lambda x: -len(x[1])
        )[:5]:
            variants = ", ".join(sorted(members - {canonical}))
            logger.info(
                f"    \"{canonical}\" ({len(members)} variants: {variants})"
            )

    # ------------------------------------------------------------------
    # Compute cluster sizes for each entity type.
    # After all clustering is done, count rows per cluster_id so that
    # each row carries the size of the cluster it belongs to.
    # ------------------------------------------------------------------
    for entity_type in entity_types:
        cluster_col = f"entity_{entity_type}_cluster_id"
        size_col = f"entity_{entity_type}_cluster_size"
        if cluster_col not in df.columns:
            df[size_col] = 0
            continue
        # Count rows per cluster_id (excluding empty)
        non_empty = df[cluster_col] != ""
        if non_empty.any():
            counts = df.loc[non_empty, cluster_col].map(
                df.loc[non_empty, cluster_col].value_counts()
            )
            df[size_col] = 0
            df.loc[non_empty, size_col] = counts
        else:
            df[size_col] = 0

    return df


# ---------------------------------------------------------------------------
# Shared normalisation and clustering utilities
# ---------------------------------------------------------------------------

def normalise_entity_value(value: str) -> str:
    """Standardised normalisation for entity values.

    Purely mechanical — no abbreviation maps:
    1. Lowercase
    2. Strip punctuation (keep hyphens and ampersands)
    3. Collapse whitespace
    4. Strip leading/trailing whitespace

    Args:
        value: Raw entity value from extraction.

    Returns:
        Normalised string.
    """
    text = value.lower()
    text = _STRIP_CHARS_RE.sub("", text)
    text = _COLLAPSE_WHITESPACE_RE.sub(" ", text)
    return text.strip()


def strip_scheme_noise(value: str) -> str:
    """Strip universal metadata noise from scheme names.

    Removes phase indicators, dates, year ranges, and admin
    annotations so that similarity comparison focuses on the
    core scheme identity (location / development name).

    Applied only during the second-pass clustering for schemes,
    NOT to the stored canonical forms.

    Args:
        value: Normalised scheme name (lowercase, punctuation stripped).

    Returns:
        Scheme name with noise tokens removed.
    """
    text = _NEW_ANNOTATION_RE.sub("", value)
    text = _PHASE_RE.sub("", text)
    text = _DAY_MONTH_YEAR_RE.sub("", text)
    text = _MONTH_YEAR_RE.sub("", text)
    text = _DATE_RANGE_RE.sub("", text)
    text = _STANDALONE_YEAR_RE.sub("", text)
    text = _ORPHAN_SEPARATOR_RE.sub("", text)
    text = _COLLAPSE_WHITESPACE_RE.sub(" ", text)
    return text.strip()


def _cluster_values(
    unique_values: list[str],
    value_counts: Counter,
    similarity_threshold: float,
    entity_type: str = "",
) -> list[tuple[str, set[str]]]:
    """Cluster entity values by character-trigram similarity.

    Pass 1 (all entity types): standard char-trigram Jaccard >= threshold.

    Pass 2 (scheme entities only): for pairs not yet merged, strip phase/
    date noise and check token-level overlap coefficient.  This catches
    multi-phase developments with different naming conventions
    (e.g. "Phase 2 Liberty Park Hoo Road" vs "Liberty Park Phase 6").

    Returns list of (canonical_form, set_of_member_values).
    """
    n = len(unique_values)
    if n == 0:
        return []

    # Normalise and pre-compute trigrams
    normalised = [normalise_entity_value(v) for v in unique_values]
    trigrams = [char_trigrams(nv) for nv in normalised]

    # Pass 1: standard char-trigram Jaccard
    uf = UnionFind(n)
    for i in range(n):
        for j in range(i + 1, n):
            sim = jaccard_similarity(trigrams[i], trigrams[j])
            if sim >= similarity_threshold:
                uf.union(i, j)

    # Pass 2: noise-stripped overlap for scheme entities only.
    # Merges pairs where the core name (minus phases/dates) has high
    # token overlap, even if the full strings are too different for
    # char-trigram Jaccard.
    if entity_type == "scheme":
        stripped = [strip_scheme_noise(nv) for nv in normalised]
        stripped_tokens = [set(s.split()) for s in stripped]
        stripped_trigrams = [char_trigrams(s) for s in stripped]

        for i in range(n):
            for j in range(i + 1, n):
                    # Skip if already in the same cluster
                    if uf.find(i) == uf.find(j):
                        continue

                    toks_i, toks_j = stripped_tokens[i], stripped_tokens[j]

                    # Require both sides to have >= 2 meaningful tokens
                    if len(toks_i) < 2 or len(toks_j) < 2:
                        continue

                    # Token overlap: how much of the smaller token set
                    # appears in the larger
                    tok_overlap = (
                        len(toks_i & toks_j)
                        / min(len(toks_i), len(toks_j))
                    )

                    # Char-trigram overlap on stripped forms: catches
                    # partial word matches and word-order differences
                    char_overlap = overlap_coefficient(
                        stripped_trigrams[i], stripped_trigrams[j],
                    )

                    # Merge if token overlap is very high (core name
                    # is a subset of the other), with a char-trigram
                    # sanity floor to prevent coincidental word matches
                    if tok_overlap >= 0.80 and char_overlap >= 0.60:
                        uf.union(i, j)

    # Extract clusters
    index_clusters = uf.clusters()

    result = []
    for members_indices in index_clusters:
        members = {unique_values[i] for i in members_indices}
        canonical = _pick_canonical(members, value_counts)
        result.append((canonical, members))

    # Sort by cluster size descending for stable output
    result.sort(key=lambda x: (-len(x[1]), x[0]))
    return result


def _pick_canonical(members: set[str], frequencies: Counter) -> str:
    """Pick canonical form: most frequent variant, ties broken by longest."""
    return max(
        members,
        key=lambda v: (frequencies.get(v, 0), len(v)),
    )


def _entity_prefix(entity_type: str) -> str:
    """Short prefix for cluster IDs."""
    prefixes = {
        "plot": "plot",
        "address": "addr",
        "scheme": "schm",
    }
    return prefixes.get(entity_type, entity_type[:4])
