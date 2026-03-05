"""
Scheme detection via structural inference.

Simplified algorithm:
1. Aggregate unique plot/address entity counts per folder.
2. Gate: folder name must contain a place name OR address match.
3. Exclude category/type folders, entity-only folders, config exclusions.
4. Score: base confidence from entity count + small boosts.
5. Remove containers (folders whose children are other schemes).
6. Assign each file the shallowest valid scheme ancestor.

This runs as a post-processing step AFTER per-row entity extraction.
"""

from __future__ import annotations

import logging
import re
from collections import defaultdict
from dataclasses import dataclass

import pandas as pd

from classification.matcher import match_category_signals, match_type_signals
from classification.models import ClassificationConfig, MatchMethod, SchemeExclusions, SignalSource
from classification.entity_extractor import extract_entities_with_confidence
from classification.tokeniser import tokenise

logger = logging.getLogger(__name__)

# Scheme detection defaults — overridden by weights.yaml "scheme_detection" section.
_SCHEME_DEFAULTS = {
    "min_entities": 3,
    "high_entity_threshold": 5,
    "high_entity_confidence": 0.85,
    "med_entity_confidence": 0.70,
    "direct_plot_children_boost": 0.10,
    "development_categories_boost": 0.10,
    "min_scheme_name_length": 3,
    "max_entity_only_tokens": 3,
    "sibling_quality_threshold": 0.50,
}


def _load_scheme_weights(weights: dict) -> dict:
    """Load scheme detection weights from config, falling back to defaults."""
    overrides = weights.get("scheme_detection", {})
    return {k: overrides.get(k, v) for k, v in _SCHEME_DEFAULTS.items()}


# Plot-number detector for long folder names.  If a name contains a specific
# unit reference (Plot 16, Flat 3, Unit 5A) ANYWHERE, it identifies a single
# unit — not a scheme — regardless of how many other tokens surround it.
_PLOT_NUMBER_RE = re.compile(
    r"\b(?:plot|flat|unit|apt)\s*[-–_ ]?\s*\d{1,4}[a-z]?\b",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class SchemeMatch:
    """A detected scheme assigned to a file."""
    value: str        # The folder name identified as the scheme
    confidence: float # Confidence score
    depth: int        # Folder depth where the scheme sits
    folder_path: str  # Full relative folder path of the scheme

    # Diagnostic fields (kept for container detection and reporting)
    place_name_match: bool = False
    address_like_name: bool = False
    direct_category_children: int = 0
    direct_plot_children: int = 0


@dataclass
class _FolderStats:
    """Aggregated entity stats for a single folder path."""
    plots: set[str]
    addresses: set[str]
    file_count: int
    direct_plot_children: int  # folders one level below that ARE plot entities
    category_signals: set[str]  # development categories from ALL descendants
    direct_category_children: int  # immediate child folders matching dev categories
    total_direct_children: int = 0  # all direct child folders in hierarchy

    @property
    def unique_entities(self) -> int:
        return len(self.plots) + len(self.addresses)


def detect_and_assign_schemes(
    df: pd.DataFrame,
    config: ClassificationConfig,
    min_entities: int | None = None,
    folder_categories: dict[str, set[str]] | None = None,
    root_depth: int = -1,
) -> tuple[pd.DataFrame, dict[str, "SchemeMatch"]]:
    """Detect schemes estate-wide and assign to each file row.

    Expects df to already have:
      - segments: list[str] (from path_parser)
      - entity_plot: str (from entity extraction)
      - entity_address: str (from entity extraction)

    Adds columns:
      - entity_scheme: str (scheme folder name, or "")
      - entity_scheme_confidence: float
      - entity_scheme_path: str (full folder path of the scheme)
      - entity_scheme_depth: int (folder depth, or -1 if no scheme)
      - entity_scheme_place_name_match: bool
      - entity_scheme_address_like_name: bool

    Returns:
        (DataFrame with scheme columns, all_candidates dict for sidecar).
    """
    sw = _load_scheme_weights(config.weights)
    min_entities = min_entities if min_entities is not None else sw["min_entities"]
    logger.info("Starting scheme detection...")

    # Step 1: Aggregate entity counts per folder path
    folder_stats = _aggregate_folder_stats(df, folder_categories or {})
    logger.info(f"  Unique folder paths: {len(folder_stats):,}")

    # Step 1b: Enrich with direct category children counts
    dev_categories = {
        name for name, cat_def in config.categories.items()
        if cat_def.scheme_signal
    }
    _compute_direct_category_children(
        folder_stats, folder_categories or {}, dev_categories,
    )

    # Step 1c: Count total direct children per folder
    _compute_total_direct_children(folder_stats)

    # Step 2: Identify scheme candidates (filtered)
    candidates, pre_filter_excluded = _identify_candidates(folder_stats, config, min_entities)
    logger.info(f"  Scheme candidates after filtering: {len(candidates):,}")

    # Step 3: Score candidates
    scored_schemes = _score_candidates(candidates, config, root_depth=root_depth)
    logger.info(f"  Scored scheme candidates: {len(scored_schemes):,}")

    # Snapshot all scored candidates BEFORE exclusion for the sidecar
    pre_exclusion_schemes: dict[str, SchemeMatch] = dict(scored_schemes)

    # Step 3b: Remove container folders (parents of other schemes)
    scored_schemes, excluded_containers, excluded_subsumed = \
        _exclude_container_folders(scored_schemes, config)

    # Step 3c: Deduplicate same-name schemes at different paths
    scored_schemes = _deduplicate_schemes(scored_schemes)

    # Step 4: Assign each file its shallowest scheme ancestor
    schemes = []
    confidences = []
    paths = []
    depths = []
    place_matches = []
    addr_likes = []

    for _, row in df.iterrows():
        segs = row.get("segments")
        if not isinstance(segs, list) or not segs:
            match = None
        else:
            match = _assign_scheme_to_row(segs, scored_schemes)

        if match:
            schemes.append(match.value)
            confidences.append(match.confidence)
            paths.append(match.folder_path)
            depths.append(match.depth)
            place_matches.append(match.place_name_match)
            addr_likes.append(match.address_like_name)
        else:
            schemes.append("")
            confidences.append(0.0)
            paths.append("")
            depths.append(-1)
            place_matches.append(False)
            addr_likes.append(False)

    df = df.copy()
    df["entity_scheme"] = schemes
    df["entity_scheme_confidence"] = confidences
    df["entity_scheme_path"] = paths
    df["entity_scheme_depth"] = depths
    df["entity_scheme_place_name_match"] = place_matches
    df["entity_scheme_address_like_name"] = addr_likes

    assigned = sum(1 for s in schemes if s)
    logger.info(
        f"  Scheme assignment: {assigned:,}/{len(df):,} files "
        f"({assigned / len(df) * 100:.1f}%)"
    )

    # Build candidate map for the sidecar.
    all_candidates: dict[str, tuple[SchemeMatch, str, int]] = {}

    # Pre-filter excluded: zero-score stubs so they appear in the report
    for path, (reason, unique_ents, dev_cat_cnt) in pre_filter_excluded.items():
        stub_parts = path.split("\\")
        stub = SchemeMatch(
            value=stub_parts[-1],
            confidence=0.0,
            depth=len(stub_parts),
            folder_path=path,
        )
        all_candidates[path] = (stub, reason, unique_ents)

    # Scored candidates: tagged with their outcome
    for path, match in pre_exclusion_schemes.items():
        if path in scored_schemes:
            status = "chosen"
        elif path in excluded_containers:
            status = "container"
        elif path in excluded_subsumed:
            status = "subsumed"
        else:
            status = "unknown"
        all_candidates[path] = (match, status, 0)

    return df, all_candidates


def _aggregate_folder_stats(
    df: pd.DataFrame,
    folder_categories: dict[str, set[str]],
) -> dict[str, _FolderStats]:
    """Build entity counts for every folder path in the estate."""
    stats: dict[str, _FolderStats] = {}

    # Track which folders directly contain plot-entity children
    plot_entity_depths: dict[str, set[int]] = defaultdict(set)

    for _, row in df.iterrows():
        segs = row.get("segments")
        if not isinstance(segs, list) or not segs:
            continue

        plot = row.get("entity_plot", "")
        address = row.get("entity_address", "")

        # Track plot entity depth for "above plots" detection
        plot_depth = row.get("entity_plot_depth", -1)
        if plot and isinstance(plot_depth, (int, float)) and plot_depth >= 0:
            for level in range(1, len(segs) + 1):
                folder_key = "\\".join(segs[:level])
                folder_depth = level - 1
                if folder_depth == int(plot_depth) - 1:
                    plot_entity_depths[folder_key].add(int(plot_depth))

        # Propagate entities to all ancestor folders
        for level in range(1, len(segs) + 1):
            folder_key = "\\".join(segs[:level])
            if folder_key not in stats:
                stats[folder_key] = _FolderStats(
                    plots=set(), addresses=set(), file_count=0,
                    direct_plot_children=0,
                    category_signals=set(),
                    direct_category_children=0,
                )
            stats[folder_key].file_count += 1
            if plot:
                stats[folder_key].plots.add(plot)
            if address:
                stats[folder_key].addresses.add(address)

    # Set direct_plot_children counts
    for folder_key, depths in plot_entity_depths.items():
        if folder_key in stats:
            stats[folder_key].direct_plot_children = len(depths)

    # Propagate category signals to ancestor folders.
    for seg_path, categories in folder_categories.items():
        parts = seg_path.split("\\")
        for level in range(1, len(parts) + 1):
            ancestor_key = "\\".join(parts[:level])
            if ancestor_key in stats:
                stats[ancestor_key].category_signals.update(categories)

    return stats


def _compute_direct_category_children(
    folder_stats: dict[str, _FolderStats],
    folder_categories: dict[str, set[str]],
    dev_categories: set[str],
) -> None:
    """Count immediate child folders with development categories.

    Mutates folder_stats in place.
    """
    for seg_path, categories in folder_categories.items():
        if not (categories & dev_categories):
            continue

        sep_idx = seg_path.rfind("\\")
        if sep_idx < 0:
            continue
        parent_path = seg_path[:sep_idx]

        if parent_path in folder_stats:
            folder_stats[parent_path].direct_category_children += 1


def _compute_total_direct_children(
    folder_stats: dict[str, _FolderStats],
) -> None:
    """Count total direct child folders for each folder.

    Mutates folder_stats in place.
    """
    for path in folder_stats:
        sep_idx = path.rfind("\\")
        if sep_idx >= 0:
            parent = path[:sep_idx]
            if parent in folder_stats:
                folder_stats[parent].total_direct_children += 1


def _contains_place_name(
    folder_name: str, place_names: set[str],
) -> bool:
    """Check if a folder name contains a known UK place name."""
    if not place_names:
        return False

    tokens = tokenise(folder_name)
    meaningful = [t.lower() for t in tokens if any(c.isalnum() for c in t)]

    for token in meaningful:
        if token in place_names:
            return True

    for i in range(len(meaningful) - 1):
        bigram = meaningful[i] + " " + meaningful[i + 1]
        if bigram in place_names:
            return True

    return False


def _is_category_or_type_folder(
    folder_name: str, config: ClassificationConfig,
) -> bool:
    """Check if a folder name matches a known category OR document type.

    Guards:
    0. Place name present → NOT a category folder.
    1. Long name (4+ tokens): category abbreviations suppressed.
    2. Long name (4+ tokens): single-word base type tokens suppressed.
    """
    if config.place_names and _contains_place_name(folder_name, config.place_names):
        return False

    tokens = tokenise(folder_name)
    meaningful = [t for t in tokens if any(c.isalnum() for c in t)]
    is_long_name = len(meaningful) >= 4

    cat_signals = match_category_signals(folder_name, config, depth=0)
    if is_long_name:
        cat_signals = [
            s for s in cat_signals
            if s.match_method != MatchMethod.ABBREVIATION
        ]
    if cat_signals:
        return True

    type_signals = match_type_signals(
        folder_name, config, source=SignalSource.FOLDER_TYPE, depth=0,
    )
    if is_long_name:
        type_signals = [
            s for s in type_signals
            if not (
                s.match_method == MatchMethod.TOKEN
                and " " not in s.match_term
                and s.origin_layer == 0
            )
        ]
    if type_signals:
        return True

    return False


# Plot tokens that can also serve as scheme-level development identifiers.
_SCHEME_COMPATIBLE_PLOT_TOKENS = {"phase", "block"}


def _is_entity_only_folder(
    folder_name: str, config: ClassificationConfig,
) -> bool:
    """Check if a folder is ONLY a plot/address entity (not a scheme).

    Returns True only for plot-type entities, since addresses CAN be schemes.
    """
    sw = _load_scheme_weights(config.weights)
    tokens = tokenise(folder_name)
    meaningful = [t for t in tokens if any(c.isalnum() for c in t)]
    if len(meaningful) > sw["max_entity_only_tokens"]:
        if _PLOT_NUMBER_RE.search(folder_name):
            return True
        return False

    if any(t in _SCHEME_COMPATIBLE_PLOT_TOKENS for t in meaningful):
        return False

    matches = extract_entities_with_confidence([folder_name], config)
    plot_match = matches.get("plot")
    return plot_match is not None


def _is_excluded_by_config(
    folder_name: str, exclusions: SchemeExclusions,
) -> bool:
    """Check if folder name is excluded by config-driven rules."""
    name_lower = folder_name.lower().strip()
    stripped_lower = re.sub(r"^\d+[\.\s]+", "", name_lower).strip()

    if name_lower in exclusions.names or stripped_lower in exclusions.names:
        return True

    for sub in exclusions.substrings:
        if sub in name_lower:
            return True

    for pattern in exclusions.compiled_patterns:
        if pattern.search(folder_name):
            return True

    return False


def _is_address_like_name(
    folder_name: str, config: ClassificationConfig,
) -> bool:
    """Check if a folder name matches an address entity."""
    matches = extract_entities_with_confidence([folder_name], config)
    return "address" in matches


def _identify_candidates(
    folder_stats: dict[str, _FolderStats],
    config: ClassificationConfig,
    min_entities: int,
) -> tuple[dict[str, _FolderStats], dict[str, tuple[str, int, int]]]:
    """Filter folder stats down to scheme candidates.

    Gate: place name OR address match in folder name.
    Structure: enough entities or development categories beneath.
    """
    sw = _load_scheme_weights(config.weights)
    candidates = {}
    pre_filter_excluded: dict[str, tuple[str, int, int]] = {}
    exclusions = config.scheme_exclusions

    dev_categories = {
        name for name, cat_def in config.categories.items()
        if cat_def.scheme_signal
    }

    for folder_path, fstats in folder_stats.items():
        # Structural gate: enough entities or development categories
        dev_cats = fstats.category_signals & dev_categories
        has_enough_entities = fstats.unique_entities >= min_entities
        has_entity_plus_categories = (
            fstats.unique_entities >= 1 and len(dev_cats) >= 3
        )
        has_strong_category_signal = len(dev_cats) >= 4

        if not (
            has_enough_entities
            or has_entity_plus_categories
            or has_strong_category_signal
        ):
            continue

        parts = folder_path.split("\\")
        folder_name = parts[-1]

        _pfe = (fstats.unique_entities, len(dev_cats))

        # Too short to be a scheme name
        if len(folder_name.strip()) < sw["min_scheme_name_length"]:
            pre_filter_excluded[folder_path] = ("too_short", *_pfe)
            continue

        # Config-driven exclusions
        if _is_excluded_by_config(folder_name, exclusions):
            pre_filter_excluded[folder_path] = ("cfg_excl", *_pfe)
            continue

        # Exclude category/type folders
        if _is_category_or_type_folder(folder_name, config):
            pre_filter_excluded[folder_path] = ("cat_folder", *_pfe)
            continue

        # Exclude folders nested inside a category/type ancestor
        if any(
            _is_category_or_type_folder(parts[i], config)
            and folder_stats.get(
                "\\".join(parts[: i + 1]),
                _FolderStats(plots=set(), addresses=set(), file_count=0,
                             direct_plot_children=0, category_signals=set(),
                             direct_category_children=0),
            ).direct_plot_children == 0
            for i in range(len(parts) - 1)
        ):
            pre_filter_excluded[folder_path] = ("nested", *_pfe)
            continue

        # Exclude pure plot-entity folders
        if _is_entity_only_folder(folder_name, config):
            pre_filter_excluded[folder_path] = ("entity_only", *_pfe)
            continue

        # Gate: require place name OR address signal in folder name
        if config.place_names:
            has_place = _contains_place_name(folder_name, config.place_names)
            has_addr = _is_address_like_name(folder_name, config)
            if not has_place and not has_addr:
                pre_filter_excluded[folder_path] = ("no_location", *_pfe)
                continue

        candidates[folder_path] = fstats

    return candidates, pre_filter_excluded


# ---- Scoring ----


def _score_candidates(
    candidates: dict[str, _FolderStats],
    config: ClassificationConfig,
    root_depth: int = -1,
) -> dict[str, SchemeMatch]:
    """Score each scheme candidate.

    Simple scoring:
      - Base confidence: 0.85 (>=5 entities) or 0.70 (<5)
      - +0.10 if direct plot children present
      - +0.10 if >=3 development categories

    Container hard-exclusions are applied inline to prevent
    obvious containers from becoming scheme candidates.
    """
    sw = _load_scheme_weights(config.weights)
    scored: dict[str, SchemeMatch] = {}

    dev_categories = {
        name for name, cat_def in config.categories.items()
        if cat_def.scheme_signal
    }

    for folder_path, fstats in candidates.items():
        parts = folder_path.split("\\")
        folder_name = parts[-1]
        depth = len(parts) - 1

        # Never score the estate root as a scheme
        if depth <= root_depth:
            continue

        # Base confidence from entity count
        if fstats.unique_entities >= sw["high_entity_threshold"]:
            score = sw["high_entity_confidence"]
        else:
            score = sw["med_entity_confidence"]

        # Boost: sits one level above plot entity folders
        if fstats.direct_plot_children > 0:
            score += sw["direct_plot_children_boost"]

        # Boost: has multiple development-category subfolders
        dev_cats = fstats.category_signals & dev_categories
        if len(dev_cats) >= 3:
            score += sw["development_categories_boost"]

        # Hard exclusion: structural container detection.
        # A folder where the majority of children are scheme candidates
        # is a container, not a genuine scheme.
        total_children = fstats.total_direct_children
        child_candidate_count = 0
        prefix = folder_path + "\\"
        for other_path in candidates:
            if other_path.startswith(prefix):
                remainder = other_path[len(prefix):]
                if "\\" not in remainder:
                    child_candidate_count += 1

        if child_candidate_count >= 3 and total_children > 0:
            candidate_ratio = child_candidate_count / total_children
            if candidate_ratio > 0.5:
                logger.info(
                    f"  Structural container excluded: {folder_name} "
                    f"({child_candidate_count}/{total_children} children "
                    f"are scheme candidates)"
                )
                continue

        # Hard exclusion: geographic/organisational sub-container.
        non_candidate_children = total_children - child_candidate_count
        if (
            child_candidate_count >= 1
            and fstats.direct_category_children == 0
            and fstats.direct_plot_children == 0
            and non_candidate_children == 0
        ):
            logger.info(
                f"  Sub-container excluded: {folder_name} "
                f"(has {child_candidate_count} child scheme(s) but "
                f"no direct category/plot children)"
            )
            continue

        # Diagnostic signals for this candidate
        place_match = _contains_place_name(folder_name, config.place_names)
        addr_like = _is_address_like_name(folder_name, config)

        score = max(0.0, min(1.0, score))

        scored[folder_path] = SchemeMatch(
            value=folder_name,
            confidence=round(score, 3),
            depth=depth,
            folder_path=folder_path,
            place_name_match=place_match,
            address_like_name=addr_like,
            direct_category_children=fstats.direct_category_children,
            direct_plot_children=fstats.direct_plot_children,
        )

    return scored


# A parent with this many direct child schemes is always a container
_CONTAINER_CHILD_THRESHOLD = 3


def _exclude_container_folders(
    scored_schemes: dict[str, SchemeMatch],
    config: ClassificationConfig,
) -> tuple[dict[str, SchemeMatch], set[str], set[str]]:
    """Resolve parent/child scheme conflicts.

    - Any quality child scores higher → parent is a container, remove it.
    - All children score lower, but 3+ quality children → parent is
      still a container (inflated score from aggregated entities).
    - 1–2 quality children, all lower → parent wins if it has direct
      structure (category/plot children), otherwise parent is a container.
    """
    sw = _load_scheme_weights(config.weights)
    containers = set()
    subsumed = set()
    scheme_paths = set(scored_schemes.keys())

    for path in scheme_paths:
        parent_conf = scored_schemes[path].confidence
        direct_children = []

        for other_path in scheme_paths:
            if other_path == path:
                continue
            if other_path.startswith(path + "\\"):
                remainder = other_path[len(path) + 1:]
                if "\\" not in remainder:
                    direct_children.append(other_path)

        if not direct_children:
            continue

        quality_children = [
            c for c in direct_children
            if scored_schemes[c].confidence > sw["sibling_quality_threshold"]
        ]

        any_quality_child_higher = any(
            scored_schemes[c].confidence > parent_conf
            for c in quality_children
        )

        if any_quality_child_higher:
            containers.add(path)
        elif len(quality_children) >= _CONTAINER_CHILD_THRESHOLD:
            containers.add(path)
        elif quality_children:
            parent_match = scored_schemes[path]
            parent_has_structure = (
                parent_match.direct_category_children > 0
                or parent_match.direct_plot_children > 0
            )
            if parent_has_structure:
                subsumed.update(quality_children)
                subsumed.update(c for c in direct_children if c not in quality_children)
            else:
                containers.add(path)

    excluded = containers | subsumed

    if containers:
        logger.info(
            f"  Container exclusion: removed {len(containers)} container folders"
        )
    if subsumed:
        logger.info(
            f"  Container exclusion: suppressed {len(subsumed)} "
            f"subsumed child schemes"
        )

    return (
        {p: m for p, m in scored_schemes.items() if p not in excluded},
        containers,
        subsumed,
    )


def _deduplicate_schemes(
    scored_schemes: dict[str, SchemeMatch],
) -> dict[str, SchemeMatch]:
    """Merge duplicate scheme names that appear at multiple paths.

    Groups by name, picks the primary (highest confidence, then
    shallowest depth), and redirects duplicates to the primary.
    """
    by_name: dict[str, list[str]] = defaultdict(list)
    for path, match in scored_schemes.items():
        by_name[match.value].append(path)

    merged_count = 0
    result = dict(scored_schemes)

    for name, paths in by_name.items():
        if len(paths) < 2:
            continue

        paths.sort(
            key=lambda p: (-scored_schemes[p].confidence, scored_schemes[p].depth),
        )
        primary_match = scored_schemes[paths[0]]

        for dup_path in paths[1:]:
            result[dup_path] = primary_match
            merged_count += 1

    if merged_count:
        logger.info(
            f"  Deduplication: {merged_count} duplicate scheme entries "
            f"merged into primary instances"
        )

    return result


def _assign_scheme_to_row(
    segments: list[str],
    scored_schemes: dict[str, SchemeMatch],
) -> SchemeMatch | None:
    """Find the shallowest valid scheme ancestor for a file's path.

    Walks top-down (shallowest to deepest). First match wins.
    """
    for level in range(1, len(segments) + 1):
        folder_key = "\\".join(segments[:level])
        if folder_key in scored_schemes:
            return scored_schemes[folder_key]

    return None


def build_score_trace(m: SchemeMatch, weights: dict | None = None) -> str:
    """Build a compact human-readable scoring trace for a SchemeMatch.

    Example: base=0.85; dir_plots=+0.10; place_name=yes → 0.950
    """
    sw = _load_scheme_weights(weights) if weights else _SCHEME_DEFAULTS
    parts: list[str] = []

    # Base confidence
    if m.confidence >= sw["high_entity_confidence"]:
        parts.append(f"base={sw['high_entity_confidence']:.2f}")
    else:
        parts.append(f"base={sw['med_entity_confidence']:.2f}")

    if m.direct_plot_children > 0:
        parts.append(f"dir_plots({m.direct_plot_children})=+{sw['direct_plot_children_boost']:.2f}")

    if m.place_name_match:
        parts.append("place_name=yes")

    if m.address_like_name:
        parts.append("addr_match=yes")

    return "; ".join(parts) + f" → {m.confidence:.3f}"
