"""
Cross-estate duplicate file detection (metadata-only).

Identifies duplicate and near-duplicate files across estates using
metadata signals available from TreeSize exports: filename, extension,
file size, last_modified, owner, and folder path.

Three detection tiers (highest → lowest confidence):

1. Exact metadata match — (filename + extension + size).
   Confidence boosted if last_modified and/or owner also match.
3. Folder-level duplication — Jaccard similarity on per-folder
   filename sets.  Computed at report time (no columns added).
4. Copy-path heuristics — flag files in backup/archive/copy paths
   where a matching file exists elsewhere.

Runs as a post-processing step on the combined multi-estate DataFrame,
after entity clustering.
"""

from __future__ import annotations

import hashlib
import logging
from collections import defaultdict

import pandas as pd

logger = logging.getLogger(__name__)

# Duplicate detection defaults — overridden by weights.yaml "duplicate_detection" section.
_DUP_DEFAULTS = {
    "base_confidence": 0.85,
    "date_match_boost": 0.10,
    "owner_match_boost": 0.05,
    "copy_path_keywords": [
        "backup", "backups", "archive", "archived", "copy", "copies",
        "old", "temp", "tmp", "desktop", "bak",
    ],
}


def _load_dup_weights(weights: dict) -> dict:
    """Load duplicate detection weights from config, falling back to defaults."""
    overrides = weights.get("duplicate_detection", {})
    result = {k: overrides.get(k, v) for k, v in _DUP_DEFAULTS.items()}
    # Ensure copy_path_keywords is a frozenset for O(1) lookup
    result["copy_path_keywords"] = frozenset(result["copy_path_keywords"])
    return result


# ====================================================================
# Main entry point
# ====================================================================

def detect_duplicates(
    df: pd.DataFrame,
    weights: dict | None = None,
) -> pd.DataFrame:
    """Detect duplicate files across all estates.

    Adds columns to the DataFrame for downstream reporting:

      Tier 1 (exact):
        dup_group_id          — deterministic hash for the group
        dup_group_size        — number of files in the exact group
        dup_confidence        — 0.85 base, boosted by date/owner match
        dup_is_cross_estate   — group spans multiple source_systems
        dup_is_cross_folder   — group spans multiple parent folders
        dup_is_latest         — True for the newest file(s) in the group
                                (all True when dates match or are null)

      Derived:
        potential_duplicate   — True when dup_group_size > 1

      Tier 4 (copy paths):
        dup_copy_path_suspect — file sits in a backup/archive/copy path

    Tier 3 (folder duplication) is computed at report time — no columns.
    """
    logger.info("Starting cross-estate duplicate detection...")

    dw = _load_dup_weights(weights or {})

    df = df.copy()

    # Initialise all columns with defaults
    df["dup_group_id"] = ""
    df["dup_group_size"] = 1
    df["dup_confidence"] = 0.0
    df["dup_is_cross_estate"] = False
    df["dup_is_cross_folder"] = False
    df["dup_is_latest"] = True
    df["potential_duplicate"] = False
    df["dup_copy_path_suspect"] = False

    # Tier 1: exact metadata match
    df = _detect_exact_duplicates(df, dw)

    # Tier 4: copy-path heuristics
    df = _detect_copy_path_suspects(df, dw)

    # Derived fields
    df["potential_duplicate"] = df["dup_group_size"] > 1

    return df


# ====================================================================
# Tier 1 — Exact metadata match
# ====================================================================

def _normalise_size(size) -> int:
    """Normalise a file size value to int bytes.  Returns -1 on failure."""
    try:
        return int(float(size))
    except (TypeError, ValueError):
        return -1


def _detect_exact_duplicates(df: pd.DataFrame, dw: dict) -> pd.DataFrame:
    """Group files by (filename_stem, extension, size).

    Files with identical names, types, and byte sizes are flagged as
    probable exact duplicates.
    """
    if "filename_stem" not in df.columns:
        logger.warning("  Tier 1 skipped — no filename_stem column")
        return df

    # Build composite key (vectorised where possible)
    stems = df["filename_stem"].fillna("").str.lower().str.strip()
    exts = df["extension"].fillna("").str.lower().str.strip() if "extension" in df.columns else pd.Series("", index=df.index)
    sizes = df["size"].apply(_normalise_size) if "size" in df.columns else pd.Series(-1, index=df.index)

    # Key is only valid when stem is non-empty, size >= 0, and stem
    # contains at least one alphanumeric char (excludes "*.*" placeholders)
    valid = (stems != "") & (sizes >= 0) & stems.str.contains(r"[a-z0-9]", na=False)
    keys = stems + "|" + exts + "|" + sizes.astype(str)
    keys[~valid] = ""
    df["_exact_key"] = keys

    # Find groups with 2+ members
    key_counts = df["_exact_key"].value_counts()
    dup_keys = set(key_counts[key_counts >= 2].index) - {""}

    if not dup_keys:
        logger.info("  Tier 1 — Exact duplicates: 0 groups")
        df.drop(columns=["_exact_key"], inplace=True)
        return df

    # Process each duplicate group
    dup_mask = df["_exact_key"].isin(dup_keys)
    dup_subset = df[dup_mask]

    group_ids: dict[int, str] = {}
    group_sizes: dict[int, int] = {}
    confidences: dict[int, float] = {}
    cross_estate: dict[int, bool] = {}
    cross_folder: dict[int, bool] = {}

    n_groups = 0
    n_files = 0
    n_cross_estate = 0

    for key, group in dup_subset.groupby("_exact_key"):
        n_groups += 1
        n_files += len(group)
        gid = hashlib.sha256(key.encode()).hexdigest()[:12]
        size = len(group)

        # Confidence scoring
        conf = dw["base_confidence"]

        if "last_modified" in group.columns:
            dates = group["last_modified"].dropna()
            if len(dates) == size and dates.nunique() == 1:
                conf += dw["date_match_boost"]

        if "owner" in group.columns:
            owners = group["owner"].dropna()
            if len(owners) == size and owners.nunique() == 1:
                conf += dw["owner_match_boost"]

        conf = min(conf, 1.0)

        # Cross-estate
        is_ce = False
        if "source_system" in group.columns:
            is_ce = group["source_system"].nunique() > 1
            if is_ce:
                n_cross_estate += 1

        # Cross-folder
        is_cf = _has_multiple_folders(group)

        for idx in group.index:
            group_ids[idx] = gid
            group_sizes[idx] = size
            confidences[idx] = conf
            cross_estate[idx] = is_ce
            cross_folder[idx] = is_cf

    # Apply results
    df["dup_group_id"] = df.index.map(lambda i: group_ids.get(i, ""))
    df["dup_group_size"] = df.index.map(lambda i: group_sizes.get(i, 1))
    df["dup_confidence"] = df.index.map(lambda i: confidences.get(i, 0.0))
    df["dup_is_cross_estate"] = df.index.map(lambda i: cross_estate.get(i, False))
    df["dup_is_cross_folder"] = df.index.map(lambda i: cross_folder.get(i, False))

    # dup_is_latest: within each group, the row(s) with the newest
    # last_modified get True, others get False.  When all dates match
    # (or are null), all rows get True since they're indistinguishable.
    is_latest: dict[int, bool] = {}
    if "last_modified" in df.columns:
        for key, group in dup_subset.groupby("_exact_key"):
            dates = pd.to_datetime(group["last_modified"], errors="coerce")
            max_date = dates.max()
            if pd.isna(max_date) or dates.nunique() <= 1:
                # All dates match or all null — all are equally "latest"
                for idx in group.index:
                    is_latest[idx] = True
            else:
                for idx in group.index:
                    is_latest[idx] = dates.at[idx] == max_date

    df["dup_is_latest"] = df.index.map(lambda i: is_latest.get(i, True))

    df.drop(columns=["_exact_key"], inplace=True)

    logger.info(
        f"  Tier 1 — Exact duplicates: {n_groups:,} groups "
        f"({n_files:,} files, {n_cross_estate:,} cross-estate)"
    )
    return df


# ====================================================================
# Tier 4 — Copy-path heuristics
# ====================================================================

def _detect_copy_path_suspects(df: pd.DataFrame, dw: dict) -> pd.DataFrame:
    """Flag files in backup/archive/copy paths.

    A file is a copy-path suspect if:
      1. It sits under a folder whose name matches a copy keyword, AND
      2. A file with the same (filename_stem, extension) exists in a
         non-copy path somewhere in the combined dataset.
    """
    if "segments" not in df.columns or "filename_stem" not in df.columns:
        logger.warning("  Tier 4 skipped — missing segments or filename_stem")
        return df

    copy_keywords = dw["copy_path_keywords"]

    # Identify rows in copy-like paths
    def _in_copy_path(segments) -> bool:
        if not isinstance(segments, list):
            return False
        for seg in segments:
            if isinstance(seg, str) and seg.lower().strip() in copy_keywords:
                return True
        return False

    copy_mask = df["segments"].apply(_in_copy_path)

    if not copy_mask.any():
        logger.info("  Tier 4 — Copy-path suspects: 0 files")
        return df

    # Build lookup of files NOT in copy paths
    non_copy = df[~copy_mask]
    exts = non_copy["extension"].fillna("").str.lower() if "extension" in non_copy.columns else pd.Series("", index=non_copy.index)
    non_copy_file_keys = set(
        non_copy["filename_stem"].fillna("").str.lower() + "|" + exts
    )

    # Check each copy-path file against the non-copy set
    copy_rows = df[copy_mask]
    copy_exts = copy_rows["extension"].fillna("").str.lower() if "extension" in copy_rows.columns else pd.Series("", index=copy_rows.index)
    copy_file_keys = copy_rows["filename_stem"].fillna("").str.lower() + "|" + copy_exts

    suspect_mask = copy_file_keys.isin(non_copy_file_keys)
    suspect_indices = copy_rows.index[suspect_mask]

    df.loc[suspect_indices, "dup_copy_path_suspect"] = True

    logger.info(
        f"  Tier 4 — Copy-path suspects: {len(suspect_indices):,} files "
        f"(of {copy_mask.sum():,} in copy-like paths)"
    )
    return df


# ====================================================================
# Helpers
# ====================================================================

def _has_multiple_folders(group: pd.DataFrame) -> bool:
    """Check if group members span multiple parent folders."""
    if "segments" not in group.columns:
        return False
    folders = set()
    for _, row in group.iterrows():
        segs = row.get("segments")
        if isinstance(segs, list) and segs:
            folders.add(tuple(segs))
        if len(folders) > 1:
            return True
    return False
