"""
Row filtering with auditable summary.

Removes rows that are not classifiable documents:
folder-only rows, system junk, null paths.

Every filter logs what it removes and why. At scale,
silently dropping rows is dangerous — someone will ask
"where did those files go?"
"""

import logging
from dataclasses import dataclass, field

import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class FilterSummary:
    """Audit trail for what was removed."""

    initial_count: int = 0
    null_paths: int = 0
    folder_rows: int = 0
    system_files: int = 0
    final_count: int = 0
    details: dict[str, int] = field(default_factory=dict)

    @property
    def total_removed(self) -> int:
        return self.initial_count - self.final_count

    def log(self) -> None:
        logger.info(
            f"Filtering: {self.initial_count:,} → {self.final_count:,} "
            f"({self.total_removed:,} removed)"
        )
        if self.null_paths:
            logger.info(f"  Null/empty paths: {self.null_paths:,}")
        if self.folder_rows:
            logger.info(f"  Folder-only rows: {self.folder_rows:,}")
        if self.system_files:
            detail_str = ", ".join(
                f"{k}: {v}" for k, v in self.details.items()
            )
            logger.info(f"  System/junk files: {self.system_files:,} ({detail_str})")


def apply_filters(
    df: pd.DataFrame,
    exclusion_config: dict,
) -> tuple[pd.DataFrame, FilterSummary]:
    """Apply all filters and return cleaned DataFrame + audit summary.

    Filter order matters for correctness:
    1. Null paths first (can't inspect what doesn't exist)
    2. Folder rows (structural, based on path shape + metadata)
    3. System files (content-based, pattern matching on filenames)
    """
    summary = FilterSummary(initial_count=len(df))

    df = _remove_null_paths(df, summary)
    df = _remove_folder_rows(df, summary)
    df = _remove_system_files(df, exclusion_config, summary)

    summary.final_count = len(df)
    summary.log()

    return df, summary


def _remove_null_paths(df: pd.DataFrame, summary: FilterSummary) -> pd.DataFrame:
    """Remove rows where path is null, empty, non-string, or not a valid path.

    TreeSize exports can corrupt path values beyond certain row counts,
    replacing them with numeric 0. We catch these by requiring paths
    to be actual strings containing at least one path separator.
    """
    is_string = df["full_path"].apply(lambda x: isinstance(x, str))
    path_str = df["full_path"].astype(str).str.strip()

    valid = (
        df["full_path"].notna()
        & is_string
        & (path_str != "")
        & (path_str != "nan")
        & (path_str.str.contains("\\", regex=False) | path_str.str.contains("/", regex=False))
    )
    summary.null_paths = (~valid).sum()
    return df[valid].copy()


def _remove_folder_rows(df: pd.DataFrame, summary: FilterSummary) -> pd.DataFrame:
    """Remove folder-only rows.

    TreeSize marks folders by:
    - Path ending with backslash
    - folder_count > 0 (if the column exists)
    """
    is_folder = df["full_path"].astype(str).str.rstrip().str.endswith("\\")

    if "folder_count" in df.columns:
        has_subfolders = pd.to_numeric(df["folder_count"], errors="coerce").fillna(0) > 0
        is_folder = is_folder | has_subfolders

    summary.folder_rows = is_folder.sum()
    return df[~is_folder].copy()


def _remove_system_files(
    df: pd.DataFrame,
    exclusion_config: dict,
    summary: FilterSummary,
) -> pd.DataFrame:
    """Remove system artifacts and junk files by pattern matching."""
    # Extract just the filename from the full path
    filenames = df["full_path"].astype(str).str.rsplit("\\", n=1).str[-1]
    filenames_lower = filenames.str.lower()

    exclude_mask = pd.Series(False, index=df.index)

    # Exact filename matches (case-insensitive)
    exact_names = exclusion_config.get("filenames", [])
    if exact_names:
        exact_lower = {n.lower() for n in exact_names}
        match = filenames_lower.isin(exact_lower)
        if match.any():
            summary.details["exact_filename"] = int(match.sum())
        exclude_mask = exclude_mask | match

    # Prefix matches (e.g. ~$ for Office temp files)
    for prefix in exclusion_config.get("prefixes", []):
        match = filenames.str.startswith(prefix)
        if match.any():
            summary.details[f"prefix:'{prefix}'"] = int(match.sum())
        exclude_mask = exclude_mask | match

    # Extension matches (case-insensitive)
    excluded_exts = exclusion_config.get("extensions", [])
    if excluded_exts:
        # Extract extension: everything after the last dot
        exts = filenames_lower.str.rsplit(".", n=1).str[-1]
        ext_lower = {e.lower() for e in excluded_exts}
        match = exts.isin(ext_lower)
        if match.any():
            summary.details["extension"] = int(match.sum())
        exclude_mask = exclude_mask | match

    summary.system_files = int(exclude_mask.sum())
    return df[~exclude_mask].copy()
