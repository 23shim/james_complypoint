"""
Path decomposition.

Strips the configurable root prefix and splits file paths into
ordered folder segments, filename stem, and extension.

Design principle: No segment is pre-assigned a role (project,
category, etc.). The classification engine decides what each
segment means. This module just produces clean, ordered parts.

Scale: Uses vectorised string splits where possible. The single
apply() pass for extraction handles ~1M rows in a few seconds.
"""

import logging

import pandas as pd

logger = logging.getLogger(__name__)

# Common path separators — handle both Windows and forward-slash variants
_SEPARATORS = "/\\"


def decompose_paths(df: pd.DataFrame, root_prefix: str) -> pd.DataFrame:
    """Decompose full_path into structured components.

    Adds columns:
        relative_path  - path after root stripping (for debugging)
        segments        - list[str] of folder names in order
        filename_stem   - filename without extension
        extension       - lowercase, no dot (empty string if none)
        depth           - number of folder segments

    Args:
        df: DataFrame with 'full_path' column.
        root_prefix: The path prefix to strip (e.g. "\\\\server\\share\\root").

    Returns:
        DataFrame with new columns added.
    """
    df = df.copy()

    root_parts = _parse_prefix(root_prefix)
    root_depth = len(root_parts)

    logger.info(
        f"Root prefix: '{root_prefix}' ({root_depth} segments to strip)"
    )

    # Normalise separators and split all paths in one vectorised pass
    normalised = df["full_path"].astype(str).apply(_normalise_path)
    all_parts = normalised.str.split("\\")

    # Extract components — single apply pass over lists (fast)
    extracted = all_parts.apply(lambda parts: _extract(parts, root_parts, root_depth))
    result = pd.DataFrame(extracted.tolist(), index=df.index)

    for col in result.columns:
        df[col] = result[col]

    _log_summary(df)

    return df


def _normalise_path(path: str) -> str:
    """Normalise all separators to backslash."""
    return path.replace("/", "\\")


def _parse_prefix(root_prefix: str) -> list[str]:
    """Parse root prefix into lowercase parts for case-insensitive matching."""
    cleaned = root_prefix.strip().replace("/", "\\").rstrip("\\")
    return [p.lower() for p in cleaned.split("\\") if p]


def _extract(parts: list[str], root_parts: list[str], root_depth: int) -> dict:
    """Extract structured components from a split path.

    This runs once per row. Kept as a plain function (not a method)
    for apply() performance.
    """
    # Filter empty strings from split (handles leading \\)
    parts = [p for p in parts if p]

    if not parts:
        return _empty_result()

    # Strip root prefix (case-insensitive)
    parts_lower = [p.lower() for p in parts]
    if parts_lower[:root_depth] == root_parts:
        meaningful = parts[root_depth:]
    else:
        # Path doesn't match root — keep everything
        meaningful = parts

    if not meaningful:
        return _empty_result()

    # Last element is the filename; everything before it is folder segments
    filename_full = meaningful[-1]
    segments = meaningful[:-1]

    # Split filename into stem + extension
    stem, ext = _split_filename(filename_full)

    return {
        "relative_path": "\\".join(meaningful),
        "segments": segments,
        "filename_stem": stem,
        "extension": ext,
        "depth": len(segments),
    }


def _split_filename(filename: str) -> tuple[str, str]:
    """Split a filename into (stem, extension).

    Handles:
    - Normal: "report.pdf" -> ("report", "pdf")
    - No extension: "README" -> ("README", "")
    - Multiple dots: "report.v2.pdf" -> ("report.v2", "pdf")
    - Hidden files: ".gitignore" -> (".gitignore", "")
    """
    if "." not in filename or filename.startswith(".") and filename.count(".") == 1:
        return filename, ""

    last_dot = filename.rfind(".")
    stem = filename[:last_dot]
    ext = filename[last_dot + 1:].lower()

    return stem, ext


def _empty_result() -> dict:
    """Return an empty decomposition result."""
    return {
        "relative_path": "",
        "segments": [],
        "filename_stem": "",
        "extension": "",
        "depth": 0,
    }


def _log_summary(df: pd.DataFrame) -> None:
    """Log decomposition statistics."""
    depth_stats = df["depth"].describe()
    logger.info(
        f"Path decomposition complete: "
        f"{len(df):,} files, "
        f"median depth={depth_stats['50%']:.0f}, "
        f"max depth={depth_stats['max']:.0f}"
    )

    root_level = (df["depth"] == 0).sum()
    if root_level > 0:
        logger.info(f"  {root_level:,} files at root level (no folder context)")

    no_ext = (df["extension"] == "").sum()
    if no_ext > 0:
        logger.info(f"  {no_ext:,} files with no extension")

    # Show depth distribution
    depth_dist = df["depth"].value_counts().sort_index()
    logger.info("  Depth distribution:")
    for depth, count in depth_dist.items():
        pct = count / len(df) * 100
        logger.info(f"    depth {depth}: {count:,} ({pct:.1f}%)")
