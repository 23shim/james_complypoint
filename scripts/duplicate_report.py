"""
Generate a cross-estate duplicate detection report from classified JSONL.

Outputs a single text file in the same style as cluster_review.py with
three sections:

  1. Overview — summary statistics
  2. Tier 1  — exact metadata duplicates (filename + ext + size)
  3. Tier 3  — duplicate folders (Jaccard similarity, computed here)
  4. Tier 4  — copy-path suspects

Each duplicate group lists its members with size and last_modified,
following the cluster_review layout.

Performance: designed for 500K+ rows.  All heavy lifting uses vectorised
pandas/groupby operations — no df.iterrows() on the full dataset.
"""

import sys
import os

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import json
import time
from pathlib import Path
from collections import defaultdict

import click
import pandas as pd


def load_jsonl(path: Path) -> pd.DataFrame:
    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    return pd.DataFrame(records)


# ====================================================================
# Formatting helpers
# ====================================================================

def _format_size(size) -> str:
    try:
        s = float(size)
    except (TypeError, ValueError):
        return "?"
    if s < 1024:
        return f"{s:.0f} B"
    if s < 1048576:
        return f"{s / 1024:.1f} KB"
    if s < 1073741824:
        return f"{s / 1048576:.1f} MB"
    return f"{s / 1073741824:.1f} GB"


def _format_date(val) -> str:
    s = str(val) if val is not None else ""
    return s[:10] if len(s) >= 10 else s or "?"


def _build_filename(stem, ext) -> str:
    """Build a displayable filename from stem + extension."""
    return f"{stem}.{ext}" if ext else str(stem)


# ====================================================================
# Section: Overview
# ====================================================================

def overview_section(df: pd.DataFrame) -> str:
    lines = []
    lines.append("=" * 80)
    lines.append("CROSS-ESTATE DUPLICATE DETECTION")
    lines.append("=" * 80)
    lines.append("")
    lines.append(f"Total files analysed: {len(df):,}")

    if "source_system" in df.columns:
        estate_counts = df["source_system"].value_counts().sort_index()
        if len(estate_counts) > 0:
            parts = [f"{name}: {cnt:,}" for name, cnt in estate_counts.items()]
            lines.append(f"  Estates: {', '.join(parts)}")
    lines.append("")

    # Tier 1 stats — vectorised redundant-bytes calculation
    if "dup_group_id" in df.columns:
        exact = df[df["dup_group_id"] != ""]
        n_groups = exact["dup_group_id"].nunique()
        n_files = len(exact)
        n_cross = 0
        if "dup_is_cross_estate" in exact.columns:
            n_cross = int(
                exact.drop_duplicates("dup_group_id")["dup_is_cross_estate"].sum()
            )

        # Redundant bytes: per group = size * (count - 1)
        redundant = 0.0
        if "size" in exact.columns and n_groups > 0:
            agg = exact.groupby("dup_group_id").agg(
                first_size=("size", "first"),
                cnt=("size", "size"),
            )
            redundant = (agg["first_size"].fillna(0) * (agg["cnt"] - 1)).sum()

        lines.append(
            f"  Tier 1 \u2014 Exact duplicates: {n_groups:,} groups "
            f"({n_files:,} files, {n_cross:,} cross-estate, "
            f"~{_format_size(redundant)} redundant)"
        )

    if "dup_copy_path_suspect" in df.columns:
        n_suspects = int(df["dup_copy_path_suspect"].sum())
        lines.append(f"  Tier 4 \u2014 Copy-path suspects: {n_suspects:,} files")

    lines.append("")
    return "\n".join(lines)


# ====================================================================
# Section: Tier 1 — Exact metadata duplicates
# ====================================================================

def exact_duplicates_section(df: pd.DataFrame, max_groups: int = 500) -> str:
    lines = []
    lines.append("=" * 80)
    lines.append("TIER 1: EXACT METADATA DUPLICATES  (filename + extension + size)")
    lines.append("=" * 80)
    lines.append("")

    if "dup_group_id" not in df.columns:
        lines.append("No exact duplicate data found.")
        return "\n".join(lines)

    exact = df[df["dup_group_id"] != ""]
    if len(exact) == 0:
        lines.append("No exact duplicates detected.")
        return "\n".join(lines)

    # Pre-compute per-group aggregates (vectorised)
    agg = exact.groupby("dup_group_id").agg(
        count=("dup_group_id", "size"),
        first_size=("size", "first"),
        confidence=("dup_confidence", "first"),
        is_cross_estate=("dup_is_cross_estate", "first"),
        is_cross_folder=("dup_is_cross_folder", "first"),
    )
    agg["redundant"] = agg["first_size"].fillna(0) * (agg["count"] - 1)
    agg = agg.sort_values("redundant", ascending=False)

    total_files = int(agg["count"].sum())
    total_redundant = agg["redundant"].sum()
    n_cross = int(agg["is_cross_estate"].sum())

    lines.append(
        f"  {len(agg):,} groups | {total_files:,} files | "
        f"{n_cross:,} cross-estate | ~{_format_size(total_redundant)} redundant"
    )
    lines.append("")

    # Size distribution
    size_dist = defaultdict(int)
    for c in agg["count"]:
        if c == 2:
            size_dist["2 copies"] += 1
        elif c <= 5:
            size_dist["3-5 copies"] += 1
        elif c <= 10:
            size_dist["6-10 copies"] += 1
        elif c <= 20:
            size_dist["11-20 copies"] += 1
        else:
            size_dist["20+ copies"] += 1

    lines.append("  Group size distribution:")
    for band in ("2 copies", "3-5 copies", "6-10 copies", "11-20 copies", "20+ copies"):
        if band in size_dist:
            lines.append(f"    {band:>12}: {size_dist[band]:,}")
    lines.append("")

    # Detailed listing — only the top N groups by redundant storage
    top_group_ids = agg.index[:max_groups]
    top_rows = exact[exact["dup_group_id"].isin(top_group_ids)]

    # Pre-build lookup columns to avoid repeated .get() calls
    has_scheme = "entity_scheme_canonical" in top_rows.columns or "entity_scheme" in top_rows.columns

    for rank, gid in enumerate(top_group_ids, 1):
        g = agg.loc[gid]
        group_rows = top_rows[top_rows["dup_group_id"] == gid]

        # Representative filename
        first = group_rows.iloc[0]
        stem = first.get("filename_stem", "")
        ext = first.get("extension", "")
        fn = f"{stem}.{ext}" if ext else str(stem)

        tags = []
        if g["is_cross_estate"]:
            tags.append("CROSS-ESTATE")
        if g["is_cross_folder"]:
            tags.append("CROSS-FOLDER")
        tag_str = f"  [{', '.join(tags)}]" if tags else ""

        # Estate breakdown
        estate_tag = ""
        if "source_system" in group_rows.columns:
            ec = group_rows["source_system"].value_counts()
            parts = [f"{name}: {cnt}" for name, cnt in sorted(ec.items())]
            estate_tag = f" | {', '.join(parts)}"

        lines.append(
            f"  [{rank:>4}] \"{fn}\"  "
            f"({int(g['count'])} copies, {_format_size(g['first_size'])}, "
            f"confidence: {g['confidence']:.2f}){tag_str}{estate_tag}"
        )

        # Members — use itertuples for speed (3-10x faster than iterrows)
        members = []
        for row in group_rows.itertuples(index=False):
            r_stem = getattr(row, "filename_stem", "")
            r_ext = getattr(row, "extension", "")
            scheme = ""
            if has_scheme:
                scheme = getattr(row, "entity_scheme_canonical", "") or getattr(row, "entity_scheme", "") or ""
            members.append((
                getattr(row, "source_system", ""),
                _build_filename(r_stem, r_ext),
                _format_date(getattr(row, "last_modified", None)),
                getattr(row, "size", 0),
                scheme,
            ))
        members.sort()  # by estate, then filename

        for estate, filename, mod, sz, scheme in members:
            scheme_tag = f" | {scheme}" if scheme else ""
            estate_tag2 = f" [{estate}]" if estate else ""
            lines.append(
                f"           - {filename}  "
                f"({mod}, {_format_size(sz)}){estate_tag2}{scheme_tag}"
            )

        lines.append("")

    if len(agg) > max_groups:
        lines.append(f"  ... and {len(agg) - max_groups:,} more groups")
        lines.append("")

    return "\n".join(lines)


# ====================================================================
# Section: Tier 3 — Duplicate folders (computed at report time)
# ====================================================================

# Max folders a single filename can appear in before we skip it.
# Generic files (DSC00001.jpg, photo 1.jpg) appear in hundreds of
# folders and create O(n^2) pair explosions without adding signal.
_MAX_FOLDER_SPREAD = 50

# Minimum unique filenames in a folder to be a candidate.
_MIN_FOLDER_FILES = 5


def folder_duplicates_section(df: pd.DataFrame, max_pairs: int = 100) -> str:
    t0 = time.time()

    lines = []
    lines.append("=" * 80)
    lines.append("TIER 3: DUPLICATE FOLDERS  (>50% filename overlap)")
    lines.append("=" * 80)
    lines.append("")

    if "segments" not in df.columns or "filename_stem" not in df.columns:
        lines.append("Insufficient data for folder comparison.")
        return "\n".join(lines)

    # ---- Step 1: Build folder fingerprints (vectorised) ----

    # Build folder_key and file_sig columns without iterrows
    has_ext = "extension" in df.columns
    has_estate = "source_system" in df.columns

    # Filter to rows with usable segments and filename
    mask = df["filename_stem"].notna() & (df["filename_stem"] != "")
    if "segments" in df.columns:
        mask = mask & df["segments"].apply(lambda s: isinstance(s, list) and len(s) > 0)
    usable = df[mask].copy()

    if len(usable) == 0:
        lines.append("No usable data for folder comparison.")
        return "\n".join(lines)

    usable["_folder_key"] = usable["segments"].apply(lambda s: "/".join(s))
    stem_lower = usable["filename_stem"].str.lower()
    ext_lower = usable["extension"].fillna("").str.lower() if has_ext else pd.Series("", index=usable.index)
    usable["_file_sig"] = stem_lower + "|" + ext_lower

    # Per-folder: set of unique file signatures, file count, estate
    folder_groups = usable.groupby("_folder_key")
    folder_file_count = folder_groups.size()
    folder_unique_sigs = folder_groups["_file_sig"].apply(set)

    if has_estate:
        folder_estate = folder_groups["source_system"].first().to_dict()
    else:
        folder_estate = {}

    # Filter to folders with enough unique files
    candidates = {k: v for k, v in folder_unique_sigs.items() if len(v) >= _MIN_FOLDER_FILES}

    if len(candidates) < 2:
        lines.append("Not enough qualifying folders (need 5+ unique files each).")
        return "\n".join(lines)

    # ---- Step 2: Inverted index with spread cap ----

    file_to_folders: dict[str, list[str]] = defaultdict(list)
    for folder, sigs in candidates.items():
        for sig in sigs:
            file_to_folders[sig].append(folder)

    # ---- Step 3: Build pair-shared counts, skipping high-spread files ----

    pair_shared: dict[tuple[str, str], int] = defaultdict(int)
    skipped_files = 0

    for sig, folders in file_to_folders.items():
        if len(folders) < 2:
            continue
        if len(folders) > _MAX_FOLDER_SPREAD:
            skipped_files += 1
            continue
        folders_sorted = sorted(folders)
        for i in range(len(folders_sorted)):
            for j in range(i + 1, len(folders_sorted)):
                pair_shared[(folders_sorted[i], folders_sorted[j])] += 1

    if skipped_files:
        lines.append(
            f"  (Skipped {skipped_files:,} high-spread filenames appearing "
            f"in >{_MAX_FOLDER_SPREAD} folders)"
        )
        lines.append("")

    # ---- Step 4: Compute Jaccard for candidate pairs ----

    pairs = []
    for (f1, f2), shared_count in pair_shared.items():
        set1 = candidates[f1]
        set2 = candidates[f2]
        union_size = len(set1 | set2)
        if union_size == 0:
            continue
        jaccard = shared_count / union_size
        if jaccard >= 0.50:
            pairs.append({
                "folder_a": f1,
                "folder_b": f2,
                "jaccard": jaccard,
                "shared_files": shared_count,
                "files_a": len(set1),
                "files_b": len(set2),
                "total_files_a": int(folder_file_count.get(f1, 0)),
                "total_files_b": int(folder_file_count.get(f2, 0)),
                "estate_a": folder_estate.get(f1, ""),
                "estate_b": folder_estate.get(f2, ""),
            })

    elapsed = time.time() - t0

    if not pairs:
        lines.append(
            f"No duplicate folders detected (Jaccard >= 0.50).  "
            f"({elapsed:.1f}s)"
        )
        return "\n".join(lines)

    pairs.sort(key=lambda p: -p["jaccard"])

    lines.append(
        f"  {len(pairs):,} folder pairs with >50% filename overlap  "
        f"({elapsed:.1f}s)"
    )
    lines.append("")

    displayed = pairs[:max_pairs]
    for i, p in enumerate(displayed, 1):
        estate_a = f" [{p['estate_a']}]" if p["estate_a"] else ""
        estate_b = f" [{p['estate_b']}]" if p["estate_b"] else ""
        pct = p["jaccard"] * 100

        lines.append(
            f"  [{i:>4}] Similarity: {pct:.0f}% "
            f"({p['shared_files']} shared files)"
        )
        name_a = "/".join(p["folder_a"].split("/")[-2:]) if "/" in p["folder_a"] else p["folder_a"]
        name_b = "/".join(p["folder_b"].split("/")[-2:]) if "/" in p["folder_b"] else p["folder_b"]
        lines.append(
            f"         Folder A: {name_a}  "
            f"({p['total_files_a']} files){estate_a}"
        )
        lines.append(
            f"         Folder B: {name_b}  "
            f"({p['total_files_b']} files){estate_b}"
        )
        lines.append(f"         Full A: {p['folder_a']}")
        lines.append(f"         Full B: {p['folder_b']}")
        lines.append("")

    if len(pairs) > max_pairs:
        lines.append(f"  ... and {len(pairs) - max_pairs:,} more pairs")
        lines.append("")

    return "\n".join(lines)


# ====================================================================
# Section: Tier 4 — Copy-path suspects
# ====================================================================

_COPY_KEYWORDS = frozenset({
    "backup", "backups", "archive", "archived", "copy",
    "copies", "old", "temp", "tmp", "desktop", "bak",
})


def copy_path_section(df: pd.DataFrame, max_items: int = 200) -> str:
    lines = []
    lines.append("=" * 80)
    lines.append("TIER 4: COPY-PATH SUSPECTS  (files in backup/archive/copy paths)")
    lines.append("=" * 80)
    lines.append("")

    if "dup_copy_path_suspect" not in df.columns:
        lines.append("No copy-path data found.")
        return "\n".join(lines)

    suspects = df[df["dup_copy_path_suspect"] == True]
    if len(suspects) == 0:
        lines.append("No copy-path suspects detected.")
        return "\n".join(lines)

    total_size = suspects["size"].sum() if "size" in suspects.columns else 0

    lines.append(
        f"  {len(suspects):,} files in copy-like paths "
        f"(~{_format_size(total_size)} total)"
    )
    lines.append("")

    # Keyword breakdown — vectorised via exploded segments
    if "segments" in suspects.columns:
        exploded = suspects["segments"].explode()
        if len(exploded) > 0:
            lower_segs = exploded.dropna().str.lower().str.strip()
            keyword_hits = lower_segs[lower_segs.isin(_COPY_KEYWORDS)]
            if len(keyword_hits) > 0:
                kw_counts = keyword_hits.value_counts()
                lines.append("  By path keyword:")
                for kw, cnt in kw_counts.items():
                    lines.append(f"    {kw:>12}: {cnt:,}")
                lines.append("")

    # Sample listing — itertuples for speed
    displayed = suspects.head(max_items)
    for i, row in enumerate(displayed.itertuples(index=False), 1):
        stem = getattr(row, "filename_stem", "")
        ext = getattr(row, "extension", "")
        fn = _build_filename(stem, ext)
        estate = f" [{getattr(row, 'source_system', '')}]" if getattr(row, "source_system", "") else ""
        sz = _format_size(getattr(row, "size", 0))
        mod = _format_date(getattr(row, "last_modified", None))

        lines.append(f"  [{i:>4}] {fn}  ({mod}, {sz}){estate}")

    if len(suspects) > max_items:
        lines.append(f"\n  ... and {len(suspects) - max_items:,} more suspects")

    lines.append("")
    return "\n".join(lines)


# ====================================================================
# CLI entry point
# ====================================================================

@click.command()
@click.argument("input_path", type=click.Path(exists=True))
@click.option(
    "--output-dir", default="reports",
    help="Directory for output (default: reports/)",
)
def main(input_path: str, output_dir: str):
    input_path = Path(input_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    print(f"Loading {input_path.name}...")
    df = load_jsonl(input_path)
    print(f"  {len(df):,} records loaded ({time.time() - t0:.1f}s)")

    report_parts = []

    for label, fn in [
        ("Overview", overview_section),
        ("Tier 1 — Exact duplicates", exact_duplicates_section),
        ("Tier 3 — Duplicate folders", folder_duplicates_section),
        ("Tier 4 — Copy-path suspects", copy_path_section),
    ]:
        t1 = time.time()
        report_parts.append(fn(df))
        print(f"  {label}: {time.time() - t1:.1f}s")

    stem = input_path.stem.replace("_classified", "")
    out_path = output_dir / f"{stem}_duplicate_report.txt"

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n\n".join(report_parts))

    elapsed = time.time() - t0
    print(f"\nWrote duplicate report to: {out_path}")
    print(f"  ({os.path.getsize(out_path) / 1024:.0f} KB, {elapsed:.1f}s total)")


if __name__ == "__main__":
    main()
