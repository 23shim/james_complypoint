"""
Generate per-entity review CSVs from classified JSONL.

Produces CSVs mirroring the schemes_*_v3.csv format:
  1. addresses_<estate>.csv — one row per unique address (or canonical cluster)
  2. plots_<estate>.csv — one row per unique plot (or canonical cluster)

Usage:
    python scripts/entity_report.py <classified.jsonl> [--output-dir reports/]
"""

import sys
import os
sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import json
from pathlib import Path
from collections import Counter, defaultdict

import click
import pandas as pd


def load_jsonl(path: Path) -> pd.DataFrame:
    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    return pd.DataFrame(records)


def _safe_str(val) -> str:
    if pd.isna(val) or val is None:
        return ""
    return str(val).strip()


def _top_n(values, n=5) -> str:
    """Return top-n most common non-empty values, semicolon-separated."""
    counts = Counter(v for v in values if v)
    return "; ".join(v for v, _ in counts.most_common(n))


def _build_entity_paths(group: pd.DataFrame, depth_col: str) -> tuple[str, str, str]:
    """Build full_path, parent_folder, and sample_full_paths for an entity group.

    Uses entity depth + segments to reconstruct the path from root
    to the matched folder segment.

    Returns:
        (most_common_full_path, parent_folder, sample_full_paths_str)
    """
    full_paths = []
    for _, row in group.iterrows():
        segs = row.get("segments")
        depth = row.get(depth_col, -1)
        if not isinstance(segs, list) or not segs:
            continue
        try:
            d = int(depth)
        except (TypeError, ValueError):
            continue
        if d < 0 or d >= len(segs):
            # Fallback: use full segment path
            full_paths.append("\\".join(segs))
        else:
            # Path from root through the matched segment
            full_paths.append("\\".join(segs[: d + 1]))

    if not full_paths:
        return ("", "", "")

    path_counts = Counter(full_paths)
    most_common_path = path_counts.most_common(1)[0][0]

    # Parent folder: one level above the matched segment
    parts = most_common_path.split("\\")
    parent = parts[-2] if len(parts) >= 2 else "(root)"

    # Sample distinct full paths (top 5 by frequency)
    sample = "; ".join(p for p, _ in path_counts.most_common(5))

    return (most_common_path, parent, sample)


def _collect_other_folder_addresses(group: pd.DataFrame, selected_address: str) -> str:
    """Collect other address matches from raw_addresss that aren't the selected one.

    For each file in the group, inspects raw_addresss to find address-bearing
    folders other than the one assigned as entity_address.  Returns a
    semicolon-separated string of unique other addresses, sorted by frequency.
    """
    raw_col = "raw_addresss"
    if raw_col not in group.columns:
        return ""

    other_counts: Counter = Counter()
    selected_lower = selected_address.lower().strip() if selected_address else ""

    for _, row in group.iterrows():
        raw_list = row.get(raw_col)
        if not isinstance(raw_list, list):
            continue
        for match in raw_list:
            if isinstance(match, dict):
                v = match.get("value", "")
            elif isinstance(match, str):
                v = match
            else:
                continue
            if v and isinstance(v, str) and v.strip():
                if v.lower().strip() != selected_lower:
                    other_counts[v] += 1

    if not other_counts:
        return ""

    # Return sorted by frequency descending
    return "; ".join(v for v, _ in other_counts.most_common())


def build_address_report(df: pd.DataFrame) -> pd.DataFrame:
    """One row per address (using canonical form if clustered, else raw value).

    Includes both the validated/preferred address and any other address
    matches found in the same files' folder paths (from raw_addresss).
    """

    # Determine the grouping key: canonical if available, else raw
    has_canonical = "entity_address_canonical" in df.columns
    has_cluster = "entity_address_cluster_id" in df.columns
    value_col = "entity_address"

    if value_col not in df.columns:
        return pd.DataFrame()

    # Filter to rows with an address
    subset = df[df[value_col].apply(lambda v: isinstance(v, str) and v.strip() != "")].copy()
    if len(subset) == 0:
        return pd.DataFrame()

    # Group key: cluster_id if available, else raw value
    if has_cluster:
        subset["_group"] = subset["entity_address_cluster_id"].apply(
            lambda v: v if isinstance(v, str) and v else ""
        )
        # Fallback for unclustered rows
        mask = subset["_group"] == ""
        subset.loc[mask, "_group"] = "raw:" + subset.loc[mask, value_col]
    else:
        subset["_group"] = subset[value_col]

    rows = []
    for group_key, group in subset.groupby("_group"):
        # Canonical or most-common raw value
        if has_canonical:
            canonical = _safe_str(group["entity_address_canonical"].mode().iloc[0])
            if not canonical:
                canonical = _safe_str(group[value_col].mode().iloc[0])
        else:
            canonical = _safe_str(group[value_col].mode().iloc[0])

        # All raw variants
        raw_variants = sorted(set(
            _safe_str(v) for v in group[value_col] if _safe_str(v)
        ))
        variant_count = len(raw_variants)

        # Confidence
        conf_col = "entity_address_confidence"
        avg_conf = group[conf_col].mean() if conf_col in group.columns else 0.0

        # Depth
        depth_col = "entity_address_depth"
        avg_depth = group[depth_col].mean() if depth_col in group.columns else -1
        # Most common depth
        if depth_col in group.columns:
            mode_depth = group[depth_col].mode()
            common_depth = int(mode_depth.iloc[0]) if len(mode_depth) > 0 else -1
        else:
            common_depth = -1

        # Full path to the matched entity segment
        full_path, parent_folder, sample_full_paths = _build_entity_paths(
            group, "entity_address_depth",
        )

        # Associated schemes
        scheme_col = "entity_scheme_canonical" if "entity_scheme_canonical" in group.columns else "entity_scheme"
        schemes = []
        if scheme_col in group.columns:
            schemes = sorted(set(
                _safe_str(v) for v in group[scheme_col] if _safe_str(v)
            ))

        # Associated plots
        plot_col = "entity_plot_canonical" if "entity_plot_canonical" in group.columns else "entity_plot"
        unique_plots = 0
        sample_plots = ""
        if plot_col in group.columns:
            plots = [_safe_str(v) for v in group[plot_col] if _safe_str(v)]
            unique_plots = len(set(plots))
            sample_plots = _top_n(plots, 5)

        # Inferred types
        type_counts = Counter(
            _safe_str(v) for v in group.get("inferred_type", []) if _safe_str(v)
        )
        top_types = "; ".join(f"{t} ({c})" for t, c in type_counts.most_common(3))

        # Sample filenames
        sample_fns = _top_n(group.get("filename", []), 5)

        # Other address matches from other folders in the same paths
        other_addresses = _collect_other_folder_addresses(group, canonical)

        rows.append({
            "address": canonical,
            "variant_count": variant_count,
            "file_count": len(group),
            "avg_confidence": round(avg_conf, 2),
            "typical_depth": common_depth,
            "parent_folder": parent_folder,
            "full_path": full_path,
            "schemes": "; ".join(schemes),
            "unique_plots": unique_plots,
            "top_types": top_types,
            "sample_filenames": sample_fns,
            "sample_full_paths": sample_full_paths,
            "sample_plots": sample_plots,
            "all_variants": "; ".join(raw_variants) if variant_count > 1 else "",
            "other_folder_addresses": other_addresses,
        })

    report = pd.DataFrame(rows)
    report = report.sort_values("file_count", ascending=False).reset_index(drop=True)
    report.index += 1
    report.index.name = "#"
    return report


def build_plot_report(df: pd.DataFrame) -> pd.DataFrame:
    """One row per plot (using canonical form if clustered, else raw value)."""

    has_canonical = "entity_plot_canonical" in df.columns
    has_cluster = "entity_plot_cluster_id" in df.columns
    value_col = "entity_plot"

    if value_col not in df.columns:
        return pd.DataFrame()

    subset = df[df[value_col].apply(lambda v: isinstance(v, str) and v.strip() != "")].copy()
    if len(subset) == 0:
        return pd.DataFrame()

    if has_cluster:
        subset["_group"] = subset["entity_plot_cluster_id"].apply(
            lambda v: v if isinstance(v, str) and v else ""
        )
        mask = subset["_group"] == ""
        subset.loc[mask, "_group"] = "raw:" + subset.loc[mask, value_col]
    else:
        subset["_group"] = subset[value_col]

    rows = []
    for group_key, group in subset.groupby("_group"):
        if has_canonical:
            canonical = _safe_str(group["entity_plot_canonical"].mode().iloc[0])
            if not canonical:
                canonical = _safe_str(group[value_col].mode().iloc[0])
        else:
            canonical = _safe_str(group[value_col].mode().iloc[0])

        raw_variants = sorted(set(
            _safe_str(v) for v in group[value_col] if _safe_str(v)
        ))
        variant_count = len(raw_variants)

        conf_col = "entity_plot_confidence"
        avg_conf = group[conf_col].mean() if conf_col in group.columns else 0.0

        depth_col = "entity_plot_depth"
        if depth_col in group.columns:
            mode_depth = group[depth_col].mode()
            common_depth = int(mode_depth.iloc[0]) if len(mode_depth) > 0 else -1
        else:
            common_depth = -1

        # Full path to the matched entity segment
        full_path, parent_folder, sample_full_paths = _build_entity_paths(
            group, "entity_plot_depth",
        )

        # Associated schemes
        scheme_col = "entity_scheme_canonical" if "entity_scheme_canonical" in group.columns else "entity_scheme"
        schemes = []
        if scheme_col in group.columns:
            schemes = sorted(set(
                _safe_str(v) for v in group[scheme_col] if _safe_str(v)
            ))

        # Associated addresses
        addr_col = "entity_address_canonical" if "entity_address_canonical" in group.columns else "entity_address"
        unique_addrs = 0
        sample_addrs = ""
        if addr_col in group.columns:
            addrs = [_safe_str(v) for v in group[addr_col] if _safe_str(v)]
            unique_addrs = len(set(addrs))
            sample_addrs = _top_n(addrs, 5)

        type_counts = Counter(
            _safe_str(v) for v in group.get("inferred_type", []) if _safe_str(v)
        )
        top_types = "; ".join(f"{t} ({c})" for t, c in type_counts.most_common(3))

        sample_fns = _top_n(group.get("filename", []), 5)

        rows.append({
            "plot": canonical,
            "variant_count": variant_count,
            "file_count": len(group),
            "avg_confidence": round(avg_conf, 2),
            "typical_depth": common_depth,
            "parent_folder": parent_folder,
            "full_path": full_path,
            "schemes": "; ".join(schemes),
            "unique_addresses": unique_addrs,
            "top_types": top_types,
            "sample_filenames": sample_fns,
            "sample_full_paths": sample_full_paths,
            "sample_addresses": sample_addrs,
            "all_variants": "; ".join(raw_variants) if variant_count > 1 else "",
        })

    report = pd.DataFrame(rows)
    report = report.sort_values("file_count", ascending=False).reset_index(drop=True)
    report.index += 1
    report.index.name = "#"
    return report


@click.command()
@click.argument("input_path", type=click.Path(exists=True))
@click.option("--output-dir", default="reports", help="Output directory")
def main(input_path: str, output_dir: str):
    input_path = Path(input_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading {input_path.name}...")
    df = load_jsonl(input_path)
    print(f"  {len(df):,} records loaded")

    stem = input_path.stem.replace("_classified", "")

    # Address report
    addr_report = build_address_report(df)
    if len(addr_report) > 0:
        out_path = output_dir / f"addresses_{stem}_v5.csv"
        addr_report.to_csv(out_path, encoding="utf-8-sig")
        print(f"\nAddresses: {len(addr_report):,} unique entries")
        print(f"  Wrote to {out_path}")

        multi = addr_report[addr_report["variant_count"] > 1]
        print(f"  {len(multi)} with multiple variants (clustered)")

        # Quick stats
        total_files = addr_report["file_count"].sum()
        print(f"  Covers {total_files:,} files total")

        # Top 15
        print(f"\n  Top 15 addresses by file count:")
        print(f"  {'#':>4}  {'Address':<55} {'Files':>6} {'Conf':>5} {'Schemes'}")
        print(f"  {'-'*4}  {'-'*55} {'-'*6} {'-'*5} {'-'*30}")
        for idx, row in addr_report.head(15).iterrows():
            addr = str(row["address"])[:54]
            schemes = str(row["schemes"])[:30]
            print(
                f"  {idx:>4}  {addr:<55} {row['file_count']:>6} "
                f"{row['avg_confidence']:>5.2f} {schemes}"
            )
    else:
        print("\nNo addresses found.")

    # Plot report
    plot_report = build_plot_report(df)
    if len(plot_report) > 0:
        out_path = output_dir / f"plots_{stem}_v5.csv"
        plot_report.to_csv(out_path, encoding="utf-8-sig")
        print(f"\nPlots: {len(plot_report):,} unique entries")
        print(f"  Wrote to {out_path}")

        multi = plot_report[plot_report["variant_count"] > 1]
        print(f"  {len(multi)} with multiple variants (clustered)")

        total_files = plot_report["file_count"].sum()
        print(f"  Covers {total_files:,} files total")

        # Top 15
        print(f"\n  Top 15 plots by file count:")
        print(f"  {'#':>4}  {'Plot':<55} {'Files':>6} {'Conf':>5} {'Schemes'}")
        print(f"  {'-'*4}  {'-'*55} {'-'*6} {'-'*5} {'-'*30}")
        for idx, row in plot_report.head(15).iterrows():
            plot = str(row["plot"])[:54]
            schemes = str(row["schemes"])[:30]
            print(
                f"  {idx:>4}  {plot:<55} {row['file_count']:>6} "
                f"{row['avg_confidence']:>5.2f} {schemes}"
            )
    else:
        print("\nNo plots found.")

    print("\nDone.")


if __name__ == "__main__":
    main()
