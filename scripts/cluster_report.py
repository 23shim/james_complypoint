"""
Generate cluster summary reports from classified JSONL output.

Produces two CSV reports:
  1. Version/duplicate cluster summary (one row per multi-file cluster)
  2. Entity cluster summary (one row per entity cluster)

Usage:
    python scripts/cluster_report.py <classified.jsonl> [--output-dir reports/]
"""

import sys
import os
sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import json
from pathlib import Path

import click
import pandas as pd


def load_jsonl(path: Path) -> pd.DataFrame:
    """Load a JSONL file into a DataFrame."""
    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    return pd.DataFrame(records)


def build_version_cluster_report(df: pd.DataFrame) -> pd.DataFrame:
    """Build summary of multi-file version/duplicate clusters."""
    if "version_cluster_id" not in df.columns:
        return pd.DataFrame()

    # Only multi-file clusters
    multi = df[df["version_cluster_size"] > 1].copy()
    if len(multi) == 0:
        return pd.DataFrame()

    rows = []
    for cid, group in multi.groupby("version_cluster_id"):
        # Representative filename
        norm_fn = group["normalized_filename"].iloc[0] if "normalized_filename" in group.columns else ""

        # Extension family
        exts = group["extension"].dropna().unique().tolist() if "extension" in group.columns else []

        # Inferred type
        types = group["inferred_type"].dropna().unique().tolist() if "inferred_type" in group.columns else []
        type_str = types[0] if len(types) == 1 else "mixed" if len(types) > 1 else ""

        # Entity context
        entity_ctx = ""
        for col in ("entity_scheme_canonical", "entity_address_canonical", "entity_scheme", "entity_address"):
            if col in group.columns:
                vals = group[col][group[col] != ""].unique()
                if len(vals) > 0:
                    entity_ctx = vals[0]
                    break

        # Scheme
        scheme = ""
        for col in ("entity_scheme_canonical", "entity_scheme"):
            if col in group.columns:
                vals = group[col][group[col] != ""].unique()
                if len(vals) > 0:
                    scheme = vals[0]
                    break

        # Dates
        latest_mod = ""
        oldest_mod = ""
        span_days = 0
        if "last_modified" in group.columns:
            dates = pd.to_datetime(group["last_modified"], errors="coerce").dropna()
            if len(dates) > 0:
                latest_mod = str(dates.max())
                oldest_mod = str(dates.min())
                span_days = (dates.max() - dates.min()).days

        # Latest file
        latest_file = ""
        if "is_latest_in_cluster" in group.columns and "filename" in group.columns:
            latest_rows = group[group["is_latest_in_cluster"] == True]
            if len(latest_rows) > 0:
                latest_file = latest_rows["filename"].iloc[0]

        # Unique folders
        unique_folders = 0
        if "segments" in group.columns:
            folders = set()
            for segs in group["segments"]:
                if isinstance(segs, list) and segs:
                    folders.add("\\".join(segs))
            unique_folders = len(folders)

        # Potential duplicate
        is_dup = group["potential_duplicate"].any() if "potential_duplicate" in group.columns else False

        # Version markers
        has_markers = False
        if "version_marker" in group.columns:
            has_markers = (group["version_marker"] != "").any()

        # Sample filenames
        sample_fns = ""
        if "filename" in group.columns:
            fns = group["filename"].dropna().tolist()[:5]
            sample_fns = "; ".join(fns)

        # Anomaly flags
        flags = []
        if len(group) > 20:
            flags.append("oversized_cluster")
        if len(types) > 1:
            flags.append("mixed_types")
        if len(group) > 1 and not has_markers:
            flags.append("no_version_markers")
        if span_days > 365 * 5:
            flags.append("wide_date_range")
        if is_dup and unique_folders == 1:
            flags.append("single_folder_duplicates")

        rows.append({
            "cluster_id": cid,
            "cluster_size": len(group),
            "representative_filename": norm_fn,
            "extension": ", ".join(exts),
            "inferred_type": type_str,
            "entity_context": entity_ctx,
            "scheme": scheme,
            "latest_file": latest_file,
            "latest_modified": latest_mod,
            "oldest_modified": oldest_mod,
            "span_days": span_days,
            "unique_folders": unique_folders,
            "potential_duplicate": is_dup,
            "has_version_markers": has_markers,
            "sample_filenames": sample_fns,
            "anomaly_flags": "|".join(flags),
        })

    report = pd.DataFrame(rows)
    report = report.sort_values("cluster_size", ascending=False).reset_index(drop=True)
    report.index += 1
    report.index.name = "#"
    return report


def _build_entity_canonical_lookups(df: pd.DataFrame) -> tuple[dict, dict]:
    """Build scheme and address cluster-ID → canonical-name lookups."""
    scheme_lookup: dict[str, str] = {}
    addr_lookup: dict[str, str] = {}

    if "entity_scheme_cluster_id" in df.columns and "entity_scheme_canonical" in df.columns:
        for cid, canon in zip(df["entity_scheme_cluster_id"], df["entity_scheme_canonical"]):
            if isinstance(cid, str) and cid and isinstance(canon, str) and canon:
                scheme_lookup[cid] = canon

    if "entity_address_cluster_id" in df.columns and "entity_address_canonical" in df.columns:
        for cid, canon in zip(df["entity_address_cluster_id"], df["entity_address_canonical"]):
            if isinstance(cid, str) and cid and isinstance(canon, str) and canon:
                addr_lookup[cid] = canon

    return scheme_lookup, addr_lookup


def _decode_plot_context(ctx: str, scheme_lookup: dict, addr_lookup: dict) -> str:
    """Decode a raw plot context key into a human-readable linked entity string.

    Context key formats produced by entity_cluster.py:
      - <scheme_cluster_id>       e.g. "schm_0042"  → "scheme: <canonical>"
      - "addr:<address_cluster_id>"                   → "address: <canonical>"
      - "__nocontext__<folder_path>"                  → "(isolated)"
      - ""                                            → ""
    """
    if not ctx:
        return ""
    if ctx.startswith("__nocontext__"):
        return "(isolated — no scheme/address context)"
    if ctx.startswith("addr:"):
        addr_id = ctx[5:]
        canon = addr_lookup.get(addr_id, addr_id)
        return f"address: {canon}"
    # Scheme cluster ID
    canon = scheme_lookup.get(ctx, ctx)
    return f"scheme: {canon}"


def build_entity_cluster_report(df: pd.DataFrame) -> pd.DataFrame:
    """Build summary of entity clusters."""
    rows = []

    scheme_lookup, addr_lookup = _build_entity_canonical_lookups(df)

    for entity_type in ("scheme", "address", "plot"):
        cluster_col = f"entity_{entity_type}_cluster_id"
        canonical_col = f"entity_{entity_type}_canonical"
        value_col = f"entity_{entity_type}"

        if cluster_col not in df.columns:
            continue

        clustered = df[df[cluster_col] != ""].copy()
        if len(clustered) == 0:
            continue

        for cid, group in clustered.groupby(cluster_col):
            canonical = group[canonical_col].iloc[0] if canonical_col in group.columns else ""

            # Unique raw variants
            variants = set()
            if value_col in group.columns:
                variants = set(group[value_col][group[value_col] != ""].unique())

            # Linked scheme/address context (plot clusters only)
            linked_context = ""
            if entity_type == "plot":
                ctx_col = "entity_plot_cluster_context"
                if ctx_col in group.columns:
                    ctx_vals = group[ctx_col][group[ctx_col] != ""]
                    if len(ctx_vals) > 0:
                        linked_context = _decode_plot_context(
                            ctx_vals.iloc[0], scheme_lookup, addr_lookup,
                        )

            rows.append({
                "entity_type": entity_type,
                "cluster_id": cid,
                "canonical_form": canonical,
                "variant_count": len(variants),
                "file_count": len(group),
                "variants": "; ".join(sorted(variants)),
                "linked_context": linked_context,
            })

    report = pd.DataFrame(rows)
    if len(report) > 0:
        report = report.sort_values(
            ["entity_type", "file_count"], ascending=[True, False]
        ).reset_index(drop=True)
        report.index += 1
        report.index.name = "#"
    return report


@click.command()
@click.argument("input_path", type=click.Path(exists=True))
@click.option(
    "--output-dir", default="reports",
    help="Directory for output CSVs (default: reports/)",
)
def main(input_path: str, output_dir: str):
    """Generate cluster summary reports from classified JSONL."""
    input_path = Path(input_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading {input_path.name}...")
    df = load_jsonl(input_path)
    print(f"  {len(df):,} records loaded")

    # Version cluster report
    stem = input_path.stem.replace("_classified", "")
    version_report = build_version_cluster_report(df)
    if len(version_report) > 0:
        out_path = output_dir / f"{stem}_version_clusters.csv"
        version_report.to_csv(out_path, encoding="utf-8-sig")
        print(f"\nVersion clusters: {len(version_report):,} multi-file clusters")
        print(f"  Wrote to {out_path}")

        # Summary stats
        anomalies = version_report["anomaly_flags"].apply(lambda x: x.split("|") if x else [])
        oversized = sum(1 for flags in anomalies if "oversized_cluster" in flags)
        mixed = sum(1 for flags in anomalies if "mixed_types" in flags)
        no_markers = sum(1 for flags in anomalies if "no_version_markers" in flags)

        print(f"  Anomalies: {oversized} oversized, {mixed} mixed types, {no_markers} no version markers")
        print(f"  Total duplicate files: {version_report['cluster_size'].sum():,}")

        # Top 10
        print(f"\n  Top 10 clusters:")
        print(f"  {'#':>4}  {'Filename':<40} {'Type':<15} {'Size':>5} {'Folders':>7} {'Dup':>4} {'Flags'}")
        print(f"  {'-'*4}  {'-'*40} {'-'*15} {'-'*5} {'-'*7} {'-'*4} {'-'*30}")
        for idx, row in version_report.head(10).iterrows():
            dup = "Y" if row["potential_duplicate"] else "N"
            fn = str(row["representative_filename"])[:39]
            tp = str(row["inferred_type"])[:14]
            print(
                f"  {idx:>4}  {fn:<40} {tp:<15} "
                f"{row['cluster_size']:>5} {row['unique_folders']:>7} "
                f"{dup:>4} {row['anomaly_flags']}"
            )
    else:
        print("\nNo multi-file version clusters found.")

    # Entity cluster report
    entity_report = build_entity_cluster_report(df)
    if len(entity_report) > 0:
        out_path = output_dir / f"{stem}_entity_clusters.csv"
        entity_report.to_csv(out_path, encoding="utf-8-sig")
        print(f"\nEntity clusters: {len(entity_report):,} total")
        print(f"  Wrote to {out_path}")

        for etype in entity_report["entity_type"].unique():
            subset = entity_report[entity_report["entity_type"] == etype]
            multi = subset[subset["variant_count"] > 1]
            print(f"  {etype}: {len(subset)} clusters ({len(multi)} with multiple variants)")
    else:
        print("\nNo entity clusters found.")

    print("\nDone.")


if __name__ == "__main__":
    main()
