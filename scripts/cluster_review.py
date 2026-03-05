"""
Generate a human-skimmable cluster review report from classified JSONL.

Outputs a single text file with:
  - Entity clusters (scheme, address, plot) — canonical, variants, file counts
  - File version clusters — normalised filename, members, duplicate flags
"""

import sys
import os
sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import json
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
    """Decode a raw plot context key into a human-readable linked entity string."""
    if not ctx:
        return ""
    if ctx.startswith("__nocontext__"):
        return "(isolated — no scheme/address context)"
    if ctx.startswith("addr:"):
        addr_id = ctx[5:]
        canon = addr_lookup.get(addr_id, addr_id)
        return f"address: {canon}"
    canon = scheme_lookup.get(ctx, ctx)
    return f"scheme: {canon}"


def entity_cluster_section(df: pd.DataFrame) -> str:
    lines = []
    lines.append("=" * 80)
    lines.append("ENTITY CLUSTERS")
    lines.append("=" * 80)
    lines.append("")

    scheme_lookup, addr_lookup = _build_entity_canonical_lookups(df)

    for entity_type in ("scheme", "address", "plot"):
        cluster_col = f"entity_{entity_type}_cluster_id"
        canonical_col = f"entity_{entity_type}_canonical"
        value_col = f"entity_{entity_type}"

        if cluster_col not in df.columns:
            continue

        clustered = df[df[cluster_col] != ""].copy()
        if len(clustered) == 0:
            lines.append(f"--- {entity_type.upper()} --- (no clusters)")
            lines.append("")
            continue

        # Group by cluster
        clusters = []
        for cid, group in clustered.groupby(cluster_col):
            canonical = group[canonical_col].iloc[0] if canonical_col in group.columns else ""
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

            # Collect matched folders with estate and file count.
            # The "matched folder" is the path up to and including the
            # entity segment (segments[:depth+1]).
            depth_col = f"entity_{entity_type}_depth"
            folder_estate_files: dict[tuple[str, str], int] = {}  # (folder, estate) -> file_count
            if "segments" in group.columns and depth_col in group.columns:
                for _, row in group.iterrows():
                    segs = row.get("segments")
                    depth = row.get(depth_col)
                    estate = row.get("source_system", "") or ""
                    if isinstance(segs, list) and segs:
                        if isinstance(depth, (int, float)) and int(depth) >= 0:
                            folder = "/".join(segs[:int(depth) + 1])
                        else:
                            folder = "/".join(segs)
                        key = (folder, estate)
                        folder_estate_files[key] = folder_estate_files.get(key, 0) + 1

            # Aggregate: unique folders, per-estate file totals
            unique_folders = set(f for f, _ in folder_estate_files)
            estate_file_totals: dict[str, int] = {}
            for (_, estate), cnt in folder_estate_files.items():
                estate_file_totals[estate] = estate_file_totals.get(estate, 0) + cnt

            # Build folder list with estate tag and file count
            folder_details: list[dict] = []
            # Group by folder path
            folder_map: dict[str, list[tuple[str, int]]] = {}
            for (folder, estate), cnt in folder_estate_files.items():
                folder_map.setdefault(folder, []).append((estate, cnt))
            for folder in sorted(folder_map):
                estates = sorted(folder_map[folder])
                folder_details.append({
                    "path": folder,
                    "estates": estates,  # list of (estate_name, file_count)
                })

            clusters.append({
                "canonical": canonical,
                "variants": sorted(variants),
                "file_count": len(group),
                "folder_count": len(unique_folders),
                "variant_count": len(variants),
                "linked_context": linked_context,
                "estate_file_totals": estate_file_totals,
                "folder_details": folder_details,
            })

        # Sort by file count descending
        clusters.sort(key=lambda c: -c["file_count"])

        total_files = sum(c["file_count"] for c in clusters)
        multi_variant = sum(1 for c in clusters if c["variant_count"] > 1)

        lines.append(f"--- {entity_type.upper()} ---  "
                      f"{len(clusters)} clusters | {total_files:,} files | "
                      f"{multi_variant} with multiple variants")
        lines.append("")

        for i, c in enumerate(clusters, 1):
            marker = "*" if c["variant_count"] > 1 else " "
            # Estate file totals
            estate_tag = ""
            if c["estate_file_totals"]:
                parts = [f"{name}: {cnt}" for name, cnt in sorted(c["estate_file_totals"].items())]
                estate_tag = f" | {', '.join(parts)}"
            lines.append(f"  {marker} [{i:>3}] \"{c['canonical']}\"  "
                          f"({c['file_count']:,} files, {c['folder_count']} folders{estate_tag})")
            if c["linked_context"]:
                lines.append(f"           context: {c['linked_context']}")
            # Show all matched folders with estate tag
            for fd in c["folder_details"]:
                estate_parts = []
                for estate_name, fcnt in fd["estates"]:
                    if estate_name:
                        estate_parts.append(f"{estate_name}: {fcnt}")
                    else:
                        estate_parts.append(str(fcnt))
                file_info = f"  ({', '.join(estate_parts)} files)" if estate_parts else ""
                folder_name = fd["path"].split("/")[-1]
                lines.append(f"           - {folder_name}{file_info}")

        lines.append("")

    return "\n".join(lines)


def file_version_cluster_section(df: pd.DataFrame) -> str:
    lines = []
    lines.append("=" * 80)
    lines.append("FILE VERSION / DUPLICATE CLUSTERS")
    lines.append("=" * 80)
    lines.append("")

    if "version_cluster_id" not in df.columns:
        lines.append("No file version clustering data found.")
        return "\n".join(lines)

    # Only multi-file clusters
    multi = df[df["version_cluster_size"] > 1].copy()
    if len(multi) == 0:
        lines.append("No multi-file clusters found.")
        return "\n".join(lines)

    # Group by cluster
    clusters = []
    for cid, group in multi.groupby("version_cluster_id"):
        norm_fn = group["normalized_filename"].iloc[0] if "normalized_filename" in group.columns else ""

        # Gather members
        members = []
        for _, row in group.iterrows():
            fn = row.get("filename", "")
            rel = row.get("relative_path", "")
            mod = str(row.get("last_modified", ""))[:10]
            sz = row.get("size", 0)
            latest = row.get("is_latest_in_cluster", False)
            ver = row.get("version_marker", "")
            estate = row.get("source_system", "")
            members.append({
                "filename": fn,
                "relative_path": rel,
                "last_modified": mod,
                "size": sz,
                "is_latest": latest,
                "version_marker": ver,
                "estate": estate,
            })

        # Sort members: latest first, then by date desc
        members.sort(key=lambda m: (not m["is_latest"], m["last_modified"]), reverse=False)

        # Metadata
        exts = group["extension"].dropna().unique().tolist() if "extension" in group.columns else []
        doc_type = ""
        if "inferred_type" in group.columns:
            types = group["inferred_type"].dropna().unique().tolist()
            doc_type = types[0] if len(types) == 1 else f"mixed({', '.join(types)})" if types else ""

        is_dup = group["potential_duplicate"].any() if "potential_duplicate" in group.columns else False

        # Unique folders
        unique_folders = set()
        for _, row in group.iterrows():
            segs = row.get("segments")
            if isinstance(segs, list) and segs:
                unique_folders.add("/".join(segs))

        # Entity context
        entity_ctx = ""
        for col in ("entity_scheme_canonical", "entity_address_canonical"):
            if col in group.columns:
                vals = group[col][group[col] != ""].unique()
                if len(vals) > 0:
                    entity_ctx = vals[0]
                    break

        # Estate source breakdown
        estate_counts = {}
        if "source_system" in group.columns:
            estate_counts = group["source_system"].value_counts().to_dict()

        clusters.append({
            "cluster_id": cid,
            "normalized_filename": norm_fn,
            "size": len(group),
            "members": members,
            "extensions": exts,
            "doc_type": doc_type,
            "is_duplicate": is_dup,
            "unique_folders": len(unique_folders),
            "entity_context": entity_ctx,
            "estate_counts": estate_counts,
        })

    # Sort by cluster size descending
    clusters.sort(key=lambda c: -c["size"])

    total_files = sum(c["size"] for c in clusters)
    dups = sum(1 for c in clusters if c["is_duplicate"])

    lines.append(f"Summary: {len(clusters):,} multi-file clusters | "
                  f"{total_files:,} files | {dups:,} cross-folder duplicate clusters")
    lines.append("")

    # Size distribution
    size_dist = defaultdict(int)
    for c in clusters:
        if c["size"] <= 2:
            size_dist["2 files"] += 1
        elif c["size"] <= 5:
            size_dist["3-5 files"] += 1
        elif c["size"] <= 10:
            size_dist["6-10 files"] += 1
        elif c["size"] <= 20:
            size_dist["11-20 files"] += 1
        else:
            size_dist["20+ files"] += 1

    lines.append("Cluster size distribution:")
    for band in ("2 files", "3-5 files", "6-10 files", "11-20 files", "20+ files"):
        if band in size_dist:
            lines.append(f"  {band:>12}: {size_dist[band]:,}")
    lines.append("")

    # List all clusters
    for i, c in enumerate(clusters, 1):
        dup_tag = " [CROSS-FOLDER DUP]" if c["is_duplicate"] else ""
        entity_tag = f" | entity: {c['entity_context']}" if c["entity_context"] else ""
        estate_tag = ""
        if c["estate_counts"]:
            parts = [f"{name}: {cnt}" for name, cnt in sorted(c["estate_counts"].items())]
            estate_tag = f" | estates: {', '.join(parts)}"
        lines.append(
            f"  [{i:>4}] \"{c['normalized_filename']}\"  "
            f"({c['size']} files, {c['unique_folders']} folders, "
            f"{', '.join(c['extensions'])}){dup_tag}{entity_tag}{estate_tag}"
        )
        lines.append(f"         type: {c['doc_type']}")

        for m in c["members"]:
            latest_tag = " <-- LATEST" if m["is_latest"] else ""
            ver_tag = f" [{m['version_marker']}]" if m["version_marker"] else ""
            size_str = _format_size(m["size"])
            estate_member_tag = f" [{m['estate']}]" if m["estate"] else ""
            lines.append(
                f"           - {m['filename']}{ver_tag}  "
                f"({m['last_modified']}, {size_str}){latest_tag}{estate_member_tag}"
            )
            # Show path (truncated)
            path = m["relative_path"]
            if path:
                if len(path) > 90:
                    path = "..." + path[-87:]
                lines.append(f"             {path}")
        lines.append("")

    return "\n".join(lines)


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


def summary_section(df: pd.DataFrame) -> str:
    lines = []
    lines.append("=" * 80)
    lines.append("CLUSTERING OVERVIEW")
    lines.append("=" * 80)
    lines.append("")
    lines.append(f"Total records: {len(df):,}")

    # Per-estate breakdown
    if "source_system" in df.columns:
        estate_counts = df["source_system"].value_counts().sort_index()
        if len(estate_counts) > 1:
            for estate_name, cnt in estate_counts.items():
                lines.append(f"  {estate_name}: {cnt:,}")

    lines.append("")

    # Entity extraction coverage
    for entity_type in ("scheme", "address", "plot"):
        col = f"entity_{entity_type}"
        if col in df.columns:
            found = (df[col] != "").sum()
            pct = found / len(df) * 100 if len(df) > 0 else 0
            lines.append(f"  {entity_type}: {found:,} extracted ({pct:.1f}%)")

    lines.append("")

    # Entity cluster stats
    for entity_type in ("scheme", "address", "plot"):
        cluster_col = f"entity_{entity_type}_cluster_id"
        if cluster_col in df.columns:
            clustered = (df[cluster_col] != "").sum()
            n_clusters = df[cluster_col][df[cluster_col] != ""].nunique()
            lines.append(f"  {entity_type} clusters: {n_clusters:,} "
                          f"(covering {clustered:,} files)")

    lines.append("")

    # File version cluster stats
    if "version_cluster_size" in df.columns:
        multi = df[df["version_cluster_size"] > 1]
        n_multi_clusters = multi["version_cluster_id"].nunique()
        n_multi_files = len(multi)
        n_dups = (multi["potential_duplicate"] == True).sum() if "potential_duplicate" in multi.columns else 0
        lines.append(f"  File version clusters: {n_multi_clusters:,} multi-file "
                      f"({n_multi_files:,} files)")
        lines.append(f"  Potential cross-folder duplicates: {n_dups:,} files")

    lines.append("")
    return "\n".join(lines)


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

    print(f"Loading {input_path.name}...")
    df = load_jsonl(input_path)
    print(f"  {len(df):,} records loaded")

    report_parts = []
    report_parts.append(summary_section(df))
    report_parts.append(entity_cluster_section(df))
    report_parts.append(file_version_cluster_section(df))

    stem = input_path.stem.replace("_classified", "")
    out_path = output_dir / f"{stem}_cluster_review.txt"

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(report_parts))

    print(f"\nWrote cluster review to: {out_path}")
    print(f"  ({os.path.getsize(out_path) / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
