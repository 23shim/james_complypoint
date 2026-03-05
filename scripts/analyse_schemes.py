"""
Scheme detection analysis script.

Loads both test datasets, runs entity extraction + scheme detection,
and displays the results for review and iteration.
"""

import sys
import os
# Force UTF-8 output on Windows
sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import pandas as pd
from collections import Counter, defaultdict

import yaml

from ingestion.reader import read_file
from ingestion.schema import map_columns
from ingestion.filters import apply_filters
from ingestion.path_parser import decompose_paths
from classification.config_loader import load_config
from classification.entity_extractor import extract_entities_with_confidence
from classification.scheme_detector import (
    detect_and_assign_schemes,
    _aggregate_folder_stats,
    _identify_candidates,
    _find_lowest_common_parents,
    _score_schemes,
)


# Load config once
config = load_config("config", "housing")


def load_dataset(input_path: str, root_prefix: str, source_format: str = "treesize") -> pd.DataFrame:
    """Load and prepare a dataset through ingestion pipeline."""
    with open("config/schema.yaml") as f:
        schema_cfg = yaml.safe_load(f)

    fmt_cfg = schema_cfg["formats"][source_format]

    df = read_file(input_path, source_format, fmt_cfg)
    df = map_columns(df, fmt_cfg["column_map"])
    df, _ = apply_filters(df, schema_cfg.get("exclusion_patterns", {}))
    df = decompose_paths(df, root_prefix)
    return df


def extract_entities_for_df(df: pd.DataFrame) -> pd.DataFrame:
    """Run entity extraction on every row (plot + address)."""
    plots = []
    addresses = []
    plot_depths = []
    address_depths = []

    for _, row in df.iterrows():
        segs = row.get("segments", [])
        if not isinstance(segs, list):
            segs = []
        matches = extract_entities_with_confidence(segs, config)

        plot_match = matches.get("plot")
        addr_match = matches.get("address")

        plots.append(plot_match.value if plot_match else "")
        addresses.append(addr_match.value if addr_match else "")
        plot_depths.append(plot_match.depth if plot_match else -1)
        address_depths.append(addr_match.depth if addr_match else -1)

    df = df.copy()
    df["entity_plot"] = plots
    df["entity_address"] = addresses
    df["entity_plot_depth"] = plot_depths
    df["entity_address_depth"] = address_depths
    return df


def analyse_dataset(name: str, input_path: str, root_prefix: str):
    """Full analysis for one dataset."""
    print(f"\n{'='*80}")
    print(f"  DATASET: {name}")
    print(f"{'='*80}")

    # Load and prepare
    print(f"\nLoading {input_path}...")
    df = load_dataset(input_path, root_prefix)
    print(f"  Loaded {len(df):,} rows after filtering")

    # Entity extraction (plot + address)
    print(f"  Running entity extraction...")
    df = extract_entities_for_df(df)

    has_plot = (df["entity_plot"] != "").sum()
    has_addr = (df["entity_address"] != "").sum()
    print(f"  Rows with plot entity: {has_plot:,} ({has_plot/len(df)*100:.1f}%)")
    print(f"  Rows with address entity: {has_addr:,} ({has_addr/len(df)*100:.1f}%)")
    print(f"  Unique plots: {df[df['entity_plot'] != '']['entity_plot'].nunique()}")
    print(f"  Unique addresses: {df[df['entity_address'] != '']['entity_address'].nunique()}")

    # Run scheme detection
    print(f"\n--- Scheme Detection ---")
    df = detect_and_assign_schemes(df, config)

    has_scheme = (df["entity_scheme"] != "").sum()
    unique_schemes = df[df["entity_scheme"] != ""]["entity_scheme"].nunique()
    print(f"  Rows with scheme: {has_scheme:,} ({has_scheme/len(df)*100:.1f}%)")
    print(f"  Unique schemes: {unique_schemes}")

    # Show all unique schemes with stats
    print(f"\n--- All Detected Schemes ({unique_schemes} total) ---")
    print(f"{'Scheme Name':<55} {'Conf':>5} {'Depth':>5} {'Files':>7} {'Plots':>6} {'Addrs':>6}")
    print("-" * 100)

    scheme_df = df[df["entity_scheme"] != ""]
    scheme_summary = []

    for scheme_name in scheme_df["entity_scheme"].unique():
        subset = scheme_df[scheme_df["entity_scheme"] == scheme_name]
        conf = subset["entity_scheme_confidence"].iloc[0]
        path = subset["entity_scheme_path"].iloc[0]
        unique_plots = subset[subset["entity_plot"] != ""]["entity_plot"].nunique()
        unique_addrs = subset[subset["entity_address"] != ""]["entity_address"].nunique()
        parts = path.split("\\")
        depth = len(parts) - 1

        scheme_summary.append({
            "name": scheme_name,
            "confidence": conf,
            "depth": depth,
            "files": len(subset),
            "plots": unique_plots,
            "addrs": unique_addrs,
            "path": path,
        })

    # Sort by files descending
    scheme_summary.sort(key=lambda x: -x["files"])

    for s in scheme_summary:
        print(
            f"{s['name'][:54]:<55} {s['confidence']:>5.2f} {s['depth']:>5} "
            f"{s['files']:>7} {s['plots']:>6} {s['addrs']:>6}"
        )

    # Show top 20 by file count with full path
    print(f"\n--- Top 20 Schemes (by file count) with paths ---")
    for s in scheme_summary[:20]:
        print(f"  {s['name']}")
        print(f"    Path:  {s['path']}")
        print(f"    Files: {s['files']:,} | Plots: {s['plots']} | Addrs: {s['addrs']} | Conf: {s['confidence']:.2f}")
        print()

    # Show files WITHOUT a scheme
    no_scheme = df[df["entity_scheme"] == ""]
    print(f"\n--- Files without scheme: {len(no_scheme):,} ({len(no_scheme)/len(df)*100:.1f}%) ---")
    if len(no_scheme) > 0:
        # Show sample paths
        sample = no_scheme.head(10)
        print("  Sample paths without scheme:")
        for _, row in sample.iterrows():
            segs = row.get("segments", [])
            if isinstance(segs, list) and segs:
                print(f"    {' > '.join(segs[:4])}")

    # Show scheme + address overlap
    print(f"\n--- Scheme/Address Overlap ---")
    overlap = scheme_df[scheme_df["entity_address"] != ""]
    if len(overlap) > 0:
        # Cases where scheme name matches the address entity
        matching = overlap[overlap["entity_scheme"] == overlap["entity_address"]]
        print(f"  Files where scheme == address: {len(matching):,}")
        diff = overlap[overlap["entity_scheme"] != overlap["entity_address"]]
        print(f"  Files where scheme != address: {len(diff):,}")
        if len(diff) > 0:
            # Show some examples of scheme != address (the good case)
            sample = diff.head(5)
            print(f"\n  Examples (scheme + separate address):")
            for _, row in sample.iterrows():
                print(f"    Scheme: {row['entity_scheme']}")
                print(f"    Address: {row['entity_address']}")
                segs = row.get("segments", [])
                if isinstance(segs, list):
                    print(f"    Path: {' > '.join(segs[:5])}")
                print()

    # Confidence distribution
    print(f"\n--- Scheme Confidence Distribution ---")
    if has_scheme > 0:
        confs = scheme_df["entity_scheme_confidence"]
        print(f"  Mean: {confs.mean():.2f}")
        print(f"  Median: {confs.median():.2f}")
        print(f"  Min: {confs.min():.2f}")
        print(f"  Max: {confs.max():.2f}")
        conf_counts = confs.value_counts().sort_index()
        for conf_val, count in conf_counts.items():
            print(f"    {conf_val:.2f}: {count:,} files")

    return df, scheme_summary


# Run analysis
if __name__ == "__main__":
    os.chdir(os.path.join(os.path.dirname(__file__), ".."))

    import argparse

    parser = argparse.ArgumentParser(
        description="Analyse scheme detection patterns from raw TreeSize XLSX exports.",
    )
    parser.add_argument("--name", required=True, help="Estate label (e.g. TechServe)")
    parser.add_argument("--input", required=True, help="Path to TreeSize XLSX export")
    parser.add_argument("--root", required=True, help="UNC root prefix to strip")
    args = parser.parse_args()

    df, schemes = analyse_dataset(args.name, args.input, args.root)
    print(f"\nDone — {len(schemes)} scheme patterns found.")
