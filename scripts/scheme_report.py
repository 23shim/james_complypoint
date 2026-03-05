"""
Generate scheme detection reports as CSV files for review.

Outputs:
  reports/schemes_techserve.csv — all schemes for TechServe
  reports/schemes_win95.csv — all schemes for Win95
"""

import sys
import os
sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import pandas as pd
import yaml

from ingestion.reader import read_file
from ingestion.schema import map_columns
from ingestion.filters import apply_filters
from ingestion.path_parser import decompose_paths
from classification.config_loader import load_config
from classification.entity_extractor import extract_entities_with_confidence
from classification.scheme_detector import detect_and_assign_schemes

config = load_config("config", "housing")


def load_and_extract(input_path, root_prefix, source_format="treesize"):
    with open("config/schema.yaml") as f:
        schema_cfg = yaml.safe_load(f)
    fmt_cfg = schema_cfg["formats"][source_format]

    df = read_file(input_path, source_format, fmt_cfg)
    df = map_columns(df, fmt_cfg["column_map"])
    df, _ = apply_filters(df, schema_cfg.get("exclusion_patterns", {}))
    df = decompose_paths(df, root_prefix)

    # Entity extraction
    plots, addresses, plot_depths, address_depths = [], [], [], []
    for _, row in df.iterrows():
        segs = row.get("segments", [])
        if not isinstance(segs, list):
            segs = []
        matches = extract_entities_with_confidence(segs, config)
        pm = matches.get("plot")
        am = matches.get("address")
        plots.append(pm.value if pm else "")
        addresses.append(am.value if am else "")
        plot_depths.append(pm.depth if pm else -1)
        address_depths.append(am.depth if am else -1)

    df = df.copy()
    df["entity_plot"] = plots
    df["entity_address"] = addresses
    df["entity_plot_depth"] = plot_depths
    df["entity_address_depth"] = address_depths

    # Scheme detection
    df = detect_and_assign_schemes(df, config)
    return df


def build_scheme_report(df, dataset_name, root_prefix=""):
    """Build a summary DataFrame of all detected schemes."""
    scheme_df = df[df["entity_scheme"] != ""]
    if len(scheme_df) == 0:
        print(f"  No schemes detected for {dataset_name}")
        return pd.DataFrame()

    rows = []
    for scheme_path in scheme_df["entity_scheme_path"].unique():
        subset = scheme_df[scheme_df["entity_scheme_path"] == scheme_path]
        scheme_name = subset["entity_scheme"].iloc[0]
        conf = subset["entity_scheme_confidence"].iloc[0]

        # Organisational (validated single-best) plots and addresses
        unique_plots = sorted(subset[subset["entity_plot"] != ""]["entity_plot"].unique().tolist())
        unique_addrs = sorted(subset[subset["entity_address"] != ""]["entity_address"].unique().tolist())

        # Raw (all string/pattern matches) — aggregated across every file in the scheme
        def _collect_raw(col):
            if col not in subset.columns:
                return []
            vals: set[str] = set()
            for cell in subset[col]:
                if isinstance(cell, list):
                    for v in cell:
                        if isinstance(v, dict):
                            val = v.get("value", "")
                        else:
                            val = v
                        if val:
                            vals.add(val)
                elif isinstance(cell, str) and cell:
                    # Fallback if loaded from CSV as a repr string
                    vals.add(cell)
            return sorted(vals)

        raw_plots = _collect_raw("raw_plots")
        raw_addrs = _collect_raw("raw_addresss")

        # Depth from path
        parts = scheme_path.split("\\")
        depth = len(parts) - 1

        # Parent path (what folder the scheme sits inside)
        parent = "\\".join(parts[:-1]) if len(parts) > 1 else "(root)"

        # Full original path with root prefix
        if root_prefix:
            full_original = root_prefix.rstrip("\\") + "\\" + scheme_path
        else:
            full_original = scheme_path

        def _sig(col, default=0):
            return subset[col].iloc[0] if col in subset.columns else default

        rows.append({
            "scheme_name": scheme_name,
            "confidence": conf,
            "depth": depth,
            "file_count": len(subset),
            # Organisational (validated single-best) entity counts
            "org_plots": len(unique_plots),
            "org_addresses": len(unique_addrs),
            "org_total_entities": len(unique_plots) + len(unique_addrs),
            # Raw (all string/pattern matches) entity counts
            "raw_plot_count": len(raw_plots),
            "raw_address_count": len(raw_addrs),
            "raw_total_entities": len(raw_plots) + len(raw_addrs),
            "place_name_match": bool(_sig("entity_scheme_place_name_match", False)),
            "address_like_name": bool(_sig("entity_scheme_address_like_name", False)),
            "parent_folder": parent,
            "full_path": full_original,
            # Sample values for review
            "sample_org_plots": "; ".join(unique_plots[:5]),
            "sample_org_addresses": "; ".join(unique_addrs[:5]),
            "all_raw_plots": "; ".join(raw_plots),
            "all_raw_addresses": "; ".join(raw_addrs),
        })

    report = pd.DataFrame(rows)
    report = report.sort_values("file_count", ascending=False).reset_index(drop=True)
    report.index += 1  # 1-based index
    report.index.name = "#"
    return report


def load_jsonl(path):
    """Load a classified JSONL file into a DataFrame."""
    import json
    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    return pd.DataFrame(records)


def _print_report(report, name, out_path):
    """Print and save a scheme report."""
    report.to_csv(out_path, encoding="utf-8-sig")
    print(f"  Wrote {len(report)} schemes to {out_path}")

    print(f"\n  {'#':>4}  {'Scheme Name':<50} {'Conf':>5} {'Dep':>3} {'Files':>6} {'OrgP':>5} {'RawP':>5} {'RawA':>5} {'Parent Folder':<40}")
    print(f"  {'-'*4}  {'-'*50} {'-'*5} {'-'*3} {'-'*6} {'-'*5} {'-'*5} {'-'*5} {'-'*40}")
    for idx, row in report.iterrows():
        parent = row["parent_folder"]
        if len(parent) > 39:
            parent = "..." + parent[-36:]
        print(
            f"  {idx:>4}  {row['scheme_name'][:49]:<50} "
            f"{row['confidence']:>5.2f} {row['depth']:>3} "
            f"{row['file_count']:>6} {row['org_plots']:>5} "
            f"{row['raw_plot_count']:>5} {row['raw_address_count']:>5} {parent:<40}"
        )


if __name__ == "__main__":
    os.chdir(os.path.join(os.path.dirname(__file__), ".."))
    os.makedirs("reports", exist_ok=True)

    # Mode 1: --jsonl <path> — read from classified JSONL (fast, uses latest output)
    if len(sys.argv) >= 3 and sys.argv[1] == "--jsonl":
        jsonl_path = sys.argv[2]
        root_prefix = sys.argv[3] if len(sys.argv) > 3 else ""
        stem = os.path.splitext(os.path.basename(jsonl_path))[0].replace("_classified", "")
        out_path = f"reports/schemes_{stem}_v7.csv"

        print(f"\n{'='*80}")
        print(f"  Scheme report from JSONL: {jsonl_path}")
        print(f"{'='*80}")

        df = load_jsonl(jsonl_path)
        total = len(df)
        with_scheme = (df["entity_scheme"] != "").sum()
        print(f"  Total files: {total:,}")
        print(f"  With scheme: {with_scheme:,} ({with_scheme/total*100:.1f}%)")
        print(f"  Without scheme: {total - with_scheme:,} ({(total-with_scheme)/total*100:.1f}%)")

        report = build_scheme_report(df, stem, root_prefix=root_prefix)
        if len(report) > 0:
            _print_report(report, stem, out_path)

        print("\nDone.")
    else:
        # Mode 2: re-run pipeline from scratch (requires raw XLSX)
        if len(sys.argv) < 5:
            sys.exit(
                "Usage:\n"
                "  Mode 1 (from JSONL):  python scripts/scheme_report.py --jsonl <classified.jsonl> [root_prefix]\n"
                "  Mode 2 (from XLSX):   python scripts/scheme_report.py --name <label> --input <file.xlsx> --root <prefix> [--output <report.csv>]"
            )

        import argparse
        parser = argparse.ArgumentParser()
        parser.add_argument("--name", required=True, help="Estate label")
        parser.add_argument("--input", required=True, help="TreeSize XLSX path")
        parser.add_argument("--root", required=True, help="UNC root prefix")
        parser.add_argument("--output", default=None, help="Output CSV path")
        args = parser.parse_args()

        out_path = args.output or f"reports/schemes_{args.name.lower()}.csv"

        print(f"\n{'='*80}")
        print(f"  {args.name}")
        print(f"{'='*80}")

        df = load_and_extract(args.input, args.root)
        total = len(df)
        with_scheme = (df["entity_scheme"] != "").sum()
        print(f"  Total files: {total:,}")
        print(f"  With scheme: {with_scheme:,} ({with_scheme/total*100:.1f}%)")
        print(f"  Without scheme: {total - with_scheme:,} ({(total-with_scheme)/total*100:.1f}%)")

        report = build_scheme_report(df, args.name, root_prefix=args.root)
        if len(report) > 0:
            _print_report(report, args.name, out_path)

        print("\nDone.")
