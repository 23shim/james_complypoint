"""Folder Entity Analysis — leaf-folder view with plot/address conflicts.

Loads a classified JSONL and produces a CSV showing all leaf folders
(deepest unique folder paths, with the last segment removed for grouping)
annotated with their plot and address entity matches.  Any folder where
files received *different* entity assignments is flagged as a conflict,
making it easy to assess where the algorithm is and isn't consistent.

Algorithm
---------
1. For each file, compute a "display path" = segments[:-1] joined.
   (Strips the immediate parent folder of the file, giving a slightly
   higher-level grouping that reduces total row count.)
2. Find leaf display paths — paths that are not a prefix of any other
   display path in the dataset.  This ensures we see only the deepest
   unique folder, not every ancestor.
3. For each leaf folder, aggregate across all files that fall in it:
   - Most-selected plot / address entity (mode)
   - All distinct values seen (to reveal conflicts)
   - Average confidence of matched files
4. Filter to only folders with at least one plot or address match.
5. Sort: conflicts first, then by path.

Usage
-----
    python scripts/folder_entity_analysis.py <jsonl_path>
    python scripts/folder_entity_analysis.py <jsonl_path> --output-dir reports/

Output
------
    reports/folder_entity_analysis_<stem>.csv
"""

import argparse
import json
import os
import sys
from collections import Counter
from pathlib import Path

import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")

SEP = "\\"


# ------------------------------------------------------------------ #
#  I/O helpers                                                        #
# ------------------------------------------------------------------ #

def _load_jsonl(path: str) -> pd.DataFrame:
    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    df = pd.DataFrame(records)
    print(f"Loaded {len(df):,} records from {Path(path).name}")
    return df


# ------------------------------------------------------------------ #
#  Leaf-path detection                                                #
# ------------------------------------------------------------------ #

def _find_leaf_paths(paths: set) -> list[str]:
    """Return only paths that have no descendant in *paths*.

    A path P is a leaf when no other path Q in the set starts with
    P + SEP (i.e. P is not an ancestor of Q).
    """
    result = []
    for p in paths:
        prefix = p + SEP
        if not any(other.startswith(prefix) for other in paths):
            result.append(p)
    return sorted(result)


# ------------------------------------------------------------------ #
#  Per-entity aggregation                                             #
# ------------------------------------------------------------------ #

def _safe_str(v) -> str:
    s = str(v).strip() if v is not None else ""
    return "" if s in ("nan", "None", "none") else s


def _agg_entity(
    group: pd.DataFrame,
    val_col: str,
    conf_col: str,
) -> dict:
    """Aggregate entity values and confidences across a folder's files."""
    raw_vals = [_safe_str(v) for v in group.get(val_col, pd.Series(dtype=str))]
    raw_vals = [v for v in raw_vals if v]

    raw_confs = []
    if conf_col in group.columns:
        for val, conf in zip(
            group.get(val_col, pd.Series(dtype=str)),
            group.get(conf_col, pd.Series(dtype=float)),
        ):
            sv = _safe_str(val)
            if sv:
                try:
                    c = float(conf)
                    if c > 0:
                        raw_confs.append(c)
                except (TypeError, ValueError):
                    pass

    counter = Counter(raw_vals)
    if not counter:
        return {
            "selected": "",
            "confidence": "",
            "match_count": 0,
            "n_distinct": 0,
            "all_values": "",
            "conflict": "",
        }

    selected = counter.most_common(1)[0][0]
    selected_count = counter[selected]
    conf_avg = round(sum(raw_confs) / len(raw_confs), 2) if raw_confs else ""
    distinct_vals = sorted(set(raw_vals))
    n_distinct = len(distinct_vals)

    return {
        "selected": selected,
        "selected_pct": f"{selected_count}/{len(raw_vals)}",
        "confidence": conf_avg,
        "match_count": len(raw_vals),
        "n_distinct": n_distinct,
        "all_values": "; ".join(distinct_vals),
        "conflict": "Yes" if n_distinct > 1 else "",
    }


# ------------------------------------------------------------------ #
#  Main                                                               #
# ------------------------------------------------------------------ #

def main():
    parser = argparse.ArgumentParser(
        description="Folder entity analysis with conflict detection"
    )
    parser.add_argument("jsonl_path", help="Path to classified JSONL")
    parser.add_argument(
        "--output-dir", default="reports", help="Output directory (default: reports/)"
    )
    args = parser.parse_args()

    # Work from project root so relative paths resolve correctly
    os.chdir(os.path.join(os.path.dirname(__file__), ".."))

    df = _load_jsonl(args.jsonl_path)

    # ---------------------------------------------------------------
    # Build display paths: strip last folder segment
    # segments = folder components only (filename already excluded)
    # depth >= 2 required to have anything left after stripping
    # ---------------------------------------------------------------
    def _display_path(segs):
        if not isinstance(segs, list) or len(segs) < 2:
            return None
        return SEP.join(segs[:-1])

    df["_display_path"] = df["segments"].apply(_display_path)
    df_valid = df[df["_display_path"].notna()].copy()

    skipped = len(df) - len(df_valid)
    print(
        f"Files with display path (depth >= 2): {len(df_valid):,}  "
        f"({skipped:,} skipped — at root or top-level folder)"
    )

    # ---------------------------------------------------------------
    # Find leaf display paths
    # ---------------------------------------------------------------
    unique_display = set(df_valid["_display_path"].unique())
    print(f"Unique display paths: {len(unique_display):,}")

    leaf_paths = _find_leaf_paths(unique_display)
    print(f"Leaf display paths:   {len(leaf_paths):,}")

    # ---------------------------------------------------------------
    # Restrict to files inside leaf folders and group
    # ---------------------------------------------------------------
    leaf_set = set(leaf_paths)
    df_leaf = df_valid[df_valid["_display_path"].isin(leaf_set)].copy()
    print(f"Files in leaf folders: {len(df_leaf):,}\n")

    rows = []

    for display_path, group in df_leaf.groupby("_display_path", sort=True):
        plot  = _agg_entity(group, "entity_plot",    "entity_plot_confidence")
        addr  = _agg_entity(group, "entity_address", "entity_address_confidence")

        # Skip folders with no entity matches at all
        if not plot["selected"] and not addr["selected"]:
            continue

        # Scheme context — show unique non-empty values
        scheme_vals = sorted({
            _safe_str(v) for v in group.get("entity_scheme", pd.Series(dtype=str))
            if _safe_str(v)
        })
        scheme_str = "; ".join(scheme_vals)

        has_conflict = "Yes" if (plot["conflict"] or addr["conflict"]) else ""
        depth = display_path.count(SEP) + 1

        rows.append({
            "folder_path":          display_path,
            "depth":                depth,
            "file_count":           len(group),
            "has_conflict":         has_conflict,

            # Plot
            "plot_selected":        plot["selected"],
            "plot_selected_pct":    plot.get("selected_pct", ""),
            "plot_confidence":      plot["confidence"],
            "plot_conflict":        plot["conflict"],
            "plot_n_distinct":      plot["n_distinct"] if plot["selected"] else "",
            "plot_all_values":      plot["all_values"],

            # Address
            "address_selected":     addr["selected"],
            "address_selected_pct": addr.get("selected_pct", ""),
            "address_confidence":   addr["confidence"],
            "address_conflict":     addr["conflict"],
            "address_n_distinct":   addr["n_distinct"] if addr["selected"] else "",
            "address_all_values":   addr["all_values"],

            # Context
            "scheme":               scheme_str,
        })

    # ---------------------------------------------------------------
    # Sort: conflicts first, then alphabetically by path
    # ---------------------------------------------------------------
    out_df = pd.DataFrame(rows)
    if not out_df.empty:
        out_df = out_df.sort_values(
            ["has_conflict", "folder_path"],
            ascending=[False, True],
        ).reset_index(drop=True)

    # ---------------------------------------------------------------
    # Summary
    # ---------------------------------------------------------------
    total = len(out_df)
    print(f"Leaf folders with entity matches: {total:,}")
    if total:
        n_conflicts = (out_df["has_conflict"] == "Yes").sum()
        n_plot      = (out_df["plot_selected"] != "").sum()
        n_addr      = (out_df["address_selected"] != "").sum()
        print(f"  Has conflict:    {n_conflicts:,}  ({n_conflicts/total*100:.1f}%)")
        print(f"  Has plot match:  {n_plot:,}  ({n_plot/total*100:.1f}%)")
        print(f"  Has addr match:  {n_addr:,}  ({n_addr/total*100:.1f}%)")

    # ---------------------------------------------------------------
    # Write output
    # ---------------------------------------------------------------
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = Path(args.jsonl_path).stem.replace("_classified", "")
    out_path = output_dir / f"folder_entity_analysis_{stem}.csv"
    out_df.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"\nWritten: {out_path}")


if __name__ == "__main__":
    main()
