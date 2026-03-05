"""
ComplyPoint Phase 1 — Metadata Discovery & Readiness Engine

CLI entry point and pipeline orchestration.

Usage (single estate):
    python src/main.py --config config/run_config.yaml --input data/export.xlsx

Usage (multi-estate):
    python src/main.py --multi config/multi_estate.yaml

The pipeline runs in a strict linear sequence:
    Read → Map Columns → Validate → Filter → Normalise → Decompose Paths
    → Classify → Apply Retention → Add Audit Fields → Write
"""

import hashlib
import json as _json
import logging
import subprocess
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

# Ensure src/ is on the Python path so ingestion.* imports resolve
sys.path.insert(0, str(Path(__file__).parent))

import click
import pandas as pd
import yaml

from ingestion.reader import read_file
from ingestion.schema import map_columns, validate
from ingestion.filters import apply_filters, FilterSummary
from ingestion.normaliser import normalise
from ingestion.path_parser import decompose_paths
from classification.config_loader import load_config
from classification.engine import classify
from classification.entity_cluster import cluster_entities
from classification.duplicate_detector import detect_duplicates
from classification.retention import load_retention_rules, apply_retention

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("complypoint")


# ------------------------------------------------------------------
# Configuration loading
# ------------------------------------------------------------------

def load_configs(config_path: Path) -> dict:
    """Load run config and merge with schema config."""
    with open(config_path) as f:
        run_config = yaml.safe_load(f)

    schema_path = config_path.parent / "schema.yaml"
    if not schema_path.exists():
        raise FileNotFoundError(
            f"Schema config not found at {schema_path}. "
            f"Expected it alongside the run config."
        )

    with open(schema_path) as f:
        schema_config = yaml.safe_load(f)

    run_config["schema"] = schema_config
    return run_config


# ------------------------------------------------------------------
# Per-estate pipeline (Steps 1–6)
# ------------------------------------------------------------------

def run_single_estate(
    run_config: dict,
    input_path: Path,
    cls_config,
    skip_clustering: bool = True,
    candidates_path: str | None = None,
) -> tuple[pd.DataFrame, FilterSummary]:
    """Run ingestion + classification for one estate.

    Executes Steps 1–6: Read → Map → Validate → Filter → Normalise →
    Decompose → Classify.  Returns the classified DataFrame and filter
    summary.  When called from multi-estate mode, skip_clustering=True
    so that clustering runs later on the combined data.

    Args:
        run_config: Loaded per-estate run config dict (with schema merged).
        input_path: Path to the estate's input file.
        cls_config: Pre-loaded ClassificationConfig.
        skip_clustering: Whether to skip the clustering step.

    Returns:
        (classified_df, filter_summary) tuple.
    """
    format_name = run_config["source"]["format"]
    schema_config = run_config["schema"]
    format_config = schema_config["formats"].get(format_name)
    source_system = run_config.get("source", {}).get("source_system", "")

    if not format_config:
        available = list(schema_config["formats"].keys())
        raise ValueError(
            f"Format '{format_name}' not found in schema.yaml. Available: {available}"
        )

    # STEP 1: Read raw file
    logger.info("=" * 60)
    logger.info("STEP 1: Reading input file — %s", input_path.name)
    df = read_file(input_path, format_name, format_config)

    # STEP 2: Map columns to internal schema
    logger.info("=" * 60)
    logger.info("STEP 2: Mapping columns to internal schema")
    column_map = format_config["column_map"]
    df = map_columns(df, column_map)

    required = schema_config["internal_schema"]["required_columns"]
    validate(df, required)

    # STEP 3: Filter rows
    logger.info("=" * 60)
    logger.info("STEP 3: Filtering rows")
    exclusion_config = schema_config.get("exclusion_patterns", {})
    df, filter_summary = apply_filters(df, exclusion_config)

    # STEP 4: Normalise data
    logger.info("=" * 60)
    logger.info("STEP 4: Normalising data")
    df = normalise(df)

    # STEP 5: Decompose paths
    logger.info("=" * 60)
    logger.info("STEP 5: Decomposing paths")
    root_prefix = run_config["source"]["root_prefix"]
    df = decompose_paths(df, root_prefix)

    # STEP 6: Classify
    logger.info("=" * 60)
    logger.info("STEP 6: Classifying documents")
    df = classify(
        df, cls_config,
        skip_clustering=skip_clustering,
        candidates_path=candidates_path,
    )

    # Tag rows with source metadata
    df["source_system"] = source_system
    df["root_prefix"] = root_prefix

    return df, filter_summary


# ------------------------------------------------------------------
# Output helpers
# ------------------------------------------------------------------

def write_jsonl(
    df: pd.DataFrame, output_path: Path, weights: dict | None = None,
) -> None:
    """Write DataFrame to JSONL in chunks."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    chunk_size = (weights or {}).get("pipeline", {}).get("jsonl_chunk_size", 10_000)
    with open(output_path, "w", encoding="utf-8") as fh:
        for i in range(0, len(df), chunk_size):
            for record in df.iloc[i : i + chunk_size].to_dict(orient="records"):
                fh.write(_json.dumps(record, default=str) + "\n")


def add_audit_fields(
    df: pd.DataFrame, batch_run_id: str, weights: dict | None = None,
) -> pd.DataFrame:
    """Add identity and audit columns to a DataFrame."""
    hash_len = (weights or {}).get("pipeline", {}).get("file_id_hash_length", 16)
    df["file_id"] = df["full_path"].apply(
        lambda p: hashlib.sha256(str(p).encode("utf-8")).hexdigest()[:hash_len]
    )
    df["filename"] = df.apply(
        lambda r: f"{r['filename_stem']}.{r['extension']}"
        if r.get("extension") else r.get("filename_stem", ""),
        axis=1,
    )
    df["batch_run_id"] = batch_run_id
    df["run_timestamp"] = datetime.now(timezone.utc).isoformat()
    return df


def print_summary(
    df: pd.DataFrame,
    label: str,
    filter_summary: FilterSummary | None,
    elapsed: float,
    output_path: Path,
) -> None:
    """Print pipeline summary statistics."""
    high = (df["confidence_band"] == "High").sum()
    medium = (df["confidence_band"] == "Medium").sum()
    low = (df["confidence_band"] == "Low").sum()

    # Clustering statistics
    entity_cluster_stats = []
    for col in df.columns:
        if col.endswith("_cluster_id") and col.startswith("entity_"):
            entity_type = col.replace("entity_", "").replace("_cluster_id", "")
            n_clusters = df[col][df[col] != ""].nunique()
            if n_clusters:
                entity_cluster_stats.append(f"{n_clusters} {entity_type}")

    potential_dups = 0
    if "potential_duplicate" in df.columns:
        potential_dups = df["potential_duplicate"].sum()
    dup_groups = 0
    if "dup_group_id" in df.columns:
        non_empty = df["dup_group_id"] != ""
        dup_groups = df.loc[non_empty, "dup_group_id"].nunique() if non_empty.any() else 0

    click.echo(f"\n{'=' * 60}")
    click.echo(f"Pipeline Summary — {label}")
    click.echo(f"{'=' * 60}")
    if filter_summary:
        click.echo(
            f"  Records:   {filter_summary.initial_count:,} "
            f"-> {filter_summary.final_count:,}"
        )
        click.echo(f"  Removed:   {filter_summary.total_removed:,}")
    else:
        click.echo(f"  Records:   {len(df):,}")
    click.echo(f"  Confidence: {high:,} High, {medium:,} Medium, {low:,} Low")
    if len(df):
        click.echo(f"  Phase 2:   {high:,} ready ({high / len(df) * 100:.1f}%)")
    else:
        click.echo("  Phase 2:   0 ready")
    if entity_cluster_stats:
        click.echo(f"  Entity clusters: {', '.join(entity_cluster_stats)}")
    if dup_groups:
        click.echo(f"  Duplicate groups: {dup_groups:,}")
    if potential_dups:
        click.echo(f"  Potential duplicates: {potential_dups:,} files")
    click.echo(f"  Output:    {output_path}")
    click.echo(f"  Time:      {elapsed:.1f}s")


def run_reports(
    output_path: Path,
    root_prefix: str,
    candidates_path: str | None = None,
    weights: dict | None = None,
) -> None:
    """Run report generation scripts."""
    report_timeout = (weights or {}).get("pipeline", {}).get("report_timeout", 600)
    logger.info("=" * 60)
    logger.info("Generating reports")

    scripts_dir = Path(__file__).parent.parent / "scripts"
    reports_dir = Path(__file__).parent.parent / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    folder_tree_cmd = [
        sys.executable, str(scripts_dir / "folder_tree.py"),
        str(output_path), root_prefix,
    ]
    if candidates_path:
        folder_tree_cmd += ["--scheme-candidates", candidates_path]

    report_scripts = [
        {
            "name": "scheme_report",
            "cmd": [
                sys.executable, str(scripts_dir / "scheme_report.py"),
                "--jsonl", str(output_path), root_prefix,
            ],
        },
        {
            "name": "entity_report",
            "cmd": [
                sys.executable, str(scripts_dir / "entity_report.py"),
                str(output_path), "--output-dir", str(reports_dir),
            ],
        },
        {
            "name": "folder_tree",
            "cmd": folder_tree_cmd,
        },
        {
            "name": "cluster_review",
            "cmd": [
                sys.executable, str(scripts_dir / "cluster_review.py"),
                str(output_path), "--output-dir", str(reports_dir),
            ],
        },
        {
            "name": "duplicate_report",
            "cmd": [
                sys.executable, str(scripts_dir / "duplicate_report.py"),
                str(output_path), "--output-dir", str(reports_dir),
            ],
        },
    ]

    for script in report_scripts:
        try:
            result = subprocess.run(
                script["cmd"],
                capture_output=True, text=True, timeout=report_timeout,
            )
            if result.returncode == 0:
                logger.info(f"  {script['name']}: OK")
            else:
                stderr_snippet = result.stderr.strip().split("\n")[-1][:200]
                logger.warning(f"  {script['name']}: FAILED — {stderr_snippet}")
        except subprocess.TimeoutExpired:
            logger.warning(f"  {script['name']}: TIMEOUT (>{report_timeout}s)")
        except Exception as e:
            logger.warning(f"  {script['name']}: ERROR — {e}")

    logger.info("Reports written to: %s", reports_dir)


def run_csv_conversion(jsonl_path: Path) -> None:
    """Convert JSONL output to Dataverse CSV and full CSV."""
    scripts_dir = Path(__file__).parent.parent / "scripts"
    for script_name, suffix in [
        ("jsonl_to_csv_slim.py", "_dataverse.csv"),
        ("jsonl_to_csv.py", ".csv"),
    ]:
        script = scripts_dir / script_name
        if not script.exists():
            logger.warning("  CSV conversion script not found: %s", script)
            continue
        csv_path = jsonl_path.with_name(jsonl_path.stem + suffix)
        try:
            result = subprocess.run(
                [sys.executable, str(script), str(jsonl_path), str(csv_path)],
                capture_output=True, text=True, timeout=600,
            )
            if result.returncode == 0:
                logger.info("  CSV: %s", csv_path)
            else:
                stderr_snippet = result.stderr.strip().split("\n")[-1][:200]
                logger.warning("  %s FAILED: %s", script_name, stderr_snippet)
        except Exception as e:
            logger.warning("  %s ERROR: %s", script_name, e)


# ------------------------------------------------------------------
# Single-estate pipeline
# ------------------------------------------------------------------

def run_single_pipeline(
    config_path: Path,
    input_path: Path,
    output_path: str | None,
    no_cluster: bool,
) -> None:
    """Run the full pipeline for a single estate."""
    start = time.time()

    logger.info("ComplyPoint Phase 1 — Discovery & Readiness Pipeline")
    logger.info(f"Input: {input_path.name}")

    # Load configs
    run_config = load_configs(config_path)
    config_dir = config_path.parent
    industry = run_config["industry"]
    cls_config = load_config(config_dir, industry)

    # Resolve output path early so scheme candidates sidecar can be co-located.
    if output_path is None:
        output_dir = Path(run_config.get("output", {}).get("directory", "output"))
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"{input_path.stem}_classified.jsonl"
    else:
        output_path = Path(output_path)

    stem = output_path.stem.replace("_classified", "")
    candidates_path = str(output_path.parent / f"{stem}_scheme_candidates.json")

    # Run estate pipeline (classification with clustering as requested)
    df, filter_summary = run_single_estate(
        run_config, input_path, cls_config,
        skip_clustering=no_cluster,
        candidates_path=candidates_path,
    )

    # Apply retention rules
    logger.info("=" * 60)
    logger.info("Applying retention rules")
    retention_rules = load_retention_rules(config_dir, industry)
    df = apply_retention(df, retention_rules)

    # Add audit fields
    logger.info("=" * 60)
    logger.info("Adding identity & audit fields")
    batch_run_id = str(uuid.uuid4())
    df = add_audit_fields(df, batch_run_id, weights=cls_config.weights)
    logger.info(f"  batch_run_id: {batch_run_id}")
    source_system = run_config.get("source", {}).get("source_system", "")
    if source_system:
        logger.info(f"  source_system: {source_system}")

    # Write output
    logger.info("=" * 60)
    logger.info("Writing output")

    write_jsonl(df, output_path, weights=cls_config.weights)

    elapsed = time.time() - start
    logger.info(f"Output: {output_path} ({len(df):,} records)")
    logger.info(f"Pipeline complete in {elapsed:.1f}s")

    # Summary
    root_prefix = run_config["source"]["root_prefix"]
    format_name = run_config["source"]["format"]
    print_summary(df, input_path.name, filter_summary, elapsed, output_path)

    # Reports
    run_reports(
        output_path, root_prefix,
        candidates_path=candidates_path, weights=cls_config.weights,
    )

    # CSV conversion
    logger.info("=" * 60)
    logger.info("Converting to CSV")
    run_csv_conversion(output_path)


# ------------------------------------------------------------------
# Multi-estate pipeline
# ------------------------------------------------------------------

def run_multi_estate(
    multi_path: Path,
    no_cluster: bool,
    output_path: str | None,
) -> None:
    """Run the pipeline across multiple estates with cross-estate clustering.

    Each estate is classified independently.  Entity clustering then runs
    on the combined results so that entities (addresses, plots, schemes)
    are grouped across all estates.
    """
    start = time.time()

    logger.info("ComplyPoint Phase 1 — Multi-Estate Pipeline")
    logger.info(f"Config: {multi_path.name}")

    # Load multi-estate config
    with open(multi_path) as f:
        multi_config = yaml.safe_load(f)

    estates = multi_config.get("estates", [])
    if not estates:
        raise click.UsageError("No estates defined in multi-estate config")

    industry = multi_config["industry"]
    config_dir = multi_path.parent
    output_dir = Path(multi_config.get("output", {}).get("directory", "output"))
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load classification config once (shared across estates)
    cls_config = load_config(config_dir, industry)
    retention_rules = load_retention_rules(config_dir, industry)

    # ---- Phase A: Process each estate independently ----

    estate_dfs: list[pd.DataFrame] = []
    # Track (source_system, filter_summary) per estate for the summary
    estate_summaries: list[tuple[str, FilterSummary]] = []
    estate_root_prefixes: dict[str, str] = {}

    for estate in estates:
        estate_name = estate["name"]
        estate_config_path = Path(estate["config"])
        estate_input_path = Path(estate["input"])

        logger.info("=" * 60)
        logger.info(f"ESTATE: {estate_name}")
        logger.info("=" * 60)

        # Validate paths exist
        if not estate_config_path.exists():
            logger.error(
                f"  Config not found: {estate_config_path} — skipping estate"
            )
            continue
        if not estate_input_path.exists():
            logger.error(
                f"  Input not found: {estate_input_path} — skipping estate"
            )
            continue

        try:
            run_config = load_configs(estate_config_path)
            source_system = run_config.get("source", {}).get(
                "source_system", estate_name,
            )

            df, filter_summary = run_single_estate(
                run_config, estate_input_path, cls_config,
                skip_clustering=True,  # always skip — clustering runs cross-estate
            )

            estate_summaries.append((source_system, filter_summary))
            estate_root_prefixes[source_system] = run_config["source"]["root_prefix"]

            # Write per-estate JSONL (before cross-estate clustering)
            per_estate_path = output_dir / f"{estate_input_path.stem}_classified.jsonl"
            per_estate_df = df.copy()
            per_estate_df = add_audit_fields(
                per_estate_df, str(uuid.uuid4()), weights=cls_config.weights,
            )
            per_estate_df = apply_retention(per_estate_df, retention_rules)
            write_jsonl(per_estate_df, per_estate_path, weights=cls_config.weights)
            logger.info(
                f"  Per-estate output: {per_estate_path} ({len(df):,} records)"
            )

            estate_dfs.append(df)

        except Exception as e:
            logger.error(f"  Estate '{estate_name}' FAILED: {e}")
            continue

    if not estate_dfs:
        raise RuntimeError("All estates failed — nothing to combine")

    # ---- Phase B: Cross-estate processing ----

    logger.info("=" * 60)
    logger.info("CROSS-ESTATE: Combining %d estates", len(estate_dfs))
    logger.info("=" * 60)

    combined = pd.concat(estate_dfs, ignore_index=True)
    logger.info(f"  Combined: {len(combined):,} rows from {len(estate_dfs)} estates")

    # Cross-estate entity clustering
    if not no_cluster:
        logger.info("=" * 60)
        logger.info("CROSS-ESTATE: Entity clustering")
        combined = cluster_entities(combined, cls_config)

    # Cross-estate duplicate detection
    logger.info("=" * 60)
    logger.info("CROSS-ESTATE: Duplicate detection")
    combined = detect_duplicates(combined, cls_config.weights)

    # Apply retention rules on combined data
    logger.info("=" * 60)
    logger.info("Applying retention rules")
    combined = apply_retention(combined, retention_rules)

    # Shared audit fields across all estates
    logger.info("=" * 60)
    logger.info("Adding identity & audit fields")
    batch_run_id = str(uuid.uuid4())
    combined = add_audit_fields(combined, batch_run_id, weights=cls_config.weights)
    logger.info(f"  batch_run_id: {batch_run_id}")

    # ---- Write combined output ----

    logger.info("=" * 60)
    logger.info("Writing combined output")

    if output_path is not None:
        combined_path = Path(output_path)
    else:
        combined_path = output_dir / "combined_classified.jsonl"

    write_jsonl(combined, combined_path, weights=cls_config.weights)

    elapsed = time.time() - start
    logger.info(f"Combined output: {combined_path} ({len(combined):,} records)")
    logger.info(f"Pipeline complete in {elapsed:.1f}s")

    # ---- Summary ----

    # Per-estate breakdown
    for estate_name, fs in estate_summaries:
        click.echo(f"\n{'=' * 60}")
        click.echo(f"Estate: {estate_name}")
        click.echo(f"{'=' * 60}")
        click.echo(f"  Records: {fs.initial_count:,} -> {fs.final_count:,}")
        click.echo(f"  Removed: {fs.total_removed:,}")
        estate_mask = combined["source_system"] == estate_name
        if estate_mask.any():
            estate_subset = combined[estate_mask]
            h = (estate_subset["confidence_band"] == "High").sum()
            m = (estate_subset["confidence_band"] == "Medium").sum()
            l_ = (estate_subset["confidence_band"] == "Low").sum()
            click.echo(f"  Confidence: {h:,} High, {m:,} Medium, {l_:,} Low")

    # Combined summary
    print_summary(
        combined, "Combined (all estates)",
        filter_summary=None, elapsed=elapsed, output_path=combined_path,
    )

    # Reports on combined output (use first estate's root_prefix for
    # report scripts that need one; reports should group by source_system)
    first_root = next(iter(estate_root_prefixes.values()), "")
    run_reports(combined_path, first_root, weights=cls_config.weights)

    # CSV conversion
    logger.info("=" * 60)
    logger.info("Converting to CSV")
    run_csv_conversion(combined_path)


# ------------------------------------------------------------------
# CLI entry point
# ------------------------------------------------------------------

@click.command()
@click.option(
    "--config", "config_path",
    default=None,
    type=click.Path(exists=True),
    help="Path to run config YAML (e.g. config/run_config.yaml)",
)
@click.option(
    "--input", "input_path",
    default=None,
    type=click.Path(exists=True),
    help="Path to input file (XLSX or CSV)",
)
@click.option(
    "--multi", "multi_path",
    default=None,
    type=click.Path(exists=True),
    help="Path to multi-estate config YAML (e.g. config/multi_estate.yaml)",
)
@click.option(
    "--output", "output_path",
    default=None,
    type=click.Path(),
    help="Output path (default: output/<input_name>_classified.jsonl)",
)
@click.option(
    "--no-cluster", "no_cluster",
    is_flag=True,
    default=False,
    help="Skip entity clustering and file version/duplicate detection",
)
def main(
    config_path: str | None,
    input_path: str | None,
    multi_path: str | None,
    output_path: str | None,
    no_cluster: bool,
):
    """Run the Phase 1 ingestion pipeline."""
    if multi_path:
        if config_path or input_path:
            raise click.UsageError(
                "--multi cannot be used with --config/--input"
            )
        run_multi_estate(Path(multi_path), no_cluster, output_path)
    elif config_path and input_path:
        run_single_pipeline(
            Path(config_path), Path(input_path), output_path, no_cluster,
        )
    else:
        # Default: run multi-estate config
        default_multi = Path(__file__).parent.parent / "config" / "multi_estate.yaml"
        if not default_multi.exists():
            raise click.UsageError(
                f"No arguments given and default config not found at {default_multi}.\n"
                "Either run with --multi <path> or --config <path> --input <path>."
            )
        logger.info("No arguments given — using default: %s", default_multi)
        run_multi_estate(Path(default_multi), no_cluster, output_path)


if __name__ == "__main__":
    main()
