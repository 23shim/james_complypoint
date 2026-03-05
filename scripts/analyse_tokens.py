"""
Token Frequency Analysis — Dictionary Gap Discovery Tool

Runs the standard ingestion pipeline (read → filter → normalise → decompose),
then tokenises every folder segment and filename stem to produce a ranked
frequency table.  Tokens already covered by the loaded classification config
are flagged so you can focus on UNKNOWN terms — the ones most likely to
represent missing dictionary entries.

KEY METRIC: "project diversity" — how many distinct top-level folders
(i.e. project sites) contain each token.  A token like "inspections"
that appears across 20+ projects is far more likely to be a real document
type signal than "dalefield" which only appears in one project.

Usage:
    python scripts/analyse_tokens.py \
        --config config/run_config.yaml \
        --input  context/wkh_apps1_wkh_apps1Techserv_Development.xlsx \
        --min-count 5 \
        --unknown-only
"""

import csv
import logging
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

# Ensure src/ is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import click
import yaml

from ingestion.reader import read_file
from ingestion.schema import map_columns, validate
from ingestion.filters import apply_filters
from ingestion.normaliser import normalise
from ingestion.path_parser import decompose_paths
from classification.config_loader import load_config
from classification.tokeniser import tokenise, ngrams

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("token_analysis")


# ── Stopwords ────────────────────────────────────────────────
# Common English words, path noise, and numeric-only tokens
# that would overwhelm the frequency table without adding
# any classification value.
# ─────────────────────────────────────────────────────────────
STOPWORDS: set[str] = {
    # Articles / prepositions / conjunctions
    "a", "an", "the", "of", "in", "on", "at", "to", "for", "by",
    "with", "from", "up", "out", "is", "it", "as", "or", "if",
    "be", "no", "not", "but", "so", "do", "has", "had", "was",
    "are", "am", "its", "we", "he", "she", "us", "my", "me",
    "all", "any", "our", "you", "your", "this", "that", "each",
    "than", "and", "&", "-", "re", "fw", "fwd", "per", "via",
    "also", "will", "can", "would", "could", "should", "been",
    "being", "have", "were",
    # Common file / folder noise
    "new", "old", "copy", "final", "draft", "v1", "v2", "v3", "v4",
    "v5", "rev", "revised", "updated", "original", "various", "misc",
    "other", "general", "documents", "document", "doc", "docs",
    "files", "file", "folder", "data", "info", "information",
    "scan", "scans", "scanned", "pdf", "pdfs", "xlsx", "csv",
    "archive", "archived", "backup", "etc", "item", "items",
    "version", "type", "note", "notes", "details", "list",
    "number", "no", "ref", "reference", "see", "attached",
    "dated", "date", "received", "sent", "requested",
    "latest", "current", "previous", "existing", "proposed",
    "working", "superseded", "superceded", "obsolete",
    # Date fragments that survive tokenisation
    "jan", "feb", "mar", "apr", "may", "jun", "jul", "aug",
    "sep", "oct", "nov", "dec",
    "january", "february", "march", "april", "june", "july",
    "august", "september", "october", "november", "december",
    "monday", "tuesday", "wednesday", "thursday", "friday",
    "saturday", "sunday", "sept",
    # Ordinals and floor levels
    "first", "second", "third", "fourth", "fifth",
    "ground", "upper", "lower", "top", "bottom",
    # Common adjective/adverbs in path names
    "additional", "alternative", "main", "full", "part",
    "required", "included", "used", "based", "completed",
    "signed", "approved", "issued", "amended",
    # Housing noise that isn't a document type
    "external", "internal", "communal", "rear", "front",
    "left", "right", "north", "south", "east", "west",
}


def _is_noise(token: str) -> bool:
    """Return True if token is a stopword, pure number, or single char."""
    if len(token) <= 1:
        return True
    if token in STOPWORDS:
        return True
    # Pure numeric (dates, plot numbers, reference numbers)
    if token.isdigit():
        return True
    # Version-style tokens: v2, v10, v2a
    if token[0] == "v" and (token[1:].isdigit() or
       (len(token) <= 4 and token[1:-1].isdigit() and token[-1].isalpha())):
        return True
    # Revision-style: r1, r2, rev3
    if token[0] == "r" and token[1:].isdigit() and len(token) <= 3:
        return True
    # Alphanumeric codes that are mostly digits: 20230428, 150-23-00046
    # These are reference numbers / dates, not document types
    digit_count = sum(1 for c in token if c.isdigit())
    if len(token) >= 4 and digit_count / len(token) > 0.6:
        return True
    return False


def _build_known_terms(config) -> set[str]:
    """Extract every term the classification config already handles.

    Returns a set of lowercase strings — unigrams AND multi-word phrases.
    If a token appears here, we already match it.
    """
    known: set[str] = set()

    # Type tokens and abbreviations
    for td in config.types.values():
        for t in td.tokens:
            known.add(t)
            # Also add individual words from multi-word tokens
            # so "fire risk assessment" marks "fire", "risk", "assessment"
            for word in t.split():
                known.add(word)
        for a in td.abbreviations:
            known.add(a)

    # Category signals and aliases
    for cd in config.categories.values():
        for s in cd.signals:
            known.add(s)
            for word in s.split():
                known.add(word)
        for a in cd.aliases:
            known.add(a)

    # Entity tokens and abbreviations
    for ed in config.entities.values():
        for t in ed.tokens:
            known.add(t)
        for a in ed.abbreviations:
            known.add(a)

    return known


def _run_ingestion(config_path: Path, input_path: Path):
    """Run ingestion pipeline up to path decomposition (reuses main.py logic)."""
    with open(config_path) as f:
        run_config = yaml.safe_load(f)

    schema_path = config_path.parent / "schema.yaml"
    with open(schema_path) as f:
        schema_config = yaml.safe_load(f)

    format_name = run_config["source"]["format"]
    format_config = schema_config["formats"][format_name]

    logger.info("Reading input file...")
    df = read_file(input_path, format_name, format_config)

    logger.info("Mapping columns...")
    df = map_columns(df, format_config["column_map"])
    validate(df, schema_config["internal_schema"]["required_columns"])

    logger.info("Filtering...")
    exclusion_config = schema_config.get("exclusion_patterns", {})
    df, summary = apply_filters(df, exclusion_config)
    logger.info(
        f"  {summary.initial_count:,} -> {summary.final_count:,} "
        f"({summary.total_removed:,} removed)"
    )

    logger.info("Normalising...")
    df = normalise(df)

    logger.info("Decomposing paths...")
    root_prefix = run_config["source"]["root_prefix"]
    df = decompose_paths(df, root_prefix)

    return df, run_config


def _count_tokens(df, max_ngram: int = 3):
    """Tokenise every folder segment and filename stem, count frequencies.

    Also tracks which top-level project folders each token appears in
    (project diversity).

    Returns:
        folder_counts:   Counter for tokens from folder segments
        filename_counts: Counter for tokens from filename stems
        folder_ngrams:   dict[int, Counter] for bigrams/trigrams from folders
        filename_ngrams: dict[int, Counter] for bigrams/trigrams from filenames
        project_sets:    dict[token, set[str]] — which projects each token appears in
        sample_paths:    dict[token, list[str]] — up to 3 example paths per token
    """
    folder_counts: Counter = Counter()
    filename_counts: Counter = Counter()
    folder_ngram_counts: dict[int, Counter] = {
        n: Counter() for n in range(2, max_ngram + 1)
    }
    filename_ngram_counts: dict[int, Counter] = {
        n: Counter() for n in range(2, max_ngram + 1)
    }
    project_sets: dict[str, set[str]] = defaultdict(set)
    sample_paths: dict[str, list[str]] = {}

    total = len(df)
    log_interval = max(total // 10, 1)

    for idx, row in enumerate(df.itertuples(index=False)):
        if idx % log_interval == 0:
            logger.info(f"  Tokenising row {idx:,}/{total:,}...")

        rel_path = getattr(row, "relative_path", "")
        segments = getattr(row, "segments", [])
        stem = getattr(row, "filename_stem", "")

        # Project = first segment (top-level folder)
        project = segments[0] if segments and isinstance(segments, list) and len(segments) > 0 else "_root_"

        # ── Folder segments ──
        if segments and isinstance(segments, list):
            for seg in segments:
                tokens = tokenise(seg)
                clean = [t for t in tokens if not _is_noise(t)]
                for t in clean:
                    folder_counts[t] += 1
                    project_sets[t].add(project)
                    _add_sample(sample_paths, t, rel_path)

                for n in range(2, max_ngram + 1):
                    for ng in ngrams(clean, n):
                        folder_ngram_counts[n][ng] += 1
                        project_sets[ng].add(project)
                        _add_sample(sample_paths, ng, rel_path)

        # ── Filename stem ──
        if stem and isinstance(stem, str):
            tokens = tokenise(stem)
            clean = [t for t in tokens if not _is_noise(t)]
            for t in clean:
                filename_counts[t] += 1
                project_sets[t].add(project)
                _add_sample(sample_paths, t, rel_path)

            for n in range(2, max_ngram + 1):
                for ng in ngrams(clean, n):
                    filename_ngram_counts[n][ng] += 1
                    project_sets[ng].add(project)
                    _add_sample(sample_paths, ng, rel_path)

    return (
        folder_counts,
        filename_counts,
        folder_ngram_counts,
        filename_ngram_counts,
        project_sets,
        sample_paths,
    )


def _add_sample(samples: dict, token: str, path: str, max_samples: int = 3):
    """Keep up to max_samples example paths per token."""
    if token not in samples:
        samples[token] = []
    if len(samples[token]) < max_samples:
        samples[token].append(path)


def _write_csv(
    output_path: Path,
    folder_counts: Counter,
    filename_counts: Counter,
    folder_ngram_counts: dict[int, Counter],
    filename_ngram_counts: dict[int, Counter],
    project_sets: dict[str, set[str]],
    sample_paths: dict[str, list[str]],
    known_terms: set[str],
    min_count: int,
    min_projects: int,
    unknown_only: bool,
):
    """Write the analysis results to a CSV file."""
    rows = []

    def _add_row(token: str, fc: int, fnc: int, ngram_size: int):
        total = fc + fnc
        if total < min_count:
            return
        n_projects = len(project_sets.get(token, set()))
        if n_projects < min_projects:
            return

        is_known = token in known_terms
        if unknown_only and is_known:
            return

        source = (
            "both" if fc > 0 and fnc > 0
            else "folder" if fc > 0
            else "filename"
        )
        samples = " | ".join(sample_paths.get(token, []))
        rows.append({
            "ngram": ngram_size,
            "token": token,
            "total_count": total,
            "folder_count": fc,
            "filename_count": fnc,
            "projects": n_projects,
            "source": source,
            "known": "YES" if is_known else "",
            "sample_paths": samples,
        })

    # ── Unigrams ──
    all_tokens = set(folder_counts.keys()) | set(filename_counts.keys())
    for token in all_tokens:
        _add_row(token, folder_counts.get(token, 0),
                 filename_counts.get(token, 0), 1)

    # ── N-grams ──
    for n in sorted(folder_ngram_counts.keys()):
        fc_counter = folder_ngram_counts[n]
        fnc_counter = filename_ngram_counts[n]
        all_ng = set(fc_counter.keys()) | set(fnc_counter.keys())
        for ng in all_ng:
            _add_row(ng, fc_counter.get(ng, 0), fnc_counter.get(ng, 0), n)

    # Sort: unknown first, then by project diversity desc, then count desc
    rows.sort(key=lambda r: (r["known"] == "YES", -r["projects"], -r["total_count"]))

    fieldnames = [
        "ngram", "token", "total_count", "folder_count",
        "filename_count", "projects", "source", "known", "sample_paths",
    ]
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    return len(rows)


@click.command()
@click.option(
    "--config", "config_path", required=True,
    type=click.Path(exists=True),
    help="Path to run config YAML",
)
@click.option(
    "--input", "input_path", required=True,
    type=click.Path(exists=True),
    help="Path to input file (XLSX or CSV)",
)
@click.option(
    "--output", "output_path", default=None,
    type=click.Path(),
    help="Output CSV path (default: reports/<input_stem>_token_analysis.csv)",
)
@click.option(
    "--min-count", default=5, show_default=True,
    help="Minimum occurrence count to include a token",
)
@click.option(
    "--min-projects", default=1, show_default=True,
    help="Minimum number of distinct projects a token must appear in",
)
@click.option(
    "--unknown-only", is_flag=True, default=False,
    help="Only show tokens NOT already in the classification config",
)
@click.option(
    "--max-ngram", default=3, show_default=True,
    help="Maximum n-gram size (2=bigrams, 3=trigrams, etc.)",
)
def main(
    config_path: str,
    input_path: str,
    output_path: str | None,
    min_count: int,
    min_projects: int,
    unknown_only: bool,
    max_ngram: int,
):
    """Analyse token frequencies to discover missing dictionary terms."""
    start = time.time()
    config_path = Path(config_path)
    input_path = Path(input_path)

    logger.info("Token Frequency Analysis")
    logger.info(f"Input: {input_path.name}")

    # ── Step 1: Ingest ──
    df, run_config = _run_ingestion(config_path, input_path)
    logger.info(f"  {len(df):,} rows after ingestion")

    # ── Step 2: Load classification config to build known-terms set ──
    logger.info("Loading classification config...")
    industry = run_config["industry"]
    cls_config = load_config(config_path.parent, industry)
    known_terms = _build_known_terms(cls_config)
    logger.info(f"  {len(known_terms):,} known terms from config")

    # ── Step 3: Count tokens ──
    logger.info("Counting tokens...")
    (
        folder_counts,
        filename_counts,
        folder_ngram_counts,
        filename_ngram_counts,
        project_sets,
        sample_paths,
    ) = _count_tokens(df, max_ngram=max_ngram)

    # Count distinct projects
    all_projects = set()
    for ps in project_sets.values():
        all_projects.update(ps)

    logger.info(
        f"  Unigrams: {len(folder_counts):,} folder, "
        f"{len(filename_counts):,} filename"
    )
    for n in sorted(folder_ngram_counts.keys()):
        fc = len(folder_ngram_counts[n])
        fnc = len(filename_ngram_counts[n])
        logger.info(f"  {n}-grams: {fc:,} folder, {fnc:,} filename")
    logger.info(f"  Distinct projects (top-level folders): {len(all_projects):,}")

    # ── Step 4: Write output ──
    if output_path is None:
        out_dir = Path("reports")
        out_dir.mkdir(parents=True, exist_ok=True)
        output_path = out_dir / f"{input_path.stem}_token_analysis.csv"
    else:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

    row_count = _write_csv(
        output_path,
        folder_counts,
        filename_counts,
        folder_ngram_counts,
        filename_ngram_counts,
        project_sets,
        sample_paths,
        known_terms,
        min_count,
        min_projects,
        unknown_only,
    )

    elapsed = time.time() - start

    # ── Summary ──
    total_unique = len(set(folder_counts.keys()) | set(filename_counts.keys()))

    click.echo(f"\n{'=' * 60}")
    click.echo("Token Analysis Summary")
    click.echo(f"{'=' * 60}")
    click.echo(f"  Input rows:       {len(df):,}")
    click.echo(f"  Unique unigrams:  {total_unique:,}")
    click.echo(f"  Known terms:      {len(known_terms):,}")
    click.echo(f"  Projects found:   {len(all_projects):,}")
    click.echo(f"  Min count:        {min_count}")
    click.echo(f"  Min projects:     {min_projects}")
    click.echo(f"  Unknown only:     {unknown_only}")
    click.echo(f"  Rows written:     {row_count:,}")
    click.echo(f"  Output:           {output_path}")
    click.echo(f"  Time:             {elapsed:.1f}s")

    # ── Top unknown unigrams by project diversity ──
    combined = Counter()
    combined.update(folder_counts)
    combined.update(filename_counts)
    unknown_tokens = [
        (t, combined[t], len(project_sets.get(t, set())))
        for t in combined
        if t not in known_terms and combined[t] >= min_count
           and len(project_sets.get(t, set())) >= min_projects
    ]
    if unknown_tokens:
        # Sort by project count desc, then total count desc
        top_unknown = sorted(unknown_tokens, key=lambda x: (-x[2], -x[1]))[:25]
        click.echo(f"\n  Top 25 unknown unigrams by project diversity:")
        click.echo(f"  {'token':<30} {'count':>8} {'projects':>8}")
        click.echo(f"  {'-'*30} {'-'*8} {'-'*8}")
        for token, count, n_proj in top_unknown:
            click.echo(f"  {token:<30} {count:>8,} {n_proj:>8}")


if __name__ == "__main__":
    main()
