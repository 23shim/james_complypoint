"""
Format-specific file readers.

Each reader's only job: get raw data into a pandas DataFrame with
the original column names. No normalisation, no filtering, no path logic.

Adding a new format = writing one function + one registry entry.
"""

import logging
import re
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

# Matches HYPERLINK("path", "display") or HYPERLINK("path") in formula text
_HYPERLINK_RE = re.compile(r'HYPERLINK\("([^"]+)"')

# SpreadsheetML namespace
_SS_NS = "http://schemas.openxmlformats.org/spreadsheetml/2006/main"
_RELS_NS = "http://schemas.openxmlformats.org/package/2006/relationships"
_RELS_OFFDOC_NS = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"

# Pre-computed qualified tag names for fast comparison
_TAG_C = f"{{{_SS_NS}}}c"
_TAG_F = f"{{{_SS_NS}}}f"
_TAG_ROW = f"{{{_SS_NS}}}row"


def read_treesize_xlsx(file_path: Path, format_config: dict) -> pd.DataFrame:
    """Read a TreeSize Pro XLSX export into a raw DataFrame.

    TreeSize exports have:
    - Multiple sheets (Extensions, Details, and the main scan sheet)
    - 4 metadata rows above the actual column headers
    - The main sheet is named with a timestamp (changes every export)
    - Beyond row ~65536, paths switch from plain strings to
      =HYPERLINK() formulas (calamine reads these as 0)
    """
    header_row = format_config.get("header_row", 4)
    sheet_strategy = format_config.get("sheet_strategy", "largest")
    sheet_name = format_config.get("sheet_name", None)

    if sheet_name is None:
        sheet_name = _find_sheet(file_path, sheet_strategy)

    df = _read_xlsx(file_path, sheet_name, header_row)

    # Repair HYPERLINK formula paths that calamine reads as 0.
    # column_map keys are source column names; the DataFrame hasn't
    # been renamed yet, so we look for the source name "Path".
    column_map = format_config.get("column_map", {})
    path_col = "Path" if "Path" in df.columns else None
    if path_col is None:
        # Try to find the path column from config keys
        for source_name in column_map:
            if source_name in df.columns and column_map[source_name] == "full_path":
                path_col = source_name
                break

    if path_col and path_col in df.columns:
        df = _repair_hyperlink_paths(df, file_path, sheet_name, header_row, path_col)

    return df


def read_csv(file_path: Path, format_config: dict) -> pd.DataFrame:
    """Read a CSV file into a raw DataFrame."""
    header_row = format_config.get("header_row", 0)
    encoding = format_config.get("encoding", "utf-8")

    df = pd.read_csv(
        file_path,
        header=header_row,
        encoding=encoding,
        low_memory=False,
    )
    logger.info(f"Read {len(df):,} rows from CSV")
    return df


def _repair_hyperlink_paths(
    df: pd.DataFrame,
    file_path: Path,
    sheet_name: str,
    header_row: int,
    path_col: str,
) -> pd.DataFrame:
    """Fix path values that calamine reads as 0 due to HYPERLINK formulas.

    TreeSize stores the first ~65K paths as plain strings. Beyond that,
    it switches to =HYPERLINK("path","path") formulas. Calamine reads
    the cached result value (0) instead of the formula text.

    Extracts real paths by streaming the worksheet XML directly from the
    XLSX ZIP archive using stdlib iterparse. Only processes the path
    column, skipping all other cells. Uses constant memory regardless
    of file size.
    """
    is_string = df[path_col].apply(lambda x: isinstance(x, str))
    broken_count = int((~is_string).sum())

    if broken_count == 0:
        return df

    logger.warning(
        f"Found {broken_count:,} paths stored as HYPERLINK formulas. "
        f"Extracting paths via direct XML parsing..."
    )

    repairs = _extract_formula_paths(
        file_path, sheet_name, header_row, path_col_letter="A",
    )

    # Apply repairs
    for idx, path in repairs.items():
        if idx in df.index:
            df.at[idx, path_col] = path

    still_broken = int(df[path_col].apply(lambda x: not isinstance(x, str)).sum())
    logger.info(
        f"Repaired {len(repairs):,} HYPERLINK paths via XML extraction. "
        f"Remaining non-string paths: {still_broken:,}"
    )

    return df


def _resolve_sheet_xml_path(zf: zipfile.ZipFile, sheet_name: str) -> str:
    """Map a sheet name to its xl/worksheets/sheetN.xml path inside the ZIP."""
    with zf.open("xl/workbook.xml") as f:
        tree = ET.parse(f)

    r_ns = _RELS_OFFDOC_NS
    target_rid = None
    for sheet_el in tree.getroot().iter(f"{{{_SS_NS}}}sheet"):
        if sheet_el.get("name") == sheet_name:
            target_rid = sheet_el.get(f"{{{r_ns}}}id")
            break

    if target_rid is None:
        raise ValueError(f"Sheet '{sheet_name}' not found in workbook.xml")

    with zf.open("xl/_rels/workbook.xml.rels") as f:
        tree = ET.parse(f)
    for rel in tree.getroot():
        if rel.get("Id") == target_rid:
            target = rel.get("Target")
            return f"xl/{target}"

    raise ValueError(f"Relationship {target_rid} not found in workbook.xml.rels")


def _extract_formula_paths(
    file_path: Path,
    sheet_name: str,
    header_row: int,
    path_col_letter: str = "A",
) -> dict[int, str]:
    """Extract HYPERLINK formula paths from a single column of an XLSX worksheet.

    Streams the worksheet XML via iterparse — constant memory (~5-15 MB)
    regardless of file size. Only examines cells in the target column.

    Returns:
        dict mapping pandas DataFrame index -> extracted path string.
    """
    data_start_excel_row = header_row + 2  # +1 for 1-indexing, +1 for header

    repairs: dict[int, str] = {}

    with zipfile.ZipFile(file_path, "r") as zf:
        sheet_path = _resolve_sheet_xml_path(zf, sheet_name)

        with zf.open(sheet_path) as sheet_file:
            for _, elem in ET.iterparse(sheet_file, events=("end",)):
                if elem.tag != _TAG_C:
                    # Free completed rows to keep memory flat
                    if elem.tag == _TAG_ROW:
                        elem.clear()
                    continue

                ref = elem.get("r", "")
                col_letters = ref.rstrip("0123456789")

                if col_letters == path_col_letter:
                    f_elem = elem.find(_TAG_F)
                    if f_elem is not None and f_elem.text:
                        row_num = int(ref[len(col_letters):])
                        pandas_idx = row_num - data_start_excel_row
                        if pandas_idx >= 0:
                            match = _HYPERLINK_RE.search(f_elem.text)
                            if match:
                                repairs[pandas_idx] = match.group(1)

                elem.clear()

    return repairs


def _find_sheet(file_path: Path, strategy: str) -> str:
    """Discover the right sheet in an XLSX workbook.

    Reads sheet names from xl/workbook.xml and estimates row counts
    by scanning each worksheet's XML for the last <row> element.
    No openpyxl dependency — pure stdlib ZIP + XML.
    """
    with zipfile.ZipFile(file_path, "r") as zf:
        # Parse workbook.xml for sheet names and relationship IDs
        with zf.open("xl/workbook.xml") as f:
            wb_tree = ET.parse(f)
        r_ns = _RELS_OFFDOC_NS
        sheets = []
        for el in wb_tree.getroot().iter(f"{{{_SS_NS}}}sheet"):
            sheets.append((el.get("name"), el.get(f"{{{r_ns}}}id")))

        if not sheets:
            raise ValueError("No sheets found in workbook")

        sheet_names = [s[0] for s in sheets]

        if strategy != "largest":
            logger.info(f"Using first sheet: '{sheet_names[0]}'")
            return sheet_names[0]

        # Resolve rIds to file paths
        with zf.open("xl/_rels/workbook.xml.rels") as f:
            rels_tree = ET.parse(f)
        rid_to_target = {}
        for rel in rels_tree.getroot():
            rid_to_target[rel.get("Id")] = rel.get("Target")

        # Find the sheet with the most rows
        max_rows = 0
        selected = sheet_names[0]
        for name, rid in sheets:
            target = rid_to_target.get(rid)
            if not target:
                continue
            xml_path = f"xl/{target}"
            if xml_path not in zf.namelist():
                continue
            # Stream through XML, tracking the last row number seen
            last_row = 0
            with zf.open(xml_path) as f:
                for _, elem in ET.iterparse(f, events=("end",)):
                    if elem.tag == _TAG_ROW:
                        r = elem.get("r")
                        if r:
                            last_row = max(last_row, int(r))
                        elem.clear()
            if last_row > max_rows:
                max_rows = last_row
                selected = name

        logger.info(
            f"Auto-selected sheet '{selected}' "
            f"({max_rows:,} rows) from {len(sheet_names)} sheets"
        )
        return selected


def _read_xlsx(file_path: Path, sheet_name: str, header_row: int) -> pd.DataFrame:
    """Read an XLSX sheet, preferring calamine engine for speed."""
    try:
        df = pd.read_excel(
            file_path,
            sheet_name=sheet_name,
            header=header_row,
            engine="calamine",
        )
        logger.info(f"Read {len(df):,} rows using calamine engine")
    except ImportError:
        logger.warning(
            "python-calamine not installed — falling back to openpyxl. "
            "For large files (1M+ rows), install python-calamine for ~10x faster reads."
        )
        df = pd.read_excel(
            file_path,
            sheet_name=sheet_name,
            header=header_row,
            engine="openpyxl",
        )
        logger.info(f"Read {len(df):,} rows using openpyxl engine")

    return df


# ---------------------------------------------------------------
# Reader registry
# Format name (from run_config.yaml) -> reader function
# ---------------------------------------------------------------
READERS: dict[str, callable] = {
    "treesize": read_treesize_xlsx,
    "csv": read_csv,
}


def read_file(file_path: str | Path, format_name: str, format_config: dict) -> pd.DataFrame:
    """Read an input file using the appropriate format-specific reader.

    Args:
        file_path: Path to the input file.
        format_name: Key into the READERS registry (e.g. "treesize", "csv").
        format_config: Format-specific settings from schema.yaml.

    Returns:
        Raw DataFrame with original column names.

    Raises:
        ValueError: If format_name is not in the registry.
        FileNotFoundError: If the file does not exist.
    """
    reader = READERS.get(format_name)
    if reader is None:
        raise ValueError(
            f"Unknown format '{format_name}'. "
            f"Available formats: {list(READERS.keys())}"
        )

    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"Input file not found: {file_path}")

    logger.info(f"Reading '{file_path.name}' with format '{format_name}'")
    return reader(file_path, format_config)
