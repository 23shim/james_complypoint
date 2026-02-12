"""Tests for iterparse-based XLSX reading (sheet discovery + formula extraction)."""

import io
import zipfile
from pathlib import Path

import pytest

from ingestion.reader import (
    _extract_formula_paths,
    _find_sheet,
    _resolve_sheet_xml_path,
)

# SpreadsheetML namespace
_NS = "http://schemas.openxmlformats.org/spreadsheetml/2006/main"
_RELS_NS = "http://schemas.openxmlformats.org/package/2006/relationships"
_OFFDOC_NS = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"


def _make_xlsx(sheets: dict[str, str], tmp_path: Path) -> Path:
    """Build a minimal XLSX (ZIP) with given sheet names and XML content.

    Args:
        sheets: mapping of sheet name -> worksheet XML body (the <sheetData> content).
        tmp_path: pytest fixture for temp directory.

    Returns:
        Path to the created .xlsx file.
    """
    xlsx_path = tmp_path / "test.xlsx"

    workbook_sheets = ""
    rels = ""
    files: dict[str, str] = {}

    for idx, name in enumerate(sheets, start=1):
        rid = f"rId{idx}"
        target = f"worksheets/sheet{idx}.xml"
        workbook_sheets += (
            f'<sheet name="{name}" sheetId="{idx}" '
            f'r:id="{rid}" '
            f'xmlns:r="{_OFFDOC_NS}"/>'
        )
        rels += (
            f'<Relationship Id="{rid}" '
            f'Type="{_OFFDOC_NS}/worksheet" '
            f'Target="{target}"/>'
        )
        files[f"xl/{target}"] = (
            f'<?xml version="1.0" encoding="UTF-8"?>'
            f'<worksheet xmlns="{_NS}">'
            f"<sheetData>{sheets[name]}</sheetData>"
            f"</worksheet>"
        )

    files["xl/workbook.xml"] = (
        f'<?xml version="1.0" encoding="UTF-8"?>'
        f'<workbook xmlns="{_NS}">'
        f"<sheets>{workbook_sheets}</sheets>"
        f"</workbook>"
    )
    files["xl/_rels/workbook.xml.rels"] = (
        f'<?xml version="1.0" encoding="UTF-8"?>'
        f'<Relationships xmlns="{_RELS_NS}">'
        f"{rels}"
        f"</Relationships>"
    )

    with zipfile.ZipFile(xlsx_path, "w") as zf:
        for arcname, content in files.items():
            zf.writestr(arcname, content)

    return xlsx_path


# ---------------------------------------------------------------
# _find_sheet tests
# ---------------------------------------------------------------

class TestFindSheet:

    def test_selects_largest_sheet(self, tmp_path):
        sheets = {
            "Summary": '<row r="1"><c r="A1"><v>1</v></c></row>',
            "Data": (
                '<row r="1"><c r="A1"><v>x</v></c></row>'
                '<row r="2"><c r="A2"><v>y</v></c></row>'
                '<row r="100"><c r="A100"><v>z</v></c></row>'
            ),
        }
        xlsx = _make_xlsx(sheets, tmp_path)
        result = _find_sheet(xlsx, "largest")
        assert result == "Data"

    def test_selects_first_sheet_when_not_largest(self, tmp_path):
        sheets = {"Only": '<row r="1"><c r="A1"><v>1</v></c></row>'}
        xlsx = _make_xlsx(sheets, tmp_path)
        result = _find_sheet(xlsx, "first")
        assert result == "Only"

    def test_single_sheet(self, tmp_path):
        sheets = {"Sheet1": '<row r="5"><c r="A5"><v>5</v></c></row>'}
        xlsx = _make_xlsx(sheets, tmp_path)
        result = _find_sheet(xlsx, "largest")
        assert result == "Sheet1"


# ---------------------------------------------------------------
# _resolve_sheet_xml_path tests
# ---------------------------------------------------------------

class TestResolveSheetXmlPath:

    def test_resolves_correct_path(self, tmp_path):
        sheets = {
            "Sheet1": '<row r="1"><c r="A1"><v>1</v></c></row>',
            "Sheet2": '<row r="1"><c r="A1"><v>2</v></c></row>',
        }
        xlsx = _make_xlsx(sheets, tmp_path)
        with zipfile.ZipFile(xlsx, "r") as zf:
            path = _resolve_sheet_xml_path(zf, "Sheet2")
            assert path == "xl/worksheets/sheet2.xml"

    def test_raises_on_unknown_sheet(self, tmp_path):
        sheets = {"Sheet1": '<row r="1"><c r="A1"><v>1</v></c></row>'}
        xlsx = _make_xlsx(sheets, tmp_path)
        with zipfile.ZipFile(xlsx, "r") as zf:
            with pytest.raises(ValueError, match="not found in workbook.xml"):
                _resolve_sheet_xml_path(zf, "NonExistent")


# ---------------------------------------------------------------
# _extract_formula_paths tests
# ---------------------------------------------------------------

class TestExtractFormulaPaths:

    def test_extracts_hyperlink_formulas(self, tmp_path):
        """Basic formula extraction from column A."""
        sheet_xml = (
            # header_row=0 means header is row 1, data starts row 2
            '<row r="1"><c r="A1" t="s"><v>0</v></c></row>'
            '<row r="2"><c r="A2" t="str"><v>plain_path</v></c></row>'
            '<row r="3"><c r="A3" t="str">'
            '<f>HYPERLINK("\\\\server\\share\\file.pdf","\\\\server\\share\\file.pdf")</f>'
            '<v>0</v></c></row>'
        )
        sheets = {"Data": sheet_xml}
        xlsx = _make_xlsx(sheets, tmp_path)

        repairs = _extract_formula_paths(xlsx, "Data", header_row=0)
        # Row 3 in Excel = pandas index 1 (data_start = 0 + 2 = 2, so 3 - 2 = 1)
        assert repairs == {1: "\\\\server\\share\\file.pdf"}

    def test_ignores_non_path_columns(self, tmp_path):
        """Formulas in columns other than A should be ignored."""
        sheet_xml = (
            '<row r="1"><c r="A1"><v>header</v></c><c r="B1"><v>other</v></c></row>'
            '<row r="2"><c r="A2"><v>plain</v></c>'
            '<c r="B2" t="str"><f>HYPERLINK("should_ignore","x")</f><v>0</v></c></row>'
        )
        sheets = {"Data": sheet_xml}
        xlsx = _make_xlsx(sheets, tmp_path)

        repairs = _extract_formula_paths(xlsx, "Data", header_row=0)
        assert repairs == {}

    def test_handles_multiple_formula_rows(self, tmp_path):
        """Multiple consecutive formula rows should all be extracted."""
        rows = '<row r="1"><c r="A1"><v>hdr</v></c></row>'
        for i in range(2, 12):
            rows += (
                f'<row r="{i}"><c r="A{i}" t="str">'
                f'<f>HYPERLINK("\\\\srv\\path{i}","\\\\srv\\path{i}")</f>'
                f'<v>0</v></c></row>'
            )
        sheets = {"Data": rows}
        xlsx = _make_xlsx(sheets, tmp_path)

        repairs = _extract_formula_paths(xlsx, "Data", header_row=0)
        assert len(repairs) == 10
        assert repairs[0] == "\\\\srv\\path2"
        assert repairs[9] == "\\\\srv\\path11"

    def test_header_row_offset(self, tmp_path):
        """TreeSize has 4 metadata rows (header_row=4), so data starts at Excel row 6."""
        rows = ""
        # 4 metadata rows + 1 header row = rows 1-5
        for i in range(1, 6):
            rows += f'<row r="{i}"><c r="A{i}"><v>meta{i}</v></c></row>'
        # Data row at Excel row 6 = pandas index 0
        rows += (
            '<row r="6"><c r="A6" t="str">'
            '<f>HYPERLINK("\\\\server\\data\\report.pdf","display")</f>'
            '<v>0</v></c></row>'
        )
        sheets = {"Scan": rows}
        xlsx = _make_xlsx(sheets, tmp_path)

        repairs = _extract_formula_paths(xlsx, "Scan", header_row=4)
        assert repairs == {0: "\\\\server\\data\\report.pdf"}

    def test_no_formulas_returns_empty(self, tmp_path):
        """Plain string cells should produce no repairs."""
        sheet_xml = (
            '<row r="1"><c r="A1"><v>hdr</v></c></row>'
            '<row r="2"><c r="A2" t="s"><v>0</v></c></row>'
            '<row r="3"><c r="A3" t="s"><v>1</v></c></row>'
        )
        sheets = {"Data": sheet_xml}
        xlsx = _make_xlsx(sheets, tmp_path)

        repairs = _extract_formula_paths(xlsx, "Data", header_row=0)
        assert repairs == {}

    def test_mixed_plain_and_formula_rows(self, tmp_path):
        """Only formula rows should appear in repairs dict."""
        sheet_xml = (
            '<row r="1"><c r="A1"><v>hdr</v></c></row>'
            '<row r="2"><c r="A2" t="str"><v>\\\\server\\plain.txt</v></c></row>'
            '<row r="3"><c r="A3" t="str">'
            '<f>HYPERLINK("\\\\server\\formula.pdf","x")</f><v>0</v></c></row>'
            '<row r="4"><c r="A4" t="str"><v>\\\\server\\also_plain.doc</v></c></row>'
        )
        sheets = {"Data": sheet_xml}
        xlsx = _make_xlsx(sheets, tmp_path)

        repairs = _extract_formula_paths(xlsx, "Data", header_row=0)
        # Only row 3 (pandas idx 1) should be repaired
        assert repairs == {1: "\\\\server\\formula.pdf"}

    def test_special_characters_in_path(self, tmp_path):
        """Paths with ampersands, spaces, brackets should survive."""
        path = "\\\\server\\Fire safety &amp; risk\\Plot (1)\\cert.pdf"
        expected = "\\\\server\\Fire safety & risk\\Plot (1)\\cert.pdf"
        sheet_xml = (
            '<row r="1"><c r="A1"><v>hdr</v></c></row>'
            f'<row r="2"><c r="A2" t="str">'
            f'<f>HYPERLINK("{path}","{path}")</f><v>0</v></c></row>'
        )
        sheets = {"Data": sheet_xml}
        xlsx = _make_xlsx(sheets, tmp_path)

        repairs = _extract_formula_paths(xlsx, "Data", header_row=0)
        assert repairs == {0: expected}
