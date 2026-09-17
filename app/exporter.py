"""
Export Engine — exporter.py

Converts extracted table data (list of table dicts) into downloadable files:
  - Excel (.xlsx) with multi-sheet support and auto-fit column widths
  - CSV (.csv) single file or ZIP archive for multiple tables
"""

import io
import os
import tempfile
import zipfile

import pandas as pd
from openpyxl.utils import get_column_letter


# ---------------------------------------------------------------------------
# Excel Export
# ---------------------------------------------------------------------------

def export_to_xlsx(tables: list[dict], filename_base: str) -> io.BytesIO:
    """
    Export tables to an Excel (.xlsx) file in memory.

    Each table becomes a separate sheet. Column widths are auto-fitted.

    Args:
        tables: List of table dicts from converter (with headers/rows).
        filename_base: Base filename (without extension) for sheet naming.

    Returns:
        BytesIO buffer containing the .xlsx file.
    """
    output = io.BytesIO()

    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        for i, table in enumerate(tables):
            headers = table.get("headers", [])
            rows = table.get("rows", [])

            df = pd.DataFrame(rows, columns=headers)

            # Generate sheet name (max 31 chars for Excel)
            sheet_name = table.get("name", f"Sheet{i + 1}")
            if len(sheet_name) > 31:
                sheet_name = sheet_name[:28] + "..."

            df.to_excel(writer, sheet_name=sheet_name, index=False)

            # Auto-fit column widths
            worksheet = writer.sheets[sheet_name]
            for col_idx, col_name in enumerate(headers, start=1):
                # Calculate max width from header and data
                max_length = len(str(col_name))
                for row in rows:
                    if col_idx - 1 < len(row):
                        cell_len = len(str(row[col_idx - 1]))
                        if cell_len > max_length:
                            max_length = cell_len

                # Add padding and cap at reasonable width
                adjusted_width = min(max_length + 3, 60)
                col_letter = get_column_letter(col_idx)
                worksheet.column_dimensions[col_letter].width = adjusted_width

    output.seek(0)
    return output


# ---------------------------------------------------------------------------
# CSV Export
# ---------------------------------------------------------------------------

def export_to_csv_single(table: dict) -> io.BytesIO:
    """
    Export a single table to CSV in memory.

    Args:
        table: Single table dict from converter.

    Returns:
        BytesIO buffer containing the .csv file.
    """
    headers = table.get("headers", [])
    rows = table.get("rows", [])

    df = pd.DataFrame(rows, columns=headers)

    output = io.BytesIO()
    df.to_csv(output, index=False, encoding="utf-8-sig")
    output.seek(0)
    return output


def export_to_csv_zip(tables: list[dict], filename_base: str) -> io.BytesIO:
    """
    Export multiple tables to a ZIP archive containing individual CSV files.

    Args:
        tables: List of table dicts from converter.
        filename_base: Base filename for the CSV files.

    Returns:
        BytesIO buffer containing the .zip file.
    """
    output = io.BytesIO()

    with zipfile.ZipFile(output, "w", zipfile.ZIP_DEFLATED) as zf:
        for i, table in enumerate(tables):
            headers = table.get("headers", [])
            rows = table.get("rows", [])

            df = pd.DataFrame(rows, columns=headers)

            csv_buffer = io.StringIO()
            df.to_csv(csv_buffer, index=False, encoding="utf-8")

            # Name each CSV file
            csv_filename = f"{filename_base}_tabel_{i + 1}.csv"
            zf.writestr(csv_filename, csv_buffer.getvalue())

    output.seek(0)
    return output


# ---------------------------------------------------------------------------
# TSV for Clipboard
# ---------------------------------------------------------------------------

def table_to_tsv(table: dict) -> str:
    """
    Convert a single table to TSV string for clipboard copy.

    Args:
        table: Single table dict from converter.

    Returns:
        TSV-formatted string (tab-separated, newline-delimited).
    """
    headers = table.get("headers", [])
    rows = table.get("rows", [])

    lines = ["\t".join(str(h) for h in headers)]
    for row in rows:
        lines.append("\t".join(str(cell) for cell in row))

    return "\n".join(lines)
