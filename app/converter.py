"""
Table Extraction Engine — converter.py

Detects file type (digital PDF / scanned PDF / image) and extracts tables
using the appropriate engine:
  - Digital PDF  → pdfplumber (fast, no OCR needed)
  - Scanned PDF  → img2table + Tesseract OCR
  - Image files  → img2table + Tesseract OCR

Returns a list of table dicts ready for JSON preview and export.
"""

import os
import tempfile
from typing import Optional

import pandas as pd
import pdfplumber
from img2table.document import Image as Img2TableImage
from img2table.document import PDF as Img2TablePDF
from img2table.ocr import TesseractOCR
from pypdf import PdfReader

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SUPPORTED_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp"}
SUPPORTED_PDF_EXTENSION = ".pdf"

DEFAULT_OCR_LANG = "ind+eng"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _is_digital_pdf(file_path: str) -> bool:
    """
    Check if a PDF contains extractable vector text on at least one page.
    Returns True for digital (text-based) PDFs, False for scanned/image PDFs.
    """
    try:
        with pdfplumber.open(file_path) as pdf:
            for page in pdf.pages[:5]:  # Sample first 5 pages
                text = page.extract_text() or ""
                if len(text.strip()) > 20:
                    return True
        return False
    except Exception:
        return False


def _is_encrypted_pdf(file_path: str) -> bool:
    """Check if PDF is password-protected using pypdf."""
    try:
        reader = PdfReader(file_path)
        return reader.is_encrypted
    except Exception:
        return False


def _get_pdf_page_count(file_path: str) -> int:
    """Get total page count of a PDF."""
    try:
        reader = PdfReader(file_path)
        return len(reader.pages)
    except Exception:
        return 0


def _dataframe_to_table_dict(
    df: pd.DataFrame,
    table_id: int,
    page: int,
    table_index_on_page: int,
) -> dict:
    """Convert a pandas DataFrame to the API response table dict format."""
    # Clean up the dataframe
    df = df.fillna("")
    df = df.astype(str)

    # Use first row as headers if they look like headers, otherwise generate
    headers = [str(col) for col in df.columns.tolist()]

    # If columns are just integers (0, 1, 2...), use first row as header
    if all(isinstance(col, int) or str(col).isdigit() for col in df.columns):
        if len(df) > 0:
            headers = [str(val) for val in df.iloc[0].tolist()]
            df = df.iloc[1:].reset_index(drop=True)

    rows = df.values.tolist()

    return {
        "id": table_id,
        "name": f"Halaman {page} - Tabel {table_index_on_page}",
        "page": page,
        "headers": headers,
        "rows": rows,
        "row_count": len(rows),
        "col_count": len(headers),
    }


# ---------------------------------------------------------------------------
# Extraction: Digital PDF via pdfplumber
# ---------------------------------------------------------------------------

def _extract_tables_pdfplumber(file_path: str) -> list[dict]:
    """Extract tables from a digital PDF using pdfplumber."""
    tables = []
    table_id = 1

    with pdfplumber.open(file_path) as pdf:
        for page_num, page in enumerate(pdf.pages, start=1):
            page_tables = page.extract_tables()

            if not page_tables:
                continue

            for tbl_idx, raw_table in enumerate(page_tables, start=1):
                if not raw_table or len(raw_table) < 1:
                    continue

                # Convert to DataFrame
                df = pd.DataFrame(raw_table)
                table_dict = _dataframe_to_table_dict(
                    df, table_id, page_num, tbl_idx
                )
                tables.append(table_dict)
                table_id += 1

    return tables


# ---------------------------------------------------------------------------
# Extraction: Scanned PDF / Image via img2table + Tesseract
# ---------------------------------------------------------------------------

def _extract_tables_img2table(
    file_path: str,
    file_ext: str,
    ocr_lang: str = DEFAULT_OCR_LANG,
) -> list[dict]:
    """Extract tables from scanned PDF or image using img2table + Tesseract OCR."""
    tables = []
    table_id = 1

    # Initialize Tesseract OCR
    try:
        ocr = TesseractOCR(lang=ocr_lang)
    except OSError as e:
        raise RuntimeError(
            "Tesseract OCR tidak ditemukan di sistem. "
            "Untuk mengekstrak tabel dari gambar atau PDF hasil scan, silakan install Tesseract OCR "
            "(contoh: 'sudo apt install tesseract-ocr tesseract-ocr-ind') atau jalankan aplikasi via Docker."
        ) from e

    if file_ext == SUPPORTED_PDF_EXTENSION:
        # Process PDF via img2table
        doc = Img2TablePDF(src=file_path)
        extracted = doc.extract_tables(ocr=ocr, implicit_rows=True, implicit_columns=True)

        # extracted is a dict: {page_number: [Table, ...]}
        for page_num, page_tables in extracted.items():
            for tbl_idx, table in enumerate(page_tables, start=1):
                df = table.df
                if df is None or df.empty:
                    continue
                table_dict = _dataframe_to_table_dict(
                    df, table_id, page_num + 1, tbl_idx  # img2table pages are 0-indexed
                )
                tables.append(table_dict)
                table_id += 1
    else:
        # Process single image
        doc = Img2TableImage(src=file_path)
        extracted = doc.extract_tables(ocr=ocr, implicit_rows=True, implicit_columns=True)

        for tbl_idx, table in enumerate(extracted, start=1):
            df = table.df
            if df is None or df.empty:
                continue
            table_dict = _dataframe_to_table_dict(
                df, table_id, 1, tbl_idx
            )
            tables.append(table_dict)
            table_id += 1

    return tables


# ---------------------------------------------------------------------------
# Main Public API
# ---------------------------------------------------------------------------

def extract_tables(
    file_path: str,
    ocr_lang: str = DEFAULT_OCR_LANG,
) -> dict:
    """
    Extract tables from a PDF or image file.

    Args:
        file_path: Absolute path to the uploaded file on disk.
        ocr_lang: Tesseract OCR language string (default: "ind+eng").

    Returns:
        A dict matching the API response spec:
        {
            "file_type": "pdf" | "image",
            "tables_count": int,
            "tables": [ ... ]
        }

    Raises:
        ValueError: If the PDF is password-protected.
        RuntimeError: If no tables are found or file is corrupt.
    """
    _, ext = os.path.splitext(file_path)
    ext = ext.lower()

    # Determine file type
    if ext == SUPPORTED_PDF_EXTENSION:
        file_type = "pdf"

        # Check for encrypted PDF
        if _is_encrypted_pdf(file_path):
            raise ValueError(
                "File PDF terkunci dengan password. "
                "Silakan buka kunci PDF terlebih dahulu sebelum mengonversi."
            )

        # Try digital extraction first (fast path)
        if _is_digital_pdf(file_path):
            tables = _extract_tables_pdfplumber(file_path)

            # If pdfplumber found tables, use them
            if tables:
                return {
                    "file_type": file_type,
                    "tables_count": len(tables),
                    "tables": tables,
                }

        # Fallback to img2table + OCR for scanned PDFs or when pdfplumber finds nothing
        tables = _extract_tables_img2table(file_path, ext, ocr_lang)

    elif ext in SUPPORTED_IMAGE_EXTENSIONS:
        file_type = "image"
        tables = _extract_tables_img2table(file_path, ext, ocr_lang)

    else:
        raise RuntimeError(
            f"Ekstensi file '{ext}' tidak didukung. "
            f"Format yang didukung: .pdf, .png, .jpg, .jpeg, .webp"
        )

    if not tables:
        raise RuntimeError(
            "Tidak ditemukan tabel pada dokumen. "
            "Pastikan dokumen memuat tabel yang jelas dengan baris dan kolom."
        )

    return {
        "file_type": file_type,
        "tables_count": len(tables),
        "tables": tables,
    }
