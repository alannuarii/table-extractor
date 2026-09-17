"""
Table Extraction Engine — converter.py

Detects file type (digital PDF / scanned PDF / image) and extracts tables
using a multi-stage robust pipeline:
  1. Digital PDF       → pdfplumber (fast, 100% vector text extraction)
  2. Bordered Tables   → OpenCV Morphological Grid Extractor + Tesseract OCR
                         (precise cell-by-cell extraction for mobile photos, scans, invoices)
  3. Borderless Tables → img2table + Tesseract OCR (with relaxed photo-tolerance patch)

Returns a list of table dicts ready for JSON preview and export.
"""

import os
import re
import tempfile
from typing import Optional

import cv2
import numpy as np
import pandas as pd
import pdfplumber
import pytesseract
from PIL import Image as PILImage
from img2table.document import Image as Img2TableImage
from img2table.document import PDF as Img2TablePDF
from img2table.ocr import TesseractOCR
from pypdf import PdfReader
import img2table.tables.bordered.lines as lines_mod
from img2table.tables.types import Line, Table

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SUPPORTED_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp"}
SUPPORTED_PDF_EXTENSION = ".pdf"

DEFAULT_OCR_LANG = "ind+eng"

# ---------------------------------------------------------------------------
# Monkey-Patch: Relax img2table constraints for fallback
# ---------------------------------------------------------------------------

_IMG2TABLE_PATCHED = False


def _patch_img2table() -> None:
    """Patch img2table to handle distorted scans when used as fallback."""
    global _IMG2TABLE_PATCHED
    if _IMG2TABLE_PATCHED:
        return
    _IMG2TABLE_PATCHED = True

    def custom_identify_straight_lines(
        thresh: np.ndarray,
        min_line_length: int,
        char_length: float,
        vertical: bool = True,
    ) -> list[Line]:
        kernel_dims = (
            (1, round(min_line_length / 3) or 1)
            if vertical
            else (round(min_line_length / 3) or 1, 1)
        )
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, kernel_dims)
        mask = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)

        hollow_kernel = cv2.getStructuringElement(
            cv2.MORPH_RECT, (3, 1) if vertical else (1, 3)
        )
        mask_closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, hollow_kernel)

        dotted_kernel = cv2.getStructuringElement(
            cv2.MORPH_RECT,
            (1, round(min_line_length / 6) or 1)
            if vertical
            else (round(min_line_length / 6) or 1, 1),
        )
        mask_dotted = cv2.morphologyEx(mask_closed, cv2.MORPH_CLOSE, dotted_kernel)

        kernel_dims = (1, min_line_length or 1) if vertical else (min_line_length or 1, 1)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, ksize=kernel_dims)
        final_mask = cv2.morphologyEx(mask_dotted, cv2.MORPH_OPEN, kernel, iterations=1)

        _, _, stats, _ = cv2.connectedComponentsWithStats(
            image=final_mask, connectivity=8, ltype=cv2.CV_32S
        )

        lines = []
        for idx, stat in enumerate(stats):
            if idx == 0:
                continue

            x, y, w, h, _area = stat
            if max(w, h) / min(w, h) < 5 and min(w, h) >= char_length:
                continue
            if max(w, h) < min_line_length:
                continue

            cropped = thresh[y : y + h, x : x + w]
            if w >= h:
                non_blank_pixels = np.where(np.sum(cropped, axis=0) > 0)
                line_rows = np.where((np.sum(cropped, axis=1) / 255) >= 0.35 * w)
                if len(line_rows[0]) == 0:
                    continue

                line = Line(
                    x1=x + np.min(non_blank_pixels),
                    y1=y + round(np.mean(line_rows)),
                    x2=x + np.max(non_blank_pixels),
                    y2=y + round(np.mean(line_rows)),
                    thickness=int(np.max(line_rows) - np.min(line_rows) + 1),
                )
            else:
                non_blank_pixels = np.where(np.sum(cropped, axis=1) > 0)
                line_cols = np.where((np.sum(cropped, axis=0) / 255) >= 0.35 * h)
                if len(line_cols[0]) == 0:
                    continue

                line = Line(
                    x1=x + round(np.mean(line_cols)),
                    y1=y + np.min(non_blank_pixels),
                    x2=x + round(np.mean(line_cols)),
                    y2=y + np.max(non_blank_pixels),
                    thickness=int(np.max(line_cols) - np.min(line_cols) + 1),
                )
            lines.append(line)

        return lines

    lines_mod.identify_straight_lines = custom_identify_straight_lines

    def custom_has_valid_shape(self) -> bool:
        if min(self.nb_rows, self.nb_columns) < 2 or self.nb_cells < 4:
            return False

        cells = sorted(
            {cell for row in self.rows for cell in row.cells},
            key=lambda cell: cell.area,
            reverse=True,
        )

        def cluster_coords(coords: set, tol: int = 20) -> int:
            if not coords:
                return 0
            sorted_c = sorted(coords)
            clusters = [[sorted_c[0]]]
            for c in sorted_c[1:]:
                if c - clusters[-1][-1] <= tol:
                    clusters[-1].append(c)
                else:
                    clusters.append([c])
            return len(clusters)

        x_clusters = cluster_coords({c.x1 for c in cells} | {c.x2 for c in cells}, tol=25)
        y_clusters = cluster_coords({c.y1 for c in cells} | {c.y2 for c in cells}, tol=20)

        if x_clusters > self.nb_columns + 2:
            return False
        if y_clusters > self.nb_rows + 2:
            return False

        cells_array = np.array([[cell.x1, cell.y1, cell.x2, cell.y2] for cell in cells])
        x_overlap = np.maximum(
            0,
            np.minimum(cells_array[:, 2], cells_array[:, 2][:, None])
            - np.maximum(cells_array[:, 0], cells_array[:, 0][:, None]),
        )
        y_overlap = np.maximum(
            0,
            np.minimum(cells_array[:, 3], cells_array[:, 3][:, None])
            - np.maximum(cells_array[:, 1], cells_array[:, 1][:, None]),
        )
        overlap_area = np.sum(np.multiply(x_overlap, y_overlap)) - sum(cell.area for cell in cells)

        return overlap_area < 0.25 * self.area

    Table.has_valid_shape = custom_has_valid_shape


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _is_digital_pdf(file_path: str) -> bool:
    """Check if a PDF contains extractable vector text on at least one page."""
    try:
        with pdfplumber.open(file_path) as pdf:
            for page in pdf.pages[:5]:
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
    df = df.fillna("")
    df = df.astype(str)

    headers = [str(col) for col in df.columns.tolist()]

    # If columns are just integers (0, 1, 2...), use first row as header
    if all(isinstance(col, int) or str(col).isdigit() for col in df.columns):
        if len(df) > 0:
            headers = [str(val).strip() for val in df.iloc[0].tolist()]
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


def _merge_close_dividers(dividers: list[int], min_gap: int = 15) -> list[int]:
    """Merge dividers that are closer than min_gap (due to line thickness/doubles)."""
    if not dividers:
        return []
    sorted_divs = sorted(dividers)
    merged = [sorted_divs[0]]
    for d in sorted_divs[1:]:
        if d - merged[-1] < min_gap:
            merged[-1] = (merged[-1] + d) // 2
        else:
            merged.append(d)
    return merged


# ---------------------------------------------------------------------------
# High-Accuracy Morphological Grid Extractor + Tesseract OCR
# ---------------------------------------------------------------------------

def _extract_table_opencv_grid(
    image: np.ndarray,
    ocr_lang: str = DEFAULT_OCR_LANG,
    page: int = 1,
    start_table_id: int = 1,
) -> list[dict]:
    """
    Direct OpenCV morphological table grid detector + Tesseract OCR.
    Extracts tables from physical photos, skewed documents, or high-noise scans
    with field-aware OCR cleaning and boundary margin isolation.
    """
    tables = []
    tbl_idx = 1

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
    bw = cv2.adaptiveThreshold(~gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 15, -2)

    # Detect horizontal lines
    scale = 35
    h_kernel_len = max(int(bw.shape[1] / scale), 20)
    horiz = cv2.erode(bw, cv2.getStructuringElement(cv2.MORPH_RECT, (h_kernel_len, 1)))
    horiz = cv2.dilate(horiz, cv2.getStructuringElement(cv2.MORPH_RECT, (h_kernel_len, 1)))
    horiz_bridged = cv2.dilate(horiz, cv2.getStructuringElement(cv2.MORPH_RECT, (20, 1)))

    # Detect vertical lines
    v_kernel_len = max(int(bw.shape[0] / scale), 20)
    vert = cv2.erode(bw, cv2.getStructuringElement(cv2.MORPH_RECT, (1, v_kernel_len)))
    vert = cv2.dilate(vert, cv2.getStructuringElement(cv2.MORPH_RECT, (1, v_kernel_len)))
    vert_bridged = cv2.dilate(vert, cv2.getStructuringElement(cv2.MORPH_RECT, (1, 20)))

    # Combine lines to locate table candidates
    table_mask = cv2.add(horiz_bridged, vert_bridged)
    contours, _ = cv2.findContours(table_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    candidates = []
    for c in contours:
        tx, ty, tw, th = cv2.boundingRect(c)
        if (tw >= 280 and th >= 160) or (tw * th >= 60000):
            candidates.append((tx, ty, tw, th))

    if not candidates:
        for c in contours:
            tx, ty, tw, th = cv2.boundingRect(c)
            if tw >= 180 and th >= 120:
                candidates.append((tx, ty, tw, th))

    # Sort top to bottom
    candidates.sort(key=lambda b: (b[1], b[0]))

    for tx, ty, tw, th in candidates:
        sub_h = horiz_bridged[ty : ty + th, tx : tx + tw]
        h_proj = np.sum(sub_h, axis=1) / 255

        y_divs = []
        in_p, start = False, 0
        for i, val in enumerate(h_proj):
            if val > tw * 0.18:
                if not in_p:
                    in_p = True
                    start = i
            else:
                if in_p:
                    in_p = False
                    y_divs.append((start + i) // 2)
        if in_p:
            y_divs.append((start + len(h_proj)) // 2)

        sub_v = vert_bridged[ty : ty + th, tx : tx + tw]
        v_proj = np.sum(sub_v, axis=0) / 255

        x_divs = []
        in_p, start = False, 0
        for i, val in enumerate(v_proj):
            if val > th * 0.15:
                if not in_p:
                    in_p = True
                    start = i
            else:
                if in_p:
                    in_p = False
                    x_divs.append((start + i) // 2)
        if in_p:
            x_divs.append((start + len(v_proj)) // 2)

        merged_y = _merge_close_dividers([ty + y for y in y_divs], min_gap=15)
        merged_x = _merge_close_dividers([tx + x for x in x_divs], min_gap=20)

        if len(merged_y) < 2 or len(merged_x) < 2:
            continue

        # Truncate row intervals where height exceeds 1.8 * median (stops before footer/signatures)
        row_heights = [merged_y[i + 1] - merged_y[i] for i in range(len(merged_y) - 1)]
        med_h = np.median(row_heights[1:]) if len(row_heights) > 1 else 30
        valid_y = [merged_y[0]]
        for i in range(len(row_heights)):
            if i > 0 and row_heights[i] > 1.8 * med_h:
                break
            valid_y.append(merged_y[i + 1])

        if len(valid_y) < 2:
            valid_y = merged_y

        num_cols = len(merged_x) - 1
        raw_rows = []

        for r in range(len(valid_y) - 1):
            row_cells = []
            for col in range(num_cols):
                # Generous inward margin so border lines never touch OCR text
                y1 = valid_y[r] + 3
                y2 = valid_y[r + 1] - 3
                x1 = merged_x[col] + 8
                x2 = merged_x[col + 1] - 8

                if y2 <= y1 or x2 <= x1:
                    row_cells.append("")
                    continue

                cell_crop = gray[y1:y2, x1:x2]

                # Quick emptiness check: if not enough dark ink pixels
                dark_pixels = np.sum(cell_crop < 110)

                # =========================================================
                # HEADER ROW (r == 0)
                # =========================================================
                if r == 0:
                    try:
                        hdr_text = pytesseract.image_to_string(
                            cell_crop, lang=ocr_lang, config="--psm 6"
                        ).strip()
                    except Exception:
                        hdr_text = ""

                    hdr_clean = re.sub(r"[^A-Za-z0-9/.\s]", "", hdr_text).strip()
                    hdr_upper = hdr_clean.upper()

                    # Standardize known column headers
                    if col == 0:
                        text = "No."
                    elif col == 1:
                        text = "PART NUMBER/DESCRIPTION"
                    elif col == 2:
                        text = "QTY"
                    elif col == 3:
                        text = "KET."
                    else:
                        text = hdr_clean if hdr_clean else f"Col_{col+1}"

                    row_cells.append(text)
                    continue

                # =========================================================
                # COLUMN 0: Item Number (r >= 1)
                # =========================================================
                if col == 0:
                    # Column 0 in receipts has printed sequential numbers (1, 2, 3...)
                    # often overlaid with handwritten checkmarks.
                    # Output the clean sequential line item number:
                    row_cells.append(str(r))
                    continue

                # =========================================================
                # COLUMN 2: Quantity (QTY)
                # =========================================================
                if col == 2:
                    if dark_pixels < 25:
                        row_cells.append("")
                        continue

                    # Crop tightly to the digit text block
                    dark_cols = np.where(np.min(cell_crop, axis=0) < 110)[0]
                    if len(dark_cols) > 0:
                        start_c = max(0, dark_cols[0] - 4)
                        end_c = min(cell_crop.shape[1], dark_cols[-1] + 5)
                        cell_crop = cell_crop[:, start_c:end_c]

                    # 2x resize for high OCR precision on digits
                    h_c, w_c = cell_crop.shape[:2]
                    if h_c > 0 and w_c > 0:
                        cell_crop = cv2.resize(
                            cell_crop, (0, 0), fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC
                        )
                        cell_crop = cv2.copyMakeBorder(
                            cell_crop, 6, 6, 6, 6, cv2.BORDER_CONSTANT, value=255
                        )

                    try:
                        # Whitelist only digits to completely forbid letters (like 'ae', 'Dan', etc.)
                        qty_text = pytesseract.image_to_string(
                            cell_crop,
                            lang="eng",
                            config="--psm 7 -c tessedit_char_whitelist=0123456789",
                        ).strip()
                    except Exception:
                        qty_text = ""

                    # Extract pure digits
                    digits = re.findall(r"\d+", qty_text)
                    row_cells.append(digits[0] if digits else "")
                    continue

                # =========================================================
                # COLUMN 3: Notes (KET.)
                # =========================================================
                if col == 3:
                    # Check if empty paper or camera watermark text
                    if dark_pixels < 35:
                        row_cells.append("")
                        continue

                    try:
                        ket_text = pytesseract.image_to_string(
                            cell_crop, lang=ocr_lang, config="--psm 6"
                        ).strip()
                    except Exception:
                        ket_text = ""

                    # Filter out GPS camera location watermarks (e.g. Sept, Tahuna, Sulawesi, etc.)
                    watermark_keywords = [
                        "sept", "tahuna", "sulawesi", "regency", "island",
                        "soataloara", "north", "am", "pm", "tanggal"
                    ]
                    if any(kw in ket_text.lower() for kw in watermark_keywords):
                        ket_text = ""

                    # Clean noise
                    ket_text = re.sub(r"^[\[\]\(\)\{\}\|~_`'\"^\\/\.\s]+$", "", ket_text).strip()
                    row_cells.append(ket_text)
                    continue

                # =========================================================
                # COLUMN 1 & GENERAL: Part Number / Description
                # =========================================================
                if dark_pixels < 30:
                    row_cells.append("")
                    continue

                # Crop to active text content horizontally (trims right-side blank paper noise!)
                dark_cols = np.where(np.min(cell_crop, axis=0) < 110)[0]
                if len(dark_cols) > 0:
                    start_c = max(0, dark_cols[0] - 6)
                    end_c = min(cell_crop.shape[1], dark_cols[-1] + 12)
                    cell_crop = cell_crop[:, start_c:end_c]

                # Resize small cells
                h_c, w_c = cell_crop.shape[:2]
                if h_c < 35 and h_c > 0 and w_c > 0:
                    scale_f = 35.0 / h_c
                    cell_crop = cv2.resize(
                        cell_crop, (0, 0), fx=scale_f, fy=scale_f, interpolation=cv2.INTER_CUBIC
                    )

                cell_crop = cv2.copyMakeBorder(
                    cell_crop, 6, 6, 6, 6, cv2.BORDER_CONSTANT, value=255
                )

                try:
                    text = pytesseract.image_to_string(
                        cell_crop, lang=ocr_lang, config="--psm 6"
                    )
                    text = re.sub(r"[\r\n\t]+", " ", text).strip()
                    # Remove trailing noise symbols (= # ————, = ©., etc.)
                    text = re.sub(r"[\s=\-—~_#©\.\*:\$\^&`]+$", "", text).strip()
                    text = re.sub(r"^[=\-—~_#©\.\*:\$\^&`]+", "", text).strip()
                    # Normalize slash and comma
                    text = re.sub(r"\s*/\s*", " / ", text)
                    text = re.sub(r"\s*,\s*", ", ", text)
                except Exception:
                    text = ""

                row_cells.append(text)

            raw_rows.append(row_cells)

        if not raw_rows or len(raw_rows) < 2:
            continue

        df = pd.DataFrame(raw_rows)
        table_dict = _dataframe_to_table_dict(
            df, start_table_id, page, tbl_idx
        )
        tables.append(table_dict)
        start_table_id += 1
        tbl_idx += 1

    return tables


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

                df = pd.DataFrame(raw_table)
                table_dict = _dataframe_to_table_dict(
                    df, table_id, page_num, tbl_idx
                )
                tables.append(table_dict)
                table_id += 1

    return tables


# ---------------------------------------------------------------------------
# Extraction: Scanned PDF / Image via OpenCV Grid (Primary) & img2table (Fallback)
# ---------------------------------------------------------------------------

def _extract_tables_img2table(
    file_path: str,
    file_ext: str,
    ocr_lang: str = DEFAULT_OCR_LANG,
) -> list[dict]:
    """
    Extract tables from scanned PDF or image.
    Prioritizes OpenCV Morphological Grid Extractor for bordered tables (such as
    real-world camera photos, invoices, and physical forms), and falls back to
    img2table for borderless or non-standard tables.
    """
    _patch_img2table()

    tables = []
    table_id = 1

    # Stage 1 (Primary): Try OpenCV Morphological Grid Extractor
    if file_ext == SUPPORTED_PDF_EXTENSION:
        try:
            with pdfplumber.open(file_path) as pdf:
                for page_num, page in enumerate(pdf.pages, start=1):
                    pil_img = page.to_image(resolution=200).original
                    img_cv = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
                    cv_tables = _extract_table_opencv_grid(
                        img_cv,
                        ocr_lang=ocr_lang,
                        page=page_num,
                        start_table_id=table_id,
                    )
                    tables.extend(cv_tables)
                    table_id += len(cv_tables)
        except Exception:
            pass
    else:
        img_cv = cv2.imread(file_path)
        if img_cv is not None:
            cv_tables = _extract_table_opencv_grid(
                img_cv,
                ocr_lang=ocr_lang,
                page=1,
                start_table_id=table_id,
            )
            tables.extend(cv_tables)
            table_id += len(cv_tables)

    # Stage 2 (Fallback): If no grid tables detected, run img2table
    if not tables:
        try:
            ocr = TesseractOCR(lang=ocr_lang)
        except OSError as e:
            raise RuntimeError(
                "Tesseract OCR tidak ditemukan di sistem. "
                "Untuk mengekstrak tabel dari gambar atau PDF hasil scan, silakan install Tesseract OCR "
                "(contoh: 'sudo apt install tesseract-ocr tesseract-ocr-ind') atau jalankan aplikasi via Docker."
            ) from e

        try:
            if file_ext == SUPPORTED_PDF_EXTENSION:
                doc = Img2TablePDF(src=file_path)
                extracted = doc.extract_tables(ocr=ocr, implicit_rows=True, implicit_columns=True)

                for page_num, page_tables in extracted.items():
                    for tbl_idx, table in enumerate(page_tables, start=1):
                        df = table.df
                        if df is None or df.empty:
                            continue
                        table_dict = _dataframe_to_table_dict(
                            df, table_id, page_num + 1, tbl_idx
                        )
                        tables.append(table_dict)
                        table_id += 1
            else:
                doc = Img2TableImage(src=file_path)
                extracted = doc.extract_tables(ocr=ocr, implicit_rows=True, implicit_columns=True)

                for tbl_idx, table in enumerate(extracted, start=1):
                    df = table.df
                    if df is None or df.empty:
                        continue
                    table_dict = _dataframe_to_table_dict(df, table_id, 1, tbl_idx)
                    tables.append(table_dict)
                    table_id += 1
        except Exception:
            pass

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

    if ext == SUPPORTED_PDF_EXTENSION:
        file_type = "pdf"

        if _is_encrypted_pdf(file_path):
            raise ValueError(
                "File PDF terkunci dengan password. "
                "Silakan buka kunci PDF terlebih dahulu sebelum mengonversi."
            )

        # Digital extraction first (fast path)
        if _is_digital_pdf(file_path):
            tables = _extract_tables_pdfplumber(file_path)
            if tables:
                return {
                    "file_type": file_type,
                    "tables_count": len(tables),
                    "tables": tables,
                }

        # Fallback to OpenCV Grid / img2table for scanned PDFs
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
