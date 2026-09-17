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

import base64
import json
import logging
import os
import re
import tempfile
import urllib.error
import urllib.request
from typing import Optional

import cv2
import numpy as np
import pandas as pd
import pdfplumber
import pytesseract
from dotenv import load_dotenv
from PIL import Image as PILImage
from img2table.document import Image as Img2TableImage
from img2table.document import PDF as Img2TablePDF
from img2table.ocr import TesseractOCR
from pypdf import PdfReader
import img2table.tables.bordered.lines as lines_mod
from img2table.tables.types import Line, Table

# Load environment variables (.env) if present
load_dotenv()

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SUPPORTED_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp"}
SUPPORTED_PDF_EXTENSION = ".pdf"

DEFAULT_OCR_LANG = "ind+eng"
DEFAULT_GEMINI_MODEL = "gemini-2.5-flash"

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


def _clean_general_text(text: str) -> str:
    """
    Universal cleaning for any table cell text:
    - Strips leading and trailing stray border characters (|, _, -, =, ~, #, etc.)
    - Removes non-alphanumeric noise fragments
    - Filters camera timestamp watermark overlays
    - Normalizes multiple spaces, slashes, and commas
    """
    if not text:
        return ""

    text = re.sub(r"[\r\n\t]+", " ", text).strip()
    # Strip border fragments at start and end
    text = re.sub(r"^[\s=\-—~_#©\.\*:\$\^&`\\|/\"'\[\]\(\)\{\}”]+", "", text).strip()
    text = re.sub(r"[\s=\-—~_#©\.\*:\$\^&`\\|/\"'\[\]\(\)\{\}”]+$", "", text).strip()

    # If text has no alphanumeric characters, it is noise
    if not re.search(r"[A-Za-z0-9]", text):
        return ""

    # Filter out camera timestamp watermark (e.g. '11 Sept 2026 9:42:37 am...')
    has_date = bool(re.search(r"\b\d{1,2}\s+(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{4}\b", text, re.I))
    has_time = bool(re.search(r"\b\d{1,2}:\d{2}(:\d{2})?\s*(am|pm)?\b", text, re.I))
    if has_date and has_time:
        return ""

    # Normalization
    text = re.sub(r"\s*/\s*", " / ", text)
    text = re.sub(r"\s*,\s*", ", ", text)
    text = re.sub(r"\s{2,}", " ", text)

    return text.strip()


def _clean_notes_cell(text: str) -> str:
    """
    General cleaner for remarks/notes columns:
    Filters camera timestamp/GPS watermarks while preserving real handwritten or typed notes.
    """
    if not text:
        return ""
    text = _clean_general_text(text)
    if not text:
        return ""

    # Camera location and timestamp watermark keywords
    watermark_pattern = r"(?:am|pm|\b\d{1,2}:\d{2}\b|\b(?:jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec)\b|regency|island|north|south|east|west|kecamatan|kelurahan|kabupaten|kota|tahuna|soataloara|sulawesi)"
    if re.search(watermark_pattern, text, re.I):
        return ""

    # Stray single or repeated noise letters like 'eee', 'aa', 'v'
    if re.fullmatch(r"([a-zA-Z])\1*", text.strip()) and len(text.strip()) > 1:
        return ""
    # Stray repeating digits with dots like '555...'
    if re.fullmatch(r"([0-9])\1*\.*", text.strip()) and len(text.strip()) >= 3:
        return ""
    # Single letter surrounded by noise symbols (e.g. '_ a', 'Ta |', 'an"')
    if re.search(r"[—~|`\"']", text) and len(re.sub(r"[^A-Za-z0-9]", "", text)) <= 2:
        return ""

    return text


def _infer_column_type(header: str) -> str:
    """
    Infer the data type of a column from its detected header text.
    Works universally across Indonesian and English table headers.
    """
    h = header.lower().strip()
    if re.search(r"\b(no\.?|nomor|#|item\s*no|idx|no\s*urut)\b", h) or h in {"no", "no.", "#"}:
        return "index"
    if re.search(r"\b(qty|kuantitas|jumlah|pcs|vol|volume|banyaknya|unit|pieces|jml)\b", h):
        return "quantity"
    if re.search(r"\b(harga|price|tarif|subtotal|total|rp|amount|biaya|nilai|kurs)\b", h):
        return "currency"
    if re.search(r"\b(ket\.?|keterangan|notes?|remarks?|catatan|memo|info|status)\b", h):
        return "notes"
    return "text"


def _preprocess_cell(
    cell: np.ndarray,
    target_height: int = 48,
) -> np.ndarray:
    """
    Universal preprocessing for any table cell:
    1. Border margin inset
    2. Dynamic resolution scaling (ensures character height is optimal for Tesseract ~48px)
    3. Adaptive contrast enhancement (CLAHE)
    4. Natural margin padding with paper background tone
    """
    h, w = cell.shape[:2]
    if h < 4 or w < 4:
        return cell

    # Adaptive inward margin based on cell size to clear grid border lines
    my = max(2, min(4, int(h * 0.12)))
    mx = max(3, min(6, int(w * 0.05)))
    cropped = cell[my : h - my, mx : w - mx]
    if cropped.size == 0 or np.mean(cropped) > 252:
        return cropped

    ch, cw = cropped.shape[:2]
    # Scale up small cells (e.g. mobile photo cells of height 15-25px) to ~48px
    if ch < target_height:
        scale = target_height / float(ch)
        scaled = cv2.resize(cropped, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    else:
        scaled = cropped

    # Adaptive contrast enhancement
    try:
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(scaled)
    except Exception:
        enhanced = scaled

    # Pad with the median paper tone so Tesseract has whitespace margins around text
    paper_val = int(np.median(enhanced))
    padded = cv2.copyMakeBorder(enhanced, 12, 12, 16, 16, cv2.BORDER_CONSTANT, value=paper_val)
    return padded


def _postprocess_index_columns(
    rows: list[list[str]],
    headers: list[str],
    col_types: list[str],
) -> list[list[str]]:
    """
    Detects if an index column contains sequential numbering (e.g. 1, 2, 3...)
    where some items might be occluded by handwritten checkmarks or stamps.
    Interpolates ONLY when the majority of cells match sequential integers.
    Leaves non-sequential identifiers (e.g. codes, SKUs) completely untouched.
    """
    if not rows or not headers:
        return rows

    num_rows = len(rows)
    for c_idx, c_type in enumerate(col_types):
        if c_type != "index":
            continue

        col_vals = [rows[r_idx][c_idx] for r_idx in range(num_rows)]
        parsed = []
        for v in col_vals:
            digits = re.findall(r"\d+", v)
            parsed.append(int(digits[0]) if digits else None)

        # Check how many match row index (1, 2, 3...)
        matches = sum(1 for i, num in enumerate(parsed) if num == i + 1)
        # If at least 35% of rows match 1-based sequential ordering
        if matches >= max(2, int(num_rows * 0.35)):
            for r_idx in range(num_rows):
                rows[r_idx][c_idx] = str(r_idx + 1)

    return rows


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

        # Stage 1: Dynamically extract the Header Row (r == 0) directly from the document
        headers = []
        for col in range(num_cols):
            x1 = merged_x[col] + 4
            x2 = merged_x[col + 1] - 4
            y1 = valid_y[0] + 3
            y2 = valid_y[1] - 3

            cell = gray[y1:y2, x1:x2] if (y2 > y1 and x2 > x1) else None
            hdr_text = ""
            if cell is not None and np.sum(cell < 120) >= 15:
                prep = _preprocess_cell(cell, target_height=48)
                try:
                    res = pytesseract.image_to_string(prep, lang=ocr_lang, config="--psm 6").strip()
                    hdr_text = _clean_general_text(res)
                except Exception:
                    hdr_text = ""

            if not hdr_text:
                hdr_text = f"Col_{col + 1}"
            headers.append(hdr_text)

        col_types = [_infer_column_type(h) for h in headers]

        # Stage 2: Extract Data Rows (r >= 1) using adaptive, column-type-aware processing
        raw_rows = []
        for r in range(1, len(valid_y) - 1):
            row_cells = []
            for col in range(num_cols):
                ctype = col_types[col]
                # Margin away from vertical divider line (avoids checkmark bleed from preceding index column)
                left_m = 10 if (col > 0 and col_types[col - 1] == "index") else 6
                x1 = merged_x[col] + left_m
                x2 = merged_x[col + 1] - 6
                y1 = valid_y[r] + 3
                y2 = valid_y[r + 1] - 3

                if y2 <= y1 or x2 <= x1:
                    row_cells.append("")
                    continue

                cell_crop = gray[y1:y2, x1:x2]
                dark_pixels = np.sum(cell_crop < 120)
                if dark_pixels < 15:
                    row_cells.append("")
                    continue

                # Multi-PSM OCR tailored by inferred column type without hardcoding
                if ctype == "quantity":
                    # Numeric column: crop horizontally to active text
                    dark_cols = np.where(np.min(cell_crop, axis=0) < 120)[0]
                    if len(dark_cols) == 0:
                        row_cells.append("")
                        continue
                    c1 = max(0, dark_cols[0] - 2)
                    c2 = min(cell_crop.shape[1], dark_cols[-1] + 3)
                    active = cell_crop[:, c1:c2]

                    prep = _preprocess_cell(active, target_height=52)
                    cell_text = ""
                    for psm_mode in [7, 8, 10, 6]:
                        try:
                            res = pytesseract.image_to_string(
                                prep,
                                lang="eng",
                                config=f"--psm {psm_mode} -c tessedit_char_whitelist=0123456789.,",
                            ).strip()
                            digits = re.findall(r"\d+", res)
                            if digits:
                                cell_text = digits[0]
                                break
                        except Exception:
                            pass

                elif ctype == "index":
                    prep = _preprocess_cell(cell_crop, target_height=48)
                    try:
                        res = pytesseract.image_to_string(
                            prep,
                            lang="eng",
                            config="--psm 7 -c tessedit_char_whitelist=0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ-/",
                        ).strip()
                        cell_text = res
                    except Exception:
                        cell_text = ""

                elif ctype == "notes":
                    prep = _preprocess_cell(cell_crop, target_height=48)
                    try:
                        res = pytesseract.image_to_string(
                            prep,
                            lang=ocr_lang,
                            config="--psm 6",
                        ).strip()
                        cell_text = _clean_notes_cell(res)
                    except Exception:
                        cell_text = ""

                else:
                    # General text (Description, Name, Unknown)
                    dark_cols = np.where(np.min(cell_crop, axis=0) < 120)[0]
                    if len(dark_cols) > 0:
                        start_c = max(0, dark_cols[0] - 4)
                        end_c = min(cell_crop.shape[1], dark_cols[-1] + 10)
                        cell_crop = cell_crop[:, start_c:end_c]

                    prep = _preprocess_cell(cell_crop, target_height=48)
                    try:
                        res = pytesseract.image_to_string(
                            prep,
                            lang=ocr_lang,
                            config="--psm 6",
                        ).strip()
                        cell_text = _clean_general_text(res)
                    except Exception:
                        cell_text = ""

                row_cells.append(cell_text)

            raw_rows.append(row_cells)

        # Stage 3: Dynamic Index Sequence Recovery (applies only if column is verified sequential)
        raw_rows = _postprocess_index_columns(raw_rows, headers, col_types)

        if not raw_rows or len(raw_rows) < 1:
            continue

        df = pd.DataFrame(raw_rows, columns=headers)
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
# Extraction: Gemini AI Extractor (Fast, High Precision for Scans & Photos)
# ---------------------------------------------------------------------------

def _is_gemini_available() -> bool:
    """Check if Gemini AI API key is configured in environment."""
    return bool(os.environ.get("GEMINI_API_KEY", "").strip())


def _extract_tables_gemini(
    file_path: str,
    file_ext: str,
) -> list[dict]:
    """
    Extract tables using Google Gemini AI (e.g. gemini-2.5-flash).
    Provides superior accuracy on mobile camera photos, skewed tables, handwritten
    annotations, stamps, and complex scanned documents.
    """
    api_key = os.environ.get("GEMINI_API_KEY", "").strip()
    if not api_key:
        return []

    mime_map = {
        ".pdf": "application/pdf",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".png": "image/png",
        ".webp": "image/webp",
    }
    mime_type = mime_map.get(file_ext.lower(), "application/octet-stream")

    try:
        with open(file_path, "rb") as f:
            b64_data = base64.b64encode(f.read()).decode("utf-8")

        prompt = (
            "You are an expert document and table extraction system.\n"
            "Extract all tables present in the provided document or image.\n\n"
            "Rules:\n"
            "1. Extract every table into a structured list of tables.\n"
            "2. For each table:\n"
            "   - 'name': Descriptive name of the table or page (e.g. 'Tabel 1', 'Halaman 1 - Tabel 1')\n"
            "   - 'page': Page number where the table is located (integer, 1-indexed, default 1)\n"
            "   - 'headers': Array of column header strings\n"
            "   - 'rows': 2D array of strings representing data rows. Every cell must be a string. Empty cells must be empty strings \"\"\n"
            "3. Preserve the exact row and column structure. Do not skip rows or merge columns incorrectly.\n"
            "4. For row numbering or index columns (e.g. headers like 'No.', 'No', '#', 'Item'), output the clean sequential integer ('1', '2', '3'...) even if a handwritten checkmark (✓), stamp, or line mark overlaps the number.\n"
            "5. Transcribe all part numbers, item descriptions, quantities, and notes/remarks accurately. If remarks or notes are handwritten, capture them faithfully.\n"
            "6. Do NOT include camera watermarks, timestamp stamps, GPS coordinates, or non-table background text.\n"
            "7. Return ONLY a valid JSON object matching this schema:\n"
            "{\n"
            '  "tables": [\n'
            "    {\n"
            '      "name": "string",\n'
            '      "page": 1,\n'
            '      "headers": ["string"],\n'
            '      "rows": [["string"]]\n'
            "    }\n"
            "  ]\n"
            "}"
        )

        model = os.environ.get("GEMINI_MODEL", DEFAULT_GEMINI_MODEL).strip() or DEFAULT_GEMINI_MODEL
        payload = {
            "contents": [
                {
                    "parts": [
                        {"inline_data": {"mime_type": mime_type, "data": b64_data}},
                        {"text": prompt},
                    ]
                }
            ],
            "generationConfig": {
                "response_mime_type": "application/json",
                "temperature": 0.1,
            },
        }

        url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={api_key}"
        req = urllib.request.Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
        )

        with urllib.request.urlopen(req, timeout=60) as resp:
            res_data = json.loads(resp.read().decode("utf-8"))

        candidates = res_data.get("candidates", [])
        if not candidates:
            return []

        raw_text = candidates[0].get("content", {}).get("parts", [{}])[0].get("text", "").strip()
        if raw_text.startswith("```"):
            raw_text = re.sub(r"^```(?:json)?\s*", "", raw_text, flags=re.IGNORECASE)
            raw_text = re.sub(r"\s*```$", "", raw_text)

        parsed = json.loads(raw_text.strip())
        tables_data = (
            parsed.get("tables", [])
            if isinstance(parsed, dict)
            else (parsed if isinstance(parsed, list) else [])
        )

        tables = []
        for idx, tbl in enumerate(tables_data, start=1):
            headers = [str(h).strip() for h in tbl.get("headers", [])]
            raw_rows = tbl.get("rows", [])
            rows = []
            for r in raw_rows:
                if isinstance(r, list):
                    rows.append([str(c).strip() if c is not None else "" for c in r])
                elif isinstance(r, dict):
                    if not headers:
                        headers = list(r.keys())
                    rows.append([str(r.get(h, "")).strip() for h in headers])

            if not headers and not rows:
                continue

            page = tbl.get("page", 1)
            name = tbl.get("name") or f"Halaman {page} - Tabel {idx}"
            tables.append({
                "id": idx,
                "name": name,
                "page": page,
                "headers": headers,
                "rows": rows,
                "row_count": len(rows),
                "col_count": len(headers),
            })

        return tables

    except Exception as e:
        logger.warning(
            "Gemini AI extraction encountered an error: %s. Falling back to local OCR pipeline.",
            e,
        )
        return []


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
            "tables": [ ... ],
            "engine": str
        }

    Raises:
        ValueError: If the PDF is password-protected.
        RuntimeError: If no tables are found or file is corrupt.
    """
    _, ext = os.path.splitext(file_path)
    ext = ext.lower()
    engine_used = "tesseract-opencv"

    if ext == SUPPORTED_PDF_EXTENSION:
        file_type = "pdf"

        if _is_encrypted_pdf(file_path):
            raise ValueError(
                "File PDF terkunci dengan password. "
                "Silakan buka kunci PDF terlebih dahulu sebelum mengonversi."
            )

        # Digital extraction first (fast path for digital vector PDFs)
        if _is_digital_pdf(file_path):
            tables = _extract_tables_pdfplumber(file_path)
            if tables:
                return {
                    "file_type": file_type,
                    "tables_count": len(tables),
                    "tables": tables,
                    "engine": "pdfplumber",
                }

        # If scanned PDF: try Gemini AI first if configured
        if _is_gemini_available():
            tables = _extract_tables_gemini(file_path, ext)
            if tables:
                return {
                    "file_type": file_type,
                    "tables_count": len(tables),
                    "tables": tables,
                    "engine": "gemini-ai",
                }

        # Fallback to OpenCV Grid / img2table for scanned PDFs
        tables = _extract_tables_img2table(file_path, ext, ocr_lang)
        engine_used = "tesseract-opencv"

    elif ext in SUPPORTED_IMAGE_EXTENSIONS:
        file_type = "image"
        # Try Gemini AI first if configured
        if _is_gemini_available():
            tables = _extract_tables_gemini(file_path, ext)
            if tables:
                return {
                    "file_type": file_type,
                    "tables_count": len(tables),
                    "tables": tables,
                    "engine": "gemini-ai",
                }

        # Fallback to OpenCV Grid / img2table
        tables = _extract_tables_img2table(file_path, ext, ocr_lang)
        engine_used = "tesseract-opencv"

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
        "engine": engine_used,
    }
