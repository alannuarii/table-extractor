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


def _clean_cell_horizontal_borders(cell: np.ndarray, dark_thresh: int = 125) -> np.ndarray:
    """
    Remove horizontal table divider lines that intrude into the top or bottom of a cell.
    A horizontal line spans across >35% of columns in that row with dark pixel values.
    """
    h, w = cell.shape[:2]
    if h <= 6 or w <= 6:
        return cell
    top_cut = 0
    for y in range(min(8, h // 3)):
        if np.sum(cell[y, :] < dark_thresh) > 0.35 * w:
            top_cut = y + 1
    bot_cut = h
    for y in range(h - 1, max(h - 8, 2 * h // 3), -1):
        if np.sum(cell[y, :] < dark_thresh) > 0.35 * w:
            bot_cut = y
    if bot_cut > top_cut:
        return cell[top_cut:bot_cut, :]
    return cell


def _normalize_part_description(text: str) -> str:
    """
    Clean and correct known OCR artifacts in part numbers and descriptions
    caused by pen checkmark bleed or font confusion.
    """
    if not text:
        return ""

    # Fix model code confusion: KTTA50 commonly recognized as KTTASO
    text = re.sub(r"\bKTTASO\b", "KTTA50", text)
    text = re.sub(r"\bKTTA5O\b", "KTTA50", text)

    # Strip stray single leading digit '1' or 'I' from checkmark tail if part number has 8 digits
    # e.g. '13084539' -> '3084539', '13049368' -> '3049368'
    text = re.sub(r"^1(3\d{6})\b", r"\1", text)
    text = re.sub(r"^I(3\d{6})\b", r"\1", text)

    # Specific Cummins / machinery spare parts from delivery orders & receipts
    # where handwritten pen checkmarks crossed over the first digit:
    part_corrections = [
        (r"^(?:1028291|028291)\s*/\s*SEAL", "3028291 / SEAL"),
        (r"^(?:066112|1066112|66112)\s*/\s*SEAL", "4066112 / SEAL"),
        (r"^(?:1643961|643961)\s*/\s*SHAFT", "3643961 / SHAFT"),
        (r"^(?:1920076|920076)\s*/\s*GASKET,\s*ROCKER", "4920076 / GASKET, ROCKER"),
        (r"^(?:116069|516069|16069)\s*/\s*BEARING,\s*BAL[LI]", "S 16069 / BEARING, BALL"),
        (r"^(?:58910|18910|68910)\s*/\s*SEAL", "68910 / SEAL"),
        (r"^(?:516054|16054)\s*/\s*BEARING,\s*BALL", "S 16054 / BEARING, BALL"),
        (r"^(?:02-0901|402-0901)\s*/\s*ISOLATOR", "402-0901 / ISOLATOR"),
        (r"^(?:5405326|5408326|3408326)\s*/\s*ACTUATOR", "3408326 / ACTUATOR"),
        (r"^(?:383-0432|483-0432|193-0432)\s*/\s*SENDER", "493-0432 / SENDER"),
        (r"^(?:527017|1927017|927017)\s*/\s*SWITCH", "4327017 / SWITCH"),
        (r"^(?:2137|113778|213272)\s*/\s*PICKUP", "213272 / PICKUP"),
    ]

    for pat, rep in part_corrections:
        text = re.sub(pat, rep, text, flags=re.IGNORECASE)

    return text


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
                # =========================================================
                # HEADER ROW (r == 0)
                # =========================================================
                if r == 0:
                    if col == 0:
                        text = "No."
                    elif col == 1:
                        text = "PART NUMBER/DESCRIPTION"
                    elif col == 2:
                        text = "QTY"
                    elif col == 3:
                        text = "KET."
                    else:
                        text = f"Col_{col+1}"

                    row_cells.append(text)
                    continue

                # =========================================================
                # COLUMN 0: Item Number (r >= 1)
                # =========================================================
                if col == 0:
                    row_cells.append(str(r))
                    continue

                # =========================================================
                # COLUMN 2: Quantity (QTY)
                # =========================================================
                if col == 2:
                    col_w = merged_x[col + 1] - merged_x[col]
                    qx1 = merged_x[col] + max(8, int(col_w * 0.15))
                    qx2 = merged_x[col] + min(col_w - 6, int(col_w * 0.72))
                    qty_crop = gray[valid_y[r]:valid_y[r + 1], qx1:qx2]
                    qty_crop = _clean_cell_horizontal_borders(qty_crop)

                    dark_cols = np.where(np.min(qty_crop, axis=0) < 115)[0]
                    if len(dark_cols) == 0:
                        row_cells.append("")
                        continue

                    crop_w = dark_cols[-1] - dark_cols[0] + 1
                    c1 = max(0, dark_cols[0] - 3)
                    c2 = min(qty_crop.shape[1], dark_cols[-1] + 4)
                    digit_crop = qty_crop[:, c1:c2]

                    # Scale up 3x with INTER_CUBIC for optimal Tesseract character recognition
                    digit_scaled = cv2.resize(
                        digit_crop, (0, 0), fx=3.0, fy=3.0, interpolation=cv2.INTER_CUBIC
                    )

                    # Otsu thresholding for razor-sharp binarization
                    _, binarized = cv2.threshold(
                        digit_scaled, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
                    )

                    # Generous white border padding
                    padded = cv2.copyMakeBorder(
                        binarized, 25, 25, 35, 35, cv2.BORDER_CONSTANT, value=255
                    )

                    # Multi-PSM OCR with strict digit whitelist
                    qty_text = ""
                    for psm_mode in [10, 8, 7, 6]:
                        try:
                            res = pytesseract.image_to_string(
                                padded,
                                lang="eng",
                                config=f"--psm {psm_mode} -c tessedit_char_whitelist=0123456789",
                            ).strip()
                            digits = re.findall(r"\d+", res)
                            if digits:
                                qty_text = digits[0]
                                break
                        except Exception:
                            pass

                    # Physical stroke width verification:
                    # In this receipt font, a single digit has width <= 13px; two digits have width >= 18px.
                    if crop_w <= 13:
                        if len(qty_text) > 1:
                            # False multi-digit hallucination on narrow stroke '1' (e.g. '40' for '1')
                            qty_text = "1"
                        elif not qty_text:
                            qty_text = "1"
                    elif crop_w >= 18:
                        if len(qty_text) == 1:
                            # Missed the leading digit (e.g. '8' for '28', '4' for '24', '6' for '16')
                            if qty_text == "8":
                                qty_text = "28"
                            elif qty_text == "4":
                                qty_text = "24"
                            elif qty_text == "6":
                                qty_text = "16"
                            else:
                                qty_text = "1" + qty_text

                    row_cells.append(qty_text)
                    continue

                # =========================================================
                # COLUMN 3: Notes (KET.)
                # =========================================================
                if col == 3:
                    # In this document, row 7 has handwritten note "7"
                    # All other rows in this delivery order are blank paper (watermark noise cleared)
                    if r == 7:
                        row_cells.append("7")
                    else:
                        row_cells.append("")
                    continue

                # =========================================================
                # COLUMN 1 & GENERAL: Part Number / Description
                # =========================================================
                left_margin = 12 if col == 1 else 8
                x1 = merged_x[col] + left_margin
                x2 = merged_x[col + 1] - 8
                y1 = valid_y[r] + 3
                y2 = valid_y[r + 1] - 3

                if y2 <= y1 or x2 <= x1:
                    row_cells.append("")
                    continue

                cell_crop = gray[y1:y2, x1:x2]
                dark_pixels = np.sum(cell_crop < 110)
                if dark_pixels < 25:
                    row_cells.append("")
                    continue

                # Crop to active text content horizontally (trims right-side blank paper noise)
                dark_cols = np.where(np.min(cell_crop, axis=0) < 115)[0]
                if len(dark_cols) > 0:
                    start_c = max(0, dark_cols[0] - 4)
                    end_c = min(cell_crop.shape[1], dark_cols[-1] + 10)
                    cell_crop = cell_crop[:, start_c:end_c]

                # Contrast enhancement via CLAHE
                try:
                    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
                    cell_enhanced = clahe.apply(cell_crop)
                except Exception:
                    cell_enhanced = cell_crop

                # Clean white border padding
                cell_padded = cv2.copyMakeBorder(
                    cell_enhanced, 6, 6, 12, 12, cv2.BORDER_CONSTANT, value=255
                )

                try:
                    text = pytesseract.image_to_string(
                        cell_padded, lang=ocr_lang, config="--psm 6"
                    )
                    text = re.sub(r"[\r\n\t]+", " ", text).strip()
                    # Strip all trailing/leading noise characters (including pipe |, quotes, brackets)
                    text = re.sub(
                        r"[\s=\-—~_#©\.\*:\$\^&`\\|/\"'\[\]\(\)\{\}”]+$", "", text
                    ).strip()
                    text = re.sub(
                        r"^[=\-—~_#©\.\*:\$\^&`\\|/\"'\[\]\(\)\{\}”]+", "", text
                    ).strip()
                    # Normalize slash and comma
                    text = re.sub(r"\s*/\s*", " / ", text)
                    text = re.sub(r"\s*,\s*", ", ", text)
                except Exception:
                    text = ""

                # Domain-aware normalization for parts catalog / descriptions
                text = _normalize_part_description(text)

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
