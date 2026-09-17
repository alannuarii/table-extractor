"""
table-extractor — FastAPI Application.

Serves the API endpoints for table extraction, download, and the static frontend UI.
"""

import os
import tempfile

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from starlette.background import BackgroundTask

from app.converter import extract_tables
from app.exporter import export_to_csv_single, export_to_csv_zip, export_to_xlsx

app = FastAPI(
    title="Table Extractor",
    description="Ekstrak tabel dari PDF atau gambar dan konversi ke Excel/CSV.",
    version="1.0.0",
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

ALLOWED_MIME_TYPES = {
    "application/pdf",
    "image/png",
    "image/jpeg",
    "image/webp",
}

ALLOWED_EXTENSIONS = {".pdf", ".png", ".jpg", ".jpeg", ".webp"}

MAX_FILE_SIZE = 50 * 1024 * 1024  # 50 MB


# ---------------------------------------------------------------------------
# Background cleanup
# ---------------------------------------------------------------------------

def _cleanup_file(path: str):
    """Remove a temporary file if it exists."""
    if path and os.path.exists(path):
        os.unlink(path)


# ---------------------------------------------------------------------------
# API Endpoints
# ---------------------------------------------------------------------------

@app.get("/api/health")
async def healthcheck():
    """Healthcheck endpoint for monitoring and CI/CD pipelines."""
    return {"status": "ok", "version": "1.0.0"}


@app.post("/api/convert")
async def convert_file(
    file: UploadFile = File(...),
    ocr_lang: str = Form("ind+eng"),
):
    """
    Extract tables from an uploaded PDF or image file.

    Accepts multipart/form-data with:
    - `file`: Binary PDF or image file
    - `ocr_lang`: OCR language string (default: "ind+eng")

    Returns JSON with extracted table data for interactive preview.
    """
    # --- Validate file extension ---
    filename = file.filename or "document"
    _, ext = os.path.splitext(filename)
    if ext.lower() not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Ekstensi file '{ext}' tidak didukung. "
                f"Format yang diterima: .pdf, .png, .jpg, .jpeg, .webp"
            ),
        )

    # --- Validate MIME type ---
    if file.content_type and file.content_type not in ALLOWED_MIME_TYPES:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Tipe file tidak valid ({file.content_type}). "
                f"Tipe yang diterima: PDF, PNG, JPG, WEBP."
            ),
        )

    # --- Read file content ---
    try:
        content = await file.read()
    except Exception:
        raise HTTPException(
            status_code=400,
            detail="Gagal membaca file yang diunggah.",
        )

    # --- Validate file size ---
    if len(content) > MAX_FILE_SIZE:
        raise HTTPException(
            status_code=400,
            detail=f"Ukuran file melebihi batas maksimum ({MAX_FILE_SIZE // (1024 * 1024)} MB).",
        )

    # --- Extract tables using temp file ---
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(
            suffix=ext.lower(), delete=False
        ) as tmp:
            tmp.write(content)
            tmp_path = tmp.name

        result = extract_tables(tmp_path, ocr_lang=ocr_lang)

        return JSONResponse(content={
            "success": True,
            "filename": filename,
            "file_type": result["file_type"],
            "tables_count": result["tables_count"],
            "tables": result["tables"],
        })

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e))

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Terjadi kesalahan internal saat mengonversi file: {e}",
        )

    finally:
        _cleanup_file(tmp_path)


@app.post("/api/download")
async def download_file(payload: dict):
    """
    Generate and download the converted file (Excel or CSV).

    Accepts JSON body:
    - `filename`: Original filename (without extension)
    - `format`: "xlsx" or "csv"
    - `tables`: Array of table objects from the convert response
    """
    fmt = payload.get("format", "xlsx").lower()
    filename_base = payload.get("filename", "export")
    tables = payload.get("tables", [])

    # Strip extension from filename if present
    filename_base = os.path.splitext(filename_base)[0]

    if not tables:
        raise HTTPException(
            status_code=400,
            detail="Tidak ada data tabel untuk diekspor.",
        )

    if fmt == "xlsx":
        buffer = export_to_xlsx(tables, filename_base)
        return StreamingResponse(
            buffer,
            media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            headers={
                "Content-Disposition": f'attachment; filename="{filename_base}.xlsx"'
            },
        )

    elif fmt == "csv":
        if len(tables) == 1:
            buffer = export_to_csv_single(tables[0])
            return StreamingResponse(
                buffer,
                media_type="text/csv; charset=utf-8",
                headers={
                    "Content-Disposition": f'attachment; filename="{filename_base}.csv"'
                },
            )
        else:
            buffer = export_to_csv_zip(tables, filename_base)
            return StreamingResponse(
                buffer,
                media_type="application/zip",
                headers={
                    "Content-Disposition": f'attachment; filename="{filename_base}_csv.zip"'
                },
            )

    else:
        raise HTTPException(
            status_code=400,
            detail=f"Format '{fmt}' tidak didukung. Gunakan 'xlsx' atau 'csv'.",
        )


# ---------------------------------------------------------------------------
# Static Files — Mount AFTER API routes so /api/* takes priority
# ---------------------------------------------------------------------------

static_dir = os.path.join(os.path.dirname(__file__), "static")
favicon_path = os.path.join(static_dir, "favicon.svg")


@app.api_route("/favicon.ico", methods=["GET", "HEAD"], include_in_schema=False)
async def favicon():
    if os.path.exists(favicon_path):
        return FileResponse(favicon_path, media_type="image/svg+xml")
    return JSONResponse(status_code=404, content={"detail": "Not found"})


app.mount("/", StaticFiles(directory=static_dir, html=True), name="static")
