# Table Extractor

Aplikasi web self-hosted untuk mengekstrak tabel dari file **PDF** atau **Gambar** (PNG, JPG, JPEG, WEBP) dan mengonversinya menjadi file **Excel (.xlsx)** atau **CSV (.csv)**.

## Fitur Utama

- 📄 **Ekstraksi Tabel Otomatis** — Deteksi otomatis PDF digital vs scanned, tabel bergaris maupun tanpa garis
- 🖼️ **Multi-Format Input** — PDF, PNG, JPG, JPEG, WEBP
- 📊 **Preview Interaktif** — Pratinjau tabel langsung di browser dengan navigasi tab multi-tabel
- 🔍 **Quick Search** — Pencarian cepat dalam data tabel
- 📥 **Export Fleksibel** — Download sebagai Excel (multi-sheet) atau CSV (ZIP jika multi-tabel)
- 📋 **Copy to Clipboard** — Salin tabel sebagai TSV langsung ke Excel/Google Sheets
- 🌗 **Dark/Light Mode** — Toggle tema gelap/terang
- 🔒 **Privasi** — Semua pemrosesan berjalan lokal, tanpa API pihak ketiga

## Tech Stack

| Komponen | Teknologi |
|----------|-----------|
| Backend | FastAPI + Uvicorn |
| Table Extraction | pdfplumber + img2table + Tesseract OCR |
| Data Processing | pandas + openpyxl |
| Frontend | HTML5 + Vanilla JS + Tailwind CSS + Lucide Icons |
| Container | Docker + Docker Compose |
| CI/CD | Jenkins |

## Quick Start

### Menggunakan Docker (Direkomendasikan)

```bash
docker-compose up -d --build
```

Aplikasi berjalan di: **http://localhost:3022**

### Pengembangan Lokal

```bash
# Install system dependencies (Ubuntu/Debian)
sudo apt-get install tesseract-ocr tesseract-ocr-ind poppler-utils

# Install Python dependencies
pip install -r requirements.txt

# Jalankan server
uvicorn app.main:app --reload --port 8000
```

Aplikasi berjalan di: **http://localhost:8000**

## API Endpoints

| Method | Endpoint | Deskripsi |
|--------|----------|-----------|
| `GET` | `/api/health` | Healthcheck |
| `POST` | `/api/convert` | Upload & ekstrak tabel (multipart/form-data) |
| `POST` | `/api/download` | Download hasil konversi (JSON body) |

## Struktur Project

```
table-extractor/
├── app/
│   ├── __init__.py
│   ├── main.py             # FastAPI routing & endpoint handlers
│   ├── converter.py         # Table extraction engine
│   ├── exporter.py          # Excel/CSV export engine
│   └── static/
│       ├── index.html       # Single Page App (Frontend)
│       └── favicon.svg      # App icon
├── Dockerfile
├── docker-compose.yml
├── Jenkinsfile
├── requirements.txt
├── README.md
└── PRD.md
```
