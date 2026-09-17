FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies required by Tesseract OCR, Poppler, and OpenCV
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    tesseract-ocr \
    tesseract-ocr-ind \
    poppler-utils \
    libgl1 \
    libglib2.0-0 && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Configure pip for network resilience
ENV PIP_DEFAULT_TIMEOUT=1000 \
    PIP_RETRIES=10

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir --timeout 1000 --retries 10 -r requirements.txt

# Copy application code
COPY app/ ./app/

# Expose the application port
EXPOSE 8000

# Healthcheck
HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/api/health')" || exit 1

# Run the application
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
