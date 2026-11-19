FROM python:3.11-slim

# Set INSTALL_WHISPER=1 to add speech-to-text (needs more RAM - use a paid Render plan)
ARG INSTALL_WHISPER=0

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    MPLCONFIGDIR=/tmp/mpl \
    TRUTHLENS_DATA_DIR=/app/data

# ffmpeg = video/audio decoding, tesseract = OCR, libsndfile = audio I/O
RUN apt-get update && apt-get install -y --no-install-recommends \
        ffmpeg tesseract-ocr tesseract-ocr-eng libsndfile1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt requirements-optional.txt ./
RUN pip install -r requirements.txt && \
    if [ "$INSTALL_WHISPER" = "1" ]; then pip install -r requirements-optional.txt; fi

COPY app ./app
COPY static ./static
COPY scripts ./scripts
RUN mkdir -p /app/data

EXPOSE 8000
# Render injects $PORT; default 8000 locally
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000} --workers 1"]
