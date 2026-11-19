"""Central configuration for TruthLens. Values can be overridden with environment variables."""
import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = Path(os.getenv("TRUTHLENS_DATA_DIR", BASE_DIR / "data"))
UPLOAD_DIR = DATA_DIR / "uploads"      # original files, never modified
EVIDENCE_DIR = DATA_DIR / "evidence"   # generated evidence images / audio plots
REPORT_DIR = DATA_DIR / "reports"      # exported JSON / PDF reports
DB_PATH = DATA_DIR / "truthlens.db"
STATIC_DIR = BASE_DIR / "static"

for d in (UPLOAD_DIR, EVIDENCE_DIR, REPORT_DIR):
    d.mkdir(parents=True, exist_ok=True)

MAX_UPLOAD_MB = int(os.getenv("MAX_UPLOAD_MB", "60"))
MAX_VIDEO_SECONDS = float(os.getenv("MAX_VIDEO_SECONDS", "120"))   # only analyse the first N seconds
VIDEO_SAMPLE_EVERY_SEC = float(os.getenv("VIDEO_SAMPLE_EVERY_SEC", "1.0"))
SYNC_ANALYSIS_FPS = float(os.getenv("SYNC_ANALYSIS_FPS", "12"))
AUDIO_SAMPLE_RATE = 32000   # analysis sample rate (keeps up to 16 kHz detail)

# Optional speech-to-text (faster-whisper). Disabled automatically if the package is missing.
ENABLE_WHISPER = os.getenv("ENABLE_WHISPER", "1") == "1"
WHISPER_MODEL = os.getenv("WHISPER_MODEL", "tiny")

APP_VERSION = "1.3.0"

ALLOWED_EXT = {
    "image": {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"},
    "video": {".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v"},
    "audio": {".wav", ".mp3", ".m4a", ".ogg", ".flac", ".aac"},
}

DISCLAIMER = (
    "TruthLens provides forensic indicators and evidence references. It does not provide a "
    "final legal determination of authenticity. A detection score is only an indicator, not a final verdict."
)
