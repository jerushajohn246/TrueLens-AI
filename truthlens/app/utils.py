"""Shared helpers: hashing, media type detection, FFmpeg access, evidence saving."""
import hashlib
import json
import re
import shutil
import subprocess
import uuid
from pathlib import Path

import cv2
import numpy as np

from .config import ALLOWED_EXT, EVIDENCE_DIR


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def md5_file(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# Magic-byte signatures so a renamed file is still detected correctly
_MAGIC = [
    (b"\xff\xd8\xff", "image"), (b"\x89PNG", "image"), (b"BM", "image"),
    (b"II*\x00", "image"), (b"MM\x00*", "image"),
    (b"ID3", "audio"), (b"fLaC", "audio"), (b"OggS", "audio"),
    (b"\x1aE\xdf\xa3", "video"),
]


def detect_media_type(path: Path, filename: str) -> str | None:
    ext = Path(filename).suffix.lower()
    with open(path, "rb") as f:
        head = f.read(16)
    if head[:4] == b"RIFF":
        if head[8:12] == b"WAVE":
            return "audio"
        if head[8:12] == b"WEBP":
            return "image"
        if head[8:11] == b"AVI":
            return "video"
    if head[4:8] == b"ftyp":
        return "audio" if ext in {".m4a", ".aac"} else "video"
    for sig, kind in _MAGIC:
        if head.startswith(sig):
            return kind
    if head[:2] in (b"\xff\xfb", b"\xff\xf3", b"\xff\xf2"):
        return "audio"
    for kind, exts in ALLOWED_EXT.items():
        if ext in exts:
            return kind
    return None


def ffmpeg_bin() -> str:
    """Use system ffmpeg if present, otherwise the binary bundled with imageio-ffmpeg."""
    sys_ff = shutil.which("ffmpeg")
    if sys_ff:
        return sys_ff
    import imageio_ffmpeg
    return imageio_ffmpeg.get_ffmpeg_exe()


def run_ffmpeg(args: list[str], timeout: int = 180) -> subprocess.CompletedProcess:
    return subprocess.run([ffmpeg_bin(), "-hide_banner", *args], capture_output=True,
                          text=True, timeout=timeout)


def probe_media(path: Path) -> dict:
    """Parse `ffmpeg -i` output (works without ffprobe) for container metadata."""
    res = run_ffmpeg(["-i", str(path)], timeout=60)
    txt = res.stderr
    info: dict = {"raw_tags": {}}
    m = re.search(r"Duration: (\d+):(\d+):([\d.]+)", txt)
    if m:
        info["duration_sec"] = int(m.group(1)) * 3600 + int(m.group(2)) * 60 + float(m.group(3))
    m = re.search(r"bitrate: (\d+) kb/s", txt)
    if m:
        info["bitrate_kbps"] = int(m.group(1))
    m = re.search(r"Video: (\w+).*?, (\d{2,5})x(\d{2,5})", txt)
    if m:
        info["video_codec"], info["width"], info["height"] = m.group(1), int(m.group(2)), int(m.group(3))
    m = re.search(r"([\d.]+) fps", txt)
    if m:
        info["fps"] = float(m.group(1))
    m = re.search(r"Audio: (\w+)", txt)
    info["audio_codec"] = m.group(1) if m else None
    info["has_audio"] = bool(m)
    for key in ("creation_time", "encoder", "major_brand", "com.apple.quicktime.make",
                "com.apple.quicktime.model", "location", "com.apple.quicktime.location.ISO6709",
                "handler_name", "comment", "title"):
        mm = re.search(rf"^\s*{re.escape(key)}\s*:\s*(.+)$", txt, re.MULTILINE)
        if mm:
            info["raw_tags"][key] = mm.group(1).strip()
    return info


def extract_audio_wav(src: Path, dst: Path, sr: int = 16000, max_seconds: float | None = None) -> bool:
    dst.parent.mkdir(parents=True, exist_ok=True)
    args = ["-y", "-i", str(src), "-vn", "-ac", "1", "-ar", str(sr)]
    if max_seconds:
        args += ["-t", str(max_seconds)]
    args += ["-f", "wav", str(dst)]
    res = run_ffmpeg(args)
    return res.returncode == 0 and dst.exists() and dst.stat().st_size > 1000


def new_id() -> str:
    return uuid.uuid4().hex[:12]


def evidence_path(job_id: str, name: str) -> Path:
    d = EVIDENCE_DIR / job_id
    d.mkdir(parents=True, exist_ok=True)
    return d / name


def save_evidence_image(job_id: str, name: str, img_bgr: np.ndarray, max_side: int = 1100) -> str:
    """Save an evidence image and return its public URL path."""
    h, w = img_bgr.shape[:2]
    scale = min(1.0, max_side / max(h, w))
    if scale < 1.0:
        img_bgr = cv2.resize(img_bgr, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    p = evidence_path(job_id, name)
    cv2.imwrite(str(p), img_bgr, [cv2.IMWRITE_JPEG_QUALITY, 90])
    return f"/evidence/{job_id}/{name}"


def url_to_path(url: str) -> Path:
    # "/evidence/<job>/<file>" -> EVIDENCE_DIR/<job>/<file>
    parts = url.strip("/").split("/")
    return EVIDENCE_DIR.joinpath(*parts[1:])


def fmt_ts(t: float) -> str:
    m, s = divmod(max(t, 0.0), 60)
    return f"{int(m):02d}:{s:04.1f}"


def make_finding(category: str, title: str, strength: float, confidence: float, explanation: str,
                 location: dict | None = None, evidence: list[str] | None = None,
                 check: str = "", severity: str | None = None, ctx: dict | None = None) -> dict:
    """Standard finding object. strength = how strongly the indicator is present (0..1)."""
    strength = float(np.clip(strength, 0, 1))
    confidence = float(np.clip(confidence, 0, 1))
    if severity is None:
        severity = ("high" if strength >= 0.7 else "medium" if strength >= 0.45
                    else "low" if strength >= 0.2 else "info")
    return {
        "id": new_id(), "category": category, "check": check, "title": title,
        "severity": severity, "strength": round(strength, 3), "confidence": round(confidence, 3),
        "explanation": explanation, "location": location or {}, "evidence": evidence or [],
        "review_status": "pending", "ctx": ctx or {},
    }


def to_jsonable(obj):
    return json.loads(json.dumps(obj, default=lambda o: o.item() if hasattr(o, "item") else str(o)))
