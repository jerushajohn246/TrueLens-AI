"""Image authenticity analysis.

Checks (all explainable, classic forensics - no black-box verdict):
  1. Metadata / EXIF + AI-generator tags
  2. Dimensions & JPEG compression quality
  3. Error Level Analysis (ELA)  -> regions with different compression history
  4. Noise-residual inconsistency -> spliced regions often carry different sensor noise
  5. Copy-move detection (ORB keypoint self-matching) -> duplicated / cloned areas
  6. Face-region consistency (lighting, sharpness, noise, compression vs surroundings)
  7. Frequency-spectrum artifacts -> periodic peaks typical of GAN/diffusion upsampling
"""
from __future__ import annotations

import io
from pathlib import Path

import cv2
import numpy as np
from PIL import ExifTags, Image

from ..utils import make_finding, save_evidence_image

AI_TAG_WORDS = ["stable diffusion", "midjourney", "dall-e", "dall·e", "dalle", "novelai", "comfyui",
                "automatic1111", "firefly", "imagen", "leonardo", "sdxl", "flux", "generated", "ai generated",
                "c2pa", "openai"]
EDITOR_WORDS = ["photoshop", "gimp", "lightroom", "snapseed", "picsart", "canva", "affinity", "pixlr",
                "facetune", "meitu", "paint.net", "fotor", "remini"]

_FACE_CASCADE = None
_YUNET = None
YUNET_PATH = Path(__file__).resolve().parent.parent / "models" / "face_detection_yunet_2023mar.onnx"


def face_detector():
    global _FACE_CASCADE
    if _FACE_CASCADE is None:
        _FACE_CASCADE = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    return _FACE_CASCADE


def _yunet():
    global _YUNET
    if _YUNET is None and YUNET_PATH.exists() and hasattr(cv2, "FaceDetectorYN"):
        try:
            _YUNET = cv2.FaceDetectorYN.create(str(YUNET_PATH), "", (320, 320), 0.7, 0.3, 50)
        except Exception:
            _YUNET = False
    return _YUNET or None


def detect_faces_full(img_bgr: np.ndarray, min_size_ratio: float = 0.06) -> list[dict]:
    """Detect faces. Uses the YuNet CNN detector (bundled, with landmarks); falls back to Haar cascade.
    Returns dicts: {"bbox": (x, y, w, h), "mouth": ((x1, y1), (x2, y2)) | None, "score": float}."""
    h, w = img_bgr.shape[:2]
    ms = max(20, int(min(h, w) * min_size_ratio))
    out = []
    det = _yunet()
    if det is not None:
        det.setInputSize((w, h))
        _, faces = det.detect(img_bgr)
        if faces is not None:
            for f in faces:
                x, y, fw, fh = [int(round(v)) for v in f[:4]]
                x, y = max(0, x), max(0, y)
                fw, fh = min(fw, w - x), min(fh, h - y)
                if fw >= ms and fh >= ms:
                    out.append({"bbox": (x, y, fw, fh), "score": float(f[14]),
                                "mouth": ((float(f[10]), float(f[11])), (float(f[12]), float(f[13])))})
    else:
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY) if img_bgr.ndim == 3 else img_bgr
        for f in face_detector().detectMultiScale(gray, scaleFactor=1.1, minNeighbors=6, minSize=(ms, ms)):
            out.append({"bbox": tuple(int(v) for v in f), "score": 0.8, "mouth": None})
    out.sort(key=lambda d: -d["bbox"][2] * d["bbox"][3])
    return out[:5]


def detect_faces(img_bgr: np.ndarray, min_size_ratio: float = 0.06) -> list[tuple[int, int, int, int]]:
    return [d["bbox"] for d in detect_faces_full(img_bgr, min_size_ratio)]


# ---------------------------------------------------------------- metadata
def extract_metadata(path: Path) -> dict:
    meta: dict = {"exif": {}, "info": {}}
    with Image.open(path) as im:
        meta["format"] = im.format
        meta["mode"] = im.mode
        meta["width"], meta["height"] = im.size
        try:
            exif = im.getexif()
            for k, v in exif.items():
                name = ExifTags.TAGS.get(k, str(k))
                meta["exif"][name] = _clean(v)
            try:  # sub-IFD (DateTimeOriginal etc.)
                for k, v in exif.get_ifd(0x8769).items():
                    meta["exif"][ExifTags.TAGS.get(k, str(k))] = _clean(v)
            except Exception:
                pass
        except Exception:
            pass
        for k, v in (im.info or {}).items():
            if k in ("exif", "icc_profile", "xmp"):
                meta["info"][k] = f"<{len(v) if hasattr(v, '__len__') else 0} bytes>"
            else:
                meta["info"][k] = _clean(v)
        if im.format == "JPEG":
            try:
                q = im.quantization
                meta["jpeg_quality_estimate"] = estimate_jpeg_quality(q)
                meta["jpeg_quant_tables"] = len(q)
            except Exception:
                pass
    return meta


def _clean(v):
    if isinstance(v, bytes):
        try:
            v = v.decode("utf-8", "ignore")
        except Exception:
            return f"<{len(v)} bytes>"
    s = str(v)
    return s[:300]


def estimate_jpeg_quality(qtables: dict) -> int | None:
    """Estimate libjpeg quality from the luminance quantisation table."""
    if not qtables:
        return None
    std_lum = np.array([16, 11, 10, 16, 24, 40, 51, 61, 12, 12, 14, 19, 26, 58, 60, 55, 14, 13, 16, 24, 40, 57, 69, 56,
                        14, 17, 22, 29, 51, 87, 80, 62, 18, 22, 37, 56, 68, 109, 103, 77, 24, 35, 55, 64, 81, 104, 113,
                        92, 49, 64, 78, 87, 103, 121, 120, 101, 72, 92, 95, 98, 112, 100, 103, 99], dtype=float)
    t = np.array(list(qtables[0]), dtype=float)[:64]
    if t.size < 64:
        return None
    best, best_err = None, 1e18
    for q in range(1, 101):
        scale = 5000 / q if q < 50 else 200 - 2 * q
        est = np.clip(np.floor((std_lum * scale + 50) / 100), 1, 255)
        err = np.abs(np.sort(est) - np.sort(t)).sum()
        if err < best_err:
            best, best_err = q, err
    return int(best)


def metadata_findings(meta: dict, path: Path) -> list[dict]:
    out = []
    exif = meta.get("exif", {})
    info_text = " ".join(f"{k} {v}" for k, v in {**meta.get("info", {}), **exif}.items()).lower()
    software = str(exif.get("Software", "")).strip()

    ai_hits = [w for w in AI_TAG_WORDS if w in info_text]
    # PNG "parameters" chunk is written by Stable Diffusion UIs
    if "parameters" in meta.get("info", {}) or ai_hits:
        out.append(make_finding(
            "metadata", "AI-generation tag found in metadata", 0.85, 0.85,
            f"The file's embedded metadata contains generator-related text ({', '.join(ai_hits) or 'parameters chunk'}). "
            "This strongly suggests the image was produced or processed by an AI image tool.",
            {"type": "metadata", "field": "PNG text / EXIF"}, check="metadata.ai_tags"))

    ed_hits = [w for w in EDITOR_WORDS if w in info_text]
    if ed_hits:
        out.append(make_finding(
            "metadata", "Editing software recorded in metadata", 0.45, 0.8,
            f"Metadata records editing software ({software or ', '.join(ed_hits)}). Editing is not proof of "
            "deception (cropping and colour correction are common), but the file is not a camera original.",
            {"type": "metadata", "field": "Software"}, check="metadata.software"))

    has_camera = bool(exif.get("Make") or exif.get("Model"))
    if not exif:
        out.append(make_finding(
            "metadata", "No EXIF metadata present", 0.3, 0.5,
            "The image contains no camera metadata. Social-media platforms and screenshots strip metadata, so this "
            "is common for genuine images too - but it removes the ability to verify source, device and capture time.",
            {"type": "metadata", "field": "EXIF"}, check="metadata.missing"))
    elif not has_camera:
        out.append(make_finding(
            "metadata", "Camera make/model missing", 0.15, 0.4,
            "EXIF exists but no camera make/model is recorded, so the capture device cannot be verified.",
            {"type": "metadata", "field": "Make/Model"}, check="metadata.no_camera"))

    dto, dtm = exif.get("DateTimeOriginal"), exif.get("DateTime")
    if dto and dtm and dto != dtm:
        out.append(make_finding(
            "metadata", "Capture time and modification time differ", 0.3, 0.6,
            f"DateTimeOriginal = {dto} but DateTime (last modified) = {dtm}. The file was saved again after capture.",
            {"type": "metadata", "field": "DateTime"}, check="metadata.dates"))

    w, h = meta.get("width", 0), meta.get("height", 0)
    if w and h and w % 64 == 0 and h % 64 == 0 and max(w, h) <= 2048 and not has_camera \
            and (w, h) in {(512, 512), (768, 768), (1024, 1024), (512, 768), (768, 512), (1024, 1536),
                           (1536, 1024), (896, 1152), (1152, 896), (832, 1216), (1216, 832), (1344, 768),
                           (768, 1344), (2048, 2048)}:
        out.append(make_finding(
            "image", "Dimensions typical of AI image generators", 0.3, 0.35,
            f"The image is {w}x{h} px, a resolution commonly produced by diffusion models, and has no camera data. "
            "Weak indicator on its own.", {"type": "metadata", "field": "dimensions"}, check="image.ai_dimensions"))

    q = meta.get("jpeg_quality_estimate")
    if q is not None and q < 70:
        out.append(make_finding(
            "image", f"Low JPEG quality (~{q})", 0.15, 0.5,
            "Heavy compression hides fine forensic traces, which lowers the reliability of pixel-level checks.",
            {"type": "metadata", "field": "JPEG quantisation"}, check="image.low_quality"))
    return out


# ---------------------------------------------------------------- pixel checks
def ela_map(img_bgr: np.ndarray, quality: int = 90) -> np.ndarray:
    """Error Level Analysis: difference between the image and a re-compressed copy (grayscale float)."""
    # computed on luminance only: chroma subsampling makes saturated colours light up in RGB-ELA
    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    buf = io.BytesIO()
    Image.fromarray(rgb).save(buf, "JPEG", quality=quality)
    re = np.array(Image.open(io.BytesIO(buf.getvalue())).convert("RGB"))
    y0 = cv2.cvtColor(rgb, cv2.COLOR_RGB2YCrCb)[..., 0].astype(np.float32)
    y1 = cv2.cvtColor(re, cv2.COLOR_RGB2YCrCb)[..., 0].astype(np.float32)
    return np.abs(y0 - y1) * 4.0


def noise_residual(gray: np.ndarray) -> np.ndarray:
    g = gray.astype(np.float32)
    return g - cv2.medianBlur(gray, 3).astype(np.float32)


def block_stat(arr: np.ndarray, block: int, fn=np.mean) -> np.ndarray:
    h, w = arr.shape
    hb, wb = h // block, w // block
    a = arr[:hb * block, :wb * block].reshape(hb, block, wb, block)
    return fn(fn(a, axis=3), axis=1) if fn is np.mean else a.std(axis=(1, 3))


def outlier_region(grid: np.ndarray, block: int, z_thr: float = 3.0, min_blocks: int = 3, side: str = "high",
                   valid: np.ndarray | None = None):
    """Return (ratio_of_outlier_blocks, bbox_in_pixels or None, z_grid). side = high | low | both."""
    vals = grid[valid] if valid is not None and valid.any() else grid
    med = np.median(vals)
    mad = np.median(np.abs(vals - med)) * 1.4826 + 1e-6
    z = (grid - med) / mad
    zz = z if side == "high" else -z if side == "low" else np.abs(z)
    m = zz > z_thr
    if valid is not None:
        m &= valid
    mask = m.astype(np.uint8)
    n = int(mask.sum())
    if n < min_blocks:
        return n / grid.size, None, z
    num, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if num <= 1:
        return n / grid.size, None, z
    i = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    x, y, bw, bh, area = stats[i]
    if area < min_blocks:
        return n / grid.size, None, z
    return n / grid.size, (int(x * block), int(y * block), int(bw * block), int(bh * block)), z


def robust_residual(y: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Residual of y after a robust (Huber-like, iteratively re-weighted) linear fit y ~ a + b*x."""
    X = np.stack([np.ones(x.size), x.ravel()], 1)
    Y = y.ravel()
    w = np.ones_like(Y)
    for _ in range(5):
        coef, *_ = np.linalg.lstsq(X * w[:, None], Y * w, rcond=None)
        r = Y - X @ coef
        s = np.median(np.abs(r)) * 1.4826 + 1e-6
        w = 1 / np.maximum(1, np.abs(r) / (1.5 * s))
    return (Y - X @ coef).reshape(y.shape)


def heatmap_overlay(img_bgr: np.ndarray, m: np.ndarray, alpha: float = 0.55) -> np.ndarray:
    m = m.astype(np.float32)
    hi = np.percentile(m, 99.5) + 1e-6
    norm = np.clip(m / hi * 255, 0, 255).astype(np.uint8)
    norm = cv2.resize(norm, (img_bgr.shape[1], img_bgr.shape[0]), interpolation=cv2.INTER_NEAREST)
    col = cv2.applyColorMap(norm, cv2.COLORMAP_JET)
    return cv2.addWeighted(img_bgr, 1 - alpha, col, alpha, 0)


def copy_move_detect(gray: np.ndarray):
    """Find clusters of keypoints that match other keypoints in the same image with a shared offset."""
    h, w = gray.shape
    orb = cv2.ORB_create(nfeatures=4000, fastThreshold=12)
    kps, des = orb.detectAndCompute(gray, None)
    if des is None or len(kps) < 30:
        return [], 0
    bf = cv2.BFMatcher(cv2.NORM_HAMMING)
    matches = bf.knnMatch(des, des, k=3)
    min_dist = max(20, 0.04 * min(h, w))
    pairs = []
    for ms in matches:
        for m in ms[1:]:  # ms[0] is the self-match
            if m.distance > 28:
                continue
            p1, p2 = np.array(kps[m.queryIdx].pt), np.array(kps[m.trainIdx].pt)
            if np.linalg.norm(p1 - p2) < min_dist:
                continue
            if p1[0] > p2[0] or (p1[0] == p2[0] and p1[1] > p2[1]):
                p1, p2 = p2, p1
            pairs.append((p1, p2))
            break
    if len(pairs) < 8:
        return [], len(pairs)
    # cluster by displacement vector (quantised)
    q = max(6, int(0.01 * max(h, w)))
    bins: dict = {}
    for p1, p2 in pairs:
        d = p2 - p1
        key = (int(round(d[0] / q)), int(round(d[1] / q)))
        bins.setdefault(key, []).append((p1, p2))
    best = max(bins.values(), key=len)
    # de-duplicate the same source point
    uniq = {(int(a[0]) // 4, int(a[1]) // 4): (a, b) for a, b in best}
    best = list(uniq.values())
    if len(best) < 12:
        return [], len(pairs)
    src = np.array([a for a, _ in best])
    ext = src.max(axis=0) - src.min(axis=0)
    if min(ext) < 0.06 * min(h, w):   # tiny clusters (e.g. two similar eyes, logo letters) are ignored
        return [], len(pairs)
    dst = np.array([b for _, b in best])
    s0, s1, d0, d1 = src.min(0), src.max(0), dst.min(0), dst.max(0)
    ix = max(0, min(s1[0], d1[0]) - max(s0[0], d0[0]))
    iy = max(0, min(s1[1], d1[1]) - max(s0[1], d0[1]))
    if ix * iy > 0.1 * np.prod(np.maximum(s1 - s0, 1)):   # source and copy overlap -> periodic texture
        return [], len(pairs)
    return best, len(pairs)


def face_consistency(img_bgr: np.ndarray, gray: np.ndarray, face, ela: np.ndarray | None = None) -> dict:
    """Compare a face box with a ring around it (lighting, sharpness, noise, compression)."""
    x, y, fw, fh = face
    H, W = gray.shape
    pad = int(0.35 * max(fw, fh))
    X0, Y0, X1, Y1 = max(0, x - pad), max(0, y - pad), min(W, x + fw + pad), min(H, y + fh + pad)
    inner = np.zeros(gray.shape, bool)
    # use central 80% of face box (avoid hair / background inside the box)
    ix, iy = x + int(0.1 * fw), y + int(0.1 * fh)
    inner[iy:y + int(0.9 * fh), ix:x + int(0.9 * fw)] = True
    ring = np.zeros(gray.shape, bool)
    ring[Y0:Y1, X0:X1] = True
    ring[y:y + fh, x:x + fw] = False
    if ring.sum() < 200 or inner.sum() < 200:
        return {}
    lap = np.abs(cv2.Laplacian(gray, cv2.CV_32F))
    res = np.abs(noise_residual(gray))
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    v = hsv[..., 2].astype(np.float32)

    def ratio(a, b):
        return float((a + 1e-3) / (b + 1e-3))

    out = {
        "sharpness_ratio": ratio(lap[inner].mean(), lap[ring].mean()),
        "noise_ratio": ratio(res[inner].mean(), res[ring].mean()),
        "brightness_diff": float(v[inner].mean() - v[ring].mean()),
    }
    if ela is not None:
        out["ela_ratio"] = ratio(ela[inner].mean(), ela[ring].mean())
    # boundary seam: gradient energy exactly on the face-box edge vs just inside
    edge = np.zeros(gray.shape, bool)
    t = max(2, fw // 40)
    edge[y:y + fh, x:x + fw] = True
    edge[y + t:y + fh - t, x + t:x + fw - t] = False
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1)
    gm = np.hypot(gx, gy)
    out["boundary_ratio"] = ratio(gm[edge].mean(), gm[inner].mean())
    return out


def face_anomaly_strength(fc: dict) -> tuple[float, list[str]]:
    reasons, s = [], 0.0
    if not fc:
        return 0.0, reasons
    sr, nr = fc["sharpness_ratio"], fc["noise_ratio"]
    if sr < 0.45 or sr > 2.6:
        s += 0.35
        reasons.append(f"face is {'much smoother' if sr < 1 else 'much sharper'} than the surrounding area "
                       f"(sharpness ratio {sr:.2f})")
    if nr < 0.5 or nr > 2.2:
        s += 0.3
        reasons.append(f"noise level on the face differs from nearby regions (noise ratio {nr:.2f})")
    er = fc.get("ela_ratio")
    if er is not None and (er > 2.0 or er < 0.5):
        s += 0.3
        reasons.append(f"compression error level on the face differs from surroundings (ELA ratio {er:.2f})")
    if abs(fc["brightness_diff"]) > 70:
        s += 0.15
        reasons.append(f"strong lighting difference between face and surroundings ({fc['brightness_diff']:+.0f} levels)")
    if fc["boundary_ratio"] > 2.2:
        s += 0.2
        reasons.append(f"a visible seam/edge follows the face boundary (boundary ratio {fc['boundary_ratio']:.2f})")
    return min(s, 1.0), reasons


def spectrum_artifacts(gray: np.ndarray):
    """Look for isolated periodic peaks in the high-frequency spectrum (upsampling grid artefacts)."""
    g = gray.astype(np.float32)
    side = 512
    g = cv2.resize(g, (side, side), interpolation=cv2.INTER_AREA) if max(g.shape) > side else \
        cv2.resize(g, (side, side), interpolation=cv2.INTER_CUBIC)
    g = g - cv2.GaussianBlur(g, (0, 0), 3)
    win = np.outer(np.hanning(side), np.hanning(side))
    F = np.fft.fftshift(np.abs(np.fft.fft2(g * win)))
    logF = np.log1p(F)
    yy, xx = np.mgrid[:side, :side]
    r = np.hypot(yy - side / 2, xx - side / 2)
    hf = (r > side * 0.18) & (r < side * 0.48)
    # remove the axis cross (caused by image borders, not by generation)
    cross = (np.abs(yy - side / 2) < 3) | (np.abs(xx - side / 2) < 3)
    region = hf & ~cross
    local = cv2.medianBlur(logF.astype(np.float32), 5)
    peak = (logF - local)[region]
    score = float(np.percentile(peak, 99.97) / (peak.std() + 1e-6))
    vis = cv2.normalize(logF, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    return score, cv2.applyColorMap(vis, cv2.COLORMAP_INFERNO)


# ---------------------------------------------------------------- main entry
def analyze_image(path: Path, job_id: str, progress=lambda *_: None) -> dict:
    findings: list[dict] = []
    evidence: dict = {}
    meta = extract_metadata(path)
    findings += metadata_findings(meta, path)
    progress("Image metadata extracted")

    pil = Image.open(path).convert("RGB")
    img = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)
    H0, W0 = img.shape[:2]
    scale = min(1.0, 1600 / max(H0, W0))
    if scale < 1.0:
        img = cv2.resize(img, (int(W0 * scale), int(H0 * scale)), interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    H, W = gray.shape
    overlay = img.copy()
    evidence["original"] = save_evidence_image(job_id, "image_original.jpg", img)

    # --- ELA (edge-aware: high ELA on sharp edges is normal, so model ELA as a function of local
    #     gradient energy and flag blocks whose ELA is far from what their texture predicts)
    ela = ela_map(img)
    block = max(8, min(H, W) // 48)
    ela_grid = block_stat(ela, block)
    grad = np.hypot(cv2.Sobel(gray, cv2.CV_32F, 1, 0), cv2.Sobel(gray, cv2.CV_32F, 0, 1))
    grad_grid = block_stat(grad, block)
    ela_resid = robust_residual(ela_grid, grad_grid)
    ratio, bbox, _ = outlier_region(np.abs(ela_resid), block, z_thr=4.0, min_blocks=4)
    ela_vis = heatmap_overlay(img, ela)
    evidence["ela"] = save_evidence_image(job_id, "image_ela.jpg", ela_vis)
    if bbox and 0.004 < ratio < 0.3:
        s = float(np.clip(ratio * 5 + 0.3, 0, 0.8))
        findings.append(make_finding(
            "image", "Region with different compression history (ELA)", s, 0.55,
            f"Error Level Analysis shows a localised area ({ratio * 100:.1f}% of the image) whose re-compression "
            "error does not match what its texture would predict. Pasted or retouched regions often keep a "
            "different compression level. Verify the highlighted region visually.",
            {"type": "region", "bbox": _scale_box(bbox, scale), "label": "ELA"}, [evidence["ela"]], "image.ela",
            ctx={"edge_ratio": _region_ratio(grad, bbox)}))
        _box(overlay, bbox, (0, 0, 255), "ELA")
    progress("Error level analysis done")

    # --- noise inconsistency
    res = np.abs(noise_residual(gray))
    nblock = max(16, min(H, W) // 32)
    n_grid = block_stat(res, nblock)
    # flat areas (sky, walls) are naturally clean after JPEG -> only compare blocks that contain detail
    textured = block_stat(np.abs(cv2.Laplacian(cv2.GaussianBlur(gray, (5, 5), 0), cv2.CV_32F)), nblock) > 1.5
    sat_ok = block_stat(((gray > 8) & (gray < 247)).astype(np.float32), nblock) > 0.9
    smooth_mask = textured & sat_ok
    nratio, nbbox, _ = outlier_region(n_grid, nblock, z_thr=3.5, min_blocks=4, side="both", valid=smooth_mask)
    noise_vis = heatmap_overlay(img, cv2.resize(n_grid, (W, H), interpolation=cv2.INTER_NEAREST))
    evidence["noise"] = save_evidence_image(job_id, "image_noise.jpg", noise_vis)
    cv = float(n_grid[smooth_mask].std() / (n_grid[smooth_mask].mean() + 1e-6)) if smooth_mask.any() else 0
    if nbbox and 0.005 < nratio < 0.3:
        s = float(np.clip(nratio * 5 + 0.2, 0, 0.75))
        findings.append(make_finding(
            "image", "Inconsistent noise pattern", s, 0.5,
            f"The sensor-noise residual in one area differs strongly from the rest of the image (noise variation "
            f"coefficient {cv:.2f}). Content inserted from another photo - or a region smoothed by face-swap / "
            "beauty filters - carries a different noise signature.",
            {"type": "region", "bbox": _scale_box(nbbox, scale), "label": "Noise"}, [evidence["noise"]],
            "image.noise", ctx={"edge_ratio": _region_ratio(np.hypot(cv2.Sobel(gray, cv2.CV_32F, 1, 0),
                                                                     cv2.Sobel(gray, cv2.CV_32F, 0, 1)), nbbox),
                                "region_noise_ratio": _region_ratio(res, nbbox)}))
        _box(overlay, nbbox, (0, 165, 255), "Noise")
    progress("Noise analysis done")

    # --- copy-move
    cm_pairs, total_pairs = copy_move_detect(gray)
    if cm_pairs:
        vis = img.copy()
        for a, b in cm_pairs[:150]:
            cv2.line(vis, tuple(int(v) for v in a), tuple(int(v) for v in b), (0, 255, 255), 1, cv2.LINE_AA)
            cv2.circle(vis, tuple(int(v) for v in a), 3, (0, 0, 255), -1)
            cv2.circle(vis, tuple(int(v) for v in b), 3, (255, 0, 0), -1)
        evidence["copy_move"] = save_evidence_image(job_id, "image_copymove.jpg", vis)
        pts = np.array([p for pr in cm_pairs for p in pr])
        x0, y0 = pts.min(axis=0)
        x1, y1 = pts.max(axis=0)
        s = float(np.clip(0.35 + len(cm_pairs) / 120, 0, 0.9))
        src = np.array([a for a, _ in cm_pairs])
        dst = np.array([b for _, b in cm_pairs])
        for grp, col in ((src, (255, 0, 255)), (dst, (255, 0, 255))):
            gx0, gy0 = grp.min(axis=0)
            gx1, gy1 = grp.max(axis=0)
            _box(overlay, (int(gx0), int(gy0), int(gx1 - gx0) + 1, int(gy1 - gy0) + 1), col, "Clone")
        findings.append(make_finding(
            "image", "Duplicated (copy-move) regions detected", s, 0.7,
            f"{len(cm_pairs)} matching feature points share the same displacement, meaning one part of the image "
            "appears to be copied to another location. This is a classic sign of cloning to hide or duplicate "
            "objects. Repetitive patterns (tiles, fences, text) can trigger it falsely.",
            {"type": "region", "bbox": _scale_box((int(x0), int(y0), int(x1 - x0), int(y1 - y0)), scale),
             "label": "Copy-move"}, [evidence["copy_move"]], "image.copy_move",
            ctx={"pairs": len(cm_pairs), "candidate_pairs": total_pairs}))
    progress("Copy-move check done")

    # --- faces
    faces = detect_faces(img)
    face_reports = []
    for i, f in enumerate(faces):
        fc = face_consistency(img, gray, f, ela)
        s, reasons = face_anomaly_strength(fc)
        face_reports.append({"bbox": _scale_box(f, scale), "metrics": {k: round(v, 3) for k, v in fc.items()}})
        color = (0, 0, 255) if s >= 0.3 else (0, 200, 0)
        _box(overlay, f, color, f"Face {i + 1}")
        if s >= 0.3:
            x, y, fw, fh = f
            pad = int(0.4 * fw)
            crop = img[max(0, y - pad):y + fh + pad, max(0, x - pad):x + fw + pad].copy()
            ev = save_evidence_image(job_id, f"image_face{i + 1}.jpg", crop, 500)
            findings.append(make_finding(
                "image", f"Face {i + 1}: possible face manipulation", s, 0.55,
                "Possible face manipulation: " + "; ".join(reasons) + ". Face-swaps and retouching often leave the "
                "face with lighting, texture or compression that does not match the rest of the photo.",
                {"type": "region", "bbox": _scale_box(f, scale), "label": f"Face {i + 1}"}, [ev], "image.face",
                ctx={"global_noise": float(np.median(res)), "noise_ratio": fc.get("noise_ratio"),
                     "brightness_diff": fc.get("brightness_diff")}))
    progress("Face analysis done")

    # --- spectrum
    spec_score, spec_vis = spectrum_artifacts(gray)
    evidence["spectrum"] = save_evidence_image(job_id, "image_spectrum.jpg", spec_vis, 512)
    if spec_score > 9.0:
        s = float(np.clip((spec_score - 9) / 10 + 0.3, 0, 0.7))
        findings.append(make_finding(
            "image", "Periodic frequency artifacts (possible AI generation / resampling)", s, 0.4,
            f"The frequency spectrum contains isolated periodic peaks (peak score {spec_score:.1f}). GAN and "
            "diffusion up-sampling layers, as well as resizing, leave such grid patterns. Low-confidence indicator.",
            {"type": "region", "label": "whole image"}, [evidence["spectrum"]], "image.spectrum"))

    evidence["overlay"] = save_evidence_image(job_id, "image_overlay.jpg", overlay)
    return {
        "media_type": "image",
        "info": {"width": W0, "height": H0, "format": meta.get("format"), "mode": meta.get("mode"),
                 "jpeg_quality_estimate": meta.get("jpeg_quality_estimate"), "faces_detected": len(faces),
                 "copy_move_candidate_pairs": total_pairs, "spectrum_peak_score": round(spec_score, 2)},
        "metadata": meta,
        "faces": face_reports,
        "evidence": evidence,
        "findings": findings,
        "checks_run": ["EXIF/metadata", "AI-tag scan", "JPEG quality", "Error Level Analysis",
                       "Noise residual consistency", "Copy-move (ORB)", "Face detection (YuNet CNN)", "Face consistency", "Frequency spectrum"],
    }


def _region_ratio(arr, b):
    """Mean of arr inside bbox divided by the global mean (e.g. how edge-heavy a flagged region is)."""
    x, y, w, h = [int(v) for v in b]
    reg = arr[y:y + h, x:x + w]
    return round(float((reg.mean() + 1e-6) / (arr.mean() + 1e-6)), 3) if reg.size else 1.0


def _scale_box(b, scale):
    if scale == 1.0:
        return [int(v) for v in b]
    return [int(v / scale) for v in b]


def _box(img, b, color, label):
    x, y, w, h = [int(v) for v in b]
    cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
    cv2.putText(img, label, (x + 3, max(14, y - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2, cv2.LINE_AA)
