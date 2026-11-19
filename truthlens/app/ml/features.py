"""Feature extraction for the AI-generated media classifiers.

Every feature is designed to be independent of file format, resolution and bitrate as far as possible
(images are resized to a fixed working size, audio is resampled to 16 kHz mono), so the models learn
what the CONTENT looks/sounds like rather than which app saved the file.
Used by both scripts/train_ai_models.py (training) and app/ml/predict.py (inference).
"""
from __future__ import annotations

import cv2
import numpy as np
from scipy import signal

# ------------------------------------------------------------------ images / frames
IMAGE_FEATURES = [
    "grain_mean", "grain_std", "grain_kurtosis", "chroma_noise_ratio", "lap_var_log", "smooth_block_frac",
    "block_std_cv", "edge_density", "grad_kurtosis", "sat_mean", "sat_std", "sat_high_frac", "colorfulness",
    "hue_entropy", "lum_entropy", "lum_clip_frac", "spectrum_slope", "hf_energy_ratio", "local_contrast_cv",
    "dark_frac",
]


def _kurt(x):
    x = x.ravel().astype(np.float64)
    s = x.std() + 1e-9
    return float(np.mean(((x - x.mean()) / s) ** 4))


def image_features(bgr: np.ndarray, size: int = 512) -> np.ndarray:
    h, w = bgr.shape[:2]
    s = size / max(h, w)
    img = cv2.resize(bgr, (max(16, int(w * s)), max(16, int(h * s))), interpolation=cv2.INTER_AREA)
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gf = g.astype(np.float32)
    res = gf - cv2.medianBlur(g, 3).astype(np.float32)
    ycc = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb).astype(np.float32)
    cres = np.abs(ycc[..., 1] - cv2.medianBlur(ycc[..., 1].astype(np.uint8), 3)).mean() + \
        np.abs(ycc[..., 2] - cv2.medianBlur(ycc[..., 2].astype(np.uint8), 3)).mean()
    lap = cv2.Laplacian(gf, cv2.CV_32F)
    B = 16
    hb, wb = g.shape[0] // B, g.shape[1] // B
    blocks = gf[:hb * B, :wb * B].reshape(hb, B, wb, B).std(axis=(1, 3))
    gx, gy = cv2.Sobel(gf, cv2.CV_32F, 1, 0), cv2.Sobel(gf, cv2.CV_32F, 0, 1)
    gm = np.hypot(gx, gy)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    sat = hsv[..., 1].astype(np.float32) / 255
    hue = hsv[..., 0][hsv[..., 1] > 40]
    hh = np.bincount(hue.ravel() // 6, minlength=30).astype(np.float64) if hue.size else np.ones(30)
    hp = hh / hh.sum()
    b, gch, r = [img[..., i].astype(np.float32) for i in range(3)]
    rg, yb = r - gch, 0.5 * (r + gch) - b
    colorful = np.sqrt(rg.std() ** 2 + yb.std() ** 2) + 0.3 * np.sqrt(rg.mean() ** 2 + yb.mean() ** 2)
    lh = np.bincount(g.ravel(), minlength=256).astype(np.float64)
    lp = lh / lh.sum()
    # radial power spectrum slope
    sq = cv2.resize(gf, (256, 256))
    F = np.abs(np.fft.fftshift(np.fft.fft2((sq - sq.mean()) * np.outer(np.hanning(256), np.hanning(256))))) ** 2
    yy, xx = np.mgrid[:256, :256]
    rr = np.hypot(yy - 128, xx - 128).astype(int)
    radial = np.bincount(rr.ravel(), F.ravel()) / (np.bincount(rr.ravel()) + 1e-9)
    k = np.arange(4, 120)
    slope = float(np.polyfit(np.log(k), np.log(radial[k] + 1e-9), 1)[0])
    hf = float(radial[64:120].sum() / (radial[4:120].sum() + 1e-9))
    lc = cv2.GaussianBlur(np.abs(gf - cv2.GaussianBlur(gf, (0, 0), 3)), (0, 0), 8)
    f = [np.abs(res).mean(), res.std(), min(_kurt(res), 50), cres / (np.abs(res).mean() + 1e-3),
         np.log1p(lap.var()), float((blocks < 2.0).mean()), float(blocks.std() / (blocks.mean() + 1e-6)),
         float((gm > 60).mean()), min(_kurt(gm), 80), float(sat.mean()), float(sat.std()), float((sat > 0.7).mean()),
         colorful / 100, float(-(hp * np.log(hp + 1e-12)).sum()), float(-(lp * np.log(lp + 1e-12)).sum()),
         float(((g < 3) | (g > 252)).mean()), slope, hf, float(lc.std() / (lc.mean() + 1e-6)), float((g < 40).mean())]
    return np.nan_to_num(np.array(f, dtype=np.float64))


# ------------------------------------------------------------------ video clips
VIDEO_TEMPORAL = ["temporal_noise", "motion_mean", "motion_jitter_ratio", "bg_warp", "face_bg_sharp",
                  "frame_diff_mean", "frame_diff_cv", "face_present"]
VIDEO_FEATURES = [f"frame_{n}" for n in IMAGE_FEATURES] + VIDEO_TEMPORAL


def video_clip_features(frames: list[np.ndarray], faces: list) -> np.ndarray:
    """frames: BGR frames of one ~2 s clip sampled at ~6 fps; faces: face box or None per frame."""
    from ..analyzers.image_analyzer import noise_residual  # noqa: F401  (kept for parity)
    fe = np.mean([image_features(f, 384) for f in frames[::2]], axis=0)
    W = 320
    gs = []
    for f in frames:
        s = W / f.shape[1]
        gs.append(cv2.cvtColor(cv2.resize(f, (W, max(16, int(f.shape[0] * s)))), cv2.COLOR_BGR2GRAY))
    tn, mot, warp, diffs, sharp = [], [], [], [], []
    for i in range(1, len(gs)):
        a, b = gs[i - 1], gs[i]
        diffs.append(float(np.abs(a.astype(np.float32) - b.astype(np.float32)).mean()))
        pts = cv2.goodFeaturesToTrack(a, 200, 0.01, 6)
        if pts is None or len(pts) < 15:
            continue
        nxt, st, _ = cv2.calcOpticalFlowPyrLK(a, b, pts, None, winSize=(21, 21), maxLevel=3)
        st = st.ravel() == 1
        p0, p1 = pts[st].reshape(-1, 2), nxt[st].reshape(-1, 2)
        if len(p0) < 10:
            continue
        M, _ = cv2.estimateAffinePartial2D(p0, p1, ransacReprojThreshold=1.5)
        if M is None:
            continue
        pred = p0 @ M[:, :2].T + M[:, 2]
        warp.append(float(np.median(np.linalg.norm(p1 - pred, axis=1))))
        mot.append(M[:, 2].copy())
        al = cv2.warpAffine(a, M, (b.shape[1], b.shape[0]))
        d = np.abs(b.astype(np.float32) - al.astype(np.float32))
        grad = np.hypot(cv2.Sobel(b, cv2.CV_32F, 1, 0), cv2.Sobel(b, cv2.CV_32F, 0, 1))
        flat = grad < 10
        flat[:6] = flat[-6:] = False
        flat[:, :6] = flat[:, -6:] = False
        if flat.sum() > 300:
            tn.append(float(d[flat].mean()))
    for f, fb in zip(frames, faces):
        if fb is None:
            continue
        g = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
        x, y, w, h = fb
        lap = np.abs(cv2.Laplacian(g, cv2.CV_32F))
        m = np.ones(g.shape, bool)
        m[max(0, int(y - 0.7 * h)):, max(0, int(x - w)):int(x + 2 * w)] = False
        if m.sum() > 2000 and w * h > 100:
            sharp.append(float(lap[y:y + h, x:x + w].mean() / (lap[m].mean() + 1e-3)))
    T = np.array(mot) if mot else np.zeros((1, 2))
    vel = np.linalg.norm(T, axis=1)
    jit = np.linalg.norm(np.diff(T, axis=0), axis=1).mean() if len(T) > 2 else 0.0
    dd = np.array(diffs) if diffs else np.zeros(1)
    temporal = [np.mean(tn) if tn else 0.0, float(vel.mean()), float(jit / (vel.mean() + 0.3)),
                np.median(warp) if warp else 0.0, min(np.median(sharp), 8) if sharp else 1.0,
                float(dd.mean()), float(dd.std() / (dd.mean() + 1e-3)), float(np.mean([fb is not None for fb in faces]))]
    return np.nan_to_num(np.concatenate([fe, np.array(temporal, dtype=np.float64)]))


def read_video_clips(path, clip_sec: float = 2.0, fps: float = 6.0, max_sec: float = 30.0,
                     face_fn=None) -> list[tuple[list, list]]:
    cap = cv2.VideoCapture(str(path))
    vfps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    step = max(1, int(round(vfps / fps)))
    per_clip = int(clip_sec * fps)
    clips, frames, faces = [], [], []
    i = 0
    while True:
        ok = cap.grab()
        if not ok or i / vfps > max_sec:
            break
        if i % step == 0:
            ok, fr = cap.retrieve()
            if ok:
                s = 480 / max(fr.shape[:2])
                fr = cv2.resize(fr, (int(fr.shape[1] * s), int(fr.shape[0] * s)), interpolation=cv2.INTER_AREA)
                faces.append(face_fn(fr) if face_fn else None)
                frames.append(fr)
                if len(frames) == per_clip:
                    clips.append((frames, faces))
                    frames, faces = [], []
        i += 1
    cap.release()
    if len(frames) >= per_clip // 2:
        clips.append((frames, faces))
    return clips


# ------------------------------------------------------------------ audio segments
AUDIO_FEATURES = ([f"mfcc{i}_mean" for i in range(1, 14)] + [f"mfcc{i}_std" for i in range(1, 14)] +
                  [f"dmfcc{i}_std" for i in range(1, 14)] +
                  ["centroid_mean", "centroid_std", "rolloff_mean", "flatness_mean", "flatness_std", "zcr_mean",
                   "zcr_std", "flux_mean", "flux_std", "rms_cv", "silence_frac", "noise_floor_rel", "pitch_std",
                   "pitch_jitter", "voiced_frac", "hnr_mean", "low_band_ratio", "mid_band_ratio"])


def _mel_fb(sr, n_fft, n_mels=40, fmax=4000):
    def hz2mel(f):
        return 2595 * np.log10(1 + f / 700)

    def mel2hz(m):
        return 700 * (10 ** (m / 2595) - 1)
    mels = np.linspace(hz2mel(60), hz2mel(fmax), n_mels + 2)
    bins = np.floor((n_fft + 1) * mel2hz(mels) / sr).astype(int)
    fb = np.zeros((n_mels, n_fft // 2 + 1))
    for i in range(1, n_mels + 1):
        l, c, r = bins[i - 1], bins[i], bins[i + 1]
        if c > l:
            fb[i - 1, l:c] = (np.arange(l, c) - l) / (c - l)
        if r > c:
            fb[i - 1, c:r] = (r - np.arange(c, r)) / (r - c)
    return fb


_FB = {}


def audio_segment_features(y: np.ndarray, sr: int = 16000) -> np.ndarray:
    """Features of one ~3 s mono segment. Band-limited to 4 kHz so codec bandwidth (opus/mp3/aac) can't leak."""
    y = y - y.mean()
    y = y / (np.abs(y).max() + 1e-9)
    n_fft, hop = 512, 160
    f, t, Z = signal.stft(y, sr, nperseg=n_fft, noverlap=n_fft - hop, boundary=None)
    P = np.abs(Z) ** 2 + 1e-12
    keep = f <= 4000
    Pk, fk = P[keep], f[keep]
    if n_fft not in _FB:
        _FB[n_fft] = _mel_fb(sr, n_fft)[:, keep]
    M = np.log(_FB[n_fft] @ Pk + 1e-10)
    from scipy.fft import dct
    mf = dct(M, axis=0, norm="ortho")[1:14]
    dmf = np.diff(mf, axis=1)
    cen = (fk[:, None] * Pk).sum(0) / Pk.sum(0)
    cum = np.cumsum(Pk, 0) / Pk.sum(0)
    roll = fk[np.argmax(cum >= 0.85, axis=0)]
    flat = np.exp(np.log(Pk).mean(0)) / Pk.mean(0)
    frames = len(y) // hop
    yy = y[:frames * hop].reshape(frames, hop)
    rms = np.sqrt((yy ** 2).mean(1) + 1e-12)
    zcr = (np.abs(np.diff(np.sign(yy), axis=1)) > 0).mean(1)
    flux = np.sqrt((np.diff(np.log(Pk), axis=1).clip(0) ** 2).mean(0))
    db = 20 * np.log10(rms)
    loud = db > np.percentile(db, 95) - 25
    noise_floor = np.percentile(db, 10) - np.percentile(db, 95)
    # pitch via autocorrelation on voiced frames
    win = 640
    lo, hi = sr // 400, sr // 70
    pit, hnr = [], []
    for s0 in range(0, len(y) - win, 320):
        fr = y[s0:s0 + win]
        if np.sqrt((fr ** 2).mean()) < 0.05:
            continue
        ac = np.correlate(fr, fr, "full")[win - 1:]
        if ac[0] <= 0:
            continue
        ac /= ac[0]
        seg = ac[lo:hi]
        kk = int(np.argmax(seg))
        if seg[kk] > 0.4:
            pit.append(sr / (lo + kk))
            hnr.append(10 * np.log10(seg[kk] / (1 - seg[kk] + 1e-6)))
    pit = np.array(pit)
    if len(pit) > 5:
        semis = 12 * np.log2(pit / np.median(pit))
        pstd, pj = float(semis.std()), float(np.median(np.abs(np.diff(pit)) / pit[:-1]))
    else:
        pstd, pj = 0.0, 0.0
    band = Pk.sum(1)
    tot = band.sum()
    feats = list(mf.mean(1)) + list(mf.std(1)) + list(dmf.std(1)) + [
        cen.mean() / 1000, cen.std() / 1000, roll.mean() / 1000, flat.mean(), flat.std(), zcr.mean(), zcr.std(),
        flux.mean(), flux.std(), rms.std() / (rms.mean() + 1e-9), float(1 - loud.mean()), noise_floor / 10, pstd,
        pj * 10, len(pit) / max(1, (len(y) - win) // 320), float(np.mean(hnr)) / 10 if hnr else 0.0,
        float(band[fk < 500].sum() / tot), float(band[(fk >= 500) & (fk < 2000)].sum() / tot)]
    return np.nan_to_num(np.array(feats, dtype=np.float64))


def audio_segments(y: np.ndarray, sr: int = 16000, seg_sec: float = 3.0, hop_sec: float = 1.5) -> list[np.ndarray]:
    n, h = int(seg_sec * sr), int(hop_sec * sr)
    segs = []
    for s0 in range(0, max(1, len(y) - n + 1), h):
        seg = y[s0:s0 + n]
        if len(seg) == n and np.sqrt((seg ** 2).mean()) > 1e-3:
            segs.append(seg)
    if not segs and len(y) > sr:
        segs.append(y)
    return segs
