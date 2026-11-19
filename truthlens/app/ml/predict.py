"""Inference for the trained AI-generated media classifiers (NumPy + OpenCV only)."""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

import numpy as np

from . import features as F

MODELS = Path(__file__).resolve().parent.parent / "models"
LABEL = {"image": "Trained ML model (EfficientNet-Lite4 features + logistic regression)",
         "video": "Trained ML model (forensic frame & motion features + logistic regression)",
         "audio": "Trained ML model (voice-behaviour features + logistic regression)"}


@lru_cache(maxsize=None)
def load(kind: str) -> dict | None:
    p = MODELS / f"ai_{kind}.json"
    if not p.exists():
        return None
    m = json.loads(p.read_text())
    for k in ("mean", "scale", "coef"):
        m[k] = np.array(m[k], dtype=np.float64)
    return m


def _prob(m, X):
    Z = (np.atleast_2d(X) - m["mean"]) / m["scale"]
    return 1 / (1 + np.exp(-(Z @ m["coef"] + m["intercept"]))), Z


def _explain(m, Z):
    """Top features pushing towards AI / real (only meaningful for hand-crafted features)."""
    if m.get("feature_type") != "handcrafted":
        return []
    contrib = (Z.mean(0) * m["coef"])
    order = np.argsort(-np.abs(contrib))[:4]
    return [{"feature": m["features"][i], "direction": "AI" if contrib[i] > 0 else "real",
             "weight": round(float(contrib[i]), 2)} for i in order]


def _result(kind, m, p, Z, n):
    return {"kind": kind, "p_ai": round(float(p), 3), "samples": n, "model": LABEL[kind],
            "trained_on": m.get("n_files"), "heldout_accuracy": m.get("loo_accuracy"),
            "trained_at": m.get("trained_at"), "explain": _explain(m, Z)}


def predict_image(bgr) -> dict | None:
    m = load("image")
    if m is None:
        return None
    from . import embed
    if m.get("feature_type") == "effnet_lite4":
        if not embed.available():
            return None
        x = embed.embed(bgr)
    else:
        x = F.image_features(bgr)
    p, Z = _prob(m, x)
    return _result("image", m, p[0], Z, 1)


def predict_video(path, face_fn=None) -> dict | None:
    m = load("video")
    if m is None:
        return None
    clips = F.read_video_clips(path, face_fn=face_fn)
    if not clips:
        return None
    X = np.array([F.video_clip_features(fr, fa) for fr, fa in clips])
    p, Z = _prob(m, X)
    r = _result("video", m, p.mean(), Z, len(X))
    r["clip_probs"] = [round(float(v), 3) for v in p]
    return r


def predict_audio(wav_path) -> dict | None:
    m = load("audio")
    if m is None:
        return None
    import soundfile as sf
    from scipy.signal import resample_poly
    y, sr = sf.read(str(wav_path), dtype="float32", always_2d=False)
    if y.ndim > 1:
        y = y.mean(1)
    if sr != 16000:
        from math import gcd
        g = gcd(int(sr), 16000)
        y = resample_poly(y, 16000 // g, int(sr) // g).astype(np.float32)
    segs = F.audio_segments(y[:16000 * 180])
    if not segs:
        return None
    full = np.array([F.audio_segment_features(s) for s in segs])
    idx = [F.AUDIO_FEATURES.index(n) for n in m["features"]]
    p, Z = _prob(m, full[:, idx])
    r = _result("audio", m, p.mean(), Z, len(segs))
    return r
