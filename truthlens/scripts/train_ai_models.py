"""Train the TruthLens AI-generated media classifiers (image, video, audio).

Put labelled examples in this layout (any file names):

    training_data/
        images/ai/   images/real/
        videos/ai/   videos/real/
        audio/ai/    audio/real/

Then run:   python scripts/train_ai_models.py            (or --data <folder>)

What it does
  * Augmentation so the model learns CONTENT, not file format:
      images -> WhatsApp-style resize + JPEG recompression, crop, flip
      videos -> original + WhatsApp-style re-encode (480 px, low bitrate, metadata stripped), cut into 2 s clips
      audio  -> re-encoded through mp3 / opus / aac, cut into 3 s segments, band-limited to 4 kHz
  * Features: images -> frozen pre-trained EfficientNet-Lite4 embedding (best on held-out images);
    videos -> forensic frame + motion features; audio -> voice-behaviour (prosody) features.
  * Model: standardisation + L2-regularised logistic regression (class-balanced). Small,
    exported to JSON so the app needs only NumPy/OpenCV at run time (works on Render's free plan).
  * Needs:  pip install scikit-learn   (training only)
  * Honest evaluation: leave-one-FILE-out - each file is predicted by a model that never saw it
    (nor any of its augmented copies).
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from app.analyzers.image_analyzer import detect_faces  # noqa: E402
from app.ml import embed as EMB  # noqa: E402
from app.ml import features as F  # noqa: E402
from app.utils import ffmpeg_bin  # noqa: E402

IMG_EXT = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
VID_EXT = {".mp4", ".mov", ".mkv", ".webm", ".avi", ".m4v"}
AUD_EXT = {".mp3", ".wav", ".m4a", ".ogg", ".opus", ".flac", ".aac"}


def files(folder: Path, exts):
    return sorted(p for p in folder.glob("*") if p.suffix.lower() in exts) if folder.exists() else []


def load_bgr(p: Path):
    im = cv2.imread(str(p))
    if im is None:
        im = cv2.cvtColor(np.array(Image.open(p).convert("RGB")), cv2.COLOR_RGB2BGR)
    return im


def jpeg(img, q):
    return cv2.imdecode(cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, q])[1], cv2.IMREAD_COLOR)


def resize_max(img, m):
    s = m / max(img.shape[:2])
    return cv2.resize(img, (max(16, int(img.shape[1] * s)), max(16, int(img.shape[0] * s))), interpolation=cv2.INTER_AREA)


def image_variants(img, rng):
    h, w = img.shape[:2]
    out = [img, jpeg(resize_max(img, 1280), 75), jpeg(resize_max(img, 640), 65), jpeg(resize_max(img, 480), 60)]
    c = 0.8
    y0, x0 = int(rng.uniform(0, 1 - c) * h), int(rng.uniform(0, 1 - c) * w)
    out.append(jpeg(img[y0:y0 + int(c * h), x0:x0 + int(c * w)], 80))
    out.append(jpeg(cv2.flip(img, 1), 85))
    return out


def ffmpeg(*args):
    subprocess.run([ffmpeg_bin(), "-hide_banner", "-loglevel", "error", "-y", *args], check=True)


def decode_audio(path: Path) -> np.ndarray:
    import soundfile as sf
    with tempfile.TemporaryDirectory() as td:
        out = Path(td) / "a.wav"
        ffmpeg("-i", str(path), "-vn", "-ac", "1", "-ar", "16000", "-t", "180", str(out))
        y, _ = sf.read(str(out), dtype="float32")
    return y


def audio_variants(path: Path):
    ys = [decode_audio(path)]
    with tempfile.TemporaryDirectory() as td:
        for codec, ext, br in (("libmp3lame", "mp3", "48k"), ("libopus", "ogg", "24k"), ("aac", "m4a", "64k")):
            out = Path(td) / f"v.{ext}"
            try:
                ffmpeg("-i", str(path), "-vn", "-ac", "1", "-ar", "48000" if codec == "libopus" else "44100",
                       "-c:a", codec, "-b:a", br, str(out))
                ys.append(decode_audio(out))
            except Exception as e:  # codec missing in this ffmpeg build
                print(f"    (skip {codec}: {e})")
    return ys


def video_variants(path: Path, td: Path):
    wa = td / (path.stem + "_wa.mp4")
    ffmpeg("-i", str(path), "-map_metadata", "-1", "-vf", "scale=480:-2", "-c:v", "libx264", "-preset", "veryfast",
           "-b:v", "1200k", "-an", str(wa))
    return [path, wa]


def face_fn(fr):
    fs = detect_faces(fr, 0.08)
    return fs[0] if fs else None


# ------------------------------------------------------------------ model
def fit(X, y, C=0.3):
    from sklearn.linear_model import LogisticRegression
    mu, sd = X.mean(0), X.std(0) + 1e-6
    clf = LogisticRegression(C=C, class_weight="balanced", max_iter=5000)
    clf.fit((X - mu) / sd, y)
    return {"mean": mu, "scale": sd, "coef": clf.coef_[0], "intercept": float(clf.intercept_[0])}


def prob(m, X):
    z = ((X - m["mean"]) / m["scale"]) @ m["coef"] + m["intercept"]
    return 1 / (1 + np.exp(-z))


AUDIO_SELECT = ["flatness_mean", "flatness_std", "flux_mean", "flux_std", "rms_cv", "silence_frac", "noise_floor_rel",
                "pitch_std", "pitch_jitter", "voiced_frac", "hnr_mean", "zcr_std", "centroid_std"]


def train_kind(kind, X, y, groups, names, feat_names, C=0.3, select=None, feature_type="handcrafted"):
    X, y, groups = np.array(X), np.array(y), np.array(groups)
    if select:
        idx = [feat_names.index(n) for n in select]
        X, feat_names = X[:, idx], list(select)
    print(f"\n[{kind}] {len(set(groups))} files -> {len(X)} training samples "
          f"(AI {int((y == 1).sum())} / real {int((y == 0).sum())})")
    # leave-one-file-out evaluation
    rows, correct = [], 0
    for g in sorted(set(groups)):
        tr, te = groups != g, groups == g
        if len(set(y[tr])) < 2:
            continue
        m = fit(X[tr], y[tr], C)
        p = float(prob(m, X[te]).mean())
        truth = int(y[te][0])
        ok = (p >= 0.5) == bool(truth)
        correct += ok
        rows.append({"file": names[g], "truth": "AI" if truth else "real", "p_ai": round(p, 3), "correct": bool(ok)})
        print(f"   held-out {names[g]:28s} truth={'AI  ' if truth else 'real'} p(AI)={p:.2f} {'OK' if ok else 'WRONG'}")
    acc = correct / max(1, len(rows))
    print(f"   leave-one-file-out accuracy: {correct}/{len(rows)} = {acc:.0%}")
    m = fit(X, y, C)
    train_acc = float(((prob(m, X) >= 0.5) == (y == 1)).mean())
    order = np.argsort(-np.abs(m["coef"]))[:8]  # (for embeddings these are anonymous dimensions)
    top = [(feat_names[i], round(float(m["coef"][i]), 3)) for i in order]
    print("   strongest features:", top)
    return {"kind": kind, "feature_type": feature_type, "features": feat_names, "mean": m["mean"].tolist(), "scale": m["scale"].tolist(),
            "coef": m["coef"].tolist(), "intercept": m["intercept"], "C": C,
            "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "n_files": {"ai": int(len({g for g in set(groups) if y[groups == g][0] == 1})),
                        "real": int(len({g for g in set(groups) if y[groups == g][0] == 0}))},
            "n_samples": int(len(X)), "loo_accuracy": round(acc, 3), "train_accuracy": round(train_acc, 3),
            "loo_results": rows, "top_features": top}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(ROOT / "training_data"))
    ap.add_argument("--out", default=str(ROOT / "app" / "models"))
    ap.add_argument("--only", choices=["images", "videos", "audio"])
    a = ap.parse_args()
    data, out = Path(a.data), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(7)
    summary = {}

    if a.only in (None, "images"):
        X, y, G, names = [], [], [], {}
        for label, sub in ((1, "ai"), (0, "real")):
            for p in files(data / "images" / sub, IMG_EXT):
                g = len(names)
                names[g] = f"{sub}/{p.name}"
                for v in image_variants(load_bgr(p), rng):
                    X.append(EMB.embed(v)); y.append(label); G.append(g)
        if len(set(y)) == 2:
            summary["image"] = train_kind("image", X, y, G, names, [f"effnet_{i}" for i in range(len(X[0]))],
                                          C=0.3, feature_type="effnet_lite4")
            (out / "ai_image.json").write_text(json.dumps(summary["image"], indent=1))

    if a.only in (None, "videos"):
        X, y, G, names = [], [], [], {}
        with tempfile.TemporaryDirectory() as td:
            for label, sub in ((1, "ai"), (0, "real")):
                for p in files(data / "videos" / sub, VID_EXT):
                    g = len(names)
                    names[g] = f"{sub}/{p.name}"
                    print(f"  video {names[g]} ...", flush=True)
                    for v in video_variants(p, Path(td)):
                        for frames, faces in F.read_video_clips(v, face_fn=face_fn):
                            X.append(F.video_clip_features(frames, faces)); y.append(label); G.append(g)
        if len(set(y)) == 2:
            summary["video"] = train_kind("video", X, y, G, names, F.VIDEO_FEATURES, C=0.05)
            (out / "ai_video.json").write_text(json.dumps(summary["video"], indent=1))

    if a.only in (None, "audio"):
        X, y, G, names = [], [], [], {}
        for label, sub in ((1, "ai"), (0, "real")):
            for p in files(data / "audio" / sub, AUD_EXT):
                g = len(names)
                names[g] = f"{sub}/{p.name}"
                print(f"  audio {names[g]} ...", flush=True)
                for yy in audio_variants(p):
                    for seg in F.audio_segments(yy):
                        X.append(F.audio_segment_features(seg)); y.append(label); G.append(g)
        if len(set(y)) == 2:
            summary["audio"] = train_kind("audio", X, y, G, names, F.AUDIO_FEATURES, C=0.1, select=AUDIO_SELECT)
            (out / "ai_audio.json").write_text(json.dumps(summary["audio"], indent=1))

    print("\nSaved models to", out)
    for k, v in summary.items():
        print(f"  {k:6s} files AI/real = {v['n_files']['ai']}/{v['n_files']['real']}, samples = {v['n_samples']}, "
              f"leave-one-file-out accuracy = {v['loo_accuracy']:.0%}")


if __name__ == "__main__":
    main()
