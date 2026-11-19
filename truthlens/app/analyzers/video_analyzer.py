"""Video forensic analysis.

Instead of analysing every frame (slow), TruthLens:
  * samples key frames every N seconds for detailed face / texture checks
  * samples a light-weight stream (~12 fps) for motion, cut detection and mouth movement
Indicators:
  * Sudden visual changes (hard cuts / splices) with before/after thumbnails
  * Frozen / duplicated frames
  * Face boundary / texture anomalies per key frame (same checks as images)
  * Face identity / size jumps between consecutive samples (face-swap flicker)
  * Mouth-motion signal for audio-video synchronisation (used by cross-modal module)
  * Container metadata (encoder, creation time, re-encoding)
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from ..config import MAX_VIDEO_SECONDS, SYNC_ANALYSIS_FPS, VIDEO_SAMPLE_EVERY_SEC
from ..utils import fmt_ts, make_finding, probe_media, save_evidence_image
from .image_analyzer import detect_faces, detect_faces_full, face_consistency

EDIT_ENCODERS = ["premiere", "after effects", "davinci", "final cut", "capcut", "kinemaster", "filmora",
                 "inshot", "vn video", "imovie", "shotcut", "openshot", "handbrake", "lavf", "ffmpeg"]


def _face_hist(frame, f):
    x, y, w, h = f
    crop = frame[y + h // 6:y + h * 5 // 6, x + w // 6:x + w * 5 // 6]
    if crop.size == 0:
        return None
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1], None, [24, 16], [0, 180, 0, 256])
    return cv2.normalize(hist, hist).flatten()


def analyze_video(path: Path, job_id: str, progress=lambda *_: None) -> dict:
    findings: list[dict] = []
    evidence: dict = {"keyframes": [], "suspicious_frames": []}
    probe = probe_media(path)
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise ValueError("Could not open video file (unsupported codec?)")
    fps = cap.get(cv2.CAP_PROP_FPS) or probe.get("fps") or 25.0
    if fps <= 1 or fps > 240:
        fps = probe.get("fps") or 25.0
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    duration = probe.get("duration_sec") or (n_frames / fps if n_frames else 0)
    analyse_until = min(duration or MAX_VIDEO_SECONDS, MAX_VIDEO_SECONDS)
    W, H = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    info = {"duration_sec": round(duration, 2), "fps": round(fps, 2), "frame_count": n_frames,
            "width": W, "height": H, "video_codec": probe.get("video_codec"),
            "audio_codec": probe.get("audio_codec"), "has_audio": probe.get("has_audio"),
            "bitrate_kbps": probe.get("bitrate_kbps"), "analysed_seconds": round(analyse_until, 2),
            "container_tags": probe.get("raw_tags", {})}

    sync_step = max(1, int(round(fps / SYNC_ANALYSIS_FPS)))
    key_step = max(1, int(round(fps * VIDEO_SAMPLE_EVERY_SEC)))
    proc_w = 480
    scale = min(1.0, proc_w / max(W, 1))

    times, diffs, mouth, face_seen = [], [], [], []
    prev_small, prev_mouth = None, None
    last_face, last_mouth, face_age = None, None, 99
    key_records = []
    prev_key_face = None
    idx = 0
    while True:
        ok = cap.grab()
        if not ok:
            break
        t = idx / fps
        if t > analyse_until:
            break
        is_sync, is_key = idx % sync_step == 0, idx % key_step == 0
        if not (is_sync or is_key):
            idx += 1
            continue
        ok, frame = cap.retrieve()
        if not ok:
            idx += 1
            continue
        small = cv2.resize(frame, (int(W * scale), int(H * scale))) if scale < 1 else frame
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)

        if is_sync:
            g = cv2.resize(gray, (64, 36)).astype(np.float32)
            diffs.append(float(np.abs(g - prev_small).mean()) if prev_small is not None else 0.0)
            prev_small = g
            # face tracking: re-detect every 3rd sync frame
            if face_age >= 3 or last_face is None:
                fs = detect_faces_full(small, 0.08)
                if fs:
                    last_face, last_mouth = fs[0]["bbox"], fs[0]["mouth"]
                    face_age = 0
                elif face_age >= 6:
                    last_face, last_mouth = None, None
            face_age += 1
            mv = np.nan
            if last_face is not None:
                x, y, w, h = last_face
                if last_mouth is not None:   # landmark-based mouth box (YuNet)
                    (mx1, my1), (mx2, my2) = last_mouth
                    mw = max(8.0, abs(mx2 - mx1))
                    cx, cy = (mx1 + mx2) / 2, (my1 + my2) / 2
                    x0, x1 = int(cx - 0.75 * mw), int(cx + 0.75 * mw)
                    y0, y1 = int(cy - 0.45 * mw), int(cy + 0.65 * mw)
                else:                        # geometric fallback (Haar)
                    x0, x1 = x + int(0.22 * w), x + int(0.78 * w)
                    y0, y1 = y + int(0.62 * h), y + int(0.98 * h)
                roi = gray[max(0, y0):max(0, y1), max(0, x0):max(0, x1)]
                if roi.size > 20:
                    r = cv2.resize(roi, (32, 16)).astype(np.float32)
                    r = (r - r.mean()) / (r.std() + 1e-3)
                    if prev_mouth is not None:
                        mv = float(np.abs(r - prev_mouth).mean())
                    prev_mouth = r
            else:
                prev_mouth = None
            times.append(t)
            mouth.append(mv)
            face_seen.append(last_face is not None)

        if is_key:
            fs = detect_faces(small, 0.06)
            rec = {"t": t, "frame": idx, "faces": len(fs), "anomaly": 0.0, "reasons": [], "bbox": None, "feat": {},
                   "frame_L": float(gray.mean()), "n_faces": len(fs)}
            thumb = save_evidence_image(job_id, f"key_{idx:06d}.jpg", small, 360)
            rec["thumb"] = thumb
            if fs:
                f = fs[0]
                fc = face_consistency(small, gray, f)
                rec.update(bbox=[int(v / scale) for v in f], metrics=fc, feat=_face_features(small, gray, f))
                hist = _face_hist(small, f)
                if prev_key_face is not None and hist is not None:
                    ph, pf = prev_key_face
                    rec["identity_corr"] = float(cv2.compareHist(ph, hist, cv2.HISTCMP_CORREL))
                    rec["size_jump"] = float(abs(f[2] - pf[2]) / max(pf[2], 1))
                prev_key_face = (hist, f) if hist is not None else prev_key_face
            key_records.append(rec)
            evidence["keyframes"].append({"t": round(t, 2), "frame": idx, "url": thumb, "faces": len(fs)})
            if len(key_records) % 10 == 0:
                progress(f"Video: analysed {fmt_ts(t)} of {fmt_ts(analyse_until)}")
        idx += 1
    cap.release()
    progress("Video frames sampled")

    times_a = np.array(times)
    diffs_a = np.array(diffs)
    info["sampled_frames"] = len(times)
    info["keyframes"] = len(key_records)
    info["face_visible_ratio"] = round(float(np.mean(face_seen)) if face_seen else 0.0, 3)
    info["multi_face_ratio"] = round(float(np.mean([r["n_faces"] > 1 for r in key_records])) if key_records else 0.0, 3)

    # ---- sudden visual changes
    cuts = []
    if len(diffs_a) > 10:
        med = np.median(diffs_a[1:]) + 1e-3
        mad = np.median(np.abs(diffs_a[1:] - med)) * 1.4826 + 0.5
        for i in range(1, len(diffs_a)):
            if diffs_a[i] > med + 8 * mad and diffs_a[i] > 12:
                cuts.append((times_a[i], diffs_a[i]))
    info["sudden_changes"] = len(cuts)
    if cuts:
        cap = cv2.VideoCapture(str(path))
        evs = []
        for k, (t, d) in enumerate(cuts[:4]):
            pair = []
            for tt in (max(0, t - 1 / SYNC_ANALYSIS_FPS), t):
                cap.set(cv2.CAP_PROP_POS_MSEC, tt * 1000)
                ok, fr = cap.read()
                if ok:
                    pair.append(cv2.resize(fr, (320, int(320 * H / max(W, 1)))))
            if len(pair) == 2:
                sep = np.full((pair[0].shape[0], 6, 3), 255, np.uint8)
                combo = np.hstack([pair[0], sep, pair[1]])
                evs.append(save_evidence_image(job_id, f"cut_{k}.jpg", combo, 700))
        cap.release()
        strength = float(np.clip(0.2 + 0.08 * len(cuts), 0, 0.6))
        findings.append(make_finding(
            "video", f"Sudden visual change at {fmt_ts(cuts[0][0])}", strength, 0.6,
            f"{len(cuts)} abrupt scene change(s) detected (" + ", ".join(fmt_ts(t) for t, _ in cuts[:6]) + "). "
            "If the video is presented as one continuous recording, cuts mean content may have been removed or "
            "joined. Normal for edited news packages or compilations.",
            {"type": "timestamp", "t": round(cuts[0][0], 2), "all": [round(t, 2) for t, _ in cuts[:20]]},
            evs, "video.cuts", ctx={"n_cuts": len(cuts), "duration": round(analyse_until, 2),
                                    "cut_interval_cv": _interval_cv([t for t, _ in cuts])}))

    # ---- frozen / duplicated frames (identical consecutive samples while scene otherwise moves)
    if len(diffs_a) > 20 and np.median(diffs_a) > 1.0:
        frozen = diffs_a[1:] < 0.05
        runs = _runs(frozen)
        long_runs = [(times_a[s + 1], times_a[min(e, len(times_a) - 1)]) for s, e in runs if e - s >= 4]
        if long_runs:
            a, b = long_runs[0]
            findings.append(make_finding(
                "video", "Frozen / duplicated frames", 0.35, 0.5,
                f"The picture is completely static from {fmt_ts(a)} to {fmt_ts(b)} while the rest of the video moves. "
                "Duplicated frames are used to pad edits or hide removed content.",
                {"type": "timestamp", "t": round(a, 2), "end": round(b, 2)}, check="video.frozen"))

    # ---- face texture / colour anomalies over time (temporal, robust to compression)
    face_keys = [r for r in key_records if r["faces"] and r.get("feat")]
    cut_times = [t for t, _ in cuts]
    susp, step_info = _temporal_face_anomalies(face_keys, cut_times)
    info["face_keyframes"] = len(face_keys)
    info["face_anomaly_keyframes"] = len(susp)
    if step_info:
        t_step = step_info["t"]
        info["face_step_change"] = step_info
        r = min(face_keys, key=lambda x: abs(x["t"] - t_step))
        r["reasons"] = [f"face {step_info['feature']} changes abruptly ({step_info['before']:.1f} -> "
                        f"{step_info['after']:.1f}, {step_info['score']:.1f} sigma) and stays changed"]
        ev = _frame_with_box(path, job_id, r, f"Frame {r['frame']}")
        before = max([x for x in face_keys if x["t"] < t_step], key=lambda x: x["t"], default=None)
        evs = [ev]
        if before:
            evs.insert(0, _frame_with_box(path, job_id, {**before, "reasons": []}, "before"))
        evidence["suspicious_frames"].append({"t": round(r["t"], 2), "frame": r["frame"], "url": ev,
                                              "reason": r["reasons"][0]})
        findings.append(make_finding(
            "video", f"Facial texture/colour shift starting at frame {r['frame']} ({fmt_ts(r['t'])})",
            float(np.clip(0.35 + step_info["score"] / 25, 0, 0.85)), 0.55,
            f"From {fmt_ts(t_step)} onward the face region's {step_info['feature']} differs from the earlier part of "
            f"the video ({step_info['before']:.1f} vs {step_info['after']:.1f}) without a scene cut. A face-swap or "
            "face filter that switches on mid-video produces exactly this kind of persistent change. Compare the "
            "before/after evidence frames.",
            {"type": "timestamp", "t": round(r["t"], 2), "frame": r["frame"], "bbox": r["bbox"]}, evs,
            "video.face_step", ctx=_lightness_step(face_keys, t_step)))
    if susp:
        evs = []
        for r in susp[:4]:
            evs.append(_frame_with_box(path, job_id, r, f"Frame {r['frame']}"))
            evidence["suspicious_frames"].append({"t": round(r["t"], 2), "frame": r["frame"],
                                                  "url": evs[-1], "reason": "; ".join(r["reasons"])})
        top = susp[0]
        ratio = len(susp) / max(1, len(face_keys))
        findings.append(make_finding(
            "video", f"Facial boundary / texture anomaly at frame {top['frame']} ({fmt_ts(top['t'])})",
            float(np.clip(0.3 + ratio + top["anomaly"] * 0.3, 0, 0.85)), 0.5,
            f"In {len(susp)} of {len(face_keys)} sampled frames the face region deviates strongly from how the same "
            f"face looks in the rest of the video. Strongest at {fmt_ts(top['t'])}: " + "; ".join(top["reasons"]) +
            ". Deepfake blending often fails on individual frames, causing such flicker.",
            {"type": "timestamp", "t": round(top["t"], 2), "frame": top["frame"], "bbox": top["bbox"],
             "all": [round(r["t"], 2) for r in susp[:20]]}, evs, "video.face_anomaly"))

    # ---- face identity / size flicker (not explained by a cut)
    flick = [r for r in face_keys if (r.get("identity_corr", 1) < 0.35 or r.get("size_jump", 0) > 0.6)
             and not any(abs(r["t"] - ct) < VIDEO_SAMPLE_EVERY_SEC * 1.5 for ct in cut_times)
             and not (step_info and abs(r["t"] - step_info["t"]) < VIDEO_SAMPLE_EVERY_SEC * 1.5)]
    if flick:
        r = flick[0]
        ev = _frame_with_box(path, job_id, r, "Face change")
        evidence["suspicious_frames"].append({"t": round(r["t"], 2), "frame": r["frame"], "url": ev,
                                              "reason": "face appearance changed abruptly"})
        findings.append(make_finding(
            "video", f"Face appearance changes abruptly at {fmt_ts(r['t'])}", min(0.35 + 0.1 * len(flick), 0.75), 0.45,
            f"Between consecutive samples the face's colour/texture profile (correlation "
            f"{r.get('identity_corr', 1):.2f}) or size (jump {r.get('size_jump', 0) * 100:.0f}%) changes without a "
            f"scene cut. {len(flick)} such event(s). Face-swap models can flicker when tracking fails.",
            {"type": "timestamp", "t": round(r["t"], 2), "frame": r["frame"],
             "all": [round(x["t"], 2) for x in flick[:20]]}, [ev], "video.face_flicker"))

    # ---- metadata
    tags = {k.lower(): v.lower() for k, v in probe.get("raw_tags", {}).items()}
    enc = tags.get("encoder", "") + " " + tags.get("handler_name", "") + " " + tags.get("comment", "")
    hits = [e for e in EDIT_ENCODERS if e in enc]
    if hits:
        findings.append(make_finding(
            "metadata", "Video was re-encoded / processed by software", 0.3, 0.6,
            f"Container metadata names '{probe['raw_tags'].get('encoder', hits[0])}'. The file is not a camera "
            "original; it has been exported or converted at least once (also true for most social-media downloads).",
            {"type": "metadata", "field": "encoder"}, check="metadata.video_encoder"))
    if "creation_time" not in tags:
        findings.append(make_finding(
            "metadata", "No creation time in video metadata", 0.2, 0.45,
            "The container has no recording date, so the claimed recording time cannot be checked from metadata.",
            {"type": "metadata", "field": "creation_time"}, check="metadata.video_no_date"))
    info["creation_time"] = probe.get("raw_tags", {}).get("creation_time")

    # timeline for the UI
    step = max(1, len(times) // 400)
    timeline = {"t": [round(x, 2) for x in times[::step]],
                "visual_change": [round(x, 2) for x in diffs[::step]],
                "mouth_motion": [None if np.isnan(x) else round(x, 3) for x in mouth[::step]]}
    return {"media_type": "video", "info": info, "findings": findings, "evidence": evidence,
            "timeline": timeline,
            "sync_series": {"t": times, "mouth": mouth, "fps": SYNC_ANALYSIS_FPS, "face_seen": face_seen},
            "checks_run": ["Container metadata", "Key-frame sampling", "Scene-cut detection", "Frozen frames",
                           "Temporal face texture/colour consistency", "Face identity flicker", "Mouth-motion tracking"]}


def _face_features(img, gray, f) -> dict:
    """Absolute appearance features of the central face region (compared over time, not to background)."""
    x, y, w, h = f
    cx0, cy0, cx1, cy1 = x + w // 5, y + h // 5, x + 4 * w // 5, y + 4 * h // 5
    g = gray[cy0:cy1, cx0:cx1]
    if g.size < 100:
        return {}
    lab = cv2.cvtColor(img[cy0:cy1, cx0:cx1], cv2.COLOR_BGR2LAB).reshape(-1, 3).astype(np.float32)
    lap = np.abs(cv2.Laplacian(g, cv2.CV_32F))
    res = np.abs(g.astype(np.float32) - cv2.medianBlur(g, 3).astype(np.float32))
    fc = face_consistency(img, gray, f)
    return {"sharpness": float(lap.mean()), "noise": float(res.mean() * 10),
            "lightness": float(lab[:, 0].mean()), "colour (a*)": float(lab[:, 1].mean()),
            "colour (b*)": float(lab[:, 2].mean()), "boundary seam": float(fc.get("boundary_ratio", 1) * 10)}


def _temporal_face_anomalies(face_keys, cut_times):
    """Robust z-scores per keyframe + single step-change detection for each face feature."""
    if len(face_keys) < 6:
        return [], None
    names = list(face_keys[0]["feat"].keys())
    F = np.array([[r["feat"].get(n, np.nan) for n in names] for r in face_keys])
    ts = np.array([r["t"] for r in face_keys])
    med = np.nanmedian(F, 0)
    mad = np.nanmedian(np.abs(F - med), 0) * 1.4826
    scale = np.maximum(mad, np.abs(med) * 0.04 + 0.3)        # floor: ignore tiny absolute variations
    Z = (F - med) / scale
    # step change: best split point per feature (Welch-like statistic, each side >= 3 samples)
    best = None
    for j, n in enumerate(names):
        col = F[:, j]
        for k in range(3, len(col) - 2):
            a, b = col[:k], col[k:]
            pooled = np.sqrt((a.var() + b.var()) / 2) + scale[j] * 0.25
            score = abs(a.mean() - b.mean()) / pooled
            near_cut = any(abs(ts[k] - c) < 1.5 for c in cut_times)
            if score > 4.0 and not near_cut and (best is None or score > best["score"]):
                best = {"feature": n, "t": round(float(ts[k]), 2), "score": round(float(score), 2),
                        "before": round(float(a.mean()), 2), "after": round(float(b.mean()), 2)}
    if best:   # measure deviations within each segment (before / after the step) separately
        seg = ts >= best["t"]
        for part in (seg, ~seg):
            if part.sum() >= 3:
                m = np.nanmedian(F[part], 0)
                sc = np.maximum(np.nanmedian(np.abs(F[part] - m), 0) * 1.4826, np.abs(m) * 0.04 + 0.3)
                Z[part] = (F[part] - m) / sc
    susp = []
    for i, r in enumerate(face_keys):
        if any(abs(r["t"] - c) < 1.0 for c in cut_times):
            continue
        if best and abs(r["t"] - best["t"]) < 1.0:
            continue
        bad = [(names[j], Z[i, j]) for j in range(len(names)) if abs(Z[i, j]) > 4.5]
        if bad:
            r["anomaly"] = float(min(1.0, max(abs(z) for _, z in bad) / 10))
            r["reasons"] = [f"{n} deviates {z:+.1f} sigma from the video's normal face appearance" for n, z in bad]
            susp.append(r)
    if len(susp) > 0.5 * len(face_keys):    # most frames 'deviate' -> no stable baseline, don't claim flicker
        susp = []
    susp.sort(key=lambda r: -r["anomaly"])
    return susp, best


def _interval_cv(ts):
    if len(ts) < 3:
        return None
    d = np.diff(ts)
    return round(float(d.std() / (d.mean() + 1e-6)), 3)


def _lightness_step(face_keys, t_step):
    """Did the WHOLE frame change brightness at the same moment? (camera exposure / lighting change)"""
    before = [r["frame_L"] for r in face_keys if r["t"] < t_step]
    after = [r["frame_L"] for r in face_keys if r["t"] >= t_step]
    if len(before) < 2 or len(after) < 2:
        return {}
    return {"frame_lightness_before": round(float(np.mean(before)), 1),
            "frame_lightness_after": round(float(np.mean(after)), 1),
            "frame_lightness_step": round(float(abs(np.mean(after) - np.mean(before))), 1)}


def _frame_with_box(path, job_id, rec, label):
    cap = cv2.VideoCapture(str(path))
    cap.set(cv2.CAP_PROP_POS_FRAMES, rec["frame"])
    ok, fr = cap.read()
    cap.release()
    if not ok:
        return rec.get("thumb", "")
    if rec.get("bbox"):
        x, y, w, h = rec["bbox"]
        cv2.rectangle(fr, (x, y), (x + w, y + h), (0, 0, 255), max(2, fr.shape[1] // 300))
        cv2.putText(fr, label, (x, max(20, y - 8)), cv2.FONT_HERSHEY_SIMPLEX, max(0.6, fr.shape[1] / 1200),
                    (0, 0, 255), 2, cv2.LINE_AA)
    cv2.putText(fr, f"t={fmt_ts(rec['t'])}  frame {rec['frame']}", (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                (255, 255, 255), 2, cv2.LINE_AA)
    return save_evidence_image(job_id, f"susp_{rec['frame']:06d}.jpg", fr, 900)


def _runs(mask):
    m = np.concatenate([[False], mask, [False]]).astype(np.int8)
    d = np.diff(m)
    return list(zip(np.where(d == 1)[0], np.where(d == -1)[0]))
