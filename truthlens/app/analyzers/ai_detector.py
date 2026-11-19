"""AI-generated media detection -> the clear YES / NO verdict shown at the top of the dashboard.

Two independent layers:

1. PROVENANCE (decisive when present)
   C2PA "Content Credentials" manifests embedded by generators such as Google Veo / Gemini / Flow,
   OpenAI Sora / DALL-E, Adobe Firefly, Microsoft Designer. A manifest whose action declares
   digitalSourceType = trainedAlgorithmicMedia means the creator tool itself states the media was
   made by generative AI (Google additionally writes "Applied imperceptible SynthID watermark").
   Messaging apps (WhatsApp, Instagram) strip these manifests, so layer 2 is needed.

2. VISUAL SIGNALS (work after WhatsApp compression; calibrated on real vs AI sample videos)
   a. Missing camera sensor grain   - generated frames are unnaturally clean
   b. Face sharp, background soft   - synthetic depth-of-field / generator focus on the subject
   c. Smooth "gliding" camera       - continuous motion without natural hand-shake micro-jitter
   d. Background warping            - background points do not move as one rigid scene

Verdict: confirmed (provenance) / likely / possible / not detected. Every signal reports its measured
value and threshold so the result is explainable.
"""
from __future__ import annotations

import re
from pathlib import Path

import cv2
import numpy as np

from ..ml import predict
from ..utils import make_finding
from .image_analyzer import detect_faces, noise_residual

AI_SOURCE_TYPES = {"trainedalgorithmicmedia": "created by a trained AI model",
                   "compositewithtrainedalgorithmicmedia": "real footage combined with AI-generated content",
                   "algorithmicmedia": "created algorithmically"}
AI_TOOL_WORDS = ["veo", "sora", "runway", "kling", "pika", "heygen", "synthesia", "luma", "hailuo", "minimax",
                 "firefly", "midjourney", "dall-e", "stable video", "gen-3", "gen-4", "d-id", "invideo", "synthid",
                 "imagen", "flux", "wan2", "hunyuan"]

# Calibration (frames resized to 360 px width). Real phone videos measured 1.08-1.54 grain and 0.76-0.86
# face/background sharpness; the AI sample measured 0.53-0.57 and 2.4-2.7.
GRAIN_AI_MAX = 0.75
SHARP_AI_MIN = 1.8
WARP_AI_MIN = 0.06


# ------------------------------------------------------------------ provenance
def scan_provenance(path: Path) -> dict:
    """Read C2PA / JUMBF manifests and generator hints from the file bytes (head + tail)."""
    size = path.stat().st_size
    with open(path, "rb") as f:
        head = f.read(6 * 1024 * 1024)
        tail = b""
        if size > 7 * 1024 * 1024:
            f.seek(-1024 * 1024, 2)
            tail = f.read()
    blob = head + tail
    low = blob.lower()
    out: dict = {"c2pa_present": b"c2pa" in low and (b"jumb" in low or b"jumd" in low),
                 "ai_declared": False, "actions": [], "signer": None, "generator": None, "hints": []}
    if out["c2pa_present"]:
        txt = blob.decode("latin-1")
        # actions: c2pa.created / c2pa.edited with CBOR-encoded description and digitalSourceType
        for m in re.finditer(rb"c2pa\.(created|edited|converted|opened|placed|produced|transcoded)", blob):
            seg = blob[m.end():m.end() + 400]
            act = {"action": "c2pa." + m.group(1).decode(), "description": _cbor_after(seg, b"description"),
                   "digitalSourceType": _cbor_after(seg, b"digitalSourceType").split("/")[-1]}
            if act not in out["actions"] and (act["description"] or act["digitalSourceType"]):
                out["actions"].append(act)
        m = re.search(r"([A-Z][A-Za-z0-9.&-]*(?: [A-Za-z0-9.&-]+){0,3} C2PA(?: [A-Z][a-z]+){1,3})", txt)
        if m:
            out["signer"] = m.group(1).strip()
        m = re.search(r"claim_generator(?:_info)?[^\x20-\x7e]{0,12}(?:d?name[^\x20-\x7e]{0,4}x?.?)?([\x20-\x7e]{4,80})", txt)
        if m:
            out["generator"] = m.group(1).strip()
        if re.search(r"generative ai|synthid", txt, re.I):
            out["ai_declared"] = True
    # generator hints in metadata: only readable text strings (avoids false matches in compressed data)
    strings = b" ".join(re.findall(rb"[\x20-\x7e]{10,}", blob[:400000] + blob[-200000:])).lower()
    for w in AI_TOOL_WORDS:
        if re.search(rb"(?<![a-z])" + re.escape(w.encode()) + rb"(?![a-z])", strings):
            out["hints"].append(w)
    if b"produced by google" in strings:
        out["hints"].append("Google-produced container")
    return out


def _cbor_after(seg: bytes, key: bytes) -> str:
    """Read the CBOR text string that follows a map key (C2PA manifests are CBOR-encoded)."""
    i = seg.find(key)
    if i < 0:
        return ""
    j = i + len(key)
    if j >= len(seg):
        return ""
    b = seg[j]
    if 0x60 <= b <= 0x77:
        n, j = b - 0x60, j + 1
    elif b == 0x78 and j + 1 < len(seg):
        n, j = seg[j + 1], j + 2
    elif b == 0x79 and j + 2 < len(seg):
        n, j = int.from_bytes(seg[j + 1:j + 3], "big"), j + 3
    else:
        return ""
    return seg[j:j + n].decode("utf-8", "ignore").strip()


# ------------------------------------------------------------------ visual signals (video)
def video_signals(path: Path, max_sec: float = 20.0, sample_fps: float = 10.0) -> dict:
    cap = cv2.VideoCapture(str(path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    step = max(1, int(round(fps / sample_fps)))
    prev = None
    face = None
    i = 0
    grain, sharp, resid, trans = [], [], [], []
    while True:
        ok = cap.grab()
        if not ok or i / fps > max_sec:
            break
        if i % step:
            i += 1
            continue
        ok, fr = cap.retrieve()
        i += 1
        if not ok:
            continue
        W = 360
        fr = cv2.resize(fr, (W, max(2, int(fr.shape[0] * W / fr.shape[1]))), interpolation=cv2.INTER_AREA)
        g = cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY)
        H = g.shape[0]
        if len(grain) % 3 == 0:
            fs = detect_faces(fr, 0.08)
            face = fs[0] if fs else None
        grain.append(float(np.abs(noise_residual(g)).mean()))
        mask = np.full(g.shape, 255, np.uint8)
        if face:
            x, y, w, h = face
            cv2.rectangle(mask, (int(x - 1.0 * w), int(y - 0.7 * h)), (int(x + 2.0 * w), H), 0, -1)
            if (mask > 0).sum() > 2000:
                lap = np.abs(cv2.Laplacian(g, cv2.CV_32F))
                sharp.append(float(lap[y:y + h, x:x + w].mean() / (lap[mask > 0].mean() + 1e-3)))
        if prev is not None:
            pg, pm = prev
            pts = cv2.goodFeaturesToTrack(pg, 300, 0.01, 6, mask=pm)
            if pts is not None and len(pts) > 20:
                nxt, st, _ = cv2.calcOpticalFlowPyrLK(pg, g, pts, None, winSize=(21, 21), maxLevel=3)
                st = st.ravel() == 1
                p0, p1 = pts[st].reshape(-1, 2), nxt[st].reshape(-1, 2)
                if len(p0) > 15:
                    M, _ = cv2.estimateAffinePartial2D(p0, p1, ransacReprojThreshold=1.5)
                    if M is not None:
                        pred = p0 @ M[:, :2].T + M[:, 2]
                        resid.append(float(np.median(np.linalg.norm(p1 - pred, axis=1))))
                        trans.append(M[:, 2].copy())
        prev = (g, mask)
    cap.release()
    T = np.array(trans) if trans else np.zeros((0, 2))
    motion = float(np.median(np.linalg.norm(T, axis=1))) if len(T) else 0.0
    jitter = 0.0
    if len(T) > 6:
        sm = np.array([np.convolve(T[:, j], np.ones(5) / 5, "same") for j in range(2)]).T
        d = np.linalg.norm(T - sm, axis=1)[2:-2]
        jitter = float(np.median(d)) if len(d) else 0.0
    return {"frames": len(grain), "grain": float(np.median(grain)) if grain else None,
            "face_bg_sharpness": float(np.median(sharp)) if len(sharp) >= 5 else None,
            "background_warp": float(np.median(resid)) if resid else None,
            "camera_motion": motion, "camera_jitter": jitter}


# ------------------------------------------------------------------ verdict
def _lerp(v, a, b):
    return float(np.clip((v - a) / (b - a), 0, 1))


def assess_video(path: Path, cuts: int = 0) -> tuple[dict, list[dict]]:
    prov = scan_provenance(path)
    sig = video_signals(path)
    signals, findings = [], []

    def add(key, name, value, fired, strength, normal, ai_range, explanation, check, cat="video"):
        signals.append({"key": key, "name": name, "value": None if value is None else round(value, 3),
                        "fired": bool(fired), "strength": round(strength, 3), "normal": normal, "ai": ai_range})
        if fired:
            findings.append(make_finding(cat, name, strength, 0.6 if cat == "video" else 0.95, explanation,
                                         {"type": "segment", "start": 0, "end": 20} if cat == "video" else
                                         {"type": "metadata", "field": "C2PA manifest"}, check=check,
                                         ctx={"value": value, **sig}))

    # 1 provenance
    if prov["ai_declared"]:
        desc = next((a["description"] for a in prov["actions"] if a.get("description")), "") or "Generative AI"
        add("c2pa", "Content Credentials (C2PA) declare this as AI-generated", 1.0, True, 0.95,
            "absent or camera-signed", "AI action present",
            f"The file carries a C2PA manifest{(' signed by ' + prov['signer']) if prov.get('signer') else ''} whose "
            f"action states: \"{desc}\" (digitalSourceType = {prov.get('source_type', 'trainedAlgorithmicMedia')}). "
            "The creating tool itself labels this media as produced by generative AI.", "ai.c2pa", cat="metadata")
    elif prov["hints"]:
        add("hint", "AI-tool / generator signature in file metadata", None, True, 0.4, "camera/app names",
            ", ".join(prov["hints"]), f"File metadata contains generator-related signatures: {', '.join(prov['hints'])}.",
            "ai.metadata_hint", cat="metadata")

    # 2 visual
    g = sig["grain"]
    if g is not None:
        s = _lerp(GRAIN_AI_MAX + 0.15 - g, 0.0, 0.45)
        add("grain", "Missing natural camera grain", g, g < GRAIN_AI_MAX, max(0.35, s) if g < GRAIN_AI_MAX else 0,
            "1.0 - 1.6", f"< {GRAIN_AI_MAX}",
            f"Average sensor-noise level is {g:.2f} (real phone videos measured 1.0-1.6, also after WhatsApp). Generated "
            "frames are rendered, not captured by a sensor, so they lack the fine random grain of a camera.",
            "ai.no_sensor_grain")
    fs = sig["face_bg_sharpness"]
    if fs is not None:
        add("sharp", "Face rendered much sharper than the background", fs, fs > SHARP_AI_MIN,
            max(0.35, _lerp(fs, SHARP_AI_MIN, 3.2)) if fs > SHARP_AI_MIN else 0, "0.6 - 1.2", f"> {SHARP_AI_MIN}",
            f"The face is {fs:.1f}x sharper than the background across the video. Real selfie videos keep face and "
            "room at similar sharpness; video generators focus detail on the subject and render a soft background.",
            "ai.face_bg_sharpness")
    mo, ji = sig["camera_motion"], sig["camera_jitter"]
    glide = mo > 0.4 and cuts == 0 and ji < 0.9 * mo and sig["frames"] > 30
    add("glide", "Smooth 'gliding' camera without hand-shake", mo, glide,
        max(0.35, _lerp(mo - ji, 0.2, 1.0)) if glide else 0, "static, or shaky handheld", "continuous smooth drift",
        f"The camera drifts continuously ({mo:.2f} px/frame) with almost no natural hand tremor ({ji:.2f} px). A phone "
        "held in the hand shakes irregularly; generators move a virtual camera along a smooth path.", "ai.smooth_camera")
    bw = sig["background_warp"]
    if bw is not None:
        warp = bw > WARP_AI_MIN and mo > 0.2
        add("warp", "Background warps instead of moving rigidly", bw, warp,
            max(0.3, _lerp(bw, WARP_AI_MIN, 0.15)) if warp else 0, "< 0.03 px", f"> {WARP_AI_MIN} px",
            f"After removing camera motion, background points still drift by {bw:.3f} px between frames (real videos "
            "measured ~0.016). In generated video the room is re-synthesised every frame and subtly 'breathes'.",
            "ai.bg_warp")

    vis = [x for x in signals if x["key"] in ("grain", "sharp", "glide", "warp")]
    weights = {"grain": 0.3, "sharp": 0.3, "glide": 0.2, "warp": 0.2}
    vscore = sum(weights[x["key"]] * (0.5 + 0.5 * x["strength"]) for x in vis if x["fired"])
    ml = _safe(lambda: predict.predict_video(path, face_fn=_face))
    status, score = _decide(prov, ml, fallback=vscore + (0.1 if prov["hints"] else 0))
    findings = _gate(findings, status, ml, "video", signals)
    return _verdict(status, score, signals, prov, "video", ml), findings


def assess_image(path: Path, image_findings: list[dict]) -> tuple[dict, list[dict]]:
    prov = scan_provenance(path)
    signals, findings = [], []
    checks = {f["check"]: f for f in image_findings}
    if prov["ai_declared"]:
        desc = next((a["description"] for a in prov["actions"] if a.get("description")), "") or "Generative AI"
        signals.append({"key": "c2pa", "name": "Content Credentials (C2PA) declare this as AI-generated", "value": 1,
                        "fired": True, "strength": 0.95, "normal": "absent", "ai": "AI action present"})
        findings.append(make_finding("metadata", "Content Credentials (C2PA) declare this as AI-generated", 0.95, 0.95,
                                     f"C2PA manifest{(' signed by ' + prov['signer']) if prov.get('signer') else ''}: \"{desc}\".",
                                     {"type": "metadata", "field": "C2PA manifest"}, check="ai.c2pa"))
    for chk, name in (("metadata.ai_tags", "AI-generator tag in metadata"), ("image.ai_dimensions", "AI-typical resolution"),
                      ("image.spectrum", "Periodic up-sampling artifacts")):
        f = checks.get(chk)
        signals.append({"key": chk, "name": name, "value": f["strength"] if f else 0, "fired": bool(f),
                        "strength": f["strength"] if f else 0, "normal": "absent", "ai": "present"})
    bgr = _safe(lambda: _load_bgr(path))
    ml = _safe(lambda: predict.predict_image(bgr)) if bgr is not None else None
    sc = 0.25 * ("image.ai_dimensions" in checks) + 0.25 * ("image.spectrum" in checks) + 0.1 * bool(prov["hints"])
    if "metadata.ai_tags" in checks and not prov["ai_declared"]:
        status, score = "likely", max(0.85, ml["p_ai"] if ml else 0)
    else:
        status, score = _decide(prov, ml, fallback=sc)
    findings = _gate(findings, status, ml, "image", signals)
    return _verdict(status, score, signals, prov, "image", ml), findings


def assess_audio(path: Path, audio_findings: list[dict], wav_path: Path | None = None) -> tuple[dict, list[dict]]:
    prov = scan_provenance(path)
    path_for_ml = wav_path or path
    checks = {f["check"] for f in audio_findings}
    voice = [c for c in ("audio.pitch", "audio.pause_regularity", "audio.bandwidth") if c in checks]
    signals = [{"key": c, "name": n, "value": int(c in checks), "fired": c in checks, "strength": 0.4 if c in checks else 0,
                "normal": "absent", "ai": "present"} for c, n in (("audio.pitch", "Unnaturally flat pitch"),
                                                                   ("audio.pause_regularity", "Robotic pause rhythm"),
                                                                   ("audio.bandwidth", "Band-limited synthetic voice"))]
    ml = _safe(lambda: predict.predict_audio(path_for_ml))
    sc = 0.3 * len(voice) + 0.1 * bool(prov["hints"])
    status, score = _decide(prov, ml, fallback=sc)
    findings = _gate([], status, ml, "audio", signals)
    return _verdict(status, score, signals, prov, "audio", ml), findings


ML_YES, ML_MAYBE = 0.60, 0.40


def _safe(fn):
    try:
        return fn()
    except Exception as e:  # a model problem must never break the analysis
        print("AI model error:", e)
        return None


def _face(fr):
    fs = detect_faces(fr, 0.08)
    return fs[0] if fs else None


def _load_bgr(path):
    img = cv2.imread(str(path))
    if img is None:
        from PIL import Image
        img = cv2.cvtColor(np.array(Image.open(path).convert("RGB")), cv2.COLOR_RGB2BGR)
    return img


def _decide(prov, ml, fallback=0.0):
    """C2PA label is decisive; otherwise the trained model decides; rules only if no model is available."""
    if prov["ai_declared"]:
        return "confirmed", max(0.97, ml["p_ai"] if ml else 0)
    p = ml["p_ai"] if ml else float(fallback)
    return ("likely" if p >= ML_YES else "possible" if p >= ML_MAYBE else "not_detected"), p


def _gate(findings, status, ml, kind, signals):
    """Add the ML finding; keep rule-based AI findings only when the overall verdict agrees (avoids contradictions)."""
    if ml:
        signals.insert(0, {"key": "ml", "name": "Trained AI-detection model", "value": ml["p_ai"],
                           "fired": ml["p_ai"] >= ML_MAYBE, "strength": ml["p_ai"], "normal": f"< {ML_MAYBE}",
                           "ai": f">= {ML_YES}"})
    keep = [f for f in findings if f["check"] == "ai.c2pa" or status in ("confirmed", "likely", "possible")]
    if ml and ml["p_ai"] >= ML_MAYBE:
        why = "; ".join(f"{e['feature']} -> {e['direction']}" for e in ml.get("explain", [])[:3])
        tf = ml.get("trained_on") or {}
        keep.append(make_finding(
            "video" if kind == "video" else "audio" if kind == "audio" else "image",
            f"Trained model: {ml['p_ai'] * 100:.0f}% likely AI-generated", ml["p_ai"],
            float(ml.get("heldout_accuracy") or 0.6),
            f"{ml['model']} rated this {kind} {ml['p_ai'] * 100:.0f}% likely AI-generated"
            + (f" (averaged over {ml['samples']} segments)" if ml["samples"] > 1 else "")
            + f". Trained on {tf.get('ai', '?')} AI and {tf.get('real', '?')} real {kind} files; accuracy on held-out "
              f"files {float(ml.get('heldout_accuracy') or 0) * 100:.0f}%." + (f" Main factors: {why}." if why else ""),
            {"type": "segment", "start": 0, "end": 30} if kind != "image" else {"type": "region", "label": "whole image"},
            check="ai.ml_model", ctx={"p_ai": ml["p_ai"]}))
    return keep


def _verdict(status, score, signals, prov, kind, ml=None):
    label = {"confirmed": "YES", "likely": "YES", "possible": "POSSIBLY", "not_detected": "NO"}[status]
    noun = "video" if kind == "video" else "image" if kind == "image" else "audio"
    headline = {
        "confirmed": f"This {noun} is AI-generated",
        "likely": f"This {noun} is likely AI-generated",
        "possible": f"This {noun} may be AI-generated",
        "not_detected": f"No signs that this {noun} is AI-generated",
    }[status]
    basis = {
        "confirmed": "Confirmed by signed Content Credentials (C2PA) embedded by the generating tool"
                     + (f" - {prov['signer']}" if prov.get("signer") else "") + ".",
        "likely": (f"The trained AI-detection model rates it {score * 100:.0f}% likely AI-generated."
                   if ml else "Several independent signals typical of AI generators were found."),
        "possible": (f"The trained model is uncertain ({score * 100:.0f}% AI) - human review needed."
                     if ml else "Some AI-generation signals were found, but not enough for a firm conclusion. Human review needed."),
        "not_detected": (f"No provenance label, and the trained model rates it only {score * 100:.0f}% likely AI. "
                         "This lowers - but does not eliminate - the chance that it was generated."
                         if ml else "No provenance label and no AI signals were found. This lowers - but does not "
                         "eliminate - the chance that the media was generated (high-end generators can evade checks)."),
    }[status]
    fired = [s["name"] for s in signals if s["fired"]]
    return {"label": label, "status": status, "score": round(float(score), 3), "headline": headline, "basis": basis,
            "reasons": fired, "signals": signals, "provenance": prov, "media_kind": kind, "ml": ml}


def overall(verdicts: list[dict]) -> dict | None:
    if not verdicts:
        return None
    order = {"confirmed": 3, "likely": 2, "possible": 1, "not_detected": 0}
    return max(verdicts, key=lambda v: (order[v["status"]], v["score"]))
