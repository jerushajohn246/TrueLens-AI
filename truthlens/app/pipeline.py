"""Workflow manager: runs every analyzer, correlates modalities, aggregates evidence, writes the report."""
from __future__ import annotations

import json
import threading
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

from . import alternatives, correlation, integrity
from . import database as db
from .analyzers import ai_detector, crossmodal, text_analyzer
from .analyzers.audio_analyzer import analyze_audio
from .analyzers.image_analyzer import analyze_image
from .analyzers.video_analyzer import analyze_video
from .config import (APP_VERSION, AUDIO_SAMPLE_RATE, DISCLAIMER, EVIDENCE_DIR, MAX_VIDEO_SECONDS, REPORT_DIR,
                     UPLOAD_DIR)
from .scoring import aggregate
from .utils import extract_audio_wav, probe_media, to_jsonable

_run_lock = threading.Semaphore(1)   # one heavy analysis at a time (small cloud instances)

LIMITATIONS = [
    "Automated indicators can produce false positives (e.g. heavy compression, filters, screen recordings, "
    "repetitive textures) and false negatives (high-quality deepfakes may leave no classic traces).",
    "The checks are explainable forensic heuristics, not a deepfake classifier trained and validated on this "
    "specific media; scores are not calibrated probabilities.",
    "Missing metadata is common for genuine social-media content and is only a weak indicator.",
    "Lip-sync analysis needs a clearly visible, mostly frontal face; it is skipped otherwise.",
    "Speech-to-text and OCR may misread words, names, dates and non-English text.",
    f"Only the first {int(MAX_VIDEO_SECONDS)} seconds of a video are analysed.",
    "Lighting/shadow physics and weather/location consistency are only partially assessed and need human review.",
]


def tools_used(caps: dict) -> list[str]:
    t = ["Python / FastAPI", "OpenCV (Haar face detection, ORB features, image processing)",
         "Pillow (EXIF, JPEG quantisation)", "NumPy / SciPy (signal processing, STFT)",
         "FFmpeg (frame/audio extraction, container metadata)", "Matplotlib (spectrograms)",
         "SQLite (history)", "ReportLab (PDF)"]
    if caps.get("ocr"):
        t.append("Tesseract OCR")
    if caps.get("whisper") == "available":
        t.append("faster-whisper (speech-to-text)")
    return t


def capabilities() -> dict:
    return {"ocr": text_analyzer.tesseract_available(), "whisper": text_analyzer.whisper_status(),
            "ffmpeg": True, "version": APP_VERSION}


def start(aid: str):
    th = threading.Thread(target=_run, args=(aid,), daemon=True)
    th.start()


def _run(aid: str):
    with _run_lock:
        try:
            _analyze(aid)
        except Exception as e:  # never leave a job hanging
            traceback.print_exc()
            db.update(aid, status="error", message=f"Analysis failed: {e}", error=traceback.format_exc()[-2000:])
            db.log(aid, f"ERROR: {e}")


def _analyze(aid: str):
    rec = db.get(aid)
    files, claim = rec["files"], rec["claim"] or ""
    started = time.time()
    state = {"p": 2}

    def progress(msg: str, pct: int | None = None):
        if pct is not None:
            state["p"] = pct
        else:
            state["p"] = min(state["p"] + 2, 92)
        db.update(aid, status="running", progress=state["p"], message=msg)
        db.log(aid, msg)

    progress("Analysis started", 3)
    caps = capabilities()
    results: dict = {"image": [], "video": [], "audio": []}
    findings: list[dict] = []
    ocr_texts: list[dict] = []
    transcripts: list[dict] = []
    crossmodal_info: list[dict] = []
    applicable: set[str] = set()
    ai_verdicts: list[dict] = []
    image_meta, video_info = None, None
    n = len(files)

    for i, f in enumerate(files):
        path = UPLOAD_DIR / aid / f["stored_name"]
        base = 5 + int(80 * i / n)
        kind = f["media_type"]
        progress(f"Analysing {kind}: {f['filename']}", base)

        if kind == "image":
            applicable.add("image")
            r = analyze_image(path, aid, progress)
            r["file_id"] = f["id"]
            image_meta = image_meta or r["metadata"]
            if caps["ocr"]:
                progress("Running OCR on image")
                txt = text_analyzer.ocr_file_image(path)
                r["ocr_text"] = txt
                if txt:
                    ocr_texts.append({"text": txt, "where": f["filename"]})
            progress("Checking whether the image is AI-generated")
            v_ai, f_ai = ai_detector.assess_image(path, r["findings"])
            r["findings"] += f_ai
            r["ai_verdict"] = {**v_ai, "file": f["filename"], "file_id": f["id"]}
            ai_verdicts.append(r["ai_verdict"])
            _tag(r["findings"], f)
            findings += r["findings"]
            results["image"].append(r)

        elif kind == "video":
            applicable.add("video")
            r = analyze_video(path, aid, progress)
            r["file_id"] = f["id"]
            video_info = video_info or r["info"]
            progress("Checking whether the video is AI-generated (provenance + visual signals)")
            v_ai, f_ai = ai_detector.assess_video(path, cuts=r["info"].get("sudden_changes", 0))
            r["findings"] += f_ai
            r["ai_verdict"] = {**v_ai, "file": f["filename"], "file_id": f["id"]}
            ai_verdicts.append(r["ai_verdict"])
            if any(x["check"].startswith("ai.") for x in f_ai):
                applicable.add("metadata")
            sync = r.pop("sync_series")
            if r["info"].get("has_audio"):
                wav = EVIDENCE_DIR / aid / f"{f['id']}_audio.wav"
                progress("Extracting audio track")
                if extract_audio_wav(path, wav, AUDIO_SAMPLE_RATE, MAX_VIDEO_SECONDS):
                    applicable |= {"audio", "crossmodal"}
                    ar = analyze_audio(wav, aid, label=f"{f['id']}_video_audio", progress=progress)
                    _tag(ar["findings"], f)
                    findings += ar["findings"]
                    r["audio"] = ar
                    progress("Checking audio-video synchronisation")
                    cm = crossmodal.av_sync(sync, wav, aid)
                    _tag(cm["findings"], f)
                    findings += cm["findings"]
                    crossmodal_info.append({"file": f["filename"], **cm["info"]})
                    r["av_sync"] = cm["info"]
                    progress("Transcribing speech")
                    tr = text_analyzer.transcribe(wav)
                    if tr:
                        tr["file"] = f["filename"]
                        transcripts.append(tr)
                        r["transcript"] = tr
            if caps["ocr"]:
                progress("Reading on-screen text (OCR)")
                ocr_v = text_analyzer.ocr_video(path, [k["t"] for k in r["evidence"]["keyframes"]])
                for o in ocr_v:
                    o["where"] = f["filename"]
                ocr_texts += ocr_v
                r["ocr"] = ocr_v
            _tag(r["findings"], f)
            findings += r["findings"]
            results["video"].append(r)

        elif kind == "audio":
            applicable.add("audio")
            wav = EVIDENCE_DIR / aid / f"{f['id']}_audio.wav"
            progress("Decoding audio")
            if not extract_audio_wav(path, wav, AUDIO_SAMPLE_RATE, 600):
                raise ValueError(f"Could not decode audio file {f['filename']}")
            r = analyze_audio(wav, aid, label=f"{f['id']}_audio", progress=progress)
            r["file_id"] = f["id"]
            pr = probe_media(path)
            r["info"].update(audio_codec=pr.get("audio_codec"), bitrate_kbps=pr.get("bitrate_kbps"))
            progress("Transcribing speech")
            tr = text_analyzer.transcribe(wav)
            if tr:
                tr["file"] = f["filename"]
                transcripts.append(tr)
                r["transcript"] = tr
            v_ai, f_ai = ai_detector.assess_audio(path, r["findings"], wav_path=wav)
            r["findings"] += f_ai
            r["ai_verdict"] = {**v_ai, "file": f["filename"], "file_id": f["id"]}
            ai_verdicts.append(r["ai_verdict"])
            _tag(r["findings"], f)
            findings += r["findings"]
            results["audio"].append(r)

    # ---- text & context
    progress("Comparing claim with media content", 88)
    merged_tr = {"text": " ".join(t.get("text", "") for t in transcripts),
                 "language": next((t.get("language") for t in transcripts if t.get("language")), None)}
    ctx = text_analyzer.analyze_context(claim, ocr_texts, merged_tr,
                                        text_analyzer.metadata_dates_from(image_meta, video_info))
    if claim:
        applicable.add("text")
        if merged_tr["text"]:
            applicable.add("crossmodal")
    findings += ctx["findings"]

    # ---- innocent explanations -> correlation graph -> risk aggregation
    progress("Testing innocent explanations for each finding", 91)
    file_ctx = _file_context(results)
    alternatives.apply(findings, file_ctx)
    progress("Building evidence correlation graph", 94)
    counters = correlation.counter_evidence(results)
    graph = correlation.apply(findings, counters)
    graph["png"] = _render_graph(aid, graph)
    ai_overall = ai_detector.overall(ai_verdicts)
    agg = _ai_floor(_enrich(aggregate(findings, applicable), graph), ai_overall, findings)
    finished = datetime.now(timezone.utc)
    result = {
        "analysis_id": aid,
        "app": f"TruthLens {APP_VERSION}",
        "created_at": rec["created_at"],
        "completed_at": finished.isoformat(timespec="seconds"),
        "processing_seconds": round(time.time() - started, 1),
        "claim": claim,
        "files": files,
        "applicable_categories": sorted(applicable | {"metadata"}),
        **agg,
        "findings": sorted(findings, key=lambda x: -x["strength"] * (0.5 + 0.5 * x["confidence"])),
        "media": results,
        "text_analysis": {"ocr": ocr_texts, "transcripts": transcripts, "context": ctx["info"]},
        "crossmodal": crossmodal_info,
        "correlation": graph,
        "counter_evidence": counters,
        "ai_verdict": ai_overall,
        "ai_verdicts": ai_verdicts,
        "capabilities": caps,
        "tools_used": tools_used(caps),
        "limitations": LIMITATIONS + ([] if caps["ocr"] else ["OCR engine (Tesseract) not installed - visible text was not read."])
                       + ([] if caps["whisper"] == "available" else
                          ["Speech-to-text not enabled - transcript-based checks were skipped."]),
        "disclaimer": DISCLAIMER,
    }
    result = to_jsonable(result)
    head = db.log(aid, f"Analysis complete - risk {agg['risk_score']}/100 ({agg['risk_label']}), "
                       f"{len(findings)} indicator(s), {len(graph['hypotheses'])} correlated hypothesis(es)")
    s = integrity.seal(result, head, "initial analysis")
    save_result(aid, result)
    db.log(aid, f"Report sealed v{s['version']} - SHA-256 {s['report_sha256'][:16]}...")
    db.update(aid, status="done", progress=100, message="Analysis complete")


def _tag(fs, f):
    for x in fs:
        x["file"] = f["filename"]
        x["file_id"] = f["id"]


def save_result(aid: str, result: dict):
    db.update(aid, result=result, risk_score=result["risk_score"], risk_level=result["risk_level"])
    (REPORT_DIR / f"{aid}.json").write_text(json.dumps(result, indent=2))


def rescore(aid: str, reason: str = "investigator review") -> dict:
    rec = db.get(aid)
    res = rec["result"]
    graph = correlation.apply(res["findings"], res.get("counter_evidence", []))
    graph["png"] = _render_graph(aid, graph)
    res["correlation"] = graph
    agg = _ai_floor(_enrich(aggregate(res["findings"], set(res["applicable_categories"])), graph),
                    res.get("ai_verdict"), res["findings"])
    res.update(agg)
    res = to_jsonable(res)
    s = integrity.seal(res, db.chain_head(aid), reason)
    save_result(aid, res)
    db.log(aid, f"Report re-sealed v{s['version']} ({reason}) - SHA-256 {s['report_sha256'][:16]}...")
    return res


AI_FLOOR = {"confirmed": 85.0, "likely": 65.0, "possible": 35.0}


def _ai_floor(agg: dict, ai: dict | None, findings: list[dict]) -> dict:
    """An AI-generated verdict sets a minimum risk: generated media presented as real is itself the core risk.
    The floor is dropped if the investigator rejects every AI-generation finding."""
    agg["ai_floor"] = None
    if not ai or ai["status"] not in AI_FLOOR:
        return agg
    ai_f = [f for f in findings if f.get("check", "").startswith("ai.") or f.get("check") == "metadata.ai_tags"]
    if ai_f and all(f.get("review_status") == "rejected" for f in ai_f):
        return agg
    floor = AI_FLOOR[ai["status"]]
    if agg["risk_score"] < floor:
        old = f"{agg['risk_label']} ({agg['risk_score']}/100)"
        agg["ai_floor"] = {"from": agg["risk_score"], "to": floor, "reason": ai["headline"]}
        agg["risk_score"] = floor
        from .scoring import LEVELS
        agg["risk_label"], agg["risk_level"] = next((lab, key) for thr, lab, key in LEVELS if floor >= thr)
        agg["recommendation"] = "Human review required before this media is trusted or shared." if floor >= 50 else \
            "Human review recommended - verify the flagged evidence and the original source."
        agg["summary"] = agg["summary"].replace(old, f"{agg['risk_label']} ({floor}/100)")
    agg["summary"] = f"{ai['headline']} ({ai['basis']}) " + agg["summary"]
    return agg


def _enrich(agg: dict, graph: dict) -> dict:
    hy = [h for h in graph.get("hypotheses", []) if h["strength"] >= 0.4]
    if hy:
        agg["summary"] += " Independent checks agree on: " + "; ".join(
            f"{h['title'].lower()} ({len(h['modalities'])} modalities)" for h in hy[:3]) + "."
    return agg


def _file_context(results: dict) -> dict:
    ctx = {}
    for im in results.get("image", []):
        ctx[im["file_id"]] = {"kind": "image", "info": im["info"], "metadata": im.get("metadata", {})}
    for vd in results.get("video", []):
        info = dict(vd["info"])
        if vd.get("audio"):
            info.update({k: v for k, v in vd["audio"]["info"].items() if k not in info})
        ctx[vd["file_id"]] = {"kind": "video", "info": info, "av_sync": vd.get("av_sync"),
                              "provenance": (vd.get("ai_verdict") or {}).get("provenance")}
    for au in results.get("audio", []):
        ctx[au["file_id"]] = {"kind": "audio", "info": au["info"]}
    return ctx


def _render_graph(aid: str, graph: dict) -> str | None:
    if not graph["nodes"]:
        return None
    p = EVIDENCE_DIR / aid / "correlation_graph.png"
    p.parent.mkdir(parents=True, exist_ok=True)
    try:
        correlation.render_png(graph, p)
        return f"/evidence/{aid}/correlation_graph.png"
    except Exception:
        return None


def stored_path(aid: str, stored_name: str) -> Path:
    return UPLOAD_DIR / aid / stored_name
