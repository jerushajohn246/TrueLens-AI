"""Text & context analysis: OCR, speech-to-text and claim-vs-media consistency checks."""
from __future__ import annotations

import re
import shutil
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np

from ..config import ENABLE_WHISPER, WHISPER_MODEL
from ..utils import fmt_ts, make_finding

STOP = set("""a an the and or but if of to in on at by for with from as is are was were be been being this that these
those it its he she they them his her their we our you your i me my not no yes do does did done have has had will would
can could should may might shall just very so than then there here what which who whom when where why how all any some
such into over under again also only own same too s t about up down out off video image photo audio clip shows show
showing seen see watch new news""".split())

RECENT_WORDS = ["today", "yesterday", "tonight", "this morning", "this evening", "just now", "right now", "live",
                "breaking", "few hours ago", "hours ago", "this week", "recently", "just happened", "abhi", "aaj", "kal"]

PLACES = ["delhi", "new delhi", "mumbai", "bangalore", "bengaluru", "chennai", "kolkata", "hyderabad", "pune",
          "ahmedabad", "jaipur", "lucknow", "kanpur", "nagpur", "indore", "bhopal", "patna", "surat", "kochi",
          "goa", "guwahati", "chandigarh", "noida", "gurgaon", "gurugram", "mysore", "mysuru", "kerala", "assam",
          "karnataka", "tamil nadu", "maharashtra", "gujarat", "punjab", "bihar", "uttar pradesh", "rajasthan",
          "india", "pakistan", "bangladesh", "nepal", "china", "usa", "america", "london", "dubai", "new york",
          "paris", "tokyo", "moscow", "ukraine", "russia", "israel", "gaza", "turkey", "syria", "iran", "sri lanka",
          "university", "campus"]

MONTHS = {m: i for i, m in enumerate(["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov",
                                      "dec"], 1)}

_whisper_model = None
_whisper_error = None


# ---------------------------------------------------------------- OCR
def tesseract_available() -> bool:
    if not shutil.which("tesseract"):
        return False
    try:
        import pytesseract  # noqa: F401
        return True
    except Exception:
        return False


def ocr_image(img_bgr: np.ndarray) -> str:
    if not tesseract_available():
        return ""
    import pytesseract
    g = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    if max(g.shape) < 1200:
        f = 1200 / max(g.shape)
        g = cv2.resize(g, None, fx=f, fy=f, interpolation=cv2.INTER_CUBIC)
    texts = []
    otsu = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    for variant, psm in ((g, 3), (g, 11), (otsu, 3)):
        try:
            raw = pytesseract.image_to_string(variant, config=f"--psm {psm}", timeout=20)
        except Exception:
            continue
        lines = []
        for ln in raw.splitlines():
            ln = ln.strip()
            alnum = sum(ch.isalnum() for ch in ln)
            if alnum >= 3 and alnum / max(len(ln), 1) > 0.5:     # drop OCR garbage like "-?)"
                lines.append(ln)
        texts.append(" ".join(lines))
    return max(texts, key=lambda t: sum(ch.isalnum() for ch in t)) if texts else ""


def ocr_file_image(path: Path) -> str:
    img = cv2.imread(str(path))
    if img is None:
        from PIL import Image
        img = cv2.cvtColor(np.array(Image.open(path).convert("RGB")), cv2.COLOR_RGB2BGR)
    return ocr_image(img)


def ocr_video(path: Path, times: list[float], max_frames: int = 8) -> list[dict]:
    if not tesseract_available() or not times:
        return []
    sel = times if len(times) <= max_frames else [times[int(i)] for i in np.linspace(0, len(times) - 1, max_frames)]
    cap = cv2.VideoCapture(str(path))
    out = []
    for t in sel:
        cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
        ok, fr = cap.read()
        if not ok:
            continue
        txt = ocr_image(fr)
        if txt.strip():
            out.append({"t": round(t, 2), "text": txt})
    cap.release()
    return out


# ---------------------------------------------------------------- speech-to-text
def whisper_status() -> str:
    if not ENABLE_WHISPER:
        return "disabled"
    try:
        import faster_whisper  # noqa: F401
        return "available"
    except Exception:
        return "not installed"


def transcribe(wav: Path) -> dict:
    """Transcribe with faster-whisper if installed. Returns {} when unavailable."""
    global _whisper_model, _whisper_error
    if whisper_status() != "available":
        return {}
    try:
        if _whisper_model is None:
            from faster_whisper import WhisperModel
            _whisper_model = WhisperModel(WHISPER_MODEL, device="cpu", compute_type="int8")
        segs, info = _whisper_model.transcribe(str(wav), beam_size=1, vad_filter=True)
        segments = [{"start": round(s.start, 2), "end": round(s.end, 2), "text": s.text.strip()} for s in segs]
        return {"language": info.language, "language_probability": round(info.language_probability, 2),
                "segments": segments, "text": " ".join(s["text"] for s in segments)}
    except Exception as e:  # model download blocked, OOM, etc.
        _whisper_error = str(e)
        return {"error": str(e)}


# ---------------------------------------------------------------- claim analysis helpers
def tokens(s: str) -> list[str]:
    return [w for w in re.findall(r"[a-zA-Zऀ-ॿ]{3,}", s.lower()) if w not in STOP]


def cosine(a: str, b: str) -> float:
    ta, tb = tokens(a), tokens(b)
    if not ta or not tb:
        return 0.0
    va, vb = {}, {}
    for w in ta:
        va[w[:6]] = va.get(w[:6], 0) + 1   # crude stemming: first 6 chars
    for w in tb:
        vb[w[:6]] = vb.get(w[:6], 0) + 1
    dot = sum(va[k] * vb.get(k, 0) for k in va)
    na = np.sqrt(sum(v * v for v in va.values()))
    nb = np.sqrt(sum(v * v for v in vb.values()))
    return float(dot / (na * nb + 1e-9))


def extract_dates(s: str) -> list[dict]:
    out = []
    low = s.lower()
    for m in re.finditer(r"\b(\d{1,2})[/\-.](\d{1,2})[/\-.]((?:19|20)\d{2})\b", low):
        d, mo, y = int(m.group(1)), int(m.group(2)), int(m.group(3))
        if mo > 12 and d <= 12:
            d, mo = mo, d
        out.append({"text": m.group(0), "year": y, "month": mo if 1 <= mo <= 12 else None})
    for m in re.finditer(r"\b((?:19|20)\d{2})[/\-.](\d{1,2})[/\-.](\d{1,2})\b", low):
        out.append({"text": m.group(0), "year": int(m.group(1)), "month": int(m.group(2))})
    for m in re.finditer(r"\b(\d{1,2})?\s*(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\.?,?\s+((?:19|20)\d{2})\b",
                         low):
        out.append({"text": m.group(0).strip(), "year": int(m.group(3)), "month": MONTHS[m.group(2)]})
    seen = {d["text"] for d in out}
    for m in re.finditer(r"\b(19[89]\d|20[0-3]\d)\b", low):
        if not any(m.group(0) in t for t in seen):
            out.append({"text": m.group(0), "year": int(m.group(0)), "month": None})
    return out


def find_places(s: str) -> set[str]:
    low = " " + re.sub(r"[^a-z ]", " ", s.lower()) + " "
    return {p for p in PLACES if f" {p} " in low}


def quoted(s: str) -> list[str]:
    return [q.strip() for q in re.findall(r"[\"“”']([^\"“”']{8,200})[\"“”']", s)]


# ---------------------------------------------------------------- main
def analyze_context(claim: str, ocr_texts: list[dict], transcript: dict, metadata_dates: list[dict],
                    now: datetime | None = None) -> dict:
    """Compare the uploader's claim/caption with what is actually in the media."""
    now = now or datetime.now(timezone.utc)
    findings: list[dict] = []
    claim = (claim or "").strip()
    ocr_all = " ".join(o["text"] for o in ocr_texts)
    tr_text = transcript.get("text", "") if transcript else ""
    info = {"claim": claim, "ocr_chars": len(ocr_all), "transcript_chars": len(tr_text),
            "transcript_language": transcript.get("language") if transcript else None}

    media_dates = []
    for o in ocr_texts:
        for d in extract_dates(o["text"]):
            media_dates.append({**d, "source": "visible text (OCR)", "where": o.get("where", "image"),
                                "t": o.get("t")})
    for d in extract_dates(tr_text):
        media_dates.append({**d, "source": "spoken audio (transcript)", "where": "audio"})
    info["dates_in_media"] = media_dates

    if not claim:
        if media_dates:
            info["note"] = "No claim supplied; dates found in media are listed for the investigator."
        return {"info": info, "findings": findings}

    low = claim.lower()
    recent = [w for w in RECENT_WORDS if re.search(rf"\b{re.escape(w)}\b", low)]
    claim_dates = extract_dates(claim)
    info["claim_recency_words"] = recent
    info["claim_dates"] = claim_dates
    claimed_year = max([d["year"] for d in claim_dates], default=now.year if recent else None)

    # 1. older visible / spoken date
    if claimed_year:
        older = [d for d in media_dates if d["year"] < claimed_year - (0 if recent else 1)]
        if recent:
            older = [d for d in media_dates if d["year"] < now.year]
        if older:
            d = min(older, key=lambda x: x["year"])
            loc = {"type": "text", "passage": d["text"], "source": d["source"]}
            if d.get("t") is not None:
                loc.update(type="timestamp", t=d["t"])
            findings.append(make_finding(
                "text", f"Media contains an older date ({d['text']})", 0.75, 0.7,
                f"The claim implies {'a recent event (' + ', '.join(recent) + ')' if recent else 'the year ' + str(claimed_year)}, "
                f"but the {d['source']} shows '{d['text']}'. The claim cannot be verified from the supplied media; "
                "the content may be old material presented as new.", loc, check="text.old_date"))

    # 2. metadata date vs recent claim
    for md in metadata_dates:
        try:
            dt = md["datetime"]
            age_days = (now - dt).days
        except Exception:
            continue
        if recent and age_days > 7:
            findings.append(make_finding(
                "metadata", f"Metadata date is {age_days} days old", float(np.clip(0.4 + age_days / 3650, 0, 0.8)),
                0.55, f"The claim says the media is recent ({', '.join(recent)}), but the {md['field']} metadata "
                f"says it was created on {dt.date()}. Metadata can be edited or reset, so treat as an indicator.",
                {"type": "metadata", "field": md["field"]}, check="text.metadata_date"))
            break
        if claim_dates and dt.year < min(d["year"] for d in claim_dates) - 1:
            findings.append(make_finding(
                "metadata", "Metadata date earlier than claimed date", 0.5, 0.5,
                f"Claim refers to {claim_dates[0]['text']}, metadata says {dt.date()}.",
                {"type": "metadata", "field": md["field"]}, check="text.metadata_date"))
            break

    # 3. location mismatch
    claim_places = find_places(claim)
    media_places = find_places(ocr_all + " " + tr_text)
    info["claim_places"], info["media_places"] = sorted(claim_places), sorted(media_places)
    if claim_places and media_places and not (claim_places & media_places):
        findings.append(make_finding(
            "text", "Location in claim differs from location in media", 0.6, 0.55,
            f"The claim mentions {', '.join(sorted(claim_places))}, but the media's text/speech mentions "
            f"{', '.join(sorted(media_places))}. The content may be from a different place.",
            {"type": "text", "passage": ", ".join(sorted(media_places))}, check="text.location"))

    # 4. caption vs transcript / OCR similarity
    if tr_text and len(tokens(tr_text)) >= 8:
        sim = cosine(claim, tr_text)
        info["claim_transcript_similarity"] = round(sim, 3)
        if sim < 0.08:
            findings.append(make_finding(
                "crossmodal", "Caption is not supported by the spoken content", 0.45, 0.45,
                f"Word overlap between the claim and the transcript is very low (similarity {sim:.2f}). Nothing that "
                "is said in the audio clearly relates to the claim.",
                {"type": "text", "passage": tr_text[:160]}, check="crossmodal.caption_transcript"))
    if ocr_all and len(tokens(ocr_all)) >= 5:
        info["claim_ocr_similarity"] = round(cosine(claim, ocr_all), 3)

    # 5. quoted statements must appear in the audio
    qs = quoted(claim)
    if qs and tr_text:
        for q in qs:
            sim = cosine(q, tr_text)
            hit = all(w[:6] in {t[:6] for t in tokens(tr_text)} for w in tokens(q)[:6]) if tokens(q) else False
            if not hit and sim < 0.35:
                findings.append(make_finding(
                    "crossmodal", "Quoted statement not found in the audio", 0.65, 0.55,
                    f"The claim attributes the words \"{q[:120]}\" to the speaker, but they do not appear in the "
                    "transcript of the audio track.", {"type": "text", "passage": q[:200]},
                    check="crossmodal.quote"))

    # 6. language mismatch (informative)
    lang = transcript.get("language") if transcript else None
    if lang and lang != "en" and re.fullmatch(r"[\x00-\x7F\s]+", claim) and not re.search(
            r"\b(hindi|kannada|tamil|telugu|urdu|bengali|marathi|translated)\b", low):
        findings.append(make_finding(
            "text", f"Spoken language is '{lang}' (claim written in English)", 0.15, 0.5,
            "Not suspicious by itself; check that any translation in the caption is accurate.",
            {"type": "text", "passage": f"language={lang}"}, check="text.language"))

    # 7. missing source
    if not re.search(r"https?://|source\s*:|via\s+@|credit|courtesy", low):
        findings.append(make_finding(
            "metadata", "No source provided with the claim", 0.2, 0.6,
            "The claim gives no original source, link or credit, so provenance cannot be traced.",
            {"type": "text", "passage": claim[:120]}, check="text.no_source"))
    return {"info": info, "findings": findings}


def metadata_dates_from(image_meta: dict | None, video_info: dict | None) -> list[dict]:
    out = []
    if image_meta:
        for field in ("DateTimeOriginal", "DateTime", "DateTimeDigitized"):
            v = image_meta.get("exif", {}).get(field)
            if v:
                try:
                    out.append({"field": f"EXIF {field}",
                                "datetime": datetime.strptime(v[:19], "%Y:%m:%d %H:%M:%S").replace(tzinfo=timezone.utc)})
                    break
                except Exception:
                    pass
    if video_info and video_info.get("creation_time"):
        v = video_info["creation_time"]
        try:
            out.append({"field": "container creation_time",
                        "datetime": datetime.fromisoformat(v.replace("Z", "+00:00")[:25]).astimezone(timezone.utc)})
        except Exception:
            pass
    return out


def transcript_ts_text(tr: dict) -> str:
    return "\n".join(f"[{fmt_ts(s['start'])}] {s['text']}" for s in tr.get("segments", []))
