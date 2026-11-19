"""Forensic PDF report generation (ReportLab)."""
from __future__ import annotations

from pathlib import Path
from xml.sax.saxutils import escape

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import (Image, KeepTogether, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table,
                                TableStyle)

import io as _io

from . import integrity
from .config import REPORT_DIR
from .utils import fmt_ts, url_to_path

NAVY = colors.HexColor("#14213d")
ACCENT = colors.HexColor("#2563eb")
LEVEL_COL = {"low": "#16a34a", "medium": "#d97706", "high": "#dc2626", "very_high": "#991b1b"}
SEV_COL = {"info": "#64748b", "low": "#16a34a", "medium": "#d97706", "high": "#dc2626"}


def _styles():
    ss = getSampleStyleSheet()
    ss.add(ParagraphStyle("H1x", parent=ss["Title"], fontSize=22, textColor=NAVY, spaceAfter=2))
    ss.add(ParagraphStyle("Sub", parent=ss["Normal"], fontSize=9, textColor=colors.HexColor("#475569"),
                          alignment=TA_CENTER))
    ss.add(ParagraphStyle("H2x", parent=ss["Heading2"], fontSize=13, textColor=NAVY, spaceBefore=10, spaceAfter=5))
    ss.add(ParagraphStyle("H3x", parent=ss["Heading3"], fontSize=10.5, textColor=NAVY, spaceBefore=6, spaceAfter=2))
    ss.add(ParagraphStyle("Body", parent=ss["Normal"], fontSize=9, leading=12.5))
    ss.add(ParagraphStyle("Small", parent=ss["Normal"], fontSize=7.8, leading=10, textColor=colors.HexColor("#334155")))
    ss.add(ParagraphStyle("Disc", parent=ss["Normal"], fontSize=8.5, leading=11.5, textColor=colors.HexColor("#7c2d12"),
                          backColor=colors.HexColor("#fff7ed"), borderColor=colors.HexColor("#fdba74"),
                          borderWidth=0.6, borderPadding=6, spaceBefore=6, spaceAfter=8))
    return ss


def _p(text, style):
    return Paragraph(escape(str(text)), style)


def _tbl(data, widths, header=True):
    t = Table(data, colWidths=widths, repeatRows=1 if header else 0)
    st = [("FONTSIZE", (0, 0), (-1, -1), 8), ("VALIGN", (0, 0), (-1, -1), "TOP"),
          ("GRID", (0, 0), (-1, -1), 0.3, colors.HexColor("#cbd5e1")),
          ("ROWBACKGROUNDS", (0, 1 if header else 0), (-1, -1), [colors.white, colors.HexColor("#f8fafc")]),
          ("LEFTPADDING", (0, 0), (-1, -1), 4), ("RIGHTPADDING", (0, 0), (-1, -1), 4)]
    if header:
        st += [("BACKGROUND", (0, 0), (-1, 0), NAVY), ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
               ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold")]
    t.setStyle(TableStyle(st))
    return t


def _img(url, max_w, max_h):
    p = url_to_path(url) if url.startswith("/evidence/") else Path(url)
    if not p.exists():
        return None
    try:
        im = Image(str(p))
        r = min(max_w / im.imageWidth, max_h / im.imageHeight)
        im.drawWidth, im.drawHeight = im.imageWidth * r, im.imageHeight * r
        return im
    except Exception:
        return None


def _loc(loc: dict) -> str:
    if not loc:
        return "-"
    t = loc.get("type")
    if t == "timestamp":
        s = f"t = {fmt_ts(loc['t'])}"
        if loc.get("frame") is not None:
            s += f" (frame {loc['frame']})"
        return s
    if t == "segment":
        return f"{fmt_ts(loc.get('start', 0))} - {fmt_ts(loc.get('end', 0))}"
    if t == "region":
        b = loc.get("bbox")
        return f"{loc.get('label', 'region')}" + (f" at x={b[0]}, y={b[1]}, {b[2]}x{b[3]} px" if b else "")
    if t == "metadata":
        return f"metadata field: {loc.get('field')}"
    if t == "text":
        return f"text: \"{str(loc.get('passage', ''))[:80]}\""
    return str(loc)


def build_pdf(rec: dict, base_url: str = "http://localhost:8000") -> Path:
    res = rec["result"]
    ss = _styles()
    out = REPORT_DIR / f"TruthLens_Report_{res['analysis_id']}.pdf"
    doc = SimpleDocTemplate(str(out), pagesize=A4, leftMargin=16 * mm, rightMargin=16 * mm,
                            topMargin=14 * mm, bottomMargin=14 * mm,
                            title=f"TruthLens Forensic Report {res['analysis_id']}", author="TruthLens")
    W = A4[0] - 32 * mm
    el = []
    el.append(_p("TruthLens Forensic Report", ss["H1x"]))
    el.append(_p("Multimodal Digital Forensics & Deepfake Indicator Analysis", ss["Sub"]))
    el.append(Spacer(1, 6))
    ai = res.get("ai_verdict")
    if ai:
        kind = {"image": "IMAGE", "audio": "AUDIO"}.get(ai.get("media_kind"), "VIDEO")
        col = {"confirmed": "#b91c1c", "likely": "#dc2626", "possible": "#d97706", "not_detected": "#15803d"}[ai["status"]]
        bg = {"confirmed": "#fef2f2", "likely": "#fef2f2", "possible": "#fffbeb", "not_detected": "#f0fdf4"}[ai["status"]]
        prov = ai.get("provenance") or {}
        lines = [f"<font size=15 color='{col}'><b>AI-GENERATED {kind}: {ai['label']}</b></font>",
                 f"<b>{escape(ai['headline'])}</b> (AI likelihood {ai['score'] * 100:.0f}%)", escape(ai["basis"])]
        if prov.get("ai_declared"):
            lines.append("<b>Content Credentials:</b> " + escape((prov.get("signer") or "") + " - " + "; ".join(
                f"{a['action']}: {a['description']}" for a in prov.get("actions", []))))
        fired = [s_["name"] for s_ in ai.get("signals", []) if s_["fired"]]
        if fired:
            lines.append("<b>Signals:</b> " + escape("; ".join(fired)))
        box = Table([[Paragraph("<br/>".join(lines), ss["Body"])]], colWidths=[W])
        box.setStyle(TableStyle([("BOX", (0, 0), (-1, -1), 1.4, colors.HexColor(col)),
                                 ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor(bg)),
                                 ("LEFTPADDING", (0, 0), (-1, -1), 8), ("TOPPADDING", (0, 0), (-1, -1), 7),
                                 ("BOTTOMPADDING", (0, 0), (-1, -1), 7)]))
        el += [box, Spacer(1, 4)]
    el.append(_p(res["disclaimer"], ss["Disc"]))

    lc = LEVEL_COL.get(res["risk_level"], "#334155")
    summary = [
        ["Analysis ID", res["analysis_id"], "Overall risk", Paragraph(
            f"<font color='{lc}'><b>{res['risk_score']}/100 - {escape(res['risk_label'])}</b></font>", ss["Body"])],
        ["Created", res["created_at"], "Confidence", f"{res['confidence'] * 100:.0f}%"],
        ["Completed", res["completed_at"], "Indicators", str(len(res["findings"]))],
        ["Processing time", f"{res['processing_seconds']} s", "Engine", res["app"]],
    ]
    t = _tbl(summary, [W * .17, W * .33, W * .17, W * .33], header=False)
    t.setStyle(TableStyle([("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"), ("FONTNAME", (2, 0), (2, -1), "Helvetica-Bold")]))
    el += [t, Spacer(1, 6)]
    seal = res.get("seal")
    if seal:
        url = integrity.verify_url(base_url, res["analysis_id"], seal)
        qr = Image(_io.BytesIO(integrity.qr_png(url)), width=30 * mm, height=30 * mm)
        txt = Paragraph(
            f"<b>Tamper-proof seal v{seal['version']}</b> ({escape(seal['reason'])}, {escape(seal['sealed_at'])})<br/>"
            f"<font size=7>Report SHA-256: {seal['report_sha256']}<br/>Audit-chain head: {seal['chain_head']}<br/>"
            f"Signature (HMAC-SHA256): {seal['signature'][:32]}...</font><br/>"
            f"<font size=8>Scan the QR code or open <font color='#2563eb'>{escape(url)}</font> to verify that this report, "
            "its audit trail and the original evidence have not been altered.</font>", ss["Small"])
        st = Table([[qr, txt]], colWidths=[34 * mm, W - 34 * mm])
        st.setStyle(TableStyle([("VALIGN", (0, 0), (-1, -1), "MIDDLE"), ("BOX", (0, 0), (-1, -1), 0.8, colors.HexColor("#16a34a")),
                                ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#f0fdf4")),
                                ("LEFTPADDING", (0, 0), (-1, -1), 6), ("TOPPADDING", (0, 0), (-1, -1), 5),
                                ("BOTTOMPADDING", (0, 0), (-1, -1), 5)]))
        el += [st, Spacer(1, 4)]
    el.append(_p("Summary", ss["H2x"]))
    el.append(_p(res["summary"], ss["Body"]))
    el.append(Spacer(1, 3))
    el.append(Paragraph(f"<b>Recommendation:</b> {escape(res['recommendation'])}", ss["Body"]))
    hyps = (res.get("correlation") or {}).get("hypotheses", [])
    if hyps:
        el.append(Spacer(1, 3))
        el.append(Paragraph("<b>Correlated hypotheses:</b> " + "; ".join(
            f"{escape(h['title'])} ({h['strength'] * 100:.0f}%, {escape(', '.join(h['modalities']))}{', ' + escape(h['when']) if h.get('when') else ''})"
            for h in hyps), ss["Body"]))
    if res.get("claim"):
        el.append(Spacer(1, 3))
        el.append(Paragraph(f"<b>Claim supplied by uploader:</b> \"{escape(res['claim'])}\"", ss["Body"]))

    el.append(_p("Evidence files", ss["H2x"]))
    rows = [["File", "Type", "Size", "SHA-256 hash", "Uploaded"]]
    for f in res["files"]:
        rows.append([Paragraph(escape(f["filename"]), ss["Small"]), f["media_type"], f"{f['size_bytes'] / 1e6:.2f} MB",
                     Paragraph(f["sha256"], ss["Small"]), Paragraph(f["uploaded_at"], ss["Small"])])
    el.append(_tbl(rows, [W * .22, W * .09, W * .1, W * .39, W * .2]))

    el.append(_p("Risk score breakdown", ss["H2x"]))
    rows = [["Category", "Indicators", "Points", "Max", "Counted"]]
    for b in res["breakdown"]:
        rows.append([b["label"], str(b["indicators"]), f"{b['points']}", str(b["max_points"]),
                     "yes" if b["applicable"] else "n/a"])
    el.append(_tbl(rows, [W * .46, W * .14, W * .13, W * .12, W * .15]))
    el.append(Spacer(1, 3))
    el.append(_p("The score represents the presence of manipulation indicators, normalised over the categories that "
                 "apply to the supplied media. It is not a legal or final authenticity verdict.", ss["Small"]))

    el.append(_p("Findings", ss["H2x"]))
    if not res["findings"]:
        el.append(_p("No indicators were raised by the automated checks.", ss["Body"]))
    for i, f in enumerate(res["findings"], 1):
        sc = SEV_COL.get(f["severity"], "#334155")
        rv = f.get("review_status", "pending")
        block = [Paragraph(f"<b>{i}. {escape(f['title'])}</b> &nbsp;<font color='{sc}' size=8>[{f['severity'].upper()}]"
                           f"</font> &nbsp;<font size=8 color='#475569'>strength {f['strength']:.2f} | confidence "
                           f"{f['confidence'] * 100:.0f}% | review: {rv}</font>", ss["H3x"]),
                 Paragraph(f"<b>What / where:</b> {escape(f['category'])} - {escape(_loc(f['location']))}"
                           f"{' - file ' + escape(f.get('file', '')) if f.get('file') else ''}", ss["Small"]),
                 Paragraph(f"<b>Why suspicious:</b> {escape(f['explanation'])}", ss["Body"])]
        if f.get("original_confidence") is not None:
            c = f.get("correlation") or {}
            trail = f"detector {f['original_confidence'] * 100:.0f}%"
            if f.get("base_confidence") != f["original_confidence"]:
                trail += f" -> innocent-cause check {f['base_confidence'] * 100:.0f}%"
            if c.get("delta"):
                trail += f" -> correlation {c['delta'] * 100:+.0f}%"
                if c.get("independent_modalities"):
                    trail += f" (corroborated by {', '.join(c['independent_modalities'])})"
            trail += f" -> final {f['confidence'] * 100:.0f}%"
            block.append(Paragraph(f"<b>Confidence trail:</b> {escape(trail)}", ss["Small"]))
        alts = f.get("alternatives") or []
        if alts:
            col = {"likely": "#b45309", "possible": "#1d4ed8", "unlikely": "#15803d", "unverifiable": "#52525b"}
            block.append(Paragraph("<b>Could this be innocent?</b> " + " ".join(
                f"<font color='{col[a['status']]}'><b>[{a['status'].upper()}]</b></font> {escape(a['cause'])}"
                + (f" - <i>{escape(a['detail'])}</i>" if a.get("detail") else "") + "."
                for a in alts), ss["Small"]))
        if f.get("review_note"):
            block.append(Paragraph(f"<b>Investigator note:</b> {escape(f['review_note'])}", ss["Small"]))
        if f.get("evidence"):
            block.append(Paragraph("<b>Evidence:</b> " + escape(", ".join(e.split('/')[-1] for e in f["evidence"])),
                                   ss["Small"]))
            im = _img(f["evidence"][0], W * 0.7, 62 * mm)
            if im:
                block += [Spacer(1, 3), im]
        block.append(Spacer(1, 5))
        el.append(KeepTogether(block))

    # media details
    media = res.get("media", {})
    for r in media.get("image", []):
        el.append(PageBreak())
        el.append(_p("Image analysis evidence", ss["H2x"]))
        info = r["info"]
        el.append(_p(", ".join(f"{k}: {v}" for k, v in info.items() if v is not None and not isinstance(v, (dict, list))), ss["Small"]))
        imgs = [(r["evidence"].get(k), lab) for k, lab in (("overlay", "Flagged regions"), ("ela", "Error Level Analysis"),
                                                           ("noise", "Noise residual map"), ("spectrum", "Frequency spectrum"))]
        cells = []
        for u, lab in imgs:
            im = _img(u, W * .47, 60 * mm) if u else None
            if im:
                cells.append([im, _p(lab, ss["Small"])])
        rows = []
        for j in range(0, len(cells), 2):
            pair = cells[j:j + 2]
            rows.append([c[0] for c in pair] + [""] * (2 - len(pair)))
            rows.append([c[1] for c in pair] + [""] * (2 - len(pair)))
        if rows:
            el.append(Table(rows, colWidths=[W / 2, W / 2]))
        exif = r["metadata"].get("exif", {})
        if exif:
            el.append(_p("EXIF metadata", ss["H3x"]))
            rows = [["Field", "Value"]] + [[k, Paragraph(escape(str(v))[:150], ss["Small"])]
                                           for k, v in list(exif.items())[:25]]
            el.append(_tbl(rows, [W * .3, W * .7]))
        if r.get("ocr_text"):
            el.append(_p("Text found in image (OCR)", ss["H3x"]))
            el.append(_p(r["ocr_text"][:1500], ss["Small"]))

    for r in media.get("video", []):
        el.append(PageBreak())
        el.append(_p("Video analysis evidence", ss["H2x"]))
        el.append(_p(", ".join(f"{k}: {v}" for k, v in r["info"].items() if not isinstance(v, (dict, list)) and v is not None),
                     ss["Small"]))
        sus = r["evidence"].get("suspicious_frames", [])
        if sus:
            el.append(_p("Suspicious frames", ss["H3x"]))
            for s in sus[:4]:
                im = _img(s["url"], W * .6, 55 * mm)
                if im:
                    el.append(KeepTogether([im, _p(f"t = {fmt_ts(s['t'])}, frame {s['frame']}: {s['reason']}",
                                                   ss["Small"]), Spacer(1, 4)]))
        if r.get("av_sync", {}).get("plot"):
            el.append(_p("Audio-video synchronisation", ss["H3x"]))
            im = _img(r["av_sync"]["plot"], W, 50 * mm)
            if im:
                el.append(im)
            a = r["av_sync"]
            el.append(_p(f"Correlation at zero lag: {a.get('corr_at_zero')}, best correlation {a.get('best_corr')} at "
                         f"lag {a.get('best_lag_sec')} s. Mismatch windows: {a.get('mismatch_windows')}", ss["Small"]))
        elif r.get("av_sync"):
            el.append(_p(f"Audio-video sync: {r['av_sync'].get('status')}", ss["Small"]))
        if r.get("audio", {}).get("evidence", {}).get("spectrogram"):
            el.append(_p("Audio track spectrogram", ss["H3x"]))
            im = _img(r["audio"]["evidence"]["spectrogram"], W, 62 * mm)
            if im:
                el.append(im)
        if r.get("transcript", {}).get("segments"):
            el.append(_p("Transcript", ss["H3x"]))
            for sgm in r["transcript"]["segments"][:40]:
                el.append(_p(f"[{fmt_ts(sgm['start'])}] {sgm['text']}", ss["Small"]))
        if r.get("ocr"):
            el.append(_p("On-screen text (OCR)", ss["H3x"]))
            for o in r["ocr"]:
                el.append(_p(f"[{fmt_ts(o['t'])}] {o['text'][:300]}", ss["Small"]))

    for r in media.get("audio", []):
        el.append(PageBreak())
        el.append(_p("Audio analysis evidence", ss["H2x"]))
        el.append(_p(", ".join(f"{k}: {v}" for k, v in r["info"].items() if not isinstance(v, (dict, list))), ss["Small"]))
        if r["evidence"].get("spectrogram"):
            im = _img(r["evidence"]["spectrogram"], W, 70 * mm)
            if im:
                el.append(im)
        if r.get("transcript", {}).get("segments"):
            el.append(_p("Transcript", ss["H3x"]))
            for sgm in r["transcript"]["segments"][:40]:
                el.append(_p(f"[{fmt_ts(sgm['start'])}] {sgm['text']}", ss["Small"]))

    g = res.get("correlation") or {}
    if g.get("nodes"):
        el.append(PageBreak())
        el.append(_p("Evidence correlation graph", ss["H2x"]))
        el.append(_p("Findings are linked when independent checks agree (same moment, same region, or the same "
                     "manipulation hypothesis from a different modality) and when counter-evidence or a likely innocent "
                     "cause argues against them. Agreement raises a finding's confidence; contradiction lowers it.", ss["Small"]))
        if g.get("png"):
            im = _img(g["png"], W, 120 * mm)
            if im:
                el.append(im)
        if g.get("hypotheses"):
            rows = [["Hypothesis", "Strength", "Modalities", "When", "Indicators"]]
            for h in g["hypotheses"]:
                rows.append([Paragraph(escape(h["title"]), ss["Small"]), f"{h['strength'] * 100:.0f}%",
                             Paragraph(escape(", ".join(h["modalities"])), ss["Small"]), h.get("when") or "-",
                             str(len(h["members"]))])
            el.append(Spacer(1, 4))
            el.append(_tbl(rows, [W * .36, W * .12, W * .22, W * .18, W * .12]))
        lab = {n["id"]: n["label"] for n in g["nodes"]}
        rows = [["Link", "From", "To", "Reason"]]
        for e in g["edges"][:40]:
            rows.append(["supports" if e["type"] == "support" else "contradicts",
                         Paragraph(escape(lab.get(e["source"], "")[:60]), ss["Small"]),
                         Paragraph(escape(lab.get(e["target"], "")[:60]), ss["Small"]),
                         Paragraph(escape(e["reason"][:140]), ss["Small"])])
        el.append(Spacer(1, 6))
        el.append(_tbl(rows, [W * .12, W * .26, W * .26, W * .36]))

    el.append(PageBreak())
    el.append(_p("Tools and models used", ss["H2x"]))
    for t in res["tools_used"]:
        el.append(_p("- " + t, ss["Body"]))
    el.append(_p("Checks performed", ss["H3x"]))
    checks = sorted({c for k in ("image", "video", "audio") for r in media.get(k, []) for c in r.get("checks_run", [])}
                    | {c for r in media.get("video", []) for c in r.get("audio", {}).get("checks_run", [])})
    el.append(_p(", ".join(checks) + ", Claim/context consistency", ss["Small"]))
    el.append(_p("Limitations and uncertainty", ss["H2x"]))
    for l in res["limitations"]:
        el.append(_p("- " + l, ss["Body"]))
    el.append(_p("Analysis history (audit trail)", ss["H2x"]))
    rows = [["Time (UTC)", "Event", "Chain hash"]] + [[Paragraph(h["ts"], ss["Small"]), Paragraph(escape(h["event"]), ss["Small"]),
                                                      Paragraph(f"<font face='Courier' size=6>{(h.get('hash') or '-')[:16]}</font>", ss["Small"])]
                                                     for h in rec.get("history", [])[-60:]]
    el.append(_tbl(rows, [W * .22, W * .58, W * .20]))
    ch = integrity.verify_chain(rec.get("history", []))
    el.append(_p(("Hash chain verified: " if ch["ok"] else "HASH CHAIN BROKEN: ") + ch["detail"], ss["Small"]))
    el.append(Spacer(1, 8))
    el.append(_p("Human review recommendation: " + res["recommendation"], ss["Disc"]))

    def footer(canvas, d):
        canvas.saveState()
        canvas.setFont("Helvetica", 7)
        canvas.setFillColor(colors.HexColor("#64748b"))
        canvas.drawString(16 * mm, 8 * mm, f"TruthLens report {res['analysis_id']}  -  indicators, not a verdict")
        canvas.drawRightString(A4[0] - 16 * mm, 8 * mm, f"Page {d.page}")
        canvas.restoreState()

    doc.build(el, onFirstPage=footer, onLaterPages=footer)
    return out
