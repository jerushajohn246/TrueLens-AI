"""TruthLens FastAPI application - API + static web dashboard."""
from __future__ import annotations

import shutil
from datetime import datetime, timezone
from pathlib import Path

import json

from fastapi import FastAPI, File, Form, HTTPException, Request, Response, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from . import database as db
from . import integrity, pipeline
from .config import (ALLOWED_EXT, APP_VERSION, DISCLAIMER, EVIDENCE_DIR, MAX_UPLOAD_MB, REPORT_DIR, STATIC_DIR,
                     UPLOAD_DIR)
from .report_pdf import build_pdf
from .utils import detect_media_type, md5_file, new_id, sha256_file

app = FastAPI(title="TruthLens", version=APP_VERSION,
              description="Explainable multimodal digital forensics & deepfake indicator analysis")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
db.init_db()


@app.get("/api/health")
def health():
    return {"status": "ok", "version": APP_VERSION}


@app.get("/api/capabilities")
def caps():
    return {**pipeline.capabilities(), "max_upload_mb": MAX_UPLOAD_MB,
            "allowed": {k: sorted(v) for k, v in ALLOWED_EXT.items()}, "disclaimer": DISCLAIMER}


@app.post("/api/analyze")
async def analyze(files: list[UploadFile] = File(...), claim: str = Form("")):
    if not files or all(not f.filename for f in files):
        raise HTTPException(400, "Upload at least one file")
    if len(files) > 5:
        raise HTTPException(400, "Maximum 5 files per analysis")
    aid = new_id()
    folder = UPLOAD_DIR / aid
    folder.mkdir(parents=True, exist_ok=True)
    stored = []
    try:
        for uf in files:
            if not uf.filename:
                continue
            fid = new_id()
            ext = Path(uf.filename).suffix.lower()[:6]
            dest = folder / f"{fid}{ext}"
            size = 0
            with open(dest, "wb") as out:
                while chunk := await uf.read(1 << 20):
                    size += len(chunk)
                    if size > MAX_UPLOAD_MB * 1024 * 1024:
                        raise HTTPException(413, f"{uf.filename} is larger than {MAX_UPLOAD_MB} MB")
                    out.write(chunk)
            kind = detect_media_type(dest, uf.filename)
            if kind is None:
                raise HTTPException(415, f"Unsupported file type: {uf.filename}")
            dest.chmod(0o444)   # preserve the original - read only
            stored.append({
                "id": fid, "filename": Path(uf.filename).name, "stored_name": dest.name, "media_type": kind,
                "size_bytes": size, "sha256": sha256_file(dest), "md5": md5_file(dest),
                "content_type": uf.content_type,
                "uploaded_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            })
    except HTTPException:
        shutil.rmtree(folder, ignore_errors=True)
        raise
    db.create(aid, claim.strip()[:2000], stored)
    for s in stored:
        db.log(aid, f"Stored {s['filename']} ({s['media_type']}, sha256 {s['sha256'][:16]}...) as read-only original")
    pipeline.start(aid)
    return {"analysis_id": aid, "files": stored}


@app.get("/api/analyses")
def history():
    return db.list_all()


@app.get("/api/analyses/{aid}")
def get_analysis(aid: str):
    rec = db.get(aid)
    if not rec:
        raise HTTPException(404, "Analysis not found")
    rec.pop("error", None)
    return rec


class Review(BaseModel):
    status: str  # accepted | rejected | needs_review | pending
    note: str = ""


@app.post("/api/analyses/{aid}/findings/{fid}/review")
def review(aid: str, fid: str, body: Review):
    if body.status not in {"accepted", "rejected", "needs_review", "pending"}:
        raise HTTPException(400, "Invalid status")
    rec = db.get(aid)
    if not rec or not rec["result"]:
        raise HTTPException(404, "Analysis not found or not finished")
    res = rec["result"]
    f = next((x for x in res["findings"] if x["id"] == fid), None)
    if not f:
        raise HTTPException(404, "Finding not found")
    f["review_status"] = body.status
    f["review_note"] = body.note[:500]
    f["reviewed_at"] = db.now_iso()
    pipeline.save_result(aid, res)
    db.log(aid, f"Investigator marked '{f['title']}' as {body.status}" + (f" - note: {body.note[:120]}" if body.note else ""))
    res = pipeline.rescore(aid)
    f = next(x for x in res["findings"] if x["id"] == fid)
    return {"finding": f, "risk_score": res["risk_score"], "risk_label": res["risk_label"],
            "risk_level": res["risk_level"], "confidence": res["confidence"], "breakdown": res["breakdown"],
            "summary": res["summary"], "recommendation": res["recommendation"], "findings": res["findings"],
            "correlation": res["correlation"], "seal": res["seal"]}


@app.get("/api/analyses/{aid}/report.json")
def report_json(aid: str):
    rec = db.get(aid)
    if not rec or not rec["result"]:
        raise HTTPException(404, "Report not ready")
    data = {**rec["result"], "analysis_history": rec["history"]}
    return JSONResponse(data, headers={"Content-Disposition": f'attachment; filename="TruthLens_Report_{aid}.json"'})


@app.get("/api/analyses/{aid}/report.pdf")
def report_pdf(aid: str, request: Request):
    rec = db.get(aid)
    if not rec or not rec["result"]:
        raise HTTPException(404, "Report not ready")
    db.log(aid, f"PDF report exported (seal v{rec['result'].get('seal', {}).get('version', '?')})")
    rec = db.get(aid)
    path = build_pdf(rec, _base_url(request))
    return FileResponse(path, media_type="application/pdf", filename=path.name)


def _base_url(request: Request) -> str:
    # honour proxies such as Render (X-Forwarded-Proto) so QR links use https
    proto = request.headers.get("x-forwarded-proto", request.url.scheme)
    host = request.headers.get("x-forwarded-host", request.url.netloc)
    return f"{proto}://{host}"


@app.get("/api/analyses/{aid}/qr.png")
def qr(aid: str, request: Request):
    rec = db.get(aid)
    if not rec or not rec["result"] or not rec["result"].get("seal"):
        raise HTTPException(404, "No seal yet")
    url = integrity.verify_url(_base_url(request), aid, rec["result"]["seal"])
    return Response(integrity.qr_png(url), media_type="image/png", headers={"X-Verify-Url": url})


@app.get("/api/analyses/{aid}/verify")
def verify(aid: str, v: int | None = None, d: str = "", s: str = ""):
    """Live verification used by the QR code / verify page."""
    rec = db.get(aid)
    if not rec or not rec["result"]:
        return {"analysis_id": aid, "found": False, "overall": False,
                "checks": [{"name": "Analysis exists on this server", "ok": False,
                            "detail": "Not found. It may have been deleted, or this server's storage was reset."}]}
    res = rec["result"]
    seals = res.get("seals", [])
    cur = res.get("seal")
    checks = []
    dg = integrity.digest(res)
    content_ok = bool(cur) and dg == cur["report_sha256"]
    sig_ok = bool(cur) and integrity.check_seal(res, cur)
    checks.append({"name": "Stored report matches its latest seal", "ok": content_ok and sig_ok,
                   "detail": f"SHA-256 {dg[:24]}... " + ("matches seal v%d and the signature is valid." % cur["version"]
                                                          if content_ok and sig_ok else "DOES NOT MATCH - the stored report was altered.")})
    if d:
        match = next((x for x in seals if x["report_sha256"].startswith(d) and x["signature"].startswith(s)
                      and (v is None or x["version"] == v)), None)
        if match:
            latest = match["version"] == cur["version"]
            checks.append({"name": "Scanned report (QR) is a genuine TruthLens seal", "ok": True,
                           "detail": f"Matches seal v{match['version']} issued {match['sealed_at']} ({match['reason']}). "
                                     + ("This is the latest version." if latest else
                                        f"NOTE: the analysis was updated afterwards - latest is v{cur['version']} "
                                        f"({cur['reason']}). See audit trail.")})
        else:
            checks.append({"name": "Scanned report (QR) is a genuine TruthLens seal", "ok": False,
                           "detail": "No seal with this fingerprint exists - the printed report may be forged."})
    ch = integrity.verify_chain(rec["history"])
    checks.append({"name": "Audit trail hash chain intact", "ok": ch["ok"], "detail": ch["detail"]})
    head_ok = bool(cur) and any(h.get("hash") == cur["chain_head"] for h in rec["history"])
    checks.append({"name": "Seal is anchored in the audit trail", "ok": head_ok,
                   "detail": "Chain head recorded in the seal exists in the audit trail." if head_ok else
                             "Seal's chain head is missing from the trail."})
    for f in rec["files"]:
        p = UPLOAD_DIR / aid / f["stored_name"]
        if p.exists():
            now = sha256_file(p)
            checks.append({"name": f"Original evidence unchanged: {f['filename']}", "ok": now == f["sha256"],
                           "detail": f"SHA-256 {now[:24]}... " + ("matches the hash recorded at upload." if now == f["sha256"]
                                                                  else "DIFFERS from the hash recorded at upload!")})
        else:
            checks.append({"name": f"Original evidence present: {f['filename']}", "ok": False,
                           "detail": "Original file is missing from storage."})
    return {"analysis_id": aid, "found": True, "overall": all(c["ok"] for c in checks), "checks": checks,
            "current_seal": cur, "seal_versions": [{k: x[k] for k in ("version", "sealed_at", "reason", "report_sha256")}
                                                   for x in seals],
            "risk_score": res["risk_score"], "risk_label": res["risk_label"], "created_at": res["created_at"],
            "files": [{"filename": f["filename"], "sha256": f["sha256"]} for f in rec["files"]]}


@app.post("/api/analyses/{aid}/verify-file")
async def verify_file(aid: str, file: UploadFile = File(...)):
    """Is this file byte-for-byte the evidence that was analysed?"""
    rec = db.get(aid)
    if not rec:
        raise HTTPException(404)
    import hashlib
    h = hashlib.sha256()
    while chunk := await file.read(1 << 20):
        h.update(chunk)
    dg = h.hexdigest()
    match = next((f for f in rec["files"] if f["sha256"] == dg), None)
    return {"sha256": dg, "match": bool(match), "matched_file": match["filename"] if match else None,
            "detail": (f"Identical to analysed evidence '{match['filename']}'." if match else
                       "This file is NOT the analysed evidence (even a 1-byte change produces a different hash).")}


@app.post("/api/verify-report")
async def verify_report(file: UploadFile = File(...)):
    """Upload a downloaded JSON report to check it was not edited."""
    try:
        data = json.loads((await file.read()).decode("utf-8"))
    except Exception:
        raise HTTPException(400, "Not a valid JSON report")
    out = integrity.verify_uploaded_report(data)
    rec = db.get(data.get("analysis_id", "")) if data.get("analysis_id") else None
    if rec and rec["result"]:
        out["known_to_server"] = any(x["report_sha256"] == data.get("seal", {}).get("report_sha256")
                                     for x in rec["result"].get("seals", []))
    return out


@app.get("/verify/{aid}")
def verify_page(aid: str):
    return FileResponse(STATIC_DIR / "verify.html", media_type="text/html")


@app.get("/api/analyses/{aid}/media/{fid}")
def media(aid: str, fid: str):
    rec = db.get(aid)
    if not rec:
        raise HTTPException(404)
    f = next((x for x in rec["files"] if x["id"] == fid), None)
    if not f:
        raise HTTPException(404)
    return FileResponse(UPLOAD_DIR / aid / f["stored_name"], media_type=f.get("content_type") or None)


@app.delete("/api/analyses/{aid}")
def delete(aid: str):
    rec = db.get(aid)
    if not rec:
        raise HTTPException(404)
    if rec["status"] in ("queued", "running"):
        raise HTTPException(409, "Analysis still running")
    for d in (UPLOAD_DIR / aid, EVIDENCE_DIR / aid):
        if d.exists():
            for p in d.rglob("*"):
                try:
                    p.chmod(0o644)
                except Exception:
                    pass
            shutil.rmtree(d, ignore_errors=True)
    for p in REPORT_DIR.glob(f"*{aid}*"):
        p.unlink(missing_ok=True)
    db.delete(aid)
    return {"deleted": aid}


app.mount("/evidence", StaticFiles(directory=EVIDENCE_DIR), name="evidence")
app.mount("/", StaticFiles(directory=STATIC_DIR, html=True), name="static")
