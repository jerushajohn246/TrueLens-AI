"""Tamper-evident integrity layer.

1. Hash-chained audit trail - every history event stores  hash = SHA256(prev_hash | time | event).
   Editing or deleting any past event breaks every hash after it.
2. Report seal - SHA-256 digest of the canonical report JSON, bound to the evidence hashes and the
   audit-chain head, signed with an HMAC server key. Every change (e.g. investigator review) creates a
   new seal version; old versions stay listed, so an older PDF can still be matched.
3. QR code - printed on the PDF; scanning it opens /verify/<id>, which re-checks seal, chain and
   original files live.
"""
from __future__ import annotations

import hashlib
import hmac
import io
import json
import os
import secrets
from datetime import datetime, timezone

from .config import DATA_DIR

EXCLUDE = {"seal", "seals", "analysis_history", "correlation_graph_png"}
GENESIS = "0" * 64


def _key() -> bytes:
    env = os.getenv("TRUTHLENS_SECRET")
    if env:
        return env.encode()
    p = DATA_DIR / "secret.key"
    if not p.exists():
        p.write_text(secrets.token_hex(32))
    return p.read_text().strip().encode()


def chain_hash(prev: str, ts: str, event: str) -> str:
    return hashlib.sha256(f"{prev}|{ts}|{event}".encode()).hexdigest()


def verify_chain(history: list[dict]) -> dict:
    prev = GENESIS
    for i, h in enumerate(history):
        if not h.get("hash"):
            return {"ok": False, "detail": f"event {i + 1} has no hash (created before sealing was enabled)"}
        if h.get("prev_hash") != prev or chain_hash(prev, h["ts"], h["event"]) != h["hash"]:
            return {"ok": False, "broken_at": i + 1,
                    "detail": f"event {i + 1} ('{h['event'][:60]}') was modified, inserted or removed"}
        prev = h["hash"]
    return {"ok": True, "events": len(history), "head": prev, "detail": f"all {len(history)} events link correctly"}


def canonical(result: dict) -> str:
    clean = {k: v for k, v in result.items() if k not in EXCLUDE}
    return json.dumps(clean, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def digest(result: dict) -> str:
    return hashlib.sha256(canonical(result).encode()).hexdigest()


def _sign(aid, dg, head, version, sealed_at, files) -> str:
    msg = f"{aid}|{dg}|{head}|{version}|{sealed_at}|{','.join(files)}"
    return hmac.new(_key(), msg.encode(), hashlib.sha256).hexdigest()


def seal(result: dict, chain_head: str, reason: str) -> dict:
    seals = result.get("seals", [])
    version = len(seals) + 1
    sealed_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    files = [f["sha256"] for f in result["files"]]
    dg = digest(result)
    s = {"version": version, "report_sha256": dg, "chain_head": chain_head, "sealed_at": sealed_at,
         "evidence_sha256": files, "reason": reason,
         "signature": _sign(result["analysis_id"], dg, chain_head, version, sealed_at, files),
         "algorithm": "SHA-256 + HMAC-SHA256"}
    result["seals"] = seals + [s]
    result["seal"] = s
    return s


def check_seal(result: dict, s: dict) -> bool:
    exp = _sign(result["analysis_id"], s["report_sha256"], s["chain_head"], s["version"], s["sealed_at"],
                s["evidence_sha256"])
    return hmac.compare_digest(exp, s["signature"])


def short(s: dict) -> tuple[str, str]:
    return s["report_sha256"][:16], s["signature"][:16]


def verify_url(base_url: str, aid: str, s: dict) -> str:
    d, sg = short(s)
    return f"{base_url.rstrip('/')}/verify/{aid}?v={s['version']}&d={d}&s={sg}"


def qr_png(url: str) -> bytes:
    import qrcode
    img = qrcode.make(url, box_size=8, border=2)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def verify_uploaded_report(data: dict) -> dict:
    """Check a downloaded JSON report: does its content still match its own signed seal?"""
    s = data.get("seal")
    if not s:
        return {"ok": False, "detail": "File has no TruthLens seal."}
    dg = digest(data)
    content_ok = dg == s.get("report_sha256")
    sig_ok = False
    try:
        sig_ok = check_seal(data, s)
    except Exception:
        pass
    return {"ok": content_ok and sig_ok, "content_matches_seal": content_ok, "signature_valid": sig_ok,
            "version": s.get("version"), "computed_sha256": dg, "sealed_sha256": s.get("report_sha256"),
            "detail": ("Report is authentic and unmodified." if content_ok and sig_ok else
                       "Report content was modified after sealing." if not content_ok else
                       "Seal signature is invalid (forged or signed by a different server key).")}
