"""End-to-end smoke tests: upload media through the API and check the forensic report."""
import io
import time

import cv2
import numpy as np
import soundfile as sf
from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def _wait(aid, timeout=120):
    t0 = time.time()
    while time.time() - t0 < timeout:
        r = client.get(f"/api/analyses/{aid}").json()
        if r["status"] in ("done", "error"):
            return r
        time.sleep(0.3)
    raise TimeoutError


def _jpeg(img, q=92):
    return io.BytesIO(cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, q])[1].tobytes())


def test_health():
    assert client.get("/api/health").json()["status"] == "ok"


def test_image_copy_move_detected():
    rng = np.random.default_rng(0)
    img = np.clip(rng.normal(120, 25, (480, 640, 3)), 0, 255).astype(np.uint8)
    img = cv2.GaussianBlur(img, (3, 3), 0)
    patch = img[60:200, 60:220].copy()
    img[260:400, 400:560] = patch                 # clone
    r = client.post("/api/analyze", files={"files": ("clone.jpg", _jpeg(img), "image/jpeg")},
                    data={"claim": "Photo taken today in Delhi"})
    assert r.status_code == 200
    rec = _wait(r.json()["analysis_id"])
    assert rec["status"] == "done", rec.get("message")
    res = rec["result"]
    assert any(f["check"] == "image.copy_move" for f in res["findings"])
    assert 0 <= res["risk_score"] <= 100
    assert len(res["files"][0]["sha256"]) == 64
    pdf = client.get(f"/api/analyses/{res['analysis_id']}/report.pdf")
    assert pdf.status_code == 200 and pdf.content[:4] == b"%PDF"


def test_audio_splice_detected():
    sr = 32000
    t = np.arange(sr * 8) / sr
    y = 0.3 * np.sin(2 * np.pi * 150 * t) * (np.sin(2 * np.pi * 3 * t) > 0) + np.random.default_rng(1).normal(0, 0.004, len(t))
    y[int(4 * sr):int(4.3 * sr)] = 0.0             # digital silence
    buf = io.BytesIO()
    sf.write(buf, y.astype(np.float32), sr, format="WAV")
    buf.seek(0)
    r = client.post("/api/analyze", files={"files": ("a.wav", buf, "audio/wav")})
    rec = _wait(r.json()["analysis_id"])
    assert rec["status"] == "done", rec.get("message")
    assert any(f["check"] == "audio.digital_silence" for f in rec["result"]["findings"])


def test_review_updates_score():
    rows = client.get("/api/analyses").json()
    done = next(a for a in rows if a["status"] == "done")
    rec = client.get(f"/api/analyses/{done['id']}").json()
    f = rec["result"]["findings"][0]
    r = client.post(f"/api/analyses/{done['id']}/findings/{f['id']}/review", json={"status": "rejected", "note": "test"})
    assert r.status_code == 200
    assert r.json()["finding"]["review_status"] == "rejected"


def test_rejects_unknown_type():
    r = client.post("/api/analyze", files={"files": ("x.txt", io.BytesIO(b"hello world"), "text/plain")})
    assert r.status_code == 415


# ---------------- v1.1 features
import json as _json
import sqlite3 as _sqlite3


def _done_with_findings():
    rows = client.get("/api/analyses").json()
    for a in rows:
        if a["status"] == "done":
            rec = client.get(f"/api/analyses/{a['id']}").json()
            if rec["result"]["findings"]:
                return rec
    raise AssertionError("no finished analysis")


def test_innocent_explanations_and_graph_present():
    rec = _done_with_findings()
    res = rec["result"]
    for f in res["findings"]:
        assert "alternatives" in f and "base_confidence" in f and "original_confidence" in f
        for a in f["alternatives"]:
            assert a["status"] in ("likely", "possible", "unlikely", "unverifiable")
    g = res["correlation"]
    assert {"nodes", "edges", "hypotheses"} <= set(g)


def test_seal_verifies_and_detects_tampering():
    rec = _done_with_findings()
    aid = rec["id"]
    v = client.get(f"/api/analyses/{aid}/verify").json()
    assert v["overall"] is True
    # genuine downloaded report verifies, edited one fails
    rep = client.get(f"/api/analyses/{aid}/report.json").json()
    ok = client.post("/api/verify-report", files={"file": ("r.json", io.BytesIO(_json.dumps(rep).encode()))}).json()
    assert ok["ok"] is True
    rep["risk_score"] = 0.0
    bad = client.post("/api/verify-report", files={"file": ("r.json", io.BytesIO(_json.dumps(rep).encode()))}).json()
    assert bad["ok"] is False
    # QR image
    q = client.get(f"/api/analyses/{aid}/qr.png")
    assert q.status_code == 200 and q.content[:4] == b"\x89PNG"
    # tamper with the audit log in the database -> chain broken
    from app.config import DB_PATH
    con = _sqlite3.connect(DB_PATH)
    con.execute("UPDATE history SET event = event || ' (edited)' WHERE id = (SELECT MIN(id) FROM history WHERE analysis_id=?)", (aid,))
    con.commit()
    v = client.get(f"/api/analyses/{aid}/verify").json()
    assert v["overall"] is False


# ---------------- v1.2 AI-generated verdict
def test_c2pa_ai_label_detected(tmp_path):
    from app.analyzers.ai_detector import scan_provenance
    desc = b"Created by Google Generative AI."
    blob = (b"\x00\x00\x00\x18ftypmp42" + b"jumb" + b"jumdc2pa" + b"c2pa.actions.v2" + b"c2pa.created" + b"kdescription" +
            b"x" + bytes([len(desc)]) + desc + b"qdigitalSourceTypexFhttp://cv.iptc.org/newscodes/digitalsourcetype/trainedAlgorithmicMedia" +
            b"\x0cGoogle C2PA Media Services 1P" + b"\x00" * 2000)
    p = tmp_path / "x.mp4"
    p.write_bytes(blob)
    prov = scan_provenance(p)
    assert prov["ai_declared"] is True
    assert prov["actions"][0]["description"] == "Created by Google Generative AI."


def test_ai_verdict_in_result():
    rows = client.get("/api/analyses").json()
    rec = client.get(f"/api/analyses/{rows[0]['id']}").json()
    v = rec["result"]["ai_verdict"]
    assert v["label"] in ("YES", "NO", "POSSIBLY") and 0 <= v["score"] <= 1


# ---------------- v1.3 trained AI-detection models
def test_trained_models_load_and_predict():
    from app.ml import predict
    for k in ("image", "video", "audio"):
        m = predict.load(k)
        assert m is not None and len(m["coef"]) == len(m["features"])
    rng = np.random.default_rng(3)
    img = np.clip(rng.normal(120, 30, (300, 400, 3)), 0, 255).astype(np.uint8)
    r = predict.predict_image(img)
    assert r is not None and 0 <= r["p_ai"] <= 1
