"""Tiny SQLite persistence layer (analysis records + audit history)."""
from __future__ import annotations

import json
import sqlite3
import threading
from datetime import datetime, timezone

from .config import DB_PATH

_lock = threading.Lock()


def _conn():
    c = sqlite3.connect(DB_PATH, check_same_thread=False)
    c.row_factory = sqlite3.Row
    return c


def init_db():
    with _lock, _conn() as c:
        c.executescript("""
        CREATE TABLE IF NOT EXISTS analyses (
            id TEXT PRIMARY KEY,
            created_at TEXT,
            status TEXT,
            progress INTEGER DEFAULT 0,
            message TEXT,
            claim TEXT,
            files TEXT,
            result TEXT,
            risk_score REAL,
            risk_level TEXT,
            error TEXT
        );
        CREATE TABLE IF NOT EXISTS history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            analysis_id TEXT,
            ts TEXT,
            event TEXT,
            prev_hash TEXT,
            hash TEXT
        );
        """)
        cols = {r[1] for r in c.execute("PRAGMA table_info(history)").fetchall()}
        for col in ("prev_hash", "hash"):            # migrate databases from v1.0
            if col not in cols:
                c.execute(f"ALTER TABLE history ADD COLUMN {col} TEXT")


def now_iso():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def create(aid: str, claim: str, files: list[dict]):
    with _lock, _conn() as c:
        c.execute("INSERT INTO analyses (id, created_at, status, progress, message, claim, files) VALUES (?,?,?,?,?,?,?)",
                  (aid, now_iso(), "queued", 0, "Queued", claim, json.dumps(files)))
    log(aid, f"Analysis created with {len(files)} file(s)")


def update(aid: str, **fields):
    if "result" in fields and not isinstance(fields["result"], str):
        fields["result"] = json.dumps(fields["result"])
    if "files" in fields and not isinstance(fields["files"], str):
        fields["files"] = json.dumps(fields["files"])
    keys = ", ".join(f"{k}=?" for k in fields)
    with _lock, _conn() as c:
        c.execute(f"UPDATE analyses SET {keys} WHERE id=?", (*fields.values(), aid))


def log(aid: str, event: str) -> str:
    """Append an audit event to the hash chain; returns the new chain head."""
    from .integrity import GENESIS, chain_hash
    with _lock, _conn() as c:
        row = c.execute("SELECT hash FROM history WHERE analysis_id=? ORDER BY id DESC LIMIT 1", (aid,)).fetchone()
        prev = (row[0] if row and row[0] else GENESIS)
        ts = now_iso()
        h = chain_hash(prev, ts, event)
        c.execute("INSERT INTO history (analysis_id, ts, event, prev_hash, hash) VALUES (?,?,?,?,?)",
                  (aid, ts, event, prev, h))
        return h


def chain_head(aid: str) -> str:
    from .integrity import GENESIS
    with _lock, _conn() as c:
        row = c.execute("SELECT hash FROM history WHERE analysis_id=? ORDER BY id DESC LIMIT 1", (aid,)).fetchone()
        return row[0] if row and row[0] else GENESIS


def get(aid: str) -> dict | None:
    with _lock, _conn() as c:
        r = c.execute("SELECT * FROM analyses WHERE id=?", (aid,)).fetchone()
        if not r:
            return None
        d = dict(r)
        d["files"] = json.loads(d["files"] or "[]")
        d["result"] = json.loads(d["result"]) if d["result"] else None
        d["history"] = [dict(h) for h in c.execute(
            "SELECT ts, event, prev_hash, hash FROM history WHERE analysis_id=? ORDER BY id", (aid,)).fetchall()]
        return d


def list_all(limit: int = 100) -> list[dict]:
    with _lock, _conn() as c:
        rows = c.execute("SELECT id, created_at, status, claim, files, risk_score, risk_level FROM analyses "
                         "ORDER BY created_at DESC LIMIT ?", (limit,)).fetchall()
    out = []
    for r in rows:
        d = dict(r)
        d["files"] = json.loads(d["files"] or "[]")
        out.append(d)
    return out


def delete(aid: str):
    with _lock, _conn() as c:
        c.execute("DELETE FROM analyses WHERE id=?", (aid,))
        c.execute("DELETE FROM history WHERE analysis_id=?", (aid,))
