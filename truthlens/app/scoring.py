"""Evidence & risk aggregator.

The risk score is NOT a 'fake probability'. It measures how many / how strong manipulation
indicators were found, split into transparent categories (weights from the problem statement):

    Image manipulation indicators   25
    Video frame inconsistency       20
    Audio irregularity              20
    Audio-video / cross-modal       20
    Metadata / source issue         10
    Text-media contradiction         5

Only categories that apply to the uploaded media are counted, and the result is normalised to 0-100.
"""
from __future__ import annotations

import numpy as np

CATEGORIES = {
    "image": {"label": "Image manipulation indicators", "weight": 25},
    "video": {"label": "Video frame inconsistency", "weight": 20},
    "audio": {"label": "Audio irregularity", "weight": 20},
    "crossmodal": {"label": "Audio-video / cross-modal mismatch", "weight": 20},
    "metadata": {"label": "Metadata / source issue", "weight": 10},
    "text": {"label": "Text-media contradiction", "weight": 5},
}

LEVELS = [(75, "Very high suspicion", "very_high"), (50, "High suspicion", "high"),
          (25, "Needs attention", "medium"), (0, "Low indicators", "low")]


def contribution(f: dict) -> float:
    if f.get("review_status") == "rejected":
        return 0.0
    conf = 1.0 if f.get("review_status") == "accepted" else f["confidence"]
    return f["strength"] * (0.5 + 0.5 * conf)


def aggregate(findings: list[dict], applicable: set[str]) -> dict:
    applicable = set(applicable) | {"metadata"}
    breakdown = []
    total_w = sum(CATEGORIES[c]["weight"] for c in applicable)
    score = 0.0
    strong_cats = 0
    for cat, meta in CATEGORIES.items():
        fs = [f for f in findings if f["category"] == cat]
        contribs = [contribution(f) for f in fs]
        frac = 1 - float(np.prod([1 - c for c in contribs])) if contribs else 0.0
        pts = meta["weight"] * frac
        if cat in applicable:
            score += pts
            if frac >= 0.3:
                strong_cats += 1
        breakdown.append({
            "category": cat, "label": meta["label"], "max_points": meta["weight"],
            "points": round(pts, 1), "applicable": cat in applicable, "indicators": len(fs),
            "fraction": round(frac, 3)})
    risk = round(100 * score / total_w, 1) if total_w else 0.0
    level_label, level = next((lab, key) for thr, lab, key in LEVELS if risk >= thr)

    # Confidence: reliability of the indicators that drive the score (not P(fake)).
    active = [f for f in findings if contribution(f) > 0.15]
    if active:
        w = np.array([contribution(f) for f in active])
        c = np.array([1.0 if f.get("review_status") == "accepted" else f["confidence"] for f in active])
        conf = float((w * c).sum() / w.sum())
        conf += 0.05 * max(0, strong_cats - 1)        # independent categories agree
    else:
        conf = 0.6 if len(applicable) > 2 else 0.5    # "nothing found" with limited coverage
    reviewed = [f for f in findings if f.get("review_status") in ("accepted", "rejected")]
    if findings:
        conf += 0.1 * len(reviewed) / len(findings)   # human review increases reliability
    conf = float(np.clip(conf, 0.3, 0.95))

    if risk >= 50:
        rec = "Human review required before this media is trusted or shared."
    elif risk >= 25:
        rec = "Human review recommended - verify the flagged evidence and the original source."
    else:
        rec = "Few indicators found. Still verify the source; absence of indicators is not proof of authenticity."
    return {"risk_score": risk, "risk_level": level, "risk_label": level_label,
            "confidence": round(conf, 3), "breakdown": breakdown, "recommendation": rec,
            "summary": _summary(risk, level_label, findings)}


def _summary(risk, label, findings):
    top = sorted([f for f in findings if f.get("review_status") != "rejected"],
                 key=lambda f: -contribution(f))[:3]
    if not top or contribution(top[0]) < 0.15:
        return ("No strong manipulation indicators were detected by the automated checks. "
                "This does not prove the media is authentic.")
    return (f"{label} ({risk}/100). This media contains indicators commonly associated with manipulation, "
            "mainly: " + "; ".join(t["title"].lower() for t in top) +
            ". It should be reviewed by a human investigator.")
