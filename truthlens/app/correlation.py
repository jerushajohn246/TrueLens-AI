"""Evidence Correlation Graph.

Findings are nodes. Edges say how findings relate:
  SUPPORT     - independent checks agree: same moment in time, same image region, or the same
                manipulation hypothesis seen from a different modality (e.g. face anomaly at 00:03
                AND lip-sync break at 00:03).
  CONTRADICT  - counter-evidence (e.g. camera metadata present vs "AI-generated"; lips match the
                audio vs "voice clone"), or a tested innocent explanation that is likely.

Confidence is then adjusted transparently:
  +0.08 per independent supporting modality (max +0.20), +0.03 per same-modality support,
  -0.10 per contradiction.
Connected groups of supporting findings become "hypotheses" (e.g. "Face-swap / dubbed video").
"""
from __future__ import annotations

import itertools

import numpy as np

THEMES = {
    "ai_generation": {"label": "AI-generated media", "checks": {"metadata.ai_tags", "image.ai_dimensions", "image.spectrum",
                                                                  "ai.c2pa", "ai.metadata_hint", "ai.no_sensor_grain",
                                                                  "ai.face_bg_sharpness", "ai.smooth_camera", "ai.bg_warp",
                                                                  "ai.ml_model"}},
    "image_edit": {"label": "Edited / composited image",
                   "checks": {"metadata.software", "image.ela", "image.noise", "image.copy_move", "image.face", "metadata.dates"}},
    "face_swap": {"label": "Face-swap / manipulated face",
                  "checks": {"video.face_step", "video.face_anomaly", "video.face_flicker", "image.face",
                             "crossmodal.av_mismatch", "crossmodal.av_local"}},
    "voice_replaced": {"label": "Dubbed or cloned voice",
                       "checks": {"crossmodal.av_mismatch", "crossmodal.av_offset", "crossmodal.av_local", "audio.pitch",
                                  "audio.pause_regularity", "audio.bandwidth", "audio.speaker", "crossmodal.quote",
                                  "crossmodal.caption_transcript"}},
    "splice": {"label": "Spliced / cut-and-joined recording",
               "checks": {"audio.digital_silence", "audio.noise_floor", "audio.repeat", "audio.speaker", "video.cuts",
                          "video.frozen", "crossmodal.av_local"}},
    "old_content": {"label": "Old or out-of-context content presented as new",
                    "checks": {"text.old_date", "text.metadata_date", "text.location", "metadata.video_no_date"}},
}

CAT_LABEL = {"image": "Image", "video": "Video", "audio": "Audio", "crossmodal": "Cross-modal",
             "metadata": "Metadata", "text": "Text", "counter": "Counter-evidence"}


def _themes(check):
    return {k for k, v in THEMES.items() if check in v["checks"]}


def _ranges(f):
    loc = f.get("location") or {}
    t = loc.get("type")
    out = []
    if t == "timestamp" and loc.get("t") is not None:
        out.append((loc["t"] - 0.6, (loc.get("end") or loc["t"]) + 0.6))
        out += [(v - 0.6, v + 0.6) for v in loc.get("all", []) if isinstance(v, (int, float))]
    elif t == "segment" and loc.get("start") is not None:
        a, b = loc["start"], loc.get("end", loc["start"])
        if b - a <= 8:                        # whole-file segments say nothing about timing
            out.append((a - 0.4, b + 0.4))
        for v in loc.get("all", []):
            if isinstance(v, list) and len(v) == 2 and v[1] - v[0] <= 8:
                out.append((v[0] - 0.4, v[1] + 0.4))
            elif isinstance(v, (int, float)):
                out.append((v - 0.6, v + 0.6))
    return out


def _time_overlap(a, b):
    for (a0, a1), (b0, b1) in itertools.product(_ranges(a), _ranges(b)):
        if a0 <= b1 and b0 <= a1:
            return max(a0, b0) + 0.5
    return None


def _bbox(f):
    loc = f.get("location") or {}
    return loc.get("bbox") if loc.get("type") in ("region", "timestamp") and loc.get("bbox") else None


def _region_overlap(a, b):
    ba, bb = _bbox(a), _bbox(b)
    if not ba or not bb:
        return False
    ax, ay, aw, ah = ba
    bx, by, bw, bh = bb
    ix = max(0, min(ax + aw, bx + bw) - max(ax, bx))
    iy = max(0, min(ay + ah, by + bh) - max(ay, by))
    return ix * iy > 0.25 * min(aw * ah, bw * bh)


def _fmt(t):
    m, s = divmod(max(t, 0), 60)
    return f"{int(m):02d}:{s:04.1f}"


def _when(times):
    if not times:
        return None
    a, b = min(t for t, _ in times) + 0.5, max(t for _, t in times) - 0.5
    return f"at {_fmt(a)}" if b - a < 0.4 else f"{_fmt(a)} - {_fmt(b)}"


def counter_evidence(media: dict) -> list[dict]:
    """Observations that argue AGAINST certain manipulation hypotheses."""
    out = []
    for im in media.get("image", []):
        exif = im.get("metadata", {}).get("exif", {})
        if exif.get("Make") or exif.get("Model"):
            out.append({"id": f"ce-cam-{im['file_id']}", "file_id": im["file_id"], "title": f"Camera metadata present ({exif.get('Make', '')} {exif.get('Model', '')})".strip(),
                        "contradicts": ["metadata.ai_tags", "image.ai_dimensions", "image.spectrum", "metadata.missing"],
                        "reason": "A real camera make/model argues against pure AI generation (but can be forged)."})
    for vd in media.get("video", []):
        tags = {k.lower(): v for k, v in (vd.get("info", {}).get("container_tags") or {}).items()}
        cam = tags.get("com.apple.quicktime.make") or tags.get("com.android.manufacturer")
        if cam:
            out.append({"id": f"ce-vcam-{vd['file_id']}", "file_id": vd["file_id"], "title": f"Recorded-on-device metadata ({cam})",
                        "contradicts": ["ai.no_sensor_grain", "ai.face_bg_sharpness", "ai.smooth_camera", "ai.bg_warp", "ai.metadata_hint"],
                        "reason": "Phone make/model written by the camera app argues against generation (can be forged)."})
        av = vd.get("av_sync") or {}
        if av.get("status") == "assessed" and (av.get("best_corr") or 0) >= 0.45 and not av.get("mismatch_windows"):
            out.append({"id": f"ce-sync-{vd['file_id']}", "file_id": vd["file_id"],
                        "title": f"Lips match the audio (correlation {av['best_corr']:.2f})",
                        "contradicts": ["audio.pitch", "audio.pause_regularity", "audio.bandwidth", "audio.speaker", "crossmodal.quote"],
                        "reason": "Good lip-sync argues against a replaced or cloned voice track."})
        if vd.get("info", {}).get("sudden_changes", 0) == 0 and vd.get("info", {}).get("duration_sec", 0) > 5:
            out.append({"id": f"ce-cont-{vd['file_id']}", "file_id": vd["file_id"], "title": "Single continuous shot (no cuts found)",
                        "contradicts": ["audio.digital_silence", "audio.repeat"],
                        "reason": "No visual cut at all makes an audio splice less likely (audio could still be replaced)."})
    return out


def build(findings: list[dict], counters: list[dict]) -> dict:
    active = [f for f in findings if f.get("review_status") != "rejected" and f["strength"] >= 0.15]
    nodes, edges = [], []
    for f in active:
        nodes.append({"id": f["id"], "kind": "finding", "label": f["title"], "category": f["category"],
                      "check": f.get("check"), "strength": f["strength"], "file_id": f.get("file_id")})
    ids = {f["id"] for f in active}

    # finding <-> finding support
    for a, b in itertools.combinations(active, 2):
        if a.get("check") == b.get("check"):
            continue
        same_file = a.get("file_id") and a.get("file_id") == b.get("file_id")
        reason = None
        if same_file:
            t = _time_overlap(a, b)
            if t is not None:
                reason = f"both occur around {_fmt(t)}"
            elif _region_overlap(a, b):
                reason = "both point to the same image region"
        shared = _themes(a.get("check")) & _themes(b.get("check"))
        if not reason and shared and (a["category"] != b["category"] or "ai_generation" in shared):
            reason = "independent signals of the same hypothesis: " + THEMES[sorted(shared)[0]]["label"].lower()
        if reason:
            edges.append({"source": a["id"], "target": b["id"], "type": "support", "reason": reason,
                          "theme": sorted(shared)[0] if shared else None})

    # counter-evidence
    for ce in counters:
        hits = [f for f in active if f.get("check") in ce["contradicts"] and (not f.get("file_id") or f.get("file_id") == ce.get("file_id"))]
        if not hits:
            continue
        nodes.append({"id": ce["id"], "kind": "counter", "label": ce["title"], "category": "counter", "strength": 0.5})
        for f in hits:
            edges.append({"source": ce["id"], "target": f["id"], "type": "contradict", "reason": ce["reason"]})
    # likely innocent explanations become contradiction nodes
    for f in active:
        for i, alt in enumerate(f.get("alternatives", [])):
            if alt["status"] == "likely":
                nid = f"alt-{f['id']}-{i}"
                nodes.append({"id": nid, "kind": "innocent", "label": "Innocent cause likely: " + alt["cause"],
                              "category": "counter", "strength": 0.4})
                edges.append({"source": nid, "target": f["id"], "type": "contradict", "reason": alt["detail"] or alt["cause"]})

    # confidence adjustment
    cat = {f["id"]: f["category"] for f in active}
    adjust = {}
    for f in active:
        sup = [e for e in edges if e["type"] == "support" and f["id"] in (e["source"], e["target"])]
        other = {e["target"] if e["source"] == f["id"] else e["source"] for e in sup}
        cross_cats = {cat[o] for o in other if o in cat and cat[o] != f["category"]}
        same = [o for o in other if o in cat and cat[o] == f["category"]]
        n_con = sum(1 for e in edges if e["type"] == "contradict" and e["target"] == f["id"])
        delta = min(0.20, 0.08 * len(cross_cats) + 0.03 * len(same)) - 0.10 * n_con
        adjust[f["id"]] = {"delta": round(delta, 3), "supporting": sorted(other), "contradictions": n_con,
                           "independent_modalities": sorted(cross_cats)}

    # hypotheses: per manipulation theme, groups of findings of that theme linked by support edges
    by_id = {f["id"]: f for f in active}
    hyps = []
    for th, spec in THEMES.items():
        mem = [f["id"] for f in active if f.get("check") in spec["checks"]]
        if len(mem) < 2:
            continue
        parent = {i: i for i in mem}

        def find(i):
            while parent[i] != i:
                parent[i] = parent[parent[i]]
                i = parent[i]
            return i
        for e in edges:
            if e["type"] == "support" and e["source"] in parent and e["target"] in parent:
                parent[find(e["source"])] = find(e["target"])
        comps: dict = {}
        for i in mem:
            comps.setdefault(find(i), []).append(i)
        for members in comps.values():
            if len(members) < 2:
                continue
            ms = [by_id[m] for m in members]
            strength = 1 - float(np.prod([1 - m["strength"] * m["confidence"] for m in ms]))
            cats = sorted({m["category"] for m in ms})
            times = [r for m in ms for r in _ranges(m)]
            hyps.append({"title": spec["label"], "theme": th, "members": members, "modalities": cats,
                         "strength": round(strength, 3),
                         "when": _when(times),
                         "contradicted": sum(adjust[m]["contradictions"] for m in members),
                         "summary": f"{len(members)} indicators from {len(cats)} modalit{'y' if len(cats) == 1 else 'ies'} "
                                    f"({', '.join(CAT_LABEL.get(c, c) for c in cats)}) point the same way."})
    hyps.sort(key=lambda h: -h["strength"])
    return {"nodes": nodes, "edges": edges, "adjustments": adjust, "hypotheses": hyps}


def apply(findings: list[dict], counters: list[dict]) -> dict:
    """Recompute graph and set each finding's final confidence = base_confidence + correlation delta."""
    for f in findings:
        f.setdefault("base_confidence", f["confidence"])
        f["confidence"] = f["base_confidence"]
        f["correlation"] = None
    g = build(findings, counters)
    for f in findings:
        a = g["adjustments"].get(f["id"])
        if a:
            f["correlation"] = a
            f["confidence"] = round(float(np.clip(f["base_confidence"] + a["delta"], 0.1, 0.95)), 3)
    return g


def render_png(graph: dict, path) -> None:
    """Static picture of the graph for the PDF report."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    nodes = graph["nodes"]
    if not nodes:
        return
    colors = {"image": "#5eb1ff", "video": "#8b7bff", "audio": "#ffb020", "crossmodal": "#ff6b9a",
              "metadata": "#94a3b8", "text": "#4dd4ac", "counter": "#22c55e"}
    n = len(nodes)
    order = sorted(range(n), key=lambda i: (nodes[i]["category"] == "counter", nodes[i]["category"]))
    pos = {}
    for k, i in enumerate(order):
        ang = 2 * np.pi * k / n + np.pi / 2
        pos[nodes[i]["id"]] = (np.cos(ang), np.sin(ang))
    fig, ax = plt.subplots(figsize=(9, 6.2))
    fig.patch.set_facecolor("white")
    for e in graph["edges"]:
        (x0, y0), (x1, y1) = pos[e["source"]], pos[e["target"]]
        sup = e["type"] == "support"
        ax.plot([x0, x1], [y0, y1], color="#16a34a" if sup else "#dc2626", lw=2 if sup else 1.6,
                ls="-" if sup else "--", alpha=0.75, zorder=1)
    for nd in nodes:
        x, y = pos[nd["id"]]
        ax.scatter([x], [y], s=250 + 900 * nd["strength"], color=colors.get(nd["category"], "#888"),
                   edgecolors="#14532d" if nd["category"] == "counter" else "#1e293b", linewidths=1.5, zorder=2)
        lab = nd["label"] if len(nd["label"]) < 38 else nd["label"][:36] + "..."
        ax.text(x * 1.16, y * 1.16, lab, fontsize=7, ha="center", va="center", color="#0f172a", zorder=3)
    ax.set_xlim(-1.75, 1.75)
    ax.set_ylim(-1.35, 1.35)
    ax.axis("off")
    ax.set_title("Evidence correlation graph  (green = supports, red dashed = contradicts)", fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
