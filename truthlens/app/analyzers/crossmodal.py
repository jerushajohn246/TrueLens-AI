"""Cross-modal correlation: compare modalities with each other.

Main check here: does the visible mouth movement follow the speech energy in the audio track?
(Caption-vs-transcript, caption-vs-OCR and metadata-vs-claim comparisons live in text_analyzer.)
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from ..utils import evidence_path, fmt_ts, make_finding  # noqa: E402
from .audio_analyzer import load_wav  # noqa: E402


def _z(x):
    x = np.asarray(x, float)
    return (x - np.nanmean(x)) / (np.nanstd(x) + 1e-9)


def _smooth(x, k=3):
    if len(x) < k:
        return x
    return np.convolve(x, np.ones(k) / k, mode="same")


def audio_envelope(wav: Path, times: np.ndarray, fps: float) -> np.ndarray:
    y, sr = load_wav(wav)
    half = int(sr / fps / 2)
    env = []
    for t in times:
        c = int(t * sr)
        seg = y[max(0, c - half):c + half]
        env.append(np.sqrt(np.mean(seg ** 2)) if len(seg) else 0.0)
    env = np.array(env)
    # speech-band onset energy: positive changes in envelope are what lips correlate with
    return env


def av_sync(sync_series: dict, wav: Path, job_id: str) -> dict:
    t = np.array(sync_series["t"])
    mouth = np.array(sync_series["mouth"], float)
    fps = sync_series["fps"]
    seen = np.array(sync_series["face_seen"], bool)
    info: dict = {"method": "cross-correlation of mouth-region motion and audio energy envelope"}
    findings: list[dict] = []
    if len(t) < fps * 4:
        info["status"] = "Video too short for a synchronisation check (need 4 s+)."
        return {"info": info, "findings": findings}
    if seen.mean() < 0.4 or np.isfinite(mouth).sum() < fps * 3:
        info["status"] = "Face/mouth not visible in enough frames - synchronisation not assessed."
        return {"info": info, "findings": findings}

    env = audio_envelope(wav, t, fps)
    if env.max() < 1e-4:
        info["status"] = "Audio track is silent."
        return {"info": info, "findings": findings}
    env_db = 20 * np.log10(env + 1e-6)
    speaking = env_db > np.percentile(env_db, 95) - 22
    info["speech_ratio_in_video"] = round(float(speaking.mean()), 3)

    valid = np.isfinite(mouth)
    m = mouth.copy()
    m[~valid] = np.nanmedian(mouth)
    m = _smooth(m, 3)
    a = _smooth(np.abs(np.diff(env, prepend=env[0])) + env * 0.5, 3)
    mz, az = _z(m), _z(a)

    max_lag = int(fps * 1.0)
    lags = np.arange(-max_lag, max_lag + 1)
    corrs = []
    for L in lags:
        if L >= 0:
            x, y_, v = mz[L:], az[:len(az) - L], valid[L:] & valid[:len(valid) - L]
        else:
            x, y_, v = mz[:L], az[-L:], valid[:L] & valid[-L:]
        corrs.append(float(np.corrcoef(x[v], y_[v])[0, 1]) if v.sum() > 10 else 0.0)
    corrs = np.nan_to_num(np.array(corrs))
    k = int(np.argmax(corrs))
    best_lag_s = lags[k] / fps
    c0 = corrs[max_lag]
    info.update(corr_at_zero=round(float(c0), 3), best_corr=round(float(corrs[k]), 3),
                best_lag_sec=round(float(best_lag_s), 2))

    # sliding windows: where is the mismatch?
    win = int(fps * 4)
    bad_windows = []
    for s in range(0, len(t) - win + 1, win // 2):
        sl = slice(s, s + win)
        v = valid[sl]
        if v.sum() < win * 0.5 or speaking[sl].mean() < 0.3:
            continue
        c = np.corrcoef(mz[sl][v], az[sl][v])[0, 1] if np.std(mz[sl][v]) > 0 else 0
        if np.isfinite(c) and c < 0.05:
            bad_windows.append((t[s], t[min(s + win, len(t) - 1)], float(c)))
    # merge
    merged = []
    for s, e, c in bad_windows:
        if merged and s <= merged[-1][1]:
            merged[-1] = (merged[-1][0], e, min(merged[-1][2], c))
        else:
            merged.append((s, e, c))
    info["mismatch_windows"] = [[round(s, 2), round(e, 2)] for s, e, _ in merged]

    # mouth still while audio clearly speaks
    talk_motion = np.nanmean(m[speaking & valid]) if (speaking & valid).any() else 0
    quiet_motion = np.nanmean(m[~speaking & valid]) if (~speaking & valid).any() else talk_motion
    info["mouth_motion_speech_vs_silence"] = round(float(talk_motion / (quiet_motion + 1e-6)), 2)

    plot = _plot(job_id, t, mz, az, merged, best_lag_s)
    info["plot"] = plot

    offset = abs(best_lag_s) >= 0.2 and corrs[k] > 0.25 and corrs[k] - c0 > 0.12
    no_relation = corrs[k] < 0.2 and c0 < 0.15
    if offset:
        findings.append(make_finding(
            "crossmodal", f"Audio appears shifted by {best_lag_s * 1000:+.0f} ms against lip movement", 0.6, 0.6,
            f"Mouth movement best matches the audio when the audio is shifted by {best_lag_s:+.2f}s (correlation "
            f"{corrs[k]:.2f} vs {c0:.2f} without shift). A constant offset above ~200 ms suggests the audio track was "
            "replaced or re-aligned.", {"type": "segment", "start": 0, "end": round(float(t[-1]), 2)}, [plot],
            "crossmodal.av_offset"))
    elif no_relation and speaking.mean() > 0.25:
        s0, e0 = (merged[0][0], merged[0][1]) if merged else (0.0, float(t[-1]))
        findings.append(make_finding(
            "crossmodal", "Audio and visible mouth movement may not be synchronized", 0.75, 0.6,
            f"Mouth movement shows no meaningful relationship with the speech energy (best correlation "
            f"{corrs[k]:.2f}). The voice may not belong to the visible speaker, or the audio was dubbed/cloned. "
            f"Recommended action: manual review of timestamps {fmt_ts(s0)}-{fmt_ts(e0)}.",
            {"type": "segment", "start": round(s0, 2), "end": round(e0, 2),
             "all": info["mismatch_windows"]}, [plot], "crossmodal.av_mismatch"))
    elif merged:
        s0, e0, c = merged[0]
        findings.append(make_finding(
            "crossmodal", f"Local lip-sync mismatch {fmt_ts(s0)}-{fmt_ts(e0)}", 0.45, 0.5,
            f"Overall the lips follow the audio (correlation {corrs[k]:.2f}), but in {len(merged)} window(s) they do "
            f"not (local correlation {c:.2f}). This may indicate an inserted or replaced audio segment.",
            {"type": "segment", "start": round(s0, 2), "end": round(e0, 2), "all": info["mismatch_windows"]},
            [plot], "crossmodal.av_local"))
    info["status"] = "assessed"
    return {"info": info, "findings": findings}


def _plot(job_id, t, mz, az, bad, lag):
    fig, ax = plt.subplots(figsize=(11, 2.8))
    fig.patch.set_facecolor("#0f1420")
    ax.set_facecolor("#0f1420")
    ax.plot(t, _smooth(az, 5), color="#5eb1ff", lw=1, label="audio energy")
    ax.plot(t, _smooth(mz, 5), color="#ffb020", lw=1, label="mouth motion")
    for s, e, _ in bad:
        ax.axvspan(s, e, color="#ff4d6d", alpha=0.25)
    ax.set_title(f"Audio-video sync (best lag {lag:+.2f}s) - red = possible mismatch", color="#dfe6f3", fontsize=9)
    ax.tick_params(colors="#9aa4b8", labelsize=8)
    for sp in ax.spines.values():
        sp.set_color("#2a3347")
    ax.legend(fontsize=7, facecolor="#1a2133", labelcolor="#dfe6f3", edgecolor="#2a3347", loc="upper right")
    ax.set_xlabel("Time (s)", color="#9aa4b8", fontsize=8)
    fig.tight_layout()
    name = "av_sync.png"
    fig.savefig(evidence_path(job_id, name), dpi=110, facecolor=fig.get_facecolor())
    plt.close(fig)
    return f"/evidence/{job_id}/{name}"
