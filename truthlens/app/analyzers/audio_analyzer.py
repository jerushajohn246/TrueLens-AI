"""Audio forensic analysis (works on standalone audio files and on audio tracks extracted from video).

Indicators:
  * Spectrogram (saved as evidence, suspicious segments marked)
  * Pause analysis - unnaturally regular pauses / digital-silence gaps (splice points)
  * Background-noise floor jumps - abrupt changes in room tone suggest editing
  * Bandwidth / high-frequency cutoff - re-synthesised or heavily processed speech
  * Pitch stability (jitter) - TTS / voice clones are often unnaturally stable
  * Persistent tonal lines ("repeated frequencies")
  * Repeated / looped segments
  * Speaker-consistency across the recording
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import soundfile as sf  # noqa: E402
from scipy import signal  # noqa: E402

from ..utils import evidence_path, fmt_ts, make_finding  # noqa: E402


def load_wav(path: Path) -> tuple[np.ndarray, int]:
    y, sr = sf.read(str(path), dtype="float32", always_2d=False)
    if y.ndim > 1:
        y = y.mean(axis=1)
    return y, sr


def frame_rms(y: np.ndarray, sr: int, hop_s: float = 0.01, win_s: float = 0.025) -> np.ndarray:
    hop, win = int(sr * hop_s), int(sr * win_s)
    if len(y) < win:
        return np.array([np.sqrt(np.mean(y ** 2) + 1e-12)])
    n = 1 + (len(y) - win) // hop
    idx = np.arange(win)[None, :] + hop * np.arange(n)[:, None]
    return np.sqrt((y[idx] ** 2).mean(axis=1) + 1e-12)


def mel_filterbank(sr: int, n_fft: int, n_mels: int = 40) -> np.ndarray:
    def hz2mel(f):
        return 2595 * np.log10(1 + f / 700)

    def mel2hz(m):
        return 700 * (10 ** (m / 2595) - 1)
    mels = np.linspace(hz2mel(50), hz2mel(sr / 2), n_mels + 2)
    bins = np.floor((n_fft + 1) * mel2hz(mels) / sr).astype(int)
    fb = np.zeros((n_mels, n_fft // 2 + 1))
    for i in range(1, n_mels + 1):
        l, c, r = bins[i - 1], bins[i], bins[i + 1]
        if c > l:
            fb[i - 1, l:c] = (np.arange(l, c) - l) / (c - l)
        if r > c:
            fb[i - 1, c:r] = (r - np.arange(c, r)) / (r - c)
    return fb


def logmel(y, sr, n_fft=1024, hop=320):
    f, t, Z = signal.stft(y, sr, nperseg=n_fft, noverlap=n_fft - hop, boundary=None)
    P = np.abs(Z) ** 2
    M = mel_filterbank(sr, n_fft) @ P
    return np.log(M + 1e-10), t


def estimate_pitch(y, sr, voiced_mask_rms, hop_s=0.01, win_s=0.04):
    hop, win = int(sr * hop_s), int(sr * win_s)
    lo, hi = int(sr / 400), int(sr / 75)
    pitches = []
    n = min(len(voiced_mask_rms), 1 + (len(y) - win) // hop if len(y) > win else 0)
    for i in range(0, n, 2):
        if not voiced_mask_rms[i]:
            pitches.append(np.nan)
            continue
        fr = y[i * hop:i * hop + win]
        fr = fr - fr.mean()
        ac = np.correlate(fr, fr, "full")[win - 1:]
        if ac[0] <= 0:
            pitches.append(np.nan)
            continue
        ac = ac / ac[0]
        seg = ac[lo:hi]
        if len(seg) == 0:
            pitches.append(np.nan)
            continue
        k = int(np.argmax(seg))
        pitches.append(sr / (lo + k) if seg[k] > 0.45 else np.nan)
    return np.array(pitches)


def analyze_audio(wav_path: Path, job_id: str, label: str = "audio", progress=lambda *_: None) -> dict:
    y, sr = load_wav(wav_path)
    dur = len(y) / sr if sr else 0
    findings: list[dict] = []
    evidence: dict = {}
    marks: list[tuple[float, float, str]] = []
    info = {"duration_sec": round(dur, 2), "sample_rate": sr}
    if dur < 0.5 or np.max(np.abs(y)) < 1e-4:
        info["note"] = "Audio is silent or too short for analysis."
        return {"media_type": "audio", "info": info, "findings": findings, "evidence": evidence,
                "checks_run": ["Silence check"]}

    y = y / (np.max(np.abs(y)) + 1e-9)
    rms = frame_rms(y, sr)
    db = 20 * np.log10(rms + 1e-9)
    peak_db = np.percentile(db, 95)
    speech = db > peak_db - 25
    info["speech_ratio"] = round(float(speech.mean()), 3)
    info["rms_db"] = round(float(20 * np.log10(np.sqrt(np.mean(y ** 2)) + 1e-9)), 1)

    # ---- digital silence (exact zeros) inside the recording -> hard edit points
    zero = np.abs(y) < 1e-5
    runs = _runs(zero)
    gaps = [(s / sr, e / sr) for s, e in runs if (e - s) / sr > 0.06 and s > sr * 0.2 and e < len(y) - sr * 0.2]
    if gaps:
        for s, e in gaps[:10]:
            marks.append((s, e, "digital silence"))
        findings.append(make_finding(
            "audio", "Digital-silence gaps inside the recording", min(0.35 + 0.1 * len(gaps), 0.8), 0.65,
            f"{len(gaps)} segment(s) of perfect digital silence were found (first at {fmt_ts(gaps[0][0])}). Real "
            "microphones always record some room noise; absolute zero usually means audio was cut and joined.",
            {"type": "segment", "start": round(gaps[0][0], 2), "end": round(gaps[0][1], 2),
             "all": [[round(a, 2), round(b, 2)] for a, b in gaps[:10]]}, check="audio.digital_silence",
            ctx={"n_gaps": len(gaps), "mean_gap_sec": round(float(np.mean([b - a for a, b in gaps])), 3)}))

    # ---- pauses
    sil_runs = [(s * 0.01, e * 0.01) for s, e in _runs(~speech) if (e - s) * 0.01 >= 0.15]
    inner = [(s, e) for s, e in sil_runs if s > 0.3 and e < dur - 0.3]
    info["pauses"] = len(inner)
    if len(inner) >= 5:
        lens = np.array([e - s for s, e in inner])
        cv = lens.std() / (lens.mean() + 1e-6)
        info["pause_length_cv"] = round(float(cv), 3)
        if cv < 0.18:
            findings.append(make_finding(
                "audio", "Unnaturally regular pauses", 0.45, 0.45,
                f"Pauses between phrases have almost identical length (variation {cv:.2f}, mean {lens.mean():.2f}s). "
                "Natural speech pauses vary a lot; uniform pauses are typical of text-to-speech output.",
                {"type": "segment", "start": round(inner[0][0], 2), "end": round(inner[-1][1], 2)},
                check="audio.pause_regularity"))
        long_p = [(s, e) for s, e in inner if e - s > 2.5]
        if long_p:
            info["long_pauses"] = len(long_p)

    # ---- background noise floor jumps (1 s windows)
    win = 100  # frames of 10 ms
    floors = []
    for i in range(0, len(db) - win + 1, win // 2):
        floors.append(np.percentile(db[i:i + win], 10))
    floors = np.array(floors)
    jumps = []
    for i in range(1, len(floors)):
        d = floors[i] - floors[i - 1]
        if abs(d) > 9 and min(floors[i], floors[i - 1]) < peak_db - 20:
            jumps.append((i * 0.5, d))
    if jumps:
        t0, d0 = max(jumps, key=lambda j: abs(j[1]))
        for t, _ in jumps[:8]:
            marks.append((t, t + 1.0, "noise change"))
        findings.append(make_finding(
            "audio", "Abrupt background-noise change", float(np.clip(0.3 + abs(d0) / 40, 0, 0.8)), 0.55,
            f"The background noise level jumps by {d0:+.1f} dB around {fmt_ts(t0)} ({len(jumps)} jump(s) in total). "
            "A sudden change in room tone suggests that recordings from different sources or takes were joined.",
            {"type": "segment", "start": round(t0, 2), "end": round(t0 + 1, 2),
             "all": [round(t, 2) for t, _ in jumps[:10]]}, check="audio.noise_floor",
            ctx={"n_jumps": len(jumps), "persistent": _persistent(floors, int(t0 / 0.5), d0)}))

    progress(f"{label}: pause & noise analysis done")

    # ---- spectrum: bandwidth & tonal lines
    f, t, Z = signal.stft(y, sr, nperseg=1024, noverlap=768, boundary=None)
    S = 20 * np.log10(np.abs(Z) + 1e-9)
    sp_idx = np.clip((t / 0.01).astype(int), 0, len(speech) - 1)
    act = S[:, speech[sp_idx]]
    mean_spec = (act if act.shape[1] > 5 else S).mean(axis=1)
    ref = np.max(mean_spec)
    above = np.where(mean_spec > ref - 55)[0]
    cutoff = float(f[above[-1]]) if len(above) else 0.0
    info["effective_bandwidth_hz"] = int(cutoff)
    if sr >= 32000 and cutoff < 5500:
        findings.append(make_finding(
            "audio", f"Limited bandwidth (energy stops at ~{cutoff / 1000:.1f} kHz)", 0.3, 0.4,
            "Almost no energy is present above this frequency. Phone audio and low-bitrate compression do this, "
            "but many voice-cloning / TTS systems also generate band-limited speech (e.g. 16-22 kHz models).",
            {"type": "segment", "start": 0, "end": round(dur, 2)}, check="audio.bandwidth"))

    med = signal.medfilt(mean_spec, 31)
    prom = mean_spec - med
    tonal_bins = np.where((prom > 14) & (f > 100))[0]
    if len(tonal_bins):
        persist = [(S[b] > (np.median(S, axis=0) + 15)).mean() for b in tonal_bins]
        strong = [(f[b], p) for b, p in zip(tonal_bins, persist) if p > 0.85]
        if strong:
            freqs = ", ".join(f"{fr:.0f} Hz" for fr, _ in strong[:4])
            findings.append(make_finding(
                "audio", "Persistent tonal frequency lines", 0.3, 0.4,
                f"Narrow frequency lines are present for most of the recording ({freqs}). Could be electrical hum, "
                "but constant artificial tones can also come from synthesis or processing.",
                {"type": "segment", "start": 0, "end": round(dur, 2)}, check="audio.tonal",
                ctx={"freqs": [round(float(fr), 1) for fr, _ in strong[:6]]}))

    # ---- pitch stability
    pitches = estimate_pitch(y, sr, speech)
    vp = pitches[~np.isnan(pitches)]
    if len(vp) > 40:
        semis = 12 * np.log2(vp / np.median(vp))
        d = np.abs(np.diff(vp)) / vp[:-1]
        jitter = float(np.median(d))
        info["pitch_median_hz"] = round(float(np.median(vp)), 1)
        info["pitch_std_semitones"] = round(float(semis.std()), 2)
        info["pitch_jitter"] = round(jitter, 4)
        if semis.std() < 1.0 and jitter < 0.01:
            findings.append(make_finding(
                "audio", "Very flat / stable pitch", 0.4, 0.4,
                f"Pitch varies by only {semis.std():.2f} semitones with very low frame-to-frame jitter "
                f"({jitter * 100:.2f}%). Human speech is usually more variable; this is consistent with synthetic "
                "or voice-cloned speech (or a very monotone speaker).",
                {"type": "segment", "start": 0, "end": round(dur, 2)}, check="audio.pitch"))
    progress(f"{label}: spectral & pitch analysis done")

    # ---- repeated segments + speaker consistency (log-mel)
    L, lt = logmel(y, sr)
    chunk = max(1, int(0.5 / (lt[1] - lt[0]))) if len(lt) > 1 else 1
    vecs, times = [], []
    for i in range(0, L.shape[1] - chunk, chunk):
        seg = L[:, i:i + chunk]
        if seg.mean() > np.percentile(L, 40):
            v = (seg - seg.mean()).flatten()
            vecs.append(v / (np.linalg.norm(v) + 1e-9))
            times.append(lt[i])
    repeats = []
    if len(vecs) > 4:
        V = np.array(vecs)
        sim = V @ V.T
        for i in range(len(V)):
            for j in range(i + 2, len(V)):
                if sim[i, j] > 0.97:
                    repeats.append((times[i], times[j], float(sim[i, j])))
        if repeats:
            a, b, sv = repeats[0]
            for s, e, _ in repeats[:6]:
                marks.append((e, e + 0.5, "repeat"))
            findings.append(make_finding(
                "audio", "Repeated / looped audio segments", float(np.clip(0.4 + 0.05 * len(repeats), 0, 0.8)), 0.55,
                f"The segment at {fmt_ts(b)} is nearly identical (similarity {sv:.3f}) to the one at {fmt_ts(a)}. "
                f"{len(repeats)} near-duplicate pair(s) found. Natural speech never repeats exactly; this indicates "
                "copy-pasted or looped audio.",
                {"type": "segment", "start": round(b, 2), "end": round(b + 0.5, 2), "source": round(a, 2)},
                check="audio.repeat"))

    # speaker consistency: mean log-mel profile per ~3 s of speech
    seg_len = max(1, int(3 / (lt[1] - lt[0]))) if len(lt) > 1 else 1
    profiles, ptimes = [], []
    for i in range(0, L.shape[1] - seg_len + 1, seg_len):
        seg = L[:, i:i + seg_len]
        loud = seg.mean(axis=0) > np.percentile(L.mean(axis=0), 50)
        if loud.sum() > seg_len * 0.3:
            p = seg[:, loud].mean(axis=1)
            profiles.append(p - p.mean())
            ptimes.append(lt[i])
    if len(profiles) >= 3:
        P = np.array(profiles)
        dists = np.array([np.linalg.norm(P[i] - P[i - 1]) for i in range(1, len(P))])
        medd = np.median(dists) + 1e-6
        k = int(np.argmax(dists))
        info["speaker_profile_max_jump"] = round(float(dists[k] / medd), 2)
        if dists[k] / medd > 3.0 and dists[k] > 6:
            tt = ptimes[k + 1]
            marks.append((tt, tt + 3, "voice change"))
            findings.append(make_finding(
                "audio", "Possible speaker / recording change", 0.4, 0.4,
                f"The spectral voice profile changes sharply around {fmt_ts(tt)} ({dists[k] / medd:.1f}x the normal "
                "variation). This can mean a different speaker, a different microphone, or an inserted segment.",
                {"type": "segment", "start": round(tt, 2), "end": round(tt + 3, 2)}, check="audio.speaker"))

    # ---- spectrogram evidence
    evidence["spectrogram"] = _save_spectrogram(job_id, label, y, sr, marks)
    evidence["waveform_marks"] = [[round(a, 2), round(b, 2), m] for a, b, m in marks]
    progress(f"{label}: spectrogram generated")
    for fd in findings:
        fd["evidence"].append(evidence["spectrogram"])
    return {"media_type": "audio", "info": info, "findings": findings, "evidence": evidence,
            "checks_run": ["Spectrogram", "Digital silence", "Pause regularity", "Background noise floor",
                           "Bandwidth", "Tonal lines", "Pitch stability", "Repeated segments", "Speaker consistency"]}


def _persistent(floors, i, d):
    """True if the new background level is held for >= 3 s after the jump (not a short bump)."""
    after = floors[i:i + 6]
    if len(after) < 4:
        return False
    return bool(np.all(np.sign(after - floors[max(0, i - 1)]) == np.sign(d)))


def _runs(mask: np.ndarray):
    m = np.concatenate([[False], mask, [False]]).astype(np.int8)
    d = np.diff(m)
    return list(zip(np.where(d == 1)[0], np.where(d == -1)[0]))


def _save_spectrogram(job_id, label, y, sr, marks):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 4.6), sharex=True,
                                   gridspec_kw={"height_ratios": [1, 2.4]})
    fig.patch.set_facecolor("#0f1420")
    for ax in (ax1, ax2):
        ax.set_facecolor("#0f1420")
        ax.tick_params(colors="#9aa4b8", labelsize=8)
        for s in ax.spines.values():
            s.set_color("#2a3347")
    tt = np.arange(len(y)) / sr
    step = max(1, len(y) // 20000)
    ax1.plot(tt[::step], y[::step], color="#5eb1ff", lw=0.5)
    ax1.set_ylabel("Amp", color="#9aa4b8", fontsize=8)
    ax2.specgram(y + np.random.default_rng(0).normal(0, 1e-6, len(y)), NFFT=1024, Fs=sr, noverlap=768, cmap="magma", vmin=-130)
    ax2.set_ylabel("Hz", color="#9aa4b8", fontsize=8)
    ax2.set_xlabel("Time (s)", color="#9aa4b8", fontsize=8)
    colors = {"digital silence": "#ff4d6d", "noise change": "#ffb020", "repeat": "#c77dff",
              "voice change": "#4dd4ac"}
    seen = set()
    for a, b, m in marks:
        for ax in (ax1, ax2):
            ax.axvspan(a, b, color=colors.get(m, "#fff"), alpha=0.28,
                       label=m if (ax is ax1 and m not in seen) else None)
        seen.add(m)
    if seen:
        ax1.legend(loc="upper right", fontsize=7, facecolor="#1a2133", labelcolor="#dfe6f3", edgecolor="#2a3347")
    fig.tight_layout()
    name = f"{label}_spectrogram.png"
    fig.savefig(evidence_path(job_id, name), dpi=110, facecolor=fig.get_facecolor())
    plt.close(fig)
    return f"/evidence/{job_id}/{name}"
