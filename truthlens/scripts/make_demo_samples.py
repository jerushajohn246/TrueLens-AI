"""Generate demo media for presenting TruthLens (no internet needed).

    python scripts/make_demo_samples.py

Creates ./samples/ with:
  synthetic_scene.jpg        - consistent synthetic photo with camera EXIF
  edited_photo.jpg         - copy-move clone + pasted/smoothed face + old date text + Photoshop tag
  animated_avatar.mp4      - talking head, lips follow the audio
  manipulated_talking.mp4  - dubbed audio (lips do not match), smoothed face, hard cut, old on-screen date
  synthetic_voice.wav        - natural-sounding synthetic speech with room tone
  edited_audio.wav         - digital-silence splice, background-noise jump, repeated (looped) segment
  claims.txt               - captions to paste into the claim box during the demo

The media is synthetic (drawn faces, synthesised voice) so it is safe to show publicly.
For the strongest demo, also test with your own real photos/videos.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
import soundfile as sf
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from app.utils import ffmpeg_bin  # noqa: E402

OUT = ROOT / "samples"
OUT.mkdir(exist_ok=True)
SR = 32000
FPS = 25


# ------------------------------------------------------------------ drawing helpers
def draw_scene(rng, W=960, H=640, grain=6.0):
    img = np.zeros((H, W, 3), np.float32)
    for y in range(H):  # sky gradient
        img[y, :] = (235 - y * 0.12, 205 - y * 0.08, 170 - y * 0.05)
    cv2.rectangle(img, (0, int(H * 0.62)), (W, H), (70, 110, 80), -1)        # grass
    cv2.rectangle(img, (60, 250), (300, 420), (60, 80, 150), -1)              # building
    for i in range(3):
        for j in range(4):
            cv2.rectangle(img, (80 + j * 55, 270 + i * 48), (115 + j * 55, 300 + i * 48), (200, 220, 235), -1)
    cv2.circle(img, (820, 110), 45, (120, 230, 255), -1)                      # sun
    for k in range(6):                                                        # trees
        x = 360 + k * 90
        cv2.rectangle(img, (x - 6, 330), (x + 6, 410), (40, 60, 90), -1)
        cv2.circle(img, (x, 320), 32 + (k % 3) * 6, (50, 120 + k * 10, 60), -1)
    img = cv2.GaussianBlur(img, (3, 3), 0)
    img += rng.normal(0, grain, img.shape)                                    # sensor noise everywhere
    return np.clip(img, 0, 255).astype(np.uint8)


def draw_face(img, cx, cy, s=1.0, mouth_open=0.0, rng=None, grain=6.0):
    layer = img.copy()
    c = lambda v: int(v * s)  # noqa: E731
    cv2.ellipse(layer, (cx, cy + c(200)), (c(160), c(85)), 0, 0, 360, (60, 60, 140), -1)
    cv2.rectangle(layer, (cx - c(36), cy + c(95)), (cx + c(36), cy + c(160)), (150, 175, 215), -1)
    cv2.ellipse(layer, (cx, cy), (c(98), c(128)), 0, 0, 360, (150, 180, 225), -1)
    cv2.ellipse(layer, (cx, cy - c(72)), (c(103), c(72)), 0, 180, 360, (30, 30, 40), -1)
    for dx, tilt, pup in ((-40, -4, -2), (42, 6, 3)):   # slightly asymmetric, like real faces
        cv2.ellipse(layer, (cx + c(dx), cy - c(40)), (c(25), c(6)), tilt, 180, 360, (40, 40, 50), max(2, c(4)))
        cv2.ellipse(layer, (cx + c(dx), cy - c(18)), (c(19 + pup / 3), c(9)), tilt, 0, 360, (250, 250, 250), -1)
        cv2.circle(layer, (cx + c(dx + pup), cy - c(18)), c(7), (40, 30, 30), -1)
    cv2.fillPoly(layer, [np.array([[cx, cy - c(10)], [cx - c(13), cy + c(32)], [cx + c(13), cy + c(32)]])],
                 (135, 165, 210))
    cv2.ellipse(layer, (cx, cy + c(65)), (c(32), max(2, c(4 + 24 * mouth_open))), 0, 0, 360, (40, 40, 120), -1)
    layer = cv2.GaussianBlur(layer, (3, 3), 0).astype(np.float32)
    if rng is not None:
        layer += rng.normal(0, grain, layer.shape)
    return np.clip(layer, 0, 255).astype(np.uint8)


# ------------------------------------------------------------------ audio helpers
def synth_speech(seconds, seed, base_pitch=125, flat=False):
    rng = np.random.default_rng(seed)
    n = int(seconds * SR)
    env = np.zeros(n)
    t = 0.3
    while t < seconds - 0.4:
        d = rng.uniform(0.10, 0.28)
        a, b = int(t * SR), int((t + d) * SR)
        env[a:b] = np.hanning(b - a) * rng.uniform(0.5, 1.0)
        t += d + (rng.uniform(0.35, 0.9) if rng.random() < 0.15 else rng.uniform(0.03, 0.14))
    tt = np.arange(n) / SR
    if flat:
        f0 = np.full(n, base_pitch)
    else:
        f0 = base_pitch * (1 + 0.12 * np.sin(2 * np.pi * 0.35 * tt + seed) + 0.06 * np.sin(2 * np.pi * 1.7 * tt))
        f0 *= 1 + 0.012 * np.convolve(rng.normal(0, 1, n), np.ones(400) / 400 * 20, "same")
    phase = 2 * np.pi * np.cumsum(f0) / SR
    voice = sum((1 / k) * np.sin(k * phase) * (1.4 if 4 <= k <= 7 else 1) for k in range(1, 28))
    hiss = rng.normal(0, 1, n) * 0.05
    y = (voice + hiss) * env
    return y / (np.abs(y).max() + 1e-9) * 0.8, env


def room_tone(n, level, seed):
    rng = np.random.default_rng(seed)
    w = rng.normal(0, 1, n)
    pink = np.convolve(w, np.ones(8) / 8, "same")
    return pink * level


def mouth_track(env, n_frames):
    hop = SR // FPS
    m = np.array([env[i * hop:(i + 1) * hop].mean() if i * hop < len(env) else 0 for i in range(n_frames)])
    m = np.convolve(m, [0.25, 0.5, 0.25], "same")
    return np.clip(m / (m.max() + 1e-9), 0, 1)


# ------------------------------------------------------------------ builders
def make_photos():
    rng = np.random.default_rng(1)
    scene = draw_scene(rng)
    img = draw_face(scene, 620, 300, 0.9, 0.15, rng)
    exif = Image.Exif()
    exif[0x010F] = "Canon"
    exif[0x0110] = "Canon EOS 200D"
    exif[0x0132] = "2026:11:18 10:42:13"
    exif[0x0131] = "Firmware 1.0.1"
    Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)).save(OUT / "synthetic_scene.jpg", quality=92, exif=exif)

    # edited: clone the building, paste a smoothed face from "another photo", add old date, re-save
    ed = img.copy()
    ed[440:610, 700:940] = ed[250:420, 60:300]                       # copy-move clone (duplicate building)
    fx, fy = 620, 300
    face_box = (fx - 80, fy - 100, 160, 210)
    x, y, w, h = face_box
    patch = cv2.bilateralFilter(ed[y:y + h, x:x + w], 15, 60, 15)  # smooth "swapped" face (no sensor noise)
    patch = cv2.convertScaleAbs(patch, alpha=1.12, beta=18)         # different lighting
    buf = cv2.imencode(".jpg", patch, [cv2.IMWRITE_JPEG_QUALITY, 55])[1]
    ed[y:y + h, x:x + w] = cv2.imdecode(buf, cv2.IMREAD_COLOR)      # different compression history
    cv2.putText(ed, "15/08/2019", (30, 610), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (255, 255, 255), 3, cv2.LINE_AA)
    exif2 = Image.Exif()
    exif2[0x0131] = "Adobe Photoshop 25.0 (Windows)"
    exif2[0x0132] = "2026:11:18 11:05:40"
    Image.fromarray(cv2.cvtColor(ed, cv2.COLOR_BGR2RGB)).save(OUT / "edited_photo.jpg", quality=88, exif=exif2)
    print("photos done")


def write_video(path, frames_fn, n_frames, audio, W=960, H=640, created="2026-11-18T09:30:00Z"):
    tmp_v = path.with_suffix(".tmp.avi")
    tmp_a = path.with_suffix(".tmp.wav")
    vw = cv2.VideoWriter(str(tmp_v), cv2.VideoWriter_fourcc(*"MJPG"), FPS, (W, H))
    for i in range(n_frames):
        vw.write(frames_fn(i))
    vw.release()
    sf.write(tmp_a, audio.astype(np.float32), SR)
    subprocess.run([ffmpeg_bin(), "-y", "-loglevel", "error", "-i", str(tmp_v), "-i", str(tmp_a),
                    "-c:v", "libx264", "-pix_fmt", "yuv420p", "-preset", "veryfast", "-crf", "30",
                    "-c:a", "aac", "-b:a", "128k", "-shortest", "-metadata", f"creation_time={created}",
                    str(path)], check=True)
    tmp_v.unlink(missing_ok=True)
    tmp_a.unlink(missing_ok=True)


def make_videos(seconds=14):
    n = seconds * FPS
    rng = np.random.default_rng(2)
    bg = draw_scene(rng)
    speech, env = synth_speech(seconds, seed=11)
    audio = speech + room_tone(len(speech), 0.004, 3)
    mouth = mouth_track(env, n)

    def genuine(i):
        r = np.random.default_rng(100 + i)
        base = np.clip(bg.astype(np.float32) + r.normal(0, 3, bg.shape), 0, 255).astype(np.uint8)
        return draw_face(base, 480 + int(10 * np.sin(i / 30)), 300, 0.95, mouth[i], r)
    write_video(OUT / "animated_avatar.mp4", genuine, n, audio)
    print("genuine video done")

    # manipulated: lips driven by speech A, audio is a different speech B (dubbed / cloned voice)
    speech_b, _ = synth_speech(seconds, seed=77, base_pitch=118, flat=True)
    audio_b = speech_b + room_tone(len(speech_b), 0.004, 4)
    cut = int(8 * SR)
    audio_b[cut:cut + int(0.25 * SR)] = 0.0                           # spliced audio (digital silence)
    audio_b[cut + int(0.25 * SR):] += room_tone(len(audio_b) - cut - int(0.25 * SR), 0.03, 5)  # noise jump
    alt_bg = draw_scene(np.random.default_rng(9), grain=4)
    alt_bg = cv2.convertScaleAbs(alt_bg, alpha=0.8, beta=-25)

    def manipulated(i):
        r = np.random.default_rng(500 + i)
        t = i / FPS
        scene = alt_bg if 6.0 <= t < 6.6 else bg                      # hard cut / inserted shot
        base = np.clip(scene.astype(np.float32) + r.normal(0, 3, scene.shape), 0, 255).astype(np.uint8)
        fr = draw_face(base, 480 + int(10 * np.sin(i / 30)), 300, 0.95, mouth[i], r)
        if t >= 3.0:                                                  # "face-swapped" region from t = 3 s
            cx = 480 + int(10 * np.sin(i / 30))
            x, y, w, h = cx - 85, 185, 170, 230
            p = cv2.GaussianBlur(fr[y:y + h, x:x + w], (9, 9), 0)        # generated faces lack sensor noise
            p = cv2.convertScaleAbs(p, alpha=1.1, beta=22)               # different lighting / skin tone
            p[..., 0] = np.clip(p[..., 0].astype(int) - 18, 0, 255)      # colour cast
            mask = np.zeros((h, w), np.float32)
            cv2.ellipse(mask, (w // 2, h // 2), (w // 2 - 6, h // 2 - 6), 0, 0, 360, 1, -1)
            mask = cv2.GaussianBlur(mask, (15, 15), 0)[..., None]
            fr[y:y + h, x:x + w] = (p * mask + fr[y:y + h, x:x + w] * (1 - mask)).astype(np.uint8)
        cv2.rectangle(fr, (10, 8), (330, 52), (0, 0, 0), -1)          # CCTV-style timestamp
        cv2.putText(fr, "08-03-2019 14:22", (18, 42), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2,
                    cv2.LINE_AA)
        return fr
    write_video(OUT / "manipulated_talking.mp4", manipulated, n, audio_b, created="2019-03-08T14:22:00Z")
    print("manipulated video done")


def make_audio(seconds=16):
    y, _ = synth_speech(seconds, seed=21)
    y = y + room_tone(len(y), 0.004, 6)
    sf.write(OUT / "synthetic_voice.wav", y.astype(np.float32), SR)

    e, _ = synth_speech(seconds, seed=31, flat=True)
    e = e + room_tone(len(e), 0.004, 7)
    a, b = int(4 * SR), int(5.5 * SR)
    seg = e[a:b].copy()
    e[int(11 * SR):int(11 * SR) + len(seg)] = seg                    # repeated / looped segment
    s = int(7.5 * SR)
    e[s:s + int(0.3 * SR)] = 0.0                                     # digital-silence splice
    e[s + int(0.3 * SR):] += room_tone(len(e) - s - int(0.3 * SR), 0.035, 8)  # background-noise jump
    sf.write(OUT / "edited_audio.wav", (e / np.abs(e).max() * 0.9).astype(np.float32), SR)
    print("audio done")


def make_claims():
    (OUT / "claims.txt").write_text(
        "Use these captions in the 'Claim / caption' box during the demo:\n\n"
        "edited_photo.jpg:\n  This photo was taken today at our campus in Bangalore.\n\n"
        "manipulated_talking.mp4:\n  This video was recorded today at the university. The speaker said "
        "\"we will close the campus tomorrow\".\n\n"
        "animated_avatar.mp4:\n  Recorded on 18 November 2026 during the hackathon. Source: team camera\n")


if __name__ == "__main__":
    make_photos()
    make_audio()
    make_videos()
    make_claims()
    print(f"Samples written to {OUT}")
