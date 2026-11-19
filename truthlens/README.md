# TruthLens — Multimodal Digital Forensics & Deepfake Detection

> **Problem Statement 04.** TruthLens examines **images, videos, audio and text** and reports whether media *may* be
> manipulated, AI-generated, misleading or inconsistent. For every finding it shows **what** was detected, **where**,
> **why** it is suspicious, the **confidence**, the **evidence**, and the **limitations**.
>
> *A detection score is only an indicator, not a final verdict.*

---

## 1. Run it in VS Code (5 minutes)

**Requirements:** Python 3.10 – 3.12 and VS Code with the Python extension.

```bash
# 1. Open the folder in VS Code  (File → Open Folder → truthlens)
# 2. Open the terminal (Ctrl + `) and create a virtual environment
python -m venv .venv

# 3. Activate it
.venv\Scripts\activate          # Windows
source .venv/bin/activate       # macOS / Linux

# 4. Install dependencies
pip install -r requirements.txt

# 5. (Optional) generate demo files into ./samples
python scripts/make_demo_samples.py

# 6. Start the server
python -m uvicorn app.main:app --reload --port 8000
```

Open **http://localhost:8000**.

**Shortcuts:**
- Double-click `run.bat` (Windows) or run `./run.sh` (Mac/Linux). Both do every step above.
- In VS Code, press **F5** → "TruthLens: run server (debug)". You can set breakpoints in any analyzer.

FFmpeg is bundled through `imageio-ffmpeg`, so you do **not** need to install it separately.

### Optional extras (local)

| Feature | What it enables | How to install |
|---|---|---|
| **OCR** (Tesseract) | Reads dates/text visible in images and video frames, which powers the old-date and location-mismatch checks | Windows: install from <https://github.com/UB-Mannheim/tesseract/wiki> and tick "Add to PATH". Mac: `brew install tesseract`. Linux: `sudo apt install tesseract-ocr` |
| **Speech-to-text** (faster-whisper) | Transcript, caption-vs-speech check, and the "quoted statement not in audio" check | `pip install -r requirements-optional.txt` (it downloads a 75 MB model the first time) |

The badges in the top-right of the app show which features are switched on. The app works without them; the report lists any checks it skipped under *Limitations*.

### Run the tests
```bash
pip install -r requirements-dev.txt
pytest -q
```

---

## 2. Deploy on Render

The included `Dockerfile` installs FFmpeg **and Tesseract OCR**, so OCR is on in the cloud automatically.

1. Push this folder to a **GitHub repository**:
   ```bash
   git init && git add . && git commit -m "TruthLens"
   git branch -M main
   git remote add origin https://github.com/<you>/truthlens.git
   git push -u origin main
   ```
2. Go to <https://dashboard.render.com> → **New +** → **Blueprint**, then pick the repo. Render reads `render.yaml`.
   *(Or choose **New + → Web Service**, select the repo, and set Runtime to **Docker**. No other settings are needed.)*
3. Wait for the build (~5–8 min the first time). Your app goes live at `https://truthlens-xxxx.onrender.com`.

**Notes for the free plan**
- It has 512 MB of RAM. Images, audio and videos up to about 60 s work well. `MAX_VIDEO_SECONDS=60` is preset in `render.yaml`.
- The disk is wiped on every redeploy, and the instance sleeps when idle, so history resets. The first request after sleeping takes about 30 s. **Open the site a minute before you present.**
- To turn on speech-to-text in the cloud, set the Docker build arg `INSTALL_WHISPER=1` in the Dockerfile and use the **Starter** plan.
- On a paid plan, uncomment the `disk:` block in `render.yaml` to keep history between deploys.

---

## 3. What the system does

```
 Upload (image / video / audio + claim)
      │  magic-byte type check · SHA-256 + MD5 · original stored read-only
      ▼
 ┌──────────────┬───────────────┬───────────────┬──────────────────┐
 │ Image        │ Video         │ Audio         │ Text / context   │
 │ analyzer     │ analyzer      │ analyzer      │ analyzer         │
 └──────┬───────┴──────┬────────┴──────┬────────┴────────┬─────────┘
        └──────────────┴───── Cross-modal correlation ───┘
                                │ lip-sync · caption↔speech · visible date↔claim · metadata↔claim
                                ▼
                     Evidence & risk aggregator  →  Dashboard + PDF/JSON report
```

### Checks implemented

| Modality | Check | What it catches |
|---|---|---|
| **Image** | EXIF / PNG metadata scan | AI-generator tags (Stable Diffusion, Midjourney…), editing software, missing camera data, date mismatch |
| | JPEG quantisation → quality estimate | heavy recompression |
| | **Error Level Analysis** (edge-aware, luminance) | regions with a different compression history (pasted / retouched) |
| | **Noise-residual consistency** (two-sided) | spliced content, or a face smoothed by a swap/beauty filter |
| | **Copy-move** (ORB self-matching + offset clustering) | cloned objects. Repetitive textures are filtered out |
| | **Face consistency** (YuNet CNN detector) | face sharpness, noise, compression, lighting and seam vs surroundings |
| | Frequency spectrum | periodic up-sampling artefacts (GAN / diffusion / resizing) |
| | AI-typical resolutions | 512², 1024², 832×1216 … with no camera data |
| **Video** | Key-frame sampling (1 per second) + light stream (12 fps) | fast: does not decode-analyse every frame |
| | Scene-cut detection | splices, with before/after thumbnails and timestamps |
| | Frozen / duplicated frames | padding over removed content |
| | **Temporal face analysis** (step-change + per-frame deviation) | face-swap switching on mid-video, blending flicker |
| | Face identity / size jumps | tracking failures of swap models |
| | Container metadata | encoder, creation time, re-encoding |
| **Audio** | Spectrogram (with marked segments) | visual evidence |
| | Digital-silence gaps | cut-and-join edits |
| | Background-noise floor jumps | joined recordings / takes |
| | Pause regularity | text-to-speech rhythm |
| | Pitch stability (jitter) | synthetic / cloned voice |
| | Bandwidth cutoff, tonal lines | resynthesis / processing |
| | Repeated segments | looped / copy-pasted speech |
| | Speaker-profile change | inserted segment / different speaker |
| **Text** | OCR (Tesseract) on images and video frames | visible dates, places, captions |
| | Speech-to-text (Whisper, optional) | transcript with timestamps |
| | Claim analysis | "today" vs a 2019 date on screen, wrong city, quote not spoken, no source, metadata date vs claim |
| **Cross-modal** | **Lip-sync**: mouth-landmark motion vs audio energy (cross-correlation, sliding windows) | dubbed audio, cloned voice, offset audio track |

### 🧠 Trained AI-detection models (v1.3)

The YES/NO banner is now decided by **three trained ML models**, one per media type. The C2PA provenance label still overrides them when it's present.

| Media | Features the model learns from | Trained on | Accuracy on files it never saw* |
|---|---|---|---|
| Image | Pre-trained **EfficientNet-Lite4** (ImageNet) embedding, whole image + 5 crops | 8 AI + 9 real | **71%** (12/17) |
| Video | Forensic frame features (grain, colour, sharpness, texture) + motion (camera path, warp, face/background focus), on 2 s clips | 4 AI + 3 real | **86%** (6/7) |
| Audio | Voice-behaviour features (pitch variation and jitter, harmonicity, pauses, spectral flux), on 3 s segments, band-limited to 4 kHz | 2 AI + 2 real | **100%** (4/4) |

\*Leave-one-file-out: every file was predicted by a model trained **without** it or any of its augmented copies.

- **Augmentation:** each training file is also used as a WhatsApp-style copy (resized, recompressed, metadata stripped; audio re-encoded as mp3/opus/aac), so the models learn content rather than file format.
- **Model:** class-balanced, L2-regularised logistic regression, exported to JSON. Run time needs only NumPy + OpenCV. Image ≈ 0.5 s, video ≈ 1–3 s, about 280 MB RAM.
- **Thresholds:** ≥ 60% → **YES**, 40–60% → **POSSIBLY**, < 40% → **NO**.
- **Retrain with more data:** put files in `training_data/…` and run `python scripts/train_ai_models.py` (see `training_data/README.md`).
- **Limitation:** with this few training files, expect mistakes on very different media (e.g. polished studio/stock photos can look AI-like). Accuracy grows with more examples.

### 🤖 AI-generated media verdict (v1.2) — shown first on every report

The first thing on the dashboard (and on page 1 of the PDF) is a clear answer: **AI-GENERATED VIDEO: YES / NO / POSSIBLY**, with an AI-likelihood % and the signals behind it. All other features follow below.

**Layer 1: Provenance (decisive).** TruthLens reads **C2PA Content Credentials** embedded by generators such as Google Veo/Gemini/Flow, OpenAI Sora/DALL-E and Adobe Firefly. If the manifest declares `digitalSourceType = trainedAlgorithmicMedia`, the verdict is **YES (confirmed)**. Example from a Google Veo video: *signed by Google C2PA Media Services · c2pa.created "Created by Google Generative AI." · c2pa.edited "Applied imperceptible SynthID watermark."*

**Layer 2: Visual signals (work after WhatsApp strips the metadata).**

| Signal | Real phone video | AI video (Veo sample) |
|---|---|---|
| Camera sensor grain | 1.0 – 1.6 | **0.51 – 0.58** (rendered, too clean) |
| Face sharpness ÷ background sharpness | 0.7 – 0.9 | **2.4 – 2.7** (synthetic focus) |
| Camera path | static or shaky handheld | **smooth continuous glide**, no hand tremor |
| Background warp after removing camera motion | ~0.017 px | **0.09 – 0.12 px** (room "breathes") |

**Verdict levels:** confirmed (C2PA) → risk ≥ 85 · likely (visual) → risk ≥ 65 · possible → risk ≥ 35 · not detected. Each AI signal has its own "Could this be innocent?" check (portrait mode, gimbal, phone denoising, parallax) and joins the correlation graph as the *AI-generated media* hypothesis.

> **Honest limitation:** the visual thresholds were calibrated on a small set of real vs AI videos. A trained classifier needs hundreds of labelled videos; with more samples the thresholds in `app/analyzers/ai_detector.py` can be re-tuned. Treat "likely" as a strong indicator for human review.

### ⭐ Unique features (v1.1)

#### 1. 🛡 "Could this be innocent?" — tested alternative explanations
Every finding lists the ordinary causes that could produce the same signal, and **tests each one against the media**:

| Finding | Innocent cause tested | How it is tested |
|---|---|---|
| ELA region | sharp text / edges | edge density inside the region vs the whole image |
| Face smoother than surroundings | whole-photo beauty filter | noise level of the entire image |
| Face changes mid-video | camera exposure / lighting change | did the **whole frame** change brightness at the same moment? |
| Digital-silence gaps | Zoom/WhatsApp noise gate | number and length of the gaps |
| Tonal lines in audio | 50 Hz mains hum (India) | are the lines multiples of 50 Hz? |
| Limited bandwidth | phone-call / voice-note codec | codec and bitrate from the container |
| Lip-sync mismatch | voice-over, face too small, several speakers | face-visibility ratio, multi-face ratio |
| Missing metadata | WhatsApp/Instagram stripping, screenshot | platform image sizes, screen resolutions |

Each cause is marked **likely / possible / unlikely / unverifiable**. A *likely* innocent cause lowers the finding's confidence (×0.70), and a *possible* one lowers it slightly (×0.88). This directly answers the problem statement's **"alternative explanations"** requirement.

#### 2. 🕸 Evidence Correlation Graph
- Findings become nodes, connected by **green "supports" edges** when independent checks agree: the same moment (e.g. a face anomaly at 00:03 **and** lip-sync breaking at 00:03), the same image region, or the same hypothesis seen through a different modality.
- **Red dashed "contradicts" edges** come from counter-evidence (e.g. *camera metadata present* vs "AI-generated", *lips match the audio* vs "voice clone") and from likely innocent causes.
- Confidence is adjusted transparently: **+8% per independent modality** (max +20%), **−10% per contradiction**. Each finding shows a *confidence trail*: `detector 55% → innocent cause 48% → corroborated +16% by Audio, Video → final 64%`.
- Groups of agreeing findings become **hypotheses**, e.g. *Face-swap* (75%), *Spliced recording* (66%), *Dubbed voice* (48%), each listing which modalities agree.
- Interactive graph in the dashboard (click a node to jump to its finding), plus a static graph page in the PDF.

#### 3. 🔏 Tamper-proof report: hash chain + signed seal + QR verification
- **Hash-chained audit trail:** every event stores `SHA-256(previous_hash | time | event)`, so editing or deleting any past event breaks the chain.
- **Signed report seal:** the SHA-256 of the canonical report, bound to the evidence hashes and the chain head, signed with HMAC-SHA256. Every investigator review creates a **new seal version**, and old versions stay listed.
- **QR code** on the PDF and dashboard opens `/verify/<id>`, which checks live:
  - ✅ the stored report matches its seal and the signature is valid
  - ✅ the scanned PDF is a genuine seal (and warns if a newer version exists)
  - ✅ the audit trail chain is intact
  - ✅ the original evidence files still match their upload hashes
- The verify page also lets anyone **upload a file** ("is this the same evidence?") or **upload a JSON report** ("was it edited?").

### Risk score (transparent, not a "fake probability")

| Category | Points |
|---|---|
| Image manipulation indicators | 25 |
| Video frame inconsistency | 20 |
| Audio irregularity | 20 |
| Audio-video / cross-modal mismatch | 20 |
| Metadata / source issue | 10 |
| Text-media contradiction | 5 |

- Each finding contributes `strength × (0.5 + 0.5 × confidence)`. Contributions within a category combine as `1 − Π(1 − c)`.
- Only categories that **apply** to the uploaded media are counted, and the total is normalised to 0–100.
- **0–24** Low indicators · **25–49** Needs attention · **50–74** High suspicion · **75–100** Very high suspicion.
- An investigator can **Accept / Reject / Flag** each finding. The score and confidence recompute live, and the action is written to the audit trail.

### The report (PDF and JSON)
It contains: file name, SHA-256/MD5, upload time, tools and models used, every finding (what/where/why/strength/confidence/review status), evidence images, risk breakdown, transcript and OCR, limitations, the full analysis history, and a human-review recommendation.

---

## 4. Demo script for the judges (about 4 minutes)

Run `python scripts/make_demo_samples.py` first. Captions to paste are in `samples/claims.txt`.

| # | Upload | Claim | What to point out |
|---|---|---|---|
| 1 | your own phone photo | — | **AI-GENERATED IMAGE: NO**. Shows we don't cry wolf (the bundled `synthetic_scene.jpg` was drawn by code, so TruthLens correctly says YES for it) |
| 2 | `edited_photo.jpg` | "This photo was taken today at our campus in Bangalore." | Copy-move boxes on the cloned building. ELA + noise map + face box all on the pasted face. *"15/08/2019"* read by OCR contradicts "today". Photoshop tag |
| 3 | `manipulated_talking.mp4` | the claim from claims.txt | Face-swap onset **at frame 75 (00:03.0)** with before/after frames. Lip-sync plot shows the mismatch. Click a timestamp → the video seeks. 2019 date on screen + 2019 creation_time vs "today". Audio splice in the spectrogram |
| 4 | `animated_avatar.mp4` | — | Computer-animated avatar → **AI-GENERATED VIDEO: YES (likely)**, from missing camera grain and a face sharper than the background |
| 4b | your own AI video (e.g. made with Gemini/Veo) | — | **YES, confirmed by Google C2PA** (original file) or **YES, likely** (WhatsApp copy) |
| 4c | your own real phone video | — | **AI-GENERATED VIDEO: NO** |
| 5 | — | — | Open the **Evidence correlation graph**: four hypotheses agree across video, audio and cross-modal |
| 6 | — | — | Expand **"Could this be innocent?"** on the face finding: "Camera exposure change → UNLIKELY, whole frame brightness unchanged" |
| 7 | — | — | Reject a finding → score updates and the report is **re-sealed v2** |
| 8 | — | — | Download the PDF → **scan its QR with a phone** → live ✅ verification page |
| 9 | — | — | Edit the downloaded JSON (change the score) → upload it on the verify page → ❌ **"modified after sealing"** |

Then test something real: a photo from your phone, or a WhatsApp-forwarded image.

**Line to say:**
> "TruthLens is an explainable multimodal digital-forensics platform. Instead of labelling media real or fake, it identifies suspicious indicators, links every finding to evidence, tests innocent explanations for each one, shows where independent signals agree or contradict in a correlation graph, and seals the report so anyone can verify by QR that nothing was altered."

**Note on Render's free plan:** the disk resets on redeploy, so old QR codes will show "analysis not found" afterwards. That's honest behaviour. For the live demo, generate a fresh report on the day.

---

## 5. Project structure

```
truthlens/
├── app/
│   ├── main.py              FastAPI routes (upload, status, review, reports, media)
│   ├── pipeline.py          workflow manager – runs analyzers in a background thread
│   ├── scoring.py           evidence & risk aggregator
│   ├── report_pdf.py        PDF report (ReportLab)
│   ├── analyzers/ai_detector.py ★ AI-generated verdict (C2PA provenance + trained models)
│   ├── ml/                  ★ features.py, embed.py (EfficientNet), predict.py
│   ├── models/              ★ ai_image.json, ai_video.json, ai_audio.json, effnet_lite4.onnx, YuNet
│   ├── alternatives.py      ★ "Could this be innocent?" tested explanations
│   ├── correlation.py       ★ evidence correlation graph + hypotheses
│   ├── integrity.py         ★ hash chain, signed seals, QR verification
│   ├── database.py          SQLite – analyses + audit history
│   ├── config.py, utils.py  settings, hashing, FFmpeg helpers
│   ├── models/              YuNet face detector (ONNX, 230 KB)
│   └── analyzers/
│       ├── image_analyzer.py
│       ├── video_analyzer.py
│       ├── audio_analyzer.py
│       ├── text_analyzer.py   OCR, Whisper, claim checks
│       └── crossmodal.py      lip-sync correlation
├── static/                  dashboard + verify.html (HTML/CSS/vanilla JS – no build step)
├── scripts/make_demo_samples.py, scripts/train_ai_models.py
├── training_data/           put your labelled AI / real files here to retrain
├── tests/test_smoke.py
├── Dockerfile · render.yaml · requirements*.txt · run.bat · run.sh · .vscode/
```

### API
| Method | Endpoint | Purpose |
|---|---|---|
| POST | `/api/analyze` | multipart `files` (1–5) + `claim` → `analysis_id` |
| GET | `/api/analyses/{id}` | status, progress, live log, result |
| GET | `/api/analyses` | history |
| POST | `/api/analyses/{id}/findings/{fid}/review` | `{status: accepted\|rejected\|needs_review, note}` |
| GET | `/api/analyses/{id}/report.pdf` / `report.json` | export |
| DELETE | `/api/analyses/{id}` | remove analysis + evidence |
| GET | `/api/analyses/{id}/qr.png` | QR code for the current seal |
| GET | `/api/analyses/{id}/verify?v=&d=&s=` | live integrity verification |
| POST | `/api/analyses/{id}/verify-file` | is this file the analysed evidence? |
| POST | `/api/verify-report` | was this JSON report edited? |
| GET | `/verify/{id}` | public verification page (QR target) |
| GET | `/docs` | interactive Swagger docs |

---

## 6. Limitations (be upfront about these)
- These are explainable forensic heuristics, **not** a deepfake classifier trained and validated on a benchmark. Scores are not calibrated probabilities.
- Expect false positives from heavy compression, filters, screenshots and repetitive textures. Expect false negatives from high-quality deepfakes.
- Lip-sync needs a visible, mostly frontal face.
- OCR and speech-to-text can misread text and speech.
- Shadow physics and weather consistency need human review.

## 7. Future scope
A trained deepfake classifier (e.g. on FaceForensics++) added as one more indicator · a calibration dashboard on a labelled test set · C2PA / content-credential verification · reverse image search for reused old content · speaker verification against a reference voice · batch analysis and case management.

## Privacy
All processing runs on your own server. No media is sent to third-party AI APIs. Originals are stored read-only and hashed, and each analysis can be deleted from History.
