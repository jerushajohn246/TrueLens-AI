"""Innocent explanations ("Could this be innocent?").

For every finding TruthLens lists the ordinary, non-malicious causes that could produce the same signal,
then TESTS each one against the media where possible:

    likely        - evidence supports the innocent cause      -> finding confidence x 0.70
    possible      - cannot be ruled out, some support          -> finding confidence x 0.88
    unlikely      - test was run and argues against it
    unverifiable  - cannot be tested automatically; listed for the human investigator

This answers the problem statement's requirement for "alternative explanations" and avoids
presenting a single signal as proof.
"""
from __future__ import annotations

SOCIAL_SIDES = {1600, 1280, 1080, 2048, 960, 720, 640, 1440}
SCREEN_SIZES = {(1080, 2400), (1080, 2340), (1080, 1920), (720, 1600), (1440, 3200), (1170, 2532), (1179, 2556),
                (1284, 2778), (1366, 768), (1920, 1080), (1536, 864), (1440, 900), (1280, 720), (2560, 1440)}
MULT = {"likely": 0.70, "possible": 0.88, "unlikely": 1.0, "unverifiable": 1.0}
RANK = {"likely": 3, "possible": 2, "unverifiable": 1, "unlikely": 0}


def _alt(cause, status, test, detail):
    return {"cause": cause, "status": status, "test": test, "detail": detail}


def _image_context(fc):
    info, meta = fc.get("info", {}), fc.get("metadata", {}) or {}
    w, h = info.get("width") or 0, info.get("height") or 0
    exif = meta.get("exif", {}) or {}
    social = (max(w, h) in SOCIAL_SIDES or min(w, h) == 1080) and not (exif.get("Make") or exif.get("Model"))
    screenshot = ((w, h) in SCREEN_SIZES or (h, w) in SCREEN_SIZES) and not exif.get("Make") and \
        (meta.get("format") == "PNG" or not exif)
    q = info.get("jpeg_quality_estimate")
    return w, h, exif, social, screenshot, q


def explain(f: dict, fc: dict) -> list[dict]:
    """Return a list of tested innocent explanations for one finding. fc = per-file context."""
    c = f.get("check", "")
    x = f.get("ctx", {}) or {}
    alts: list[dict] = []

    if c.startswith("image.") or c.startswith("metadata."):
        w, h, exif, social, screenshot, q = _image_context(fc) if fc.get("kind") == "image" else (0, 0, {}, False, False, None)
    else:
        w = h = 0
        exif, social, screenshot, q = {}, False, False, None

    if c == "image.ela":
        er = x.get("edge_ratio", 1)
        alts.append(_alt("High-contrast edges or text in that region", "likely" if er > 2.2 else "possible" if er > 1.5 else "unlikely",
                         "Compared edge density inside the flagged region with the whole image",
                         f"Region is {er:.1f}x as edge-heavy as the image average. "
                         + ("Sharp text/edges naturally show high ELA." if er > 1.5 else "The region is not unusually edge-heavy.")))
        alts.append(_alt("Social-media recompression (WhatsApp / Instagram)", "possible" if social else "unlikely",
                         "Checked image size and camera metadata for platform recompression patterns",
                         f"{w}x{h} px, " + ("no camera data - typical of a forwarded image." if social else "does not match common platform sizes.")))
        alts.append(_alt("Local brightness/contrast adjustment (not deceptive)", "unverifiable",
                         "Cannot distinguish honest retouching from content insertion automatically",
                         "Investigator should check whether the region's CONTENT changed or only its tone."))
    elif c == "image.noise":
        er, nr = x.get("edge_ratio", 1), x.get("region_noise_ratio", 1)
        alts.append(_alt("Out-of-focus / blurred area (shallow depth of field)", "possible" if nr < 0.8 and er < 0.8 else "unlikely",
                         "Checked whether the flagged region is both smooth and low in detail",
                         f"Noise ratio {nr:.2f}, edge ratio {er:.2f}. "
                         + ("A blurry background or out-of-focus subject also has low noise." if nr < 0.8 and er < 0.8 else "Region keeps detail, so blur does not explain it.")))
        alts.append(_alt("Over-exposed or very dark area (clipped pixels have no noise)", "unverifiable",
                         "Clipped pixels were excluded from the test already", "Residual risk: partially clipped regions."))
        alts.append(_alt("Phone 'beauty' / noise-reduction processing", "possible",
                         "Modern phones denoise selectively (faces, sky)", "Common on genuine phone photos; compare with other photos from the same device."))
    elif c == "image.copy_move":
        alts.append(_alt("Naturally repeating pattern (windows, tiles, fence, crowd)", "unlikely",
                         "Overlapping / periodic matches were already filtered out",
                         f"{x.get('pairs', '?')} matches share ONE offset and the source and copy areas do not overlap, "
                         "which is atypical for natural repetition."))
        alts.append(_alt("Duplicate objects that genuinely exist (identical products, printed logos)", "unverifiable",
                         "Requires scene knowledge", "Check whether two identical objects plausibly exist in the scene."))
    elif c == "image.face":
        gn = x.get("global_noise", 2)
        alts.append(_alt("Whole-photo beauty filter or noise reduction", "likely" if gn < 0.9 else "unlikely",
                         "Measured noise level of the whole image, not just the face",
                         f"Global noise residual {gn:.2f}. " + ("The entire photo is smoothed, so a smooth face is expected." if gn < 0.9 else "Only the face is affected, not the whole image.")))
        bd = x.get("brightness_diff") or 0
        alts.append(_alt("Make-up, flash or a spotlight on the face", "possible" if abs(bd) > 40 else "unlikely",
                         "Compared face brightness with its surroundings",
                         f"Face is {bd:+.0f} levels brighter/darker than surroundings."))
    elif c == "image.spectrum":
        alts.append(_alt("Image was resized or sharpened by an app", "possible",
                         "Resizing leaves periodic spectral peaks similar to AI up-sampling", "Indicator stays low-confidence for this reason."))
        alts.append(_alt("Printed halftone / screen photographed (moire)", "unverifiable", "Needs visual check", "Look for a screen or printed page in the photo."))
    elif c == "image.ai_dimensions":
        alts.append(_alt("Image was cropped/resized to a square by an app or website", "possible",
                         "Many tools export 1024x1024 or 512x512", "Dimensions alone are weak evidence."))
    elif c == "metadata.missing":
        alts.append(_alt("Metadata stripped by WhatsApp / Instagram / Facebook", "likely" if social else "possible",
                         "Checked size pattern of common platforms", f"{w}x{h} px" + (" matches platform recompression." if social else ".")))
        alts.append(_alt("Screenshot instead of camera photo", "likely" if screenshot else "unlikely",
                         "Checked for screen-resolution dimensions", f"{w}x{h} px" + (" is a screen resolution." if screenshot else " is not a typical screen size.")))
    elif c == "metadata.software":
        alts.append(_alt("Routine edits only (crop, rotate, colour correction)", "unverifiable",
                         "Software tag does not say what was edited",
                         "Combine with pixel-level findings: no ELA/noise/clone findings => likely routine editing."))
    elif c == "metadata.no_camera":
        alts.append(_alt("Exported from a gallery/editing app that drops camera fields", "possible", "Common behaviour", "Weak indicator."))
    elif c == "metadata.dates":
        alts.append(_alt("Photo was rotated or re-saved by the gallery app", "likely",
                         "Gallery apps update DateTime on any save", "A later modification time alone is very common."))
    elif c == "image.low_quality":
        alts.append(_alt("Forwarded through messaging apps", "likely", "Low quality is the typical result of recompression", "Not a manipulation sign by itself."))
    elif c == "metadata.ai_tags":
        alts.append(_alt("AI tool used only for upscaling/denoising a real photo", "possible",
                         "Generator tags are also written by AI enhancement tools",
                         "Check the prompt/parameters text: an upscaler leaves no scene prompt."))

    # ---------------- AI-generation signals
    elif c == "ai.c2pa":
        acts = (fc.get("provenance") or {}).get("actions", [])
        created = any(a.get("action") == "c2pa.created" for a in acts)
        alts.append(_alt("AI was used only to edit/enhance real footage", "unlikely" if created else "possible",
                         "Checked whether the manifest says the file was CREATED by AI or only edited",
                         "Manifest action is 'c2pa.created' by generative AI." if created else "Manifest lists edits only."))
        alts.append(_alt("Manifest copied from another file (forgery)", "unverifiable",
                         "Full cryptographic signature validation needs the c2pa toolkit",
                         "Signer name is read from the embedded certificate; verify with contentcredentials.org/verify for legal use."))
    elif c == "ai.ml_model":
        alts.append(_alt("Professional / stock / heavily edited media that looks 'too perfect'", "possible",
                         "Studio lighting, retouching and beauty filters resemble AI output",
                         "In validation, polished stock portraits were the most common false alarm."))
        alts.append(_alt("Model trained on a small dataset", "unverifiable",
                         "Accuracy on unseen files is limited by the number of training examples",
                         "Use alongside provenance and the other indicators; retrain with more examples to improve."))
    elif c == "ai.metadata_hint":
        alts.append(_alt("File was only downloaded/processed by that company's service (e.g. YouTube, Google Photos)", "possible",
                         "Container names show the last tool that wrote the file, not necessarily the creator", ""))
    elif c == "ai.no_sensor_grain":
        alts.append(_alt("Phone noise reduction / beauty mode / heavy compression", "possible",
                         "Modern phones and some apps denoise strongly",
                         "Real phone videos in calibration kept grain >= 1.0 even after WhatsApp compression."))
        alts.append(_alt("Screen recording of a game / animation (synthetic but not 'deepfake')", "possible", "", ""))
    elif c == "ai.face_bg_sharpness":
        alts.append(_alt("Portrait / cinematic mode or a professional camera with shallow depth of field", "possible",
                         "These blur the background optically", "Check device metadata or ask for the original file."))
    elif c == "ai.smooth_camera":
        alts.append(_alt("Gimbal / stabiliser or software stabilisation", "possible",
                         "Stabilised footage is also smooth", "Stabilisers usually leave small corrections; check for cropping wobble."))
    elif c == "ai.bg_warp":
        alts.append(_alt("Parallax (objects at different depths move differently when the camera moves)", "possible",
                         "Rigid-motion model ignores depth", "Strongest when the camera moves sideways near objects."))

    # ---------------- video
    elif c == "video.cuts":
        n, cv = x.get("n_cuts", 0), x.get("cut_interval_cv")
        alts.append(_alt("Normally edited news package / compilation / reel", "likely" if n >= 4 else "possible",
                         "Counted cuts and their regularity",
                         f"{n} cut(s)" + (f", interval variation {cv:.2f}" if cv is not None else "") + ". "
                         + ("Many cuts are normal for edited content - only suspicious if presented as one continuous recording." if n >= 4 else "A few cuts can be ordinary editing.")))
        alts.append(_alt("Camera switched off/on or app paused recording", "possible", "Cannot be distinguished from a splice automatically", ""))
    elif c == "video.frozen":
        alts.append(_alt("Streaming buffering / dropped connection while screen-recording", "possible",
                         "Frozen segments are common in screen recordings of live streams", ""))
    elif c in ("video.face_step", "video.face_anomaly"):
        step = x.get("frame_lightness_step")
        if step is not None:
            alts.append(_alt("Camera auto-exposure or room lighting changed", "likely" if step > 12 else "unlikely",
                             "Checked whether the WHOLE frame changed brightness at the same moment",
                             f"Whole-frame brightness {x.get('frame_lightness_before')} -> {x.get('frame_lightness_after')} "
                             f"(change {step}). " + ("The whole scene changed, so lighting explains the face change." if step > 12 else "Only the face changed, the scene did not.")))
        else:
            alts.append(_alt("Lighting change or person turning the head", "possible", "Head pose / lighting not fully modelled", ""))
        alts.append(_alt("Video call / app face filter switched on", "unverifiable",
                         "A live filter is technically a face manipulation but may be harmless", "Ask for the original recording."))
        alts.append(_alt("Strong compression on moving faces", "possible" if f.get("strength", 0) < 0.6 else "unlikely",
                         "Weak anomalies are often compression", ""))
    elif c == "video.face_flicker":
        alts.append(_alt("A different person entered the frame or the camera re-framed", "possible" if fc.get("info", {}).get("multi_face_ratio", 0) > 0.1 else "unlikely",
                         "Checked how often more than one face is visible",
                         f"More than one face in {fc.get('info', {}).get('multi_face_ratio', 0) * 100:.0f}% of key frames."))
    elif c == "metadata.video_encoder":
        alts.append(_alt("Downloaded from YouTube / Instagram / WhatsApp", "likely",
                         "Almost every shared video is re-encoded by the platform", "Not a sign of manipulation by itself."))
    elif c == "metadata.video_no_date":
        alts.append(_alt("Platform removed metadata on upload", "likely", "Normal for shared videos", ""))

    # ---------------- audio
    elif c == "audio.digital_silence":
        n, g = x.get("n_gaps", 0), x.get("mean_gap_sec", 0)
        gate = n >= 4 and g < 0.4
        alts.append(_alt("Noise gate / noise suppression (Zoom, Meet, WhatsApp voice notes)", "likely" if gate else "unlikely",
                         "Checked number and length of silent gaps",
                         f"{n} gap(s), average {g:.2f}s. " + ("Many short gaps are typical of call apps muting silence." if gate else "Few isolated gaps look more like cut points.")))
        alts.append(_alt("Speaker pressed mute during the recording", "possible" if n <= 2 else "unlikely", "", ""))
    elif c == "audio.noise_floor":
        alts.append(_alt("Fan / AC / traffic started or stopped", "possible" if not x.get("persistent") else "unlikely",
                         "Checked whether the new background level is held afterwards",
                         "The change is " + ("held for several seconds (like a new recording)." if x.get("persistent") else "short-lived (like a passing noise).")))
        alts.append(_alt("Speaker moved to another room while recording", "unverifiable", "", "Listen around the marked time."))
    elif c == "audio.pitch":
        alts.append(_alt("Naturally monotone speaker or someone reading aloud", "possible",
                         "Pitch flatness also occurs in formal reading", "Compare with another known recording of the same person."))
    elif c == "audio.pause_regularity":
        alts.append(_alt("Professionally edited podcast / audiobook (pauses trimmed)", "possible", "Editors normalise pauses", ""))
    elif c == "audio.bandwidth":
        codec = (fc.get("info", {}).get("audio_codec") or "").lower()
        br = fc.get("info", {}).get("bitrate_kbps") or 999
        tel = codec in ("amr_nb", "amr_wb", "opus", "gsm") or br < 64
        alts.append(_alt("Phone call or voice-note codec", "likely" if tel else "possible",
                         "Checked audio codec and bitrate", f"codec={codec or 'pcm'}, bitrate={br if br != 999 else 'n/a'} kb/s."))
    elif c == "audio.tonal":
        fr = x.get("freqs", [])
        hum = any(abs(v / 50 - round(v / 50)) < 0.04 for v in fr if v > 40)
        alts.append(_alt("Electrical mains hum (50 Hz in India and its harmonics)", "likely" if hum else "unlikely",
                         "Checked if tonal lines sit on multiples of 50 Hz",
                         f"Lines at {', '.join(f'{v:.0f}' for v in fr) or '-'} Hz" + (" - matches mains hum." if hum else " - not mains-related.")))
    elif c == "audio.repeat":
        alts.append(_alt("Chant, slogan or song chorus that genuinely repeats", "possible",
                         "Repeated words in chants can be acoustically similar", "Natural repeats are rarely sample-identical; listen to both segments."))
    elif c == "audio.speaker":
        alts.append(_alt("A second real person started speaking", "possible", "Conversations contain several voices", ""))
        alts.append(_alt("Microphone moved / changed (e.g. headset to laptop mic)", "possible", "", ""))

    # ---------------- cross-modal
    elif c == "crossmodal.av_mismatch":
        info = fc.get("info", {})
        fv = info.get("face_visible_ratio", 1)
        alts.append(_alt("Voice-over / narrator / off-screen speaker", "possible",
                         "Lip-sync assumes the visible person is the one speaking", "Common in news reports and documentaries."))
        alts.append(_alt("Face too small, turned or partly hidden for reliable lip tracking", "likely" if fv < 0.6 else "unlikely",
                         "Checked how often a face was visible", f"Face visible in {fv * 100:.0f}% of sampled frames."))
        alts.append(_alt("Several people talking in turns", "possible" if info.get("multi_face_ratio", 0) > 0.2 else "unlikely",
                         "Checked for multiple faces", f"Multiple faces in {info.get('multi_face_ratio', 0) * 100:.0f}% of key frames."))
    elif c == "crossmodal.av_offset":
        alts.append(_alt("Encoding / streaming delay between audio and video", "possible",
                         "Constant small offsets are common after re-uploading", "A constant offset with good correlation is more often technical than malicious."))
    elif c == "crossmodal.av_local":
        alts.append(_alt("Speaker briefly turned away or covered the mouth", "possible", "", "Check the marked window visually."))
    elif c == "crossmodal.caption_transcript":
        alts.append(_alt("Caption describes the scene, not the speech", "possible", "Captions often summarise context", ""))
    elif c == "crossmodal.quote":
        alts.append(_alt("Quote was paraphrased or translated", "possible", "Transcript compares words, not meaning", "Check translation and paraphrase."))
        alts.append(_alt("Speech-to-text misheard the words", "possible", "STT errors on names, accents and noise", ""))

    # ---------------- text
    elif c == "text.old_date":
        alts.append(_alt("Camera/CCTV clock set incorrectly", "possible",
                         "Burned-in timestamps are often wrong on CCTV and dashcams", "Compare with metadata date and visible events."))
        alts.append(_alt("Archive footage used intentionally (and labelled as such elsewhere)", "unverifiable", "", "Check the original post for 'file footage' labels."))
    elif c == "text.location":
        alts.append(_alt("News ticker / text about a different story", "possible", "On-screen text may not describe the scene", ""))
    elif c == "text.metadata_date":
        alts.append(_alt("Device clock was wrong or reset", "possible", "Metadata dates are easily wrong", ""))
    elif c == "text.no_source":
        alts.append(_alt("Uploader is the original creator", "unverifiable", "", "Ask the uploader for the original file."))
    return alts


def apply(findings: list[dict], file_ctx: dict) -> None:
    """Attach tested innocent explanations and set base_confidence (used by correlation + scoring)."""
    for f in findings:
        fc = file_ctx.get(f.get("file_id"), {}) if f.get("file_id") else {}
        alts = explain(f, fc)
        f["alternatives"] = alts
        f.setdefault("original_confidence", f["confidence"])
        best = max((a["status"] for a in alts), key=lambda s: RANK[s], default="unlikely")
        f["innocent_likelihood"] = {"likely": "high", "possible": "medium"}.get(best, "low")
        mult = MULT[best]
        f["base_confidence"] = round(f["original_confidence"] * mult, 3)
        f["confidence"] = f["base_confidence"]
