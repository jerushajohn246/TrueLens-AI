/* TruthLens front-end (vanilla JS, no build step). */
"use strict";

const $ = (s, el = document) => el.querySelector(s);
const $$ = (s, el = document) => [...el.querySelectorAll(s)];
const app = $("#app");
const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
const fmtTs = (t) => { t = Math.max(0, +t || 0); const m = Math.floor(t / 60); return `${String(m).padStart(2, "0")}:${(t - m * 60).toFixed(1).padStart(4, "0")}`; };
const pct = (x) => `${Math.round((x || 0) * 100)}%`;
const LEVEL_COLOR = { low: "#22c55e", medium: "#f59e0b", high: "#ef4444", very_high: "#ff6b6b" };
const CAT_LABEL = { image: "Image", video: "Video", audio: "Audio", crossmodal: "Cross-modal", metadata: "Metadata", text: "Text / context" };

let pollTimer = null;
let current = null;           // current analysis record
let findingFilter = "all";

async function api(url, opts = {}) {
  const r = await fetch(url, opts);
  if (!r.ok) {
    let msg = r.statusText;
    try { msg = (await r.json()).detail || msg; } catch (_) { /* ignore */ }
    throw new Error(msg);
  }
  return r.json();
}

function toast(msg, ms = 2600) {
  const t = $("#toast");
  t.textContent = msg;
  t.classList.remove("hidden");
  clearTimeout(t._h);
  t._h = setTimeout(() => t.classList.add("hidden"), ms);
}

function lightbox(src, cap = "") {
  const lb = $("#lightbox");
  $("img", lb).src = src;
  $(".lb-cap", lb).textContent = cap;
  lb.classList.remove("hidden");
}
$("#lightbox").addEventListener("click", () => $("#lightbox").classList.add("hidden"));
document.addEventListener("keydown", (e) => { if (e.key === "Escape") $("#lightbox").classList.add("hidden"); });

/* ---------------------------------------------------------------- router */
function route() {
  clearInterval(pollTimer);
  const h = location.hash || "#/";
  $$("[data-nav]").forEach((a) => a.classList.toggle("active",
    (h === "#/" && a.dataset.nav === "home") || (h.startsWith("#/history") && a.dataset.nav === "history")));
  if (h.startsWith("#/analysis/")) return showAnalysis(h.split("/")[2]);
  if (h.startsWith("#/history")) return showHistory();
  return showHome();
}
window.addEventListener("hashchange", route);

async function loadCaps() {
  try {
    const c = await api("/api/capabilities");
    $("#caps").innerHTML = `
      <span class="cap on">FFmpeg</span>
      <span class="cap ${c.ocr ? "on" : "off"}" title="Tesseract OCR">OCR ${c.ocr ? "on" : "off"}</span>
      <span class="cap ${c.whisper === "available" ? "on" : "off"}" title="faster-whisper speech-to-text">Speech-to-text ${c.whisper === "available" ? "on" : "off"}</span>
      <span class="cap">v${esc(c.version)}</span>`;
  } catch (_) { /* server not ready */ }
}

/* ---------------------------------------------------------------- home / upload */
let selected = [];
function showHome() {
  app.innerHTML = $("#tpl-home").innerHTML;
  selected = [];
  const input = $("#file"), drop = $("#drop");
  drop.addEventListener("click", () => input.click());
  input.addEventListener("change", () => addFiles(input.files));
  ["dragenter", "dragover"].forEach((ev) => drop.addEventListener(ev, (e) => { e.preventDefault(); drop.classList.add("over"); }));
  ["dragleave", "drop"].forEach((ev) => drop.addEventListener(ev, (e) => { e.preventDefault(); drop.classList.remove("over"); }));
  drop.addEventListener("drop", (e) => addFiles(e.dataTransfer.files));
  $("#go").addEventListener("click", submit);
}

function kindOf(f) {
  if (f.type.startsWith("image/")) return "image";
  if (f.type.startsWith("video/")) return "video";
  if (f.type.startsWith("audio/")) return "audio";
  const ext = f.name.split(".").pop().toLowerCase();
  if (["mp4", "mov", "mkv", "webm", "avi", "m4v"].includes(ext)) return "video";
  if (["wav", "mp3", "m4a", "flac", "ogg", "aac"].includes(ext)) return "audio";
  if (["jpg", "jpeg", "png", "webp", "bmp", "tif", "tiff"].includes(ext)) return "image";
  return "unknown";
}

function addFiles(list) {
  for (const f of list) {
    if (selected.length >= 5) { toast("Maximum 5 files"); break; }
    if (kindOf(f) === "unknown") { toast(`Unsupported: ${f.name}`); continue; }
    selected.push(f);
  }
  renderFileList();
}

function renderFileList() {
  const ul = $("#filelist");
  ul.innerHTML = selected.map((f, i) => `
    <li><span class="ftype">${kindOf(f)}</span><span class="fname">${esc(f.name)}</span>
    <span class="muted">${(f.size / 1e6).toFixed(2)} MB</span><button class="x" data-i="${i}" title="Remove">✕</button></li>`).join("");
  $$(".x", ul).forEach((b) => b.addEventListener("click", () => { selected.splice(+b.dataset.i, 1); renderFileList(); }));
  $("#go").disabled = selected.length === 0;
}

async function submit() {
  const fd = new FormData();
  selected.forEach((f) => fd.append("files", f));
  fd.append("claim", $("#claim").value);
  $("#go").disabled = true;
  $("#go").textContent = "Uploading…";
  try {
    const r = await api("/api/analyze", { method: "POST", body: fd });
    location.hash = `#/analysis/${r.analysis_id}`;
  } catch (e) {
    toast("Upload failed: " + e.message, 5000);
    $("#go").disabled = false;
    $("#go").textContent = "Analyze media";
  }
}

/* ---------------------------------------------------------------- progress + report */
async function showAnalysis(id) {
  let rec;
  try { rec = await api(`/api/analyses/${id}`); } catch (e) { app.innerHTML = `<div class="card">Analysis not found.</div>`; return; }
  if (rec.status === "done") return renderReport(rec);
  if (rec.status === "error") { app.innerHTML = `<div class="card"><h2>Analysis failed</h2><p>${esc(rec.message)}</p><a class="btn" href="#/">Try again</a></div>`; return; }
  app.innerHTML = $("#tpl-progress").innerHTML;
  const tick = async () => {
    try {
      const r = await api(`/api/analyses/${id}`);
      $("#prog-bar").style.width = `${r.progress}%`;
      $("#prog-pct").textContent = `${r.progress}%`;
      $("#prog-msg").textContent = r.message || "";
      $("#prog-log").innerHTML = r.history.map((h) => `<li>${esc(h.ts.slice(11, 19))} — ${esc(h.event)}</li>`).join("");
      $("#prog-log").scrollTop = 1e6;
      if (r.status === "done" || r.status === "error") { clearInterval(pollTimer); showAnalysis(id); }
    } catch (_) { /* keep polling */ }
  };
  tick();
  pollTimer = setInterval(tick, 1000);
}

function aiBanner(r) {
  const v = r.ai_verdict;
  if (!v) return "";
  const kind = v.media_kind === "image" ? "IMAGE" : v.media_kind === "audio" ? "AUDIO" : "VIDEO";
  const icon = { confirmed: "🤖", likely: "🤖", possible: "❓", not_detected: "✅" }[v.status];
  const pctv = Math.round(v.score * 100);
  const prov = v.provenance || {};
  const sig = (v.signals || []).map((s) => `<tr class="${s.fired ? "fired" : ""}"><td>${s.fired ? "●" : "○"}</td><td>${esc(s.name)}</td>
      <td class="mono">${s.value === null || s.value === undefined ? "—" : esc(s.value)}</td><td class="muted">${esc(s.normal || "")}</td><td class="muted">${esc(s.ai || "")}</td></tr>`).join("");
  const others = (r.ai_verdicts || []).length > 1 ? `<div class="ai-files">${r.ai_verdicts.map((x) => `<span class="ai-chip s-${x.status}">${esc(x.file)}: <b>${x.label}</b></span>`).join("")}</div>` : "";
  return `<section class="ai-banner s-${v.status}">
    <div class="ai-main">
      <div class="ai-icon">${icon}</div>
      <div class="ai-text">
        <div class="ai-q">Is this ${kind.toLowerCase()} AI-generated?</div>
        <div class="ai-answer">AI-GENERATED ${kind}: <span>${v.label}</span></div>
        <div class="ai-head">${esc(v.headline)}${v.status === "confirmed" ? " — verified" : ""}</div>
        <div class="ai-basis">${esc(v.basis)}</div>
      </div>
      <div class="ai-meter"><div class="ai-pct">${pctv}%</div><div class="muted" style="font-size:.72rem">AI likelihood</div>
        <div class="bd-bar" style="width:120px"><div style="width:${pctv}%;background:currentColor"></div></div></div>
    </div>
    ${v.ml ? `<div class="ai-c2pa">🧠 <b>Trained AI-detection model:</b> ${Math.round(v.ml.p_ai * 100)}% AI
      · trained on ${v.ml.trained_on ? `${v.ml.trained_on.ai} AI + ${v.ml.trained_on.real} real files` : "labelled examples"}
      · accuracy on unseen files ${Math.round((v.ml.heldout_accuracy || 0) * 100)}%
      ${(v.ml.explain || []).length ? `<div class="muted" style="font-size:.78rem;margin-top:3px">Main factors: ${v.ml.explain.slice(0, 3).map((e) => `${esc(e.feature.replace(/_/g, " "))} → ${e.direction}`).join(" · ")}</div>` : ""}</div>` : ""}
    ${prov.ai_declared ? `<div class="ai-c2pa">🔏 <b>Content Credentials (C2PA)</b>${prov.signer ? ` signed by <b>${esc(prov.signer)}</b>` : ""}:
      ${(prov.actions || []).map((a) => `<span class="mono">${esc(a.action)}</span> “${esc(a.description)}”${a.digitalSourceType ? ` <span class="muted">(${esc(a.digitalSourceType)})</span>` : ""}`).join(" · ")}</div>` : ""}
    ${others}
    <details class="ai-details"><summary>How was this decided? (${(v.signals || []).filter((s) => s.fired).length} of ${(v.signals || []).length} signals fired)</summary>
      <table class="tbl ai-tbl"><thead><tr><th></th><th>Signal</th><th>Measured</th><th>Real media</th><th>AI media</th></tr></thead><tbody>${sig}</tbody></table>
      <p class="muted" style="font-size:.75rem;margin:8px 0 0">Provenance (C2PA) labels are decisive when present but are removed by WhatsApp/Instagram; visual signals still work on forwarded copies. Visual thresholds are calibrated on sample real and AI videos — treat “likely” as a strong indicator for human review, not legal proof.</p>
    </details>
  </section>`;
}

function gauge(score, level, label) {
  const r = 92, c = 2 * Math.PI * r, off = c * (1 - score / 100);
  return `<div class="gauge">
    <svg width="230" height="230" viewBox="0 0 230 230">
      <circle cx="115" cy="115" r="${r}" stroke="#1f2940" stroke-width="18" fill="none"/>
      <circle cx="115" cy="115" r="${r}" stroke="${LEVEL_COLOR[level]}" stroke-width="18" fill="none" stroke-linecap="round"
        stroke-dasharray="${c}" stroke-dashoffset="${off}" style="transition: stroke-dashoffset .8s"/>
    </svg>
    <div class="val"><div><div class="num lvl-${level}">${Math.round(score)}</div>
      <div class="lab lvl-${level}">${esc(label)}</div><div class="sub">risk score / 100</div></div></div></div>`;
}

function renderReport(rec) {
  current = rec;
  const r = rec.result;
  const nFind = r.findings.length;
  const high = r.findings.filter((f) => f.severity === "high").length;
  app.innerHTML = `
    ${aiBanner(r)}
    <section class="card">
      <div class="report-head">
        <div id="gauge">${gauge(r.risk_score, r.risk_level, r.risk_label)}</div>
        <div>
          <div class="row-between"><h2 style="margin:0">Forensic report <span class="mono muted">#${esc(r.analysis_id)}</span></h2>
            <span class="muted" style="font-size:.8rem">Completed ${esc(r.completed_at.replace("T", " "))} UTC · ${r.processing_seconds}s</span></div>
          <div class="kpis">
            <div class="kpi"><div class="k">Overall risk</div><div class="v lvl-${r.risk_level}" id="k-risk">${esc(r.risk_label)}</div></div>
            <div class="kpi"><div class="k">Confidence</div><div class="v" id="k-conf">${pct(r.confidence)}</div></div>
            <div class="kpi"><div class="k">Indicators</div><div class="v">${nFind} <span class="muted" style="font-size:.8rem">(${high} high)</span></div></div>
            <div class="kpi"><div class="k">Files</div><div class="v">${r.files.length}</div></div>
          </div>
          <div class="summary" id="summary">${esc(r.summary)}</div>
          ${r.ai_floor ? `<div class="muted" style="font-size:.78rem;margin-top:6px">Risk raised from ${r.ai_floor.from} to ${r.ai_floor.to} because: ${esc(r.ai_floor.reason)}.</div>` : ""}
          <div class="rec ${r.risk_score < 25 ? "ok" : ""}" id="recommend">👤 ${esc(r.recommendation)}</div>
          ${r.claim ? `<div class="claim-box"><b>Claim checked:</b> “${esc(r.claim)}”</div>` : ""}
          <div class="disclaimer">${esc(r.disclaimer)}</div>
          <div class="actions">
            <a class="btn primary" href="/api/analyses/${r.analysis_id}/report.pdf" target="_blank">⬇ Download PDF report</a>
            <a class="btn" href="/api/analyses/${r.analysis_id}/report.json" download>⬇ JSON report</a>
            <a class="btn" href="#/">+ New analysis</a>
          </div>
        </div>
      </div>
    </section>

    <div class="grid2">
      <div>
        <section class="card" id="media-card"><h2>Evidence viewer</h2><div id="media"></div></section>
      </div>
      <div>
        <section class="card"><h2>Risk breakdown</h2><div id="breakdown"></div>
          <p class="muted" style="font-size:.78rem;margin-bottom:0">Score = presence of manipulation indicators, normalised over the categories that apply to the uploaded media. Not a "fake probability".</p></section>
        <section class="card" id="seal-card"></section>
        <section class="card" id="text-card"></section>
        <section class="card" id="files-card"></section>
        <section class="card" id="meta-card"></section>
      </div>
    </div>
    <section class="card" id="graph-card"></section>
    <section class="card"><div class="row-between"><h2>Findings — what, where, why, how confident</h2><span class="muted" style="font-size:.8rem">Accept / reject each finding to update the score (investigator in the loop)</span></div>
      <div class="filter-row" id="filters"></div><div id="findings" class="findings-grid"></div></section>
    </div>`;
  renderBreakdown(r);
  renderFilters(r);
  renderFindings(r);
  renderMedia(r);
  renderText(r);
  renderFiles(rec);
  renderGraph(r);
  renderSeal(r);
}

function renderBreakdown(r) {
  $("#breakdown").innerHTML = r.breakdown.map((b) => {
    const p = b.max_points ? (b.points / b.max_points) * 100 : 0;
    const col = p >= 60 ? "var(--high)" : p >= 30 ? "var(--medium)" : "var(--low)";
    return `<div class="bd-row ${b.applicable ? "" : "na"}">
      <div class="bd-top"><span>${esc(b.label)} <span class="muted">(${b.indicators})</span></span>
      <span class="mono">${b.applicable ? `${b.points} / ${b.max_points}` : "n/a"}</span></div>
      <div class="bd-bar"><div style="width:${p}%;background:${col}"></div></div></div>`;
  }).join("");
}

function renderFilters(r) {
  const cats = ["all", ...new Set(r.findings.map((f) => f.category))];
  $("#filters").innerHTML = cats.map((c) => `<button class="tab ${c === findingFilter ? "active" : ""}" data-c="${c}">
    ${c === "all" ? `All (${r.findings.length})` : `${CAT_LABEL[c] || c} (${r.findings.filter((f) => f.category === c).length})`}</button>`).join("");
  $$("#filters .tab").forEach((b) => b.addEventListener("click", () => { findingFilter = b.dataset.c; renderFilters(r); renderFindings(r); }));
}

function locHtml(f) {
  const l = f.location || {};
  const seek = (t) => `<a data-seek="${t}" data-file="${f.file_id || ""}">${fmtTs(t)}</a>`;
  switch (l.type) {
    case "timestamp": {
      const extra = (l.all || []).filter((t) => t !== l.t).slice(0, 6).map(seek).join(" ");
      return `⏱ ${seek(l.t)}${l.frame !== undefined ? ` · frame ${l.frame}` : ""}${l.end ? ` → ${fmtTs(l.end)}` : ""}${extra ? ` · also ${extra}` : ""}`;
    }
    case "segment": return `🎚 segment ${seek(l.start)} – ${fmtTs(l.end)}${l.source !== undefined ? ` (copy of ${seek(l.source)})` : ""}`;
    case "region": return `🔲 ${esc(l.label || "region")}${l.bbox ? ` at x=${l.bbox[0]}, y=${l.bbox[1]} (${l.bbox[2]}×${l.bbox[3]} px)` : ""}`;
    case "metadata": return `🏷 metadata field: <span class="mono">${esc(l.field)}</span>`;
    case "text": return `📝 “${esc(String(l.passage || "").slice(0, 120))}”${l.source ? ` (${esc(l.source)})` : ""}`;
    default: return "";
  }
}

function renderFindings(r) {
  const list = r.findings.filter((f) => findingFilter === "all" || f.category === findingFilter);
  if (!list.length) { $("#findings").innerHTML = `<p class="muted">No indicators raised. Absence of indicators is not proof of authenticity.</p>`; return; }
  $("#findings").innerHTML = list.map((f) => `
    <div class="finding sev-${f.severity} rv-${f.review_status}" id="f-${f.id}">
      <div class="f-head"><div class="f-title">${esc(f.title)}</div><span class="chip ${f.severity}">${f.severity}</span></div>
      <div class="f-meta"><span class="chip cat">${esc(CAT_LABEL[f.category] || f.category)}</span>
        <span>strength <span class="meter"><i style="width:${f.strength * 100}%"></i></span> ${f.strength.toFixed(2)}</span>
        <span>confidence <span class="meter"><i style="width:${f.confidence * 100}%;background:var(--accent2)"></i></span> ${pct(f.confidence)}</span>
        ${f.file ? `<span>📄 ${esc(f.file)}</span>` : ""}</div>
      <div class="f-loc">${locHtml(f)}</div>
      <div class="f-exp">${esc(f.explanation)}</div>
      ${confTrail(f)}
      ${innocentBlock(f)}
      ${f.evidence?.length ? `<div class="f-ev">${f.evidence.map((u) => `<img src="${u}" data-lb="${u}" alt="evidence" loading="lazy" title="${esc(u.split("/").pop())}"/>`).join("")}</div>` : ""}
      <div class="f-review">
        ${["accepted", "rejected", "needs_review"].map((s) => `<button class="btn sm ${f.review_status === s ? "sel-" + s : ""}" data-rv="${s}" data-id="${f.id}">
          ${{ accepted: "✔ Accept", rejected: "✖ Reject", needs_review: "⚑ Needs manual review" }[s]}</button>`).join("")}
        <input placeholder="Investigator note (optional)" value="${esc(f.review_note || "")}" data-note="${f.id}" />
      </div>
    </div>`).join("");
  $$("#findings [data-lb]").forEach((im) => im.addEventListener("click", () => lightbox(im.dataset.lb, im.title)));
  $$("#findings [data-seek]").forEach((a) => a.addEventListener("click", () => seekTo(+a.dataset.seek, a.dataset.file)));
  $$("#findings [data-rv]").forEach((b) => b.addEventListener("click", () => review(b.dataset.id, b.dataset.rv)));
}

function confTrail(f) {
  if (f.original_confidence === undefined) return "";
  const c = f.correlation || {};
  const parts = [`<span>detector ${pct(f.original_confidence)}</span>`];
  if (f.base_confidence !== f.original_confidence) parts.push(`<span class="neg">innocent cause → ${pct(f.base_confidence)}</span>`);
  if (c.delta) parts.push(`<span class="${c.delta > 0 ? "pos" : "neg"}">${c.delta > 0 ? "corroborated" : "contradicted"} ${c.delta > 0 ? "+" : ""}${Math.round(c.delta * 100)}%${c.independent_modalities?.length ? " by " + c.independent_modalities.map((m) => CAT_LABEL[m] || m).join(", ") : ""}</span>`);
  if (f.review_status === "accepted") parts.push(`<span class="pos">investigator accepted → 100%</span>`);
  parts.push(`<b>final ${pct(f.review_status === "accepted" ? 1 : f.confidence)}</b>`);
  return `<div class="trail" title="How this finding's confidence was calculated">${parts.join('<i>→</i>')}</div>`;
}

function innocentBlock(f) {
  const alts = f.alternatives || [];
  if (!alts.length) return "";
  const lk = { high: "An innocent cause is LIKELY", medium: "An innocent cause is possible", low: "Innocent causes tested and unlikely" }[f.innocent_likelihood] || "";
  return `<details class="innocent ${f.innocent_likelihood}"><summary>🛡 Could this be innocent? <span class="muted">— ${lk}</span></summary>
    <ul>${alts.map((a) => `<li><span class="st st-${a.status}">${a.status}</span> <b>${esc(a.cause)}</b>
      ${a.detail ? `<div class="alt-d">${esc(a.detail)}</div>` : ""}${a.test ? `<div class="alt-t">Test: ${esc(a.test)}</div>` : ""}</li>`).join("")}</ul></details>`;
}

async function review(fid, status) {
  const r = current.result;
  const f = r.findings.find((x) => x.id === fid);
  const note = $(`[data-note="${fid}"]`)?.value || "";
  const newStatus = f.review_status === status ? "pending" : status;
  try {
    const res = await api(`/api/analyses/${r.analysis_id}/findings/${fid}/review`, {
      method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ status: newStatus, note }),
    });
    r.findings = res.findings; r.correlation = res.correlation; r.seal = res.seal;
    r.seals = [...(r.seals || []), res.seal];
    Object.assign(r, { risk_score: res.risk_score, risk_label: res.risk_label, risk_level: res.risk_level,
      confidence: res.confidence, breakdown: res.breakdown, summary: res.summary, recommendation: res.recommendation });
    $("#gauge").innerHTML = gauge(r.risk_score, r.risk_level, r.risk_label);
    $("#k-risk").textContent = r.risk_label; $("#k-risk").className = `v lvl-${r.risk_level}`;
    $("#k-conf").textContent = pct(r.confidence);
    $("#summary").textContent = r.summary;
    $("#recommend").textContent = "👤 " + r.recommendation;
    renderBreakdown(r); renderFindings(r); renderGraph(r); renderSeal(r);
    toast(`Finding marked: ${newStatus.replace("_", " ")} — score ${r.risk_score}, report re-sealed v${res.seal.version}`);
  } catch (e) { toast("Review failed: " + e.message); }
}

/* ---------------------------------------------------------------- media viewers */
function seekTo(t, fileId) {
  let v = fileId ? $(`video[data-fid="${fileId}"], audio[data-fid="${fileId}"]`) : null;
  v = v || $("video") || $("audio");
  if (!v) return;
  v.currentTime = t;
  v.scrollIntoView({ behavior: "smooth", block: "center" });
  v.play?.().catch(() => {});
  setTimeout(() => v.pause?.(), 1500);
}

function infoGrid(obj, skip = []) {
  return `<div class="info-grid">${Object.entries(obj || {}).filter(([k, v]) => v !== null && v !== undefined && typeof v !== "object" && !skip.includes(k))
    .map(([k, v]) => `<div><b>${esc(k.replace(/_/g, " "))}</b>${esc(v)}</div>`).join("")}</div>`;
}

function renderMedia(r) {
  const m = r.media, host = $("#media");
  let html = "";
  const fileById = Object.fromEntries(r.files.map((f) => [f.id, f]));

  (m.image || []).forEach((im, i) => {
    const f = fileById[im.file_id] || {};
    const views = [["overlay", "Flagged regions"], ["original", "Original"], ["ela", "Error Level Analysis"], ["noise", "Noise map"], ["copy_move", "Copy-move"], ["spectrum", "Frequency spectrum"]]
      .filter(([k]) => im.evidence[k]);
    html += `<div class="media-block" data-img="${i}">
      <div class="media-title"><span class="ftype">image</span><b>${esc(f.filename)}</b></div>
      <div class="tabs">${views.map(([k, lab], j) => `<button class="tab ${j === 0 ? "active" : ""}" data-view="${im.evidence[k]}" data-cap="${lab}">${lab}</button>`).join("")}</div>
      <div class="viewer"><img src="${im.evidence[views[0][0]]}" data-lb="${im.evidence[views[0][0]]}" alt="image evidence"/></div>
      ${infoGrid(im.info)}
    </div>`;
  });

  (m.video || []).forEach((vd) => {
    const f = fileById[vd.file_id] || {};
    const sus = new Set((vd.evidence.suspicious_frames || []).map((s) => s.frame));
    html += `<div class="media-block">
      <div class="media-title"><span class="ftype">video</span><b>${esc(f.filename)}</b></div>
      <div class="viewer"><video controls preload="metadata" data-fid="${vd.file_id}" src="/api/analyses/${r.analysis_id}/media/${vd.file_id}"></video></div>
      <div class="timeline-wrap"><canvas class="timeline" data-tl="${vd.file_id}"></canvas>
        <div class="tl-legend"><span><i style="background:#5eb1ff"></i>visual change</span><span><i style="background:#ffb020"></i>mouth motion</span>
        <span><i style="background:#ef4444"></i>suspicious timestamp</span><span><i style="background:rgba(239,68,68,.25)"></i>suspicious segment</span><span class="muted">click to seek</span></div></div>
      <div class="strip">${(vd.evidence.keyframes || []).map((k) => `<figure class="${sus.has(k.frame) ? "sus" : ""}" data-seek="${k.t}" data-file="${vd.file_id}">
        <img src="${k.url}" loading="lazy" alt="frame"/><figcaption>${fmtTs(k.t)}</figcaption></figure>`).join("")}</div>
      ${vd.av_sync?.plot ? `<img class="spec" src="${vd.av_sync.plot}" data-lb="${vd.av_sync.plot}" alt="sync plot"/>` : vd.av_sync ? `<p class="muted" style="font-size:.82rem">Lip-sync: ${esc(vd.av_sync.status)}</p>` : ""}
      ${vd.audio?.evidence?.spectrogram ? `<img class="spec" src="${vd.audio.evidence.spectrogram}" data-lb="${vd.audio.evidence.spectrogram}" alt="spectrogram"/>` : ""}
      ${infoGrid(vd.info, ["container_tags"])}
    </div>`;
  });

  (m.audio || []).forEach((au) => {
    const f = fileById[au.file_id] || {};
    html += `<div class="media-block">
      <div class="media-title"><span class="ftype">audio</span><b>${esc(f.filename)}</b></div>
      <audio controls style="width:100%" data-fid="${au.file_id}" src="/api/analyses/${r.analysis_id}/media/${au.file_id}"></audio>
      ${au.evidence.spectrogram ? `<img class="spec" src="${au.evidence.spectrogram}" data-lb="${au.evidence.spectrogram}" alt="spectrogram"/>` : ""}
      ${infoGrid(au.info)}
    </div>`;
  });

  host.innerHTML = html || `<p class="muted">No media.</p>`;
  $$(".media-block[data-img]", host).forEach((blk) => {
    $$(".tab", blk).forEach((t) => t.addEventListener("click", () => {
      $$(".tab", blk).forEach((x) => x.classList.remove("active"));
      t.classList.add("active");
      const img = $(".viewer img", blk);
      img.src = t.dataset.view; img.dataset.lb = t.dataset.view;
    }));
  });
  $$("[data-lb]", host).forEach((im) => im.addEventListener("click", () => lightbox(im.dataset.lb)));
  $$(".strip [data-seek]", host).forEach((fg) => fg.addEventListener("click", () => seekTo(+fg.dataset.seek, fg.dataset.file)));
  (m.video || []).forEach((vd) => drawTimeline(vd, r));
}

function drawTimeline(vd, r) {
  const cv = $(`canvas[data-tl="${vd.file_id}"]`);
  if (!cv) return;
  const dpr = window.devicePixelRatio || 1;
  const W = cv.clientWidth, H = cv.clientHeight;
  cv.width = W * dpr; cv.height = H * dpr;
  const ctx = cv.getContext("2d");
  ctx.scale(dpr, dpr);
  const tl = vd.timeline || { t: [] };
  const dur = vd.info.analysed_seconds || vd.info.duration_sec || (tl.t.at(-1) || 1);
  const x = (t) => (t / dur) * (W - 20) + 10;
  // suspicious segments & timestamps from findings of this file
  const fs = r.findings.filter((f) => f.file_id === vd.file_id && f.review_status !== "rejected");
  fs.forEach((f) => {
    const l = f.location || {};
    if (l.type === "segment" && l.end > l.start) {
      ctx.fillStyle = "rgba(239,68,68,.18)";
      ctx.fillRect(x(l.start), 0, Math.max(2, x(l.end) - x(l.start)), H);
    }
    (l.all && Array.isArray(l.all[0]) ? l.all : []).forEach(([a, b]) => { ctx.fillStyle = "rgba(239,68,68,.12)"; ctx.fillRect(x(a), 0, Math.max(2, x(b) - x(a)), H); });
  });
  const plot = (arr, color) => {
    const vals = arr.map((v) => (v === null ? NaN : v));
    const finite = vals.filter((v) => isFinite(v));
    if (!finite.length) return;
    const mx = Math.max(...finite) || 1;
    ctx.strokeStyle = color; ctx.lineWidth = 1.4; ctx.beginPath();
    let started = false;
    vals.forEach((v, i) => {
      if (!isFinite(v)) { started = false; return; }
      const px = x(tl.t[i]), py = H - 14 - (v / mx) * (H - 30);
      if (!started) { ctx.moveTo(px, py); started = true; } else ctx.lineTo(px, py);
    });
    ctx.stroke();
  };
  plot(tl.visual_change || [], "#5eb1ff");
  plot(tl.mouth_motion || [], "#ffb020");
  const marks = [];
  fs.forEach((f) => {
    const l = f.location || {};
    if (l.type === "timestamp") { marks.push(l.t); (l.all || []).forEach((t) => typeof t === "number" && marks.push(t)); }
    if (l.type === "segment" && l.all && typeof l.all[0] === "number") l.all.forEach((t) => marks.push(t));
  });
  ctx.fillStyle = "#ef4444";
  [...new Set(marks)].forEach((t) => { ctx.fillRect(x(t) - 1, 0, 2, H); ctx.beginPath(); ctx.arc(x(t), 7, 4, 0, 7); ctx.fill(); });
  ctx.fillStyle = "#8d98b3"; ctx.font = "10px JetBrains Mono, monospace";
  for (let i = 0; i <= 5; i++) { const t = (dur * i) / 5; ctx.fillText(fmtTs(t), Math.min(x(t), W - 44), H - 2); }
  cv.onclick = (e) => { const rect = cv.getBoundingClientRect(); seekTo(((e.clientX - rect.left - 10) / (W - 20)) * dur, vd.file_id); };
}

function renderText(r) {
  const ta = r.text_analysis || {};
  const ctx = ta.context || {};
  const ocr = ta.ocr || [];
  const trs = ta.transcripts || [];
  let html = `<h2>Text &amp; context</h2>`;
  html += `<dl class="kv">
    <dt>Claim</dt><dd>${r.claim ? `“${esc(r.claim)}”` : '<span class="muted">none supplied</span>'}</dd>
    <dt>Recency words in claim</dt><dd>${esc((ctx.claim_recency_words || []).join(", ") || "—")}</dd>
    <dt>Places in claim / media</dt><dd>${esc((ctx.claim_places || []).join(", ") || "—")} / ${esc((ctx.media_places || []).join(", ") || "—")}</dd>
    <dt>Dates found in media</dt><dd>${[...new Map((ctx.dates_in_media || []).map((d) => [d.text + d.source, d])).values()].map((d) => `<span class="mono">${esc(d.text)}</span> <span class="muted">(${esc(d.source)}${d.t !== undefined && d.t !== null ? " @ " + fmtTs(d.t) : ""})</span>`).join("<br>") || "—"}</dd>
    ${ctx.claim_transcript_similarity !== undefined ? `<dt>Claim ↔ transcript similarity</dt><dd>${ctx.claim_transcript_similarity}</dd>` : ""}
    ${ctx.claim_ocr_similarity !== undefined ? `<dt>Claim ↔ OCR similarity</dt><dd>${ctx.claim_ocr_similarity}</dd>` : ""}
  </dl>`;
  html += `<h4 style="margin-top:14px">Text visible in media (OCR)</h4>`;
  html += ocr.length ? `<div class="textbox">${ocr.map((o) => `${o.t !== undefined ? `[${fmtTs(o.t)}] ` : ""}${esc(o.text)}`).join("\n")}</div>`
    : `<p class="muted" style="font-size:.85rem">${r.capabilities?.ocr ? "No readable text found." : "OCR engine not installed on this server."}</p>`;
  html += `<h4 style="margin-top:14px">Transcript (speech-to-text)</h4>`;
  if (trs.length) {
    html += trs.map((t) => `<div class="textbox">${t.error ? esc("Transcription error: " + t.error) : (t.segments || []).map((s) => `[${fmtTs(s.start)}] ${esc(s.text)}`).join("\n") || "No speech detected."}</div>
      <p class="muted" style="font-size:.75rem">${esc(t.file || "")} · language: ${esc(t.language || "?")}</p>`).join("");
  } else {
    html += `<p class="muted" style="font-size:.85rem">${r.capabilities?.whisper === "available" ? "No speech to transcribe." : "Speech-to-text is not enabled (install faster-whisper — see README)."}</p>`;
  }
  $("#text-card").innerHTML = html;
}

function renderFiles(rec) {
  const r = rec.result;
  $("#files-card").innerHTML = `<h2>Evidence integrity</h2>
    ${r.files.map((f) => `<dl class="kv" style="margin-bottom:12px">
      <dt>File</dt><dd><b>${esc(f.filename)}</b> (${esc(f.media_type)}, ${(f.size_bytes / 1e6).toFixed(2)} MB)</dd>
      <dt>SHA-256</dt><dd class="mono">${esc(f.sha256)}</dd>
      <dt>MD5</dt><dd class="mono">${esc(f.md5)}</dd>
      <dt>Uploaded (UTC)</dt><dd>${esc(f.uploaded_at)}</dd></dl>`).join("")}
    <p class="muted" style="font-size:.78rem">Originals are stored read-only and never modified; all evidence images are derived copies.</p>`;
  const img = (r.media.image || [])[0];
  const exif = img?.metadata?.exif || {};
  const vtags = (r.media.video || [])[0]?.info?.container_tags || {};
  $("#meta-card").innerHTML = `<h2>Metadata, limitations &amp; audit trail</h2>
    ${Object.keys(exif).length ? `<details open><summary>EXIF metadata (${Object.keys(exif).length} fields)</summary><dl class="kv" style="margin-top:8px">${Object.entries(exif).map(([k, v]) => `<dt>${esc(k)}</dt><dd>${esc(v)}</dd>`).join("")}</dl></details>` : img ? `<p class="muted">No EXIF metadata in image.</p>` : ""}
    ${Object.keys(vtags).length ? `<details open><summary>Video container tags</summary><dl class="kv" style="margin-top:8px">${Object.entries(vtags).map(([k, v]) => `<dt>${esc(k)}</dt><dd>${esc(v)}</dd>`).join("")}</dl></details>` : ""}
    <details><summary>Limitations &amp; uncertainty</summary><ul class="plain">${r.limitations.map((l) => `<li>${esc(l)}</li>`).join("")}</ul></details>
    <details><summary>Tools / models used</summary><ul class="plain">${r.tools_used.map((l) => `<li>${esc(l)}</li>`).join("")}</ul></details>
    <details><summary>Analysis history (${rec.history.length} events)</summary><ol class="log">${rec.history.map((h) => `<li>${esc(h.ts.replace("T", " ").slice(0, 19))} — ${esc(h.event)}</li>`).join("")}</ol></details>`;
}

/* ---------------------------------------------------------------- correlation graph */
const NODE_COLOR = { image: "#5eb1ff", video: "#8b7bff", audio: "#ffb020", crossmodal: "#ff6b9a", metadata: "#94a3b8", text: "#4dd4ac", counter: "#22c55e" };

function renderGraph(r) {
  const g = r.correlation || { nodes: [], edges: [], hypotheses: [] };
  const card = $("#graph-card");
  const nSup = g.edges.filter((e) => e.type === "support").length, nCon = g.edges.length - nSup;
  card.innerHTML = `<div class="row-between"><h2>🕸 Evidence correlation graph</h2>
      <span class="muted" style="font-size:.8rem">${nSup} supporting · ${nCon} contradicting links · click a node to open its finding</span></div>
    ${g.nodes.length ? `<div class="graph-wrap"><div class="graph-box"><svg id="graph-svg"></svg>
      <div class="tl-legend"><span><i style="background:#16a34a"></i>supports</span><span><i style="background:#dc2626"></i>contradicts (dashed)</span>
      ${Object.entries(NODE_COLOR).map(([k, c]) => `<span><i style="background:${c};border-radius:50%"></i>${k === "counter" ? "counter-evidence / innocent cause" : CAT_LABEL[k]}</span>`).join("")}</div></div>
      <div class="hyps"><h4>Correlated hypotheses</h4>
      ${g.hypotheses.length ? g.hypotheses.map((h) => `<div class="hyp"><div class="row-between"><b>${esc(h.title)}</b><span class="mono">${Math.round(h.strength * 100)}%</span></div>
        <div class="bd-bar"><div style="width:${h.strength * 100}%;background:${h.strength > .7 ? "var(--high)" : h.strength > .45 ? "var(--medium)" : "var(--low)"}"></div></div>
        <div class="muted" style="font-size:.78rem;margin-top:4px">${esc(h.summary)}${h.when ? ` · ${esc(h.when)}` : ""}${h.contradicted ? ` · ${h.contradicted} contradiction(s)` : ""}</div></div>`).join("")
        : `<p class="muted" style="font-size:.85rem">No group of independent indicators agrees on one hypothesis. Isolated findings deserve extra caution.</p>`}
      <p class="muted" style="font-size:.74rem">A hypothesis appears only when two or more checks independently point the same way — agreement across modalities is much stronger than any single signal.</p></div></div>`
    : `<p class="muted">No indicators to correlate.</p>`}`;
  if (g.nodes.length) drawGraph(g);
}

function drawGraph(g) {
  const svg = $("#graph-svg");
  const W = svg.clientWidth || 700, H = 420;
  svg.setAttribute("viewBox", `0 0 ${W} ${H}`);
  const cats = [...new Set(g.nodes.map((n) => n.category))];
  const N = g.nodes.map((n, i) => {
    const a = (cats.indexOf(n.category) / cats.length) * Math.PI * 2;
    return { ...n, x: W / 2 + Math.cos(a) * W * 0.25 + (i % 5) * 7, y: H / 2 + Math.sin(a) * H * 0.28 + (i % 3) * 9, vx: 0, vy: 0 };
  });
  const idx = Object.fromEntries(N.map((n, i) => [n.id, i]));
  const E = g.edges.filter((e) => e.source in idx && e.target in idx);
  for (let it = 0; it < 400; it++) {                  // small force-directed layout
    for (let i = 0; i < N.length; i++) for (let j = i + 1; j < N.length; j++) {
      const a = N[i], b = N[j]; let dx = a.x - b.x, dy = a.y - b.y; const d2 = dx * dx + dy * dy + 0.01, f = 6000 / d2;
      const d = Math.sqrt(d2); dx /= d; dy /= d; a.vx += dx * f; a.vy += dy * f; b.vx -= dx * f; b.vy -= dy * f;
    }
    E.forEach((e) => { const a = N[idx[e.source]], b = N[idx[e.target]]; const dx = b.x - a.x, dy = b.y - a.y, d = Math.hypot(dx, dy) || 1, f = (d - 150) * 0.02;
      a.vx += dx / d * f; a.vy += dy / d * f; b.vx -= dx / d * f; b.vy -= dy / d * f; });
    N.forEach((n) => { n.vx += (W / 2 - n.x) * 0.004; n.vy += (H / 2 - n.y) * 0.006; n.x += n.vx * 0.5; n.y += n.vy * 0.5; n.vx *= 0.6; n.vy *= 0.6;
      n.x = Math.max(60, Math.min(W - 60, n.x)); n.y = Math.max(30, Math.min(H - 42, n.y)); });
  }
  const r = (n) => 9 + n.strength * 14;
  let html = E.map((e) => { const a = N[idx[e.source]], b = N[idx[e.target]], sup = e.type === "support";
    return `<line x1="${a.x}" y1="${a.y}" x2="${b.x}" y2="${b.y}" stroke="${sup ? "#16a34a" : "#dc2626"}" stroke-width="${sup ? 2.4 : 1.8}" ${sup ? "" : 'stroke-dasharray="6 4"'} opacity=".8"><title>${esc((sup ? "SUPPORTS: " : "CONTRADICTS: ") + e.reason)}</title></line>`; }).join("");
  html += N.map((n) => { const lab = n.label.length > 28 ? n.label.slice(0, 26) + "…" : n.label; const counter = n.category === "counter";
    return `<g class="gnode" data-id="${n.id}" style="cursor:${counter ? "default" : "pointer"}">
      <circle cx="${n.x}" cy="${n.y}" r="${r(n)}" fill="${NODE_COLOR[n.category] || "#888"}" fill-opacity="${counter ? .25 : .9}" stroke="${counter ? "#22c55e" : "#0b0f19"}" stroke-width="2"><title>${esc(n.label)}</title></circle>
      <text x="${n.x}" y="${n.y + r(n) + 12}" text-anchor="middle" font-size="10.5" fill="#dbe3f0" stroke="#111726" stroke-width="3.5" paint-order="stroke">${esc(lab)}</text></g>`; }).join("");
  svg.innerHTML = html;
  $$(".gnode", svg).forEach((gn) => gn.addEventListener("click", () => {
    const el = $(`#f-${gn.dataset.id}`); if (!el) return;
    findingFilter = "all"; el.scrollIntoView({ behavior: "smooth", block: "center" }); el.classList.add("flash"); setTimeout(() => el.classList.remove("flash"), 1600);
  }));
}

/* ---------------------------------------------------------------- integrity seal */
function renderSeal(r) {
  const s = r.seal;
  const card = $("#seal-card");
  if (!s) { card.innerHTML = `<h2>🔏 Integrity seal</h2><p class="muted">Not sealed.</p>`; return; }
  const vurl = `/verify/${r.analysis_id}?v=${s.version}&d=${s.report_sha256.slice(0, 16)}&s=${s.signature.slice(0, 16)}`;
  card.innerHTML = `<h2>🔏 Tamper-proof seal</h2>
    <div class="seal">
      <img src="/api/analyses/${r.analysis_id}/qr.png?v=${s.version}" alt="verification QR" class="qr" />
      <div><div class="seal-ok">✔ Sealed — version ${s.version}</div>
        <dl class="kv" style="grid-template-columns:110px 1fr">
          <dt>Report SHA-256</dt><dd class="mono">${esc(s.report_sha256)}</dd>
          <dt>Audit chain head</dt><dd class="mono">${esc(s.chain_head.slice(0, 32))}…</dd>
          <dt>Signature</dt><dd class="mono">${esc(s.signature.slice(0, 32))}…</dd>
          <dt>Sealed (UTC)</dt><dd>${esc(s.sealed_at)} · ${esc(s.reason)}</dd></dl></div></div>
    <p class="muted" style="font-size:.78rem">Scan the QR (also printed on the PDF) to verify live that this report, its audit trail and the original evidence have not been altered. Every investigator review creates a new seal version.</p>
    <div class="actions"><a class="btn sm" href="${vurl}" target="_blank">🔍 Open verification page</a>
      <a class="btn sm" href="/verify/${r.analysis_id}" target="_blank">Verify a file / JSON report</a></div>`;
}

/* ---------------------------------------------------------------- history */
async function showHistory() {
  app.innerHTML = $("#tpl-history").innerHTML;
  const rows = await api("/api/analyses");
  if (!rows.length) { $("#hist-empty").classList.remove("hidden"); $("#hist-table").classList.add("hidden"); return; }
  $("#hist-table tbody").innerHTML = rows.map((a) => `
    <tr class="click" data-id="${a.id}">
      <td class="mono">${esc(a.created_at.replace("T", " ").slice(0, 19))}</td>
      <td>${a.files.map((f) => `${esc(f.filename)} <span class="muted">(${f.media_type})</span>`).join("<br>")}</td>
      <td style="max-width:280px">${esc((a.claim || "—").slice(0, 120))}</td>
      <td>${a.risk_score !== null ? `<span class="badge ${a.risk_level}">${a.risk_score}</span>` : "—"}</td>
      <td>${esc(a.status)}</td>
      <td><button class="btn sm danger" data-del="${a.id}">Delete</button></td>
    </tr>`).join("");
  $$("#hist-table tr.click").forEach((tr) => tr.addEventListener("click", (e) => {
    if (e.target.dataset.del) return;
    location.hash = `#/analysis/${tr.dataset.id}`;
  }));
  $$("[data-del]").forEach((b) => b.addEventListener("click", async (e) => {
    e.stopPropagation();
    if (!confirm("Delete this analysis and its evidence?")) return;
    try { await api(`/api/analyses/${b.dataset.del}`, { method: "DELETE" }); showHistory(); toast("Deleted"); } catch (err) { toast(err.message); }
  }));
}

window.addEventListener("resize", () => { if (current && location.hash.startsWith("#/analysis/")) (current.result.media.video || []).forEach((vd) => drawTimeline(vd, current.result)); });
loadCaps();
route();
