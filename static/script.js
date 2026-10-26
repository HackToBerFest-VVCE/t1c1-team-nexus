/* ===================================================================
   AccidentGuard — dashboard logic
   Calls the Flask backend, drives the demo loop, renders the rule-engine
   classification separately from the Gemma 4 summary.
   =================================================================== */

(function () {
  "use strict";

  // ---------- Element helpers ----------
  const $ = (id) => document.getElementById(id);
  const els = {
    aiBadge: $("aiBadge"), aiStateText: $("aiStateText"), aiModel: $("aiModel"),
    btnStart: $("btnStartDemo"), btnStop: $("btnStopDemo"),
    btnGenerate: $("btnGenerate"), btnReset: $("btnReset"),
    categorySelect: $("categorySelect"), demoHint: $("demoHint"),
    statTotal: $("statTotal"), statNormal: $("statNormal"),
    statSuspicious: $("statSuspicious"), statAccident: $("statAccident"),
    currentEvent: $("currentEvent"), aiPanel: $("aiPanel"), aiSrcTag: $("aiSrcTag"),
    historyList: $("historyList"), historyCount: $("historyCount"), toast: $("toast"),
  };

  let demoTimer = null;
  let demoIndex = 0;

  // ---------- API helpers ----------
  async function api(path, opts = {}) {
    const res = await fetch(path, {
      headers: { "Content-Type": "application/json" },
      ...opts,
    });
    if (!res.ok) {
      throw new Error(`Request failed (${res.status})`);
    }
    return res.json();
  }

  // ---------- Toast ----------
  let toastTimer = null;
  function toast(msg, kind = "") {
    if (!els.toast) return;
    els.toast.textContent = msg;
    els.toast.className = "toast " + kind;
    els.toast.hidden = false;
    clearTimeout(toastTimer);
    toastTimer = setTimeout(() => { els.toast.hidden = true; }, 3500);
  }

  // ---------- Rendering: AI status ----------
  function renderAiStatus(ai, model) {
    const state = ai.state; // connected | error | not_configured | pending
    els.aiBadge.dataset.state = state;
    const labels = {
      connected: "Connected",
      error: "Error",
      not_configured: "Not configured",
      pending: "Working…",
    };
    els.aiStateText.textContent = labels[state] || state;
    if (model && model.model) els.aiModel.textContent = "model: " + model.model;
  }

  async function refreshAiStatus() {
    try {
      const d = await api("/api/ai/status");
      renderAiStatus(d.ai, d.model);
      return d;
    } catch (e) {
      els.aiBadge.dataset.state = "error";
      els.aiStateText.textContent = "Error";
    }
  }

  // ---------- Rendering: current event (rule engine) ----------
  function fmtReadings(r) {
    return `
      <div class="ev-grid">
        <div class="ev-field"><span class="k">Speed</span><span class="v">${r.speed_kmh} km/h</span></div>
        <div class="ev-field"><span class="k">Speed change</span><span class="v">${r.speed_delta_kmh} km/h</span></div>
        <div class="ev-field"><span class="k">Deceleration</span><span class="v">${r.deceleration_g} g</span></div>
        <div class="ev-field"><span class="k">Impact score</span><span class="v">${r.impact_score}</span></div>
        <div class="ev-field"><span class="k">Lane deviation</span><span class="v">${r.lane_deviation_m} m</span></div>
        <div class="ev-field"><span class="k">Location</span><span class="v" style="font-size:.82rem">${escapeHtml(r.location_label)}</span></div>
      </div>`;
  }

  function renderCurrentEvent(ev) {
    if (!ev) return;
    els.currentEvent.classList.remove("event-empty");
    els.currentEvent.innerHTML = `
      <div style="display:flex;gap:10px;flex-wrap:wrap;align-items:center;margin-bottom:12px">
        <span class="cat-pill cat-${ev.category}">${prettyCat(ev.category)}</span>
        <span class="sev sev-${ev.severity}">${ev.severity}</span>
        <span class="muted" style="font-family:var(--mono);font-size:.8rem">${escapeHtml(ev.event_id)}</span>
      </div>
      <div class="ev-field" style="margin-bottom:10px"><span class="k">Timestamp</span>
        <span class="v">${formatTime(ev.timestamp)}</span></div>
      ${fmtReadings(ev.readings)}
      <div class="rule-note"><b>Rule engine verdict:</b> ${escapeHtml(ev.rule_explanation)}
        <br><span class="muted" style="font-size:.78rem">This is transparent rule-based logic, separate from the AI summary.</span></div>`;
  }

  // ---------- Rendering: AI panel ----------
  function renderAiLoading() {
    els.aiPanel.classList.remove("ai-empty");
    els.aiSrcTag.textContent = "AI";
    els.aiSrcTag.className = "src-tag";
    els.aiPanel.innerHTML = `<p><span class="spinner"></span>Asking Gemma 4 to summarise this event…</p>`;
  }

  function renderAiResult(ai) {
    if (!ai) {
      els.aiPanel.innerHTML = `<p class="placeholder">No AI summary for this event yet.</p>`;
      return;
    }
    const isGemma = ai.source === "gemma";
    els.aiSrcTag.textContent = isGemma ? "Gemma 4" : "Non-AI fallback";
    els.aiSrcTag.className = "src-tag " + (isGemma ? "gemma" : "fallback");
    els.aiPanel.classList.remove("ai-empty");

    const recs = (ai.recommendations || [])
      .map((r) => `<li>${escapeHtml(r)}</li>`).join("");
    const attn = (ai.attention_points || [])
      .map((a) => `<li>${escapeHtml(a)}</li>`).join("") || "<li>none</li>";
    const sev = ai.severity_opinion
      ? `<span class="sev sev-${ai.severity_opinion}">${ai.severity_opinion}</span>` : "";

    els.aiPanel.innerHTML = `
      <p class="ai-summary">${escapeHtml(ai.summary || "")}</p>
      <div class="ai-block">
        <h4>Attention</h4>
        <ul>${attn}</ul>
      </div>
      <div class="ai-block">
        <h4>Recommendations</h4>
        <ul>${recs || "<li>none</li>"}</ul>
      </div>
      <div class="ai-block">
        <h4>Gemma severity opinion</h4>
        ${sev || '<span class="muted">not provided</span>'}
      </div>
      ${ai.error ? `<p class="ai-error">⚠ ${escapeHtml(ai.error)}</p>` : ""}
      <p class="ai-meta">source: ${escapeHtml(ai.source)} · model: ${escapeHtml(ai.model || "n/a")}${
        ai.used_fallback ? " · fallback reason: " + escapeHtml(ai.fallback_reason || "") : ""
      }</p>`;
  }

  // ---------- Rendering: history + stats ----------
  function renderHistory(events) {
    els.historyCount.textContent = `${events.length} event${events.length === 1 ? "" : "s"}`;
    if (!events.length) {
      els.historyList.innerHTML = `<li class="placeholder">No incidents recorded yet.</li>`;
      return;
    }
    els.historyList.innerHTML = events.map((e) => {
      const c = e.classification;
      const ai = e.ai_result;
      const aiTag = ai
        ? (ai.source === "gemma"
            ? `<span class="hi-ai gemma">Gemma ✓</span>`
            : `<span class="hi-ai fallback">fallback</span>`)
        : `<span class="hi-ai none">no AI</span>`;
      return `
        <li class="history-item">
          <span class="hi-time">${formatTime(c.timestamp)}</span>
          <span class="cat-pill cat-${c.category}">${prettyCat(c.category)}</span>
          <span class="sev sev-${c.severity}">${c.severity}</span>
          <span class="hi-main"><span class="hi-loc">${escapeHtml(c.readings.location_label)}</span></span>
          ${aiTag}
        </li>`;
    }).join("");
  }

  function renderStats(s) {
    els.statTotal.textContent = s.total ?? 0;
    els.statNormal.textContent = (s.by_category || {}).normal_driving || 0;
    els.statSuspicious.textContent = (s.by_category || {}).suspicious_event || 0;
    els.statAccident.textContent = (s.by_category || {}).possible_accident || 0;
  }

  // ---------- Actions ----------
  async function loadHistoryAndStats() {
    try {
      const [h, s] = await Promise.all([api("/api/history"), api("/api/stats")]);
      renderHistory(h.events || []);
      renderStats(s);
    } catch (e) { /* ignore on first load */ }
  }

  async function generateEvent() {
    const category = els.categorySelect.value;
    const body = category === "random" ? {} : { category };
    try {
      els.btnGenerate.disabled = true;
      const d = await api("/api/event/generate", { method: "POST", body: JSON.stringify(body) });
      renderCurrentEvent(d.event);
      renderAiLoading();
      // Ask Gemma to summarise this event (works without a key -> fallback).
      const s = await api(`/api/event/${encodeURIComponent(d.event.event_id)}/summarize`,
        { method: "POST", body: "{}" });
      renderAiResult(s.ai);
      renderAiStatus(s.ai_status, null);
      await loadHistoryAndStats();
    } catch (e) {
      toast("Could not generate event: " + e.message, "error");
    } finally {
      els.btnGenerate.disabled = false;
    }
  }

  function startDemo() {
    els.btnStart.disabled = true;
    els.btnStop.disabled = false;
    els.demoHint.textContent = "Demo running — generating an event every 4s…";
    demoIndex = 0;
    // Run one immediately, then on an interval.
    runDemoStep();
    demoTimer = setInterval(runDemoStep, 4000);
  }

  function stopDemo() {
    if (demoTimer) { clearInterval(demoTimer); demoTimer = null; }
    els.btnStart.disabled = false;
    els.btnStop.disabled = true;
    els.demoHint.textContent = "Tip: Start Demo auto-generates events and asks Gemma 4 to summarise each one.";
  }

  async function runDemoStep() {
    // Cycle through categories to showcase all three for judges.
    const cycle = ["normal_driving", "suspicious_event", "possible_accident"];
    const category = cycle[demoIndex % cycle.length];
    demoIndex++;
    try {
      const d = await api("/api/event/generate",
        { method: "POST", body: JSON.stringify({ category }) });
      renderCurrentEvent(d.event);
      renderAiLoading();
      const s = await api(`/api/event/${encodeURIComponent(d.event.event_id)}/summarize`,
        { method: "POST", body: "{}" });
      renderAiResult(s.ai);
      renderAiStatus(s.ai_status, null);
      await loadHistoryAndStats();
    } catch (e) {
      stopDemo();
      toast("Demo stopped: " + e.message, "error");
    }
  }

  async function resetAll() {
    try {
      await api("/api/reset", { method: "POST", body: "{}" });
      await loadHistoryAndStats();
      els.currentEvent.classList.add("event-empty");
      els.currentEvent.innerHTML = `<p class="placeholder">Cleared. Generate a new event to begin.</p>`;
      els.aiPanel.classList.add("ai-empty");
      els.aiSrcTag.textContent = "AI";
      els.aiSrcTag.className = "src-tag";
      els.aiPanel.innerHTML = `<p class="placeholder">The Gemma 4 summary will appear here after an event is analysed.</p>`;
      toast("History cleared", "ok");
    } catch (e) {
      toast("Could not reset: " + e.message, "error");
    }
  }

  // ---------- Utilities ----------
  function prettyCat(c) {
    return ({
      normal_driving: "Normal driving",
      suspicious_event: "Suspicious event",
      possible_accident: "Possible accident",
    })[c] || c;
  }
  function formatTime(iso) {
    try {
      const d = new Date(iso);
      return d.toLocaleString();
    } catch { return iso; }
  }
  function escapeHtml(s) {
    return String(s == null ? "" : s)
      .replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;").replace(/'/g, "&#39;");
  }

  // ---------- Wire up ----------
  els.btnStart.addEventListener("click", startDemo);
  els.btnStop.addEventListener("click", stopDemo);
  els.btnGenerate.addEventListener("click", generateEvent);
  els.btnReset.addEventListener("click", resetAll);

  // Initial load.
  refreshAiStatus();
  loadHistoryAndStats();
})();
