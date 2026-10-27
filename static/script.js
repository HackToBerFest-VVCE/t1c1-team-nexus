/* AccidentGuard — main dashboard logic */
(function () {
  "use strict";
  const $ = id => document.getElementById(id);
  const els = {
    aiBadge:$("aiBadge"), aiStateText:$("aiStateText"), aiModel:$("aiModel"),
    btnStart:$("btnStartDemo"), btnStop:$("btnStopDemo"),
    btnGenerate:$("btnGenerate"), btnReset:$("btnReset"), btnAiCheck:$("btnAiCheck"),
    scenarioSelect:$("scenarioSelect"), demoHint:$("demoHint"),
    toggleAi:$("toggleAi"), toggleAutoAccept:$("toggleAutoAccept"), toggleVictimAuto:$("toggleVictimAuto"),
    statTotal:$("statTotal"), statNormal:$("statNormal"), statSuspicious:$("statSuspicious"),
    statAccident:$("statAccident"), statRejected:$("statRejected"),
    phonePanel:$("phonePanel"), vehiclePanel:$("vehiclePanel"),
    currentEvent:$("currentEvent"), aiPanel:$("aiPanel"), aiSrcTag:$("aiSrcTag"),
    incidentPanel:$("incidentPanel"), victimCheckPanel:$("victimCheckPanel"),
    vmCountdown:$("vmCountdown"), vmBtnSafe:$("vmBtnSafe"), vmBtnHelp:$("vmBtnHelp"),
    incidentList:$("incidentList"), activeCount:$("activeCount"),
    historyList:$("historyList"), historyCount:$("historyCount"),
    toast:$("toast"),
    photoUpload:$("photoUpload"), photoInput:$("photoInput"), photoResult:$("photoResult"),
    pSensor:$("pSensor"), pValidation:$("pValidation"), pFusion:$("pFusion"),
    pFilter:$("pFilter"), pGemma:$("pGemma"), pSafety:$("pSafety"), pDispatch:$("pDispatch"),
    steps: document.querySelectorAll(".pipe-step"),
  };

  let demoTimer=null, demoIndex=0, pollTimer=null;
  let currentIncidentId=null, vmTimer=null, vmSecsLeft=0;

  // ---------- Utilities ----------
  const esc = s => String(s==null?"":s).replace(/&/g,"&amp;").replace(/</g,"&lt;").replace(/>/g,"&gt;").replace(/"/g,"&quot;");
  const prettyCat = c => ({normal_driving:"Normal driving",suspicious_event:"Suspicious event",possible_accident:"Possible accident"})[c]||c;
  const sevRank = s => ({low:1,medium:2,high:3,critical:4})[s]||0;
  const fmtTime = iso => { try { return new Date(iso).toLocaleString(); } catch { return iso; } };

  async function api(path, opts={}) {
    const r = await fetch(path, { headers:{"Content-Type":"application/json"}, ...opts });
    if (!r.ok) throw new Error(`Request failed (${r.status})`);
    return r.json();
  }
  let toastTimer=null;
  function toast(msg,kind="") {
    if(!els.toast) return;
    els.toast.textContent=msg; els.toast.className="toast "+kind; els.toast.hidden=false;
    clearTimeout(toastTimer); toastTimer=setTimeout(()=>{els.toast.hidden=true;},3500);
  }

  // ---------- AI status ----------
  function renderAiStatus(ai, model) {
    if(!ai) return;
    els.aiBadge.dataset.state = ai.state;
    let lbl = ({connected:"Connected",error:"Error",not_configured:"Not configured",pending:"Working…"})[ai.state]||ai.state;
    if(ai.state==="error" && ai.reason) lbl += " — " + ai.reason;
    if(ai.state==="not_configured") lbl = "Not configured";
    els.aiStateText.textContent = lbl;
    if(model && model.model) els.aiModel.textContent = "model: " + model.model;
  }
  async function refreshAiStatus() {
    try {
      const d = await api("/api/ai/status");
      renderAiStatus(d.ai, d.model);
      if(d.toggles) {
        els.toggleAi.checked = !!d.toggles.ai_enabled;
        els.toggleAutoAccept.checked = !!d.toggles.auto_accept;
        els.toggleVictimAuto.checked = !!d.toggles.victim_auto_respond;
      }
    } catch {}
  }

  // ---------- Pipeline strip ----------
  function setPipeline(state) {
    // state: {sensor, validation, fusion, filter, gemma, safety, dispatch}
    // each: "waiting"|"done"|"skipped"|"failed"
    const icons = {done:"✓",skipped:"⊘",failed:"✗",waiting:"…"};
    const details = {
      pSensor: els.pSensor, pValidation: els.pValidation, pFusion: els.pFusion,
      pFilter: els.pFilter, pGemma: els.pGemma, pSafety: els.pSafety, pDispatch: els.pDispatch,
    };
    const stepMap = {sensor:"sensor",validation:"validation",fusion:"fusion",filter:"filter",gemma:"gemma",safety:"safety",dispatch:"dispatch"};
    if(!state) {
      els.steps.forEach(s=>{s.classList.remove("lit","fallback","failed","skipped");});
      Object.values(details).forEach(d=>{if(d)d.textContent="—";});
      return;
    }
    els.steps.forEach(s=>{
      const k = s.dataset.step;
      const st = state[k] || "waiting";
      s.classList.remove("lit","fallback","failed","skipped");
      if(st==="done") s.classList.add("lit");
      else if(st==="failed") s.classList.add("failed");
      else if(st==="skipped") s.classList.add("skipped");
      const det = state[k+"_detail"] || "";
      const el = details["p"+k.charAt(0).toUpperCase()+k.slice(1)];
      if(el) el.textContent = det || (icons[st]||"");
    });
    if(state.gemma_fallback) {
      document.querySelector('[data-step="gemma"]')?.classList.add("fallback");
    }
  }

  // ---------- Sensors ----------
  function renderPhonePanel(p) {
    if(!p || !Object.keys(p).length) { els.phonePanel.innerHTML=`<p class="placeholder">No data.</p>`; return; }
    els.phonePanel.innerHTML = `
      <div class="ev-grid">
        <div class="ev-field"><span class="k">Accelerometer peak</span><span class="v">${esc(p.accel_peak_g)} g</span></div>
        <div class="ev-field"><span class="k">Gyroscope</span><span class="v">${esc(p.gyro_deg_s)} deg/s</span></div>
        <div class="ev-field"><span class="k">GPS speed change</span><span class="v">${esc(p.gps_speed_change_kmh)} km/h</span></div>
        <div class="ev-field"><span class="k">GPS location</span><span class="v" style="font-size:.78rem">${esc(p.location_label)}</span></div>
      </div>`;
  }
  function renderVehiclePanel(v) {
    if(!v || !Object.keys(v).length) { els.vehiclePanel.innerHTML=`<p class="placeholder">No data.</p>`; return; }
    const crashCls = v.crash_flag ? "sev-critical" : "sev-low";
    const moveCls = !v.post_impact_movement ? "sev-critical" : "sev-low";
    els.vehiclePanel.innerHTML = `
      <div class="ev-grid">
        <div class="ev-field"><span class="k">Impact score (0-1)</span><span class="v">${esc(v.impact_score)}</span></div>
        <div class="ev-field"><span class="k">Crash / airbag flag</span><span class="v"><span class="sev ${crashCls}">${v.crash_flag?"TRIGGERED":"clear"}</span></span></div>
        <div class="ev-field"><span class="k">Post-impact movement</span><span class="v"><span class="sev ${moveCls}">${v.post_impact_movement?"moving":"STOPPED"}</span></span></div>
        <div class="ev-field"><span class="k">Speed change</span><span class="v">${esc(v.speed_change_kmh)} km/h</span></div>
      </div>`;
  }

  // ---------- Rule engine / current event ----------
  function renderCurrentEvent(ev, fusion) {
    if(!ev) return;
    els.currentEvent.classList.remove("event-empty");
    const sigs = fusion ? fusion.signals_detail||[] : [];
    const sigChips = sigs.map(s=>`<span class="chip chip-sig">${esc(s)}</span>`).join(" ");
    const total = fusion ? fusion.agreeing_signals||0 : 0;
    els.currentEvent.innerHTML = `
      <div style="display:flex;gap:8px;flex-wrap:wrap;align-items:center;margin-bottom:10px">
        <span class="cat-pill cat-${esc(ev.category)}">${prettyCat(ev.category)}</span>
        <span class="sev sev-${esc(ev.severity)}">${esc(ev.severity)}</span>
        <span class="muted" style="font-family:var(--mono);font-size:.8rem">${esc(ev.event_id)}</span>
      </div>
      <div class="ev-field" style="margin-bottom:8px"><span class="k">Timestamp</span><span class="v">${fmtTime(ev.timestamp)}</span></div>
      ${fusion ? `<div class="fusion-bar">Signals agreeing: <b>${total} of 6</b> ${sigChips}</div>` : ""}
      <div class="rule-note"><b>Rule engine:</b> ${esc(ev.rule_explanation)}<br>
        <span class="muted" style="font-size:.78rem">Transparent rule-based logic, separate from the AI summary.</span></div>`;
  }

  // ---------- AI panel ----------
  function renderAiLoading() {
    els.aiPanel.classList.remove("ai-empty");
    els.aiSrcTag.textContent="AI"; els.aiSrcTag.className="src-tag";
    els.aiPanel.innerHTML=`<p><span class="spinner"></span>Asking Gemma 4 to assess this event…</p>`;
  }
  function renderAiResult(ai, scene) {
    if(!ai) { els.aiPanel.innerHTML=`<p class="placeholder">No AI summary yet.</p>`; return; }
    const isGemma = ai.source==="gemma";
    els.aiSrcTag.textContent = isGemma ? "Gemma 4" : "Non-AI fallback";
    els.aiSrcTag.className = "src-tag " + (isGemma ? "gemma" : "fallback");
    els.aiPanel.classList.remove("ai-empty");
    const ev = (ai.evidence||[]).map(e=>`<li>${esc(e)}</li>`).join("")||"<li>none</li>";
    const recs = (ai.recommendations||ai.responder_briefing||[]).map(r=>`<li>${esc(r)}</li>`).join("")||"<li>—</li>";
    const sev = ai.severity_opinion ? `<span class="sev sev-${esc(ai.severity_opinion)}">${esc(ai.severity_opinion)}</span>` : "";
    const conf = (typeof ai.confidence==="number")
      ? `<div class="conf-bar"><span style="width:${Math.round(ai.confidence*100)}%"></span></div><span class="muted">${Math.round(ai.confidence*100)}% confidence</span>` : "";
    let sceneHtml = "";
    if(scene) {
      if(!scene.available) {
        sceneHtml = `<div class="ai-block"><h4>Scene analysis</h4><p class="muted">${esc(scene.notes||"Scene analysis unavailable.")}</p></div>`;
      } else {
        sceneHtml = `<div class="ai-block"><h4>Scene analysis (Gemma 4 photo)</h4>
          <p>Vehicles: ${esc(scene.vehicles)} · Injuries: ${esc(scene.visible_injuries)} · Fire/fuel: ${esc(scene.fire_or_fuel_risk)} · Road blocked: ${esc(scene.road_blocked)}</p>
          <p class="muted">${esc(scene.notes||"")}</p></div>`;
      }
    }
    els.aiPanel.innerHTML = `
      <p class="ai-summary"><b>Action:</b> ${esc(ai.action||ai.summary||"")}</p>
      <div class="ai-block"><h4>Evidence</h4><ul>${ev}</ul></div>
      <div class="ai-block"><h4>Uncertainty</h4><p>${esc(ai.uncertainty||"—")}</p></div>
      ${conf ? `<div class="ai-block"><h4>Confidence</h4>${conf}</div>` : ""}
      <div class="ai-block"><h4>Recommendations</h4><ul>${recs}</ul></div>
      <div class="ai-block"><h4>Gemma severity opinion</h4>${sev||'<span class="muted">not provided</span>'}</div>
      ${sceneHtml}
      ${ai.error ? `<p class="ai-error">⚠ ${esc(ai.error)}</p>` : ""}
      <p class="ai-meta">source: ${esc(ai.source)} · model: ${esc(ai.model||"n/a")}</p>`;
    // show photo upload when incident is active
    els.photoUpload.hidden = false;
  }

  // ---------- Safety & incident panel ----------
  function renderIncident(inc) {
    if(!inc) { els.incidentPanel.innerHTML=`<p class="placeholder">No incident yet.</p>`; return; }
    const s=inc.safety||{}, d=inc.dispatch||{}, fr=inc.filter_result||{};
    const disTag = s.disagreement
      ? `<span class="chip chip-warn">⚠ disagreement handled</span>`
      : `<span class="chip chip-ok">agreed</span>`;
    let disp = "—";
    if(inc.status==="awaiting_victim") disp="Victim check running";
    else if(inc.status==="alerted"||inc.status==="escalated") disp = d.offered_to ? `Offered: ${esc(d.offered_to)}` : "alerting…";
    else if(d.accepted_unit) disp = `${esc(d.accepted_unit)} · ETA ${d.eta_min} min`;
    else if(d.control_room) disp = "Escalated to control room (simulated)";
    if(!fr.passed && fr.filter_label) {
      els.incidentPanel.innerHTML = `
        <div class="reject-banner"><b>False alarm rejected:</b> ${esc(fr.reason||fr.filter_label)}</div>
        <p class="muted">Event still recorded in history with "Rejected" tag.</p>`;
      els.victimCheckPanel.hidden=true;
      return;
    }
    els.incidentPanel.innerHTML = `
      <div style="display:flex;gap:8px;flex-wrap:wrap;align-items:center;margin-bottom:10px">
        <span class="sev sev-${esc(inc.severity)}">${esc(inc.severity)}</span>
        <span class="cat-pill cat-${esc(inc.classification.category)}">${prettyCat(inc.classification.category)}</span>
        <span class="chip">${esc(inc.status)}</span>${disTag}
      </div>
      <p><b>Safety layer (Python):</b> ${esc(s.note||"—")}</p>
      <p><b>Dispatch:</b> ${disp}</p>
      <p class="muted" style="font-size:.78rem">${esc(inc.id)} · ${esc(inc.location_label)}</p>`;
    // show victim check mirror if awaiting
    if(inc.status==="awaiting_victim") {
      els.victimCheckPanel.hidden=false;
      startVmCountdown(inc);
    } else {
      els.victimCheckPanel.hidden=true;
      stopVmCountdown();
    }
  }

  // ---------- Victim mirror countdown ----------
  function startVmCountdown(inc) {
    stopVmCountdown();
    vmSecsLeft = inc.victim?.deadline ? Math.max(0, Math.round(inc.victim.deadline - Date.now()/1000)) : 10;
    updateVmDisplay();
    vmTimer = setInterval(()=>{
      vmSecsLeft = Math.max(0, vmSecsLeft-1);
      updateVmDisplay();
      if(vmSecsLeft<=0) stopVmCountdown();
    }, 1000);
  }
  function stopVmCountdown() { if(vmTimer){ clearInterval(vmTimer); vmTimer=null; } }
  function updateVmDisplay() { if(els.vmCountdown) els.vmCountdown.textContent = `${vmSecsLeft}s`; }

  async function vmRespond(answer) {
    if(!currentIncidentId) return;
    try {
      await api(`/api/incidents/${encodeURIComponent(currentIncidentId)}/respond`,
        { method:"POST", body:JSON.stringify({answer}) });
      await poll();
    } catch(e) { toast("Victim respond failed: "+e.message, "error"); }
  }

  // ---------- Photo upload ----------
  async function handlePhoto(incidentId) {
    const f = els.photoInput?.files?.[0];
    if(!f) return;
    const reader = new FileReader();
    reader.onload = async ev => {
      const b64 = ev.target.result.split(",")[1];
      els.photoResult.hidden=false;
      els.photoResult.innerHTML=`<span class="spinner"></span> Analysing photo with Gemma 4…`;
      try {
        const d = await api(`/api/incidents/${encodeURIComponent(incidentId)}/photo`, {
          method:"POST", body:JSON.stringify({ image_data:b64, mime_type:f.type||"image/jpeg" }),
        });
        const sc = d.scene_analysis;
        if(sc && sc.available) {
          els.photoResult.innerHTML=`<b>Scene analysis:</b> ${esc(sc.notes)} Vehicles: ${sc.vehicles}, Injuries: ${esc(sc.visible_injuries)}, Fire: ${esc(sc.fire_or_fuel_risk)}, Road blocked: ${esc(sc.road_blocked)}.`;
        } else {
          els.photoResult.innerHTML=`<span class="muted">${esc(sc?.notes||"Scene analysis unavailable.")}</span>`;
        }
        // re-render AI panel with scene data
        if(d.incident) renderAiResult(d.incident.ai_result, sc);
      } catch(e) {
        els.photoResult.innerHTML=`<span class="ai-error">Photo analysis failed: ${esc(e.message)}</span>`;
      }
    };
    reader.readAsDataURL(f);
  }

  // ---------- Active incidents ----------
  function renderActiveIncidents(incidents) {
    const active = (incidents||[]).filter(i=>["new","awaiting_victim","alerted","assigned","en_route","arrived","escalated"].includes(i.status));
    els.activeCount.textContent=active.length;
    if(!active.length) { els.incidentList.innerHTML=`<li class="placeholder">No active incidents.</li>`; return; }
    active.sort((a,b)=>sevRank(b.severity)-sevRank(a.severity));
    els.incidentList.innerHTML=active.map(i=>{
      const d=i.dispatch||{};
      const line = d.accepted_unit ? `${esc(d.accepted_unit)} · ETA ${d.eta_min}m` : d.offered_to ? `offered: ${esc(d.offered_to)}` : i.status;
      return `<li class="history-item"><span class="sev sev-${esc(i.severity)}">${esc(i.severity)}</span>
        <span class="cat-pill cat-${esc(i.classification.category)}">${prettyCat(i.classification.category)}</span>
        <span class="hi-main">${esc(i.location_label)}<br><span class="muted" style="font-size:.74rem">${line}</span></span>
        <span class="chip">${esc(i.status)}</span></li>`;
    }).join("");
  }

  // ---------- History ----------
  function renderHistory(events) {
    els.historyCount.textContent=`${events.length} event${events.length===1?"":"s"}`;
    if(!events.length) { els.historyList.innerHTML=`<li class="placeholder">No incidents recorded yet.</li>`; return; }
    els.historyList.innerHTML=events.map(e=>{
      const c=e.classification, ai=e.ai_result, inc=e.incident||{}, fr=e.filter_result||{};
      const rejected = !fr.passed;
      const aiTag = rejected
        ? `<span class="chip chip-warn">Rejected: ${esc(fr.filter_label)}</span>`
        : (ai ? (ai.source==="gemma"?`<span class="hi-ai gemma">Gemma ✓</span>`:`<span class="hi-ai fallback">fallback</span>`) : `<span class="hi-ai none">no AI</span>`);
      return `<li class="history-item"><span class="hi-time">${fmtTime(c.timestamp)}</span>
        <span class="cat-pill cat-${esc(c.category)}">${prettyCat(c.category)}</span>
        <span class="sev sev-${esc(c.severity)}">${esc(c.severity)}</span>
        <span class="hi-main"><span class="hi-loc">${esc(c.readings.location_label)}</span></span>
        ${aiTag}</li>`;
    }).join("");
  }

  function renderStats(s) {
    els.statTotal.textContent = s.total??0;
    els.statNormal.textContent = (s.by_category||{}).normal_driving||0;
    els.statSuspicious.textContent = (s.by_category||{}).suspicious_event||0;
    els.statAccident.textContent = (s.by_category||{}).possible_accident||0;
    els.statRejected.textContent = s.rejected??0;
  }

  async function loadAll() {
    try {
      const [h,s,inc] = await Promise.all([api("/api/history"),api("/api/stats"),api("/api/incidents")]);
      renderHistory(h.events||[]); renderStats(s); renderActiveIncidents(inc.incidents||[]);
    } catch {}
  }

  // ---------- Poll active incidents ----------
  function startPolling() { if(pollTimer) return; pollTimer=setInterval(async()=>{ try { const d=await api("/api/incidents"); renderActiveIncidents(d.incidents||[]); const cur=currentIncidentId?(d.incidents||[]).find(i=>i.id===currentIncidentId):null; if(cur) { renderIncident(cur); } } catch {} }, 2000); }
  function stopPolling() { if(pollTimer){ clearInterval(pollTimer); pollTimer=null; } }

  // ---------- Generate ----------
  async function generateEvent() {
    const scenario = els.scenarioSelect.value;
    try {
      els.btnGenerate.disabled=true;
      renderAiLoading();
      const d = await api("/api/event/generate", {method:"POST", body:JSON.stringify({scenario})});
      currentIncidentId = d.incident ? d.incident.id : null;
      renderPhonePanel(d.phone_stream||{});
      renderVehiclePanel(d.vehicle_stream||{});
      renderCurrentEvent(d.event, d.fusion);
      renderAiResult(d.ai, null);
      renderAiStatus(d.ai_status, null);
      renderIncident(d.incident);
      // pipeline
      const fr = d.filter_result||{};
      const ai = d.ai||{};
      const inc = d.incident||{};
      const pState = {
        sensor:"done", sensor_detail:`${d.phone_stream?Object.keys(d.phone_stream).length:0} phone + ${d.vehicle_stream?Object.keys(d.vehicle_stream).length:0} vehicle fields`,
        validation:"done", validation_detail:"passed",
        fusion:"done", fusion_detail:`${(d.fusion||{}).agreeing_signals||0} of 6 signals`,
        filter: fr.passed?"done":"failed",
        filter_detail: fr.passed?"passed":`Rejected: ${fr.filter_label||""}`,
        gemma: ai.source==="gemma"?"done":(ai.source==="fallback"?"done":"skipped"),
        gemma_detail: ai.source==="gemma"?`${ai.severity_opinion?.toUpperCase()||""}, ${Math.round((ai.confidence||0)*100)}%`:"Non-AI fallback",
        gemma_fallback: ai.source!=="gemma",
        safety:"done", safety_detail: inc.safety?inc.safety.note:"—",
        dispatch: (inc.status==="alerted"||inc.status==="assigned"||inc.dispatch?.offered_to)?"done":"skipped",
        dispatch_detail: inc.dispatch?.accepted_unit ? `${inc.dispatch.accepted_unit}, ETA ${inc.dispatch.eta_min}m` : (inc.dispatch?.offered_to ? `offered ${inc.dispatch.offered_to}` : (fr.passed?"dispatching…":"not needed")),
      };
      setPipeline(pState);
      // photo upload target
      if(els.photoInput) {
        els.photoInput.onchange = () => handlePhoto(currentIncidentId);
      }
      await loadAll(); startPolling();
    } catch(e) { toast("Could not generate event: "+e.message, "error"); }
    finally { els.btnGenerate.disabled=false; }
  }

  function startDemo() {
    els.btnStart.disabled=true; els.btnStop.disabled=false;
    els.demoHint.textContent="Demo running — cycling scenarios every 4s…"; demoIndex=0;
    runDemoStep(); demoTimer=setInterval(runDemoStep, 4000);
  }
  function stopDemo() {
    if(demoTimer){ clearInterval(demoTimer); demoTimer=null; }
    els.btnStart.disabled=false; els.btnStop.disabled=true;
    els.demoHint.textContent="Start Demo cycles through scenarios every 4 s.";
  }
  async function runDemoStep() {
    const cycle=["normal_driving","phone_drop","hard_braking","pothole","suspicious_event","possible_accident","real_crash"];
    const scenario=cycle[demoIndex%cycle.length]; demoIndex++;
    try {
      renderAiLoading();
      const d = await api("/api/event/generate",{method:"POST",body:JSON.stringify({scenario})});
      currentIncidentId=d.incident?d.incident.id:null;
      renderPhonePanel(d.phone_stream||{}); renderVehiclePanel(d.vehicle_stream||{});
      renderCurrentEvent(d.event,d.fusion); renderAiResult(d.ai,null);
      renderAiStatus(d.ai_status,null); renderIncident(d.incident);
      const fr=d.filter_result||{}, ai=d.ai||{}, inc=d.incident||{};
      setPipeline({
        sensor:"done",sensor_detail:"2 streams",
        validation:"done",validation_detail:"passed",
        fusion:"done",fusion_detail:`${(d.fusion||{}).agreeing_signals||0} of 6`,
        filter:fr.passed?"done":"failed",filter_detail:fr.passed?"passed":`Rejected: ${fr.filter_label||""}`,
        gemma:ai.source==="gemma"?"done":"done",gemma_detail:ai.source==="gemma"?`${ai.severity_opinion?.toUpperCase()||""}, ${Math.round((ai.confidence||0)*100)}%`:"fallback",
        gemma_fallback:ai.source!=="gemma",
        safety:"done",safety_detail:inc.safety?inc.safety.note:"—",
        dispatch:(inc.status==="alerted"||inc.dispatch?.offered_to)?"done":"skipped",
        dispatch_detail:inc.dispatch?.accepted_unit?`${inc.dispatch.accepted_unit}, ETA ${inc.dispatch.eta_min}m`:(fr.passed?"dispatching…":"not needed"),
      });
      await loadAll(); startPolling();
    } catch(e) { stopDemo(); toast("Demo stopped: "+e.message,"error"); }
  }

  async function resetAll() {
    try {
      stopDemo(); stopPolling(); stopVmCountdown();
      await api("/api/reset",{method:"POST",body:"{}"}); await loadAll();
      currentIncidentId=null;
      els.currentEvent.classList.add("event-empty");
      els.currentEvent.innerHTML=`<p class="placeholder">Cleared. Generate a new event to begin.</p>`;
      els.aiPanel.classList.add("ai-empty"); els.aiSrcTag.textContent="AI"; els.aiSrcTag.className="src-tag";
      els.aiPanel.innerHTML=`<p class="placeholder">The Gemma 4 assessment will appear here.</p>`;
      els.incidentPanel.innerHTML=`<p class="placeholder">No incident yet.</p>`;
      els.phonePanel.innerHTML=`<p class="placeholder">No event yet.</p>`;
      els.vehiclePanel.innerHTML=`<p class="placeholder">No event yet.</p>`;
      els.victimCheckPanel.hidden=true; els.photoUpload.hidden=true;
      setPipeline(null); toast("History cleared","ok");
    } catch(e) { toast("Could not reset: "+e.message,"error"); }
  }
  async function setToggle(field,value) {
    try { const d=await api("/api/toggle-ai",{method:"POST",body:JSON.stringify({[field]:value})}); renderAiStatus(d.ai,null); }
    catch { toast("Toggle failed","error"); }
  }
  async function aiCheck() {
    els.btnAiCheck.disabled=true; const old=els.btnAiCheck.textContent; els.btnAiCheck.textContent="Testing…";
    try {
      const d=await api("/api/ai-check");
      if(d.status==="connected") toast("AI connected — real Gemma response received.","ok");
      else toast(`AI ${d.status}${d.reason?": "+d.reason:""}`,d.status==="not_configured"?"":"error");
      refreshAiStatus();
    } catch { toast("AI check failed","error"); }
    finally { els.btnAiCheck.disabled=false; els.btnAiCheck.textContent=old; }
  }

  // ---------- Wire up ----------
  els.btnStart.addEventListener("click",startDemo);
  els.btnStop.addEventListener("click",stopDemo);
  els.btnGenerate.addEventListener("click",generateEvent);
  els.btnReset.addEventListener("click",resetAll);
  els.btnAiCheck.addEventListener("click",aiCheck);
  els.toggleAi.addEventListener("change",()=>setToggle("ai_enabled",els.toggleAi.checked));
  els.toggleAutoAccept.addEventListener("change",()=>setToggle("auto_accept",els.toggleAutoAccept.checked));
  els.toggleVictimAuto.addEventListener("change",()=>setToggle("victim_auto_respond",els.toggleVictimAuto.checked));
  els.vmBtnSafe?.addEventListener("click",()=>vmRespond("ok"));
  els.vmBtnHelp?.addEventListener("click",()=>vmRespond("help"));

  refreshAiStatus(); loadAll();

  async function poll() {
    try {
      const d=await api("/api/incidents"); renderActiveIncidents(d.incidents||[]);
      const cur=currentIncidentId?(d.incidents||[]).find(i=>i.id===currentIncidentId):null;
      if(cur) renderIncident(cur);
    } catch {}
  }
  window.quickScenario = s => { els.scenarioSelect.value=s; els.btnGenerate.click(); };
})();
