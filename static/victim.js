/* AccidentGuard — Victim screen */
(function () {
  "use strict";
  const $ = id => document.getElementById(id);
  const els = {
    safe:$("safeState"), countdown:$("countdownState"), alert:$("alertState"),
    countdownNum:$("countdownNum"), ringProgress:$("ringProgress"),
    btnOk:$("btnOk"), btnHelp:$("btnHelp"),
    alertStatus:$("alertStatus"), alertTitle:$("alertTitle"), alertBadge:$("alertBadge"),
    tipText:$("tipText"), vTimeline:$("vTimeline"), vLoc:$("vLoc"), vUnit:$("vUnit"),
    tabs:document.querySelectorAll(".v-tab"),
    victimPhotoInput:$("victimPhotoInput"), victimPhotoResult:$("victimPhotoResult"),
  };
  const CIRCUMFERENCE = 276.46;
  let current=null, lang="en", tickTimer=null, displayedCountdown=null;

  async function api(path, opts={}) {
    const r=await fetch(path,{headers:{"Content-Type":"application/json"},...opts}); return r.json();
  }
  const esc = s => String(s==null?"":s).replace(/&/g,"&amp;").replace(/</g,"&lt;").replace(/>/g,"&gt;");

  let actx=null;
  function beep() {
    try {
      actx = actx || new(window.AudioContext||window.webkitAudioContext)();
      const o=actx.createOscillator(), g=actx.createGain();
      o.type="sine"; o.frequency.value=880; o.connect(g); g.connect(actx.destination);
      g.gain.setValueAtTime(0.0001,actx.currentTime);
      g.gain.exponentialRampToValueAtTime(0.3,actx.currentTime+0.02);
      g.gain.exponentialRampToValueAtTime(0.0001,actx.currentTime+0.4);
      o.start(); o.stop(actx.currentTime+0.4);
    } catch {}
  }
  function vibrate(p) { try { navigator.vibrate&&navigator.vibrate(p); } catch {} }
  document.addEventListener("click", function once() { try { (actx||new(window.AudioContext||window.webkitAudioContext)()).resume(); } catch {} document.removeEventListener("click",once); }, {once:true});

  function showState(name) {
    els.safe.classList.toggle("v-hidden", name!=="safe");
    els.countdown.classList.toggle("v-hidden", name!=="countdown");
    els.alert.classList.toggle("v-hidden", name!=="alert");
  }

  function updateRing(secs, total=10) {
    if(!els.ringProgress) return;
    const frac = Math.max(0, Math.min(1, secs/total));
    els.ringProgress.style.strokeDashoffset = CIRCUMFERENCE * (1 - frac);
    els.ringProgress.classList.toggle("urgent", secs <= 3);
    if(els.countdownNum) { els.countdownNum.textContent=secs; els.countdownNum.style.color=secs<=3?"var(--crit)":""; }
  }

  function renderTip() {
    if(!current) return;
    const tips = current.first_aid || current.alert || {};
    els.tipText.textContent = tips[lang] || tips.en || "—";
  }
  function renderTimeline() {
    if(!current) return;
    const stages = [
      {label:"Alert sent", done:["alerted","assigned","en_route","arrived"].includes(current.status)},
      {label:"Ambulance assigned", done:!!current.accepted_unit},
      {label:current.eta_min?`On the way · ETA ${current.eta_min} min`:"On the way", done:["en_route","arrived"].includes(current.status)},
      {label:"Arrived", done:current.status==="arrived"},
    ];
    els.vTimeline.innerHTML=stages.map(s=>`<div class="tl-item ${s.done?"done":""}"><span>${s.label}</span></div>`).join("");
  }
  function renderAlert() {
    els.alertStatus.textContent=`status: ${current.status}`;
    els.vLoc.textContent = current.location_label||"Simulated location";
    els.vUnit.textContent = current.accepted_unit
      ? `${current.accepted_unit} · ETA ${current.eta_min} min (simulated)`
      : "Dispatching nearest unit (simulated)…";
    const badge = current.status==="arrived"?"Arrived":(current.status==="assigned"?"Assigned":"Alert sent");
    els.alertBadge.textContent=badge;
    els.alertBadge.className="sev "+(current.status==="arrived"?"sev-low":"sev-critical");
    els.alertTitle.textContent = current.accepted_unit?"Help is on the way":"Alert sent — dispatching help";
    renderTip(); renderTimeline();
  }

  function applyState() {
    if(!current) { showState("safe"); stopTick(); return; }
    if(current.status==="awaiting_victim") {
      showState("countdown");
      displayedCountdown = current.countdown!=null ? current.countdown : 10;
      updateRing(displayedCountdown);
      beep(); vibrate([200,100,200]);
      startTick();
    } else if(["alerted","assigned","en_route","arrived","resolved","escalated","new"].includes(current.status)) {
      showState("alert"); renderAlert(); stopTick();
    } else {
      showState("safe"); stopTick();
    }
  }
  function startTick() {
    if(tickTimer) return;
    tickTimer=setInterval(()=>{
      if(displayedCountdown==null) return;
      displayedCountdown=Math.max(0,displayedCountdown-1);
      updateRing(displayedCountdown);
    },1000);
  }
  function stopTick() { if(tickTimer){ clearInterval(tickTimer); tickTimer=null; } }

  async function respond(answer) {
    if(!current) return;
    try { await api(`/api/incidents/${encodeURIComponent(current.id)}/respond`,{method:"POST",body:JSON.stringify({answer})}); } catch {}
    poll();
  }

  // photo upload
  if(els.victimPhotoInput) {
    els.victimPhotoInput.addEventListener("change", () => {
      const f=els.victimPhotoInput.files?.[0];
      if(!f||!current) return;
      const reader=new FileReader();
      reader.onload=async ev=>{
        const b64=ev.target.result.split(",")[1];
        els.victimPhotoResult.hidden=false; els.victimPhotoResult.innerHTML="<span class='spinner'></span> Analysing…";
        try {
          const d=await api(`/api/incidents/${encodeURIComponent(current.id)}/photo`,{method:"POST",body:JSON.stringify({image_data:b64,mime_type:f.type||"image/jpeg"})});
          const sc=d.scene_analysis;
          els.victimPhotoResult.innerHTML=sc?.available
            ?`Scene: ${esc(sc.notes)} Vehicles: ${sc.vehicles}, Injuries: ${esc(sc.visible_injuries)}`
            :`<span class="muted">${esc(sc?.notes||"Scene analysis unavailable.")}</span>`;
        } catch(e) { els.victimPhotoResult.innerHTML=`<span class="ai-error">Failed: ${esc(e.message)}</span>`; }
      };
      reader.readAsDataURL(f);
    });
  }

  async function poll() {
    try {
      const d=await api("/api/my-incident");
      const inc=d.incident;
      if(!inc) { current=null; applyState(); return; }
      const same=current&&current.id===inc.id;
      const wasCountdown=current&&current.status==="awaiting_victim";
      current=inc;
      if(!same||(wasCountdown&&inc.status!=="awaiting_victim")) applyState();
      else if(inc.status!=="awaiting_victim") applyState();
      else { displayedCountdown=inc.countdown!=null?inc.countdown:displayedCountdown; updateRing(displayedCountdown||0); }
    } catch {}
  }

  els.btnOk?.addEventListener("click",()=>respond("ok"));
  els.btnHelp?.addEventListener("click",()=>respond("help"));
  els.tabs.forEach(t=>t.addEventListener("click",()=>{ els.tabs.forEach(x=>x.classList.remove("active")); t.classList.add("active"); lang=t.dataset.lang; renderTip(); }));

  poll();
  setInterval(poll,2000);
})();
