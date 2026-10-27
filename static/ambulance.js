/* AccidentGuard — Ambulance console */
(function () {
  "use strict";
  const $ = id => document.getElementById(id);
  const els = {
    unitSelect:$("unitSelect"), ambList:$("ambList"), listCount:$("listCount"),
    ambDetail:$("ambDetail"), ambActions:$("ambActions"),
    btnAccept:$("btnAccept"), btnEnRoute:$("btnEnRoute"), btnArrived:$("btnArrived"), btnResolved:$("btnResolved"),
    cActive:$("cActive"), cCritical:$("cCritical"), cRespTime:$("cRespTime"),
    map:$("map"), nearestLine:$("nearestLine"), mapNote:$("mapNote"), detailSrc:$("detailSrc"),
    unitsBody:$("unitsBody"),
  };
  let myUnit=null, selectedId=null, serverNow=0, seenCritical=new Set();
  let units=[], incidents=[];

  async function api(path,opts={}) {
    const r=await fetch(path,{headers:{"Content-Type":"application/json"},...opts});
    if(!r.ok) throw new Error(`Request failed (${r.status})`); return r.json();
  }
  const esc=s=>String(s==null?"":s).replace(/&/g,"&amp;").replace(/</g,"&lt;").replace(/>/g,"&gt;");
  const prettyCat=c=>({normal_driving:"Normal",suspicious_event:"Suspicious",possible_accident:"Possible accident"})[c]||c;
  const sevRank=s=>({low:1,medium:2,high:3,critical:4})[s]||0;
  const fmtTime=iso=>{try{return new Date(iso).toLocaleTimeString();}catch{return iso;}};

  let actx=null;
  function beep() {
    try { actx=actx||new(window.AudioContext||window.webkitAudioContext)();
      const o=actx.createOscillator(),g=actx.createGain();
      o.type="square";o.frequency.value=660;o.connect(g);g.connect(actx.destination);
      g.gain.setValueAtTime(0.0001,actx.currentTime);
      g.gain.exponentialRampToValueAtTime(0.25,actx.currentTime+0.02);
      g.gain.exponentialRampToValueAtTime(0.0001,actx.currentTime+0.5);
      o.start();o.stop(actx.currentTime+0.5);
    } catch {}
  }
  document.addEventListener("click",function once(){try{(actx=actx||new(window.AudioContext||window.webkitAudioContext)()).resume();}catch{}document.removeEventListener("click",once);},{once:true});

  async function loadUnits() {
    const d=await api("/api/units"); units=d.units||[];
    if(!els.unitSelect.options.length) {
      els.unitSelect.innerHTML=units.map(u=>`<option value="${esc(u.name)}">${esc(u.name)}</option>`).join("");
      myUnit=units[0]?units[0].name:null;
    }
  }

  function mine(i) { const d=i.dispatch||{}; return d.offered_to===myUnit||d.accepted_unit===myUnit; }

  // Auto-select highest-severity incident on load
  function autoSelect() {
    if(selectedId) return;
    const best = incidents.slice().sort((a,b)=>sevRank(b.severity)-sevRank(a.severity))[0];
    if(best && mine(best)) { selectedId=best.id; renderList(); renderDetail(); renderMap(); }
    else if(best) { selectedId=best.id; renderList(); renderDetail(); renderMap(); }
  }

  function renderList() {
    const listed = incidents.filter(i=>!["resolved","cancelled"].includes(i.status));
    listed.sort((a,b)=>(sevRank(b.severity)-sevRank(a.severity))||(a.created_ts-b.created_ts));
    els.listCount.textContent=listed.length;
    if(!listed.length){els.ambList.innerHTML=`<li class="placeholder">No active incidents.</li>`;return;}
    els.ambList.innerHTML=listed.map(i=>{
      const d=i.dispatch||{};
      let escBadge="";
      if(!d.accepted_unit&&d.offer_deadline){
        const rem=Math.max(0,Math.round(d.offer_deadline-serverNow));
        escBadge=`<span class="ai-esc">esc in ${rem}s</span>`;
      }
      const isNewCrit=i.severity==="critical"&&!seenCritical.has(i.id);
      if(i.severity==="critical") seenCritical.add(i.id);
      if(isNewCrit) beep();
      return `<li class="amb-item ${i.id===selectedId?"selected":""} ${isNewCrit?"new-critical":""}" data-id="${esc(i.id)}">
        <span class="sev sev-${esc(i.severity)}">${esc(i.severity)}</span>
        <span><b>${prettyCat(i.classification.category)}</b><br><span class="muted" style="font-size:.74rem">${esc(i.location_label)} · ${fmtTime(i.timestamp)}</span></span>
        <span><span class="chip">${esc(i.status)}</span><br>${escBadge}</span>
      </li>`;
    }).join("");
    els.ambList.querySelectorAll(".amb-item").forEach(el=>el.addEventListener("click",()=>{
      selectedId=el.dataset.id; renderList(); renderDetail(); renderMap();
    }));
  }

  function selectedIncident() { return incidents.find(i=>i.id===selectedId)||null; }

  function renderDetail() {
    const i=selectedIncident();
    if(!i){els.ambDetail.innerHTML=`<p class="placeholder">Select an incident.</p>`;els.ambActions.hidden=true;return;}
    const ai=i.ai_result||{}, s=i.safety||{}, d=i.dispatch||{}, fr=i.filter_result||{}, fusion=i.fusion||{};
    const briefing = (i.dispatch_briefing && i.dispatch_briefing.responder_briefing) || ai.responder_briefing || [];
    els.detailSrc.textContent=ai.source==="gemma"?"Gemma 4":(ai.source==="fallback"?"Non-AI fallback":"—");
    const ev=(ai.evidence||[]).map(e=>`<li>${esc(e)}</li>`).join("")||"<li>none</li>";
    const brief=briefing.map(b=>`<li>${esc(b)}</li>`).join("")||"<li>—</li>";
    const conf=(typeof ai.confidence==="number")
      ?`<div class="conf-bar"><span style="width:${Math.round(ai.confidence*100)}%"></span></div>${Math.round(ai.confidence*100)}%`:"—";
    const sigs=(fusion.signals_detail||[]).map(sig=>`<span class="chip chip-sig">${esc(sig)}</span>`).join(" ")||"none";
    const sc=i.scene_analysis;
    const sceneHtml=sc?(!sc.available?`<div class="ai-block"><h4>Scene analysis</h4><p class="muted">${esc(sc.notes||"Unavailable.")}</p></div>`
      :`<div class="ai-block"><h4>Scene analysis (Gemma 4)</h4><p>Vehicles: ${esc(sc.vehicles)} · Injuries: ${esc(sc.visible_injuries)} · Fire: ${esc(sc.fire_or_fuel_risk)} · Road blocked: ${esc(sc.road_blocked)}</p><p class="muted">${esc(sc.notes||"")}</p></div>`):"";
    const victimStatus=i.victim?.answer==="ok"?"Responded safe ✓":i.victim?.answer==="help"?"Asked for help 🔴":"No response yet";
    const briefNote=i.dispatch_briefing?.note?`<p class="muted" style="font-size:.75rem">${esc(i.dispatch_briefing.note)}</p>`:"";
    const filterRow=!fr.passed?`<div class="reject-banner" style="font-size:.82rem">False alarm: ${esc(fr.reason||fr.filter_label)}</div>`:"";
    els.ambDetail.innerHTML=`
      ${filterRow}
      <div class="detail-row"><span class="sev sev-${esc(i.severity)}">${esc(i.severity)}</span>
        <span class="cat-pill cat-${esc(i.classification.category)}">${prettyCat(i.classification.category)}</span>
        <span class="chip">${esc(i.status)}</span></div>
      <div class="ai-block"><h4>Confidence</h4>${conf}</div>
      <div class="ai-block"><h4>Gemma evidence</h4><ul>${ev}</ul></div>
      <div class="ai-block"><h4>Uncertainty</h4><p>${esc(ai.uncertainty||"—")}</p></div>
      <div class="ai-block"><h4>Action</h4><p>${esc(ai.action||"—")}</p></div>
      <div class="ai-block"><h4>Responder briefing (3 lines)</h4><ul>${brief}</ul>${briefNote}</div>
      ${sceneHtml}
      <div class="ai-block"><h4>Sensor fusion — signals agreeing</h4><p>${sigs} (${esc(fusion.agreeing_signals||0)} of 6)</p></div>
      <div class="ai-block"><h4>Safety layer (Python)</h4><p>${esc(s.note||"—")}${s.disagreement?" — <b>disagreement handled</b>":""}</p></div>
      <div class="ai-block"><h4>Victim response</h4><p>${victimStatus}</p></div>
      <p class="detail-meta">${esc(i.id)} · ${esc(i.location_label)} · ${fmtTime(i.timestamp)}${d.accepted_unit?" · "+esc(d.accepted_unit)+" · ETA "+d.eta_min+"m":""}</p>
      <p class="disclaimer">Assessment based only on supplied simulated data. Not a medical diagnosis. For real emergencies contact actual emergency services.</p>`;
    renderActions(i);
  }

  function renderActions(i) {
    const d=i.dispatch||{}, st=i.status;
    const offeredToMe=d.offered_to===myUnit;
    els.ambActions.hidden=false;
    els.btnAccept.disabled=!(offeredToMe&&(st==="alerted"||st==="escalated"));
    els.btnEnRoute.disabled=!(d.accepted_unit===myUnit&&st==="assigned");
    els.btnArrived.disabled=!(d.accepted_unit===myUnit&&st==="en_route");
    els.btnResolved.disabled=!(d.accepted_unit===myUnit&&(st==="arrived"||st==="en_route"||st==="assigned"));
  }

  async function act(path,body,btn) {
    if(btn) btn.disabled=true;
    try { await api(path,{method:"POST",body:JSON.stringify(body)}); } catch {}
    finally { await poll(); }
  }

  function renderCounters() {
    const active=incidents.filter(i=>["new","awaiting_victim","alerted","assigned","en_route","arrived","escalated"].includes(i.status));
    els.cActive.textContent=active.length;
    els.cCritical.textContent=active.filter(i=>i.severity==="critical").length;
    const etas=incidents.map(i=>i.dispatch&&i.dispatch.eta_min).filter(x=>x);
    els.cRespTime.textContent=etas.length?Math.round(etas.reduce((a,b)=>a+b,0)/etas.length)+"m":"—";
  }

  // Ambulance services table
  function renderUnitsTable() {
    if(!units.length){els.unitsBody.innerHTML=`<tr><td colspan="4" class="placeholder">—</td></tr>`;return;}
    const inc=selectedIncident();
    const offeredTo=inc?.dispatch?.offered_to;
    let sorted=[...units].sort((a,b)=>(a.distance_km??999)-(b.distance_km??999));
    els.unitsBody.innerHTML=sorted.map(u=>{
      const isOffered=u.name===offeredTo;
      const isNearest=sorted[0]===u&&u.available;
      const cls=isOffered?"offered-row":isNearest?"nearest-row":"";
      const status=u.available?"available":`busy (${esc(u.current_incident||"—")})`;
      return `<tr class="${cls}"><td>${esc(u.name)}</td><td>${status}</td><td>${u.distance_km!=null?u.distance_km.toFixed(1):"—"}</td><td>${u.eta_min??("—")}</td></tr>`;
    }).join("");
  }

  // SVG map
  const BOX={latMin:12.90,latMax:13.05,lngMin:77.53,lngMax:77.72};
  function project(lat,lng){
    return [((lng-BOX.lngMin)/(BOX.lngMax-BOX.lngMin))*100,
            80-((lat-BOX.latMin)/(BOX.latMax-BOX.latMin))*80];
  }
  function renderMap() {
    const i=selectedIncident();
    let parts=[];
    let pinX=null,pinY=null,nearestX=null,nearestY=null,nearestName=null;
    if(i){
      [pinX,pinY]=project(i.lat,i.lng);
      parts.push(`<circle class="pin" cx="${pinX}" cy="${pinY}" r="2.2"/><circle cx="${pinX}" cy="${pinY}" r="4" fill="none" stroke="#f87171" stroke-width="0.6"/>`);
    }
    // Find nearest available unit (smallest distance_km in units array)
    const sortedU=[...units].sort((a,b)=>(a.distance_km??999)-(b.distance_km??999));
    const nearest=sortedU.find(u=>u.available);
    if(nearest){nearestName=nearest.name;[nearestX,nearestY]=project(nearest.lat,nearest.lng);}
    parts=parts.concat(units.map(u=>{
      const[ux,uy]=project(u.lat,u.lng);
      const cls=u.current_incident?"unit-dot busy":(u.name===nearestName?"unit-dot nearest":"unit-dot");
      return `<circle class="${cls}" cx="${ux}" cy="${uy}" r="1.6"><title>${esc(u.name)}</title></circle><text x="${ux+2}" y="${uy}">${esc(u.name.replace("Ambulance ",""))}</text>`;
    }));
    els.map.innerHTML=parts.join("");
    // Draw line from nearest unit to pin
    if(els.nearestLine){
      if(pinX!=null&&nearestX!=null&&nearestName){
        els.nearestLine.setAttribute("x1",nearestX); els.nearestLine.setAttribute("y1",nearestY);
        els.nearestLine.setAttribute("x2",pinX); els.nearestLine.setAttribute("y2",pinY);
        els.nearestLine.style.opacity="1";
      } else { els.nearestLine.style.opacity="0"; }
    }
    els.mapNote.textContent=i?`${esc(i.location_label)} · nearest: ${esc(nearestName||"—")}`:"No incident selected.";
  }

  async function poll() {
    try {
      const d=await api("/api/incidents");
      incidents=d.incidents||[]; serverNow=d.now||(Date.now()/1000);
      if(selectedId){
        try{const ud=await api(`/api/units?incident=${encodeURIComponent(selectedId)}`);units=ud.units||[];}catch{}
      }
      renderList(); renderCounters(); renderUnitsTable();
      if(selectedId){renderDetail();renderMap();}
      else autoSelect();
    } catch {}
  }

  els.unitSelect.addEventListener("change",()=>{myUnit=els.unitSelect.value;selectedId=null;renderList();renderDetail();renderMap();});
  els.btnAccept.addEventListener("click",()=>act(`/api/incidents/${encodeURIComponent(selectedId)}/accept`,{unit:myUnit},els.btnAccept));
  els.btnEnRoute.addEventListener("click",()=>act(`/api/incidents/${encodeURIComponent(selectedId)}/status`,{status:"en_route"},els.btnEnRoute));
  els.btnArrived.addEventListener("click",()=>act(`/api/incidents/${encodeURIComponent(selectedId)}/status`,{status:"arrived"},els.btnArrived));
  els.btnResolved.addEventListener("click",()=>act(`/api/incidents/${encodeURIComponent(selectedId)}/status`,{status:"resolved"},els.btnResolved));

  (async function init(){await loadUnits();await poll();setInterval(poll,2000);})();
})();
