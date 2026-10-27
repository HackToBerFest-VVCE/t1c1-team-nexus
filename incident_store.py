"""
AccidentGuard — Incident store (in-memory, shared state)
========================================================

Central, thread-safe, in-memory state for the full pipeline so the dashboard,
the victim screen and the ambulance screen all see the same incidents.

It owns:
  * creating incidents from a (classification, ai_result, gemma_enabled) triple,
  * running the safety layer,
  * the victim response countdown (10s; skipped for critical),
  * dispatch offering + 20s escalation to the next nearest unit,
  * status transitions (validated via dispatch.validate_transition),
  * sanitised views for the victim screen (no evidence/confidence/sensors),
  * runtime toggles (AI on/off, auto-accept for solo demo).

Timing uses a swappable clock so tests are deterministic without real sleeps.

SIMULATION ONLY — no real emergency calls. "emergency contact notified" is a
simulated log message.
"""

from __future__ import annotations

import threading
import time
import random
import uuid
from datetime import datetime, timezone

import accident_detection as det
import dispatch
import safety_layer
import gemma_service as gs

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

VICTIM_COUNTDOWN_S = 10
ESCALATION_S = 20
ETA_MIN, ETA_MAX = 4, 12  # minutes (simulated)

_lock = threading.RLock()
_incidents: dict[str, dict] = {}

# Swappable clock for deterministic tests. Default: wall time.
_clock = time.time


def set_clock(fn):
    global _clock
    _clock = fn


def now() -> float:
    return _clock()


def _iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


# Runtime toggles (for the dashboard)
_toggles = {"ai_enabled": True, "auto_accept": True, "victim_auto_respond": False}


def get_toggles() -> dict:
    with _lock:
        return dict(_toggles)


def set_ai_enabled(value: bool):
    with _lock:
        _toggles["ai_enabled"] = bool(value)
    return get_toggles()


def set_auto_accept(value: bool):
    with _lock:
        _toggles["auto_accept"] = bool(value)
    return get_toggles()


def set_victim_auto_respond(value: bool):
    with _lock:
        _toggles["victim_auto_respond"] = bool(value)
    return get_toggles()


def add_scene_analysis(incident_id: str, analysis: dict) -> dict | None:
    """Attach a photo scene-analysis result to an existing incident."""
    with _lock:
        inc = _incidents.get(incident_id)
        if inc is None:
            return None
        inc["scene_analysis"] = analysis
        return _public(inc)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _log(incident, scope, message):
    if scope in ("victim", "dispatch"):
        incident[scope]["log"].append(message)
    incident["timeline"].append({"t": _iso(), "event": message})


def _reset_units():
    for u in dispatch.all_units():
        u.available = True
        u.current_incident = None


def reset():
    with _lock:
        _incidents.clear()
        _reset_units()
    return {"ok": True}


# ---------------------------------------------------------------------------
# Incident creation (runs the safety layer)
# ---------------------------------------------------------------------------

def create_incident(classification: det.EventClassification,
                    ai_result: dict,
                    gemma_enabled: bool,
                    fusion: dict | None = None,
                    filter_result: dict | None = None,
                    phone_stream: dict | None = None,
                    vehicle_stream: dict | None = None) -> dict:
    """
    Build an incident from a rule-engine classification + a (Gemma or fallback)
    AI result, run the safety layer, and start the victim countdown / alert.
    """
    readings = classification.readings
    loc = dispatch.coords_for(readings.location_label)
    gemma_sev = None
    if ai_result and ai_result.get("source") == "gemma":
        gemma_sev = ai_result.get("severity_opinion") or ai_result.get("severity")

    safety = safety_layer.decide_final_severity(
        classification.severity, readings, gemma_sev
    )
    final_sev = safety["final_severity"]

    incident = {
        "id": f"INC-{uuid.uuid4().hex[:8].upper()}",
        "classification": classification.to_dict(),
        "ai_result": ai_result,
        "gemma_enabled": gemma_enabled,
        "safety": safety,
        "severity": final_sev,
        "status": "new",
        "created_ts": now(),
        "timestamp": classification.timestamp,
        "location_label": readings.location_label,
        "lat": loc[0],
        "lng": loc[1],
        "fusion": fusion or {},
        "filter_result": filter_result or {"passed": True, "reason": "", "filter_label": "passed"},
        "phone_stream": phone_stream or {},
        "vehicle_stream": vehicle_stream or {},
        "scene_analysis": None,
        "dispatch_briefing": None,
        "victim": {"answer": None, "deadline": None, "log": []},
        "dispatch": {
            "offered_to": None,
            "offer_deadline": None,
            "offer_tried": [],
            "accepted_unit": None,
            "eta_min": None,
            "control_room": False,
            "log": [],
        },
        "timeline": [],
        "ai_evidence": (ai_result or {}).get("evidence", []) if gemma_enabled else [],
    }

    _log(incident, "victim", f"Incident created (simulated). Final severity: {final_sev}.")

    # If the false-alarm filter rejected this event, resolve it immediately — no alert.
    if not (filter_result or {}).get("passed", True):
        incident["status"] = "resolved"
        label = (filter_result or {}).get("filter_label", "false alarm")
        _log(incident, "victim", f"False alarm rejected: {label}. No alert sent (simulated).")
    elif not safety["alert_needed"]:
        incident["status"] = "resolved"
        _log(incident, "victim", "No alert needed (low severity).")
    elif final_sev == "critical":
        _start_alert(incident)
    else:  # medium or high -> victim countdown
        incident["status"] = "awaiting_victim"
        incident["victim"]["deadline"] = now() + VICTIM_COUNTDOWN_S
        _log(incident, "victim",
             f"Victim check started — {VICTIM_COUNTDOWN_S}s countdown (simulated).")

    with _lock:
        _incidents[incident["id"]] = incident
    return incident


def _start_alert(incident: dict):
    """Send the alert now and begin dispatch offering."""
    was_countdown = incident["status"] == "awaiting_victim"
    incident["status"] = "alerted"
    incident["victim"]["deadline"] = None
    msg = "No victim response — alert sent now (simulated)." if was_countdown \
        else "Critical severity — alert sent immediately (simulated)."
    _log(incident, "victim", msg)
    _log(incident, "dispatch", "Simulated emergency contact notified.")
    _offer_nearest(incident)


def _offer_nearest(incident: dict):
    """Offer the incident to the nearest available unit, set escalation deadline."""
    loc = (incident["lat"], incident["lng"])
    unit = dispatch.next_unit_to_offer(dispatch.all_units(), loc, set(incident["dispatch"]["offer_tried"]))
    if unit is None:
        incident["dispatch"]["control_room"] = True
        incident["dispatch"]["offered_to"] = None
        incident["dispatch"]["offer_deadline"] = None
        _log(incident, "dispatch", "No available units — escalated to control room (simulated).")
        return
    incident["dispatch"]["offered_to"] = unit.name
    incident["dispatch"]["offer_deadline"] = now() + ESCALATION_S
    incident["dispatch"]["offer_tried"].append(unit.name)
    dist = dispatch.haversine_km((unit.lat, unit.lng), loc)
    _log(incident, "dispatch",
         f"Offered to {unit.name} ({dist:.1f} km). Waiting up to {ESCALATION_S}s (simulated).")


# ---------------------------------------------------------------------------
# Time-driven processing (call from poll endpoints / a worker)
# ---------------------------------------------------------------------------

def tick():
    """Advance all due countdowns and escalations. Safe to call often."""
    with _lock:
        for inc in list(_incidents.values()):
            _process(inc)


def _process(incident: dict):
    st = incident["status"]

    # Victim countdown expired with no answer -> auto alert
    if st == "awaiting_victim":
        dl = incident["victim"]["deadline"]
        # victim_auto_respond toggle: auto-answer "ok" or "help" for demo
        if _toggles.get("victim_auto_respond") and incident["victim"]["answer"] is None:
            # auto-respond with "help" after 2 seconds for demo (safe choice)
            if dl is not None and now() >= incident["created_ts"] + 2.0:
                incident["victim"]["answer"] = "help"
                _log(incident, "victim", "Auto-responded: I need help (demo toggle).")
                _start_alert(incident)
                return
        if dl is not None and now() >= dl and incident["victim"]["answer"] is None:
            _start_alert(incident)
            st = incident["status"]

    # Dispatch escalation: offer not accepted in time -> next nearest unit
    if st == "alerted":
        dl = incident["dispatch"]["offer_deadline"]
        if dl is not None and now() >= dl and incident["dispatch"]["accepted_unit"] is None:
            offered = incident["dispatch"]["offered_to"]
            if offered:
                _log(incident, "dispatch", f"{offered} did not accept in time — escalating.")
            _offer_nearest(incident)
        # Auto-accept demo: a unit accepts automatically
        if _toggles["auto_accept"] and incident["dispatch"]["offered_to"] \
                and incident["dispatch"]["accepted_unit"] is None:
            _accept_unit(incident, incident["dispatch"]["offered_to"])


# ---------------------------------------------------------------------------
# Victim response
# ---------------------------------------------------------------------------

def respond(incident_id: str, answer: str) -> dict:
    """answer: 'ok' or 'help'. Returns {ok, incident} or {ok:False, error}."""
    with _lock:
        inc = _incidents.get(incident_id)
        if inc is None:
            return {"ok": False, "error": "Incident not found."}
        st = inc["status"]
        if answer == "ok":
            if st != "awaiting_victim":
                return {"ok": False, "error": "Cannot cancel — alert already sent."}
            inc["victim"]["answer"] = "ok"
            inc["victim"]["deadline"] = None
            inc["status"] = "cancelled"
            _log(inc, "victim", "Cancelled by victim (I'm OK).")
            return {"ok": True, "incident": _public(inc)}
        if answer == "help":
            inc["victim"]["answer"] = "help"
            _log(inc, "victim", "Victim requested help — alert sent now (simulated).")
            if st == "awaiting_victim":
                _start_alert(inc)
            elif st == "new":
                _start_alert(inc)
            # if already alerted, dispatch already running
            return {"ok": True, "incident": _public(inc)}
        return {"ok": False, "error": "answer must be 'ok' or 'help'."}


# ---------------------------------------------------------------------------
# Dispatch accept + status updates
# ---------------------------------------------------------------------------

def accept(incident_id: str, unit_name: str) -> dict:
    with _lock:
        inc = _incidents.get(incident_id)
        if inc is None:
            return {"ok": False, "error": "Incident not found."}
        if inc["status"] not in ("alerted", "escalated"):
            return {"ok": False, "error": f"Cannot accept: incident is {inc['status']}.",
                     "allowed": False}
        # The unit must be the one offered (and available), unless auto-accept.
        unit = next((u for u in dispatch.all_units() if u.name == unit_name), None)
        if unit is None:
            return {"ok": False, "error": f"Unknown unit: {unit_name}."}
        if not unit.available:
            return {"ok": False, "error": f"{unit_name} is not available."}
        if inc["dispatch"]["offered_to"] != unit_name:
            return {"ok": False, "error": f"Incident was not offered to {unit_name}."}
        _accept_unit(inc, unit_name)
        return {"ok": True, "incident": _public(inc)}


def _accept_unit(incident: dict, unit_name: str):
    unit = next((u for u in dispatch.all_units() if u.name == unit_name), None)
    if unit is None or not unit.available:
        return
    unit.available = False
    unit.current_incident = incident["id"]
    eta = random.randint(ETA_MIN, ETA_MAX)
    incident["dispatch"]["accepted_unit"] = unit_name
    incident["dispatch"]["offer_deadline"] = None
    incident["dispatch"]["eta_min"] = eta
    ok, err = dispatch.validate_transition(incident["status"], "assigned")
    if ok:
        incident["status"] = "assigned"
    _log(incident, "dispatch", f"{unit_name} accepted — ETA {eta} min (simulated).")
    # Call 2: update responder briefing now that we know the unit and ETA
    ai = incident.get("ai_result") or {}
    victim_answer = incident["victim"]["answer"]
    elapsed = now() - incident["created_ts"]
    try:
        briefing2 = gs.generate_dispatch_briefing(
            assessment=ai,
            victim_answer=victim_answer,
            seconds_without_response=elapsed,
            nearest_unit=unit_name,
            eta_min=eta,
        )
        incident["dispatch_briefing"] = briefing2
        _log(incident, "dispatch",
             f"Dispatch briefing {'updated by Gemma' if briefing2.get('source') == 'gemma' else 'from fallback'} (simulated).")
    except Exception:
        incident["dispatch_briefing"] = None


def set_status(incident_id: str, new_status: str) -> dict:
    with _lock:
        inc = _incidents.get(incident_id)
        if inc is None:
            return {"ok": False, "error": "Incident not found."}
        ok, err = dispatch.validate_transition(inc["status"], new_status)
        if not ok:
            return {"ok": False, "error": err}
        prev = inc["status"]
        inc["status"] = new_status
        _log(inc, "dispatch", f"Status: {prev} -> {new_status}.")
        if new_status == "resolved":
            # free the assigned unit, if any
            u = inc["dispatch"]["accepted_unit"]
            if u:
                unit = next((x for x in dispatch.all_units() if x.name == u), None)
                if unit:
                    unit.available = True
                    unit.current_incident = None
            _log(inc, "victim", "Incident resolved.")
        return {"ok": True, "incident": _public(inc)}


# ---------------------------------------------------------------------------
# Views
# ---------------------------------------------------------------------------

def _public(incident: dict) -> dict:
    """Full incident view (dashboard / ambulance)."""
    return {
        "id": incident["id"],
        "classification": incident["classification"],
        "ai_result": incident["ai_result"],
        "gemma_enabled": incident["gemma_enabled"],
        "safety": incident["safety"],
        "severity": incident["severity"],
        "status": incident["status"],
        "timestamp": incident["timestamp"],
        "created_ts": incident["created_ts"],
        "location_label": incident["location_label"],
        "lat": incident["lat"],
        "lng": incident["lng"],
        "fusion": incident.get("fusion", {}),
        "filter_result": incident.get("filter_result", {}),
        "phone_stream": incident.get("phone_stream", {}),
        "vehicle_stream": incident.get("vehicle_stream", {}),
        "scene_analysis": incident.get("scene_analysis"),
        "dispatch_briefing": incident.get("dispatch_briefing"),
        "victim": {
            "answer": incident["victim"]["answer"],
            "deadline": incident["victim"]["deadline"],
            "log": list(incident["victim"]["log"]),
        },
        "dispatch": {
            "offered_to": incident["dispatch"]["offered_to"],
            "offer_deadline": incident["dispatch"]["offer_deadline"],
            "offer_tried": list(incident["dispatch"]["offer_tried"]),
            "accepted_unit": incident["dispatch"]["accepted_unit"],
            "eta_min": incident["dispatch"]["eta_min"],
            "control_room": incident["dispatch"]["control_room"],
            "log": list(incident["dispatch"]["log"]),
        },
        "timeline": list(incident["timeline"]),
        "disagreement": incident["safety"].get("disagreement", False),
        "safety_note": incident["safety"].get("note", ""),
    }


def list_incidents() -> list[dict]:
    with _lock:
        items = sorted(_incidents.values(),
                       key=lambda i: i["created_ts"], reverse=True)
        return [_public(i) for i in items]


def get_incident(incident_id: str):
    with _lock:
        inc = _incidents.get(incident_id)
        return _public(inc) if inc else None


def _countdown_remaining(incident: dict) -> int:
    dl = incident["victim"]["deadline"]
    if dl is None or incident["status"] != "awaiting_victim":
        return 0
    return max(0, int(round(dl - now())))


def my_incident() -> dict:
    """
    The victim's current incident, SANITISED: no evidence, no confidence,
    no sensor numbers. Used by /api/my-incident.
    """
    with _lock:
        active = [i for i in _incidents.values()
                  if i["status"] in ("new", "awaiting_victim", "alerted",
                                     "assigned", "en_route", "arrived")]
        if not active:
            return {"incident": None}
        inc = max(active, key=lambda i: i["created_ts"])
        ai = inc["ai_result"] or {}
        alert = {
            "en": ai.get("alert_en", ""),
            "kn": ai.get("alert_kn", ""),
            "hi": ai.get("alert_hi", ""),
        }
        tips = {
            "en": ai.get("alert_en", ""),
            "kn": ai.get("alert_kn", ""),
            "hi": ai.get("alert_hi", ""),
        }
        return {
            "incident": {
                "id": inc["id"],
                "status": inc["status"],
                "timestamp": inc["timestamp"],
                "location_label": inc["location_label"],
                "lat": inc["lat"],
                "lng": inc["lng"],
                "countdown": _countdown_remaining(inc),
                "victim_answer": inc["victim"]["answer"],
                "alert": alert,
                "first_aid": tips,
                "accepted_unit": inc["dispatch"]["accepted_unit"],
                "eta_min": inc["dispatch"]["eta_min"],
                "timeline": list(inc["timeline"]),
            }
        }


def list_units(incident_id=None):
    """Units with optional distance and estimated time to a given incident."""
    loc = None
    if incident_id:
        with _lock:
            inc = _incidents.get(incident_id)
            if inc:
                loc = (inc["lat"], inc["lng"])
    out = []
    for u in dispatch.all_units():
        d = {
            "name": u.name,
            "lat": u.lat,
            "lng": u.lng,
            "available": u.available,
            "current_incident": u.current_incident,
        }
        if loc is not None:
            km = round(dispatch.haversine_km((u.lat, u.lng), loc), 2)
            d["distance_km"] = km
            # ETA: distance / 30 km/h + small traffic factor, rounded to 1 min
            traffic_factor = 1.25
            d["eta_min"] = max(1, round(km / 30.0 * 60 * traffic_factor))
        out.append(d)
    return out
