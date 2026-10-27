"""
Tests for the incident store: victim cancel, critical skip, countdown expiry,
dispatch escalation, accept/eta, status transitions, and the sanitised view.
"""
import time

import accident_detection as det
import incident_store as store


# --- helpers ---------------------------------------------------------------

def _classify(impact, decel, lane, delta, speed=60.0, loc="Simulated · MG Road, Bengaluru"):
    r = det.DrivingReadings(
        timestamp="2026-10-26T10:00:00Z", speed_kmh=speed, speed_delta_kmh=delta,
        deceleration_g=decel, impact_score=impact, lane_deviation_m=lane,
        location_label=loc, event_id="EVT-T")
    return det.classify_event(r)


def _fallback_ai():
    return {
        "source": "fallback", "severity_opinion": None, "evidence": [],
        "uncertainty": "Gemma unavailable", "action": "a", "alert_en": "e",
        "alert_kn": "k", "alert_hi": "h", "responder_briefing": ["b1", "b2", "b3"],
        "summary": "s", "attention_points": [], "recommendations": ["b1"],
    }


def _gemma_ai(sev="critical"):
    return {
        "source": "gemma", "severity_opinion": sev, "confidence": 0.9,
        "evidence": ["e1", "e2"], "uncertainty": "u", "action": "a",
        "alert_en": "e", "alert_kn": "k", "alert_hi": "h",
        "responder_briefing": ["b1", "b2", "b3"], "summary": "s",
        "attention_points": ["e1"], "recommendations": ["b1"],
    }


# --- victim check ----------------------------------------------------------

def test_medium_starts_victim_countdown():
    store.set_clock(lambda: 0.0)
    inc = store.create_incident(_classify(0.4, 0.5, 0.7, -28), _fallback_ai(), False)
    assert inc["status"] == "awaiting_victim"
    assert inc["victim"]["deadline"] == 10.0


def test_critical_skips_countdown_and_alerts():
    store.set_clock(lambda: 0.0)
    inc = store.create_incident(_classify(0.9, 0.9, 1.2, -50), _fallback_ai(), False)
    assert inc["status"] == "alerted"
    assert inc["victim"]["deadline"] is None
    assert inc["dispatch"]["offered_to"] is not None  # dispatch started


def test_victim_ok_cancels():
    store.set_clock(lambda: 0.0)
    inc = store.create_incident(_classify(0.4, 0.5, 0.7, -28), _fallback_ai(), False)
    r = store.respond(inc["id"], "ok")
    assert r["ok"] is True
    assert store.get_incident(inc["id"])["status"] == "cancelled"
    assert store.get_incident(inc["id"])["victim"]["answer"] == "ok"


def test_victim_help_alerts_now():
    store.set_clock(lambda: 0.0)
    inc = store.create_incident(_classify(0.4, 0.5, 0.7, -28), _fallback_ai(), False)
    r = store.respond(inc["id"], "help")
    assert r["ok"] is True
    assert store.get_incident(inc["id"])["status"] == "alerted"


def test_ok_after_alert_rejected():
    store.set_clock(lambda: 0.0)
    inc = store.create_incident(_classify(0.9, 0.9, 1.2, -50), _fallback_ai(), False)
    r = store.respond(inc["id"], "ok")
    assert r["ok"] is False  # already alerted


def test_no_response_auto_alerts_after_countdown():
    t = [0.0]
    store.set_clock(lambda: t[0])
    inc = store.create_incident(_classify(0.4, 0.5, 0.7, -28), _fallback_ai(), False)
    t[0] = 5.0
    store.tick()
    assert store.get_incident(inc["id"])["status"] == "awaiting_victim"
    t[0] = 11.0
    store.tick()
    assert store.get_incident(inc["id"])["status"] == "alerted"


# --- dispatch --------------------------------------------------------------

def test_escalation_moves_to_next_unit():
    t = [0.0]
    store.set_clock(lambda: t[0])
    inc = store.create_incident(_classify(0.9, 0.9, 1.2, -50), _fallback_ai(), False)
    first = inc["dispatch"]["offered_to"]
    t[0] = 21.0  # past 20s escalation
    store.tick()
    inc2 = store.get_incident(inc["id"])
    assert inc2["dispatch"]["offered_to"] != first
    assert first in inc2["dispatch"]["offer_tried"]


def test_escalation_to_control_room_when_units_exhausted():
    t = [0.0]
    store.set_clock(lambda: t[0])
    inc = store.create_incident(_classify(0.9, 0.9, 1.2, -50), _fallback_ai(), False)
    # Exhaust all units by advancing the clock past each 20s escalation.
    for i in range(10):
        t[0] += 21.0
        store.tick()
    inc2 = store.get_incident(inc["id"])
    assert inc2["dispatch"]["control_room"] is True


def test_accept_sets_eta_and_assigned():
    t = [0.0]
    store.set_clock(lambda: t[0])
    inc = store.create_incident(_classify(0.9, 0.9, 1.2, -50), _fallback_ai(), False)
    unit = inc["dispatch"]["offered_to"]
    r = store.accept(inc["id"], unit)
    assert r["ok"] is True
    inc2 = store.get_incident(inc["id"])
    assert inc2["status"] == "assigned"
    assert inc2["dispatch"]["accepted_unit"] == unit
    assert 4 <= inc2["dispatch"]["eta_min"] <= 12


def test_accept_wrong_unit_rejected():
    store.set_clock(lambda: 0.0)
    inc = store.create_incident(_classify(0.9, 0.9, 1.2, -50), _fallback_ai(), False)
    r = store.accept(inc["id"], "Ambulance 108-Z")
    assert r["ok"] is False


def test_invalid_status_transition_rejected():
    t = [0.0]
    store.set_clock(lambda: t[0])
    inc = store.create_incident(_classify(0.9, 0.9, 1.2, -50), _fallback_ai(), False)
    unit = inc["dispatch"]["offered_to"]
    store.accept(inc["id"], unit)
    r = store.set_status(inc["id"], "arrived")  # assigned -> arrived is invalid
    assert r["ok"] is False and r["error"]


def test_full_status_path_resolves_and_frees_unit():
    t = [0.0]
    store.set_clock(lambda: t[0])
    inc = store.create_incident(_classify(0.9, 0.9, 1.2, -50), _fallback_ai(), False)
    unit = inc["dispatch"]["offered_to"]
    store.accept(inc["id"], unit)
    for st in ("en_route", "arrived", "resolved"):
        assert store.set_status(inc["id"], st)["ok"] is True
    assert store.get_incident(inc["id"])["status"] == "resolved"
    u = next(x for x in __import__("dispatch").all_units() if x.name == unit)
    assert u.available is True


# --- sanitised view --------------------------------------------------------

def test_my_incident_excludes_evidence_and_confidence():
    store.set_clock(lambda: 0.0)
    store.create_incident(_classify(0.9, 0.9, 1.2, -50), _gemma_ai("critical"), True)
    out = store.my_incident()
    inc = out["incident"]
    assert inc is not None
    forbidden = {"evidence", "confidence", "uncertainty", "classification",
                 "readings", "safety", "ai_result", "impact_score",
                 "deceleration_g", "speed_kmh"}
    assert not (forbidden & set(inc.keys()))
    import json
    blob = json.dumps(inc).lower()
    for word in ("evidence", "confidence", "impact_score", "deceleration_g", "speed_kmh"):
        assert word not in blob, f"leaked field: {word}"


def test_my_incident_none_when_no_active():
    assert store.my_incident()["incident"] is None
