"""
API tests via the Flask test client, with NO API key (forced). These confirm the
full flow works end-to-end on the fallback path and that the victim view never
leaks evidence/confidence/sensor numbers.
"""
import json

import gemma_service as gs
import dispatch
import incident_store as store

import app as A


def _client(monkeypatch):
    # Force the no-key path deterministically, even if the host has a .env key.
    monkeypatch.setattr(gs, "_get_api_key", lambda: None)
    store.reset()
    return A.app.test_client()


def test_ai_check_no_key(monkeypatch):
    c = _client(monkeypatch)
    r = c.get("/api/ai-check")
    d = r.get_json()
    assert d["status"] == "not_configured"
    assert d["reason"] == "no key found"


def test_generate_without_key_uses_fallback(monkeypatch):
    c = _client(monkeypatch)
    r = c.post("/api/event/generate", json={"category": "possible_accident"})
    d = r.get_json()
    assert r.status_code == 200
    assert d["ai"]["source"] == "fallback"
    assert d["ai"]["used_fallback"] is True
    assert d["incident"]["status"] == "alerted"  # critical -> alert immediately


def test_my_incident_api_excludes_sensitive_fields(monkeypatch):
    c = _client(monkeypatch)
    c.post("/api/event/generate", json={"category": "possible_accident"})
    r = c.get("/api/my-incident")
    inc = r.get_json()["incident"]
    assert inc is not None
    forbidden = {"evidence", "confidence", "uncertainty", "classification",
                 "readings", "safety", "ai_result"}
    assert not (forbidden & set(inc.keys()))
    blob = json.dumps(inc).lower()
    for w in ("evidence", "confidence", "impact_score", "deceleration_g", "speed_kmh"):
        assert w not in blob


def test_full_flow_no_key_critical(monkeypatch):
    c = _client(monkeypatch)
    g = c.post("/api/event/generate", json={"category": "possible_accident"}).get_json()
    inc_id = g["incident"]["id"]
    offered = g["incident"]["dispatch"]["offered_to"]
    assert offered is not None

    # accept
    a = c.post(f"/api/incidents/{inc_id}/accept", json={"unit": offered}).get_json()
    assert a["ok"] is True

    # status path
    for st in ("en_route", "arrived", "resolved"):
        r = c.post(f"/api/incidents/{inc_id}/status", json={"status": st}).get_json()
        assert r["ok"] is True, r.get("error")
    final = c.get(f"/api/incidents").get_json()
    inc = next(i for i in final["incidents"] if i["id"] == inc_id)
    assert inc["status"] == "resolved"

    # unit freed
    units = c.get("/api/units").get_json()["units"]
    u = next(x for x in units if x["name"] == offered)
    assert u["available"] is True


def test_victim_help_alerts_and_dispatches(monkeypatch):
    """Victim taps I need help → event becomes alerted and dispatch fires."""
    c = _client(monkeypatch)
    # Create a medium-severity incident directly in the store so we control the status.
    import accident_detection as _det
    _cls = _det.generate_sample_event(category=_det.EventCategory.SUSPICIOUS, seed=99)
    # Force medium severity by clamping readings
    _cls.severity = _det.Severity.MEDIUM
    _ai = gs._fallback_summary(
        {"category": "suspicious_event", "severity": "medium", "simulated_readings": {}},
        "not_configured",
    )
    # Patch safety layer to always return medium so incident goes to awaiting_victim
    import incident_store as _st
    import safety_layer as _sl
    original_decide = _sl.decide_final_severity
    _sl.decide_final_severity = lambda *a, **kw: {
        "final_severity": "medium", "rule_severity": "medium",
        "gemma_severity": None, "disagreement": False,
        "note": "medium (test override)", "alert_needed": True,
    }
    try:
        inc_raw = _st.create_incident(_cls, _ai, False)
        _st.tick()
    finally:
        _sl.decide_final_severity = original_decide

    assert _st.get_incident(inc_raw["id"])["status"] == "awaiting_victim"
    inc_id = inc_raw["id"]
    r = c.post(f"/api/incidents/{inc_id}/respond", json={"answer": "help"}).get_json()
    assert r["ok"] is True
    inc = c.get("/api/incidents").get_json()["incidents"][0]
    assert inc["status"] == "alerted"
    assert inc["dispatch"]["offered_to"] is not None


def test_victim_ok_cancels(monkeypatch):
    """Victim taps I'M SAFE → incident becomes cancelled."""
    c = _client(monkeypatch)
    import accident_detection as _det
    import incident_store as _st
    import safety_layer as _sl
    _cls = _det.generate_sample_event(category=_det.EventCategory.SUSPICIOUS, seed=77)
    _cls.severity = _det.Severity.MEDIUM
    _ai = gs._fallback_summary(
        {"category": "suspicious_event", "severity": "medium", "simulated_readings": {}},
        "not_configured",
    )
    orig = _sl.decide_final_severity
    _sl.decide_final_severity = lambda *a, **kw: {
        "final_severity": "medium", "rule_severity": "medium",
        "gemma_severity": None, "disagreement": False,
        "note": "medium (test)", "alert_needed": True,
    }
    try:
        inc_raw = _st.create_incident(_cls, _ai, False)
    finally:
        _sl.decide_final_severity = orig
    inc_id = inc_raw["id"]
    assert _st.get_incident(inc_id)["status"] == "awaiting_victim"
    r = c.post(f"/api/incidents/{inc_id}/respond", json={"answer": "ok"}).get_json()
    assert r["ok"] is True
    inc = c.get("/api/incidents").get_json()["incidents"][0]
    assert inc["status"] == "cancelled"


def test_toggle_ai_off(monkeypatch):
    c = _client(monkeypatch)
    r = c.post("/api/toggle-ai", json={"ai_enabled": False}).get_json()
    assert r["toggles"]["ai_enabled"] is False
    g = c.post("/api/event/generate", json={"category": "possible_accident"}).get_json()
    assert g["ai"]["source"] == "fallback"


def test_toggle_auto_accept(monkeypatch):
    c = _client(monkeypatch)
    c.post("/api/toggle-ai", json={"auto_accept": True})
    g = c.post("/api/event/generate", json={"category": "possible_accident"}).get_json()
    inc_id = g["incident"]["id"]
    # auto-accept happens on tick
    c.get("/api/incidents")
    inc = c.get(f"/api/incidents").get_json()["incidents"][0]
    # critical -> alerted -> auto-accept -> assigned
    assert inc["status"] in ("assigned", "en_route", "arrived", "resolved")
    assert inc["dispatch"]["accepted_unit"] is not None


def test_units_list(monkeypatch):
    c = _client(monkeypatch)
    u = c.get("/api/units").get_json()["units"]
    assert len(u) == 5
    assert all("name" in x and "available" in x for x in u)


def test_reset(monkeypatch):
    c = _client(monkeypatch)
    c.post("/api/event/generate", json={"category": "possible_accident"})
    assert c.post("/api/reset").get_json()["ok"] is True
    assert c.get("/api/incidents").get_json()["incidents"] == []


def test_incidents_response_includes_now_field(monkeypatch):
    """ambulance.js uses 'now' from /api/incidents for escalation countdowns."""
    c = _client(monkeypatch)
    d = c.get("/api/incidents").get_json()
    assert "now" in d
    assert isinstance(d["now"], (int, float))
    assert d["now"] > 0


def test_incidents_location_label_is_present_and_simulated(monkeypatch):
    """Regression: ambulance.js renderList() was calling esc(i.location_label) which
    crashed with TypeError when a local variable named 'esc' shadowed the esc() function.
    This test verifies the data path is intact (the JS fix is in static/ambulance.js)."""
    c = _client(monkeypatch)
    c.post("/api/event/generate", json={"category": "possible_accident"})
    incidents = c.get("/api/incidents").get_json()["incidents"]
    assert len(incidents) >= 1
    loc = incidents[0]["location_label"]
    assert isinstance(loc, str) and len(loc) > 0
    # Location labels must carry the "Simulated" marker
    assert "Simulated" in loc


def test_units_with_incident_include_distance(monkeypatch):
    """ambulance.js requests /api/units?incident=<id> to get distances for the SVG map."""
    c = _client(monkeypatch)
    g = c.post("/api/event/generate", json={"category": "possible_accident"}).get_json()
    inc_id = g["incident"]["id"]
    units = c.get(f"/api/units?incident={inc_id}").get_json()["units"]
    assert len(units) == 5
    # All units should have a distance_km when incident is provided
    assert all("distance_km" in u for u in units)
    # Distances must be positive numbers
    assert all(isinstance(u["distance_km"], float) and u["distance_km"] >= 0 for u in units)
