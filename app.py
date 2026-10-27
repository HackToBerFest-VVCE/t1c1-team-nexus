"""
AccidentGuard — AI-Assisted Accident Detection & Emergency Alert Platform
=========================================================================

Team Nexus · Team 11 · Hacktoberfest × VectorFlow 2026 · Challenge A: Best Use of Gemma 4

Pipeline:
  Sensor data → Python validation → Sensor fusion → False-alarm filter →
  Gemma 4 → Python safety layer → Victim check → Dispatch → Dashboard

Slogans: "Sensors detect. Gemma 4 interprets and communicates."
         "Gemma 4 reasons. Python decides."

SIMULATION ONLY — no real sensors, GPS, emergency calls or ambulances.
"""

from __future__ import annotations

import os

from flask import Flask, jsonify, render_template, request

# Load .env from an explicit path next to app.py with override=True.
from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env"),
            override=True)

import accident_detection as det
import gemma_service as gs
import incident_store as store
import sensor_fusion as sf


app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024   # 10 MB for photo uploads


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_SCENARIO_TO_CATEGORY = {
    "normal_driving":    det.EventCategory.NORMAL,
    "hard_braking":      det.EventCategory.SUSPICIOUS,
    "pothole":           det.EventCategory.SUSPICIOUS,
    "phone_drop":        det.EventCategory.NORMAL,
    "suspicious_event":  det.EventCategory.SUSPICIOUS,
    "possible_accident": det.EventCategory.POSSIBLE_ACCIDENT,
    "real_crash":        det.EventCategory.POSSIBLE_ACCIDENT,
}


def _validate_event(classification: det.EventClassification):
    r = classification.readings
    if not (0.0 <= r.impact_score <= 1.0):
        return False, "impact_score out of range (0..1)"
    if not (0.0 <= r.deceleration_g <= 2.0):
        return False, "deceleration_g out of range (0..2)"
    if not (0.0 <= r.speed_kmh <= 300.0):
        return False, "speed out of range"
    if not (-5.0 <= r.lane_deviation_m <= 5.0):
        return False, "lane deviation out of range"
    return True, "ok"


def _assess(classification: det.EventClassification,
            gemma_enabled: bool,
            fusion: dict,
            filter_res: dict):
    payload = det.event_for_ai(classification)
    if not gemma_enabled:
        fb = gs._fallback_summary(payload, "ai_disabled")
        fb["status"] = "disabled"
        fb["fallback_reason"] = "ai_disabled"
        fb["error"] = None
        return fb, False
    res = gs.generate_incident_summary(payload, fusion=fusion, filter_res=filter_res)
    return res, res.get("source") == "gemma"


# ---------------------------------------------------------------------------
# Pages
# ---------------------------------------------------------------------------

@app.route("/")
def index():
    return render_template("index.html", model_info=gs.get_model_info(),
                           toggles=store.get_toggles())


@app.route("/victim")
def victim_page():
    return render_template("victim.html", model_info=gs.get_model_info())


@app.route("/ambulance")
def ambulance_page():
    return render_template("ambulance.html", model_info=gs.get_model_info())


# ---------------------------------------------------------------------------
# AI status + connection check
# ---------------------------------------------------------------------------

@app.route("/api/ai/status")
def ai_status():
    return jsonify({
        "ai": gs.get_status(),
        "model": gs.get_model_info(),
        "configured": gs.is_configured(),
        "toggles": store.get_toggles(),
    })


@app.route("/api/ai/info")
def ai_info():
    return jsonify(gs.get_model_info())


@app.route("/api/ai-check")
def ai_check():
    return jsonify(gs.check_connection())


@app.route("/api/ai/test", methods=["POST"])
def ai_test():
    return jsonify(gs.quick_self_test())


# ---------------------------------------------------------------------------
# Event / incident pipeline
# ---------------------------------------------------------------------------

@app.route("/api/event/generate", methods=["POST"])
def generate_event():
    """
    Full pipeline: sensor streams → fusion → false-alarm filter →
    validate → classify → Gemma 4 → safety layer → incident.
    """
    data = request.get_json(silent=True) or {}
    scenario = data.get("scenario") or data.get("category") or "random"

    # 1. Generate dual-stream sensor data
    phone, vehicle = sf.generate_streams(scenario)
    legacy = sf.streams_to_legacy_readings(phone, vehicle)

    # 2. Classify with the rule engine
    category_override = _SCENARIO_TO_CATEGORY.get(scenario)
    if category_override is None:
        # try as raw category name
        try:
            category_override = det.EventCategory(scenario)
        except ValueError:
            category_override = None
    if category_override is not None:
        classification = det.generate_sample_event(category=category_override)
        # Overlay location from stream so UI stays consistent
        classification.readings.location_label = phone.location_label
    else:
        classification = det.generate_from_streams(phone, vehicle, legacy)

    # 3. Sensor fusion
    fusion = sf.fuse(phone, vehicle)

    # 4. False-alarm filter
    filter_result = sf.false_alarm_filter(phone, vehicle, fusion)

    # 5. Validation
    ok, reason = _validate_event(classification)
    if not ok:
        return jsonify({"error": reason, "event": classification.to_dict()}), 400

    # 6. Gemma 4 assessment (or fallback)
    toggles = store.get_toggles()
    gemma_enabled = toggles["ai_enabled"] and gs.is_configured()
    ai_result, used_gemma = _assess(
        classification, gemma_enabled,
        fusion=fusion.to_dict(), filter_res=filter_result.to_dict()
    )

    # 7. Create incident (safety layer runs inside)
    incident = store.create_incident(
        classification, ai_result, used_gemma,
        fusion=fusion.to_dict(),
        filter_result=filter_result.to_dict(),
        phone_stream=phone.to_dict(),
        vehicle_stream=vehicle.to_dict(),
    )
    store.tick()

    return jsonify({
        "event": classification.to_dict(),
        "incident": store.get_incident(incident["id"]),
        "ai": ai_result,
        "ai_status": gs.get_status(),
        "fusion": fusion.to_dict(),
        "filter_result": filter_result.to_dict(),
        "phone_stream": phone.to_dict(),
        "vehicle_stream": vehicle.to_dict(),
    })


@app.route("/api/event/<event_id>/summarize", methods=["POST"])
def summarize_event(event_id: str):
    store.tick()
    incident = next((i for i in store.list_incidents()
                     if i["classification"]["event_id"] == event_id), None)
    if incident is None:
        return jsonify({"error": "Event not found", "event_id": event_id}), 404
    toggles = store.get_toggles()
    gemma_enabled = toggles["ai_enabled"] and gs.is_configured()
    payload = {
        "event_id": event_id,
        "timestamp": incident["classification"]["timestamp"],
        "category": incident["classification"]["category"],
        "severity": incident["classification"]["severity"],
        "simulated_readings": incident["classification"]["readings"],
        "rule_engine_notes": incident["classification"]["rule_explanation"],
    }
    if not gemma_enabled:
        ai_result = gs._fallback_summary(payload, "ai_disabled")
        ai_result["status"] = "disabled"
        ai_result["error"] = None
    else:
        ai_result = gs.generate_incident_summary(
            payload,
            fusion=incident.get("fusion", {}),
            filter_res=incident.get("filter_result", {}),
        )
    return jsonify({"ai": ai_result, "ai_status": gs.get_status()})


# ---------------------------------------------------------------------------
# Photo analysis
# ---------------------------------------------------------------------------

@app.route("/api/incidents/<incident_id>/photo", methods=["POST"])
def upload_photo(incident_id: str):
    """
    Analyse a crash photo with Gemma 4 vision (optional, simulated bystander).
    Accepts multipart/form-data with field "photo", or JSON with base64 "image_data"
    and "mime_type".
    """
    store.tick()
    inc = store.get_incident(incident_id)
    if inc is None:
        return jsonify({"error": "Incident not found"}), 404

    # Extract image bytes
    image_bytes = None
    mime_type = "image/jpeg"

    if request.content_type and "multipart" in request.content_type:
        f = request.files.get("photo")
        if f:
            image_bytes = f.read()
            mime_type = f.mimetype or "image/jpeg"
    else:
        body = request.get_json(silent=True) or {}
        import base64 as _b64
        raw = body.get("image_data", "")
        if raw:
            try:
                image_bytes = _b64.b64decode(raw)
            except Exception:
                pass
        mime_type = body.get("mime_type", "image/jpeg")

    if not image_bytes:
        return jsonify({"error": "No image data received"}), 400

    result = gs.analyse_crash_photo(image_bytes, mime_type)
    updated = store.add_scene_analysis(incident_id, result)
    return jsonify({"scene_analysis": result,
                    "incident": updated})


# ---------------------------------------------------------------------------
# Incidents + units + victim + dispatch
# ---------------------------------------------------------------------------

@app.route("/api/incidents")
def api_incidents():
    store.tick()
    return jsonify({"incidents": store.list_incidents(), "now": store.now()})


@app.route("/api/my-incident")
def api_my_incident():
    store.tick()
    return jsonify(store.my_incident())


@app.route("/api/incidents/<incident_id>/respond", methods=["POST"])
def api_respond(incident_id: str):
    store.tick()
    data = request.get_json(silent=True) or {}
    result = store.respond(incident_id, data.get("answer"))
    return jsonify(result), 200 if result.get("ok") else 400


@app.route("/api/incidents/<incident_id>/accept", methods=["POST"])
def api_accept(incident_id: str):
    store.tick()
    data = request.get_json(silent=True) or {}
    result = store.accept(incident_id, data.get("unit"))
    return jsonify(result), 200 if result.get("ok") else 400


@app.route("/api/incidents/<incident_id>/status", methods=["POST"])
def api_status(incident_id: str):
    store.tick()
    data = request.get_json(silent=True) or {}
    result = store.set_status(incident_id, data.get("status"))
    return jsonify(result), 200 if result.get("ok") else 400


@app.route("/api/units")
def api_units():
    store.tick()
    return jsonify({"units": store.list_units(incident_id=request.args.get("incident"))})


@app.route("/api/toggle-ai", methods=["POST"])
def api_toggle_ai():
    data = request.get_json(silent=True) or {}
    if "ai_enabled" in data:
        store.set_ai_enabled(bool(data["ai_enabled"]))
    if "auto_accept" in data:
        store.set_auto_accept(bool(data["auto_accept"]))
    if "victim_auto_respond" in data:
        store.set_victim_auto_respond(bool(data["victim_auto_respond"]))
    return jsonify({"toggles": store.get_toggles(), "ai": gs.get_status()})


# ---------------------------------------------------------------------------
# History + stats
# ---------------------------------------------------------------------------

@app.route("/api/history")
def history():
    store.tick()
    incidents = store.list_incidents()
    items = [{
        "classification": i["classification"],
        "ai_result": i["ai_result"],
        "incident": {"id": i["id"], "status": i["status"], "severity": i["severity"]},
        "filter_result": i.get("filter_result", {}),
        "fusion": i.get("fusion", {}),
    } for i in incidents]
    return jsonify({"events": items, "count": len(items)})


@app.route("/api/stats")
def stats():
    store.tick()
    counts = {c.value: 0 for c in det.EventCategory}
    rejected = 0
    for i in store.list_incidents():
        cat = i["classification"]["category"]
        counts[cat] = counts.get(cat, 0) + 1
        if not i.get("filter_result", {}).get("passed", True):
            rejected += 1
    total = sum(counts.values())
    return jsonify({"total": total, "by_category": counts, "rejected": rejected})


@app.route("/api/reset", methods=["POST"])
def reset():
    store.reset()
    return jsonify({"ok": True})


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    port = int(os.environ.get("FLASK_PORT", "5000"))
    debug = os.environ.get("FLASK_DEBUG", "True").lower() in ("true", "1", "yes")
    print("=" * 62)
    print(" AccidentGuard — AI-Assisted Accident Detection & Emergency Alert")
    print(" Team Nexus · Team 11 · Hacktoberfest × VectorFlow 2026")
    print(f" http://127.0.0.1:{port}  |  /victim  |  /ambulance")
    print(f" AI: {'configured' if gs.is_configured() else 'NOT configured (fallback mode)'}  "
          f"| Model: {gs.get_model_info()['model']}")
    print("=" * 62)
    app.run(host="127.0.0.1", port=port, debug=debug)
