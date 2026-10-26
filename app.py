"""
AccidentGuard — Flask backend
=============================

Wires together the simulated event classification engine (accident_detection.py)
and the Gemma 4 service (gemma_service.py) and serves a responsive dashboard.

Routes:
    GET  /                          Dashboard page
    GET  /api/ai/status             Current AI status indicator
    GET  /api/ai/info               Model id + whether it is verified
    POST /api/ai/test               Make one real Gemma ping (proof of integration)
    POST /api/event/generate        Generate + classify a simulated event
    POST /api/event/<id>/summarize  Ask Gemma 4 to summarise an event
    GET  /api/history               Incident history (newest first)
    GET  /api/stats                 Total events + per-category counts
    POST /api/reset                 Clear history

The app stores events in memory (no database) — appropriate for a demo.
"""

from __future__ import annotations

import os
import threading

from flask import Flask, jsonify, render_template, request

# Load variables from .env (GEMINI_API_KEY, GEMMA_MODEL_ID, ...).
from dotenv import load_dotenv
load_dotenv()

import accident_detection as det
import gemma_service as gs


app = Flask(__name__)

# Simple in-memory history. A lock guards writes from the demo loop.
_history: list[dict] = []
_lock = threading.Lock()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _store_event(classification: det.EventClassification) -> dict:
    """Add a classified event to history and return the stored entry."""
    entry = {
        "classification": classification.to_dict(),
        "ai_payload": det.event_for_ai(classification),  # privacy-safe payload for Gemma
        "ai_result": None,
    }
    with _lock:
        _history.append(entry)
    return entry


def _find_entry(event_id: str) -> dict | None:
    with _lock:
        return next((e for e in _history
                     if e["classification"]["event_id"] == event_id), None)


# ---------------------------------------------------------------------------
# Page + AI status routes
# ---------------------------------------------------------------------------

@app.route("/")
def index():
    return render_template("index.html", model_info=gs.get_model_info())


@app.route("/api/ai/status")
def ai_status():
    return jsonify({
        "ai": gs.get_status(),
        "model": gs.get_model_info(),
        "configured": gs.is_configured(),
    })


@app.route("/api/ai/info")
def ai_info():
    return jsonify(gs.get_model_info())


@app.route("/api/ai/test", methods=["POST"])
def ai_test():
    """Make one real, tiny Gemma call to prove the integration works."""
    return jsonify(gs.quick_self_test())


# ---------------------------------------------------------------------------
# Event routes
# ---------------------------------------------------------------------------

@app.route("/api/event/generate", methods=["POST"])
def generate_event():
    """Generate and classify a simulated event. Optionally force a category."""
    data = request.get_json(silent=True) or {}
    requested = data.get("category")
    category = None
    if requested in {c.value for c in det.EventCategory}:
        category = det.EventCategory(requested)

    classification = det.generate_sample_event(category=category)
    entry = _store_event(classification)
    return jsonify({"event": entry["classification"]})


@app.route("/api/event/<event_id>/summarize", methods=["POST"])
def summarize_event(event_id: str):
    """Send an event to Gemma 4 and store the result on the event."""
    entry = _find_entry(event_id)
    if entry is None:
        return jsonify({"error": "Event not found", "event_id": event_id}), 404

    result = gs.generate_incident_summary(entry["ai_payload"])
    entry["ai_result"] = result
    return jsonify({"ai": result, "ai_status": gs.get_status()})


# ---------------------------------------------------------------------------
# History + stats
# ---------------------------------------------------------------------------

@app.route("/api/history")
def history():
    with _lock:
        items = [
            {
                "classification": e["classification"],
                "ai_result": e["ai_result"],
            }
            for e in reversed(_history)
        ]
    return jsonify({"events": items, "count": len(items)})


@app.route("/api/stats")
def stats():
    counts: dict[str, int] = {}
    with _lock:
        for e in _history:
            cat = e["classification"]["category"]
            counts[cat] = counts.get(cat, 0) + 1
        total = len(_history)

    # Ensure all three categories always appear (good for the UI).
    for c in det.EventCategory:
        counts.setdefault(c.value, 0)
    return jsonify({"total": total, "by_category": counts})


@app.route("/api/reset", methods=["POST"])
def reset():
    with _lock:
        _history.clear()
    return jsonify({"ok": True})


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    port = int(os.environ.get("FLASK_PORT", "5000"))
    debug = os.environ.get("FLASK_DEBUG", "True").lower() in ("true", "1", "yes")
    print("=" * 60)
    print(" AccidentGuard — starting dashboard")
    print(f" Open: http://127.0.0.1:{port}")
    print(f" AI configured: {gs.is_configured()}  |  Model: {gs.get_model_info()['model']}")
    if not gs.is_configured():
        print(" ! GEMINI_API_KEY not set — dashboard runs in non-AI fallback mode.")
    print("=" * 60)
    app.run(host="127.0.0.1", port=port, debug=debug)
