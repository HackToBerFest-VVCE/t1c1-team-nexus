"""
AccidentGuard — Gemma 4 integration service
==========================================

Verified model: gemma-4-26b-a4b-it (Gemini API, google-genai SDK).
Docs: https://ai.google.dev/gemma/docs/core/gemma_on_gemini_api

Two real calls:
  Call 1 — incident assessment (JSON: severity, confidence, evidence, uncertainty,
            action, alert_en/kn/hi, responder_briefing).
  Call 2 — dispatch briefing update (JSON: responder_briefing, updated_action).

Optional third call: crash photo analysis (inline image, JSON result).

Key loading:
  load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env"),
              override=True)
  Key is stripped of whitespace and quotes. Never logged, never sent to browser.

Python (safety_layer.py) makes the FINAL severity decision — never Gemma.
"""

from __future__ import annotations

import base64
import json
import os
import re
from typing import Optional

from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env"),
            override=True)

from google import genai
from google.genai import types
from google.genai import errors as genai_errors


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

VERIFIED_MODELS = ("gemma-4-26b-a4b-it", "gemma-4-31b-it")
DEFAULT_MODEL = "gemma-4-26b-a4b-it"

ENV_KEY = "GEMINI_API_KEY"
ENV_MODEL = "GEMMA_MODEL_ID"
ENV_TIMEOUT = "GEMMA_TIMEOUT"
DEFAULT_TIMEOUT_MS = 20_000   # 20 s for real calls; check endpoint allows 15 s


def _get_api_key() -> Optional[str]:
    """Read key from env, strip whitespace and quotes."""
    raw = os.environ.get(ENV_KEY, "")
    return raw.strip().strip('"').strip("'").strip() or None


def _get_model_id() -> str:
    model = os.environ.get(ENV_MODEL, "").strip() or DEFAULT_MODEL
    return model if model in VERIFIED_MODELS else DEFAULT_MODEL


def _get_timeout_ms() -> int:
    try:
        s = int(os.environ.get(ENV_TIMEOUT, "20"))
        return (s if s > 0 else 20) * 1000
    except ValueError:
        return DEFAULT_TIMEOUT_MS


def _make_client(api_key: str):
    return genai.Client(
        api_key=api_key,
        http_options=types.HttpOptions(timeout=_get_timeout_ms()),
    )


# ---------------------------------------------------------------------------
# Status tracking
# ---------------------------------------------------------------------------

_last_status: dict = {
    "state": "not_configured",
    "reason": None,
    "detail": "No API call made yet.",
    "model": DEFAULT_MODEL,
    "verified": True,
    "configured": False,
}


def _set_status(state, reason, model, configured, detail=None):
    _last_status.update(state=state, reason=reason, model=model,
                        configured=configured,
                        detail=detail or reason or state)


def is_configured() -> bool:
    return _get_api_key() is not None


def get_status() -> dict:
    _last_status["configured"] = is_configured()
    if not is_configured() and _last_status["state"] == "connected":
        _last_status.update(state="not_configured", reason="no key found",
                            detail="API key is missing.")
    return dict(_last_status)


def get_model_info() -> dict:
    return {
        "model": _get_model_id(),
        "verified_supported": _get_model_id() in VERIFIED_MODELS,
        "verified_models": list(VERIFIED_MODELS),
        "configured": is_configured(),
    }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _sanitize(msg: str) -> str:
    key = _get_api_key()
    if key and key in msg:
        msg = msg.replace(key, "***REDACTED***")
    return re.sub(r"(key=)[^\s&]+", r"\1***REDACTED***", msg)


def _classify_error(exc) -> str:
    msg = _sanitize(str(exc)).lower()
    if "timeout" in msg or "timed out" in msg:
        return "timeout or quota"
    if "quota" in msg or "rate" in msg or "resourceexhausted" in msg or "429" in msg:
        return "timeout or quota"
    if "not found" in msg or "model not found" in msg or "404" in msg:
        return "model not found"
    if ("api key" in msg or "not valid" in msg or "invalid" in msg or
            "permission" in msg or "forbidden" in msg or "denied" in msg or
            "unauth" in msg):
        return "key rejected"
    return "api error"


def _strip_fences(text: str) -> str:
    t = text.strip()
    if t.startswith("```"):
        nl = t.find("\n")
        t = t[nl + 1:] if nl != -1 else ""
        if t.endswith("```"):
            t = t[: t.rfind("```")]
    return t.strip()


def _parse_json(text: str) -> dict:
    """Strip fences, then extract the first {...} block and parse it."""
    t = _strip_fences(text)
    start = t.find("{")
    end = t.rfind("}")
    if start == -1 or end <= start:
        raise ValueError("No JSON object found in model response.")
    try:
        return json.loads(t[start: end + 1])
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON from model: {exc.msg}")


_VALID_SEV = {"low", "medium", "high", "critical"}


def _norm_sev(v) -> str:
    s = str(v or "low").lower().strip()
    return s if s in _VALID_SEV else "low"


def _normalize_assessment(obj: dict) -> dict:
    sev = _norm_sev(obj.get("severity", "low"))
    try:
        conf = max(0.0, min(1.0, float(obj.get("confidence", 0.5))))
    except (TypeError, ValueError):
        conf = 0.5
    evidence = obj.get("evidence", [])
    if not isinstance(evidence, list):
        evidence = [str(evidence)] if evidence else []
    evidence = [str(e).strip() for e in evidence if str(e).strip()]
    briefing = obj.get("responder_briefing", [])
    if not isinstance(briefing, list):
        briefing = [str(briefing)] if briefing else []
    briefing = [str(b).strip() for b in briefing if str(b).strip()]
    action = str(obj.get("action", "") or "").strip()
    uncertainty = str(obj.get("uncertainty", "") or "").strip()
    alerts = {k: str(obj.get(k, "") or "").strip() for k in ("alert_en", "alert_kn", "alert_hi")}
    summary = action or f"Assessed severity: {sev}."
    return {
        "severity_opinion": sev, "confidence": conf, "evidence": evidence,
        "uncertainty": uncertainty, "action": action,
        "alert_en": alerts["alert_en"], "alert_kn": alerts["alert_kn"],
        "alert_hi": alerts["alert_hi"], "responder_briefing": briefing,
        "summary": summary, "attention_points": evidence, "recommendations": briefing,
    }


# ---------------------------------------------------------------------------
# System instruction (placed at start of user prompt for Gemma on Gemini API)
# ---------------------------------------------------------------------------

_SYSTEM_HEADER = (
    "You are AccidentGuard's safety assistant for a SIMULATION PROTOTYPE. "
    "The data is SIMULATED — there are NO real sensors, NO real GPS, NO real accidents. "
    "Rules you MUST follow:\n"
    "1. Every reading is illustrative simulation data — do NOT invent values.\n"
    "2. Do NOT claim certainty about a real accident; use cautious language.\n"
    "3. You are NOT a replacement for real emergency professionals.\n"
    "4. Output ONLY a single JSON object with no markdown, no code fences, no prose.\n\n"
)


# ---------------------------------------------------------------------------
# Call 1 — Incident assessment
# ---------------------------------------------------------------------------

def _build_assessment_prompt(payload: dict, fusion: dict, filter_res: dict) -> str:
    r = payload.get("simulated_readings", {})
    fusion_summary = (
        f"Fusion: {fusion.get('agreeing_signals', 0)} of 6 signals agree. "
        f"Signals: {', '.join(fusion.get('signals_detail', [])) or 'none'}. "
        f"Has vehicle/stop signal: {fusion.get('has_vehicle_or_stop', False)}."
    )
    filter_summary = (
        f"False-alarm filter: {'PASSED' if filter_res.get('passed') else 'REJECTED — ' + filter_res.get('reason', '')}."
    )
    lines = [
        _SYSTEM_HEADER,
        "Return ONLY a JSON object with these exact keys:",
        '{"severity":"low|medium|high|critical","confidence":0.0-1.0,'
        '"evidence":["string"],"uncertainty":"string","action":"string",'
        '"alert_en":"first-aid tip","alert_kn":"tip in Kannada","alert_hi":"tip in Hindi",'
        '"responder_briefing":["line1","line2","line3"]}',
        "",
        "AccidentGuard simulated event review request.",
        f"Event ID: {payload.get('event_id', 'unknown')}",
        f"Timestamp: {payload.get('timestamp', 'unknown')}",
        f"Rule-engine classification: {payload.get('category', 'unknown')}",
        f"Rule-engine severity: {payload.get('severity', 'unknown')}",
        f"Location (simulated): {r.get('location_label', 'unknown')}",
        "",
        "Simulated readings:",
        f"  speed: {r.get('speed_kmh', 'N/A')} km/h",
        f"  speed change: {r.get('speed_delta_kmh', 'N/A')} km/h",
        f"  deceleration: {r.get('deceleration_g', 'N/A')} g",
        f"  impact score (0-1): {r.get('impact_score', 'N/A')}",
        f"  lane deviation: {r.get('lane_deviation_m', 'N/A')} m",
        "",
        fusion_summary,
        filter_summary,
        f"Rule-engine notes: {payload.get('rule_engine_notes', '')}",
        "",
        "Return ONLY the JSON object.",
    ]
    return "\n".join(lines)


def generate_incident_summary(payload: dict,
                               fusion: dict | None = None,
                               filter_res: dict | None = None) -> dict:
    """Call 1: assess the incident. Returns structured dict."""
    if not isinstance(payload, dict):
        payload = {}
    fusion = fusion or {}
    filter_res = filter_res or {"passed": True, "reason": "", "filter_label": "passed"}

    api_key = _get_api_key()
    model = _get_model_id()

    if not api_key:
        _set_status("not_configured", "no key found", model, False,
                    "GEMINI_API_KEY not set.")
        return _fallback_summary(payload, "not_configured")

    prompt = _build_assessment_prompt(payload, fusion, filter_res)
    try:
        client = _make_client(api_key)
        response = client.models.generate_content(
            model=model,
            contents=prompt,
            config=types.GenerateContentConfig(
                temperature=0.2,
                thinking_config=types.ThinkingConfig(thinking_level="minimal"),
            ),
        )
        text = response.text
        if not text or not text.strip():
            raise ValueError("Gemma returned an empty response.")
        norm = _normalize_assessment(_parse_json(text))
        _set_status("connected", None, model, True,
                    f"Real response received from {model}.")
        return {
            "status": "connected", "source": "gemma", "model": model,
            "used_fallback": False, "error": None, **norm,
        }
    except (genai_errors.APIError, genai_errors.ClientError,
            genai_errors.ServerError) as exc:
        reason = _classify_error(exc)
        safe = _sanitize(str(exc))
        _set_status("error", reason, model, True, safe)
        r = _fallback_summary(payload, reason)
        r["error"] = safe
        return r
    except ValueError as exc:
        safe = _sanitize(str(exc))
        _set_status("error", "invalid JSON", model, True,
                    f"Model did not return valid JSON: {safe}")
        r = _fallback_summary(payload, "invalid JSON")
        r["error"] = f"Model did not return valid JSON: {safe}"
        return r
    except Exception as exc:
        reason = _classify_error(exc)
        safe = _sanitize(str(exc))
        _set_status("error", reason, model, True, safe)
        r = _fallback_summary(payload, reason)
        r["error"] = safe
        return r


# ---------------------------------------------------------------------------
# Call 2 — Dispatch briefing update
# ---------------------------------------------------------------------------

def generate_dispatch_briefing(assessment: dict,
                                victim_answer: Optional[str],
                                seconds_without_response: float,
                                nearest_unit: str,
                                eta_min: int) -> dict:
    """Call 2: update the 3-line responder briefing at dispatch time."""
    api_key = _get_api_key()
    model = _get_model_id()

    victim_status = (
        f"Victim answered: {victim_answer}" if victim_answer
        else f"No victim response for {int(seconds_without_response)}s."
    )
    prompt = (
        f"{_SYSTEM_HEADER}"
        "Return ONLY a JSON object:\n"
        '{"responder_briefing":["line1","line2","line3"],"updated_action":"string"}\n\n'
        "Update the responder briefing for this dispatch event (SIMULATED):\n"
        f"Severity: {assessment.get('severity_opinion', 'unknown')}\n"
        f"Confidence: {assessment.get('confidence', 'N/A')}\n"
        f"Action: {assessment.get('action', '')}\n"
        f"Original briefing: {assessment.get('responder_briefing', [])}\n"
        f"Victim status: {victim_status}\n"
        f"Unit dispatched: {nearest_unit} (simulated), ETA {eta_min} min.\n"
        "Keep briefing to 3 short lines. Output ONLY the JSON object."
    )

    if not api_key:
        return {"source": "fallback", "responder_briefing": assessment.get("responder_briefing", []),
                "updated_action": assessment.get("action", ""),
                "note": "Non-AI fallback (no key)"}

    try:
        client = _make_client(api_key)
        response = client.models.generate_content(
            model=model, contents=prompt,
            config=types.GenerateContentConfig(
                temperature=0.2,
                thinking_config=types.ThinkingConfig(thinking_level="minimal"),
            ),
        )
        obj = _parse_json(response.text or "")
        briefing = obj.get("responder_briefing", assessment.get("responder_briefing", []))
        if not isinstance(briefing, list):
            briefing = assessment.get("responder_briefing", [])
        return {
            "source": "gemma",
            "responder_briefing": [str(b).strip() for b in briefing if str(b).strip()],
            "updated_action": str(obj.get("updated_action", "") or "").strip(),
            "note": None,
        }
    except Exception as exc:
        return {
            "source": "fallback",
            "responder_briefing": assessment.get("responder_briefing", []),
            "updated_action": assessment.get("action", ""),
            "note": f"Briefing reused from Call 1 (Call 2 failed: {_sanitize(str(exc))[:80]})",
        }


# ---------------------------------------------------------------------------
# Call 3 — Crash photo analysis (optional)
# ---------------------------------------------------------------------------

def analyse_crash_photo(image_bytes: bytes, mime_type: str) -> dict:
    """
    Attempt to analyse a crash photo using Gemma 4 vision.
    Returns a structured dict. If the model doesn't support images, returns
    a clearly-labelled 'scene_analysis_unavailable' result.
    """
    api_key = _get_api_key()
    model = _get_model_id()

    if not api_key:
        return _photo_unavailable("No API key configured.")

    if not image_bytes:
        return _photo_unavailable("No image data provided.")

    prompt_text = (
        f"{_SYSTEM_HEADER}"
        "You are viewing an image submitted by a simulated bystander at an accident scene. "
        "This is a SIMULATION — treat the image as illustrative only.\n"
        "Return ONLY a JSON object:\n"
        '{"vehicles":0-9,"visible_injuries":"yes|no|unclear",'
        '"fire_or_fuel_risk":"yes|no|unclear","road_blocked":"yes|no|unclear",'
        '"notes":"short string","confidence":0.0-1.0,"unclear":false}\n'
        "If you cannot see the scene clearly, set unclear=true and use 'unclear' for the fields. "
        "Do NOT invent details. This is NOT a medical diagnosis. "
        "Output ONLY the JSON object."
    )

    try:
        client = _make_client(api_key)
        b64 = base64.b64encode(image_bytes).decode("utf-8")
        image_part = types.Part.from_bytes(data=base64.b64decode(b64), mime_type=mime_type)
        response = client.models.generate_content(
            model=model,
            contents=[prompt_text, image_part],
            config=types.GenerateContentConfig(
                temperature=0.1,
                thinking_config=types.ThinkingConfig(thinking_level="minimal"),
            ),
        )
        text = response.text or ""
        obj = _parse_json(text)
        return {
            "available": True,
            "source": "gemma",
            "vehicles": int(obj.get("vehicles", 0)),
            "visible_injuries": str(obj.get("visible_injuries", "unclear")),
            "fire_or_fuel_risk": str(obj.get("fire_or_fuel_risk", "unclear")),
            "road_blocked": str(obj.get("road_blocked", "unclear")),
            "notes": str(obj.get("notes", "") or "").strip(),
            "confidence": float(obj.get("confidence", 0.5)),
            "unclear": bool(obj.get("unclear", False)),
            "error": None,
        }
    except Exception as exc:
        safe = _sanitize(str(exc))
        low = safe.lower()
        # Detect unsupported image for this model gracefully
        if any(w in low for w in ("image", "vision", "multimodal", "content type",
                                   "invalid", "unsupported", "mime")):
            return _photo_unavailable(f"Scene analysis unavailable for this model. ({safe[:80]})")
        return _photo_unavailable(f"Photo analysis failed: {safe[:120]}")


def _photo_unavailable(reason: str) -> dict:
    return {
        "available": False,
        "source": "unavailable",
        "vehicles": None, "visible_injuries": None,
        "fire_or_fuel_risk": None, "road_blocked": None,
        "notes": reason, "confidence": None, "unclear": True,
        "error": reason,
    }


# ---------------------------------------------------------------------------
# Fallback (NON-AI) — clearly labelled, never called Gemma
# ---------------------------------------------------------------------------

_FALLBACK_ALERTS = {
    "possible_accident": {
        "en": "If hurt, stay still and call for help. Do not move unless in danger.",
        "kn": "ಗಾಯವಾಗಿದ್ದರೆ ಆಚೆ ಕರೆದು ಸಹಾಯ ಪಡೆಯಿರಿ. ಅಪಾಯವಿಲ್ಲದಿದ್ದರೆ ಆಚೆ ಚಲಿಸಬೇಡಿ.",
        "hi": "चोट लगी हो तो सहायता बुलाएँ। खतरा न हो तो हिलें नहीं।",
    },
    "suspicious_event": {
        "en": "Slow down and check your surroundings. Pull over if something feels wrong.",
        "kn": "ವೇಗ ಕಡಿಮೆ ಮಾಡಿ, ಸುತ್ತ ಗಮನಿಸಿ. ಏನಾದರೂ ತಪ್ಪಿದ್ದರೆ ಬದಿಗೆ ನಿಲ್ಲಿಸಿ.",
        "hi": "रफ़्तार घटाएँ और ध्यान दें। ग़लती महसूस हो तो बाज़ू में खड़े करें।",
    },
    "normal_driving": {
        "en": "No action needed — driving looks normal.",
        "kn": "ಯಾವುದೇ ಕ್ರಮ ಬೇಡ — ಚಾಲನೆ ಸಾಮಾನ್ಯ.",
        "hi": "कोई कार्रवाई आवश्यक नहीं — ड्राइविंग सामान्य।",
    },
}


def _fallback_summary(payload: dict, reason: str) -> dict:
    cat = payload.get("category", "normal_driving")
    sev = payload.get("severity", "low")
    readings = payload.get("simulated_readings", {})
    alerts = _FALLBACK_ALERTS.get(cat, _FALLBACK_ALERTS["normal_driving"])

    if cat == "possible_accident":
        action = ("Several simulated readings show hard braking, high impact and "
                  "large lane deviation. In a real vehicle this would warrant immediate "
                  "attention — but this is simulated data only.")
        briefing = [
            "If real, stop safely and check for injury.",
            "Contact actual emergency services if anyone may be hurt.",
            "Do not move an injured person unless in immediate danger.",
        ]
    elif cat == "suspicious_event":
        action = ("Some simulated readings look unusual — sudden speed change or "
                  "notable braking — but do not by themselves indicate an accident.")
        briefing = [
            "Treat as caution. Keep monitoring the simulated data.",
            "In a real vehicle, slow down and increase following distance.",
            "Pull over when safe if something felt wrong.",
        ]
    else:
        action = "All simulated readings within normal ranges. No concern triggered."
        briefing = [
            "No action needed for this simulated event.",
            "Continue normal driving in a real-world equivalent.",
        ]

    attention = [k for k, v in readings.items()
                 if isinstance(v, (int, float)) and v != 0 and k != "speed_kmh"]

    return {
        "status": "error" if reason not in ("not_configured", "ai_disabled") else reason,
        "source": "fallback",
        "model": "Non-AI fallback (rule-based template)",
        "summary": action,
        "attention_points": [f"simulated {a}" for a in attention] or ["none"],
        "recommendations": briefing,
        "severity_opinion": sev,
        "confidence": None,
        "evidence": [],
        "uncertainty": "Gemma unavailable — rule-based assessment only.",
        "action": action,
        "alert_en": alerts["en"],
        "alert_kn": alerts["kn"],
        "alert_hi": alerts["hi"],
        "responder_briefing": briefing,
        "used_fallback": True,
        "fallback_reason": reason,
        "error": reason if reason not in ("not_configured", "ai_disabled") else None,
    }


# ---------------------------------------------------------------------------
# Connection check  (/api/ai-check)
# ---------------------------------------------------------------------------

def check_connection() -> dict:
    """Make one tiny real Gemma call. Returns connected or a specific reason."""
    api_key = _get_api_key()
    model = _get_model_id()
    if not api_key:
        _set_status("not_configured", "no key found", model, False,
                    "GEMINI_API_KEY not set.")
        return {"status": "not_configured", "reason": "no key found",
                "model": model, "detail": "GEMINI_API_KEY not set.", "response": None}
    try:
        client = _make_client(api_key)
        resp = client.models.generate_content(
            model=model, contents="Reply with exactly one word: READY",
        )
        text = (resp.text or "").strip()
        if not text:
            raise ValueError("empty response")
        _set_status("connected", None, model, True, "Real Gemma response received.")
        return {"status": "connected", "reason": None, "model": model,
                "detail": "Real Gemma response received.",
                "response": _sanitize(text)}
    except (genai_errors.APIError, genai_errors.ClientError,
            genai_errors.ServerError) as exc:
        reason = _classify_error(exc)
        safe = _sanitize(str(exc))
        _set_status("error", reason, model, True, safe)
        return {"status": "error", "reason": reason, "model": model,
                "detail": safe, "response": None}
    except Exception as exc:
        reason = _classify_error(exc)
        safe = _sanitize(str(exc))
        _set_status("error", reason, model, True, safe)
        return {"status": "error", "reason": reason, "model": model,
                "detail": safe, "response": None}


def quick_self_test() -> dict:
    """Compatibility wrapper used by manual_integration_check.py."""
    res = check_connection()
    return {
        "status": res["status"], "model": res["model"], "response": res["response"],
        "error": None if res["status"] == "connected" else (res.get("detail") or res.get("reason")),
    }
