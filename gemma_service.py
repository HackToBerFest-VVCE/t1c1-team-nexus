"""
AccidentGuard — Gemma 4 integration service
==========================================

This module performs the REAL API call to a hosted Gemma 4 model through the
Google Gemini API (the new `google-genai` SDK), as documented at:
    https://ai.google.dev/gemma/docs/core/gemma_on_gemini_api

Verified supported model IDs (checked against the official docs):
    - gemma-4-26b-a4b-it   (preferred / default)
    - gemma-4-31b-it       (documented fallback)

Design rules (project requirements):
  * The API key is read ONLY from the GEMINI_API_KEY environment variable.
    It is never logged, never returned to the frontend, and never placed in
    error messages.
  * Status indicator has exactly three real states:
        connected      -> a real Gemma API response was received
        error          -> the request failed
        not_configured -> the API key is missing
  * A clearly-labelled NON-AI fallback keeps the demo running when the API is
    unavailable. It is NEVER labelled as coming from Gemma.
  * Gemma is constrained by a system instruction so it must not invent sensor
    readings, must not claim certainty about a real accident, and must not
    replace emergency professionals.
"""

from __future__ import annotations

import json
import os
import re
from typing import Optional

from google import genai
from google.genai import types
from google.genai import errors as genai_errors


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Verified against the official Gemma-on-Gemini-API docs.
VERIFIED_MODELS = ("gemma-4-26b-a4b-it", "gemma-4-31b-it")
DEFAULT_MODEL = "gemma-4-26b-a4b-it"

ENV_KEY = "GEMINI_API_KEY"
ENV_MODEL = "GEMMA_MODEL_ID"
ENV_TIMEOUT = "GEMMA_TIMEOUT"
DEFAULT_TIMEOUT_MS = 30_000  # 30 seconds


def _get_api_key() -> Optional[str]:
    """Return the API key from the environment, or None if not set."""
    key = os.environ.get(ENV_KEY, "").strip()
    return key or None


def _get_model_id() -> str:
    """Return the configured model ID, defaulting to the preferred verified one."""
    model = os.environ.get(ENV_MODEL, "").strip() or DEFAULT_MODEL
    if model not in VERIFIED_MODELS:
        # Don't silently use an unknown model: fall back to the verified default.
        return DEFAULT_MODEL
    return model


def _get_timeout_ms() -> int:
    """Return the API timeout in milliseconds.

    GEMMA_TIMEOUT is read in SECONDS (matching .env.example), then converted to
    milliseconds for the google-genai SDK's HttpOptions.timeout. A value <= 0 or a
    non-numeric value falls back to the 30-second default.
    """
    try:
        seconds = int(os.environ.get(ENV_TIMEOUT, "30"))
        if seconds <= 0:
            seconds = 30
        return seconds * 1000
    except ValueError:
        return DEFAULT_TIMEOUT_MS


# ---------------------------------------------------------------------------
# Public status helpers
# ---------------------------------------------------------------------------

# Tracks the outcome of the most recent generate call so the dashboard can show
# a live AI status. Updated only by generate_incident_summary().
_last_status: dict = {
    "state": "not_configured",     # connected | error | not_configured
    "detail": "No API call made yet.",
    "model": DEFAULT_MODEL,
    "verified": True,
    "configured": False,
}


def is_configured() -> bool:
    """True if an API key is present in the environment."""
    return _get_api_key() is not None


def get_status() -> dict:
    """Return the current AI status snapshot for the dashboard."""
    _last_status["configured"] = is_configured()
    if not is_configured() and _last_status["state"] == "connected":
        # Key was removed after a successful call — reflect reality.
        _last_status["state"] = "not_configured"
        _last_status["detail"] = "API key is missing."
    return dict(_last_status)


def get_model_info() -> dict:
    return {
        "model": _get_model_id(),
        "verified_supported": _get_model_id() in VERIFIED_MODELS,
        "verified_models": list(VERIFIED_MODELS),
        "configured": is_configured(),
    }


# ---------------------------------------------------------------------------
# Prompt construction (constrains the model's behaviour)
# ---------------------------------------------------------------------------

SYSTEM_INSTRUCTION = """You are AccidentGuard's safety assistant. You review \
SIMULATED driving-event data produced by a hackathon prototype. There are NO \
real sensors, NO real GPS, and NO real accidents — the readings are illustrative.

Strict rules you MUST follow:
1. Treat every reading as illustrative simulation data, not a measurement.
2. Do NOT invent sensor readings, speeds, locations, or values that are not in the input.
3. Do NOT claim certainty that a real accident occurred. Use cautious language \
such as "the data suggests" or "if this pattern occurred in a real vehicle".
4. You are NOT a replacement for emergency professionals. For anything that could \
be a real-world incident, advise contacting actual emergency services.
5. Keep language simple and non-technical so a layperson can understand it.

Respond using EXACTLY these four labelled sections, each starting on its own line, in order:
SUMMARY: 2 to 4 sentences in plain language.
ATTENTION: comma-separated details from the supplied data that may need attention, or the word none.
RECOMMENDATIONS: cautious next steps, each on its own line starting with "- ".
SEVERITY_OPINION: exactly one of low, medium, high, or critical."""


def _build_user_prompt(payload: dict) -> str:
    """Render the incident payload as a readable, privacy-safe text block."""
    readings = payload.get("simulated_readings", {})
    lines = [
        "AccidentGuard simulated event review request.",
        "Note: the data below is SIMULATED for a prototype demo.",
        "",
        f"Event ID: {payload.get('event_id', 'unknown')}",
        f"Timestamp: {payload.get('timestamp', 'unknown')}",
        f"Rule-engine classification: {payload.get('category', 'unknown')}",
        f"Rule-engine severity: {payload.get('severity', 'unknown')}",
        f"Simulated location label: {readings.get('location_label', 'unknown')}",
        "",
        "Simulated readings:",
        f"- speed: {readings.get('speed_kmh', 'N/A')} km/h",
        f"- speed change: {readings.get('speed_delta_kmh', 'N/A')} km/h",
        f"- deceleration: {readings.get('deceleration_g', 'N/A')} g",
        f"- impact score (0..1): {readings.get('impact_score', 'N/A')}",
        f"- lane deviation: {readings.get('lane_deviation_m', 'N/A')} m",
        "",
        f"Rule-engine notes: {payload.get('rule_engine_notes', '')}",
        "",
        "Provide the SUMMARY, ATTENTION, RECOMMENDATIONS, and SEVERITY_OPINION sections now.",
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Response parsing
# ---------------------------------------------------------------------------

def _parse_structured(text: str) -> dict:
    """Extract the four labelled sections from the model response."""
    result = {
        "summary": text.strip(),
        "attention_points": [],
        "recommendations": [],
        "severity_opinion": None,
        "raw": text,
    }

    def _section(name: str) -> Optional[str]:
        # Match "NAME:" up to the next section label or end of text.
        pattern = rf"{name}:\s*(.*?)(?=\n\s*(?:SUMMARY|ATTENTION|RECOMMENDATIONS|SEVERITY_OPINION):|\Z)"
        m = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
        return m.group(1).strip() if m else None

    summary = _section("SUMMARY")
    if summary:
        result["summary"] = summary

    attention = _section("ATTENTION")
    if attention and attention.lower() != "none":
        result["attention_points"] = [a.strip() for a in attention.split(",") if a.strip()]

    recs = _section("RECOMMENDATIONS")
    if recs:
        result["recommendations"] = [
            r.lstrip("- ").strip() for r in recs.splitlines() if r.strip()
        ]

    sev = _section("SEVERITY_OPINION")
    if sev:
        sev_low = sev.strip().lower()
        if sev_low in ("low", "medium", "high", "critical"):
            result["severity_opinion"] = sev_low

    return result


# ---------------------------------------------------------------------------
# Fallback (NON-AI) — for demo continuity only
# ---------------------------------------------------------------------------

def _fallback_summary(payload: dict, reason: str) -> dict:
    """
    Build a clearly-labelled, NON-AI summary from the rule-engine data alone.
    This is used when the API is unavailable. It is NEVER presented as Gemma output.
    """
    cat = payload.get("category", "unknown")
    sev = payload.get("severity", "unknown")
    readings = payload.get("simulated_readings", {})

    if cat == "possible_accident":
        summary_text = (
            "Several simulated readings show a sharp combination of hard braking, a "
            "high impact score, and large lane deviation. In a real vehicle this "
            "pattern would warrant immediate attention, but this is simulated data only."
        )
        recs = [
            "If this were a real event, stop the vehicle safely and check for injury.",
            "Contact actual emergency services if anyone may be hurt.",
            "Do not move an injured person unless there is immediate danger.",
            "Record the time and circumstances for a real-world follow-up.",
        ]
    elif cat == "suspicious_event":
        summary_text = (
            "Some simulated readings look unusual — for example a sudden speed change "
            "or notable braking — but they do not by themselves indicate an accident."
        )
        recs = [
            "Treat as a caution flag and continue monitoring the simulated data.",
            "In a real vehicle, slow down and increase following distance.",
            "If something felt wrong in a real situation, pull over when safe to check.",
        ]
    else:
        summary_text = (
            "All simulated readings are within normal ranges. No rule-based concern "
            "was triggered for this event."
        )
        recs = [
            "No action needed for this simulated event.",
            "Continue normal driving behaviour in a real-world equivalent.",
        ]

    attention = [k for k, v in readings.items()
                 if isinstance(v, (int, float)) and v != 0 and k != "speed_kmh"]

    return {
        "status": "error" if reason != "not_configured" else "not_configured",
        "source": "fallback",
        "model": "rule-based fallback (non-AI)",
        "summary": summary_text,
        "attention_points": [f"simulated {a}" for a in attention] or ["none"],
        "recommendations": recs,
        "severity_opinion": sev,
        "used_fallback": True,
        "fallback_reason": reason,
        "error": reason if reason != "not_configured" else None,
    }


# ---------------------------------------------------------------------------
# Sanitization — never leak the API key
# ---------------------------------------------------------------------------

def _sanitize(message: str) -> str:
    """Remove the API key from any string before it reaches the user/logs."""
    key = _get_api_key()
    if key and key in message:
        message = message.replace(key, "***REDACTED***")
    # Also redact common bearer patterns defensively.
    message = re.sub(r"(key=)[^\s&]+", r"\1***REDACTED***", message)
    return message


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def generate_incident_summary(payload: dict) -> dict:
    """
    Send an incident payload to Gemma 4 and return a structured result.

    Returns a dict with:
        status            connected | error | not_configured
        source            gemma | fallback
        model             the model id used, or the fallback label
        summary           plain-language summary
        attention_points  list[str]
        recommendations   list[str]
        severity_opinion  low|medium|high|critical | None
        used_fallback     bool
        error             safe message or None (never contains the key)
    """
    if not isinstance(payload, dict):
        payload = {}

    api_key = _get_api_key()
    model = _get_model_id()

    # --- State 1: Not configured -------------------------------------------
    if not api_key:
        _last_status.update(
            state="not_configured",
            detail="GEMINI_API_KEY is not set. Provide a key in .env to enable AI.",
            model=model,
            configured=False,
        )
        return _fallback_summary(payload, "not_configured")

    # --- State 2/3: Make the real API call --------------------------------
    prompt = _build_user_prompt(payload)
    try:
        client = genai.Client(
            api_key=api_key,
            http_options=types.HttpOptions(timeout=_get_timeout_ms()),
        )
        response = client.models.generate_content(
            model=model,
            contents=prompt,
            config=types.GenerateContentConfig(
                system_instruction=SYSTEM_INSTRUCTION,
                temperature=0.3,
                # Keep responses fast for a live demo; "minimal" disables long thinking.
                thinking_config=types.ThinkingConfig(thinking_level="minimal"),
            ),
        )

        # response.text raises if the output is empty/blocked.
        text = response.text
        if not text or not text.strip():
            raise RuntimeError("Gemma returned an empty response.")

        parsed = _parse_structured(text)
        _last_status.update(
            state="connected",
            detail=f"Real response received from {model}.",
            model=model,
            configured=True,
        )
        return {
            "status": "connected",
            "source": "gemma",
            "model": model,
            "summary": parsed["summary"],
            "attention_points": parsed["attention_points"],
            "recommendations": parsed["recommendations"],
            "severity_opinion": parsed["severity_opinion"],
            "used_fallback": False,
            "error": None,
        }

    except (genai_errors.APIError, genai_errors.ClientError,
            genai_errors.ServerError) as exc:
        safe = _sanitize(str(exc))
        _last_status.update(
            state="error",
            detail=f"API request failed: {safe}",
            model=model,
            configured=True,
        )
        result = _fallback_summary(payload, "api_error")
        result["error"] = safe
        return result

    except Exception as exc:  # network / timeout / parse / unknown
        safe = _sanitize(str(exc))
        # Classify likely timeouts for a clearer message.
        lowered = safe.lower()
        is_timeout = "timeout" in lowered or "timed out" in lowered
        reason = "timeout" if is_timeout else "unknown_error"
        _last_status.update(
            state="error",
            detail=f"Request failed ({reason}): {safe}",
            model=model,
            configured=True,
        )
        result = _fallback_summary(payload, reason)
        result["error"] = safe if reason != "timeout" else "The request timed out."
        return result


# ---------------------------------------------------------------------------
# Convenience for manual / integration testing
# ---------------------------------------------------------------------------

def quick_self_test() -> dict:
    """
    Make one real API call with a tiny prompt to confirm the key + model work.
    Used by the integration test script and the /api/ai/test endpoint.
    """
    api_key = _get_api_key()
    model = _get_model_id()
    if not api_key:
        return {"status": "not_configured", "model": model, "response": None,
                "error": "GEMINI_API_KEY is not set."}
    try:
        client = genai.Client(api_key=api_key,
                              http_options=types.HttpOptions(timeout=_get_timeout_ms()))
        resp = client.models.generate_content(
            model=model,
            contents="Reply with exactly the word: READY",
        )
        return {"status": "connected", "model": model,
                "response": _sanitize(resp.text or ""), "error": None}
    except Exception as exc:
        return {"status": "error", "model": model, "response": None,
                "error": _sanitize(str(exc))}
