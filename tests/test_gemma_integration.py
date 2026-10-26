"""
AccidentGuard — REAL Gemma 4 integration tests.

These tests make ACTUAL API calls to the hosted Gemma 4 model and require a
valid GEMINI_API_KEY in the environment (or a .env file). They are SEPARATE from
the mock tests in test_accident_detection.py.

If no key is configured, every test here SKIPS (rather than failing), so the
full suite stays green for offline / mock runs:

    # Offline mock tests (no key needed):
    python -m pytest tests/test_accident_detection.py -v

    # Real API integration tests (needs your key in .env):
    python -m pytest tests/test_gemma_integration.py -v
"""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Load .env so a key placed there is picked up during local runs.
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

import pytest  # noqa: E402

import gemma_service as gs  # noqa: E402


HAS_KEY = bool(os.environ.get("GEMINI_API_KEY", "").strip())

# Skip the whole module when there is no key — these are REAL API tests.
pytestmark = pytest.mark.skipif(
    not HAS_KEY,
    reason="GEMINI_API_KEY not set — skipping real Gemma 4 API tests. "
           "Add a key to .env to run these.",
)


def _accident_payload():
    return {
        "event_id": "EVT-INTEG",
        "timestamp": "2026-10-26T10:00:00Z",
        "category": "possible_accident",
        "severity": "critical",
        "simulated_readings": {
            "speed_kmh": 62.0,
            "speed_delta_kmh": -48.0,
            "deceleration_g": 0.88,
            "impact_score": 0.82,
            "lane_deviation_m": 1.15,
            "location_label": "Simulated · Test Highway",
        },
        "rule_engine_notes": "Very hard deceleration; High impact score.",
    }


def test_quick_self_test_connects():
    """A trivial real call confirms the key + model are valid."""
    result = gs.quick_self_test()
    assert result["status"] == "connected", (
        f"Expected connected, got {result['status']}: {result.get('error')}"
    )
    assert result["response"], "Expected a non-empty response from Gemma."


def test_real_gemma_summary_for_possible_accident():
    """The headline test: real Gemma summarises a simulated accident event."""
    result = gs.generate_incident_summary(_accident_payload())

    assert result["status"] == "connected", (
        f"Expected connected, got {result['status']}: {result.get('error')}"
    )
    assert result["source"] == "gemma", "Response must come from Gemma, not fallback."
    assert result["used_fallback"] is False
    assert result["model"] in gs.VERIFIED_MODELS, (
        f"Model {result['model']} is not a verified Gemma 4 id."
    )
    assert result["summary"], "Gemma must return a non-empty summary."
    assert isinstance(result["recommendations"], list)
    assert len(result["recommendations"]) >= 1


def test_status_reflects_success_after_call():
    gs.generate_incident_summary(_accident_payload())
    status = gs.get_status()
    assert status["state"] == "connected"
    assert status["configured"] is True
