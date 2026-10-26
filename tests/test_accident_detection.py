"""
Tests for the AccidentGuard event classification engine.

These are MOCK-free unit tests — they do not call the Gemma API and do not need
a network connection or API key. Real API tests live separately
(see test_gemma_integration.py).
"""

import os
import sys

# Make the project root importable when running pytest from anywhere.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from accident_detection import (  # noqa: E402
    EventCategory,
    Severity,
    DrivingReadings,
    classify_event,
    generate_sample_event,
    event_for_ai,
    SIMULATED_LOCATIONS,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _readings(**overrides) -> DrivingReadings:
    base = dict(
        timestamp="2026-10-26T10:00:00Z",
        speed_kmh=50.0,
        speed_delta_kmh=-3.0,
        deceleration_g=0.10,
        impact_score=0.05,
        lane_deviation_m=0.10,
        location_label="Simulated · Test Road",
        event_id="EVT-TEST",
    )
    base.update(overrides)
    return DrivingReadings(**base)


# ---------------------------------------------------------------------------
# Category tests (the three required cases)
# ---------------------------------------------------------------------------

def test_normal_driving_event():
    result = generate_sample_event(category=EventCategory.NORMAL, seed=1)
    assert result.category is EventCategory.NORMAL
    assert result.severity is Severity.LOW
    assert result.triggered_rules == []
    assert result.readings.impact_score < 0.35


def test_suspicious_driving_event():
    result = generate_sample_event(category=EventCategory.SUSPICIOUS, seed=2)
    assert result.category is EventCategory.SUSPICIOUS
    assert result.severity is Severity.MEDIUM
    assert len(result.triggered_rules) >= 1


def test_possible_accident_event():
    result = generate_sample_event(category=EventCategory.POSSIBLE_ACCIDENT, seed=3)
    assert result.category is EventCategory.POSSIBLE_ACCIDENT
    assert result.severity in (Severity.HIGH, Severity.CRITICAL)
    assert len(result.triggered_rules) >= 2  # accident needs 2+ signals


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------

def test_seed_is_reproducible():
    a = generate_sample_event(category=EventCategory.POSSIBLE_ACCIDENT, seed=42)
    b = generate_sample_event(category=EventCategory.POSSIBLE_ACCIDENT, seed=42)
    assert a.readings.to_dict() == b.readings.to_dict()
    assert a.category == b.category


# ---------------------------------------------------------------------------
# Direct classification rules
# ---------------------------------------------------------------------------

def test_classify_normal_readings_directly():
    result = classify_event(_readings())  # all low values
    assert result.category is EventCategory.NORMAL
    assert result.severity is Severity.LOW
    # "source" is a marker added in to_dict() so the UI never confuses
    # engine output with AI output.
    assert result.to_dict()["source"] == "rule_engine"


def test_classify_accident_requires_two_signals():
    # High impact alone should not be CRITICAL accident unless another signal fires
    only_impact = classify_event(
        _readings(impact_score=0.85, deceleration_g=0.10, lane_deviation_m=0.10, speed_delta_kmh=-3.0)
    )
    assert only_impact.category in (EventCategory.SUSPICIOUS, EventCategory.POSSIBLE_ACCIDENT)

    # High impact + hard deceleration => possible accident
    two_signals = classify_event(
        _readings(impact_score=0.85, deceleration_g=0.90, lane_deviation_m=0.10, speed_delta_kmh=-3.0)
    )
    assert two_signals.category is EventCategory.POSSIBLE_ACCIDENT
    assert two_signals.severity is Severity.CRITICAL  # high impact => critical


def test_suspicious_classification_directly():
    result = classify_event(
        _readings(deceleration_g=0.55, impact_score=0.10, lane_deviation_m=0.10, speed_delta_kmh=-10.0)
    )
    assert result.category is EventCategory.SUSPICIOUS
    assert result.severity is Severity.MEDIUM


# ---------------------------------------------------------------------------
# Serialization
# ---------------------------------------------------------------------------

def test_to_dict_structure():
    result = generate_sample_event(category=EventCategory.NORMAL, seed=5)
    d = result.to_dict()
    assert set(d.keys()) >= {"event_id", "timestamp", "category", "severity",
                             "readings", "rule_explanation", "triggered_rules", "source"}
    assert d["source"] == "rule_engine"
    assert d["category"] == "normal_driving"
    assert isinstance(d["readings"], dict)


def test_event_for_ai_payload_is_privacy_safe():
    result = generate_sample_event(category=EventCategory.POSSIBLE_ACCIDENT, seed=7)
    payload = event_for_ai(result)
    assert "simulated_readings" in payload
    # No personal-data keys should be present
    forbidden = {"driver_name", "plate", "phone", "user_id", "name"}
    assert not (forbidden & set(payload.keys()))
    assert payload["simulated_readings"]["location_label"].startswith("Simulated")


def test_locations_are_clearly_labelled_as_simulated():
    for loc in SIMULATED_LOCATIONS:
        assert "Simulated" in loc
