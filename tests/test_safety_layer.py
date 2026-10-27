"""Tests for the safety layer (Python makes the final severity decision)."""
import accident_detection as det
import safety_layer as sl


def _r(impact=0.0, decel=0.0):
    return {"impact_score": impact, "deceleration_g": decel}


def test_no_impact_no_sharp_decel_never_above_medium():
    # Gemma says critical, but with no impact and no sharp deceleration it is capped.
    res = sl.decide_final_severity("low", _r(impact=0.1, decel=0.2), "critical")
    assert res["final_severity"] in ("low", "medium")
    assert res["disagreement"] is True  # Gemma wanted critical
    assert "disagreement handled" in res["note"]


def test_strong_impact_with_hard_decel_at_least_high():
    res = sl.decide_final_severity("medium", _r(impact=0.85, decel=0.90), "medium")
    assert res["final_severity"] in ("high", "critical")


def test_very_high_impact_critical_floor():
    res = sl.decide_final_severity("high", _r(impact=0.92, decel=0.90), "medium")
    assert res["final_severity"] == "critical"


def test_gemma_can_raise_within_band():
    # Sharp deceleration present so no ceiling; Gemma raises medium -> high.
    res = sl.decide_final_severity("medium", _r(impact=0.5, decel=0.6), "high")
    assert res["final_severity"] == "high"
    assert res["disagreement"] is True


def test_low_severity_means_no_alert():
    res = sl.decide_final_severity("low", _r(impact=0.1, decel=0.1), "low")
    assert res["final_severity"] == "low"
    assert res["alert_needed"] is False


def test_medium_and_above_need_alert():
    res = sl.decide_final_severity("medium", _r(impact=0.4, decel=0.5), "medium")
    assert res["final_severity"] == "medium"
    assert res["alert_needed"] is True


def test_gemma_unavailable_uses_rules_no_disagreement():
    res = sl.decide_final_severity("medium", _r(impact=0.4, decel=0.5), None)
    assert res["final_severity"] == "medium"
    assert res["disagreement"] is False
    assert "unavailable" in res["note"]


def test_accepts_severity_enum():
    res = sl.decide_final_severity(det.Severity.CRITICAL,
                                   _r(impact=0.95, decel=0.95), det.Severity.HIGH)
    assert res["final_severity"] == "critical"
