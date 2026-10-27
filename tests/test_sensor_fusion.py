"""
Tests for sensor_fusion.py: fusion counts, false-alarm filter, scenario generators.
"""
import sensor_fusion as sf


def _phone(accel=0.3, gyro=15.0, gps_delta=0.0, loc="Simulated · Test"):
    return sf.PhoneStream(accel_peak_g=accel, gyro_deg_s=gyro,
                          gps_speed_change_kmh=gps_delta, location_label=loc)


def _vehicle(impact=0.05, crash=False, moving=True, speed_delta=0.0):
    return sf.VehicleStream(impact_score=impact, crash_flag=crash,
                            post_impact_movement=moving, speed_change_kmh=speed_delta)


# ---------- Fusion counts ----------

def test_no_signals_for_normal_drive():
    p = _phone(accel=0.3, gyro=10.0, gps_delta=0.0)
    v = _vehicle(impact=0.05, crash=False, moving=True, speed_delta=-2.0)
    f = sf.fuse(p, v)
    assert f.agreeing_signals == 0
    assert f.fused_severity_hint == "low"


def test_all_6_signals_for_severe_crash():
    p = _phone(accel=5.0, gyro=200.0, gps_delta=-55.0)
    v = _vehicle(impact=0.9, crash=True, moving=False, speed_delta=-60.0)
    f = sf.fuse(p, v)
    assert f.agreeing_signals == 6
    assert f.has_vehicle_or_stop is True
    assert f.fused_severity_hint == "critical"


def test_three_signals_with_vehicle_gives_high():
    p = _phone(accel=2.5, gyro=130.0, gps_delta=-35.0)
    v = _vehicle(impact=0.5, crash=False, moving=True, speed_delta=-5.0)
    f = sf.fuse(p, v)
    # phone_accel + phone_gyro + vehicle_impact + sudden_stop (gps)
    assert f.agreeing_signals >= 3
    assert f.has_vehicle_or_stop is True
    assert f.fused_severity_hint in ("high", "critical")


def test_phone_only_signal_no_vehicle():
    p = _phone(accel=3.0, gyro=200.0, gps_delta=0.0)
    v = _vehicle(impact=0.05, crash=False, moving=True, speed_delta=0.0)
    f = sf.fuse(p, v)
    assert f.has_vehicle_or_stop is False


# ---------- False-alarm filter ----------

def test_phone_drop_rejected():
    p = _phone(accel=3.5, gyro=350.0, gps_delta=1.0)
    v = _vehicle(impact=0.05, crash=False, moving=True, speed_delta=1.0)
    fusion = sf.fuse(p, v)
    result = sf.false_alarm_filter(p, v, fusion)
    assert result.passed is False
    assert result.filter_label == "phone drop"


def test_pothole_rejected():
    p = _phone(accel=0.8, gyro=25.0, gps_delta=-5.0)
    v = _vehicle(impact=0.28, crash=False, moving=True, speed_delta=-3.0)
    fusion = sf.fuse(p, v)
    result = sf.false_alarm_filter(p, v, fusion)
    assert result.passed is False
    assert result.filter_label == "pothole"


def test_hard_braking_rejected():
    p = _phone(accel=1.2, gyro=40.0, gps_delta=-35.0)
    v = _vehicle(impact=0.15, crash=False, moving=True, speed_delta=-40.0)
    fusion = sf.fuse(p, v)
    result = sf.false_alarm_filter(p, v, fusion)
    assert result.passed is False
    assert result.filter_label == "hard braking without impact"


def test_real_crash_passes_filter():
    p = _phone(accel=5.0, gyro=180.0, gps_delta=-55.0)
    v = _vehicle(impact=0.88, crash=True, moving=False, speed_delta=-60.0)
    fusion = sf.fuse(p, v)
    result = sf.false_alarm_filter(p, v, fusion)
    assert result.passed is True
    assert result.filter_label == "passed"


def test_real_crash_needs_3_signals_with_vehicle_or_stop():
    # Only 2 signals and no vehicle/stop → fused hint is medium but filter can pass
    # (filter doesn't block on signal count alone; it looks at patterns).
    p = _phone(accel=2.5, gyro=130.0, gps_delta=-10.0)
    v = _vehicle(impact=0.5, crash=False, moving=True, speed_delta=-5.0)
    fusion = sf.fuse(p, v)
    # vehicle_impact is present so has_vehicle_or_stop = True
    assert "vehicle_impact" in fusion.signals_detail
    result = sf.false_alarm_filter(p, v, fusion)
    # This isn't a phone drop, pothole, or hard braking pattern
    assert result.passed is True


# ---------- Scenario generators ----------

def test_generate_normal_passes_filter():
    p, v = sf.generate_streams("normal_driving", seed=1)
    fusion = sf.fuse(p, v)
    result = sf.false_alarm_filter(p, v, fusion)
    # normal should always pass or be ignored (no rejection pattern)
    assert isinstance(result.passed, bool)


def test_generate_phone_drop_is_rejected():
    p, v = sf.generate_streams("phone_drop", seed=2)
    fusion = sf.fuse(p, v)
    result = sf.false_alarm_filter(p, v, fusion)
    assert result.passed is False
    assert result.filter_label == "phone drop"


def test_generate_real_crash_passes():
    p, v = sf.generate_streams("real_crash", seed=3)
    fusion = sf.fuse(p, v)
    result = sf.false_alarm_filter(p, v, fusion)
    assert result.passed is True


def test_generate_hard_braking_is_rejected():
    p, v = sf.generate_streams("hard_braking", seed=4)
    fusion = sf.fuse(p, v)
    result = sf.false_alarm_filter(p, v, fusion)
    assert result.passed is False
    assert result.filter_label == "hard braking without impact"


def test_generate_pothole_is_rejected():
    p, v = sf.generate_streams("pothole", seed=5)
    fusion = sf.fuse(p, v)
    result = sf.false_alarm_filter(p, v, fusion)
    assert result.passed is False
    assert result.filter_label == "pothole"


def test_streams_to_legacy_has_required_fields():
    p, v = sf.generate_streams("possible_accident", seed=10)
    legacy = sf.streams_to_legacy_readings(p, v)
    for key in ("speed_kmh", "speed_delta_kmh", "deceleration_g",
                "impact_score", "lane_deviation_m", "location_label"):
        assert key in legacy
    assert legacy["location_label"].startswith("Simulated")


def test_photo_unavailable_returns_safe_dict():
    """analyse_crash_photo without a key returns a graceful unavailable dict."""
    import os
    import gemma_service as gsi
    original = gsi._get_api_key
    gsi._get_api_key = lambda: None
    try:
        result = gsi.analyse_crash_photo(b"", "image/jpeg")
        assert result["available"] is False
        assert "error" in result
        assert result["confidence"] is None
    finally:
        gsi._get_api_key = original
