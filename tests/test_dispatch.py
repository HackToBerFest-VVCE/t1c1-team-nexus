"""Tests for dispatch: geometry, nearest-unit selection, escalation, transitions."""
import dispatch


def test_haversine_zero_for_same_point():
    p = (12.97, 77.59)
    assert dispatch.haversine_km(p, p) == 0.0


def test_haversine_positive_distance():
    d = dispatch.haversine_km((12.97, 77.59), (13.00, 77.62))
    assert 3.0 < d < 6.0  # roughly a few km


def test_sorted_units_by_distance_is_ascending():
    loc = (12.9700, 77.5950)  # exactly at unit A
    ordered = dispatch.sorted_units_by_distance(dispatch.all_units(), loc)
    assert ordered[0].name == "Ambulance 108-A"
    dists = [dispatch.haversine_km((u.lat, u.lng), loc) for u in ordered]
    assert dists == sorted(dists)


def test_nearest_available_unit_returns_closest():
    loc = (12.9700, 77.5950)
    u = dispatch.nearest_available_unit(dispatch.all_units(), loc)
    assert u is not None
    assert u.name == "Ambulance 108-A"


def test_next_unit_skips_tried_and_unavailable():
    units = dispatch.all_units()
    loc = (12.9700, 77.5950)
    # Make A unavailable; nearest available should skip it.
    units[0].available = False
    u = dispatch.next_unit_to_offer(units, loc, tried=set())
    assert u.name != "Ambulance 108-A"
    units[0].available = True  # restore


def test_next_unit_none_when_all_tried_or_busy():
    units = dispatch.all_units()
    loc = (12.9700, 77.5950)
    tried = {u.name for u in units}
    assert dispatch.next_unit_to_offer(units, loc, tried=tried) is None


def test_validate_transition_valid():
    assert dispatch.validate_transition("new", "awaiting_victim") == (True, None)
    assert dispatch.validate_transition("alerted", "assigned") == (True, None)
    assert dispatch.validate_transition("assigned", "en_route") == (True, None)
    assert dispatch.validate_transition("en_route", "arrived") == (True, None)
    assert dispatch.validate_transition("arrived", "resolved") == (True, None)


def test_validate_transition_invalid():
    ok, err = dispatch.validate_transition("awaiting_victim", "en_route")
    assert ok is False and err
    ok, err = dispatch.validate_transition("resolved", "assigned")
    assert ok is False and err  # terminal


def test_validate_transition_unknown_status():
    ok, err = dispatch.validate_transition("new", "bogus")
    assert ok is False and err
    ok, err = dispatch.validate_transition("bogus", "new")
    assert ok is False and err
