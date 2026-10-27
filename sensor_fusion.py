"""
AccidentGuard — Sensor Fusion & False-Alarm Filter
===================================================

Two simulated sensor streams:
  Phone   : accelerometer peak (g), gyroscope rotation (deg/s),
            GPS speed change (km/h), GPS lat/lng location label.
  Vehicle : impact sensor score (0-1), airbag / crash-sensor flag,
            post-impact movement flag (True = still moving, False = stopped),
            vehicle speed change (km/h).

Fusion rule: count how many of 6 independent signals agree on an impact.
  1  phone_accel_peak >= ACCEL_IMPACT_G
  2  phone_gyro_deg_s >= GYRO_IMPACT_DEG
  3  vehicle_impact_score >= VEHICLE_IMPACT_THRESHOLD
  4  vehicle_crash_flag == True
  5  |gps_speed_change| >= SUDDEN_STOP_KMH  (GPS sudden stop)
  6  post_impact_movement == False           (no movement after event)

False-alarm filter (BEFORE Gemma):
  * Phone drop      : phone accel high, vehicle quiet, no GPS speed change
                     → REJECTED as "phone drop"
  * Pothole         : vehicle impact moderate, no crash flag, slight decel,
                     no GPS speed change, phone accel low
                     → REJECTED as "pothole"
  * Hard braking    : high decel from vehicle, crash flag False, impact low
                     → REJECTED as "hard braking without impact"
  * Real crash      : impact + sudden stop + no movement (post_impact_movement=False)
                     → PASSED

A possible accident requires ≥3 agreeing signals AND at least one of:
  vehicle_impact_score >= threshold, crash_flag, or sudden stop.

SIMULATION ONLY — no real sensors.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, asdict
from typing import Optional

# Signal thresholds (illustrative, not calibrated against real data)
ACCEL_IMPACT_G = 2.0        # phone accelerometer peak indicating probable impact
GYRO_IMPACT_DEG = 120.0     # phone gyroscope rotation speed (deg/s)
VEHICLE_IMPACT_THRESHOLD = 0.40   # vehicle impact sensor score
SUDDEN_STOP_KMH = 30.0      # sudden GPS speed drop

# Scenario → (category hint, stream characteristics)
SCENARIOS = {
    "normal_driving":   "Normal driving",
    "hard_braking":     "Hard braking",
    "pothole":          "Pothole",
    "phone_drop":       "Phone drop",
    "suspicious_event": "Suspicious event",
    "possible_accident":"Possible accident",
    "real_crash":       "Real crash",
    "random":           "Random",
}


@dataclass
class PhoneStream:
    accel_peak_g: float          # accelerometer peak
    gyro_deg_s: float            # gyroscope rotation
    gps_speed_change_kmh: float  # GPS speed delta (negative = decel)
    location_label: str

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class VehicleStream:
    impact_score: float          # 0..1 vehicle impact sensor
    crash_flag: bool             # airbag / crash sensor triggered
    post_impact_movement: bool   # True = still moving after event
    speed_change_kmh: float      # vehicle ECU speed delta

    def to_dict(self) -> dict:
        d = asdict(self)
        # bool -> friendly string for JSON
        d["crash_flag"] = self.crash_flag
        d["post_impact_movement"] = self.post_impact_movement
        return d


@dataclass
class FusionResult:
    agreeing_signals: int        # 0..6
    signals_detail: list[str]    # which signals fired
    has_vehicle_or_stop: bool    # at least one vehicle/stop signal
    fused_severity_hint: str     # "low" | "medium" | "high" | "critical"

    def to_dict(self) -> dict:
        return {
            "agreeing_signals": self.agreeing_signals,
            "signals_detail": self.signals_detail,
            "has_vehicle_or_stop": self.has_vehicle_or_stop,
            "fused_severity_hint": self.fused_severity_hint,
        }


@dataclass
class FilterResult:
    passed: bool
    reason: str          # empty string if passed
    filter_label: str    # e.g. "phone drop", "pothole", "hard braking", "passed"

    def to_dict(self) -> dict:
        return {
            "passed": self.passed,
            "reason": self.reason,
            "filter_label": self.filter_label,
        }


def fuse(phone: PhoneStream, vehicle: VehicleStream) -> FusionResult:
    """Count agreeing signals and determine fused severity hint."""
    signals = []

    if phone.accel_peak_g >= ACCEL_IMPACT_G:
        signals.append("phone_accel")
    if phone.gyro_deg_s >= GYRO_IMPACT_DEG:
        signals.append("phone_gyro")
    if vehicle.impact_score >= VEHICLE_IMPACT_THRESHOLD:
        signals.append("vehicle_impact")
    if vehicle.crash_flag:
        signals.append("vehicle_crash_flag")
    if abs(phone.gps_speed_change_kmh) >= SUDDEN_STOP_KMH or abs(vehicle.speed_change_kmh) >= SUDDEN_STOP_KMH:
        signals.append("sudden_stop")
    if not vehicle.post_impact_movement:
        signals.append("no_post_impact_movement")

    n = len(signals)
    has_vehicle_or_stop = any(s in signals for s in (
        "vehicle_impact", "vehicle_crash_flag", "sudden_stop", "no_post_impact_movement"
    ))

    if n >= 4 and has_vehicle_or_stop:
        hint = "critical"
    elif n >= 3 and has_vehicle_or_stop:
        hint = "high"
    elif n >= 2:
        hint = "medium"
    elif n == 1:
        hint = "low"
    else:
        hint = "low"

    return FusionResult(
        agreeing_signals=n,
        signals_detail=signals,
        has_vehicle_or_stop=has_vehicle_or_stop,
        fused_severity_hint=hint,
    )


def false_alarm_filter(phone: PhoneStream, vehicle: VehicleStream,
                       fusion: FusionResult) -> FilterResult:
    """
    Apply false-alarm detection. Returns a FilterResult with passed=True
    when the event should proceed to Gemma / alert, or passed=False with
    a reason when it is a known false-alarm pattern.
    """
    # Phone drop: phone alone triggered, vehicle silent, no GPS speed change
    phone_only = "phone_accel" in fusion.signals_detail and not fusion.has_vehicle_or_stop
    no_gps = abs(phone.gps_speed_change_kmh) < 10.0
    vehicle_quiet = vehicle.impact_score < 0.20 and not vehicle.crash_flag
    if phone_only and vehicle_quiet and no_gps:
        return FilterResult(
            passed=False,
            reason="Phone impact only — vehicle sensors quiet, no GPS speed change. Likely a phone drop.",
            filter_label="phone drop",
        )

    # Hard braking: decel without meaningful impact
    hard_brake_no_impact = (
        abs(vehicle.speed_change_kmh) >= 25.0
        and vehicle.impact_score < VEHICLE_IMPACT_THRESHOLD
        and not vehicle.crash_flag
        and fusion.agreeing_signals <= 1
    )
    if hard_brake_no_impact:
        return FilterResult(
            passed=False,
            reason="Hard braking detected but no impact signal from vehicle or phone. Classified as hard braking.",
            filter_label="hard braking without impact",
        )

    # Pothole: moderate vehicle impact, no crash flag, no GPS speed change, low phone accel
    pothole = (
        0.20 <= vehicle.impact_score < VEHICLE_IMPACT_THRESHOLD
        and not vehicle.crash_flag
        and abs(phone.gps_speed_change_kmh) < 15.0
        and phone.accel_peak_g < ACCEL_IMPACT_G
        and fusion.agreeing_signals <= 2
    )
    if pothole:
        return FilterResult(
            passed=False,
            reason="Moderate vehicle impact with no crash flag or GPS speed change. Likely a pothole.",
            filter_label="pothole",
        )

    return FilterResult(passed=True, reason="", filter_label="passed")


# -------------------------------------------------------------------------
# Simulated stream generators
# -------------------------------------------------------------------------

_LOCATIONS = [
    "Simulated · MG Road, Bengaluru",
    "Simulated · Outer Ring Road, Bengaluru",
    "Simulated · Highway NH-48 (simulated stretch)",
    "Simulated · Jayanagar 4th Block, Bengaluru",
    "Simulated · Airport Flyover (simulated)",
    "Simulated · Residential Area, Indiranagar",
]


def generate_streams(scenario: str, seed: Optional[int] = None) -> tuple[PhoneStream, VehicleStream]:
    """Generate two simulated sensor streams consistent with the given scenario."""
    if seed is not None:
        random.seed(seed)

    loc = random.choice(_LOCATIONS)

    if scenario in ("normal_driving", "random"):
        p = PhoneStream(
            accel_peak_g=round(random.uniform(0.1, 0.8), 2),
            gyro_deg_s=round(random.uniform(5, 60), 1),
            gps_speed_change_kmh=round(random.uniform(-5, 5), 1),
            location_label=loc,
        )
        v = VehicleStream(
            impact_score=round(random.uniform(0.0, 0.15), 2),
            crash_flag=False,
            post_impact_movement=True,
            speed_change_kmh=round(random.uniform(-6, 4), 1),
        )

    elif scenario == "hard_braking":
        p = PhoneStream(
            accel_peak_g=round(random.uniform(0.5, 1.5), 2),
            gyro_deg_s=round(random.uniform(20, 80), 1),
            gps_speed_change_kmh=round(random.uniform(-40, -20), 1),
            location_label=loc,
        )
        v = VehicleStream(
            impact_score=round(random.uniform(0.05, 0.30), 2),
            crash_flag=False,
            post_impact_movement=True,
            speed_change_kmh=round(random.uniform(-45, -25), 1),
        )

    elif scenario == "pothole":
        p = PhoneStream(
            accel_peak_g=round(random.uniform(0.3, 1.2), 2),
            gyro_deg_s=round(random.uniform(10, 50), 1),
            gps_speed_change_kmh=round(random.uniform(-10, 5), 1),
            location_label=loc,
        )
        v = VehicleStream(
            impact_score=round(random.uniform(0.22, 0.38), 2),
            crash_flag=False,
            post_impact_movement=True,
            speed_change_kmh=round(random.uniform(-8, 2), 1),
        )

    elif scenario == "phone_drop":
        p = PhoneStream(
            accel_peak_g=round(random.uniform(2.5, 5.0), 2),
            gyro_deg_s=round(random.uniform(200, 500), 1),
            gps_speed_change_kmh=round(random.uniform(-3, 3), 1),
            location_label=loc,
        )
        v = VehicleStream(
            impact_score=round(random.uniform(0.0, 0.10), 2),
            crash_flag=False,
            post_impact_movement=True,
            speed_change_kmh=round(random.uniform(-3, 3), 1),
        )

    elif scenario == "suspicious_event":
        p = PhoneStream(
            accel_peak_g=round(random.uniform(1.5, 2.8), 2),
            gyro_deg_s=round(random.uniform(70, 160), 1),
            gps_speed_change_kmh=round(random.uniform(-30, -15), 1),
            location_label=loc,
        )
        v = VehicleStream(
            impact_score=round(random.uniform(0.30, 0.55), 2),
            crash_flag=False,
            post_impact_movement=True,
            speed_change_kmh=round(random.uniform(-35, -18), 1),
        )

    elif scenario in ("possible_accident", "real_crash"):
        p = PhoneStream(
            accel_peak_g=round(random.uniform(3.0, 8.0), 2),
            gyro_deg_s=round(random.uniform(150, 400), 1),
            gps_speed_change_kmh=round(random.uniform(-60, -35), 1),
            location_label=loc,
        )
        v = VehicleStream(
            impact_score=round(random.uniform(0.65, 0.98), 2),
            crash_flag=True,
            post_impact_movement=False,
            speed_change_kmh=round(random.uniform(-65, -40), 1),
        )

    else:
        # fallback to normal
        p = PhoneStream(
            accel_peak_g=0.3, gyro_deg_s=15.0,
            gps_speed_change_kmh=0.0, location_label=loc,
        )
        v = VehicleStream(
            impact_score=0.05, crash_flag=False,
            post_impact_movement=True, speed_change_kmh=0.0,
        )

    return p, v


def streams_to_legacy_readings(phone: PhoneStream, vehicle: VehicleStream) -> dict:
    """
    Map the new dual-stream data to the legacy DrivingReadings fields used by the
    rule engine and safety layer. This keeps backward compatibility.
    """
    # deceleration_g: worst of phone accel and derived vehicle decel
    vehicle_decel_g = abs(vehicle.speed_change_kmh) / 27.78 / 9.81  # rough km/h -> g
    decel_g = round(max(
        phone.accel_peak_g * 0.4,  # partial weight — phone accel isn't pure decel
        vehicle_decel_g,
    ), 2)

    speed_delta = vehicle.speed_change_kmh  # prefer vehicle ECU speed

    return {
        "speed_kmh": max(0.0, 50.0 + speed_delta),  # approximate current speed
        "speed_delta_kmh": speed_delta,
        "deceleration_g": min(decel_g, 2.0),
        "impact_score": vehicle.impact_score,
        "lane_deviation_m": 0.5 if abs(phone.gyro_deg_s) >= 60 else 0.1,
        "location_label": phone.location_label,
    }
