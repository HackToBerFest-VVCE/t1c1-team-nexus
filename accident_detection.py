"""
AccidentGuard — Simulated Event Classification Engine
=====================================================

This module generates and classifies SIMULATED driving events using transparent,
rule-based logic. It contains NO real sensors, GPS hardware, or live data.

Two important boundaries:
  1. This is a simulation. The readings are illustrative, not measured.
  2. Classification here is a heuristic, NOT an accident verdict. The real
     interpretation is done later by the Gemma 4 model (see gemma_service.py),
     which is explicitly told it must not invent sensor data or claim certainty
     about a real accident.

The engine is intentionally simple and readable so a beginner can follow the
rules and the test cases.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from enum import Enum
from typing import Optional


# ---------------------------------------------------------------------------
# Categories and severity
# ---------------------------------------------------------------------------

class EventCategory(str, Enum):
    """The three classification buckets required by the project."""
    NORMAL = "normal_driving"
    SUSPICIOUS = "suspicious_event"
    POSSIBLE_ACCIDENT = "possible_accident"


class Severity(str, Enum):
    """Incident severity indicator shown on the dashboard."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


# ---------------------------------------------------------------------------
# Rule thresholds (illustrative, rule-based — NOT calibrated against real data)
# ---------------------------------------------------------------------------
# These constants make the rules auditable. Change them here and the behaviour
# of classify_event() updates everywhere, including the tests.

SUSPICIOUS_DECEL_G = 0.45      # g-force of sudden braking that looks odd
ACCIDENT_DECEL_G = 0.70        # very hard braking

SUSPICIOUS_IMPACT = 0.35       # impact-score units (0..1 scale, simulated)
ACCIDENT_IMPACT = 0.60

SUSPICIOUS_LANE_DEV_M = 0.60   # lane deviation in metres
ACCIDENT_LANE_DEV_M = 1.00

SUSPICIOUS_SPEED_DELTA_KMH = 25.0   # sudden speed drop
ACCIDENT_SPEED_DELTA_KMH = 40.0

# Clearly-labelled simulated locations (NO real GPS).
SIMULATED_LOCATIONS = [
    "Simulated · MG Road, Bengaluru",
    "Simulated · Outer Ring Road, Bengaluru",
    "Simulated · Highway NH-48 (simulated stretch)",
    "Simulated · Jayanagar 4th Block, Bengaluru",
    "Simulated · Airport Flyover (simulated)",
    "Simulated · Residential Area, Indiranagar",
]


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class DrivingReadings:
    """Simulated sensor-like readings for one driving event."""
    timestamp: str                       # ISO-8601 UTC
    speed_kmh: float                     # current speed
    speed_delta_kmh: float               # sudden change in speed (negative = drop)
    deceleration_g: float                # braking intensity in g
    impact_score: float                  # 0..1 simulated impact indicator
    lane_deviation_m: float              # how far off the lane centre (metres)
    location_label: str                  # clearly labelled simulated location
    event_id: str = ""                   # filled in by the generator

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class EventClassification:
    """The engine's output: the readings plus the rule-based verdict."""
    event_id: str
    timestamp: str
    category: EventCategory
    severity: Severity
    readings: DrivingReadings
    rule_explanation: str               # human-readable, NON-AI justification
    triggered_rules: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "event_id": self.event_id,
            "timestamp": self.timestamp,
            "category": self.category.value,
            "severity": self.severity.value,
            "readings": self.readings.to_dict(),
            "rule_explanation": self.rule_explanation,
            "triggered_rules": self.triggered_rules,
            # Frontend hint so the UI can never confuse engine vs. AI output.
            "source": "rule_engine",
        }


# ---------------------------------------------------------------------------
# Classification logic
# ---------------------------------------------------------------------------

def classify_event(readings: DrivingReadings) -> EventClassification:
    """
    Classify a DrivingReadings object using transparent, auditable rules.

    The function is deterministic given the same readings. It returns a
    rule_explanation string that the dashboard shows SEPARATELY from the
    Gemma-generated summary, so it is always clear which is which.
    """
    triggered: list[str] = []

    # Rule A: hard deceleration
    if readings.deceleration_g >= ACCIDENT_DECEL_G:
        triggered.append(f"Very hard deceleration ({readings.deceleration_g:.2f} g)")
    elif readings.deceleration_g >= SUSPICIOUS_DECEL_G:
        triggered.append(f"Sudden deceleration ({readings.deceleration_g:.2f} g)")

    # Rule B: impact score
    if readings.impact_score >= ACCIDENT_IMPACT:
        triggered.append(f"High impact score ({readings.impact_score:.2f})")
    elif readings.impact_score >= SUSPICIOUS_IMPACT:
        triggered.append(f"Notable impact score ({readings.impact_score:.2f})")

    # Rule C: lane deviation
    if readings.lane_deviation_m >= ACCIDENT_LANE_DEV_M:
        triggered.append(f"Large lane deviation ({readings.lane_deviation_m:.2f} m)")
    elif readings.lane_deviation_m >= SUSPICIOUS_LANE_DEV_M:
        triggered.append(f"Lane deviation ({readings.lane_deviation_m:.2f} m)")

    # Rule D: speed drop
    if abs(readings.speed_delta_kmh) >= ACCIDENT_SPEED_DELTA_KMH:
        triggered.append(f"Sharp speed change ({readings.speed_delta_kmh:.1f} km/h)")
    elif abs(readings.speed_delta_kmh) >= SUSPICIOUS_SPEED_DELTA_KMH:
        triggered.append(f"Speed change ({readings.speed_delta_kmh:.1f} km/h)")

    # Decide the category from how many "strong" signals fired.
    # An accident needs at least one ACCIDENT-level signal AND one more signal.
    accident_signals = (
        readings.deceleration_g >= ACCIDENT_DECEL_G
        or readings.impact_score >= ACCIDENT_IMPACT
        or readings.lane_deviation_m >= ACCIDENT_LANE_DEV_M
        or abs(readings.speed_delta_kmh) >= ACCIDENT_SPEED_DELTA_KMH
    )
    suspicious_signals = (
        readings.deceleration_g >= SUSPICIOUS_DECEL_G
        or readings.impact_score >= SUSPICIOUS_IMPACT
        or readings.lane_deviation_m >= SUSPICIOUS_LANE_DEV_M
        or abs(readings.speed_delta_kmh) >= SUSPICIOUS_SPEED_DELTA_KMH
    )

    if accident_signals and len(triggered) >= 2:
        category = EventCategory.POSSIBLE_ACCIDENT
        severity = Severity.CRITICAL if readings.impact_score >= ACCIDENT_IMPACT else Severity.HIGH
    elif suspicious_signals:
        category = EventCategory.SUSPICIOUS
        severity = Severity.MEDIUM
    else:
        category = EventCategory.NORMAL
        severity = Severity.LOW

    if not triggered:
        explanation = "All simulated readings within normal ranges. No rule triggered."
    else:
        explanation = "Rule engine flagged: " + "; ".join(triggered) + "."

    return EventClassification(
        event_id=readings.event_id,
        timestamp=readings.timestamp,
        category=category,
        severity=severity,
        readings=readings,
        rule_explanation=explanation,
        triggered_rules=triggered,
    )


# ---------------------------------------------------------------------------
# Simulated data generation
# ---------------------------------------------------------------------------

def _now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _make_event_id() -> str:
    return f"EVT-{random.randint(10000, 99999)}"


def generate_sample_event(
    category: Optional[EventCategory] = None,
    seed: Optional[int] = None,
) -> EventClassification:
    """
    Generate a single simulated driving event and classify it.

    Args:
        category: Force a specific category. If None, one is chosen at random.
        seed: Optional RNG seed for reproducible events (used by tests).

    Returns:
        An EventClassification ready to show on the dashboard and send to Gemma.
    """
    if seed is not None:
        random.seed(seed)

    if category is None:
        category = random.choices(
            list(EventCategory),
            weights=[0.6, 0.25, 0.15],  # mostly normal driving
        )[0]

    # Build readings consistent with the requested category.
    if category == EventCategory.NORMAL:
        readings = DrivingReadings(
            timestamp=_now_iso(),
            speed_kmh=round(random.uniform(30, 70), 1),
            speed_delta_kmh=round(random.uniform(-5, 5), 1),
            deceleration_g=round(random.uniform(0.05, 0.3), 2),
            impact_score=round(random.uniform(0.0, 0.2), 2),
            lane_deviation_m=round(random.uniform(0.0, 0.3), 2),
            location_label=random.choice(SIMULATED_LOCATIONS),
        )
    elif category == EventCategory.SUSPICIOUS:
        readings = DrivingReadings(
            timestamp=_now_iso(),
            speed_kmh=round(random.uniform(35, 80), 1),
            speed_delta_kmh=round(random.uniform(-32, -18), 1),   # moderate drop
            deceleration_g=round(random.uniform(SUSPICIOUS_DECEL_G, ACCIDENT_DECEL_G), 2),
            impact_score=round(random.uniform(SUSPICIOUS_IMPACT, ACCIDENT_IMPACT), 2),
            lane_deviation_m=round(random.uniform(SUSPICIOUS_LANE_DEV_M, ACCIDENT_LANE_DEV_M), 2),
            location_label=random.choice(SIMULATED_LOCATIONS),
        )
    else:  # POSSIBLE_ACCIDENT
        readings = DrivingReadings(
            timestamp=_now_iso(),
            speed_kmh=round(random.uniform(40, 90), 1),
            speed_delta_kmh=round(random.uniform(-55, -38), 1),   # sharp drop
            deceleration_g=round(random.uniform(ACCIDENT_DECEL_G, 1.1), 2),
            impact_score=round(random.uniform(ACCIDENT_IMPACT, 0.95), 2),
            lane_deviation_m=round(random.uniform(ACCIDENT_LANE_DEV_M, 1.6), 2),
            location_label=random.choice(SIMULATED_LOCATIONS),
        )

    readings.event_id = _make_event_id()
    return classify_event(readings)


def event_for_ai(classification: EventClassification) -> dict:
    """
    Build the compact, privacy-safe payload we send to Gemma.
    We intentionally do NOT send personal data — only simulated readings.
    """
    r = classification.readings
    return {
        "event_id": classification.event_id,
        "timestamp": classification.timestamp,
        "category": classification.category.value,
        "severity": classification.severity.value,
        "simulated_readings": {
            "speed_kmh": r.speed_kmh,
            "speed_delta_kmh": r.speed_delta_kmh,
            "deceleration_g": r.deceleration_g,
            "impact_score": r.impact_score,
            "lane_deviation_m": r.lane_deviation_m,
            "location_label": r.location_label,
        },
        "rule_engine_notes": classification.rule_explanation,
    }


def generate_from_streams(phone_stream, vehicle_stream,
                          legacy_readings: dict) -> EventClassification:
    """
    Classify an event that was generated from dual sensor streams.
    `legacy_readings` is the dict produced by sensor_fusion.streams_to_legacy_readings().
    This keeps the rule engine working unchanged on backward-compatible fields.
    """
    r = DrivingReadings(
        timestamp=_now_iso(),
        speed_kmh=legacy_readings.get("speed_kmh", 50.0),
        speed_delta_kmh=legacy_readings.get("speed_delta_kmh", 0.0),
        deceleration_g=legacy_readings.get("deceleration_g", 0.0),
        impact_score=legacy_readings.get("impact_score", 0.0),
        lane_deviation_m=legacy_readings.get("lane_deviation_m", 0.0),
        location_label=legacy_readings.get("location_label", "Simulated · Unknown"),
    )
    r.event_id = _make_event_id()
    return classify_event(r)
