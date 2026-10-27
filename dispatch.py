"""
AccidentGuard — Dispatch (simulation)
=====================================

Simulated ambulance units, nearest-unit selection (Haversine), escalation order,
and incident status-transition validation.

IMPORTANT: This is a SIMULATION. Units, coordinates and "emergency contact
notified" messages are illustrative. No real emergency calls are made.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

# ---------------------------------------------------------------------------
# Simulated location coordinates (Bengaluru area) — match the labels used by the
# event engine. These are for map illustration only, not real GPS tracking.
# ---------------------------------------------------------------------------

DEFAULT_COORD = (12.9716, 77.5946)  # Bengaluru centre

LOCATION_COORDS = {
    "Simulated · MG Road, Bengaluru": (12.9756, 77.6066),
    "Simulated · Outer Ring Road, Bengaluru": (12.9350, 77.6260),
    "Simulated · Highway NH-48 (simulated stretch)": (12.9440, 77.5460),
    "Simulated · Jayanagar 4th Block, Bengaluru": (12.9250, 77.5940),
    "Simulated · Airport Flyover (simulated)": (13.1980, 77.7060),
    "Simulated · Residential Area, Indiranagar": (12.9719, 77.6412),
}


def coords_for(location_label: str):
    """Return (lat, lng) for a simulated location label, with a safe default."""
    if location_label in LOCATION_COORDS:
        return LOCATION_COORDS[location_label]
    return DEFAULT_COORD


# ---------------------------------------------------------------------------
# Simulated ambulance units (5, around the same area as the locations).
# ---------------------------------------------------------------------------

@dataclass
class Unit:
    name: str
    lat: float
    lng: float
    available: bool = True
    current_incident: str | None = None

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "lat": self.lat,
            "lng": self.lng,
            "available": self.available,
            "current_incident": self.current_incident,
        }


UNITS = [
    Unit("Ambulance 108-A", 12.9700, 77.5950),
    Unit("Ambulance 108-B", 12.9900, 77.6150),
    Unit("Ambulance 108-C", 12.9550, 77.5750),
    Unit("Ambulance 108-D", 13.0000, 77.5850),
    Unit("Ambulance 108-E", 12.9450, 77.6050),
]


def all_units() -> list[Unit]:
    return UNITS


# ---------------------------------------------------------------------------
# Geometry
# ---------------------------------------------------------------------------

def haversine_km(a, b) -> float:
    """Distance in km between two (lat, lng) points."""
    lat1, lon1 = a
    lat2, lon2 = b
    r = 6371.0
    p1 = math.radians(lat1)
    p2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlam = math.radians(lon2 - lon1)
    h = math.sin(dphi / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dlam / 2) ** 2
    return 2 * r * math.asin(math.sqrt(h))


def sorted_units_by_distance(units: list[Unit], location) -> list[Unit]:
    """Return units sorted by distance to the incident location (ascending)."""
    return sorted(units, key=lambda u: haversine_km((u.lat, u.lng), location))


def next_unit_to_offer(units: list[Unit], location, tried: set[str]) -> Unit | None:
    """
    Nearest AVAILABLE unit not already tried (for escalation).
    Returns None when no available unit remains.
    """
    for u in sorted_units_by_distance(units, location):
        if u.available and u.name not in tried:
            return u
    return None


def nearest_available_unit(units: list[Unit], location) -> Unit | None:
    """Convenience: the single nearest available unit."""
    return next_unit_to_offer(units, location, tried=set())


# ---------------------------------------------------------------------------
# Incident status transitions
# ---------------------------------------------------------------------------

# Lifecycle (terminal states have no outgoing transitions):
#   new -> awaiting_victim | alerted | resolved | cancelled
#   awaiting_victim -> alerted | cancelled
#   alerted -> assigned | escalated | cancelled | resolved
#   escalated -> assigned | resolved | cancelled
#   assigned -> en_route | resolved | cancelled
#   en_route -> arrived | resolved
#   arrived -> resolved
VALID_TRANSITIONS = {
    "new": {"awaiting_victim", "alerted", "resolved", "cancelled"},
    "awaiting_victim": {"alerted", "cancelled"},
    "alerted": {"assigned", "escalated", "cancelled", "resolved"},
    "escalated": {"assigned", "resolved", "cancelled"},
    "assigned": {"en_route", "resolved", "cancelled"},
    "en_route": {"arrived", "resolved"},
    "arrived": {"resolved"},
    "resolved": set(),
    "cancelled": set(),
}

ALL_STATUSES = sorted(VALID_TRANSITIONS.keys())


def validate_transition(current: str, new: str):
    """
    Returns (ok: bool, error: str|None).
    """
    if current not in VALID_TRANSITIONS:
        return False, f"Unknown current status: {current}"
    if new not in VALID_TRANSITIONS:
        return False, f"Unknown target status: {new}"
    if current == new:
        return True, None  # idempotent
    if new in VALID_TRANSITIONS[current]:
        return True, None
    return False, (
        f"Invalid transition: {current} -> {new}. "
        f"Allowed: {sorted(VALID_TRANSITIONS[current]) or 'none (terminal)'}"
    )
