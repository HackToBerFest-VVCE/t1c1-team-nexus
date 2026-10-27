"""
AccidentGuard — Safety layer
============================

The safety layer makes the FINAL severity decision. The rule engine and the
Gemma 4 assessment are both inputs, but Python (not the model) decides the
final severity. This is deliberate: an AI suggestion can raise a concern, but
fixed, auditable rules cap or floor the result so the system is never silently
misled.

Fixed rules (illustrative, auditable — NOT calibrated against real data):
  * If there is no impact AND no sharp deceleration, the final severity is
    NEVER above "medium".
  * If impact_score > 0.8 with very hard deceleration (>= 0.70 g), the final
    severity is AT LEAST "high" (at least "critical" if impact_score >= 0.9).
  * If Gemma and the rules disagree, we use the SAFER (higher) validated result
    that still respects the fixed rules, and we flag "disagreement handled".

Alert rule: "low" severity means NO alert ("no alert needed").
"""

from __future__ import annotations

from accident_detection import Severity, SUSPICIOUS_DECEL_G, SUSPICIOUS_IMPACT, ACCIDENT_DECEL_G

RANK = {"low": 1, "medium": 2, "high": 3, "critical": 4}
RANK_TO_SEV = {v: k for k, v in RANK.items()}


def _sev_str(v):
    """Normalise a severity from an Enum, a plain string, or None."""
    if v is None:
        return None
    if hasattr(v, "value"):          # Severity enum -> "medium" etc.
        v = v.value
    s = str(v).replace("Severity.", "").strip().lower()
    return s if s in RANK else None


def _reading(readings, key, default=0.0):
    """Read a numeric field from a DrivingReadings or a dict."""
    if readings is None:
        return default
    if isinstance(readings, dict):
        return float(readings.get(key, default))
    return float(getattr(readings, key, default))


def decide_final_severity(rule_severity, readings, gemma_severity=None) -> dict:
    """
    Compute the final severity.

    Args:
        rule_severity: severity from the rule engine (str or Severity).
        readings: DrivingReadings or dict with impact_score / deceleration_g.
        gemma_severity: severity returned by Gemma (str) or None when unavailable.

    Returns:
        {
          "final_severity": "low|medium|high|critical",
          "rule_severity": str,
          "gemma_severity": str|None,
          "disagreement": bool,
          "note": str,                 # human-readable explanation
          "alert_needed": bool,        # False for low severity
        }
    """
    rule = _sev_str(rule_severity) or "low"
    gemma = _sev_str(gemma_severity)

    impact = _reading(readings, "impact_score", 0.0)
    decel = _reading(readings, "deceleration_g", 0.0)

    no_impact = impact <= SUSPICIOUS_IMPACT          # no notable impact
    no_sharp_decel = decel < SUSPICIOUS_DECEL_G      # no sudden braking
    very_hard_decel = decel >= ACCIDENT_DECEL_G

    # Start from the safer (higher) of rule and Gemma.
    candidates = [RANK[rule]]
    if gemma is not None:
        candidates.append(RANK[gemma])
    final_rank = max(candidates)

    # Fixed-rule CEILING: no impact and no sharp deceleration -> never above medium.
    if no_impact and no_sharp_decel:
        final_rank = min(final_rank, RANK["medium"])

    # Fixed-rule FLOOR: strong impact + very hard deceleration -> at least high.
    if impact > 0.8 and very_hard_decel:
        final_rank = max(final_rank, RANK["high"])
        if impact >= 0.9:
            final_rank = max(final_rank, RANK["critical"])

    final = RANK_TO_SEV[final_rank]

    # Disagreement is between the rule's value and Gemma's value (Python then
    # picks the safer validated result that respects the fixed rules).
    disagreement = gemma is not None and gemma != rule
    if disagreement:
        note = ("disagreement handled — used safer validated result "
                 f"({final}); Gemma suggested {gemma}")
    elif gemma is not None:
        note = f"rules and Gemma agree ({final})"
    else:
        note = f"rule-based result ({final}) — Gemma unavailable"

    return {
        "final_severity": final,
        "rule_severity": rule,
        "gemma_severity": gemma,
        "disagreement": disagreement,
        "note": note,
        "alert_needed": final != "low",
    }
