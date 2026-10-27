"""
AccidentGuard — Manual Gemma 4 integration check
=================================================

Run after adding GEMINI_API_KEY to .env:
    python tests/manual_integration_check.py

Expected output with a valid key:
    STEP 1 ... PASS  (real Gemma ping returned a response)
    STEP 2 ... PASS  (Call 1: Gemma assessed a simulated accident event)
    STEP 3 ... PASS  (Call 2: Gemma updated the responder briefing)
    MODEL USED: gemma-4-26b-a4b-it
    RESULT: CONNECTED — real Gemma 4 integration confirmed.
"""

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

try:
    from dotenv import load_dotenv
    load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".env"),
                override=True)
except Exception:
    pass

import json
import accident_detection as det
import gemma_service as gs
import sensor_fusion as sf


def line(s=""):
    print(s)


def main() -> int:
    print("=" * 64)
    print(" AccidentGuard — manual Gemma 4 integration check")
    print(" Team Nexus · Team 11 · Hacktoberfest × VectorFlow 2026")
    print("=" * 64)
    print(f" Configured model : {gs.get_model_info()['model']}")
    print(f" Verified Gemma 4 : {gs.get_model_info()['verified_supported']}")
    print(f" API key present  : {gs.is_configured()}")
    if not gs.is_configured():
        line()
        print(" ✗ GEMINI_API_KEY is NOT set.")
        print("   Create .env from .env.example and add your key from")
        print("   Google AI Studio: https://aistudio.google.com/app/apikey")
        print("   RESULT: FAIL — reason: no key found")
        return 1

    failures = 0

    # STEP 1: tiny real ping
    line()
    print(" STEP 1 — real Gemma ping (check_connection)")
    ping = gs.check_connection()
    if ping["status"] == "connected" and ping.get("response"):
        print(f"   ✓ PASS — Gemma responded: {repr(ping['response'])[:80]}")
        print(f"   model used: {ping['model']}")
    else:
        failures += 1
        print(f"   ✗ FAIL — status: {ping['status']} | reason: {ping.get('reason')} | {ping.get('detail','')[:80]}")

    # STEP 2: Call 1 — incident assessment
    line()
    print(" STEP 2 — Call 1: Gemma assesses a simulated possible-accident event")
    ev = det.generate_sample_event(category=det.EventCategory.POSSIBLE_ACCIDENT, seed=11)
    phone, vehicle = sf.generate_streams("real_crash", seed=11)
    fusion = sf.fuse(phone, vehicle)
    filt = sf.false_alarm_filter(phone, vehicle, fusion)
    payload = det.event_for_ai(ev)
    result = gs.generate_incident_summary(payload, fusion=fusion.to_dict(), filter_res=filt.to_dict())
    print(f"   event category : {payload['category']}")
    print(f"   AI source      : {result['source']}")
    print(f"   AI status      : {result['status']}")
    print(f"   used fallback  : {result['used_fallback']}")
    print(f"   model          : {result['model']}")
    print(f"   severity       : {result.get('severity_opinion')}")
    print(f"   confidence     : {result.get('confidence')}")
    print()
    print(f"   action         : {(result.get('action',''))[:120]}")
    if result.get("evidence"):
        print(f"   evidence       : {result['evidence'][:2]}")
    if result.get("uncertainty"):
        print(f"   uncertainty    : {result['uncertainty'][:80]}")
    if result.get("responder_briefing"):
        print(f"   briefing       : {result['responder_briefing'][:2]}")
    if result.get("alert_en"):
        print(f"   alert_en       : {result['alert_en'][:80]}")

    if result["status"] == "connected" and result["source"] == "gemma":
        print("   ✓ PASS — real Gemma Call 1 response received and parsed.")
    else:
        failures += 1
        print(f"   ✗ FAIL — {result.get('error','')[:80]}")

    # STEP 3: Call 2 — dispatch briefing update
    line()
    print(" STEP 3 — Call 2: Gemma updates responder briefing at dispatch")
    b2 = gs.generate_dispatch_briefing(
        assessment=result,
        victim_answer=None,
        seconds_without_response=12.0,
        nearest_unit="Ambulance 108-A",
        eta_min=6,
    )
    print(f"   source         : {b2.get('source')}")
    print(f"   briefing       : {b2.get('responder_briefing',['—'])[:3]}")
    if b2.get("note"):
        print(f"   note           : {b2['note'][:80]}")
    if b2.get("source") == "gemma":
        print("   ✓ PASS — real Gemma Call 2 response received.")
    else:
        failures += 1
        print("   ✗ FAIL — Call 2 returned fallback (check key / quota).")

    line()
    print("=" * 64)
    if failures == 0:
        print(" RESULT: CONNECTED — real Gemma 4 integration confirmed.")
        print(f" MODEL USED: {gs.get_model_info()['model']}")
        print(" You can show this output as evidence of genuine integration.")
        return 0
    print(f" RESULT: FAIL — {failures} check(s) failed.")
    print(f" Reason: {gs.get_status().get('reason','unknown')}")
    print(" Common causes: invalid/expired key, quota exceeded, network error.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
