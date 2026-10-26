"""
AccidentGuard — manual Gemma 4 integration check
=================================================

A standalone script (no pytest needed) that PROVES the real Gemma 4 API call
succeeds. Run it after adding your key to .env:

    python tests/manual_integration_check.py

Expected output when a valid key is set:
    STEP 1 ... PASS  (real Gemma ping returned a response)
    STEP 2 ... PASS  (Gemma summarised a simulated accident event)
    MODEL USED: gemma-4-26b-a4b-it
    RESULT: CONNECTED (real Gemma response received and displayed)

This file makes REAL network calls and requires GEMINI_API_KEY.
"""

import json
import os
import sys

# Make the project root importable.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

import accident_detection as det  # noqa: E402
import gemma_service as gs  # noqa: E402


def line(s=""):
    print(s)


def main() -> int:
    print("=" * 64)
    print(" AccidentGuard — manual Gemma 4 integration check")
    print("=" * 64)
    print(f" Configured model : {gs.get_model_info()['model']}")
    print(f" Verified Gemma 4 : {gs.get_model_info()['verified_supported']}")
    print(f" API key present   : {gs.is_configured()}")

    if not gs.is_configured():
        line()
        print(" ✗ GEMINI_API_KEY is NOT set.")
        print("   Create a .env file (copy .env.example) and add your key from")
        print("   Google AI Studio: https://aistudio.google.com/app/apikey")
        print("   Then re-run this script.")
        return 1

    failures = 0

    # --- STEP 1: trivial real ping ---------------------------------------
    line()
    print(" STEP 1 — real Gemma ping (quick_self_test)")
    ping = gs.quick_self_test()
    if ping["status"] == "connected" and ping.get("response"):
        print("   ✓ PASS — Gemma responded:", repr(ping["response"])[:80])
        print("   model used:", ping["model"])
    else:
        failures += 1
        print("   ✗ FAIL — status:", ping["status"], "| error:", ping.get("error"))

    # --- STEP 2: full incident summary -----------------------------------
    line()
    print(" STEP 2 — Gemma summarises a simulated possible-accident event")
    ev = det.generate_sample_event(category=det.EventCategory.POSSIBLE_ACCIDENT, seed=11)
    payload = det.event_for_ai(ev)
    result = gs.generate_incident_summary(payload)

    print("   event category :", payload["category"])
    print("   AI source       :", result["source"])
    print("   AI status       :", result["status"])
    print("   used fallback   :", result["used_fallback"])
    print("   model           :", result["model"])
    line()
    print("   --- Gemma summary ---")
    print("   " + (result["summary"] or "(empty)").replace("\n", "\n   "))
    if result.get("recommendations"):
        print("   --- Recommendations ---")
        for r in result["recommendations"]:
            print("   - " + r)
    if result.get("severity_opinion"):
        print("   Gemma severity opinion:", result["severity_opinion"])

    if result["status"] == "connected" and result["source"] == "gemma":
        line()
        print("   ✓ PASS — a REAL Gemma response was received and parsed.")
    else:
        failures += 1
        line()
        print("   ✗ FAIL — did not receive a real Gemma response.")
        print("   error:", result.get("error"))

    # --- Verdict ----------------------------------------------------------
    line()
    print("=" * 64)
    if failures == 0:
        print(" RESULT: CONNECTED — real Gemma 4 integration confirmed.")
        print(" MODEL USED:", gs.get_model_info()["model"])
        print(" You can show this output as evidence of genuine integration.")
        return 0
    print(f" RESULT: {failures} check(s) FAILED. Review the error above.")
    print(" Common causes: invalid/expired key, no network, model id issue.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
