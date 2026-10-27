# AccidentGuard — AI-Assisted Accident Detection & Emergency Alert Platform

**Team Nexus · Team 11 · Hacktoberfest × VectorFlow 2026 · Challenge A: Best Use of Gemma 4**

*Sensors detect. Gemma 4 interprets and communicates. Gemma 4 reasons. Python decides.*

AccidentGuard simulates the full chain from dual-sensor data to ambulance dispatch — driven by real Google Gemma 4 API calls — on a responsive three-screen web app. All data is simulated. No real sensors, GPS, emergency calls or ambulances.

---

## Full pipeline (7 steps)

```
Sensor data → Python validation → Sensor fusion → False-alarm filter →
Gemma 4 → Python safety layer → Victim check → Dispatch → Dashboard
```

1. **Sensor data** — Two simulated streams per event:
   - *Phone*: accelerometer peak (g), gyroscope (deg/s), GPS speed change, GPS location label.
   - *Vehicle/IoT*: impact sensor score (0–1), airbag/crash-sensor flag, post-impact movement, ECU speed change.
2. **Python validation** — Sanity-checks ranges before the pipeline proceeds.
3. **Sensor fusion** — Counts how many of 6 independent signals agree on an impact. Shows "Signals agreeing: N of 6" with chips. A possible accident needs ≥3 signals including at least one vehicle or stop signal.
4. **False-alarm filter** — Rule-based Python filter rejects known false-alarm patterns *before* calling Gemma:
   - **Phone drop** → phone accel high, vehicle quiet, no GPS speed change → rejected.
   - **Pothole** → moderate vehicle impact, no crash flag, low phone accel → rejected.
   - **Hard braking** → high decel, no impact signal → rejected.
   - **Real crash** → impact + sudden stop + no movement → passes.
   - Rejected events are still logged in history with a "Rejected" tag.
5. **Gemma 4** — Two real API calls (see below). Falls back to a clearly-labelled rule-based template when the API is unavailable.
6. **Python safety layer** — Makes the **FINAL** severity decision with fixed, auditable rules. If Gemma and the rules disagree, the safer validated result is used and "disagreement handled" is flagged.
7. **Victim check → Dispatch** — Medium/high: 10-second countdown. Critical: skips countdown and alerts immediately. Nearest available ambulance offered first (Haversine); escalates after 20 s.

---

## How we use Gemma 4

**Model: `gemma-4-26b-a4b-it`** (verified against the [official Gemini API docs](https://ai.google.dev/gemma/docs/core/gemma_on_gemini_api); falls back to `gemma-4-31b-it`). Never substitutes Gemini or another model.

**Call 1 — Incident assessment.** Gemma receives: event category, rule-engine severity, simulated readings, fusion signal count, and filter result.
Returns only a JSON object: `{severity, confidence, evidence[], uncertainty, action, alert_en, alert_kn, alert_hi, responder_briefing[3]}`.

**Call 2 — Dispatch briefing update.** Triggered when a unit accepts. Gemma receives the Call 1 assessment, victim response status, elapsed seconds, and the unit name with ETA. Returns an updated 3-line responder briefing. If Call 2 fails, the Call 1 briefing is reused and labelled.

**Optional Call 3 — Crash photo analysis.** The dashboard and victim screen allow uploading a simulated bystander photo. Gemma analyses it and returns: vehicles, visible injuries, fire/fuel risk, road blocked, notes, confidence. If the model rejects images, the result shows "Scene analysis unavailable for this model" and the pipeline continues normally; no invented scene descriptions.

**Python makes the final decision.** Gemma can raise or lower severity concerns, but the safety layer's fixed rules always have the last word.

**Gemma ON/OFF toggle.** When OFF, the system uses a rule-based template labelled "Non-AI fallback" — useful to show judges the difference.

---

## Exact Gemma 4 model ID used

`gemma-4-26b-a4b-it` — documented fallback: `gemma-4-31b-it`.  
SDK: `from google import genai; client.models.generate_content(model=..., ...)` (new `google-genai` SDK, not the legacy `google-generativeai`).

---

## Three screens

- **Dashboard (`/`)** — Scenario selector (Normal, Suspicious, Possible accident, Real crash, Phone drop, Pothole, Hard braking, Random), quick-demo buttons (Real crash / Phone drop), 7-step pipeline strip with one-line result per step, phone + vehicle sensor panels, rule-engine verdict, Gemma assessment card (evidence, uncertainty, confidence, first-aid tips in EN/KN/HI, Call 2 briefing, scene analysis), safety + incident panel with victim-check mirror, active incidents, history, stats.
- **Victim screen (`/victim`, mobile-first)** — Safe by default. On incident: full-screen "Are you OK?" with 10-second countdown ring, **I'M SAFE** and **I NEED HELP** buttons, beep + vibration, then post-alert timeline, EN/KN/HI first-aid tabs, simulated location + unit ETA. Never shows evidence, confidence or sensor numbers.
- **Ambulance screen (`/ambulance`, desktop)** — "Viewing as unit" dropdown, auto-selects highest-severity incident, incident list with escalation countdown, Victim analysis card (severity, confidence bar, evidence, uncertainty, action, scene analysis, 3-line briefing, fusion signals, safety-layer note, victim response status, disclaimer), SVG map with all units + incident pin + nearest-unit line, Ambulance services table (availability, distance km, ETA min, nearest first), Accept/En route/Arrived/Resolved buttons (transitions enforced), counters.

---

## Installation & setup

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows; macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
copy .env.example .env          # macOS/Linux: cp .env.example .env
```

Edit `.env` and set:
```
GEMINI_API_KEY=your_real_key_here
GEMMA_MODEL_ID=gemma-4-26b-a4b-it
```
Get a free key at: https://aistudio.google.com/app/apikey

The key is loaded with `load_dotenv(..., override=True)`, stripped of whitespace/quotes, and **never** sent to the browser, logged, or included in error messages.

## How to run

```bash
python app.py
# Open: http://127.0.0.1:5000 | /victim | /ambulance
```

## How to test

```bash
# Mock tests (no key needed) — 70+ pass
python -m pytest -q

# Real Gemma 4 integration (needs key in .env; skip without a key)
python -m pytest tests/test_gemma_integration.py -v

# Standalone proof-of-integration (prints CONNECTED or FAIL with reason)
python tests/manual_integration_check.py
```

---

## 2-minute demo script

**Setup:** add key to `.env`, run `python tests/manual_integration_check.py` → leave `RESULT: CONNECTED` visible.

1. **(30 s) Phone drop gets rejected** — Select *Phone drop*, click Generate Event. Pipeline strip shows "False-alarm filter: FAILED". No victim countdown, no dispatch. History shows "Rejected: phone drop".

2. **(30 s) Real crash passes** — Select *Real crash*, click Generate. Pipeline lights up all 7 steps. Fusion shows "6 of 6 signals". Gemma 4 card shows real evidence, uncertainty, action. Safety layer: "agreed". Dispatch: auto-assigned, ETA shown.

3. **(15 s) Victim screen asks "Are you OK?"** — Select *Suspicious event*, Generate. Open `/victim`. Full-screen countdown ring appears with I'M SAFE / I NEED HELP buttons.

4. **(15 s) No answer → alert sent** — Let the 10 s countdown expire. Timeline on victim screen shows: Alert sent → Ambulance assigned → On the way with ETA.

5. **(15 s) Ambulance screen** — Open `/ambulance`. Incident auto-selected. Ambulance services table shows distances and ETAs. SVG map shows unit-to-incident line. Click Accept (if auto-accept is OFF).

6. **(15 s) Gemma ON vs OFF** — Toggle *Gemma 4 OFF*, generate a Real crash. Assessment card shows "Non-AI fallback". Toggle it back ON — Gemma gives real evidence and briefing.

---

## Limitations (simulation only)

- No real sensors, hardware, GPS tracking, or emergency service integration.
- All sensor readings, ambulance positions, and "emergency contact notified" messages are illustrative.
- Rule thresholds are not calibrated against real crash data.
- In-memory state resets on server restart.
- AI responses are real but the Gemma model is constrained to not claim certainty about real accidents.
- Not for real-world or safety-critical use.

## Security

- API key in `.env` only; `.gitignore` covers `.env`; `.env.example` contains only placeholders.
- Key is never sent to the browser, logged, or included in error messages.
- Key is redacted in all error strings before they leave Python.
