# AccidentGuard — AI-Powered Accident Detection & Alert System

**Project summary**

- AccidentGuard turns simulated driving telemetry into clear incident understanding.
- A transparent Python rule engine classifies each event — normal driving, suspicious event, or possible accident — with a severity indicator.
- The real **Google Gemma 4** model then writes a plain-language summary, flags details to watch, and suggests cautious next steps.
- A responsive dark dashboard shows the rule-engine verdict and the AI summary separately, with a live AI connection status.
- This is a hackathon simulation prototype: no real sensors, GPS, or emergency notifications.

This project is a **hackathon prototype**. It uses **illustrative, simulated data
only** — there are no real sensors, no live GPS, and no emergency-service
notifications. The real AI integration is genuine: the app makes an actual API
call to the hosted Gemma 4 model and displays the model's real response.

---

## Problem statement & proposed solution

**Problem:** After a road incident, the raw telemetry (sudden deceleration,
impact scores, lane deviation) is hard for a non-expert to interpret. People need
a quick, plain-language understanding of what the data suggests and what cautious
next steps make sense — without falsely claiming a real accident occurred.

**Solution:** AccidentGuard demonstrates a three-stage pipeline:

1. **Simulated event generation** — produces labelled driving readings (no real
   hardware).
2. **Transparent rule-based classification** — a simple, auditable engine sorts
   each event into *normal driving*, *suspicious event*, or *possible accident*,
   with a severity indicator. This logic is clearly separated from the AI.
3. **Gemma 4 interpretation** — the real Gemma model receives the (privacy-safe)
   event data and returns a summary, points to watch, and cautious recommendations.
   It is constrained to avoid inventing readings or claiming certainty.

---

## Features & architecture

**Features**
- Start Demo and Generate Sample Event controls (with event-type selector).
- Simulated event classification with severity indicator (low/medium/high/critical).
- Event timestamp and clearly-labelled *simulated* location.
- Real **Gemma 4** incident summary, attention points, recommendations, and
  severity opinion.
- AI connection status indicator with three states:
  **Connected** (real Gemma response received), **Error** (request failed),
  **Not configured** (API key missing). The UI never shows "Connected" unless a
  real response was received.
- Incident history with per-event AI badge (Gemma vs. fallback).
- Total events and per-category counts.
- Loading indicators and clear error messages.
- Accessible, mobile-friendly dark dashboard.

**Architecture / pipeline**

```
Simulated event ─▶ Python rule engine (classification + severity)
                                       │
                                       ▼
                          privacy-safe payload ─▶ Gemma 4 API ─▶ structured summary
                                       │                                   │
                                       ▼                                   ▼
                                 dashboard display (rule verdict shown separately from AI summary)
```

The Flask backend (`app.py`) exposes JSON endpoints; the frontend
(`templates/index.html`, `static/script.js`) renders results and drives the demo
loop. State is held in memory (no database) — appropriate for a demo.

**File structure**

```
AccidentGuard/
├── app.py                      # Flask backend + routes
├── accident_detection.py       # Simulated events + transparent rule classifier
├── gemma_service.py            # REAL Gemma 4 API integration + status + fallback
├── requirements.txt
├── .env.example                # placeholders only (safe to commit)
├── .gitignore                  # ignores .env and secrets
├── README.md
├── templates/
│   └── index.html              # dashboard
├── static/
│   ├── style.css               # dark responsive theme
│   └── script.js               # frontend logic + demo loop
└── tests/
    ├── test_accident_detection.py   # mock tests (no key needed)
    ├── test_gemma_integration.py    # REAL API tests (skip without a key)
    └── manual_integration_check.py  # standalone proof-of-integration script
```

---

## Exact Gemma 4 model ID used

**Primary model: `gemma-4-26b-a4b-it`**

This was **verified against Google's official documentation**
([Run Gemma with the Gemini API](https://ai.google.dev/gemma/docs/core/gemma_on_gemini_api)),
which lists exactly two supported Gemma 4 models on the Gemini API:

| Model ID | Role |
|---|---|
| `gemma-4-26b-a4b-it` | Primary (used by this project) |
| `gemma-4-31b-it` | Documented fallback |

The integration uses the official **`google-genai`** SDK (`from google import
genai`), **not** the legacy `google-generativeai` package, and calls
`client.models.generate_content(model="gemma-4-26b-a4b-it", ...)` with a system
instruction that constrains the model's behaviour. The model ID is set via
`GEMMA_MODEL_ID` in `.env` and validated against the verified list; an unknown ID
falls back to `gemma-4-26b-a4b-it` rather than silently substituting another model.
The app never substitutes Gemini or another model for Gemma 4.

### How Gemma contributes to the solution

The rule engine decides the *category* using fixed thresholds. Gemma provides the
*interpretation* the rule engine cannot: it rephrases the simulated event in plain
language, highlights which supplied details may need attention, and proposes
cautious next steps. Because it receives a constrained system instruction, it does
not invent sensor readings, does not claim certainty that a real accident
occurred, and reminds users to contact actual emergency services for real
incidents. The engine's verdict and Gemma's summary are always shown separately on
the dashboard so their roles stay distinct.

---

## Installation & setup

**Prerequisites:** Python 3.10+ and pip.

```bash
# 1. (Recommended) create a virtual environment
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Create your local config from the example (placeholders only)
#    Windows:
copy .env.example .env
#    macOS/Linux:
cp .env.example .env
```

## API key configuration

1. Get a free key from **Google AI Studio**: https://aistudio.google.com/app/apikey
2. Open your local `.env` file (created above) and set:
   ```
   GEMINI_API_KEY=your_real_key_here
   GEMMA_MODEL_ID=gemma-4-26b-a4b-it
   ```
3. The key is read **only** from the `GEMINI_API_KEY` environment variable. It is
   never sent to the frontend, never logged, and never included in error messages.
4. `.env` is in `.gitignore` — **never commit it**. Only `.env.example`
   (placeholders) is tracked.

> If you run without a key, the dashboard still works in **non-AI fallback mode**:
   summaries are produced by a clearly-labelled rule-based fallback (never
   presented as Gemma output), and the AI badge shows **Not configured**.

## How to run the application

```bash
python app.py
```

Then open **http://127.0.0.1:5000** in your browser.

- Click **Generate Sample Event** (pick an event type first) to create one event,
  or **Start Demo** to auto-generate events every few seconds. Each event is
  summarised by Gemma 4 (if a key is set) or by the fallback (if not).
- Watch the **AI badge** in the header: it flips to **Connected** only after a
  real Gemma response arrives.

## How to test

**Mock tests** (no key, no network):
```bash
python -m pytest tests/test_accident_detection.py -v
```

**Real Gemma 4 API tests** (require your key in `.env`):
```bash
python -m pytest tests/test_gemma_integration.py -v
```
These tests are **skipped automatically** when no key is set, so the full suite
stays green for offline runs.

**Full suite:**
```bash
python -m pytest -v
```

**Standalone proof-of-integration** (no pytest, prints a clear PASS/FAIL):
```bash
python tests/manual_integration_check.py
```
With a valid key it prints `RESULT: CONNECTED — real Gemma 4 integration
confirmed.` along with the model ID and the actual Gemma response — this is the
cleanest artifact to show a judge.

**Required test cases covered**

| # | Case | How |
|---|---|---|
| 1 | Normal driving event | `test_normal_driving_event` |
| 2 | Suspicious event | `test_suspicious_driving_event` |
| 3 | Possible accident event | `test_possible_accident_event` |
| 4 | Successful real Gemma 4 request | `test_real_gemma_summary_for_possible_accident` (+ manual script) |
| 5 | Missing API key | app runs in fallback; badge = Not configured |
| 6 | API failure / timeout | error path returns status `error` + fallback (verified live with an invalid key) |
| 7 | Dashboard display of the AI response | verified by booting the server and loading the page |

---

## Evidence of the real Gemma integration

- **`gemma_service.py`** makes a real `client.models.generate_content(...)`
  call to `gemma-4-26b-a4b-it` via the official `google-genai` SDK.
- The **error path was verified live**: with an intentionally invalid key, Google's
  API returned `400 INVALID_ARGUMENT ... API key not valid`, which the app caught
  and surfaced as status `error` while redacting the key. This proves the request
  reaches Google's `generativelanguage.googleapis.com` endpoint with the correct
  model ID.
- To record final evidence with a real key, run `python
  tests/manual_integration_check.py` and save its output — it shows the model ID
  and the actual response text.
- The dashboard AI badge shows **Connected** only after a real response; the
  fallback is always labelled "Non-AI fallback", never "Gemma".

## Current limitations

- **Simulation only.** No real sensors, GPS, vehicle hardware, or live data.
- **No emergency notifications.** The app recommends contacting emergency services
  in text only; it does not place calls or messages.
- **In-memory storage.** Event history resets when the server restarts.
- **API quota / latency.** Real Gemma calls depend on your key's quota and network.
- **Rule thresholds are illustrative.** The classifier uses example thresholds,
  not values calibrated against real crash data — intentionally, because the data
  is simulated.
- **Safety scope.** This prototype must not be used for real-world accident
  detection or any safety-critical decision.

---

## GitHub workflow (for the team)

We use the organizers' existing repository. Do **not** create a replacement repo.

**1. Review generated code first**
```bash
git status                 # see what changed
git diff <file>            # review a specific file before staging
```
Never overwrite existing team files without reviewing them.

**2. Make the initial commit** (after reviewing):
```bash
git add app.py accident_detection.py gemma_service.py requirements.txt .env.example .gitignore README.md templates/ static/ tests/
git status                 # confirm .env is NOT staged
git commit -m "AccidentGuard: simulation engine + Gemma 4 integration + dashboard"
```

**3. Push to the designated repository:**
```bash
git push origin clean-main     # or the branch your organizers specified
```
Always push to a feature branch rather than directly to `main`/`master` unless
the organizers ask otherwise.

**4. Commit progress daily.** Small, frequent commits are better than one big
push at the end.

**5. Verify no secrets are tracked** before every push:
```bash
git diff --cached | Select-String -Pattern "AIza"      # Windows PowerShell
# or on macOS/Linux:
git diff --cached | grep -iE "AIza|api_key=.{10,}"
```
`.env` is gitignored; only `.env.example` (placeholders) should be tracked. If a
key was ever committed by mistake, rotate it immediately in Google AI Studio.

---

## Demo script (for judges)

> **1 min setup + 2 min demo.**

1. **Setup (pre-demo):** `pip install -r requirements.txt`, add your
   `GEMINI_API_KEY` to `.env`, run `python tests/manual_integration_check.py`
   and leave its `RESULT: CONNECTED` output visible as evidence.

2. **The problem (20s):** Raw driving telemetry is hard to interpret. We need a
   plain-language reading plus cautious next steps — without falsely claiming a
   real accident.

3. **The pipeline (30s):** Explain the three stages — simulated event → Python
   rule engine → real Gemma 4 summary — and note the engine's verdict is shown
   separately from the AI's summary.

4. **Live demo (45s):** Run `python app.py`, open the dashboard. Click
   **Start Demo**. Point out: events appear and get classified; the AI badge
   flips to **Connected** as real Gemma responses arrive; the summary, attention
   points, recommendations, and Gemma severity opinion update live; history and
   stats increment.

5. **Real integration proof (20s):** Show the header model ID
   (`gemma-4-26b-a4b-it`), the "Gemma ✓" badges in history, and the
   `manual_integration_check.py` output from step 1.

6. **Limitations (15s):** State clearly that this is a simulation prototype — no
   real sensors, GPS, or emergency notifications — and that Gemma never claims
   certainty about a real accident.

---

## License

Project code is provided for the hackathon. The Gemma 4 model is accessed under
Google's terms; see Google's Gemma documentation for usage and license details.
