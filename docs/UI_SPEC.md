# UI_SPEC.md — RiskTrail Frontend

Stack: React + Vite + JavaScript + CSS. Must consume `POST /api/investigate`
exactly as defined in API_CONTRACT.md — no inventing a different response
shape.

## Screens / States

### 1. Input State
- RiskTrail branding (name/logo, short tagline)
- Transaction ID text input
- "Investigate" button

### 2. Loading State
- Shown while `POST /api/investigate` is in flight
- Simple spinner or loading text — no need for anything elaborate

### 3. Result State
Render, top to bottom:
- **Risk score** — numeric, 0–100
- **Risk level** — badge/label (e.g. color-coded LOW/MEDIUM/HIGH)
- **Risk factors** — list of short strings
- **Evidence cards** — one per item in `evidence[]`, showing `type`, `id`,
  and `text`
- **AI investigation** — the `investigation` narrative text
- **Recommendation** — the `recommendation` text, visually distinct (e.g.
  highlighted box) since it's the actionable takeaway

### 4. Error State
Shown when the backend returns an error or the request fails:
- Clear, human-readable error message
- Include `transaction_id` if it was provided
- Allow the user to try again (e.g. re-enable the input/button)

## Demo Story This UI Must Support
1. Enter a transaction ID.
2. Click Investigate.
3. See loading, then: risk score → risk factors → evidence → AI
   investigation → recommendation.
4. Message the UI should make obvious: RiskTrail doesn't just flag the
   transaction — it traces the evidence behind the risk.

## Non-Goals
No auth, no multi-page routing complexity, no state-management library
beyond what React provides out of the box, no design system beyond simple
CSS. Keep it demo-able and easy for a fresher to explain.
