# hanuman.ai — AI Call Platform for Nepal

Multi-tenant AI phone agent SaaS: answers business calls in Nepali/English via Claude,
qualifies leads, serves many customer companies from one deployment.

## Commands

```bash
# run server (needs ANTHROPIC_API_KEY + ADMIN_API_KEY in .env)
uvicorn app.main:app --reload     # text only on Windows: piper lives in WSL

# run server WITH voice (mic in the portal) — dev machine is Windows + WSL
serve_voice.cmd                   # serves from .venv-wsl on the same port 8000

# simulated call in terminal (no telephony)
python test_call.py [tenant_id]

# syntax check everything
python -m py_compile app/*.py media/voice_bridge.py test_call.py
```

Run the test suite with `pytest` (91 tests, SQLite-backed, no network). Every
test gets a fresh DB via the `tmp_db` fixture in `tests/conftest.py`.

## Architecture (read docs/ARCHITECTURE.md for the full picture)

- `app/agent.py` — the brain. Telephony-agnostic: text in → Claude → JSON envelope
  `{"say", "lead_update", "end_call"}` out. System prompt = PLATFORM_RULES (static,
  prompt-cached across all tenants — keep it byte-identical between calls!) +
  small per-tenant block. Default model: Haiku (cost = the business margin; don't
  switch to Sonnet globally).
- `app/tenants.py` — tenant CRUD, phone-number→tenant routing, portal api_keys (`tk_...`).
- `app/auth.py` — portal logins: users, scrypt password hashing, sessions, invites.
  Stdlib crypto only, no auth dependency. Session and invite tokens are stored
  **hashed** — never add a code path that writes or logs a raw token.
- `app/main.py` — FastAPI. Three surfaces: Twilio webhooks (`/twilio/*`), generic
  text WebSocket (`/ws/chat`, for future Asterisk/SIP integration), admin API +
  customer portal API. Static UIs: `app/static/admin.html`, `app/static/portal.html`
  (portal is a mobile-first PWA — single file, no build step, keep it that way).
- `app/storage.py`, `app/usage.py` — SQLite (pilot). Postgres schema for scale
  lives in `db/schema.sql`; keep module interfaces stable so the swap is clean.
- `media/voice_bridge.py` — Whisper STT + Piper TTS reference pipeline.
- `deploy/` — install.sh (Ubuntu VPS), Dockerfile, nginx, systemd.

## Conventions

- Every data query MUST be tenant-scoped. Never leak data across tenants.
- Portal endpoints: customers may only edit fields in `PORTAL_EDITABLE` (main.py).
  Never expose `api_key`, `status`, `included_minutes` to portal writes.
- Portal auth has two doors, both tenant-scoped: a session cookie (people, via
  email + password) and `X-Tenant-Key` (machines/integrations). New portal
  routes must accept both — take `session` and `x_tenant_key`, call
  `_require_capability(...)` (or `_require_tenant` where no role gates it).
- Portal users have roles: owner > manager > staff, with the capability table
  in `auth.PERMISSIONS`. A signed-in person is limited by their role; the
  tenant API key is the company's integration credential and is not.
- Signup is public (`POST /auth/signup` creates a company plus its owner) and
  operators can still invite users directly. Password reset is self-service via
  `POST /auth/forgot`; with no SMTP configured the link is logged, not mailed.
- Metering (`usage.record_call`) must never raise into call handling.
- Agent speech style: short sentences, one question per turn, no markdown — it's spoken.
- Platform-level agent rules that must never be removed: admits it's an AI when asked;
  never invents company facts; no pressure on uninterested callers.
- Legal guardrails (Nepal): licensed telephony only (no SIM boxes), do-not-call list
  before outbound dialing, recording disclosure. See docs/ROADMAP.md "Standing rules".

## Current phase

Pilot (roadmap Phase 1-2): Twilio outbound testing via `POST /admin/test-call`,
customers onboarded BYO-number. Next engineering milestones: pytest suite,
Asterisk↔voice_bridge integration, Postgres migration, eSewa/Khalti billing.
