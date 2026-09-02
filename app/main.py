"""
FastAPI server — multi-tenant AI call platform.

Call paths:
  POST /twilio/voice     Twilio webhook (dialed number → tenant)
  POST /twilio/turn      Twilio webhook, each speech turn
  WS   /ws/chat          Generic text WebSocket for any telephony stack
                         (first client message: {"tenant_id": "..."} handshake,
                          or plain text to use the default tenant)

Admin API (header: X-Admin-Key = ADMIN_API_KEY from .env):
  GET/POST        /admin/tenants
  GET/PUT/DELETE  /admin/tenants/{tenant_id}
  POST            /admin/numbers            {"e164": "...", "tenant_id": "..."}
  GET             /admin/leads?tenant_id=
  GET/POST        /admin/dnc                 do-not-call list (checked before outbound dial)
  DELETE          /admin/dnc/{e164}

Dashboard: GET /admin  (simple HTML UI, same admin key)
"""
import asyncio
import json
import uuid
from pathlib import Path

from fastapi import FastAPI, Form, Header, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import Response, JSONResponse, HTMLResponse, RedirectResponse

from . import agent, dnc, storage, tenants, usage, config

app = FastAPI(title="hanuman.ai — AI Call Platform")

SESSIONS: dict[str, agent.CallSession] = {}   # single-process; Redis when scaling out


@app.get("/")
async def root():
    return RedirectResponse("/admin")

STATIC_DIR = Path(__file__).parent / "static"


# ------------------------------------------------------------------ helpers
def _require_admin(x_admin_key: str | None):
    if not config.ADMIN_API_KEY or x_admin_key != config.ADMIN_API_KEY:
        raise HTTPException(status_code=401, detail="invalid admin key")


def _xml_escape(s: str) -> str:
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def _twiml(text: str, gather: bool = True, action: str = "/twilio/turn") -> Response:
    # Twilio has no Nepali (ne-NP) voice; hi-IN is the closest supported
    # language — same Devanagari script, far more intelligible than en-IN
    # misreading Nepali text. Real Nepali needs the Asterisk/Piper pipeline
    # (see docs/ASTERISK_SETUP.md) — this is a stopgap for Twilio testing.
    say = f'<Say language="hi-IN">{_xml_escape(text)}</Say>'
    if gather:
        body = (
            f'<Gather input="speech" action="{action}" method="POST" '
            f'speechTimeout="auto" language="hi-IN">{say}</Gather>'
            f'<Redirect method="POST">{action}</Redirect>'
        )
    else:
        body = say + "<Hangup/>"
    xml = f'<?xml version="1.0" encoding="UTF-8"?><Response>{body}</Response>'
    return Response(content=xml, media_type="application/xml")


# ------------------------------------------------------------- Twilio path
@app.post("/twilio/voice")
async def twilio_voice(
    CallSid: str = Form(...),
    From: str = Form("unknown"),
    To: str = Form(""),
    tenant_id: str | None = None,          # query param, used for outbound calls
):
    tenant_id = tenant_id or tenants.tenant_for_number(To)
    cfg = tenants.get_or_default(tenant_id)
    if usage.over_cap(cfg.tenant_id, cfg.included_minutes):
        return _twiml(
            "We are unable to take your call right now. Please try again later.",
            gather=False,
        )
    session = agent.CallSession(call_id=CallSid, caller_number=From, tenant=cfg)
    SESSIONS[CallSid] = session
    return _twiml(agent.greeting(session))


@app.post("/twilio/turn")
async def twilio_turn(CallSid: str = Form(...), SpeechResult: str = Form("")):
    session = SESSIONS.get(CallSid)
    if session is None:
        return _twiml("Sorry, something went wrong. Goodbye.", gather=False)
    if not SpeechResult.strip():
        return _twiml("Sorry, I didn't catch that. Could you repeat?")

    reply = agent.respond(session, SpeechResult)
    if session.ended:
        storage.save_session(session)
        SESSIONS.pop(CallSid, None)
        return _twiml(reply, gather=False)
    return _twiml(reply)


# ------------------------------------------------------ generic WS path
@app.websocket("/ws/chat")
async def ws_chat(ws: WebSocket):
    """
    Protocol: optional first message {"tenant_id": "..."} selects the tenant.
    Then plain-text utterances in, plain-text agent replies out.
    First server message is the greeting. Server closes when call ends.
    """
    await ws.accept()
    first = await ws.receive_text()
    tenant_id = None
    pending_user_text = None
    try:
        handshake = json.loads(first)
        if isinstance(handshake, dict) and "tenant_id" in handshake:
            tenant_id = handshake["tenant_id"]
        else:
            pending_user_text = first
    except json.JSONDecodeError:
        pending_user_text = first

    session = agent.CallSession(
        call_id=str(uuid.uuid4()), tenant=tenants.get_or_default(tenant_id)
    )
    SESSIONS[session.call_id] = session
    try:
        # agent.* calls block on a synchronous Anthropic HTTP request — running them
        # inline would freeze this connection's event loop turn (and its keepalive
        # ping/pong) for the call's full duration, so offload to a thread.
        await ws.send_text(await asyncio.to_thread(agent.greeting, session))
        if pending_user_text and not session.ended:
            await ws.send_text(await asyncio.to_thread(agent.respond, session, pending_user_text))
        while not session.ended:
            user_text = await ws.receive_text()
            await ws.send_text(await asyncio.to_thread(agent.respond, session, user_text))
        storage.save_session(session)
        await ws.close()
    except WebSocketDisconnect:
        storage.save_session(session)
    finally:
        SESSIONS.pop(session.call_id, None)


# ------------------------------------------------------------- admin API
@app.get("/admin/tenants")
async def admin_list_tenants(x_admin_key: str | None = Header(default=None)):
    _require_admin(x_admin_key)
    return tenants.list_all()


@app.post("/admin/tenants")
async def admin_create_tenant(
    body: dict, x_admin_key: str | None = Header(default=None)
):
    _require_admin(x_admin_key)
    name = body.pop("company_name", None)
    if not name:
        raise HTTPException(400, "company_name required")
    allowed = {k: v for k, v in body.items() if k in tenants.TenantConfig.__dataclass_fields__}
    cfg = tenants.create(company_name=name, **allowed)
    return cfg.to_dict()


@app.get("/admin/tenants/{tenant_id}")
async def admin_get_tenant(tenant_id: str, x_admin_key: str | None = Header(default=None)):
    _require_admin(x_admin_key)
    cfg = tenants.get(tenant_id)
    if cfg is None:
        raise HTTPException(404, "tenant not found")
    return cfg.to_dict()


@app.put("/admin/tenants/{tenant_id}")
async def admin_update_tenant(
    tenant_id: str, body: dict, x_admin_key: str | None = Header(default=None)
):
    _require_admin(x_admin_key)
    cfg = tenants.get(tenant_id)
    if cfg is None:
        raise HTTPException(404, "tenant not found")
    for k, v in body.items():
        if k != "tenant_id" and hasattr(cfg, k):
            setattr(cfg, k, v)
    tenants.save(cfg)
    return cfg.to_dict()


@app.delete("/admin/tenants/{tenant_id}")
async def admin_delete_tenant(tenant_id: str, x_admin_key: str | None = Header(default=None)):
    _require_admin(x_admin_key)
    tenants.delete(tenant_id)
    return {"deleted": tenant_id}


@app.post("/admin/numbers")
async def admin_map_number(body: dict, x_admin_key: str | None = Header(default=None)):
    _require_admin(x_admin_key)
    e164, tenant_id = body.get("e164"), body.get("tenant_id")
    if not e164 or not tenant_id:
        raise HTTPException(400, "e164 and tenant_id required")
    tenants.map_number(e164, tenant_id)
    return {"mapped": e164, "tenant_id": tenant_id}


@app.get("/admin/leads")
async def admin_leads(
    tenant_id: str | None = None, x_admin_key: str | None = Header(default=None)
):
    _require_admin(x_admin_key)
    return JSONResponse(storage.list_leads(tenant_id=tenant_id))


@app.get("/admin/dnc")
async def admin_list_dnc(x_admin_key: str | None = Header(default=None)):
    _require_admin(x_admin_key)
    return dnc.list_all()


@app.post("/admin/dnc")
async def admin_add_dnc(body: dict, x_admin_key: str | None = Header(default=None)):
    _require_admin(x_admin_key)
    e164 = body.get("e164")
    if not e164:
        raise HTTPException(400, "e164 required")
    dnc.add(e164, reason=body.get("reason", ""))
    return {"added": e164}


@app.delete("/admin/dnc/{e164}")
async def admin_remove_dnc(e164: str, x_admin_key: str | None = Header(default=None)):
    _require_admin(x_admin_key)
    dnc.remove(e164)
    return {"removed": e164}


@app.get("/admin")
async def admin_dashboard():
    html = (STATIC_DIR / "admin.html").read_text(encoding="utf-8")
    return HTMLResponse(html)


# ---------------------------------------------------- customer portal API
# Auth: header X-Tenant-Key = the tenant's api_key (given at onboarding).
# Customers can see their own data and edit their own agent — nothing else.

PORTAL_EDITABLE = {
    "agent_name", "language", "greeting", "facts", "questions", "transfer_to",
}


def _require_tenant(x_tenant_key: str | None) -> tenants.TenantConfig:
    cfg = tenants.get_by_api_key(x_tenant_key or "")
    if cfg is None or cfg.status != "active":
        raise HTTPException(status_code=401, detail="invalid tenant key")
    return cfg


@app.get("/portal/me")
async def portal_me(x_tenant_key: str | None = Header(default=None)):
    cfg = _require_tenant(x_tenant_key)
    d = cfg.to_dict()
    d.pop("api_key", None)   # never echo the key back
    return d


@app.put("/portal/me")
async def portal_update(body: dict, x_tenant_key: str | None = Header(default=None)):
    cfg = _require_tenant(x_tenant_key)
    for k, v in body.items():
        if k in PORTAL_EDITABLE:
            setattr(cfg, k, v)
    tenants.save(cfg)
    d = cfg.to_dict()
    d.pop("api_key", None)
    return d


@app.get("/portal/leads")
async def portal_leads(x_tenant_key: str | None = Header(default=None)):
    cfg = _require_tenant(x_tenant_key)
    return JSONResponse(storage.list_leads(tenant_id=cfg.tenant_id))


@app.get("/portal/usage")
async def portal_usage(x_tenant_key: str | None = Header(default=None)):
    cfg = _require_tenant(x_tenant_key)
    s = usage.summary(cfg.tenant_id)
    s["included_minutes"] = cfg.included_minutes
    return s


@app.get("/admin/usage")
async def admin_usage(x_admin_key: str | None = Header(default=None)):
    _require_admin(x_admin_key)
    return usage.all_tenants_summary()


@app.post("/admin/test-call")
async def admin_test_call(body: dict, x_admin_key: str | None = Header(default=None)):
    """
    Outbound pilot call via Twilio: the agent CALLS a phone (e.g. yours).
    Body: {"to": "+9779803250775", "tenant_id": "<optional>"}
    Requires TWILIO_* and PUBLIC_BASE_URL in .env.
    Note: Twilio trial accounts can only call numbers verified in the console.
    """
    _require_admin(x_admin_key)
    to = body.get("to")
    if not to:
        raise HTTPException(400, "to (E.164 number) required")
    if dnc.is_listed(to):
        raise HTTPException(403, f"{to} is on the do-not-call list")
    if not (config.TWILIO_ACCOUNT_SID and config.TWILIO_AUTH_TOKEN
            and config.TWILIO_PHONE_NUMBER and config.PUBLIC_BASE_URL):
        raise HTTPException(400, "TWILIO_* and PUBLIC_BASE_URL must be set in .env")

    from twilio.base.exceptions import TwilioRestException
    from twilio.rest import Client as TwilioClient
    tw = TwilioClient(config.TWILIO_ACCOUNT_SID, config.TWILIO_AUTH_TOKEN)
    url = config.PUBLIC_BASE_URL.rstrip("/") + "/twilio/voice"
    if body.get("tenant_id"):
        url += f"?tenant_id={body['tenant_id']}"
    try:
        call = tw.calls.create(to=to, from_=config.TWILIO_PHONE_NUMBER, url=url)
    except TwilioRestException as e:
        raise HTTPException(400, f"Twilio error: {e.msg}")
    return {"queued": True, "call_sid": call.sid, "to": to}


@app.websocket("/portal/ws/test-chat")
async def portal_test_chat(ws: WebSocket):
    """In-browser test call: first message is the tenant api_key, then text turns."""
    await ws.accept()
    key = await ws.receive_text()
    cfg = tenants.get_by_api_key(key.strip())
    if cfg is None:
        await ws.send_text("[error] invalid key")
        await ws.close()
        return
    session = agent.CallSession(
        call_id="portal-test-" + str(uuid.uuid4())[:8],
        caller_number="portal-test",
        tenant=cfg,
    )
    try:
        await ws.send_text(await asyncio.to_thread(agent.greeting, session))
        while not session.ended:
            user_text = await ws.receive_text()
            await ws.send_text(await asyncio.to_thread(agent.respond, session, user_text))
        await ws.close()
    except WebSocketDisconnect:
        pass


@app.get("/portal")
async def portal_page():
    return HTMLResponse((STATIC_DIR / "portal.html").read_text(encoding="utf-8"))


@app.get("/manifest.json")
async def manifest():
    return JSONResponse({
        "name": "hanuman.ai",
        "short_name": "hanuman.ai",
        "start_url": "/portal",
        "display": "standalone",
        "background_color": "#0f1420",
        "theme_color": "#0f1420",
        "icons": [{
            "src": "data:image/svg+xml,<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 100 100'><text y='0.9em' font-size='90'>📞</text></svg>",
            "sizes": "any", "type": "image/svg+xml",
        }],
    })


@app.get("/health")
async def health():
    return {"status": "ok", "model": config.CLAUDE_MODEL}
