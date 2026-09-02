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
  GET/POST        /admin/tenants/{id}/users  portal logins; POST returns a one-time invite link
  POST            /admin/users/{id}/reissue-invite
  DELETE          /admin/users/{id}

Portal auth (customers sign in with email + password; no public signup):
  POST            /auth/login                {"email","password"} → session cookie
  POST            /auth/logout
  GET             /auth/me
  GET/POST        /auth/invite/{token}       check / accept an invite, sets first password
  POST            /auth/password             {"current","new"}

Dashboard: GET /admin  (simple HTML UI, same admin key)
"""
import asyncio
import json
import uuid
from pathlib import Path

from fastapi import (
    Cookie, FastAPI, Form, Header, HTTPException, WebSocket, WebSocketDisconnect,
)
from fastapi.responses import Response, JSONResponse, HTMLResponse, RedirectResponse

from . import agent, auth, dnc, storage, tenants, usage, config

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
# Two ways in, deliberately:
#   - session cookie  → people, signed in with email + password (the portal UI)
#   - X-Tenant-Key    → machines, using the tenant api_key (integrations)
# Customers can see their own data and edit their own agent — nothing else.

PORTAL_EDITABLE = {
    "agent_name", "language", "greeting", "facts", "questions", "transfer_to",
}

SESSION_COOKIE = "hanuman_session"


def _tenant_for_session(token: str | None) -> tenants.TenantConfig | None:
    user = auth.user_for_session(token or "")
    if user is None:
        return None
    return tenants.get(user.tenant_id)


def _require_tenant(
    x_tenant_key: str | None,
    session: str | None = None,
) -> tenants.TenantConfig:
    cfg = _tenant_for_session(session) if session else None
    if cfg is None:
        cfg = tenants.get_by_api_key(x_tenant_key or "")
    if cfg is None or cfg.status != "active":
        raise HTTPException(status_code=401, detail="not signed in")
    return cfg


def _set_session_cookie(response: Response, token: str) -> None:
    # secure=True only when we know we're behind HTTPS — otherwise the cookie
    # would be dropped on a plain-HTTP pilot box and nobody could sign in.
    response.set_cookie(
        SESSION_COOKIE, token,
        max_age=auth.SESSION_TTL_DAYS * 24 * 3600,
        httponly=True, samesite="lax",
        secure=config.PUBLIC_BASE_URL.startswith("https://"),
        path="/",
    )


# ------------------------------------------------------------ portal auth API

@app.post("/auth/login")
async def auth_login(body: dict):
    try:
        user, token = auth.login(body.get("email", ""), body.get("password", ""))
    except auth.AuthError as exc:
        raise HTTPException(status_code=401, detail=str(exc))
    cfg = tenants.get(user.tenant_id)
    if cfg is None or cfg.status != "active":
        raise HTTPException(status_code=403, detail="this account's company is not active")
    resp = JSONResponse({"user": user.to_dict(), "company_name": cfg.company_name})
    _set_session_cookie(resp, token)
    return resp


@app.post("/auth/logout")
async def auth_logout(session: str | None = Cookie(default=None, alias=SESSION_COOKIE)):
    auth.logout(session or "")
    resp = JSONResponse({"ok": True})
    resp.delete_cookie(SESSION_COOKIE, path="/")
    return resp


@app.get("/auth/me")
async def auth_me(session: str | None = Cookie(default=None, alias=SESSION_COOKIE)):
    user = auth.user_for_session(session or "")
    if user is None:
        raise HTTPException(status_code=401, detail="not signed in")
    cfg = tenants.get(user.tenant_id)
    return {"user": user.to_dict(),
            "company_name": cfg.company_name if cfg else ""}


@app.get("/auth/invite/{token}")
async def auth_invite_peek(token: str):
    """Check an invite link without consuming it, so the page can greet the
    right person before they choose a password."""
    user = auth.peek_invite(token)
    if user is None:
        raise HTTPException(status_code=404, detail="this invite link is invalid or has expired")
    cfg = tenants.get(user.tenant_id)
    return {"email": user.email,
            "company_name": cfg.company_name if cfg else ""}


@app.post("/auth/invite/{token}")
async def auth_invite_accept(token: str, body: dict):
    try:
        user = auth.accept_invite(token, body.get("password", ""))
        # Sign them straight in — a freshly set password is proof enough.
        _, session_token = auth.login(user.email, body.get("password", ""))
    except auth.AuthError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    resp = JSONResponse({"user": user.to_dict()})
    _set_session_cookie(resp, session_token)
    return resp


@app.post("/auth/password")
async def auth_change_password(
    body: dict,
    session: str | None = Cookie(default=None, alias=SESSION_COOKIE),
):
    user = auth.user_for_session(session or "")
    if user is None:
        raise HTTPException(status_code=401, detail="not signed in")
    try:
        auth.change_password(user.user_id, body.get("current", ""), body.get("new", ""))
    except auth.AuthError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    # change_password drops every session, this one included — sign back in.
    _, token = auth.login(user.email, body.get("new", ""))
    resp = JSONResponse({"ok": True})
    _set_session_cookie(resp, token)
    return resp


# ------------------------------------------- admin: portal user management

@app.get("/admin/tenants/{tenant_id}/users")
async def admin_list_users(tenant_id: str, x_admin_key: str | None = Header(default=None)):
    _require_admin(x_admin_key)
    return auth.list_users(tenant_id)


@app.post("/admin/tenants/{tenant_id}/users")
async def admin_create_user(
    tenant_id: str, body: dict, x_admin_key: str | None = Header(default=None)
):
    """Create a portal login for a tenant. Returns a one-time invite link —
    it is shown once and cannot be recovered, only reissued."""
    _require_admin(x_admin_key)
    if tenants.get(tenant_id) is None:
        raise HTTPException(404, "tenant not found")
    try:
        user, token = auth.create_user(tenant_id, body.get("email", ""))
    except auth.AuthError as exc:
        raise HTTPException(400, str(exc))
    base = config.PUBLIC_BASE_URL.rstrip("/") or "http://127.0.0.1:8000"
    return {"user": user.to_dict(), "invite_url": f"{base}/portal?invite={token}"}


@app.post("/admin/users/{user_id}/reissue-invite")
async def admin_reissue_invite(user_id: str, x_admin_key: str | None = Header(default=None)):
    """Password reset, operator-style: issue a fresh invite link."""
    _require_admin(x_admin_key)
    try:
        token = auth.reissue_invite(user_id)
    except auth.AuthError as exc:
        raise HTTPException(404, str(exc))
    base = config.PUBLIC_BASE_URL.rstrip("/") or "http://127.0.0.1:8000"
    return {"invite_url": f"{base}/portal?invite={token}"}


@app.delete("/admin/users/{user_id}")
async def admin_delete_user(user_id: str, x_admin_key: str | None = Header(default=None)):
    _require_admin(x_admin_key)
    auth.delete_user(user_id)
    return {"deleted": user_id}


@app.get("/portal/me")
async def portal_me(
    x_tenant_key: str | None = Header(default=None),
    session: str | None = Cookie(default=None, alias=SESSION_COOKIE),
):
    cfg = _require_tenant(x_tenant_key, session)
    d = cfg.to_dict()
    d.pop("api_key", None)   # never echo the key back
    return d


@app.put("/portal/me")
async def portal_update(
    body: dict,
    x_tenant_key: str | None = Header(default=None),
    session: str | None = Cookie(default=None, alias=SESSION_COOKIE),
):
    cfg = _require_tenant(x_tenant_key, session)
    for k, v in body.items():
        if k in PORTAL_EDITABLE:
            setattr(cfg, k, v)
    tenants.save(cfg)
    d = cfg.to_dict()
    d.pop("api_key", None)
    return d


@app.get("/portal/leads")
async def portal_leads(
    x_tenant_key: str | None = Header(default=None),
    session: str | None = Cookie(default=None, alias=SESSION_COOKIE),
):
    cfg = _require_tenant(x_tenant_key, session)
    return JSONResponse(storage.list_leads(tenant_id=cfg.tenant_id))


@app.get("/portal/usage")
async def portal_usage(
    x_tenant_key: str | None = Header(default=None),
    session: str | None = Cookie(default=None, alias=SESSION_COOKIE),
):
    cfg = _require_tenant(x_tenant_key, session)
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
    """In-browser test call, then text turns.

    Authenticates from the session cookie the browser sends with the upgrade
    request. The first message is still read and accepted as a tenant api_key
    so existing integrations keep working; signed-in browsers send "".
    """
    await ws.accept()
    key = await ws.receive_text()
    cfg = _tenant_for_session(ws.cookies.get(SESSION_COOKIE))
    if cfg is None:
        cfg = tenants.get_by_api_key(key.strip())
    if cfg is None or cfg.status != "active":
        await ws.send_text("[error] not signed in")
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
