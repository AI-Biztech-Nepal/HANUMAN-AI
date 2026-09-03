"""
Core conversation agent: Claude decides what to say on each turn,
tracks the lead-qualification state, and signals when the call should end.

Multi-tenant: each CallSession carries a TenantConfig; the system prompt is
built as [static platform rules — prompt-cached across ALL tenants] +
[small per-tenant block]. Static block caching is what keeps per-call cost low.

Telephony-agnostic — works the same for phone (via STT), web chat, or test CLI.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field

import anthropic
import httpx2

from . import config
from .tenants import TenantConfig, get_or_default

_client: anthropic.Anthropic | None = None


def _get_client() -> anthropic.Anthropic:
    global _client
    if _client is None:
        # max_keepalive_connections=0: a pooled/reused connection reliably hangs on
        # the 2nd+ request in this deployment (Windows host + asyncio event loop
        # running alongside a blocking call) even though a fresh connection always
        # succeeds — force a new connection per call rather than reusing one.
        _client = anthropic.Anthropic(
            api_key=config.ANTHROPIC_API_KEY,
            http_client=httpx2.Client(limits=httpx2.Limits(max_keepalive_connections=0)),
        )
    return _client


# ---------------------------------------------------------------- cost

# USD per 1M tokens: (input, output, cache_write_5m, cache_read).
PRICING = {
    "haiku": (1.00, 5.00, 1.25, 0.10),
    "sonnet": (3.00, 15.00, 3.75, 0.30),
}


def estimate_cost_usd(usage: dict, model: str) -> float:
    """Rough $ cost for accumulated token usage, given the model name."""
    rates = next((r for key, r in PRICING.items() if key in model), None)
    if rates is None:
        return 0.0
    in_rate, out_rate, cache_write_rate, cache_read_rate = rates
    return (
        usage.get("input_tokens", 0) * in_rate
        + usage.get("output_tokens", 0) * out_rate
        + usage.get("cache_creation_input_tokens", 0) * cache_write_rate
        + usage.get("cache_read_input_tokens", 0) * cache_read_rate
    ) / 1_000_000


@dataclass
class Lead:
    """Structured info the agent tries to collect during the call."""
    name: str | None = None
    phone: str | None = None
    interest: str | None = None
    budget: str | None = None
    timeline: str | None = None
    qualified: bool | None = None
    notes: str = ""
    extra: dict = field(default_factory=dict)   # tenant-custom fields land here

    def to_dict(self) -> dict:
        d = self.__dict__.copy()
        extra = d.pop("extra")
        d.update(extra)
        return d


@dataclass
class CallSession:
    """One phone call = one session. Holds history + lead state + tenant."""
    call_id: str
    caller_number: str = "unknown"
    tenant: TenantConfig = field(default_factory=lambda: get_or_default(None))
    messages: list = field(default_factory=list)
    lead: Lead = field(default_factory=Lead)
    # Exact details heard once but not yet confirmed by the caller. See
    # CONFIRM_REQUIRED — these are held here instead of being written to lead.
    pending: dict = field(default_factory=dict)
    pending_note: str = ""      # steering text to prepend to the next caller turn
    ended: bool = False
    started_at: float = field(default_factory=lambda: __import__("time").time())
    usage_totals: dict = field(default_factory=lambda: {
        "input_tokens": 0,
        "output_tokens": 0,
        "cache_creation_input_tokens": 0,
        "cache_read_input_tokens": 0,
    })

    def duration_sec(self) -> int:
        import time
        return int(time.time() - self.started_at)

    @property
    def cost_usd(self) -> float:
        return estimate_cost_usd(self.usage_totals, config.CLAUDE_MODEL)


# ---------------------------------------------------------------- prompts

# Static across ALL tenants → sent with cache_control so Anthropic caches it.
PLATFORM_RULES = """You are an AI phone agent on a live call in Nepal. Platform rules (never overridden):

VOICE STYLE
- 1-3 SHORT sentences per turn, then let the caller speak. One question at a time.
- Spoken audio only: no lists, no markdown, no emoji.
- Language: if mode is "auto", mirror the caller — natural conversational Nepali (Devanagari) if they speak Nepali, English if English; Nepali-English code-switching is normal and fine. Use polite Nepali forms (तपाईं).
- Say numbers and phone numbers digit-by-digit. Never guess important details — ask again.

HONESTY & SAFETY
- If asked whether you are a machine/AI: answer truthfully yes, and offer a human follow-up.
- Never invent facts, prices, or offers beyond the company facts provided. If unknown, say a colleague will confirm.
- Never pressure anyone. If not interested, thank them and end quickly.
- If what you heard is garbled, unclear, or doesn't form a sensible sentence (likely a transcription error, not a real reply), don't guess its meaning or treat it as goodbye — just ask the caller to repeat themselves. Never set end_call=true for this reason alone.
- Speech recognition mangles names and numbers even when the surrounding sentence sounds perfectly sensible, so a name you heard once is not yet a fact. Repeat any name, phone number, or other exact detail back to the caller and get their confirmation before you record it in lead_update or use it to address them. Never greet or thank a caller by an unconfirmed name.
- The turn you first hear a name or number, your ENTIRE reply must be the read-back question and nothing else — do not thank them, do not use the name, do not move on to the next question. Say it back the way you heard it and ask if that is right ("तपाईंको नाम चन्दन, ठीक छ?" / "That's Chandan — did I get that right?"). Only after the caller confirms may you use it or ask anything else. If they correct you, read the new version back the same way.

CALL FLOW
1. Greet, state your name and company.
2. Understand the caller's need.
3. Collect the qualification answers listed in the company block, naturally — not as an interrogation.
4. If qualified: confirm their phone number, promise human follow-up, close politely.
5. End with a courteous closing (e.g., "धन्यवाद, शुभ दिन!" / "Thank you, have a great day!").

OUTPUT FORMAT — respond ONLY with a JSON object, no other text:
{"say": "<what to speak — short, natural>", "lead_update": {<newly learned lead fields>}, "end_call": <true when conversation is finished>}
Set end_call=true when: caller says goodbye, asks to stop, is clearly not interested, or qualification is complete and follow-up confirmed."""


def _tenant_block(t: TenantConfig) -> str:
    questions = "\n".join(f"- {q}" for q in t.questions)
    parts = [
        f"COMPANY BLOCK\nCompany: {t.company_name}\nYour name: {t.agent_name}\nLanguage mode: {t.language}",
        f"Qualification info to collect:\n{questions}",
        f"Lead fields you may set in lead_update: {', '.join(t.lead_fields)}",
    ]
    if t.greeting:
        parts.append(f"Custom opening line (use it, adapted to caller's language): {t.greeting}")
    if t.facts:
        parts.append(f"Company facts you may state:\n{t.facts}")
    if t.transfer_to:
        parts.append("A human transfer is available for hot leads — offer it when the caller is highly interested.")
    return "\n\n".join(parts)


def _system_blocks(t: TenantConfig) -> list[dict]:
    return [
        {
            "type": "text",
            "text": PLATFORM_RULES,
            "cache_control": {"type": "ephemeral"},   # cached across all tenants
        },
        {"type": "text", "text": _tenant_block(t)},
    ]


# ---------------------------------------------------------------- turns

def greeting(session: CallSession) -> str:
    """Opening line when the call connects."""
    return _turn(session, user_text=None)


def respond(session: CallSession, user_text: str) -> str:
    """Process one caller utterance, return what the agent should say."""
    return _turn(session, user_text=user_text)


def _turn(session: CallSession, user_text: str | None) -> str:
    if user_text is not None:
        if session.pending_note:
            user_text = f"{session.pending_note}\n{user_text}"
            session.pending_note = ""
        session.messages.append({"role": "user", "content": user_text})
    elif not session.messages:
        session.messages.append(
            {"role": "user", "content": "[SYSTEM: The call just connected. Greet the caller.]"}
        )

    response = _get_client().messages.create(
        model=config.CLAUDE_MODEL,
        max_tokens=500,
        system=_system_blocks(session.tenant),
        messages=session.messages,
    )
    raw = response.content[0].text
    session.messages.append({"role": "assistant", "content": raw})

    usage = response.usage
    session.usage_totals["input_tokens"] += getattr(usage, "input_tokens", 0) or 0
    session.usage_totals["output_tokens"] += getattr(usage, "output_tokens", 0) or 0
    session.usage_totals["cache_creation_input_tokens"] += (
        getattr(usage, "cache_creation_input_tokens", 0) or 0
    )
    session.usage_totals["cache_read_input_tokens"] += (
        getattr(usage, "cache_read_input_tokens", 0) or 0
    )

    say, lead_update, end_call = _parse_envelope(raw)
    _apply_lead_update(session, lead_update)
    session.ended = end_call
    return say


# Fields where a transcription error is silent and costly: the sentence around
# them sounds perfectly sensible, so nothing flags the mistake. A name heard
# once is a guess — Whisper turned "चन्दन" into "सन्धान" in testing, and the
# agent then addressed the caller by it. These are held until the caller says
# the value a second time; everything else commits on first hearing.
CONFIRM_REQUIRED = {"name", "phone"}


def _normalize(value) -> str:
    """Loose comparison key — punctuation and case shouldn't block a match."""
    return re.sub(r"[\s।.,!?'\"-]+", "", str(value)).casefold()


def _apply_lead_update(session: CallSession, lead_update: dict) -> None:
    """Commit newly learned fields, holding unconfirmed exact details back.

    A CONFIRM_REQUIRED value is committed only once the model reports the same
    value on a later turn — by then the caller has had a turn to correct it.
    A different value replaces the pending one rather than committing, which is
    exactly what a correction looks like.
    """
    newly_pending = {}
    for k, v in lead_update.items():
        if v in (None, ""):
            continue
        if k in CONFIRM_REQUIRED:
            committed = getattr(session.lead, k, None) or session.lead.extra.get(k)
            if committed is not None and _normalize(committed) == _normalize(v):
                continue    # already confirmed; models re-send known fields
                            # every turn, and re-gating would loop the agent
                            # back to "is your name X?" forever
            previous = session.pending.get(k)
            if previous is None or _normalize(previous) != _normalize(v):
                session.pending[k] = v
                newly_pending[k] = v
                continue                    # heard once — not a fact yet
            session.pending.pop(k, None)    # said twice — now it's confirmed
        if hasattr(session.lead, k) and k != "extra":
            setattr(session.lead, k, v)
        else:
            session.lead.extra[k] = v

    # Never silently drop what we heard: keep unconfirmed values on the lead,
    # clearly labelled, so a human can follow up rather than losing the detail.
    if session.pending:
        session.lead.extra["unconfirmed"] = dict(session.pending)
    else:
        session.lead.extra.pop("unconfirmed", None)

    if newly_pending:
        detail = ", ".join(f'{k}="{v}"' for k, v in newly_pending.items())
        # Carried into the next turn rather than appended now: a standalone
        # message here would put two user turns back to back.
        session.pending_note = (
            f"[SYSTEM: You recorded {detail}, but the caller has not confirmed it and "
            f"speech recognition mistakes these constantly. Read it back and ask them "
            f"to confirm before you use it. Do not address them by an unconfirmed name.]"
        )


def _parse_envelope(raw: str) -> tuple[str, dict, bool]:
    """Extract the JSON envelope; degrade gracefully if the model added prose."""
    match = re.search(r"\{.*\}", raw, re.DOTALL)
    if not match:
        return raw.strip(), {}, False
    try:
        data = json.loads(match.group(0))
        return (
            str(data.get("say", "")).strip() or raw.strip(),
            data.get("lead_update") or {},
            bool(data.get("end_call", False)),
        )
    except json.JSONDecodeError:
        return raw.strip(), {}, False
