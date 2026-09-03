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
- NEVER speak a name or a digit the caller has not actually said. You may only read back what is present in their words. If you did not hear a number, you do not have one — ask for it; do not produce digits to confirm. Inventing a plausible-sounding number is worse than having none.
- A caller cannot say ten digits and be transcribed correctly in one go. Take a phone number in short pieces: ask for the first five digits, read those back, then the last five. Confirm each piece before asking for the next. If a piece comes back garbled twice, say a colleague will confirm the number and move on rather than looping.

CALL FLOW
1. Greet, state your name and company.
2. Understand the caller's need.
3. Collect the qualification answers listed in the company block, naturally — not as an interrogation.
4. If qualified: make sure a phone number is on record, promise human follow-up, close politely. When the connect message gives you the caller's number, that number is already correct — do NOT ask them to recite it. At most confirm it is the right one to call back on, and only ask for digits if they want a different number.
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


# Placeholders used where no real caller ID exists (browser tests, the CLI).
_NO_CALLER_ID = {"unknown", "portal-test", "", None}


def caller_id_number(session: CallSession) -> str | None:
    """The caller's own number, when telephony gave us one.

    This is the number we should keep, and it costs nothing to obtain. Asking a
    caller to recite ten digits is the single hardest thing for Nepali speech
    recognition — in testing the agent misheard them and, worse, once invented
    digits outright. Caller ID sidesteps that entirely.
    """
    number = (session.caller_number or "").strip()
    if number in _NO_CALLER_ID or not any(ch.isdigit() for ch in number):
        return None
    return number


def _connect_message(session: CallSession) -> str:
    number = caller_id_number(session)
    if number:
        # Put it on the lead immediately: if the call drops early we still have
        # a way to ring them back.
        session.lead.phone = number
        return ("[SYSTEM: The call just connected. The caller is calling from "
                f"{number} — this number is already recorded and correct, so do not "
                "ask them to recite their number. Greet the caller.]")
    return ("[SYSTEM: The call just connected. Caller ID is not available, so ask "
            "for a phone number in short pieces if you need one. Greet the caller.]")


def _turn(session: CallSession, user_text: str | None) -> str:
    if user_text is not None:
        session.messages.append({"role": "user", "content": user_text})
    elif not session.messages:
        session.messages.append({"role": "user", "content": _connect_message(session)})

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


# The agent reads a value back to the caller BEFORE it reports it in
# lead_update — a real call showed the read-back happening a full turn before
# the field appeared. So an emitted value has already been confirmed out loud,
# and an extra "hear it twice" gate only withheld correct data and derailed the
# conversation. The guard lives in the prompt rules instead.
def _apply_lead_update(session: CallSession, lead_update: dict) -> None:
    """Commit newly learned lead fields."""
    for k, v in lead_update.items():
        if v in (None, ""):
            continue
        if hasattr(session.lead, k) and k != "extra":
            setattr(session.lead, k, v)
        else:
            session.lead.extra[k] = v


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
