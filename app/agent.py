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
import urllib.error
import urllib.request
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


def active_model() -> str:
    """Whichever brain answers a turn — see config.LLM_BACKEND."""
    return config.OLLAMA_MODEL if config.LLM_BACKEND == "ollama" else config.CLAUDE_MODEL


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
        # The model that actually answered, not the configured Claude one: a
        # local backend has no per-token price, and billing its tokens at
        # Haiku's rates would overstate cost per call — the number every
        # roadmap phase gate is measured on.
        return estimate_cost_usd(self.usage_totals, active_model())


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
        parts.append(
            "Opening line (ALREADY SPOKEN, word for word, as the first thing on "
            f"this call — do not greet again or repeat it): {t.greeting}")
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
    """Opening line when the call connects.

    A tenant who typed an opening line gets that line, word for word. It used
    to be passed to Claude as guidance to adapt, which paraphrased it fresh on
    every call — so the company's own wording was never quite what callers
    heard, and no two calls opened the same way.

    Saying it verbatim also lets the agent open in its own recorded voice:
    voice.speak plays a real recording when the words match one, and a line
    rewritten each call can never match. And it skips an API round trip on the
    one turn where nothing has been said yet for Claude to respond to.
    """
    line = (session.tenant.greeting or "").strip()
    if not line:
        return _turn(session, user_text=None)

    # Keep the transcript honest: the model must see what the caller heard,
    # or its next turn will answer a greeting that was never spoken.
    if not session.messages:
        session.messages.append({"role": "user", "content": _connect_message(session)})
    session.messages.append(
        {"role": "assistant", "content": json.dumps({"say": line}, ensure_ascii=False)})
    return line


# Tenants with a canonical answer worth short-circuiting the model for:
# {tenant_id: {intent_keywords: exact recorded-line text}}.
#
# Small local models do not reliably recite an exact sentence, even one
# they were just handed as fact — measured on gemma3:4b, asked "तपाईंको
# लोकेसन कहाँ होला?" ("where is YOUR location"), it answered by asking the
# CALLER for their own address instead. That is not a phrasing problem to
# prompt-engineer around; it is the model losing track of whose location
# was asked about. For a handful of guaranteed-frequent, guaranteed-true
# questions, answering directly is both more correct and lets the tenant's
# trained/recorded voice speak the reply verbatim instead of the
# synthesized fallback.
#
# Deliberately keyed by tenant_id, not applied platform-wide: the answer
# text is one tenant's real address, recorded in that tenant's own voice.
# Every other tenant on this deployment must keep going through the model —
# see CLAUDE.md "Every data query MUST be tenant-scoped."
_CANNED_ANSWERS: dict[str, list[tuple[tuple[str, ...], str]]] = {
    "d4029493": [  # Hamro G&G Auto Enterprises
        (("लोकेसन", "ठाउँ", "कहाँ"), "हाम्रो सोरुम गोठाटारमा, पहिलो पुल नजिकै छ।"),
    ],
}


def _canned_answer(session: CallSession, user_text: str) -> str | None:
    for keywords, say in _CANNED_ANSWERS.get(session.tenant.tenant_id, []):
        if any(k in user_text for k in keywords):
            return say
    return None


def respond(session: CallSession, user_text: str) -> str:
    """Process one caller utterance, return what the agent should say."""
    say = _canned_answer(session, user_text)
    if say is not None:
        # Same bookkeeping _turn() would do, minus the model call: the
        # transcript still needs the caller's turn and a parseable assistant
        # envelope, or the next real turn answers a question it never saw.
        session.messages.append({"role": "user", "content": user_text})
        session.messages.append({"role": "assistant", "content": json.dumps(
            {"say": say, "lead_update": {}, "end_call": False}, ensure_ascii=False)})
        return say
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


# ------------------------------------------------------- model backends

# The reply envelope, as a schema. Claude and Ollama both accept one, and a
# forced schema is what makes a local model usable at all: qwen3:8b emitted the
# envelope 0% of the time when merely asked for it in the prompt, and 100% of
# the time under a schema.
#
# lead_update is built from the tenant's own lead_fields rather than left as a
# free-form object, for two reasons. Claude's structured outputs reject any
# object without additionalProperties: false. And a small local model handed an
# open object invents keys ("type", "action", "question") and never fills the
# real ones — gemma3:4b captured 0 of 4 fields that way.
def _envelope_schema(t: TenantConfig) -> dict:
    lead_props = {
        f: {"type": "boolean"} if f == "qualified" else {"type": "string"}
        for f in t.lead_fields
    }
    return {
        "type": "object",
        "properties": {
            "say": {"type": "string"},
            "lead_update": {
                "type": "object",
                "properties": lead_props,
                "additionalProperties": False,
            },
            "end_call": {"type": "boolean"},
        },
        "required": ["say", "lead_update", "end_call"],
        "additionalProperties": False,
    }


def _call_claude(t: TenantConfig, messages: list[dict]) -> tuple[str, dict]:
    """A turn from the hosted model. Returns (raw envelope text, usage)."""
    response = _get_client().messages.create(
        model=config.CLAUDE_MODEL,
        max_tokens=500,
        system=_system_blocks(t),
        messages=messages,
        output_config={"format": {"type": "json_schema", "schema": _envelope_schema(t)}},
    )
    u = response.usage
    return response.content[0].text, {
        "input_tokens": getattr(u, "input_tokens", 0) or 0,
        "output_tokens": getattr(u, "output_tokens", 0) or 0,
        "cache_creation_input_tokens": getattr(u, "cache_creation_input_tokens", 0) or 0,
        "cache_read_input_tokens": getattr(u, "cache_read_input_tokens", 0) or 0,
    }


def _call_ollama(t: TenantConfig, messages: list[dict]) -> tuple[str, dict]:
    """A turn from a local model — no API key, nothing leaves the machine.

    Ollama takes the schema bare where Claude wraps it, and the cached/uncached
    split has no meaning here, so those counters stay zero and estimate_cost_usd
    reports $0 for a model it has no rates for.
    """
    body = json.dumps({
        "model": config.OLLAMA_MODEL,
        # One string, not cache-annotated blocks: local inference has no prefix cache.
        "system": "\n\n".join(b["text"] for b in _system_blocks(t)),
        "messages": messages,
        "stream": False,
        "think": False,          # reasoning models otherwise spend the budget thinking
        "keep_alive": config.OLLAMA_KEEP_ALIVE,
        "format": _envelope_schema(t),
        "options": {"temperature": 0.3, "num_predict": 500},
    }).encode()
    req = urllib.request.Request(
        config.OLLAMA_URL.rstrip("/") + "/api/chat",
        data=body, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=config.OLLAMA_TIMEOUT_S) as r:
        data = json.loads(r.read())
    raw = (data.get("message") or {}).get("content", "")
    return raw, {
        "input_tokens": data.get("prompt_eval_count", 0) or 0,
        "output_tokens": data.get("eval_count", 0) or 0,
        "cache_creation_input_tokens": 0,
        "cache_read_input_tokens": 0,
    }


def _call_model(t: TenantConfig, messages: list[dict]) -> tuple[str, dict]:
    if config.LLM_BACKEND == "ollama":
        return _call_ollama(t, messages)
    return _call_claude(t, messages)


def _turn(session: CallSession, user_text: str | None) -> str:
    if user_text is not None:
        session.messages.append({"role": "user", "content": user_text})
    elif not session.messages:
        session.messages.append({"role": "user", "content": _connect_message(session)})

    raw, usage = _call_model(session.tenant, session.messages)
    session.messages.append({"role": "assistant", "content": raw})

    for k, v in usage.items():
        session.usage_totals[k] += v

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
