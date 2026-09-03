import json
from types import SimpleNamespace
from unittest.mock import patch

from app import agent, tenants


def _fake_response(envelope: dict, prose_prefix: str = "", usage_overrides: dict | None = None):
    usage = {
        "input_tokens": 100,
        "output_tokens": 20,
        "cache_creation_input_tokens": 0,
        "cache_read_input_tokens": 0,
    }
    usage.update(usage_overrides or {})
    text = prose_prefix + json.dumps(envelope)
    return SimpleNamespace(
        content=[SimpleNamespace(text=text)],
        usage=SimpleNamespace(**usage),
    )


# ---------------------------------------------------------------- envelope parsing

def test_parse_envelope_clean_json():
    raw = json.dumps({"say": "hello", "lead_update": {"name": "Ram"}, "end_call": False})
    say, lead_update, end_call = agent._parse_envelope(raw)
    assert say == "hello"
    assert lead_update == {"name": "Ram"}
    assert end_call is False


def test_parse_envelope_json_wrapped_in_prose():
    raw = 'Sure thing! {"say": "namaste", "lead_update": {}, "end_call": true} hope that helps'
    say, lead_update, end_call = agent._parse_envelope(raw)
    assert say == "namaste"
    assert end_call is True


def test_parse_envelope_malformed_json_degrades_gracefully():
    raw = "just plain text, no envelope at all"
    say, lead_update, end_call = agent._parse_envelope(raw)
    assert say == raw
    assert lead_update == {}
    assert end_call is False


def test_parse_envelope_broken_json_braces():
    raw = '{"say": "oops", "lead_update": }'  # invalid JSON despite braces
    say, lead_update, end_call = agent._parse_envelope(raw)
    assert say == raw
    assert lead_update == {}
    assert end_call is False


# ---------------------------------------------------------------- lead_update merging

def test_turn_merges_known_and_extra_lead_fields():
    cfg = tenants.TenantConfig(tenant_id="t1", company_name="Acme")
    session = agent.CallSession(call_id="c1", tenant=cfg)
    envelope = {
        "say": "got it",
        "lead_update": {"interest": "scooter", "custom_field": "vip"},
        "end_call": False,
    }
    with patch.object(agent, "_get_client") as get_client:
        get_client.return_value.messages.create.return_value = _fake_response(envelope)
        agent.respond(session, "hello")

    assert session.lead.interest == "scooter"
    assert session.lead.extra["custom_field"] == "vip"


def test_turn_holds_an_unconfirmed_name_back_from_the_lead():
    """Names are gated by CONFIRM_REQUIRED — see tests/test_lead_confirmation.py."""
    cfg = tenants.TenantConfig(tenant_id="t1", company_name="Acme")
    session = agent.CallSession(call_id="c1", tenant=cfg)
    envelope = {"say": "got it", "lead_update": {"name": "Sita"}, "end_call": False}
    with patch.object(agent, "_get_client") as get_client:
        get_client.return_value.messages.create.return_value = _fake_response(envelope)
        agent.respond(session, "hello")

    assert session.lead.name is None
    assert session.pending["name"] == "Sita"


def test_turn_sets_ended_flag():
    cfg = tenants.TenantConfig(tenant_id="t1", company_name="Acme")
    session = agent.CallSession(call_id="c1", tenant=cfg)
    envelope = {"say": "bye", "lead_update": {}, "end_call": True}
    with patch.object(agent, "_get_client") as get_client:
        get_client.return_value.messages.create.return_value = _fake_response(envelope)
        agent.respond(session, "goodbye")

    assert session.ended is True


# ---------------------------------------------------------------- cost tracking

def test_usage_accumulates_across_turns():
    cfg = tenants.TenantConfig(tenant_id="t1", company_name="Acme")
    session = agent.CallSession(call_id="c1", tenant=cfg)
    envelope = {"say": "hi", "lead_update": {}, "end_call": False}
    with patch.object(agent, "_get_client") as get_client:
        get_client.return_value.messages.create.return_value = _fake_response(envelope)
        agent.greeting(session)
        agent.respond(session, "hello")

    assert session.usage_totals["input_tokens"] == 200
    assert session.usage_totals["output_tokens"] == 40


def test_estimate_cost_usd_haiku():
    usage = {
        "input_tokens": 1_000_000,
        "output_tokens": 1_000_000,
        "cache_creation_input_tokens": 1_000_000,
        "cache_read_input_tokens": 1_000_000,
    }
    cost = agent.estimate_cost_usd(usage, "claude-haiku-4-5-20251001")
    assert cost == 1.00 + 5.00 + 1.25 + 0.10


def test_estimate_cost_usd_unknown_model_returns_zero():
    usage = {"input_tokens": 1_000_000, "output_tokens": 1_000_000}
    assert agent.estimate_cost_usd(usage, "some-unrecognized-model") == 0.0


def test_session_cost_usd_property():
    cfg = tenants.TenantConfig(tenant_id="t1", company_name="Acme")
    session = agent.CallSession(call_id="c1", tenant=cfg)
    session.usage_totals["input_tokens"] = 1_000_000
    session.usage_totals["output_tokens"] = 1_000_000
    # config.CLAUDE_MODEL default contains "haiku"
    assert session.cost_usd > 0


# ---------------------------------------------------------------- tenant prompt block

def test_tenant_block_includes_greeting_and_facts():
    cfg = tenants.TenantConfig(
        tenant_id="t1",
        company_name="Acme",
        greeting="Welcome to Acme!",
        facts="We sell widgets.",
    )
    block = agent._tenant_block(cfg)
    assert "Welcome to Acme!" in block
    assert "We sell widgets." in block
    assert "Acme" in block
