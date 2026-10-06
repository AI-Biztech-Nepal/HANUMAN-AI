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


def test_turn_records_a_name_the_agent_reports():
    """The agent confirms a name with the caller before reporting it, so a
    reported name is committed — see tests/test_lead_confirmation.py."""
    cfg = tenants.TenantConfig(tenant_id="t1", company_name="Acme")
    session = agent.CallSession(call_id="c1", tenant=cfg)
    envelope = {"say": "got it", "lead_update": {"name": "Sita"}, "end_call": False}
    with patch.object(agent, "_get_client") as get_client:
        get_client.return_value.messages.create.return_value = _fake_response(envelope)
        agent.respond(session, "hello")

    assert session.lead.name == "Sita"


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


# ---------------------------------------------------------------- envelope schema & backends

def _objects(node):
    """Every object-typed node in a JSON schema, however deeply nested."""
    if isinstance(node, dict):
        if node.get("type") == "object":
            yield node
        for v in node.values():
            yield from _objects(v)
    elif isinstance(node, list):
        for v in node:
            yield from _objects(v)


def test_envelope_schema_closes_every_object():
    # Claude's structured outputs reject any object lacking
    # additionalProperties: false — an open lead_update 400s the whole turn.
    cfg = tenants.TenantConfig(tenant_id="t1", company_name="Acme")
    for obj in _objects(agent._envelope_schema(cfg)):
        assert obj.get("additionalProperties") is False


def test_envelope_schema_lead_update_is_the_tenants_own_fields():
    cfg = tenants.TenantConfig(tenant_id="t1", company_name="Acme",
                               lead_fields=["name", "budget", "qualified", "car_model"])
    lead = agent._envelope_schema(cfg)["properties"]["lead_update"]["properties"]
    assert set(lead) == {"name", "budget", "qualified", "car_model"}
    assert lead["qualified"] == {"type": "boolean"}
    assert lead["car_model"] == {"type": "string"}     # tenant-custom field is allowed


def test_local_backend_never_builds_a_claude_client(monkeypatch):
    # The zero-key promise: with LLM_BACKEND=ollama a turn must not touch the
    # Anthropic client, so a blank ANTHROPIC_API_KEY cannot matter.
    monkeypatch.setattr(agent.config, "LLM_BACKEND", "ollama")
    monkeypatch.setattr(agent.config, "ANTHROPIC_API_KEY", "")
    cfg = tenants.TenantConfig(tenant_id="t1", company_name="Acme")
    envelope = {"say": "नमस्ते", "lead_update": {}, "end_call": False}

    class _Resp:
        def read(self):
            return json.dumps({"message": {"content": json.dumps(envelope)},
                               "prompt_eval_count": 7, "eval_count": 3}).encode()
        def __enter__(self): return self
        def __exit__(self, *a): return False

    with patch.object(agent, "_get_client", side_effect=AssertionError("Claude client used")), \
         patch("urllib.request.urlopen", return_value=_Resp()) as opened:
        session = agent.CallSession(call_id="c1", caller_number="+977", tenant=cfg)
        assert agent.respond(session, "नमस्ते") == "नमस्ते"

    body = json.loads(opened.call_args.args[0].data)
    assert body["format"]["properties"]["lead_update"]["additionalProperties"] is False
    assert session.usage_totals["input_tokens"] == 7
    assert session.cost_usd == 0.0                     # a local model has no per-token price


# ---------------------------------------------------------------- canned answers

def test_canned_answer_skips_the_model_and_uses_the_tenants_own_wording():
    cfg = tenants.TenantConfig(tenant_id="d4029493", company_name="Hamro G&G")
    session = agent.CallSession(call_id="c1", tenant=cfg)
    with patch.object(agent, "_get_client", side_effect=AssertionError("model called")):
        say = agent.respond(session, "हजुर तपाईंको लोकेसन कहाँ होला?")
    assert say == "हाम्रो सोरुम गोठाटारमा, पहिलो पुल नजिकै छ।"


def test_canned_answer_records_a_parseable_turn_in_history():
    cfg = tenants.TenantConfig(tenant_id="d4029493", company_name="Hamro G&G")
    session = agent.CallSession(call_id="c1", tenant=cfg)
    with patch.object(agent, "_get_client", side_effect=AssertionError("model called")):
        agent.respond(session, "तपाईंको ठाउँ कहाँ हो?")

    assert session.messages[-2] == {"role": "user", "content": "तपाईंको ठाउँ कहाँ हो?"}
    say, lead_update, end_call = agent._parse_envelope(session.messages[-1]["content"])
    assert say == "हाम्रो सोरुम गोठाटारमा, पहिलो पुल नजिकै छ।"
    assert lead_update == {}
    assert end_call is False


def test_canned_answer_is_scoped_to_its_own_tenant():
    # Same question, a DIFFERENT tenant — must not get Hamro G&G's address.
    # This is the one regression that actually matters here: leaking one
    # tenant's real-world address into another tenant's calls.
    cfg = tenants.TenantConfig(tenant_id="some-other-tenant", company_name="Acme")
    session = agent.CallSession(call_id="c1", tenant=cfg)
    envelope = {"say": "model answered instead", "lead_update": {}, "end_call": False}
    with patch.object(agent, "_get_client") as get_client:
        get_client.return_value.messages.create.return_value = _fake_response(envelope)
        say = agent.respond(session, "हजुर तपाईंको लोकेसन कहाँ होला?")
    assert say == "model answered instead"


def test_unrelated_questions_still_go_to_the_model():
    cfg = tenants.TenantConfig(tenant_id="d4029493", company_name="Hamro G&G")
    session = agent.CallSession(call_id="c1", tenant=cfg)
    envelope = {"say": "स्कुटर उपलब्ध छ", "lead_update": {}, "end_call": False}
    with patch.object(agent, "_get_client") as get_client:
        get_client.return_value.messages.create.return_value = _fake_response(envelope)
        say = agent.respond(session, "के तपाईंसँग स्कुटर छ?")
    assert say == "स्कुटर उपलब्ध छ"
