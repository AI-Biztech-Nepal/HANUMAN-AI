"""How exact details (names, phone numbers) reach the lead.

History worth keeping: a "hear it twice before recording" gate lived here
briefly. A real call showed the agent reads a value back to the caller a full
turn BEFORE it reports it in lead_update — so by the time a field arrives it
has already been confirmed out loud, and the extra gate only withheld correct
data and pushed the agent into a confusing re-confirmation. The read-back is
enforced in the prompt rules; these tests cover what the code still owes:
committing what the model reports, and never making the caller recite a phone
number we already have.
"""
from app import agent


def _session(caller_number="unknown") -> agent.CallSession:
    return agent.CallSession(call_id="test-call", caller_number=caller_number)


# ------------------------------------------------------------- lead updates

def test_reported_fields_are_committed():
    s = _session()
    agent._apply_lead_update(s, {"name": "चन्दन", "interest": "Aprilia SR"})
    assert s.lead.name == "चन्दन"
    assert s.lead.interest == "Aprilia SR"


def test_a_confirmed_name_lands_on_the_first_report():
    """The regression this file exists for: the agent confirmed the name with
    the caller, then reported it, and the old gate still withheld it — so the
    lead came back with no name at all."""
    s = _session()
    agent._apply_lead_update(s, {"name": "चन्दन गुप्ता"})
    assert s.lead.name == "चन्दन गुप्ता"


def test_a_later_correction_overwrites():
    s = _session()
    agent._apply_lead_update(s, {"name": "सन्धान"})
    agent._apply_lead_update(s, {"name": "चन्दन"})
    assert s.lead.name == "चन्दन"


def test_tenant_custom_fields_go_to_extra():
    s = _session()
    agent._apply_lead_update(s, {"preferred_colour": "red"})
    assert s.lead.extra["preferred_colour"] == "red"


def test_empty_values_are_ignored():
    s = _session()
    s.lead.name = "चन्दन"
    agent._apply_lead_update(s, {"name": "", "interest": None})
    assert s.lead.name == "चन्दन"
    assert s.lead.interest is None


def test_extra_is_never_overwritten_wholesale():
    s = _session()
    agent._apply_lead_update(s, {"extra": "nonsense"})
    assert isinstance(s.lead.extra, dict)


# --------------------------------------------------------------- caller ID

def test_caller_id_is_used_when_telephony_provides_one():
    s = _session("+9779808027608")
    assert agent.caller_id_number(s) == "+9779808027608"


def test_placeholders_are_not_treated_as_caller_ids():
    for placeholder in ("unknown", "portal-test", "", "   "):
        assert agent.caller_id_number(_session(placeholder)) is None


def test_a_number_from_caller_id_is_recorded_without_asking():
    s = _session("+9779808027608")
    msg = agent._connect_message(s)
    # On the lead straight away: if the call drops we can still ring back.
    assert s.lead.phone == "+9779808027608"
    assert "+9779808027608" in msg
    assert "do not" in msg.lower() and "recite" in msg.lower()


def test_without_caller_id_the_agent_is_told_to_ask_in_pieces():
    s = _session()
    msg = agent._connect_message(s)
    assert s.lead.phone is None
    assert "short pieces" in msg


def test_the_connect_message_still_asks_for_a_greeting():
    for number in ("+9779808027608", "unknown"):
        assert "Greet the caller" in agent._connect_message(_session(number))


# ------------------------------------------------------------- prompt rules

def test_platform_rules_forbid_inventing_digits():
    """The agent once read back a phone number the caller never said."""
    rules = agent.PLATFORM_RULES
    assert "NEVER speak a name or a digit the caller has not actually said" in rules
    assert "Inventing a plausible-sounding number is worse than having none" in rules


def test_platform_rules_split_long_numbers():
    assert "cannot say ten digits" in agent.PLATFORM_RULES


def test_platform_rules_still_require_a_read_back():
    rules = agent.PLATFORM_RULES
    assert "ENTIRE reply must be the read-back question" in rules
    assert "Never greet or thank a caller by an unconfirmed name" in rules
