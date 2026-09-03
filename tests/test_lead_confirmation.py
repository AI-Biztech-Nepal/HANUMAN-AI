"""The confirmation gate on exact details.

Regression cover for a real failure: Whisper heard "मेरो नाम चन्दन हो" as
"सन्धान", the agent wrote it to the lead and addressed the caller by it. A
value that arrives once is a guess; these tests pin the rule that it takes a
second hearing to become a fact.
"""
from app import agent


def _session() -> agent.CallSession:
    return agent.CallSession(call_id="test-call")


def test_a_name_heard_once_is_not_written_to_the_lead():
    s = _session()
    agent._apply_lead_update(s, {"name": "सन्धान"})
    assert s.lead.name is None
    assert s.pending["name"] == "सन्धान"


def test_a_name_heard_twice_is_committed():
    s = _session()
    agent._apply_lead_update(s, {"name": "चन्दन"})
    agent._apply_lead_update(s, {"name": "चन्दन"})
    assert s.lead.name == "चन्दन"
    assert "name" not in s.pending


def test_a_correction_replaces_the_pending_value_without_committing():
    s = _session()
    agent._apply_lead_update(s, {"name": "सन्धान"})   # misheard
    agent._apply_lead_update(s, {"name": "चन्दन"})    # caller corrects
    assert s.lead.name is None, "a corrected name must still need confirming"
    assert s.pending["name"] == "चन्दन"
    agent._apply_lead_update(s, {"name": "चन्दन"})    # confirmed
    assert s.lead.name == "चन्दन"


def test_a_confirmed_name_is_not_re_gated_when_the_model_resends_it():
    """Models restate known fields every turn. Re-gating a value that is
    already committed sent the agent back to "is your name X?" on every
    later turn, ignoring whatever the caller had just said."""
    s = _session()
    agent._apply_lead_update(s, {"name": "चन्दन"})
    agent._apply_lead_update(s, {"name": "चन्दन"})
    assert s.lead.name == "चन्दन"

    s.pending_note = ""
    agent._apply_lead_update(s, {"name": "चन्दन", "interest": "Grazia"})
    assert s.lead.name == "चन्दन"
    assert not s.pending, "a confirmed value must not go back into the gate"
    assert s.pending_note == "", "and must not re-trigger a read-back"
    assert s.lead.interest == "Grazia", "the new information must still land"


def test_a_confirmed_name_can_still_be_corrected_later():
    s = _session()
    agent._apply_lead_update(s, {"name": "चन्दन"})
    agent._apply_lead_update(s, {"name": "चन्दन"})
    # Caller says "actually it's Chandani" — a genuinely different value must
    # re-enter the gate rather than overwrite silently.
    agent._apply_lead_update(s, {"name": "चन्दनी"})
    assert s.lead.name == "चन्दन", "the confirmed value stands until reconfirmed"
    assert s.pending["name"] == "चन्दनी"
    agent._apply_lead_update(s, {"name": "चन्दनी"})
    assert s.lead.name == "चन्दनी"


def test_confirmation_ignores_punctuation_and_case():
    s = _session()
    agent._apply_lead_update(s, {"name": "चन्दन"})
    agent._apply_lead_update(s, {"name": "चन्दन।"})
    assert s.lead.name == "चन्दन।"

    s2 = _session()
    agent._apply_lead_update(s2, {"name": "Chandan"})
    agent._apply_lead_update(s2, {"name": "chandan"})
    assert s2.lead.name == "chandan"


def test_phone_numbers_are_gated_too():
    s = _session()
    agent._apply_lead_update(s, {"phone": "9808027608"})
    assert s.lead.phone is None
    agent._apply_lead_update(s, {"phone": "9808027608"})
    assert s.lead.phone == "9808027608"


def test_ordinary_fields_still_commit_immediately():
    s = _session()
    agent._apply_lead_update(s, {"interest": "scooter", "budget": "60000"})
    assert s.lead.interest == "scooter"
    assert s.lead.budget == "60000"
    assert not s.pending


def test_tenant_custom_fields_commit_immediately():
    s = _session()
    agent._apply_lead_update(s, {"preferred_colour": "red"})
    assert s.lead.extra["preferred_colour"] == "red"


def test_unconfirmed_values_are_kept_on_the_lead_not_dropped():
    s = _session()
    agent._apply_lead_update(s, {"name": "सन्धान"})
    # The detail still reaches a human, clearly marked as unverified.
    assert s.lead.to_dict()["unconfirmed"] == {"name": "सन्धान"}


def test_the_unconfirmed_marker_clears_once_confirmed():
    s = _session()
    agent._apply_lead_update(s, {"name": "चन्दन"})
    agent._apply_lead_update(s, {"name": "चन्दन"})
    assert "unconfirmed" not in s.lead.to_dict()


def test_empty_values_are_ignored():
    s = _session()
    agent._apply_lead_update(s, {"name": "", "interest": None})
    assert not s.pending
    assert s.lead.name is None
    assert s.lead.interest is None


def test_a_first_hearing_queues_a_read_back_instruction():
    s = _session()
    agent._apply_lead_update(s, {"name": "सन्धान"})
    assert "सन्धान" in s.pending_note
    assert "confirm" in s.pending_note.lower()


def test_the_instruction_rides_on_the_next_caller_turn_not_its_own_message():
    """Two user messages back to back would be a malformed conversation."""
    s = _session()
    s.messages.append({"role": "user", "content": "hello"})
    s.messages.append({"role": "assistant", "content": "hi"})
    agent._apply_lead_update(s, {"name": "सन्धान"})
    assert s.messages[-1]["role"] == "assistant", "no message appended yet"

    # Simulate the next turn's message assembly.
    note, text = s.pending_note, "हो, चन्दन हो"
    assert note
    combined = f"{note}\n{text}"
    s.messages.append({"role": "user", "content": combined})
    s.pending_note = ""
    roles = [m["role"] for m in s.messages]
    assert all(a != b for a, b in zip(roles, roles[1:])), "roles must alternate"
    assert text in s.messages[-1]["content"]


def test_no_instruction_is_queued_for_ordinary_fields():
    s = _session()
    agent._apply_lead_update(s, {"interest": "scooter"})
    assert s.pending_note == ""
