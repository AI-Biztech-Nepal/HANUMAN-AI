import pytest

from app import auth, config


@pytest.fixture(autouse=True)
def tmp_db(tmp_path, monkeypatch):
    """Point every module's sqlite connection at a fresh per-test file."""
    db_path = tmp_path / "test_leads.db"
    monkeypatch.setattr(config, "LEADS_DB_PATH", str(db_path))
    monkeypatch.setattr(config, "ADMIN_API_KEY", "test-admin-key")
    # Tests mock the Claude client, so they must not inherit whatever brain the
    # developer's .env selects — LLM_BACKEND=ollama would send them to a real
    # local server. A test that wants the local backend sets it itself.
    monkeypatch.setattr(config, "LLM_BACKEND", "claude")
    yield db_path


@pytest.fixture(autouse=True)
def reset_throttles():
    """Clear the login and rate-limit counters between tests.

    They live in module-level dicts rather than the DB, so unlike everything
    else they survive tmp_db and would otherwise leak across tests — one that
    exhausts a limit would fail whichever test happened to run next.
    """
    auth._failed.clear()
    auth._rate.clear()
    yield
