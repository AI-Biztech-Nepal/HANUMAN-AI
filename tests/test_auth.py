import pytest

from app import auth, tenants


@pytest.fixture(autouse=True)
def clear_throttle():
    auth._failed.clear()
    yield
    auth._failed.clear()


def _invited_user(email="owner@gng.com.np"):
    cfg = tenants.create(company_name="G&G Automobiles")
    user, token = auth.create_user(cfg.tenant_id, email)
    return cfg, user, token


# ------------------------------------------------------------------ passwords

def test_hash_is_salted_and_verifies():
    a = auth.hash_password("correct horse battery")
    b = auth.hash_password("correct horse battery")
    assert a != b, "same password must not produce the same hash"
    assert auth.verify_password("correct horse battery", a)
    assert auth.verify_password("correct horse battery", b)


def test_verify_rejects_wrong_password_and_junk():
    stored = auth.hash_password("s3cret-password")
    assert not auth.verify_password("wrong-password", stored)
    assert not auth.verify_password("s3cret-password", "")
    assert not auth.verify_password("s3cret-password", "not-a-hash")
    assert not auth.verify_password("s3cret-password", "md5$x$y$z$q$w")


def test_password_is_never_stored_in_plaintext():
    _cfg, _user, token = _invited_user()
    auth.accept_invite(token, "plaintext-check-123")
    with auth._conn() as conn:
        rows = conn.execute("SELECT password_hash FROM users").fetchall()
    assert rows and "plaintext-check-123" not in rows[0][0]
    assert rows[0][0].startswith("scrypt$")


def test_short_password_rejected():
    _cfg, _user, token = _invited_user()
    with pytest.raises(auth.AuthError):
        auth.accept_invite(token, "short")


# --------------------------------------------------------------------- emails

def test_email_is_normalized_and_validated():
    assert auth.normalize_email("  Owner@GNG.com.NP ") == "owner@gng.com.np"
    for bad in ("", "nope", "no@domain", "a b@c.com"):
        with pytest.raises(auth.AuthError):
            auth.normalize_email(bad)


def test_duplicate_email_rejected_case_insensitively():
    cfg, _user, _token = _invited_user("dup@gng.com.np")
    with pytest.raises(auth.AuthError):
        auth.create_user(cfg.tenant_id, "DUP@gng.com.np")


def test_login_is_case_insensitive_on_email():
    _cfg, _user, token = _invited_user("Mixed@Case.com")
    auth.accept_invite(token, "password-1234")
    user, session = auth.login("MIXED@case.COM", "password-1234")
    assert user.email == "mixed@case.com"
    assert auth.user_for_session(session) is not None


# -------------------------------------------------------------------- invites

def test_invite_flow_activates_user():
    _cfg, user, token = _invited_user()
    assert user.status == "invited"
    assert auth.peek_invite(token).email == "owner@gng.com.np"

    activated = auth.accept_invite(token, "password-1234")
    assert activated.status == "active"
    assert auth.get_user(user.user_id).status == "active"


def test_invite_cannot_be_reused():
    _cfg, _user, token = _invited_user()
    auth.accept_invite(token, "password-1234")
    assert auth.peek_invite(token) is None
    with pytest.raises(auth.AuthError):
        auth.accept_invite(token, "another-password")


def test_invite_token_is_not_stored_in_the_clear():
    _cfg, _user, token = _invited_user()
    with auth._conn() as conn:
        stored = conn.execute("SELECT token_hash FROM invites").fetchone()[0]
    assert stored != token


def test_reissue_invalidates_the_previous_invite():
    _cfg, user, first = _invited_user()
    second = auth.reissue_invite(user.user_id)
    assert first != second
    assert auth.peek_invite(first) is None
    assert auth.peek_invite(second) is not None


def test_invited_user_cannot_log_in_before_setting_a_password():
    _cfg, _user, _token = _invited_user()
    with pytest.raises(auth.AuthError):
        auth.login("owner@gng.com.np", "")


# ------------------------------------------------------------------- sessions

def test_login_and_session_roundtrip():
    _cfg, user, token = _invited_user()
    auth.accept_invite(token, "password-1234")
    logged_in, session = auth.login("owner@gng.com.np", "password-1234")
    assert logged_in.user_id == user.user_id
    assert auth.user_for_session(session).user_id == user.user_id


def test_logout_kills_the_session():
    _cfg, _user, token = _invited_user()
    auth.accept_invite(token, "password-1234")
    _user2, session = auth.login("owner@gng.com.np", "password-1234")
    auth.logout(session)
    assert auth.user_for_session(session) is None


def test_session_token_is_not_stored_in_the_clear():
    _cfg, _user, token = _invited_user()
    auth.accept_invite(token, "password-1234")
    _u, session = auth.login("owner@gng.com.np", "password-1234")
    with auth._conn() as conn:
        stored = conn.execute("SELECT token_hash FROM sessions").fetchone()[0]
    assert stored != session


def test_expired_session_is_rejected():
    from datetime import datetime, timedelta, timezone
    _cfg, _user, token = _invited_user()
    auth.accept_invite(token, "password-1234")
    _u, session = auth.login("owner@gng.com.np", "password-1234")
    past = (datetime.now(timezone.utc) - timedelta(seconds=1)).isoformat()
    with auth._conn() as conn:
        conn.execute("UPDATE sessions SET expires_at = ?", (past,))
    assert auth.user_for_session(session) is None


def test_unknown_and_empty_session_tokens_are_rejected():
    assert auth.user_for_session("") is None
    assert auth.user_for_session("nope") is None


def test_suspending_a_user_revokes_live_sessions():
    _cfg, user, token = _invited_user()
    auth.accept_invite(token, "password-1234")
    _u, session = auth.login("owner@gng.com.np", "password-1234")
    auth.set_status(user.user_id, "suspended")
    assert auth.user_for_session(session) is None
    with pytest.raises(auth.AuthError):
        auth.login("owner@gng.com.np", "password-1234")


def test_change_password_revokes_other_sessions():
    _cfg, _user, token = _invited_user()
    auth.accept_invite(token, "password-1234")
    _u, first = auth.login("owner@gng.com.np", "password-1234")
    _u2, second = auth.login("owner@gng.com.np", "password-1234")
    auth.change_password(_u.user_id, "password-1234", "brand-new-password")
    assert auth.user_for_session(first) is None
    assert auth.user_for_session(second) is None
    assert auth.login("owner@gng.com.np", "brand-new-password")


def test_change_password_requires_the_current_one():
    _cfg, user, token = _invited_user()
    auth.accept_invite(token, "password-1234")
    with pytest.raises(auth.AuthError):
        auth.change_password(user.user_id, "wrong-current", "brand-new-password")


def test_purge_expired_removes_dead_sessions_only():
    from datetime import datetime, timedelta, timezone
    _cfg, _user, token = _invited_user()
    auth.accept_invite(token, "password-1234")
    _u, live = auth.login("owner@gng.com.np", "password-1234")
    _u2, dead = auth.login("owner@gng.com.np", "password-1234")
    past = (datetime.now(timezone.utc) - timedelta(days=1)).isoformat()
    with auth._conn() as conn:
        conn.execute("UPDATE sessions SET expires_at = ? WHERE token_hash = ?",
                     (past, auth._token_hash(dead)))
    assert auth.purge_expired() == 1
    assert auth.user_for_session(live) is not None


# ------------------------------------------------------------------- throttle

def test_repeated_failures_lock_the_account_out():
    _cfg, _user, token = _invited_user()
    auth.accept_invite(token, "password-1234")
    for _ in range(auth.MAX_FAILED_ATTEMPTS):
        with pytest.raises(auth.AuthError):
            auth.login("owner@gng.com.np", "wrong-password")
    # Even the correct password is refused while locked out.
    with pytest.raises(auth.AuthError, match="too many failed attempts"):
        auth.login("owner@gng.com.np", "password-1234")


def test_successful_login_clears_the_failure_count():
    _cfg, _user, token = _invited_user()
    auth.accept_invite(token, "password-1234")
    for _ in range(auth.MAX_FAILED_ATTEMPTS - 1):
        with pytest.raises(auth.AuthError):
            auth.login("owner@gng.com.np", "wrong-password")
    assert auth.login("owner@gng.com.np", "password-1234")
    for _ in range(auth.MAX_FAILED_ATTEMPTS - 1):
        with pytest.raises(auth.AuthError):
            auth.login("owner@gng.com.np", "wrong-password")
    assert auth.login("owner@gng.com.np", "password-1234")


# ---------------------------------------------------------------- tenant link

def test_users_are_scoped_to_their_tenant():
    cfg_a = tenants.create(company_name="A Co")
    cfg_b = tenants.create(company_name="B Co")
    auth.create_user(cfg_a.tenant_id, "a@a.com")
    auth.create_user(cfg_b.tenant_id, "b@b.com")
    assert [u["email"] for u in auth.list_users(cfg_a.tenant_id)] == ["a@a.com"]
    assert [u["email"] for u in auth.list_users(cfg_b.tenant_id)] == ["b@b.com"]


def test_delete_user_removes_sessions_and_invites():
    _cfg, user, token = _invited_user()
    auth.accept_invite(token, "password-1234")
    _u, session = auth.login("owner@gng.com.np", "password-1234")
    auth.delete_user(user.user_id)
    assert auth.get_user(user.user_id) is None
    assert auth.user_for_session(session) is None
