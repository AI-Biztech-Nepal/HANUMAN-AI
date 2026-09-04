"""Self-serve signup and password reset.

Both endpoints are reachable without signing in, so the checks here are as
much about what they refuse and what they decline to reveal as what they do.
"""
from fastapi.testclient import TestClient

from app import auth, notify, tenants
from app.main import app


def _client() -> TestClient:
    return TestClient(app)


def _signup(c, company="G&G Automobiles", email="owner@gng.com.np",
            password="a-good-password"):
    return c.post("/auth/signup", json={"company_name": company,
                                        "email": email, "password": password})


# ------------------------------------------------------------------- signup

def test_signup_creates_a_company_and_signs_the_owner_in():
    c = _client()
    r = _signup(c)
    assert r.status_code == 200, r.text
    assert r.json()["company_name"] == "G&G Automobiles"
    assert r.json()["user"]["role"] == auth.ROLE_OWNER
    # The session cookie is live: a follow-up needs no further auth.
    assert c.get("/portal/me").status_code == 200
    assert c.get("/auth/me").json()["user"]["email"] == "owner@gng.com.np"


def test_the_new_owner_can_run_their_team_immediately():
    c = _client()
    _signup(c)
    assert c.get("/portal/team").status_code == 200
    assert c.post("/portal/team",
                  json={"email": "staff@gng.com.np", "role": "staff"}).status_code == 200
    assert c.put("/portal/me", json={"agent_name": "Aashika"}).status_code == 200


def test_signup_rejects_a_duplicate_email_without_creating_a_tenant():
    c = _client()
    _signup(c)
    before = len(tenants.list_all())
    r = _signup(c, company="Second Try")
    assert r.status_code == 400
    assert "already has an account" in r.json()["detail"]
    assert len(tenants.list_all()) == before


def test_signup_requires_company_email_and_a_long_enough_password():
    c = _client()
    assert _signup(c, company="").status_code == 400
    assert _signup(c, email="not-an-email").status_code == 400
    assert _signup(c, password="short").status_code == 400
    assert tenants.list_all() == []


def test_signup_is_rate_limited_per_client():
    c = _client()
    codes = [_signup(c, company=f"Co {i}", email=f"o{i}@gng.com.np").status_code
             for i in range(7)]
    assert 400 in codes[5:], "expected the later signups to be refused"
    assert codes[0] == 200


# ------------------------------------------------------------ password reset

def test_forgot_password_answers_the_same_for_unknown_addresses():
    c = _client()
    _signup(c)
    known = c.post("/auth/forgot", json={"email": "owner@gng.com.np"})
    unknown = c.post("/auth/forgot", json={"email": "nobody@nowhere.test"})
    assert known.status_code == unknown.status_code == 200
    assert known.json() == unknown.json(), "responses must not reveal who has an account"


def test_forgot_password_never_returns_the_link():
    c = _client()
    _signup(c)
    body = c.post("/auth/forgot", json={"email": "owner@gng.com.np"}).text
    assert "invite=" not in body and "token" not in body.lower()


def test_the_reset_link_lets_the_owner_choose_a_new_password(caplog):
    c = _client()
    _signup(c)
    with caplog.at_level("WARNING"):
        c.post("/auth/forgot", json={"email": "owner@gng.com.np"})
    link = next(m.split()[-1] for m in caplog.messages if "invite=" in m)
    token = link.split("invite=")[1]

    fresh = _client()
    assert fresh.post(f"/auth/invite/{token}",
                      json={"password": "brand-new-password"}).status_code == 200
    assert fresh.get("/auth/me").status_code == 200
    # And the old password no longer works.
    other = _client()
    assert other.post("/auth/login", json={"email": "owner@gng.com.np",
                                           "password": "a-good-password"}).status_code == 401


def test_a_reset_token_works_only_once():
    c = _client()
    _signup(c)
    user = auth.get_user_by_email("owner@gng.com.np")
    token = auth.reissue_invite(user.user_id)
    assert auth.accept_invite(token, "first-new-password")
    try:
        auth.accept_invite(token, "second-new-password")
        raise AssertionError("a used reset token must not work again")
    except auth.AuthError:
        pass


def test_reset_is_declined_for_a_suspended_account():
    c = _client()
    _signup(c)
    user = auth.get_user_by_email("owner@gng.com.np")
    auth.set_status(user.user_id, "suspended", force=True)
    assert auth.start_password_reset("owner@gng.com.np") is None


# -------------------------------------------------------------------- mailer

def test_mailer_is_a_no_op_until_smtp_is_configured(monkeypatch):
    monkeypatch.setattr(notify.config, "SMTP_HOST", "")
    assert notify.configured() is False
    assert notify.send_password_reset("someone@gng.com.np", "http://x/y") is False


def test_mail_failure_does_not_raise(monkeypatch):
    monkeypatch.setattr(notify.config, "SMTP_HOST", "smtp.invalid")
    monkeypatch.setattr(notify.config, "SMTP_FROM", "bot@gng.com.np")
    # No SMTP server exists at smtp.invalid; the send must fail quietly so a
    # mail outage cannot turn a password reset into a 500.
    assert notify.send_password_reset("someone@gng.com.np", "http://x/y") is False
