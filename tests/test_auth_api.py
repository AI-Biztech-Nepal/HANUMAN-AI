import pytest
from fastapi.testclient import TestClient

from app import auth, tenants
from app.main import app, SESSION_COOKIE

ADMIN_KEY = "test-admin-key"
PASSWORD = "password-1234"


@pytest.fixture(autouse=True)
def clear_throttle():
    auth._failed.clear()
    yield
    auth._failed.clear()


def _client() -> TestClient:
    return TestClient(app)


def _tenant_with_login(c, email="owner@gng.com.np"):
    """Create a tenant + an activated portal user. Returns (tenant, invite_url)."""
    cfg = tenants.create(company_name="G&G Automobiles")
    resp = c.post(f"/admin/tenants/{cfg.tenant_id}/users",
                  json={"email": email}, headers={"X-Admin-Key": ADMIN_KEY})
    assert resp.status_code == 200
    return cfg, resp.json()["invite_url"]


def _token_from(invite_url: str) -> str:
    return invite_url.split("invite=", 1)[1]


# ------------------------------------------------------------ admin: creating

def test_creating_a_user_requires_the_admin_key():
    c = _client()
    cfg = tenants.create(company_name="G&G")
    assert c.post(f"/admin/tenants/{cfg.tenant_id}/users",
                  json={"email": "a@b.com"}).status_code == 401


def test_cannot_create_a_user_for_a_missing_tenant():
    c = _client()
    r = c.post("/admin/tenants/nope/users", json={"email": "a@b.com"},
               headers={"X-Admin-Key": ADMIN_KEY})
    assert r.status_code == 404


def test_create_user_returns_a_one_time_invite_link():
    c = _client()
    _cfg, invite_url = _tenant_with_login(c)
    assert "/portal?invite=" in invite_url


def test_there_is_no_public_signup_endpoint():
    c = _client()
    assert c.post("/auth/signup", json={"email": "x@y.com",
                                        "password": PASSWORD}).status_code == 404


# ------------------------------------------------------------- invite → login

def test_invite_peek_then_accept_signs_the_user_in():
    c = _client()
    _cfg, invite_url = _tenant_with_login(c)
    token = _token_from(invite_url)

    peek = c.get(f"/auth/invite/{token}")
    assert peek.status_code == 200
    assert peek.json()["email"] == "owner@gng.com.np"
    assert peek.json()["company_name"] == "G&G Automobiles"

    accepted = c.post(f"/auth/invite/{token}", json={"password": PASSWORD})
    assert accepted.status_code == 200
    assert SESSION_COOKIE in accepted.cookies
    # Signed in immediately, no second login needed.
    assert c.get("/portal/me").status_code == 200


def test_bad_invite_token_is_404():
    assert _client().get("/auth/invite/not-a-real-token").status_code == 404


def test_invite_rejects_a_short_password():
    c = _client()
    _cfg, invite_url = _tenant_with_login(c)
    r = c.post(f"/auth/invite/{_token_from(invite_url)}", json={"password": "short"})
    assert r.status_code == 400


# --------------------------------------------------------------------- log in

def test_login_sets_a_session_cookie_and_grants_portal_access():
    c = _client()
    _cfg, invite_url = _tenant_with_login(c)
    c.post(f"/auth/invite/{_token_from(invite_url)}", json={"password": PASSWORD})
    c.cookies.clear()
    assert c.get("/portal/me").status_code == 401

    r = c.post("/auth/login", json={"email": "owner@gng.com.np", "password": PASSWORD})
    assert r.status_code == 200
    assert r.json()["company_name"] == "G&G Automobiles"
    assert c.get("/portal/me").status_code == 200


def test_login_rejects_a_wrong_password():
    c = _client()
    _cfg, invite_url = _tenant_with_login(c)
    c.post(f"/auth/invite/{_token_from(invite_url)}", json={"password": PASSWORD})
    c.cookies.clear()
    r = c.post("/auth/login", json={"email": "owner@gng.com.np", "password": "nope-nope"})
    assert r.status_code == 401
    assert c.get("/portal/me").status_code == 401


def test_login_does_not_reveal_whether_an_email_exists():
    c = _client()
    _cfg, invite_url = _tenant_with_login(c)
    c.post(f"/auth/invite/{_token_from(invite_url)}", json={"password": PASSWORD})
    known = c.post("/auth/login", json={"email": "owner@gng.com.np", "password": "wrong-one"})
    unknown = c.post("/auth/login", json={"email": "ghost@nowhere.com", "password": "wrong-one"})
    assert known.status_code == unknown.status_code == 401
    assert known.json()["detail"] == unknown.json()["detail"]


def test_session_cookie_is_httponly():
    c = _client()
    _cfg, invite_url = _tenant_with_login(c)
    r = c.post(f"/auth/invite/{_token_from(invite_url)}", json={"password": PASSWORD})
    set_cookie = r.headers.get("set-cookie", "")
    assert "httponly" in set_cookie.lower()
    assert "samesite=lax" in set_cookie.lower()


def test_logout_clears_access():
    c = _client()
    _cfg, invite_url = _tenant_with_login(c)
    c.post(f"/auth/invite/{_token_from(invite_url)}", json={"password": PASSWORD})
    assert c.get("/portal/me").status_code == 200
    assert c.post("/auth/logout").status_code == 200
    assert c.get("/portal/me").status_code == 401


def test_auth_me_reports_the_signed_in_user():
    c = _client()
    _cfg, invite_url = _tenant_with_login(c)
    c.post(f"/auth/invite/{_token_from(invite_url)}", json={"password": PASSWORD})
    me = c.get("/auth/me")
    assert me.status_code == 200
    assert me.json()["user"]["email"] == "owner@gng.com.np"
    assert me.json()["company_name"] == "G&G Automobiles"


def test_auth_me_requires_a_session():
    assert _client().get("/auth/me").status_code == 401


# ------------------------------------------------------------------- isolation

def test_a_session_only_reaches_its_own_tenants_data():
    c = _client()
    cfg_a, invite_a = _tenant_with_login(c, "a@gng.com.np")
    cfg_b = tenants.create(company_name="Rival Motors")
    c.post(f"/admin/tenants/{cfg_b.tenant_id}/users", json={"email": "b@rival.com"},
           headers={"X-Admin-Key": ADMIN_KEY})

    c.post(f"/auth/invite/{_token_from(invite_a)}", json={"password": PASSWORD})
    me = c.get("/portal/me").json()
    assert me["tenant_id"] == cfg_a.tenant_id
    assert me["company_name"] == "G&G Automobiles"


def test_portal_me_never_returns_the_api_key():
    c = _client()
    _cfg, invite_url = _tenant_with_login(c)
    c.post(f"/auth/invite/{_token_from(invite_url)}", json={"password": PASSWORD})
    assert "api_key" not in c.get("/portal/me").json()


def test_session_cannot_edit_protected_fields():
    c = _client()
    _cfg, invite_url = _tenant_with_login(c)
    c.post(f"/auth/invite/{_token_from(invite_url)}", json={"password": PASSWORD})
    r = c.put("/portal/me", json={"agent_name": "Aashika",
                                  "included_minutes": 999999, "status": "suspended"})
    assert r.status_code == 200
    body = r.json()
    assert body["agent_name"] == "Aashika"
    assert body["included_minutes"] == 1000
    assert body["status"] == "active"


# --------------------------------------------------------- api key still works

def test_tenant_api_key_still_authenticates_for_integrations():
    c = _client()
    cfg = tenants.create(company_name="Integration Co")
    r = c.get("/portal/me", headers={"X-Tenant-Key": cfg.api_key})
    assert r.status_code == 200
    assert r.json()["tenant_id"] == cfg.tenant_id


def test_bad_api_key_is_still_rejected():
    assert _client().get("/portal/me",
                         headers={"X-Tenant-Key": "tk_bogus"}).status_code == 401


# ------------------------------------------------------------ password change

def test_change_password_requires_the_current_one():
    c = _client()
    _cfg, invite_url = _tenant_with_login(c)
    c.post(f"/auth/invite/{_token_from(invite_url)}", json={"password": PASSWORD})
    r = c.post("/auth/password", json={"current": "wrong", "new": "another-password"})
    assert r.status_code == 400


def test_change_password_keeps_the_caller_signed_in():
    c = _client()
    _cfg, invite_url = _tenant_with_login(c)
    c.post(f"/auth/invite/{_token_from(invite_url)}", json={"password": PASSWORD})
    r = c.post("/auth/password", json={"current": PASSWORD, "new": "another-password"})
    assert r.status_code == 200
    assert c.get("/portal/me").status_code == 200
    c.cookies.clear()
    assert c.post("/auth/login",
                  json={"email": "owner@gng.com.np",
                        "password": "another-password"}).status_code == 200


# ------------------------------------------------------------ operator resets

def test_reissue_invite_lets_a_locked_out_customer_back_in():
    c = _client()
    _cfg, invite_url = _tenant_with_login(c)
    c.post(f"/auth/invite/{_token_from(invite_url)}", json={"password": PASSWORD})
    user_id = c.get("/auth/me").json()["user"]["user_id"]

    r = c.post(f"/admin/users/{user_id}/reissue-invite", headers={"X-Admin-Key": ADMIN_KEY})
    assert r.status_code == 200
    c.cookies.clear()
    new_token = _token_from(r.json()["invite_url"])
    assert c.post(f"/auth/invite/{new_token}", json={"password": "reset-password"}).status_code == 200
    assert c.get("/portal/me").status_code == 200


def test_reissue_requires_admin_key():
    assert _client().post("/admin/users/whatever/reissue-invite").status_code == 401


def test_deleting_a_user_ends_their_session():
    c = _client()
    _cfg, invite_url = _tenant_with_login(c)
    c.post(f"/auth/invite/{_token_from(invite_url)}", json={"password": PASSWORD})
    user_id = c.get("/auth/me").json()["user"]["user_id"]
    assert c.delete(f"/admin/users/{user_id}",
                    headers={"X-Admin-Key": ADMIN_KEY}).status_code == 200
    assert c.get("/portal/me").status_code == 401


def test_admin_can_list_a_tenants_users():
    c = _client()
    cfg, _invite = _tenant_with_login(c)
    r = c.get(f"/admin/tenants/{cfg.tenant_id}/users", headers={"X-Admin-Key": ADMIN_KEY})
    assert r.status_code == 200
    assert [u["email"] for u in r.json()] == ["owner@gng.com.np"]
