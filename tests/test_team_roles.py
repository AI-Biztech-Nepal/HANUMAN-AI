"""Team roles: a company runs its own staff inside the portal.

Covers who may do what, the two doors (session vs tenant key), and the
lockout guards that keep a company from stranding itself without an owner.
"""
import pytest
from fastapi.testclient import TestClient

from app import auth, tenants
from app.main import app

ADMIN_KEY = "test-admin-key"


def _tenant():
    return tenants.create(company_name="G&G Automobiles")


def _member(tenant_id: str, email: str, role: str) -> str:
    """Create a user at `role` with a known password. Returns the password."""
    user, token = auth.create_user(tenant_id, email, role=role)
    auth.accept_invite(token, "a-good-password")
    return "a-good-password"


def _signed_in(tenant_id: str, email: str, role: str) -> TestClient:
    password = _member(tenant_id, email, role)
    c = TestClient(app)
    resp = c.post("/auth/login", json={"email": email, "password": password})
    assert resp.status_code == 200, resp.text
    return c


# ------------------------------------------------------------------ the table

def test_permission_table_matches_roles():
    assert auth.can(auth.ROLE_OWNER, "team:write")
    assert not auth.can(auth.ROLE_MANAGER, "team:write")
    assert not auth.can(auth.ROLE_STAFF, "agent:write")
    assert auth.can(auth.ROLE_MANAGER, "agent:write")
    # Everyone who can sign in can work the leads.
    for role in auth.ROLES:
        assert auth.can(role, "leads:read")


def test_unknown_capability_is_denied_not_granted():
    assert not auth.can(auth.ROLE_OWNER, "billing:refund")


# ------------------------------------------------------------ agent:write gate

def test_staff_cannot_edit_the_agent_script():
    cfg = _tenant()
    c = _signed_in(cfg.tenant_id, "staff@gng.com.np", auth.ROLE_STAFF)
    resp = c.put("/portal/me", json={"agent_name": "Hacked"})
    assert resp.status_code == 403
    assert tenants.get(cfg.tenant_id).agent_name != "Hacked"


def test_manager_can_edit_the_agent_script():
    cfg = _tenant()
    c = _signed_in(cfg.tenant_id, "manager@gng.com.np", auth.ROLE_MANAGER)
    resp = c.put("/portal/me", json={"agent_name": "Aashika"})
    assert resp.status_code == 200
    assert tenants.get(cfg.tenant_id).agent_name == "Aashika"


def test_staff_can_still_read_leads_and_agent_config():
    cfg = _tenant()
    c = _signed_in(cfg.tenant_id, "staff2@gng.com.np", auth.ROLE_STAFF)
    assert c.get("/portal/leads").status_code == 200
    assert c.get("/portal/me").status_code == 200


def test_staff_cannot_see_usage():
    cfg = _tenant()
    c = _signed_in(cfg.tenant_id, "staff3@gng.com.np", auth.ROLE_STAFF)
    assert c.get("/portal/usage").status_code == 403


# -------------------------------------------------------------- team endpoints

def test_owner_invites_a_colleague_and_gets_a_link():
    cfg = _tenant()
    c = _signed_in(cfg.tenant_id, "owner@gng.com.np", auth.ROLE_OWNER)
    resp = c.post("/portal/team",
                  json={"email": "newhire@gng.com.np", "role": auth.ROLE_STAFF})
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert "invite=" in body["invite_url"]
    assert body["user"]["role"] == auth.ROLE_STAFF
    assert body["user"]["status"] == "invited"


def test_invited_colleague_can_set_a_password_and_sign_in():
    cfg = _tenant()
    owner = _signed_in(cfg.tenant_id, "owner2@gng.com.np", auth.ROLE_OWNER)
    url = owner.post("/portal/team",
                     json={"email": "hire2@gng.com.np"}).json()["invite_url"]
    token = url.split("invite=")[1]

    c = TestClient(app)
    assert c.get(f"/auth/invite/{token}").status_code == 200
    assert c.post(f"/auth/invite/{token}",
                  json={"password": "another-good-one"}).status_code == 200
    assert c.get("/auth/me").json()["user"]["email"] == "hire2@gng.com.np"


def test_manager_may_read_the_team_but_not_change_it():
    cfg = _tenant()
    c = _signed_in(cfg.tenant_id, "manager2@gng.com.np", auth.ROLE_MANAGER)
    assert c.get("/portal/team").status_code == 200
    assert c.post("/portal/team", json={"email": "x@gng.com.np"}).status_code == 403


def test_staff_cannot_even_list_the_team():
    cfg = _tenant()
    c = _signed_in(cfg.tenant_id, "staff4@gng.com.np", auth.ROLE_STAFF)
    assert c.get("/portal/team").status_code == 403


def test_owner_changes_a_colleagues_role():
    cfg = _tenant()
    c = _signed_in(cfg.tenant_id, "owner3@gng.com.np", auth.ROLE_OWNER)
    uid = c.post("/portal/team",
                 json={"email": "promote@gng.com.np"}).json()["user"]["user_id"]
    resp = c.patch(f"/portal/team/{uid}", json={"role": auth.ROLE_MANAGER})
    assert resp.status_code == 200
    assert resp.json()["role"] == auth.ROLE_MANAGER


def test_suspending_a_colleague_ends_their_access():
    cfg = _tenant()
    owner = _signed_in(cfg.tenant_id, "owner4@gng.com.np", auth.ROLE_OWNER)
    password = _member(cfg.tenant_id, "temp@gng.com.np", auth.ROLE_STAFF)
    staff = TestClient(app)
    staff.post("/auth/login", json={"email": "temp@gng.com.np", "password": password})
    assert staff.get("/portal/leads").status_code == 200

    uid = auth.get_user_by_email("temp@gng.com.np").user_id
    assert owner.patch(f"/portal/team/{uid}",
                       json={"status": "suspended"}).status_code == 200
    assert staff.get("/portal/leads").status_code == 401


# --------------------------------------------------------------- tenant limits

def test_cannot_touch_a_member_of_another_company():
    a, b = _tenant(), _tenant()
    outsider = auth.create_user(b.tenant_id, "other@rival.com.np", auth.ROLE_STAFF)[0]
    c = _signed_in(a.tenant_id, "owner5@gng.com.np", auth.ROLE_OWNER)
    assert c.get("/portal/team").status_code == 200
    assert all(u["tenant_id"] == a.tenant_id for u in c.get("/portal/team").json())
    assert c.delete(f"/portal/team/{outsider.user_id}").status_code == 404
    assert auth.get_user(outsider.user_id) is not None


# ------------------------------------------------------------- lockout guards

def test_the_last_owner_cannot_be_demoted():
    cfg = _tenant()
    auth.create_user(cfg.tenant_id, "solo@gng.com.np", auth.ROLE_OWNER)
    uid = auth.get_user_by_email("solo@gng.com.np").user_id
    with pytest.raises(auth.AuthError):
        auth.set_role(uid, auth.ROLE_STAFF)
    assert auth.get_user(uid).role == auth.ROLE_OWNER


def test_the_last_owner_cannot_be_deleted_or_suspended():
    cfg = _tenant()
    auth.create_user(cfg.tenant_id, "solo2@gng.com.np", auth.ROLE_OWNER)
    uid = auth.get_user_by_email("solo2@gng.com.np").user_id
    with pytest.raises(auth.AuthError):
        auth.delete_user(uid)
    with pytest.raises(auth.AuthError):
        auth.set_status(uid, "suspended")
    assert auth.get_user(uid) is not None


def test_a_second_owner_frees_the_first_to_step_down():
    cfg = _tenant()
    auth.create_user(cfg.tenant_id, "first@gng.com.np", auth.ROLE_OWNER)
    auth.create_user(cfg.tenant_id, "second@gng.com.np", auth.ROLE_OWNER)
    uid = auth.get_user_by_email("first@gng.com.np").user_id
    auth.set_role(uid, auth.ROLE_MANAGER)
    assert auth.get_user(uid).role == auth.ROLE_MANAGER


def test_owner_cannot_demote_or_remove_themselves():
    cfg = _tenant()
    auth.create_user(cfg.tenant_id, "spare@gng.com.np", auth.ROLE_OWNER)
    c = _signed_in(cfg.tenant_id, "owner6@gng.com.np", auth.ROLE_OWNER)
    uid = auth.get_user_by_email("owner6@gng.com.np").user_id
    assert c.patch(f"/portal/team/{uid}", json={"role": auth.ROLE_STAFF}).status_code == 400
    assert c.delete(f"/portal/team/{uid}").status_code == 400


# ------------------------------------------------------------------ both doors

def test_tenant_api_key_is_not_role_limited():
    """Integrations authenticate as the company, not as a person."""
    cfg = _tenant()
    c = TestClient(app)
    headers = {"X-Tenant-Key": cfg.api_key}
    assert c.put("/portal/me", json={"agent_name": "Bot"}, headers=headers).status_code == 200
    assert c.get("/portal/team", headers=headers).status_code == 200


def test_auth_me_reports_the_role_and_capabilities():
    cfg = _tenant()
    c = _signed_in(cfg.tenant_id, "owner7@gng.com.np", auth.ROLE_OWNER)
    body = c.get("/auth/me").json()
    assert body["user"]["role"] == auth.ROLE_OWNER
    assert body["can"]["team:write"] is True
    assert body["can"]["agent:write"] is True


def test_first_operator_created_user_is_an_owner_and_the_next_is_not():
    cfg = _tenant()
    c = TestClient(app)
    h = {"X-Admin-Key": ADMIN_KEY}
    first = c.post(f"/admin/tenants/{cfg.tenant_id}/users",
                   json={"email": "boss@gng.com.np"}, headers=h).json()
    second = c.post(f"/admin/tenants/{cfg.tenant_id}/users",
                    json={"email": "clerk@gng.com.np"}, headers=h).json()
    assert first["user"]["role"] == auth.ROLE_OWNER
    assert second["user"]["role"] == auth.ROLE_STAFF
