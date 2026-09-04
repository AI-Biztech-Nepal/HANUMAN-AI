"""
Portal authentication — real user accounts for customer logins.

Pilot phase: SQLite (same DB file as leads/tenants), stdlib crypto only.
No external auth dependency; the Postgres schema is in db/schema.sql.

Onboarding is invite-based, matching Phase 1-2 hand-onboarding: an admin
creates a tenant's first user, hands over the one-time invite link, and the
customer sets their own password. There is no public signup endpoint.

Tenant API keys (tk_...) keep working for programmatic/API access — the
portal UI uses sessions, integrations use the key.
"""
from __future__ import annotations

import hashlib
import hmac
import re
import secrets
import sqlite3
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

from . import config

SESSION_TTL_DAYS = 30          # portal is a mobile PWA; long sessions are expected
INVITE_TTL_DAYS = 7
MIN_PASSWORD_LENGTH = 8

# Roles, widest first. A company runs its own team inside the portal: the owner
# invites staff, and not everyone who answers leads should be able to rewrite
# what the agent says on every call.
ROLE_OWNER = "owner"       # everything, including the team and billing
ROLE_MANAGER = "manager"   # agent script, leads, usage — but not the team
ROLE_STAFF = "staff"       # leads only
ROLES = (ROLE_OWNER, ROLE_MANAGER, ROLE_STAFF)

# capability -> roles that hold it. Checked in main.py on every portal write.
PERMISSIONS = {
    "leads:read":   {ROLE_OWNER, ROLE_MANAGER, ROLE_STAFF},
    "agent:read":   {ROLE_OWNER, ROLE_MANAGER, ROLE_STAFF},
    "agent:write":  {ROLE_OWNER, ROLE_MANAGER},
    "usage:read":   {ROLE_OWNER, ROLE_MANAGER},
    "team:read":    {ROLE_OWNER, ROLE_MANAGER},
    "team:write":   {ROLE_OWNER},
}


def can(role: str, capability: str) -> bool:
    return role in PERMISSIONS.get(capability, set())

# scrypt cost. n=2**14 keeps a login around ~50-100ms on a small VPS, which is
# slow enough to matter to an attacker and fast enough not to block a call.
_SCRYPT_N, _SCRYPT_R, _SCRYPT_P = 2 ** 14, 8, 1

# Login throttle. In-memory is correct for the single-process pilot; move to
# the DB if uvicorn ever runs multiple workers.
MAX_FAILED_ATTEMPTS = 5
LOCKOUT_SECONDS = 300
_failed: dict[str, list[float]] = {}

_EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")


class AuthError(Exception):
    """Raised for any auth failure that is safe to surface to the caller."""


@dataclass
class User:
    user_id: str
    tenant_id: str
    email: str
    status: str = "active"          # active | suspended | invited
    created_at: str = ""
    last_login_at: str = ""
    role: str = ROLE_STAFF          # least privilege unless stated otherwise

    def to_dict(self) -> dict:
        return {
            "user_id": self.user_id,
            "tenant_id": self.tenant_id,
            "email": self.email,
            "status": self.status,
            "created_at": self.created_at,
            "last_login_at": self.last_login_at,
            "role": self.role,
        }

    def can(self, capability: str) -> bool:
        return can(self.role, capability)


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _iso(dt: datetime) -> str:
    return dt.isoformat()


def _conn():
    conn = sqlite3.connect(config.LEADS_DB_PATH)
    conn.execute(
        """CREATE TABLE IF NOT EXISTS users (
            user_id       TEXT PRIMARY KEY,
            tenant_id     TEXT NOT NULL,
            email         TEXT NOT NULL UNIQUE COLLATE NOCASE,
            password_hash TEXT NOT NULL DEFAULT '',
            status        TEXT NOT NULL DEFAULT 'invited',
            created_at    TEXT NOT NULL,
            last_login_at TEXT NOT NULL DEFAULT '',
            role          TEXT NOT NULL DEFAULT 'staff'
        )"""
    )
    # Migration for databases created before roles existed. Accounts that
    # predate this were a tenant's only login and had unrestricted access, so
    # they become owners — demoting them silently would lock people out.
    if "role" not in {r[1] for r in conn.execute("PRAGMA table_info(users)")}:
        conn.execute("ALTER TABLE users ADD COLUMN role TEXT NOT NULL DEFAULT 'staff'")
        conn.execute("UPDATE users SET role = 'owner'")
    # Tokens are stored hashed: a dump of this table must not hand anyone a
    # live session or a usable invite.
    conn.execute(
        """CREATE TABLE IF NOT EXISTS sessions (
            token_hash TEXT PRIMARY KEY,
            user_id    TEXT NOT NULL,
            created_at TEXT NOT NULL,
            expires_at TEXT NOT NULL
        )"""
    )
    conn.execute(
        """CREATE TABLE IF NOT EXISTS invites (
            token_hash TEXT PRIMARY KEY,
            user_id    TEXT NOT NULL,
            expires_at TEXT NOT NULL,
            used_at    TEXT NOT NULL DEFAULT ''
        )"""
    )
    conn.execute("CREATE INDEX IF NOT EXISTS idx_users_tenant ON users (tenant_id)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_sessions_user ON sessions (user_id)")
    return conn


# ------------------------------------------------------------ password hashing

def hash_password(password: str) -> str:
    salt = secrets.token_bytes(16)
    dk = hashlib.scrypt(password.encode("utf-8"), salt=salt,
                        n=_SCRYPT_N, r=_SCRYPT_R, p=_SCRYPT_P, dklen=32)
    return f"scrypt${_SCRYPT_N}${_SCRYPT_R}${_SCRYPT_P}${salt.hex()}${dk.hex()}"


def verify_password(password: str, stored: str) -> bool:
    if not stored:
        return False
    try:
        algo, n, r, p, salt_hex, hash_hex = stored.split("$")
        if algo != "scrypt":
            return False
        dk = hashlib.scrypt(password.encode("utf-8"), salt=bytes.fromhex(salt_hex),
                            n=int(n), r=int(r), p=int(p), dklen=len(hash_hex) // 2)
    except (ValueError, TypeError):
        return False
    return hmac.compare_digest(dk.hex(), hash_hex)


def _token_hash(token: str) -> str:
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


def validate_password(password: str) -> None:
    if len(password or "") < MIN_PASSWORD_LENGTH:
        raise AuthError(f"password must be at least {MIN_PASSWORD_LENGTH} characters")


def normalize_email(email: str) -> str:
    email = (email or "").strip().lower()
    if not _EMAIL_RE.match(email):
        raise AuthError("a valid email address is required")
    return email


# --------------------------------------------------------------- user records

def _row_to_user(row) -> User:
    return User(user_id=row[0], tenant_id=row[1], email=row[2],
                status=row[4], created_at=row[5], last_login_at=row[6],
                role=row[7] if len(row) > 7 else ROLE_STAFF)


_USER_COLS = ("user_id, tenant_id, email, password_hash, status, "
              "created_at, last_login_at, role")


def validate_role(role: str) -> str:
    if role not in ROLES:
        raise AuthError(f"role must be one of: {', '.join(ROLES)}")
    return role


def create_user(tenant_id: str, email: str, role: str = ROLE_STAFF) -> tuple[User, str]:
    """Create an invited user. Returns (user, one-time invite token).

    The token is returned once and never recoverable — only its hash is
    stored, so a lost invite means issuing a new one.
    """
    email = normalize_email(email)
    validate_role(role)
    user = User(
        user_id=secrets.token_hex(8),
        tenant_id=tenant_id,
        email=email,
        status="invited",
        created_at=_iso(_now()),
        role=role,
    )
    invite_token = secrets.token_urlsafe(32)
    try:
        with _conn() as conn:
            conn.execute(
                f"INSERT INTO users ({_USER_COLS}) VALUES (?,?,?,?,?,?,?,?)",
                (user.user_id, user.tenant_id, user.email, "",
                 user.status, user.created_at, "", user.role),
            )
            conn.execute(
                "INSERT INTO invites (token_hash, user_id, expires_at) VALUES (?,?,?)",
                (_token_hash(invite_token), user.user_id,
                 _iso(_now() + timedelta(days=INVITE_TTL_DAYS))),
            )
    except sqlite3.IntegrityError as exc:
        raise AuthError("that email already has an account") from exc
    return user, invite_token


def get_user(user_id: str) -> User | None:
    with _conn() as conn:
        row = conn.execute(
            f"SELECT {_USER_COLS} FROM users WHERE user_id = ?", (user_id,)
        ).fetchone()
    return _row_to_user(row) if row else None


def get_user_by_email(email: str) -> User | None:
    with _conn() as conn:
        row = conn.execute(
            f"SELECT {_USER_COLS} FROM users WHERE email = ? COLLATE NOCASE",
            ((email or "").strip().lower(),),
        ).fetchone()
    return _row_to_user(row) if row else None


def list_users(tenant_id: str) -> list[dict]:
    with _conn() as conn:
        rows = conn.execute(
            f"SELECT {_USER_COLS} FROM users WHERE tenant_id = ? ORDER BY created_at",
            (tenant_id,),
        ).fetchall()
    return [_row_to_user(r).to_dict() for r in rows]


def count_owners(tenant_id: str, excluding: str = "") -> int:
    with _conn() as conn:
        return conn.execute(
            "SELECT COUNT(*) FROM users WHERE tenant_id = ? AND role = ? "
            "AND status != 'suspended' AND user_id != ?",
            (tenant_id, ROLE_OWNER, excluding),
        ).fetchone()[0]


def _guard_last_owner(user: User, force: bool) -> None:
    """A company that loses its last owner can no longer manage its own team
    or billing, so the portal refuses the step that would strand it.

    `force` is the platform operator's override: they are the party who
    resolves a stranded company, and offboarding one means removing its last
    login. Only admin endpoints pass it.
    """
    if force:
        return
    if user.role == ROLE_OWNER and count_owners(user.tenant_id, excluding=user.user_id) == 0:
        raise AuthError("this is the only owner — promote someone else first")


def set_role(user_id: str, role: str, force: bool = False) -> User:
    validate_role(role)
    user = get_user(user_id)
    if user is None:
        raise AuthError("no such user")
    if user.role != role:
        _guard_last_owner(user, force)
    with _conn() as conn:
        conn.execute("UPDATE users SET role = ? WHERE user_id = ?", (role, user_id))
    user.role = role
    return user


def delete_user(user_id: str, force: bool = False) -> None:
    user = get_user(user_id)
    if user is not None:
        _guard_last_owner(user, force)
    with _conn() as conn:
        conn.execute("DELETE FROM users WHERE user_id = ?", (user_id,))
        conn.execute("DELETE FROM sessions WHERE user_id = ?", (user_id,))
        conn.execute("DELETE FROM invites WHERE user_id = ?", (user_id,))


def set_status(user_id: str, status: str, force: bool = False) -> None:
    """Suspend or reactivate. Suspending also kills every live session."""
    if status != "active":
        user = get_user(user_id)
        if user is not None:
            _guard_last_owner(user, force)
    with _conn() as conn:
        conn.execute("UPDATE users SET status = ? WHERE user_id = ?", (status, user_id))
        if status != "active":
            conn.execute("DELETE FROM sessions WHERE user_id = ?", (user_id,))


# -------------------------------------------------------------------- invites

def peek_invite(token: str) -> User | None:
    """Resolve an unused, unexpired invite to its user — for rendering the
    'set your password' screen without consuming the token."""
    with _conn() as conn:
        row = conn.execute(
            "SELECT user_id, expires_at, used_at FROM invites WHERE token_hash = ?",
            (_token_hash(token or ""),),
        ).fetchone()
    if row is None or row[2]:
        return None
    if datetime.fromisoformat(row[1]) < _now():
        return None
    return get_user(row[0])


def accept_invite(token: str, password: str) -> User:
    """Consume an invite and set the user's first password."""
    validate_password(password)
    user = peek_invite(token)
    if user is None:
        raise AuthError("this invite link is invalid or has expired")
    with _conn() as conn:
        conn.execute(
            "UPDATE users SET password_hash = ?, status = 'active' WHERE user_id = ?",
            (hash_password(password), user.user_id),
        )
        conn.execute(
            "UPDATE invites SET used_at = ? WHERE token_hash = ?",
            (_iso(_now()), _token_hash(token)),
        )
    user.status = "active"
    return user


def reissue_invite(user_id: str) -> str:
    """Issue a fresh invite (lost link, or a password reset by the operator).
    Any previous invite for the user stops working."""
    if get_user(user_id) is None:
        raise AuthError("no such user")
    token = secrets.token_urlsafe(32)
    with _conn() as conn:
        conn.execute("DELETE FROM invites WHERE user_id = ?", (user_id,))
        conn.execute(
            "INSERT INTO invites (token_hash, user_id, expires_at) VALUES (?,?,?)",
            (_token_hash(token), user_id, _iso(_now() + timedelta(days=INVITE_TTL_DAYS))),
        )
    return token


# ------------------------------------------------------------------- throttle

def _throttle_key(email: str) -> str:
    return (email or "").strip().lower()


def _check_throttle(email: str) -> None:
    now = time.time()
    hits = [t for t in _failed.get(_throttle_key(email), []) if now - t < LOCKOUT_SECONDS]
    _failed[_throttle_key(email)] = hits
    if len(hits) >= MAX_FAILED_ATTEMPTS:
        wait = int(LOCKOUT_SECONDS - (now - hits[0]))
        raise AuthError(f"too many failed attempts — try again in {wait} seconds")


def _record_failure(email: str) -> None:
    _failed.setdefault(_throttle_key(email), []).append(time.time())


def _clear_failures(email: str) -> None:
    _failed.pop(_throttle_key(email), None)


# ------------------------------------------------------------------- sessions

def login(email: str, password: str) -> tuple[User, str]:
    """Verify credentials and open a session. Returns (user, session token)."""
    _check_throttle(email)
    user = get_user_by_email(email)
    with _conn() as conn:
        row = conn.execute(
            "SELECT password_hash FROM users WHERE email = ? COLLATE NOCASE",
            ((email or "").strip().lower(),),
        ).fetchone()
    stored = row[0] if row else ""

    # Always run the hash, even for an unknown email, so response time does
    # not reveal which addresses have accounts.
    ok = verify_password(password or "", stored or hash_password(secrets.token_hex(16)))
    if user is None or not ok or user.status != "active":
        _record_failure(email)
        raise AuthError("incorrect email or password")

    _clear_failures(email)
    token = secrets.token_urlsafe(32)
    now = _now()
    with _conn() as conn:
        conn.execute(
            "INSERT INTO sessions (token_hash, user_id, created_at, expires_at) VALUES (?,?,?,?)",
            (_token_hash(token), user.user_id, _iso(now),
             _iso(now + timedelta(days=SESSION_TTL_DAYS))),
        )
        conn.execute("UPDATE users SET last_login_at = ? WHERE user_id = ?",
                     (_iso(now), user.user_id))
    user.last_login_at = _iso(now)
    return user, token


def user_for_session(token: str) -> User | None:
    if not token:
        return None
    with _conn() as conn:
        row = conn.execute(
            "SELECT user_id, expires_at FROM sessions WHERE token_hash = ?",
            (_token_hash(token),),
        ).fetchone()
    if row is None:
        return None
    if datetime.fromisoformat(row[1]) < _now():
        logout(token)
        return None
    user = get_user(row[0])
    if user is None or user.status != "active":
        return None
    return user


def logout(token: str) -> None:
    if not token:
        return
    with _conn() as conn:
        conn.execute("DELETE FROM sessions WHERE token_hash = ?", (_token_hash(token),))


def change_password(user_id: str, current: str, new: str) -> None:
    """Change a password, then drop every other session for that user."""
    validate_password(new)
    with _conn() as conn:
        row = conn.execute(
            "SELECT password_hash FROM users WHERE user_id = ?", (user_id,)
        ).fetchone()
    if row is None or not verify_password(current or "", row[0]):
        raise AuthError("current password is incorrect")
    with _conn() as conn:
        conn.execute("UPDATE users SET password_hash = ? WHERE user_id = ?",
                     (hash_password(new), user_id))
        conn.execute("DELETE FROM sessions WHERE user_id = ?", (user_id,))


def purge_expired() -> int:
    """Drop expired sessions and invites. Safe to call on startup."""
    now = _iso(_now())
    with _conn() as conn:
        cur = conn.execute("DELETE FROM sessions WHERE expires_at < ?", (now,))
        removed = cur.rowcount or 0
        conn.execute("DELETE FROM invites WHERE expires_at < ? AND used_at = ''", (now,))
    return removed
