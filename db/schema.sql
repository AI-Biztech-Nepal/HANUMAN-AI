-- Postgres schema for scale (Phase 3+). Pilot runs on SQLite automatically.
-- See docs/ARCHITECTURE.md §3 for the full rationale.

CREATE TABLE tenants (
  id            UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  name          TEXT NOT NULL,
  status        TEXT NOT NULL DEFAULT 'trial',      -- trial|active|suspended
  plan          TEXT NOT NULL DEFAULT 'starter',    -- starter|business|pro
  included_min  INT  NOT NULL DEFAULT 1000,
  created_at    TIMESTAMPTZ DEFAULT now()
);

CREATE TABLE phone_numbers (
  id         UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  tenant_id  UUID NOT NULL REFERENCES tenants(id),
  e164       TEXT UNIQUE NOT NULL,
  trunk      TEXT NOT NULL,                         -- 'platform' | 'byo'
  direction  TEXT NOT NULL DEFAULT 'inbound'
);

CREATE TABLE agent_configs (
  tenant_id   UUID PRIMARY KEY REFERENCES tenants(id),
  agent_name  TEXT NOT NULL DEFAULT 'Asha',
  language    TEXT NOT NULL DEFAULT 'auto',
  greeting    TEXT,
  facts       TEXT,
  questions   JSONB,
  lead_schema JSONB,
  transfer_to TEXT,
  hours       JSONB
);

-- Portal logins. Invite-based: an operator creates the row, the customer sets
-- the password. Tokens are stored hashed so a dump grants nothing.
CREATE TABLE users (
  id            UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  tenant_id     UUID NOT NULL REFERENCES tenants(id) ON DELETE CASCADE,
  email         TEXT UNIQUE NOT NULL,
  password_hash TEXT NOT NULL DEFAULT '',
  status        TEXT NOT NULL DEFAULT 'invited',    -- invited|active|suspended
  created_at    TIMESTAMPTZ DEFAULT now(),
  last_login_at TIMESTAMPTZ
);

CREATE TABLE sessions (
  token_hash TEXT PRIMARY KEY,                      -- sha256 of the cookie value
  user_id    UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
  created_at TIMESTAMPTZ DEFAULT now(),
  expires_at TIMESTAMPTZ NOT NULL
);

CREATE TABLE invites (
  token_hash TEXT PRIMARY KEY,                      -- sha256 of the invite token
  user_id    UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
  expires_at TIMESTAMPTZ NOT NULL,
  used_at    TIMESTAMPTZ
);

CREATE TABLE calls (
  id            UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  tenant_id     UUID NOT NULL REFERENCES tenants(id),
  direction     TEXT NOT NULL,
  caller        TEXT,
  started_at    TIMESTAMPTZ NOT NULL,
  duration_sec  INT,
  outcome       TEXT,
  transcript    JSONB,
  recording_url TEXT
);

CREATE TABLE leads (
  id         UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  tenant_id  UUID NOT NULL REFERENCES tenants(id),
  call_id    UUID REFERENCES calls(id),
  data       JSONB NOT NULL,
  created_at TIMESTAMPTZ DEFAULT now()
);

CREATE TABLE usage_monthly (
  tenant_id  UUID REFERENCES tenants(id),
  month      DATE,
  minutes    NUMERIC DEFAULT 0,
  calls      INT DEFAULT 0,
  claude_usd NUMERIC DEFAULT 0,
  PRIMARY KEY (tenant_id, month)
);

CREATE TABLE do_not_call (
  e164       TEXT NOT NULL,
  tenant_id  UUID,                                  -- NULL = platform-wide block
  created_at TIMESTAMPTZ DEFAULT now(),
  PRIMARY KEY (e164, tenant_id)
);

CREATE INDEX ON calls (tenant_id, started_at DESC);
CREATE INDEX ON leads (tenant_id, created_at DESC);
CREATE INDEX ON users (tenant_id);
CREATE INDEX ON sessions (user_id);
CREATE INDEX ON sessions (expires_at);
