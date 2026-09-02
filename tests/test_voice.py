"""Voice-call plumbing.

Deliberately model-free: these run on a machine with no whisper/piper (the
Windows dev venv, CI) as well as on the host that actually serves voice. What
matters here is that the app degrades cleanly and never leaks a voice call to
someone who isn't signed in — the models themselves are exercised by hand.
"""
import pytest
from fastapi.testclient import TestClient

from app import auth, tenants, voice
from app.main import app

ADMIN_KEY = "test-admin-key"
PASSWORD = "password-1234"


@pytest.fixture(autouse=True)
def clear_throttle():
    auth._failed.clear()
    yield
    auth._failed.clear()


@pytest.fixture
def no_pipeline(monkeypatch):
    """A host without the speech pipeline installed."""
    monkeypatch.setattr(voice, "_bridge", None)
    monkeypatch.setattr(voice, "_load_error", "ModuleNotFoundError: no piper here")
    monkeypatch.setattr(voice, "_get_bridge", lambda: None)


def _client() -> TestClient:
    return TestClient(app)


def _signed_in(c) -> tenants.TenantConfig:
    cfg = tenants.create(company_name="G&G Automobiles")
    r = c.post(f"/admin/tenants/{cfg.tenant_id}/users",
               json={"email": "owner@gng.com.np"}, headers={"X-Admin-Key": ADMIN_KEY})
    token = r.json()["invite_url"].split("invite=", 1)[1]
    c.post(f"/auth/invite/{token}", json={"password": PASSWORD})
    return cfg


# ------------------------------------------------------------ degrading well

def test_status_reports_unavailable_instead_of_raising(no_pipeline):
    s = voice.status()
    assert s["available"] is False
    assert s["reason"]


def test_speak_and_transcribe_raise_a_clear_error_without_the_pipeline(no_pipeline):
    with pytest.raises(RuntimeError):
        voice.speak("hello")
    with pytest.raises(RuntimeError):
        voice.transcribe(b"\x00\x01")


def test_warmup_is_a_no_op_without_the_pipeline(no_pipeline):
    voice.warmup()      # must not raise


def test_importing_the_app_never_requires_the_pipeline():
    # The portal has to serve on hosts with no speech models at all; if this
    # import needed them, the whole product would be Linux-only.
    import app.main                                   # noqa: F401
    assert _client().get("/health").status_code == 200


# ----------------------------------------------------------------- auth gates

def test_voice_status_requires_authentication():
    assert _client().get("/portal/voice-status").status_code == 401


def test_voice_status_is_readable_when_signed_in():
    c = _client()
    _signed_in(c)
    r = c.get("/portal/voice-status")
    assert r.status_code == 200
    assert "available" in r.json()


def test_voice_status_accepts_a_tenant_api_key_too():
    c = _client()
    cfg = tenants.create(company_name="Integration Co")
    r = c.get("/portal/voice-status", headers={"X-Tenant-Key": cfg.api_key})
    assert r.status_code == 200


def test_voice_websocket_refuses_an_unauthenticated_caller():
    c = _client()
    with c.websocket_connect("/portal/ws/voice") as ws:
        msg = ws.receive_json()
    assert msg["type"] == "error"
    assert "not signed in" in msg["detail"]


def test_voice_websocket_refuses_when_the_pipeline_is_missing(no_pipeline):
    c = _client()
    _signed_in(c)
    with c.websocket_connect("/portal/ws/voice") as ws:
        msg = ws.receive_json()
    assert msg["type"] == "error"
    assert "unavailable" in msg["detail"]


def test_voice_websocket_refuses_a_suspended_tenant(monkeypatch):
    c = _client()
    cfg = _signed_in(c)
    cfg.status = "suspended"
    tenants.save(cfg)
    with c.websocket_connect("/portal/ws/voice") as ws:
        msg = ws.receive_json()
    assert msg["type"] == "error"


# ------------------------------------------------------------------ synthesis

def test_speak_picks_the_voice_matching_the_language(monkeypatch):
    calls = {}

    class FakeBridge:
        PIPER_VOICE_NE = "/voices/ne.onnx"
        PIPER_VOICE_EN = "/voices/en.onnx"

        @staticmethod
        def voice_for_text(text):
            return "/voices/auto.onnx"

        @staticmethod
        def synthesize(text, out_wav, voice, telephony):
            calls["voice"] = voice
            calls["telephony"] = telephony
            with open(out_wav, "wb") as f:
                f.write(b"RIFFfake")
            return out_wav

    monkeypatch.setattr(voice, "_get_bridge", lambda: FakeBridge)

    assert voice.speak("नमस्कार", "ne") == b"RIFFfake"
    assert calls["voice"] == "/voices/ne.onnx"
    # Browser audio must not be downsampled to the 8kHz telephony rate.
    assert calls["telephony"] is False

    voice.speak("hello", "en")
    assert calls["voice"] == "/voices/en.onnx"

    voice.speak("mixed text", None)
    assert calls["voice"] == "/voices/auto.onnx"


def test_transcribe_cleans_up_its_temp_file(monkeypatch):
    seen = {}

    class FakeBridge:
        @staticmethod
        def transcribe_wav(path, language=None):
            seen["path"] = path
            seen["language"] = language
            return "transcribed"

    monkeypatch.setattr(voice, "_get_bridge", lambda: FakeBridge)
    import os

    assert voice.transcribe(b"audio-bytes", "ne") == "transcribed"
    assert seen["language"] == "ne"
    assert seen["path"].endswith(".webm")
    assert not os.path.exists(seen["path"]), "temp audio must not be left on disk"
