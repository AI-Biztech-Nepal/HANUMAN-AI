"""Central configuration loaded from .env"""
import os
from dotenv import load_dotenv

load_dotenv()

ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
# Haiku keeps per-call cost ~5x lower than Sonnet — the margin of the business.
CLAUDE_MODEL = os.getenv("CLAUDE_MODEL", "claude-haiku-4-5-20251001")
ADMIN_API_KEY = os.getenv("ADMIN_API_KEY", "")

COMPANY_NAME = os.getenv("COMPANY_NAME", "Your Company")
AGENT_NAME = os.getenv("AGENT_NAME", "Asha")
DEFAULT_LANGUAGE = os.getenv("DEFAULT_LANGUAGE", "auto")

TWILIO_ACCOUNT_SID = os.getenv("TWILIO_ACCOUNT_SID", "")
TWILIO_AUTH_TOKEN = os.getenv("TWILIO_AUTH_TOKEN", "")
TWILIO_PHONE_NUMBER = os.getenv("TWILIO_PHONE_NUMBER", "")
PUBLIC_BASE_URL = os.getenv("PUBLIC_BASE_URL", "")

WHISPER_MODEL = os.getenv("WHISPER_MODEL", "small")
TTS_VOICE = os.getenv("TTS_VOICE", "en_US-amy-medium")

# Re-timbre Piper's speech to the agent's own voice (app/tone.py). On: the
# agent must sound like its own person, not like whoever recorded the base
# model. The conversion moves timbre but not rhythm, so it is a likeness
# rather than the person, and it costs some high-frequency detail — the
# closest we get until that voice has its own trained model, which needs no
# conversion pass at all; see tools/train_voice.py.
#
# Set VOICE_TONE_CONVERSION=0 to fall back to the plain base voice.
VOICE_TONE_CONVERSION = os.getenv("VOICE_TONE_CONVERSION", "1").lower() in (
    "1", "true", "yes", "on")

LEADS_DB_PATH = os.getenv("LEADS_DB_PATH", "./leads.db")

# Asterisk ARI bridge (media/asterisk_bridge.py) — Phase 1, groundwork.
# Bridge must run on the same host as Asterisk (EXTERNAL_MEDIA_HOST and
# ASTERISK_SOUNDS_DIR both assume local access).
ARI_BASE_URL = os.getenv("ARI_BASE_URL", "http://127.0.0.1:8088/ari")
ARI_USERNAME = os.getenv("ARI_USERNAME", "")
ARI_PASSWORD = os.getenv("ARI_PASSWORD", "")
ARI_APP_NAME = os.getenv("ARI_APP_NAME", "hanuman")
EXTERNAL_MEDIA_HOST = os.getenv("EXTERNAL_MEDIA_HOST", "127.0.0.1")
ASTERISK_SOUNDS_DIR = os.getenv("ASTERISK_SOUNDS_DIR", "/var/lib/asterisk/sounds/custom")
# Where the agent core's /ws/chat lives, from the bridge's perspective — override
# when the bridge runs on a different host/VM than uvicorn (e.g. WSL vs. Windows).
AGENT_WS_URL = os.getenv("AGENT_WS_URL", "ws://127.0.0.1:8000/ws/chat")

# Outbound email (app/notify.py). Optional: with no SMTP_HOST the portal logs
# reset and invite links instead of mailing them.
SMTP_HOST = os.getenv("SMTP_HOST", "")
SMTP_PORT = int(os.getenv("SMTP_PORT", "587"))
SMTP_USER = os.getenv("SMTP_USER", "")
SMTP_PASSWORD = os.getenv("SMTP_PASSWORD", "")
SMTP_FROM = os.getenv("SMTP_FROM", "") or SMTP_USER
SMTP_STARTTLS = os.getenv("SMTP_STARTTLS", "true").lower() not in ("0", "false", "no")
