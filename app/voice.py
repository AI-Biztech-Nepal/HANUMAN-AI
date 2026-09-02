"""
Voice for the web app — mic audio in, spoken reply out.

Wraps the media pipeline (faster-whisper STT, Piper TTS) so the FastAPI app
can run a real voice conversation in the browser, using exactly the same
transcription and synthesis the telephony bridges use. One agent, one voice,
whichever way the caller reaches it.

The pipeline is imported lazily and never at module scope. Piper is
Linux-only (see requirements.txt) and the Nepali model lives on the host that
runs the bridge, so on a machine without them the web app must still boot and
serve the portal — the voice endpoint reports itself unavailable instead of
taking the whole server down.

Everything here is blocking (model inference and subprocess work). Call it
from async code via asyncio.to_thread, never inline.
"""
from __future__ import annotations

import logging
import os
import tempfile

log = logging.getLogger(__name__)

_bridge = None
_load_error = ""


def _get_bridge():
    """Import media.voice_bridge once. Returns None if unavailable."""
    global _bridge, _load_error
    if _bridge is None and not _load_error:
        try:
            import sys
            from pathlib import Path
            root = str(Path(__file__).resolve().parent.parent)
            if root not in sys.path:
                sys.path.insert(0, root)
            from media import voice_bridge
            _bridge = voice_bridge
        except Exception as exc:                      # noqa: BLE001 - reported, not raised
            _load_error = f"{type(exc).__name__}: {exc}"
            log.warning("voice pipeline unavailable: %s", _load_error)
    return _bridge


def status() -> dict:
    """What the browser needs to decide whether to offer a voice call."""
    vb = _get_bridge()
    if vb is None:
        return {"available": False, "reason": _load_error or "pipeline not installed"}
    try:
        import piper  # noqa: F401
    except Exception:                                 # noqa: BLE001
        return {"available": False,
                "reason": "piper (text-to-speech) is not installed on this host"}
    return {
        "available": True,
        "stt_model": os.path.basename(vb.WHISPER_MODEL_NAME.rstrip("/")),
        "voices": {"ne": os.path.basename(vb.PIPER_VOICE_NE),
                   "en": os.path.basename(vb.PIPER_VOICE_EN)},
    }


_warming = False


def warmup() -> None:
    """Load the STT and TTS models ahead of the first turn.

    Cold, the first utterance of a call pays ~15s of model loading on top of
    its own latency, which reads as a broken app. The portal triggers this
    when it checks voice status at sign-in, so the models are usually ready
    by the time anyone presses the mic. Safe to call repeatedly.
    """
    global _warming
    if _warming:
        return
    _warming = True
    vb = _get_bridge()
    if vb is None:
        return
    try:
        vb._get_whisper()
        for v in (vb.PIPER_VOICE_NE, vb.PIPER_VOICE_EN):
            vb._get_voice(v)
    except Exception:                                 # noqa: BLE001
        log.exception("voice warmup failed; models will load on first use")


def transcribe(audio: bytes, language: str | None = "ne", suffix: str = ".webm") -> str:
    """Browser audio blob → text.

    faster-whisper decodes via PyAV, so the browser's webm/opus needs no
    conversion step here — it is handed over as-is.
    """
    vb = _get_bridge()
    if vb is None:
        raise RuntimeError(_load_error or "voice pipeline unavailable")
    path = ""
    try:
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as f:
            f.write(audio)
            path = f.name
        return vb.transcribe_wav(path, language=language)
    finally:
        if path:
            try:
                os.unlink(path)
            except OSError:
                pass


def speak(text: str, language: str | None = None) -> bytes:
    """Text → WAV bytes at full quality (no telephony downsampling)."""
    vb = _get_bridge()
    if vb is None:
        raise RuntimeError(_load_error or "voice pipeline unavailable")
    voice = (vb.PIPER_VOICE_NE if language == "ne"
             else vb.PIPER_VOICE_EN if language == "en"
             else vb.voice_for_text(text))
    path = ""
    try:
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            path = f.name
        vb.synthesize(text, path, voice=voice, telephony=False)
        with open(path, "rb") as f:
            return f.read()
    finally:
        if path:
            try:
                os.unlink(path)
            except OSError:
                pass
