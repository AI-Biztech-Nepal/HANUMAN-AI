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

import io
import logging
import os
import re
import struct
import tempfile
import wave

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


def warmup(agent_voice: str = "") -> None:
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
    # The tone converter is now the slowest thing on the first spoken turn —
    # slower than Whisper and Piper together — so it belongs here too. It has
    # its own try/except inside, and an agent whose voice is not converted
    # still speaks, so a failure here must not stop the models above counting
    # as warm.
    try:
        from . import tone
        tone.warmup(agent_voice)
    except Exception:                                 # noqa: BLE001
        log.exception("tone warmup failed; conversion will load on first use")


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


_recorded: dict[str, dict[str, str]] = {}


def _spoken_key(s: str) -> str:
    """Compare lines by what is actually said, not how it was typed.

    Punctuation is not spoken, and a greeting retyped in the portal rarely
    matches the prompt sheet character for character — a different comma is
    enough to miss. Devanagari letters and digits alone are the comparison
    that survives being edited by hand.

    The danda and double danda are the catch: they are sentence-ending
    punctuation, but they live inside the Devanagari block, so a filter that
    keeps "Devanagari" keeps them and a line retyped without the danda stops
    matching. Drop them explicitly.
    """
    return re.sub(r"[^ऀ-ॣ०-ॿ0-9a-zA-Z]", "", s or "")


def _recorded_lines(voice_key: str) -> dict[str, str]:
    """This agent's recorded lines: spoken-key -> wav path."""
    if voice_key in _recorded:
        return _recorded[voice_key]
    index: dict[str, str] = {}
    try:
        import sys
        from pathlib import Path
        root = Path(__file__).resolve().parent.parent
        if str(root) not in sys.path:
            sys.path.insert(0, str(root))
        from tools import voices
        v = voices.resolve(voice_key)
        if v is not None and v.metadata.exists():
            for line in v.metadata.read_text(encoding="utf-8").splitlines():
                parts = line.split("|")
                if len(parts) >= 2:
                    wav = v.wavs / f"{parts[0]}.wav"
                    if wav.exists():
                        index[_spoken_key(parts[1])] = str(wav)
    except Exception:                                 # noqa: BLE001
        log.exception("could not index recorded lines for %r", voice_key)
    _recorded[voice_key] = index
    log.info("recorded lines available for %s: %d", voice_key, len(index))
    return index


def recorded_line(agent_voice: str, text: str) -> bytes | None:
    """The agent's own recording of this exact line, if we have one.

    A greeting is the same sentence on every call, and we have the person
    saying it. Synthesising it — then re-timbring the synthesis toward a
    recording we already hold — is a worse version of playing the recording.
    So when the line matches, play her.

    Returns WAV bytes trimmed of room tone and levelled to match the rest of
    the call, or None when this line was never recorded.
    """
    if not agent_voice or not text:
        return None
    path = _recorded_lines(agent_voice).get(_spoken_key(text))
    if not path:
        return None
    try:
        return _clean_clip(path)
    except Exception:                                 # noqa: BLE001
        log.exception("could not read recorded line %s; synthesising instead", path)
        return None


def _clean_clip(path: str, target_peak: float = 0.89,
                silence: int = 300, pad_s: float = 0.06) -> bytes:
    """Recorded clip -> WAV bytes fit to play mid-call.

    Dataset takes carry the silence the recorder left around each sentence
    and whatever level the room gave that day. Played straight after
    synthesised speech, both are audible as a seam.
    """
    with wave.open(path) as w:
        n, sr, ch, sw = (w.getnframes(), w.getframerate(),
                         w.getnchannels(), w.getsampwidth())
        a = list(struct.unpack("<%dh" % n, w.readframes(n)))
    if ch != 1 or sw != 2 or not a:
        with open(path, "rb") as f:                   # hand it back untouched
            return f.read()

    first, last = 0, len(a) - 1
    while first < len(a) and abs(a[first]) < silence:
        first += 1
    while last > first and abs(a[last]) < silence:
        last -= 1
    pad = int(sr * pad_s)
    a = a[max(0, first - pad):min(len(a), last + pad)] or a

    peak = max(abs(min(a)), abs(max(a))) or 1
    gain = (target_peak * 32767) / peak
    a = [max(-32768, min(32767, int(x * gain))) for x in a]

    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(struct.pack("<%dh" % len(a), *a))
    return buf.getvalue()


def speak(text: str, language: str | None = None,
          agent_voice: str | None = None) -> bytes:
    """Text → WAV bytes at full quality (no telephony downsampling).

    `agent_voice` is a voice key from tools/voices.py ("ashika", "sagar").
    Given one, Piper's speech is re-timbred to that agent before it goes out;
    the words and the Nepali pronunciation are Piper's either way. If tone
    conversion is unavailable the caller still gets audio, in the base voice.
    """
    # Her own recording of this line beats anything we can synthesise of it.
    if agent_voice:
        clip = recorded_line(agent_voice, text)
        if clip is not None:
            log.info("speaking a recorded line in %s's own voice", agent_voice)
            return clip

    vb = _get_bridge()
    if vb is None:
        raise RuntimeError(_load_error or "voice pipeline unavailable")
    voice = (vb.PIPER_VOICE_NE if language == "ne"
             else vb.PIPER_VOICE_EN if language == "en"
             else vb.voice_for_text(text))
    # A voice we trained ourselves replaces the Nepali base model outright:
    # it speaks the agent's own voice and rhythm, so the tone conversion below
    # is not just unnecessary but harmful — it would re-timbre a voice that is
    # already correct. The datasets are Nepali, so an English line still goes
    # through the base model and the converter.
    convert_after = bool(agent_voice)
    if agent_voice and voice == vb.PIPER_VOICE_NE:
        from . import tone
        trained = tone.model_for(agent_voice)
        if trained:
            voice, convert_after = trained, False
    path = ""
    try:
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            path = f.name
        vb.synthesize(text, path, voice=voice, telephony=False)
        with open(path, "rb") as f:
            audio = f.read()
    finally:
        if path:
            try:
                os.unlink(path)
            except OSError:
                pass
    if convert_after:
        from . import tone
        audio = tone.convert(audio, agent_voice)
    return audio
