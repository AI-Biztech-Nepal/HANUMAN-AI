"""
Voice bridge — reference pipeline: audio in → Whisper STT → agent (/ws/chat)
→ Piper TTS → audio out.

Two modes:

  1. Local mic test (needs a machine with mic/speakers, e.g. a dev laptop):
       python media/voice_bridge.py mic --tenant <tenant_id>

  2. Asterisk/FreeSWITCH integration (production, on the VPS):
       Use `transcribe_wav()` and `synthesize()` from your dialplan handler
       (e.g. Asterisk ARI/AGI or FreeSWITCH ESL script). Feed each caller
       utterance WAV in, play the returned WAV back on the channel.

Install:  pip install faster-whisper piper-tts websockets sounddevice soundfile
Piper voices: download .onnx voice files from the piper-voices repository
into ./voices/ (English: en_US-amy-medium; Nepali: check community voices,
or use Google Cloud TTS ne-NP as a paid alternative).
"""
from __future__ import annotations

import asyncio
import audioop
import json
import logging
import os
import subprocess
import sys
import tempfile
import wave
from pathlib import Path

log = logging.getLogger(__name__)

TELEPHONY_SAMPLE_RATE = 8000  # Asterisk's format_wav requires an exact match to the endpoint's ulaw/alaw rate

# Import app.config FIRST: it calls load_dotenv() at module scope, and the
# os.getenv() lookups below must see .env or they silently fall back to the
# stock model no matter what .env says.
sys.path.insert(0, str(Path(__file__).parent.parent))
from app import config as _app_config  # noqa: E402

SERVER_WS = _app_config.AGENT_WS_URL

# Fine-tuned on OpenSLR 54 Nepali speech (154hrs) — far more accurate than base
# "small" on Nepali (WER 26.69% vs stock small, which drops most utterances
# entirely). Falls back to the stock model where the fine-tune isn't deployed.
_NEPALI_FT_MODEL = os.getenv("WHISPER_MODEL_PATH", "/home/mansa/whisper-small-nepali-ct2")
WHISPER_MODEL_NAME = _NEPALI_FT_MODEL if Path(_NEPALI_FT_MODEL).exists() else os.getenv("WHISPER_MODEL", "small")
_VOICES_DIR = Path(__file__).parent / "voices"
PIPER_VOICE_EN = str(_VOICES_DIR / "en_US-amy-medium.onnx")
PIPER_VOICE_NE = str(_VOICES_DIR / "ne_NP-google-medium.onnx")
PIPER_VOICE = PIPER_VOICE_EN  # default/back-compat for callers that don't pick a voice


def voice_for_text(text: str) -> str:
    """Pick the Nepali or English Piper voice by whether the text is
    Devanagari — an English voice model can't pronounce Nepali script."""
    if any("ऀ" <= ch <= "ॿ" for ch in text):
        return PIPER_VOICE_NE
    return PIPER_VOICE_EN

_whisper = None


def _get_whisper():
    global _whisper
    if _whisper is None:
        from faster_whisper import WhisperModel
        _whisper = WhisperModel(WHISPER_MODEL_NAME, device="cpu", compute_type="int8")
    return _whisper


def transcribe_wav(wav_path: str, language: str | None = None) -> str:
    """WAV file → text. language: 'ne', 'en', or None for auto-detect.

    Auto-detect crashes faster-whisper (IndexError in detect_language) on a
    chunk that's silence-only once its own VAD filter strips everything —
    always pass an explicit language on telephony audio to skip that path.
    """
    try:
        segments, _info = _get_whisper().transcribe(
            wav_path, language=language, vad_filter=True,
            condition_on_previous_text=False,
            repetition_penalty=1.2,
            no_repeat_ngram_size=3,
        )
        text = " ".join(s.text.strip() for s in segments).strip()
    except Exception:
        log.exception("transcription failed for %s", wav_path)
        return ""
    if _looks_like_hallucination(text):
        log.warning("discarding likely hallucination for %s: %r", wav_path, text)
        return ""
    return text


def _looks_like_hallucination(text: str) -> bool:
    """Whisper can loop on ambiguous/noisy audio, repeating one word dozens
    of times — treat that as noise rather than real speech."""
    words = text.split()
    if len(words) < 6:
        return False
    most_repeated = max(words.count(w) for w in set(words))
    return most_repeated / len(words) > 0.4


_voices: dict[str, object] = {}


def _get_voice(model_path: str):
    """Load a Piper voice once and keep it. Loading is the expensive part —
    the .onnx files here are 63-77MB, and re-reading one per utterance cost
    ~5s of the turn budget when synthesis was a subprocess call."""
    voice = _voices.get(model_path)
    if voice is None:
        from piper import PiperVoice
        voice = PiperVoice.load(model_path)
        _voices[model_path] = voice
    return voice


def synthesize(text: str, out_wav: str, voice: str = PIPER_VOICE) -> str:
    """Text → WAV file via Piper, resampled to 8kHz. Returns out_wav path."""
    try:
        with wave.open(out_wav, "wb") as w:
            _get_voice(voice).synthesize_wav(text, w)
    except Exception:
        # Fall back to the CLI so a piper-python API change can't take voice
        # output down — slower, but a degraded reply beats silence on a call.
        log.exception("in-process synthesis failed for %r, using piper CLI", voice)
        piper_bin = str(Path(sys.executable).parent / "piper")
        subprocess.run(
            [piper_bin, "--model", voice, "--output_file", out_wav],
            input=text.encode("utf-8"),
            check=True,
            capture_output=True,
        )
    _resample_to_telephony_rate(out_wav)
    return out_wav


def _resample_to_telephony_rate(wav_path: str) -> None:
    # Piper outputs 22050Hz; Asterisk's format_wav rejects anything that
    # doesn't exactly match the endpoint's codec rate (8000Hz for ulaw/alaw).
    with wave.open(wav_path, "rb") as w:
        channels, sampwidth, rate = w.getnchannels(), w.getsampwidth(), w.getframerate()
        pcm = w.readframes(w.getnframes())
    if rate == TELEPHONY_SAMPLE_RATE and channels == 1:
        return
    if channels == 2:
        pcm = audioop.tomono(pcm, sampwidth, 0.5, 0.5)
    pcm, _ = audioop.ratecv(pcm, sampwidth, 1, rate, TELEPHONY_SAMPLE_RATE, None)
    with wave.open(wav_path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(sampwidth)
        w.setframerate(TELEPHONY_SAMPLE_RATE)
        w.writeframes(pcm)


# --------------------------------------------------------------- mic mode

# Mirrors the VAD tuning in asterisk_bridge.py so the mic path and the
# telephony path end an utterance on the same rules.
MIC_SAMPLE_RATE = 16000
RMS_SPEECH_THRESHOLD = 500       # out of 32768 full-scale
SILENCE_TIMEOUT_MS = 700         # quiet for this long ends the utterance
MIN_UTTERANCE_MS = 300           # ignore blips shorter than this
MAX_UTTERANCE_MS = 15_000        # safety valve against a stuck-open mic
MIC_LANGUAGE = os.getenv("MIC_LANGUAGE", "ne")


def _record_until_silence(sd):
    """Capture one utterance, stopping when the speaker goes quiet.

    The old fixed 5-second window made every turn wait out the full window
    even after the caller had finished, and truncated anyone who ran long.
    """
    import numpy as np

    block = int(MIC_SAMPLE_RATE * 0.03)  # 30ms blocks
    silence_blocks = int(SILENCE_TIMEOUT_MS / 30)
    min_blocks = int(MIN_UTTERANCE_MS / 30)
    max_blocks = int(MAX_UTTERANCE_MS / 30)

    frames, quiet_run, spoken = [], 0, 0
    with sd.InputStream(samplerate=MIC_SAMPLE_RATE, channels=1,
                        dtype="int16", blocksize=block) as stream:
        while len(frames) < max_blocks:
            data, _overflowed = stream.read(block)
            frames.append(data.copy())
            rms = float(np.sqrt(np.mean(data.astype("float64") ** 2)))
            if rms >= RMS_SPEECH_THRESHOLD:
                spoken += 1
                quiet_run = 0
            elif spoken >= min_blocks:
                # Only start counting silence once real speech has happened,
                # so leading hesitation doesn't end the turn immediately.
                quiet_run += 1
                if quiet_run >= silence_blocks:
                    break

    return np.concatenate(frames) if frames else np.zeros((0, 1), dtype="int16")


async def mic_session(tenant_id: str | None):
    """Talk to the agent with your mic — full voice loop for local testing."""
    import sounddevice as sd
    import soundfile as sf
    import websockets

    async with websockets.connect(SERVER_WS) as ws:
        # /ws/chat always waits for a first client message before replying
        # (app/main.py ws_chat) — must send the handshake even with no tenant.
        await ws.send(json.dumps({"tenant_id": tenant_id}))
        greeting = await ws.recv()
        print(f"AGENT: {greeting}")
        _speak_local(greeting)

        while True:
            print("… speak now …")
            audio = _record_until_silence(sd)
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                sf.write(f.name, audio, MIC_SAMPLE_RATE)
                text = transcribe_wav(f.name, language=MIC_LANGUAGE)
            if not text:
                print("(heard nothing)")
                continue
            print(f"YOU: {text}")
            await ws.send(text)
            reply = await ws.recv()
            print(f"AGENT: {reply}")
            _speak_local(reply)


def _speak_local(text: str):
    import sounddevice as sd
    import soundfile as sf
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        try:
            synthesize(text, f.name)
            data, sr = sf.read(f.name)
            sd.play(data, sr)
            sd.wait()
        except Exception as e:
            print(f"[TTS unavailable: {e}]")


if __name__ == "__main__":
    if len(sys.argv) >= 2 and sys.argv[1] == "mic":
        tid = sys.argv[3] if len(sys.argv) >= 4 and sys.argv[2] == "--tenant" else None
        asyncio.run(mic_session(tid))
    else:
        print(__doc__)
