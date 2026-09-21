"""Re-timbre Piper's speech to a particular agent's voice.

Piper's ne_NP model says Nepali correctly but in a stranger's voice. OpenVoice's
tone colour converter reshapes the timbre to match a reference recording of the
agent, leaving the words and the pronunciation untouched. Both are MIT licensed,
so this can ship in a paid product.

    text -> Piper ne_NP -> correct Nepali, generic voice
                        -> here      -> correct Nepali, the agent's voice

This is an enhancement, never a dependency. If OpenVoice is not installed, the
reference audio is missing, or a conversion fails midway, `convert` hands back
the Piper audio it was given and the call carries on in the generic voice — a
plain-sounding agent is a small problem, a silent one is a lost customer.

Everything here is blocking. Call it from async code via asyncio.to_thread.
"""
from __future__ import annotations

import io
import logging
import os
import sys
import tempfile
import wave
from pathlib import Path

log = logging.getLogger(__name__)

REPO = Path(__file__).resolve().parent.parent
REFS = REPO / "media" / "voices" / "refs"

# Where the OpenVoice checkout and its v2 converter checkpoints live.
OPENVOICE_DIR = Path(os.getenv("OPENVOICE_DIR", str(Path.home() / "OpenVoice")))
CONVERTER_DIR = Path(os.getenv(
    "OPENVOICE_CONVERTER", str(OPENVOICE_DIR / "checkpoints_v2" / "converter")))

_converter = None
_load_error = ""
_embeddings: dict[str, object] = {}
_source_se = None


def _voice_registry():
    """The agent voice table from tools/voices.py, or None if unavailable."""
    try:
        if str(REPO) not in sys.path:
            sys.path.insert(0, str(REPO))
        from tools import voices
        return voices
    except Exception as exc:                          # noqa: BLE001
        log.debug("voice registry unavailable: %s", exc)
        return None


def voice_key_for(agent_name: str) -> str | None:
    """Map a tenant's agent_name to a voice key, tolerating how it was spelled.

    Tenants type the name by hand, so "Aashika" has to find the same voice as
    "Ashika" — the registry keeps the spellings.
    """
    reg = _voice_registry()
    if reg is None or not agent_name:
        return None
    v = reg.resolve(agent_name)
    return v.key if v else None


def model_for(voice_key: str) -> str | None:
    """This voice's own fine-tuned Piper model, or None if it isn't trained yet.

    A trained model makes the conversion pass in this module unnecessary: it
    already speaks in the agent's voice, and running it through the tone
    converter would only add artifacts on top of the real thing.
    """
    reg = _voice_registry()
    if reg is None or not voice_key:
        return None
    v = reg.resolve(voice_key)
    return str(v.model) if v and v.trained else None


def warmup(voice_key: str = "") -> None:
    """Load the converter, and one voice's embedding, ahead of the first call.

    Cold, the converter alone takes over a minute to load — on top of the
    conversion itself, which is not fast either. Paying that on the first
    spoken reply reads as a hung call, so the portal triggers this when it
    checks voice status at sign-in. Safe to call repeatedly; never raises.
    """
    conv = _get_converter()
    if conv is None or not voice_key:
        return
    try:
        _target_se(conv, voice_key)
    except Exception:                                 # noqa: BLE001
        log.exception("could not preload the %r embedding", voice_key)


def _get_converter():
    """Load the tone colour converter once. None if it cannot be had."""
    global _converter, _load_error
    if _converter is not None or _load_error:
        return _converter
    try:
        if OPENVOICE_DIR.exists() and str(OPENVOICE_DIR) not in sys.path:
            sys.path.insert(0, str(OPENVOICE_DIR))
        ckpt = CONVERTER_DIR / "checkpoint.pth"
        if not ckpt.exists():
            raise FileNotFoundError(f"no OpenVoice converter at {CONVERTER_DIR}")
        from openvoice.api import ToneColorConverter
        conv = ToneColorConverter(str(CONVERTER_DIR / "config.json"), device="cpu")
        conv.load_ckpt(str(ckpt))
        # Turn off OpenVoice's WavMark pass. It is where essentially all the
        # conversion time went: the same utterance takes ~60s with it and ~2s
        # without, because the encoder runs a neural model over every one-second
        # chunk on CPU. That is the difference between a usable call and a
        # caller hanging up.
        #
        # What it costs us is an inaudible provenance mark on the audio. The
        # platform's own rule — the agent says it is an AI when asked — is the
        # disclosure that matters here, and telephony at 8kHz would not carry
        # the watermark anyway.
        #
        # The library's own enable_watermark=False is not usable: it reads the
        # flag with kwargs.get and then forwards it to a parent __init__ that
        # rejects it, so passing it raises TypeError. Clearing the model after
        # construction is the same thing without the crash.
        conv.watermark_model = None
        _converter = conv
    except Exception as exc:                          # noqa: BLE001 - reported, not raised
        _load_error = f"{type(exc).__name__}: {exc}"
        log.warning("tone conversion unavailable: %s", _load_error)
    return _converter


def _embed(conv, audio: Path, cache: Path):
    """Speaker embedding for a reference clip, cached to disk and in memory."""
    import torch
    if cache.exists():
        return torch.load(cache, map_location="cpu")
    if str(REPO) not in sys.path:
        sys.path.insert(0, str(REPO))
    from tools.nepali_voice import split_utterances
    segs = split_utterances(audio, cache.parent / f"{cache.stem}_segs")
    if not segs:
        raise RuntimeError(f"no usable speech in {audio}")
    se = conv.extract_se(segs)
    torch.save(se, cache)
    return se


def _target_se(conv, key: str):
    if key not in _embeddings:
        ref = REFS / f"{key}_ref.wav"
        if not ref.exists():
            raise FileNotFoundError(f"no reference audio for {key!r} at {ref}")
        _embeddings[key] = _embed(conv, ref, ref.with_suffix(".se.pth"))
    return _embeddings[key]


def _piper_se(conv, sample_wav: bytes):
    """Embedding describing Piper itself — the voice we convert away from.

    OpenVoice's bundled base-speaker embeddings describe its own models, not
    Piper, and converting from the wrong source degrades the result badly. So
    take it from real Piper output the first time we see some.
    """
    global _source_se
    if _source_se is None:
        cache = REFS / "_piper_source.se.pth"
        if cache.exists():
            import torch
            _source_se = torch.load(cache, map_location="cpu")
        else:
            REFS.mkdir(parents=True, exist_ok=True)
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                f.write(sample_wav)
                probe = Path(f.name)
            try:
                _source_se = _embed(conv, probe, cache)
            finally:
                probe.unlink(missing_ok=True)
    return _source_se


def _polish(wav_bytes: bytes, hp_hz: float = 90.0, target_dbfs: float = -1.5) -> bytes:
    """Clean up converted speech: drop the rumble, bring the level back up.

    Conversion leaves two measurable marks. It comes out several dB quieter
    than the Piper audio that went in, and it carries far more energy below
    300Hz than the reference speaker actually has — which is heard as
    muddiness, because that band masks the 1-4kHz range that carries
    intelligibility. A high-pass and a normalise fix both, and neither
    touches the timbre the conversion just established.

    Returns the input untouched if anything goes wrong; polish is never worth
    losing the audio for.
    """
    try:
        import numpy as np
        with wave.open(io.BytesIO(wav_bytes)) as w:
            sr, n, ch, sw = (w.getframerate(), w.getnframes(),
                             w.getnchannels(), w.getsampwidth())
            a = np.frombuffer(w.readframes(n), dtype="<i2").astype(np.float64)
        if ch != 1 or sw != 2 or len(a) == 0:
            return wav_bytes

        # Second-order Butterworth high-pass (RBJ cookbook coefficients),
        # applied forwards then backwards so it adds no phase smear.
        w0 = 2.0 * np.pi * hp_hz / sr
        cw, alpha = np.cos(w0), np.sin(w0) / (2.0 * 0.7071)
        b = np.array([(1 + cw) / 2, -(1 + cw), (1 + cw) / 2])
        aa = np.array([1 + alpha, -2 * cw, 1 - alpha])
        b, aa = b / aa[0], aa / aa[0]

        def biquad(x):
            y = np.empty_like(x)
            x1 = x2 = y1 = y2 = 0.0
            for i, s in enumerate(x):
                out = b[0] * s + b[1] * x1 + b[2] * x2 - aa[1] * y1 - aa[2] * y2
                x2, x1, y2, y1 = x1, s, y1, out
                y[i] = out
            return y

        try:                                   # scipy is present via librosa
            from scipy.signal import filtfilt
            a = filtfilt(b, aa, a)
        except Exception:                      # noqa: BLE001
            a = biquad(a)[::-1]
            a = biquad(a)[::-1]

        # Tilt the balance back toward the reference speaker's.
        #
        # Conversion leaves the result bass-heavy: measured against Ashika's
        # own recordings it puts about twice the energy below 300Hz and loses
        # a quarter of the 1-4kHz presence band. That band is where
        # consonants live, so losing it reads as mush rather than as warmth.
        # A gentle shelf either side moves the balance back without touching
        # pitch or timbre. Kept mild on purpose — overdo it and speech turns
        # thin and sibilant, which is worse than muddy on a phone.
        spec = np.fft.rfft(a)
        freq = np.fft.rfftfreq(len(a), 1.0 / sr)
        gain_db = np.interp(freq,
                            [0, 250, 600, 1200, 2500, 5000, 7000, sr / 2],
                            [-3.5, -3.0, -0.5, 1.0, 3.5, 3.5, 1.0, 0.0])
        a = np.fft.irfft(spec * (10 ** (gain_db / 20.0)), n=len(a))

        peak = float(np.abs(a).max()) or 1.0
        a = a * (10 ** (target_dbfs / 20) * 32768.0 / peak)
        a = np.clip(a, -32768, 32767).astype("<i2")

        buf = io.BytesIO()
        with wave.open(buf, "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(sr)
            w.writeframes(a.tobytes())
        return buf.getvalue()
    except Exception:                          # noqa: BLE001
        log.exception("polish failed; using the unpolished audio")
        return wav_bytes


def available() -> bool:
    return _get_converter() is not None


def status() -> dict:
    conv = _get_converter()
    reg = _voice_registry()
    return {
        "available": conv is not None,
        "reason": _load_error or "",
        "voices": sorted(reg.VOICES) if reg else [],
        "references": sorted(p.stem.replace("_ref", "")
                             for p in REFS.glob("*_ref.wav")) if REFS.exists() else [],
    }


def convert(wav_bytes: bytes, voice_key: str) -> bytes:
    """Re-timbre Piper WAV bytes to `voice_key`. Returns the input unchanged
    if conversion is unavailable or fails — never raises into call handling."""
    conv = _get_converter()
    if conv is None or not voice_key:
        return wav_bytes
    src = dst = None
    try:
        tgt = _target_se(conv, voice_key)
        source = _piper_se(conv, wav_bytes)
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            f.write(wav_bytes)
            src = f.name
        dst = src.replace(".wav", ".out.wav")
        conv.convert(audio_src_path=src, src_se=source, tgt_se=tgt, output_path=dst)
        with open(dst, "rb") as f:
            return _polish(f.read())
    except Exception:                                 # noqa: BLE001
        log.exception("tone conversion failed for %r; using the base voice", voice_key)
        return wav_bytes
    finally:
        for p in (src, dst):
            if p:
                try:
                    os.unlink(p)
                except OSError:
                    pass
