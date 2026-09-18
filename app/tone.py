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
            return f.read()
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
