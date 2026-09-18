"""Speak Nepali in a chosen agent's voice: Piper for the words, OpenVoice for the timbre.

Piper's ne_NP model pronounces Nepali correctly but sounds like a stranger.
OpenVoice's tone colour converter reshapes a recording's timbre to match a
reference speaker without touching what was said. Chained, you get correct
Nepali in the agent's voice:

    text -> Piper ne_NP -> correct Nepali, generic voice
                        -> OpenVoice  -> correct Nepali, agent's voice

Both are MIT licensed, so unlike XTTS this can ship in a paid product.

Runs under WSL (piper-tts and OpenVoice are Linux-side here):

    ~/tts-venv/bin/python tools/nepali_voice.py --voice ashika
    ~/tts-venv/bin/python tools/nepali_voice.py --voice ashika "नमस्कार, म आशिका बोल्दै छु।"
    ~/tts-venv/bin/python tools/nepali_voice.py --voice ashika --no-convert   # Piper only

What this does and does not fix: the conversion moves timbre, not rhythm, so
the result carries Piper's cadence in the agent's voice. It is a large step up
from a Hindi accent and a step short of a real recording of the person. The
honest ceiling is a Piper fine-tune on 20+ minutes of that person's clean
speech, which gives their words, voice and rhythm together.

Speaker embeddings are cached next to the reference audio, because extracting
one costs far more than the conversion itself and never changes for a given
reference.
"""
from __future__ import annotations

import argparse
import sys
import time
import wave
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from tools import voices

PIPER_NE = REPO / "media" / "voices" / "ne_NP-google-medium.onnx"
OPENVOICE = Path.home() / "OpenVoice"
CONVERTER = OPENVOICE / "checkpoints_v2" / "converter"

DEMO_LINES = [
    "नमस्कार, म {AGENT} बोल्दै छु।",
    "तपाईंलाई कसरी सहयोग गर्न सक्छु?",
    "तपाईंको नाम के हो?",
    "यसको मूल्य पैँतालीस हजार रुपैयाँ हो।",
    "धन्यवाद, तपाईंको दिन शुभ रहोस्।",
]


def piper_say(voice_model, text: str, out: Path) -> float:
    """Synthesise one line with Piper. Returns audio seconds."""
    out.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(out), "wb") as w:
        voice_model.synthesize_wav(text, w)
    with wave.open(str(out)) as w:
        return w.getnframes() / w.getframerate()


def split_utterances(src: Path, out_dir: Path, gap_s: float = 0.35) -> list[str]:
    """Cut a clip into single utterances on its pauses.

    OpenVoice ships se_extractor.get_se for this, but it reaches for
    silero-vad over torch.hub and whisper-medium, which is a large download
    and a hang waiting to happen on a slow link. Splitting on level is the
    same job, needs nothing, and is the code already proven against these
    very recordings.
    """
    import numpy as np

    with wave.open(str(src)) as w:
        sr = w.getframerate()
        a = np.frombuffer(w.readframes(w.getnframes()), dtype="<i2")

    block = int(sr * 0.02)
    usable = len(a) // block * block
    rms = np.sqrt((a[:usable].astype("float64").reshape(-1, block) ** 2).mean(axis=1))
    floor = float(np.percentile(rms, 10))
    loud = rms >= max(floor * 4.0, 60.0)

    gap = int(gap_s / 0.02)
    spans, start, quiet = [], None, 0
    for i, is_loud in enumerate(loud):
        if is_loud:
            if start is None:
                start = i
            quiet = 0
        elif start is not None:
            quiet += 1
            if quiet >= gap:
                spans.append((start, i - quiet + 1))
                start, quiet = None, 0
    if start is not None:
        spans.append((start, len(loud)))

    pad = int(0.08 * sr)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for n, (s, e) in enumerate(spans):
        if (e - s) * 0.02 < 0.7:          # too short to describe a voice
            continue
        chunk = a[max(0, s * block - pad):min(len(a), e * block + pad)]
        p = out_dir / f"seg{n:03d}.wav"
        with wave.open(str(p), "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(sr)
            w.writeframes(chunk.tobytes())
        paths.append(str(p))
    return paths


def load_se(converter, audio: Path, cache: Path):
    """Speaker embedding for a reference clip, cached — extraction is the slow part."""
    import torch
    if cache.exists():
        return torch.load(cache, map_location="cpu")
    segs = split_utterances(audio, cache.parent / f"{cache.stem}_segs")
    if not segs:
        sys.exit(f"found no usable speech in {audio}")
    se = converter.extract_se(segs)
    cache.parent.mkdir(parents=True, exist_ok=True)
    torch.save(se, cache)
    print(f"  embedded {audio.name} from {len(segs)} segments")
    return se


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("text", nargs="*", help="what to say (default: a demo set)")
    ap.add_argument("--voice", default="ashika",
                    help=f"which agent ({', '.join(sorted(voices.VOICES))})")
    ap.add_argument("--ref", type=Path, default=None,
                    help="reference wav (default: this voice's ref in media/voices/refs)")
    ap.add_argument("--out-dir", type=Path, default=REPO / "samples_nepali")
    ap.add_argument("--no-convert", action="store_true",
                    help="stop after Piper, to hear the base voice alone")
    args = ap.parse_args()

    voice = voices.get(args.voice)
    ref = args.ref or REPO / "media" / "voices" / "refs" / f"{voice.key}_ref.wav"
    lines = ([" ".join(args.text)] if args.text
             else [ln.replace("{AGENT}", voice.name_ne) for ln in DEMO_LINES])

    if not PIPER_NE.exists():
        sys.exit(f"no Piper Nepali model at {PIPER_NE}")

    from piper import PiperVoice
    print(f"voice     : {voice.name_en} ({voice.name_ne})")
    print(f"piper     : {PIPER_NE.name}")
    piper = PiperVoice.load(str(PIPER_NE))

    converter = target_se = source_se = None
    if not args.no_convert:
        if not ref.exists():
            sys.exit(f"no reference audio at {ref}")
        if not (CONVERTER / "checkpoint.pth").exists():
            sys.exit(f"no OpenVoice converter at {CONVERTER}")
        import torch
        from openvoice.api import ToneColorConverter
        print(f"reference : {ref.name}")
        converter = ToneColorConverter(str(CONVERTER / "config.json"), device="cpu")
        converter.load_ckpt(str(CONVERTER / "checkpoint.pth"))

        # The source embedding must describe Piper's voice, so take it from a
        # Piper sample rather than any bundled base speaker.
        probe = args.out_dir / "_probe_piper.wav"
        piper_say(piper, "नमस्कार, तपाईंलाई कसरी सहयोग गर्न सक्छु?", probe)
        t0 = time.time()
        source_se = load_se(converter, probe, args.out_dir / "_piper_se.pth")
        target_se = load_se(converter, ref, ref.with_suffix(".se.pth"))
        print(f"embeddings ready in {time.time() - t0:.0f}s")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    print()
    tot_audio = tot_gen = 0.0
    for i, line in enumerate(lines, 1):
        base = args.out_dir / f"{i:03d}_piper.wav"
        t0 = time.time()
        dur = piper_say(piper, line, base)
        if converter is None:
            gen = time.time() - t0
            final = base
        else:
            final = args.out_dir / f"{i:03d}_{voice.key}.wav"
            converter.convert(audio_src_path=str(base), src_se=source_se,
                              tgt_se=target_se, output_path=str(final))
            gen = time.time() - t0
        tot_audio += dur
        tot_gen += gen
        print(f"  {i:03d}  {dur:4.1f}s audio in {gen:5.2f}s "
              f"({gen / dur:.2f}x realtime)  {line[:38]}")

    print(f"\nwrote to {args.out_dir}")
    print(f"total {tot_audio:.1f}s audio in {tot_gen:.1f}s "
          f"({tot_gen / tot_audio:.2f}x realtime)")
    if tot_gen / tot_audio < 1.0:
        print("Faster than realtime — this can synthesise during a live call.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
