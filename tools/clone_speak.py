"""Speak text in a cloned voice, from a short reference recording.

DO NOT SHIP THIS. XTTS v2 is released under the Coqui Public Model License,
which forbids commercial use — and hanuman.ai charges for calls. This script
is kept only as the record of an experiment: it is how we established that a
short reference carries enough of a speaker's timbre to clone at all. The
production path is tools/nepali_voice.py (Piper + OpenVoice, both MIT).

It is also the wrong tool for Nepali regardless of licence: XTTS speaks 17
languages and Nepali is not among them, so this falls back to Hindi and
pronounces Nepali with Hindi habits.

Runs XTTS v2 under WSL (the Windows venv is Python 3.14, which coqui-tts
does not support). Give it Nepali text and a reference wav of the voice you
want, and it writes a wav of that voice saying it.

    # inside WSL
    ~/tts-venv/bin/python tools/clone_speak.py "नमस्कार, म आशिका बोल्दै छु।"
    ~/tts-venv/bin/python tools/clone_speak.py --text-file lines.txt --out-dir samples/

A caution about language: XTTS v2 speaks 17 languages and Nepali is not one
of them. Hindi is, and shares Devanagari and much of the phonology, so Hindi
mode on Nepali text is intelligible but carries Hindi pronunciation habits.
That is a starting point for judging the VOICE, not a verdict on how the
agent will finally pronounce Nepali. If the accent is wrong but the timbre is
right, the fix is to keep Piper for Nepali phonetics and transfer only the
voice identity onto it.

Timing matters as much as quality here: this is CPU-only, and a phone caller
will not wait. The script prints how long each line took and the ratio of
generation time to audio length, so you can see whether it could ever run
live or whether lines must be pre-rendered and cached.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent

DEFAULT_REF = REPO / "media" / "voices" / "refs" / "ashika_ref.wav"
MODEL = "tts_models/multilingual/multi-dataset/xtts_v2"

# Nepali is not an XTTS language; Hindi is the closest Devanagari one.
DEFAULT_LANG = "hi"

DEMO_LINES = [
    "नमस्कार, म आशिका बोल्दै छु।",
    "तपाईंलाई कसरी सहयोग गर्न सक्छु?",
    "तपाईंको नाम के हो?",
    "यसको मूल्य पैँतालीस हजार रुपैयाँ हो।",
    "धन्यवाद, तपाईंको दिन शुभ रहोस्।",
]


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("text", nargs="*", help="what to say (default: a demo set)")
    ap.add_argument("--text-file", type=Path, help="one line per utterance")
    ap.add_argument("--ref", type=Path, default=DEFAULT_REF, help="reference voice wav")
    ap.add_argument("--lang", default=DEFAULT_LANG, help=f"XTTS language (default {DEFAULT_LANG})")
    ap.add_argument("--out-dir", type=Path, default=REPO / "samples")
    args = ap.parse_args()

    if not args.ref.exists():
        sys.exit(f"no reference audio at {args.ref}")

    if args.text_file:
        lines = [ln.strip() for ln in args.text_file.read_text(encoding="utf-8").splitlines()]
        lines = [ln for ln in lines if ln and not ln.startswith("#")]
    elif args.text:
        lines = [" ".join(args.text)]
    else:
        lines = DEMO_LINES

    from TTS.api import TTS

    print(f"reference : {args.ref}")
    print(f"language  : {args.lang}")
    print(f"loading   : {MODEL}")
    t0 = time.time()
    tts = TTS(MODEL)          # first run downloads ~1.8GB
    print(f"loaded in {time.time() - t0:.0f}s\n")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    import wave

    total_gen = total_audio = 0.0
    for i, line in enumerate(lines, 1):
        out = args.out_dir / f"{i:03d}.wav"
        t0 = time.time()
        tts.tts_to_file(text=line, speaker_wav=str(args.ref),
                        language=args.lang, file_path=str(out))
        gen = time.time() - t0
        with wave.open(str(out)) as w:
            dur = w.getnframes() / w.getframerate()
        total_gen += gen
        total_audio += dur
        print(f"  {i:03d}  {dur:4.1f}s audio in {gen:5.1f}s  "
              f"({gen / dur:.1f}x realtime)  {line[:40]}")

    print(f"\nwrote {len(lines)} files to {args.out_dir}")
    print(f"total {total_audio:.1f}s audio in {total_gen:.0f}s "
          f"({total_gen / total_audio:.1f}x realtime)")
    if total_gen / total_audio > 1.0:
        print("\nSlower than realtime: this cannot synthesise during a live call "
              "on this hardware. Either pre-render the lines the agent repeats "
              "and cache them, or put the model on a GPU.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
