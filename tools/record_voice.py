"""Record your own voice into a Piper training dataset — entirely offline.

Reads prompt sentences from a text file, records you speaking each one, and
writes the LJSpeech layout Piper's training expects:

    voice_dataset/
      wavs/0001.wav 0002.wav ...
      metadata.csv          # id|transcript|transcript

Usage:
    python tools/record_voice.py                       # start or resume
    python tools/record_voice.py --prompts my.txt      # your own sentences
    python tools/record_voice.py --status              # how much do I have?

It resumes: already-recorded prompts are skipped, so you can stop any time
and pick up later. After each take you can keep it, redo it, or skip the
sentence — a dataset with one bad take in it trains a voice with that bad
take in it.

Recording quality matters more than quantity, in this order:
  - one quiet room, one microphone, one distance from it, every session
  - no background music, fan, or traffic
  - speak the sentence as written; if you fluff it, redo rather than improvise
"""
from __future__ import annotations

import argparse
import csv
import sys
import wave
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from tools import voices

# Piper trains at 22.05kHz; recording at the training rate avoids a resample.
SAMPLE_RATE = 22050
SILENCE_HOLD_S = 0.9          # quiet for this long ends a take
RMS_SPEECH = 450              # out of 32768, matching the call pipeline's VAD
MAX_TAKE_S = 20.0
LEAD_IN_S = 0.25              # keep a little room before the first word

DEFAULT_PROMPTS = REPO / "tools" / "prompts_ne.txt"

# Set from --voice in main(); each agent records into its own dataset.
DATASET = WAVS = METADATA = None


def use_voice(voice) -> None:
    global DATASET, WAVS, METADATA
    DATASET, WAVS, METADATA = voice.dataset, voice.wavs, voice.metadata


def load_done() -> dict[str, str]:
    """Recorded id -> transcript, so a rerun resumes instead of restarting."""
    if not METADATA.exists():
        return {}
    done = {}
    with METADATA.open(encoding="utf-8", newline="") as f:
        for row in csv.reader(f, delimiter="|"):
            if row:
                done[row[0]] = row[1] if len(row) > 1 else ""
    return done


def wav_seconds(path: Path) -> float:
    with wave.open(str(path)) as w:
        return w.getnframes() / w.getframerate()


def total_seconds() -> float:
    return sum(wav_seconds(p) for p in WAVS.glob("*.wav")) if WAVS.exists() else 0.0


def report(prompts: list[str]) -> None:
    done = load_done()
    secs = total_seconds()
    print(f"recorded   : {len(done)} of {len(prompts)} prompts")
    print(f"audio       : {secs / 60:.1f} minutes")
    print(f"dataset     : {DATASET}")
    print()
    if secs < 20 * 60:
        print(f"Piper fine-tuning wants 20-60 minutes. You need roughly "
              f"{max(0, (20 * 60 - secs)) / 60:.0f} more minutes for a usable "
              f"voice, and more is better.")
    else:
        print("That is enough to attempt a fine-tune.")


def record_take(sd, np):
    """Capture one utterance, stopping when you go quiet."""
    block = int(SAMPLE_RATE * 0.03)
    silence_blocks = int(SILENCE_HOLD_S / 0.03)
    max_blocks = int(MAX_TAKE_S / 0.03)
    lead_blocks = int(LEAD_IN_S / 0.03)

    frames, quiet, spoke = [], 0, False
    with sd.InputStream(samplerate=SAMPLE_RATE, channels=1,
                        dtype="int16", blocksize=block) as stream:
        while len(frames) < max_blocks:
            data, _ = stream.read(block)
            frames.append(data.copy())
            rms = float(np.sqrt(np.mean(data.astype("float64") ** 2)))
            if rms >= RMS_SPEECH:
                spoke, quiet = True, 0
            elif spoke:
                quiet += 1
                if quiet >= silence_blocks:
                    break
    if not spoke:
        return None
    # Trim the trailing silence back to a short tail, keep a little lead-in.
    keep = max(0, len(frames) - silence_blocks + lead_blocks)
    return np.concatenate(frames[:keep])


def save_wav(path: Path, audio) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SAMPLE_RATE)
        w.writeframes(audio.tobytes())


def append_metadata(clip_id: str, text: str) -> None:
    METADATA.parent.mkdir(parents=True, exist_ok=True)
    with METADATA.open("a", encoding="utf-8", newline="") as f:
        # LJSpeech format: id|raw|normalised. Piper reads the third column.
        csv.writer(f, delimiter="|", quoting=csv.QUOTE_NONE,
                   escapechar="\\").writerow([clip_id, text, text])


def peak_dbfs(audio, np) -> float:
    peak = int(np.abs(audio).max()) or 1
    return 20.0 * np.log10(peak / 32768.0)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--voice", default=voices.DEFAULT_VOICE,
                    help=f"which agent's voice ({', '.join(sorted(voices.VOICES))})")
    ap.add_argument("--prompts", type=Path, default=DEFAULT_PROMPTS)
    ap.add_argument("--status", action="store_true", help="show progress and exit")
    args = ap.parse_args()

    voice = voices.get(args.voice)
    use_voice(voice)
    prompts = voices.load_prompts(args.prompts, voice)
    print(f"voice: {voice.name_en} ({voice.name_ne}, {voice.gender}) "
          f"-> {voice.dataset.name}\n")
    if args.status:
        report(prompts)
        return 0

    try:
        import numpy as np
        import sounddevice as sd
    except ImportError as exc:
        sys.exit(f"needs sounddevice and numpy: {exc}")

    done = load_done()
    todo = [(f"{i + 1:04d}", t) for i, t in enumerate(prompts)
            if f"{i + 1:04d}" not in done]
    if not todo:
        print("Every prompt is recorded.")
        report(prompts)
        return 0

    print(f"{len(done)} done, {len(todo)} to go — {total_seconds() / 60:.1f} min so far.")
    print("Enter = record, then k(eep) / r(edo) / s(kip) / q(uit).\n")

    for clip_id, text in todo:
        while True:
            print(f"[{clip_id}] {text}")
            try:
                cmd = input("      Enter to record (or s/q): ").strip().lower()
            except (EOFError, KeyboardInterrupt):
                print("\nstopped.")
                report(prompts)
                return 0
            if cmd == "q":
                report(prompts)
                return 0
            if cmd == "s":
                break

            print("      listening… speak now")
            audio = record_take(sd, np)
            if audio is None or len(audio) < SAMPLE_RATE * 0.3:
                print("      heard nothing — try again\n")
                continue

            secs = len(audio) / SAMPLE_RATE
            peak = peak_dbfs(audio, np)
            warn = ""
            if peak > -1.0:
                warn = "  ⚠ too loud, likely clipped — move back from the mic"
            elif peak < -30.0:
                warn = "  ⚠ very quiet — move closer or raise the input gain"
            print(f"      {secs:.1f}s, peak {peak:.0f} dBFS{warn}")

            choice = input("      k=keep  r=redo  s=skip  q=quit: ").strip().lower()
            if choice == "q":
                report(prompts)
                return 0
            if choice == "r":
                print()
                continue
            if choice == "s":
                break
            save_wav(WAVS / f"{clip_id}.wav", audio)
            append_metadata(clip_id, text)
            print(f"      saved — {total_seconds() / 60:.1f} min total\n")
            break

    print("\nAll prompts done.")
    report(prompts)
    return 0


if __name__ == "__main__":
    sys.exit(main())
