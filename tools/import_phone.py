"""Import phone recordings into the Piper training dataset.

You read a batch of prompts into your phone's voice recorder — one long file,
with a clear pause between each sentence — then this splits that file on the
pauses and files each piece against the prompt it belongs to.

    python tools/import_phone.py recording.m4a              # preview the split
    python tools/import_phone.py recording.m4a --commit     # write it
    python tools/import_phone.py recording.m4a --play 3     # hear segment 3

The whole job is alignment: segment 1 must be the prompt you read first, and
every segment after it must line up too. One missed pause shifts everything
that follows, and a shifted dataset trains a voice on the wrong words. So the
default is a preview that writes nothing — check the count and the durations,
audition anything that looks odd, and only then --commit.

By default a batch continues from wherever the dataset left off. Pass --start
to place it somewhere else.

Recording on the phone:
  - turn OFF noise reduction / "voice enhancement" in the recorder app; the
    automatic gain those apply varies loudness mid-sentence, which is the one
    thing a training set must not contain
  - prop the phone at a fixed distance, a hand's width away — don't hold it
  - pause a full second between sentences, and don't rustle in the gap
  - read straight down the prompt list, in order, without skipping
"""
from __future__ import annotations

import argparse
import csv
import shutil
import subprocess
import sys
import tempfile
import wave
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from tools import voices

SAMPLE_RATE = 22050           # Piper's training rate, same as record_voice.py
MIN_GAP_S = 0.60              # a pause at least this long separates two prompts
MIN_SEG_S = 0.40              # anything shorter is a cough, not a sentence
PAD_S = 0.15                  # keep this much room either side of the speech
BLOCK_S = 0.02

DEFAULT_PROMPTS = REPO / "tools" / "prompts_ne.txt"

# Set from --voice in main(); each agent imports into its own dataset.
DATASET = WAVS = METADATA = None


def use_voice(voice) -> None:
    global DATASET, WAVS, METADATA
    DATASET, WAVS, METADATA = voice.dataset, voice.wavs, voice.metadata


def find_ffmpeg() -> str:
    """ffmpeg from PATH, or from where winget drops it before a shell restart."""
    found = shutil.which("ffmpeg")
    if found:
        return found
    packages = Path.home() / "AppData/Local/Microsoft/WinGet/Packages"
    for exe in packages.glob("Gyan.FFmpeg*/**/bin/ffmpeg.exe"):
        return str(exe)
    sys.exit("ffmpeg not found — install it with: winget install Gyan.FFmpeg")


def load_done() -> dict[str, str]:
    if not METADATA.exists():
        return {}
    with METADATA.open(encoding="utf-8", newline="") as f:
        return {r[0]: (r[1] if len(r) > 1 else "") for r in csv.reader(f, delimiter="|") if r}


def decode(src: Path, ffmpeg: str):
    """Any phone format -> mono 22050Hz int16, via a temp wav."""
    import numpy as np

    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "decoded.wav"
        proc = subprocess.run(
            [ffmpeg, "-hide_banner", "-loglevel", "error", "-y", "-i", str(src),
             "-ac", "1", "-ar", str(SAMPLE_RATE), "-sample_fmt", "s16", str(out)],
            capture_output=True, text=True,
        )
        if proc.returncode != 0 or not out.exists():
            sys.exit(f"ffmpeg could not read {src.name}:\n{proc.stderr.strip()}")
        with wave.open(str(out)) as w:
            return np.frombuffer(w.readframes(w.getnframes()), dtype="<i2")


def block_rms(audio, np):
    block = int(SAMPLE_RATE * BLOCK_S)
    usable = len(audio) // block * block
    frames = audio[:usable].astype("float64").reshape(-1, block)
    return np.sqrt((frames ** 2).mean(axis=1))


def pick_threshold(rms, np) -> float:
    """Sit the speech/silence line above this recording's own noise floor.

    A fixed threshold can't work across phones and rooms, so take the quiet
    tenth of the file as the floor and put the line well above it.
    """
    floor = float(np.percentile(rms, 10))
    speech = float(np.percentile(rms, 90))
    return max(floor * 4.0, floor + (speech - floor) * 0.08, 60.0)


def split(audio, np, min_gap_s: float, threshold: float | None):
    """Cut on the pauses. Returns [(start_sample, end_sample), ...]."""
    rms = block_rms(audio, np)
    thr = threshold if threshold is not None else pick_threshold(rms, np)
    loud = rms >= thr

    block = int(SAMPLE_RATE * BLOCK_S)
    gap_blocks = int(min_gap_s / BLOCK_S)
    pad = int(PAD_S * SAMPLE_RATE)

    segments, start, quiet = [], None, 0
    for i, is_loud in enumerate(loud):
        if is_loud:
            if start is None:
                start = i
            quiet = 0
        elif start is not None:
            quiet += 1
            if quiet >= gap_blocks:
                segments.append((start, i - quiet + 1))
                start, quiet = None, 0
    if start is not None:
        segments.append((start, len(loud)))

    out = []
    for a, b in segments:
        s = max(0, a * block - pad)
        e = min(len(audio), b * block + pad)
        if (e - s) / SAMPLE_RATE >= MIN_SEG_S:
            out.append((s, e))
    return out, thr


def dbfs(chunk, np) -> float:
    peak = int(np.abs(chunk).max()) or 1
    return 20.0 * np.log10(peak / 32768.0)


def rms_dbfs(chunk, np) -> float:
    r = float(np.sqrt((chunk.astype("float64") ** 2).mean())) or 1.0
    return 20.0 * np.log10(max(r, 1.0) / 32768.0)


def save_wav(path: Path, audio) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SAMPLE_RATE)
        w.writeframes(audio.tobytes())


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("files", nargs="+", type=Path, help="phone recordings, in reading order")
    ap.add_argument("--voice", default=voices.DEFAULT_VOICE,
                    help=f"which agent's voice ({', '.join(sorted(voices.VOICES))})")
    ap.add_argument("--prompts", type=Path, default=DEFAULT_PROMPTS)
    ap.add_argument("--start", type=int, default=None,
                    help="1-based prompt number this batch starts at (default: resume)")
    ap.add_argument("--commit", action="store_true", help="actually write the dataset")
    ap.add_argument("--play", type=int, metavar="N", help="play segment N and exit")
    ap.add_argument("--gap", type=float, default=MIN_GAP_S,
                    help=f"pause length that separates prompts (default {MIN_GAP_S})")
    ap.add_argument("--threshold", type=float, default=None,
                    help="speech/silence RMS line (default: from the noise floor)")
    ap.add_argument("--gain", type=float, default=0.0,
                    help="dB applied to the whole batch, to match earlier batches")
    args = ap.parse_args()

    try:
        import numpy as np
    except ImportError as exc:
        sys.exit(f"needs numpy: {exc}")

    ffmpeg = find_ffmpeg()
    voice = voices.get(args.voice)
    use_voice(voice)
    prompts = voices.load_prompts(args.prompts, voice)
    done = load_done()

    if args.start is not None:
        first = args.start
    else:
        taken = {int(k) for k in done if k.isdigit()}
        first = max(taken) + 1 if taken else 1

    # One continuous stream, so a prompt split across two files still works.
    chunks = [decode(f, ffmpeg) for f in args.files]
    audio = np.concatenate(chunks) if len(chunks) > 1 else chunks[0]
    if args.gain:
        audio = np.clip(audio.astype("float64") * 10 ** (args.gain / 20),
                        -32768, 32767).astype("<i2")

    segments, thr = split(audio, np, args.gap, args.threshold)
    if not segments:
        sys.exit("no speech found — lower --threshold, or check the recording")

    if args.play is not None:
        if not 1 <= args.play <= len(segments):
            sys.exit(f"segment {args.play} out of range (1..{len(segments)})")
        s, e = segments[args.play - 1]
        tmp = Path(tempfile.gettempdir()) / f"seg{args.play:04d}.wav"
        save_wav(tmp, audio[s:e])
        idx = first + args.play - 2
        if 0 <= idx < len(prompts):
            print(f"expected text: {prompts[idx]}")
        try:
            import winsound
            winsound.PlaySound(str(tmp), winsound.SND_FILENAME)
        except Exception:
            print(f"saved to {tmp} — play it yourself")
        return 0

    available = prompts[first - 1:]
    print(f"voice       : {voice.name_en} ({voice.name_ne}, {voice.gender})")
    print(f"dataset     : {voice.dataset.name}")
    print(f"source      : {', '.join(f.name for f in args.files)}")
    print(f"length      : {len(audio) / SAMPLE_RATE / 60:.2f} min")
    print(f"threshold   : {thr:.0f} RMS  (gap {args.gap}s)")
    print(f"segments    : {len(segments)}")
    print(f"starting at : prompt {first:04d}")
    print()

    peaks = []
    for i, (s, e) in enumerate(segments):
        chunk = audio[s:e]
        pk, rd = dbfs(chunk, np), rms_dbfs(chunk, np)
        peaks.append(pk)
        idx = first + i - 1
        text = prompts[idx] if 0 <= idx < len(prompts) else "— NO PROMPT LEFT —"
        clip_id = f"{first + i:04d}"
        flags = []
        if pk > -1.0:
            flags.append("CLIPPED")
        elif pk < -30.0:
            flags.append("TOO-QUIET")
        if clip_id in done:
            flags.append("OVERWRITES")
        mark = "  <<< " + " ".join(flags) if flags else ""
        print(f"  {i + 1:3d} -> {clip_id}  {(e - s) / SAMPLE_RATE:4.1f}s  "
              f"{pk:6.1f}d {rd:6.1f}d  {text[:44]}{mark}")

    pk = np.array(peaks)
    print()
    print(f"peak dBFS   : min {pk.min():.1f}  med {np.median(pk):.1f}  max {pk.max():.1f}"
          f"   spread {pk.max() - pk.min():.1f} dB")
    if np.median(pk) < -20:
        print("  -> quiet. Re-record closer, or pass --gain to lift the batch.")
    if pk.max() - pk.min() > 12:
        print("  -> uneven. If the app applied automatic gain, turn it off and redo.")

    if len(segments) > len(available):
        print(f"\n!! {len(segments)} segments but only {len(available)} prompts left.")
    print("\nCheck that each line above shows the sentence you actually read.")
    print("If a row is off by one, the split is wrong — adjust --gap and look again.")

    if not args.commit:
        print("\nPreview only. Nothing written. Add --commit when it lines up.")
        return 0

    written = 0
    for i, (s, e) in enumerate(segments):
        idx = first + i - 1
        if idx >= len(prompts):
            break
        clip_id = f"{first + i:04d}"
        save_wav(WAVS / f"{clip_id}.wav", audio[s:e])
        done[clip_id] = prompts[idx]
        written += 1

    METADATA.parent.mkdir(parents=True, exist_ok=True)
    with METADATA.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, delimiter="|", quoting=csv.QUOTE_NONE, escapechar="\\")
        for clip_id in sorted(done):
            w.writerow([clip_id, done[clip_id], done[clip_id]])

    print(f"\nwrote {written} clips — dataset now holds {len(done)} of {len(prompts)}.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
