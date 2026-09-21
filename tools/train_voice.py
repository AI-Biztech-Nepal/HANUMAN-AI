"""Turn a recorded dataset into an agent's own Piper voice.

    record_voice.py  ->  voice_dataset_<key>/  ->  here  ->  media/voices/<key>.onnx

This is the step that ends the borrowed voice. Today an agent speaks through
Piper's ne_NP model with OpenVoice reshaping the timbre (app/tone.py): the
words and pronunciation are a stranger's, the timbre is roughly the agent's,
and the rhythm is nobody's. A fine-tune on the agent's own speech gives all
three at once, and drops the conversion pass — and its latency — entirely.

    python tools/train_voice.py --voice ashika --check    # is the data usable?
    python tools/train_voice.py --voice ashika --base <ckpt>
    python tools/train_voice.py --voice ashika --export   # checkpoint -> onnx

Run it under WSL: training and Piper are Linux-side here.

    .venv-wsl/bin/python tools/train_voice.py --voice ashika --check

--check first, always. Training is hours of compute that cannot fix a dataset
recorded too quietly or too short, and both are easy to do by accident — the
first Sagar take was 48 seconds at -26 dBFS and had to be thrown away after
the fact. Checking costs seconds and is the only cheap moment to find out.
"""
from __future__ import annotations

import argparse
import math
import struct
import subprocess
import sys
import wave
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from tools import voices

# What a usable dataset looks like. These are floors, not targets.
#
# MIN_MINUTES: VITS fine-tuning starts paying off around 20 minutes of clean
#   speech from one speaker. Less than that and the model keeps the base
#   voice's character where the data is thin, which is heard as the voice
#   drifting mid-sentence.
# PEAK_DBFS_*: recorded level. Too quiet and the noise floor gets amplified
#   along with the voice during normalisation; too hot and the clipping is
#   baked in and trains as part of the timbre.
MIN_MINUTES = 20.0
WARN_MINUTES = 30.0
PEAK_DBFS_MIN = -12.0
PEAK_DBFS_MAX = -1.0
SAMPLE_RATE = 22050

# espeak-ng's Nepali voice: the phonemiser, not the speaker. The prompts in
# tools/prompts_ne.txt are Nepali, so a voice trained from them is a Nepali
# voice — see app/voice.py, which keeps English on the base model.
ESPEAK_VOICE = "ne"


def _peak_dbfs(path: Path) -> tuple[float, float]:
    """(peak dBFS, seconds) for one 16-bit mono wav."""
    with wave.open(str(path)) as f:
        n, sr = f.getnframes(), f.getframerate()
        if n == 0:
            return -math.inf, 0.0
        a = struct.unpack("<%dh" % n, f.readframes(n))
    peak = max(abs(min(a)), abs(max(a)))
    return (20 * math.log10(peak / 32768) if peak else -math.inf), n / sr


def check(voice) -> bool:
    """Report whether this dataset is worth spending training time on."""
    if not voice.wavs.is_dir():
        print(f"no dataset at {voice.dataset}")
        print(f"  record one:  python tools/record_voice.py --voice {voice.key}")
        return False

    clips = sorted(voice.wavs.glob("*.wav"))
    if not clips:
        print(f"{voice.dataset} has no wavs yet")
        return False

    total = 0.0
    quiet, hot, wrong_format = [], [], []
    for w in clips:
        pk, secs = _peak_dbfs(w)
        total += secs
        if pk < PEAK_DBFS_MIN:
            quiet.append((w.name, pk))
        elif pk > PEAK_DBFS_MAX:
            hot.append((w.name, pk))
        with wave.open(str(w)) as f:
            if f.getframerate() != SAMPLE_RATE or f.getnchannels() != 1:
                wrong_format.append(w.name)

    minutes = total / 60
    print(f"voice:     {voice.key} ({voice.name_en} / {voice.name_ne})")
    print(f"clips:     {len(clips)}")
    print(f"audio:     {minutes:.1f} min  (floor {MIN_MINUTES:.0f}, "
          f"comfortable {WARN_MINUTES:.0f})")
    print(f"metadata:  {'ok' if voice.metadata.exists() else 'MISSING'}")

    ok = True
    if minutes < MIN_MINUTES:
        print(f"\nTOO SHORT — {minutes:.1f} min of {MIN_MINUTES:.0f}. "
              f"Keep recording; this is the single biggest lever on quality.")
        ok = False
    elif minutes < WARN_MINUTES:
        print(f"\nEnough to train, but {WARN_MINUTES:.0f}+ min gives a "
              f"noticeably steadier voice.")

    if wrong_format:
        print(f"\nWRONG FORMAT ({len(wrong_format)}) — need mono {SAMPLE_RATE}Hz: "
              f"{', '.join(wrong_format[:5])}")
        ok = False
    if quiet:
        worst = min(pk for _, pk in quiet)
        print(f"\nTOO QUIET ({len(quiet)}/{len(clips)} clips, worst {worst:.1f} dBFS) "
              f"— want peaks between {PEAK_DBFS_MIN:.0f} and {PEAK_DBFS_MAX:.0f}.")
        print("  Move closer to the mic or raise its gain and re-record. "
              "Normalising afterwards raises the room noise with the voice.")
        ok = False
    if hot:
        print(f"\nCLIPPING ({len(hot)} clips) — lower the gain and re-record "
              f"those: {', '.join(n for n, _ in hot[:5])}")
        ok = False

    print("\nREADY to train." if ok else "\nNOT ready — fix the above first.")
    return ok


def train(voice, base: Path | None, epochs: int, batch_size: int) -> int:
    """Fine-tune a Piper model on this dataset via piper.train's Lightning CLI."""
    out = REPO / "voice_training" / voice.key
    out.mkdir(parents=True, exist_ok=True)
    config = out / "config.json"

    cmd = [
        sys.executable, "-m", "piper.train", "fit",
        "--data.csv_path", str(voice.metadata),
        "--data.audio_dir", str(voice.wavs),
        "--data.cache_dir", str(out / "cache"),
        "--data.config_path", str(config),
        "--data.voice_name", voice.key,
        "--data.espeak_voice", ESPEAK_VOICE,
        "--data.batch_size", str(batch_size),
        "--model.sample_rate", str(SAMPLE_RATE),
        "--trainer.max_epochs", str(epochs),
        "--trainer.default_root_dir", str(out),
    ]
    if base:
        # Fine-tuning from an existing checkpoint, rather than training from
        # scratch. From scratch needs tens of hours of speech and days of GPU;
        # from a checkpoint, 20-30 minutes of speech is enough, because the
        # model already knows how to speak and only has to learn this voice.
        cmd += ["--ckpt_path", str(base)]

    print("$ " + " ".join(cmd) + "\n")
    return subprocess.call(cmd)


def export(voice) -> int:
    """Best checkpoint -> media/voices/<key>.onnx, where app/voice.py finds it."""
    ckpts = sorted((REPO / "voice_training" / voice.key).rglob("*.ckpt"),
                   key=lambda p: p.stat().st_mtime, reverse=True)
    if not ckpts:
        print(f"no checkpoints under voice_training/{voice.key} — train first")
        return 1
    # last.ckpt is the newest, not the best: piper.train keeps the top-5 by
    # val_mel and by val_mos alongside it. Prefer a monitored one, and say
    # which, because the difference is audible.
    best = next((c for c in ckpts if c.name != "last.ckpt"), ckpts[0])
    voice.model.parent.mkdir(parents=True, exist_ok=True)
    cmd = [sys.executable, "-m", "piper.train.export_onnx",
           "--checkpoint", str(best), "--output-file", str(voice.model)]
    print(f"exporting {best.name} -> {voice.model}")
    rc = subprocess.call(cmd)
    if rc == 0:
        print(f"\n{voice.name_en} now has a trained voice at {voice.model}.")
        print("app/voice.py picks it up automatically — restart the server.")
    return rc


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--voice", default=voices.DEFAULT_VOICE,
                    help=f"which agent: {', '.join(sorted(voices.VOICES))}")
    ap.add_argument("--check", action="store_true",
                    help="inspect the dataset and exit (do this first)")
    ap.add_argument("--export", action="store_true",
                    help="export the trained checkpoint to onnx and exit")
    ap.add_argument("--base", type=Path,
                    help="checkpoint to fine-tune from (see docs/VOICE_TRAINING.md)")
    ap.add_argument("--epochs", type=int, default=2000)
    ap.add_argument("--batch-size", type=int, default=16)
    args = ap.parse_args()

    voice = voices.get(args.voice)
    if args.check:
        return 0 if check(voice) else 1
    if args.export:
        return export(voice)
    if not check(voice):
        print("\nRefusing to train on this dataset. Use --check to re-inspect.")
        return 1
    return train(voice, args.base, args.epochs, args.batch_size)


if __name__ == "__main__":
    raise SystemExit(main())
