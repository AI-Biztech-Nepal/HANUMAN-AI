"""Turn batch recordings of the prompt sheet into a Piper training dataset.

Recording 202 prompts as 202 separate files is tedious, so the batch sheet
(voice_prompts_batches.txt) asks for them read twenty at a time, one file per
batch, with a pause between lines. That gives clean audio and no transcripts:
the recording knows what was said, the filesystem does not.

This recovers them:

    recording -> 22.05kHz mono -> split on the pauses -> transcribe each
    segment -> align the transcripts to the prompt sheet -> LJSpeech dataset

The alignment is the point. Whisper's Nepali model runs at ~27% WER, which is
far too loose to use as a training transcript directly. But we are not asking
it to transcribe — we are asking it to pick which of 202 known sentences was
read, which it is good enough for. And because the prompts are read in order,
the match is constrained to move forward through the sheet, so a segment that
resembles two prompts is resolved by its neighbours rather than by its own
score. That fixes exactly the cases where independent matching fails: "which
model do you like?" and "which colour do you like?" appear twice in the sheet
with near-identical wording.

    python tools/build_dataset.py --voice ashika --src <folder> --dry-run
    python tools/build_dataset.py --voice ashika --src <folder>

Then check what came out before spending GPU time on it:

    python tools/train_voice.py --voice ashika --check
"""
from __future__ import annotations

import argparse
import difflib
import re
import shutil
import subprocess
import sys
import wave
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "media"))

from tools import voices
from tools.nepali_voice import split_utterances

SAMPLE_RATE = 22050          # what Piper trains at; matches record_voice.py
PROMPT_SHEET = REPO / "voice_prompts_batches.txt"

# Segment length bounds. Piper learns from utterances, not paragraphs: a
# segment far longer than a prompt means the reader ran two or more lines
# together without pausing, so the audio no longer matches any single
# transcript and has to be dropped rather than mislabelled.
MIN_SEG_S = 0.7
MAX_SEG_S = 12.0

# How close a transcript must be to a prompt to be believed. Below this the
# segment is dropped: a wrong transcript is worse than no clip, because the
# model learns to say those words with that audio.
MIN_RATIO = 0.62

# Cost of skipping a segment or a prompt during alignment. Low enough that a
# genuinely missing prompt does not drag the whole sequence out of step.
GAP = 0.35


def _norm(s: str) -> str:
    """Compare on Devanagari letters alone — punctuation is not spoken, and
    Whisper's choice of it tells us nothing about what she said."""
    return re.sub(r"[^ऀ-ॿ]", "", s)


def load_prompts() -> list[tuple[int, str]]:
    if not PROMPT_SHEET.exists():
        raise SystemExit(f"no prompt sheet at {PROMPT_SHEET}")
    text = PROMPT_SHEET.read_text(encoding="utf-8")
    items = [(int(m.group(1)), m.group(2).strip())
             for m in re.finditer(r"^(\d{4})\.\s+(.+)$", text, re.M)]
    if not items:
        raise SystemExit(f"no numbered prompts found in {PROMPT_SHEET}")
    return sorted(items)


def to_wav(src: Path, dst: Path) -> None:
    """Any recording -> mono 22.05kHz wav, the one format everything downstream
    assumes. Phones record 48kHz stereo-ish; resampling later would mean
    resampling twice."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(src),
                    "-ac", "1", "-ar", str(SAMPLE_RATE), str(dst)], check=True)


def seconds(path: Path) -> float:
    with wave.open(str(path)) as w:
        return w.getnframes() / w.getframerate()


def align(hyps: list[str], prompts: list[tuple[int, str]]) -> list[int | None]:
    """Monotonic best alignment of transcripts to prompts.

    Needleman-Wunsch over (segment, prompt) with similarity as the match score.
    Monotonic because the sheet is read top to bottom: segment k+1 cannot map
    to a prompt above segment k's. Returns, per segment, the index into
    `prompts` it was matched to, or None if it is better left unmatched.
    """
    n, m = len(hyps), len(prompts)
    if not n or not m:
        return [None] * n
    H = [_norm(h) for h in hyps]
    P = [_norm(p) for _, p in prompts]

    def sim(i: int, j: int) -> float:
        if not H[i] or not P[j]:
            return 0.0
        return difflib.SequenceMatcher(None, H[i], P[j]).ratio()

    # score[i][j] = best total score aligning the first i segments to the
    # first j prompts. back[i][j] records which move produced it.
    score = [[0.0] * (m + 1) for _ in range(n + 1)]
    back = [[""] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        score[i][0] = score[i - 1][0] - GAP
        back[i][0] = "seg"
    for j in range(1, m + 1):
        score[0][j] = score[0][j - 1] - GAP
        back[0][j] = "prompt"
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            diag = score[i - 1][j - 1] + sim(i - 1, j - 1)
            skip_seg = score[i - 1][j] - GAP
            skip_prompt = score[i][j - 1] - GAP
            best = max(diag, skip_seg, skip_prompt)
            score[i][j] = best
            back[i][j] = ("match" if best == diag else
                          "seg" if best == skip_seg else "prompt")

    out: list[int | None] = [None] * n
    i, j = n, m
    while i > 0 and j > 0:
        move = back[i][j]
        if move == "match":
            out[i - 1] = j - 1
            i, j = i - 1, j - 1
        elif move == "seg":
            i -= 1
        else:
            j -= 1
    return out


def build(voice, src: Path, gap_s: float, dry_run: bool) -> int:
    prompts = load_prompts()
    print(f"{len(prompts)} prompts in the sheet")

    sources = sorted(p for p in src.rglob("*")
                     if p.suffix.lower() in
                     (".mp4", ".m4a", ".wav", ".mp3", ".opus", ".ogg", ".aac"))
    if not sources:
        raise SystemExit(f"no recordings under {src}")
    print(f"{len(sources)} recordings under {src}\n")

    work = REPO / "voice_training" / f"{voice.key}_build"
    work.mkdir(parents=True, exist_ok=True)

    import voice_bridge

    kept: list[tuple[Path, int, str]] = []       # (wav, prompt number, text)
    seen: set[int] = set()
    stats = {"segments": 0, "too_long": 0, "too_short": 0,
             "unmatched": 0, "low": 0, "duplicate": 0}

    for f in sources:
        tag = str(f.relative_to(src)).replace("/", "_").replace("\\", "_")
        tag = re.sub(r"[^A-Za-z0-9_.-]", "", tag)
        wav = work / f"{tag}.wav"
        if not wav.exists():
            to_wav(f, wav)
        segs = [Path(s) for s in split_utterances(wav, work / f"{tag}_segs",
                                                  gap_s=gap_s)]
        usable, lengths = [], []
        for s in segs:
            d = seconds(s)
            stats["segments"] += 1
            if d > MAX_SEG_S:
                stats["too_long"] += 1
                continue
            if d < MIN_SEG_S:
                stats["too_short"] += 1
                continue
            usable.append(s)
            lengths.append(d)

        hyps = [voice_bridge.transcribe_wav(str(s), language="ne")
                for s in usable]
        matched = align(hyps, prompts)

        good = 0
        for s, hyp, idx in zip(usable, hyps, matched):
            if idx is None:
                stats["unmatched"] += 1
                continue
            num, text = prompts[idx]
            ratio = difflib.SequenceMatcher(None, _norm(hyp), _norm(text)).ratio()
            if ratio < MIN_RATIO:
                stats["low"] += 1
                continue
            if num in seen:
                # A prompt read twice across takes: keep the first. Two clips
                # with one transcript teaches the model that the same words
                # have two rhythms, which is heard as hesitancy.
                stats["duplicate"] += 1
                continue
            seen.add(num)
            kept.append((s, num, text))
            good += 1

        print(f"{tag:<28} {len(segs):>3} segs  {len(usable):>3} usable  "
              f"{good:>3} matched")

    print(f"\nsegments seen      : {stats['segments']}")
    print(f"  dropped, too long: {stats['too_long']} (>{MAX_SEG_S:.0f}s "
          f"— lines run together)")
    print(f"  dropped, too short: {stats['too_short']} (<{MIN_SEG_S}s)")
    print(f"  dropped, no match : {stats['unmatched']}")
    print(f"  dropped, low conf : {stats['low']} (<{MIN_RATIO})")
    print(f"  dropped, duplicate: {stats['duplicate']}")
    print(f"KEPT               : {len(kept)} clips covering "
          f"{len(seen)}/{len(prompts)} prompts")

    total = sum(seconds(w) for w, _, _ in kept)
    print(f"usable audio       : {total/60:.1f} min")

    if dry_run:
        print("\n--dry-run: nothing written.")
        return 0
    if not kept:
        print("\nNothing to write.")
        return 1

    voice.wavs.mkdir(parents=True, exist_ok=True)
    rows = []
    for s, num, text in sorted(kept, key=lambda k: k[1]):
        # The clip id is the prompt's own number, which is what
        # record_voice.py uses too. That is what makes the two tools agree:
        # pointing the recorder at this dataset afterwards offers exactly the
        # prompts that are missing here, instead of starting again from one.
        clip_id = f"{num:04d}"
        shutil.copyfile(s, voice.wavs / f"{clip_id}.wav")
        rows.append(f"{clip_id}|{text}|{text}")
    voice.metadata.write_text("\n".join(rows) + "\n", encoding="utf-8")

    missing = [n for n, _ in prompts if n not in seen]
    print(f"\nwrote {len(rows)} clips to {voice.wavs}")
    print(f"wrote {voice.metadata}")
    print(f"\n{len(missing)} prompts still unrecorded"
          + (f", first few: {', '.join(f'{n:04d}' for n in missing[:8])}"
             if missing else ""))
    print(f"\nNext:  python tools/record_voice.py --voice {voice.key}"
          f"   (offers only what is missing)")
    print(f"       python tools/train_voice.py --voice {voice.key} --check")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--voice", default=voices.DEFAULT_VOICE,
                    help=f"which agent: {', '.join(sorted(voices.VOICES))}")
    ap.add_argument("--src", type=Path, required=True,
                    help="folder of batch recordings")
    ap.add_argument("--gap", type=float, default=0.6,
                    help="silence, in seconds, that separates two prompts")
    ap.add_argument("--dry-run", action="store_true",
                    help="report what would be kept, write nothing")
    args = ap.parse_args()

    voice = voices.get(args.voice)
    if voice.dataset.exists() and not args.dry_run:
        raise SystemExit(
            f"{voice.dataset} already exists — move it aside first, so a "
            f"rebuild never half-overwrites a dataset you recorded by hand.")
    return build(voice, args.src, args.gap, args.dry_run)


if __name__ == "__main__":
    raise SystemExit(main())
