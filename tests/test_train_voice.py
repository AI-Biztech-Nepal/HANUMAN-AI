"""The dataset audit and level-matching in tools/train_voice.py.

Synthetic clips only: what is under test is the arithmetic that decides whether a
recording session is usable, not the audio. The numbers come from real takes — the
discarded first Sagar take peaked at -28..-23 dBFS and the usable one at -17..-5.
"""
import array
import math
import wave

import pytest

from tools import train_voice, voices

RATE = train_voice.SAMPLE_RATE


def _write_clip(path, peak_dbfs, seconds=2.0):
    """A tone whose loudest sample sits at peak_dbfs."""
    amp = 32768 * 10 ** (peak_dbfs / 20)
    n = int(RATE * seconds)
    samples = array.array("h", (int(amp * math.sin(2 * math.pi * 180 * i / RATE)) for i in range(n)))
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(RATE)
        w.writeframes(samples.tobytes())


def _peak(path):
    return train_voice._peak_dbfs(path)[0]


@pytest.fixture
def voice(tmp_path, monkeypatch):
    monkeypatch.setattr(voices, "REPO", tmp_path)
    v = voices.Voice("tv", "टेस्ट", "Tv", "male")
    v.wavs.mkdir(parents=True)
    return v


def _audit(voice, capsys, *peaks):
    for i, pk in enumerate(peaks, 1):
        _write_clip(voice.wavs / f"{i:04d}.wav", pk)
    train_voice.check(voice)
    return capsys.readouterr().out


def test_takes_like_the_usable_session_pass_the_level_check(voice, capsys):
    out = _audit(voice, capsys, -17.2, -13.0, -5.0)
    assert "TOO QUIET" not in out and "CLIPPING" not in out


def test_takes_like_the_discarded_session_are_flagged(voice, capsys):
    out = _audit(voice, capsys, -25.8, -23.3, -28.3)
    assert "TOO QUIET (3/3 clips" in out


def test_one_quiet_clip_is_reported_among_good_ones(voice, capsys):
    out = _audit(voice, capsys, -8.0, -6.0, -24.0)
    assert "TOO QUIET (1/3 clips" in out


def test_clipping_is_still_refused(voice, capsys):
    out = _audit(voice, capsys, -0.2, -6.0)
    assert "CLIPPING (1 clips)" in out


def test_audit_says_what_levelling_will_do(voice, capsys):
    out = _audit(voice, capsys, -17.0, -5.0)
    assert "boost of +2..+14 dB" in out


def test_prepare_audio_levels_every_clip_and_leaves_originals_alone(voice, tmp_path):
    for i, pk in enumerate((-17.0, -8.0, -0.5), 1):     # quiet, mid, and one that must come DOWN
        _write_clip(voice.wavs / f"{i:04d}.wav", pk)
    before = {p.name: p.read_bytes() for p in voice.wavs.glob("*.wav")}

    out_dir = train_voice.prepare_audio(voice, tmp_path / "build")

    for p in sorted(out_dir.glob("*.wav")):
        assert _peak(p) == pytest.approx(train_voice.PEAK_TARGET_DBFS, abs=0.1)
    assert {p.name: p.read_bytes() for p in voice.wavs.glob("*.wav")} == before


def test_prepare_audio_drops_clips_that_no_longer_exist(voice, tmp_path):
    for i in (1, 2):
        _write_clip(voice.wavs / f"{i:04d}.wav", -10.0)
    train_voice.prepare_audio(voice, tmp_path / "build")
    (voice.wavs / "0002.wav").unlink()
    out_dir = train_voice.prepare_audio(voice, tmp_path / "build")
    assert [p.name for p in out_dir.glob("*.wav")] == ["0001.wav"]


def test_silent_clip_does_not_divide_by_zero(voice, tmp_path):
    with wave.open(str(voice.wavs / "0001.wav"), "wb") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(RATE)
        w.writeframes(bytes(2 * RATE))
    out_dir = train_voice.prepare_audio(voice, tmp_path / "build")
    assert (out_dir / "0001.wav").exists()


def test_cache_is_cleared_only_when_the_audio_changed(voice, tmp_path):
    out = tmp_path / "build"
    _write_clip(voice.wavs / "0001.wav", -10.0)
    audio = train_voice.prepare_audio(voice, out)

    assert train_voice.reset_stale_cache(out, audio) is False        # first run: nothing to clear
    (out / "cache" / "0.audio.pt").write_bytes(b"cached tensor")
    assert train_voice.reset_stale_cache(out, audio) is False        # same audio: keep the cache
    assert (out / "cache" / "0.audio.pt").exists()

    _write_clip(voice.wavs / "0001.wav", -14.0)                       # re-recorded
    audio = train_voice.prepare_audio(voice, out)
    assert train_voice.reset_stale_cache(out, audio) is True          # piper would not have noticed
    assert not (out / "cache" / "0.audio.pt").exists()


def test_a_cache_from_before_fingerprints_existed_is_not_trusted(voice, tmp_path):
    out = tmp_path / "build"
    _write_clip(voice.wavs / "0001.wav", -10.0)
    audio = train_voice.prepare_audio(voice, out)
    (out / "cache").mkdir()
    (out / "cache" / "old.audio.pt").write_bytes(b"from an unknown earlier run")
    assert train_voice.reset_stale_cache(out, audio) is True


def test_recorder_and_trainer_share_one_level_limit():
    # The recorder warned at -18 while the trainer refused at -12, so a whole
    # session passed one and failed the other. There is one source of truth now.
    from tools import record_voice
    assert record_voice.PEAK_QUIET_DBFS == train_voice.PEAK_DBFS_MIN
    assert record_voice.PEAK_HOT_DBFS == train_voice.PEAK_DBFS_MAX
