"""The agent voices we train, and where each one's dataset lives.

Each AI agent on the platform gets its own speaking voice, which means its own
recorded dataset and its own fine-tuned Piper model. They must never share a
directory: two speakers mixed into one dataset trains a model that sounds like
neither of them.

The prompt file is written once, name-neutral, with {AGENT} where the agent
says its own name. Each voice substitutes its own name at read time, so both
voices read the same 202 sentences and only the four self-introductions differ.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent


@dataclass(frozen=True)
class Voice:
    key: str            # what you type on the command line
    name_ne: str        # how the agent says its own name, in Devanagari
    name_en: str        # the tenant config's agent_name
    gender: str         # who should be at the microphone
    aliases: tuple = ()  # other spellings a tenant may have saved

    @property
    def dataset(self) -> Path:
        return REPO / f"voice_dataset_{self.key}"

    @property
    def wavs(self) -> Path:
        return self.dataset / "wavs"

    @property
    def metadata(self) -> Path:
        return self.dataset / "metadata.csv"

    @property
    def model(self) -> Path:
        """Where this voice's own fine-tuned Piper model lives, once trained.

        Until it exists the agent speaks through a base Piper model with the
        timbre converted by OpenVoice (see app/tone.py) — right words, roughly
        the right voice, someone else's rhythm. A trained model here replaces
        both halves: it says the words in this speaker's own voice and cadence,
        and needs no conversion pass at all.
        """
        return REPO / "media" / "voices" / f"{self.key}.onnx"

    @property
    def trained(self) -> bool:
        return self.model.exists()


VOICES = {
    "sagar": Voice("sagar", "सागर", "Sagar", "male", ("saagar", "sagarji")),
    "ashika": Voice("ashika", "आशिका", "Ashika", "female",
                    ("aashika", "asika", "aasika", "ashika ji")),
}

DEFAULT_VOICE = "ashika"

# Every spelling that resolves to a voice. Tenants type the agent's name by
# hand, so "Aashika" and "Ashika" both have to find her — a name that fails to
# match silently falls back to the generic voice with nothing to explain why.
_BY_NAME = {}
for _v in VOICES.values():
    _BY_NAME[_v.key] = _v
    _BY_NAME[_v.name_en.lower()] = _v
    for _a in _v.aliases:
        _BY_NAME[_a] = _v


def resolve(name: str) -> Voice | None:
    """A voice for any spelling of its name, or None if it is not one of ours."""
    return _BY_NAME.get((name or "").strip().lower())


def get(key: str) -> Voice:
    v = resolve(key)
    if v is None:
        known = ", ".join(sorted(VOICES))
        raise SystemExit(f"unknown voice {key!r} — known voices: {known}")
    return v


def load_prompts(path: Path, voice: Voice) -> list[str]:
    """Prompt lines with this voice's name filled in."""
    if not path.exists():
        raise SystemExit(f"no prompt file at {path}")
    lines = [ln.strip() for ln in path.read_text(encoding="utf-8").splitlines()]
    return [ln.replace("{AGENT}", voice.name_ne)
            for ln in lines if ln and not ln.startswith("#")]
