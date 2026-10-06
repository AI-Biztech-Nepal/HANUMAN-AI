"""Run piper.train's CLI with the ne_NP-google-medium base checkpoint loadable.

PyTorch 2.6+ changed torch.load's default to weights_only=True, which Lightning's
checkpoint-path parsing uses internally and which rejects the pickled PosixPath
object inside this (older) checkpoint. The checkpoint is Piper's own official
release from huggingface.co/datasets/rhasspy/piper-checkpoints, so allow-listing
that one type is safe — this is not a blanket switch to unsafe unpickling.

Usage: same arguments as `python -m piper.train fit ...`
"""
import pathlib
import sys

import torch

torch.serialization.add_safe_globals([pathlib.PosixPath])

sys.argv[0] = "piper.train"
from piper.train.__main__ import main  # noqa: E402

main()
