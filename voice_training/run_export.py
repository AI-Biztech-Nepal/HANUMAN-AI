"""Export a Piper checkpoint to ONNX, forcing the legacy TorchScript exporter.

This torch version's `torch.onnx.export` defaults to the newer dynamo-based
exporter, which is far stricter about data-dependent control flow than the
old tracer. It fails on a genuine, harmless assertion inside VITS's spline
transform (`rational_quadratic_spline`) that the legacy tracer has always
handled fine. Piper's own export_onnx.py doesn't pass `dynamo=`, so it
inherits whichever default this torch version ships. Forcing dynamo=False
here restores the behavior export_onnx.py was written against, without
patching the installed package.

Usage: same arguments as `python -m piper.train.export_onnx ...`
"""
import sys

import torch

_real_export = torch.onnx.export


def _export_legacy(*args, **kwargs):
    kwargs.setdefault("dynamo", False)
    return _real_export(*args, **kwargs)


torch.onnx.export = _export_legacy

sys.argv[0] = "piper.train.export_onnx"
from piper.train.export_onnx import main  # noqa: E402

main()
