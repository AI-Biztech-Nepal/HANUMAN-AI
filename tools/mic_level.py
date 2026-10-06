"""Set your mic gain BEFORE recording 202 prompts.

Speak a Nepali sentence at the volume you'll use for the whole session.
Target a green reading, then start record_voice.py and don't touch the
gain, the mic, or your distance from it again.

    python tools/mic_level.py                        # 15s meter, ctrl-c to stop
    python tools/mic_level.py --device "Microphone Array"   # pin the laptop mic
"""
import math
import sys
import time

import argparse
from pathlib import Path

import numpy as np
import sounddevice as sd

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
from tools.record_voice import select_input

RATE, BLOCK, RUN_S = 22050, int(22050 * 0.05), 15.0
# Ashika's usable dataset sits at a peak median of -1.3 dBFS.
# Sagar take 1 sat at -25.8 dBFS and had to be thrown away.
# The trainer rejects any clip peaking below -20 or above -1 dBFS, and a
# speaker's peaks wander +-3 dB from take to take. Aim well inside that
# window rather than at its edge.
GOOD_LO, GOOD_HI = -12.0, -3.0

def dbfs(x):
    return 20.0 * math.log10(max(x, 1e-9))

def _enable_ansi() -> None:
    """Let a classic Windows console draw colour. conhost ignores ANSI escapes
    until VT processing is switched on, and shows them as literal ←[90m text."""
    if sys.platform != "win32":
        return
    import ctypes
    k32 = ctypes.windll.kernel32
    handle = k32.GetStdHandle(-11)                   # STD_OUTPUT_HANDLE
    mode = ctypes.c_uint32()
    if k32.GetConsoleMode(handle, ctypes.byref(mode)):
        k32.SetConsoleMode(handle, mode.value | 0x0004)   # ENABLE_VIRTUAL_TERMINAL_PROCESSING


def main():
    _enable_ansi()
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--device", default=None,
                    help="microphone: an index or part of its name")
    args = ap.parse_args()
    print(f"input: {select_input(sd, args.device)}")
    print(f"target peak {GOOD_LO:.0f}..{GOOD_HI:.0f} dBFS — speak now\n")
    peaks, started = [], time.time()
    with sd.InputStream(samplerate=RATE, channels=1, dtype="int16",
                        blocksize=BLOCK) as stream:
        while time.time() - started < RUN_S:
            data, _ = stream.read(BLOCK)
            peak = dbfs(float(np.abs(data).max()) / 32768.0)
            if peak > -45.0:
                peaks.append(peak)
            bars = int(max(0, (peak + 60) / 60 * 40))
            if peak >= GOOD_HI:      tag, col = "HOT ", "\033[31m"
            elif peak >= GOOD_LO:    tag, col = "GOOD", "\033[32m"
            elif peak >= -25.0:      tag, col = "low ", "\033[33m"
            else:                    tag, col = "----", "\033[90m"
            sys.stdout.write(f"\r{col}{tag}\033[0m {peak:6.1f} dBFS |{'#' * bars:<40}|")
            sys.stdout.flush()
    print()
    if not peaks:
        print("\nheard nothing — wrong input device, or the mic is muted.")
        return 1
    loud = sorted(peaks)[int(len(peaks) * 0.95)]
    print(f"\nloud peaks land at {loud:.1f} dBFS")
    if loud < GOOD_LO:
        print(f"TOO QUIET by {GOOD_LO - loud:.0f} dB. Raise Windows mic level "
              f"(Settings > System > Sound > Microphone Array) or move closer.\n"
              f"This is exactly how take 1 was lost — fix it before recording.")
    elif loud > GOOD_HI:
        print("TOO HOT — clipping risk. Lower the level or back off the mic.")
    else:
        print("Good. Start recording and change nothing about the setup.")
    return 0

if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\nstopped.")
