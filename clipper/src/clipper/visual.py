"""Is there actually a live picture in this moment? (drops 'NO SIGNAL' screens, black screens, frozen frames)"""
from __future__ import annotations

import shutil
import subprocess

import numpy as np

W, H = 64, 36


def liveliness(path, start: float, dur: float) -> dict:
    """Sample the picture at 2 fps: mean brightness, mean frame-to-frame change, share of near-black frames."""
    ff = shutil.which("ffmpeg")
    r = subprocess.run([ff, "-v", "error", "-ss", f"{max(start, 0):.3f}", "-t", f"{max(dur, 1):.3f}", "-i", str(path),
                        "-an", "-vf", f"fps=2,scale={W}:{H}:flags=area,format=gray", "-f", "rawvideo", "-"],
                       capture_output=True)
    fb = W * H
    n = len(r.stdout) // fb
    if n < 3:
        return {"frames": n, "motion": 99.0, "luma": 128.0, "dark_frac": 0.0}
    a = np.frombuffer(r.stdout[: n * fb], dtype=np.uint8).reshape(n, fb).astype(np.float32)
    return {"frames": n, "motion": float(np.abs(np.diff(a, axis=0)).mean()), "luma": float(a.mean()),
            "dark_frac": float((a.mean(axis=1) < 16).mean())}


def dead_reason(info: dict, min_motion: float = 0.25) -> str | None:
    if info["dark_frac"] > 0.8:
        return "black screen"
    if min_motion > 0 and info["motion"] < min_motion:
        return "frozen / static picture (for example a NO SIGNAL screen)"
    return None
