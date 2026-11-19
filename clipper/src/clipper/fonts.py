"""Cross-platform bold font lookup for drawtext."""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

_CANDIDATES = {
    "win32": [
        r"C:\Windows\Fonts\impact.ttf", r"C:\Windows\Fonts\arialbd.ttf",
        r"C:\Windows\Fonts\segoeuib.ttf", r"C:\Windows\Fonts\verdanab.ttf",
    ],
    "darwin": [
        "/System/Library/Fonts/Supplemental/Impact.ttf",
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
        "/Library/Fonts/Arial Bold.ttf",
        "/System/Library/Fonts/Helvetica.ttc",
        "/System/Library/Fonts/HelveticaNeue.ttc",
    ],
    "linux": [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Bold.ttf",
        "/usr/share/fonts/TTF/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/noto/NotoSans-Bold.ttf",
        "/usr/share/fonts/opentype/inter/Inter-Bold.otf",
    ],
}


def find_bold_font() -> str | None:
    """Return a path to a bold font, or None (ffmpeg then uses its default)."""
    env = os.environ.get("CLIPPER_FONT")
    if env and Path(env).is_file():
        return env
    key = "win32" if sys.platform.startswith("win") else "darwin" if sys.platform == "darwin" else "linux"
    for p in _CANDIDATES[key]:
        if Path(p).is_file():
            return p
    if shutil.which("fc-match"):
        try:
            out = subprocess.run(
                ["fc-match", "-f", "%{file}", "sans:bold"],
                capture_output=True, text=True, timeout=5,
            ).stdout.strip()
            if out and Path(out).is_file():
                return out
        except (OSError, subprocess.SubprocessError):
            pass
    return None
