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


_TITLE_FIRST = {
    "win32": [r"C:\Windows\Fonts\arialbd.ttf", r"C:\Windows\Fonts\segoeuib.ttf"],
    "darwin": ["/System/Library/Fonts/Supplemental/Arial Bold.ttf", "/Library/Fonts/Arial Bold.ttf"],
    "linux": ["/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
              "/usr/share/fonts/truetype/liberation2/LiberationSans-Bold.ttf"],
}


def find_bold_font(prefer_title: bool = False) -> str | None:
    """Return a path to a bold font, or None (ffmpeg then uses its default).
    prefer_title: favour an Arial-style bold (headline box) over a display font like Impact."""
    env = os.environ.get("CLIPPER_FONT")
    if env and Path(env).is_file():
        return env
    key = "win32" if sys.platform.startswith("win") else "darwin" if sys.platform == "darwin" else "linux"
    for p in (_TITLE_FIRST[key] if prefer_title else []) + _CANDIDATES[key]:
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


def font_family(path: str | None) -> str:
    """Family name for an ASS style (fc-scan if available, else a sensible guess)."""
    if path and shutil.which("fc-scan"):
        try:
            out = subprocess.run(["fc-scan", "--format", "%{family}", path], capture_output=True,
                                 text=True, timeout=5).stdout.strip()
            if out:
                return out.split(",")[0]
        except (OSError, subprocess.SubprocessError):
            pass
    return Path(path).stem.replace("-Bold", "").replace("Bold", "") if path else "Arial"
