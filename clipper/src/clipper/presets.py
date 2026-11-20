"""Saved facecam/gameplay crops per streamer, so the split layout needs no numbers next time.

Stored in <config dir>/presets.json (the same folder as other per-user settings; CLIPPER_CONFIG_DIR overrides).
Contains only crop rectangles - no credentials.
"""
from __future__ import annotations

import json
import os
import re
from pathlib import Path

from .errors import ClipperError


def _file() -> Path:
    base = Path(os.environ.get("CLIPPER_CONFIG_DIR") or Path.home() / ".config" / "clipper")
    return base / "presets.json"


def _norm(name: str) -> str:
    n = re.sub(r"[^a-z0-9_-]+", "-", (name or "").strip().lower()).strip("-")
    if not n:
        raise ClipperError("Give the preset a name, e.g. ishowspeed.")
    return n


def _load() -> dict:
    p = _file()
    if not p.is_file():
        return {}
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        raise ClipperError(f"Could not read {p}. Fix or delete it.") from None
    return data if isinstance(data, dict) else {}


def get(name: str) -> dict:
    """-> {"facecam": [x,y,w,h], "gameplay": [x,y,w,h] or None}"""
    data = _load()
    key = _norm(name)
    if key not in data:
        known = ", ".join(sorted(data)) or "none saved yet"
        raise ClipperError(f"No preset named '{name}' (saved presets: {known}).")
    return data[key]


def names() -> list[str]:
    return sorted(_load())


def save(name: str, facecam, gameplay=None) -> str:
    from .jobs import _rect  # reuse the same validation as the job parameters
    fc, gp = _rect(facecam, "facecam"), _rect(gameplay, "gameplay")
    if not fc:
        raise ClipperError("A preset needs a facecam rectangle x,y,w,h.")
    data = _load()
    key = _norm(name)
    data[key] = {"facecam": fc, "gameplay": gp}
    p = _file()
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(data, indent=2), encoding="utf-8")
    return key
