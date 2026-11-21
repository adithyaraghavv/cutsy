"""Remembers that YouTube's daily upload limit was hit, so no more doomed uploads are queued.

Only a timestamp is stored (<config dir>/youtube_limit.json). No credentials, no network code."""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

COOLDOWN_S = 6 * 3600      # the limit resets on a rolling ~24 h; don't try again for this long after hitting it


def _file() -> Path:
    base = Path(os.environ.get("CLIPPER_CONFIG_DIR") or Path.home() / ".config" / "clipper")
    return base / "youtube_limit.json"


def hours_left() -> float:
    """Hours until uploads are worth trying again; 0 if there was no recent limit error."""
    try:
        t = float(json.loads(_file().read_text(encoding="utf-8")).get("hit_at", 0))
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        return 0.0
    return max(0.0, (t + COOLDOWN_S - time.time()) / 3600.0)


def mark_hit() -> None:
    p = _file()
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps({"hit_at": time.time()}), encoding="utf-8")
