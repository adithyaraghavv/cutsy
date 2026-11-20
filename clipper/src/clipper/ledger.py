"""Remembers which moments of a stream were already cut into clips, so a second run on the same stream
brings NEW moments instead of the same clips again.

One small file per stream: <cache_dir>/<video_id>/used_moments.json  ->  [[start_s, end_s], ...]
It holds only time ranges (no credentials, no titles). Local files are not tracked (they share one id).
"""
from __future__ import annotations

import json
from pathlib import Path

MARGIN_S = 10.0     # a new moment this close to an old clip is treated as the same event


def _file(cfg: dict, video_id: str) -> Path:
    return Path(cfg["cache_dir"]) / video_id / "used_moments.json"


def load(cfg: dict, video_id: str) -> list[tuple[float, float]]:
    p = _file(cfg, video_id)
    if not p.is_file():
        return []
    try:
        return [(float(a), float(b)) for a, b in json.loads(p.read_text(encoding="utf-8"))]
    except (json.JSONDecodeError, OSError, ValueError, TypeError):
        return []


def overlaps(start: float, end: float, used: list, margin: float = MARGIN_S) -> bool:
    return any(start < ub + margin and end > ua - margin for ua, ub in used)


def record(cfg: dict, video_id: str, spans: list) -> int:
    """Add clip time ranges to the stream's ledger (merging near-identical ones). Returns how many were new."""
    used = load(cfg, video_id)
    added = 0
    for a, b in spans:
        if not overlaps(float(a), float(b), used, margin=0.0):
            used.append((float(a), float(b)))
            added += 1
    p = _file(cfg, video_id)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps([[round(a, 2), round(b, 2)] for a, b in sorted(used)]), encoding="utf-8")
    return added


def seed_from_jobs(cfg: dict, video_id: str) -> int:
    """First time a stream is seen after upgrading: learn from clips already made in earlier job folders
    (<jobs_dir>/<job>/out/clips.json next to the cache folder). Harmless if there are none."""
    if _file(cfg, video_id).is_file():
        return 0
    spans: list = []
    for cj in Path(cfg["cache_dir"]).parent.glob("*/out/clips.json"):
        try:
            data = json.loads(cj.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        if not isinstance(data, dict) or data.get("video_id") != video_id:
            continue
        spans += [(c["start_s"], c["end_s"]) for c in data.get("clips", []) if "start_s" in c and "end_s" in c]
    return record(cfg, video_id, spans) if spans else 0


def drop_used(cands: list, cfg: dict, video_id: str) -> tuple[list, int]:
    """Remove candidates that overlap an already-clipped moment. -> (remaining, how many removed)"""
    seed_from_jobs(cfg, video_id)
    used = load(cfg, video_id)
    if not used:
        return cands, 0
    keep = [c for c in cands if not overlaps(c.start_s, c.end_s, used)]
    return keep, len(cands) - len(keep)
