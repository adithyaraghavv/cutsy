"""Evidence packs: 8 evenly spaced frames (<=512 px), a contact sheet, signals, transcript."""
from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

from .candidates import Candidate
from .errors import ClipperError
from .ingest import Media

N_FRAMES = 8
MAX_W = 512


@dataclass
class Item:
    id: str
    rank: int
    cand: Candidate
    hash: str


def trends_hash(trends: dict) -> str:
    return hashlib.sha1(json.dumps({k: trends.get(k) for k in ("keywords", "hashtags", "niche")},
                                   sort_keys=True).encode()).hexdigest()[:8]


def make_items(media: Media, cands: list[Candidate], window: float, trends: dict) -> list[Item]:
    th = trends_hash(trends)
    items = []
    for i, c in enumerate(cands, 1):
        h = hashlib.sha1(f"{media.video_id}|{c.start_s:.2f}|{c.end_s:.2f}|{window}|{th}".encode()).hexdigest()[:8]
        items.append(Item(id=f"c{i:02d}-{h}", rank=i, cand=c, hash=h))
    return items


def length_limits(c: Candidate) -> tuple[float, float]:
    """Allowed length of the trimmed clip: 45-60 s, or the whole window if it is shorter."""
    d = c.end_s - c.start_s
    return min(45.0, d), min(60.0, d)


def extract_frames(media: Media, c: Candidate, d: Path) -> list[dict]:
    ff = shutil.which("ffmpeg")
    if not ff:
        raise ClipperError("ffmpeg was not found on PATH.")
    d.mkdir(parents=True, exist_ok=True)
    start = max(0.0, c.start_s - media.offset)
    dur = max(0.5, min(c.end_s - c.start_s, media.duration - start))
    for old in d.glob("frame_*.jpg"):
        old.unlink()
    r = subprocess.run(
        [ff, "-y", "-v", "error", "-ss", f"{start:.3f}", "-t", f"{dur:.3f}", "-i", str(media.path),
         "-an", "-vf", f"fps={N_FRAMES}/{dur:.3f},scale='min({MAX_W},iw)':-2", "-frames:v", str(N_FRAMES),
         "-q:v", "4", str(d / "frame_%d.jpg")],
        capture_output=True, text=True)
    if r.returncode != 0:
        raise ClipperError(f"Could not extract evidence frames: {r.stderr.strip()[-300:]}")
    frames = sorted(d.glob("frame_*.jpg"), key=lambda p: int(p.stem.split("_")[1]))
    if not frames:
        raise ClipperError("No evidence frames could be extracted (is the clip window inside the video?).")
    subprocess.run(
        [ff, "-y", "-v", "error", "-framerate", "1", "-i", str(d / "frame_%d.jpg"),
         "-vf", "scale=384:-2,tile=4x2", "-frames:v", "1", "-q:v", "4", str(d / "contact.jpg")],
        capture_output=True, text=True)
    step = dur / N_FRAMES
    return [{"file": f.name, "t_s": round(c.start_s + i * step, 2)} for i, f in enumerate(frames)]


def write_pack(media: Media, items: list[Item], out: Path, window: float) -> Path:
    root = out / "evidence"
    for it in items:
        d = root / it.id
        frames = extract_frames(media, it.cand, d)
        lo, hi = length_limits(it.cand)
        chapters = [ch.get("title") for ch in media.chapters
                    if ch.get("start_time", 0) < it.cand.end_s and ch.get("end_time", 1e18) > it.cand.start_s]
        (d / "evidence.json").write_text(json.dumps({
            "id": it.id, "rank_by_signals": it.rank,
            "window": {"start_s": round(it.cand.start_s, 2), "end_s": round(it.cand.end_s, 2)},
            "trim_length_s": {"min": lo, "max": hi},
            "peak_s": round(it.cand.peak_s, 2), "signal_score": round(it.cand.score, 3),
            "signals_mean_z": {k: round(v, 3) for k, v in it.cand.breakdown.items()},
            "frames": frames, "contact_sheet": "contact.jpg",
            "transcript": None,   # filled once captions-from-speech exists
            "video_title": media.title, "chapters_in_window": chapters,
        }, indent=2), encoding="utf-8")
    return root
