from __future__ import annotations

import json
import re
from pathlib import Path

from . import candidates as cand
from . import ingest, signals
from .render import render_clip


def auto_hook(media: ingest.Media, start_s: float) -> str:
    """Phase 1 placeholder hook (Claude writes real ones in Phase 2)."""
    text = ""
    for ch in media.chapters:
        if ch.get("start_time", 0) <= start_s < ch.get("end_time", 1e18):
            text = ch.get("title", "")
    text = text or media.title
    words = re.sub(r"[^\w' !?-]", " ", text).split()
    return " ".join(words[:6]) if words else "WATCH THIS"


def run(source: str, *, n: int, window: float, rng, fill: bool, out: Path, cfg: dict, log=print) -> dict:
    media = ingest.resolve(source, rng, cfg)
    log(f"Source: {media.title or media.source}  ({media.duration:.0f}s, {media.width}x{media.height}"
        f"{'' if media.has_audio else ', no audio'})")
    extras = [name for name, ok in (("heatmap", media.heatmap), ("chat", media.chat_times)) if ok]
    log(f"Extra signals: {', '.join(extras) or 'none'}")

    sig = signals.analyze(media, cfg, rng)
    log(f"Analysed {sig.n * signals.BIN:.0f}s in {sig.n} bins: {', '.join(sig.raw)}")
    cands = cand.pick(sig, window, cfg)
    top = cands[:n]

    out.mkdir(parents=True, exist_ok=True)
    clips = []
    for i, c in enumerate(top, 1):
        hook = auto_hook(media, c.start_s)
        name = f"clip_{i}.mp4"
        end = min(c.end_s, media.offset + media.duration)
        log(f"[{i}/{len(top)}] {c.start_s:.1f}-{end:.1f}s score {c.score:.2f} -> {name}")
        render_clip(media, c.start_s, end, out / name, hook, cfg, fill=fill)
        clips.append({
            "file": name, "start_s": round(c.start_s, 2), "end_s": round(end, 2),
            "score": round(c.score, 4), "hook": hook,
            "signals": {k: round(v, 3) for k, v in c.breakdown.items()},
        })
    result = {"source": media.source, "video_id": media.video_id, "title": media.title, "clips": clips}
    (out / "clips.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    (out / "candidates.json").write_text(json.dumps([
        {"start_s": round(c.start_s, 2), "end_s": round(c.end_s, 2), "score": round(c.score, 4),
         "signals": {k: round(v, 3) for k, v in c.breakdown.items()}} for c in cands], indent=2),
        encoding="utf-8")
    return result
