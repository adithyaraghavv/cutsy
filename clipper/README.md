# clipper

Turn a long YouTube/Twitch VOD (URL or local file) into upload-ready vertical 9:16 clips for
Instagram Reels and YouTube Shorts, using signal analysis (and, from Phase 2, Claude as the judge).

> **Permission.** Only clip streams you have the streamer's **written permission** to clip.
> A live stream that has not ended is refused with a clear message.

## Status

| Phase | Scope | State |
|---|---|---|
| 1 | Ingest, signals, candidates, ffmpeg render (no API cost) | done |
| 2 | Claude judge, trends, captions, QA, report | not started |
| 3 | MCP server, SKILL.md, optional uploader | not started |

## Requirements

- Python 3.11+, `numpy`
- `ffmpeg` and `ffprobe` on PATH
- Optional: `yt-dlp` for URLs. **You install it yourself** (`pip install -U yt-dlp` or your package
  manager). Without it, clipper prints instructions and accepts a local file instead.
- Tests: `pytest`

clipper never installs anything and never reads credentials from files. Keys come only from the
environment (`ANTHROPIC_API_KEY`, `YOUTUBE_API_KEY`, used from Phase 2) and are never logged.

## Usage

```bash
cd clipper
PYTHONPATH=src python -m clipper <url-or-file> [--n 3] [--window 55] [--range START-END] \
    [--niche "gta6"] [--budget-usd 2.0] [--fill] [--out out]
# or, after `pip install -e .`:  clipper <url-or-file> ...
```

- `--range 3600-7200` analyses only that span (seconds). Essential for 8-hour streams; with a URL only
  that section is downloaded.
- `--fill` crops to fill 9:16 instead of a blurred background with the footage centered.
- `--niche` and `--budget-usd` are accepted now and take effect in Phase 2.

Output in `--out` (default `out/`, git-ignored):

- `clip_N.mp4` - 1080x1920, 30 fps, H.264 crf 21, yuv420p, AAC 160k, +faststart, loudnorm about -14 LUFS
- `clips.json` - file, start_s, end_s (absolute source time), score, hook, per-signal breakdown
- `candidates.json` - all (up to 15) scored windows

## How it works

1. **Ingest** - local file, or yt-dlp (<=1080p) with metadata (title, chapters, replay "Most replayed"
   heatmap) and live-chat replay when available; cached per video id in `.clipper-cache/`.
2. **Signals** per 0.5 s: audio RMS + sudden rise, frame motion (64x36 gray @ 2 fps), heatmap, chat-rate
   spike. Z-scored and combined with the weights in `config.yaml` (missing signals are skipped).
3. **Candidates** - slide a `--window` seconds window; score = mean of top 20% bins + 0.5 x mean; greedily
   keep the top 15 non-overlapping. Videos shorter than the window yield one window.
4. **Render** - blurred background + centered footage for landscape; scale-crop for portrait. A hook
   (Phase 1: chapter/title words; Phase 2: Claude) shows for the first 2.6 s via `drawtext` + `textfile`.

## Tests

```bash
cd clipper && python -m pytest
```

Uses a synthetic ffmpeg video (tone burst + moving square). No test needs a network; yt-dlp is faked.

## License

MIT
