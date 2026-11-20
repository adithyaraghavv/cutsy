# clipper

Turn a long YouTube/Twitch VOD (URL or local file) into upload-ready vertical 9:16 clips for
Instagram Reels and YouTube Shorts, using signal analysis (and, from Phase 2, Claude as the judge).

> **Permission.** Only clip streams you have the streamer's **written permission** to clip.
> A live stream that has not ended is refused with a clear message.

## Status

| Phase | Scope | State |
|---|---|---|
| 1 | Ingest, signals, candidates, ffmpeg render (no API cost) | done |
| 2 | Judge (free backends), trends, captions, QA, report | done |
| 3 | MCP server, SKILL.md, setup docs | done, plus the optional uploader |

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

## Phase 2: judge, trends, QA, report (no paid API)

Judging is a swappable backend (`--judge`), so a paid or local model can be added later without touching
the pipeline (subclass `Judge` in `judge.py`, register it in `JUDGES`).

- `--judge heuristic` (default): offline. Ranks by signals; hooks/hashtags come from `--niche`, trends and the
  title. It cannot see the video, so risk flags are reported as *not assessed*.
- `--judge chat`: **Claude in your chat judges, no API key and no per-token cost.** clipper writes
  `out/evidence/<id>/` (8 frames <=512 px, a contact sheet, signals, transcript slot) and `out/judge_prompt.md`
  for the top `--shortlist` (default 5) candidates, then stops. Claude writes `out/judgments.json`
  (score 1-10, reason, best_start/best_end trimmed to 45-60 s inside the window, hook <=6 words, caption,
  exactly 5 hashtags, risk_flags from spoilers/cutscenes/copyright/profanity). Re-run with
  `--judgments out/judgments.json`; each entry is validated and bad ones are bounced with a reason.
  Results are cached by candidate hash, so re-runs never re-ask.
- **Trends** (all optional, fail soft, never scrape Instagram/TikTok): a hand-edited `trends.json`
  (`{"keywords": [], "hashtags": [], "niches": {"gta6": {...}}}`, see `trends.example.json`) and the official
  YouTube Data API (mostPopular + search for `--niche`) when `YOUTUBE_API_KEY` is set. The key is sent in a
  header, never put in a URL or logged.
- **QA** after rendering: `--qa heuristic` (default) checks black bars, hook visible and not blank on 4 frames,
  re-rendering once with `--fill` if bars are found. It cannot judge cropping or whether the subject is visible.
  `--qa chat` also writes `out/qa/*_sheet.jpg` + `out/qa_prompt.md` for Claude; apply its `qa.json` with
  `--qa-results out/qa.json` (each failing clip is re-rendered once with the suggested fix).
- `out/report.md`: why each clip was picked, signal breakdown, judge score/reason/flags, QA results, API cost
  ($0.00 with the free backends). `--budget-usd` is reserved for paid judges.

- **Captions** (`--captions`, optional): speech recognition with faster-whisper (free, offline; install it
  yourself with `pip install faster-whisper`). The model (`base.en`, ~150 MB, set in `config.yaml`) downloads once
  from Hugging Face into `<cache_dir>/models`. Only the shortlisted candidate windows are transcribed, cached by
  candidate hash, and the transcript is added to each evidence pack so the judge can read it. Captions are
  burned in as 2-word chunks with one highlighted key word each, positioned above the Reels/Shorts UI.

## Phase 3: use it from Claude (MCP server + skill)

`python -m clipper.mcp_server` (stdio) exposes `start_clip_job`, `job_status`, `list_clips`, `approve_clip`, plus
`get_evidence` and `submit_judgments` (how Claude in your chat judges candidates without an API key). Jobs run in
detached background processes and live in `<jobs dir>/<id>/`, so they survive restarts and never block a tool call.
`approve_clip` only records your approval; nothing is ever uploaded. Setup for Claude Desktop and Claude Code:
[docs/MCP-SETUP.md](docs/MCP-SETUP.md). The skill is [skill/SKILL.md](skill/SKILL.md).

### Optional uploader
`--upload youtube|instagram|both` offers to upload each rendered clip through the official APIs (your own OAuth
credentials from environment variables). It asks `y/n` for every clip, needs a real terminal, defaults YouTube to
private, and is unreachable from the MCP server. See [docs/UPLOADER.md](docs/UPLOADER.md). Untested against the real APIs.

## Tests

```bash
cd clipper && python -m pytest
```

Uses a synthetic ffmpeg video (tone burst + moving square). No test needs a network; yt-dlp is faked.

## License

MIT
