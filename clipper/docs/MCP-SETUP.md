# Registering the clipper MCP server

> **The full chat workflow** (paste a link -> clips -> approve -> upload) needs three one-time things on the PC that
> runs Claude Code: the MCP server registered, your layout settings, and the YouTube sign-in. Windows quick start
> (Command Prompt, inside `cutsy\clipper`, with the virtual environment active):
>
> ```bat
> pip install mcp
> setx CLIPPER_JOBS_DIR "C:\Users\YOU\clipper-jobs"
> setx CLIPPER_FACECAM "0.05,0,0.20,0.30"
> setx CLIPPER_GAMEPLAY "0.30,0.05,0.70,0.90"
> setx YOUTUBE_OAUTH_CLIENT_ID "your-client-id"
> setx YOUTUBE_OAUTH_CLIENT_SECRET "your-client-secret"
> ```
> Close and reopen the terminal (so the new variables exist), then:
> ```bat
> clipper --auth youtube
> claude mcp add clipper --env PYTHONPATH="C:\path\to\cutsy\clipper\src" -- "C:\path\to\cutsy\clipper\.venv\Scripts\python.exe" -m clipper.mcp_server
> mkdir "%USERPROFILE%\.claude\skills\clipper"
> copy skill\SKILL.md "%USERPROFILE%\.claude\skills\clipper\SKILL.md"
> ```
> `setx` keeps the client id/secret in your Windows user environment instead of in Claude Code's config file.
> **Leave the `upload_clip` tool on "ask" in Claude Code's permissions**: you then get an extra confirmation prompt for
> every upload, on top of Claude asking you first. Uploads are private by default and happen at most once per clip.

The server is a local **stdio** MCP server (`python -m clipper.mcp_server`, or the `clipper-mcp` command after
`pip install -e .`). It is free: no API key, no per-use cost. Jobs run in background processes and are stored in
a jobs folder, so they survive restarts and no tool call blocks on a long stream.

## Prerequisites
- Python 3.11+ with `numpy` and `mcp` (`pip install mcp`), `ffmpeg`/`ffprobe` on PATH.
- Optional: `yt-dlp` (for URLs; install it yourself) and `faster-whisper` (for `captions=true`).
  These must be installed for the **same Python** that runs the server, and be visible on its PATH.
- Pick two **absolute** paths and substitute them below:
  - `<CLIPPER_SRC>` = the `clipper/src` folder of this repo
  - `<JOBS_DIR>` = where jobs and outputs should live (for example `~/clipper-jobs`)
- Use the full path to your Python (`which python3` / `where python`) as `command` if `python` isn't on the PATH
  that the app sees.

Environment variables the server understands: `CLIPPER_JOBS_DIR` (default `./jobs` relative to its working
directory, so always set it), `CLIPPER_TRENDS` (optional path to your `trends.json`), `YOUTUBE_API_KEY` (optional,
for trend lookups). Keys come only from environment variables, never from files in this repo.

## Claude Code
```bash
claude mcp add clipper \
  --env PYTHONPATH=<CLIPPER_SRC> \
  --env CLIPPER_JOBS_DIR=<JOBS_DIR> \
  -- python3 -m clipper.mcp_server
claude mcp list          # clipper should show as connected
```
Or put it in a project-level `.mcp.json`:
```json
{
  "mcpServers": {
    "clipper": {
      "command": "python3",
      "args": ["-m", "clipper.mcp_server"],
      "env": { "PYTHONPATH": "<CLIPPER_SRC>", "CLIPPER_JOBS_DIR": "<JOBS_DIR>" }
    }
  }
}
```
Install the skill: copy `clipper/skill/SKILL.md` to `~/.claude/skills/clipper/SKILL.md` (or
`<project>/.claude/skills/clipper/SKILL.md`).

## Claude Desktop
Edit `claude_desktop_config.json` (Settings > Developer > Edit Config), then fully restart the app:
- macOS: `~/Library/Application Support/Claude/claude_desktop_config.json`
- Windows: `%APPDATA%\Claude\claude_desktop_config.json`
```json
{
  "mcpServers": {
    "clipper": {
      "command": "python3",
      "args": ["-m", "clipper.mcp_server"],
      "env": { "PYTHONPATH": "<CLIPPER_SRC>", "CLIPPER_JOBS_DIR": "<JOBS_DIR>" }
    }
  }
}
```
On Windows use `"command": "python"` (or the full path to `python.exe`) and double backslashes in paths.
Install the skill through Settings > Capabilities (upload a zip containing the `clipper` folder with `SKILL.md`),
where your plan supports skills. These steps follow the apps' documented flow but were **not tested against a real
Claude Desktop/Claude Code install** here; if a menu name differs, check the current Claude docs.

## Using it
Paste a video link and say what you want ("make 3 Reels from this, niche gta6"). Claude starts a job, polls it,
looks at the evidence frames to judge candidates, then shows the clips from `<JOBS_DIR>/<job_id>/out/`.
Nothing is uploaded, and `approve_clip` only records your approval.

## Troubleshooting
- **Server not shown / fails to start:** run `PYTHONPATH=<CLIPPER_SRC> python3 -m clipper.mcp_server` in a
  terminal; it should wait silently on stdin (Ctrl-C to quit). An ImportError means `mcp`/`numpy` is missing for
  that Python.
- **"yt-dlp is not installed":** install it yourself for that same Python/PATH, or pass a local file path.
- **Job `failed`:** `job_status` shows the reason; full detail is in `<JOBS_DIR>/<job_id>/log.txt`.
- **Stuck `running`:** if the worker's heartbeat stops (machine slept/restarted), the next `job_status` relaunches
  it automatically (up to 3 times); downloads and analysis are cached, so it resumes quickly.
- Office/managed laptops: check your IT policy before registering local MCP servers.
