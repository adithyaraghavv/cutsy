"""Resolve a local file or URL into a Media object (+ cached metadata)."""
from __future__ import annotations

import hashlib
import json
import re
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

from .errors import ClipperError

YTDLP_HELP = (
    "yt-dlp is not installed, so URLs can't be downloaded.\n"
    "  Install it yourself (clipper never installs anything):\n"
    "    pip install -U yt-dlp      or      brew install yt-dlp\n"
    "    https://github.com/yt-dlp/yt-dlp#installation\n"
    "  Or download the VOD yourself and pass the local file:\n"
    "    clipper path/to/video.mp4"
)

LIVE_STATES = {"is_live", "is_upcoming", "post_live"}


@dataclass
class Media:
    path: Path
    duration: float                 # seconds, of `path`
    width: int
    height: int
    has_audio: bool
    offset: float = 0.0             # absolute source time of path's t=0
    video_id: str = "local"
    title: str = ""
    description: str = ""
    chapters: list = field(default_factory=list)
    heatmap: list = field(default_factory=list)      # [{start_time,end_time,value}]
    chat_times: list = field(default_factory=list)   # absolute seconds
    source: str = ""


def _need(tool: str) -> str:
    p = shutil.which(tool)
    if not p:
        raise ClipperError(f"{tool} was not found on PATH. ffmpeg/ffprobe are required.")
    return p


def probe(path: Path, need_video: bool = True) -> dict:
    try:
        out = subprocess.run(
            [_need("ffprobe"), "-v", "error", "-print_format", "json",
             "-show_format", "-show_streams", str(path)],
            capture_output=True, text=True, check=True,
        ).stdout
    except subprocess.CalledProcessError as e:
        raise ClipperError(f"Could not read {path} as a video (ffprobe: {e.stderr.strip()[:200]}).") from None
    info = json.loads(out)
    v = next((s for s in info["streams"] if s["codec_type"] == "video"), None)
    if v is None and need_video:
        raise ClipperError(f"{path} has no video stream.")
    has_audio = any(s["codec_type"] == "audio" for s in info["streams"])
    dur = float(info["format"].get("duration") or (v or {}).get("duration") or 0)
    if dur <= 0:
        raise ClipperError(f"Could not determine the duration of {path}.")
    if v is None:
        return {"duration": dur, "width": 0, "height": 0, "has_audio": has_audio}
    rot = 0
    for sd in v.get("side_data_list", []) or []:
        if "rotation" in sd:
            rot = abs(int(sd["rotation"])) % 180
    w, h = int(v["width"]), int(v["height"])
    if rot == 90:
        w, h = h, w
    return {"duration": dur, "width": w, "height": h, "has_audio": has_audio}


def is_url(s: str) -> bool:
    return bool(re.match(r"^https?://", s.strip(), re.I))


def parse_range(text: str | None) -> tuple[float, float] | None:
    if not text:
        return None
    m = re.fullmatch(r"\s*(\d+(?:\.\d+)?)\s*-\s*(\d+(?:\.\d+)?)\s*", text)
    if not m or float(m.group(2)) <= float(m.group(1)):
        raise ClipperError(f"--range must look like START-END in seconds, e.g. 3600-7200 (got {text!r}).")
    return float(m.group(1)), float(m.group(2))


def resolve(source: str, rng: tuple[float, float] | None, cfg: dict, log=None, chat: bool = True) -> Media:
    if is_url(source):
        return _resolve_url(source, rng, cfg, log or (lambda *_: None), chat)
    p = Path(source).expanduser()
    if not p.is_file():
        raise ClipperError(
            f"File not found: {source}\n  (It isn't a URL either. Check the path, or pass an http(s) link.)"
        )
    pr = probe(p)
    return Media(path=p, source=str(p), video_id=_local_id(p), title=p.stem, **pr)


def _local_id(p: Path) -> str:
    h = hashlib.sha1(f"{p.resolve()}:{p.stat().st_size}".encode()).hexdigest()[:10]
    return f"local-{h}"


# ---------------------------------------------------------------- yt-dlp ----

def _ytdlp() -> str:
    p = shutil.which("yt-dlp")
    if not p:
        raise ClipperError(YTDLP_HELP)
    return p


def _run_ytdlp(args: list[str], *, check=True, show=False) -> subprocess.CompletedProcess:
    """Single seam for tests: all yt-dlp subprocess calls go through here.
    show=True lets yt-dlp print its own progress to the terminal (used for long downloads)."""
    if show:
        return subprocess.run([_ytdlp(), *args], text=True, check=check)
    return subprocess.run([_ytdlp(), *args], capture_output=True, text=True, check=check)


def check_not_live(info: dict) -> None:
    state = info.get("live_status")
    if info.get("is_live") or state in LIVE_STATES:
        raise ClipperError(
            "This stream is live (or hasn't ended / been processed yet), so it can't be clipped.\n"
            "  Wait until the broadcast has ended and the VOD is available, then run clipper again."
        )


def fetch_info(url: str) -> dict:
    try:
        cp = _run_ytdlp(["-J", "--no-playlist", url])
    except subprocess.CalledProcessError as e:
        msg = (e.stderr or "").strip().splitlines()
        raise ClipperError(f"yt-dlp could not read that URL: {msg[-1] if msg else 'unknown error'}") from None
    try:
        return json.loads(cp.stdout)
    except json.JSONDecodeError:
        raise ClipperError("yt-dlp returned unreadable metadata for that URL.") from None


def parse_chat(path: Path) -> list[float]:
    """Parse a yt-dlp live_chat.json (JSON lines) into message times in seconds."""
    times: list[float] = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue
        act = obj.get("replayChatItemAction")
        if not act:
            continue
        try:
            times.append(int(act["videoOffsetTimeMsec"]) / 1000.0)
        except (KeyError, ValueError, TypeError):
            continue
    return sorted(times)


def _resolve_url(url: str, rng, cfg: dict, log, chat: bool) -> Media:
    _ytdlp()  # fail early with instructions
    log("Fetching video info (can take a minute for very long streams)...")
    info = fetch_info(url)
    check_not_live(info)
    vid = re.sub(r"[^A-Za-z0-9_.-]", "_", str(info.get("id") or hashlib.sha1(url.encode()).hexdigest()[:10]))
    d = Path(cfg["cache_dir"]) / vid
    d.mkdir(parents=True, exist_ok=True)
    (d / "info.json").write_text(json.dumps(info), encoding="utf-8")

    tag = f"{int(rng[0])}-{int(rng[1])}" if rng else "full"
    target = d / f"video_{tag}.mp4"
    if not target.is_file():
        log(f"Downloading {'minutes %d-%d' % (rng[0] // 60, rng[1] // 60) if rng else 'the full video'} (<= {cfg['render']['max_height_download']}p). yt-dlp progress follows...")
        h = int(cfg["render"]["max_height_download"])
        args = ["-f", f"bv*[height<={h}]+ba/b[height<={h}]/b", "--merge-output-format", "mp4",
                "--no-playlist", "-o", str(target.with_suffix(".%(ext)s"))]
        if rng:
            args += ["--download-sections", f"*{rng[0]}-{rng[1]}", "--force-keyframes-at-cuts"]
        try:
            _run_ytdlp([*args, url], show=True)
        except subprocess.CalledProcessError:
            raise ClipperError("Download failed (see the yt-dlp messages above).") from None
        if not target.is_file():
            raise ClipperError("Download finished but the video file wasn't found in the cache.")

    chat_times = _get_chat(d, url, log, chat)

    pr = probe(target)
    return Media(
        path=target, offset=rng[0] if rng else 0.0, video_id=vid,
        title=info.get("title") or "", description=info.get("description") or "",
        chapters=info.get("chapters") or [], heatmap=info.get("heatmap") or [],
        chat_times=chat_times, source=url, **pr,
    )


def _get_chat(d: Path, url: str, log, chat: bool) -> list[float]:
    """Live-chat replay message times (seconds), cached; soft-fails to []."""
    chat_file = next(d.glob("chat*.live_chat.json"), None)
    if chat_file is None and chat:
        log("Fetching chat replay (skip with --no-chat; can take several minutes for very long streams)...")
        try:
            _run_ytdlp(["--skip-download", "--write-subs", "--sub-langs", "live_chat",
                        "--no-playlist", "-o", str(d / "chat"), url], check=False)
        except ClipperError:
            pass
        chat_file = next(d.glob("chat*.live_chat.json"), None)
    return parse_chat(chat_file) if chat_file else []


def _cache_dir(cfg: dict, info: dict, url: str) -> tuple[str, Path]:
    vid = re.sub(r"[^A-Za-z0-9_.-]", "_", str(info.get("id") or hashlib.sha1(url.encode()).hexdigest()[:10]))
    d = Path(cfg["cache_dir"]) / vid
    d.mkdir(parents=True, exist_ok=True)
    (d / "info.json").write_text(json.dumps(info), encoding="utf-8")
    return vid, d


def resolve_scan(source: str, cfg: dict, log, chat: bool = True) -> Media:
    """Scan mode: for a URL, download AUDIO ONLY for the whole stream (small) plus metadata and chat replay.
    Full-quality video is fetched later, only for the chosen moments. Local files are used as they are."""
    if not is_url(source):
        return resolve(source, None, cfg, log=log, chat=chat)
    _ytdlp()
    log("Fetching video info (can take a minute for very long streams)...")
    info = fetch_info(source)
    check_not_live(info)
    vid, d = _cache_dir(cfg, info, source)
    audio = next((p for p in d.glob("audio.*") if p.suffix not in (".part", ".ytdl", ".json")), None)
    if audio is None:
        log("Downloading the audio track of the whole stream for scanning (yt-dlp progress follows)...")
        try:
            _run_ytdlp(["-f", "ba/b", "--no-playlist", "-o", str(d / "audio.%(ext)s"), source], show=True)
        except subprocess.CalledProcessError:
            raise ClipperError("Audio download failed (see the yt-dlp messages above).") from None
        audio = next((p for p in d.glob("audio.*") if p.suffix not in (".part", ".ytdl", ".json")), None)
        if audio is None:
            raise ClipperError("Audio download finished but the file wasn't found in the cache.")
    chat_times = _get_chat(d, source, log, chat)
    pr = probe(audio, need_video=False)
    return Media(path=audio, offset=0.0, video_id=vid, title=info.get("title") or "",
                 description=info.get("description") or "", chapters=info.get("chapters") or [],
                 heatmap=info.get("heatmap") or [], chat_times=chat_times, source=source, **pr)


def download_section(source: str, video_id: str, start: float, end: float, cfg: dict, log) -> Media:
    """Download one moment at full quality (<= max height), re-cut exactly; cached."""
    d = Path(cfg["cache_dir"]) / video_id
    s, e = max(0.0, start - 1.0), end + 1.0
    target = d / f"sec_{int(s * 10)}-{int(e * 10)}.mp4"
    if not target.is_file():
        h = int(cfg["render"]["max_height_download"])
        fmt = f"bv*[height<={h}][vcodec^=avc1]+ba/bv*[height<={h}]+ba/b[height<={h}]/b"
        log(f"  downloading {s:.0f}-{e:.0f}s at <= {h}p ...")
        try:
            _run_ytdlp(["-f", fmt, "--merge-output-format", "mp4", "--no-playlist",
                        "--download-sections", f"*{s:.1f}-{e:.1f}", "--force-keyframes-at-cuts",
                        "-o", str(target.with_suffix(".%(ext)s")), source], show=True)
        except subprocess.CalledProcessError:
            raise ClipperError("Section download failed (see the yt-dlp messages above).") from None
        if not target.is_file():
            raise ClipperError("Section download finished but the file wasn't found in the cache.")
    return Media(path=target, offset=s, video_id=video_id, source=source, **probe(target))
