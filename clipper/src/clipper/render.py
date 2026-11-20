"""ffmpeg rendering to 1080x1920 vertical clips."""
from __future__ import annotations

import shutil
import subprocess
import tempfile
import textwrap
from pathlib import Path

from .errors import ClipperError
from .fonts import find_bold_font
from .ingest import Media


def wrap_hook(text: str, width: int) -> str:
    text = " ".join(text.split()).upper()
    return "\n".join(textwrap.wrap(text, width=width, break_long_words=True)) or ""


def wrap_title(text: str, width: int) -> str:
    """Headline text keeps its own capitalisation (key words in CAPS are the emphasis)."""
    text = " ".join(text.split())
    return "\n".join(textwrap.wrap(text, width=width, break_long_words=True)) or ""


def _esc_path(p: str) -> str:
    """Escape a filesystem path for use inside an ffmpeg filter option value."""
    p = p.replace("\\", "/")
    return p.replace(":", r"\:").replace("'", r"\'")


def effective_layout(media: Media, cfg: dict, fill: bool) -> str:
    r = cfg["render"]
    if media.height > media.width or fill:
        return "fill"
    if r.get("layout") == "split" and r.get("facecam"):
        return "split"
    return "blur"


def _facecam(cfg: dict) -> tuple[float, float, float, float]:
    fc = cfg["render"].get("facecam")
    try:
        x, y, w, h = (float(v) for v in fc)
    except (TypeError, ValueError):
        raise ClipperError("facecam must be four numbers x,y,w,h (fractions of the frame), e.g. 0.72,0.55,0.28,0.45") from None
    if not (0 <= x < 1 and 0 <= y < 1 and 0.05 <= w <= 1 and 0.05 <= h <= 1 and x + w <= 1.001 and y + h <= 1.001):
        raise ClipperError("facecam x,y,w,h must be fractions between 0 and 1 and stay inside the frame.")
    return x, y, w, h


def title_band(cfg: dict, layout: str) -> tuple[float, float]:
    """Vertical band (fractions of the height) where the headline sits; used by QA."""
    if cfg["render"].get("title_style", "box") == "box" and layout == "split":
        t = int(cfg["render"]["split_top_px"]) / int(cfg["render"]["height"])
        return max(0.0, t - 0.12), min(1.0, t + 0.10)
    return 0.06, 0.32


def build_filter(media: Media, cfg: dict, textfile: str | None, fill: bool, ass: str | None = None) -> str:
    r = cfg["render"]
    W, H, fps = int(r["width"]), int(r["height"]), int(r["fps"])
    layout = effective_layout(media, cfg, fill)
    top_h = int(r["split_top_px"])
    if layout == "fill":
        chain = f"[0:v]scale={W}:{H}:force_original_aspect_ratio=increase,crop={W}:{H},setsar=1"
    elif layout == "split":
        x, y, w, h = _facecam(cfg)
        bot_h = H - top_h
        chain = (
            f"[0:v]split=2[camsrc][gamesrc];"
            f"[camsrc]crop=w=trunc(iw*{w}/2)*2:h=trunc(ih*{h}/2)*2:x=iw*{x}:y=ih*{y},"
            f"scale={W}:{top_h}:force_original_aspect_ratio=increase,crop={W}:{top_h},setsar=1[cam];"
            f"[gamesrc]scale={W}:{bot_h}:force_original_aspect_ratio=increase,crop={W}:{bot_h},setsar=1[game];"
            f"[cam][game]vstack=inputs=2"
        )
    else:
        # Blurred full-frame background, sharp gameplay centred on top.
        # Blur is computed on a small copy and scaled up (fast, smooth).
        chain = (
            f"[0:v]split=2[bgsrc][fgsrc];"
            f"[bgsrc]scale={W}:{H}:force_original_aspect_ratio=increase,crop={W}:{H},"
            f"scale={W // 8}:{H // 8},boxblur=6:2,scale={W}:{H},setsar=1[bg];"
            f"[fgsrc]scale={W}:-2,setsar=1[fg];"
            f"[bg][fg]overlay=(W-w)/2:(H-h)/2"
        )
    chain += f",fps={fps}"
    if ass:
        font = find_bold_font()
        fd = f":fontsdir='{_esc_path(str(Path(font).parent))}'" if font else ""
        chain += f",subtitles=filename='{_esc_path(ass)}'{fd}"
    if textfile:
        box = r.get("title_style", "box") == "box"
        font = find_bold_font(prefer_title=box)
        dt = [f"textfile='{_esc_path(textfile)}'"]
        if font:
            dt.append(f"fontfile='{_esc_path(font)}'")
        if box:   # headline style: white box, black bold text, centred on the seam of the split (or near the top)
            y = f"{top_h}-text_h/2" if layout == "split" else "h*0.12"
            dt += [f"fontsize={int(r['title_fontsize'])}", "fontcolor=black", "box=1", "boxcolor=white@0.96",
                   "boxborderw=26", "line_spacing=10", "x=(w-text_w)/2", f"y={y}"]
            secs = float(r.get("title_seconds", 0) or 0)
            if secs > 0:
                dt.append(f"enable='between(t,0,{secs})'")
        else:
            dt += [f"fontsize={int(r['hook_fontsize'])}", "fontcolor=white", "borderw=7", "bordercolor=black",
                   "line_spacing=12", "x=(w-text_w)/2", "y=h*0.10",
                   f"enable='between(t,0,{float(r['hook_seconds'])})'"]
        chain += ",drawtext=" + ":".join(dt)
    chain += ",format=yuv420p[v]"
    return chain


def render_clip(media: Media, start_abs: float, end_abs: float, out_path: Path, hook: str,
                cfg: dict, fill: bool = False, caption_words: list | None = None) -> None:
    ff = shutil.which("ffmpeg")
    if not ff:
        raise ClipperError("ffmpeg was not found on PATH.")
    r = cfg["render"]
    start = max(0.0, start_abs - media.offset)
    end = min(media.duration, end_abs - media.offset)
    dur = end - start
    if dur <= 0.2:
        raise ClipperError("Clip would be empty; check --range and --window.")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="clipper_") as td:
        textfile = None
        if r.get("title_style", "box") == "box":
            wrapped = wrap_title(hook, int(r["title_wrap_chars"])) if hook else ""
        else:
            wrapped = wrap_hook(hook, int(r["hook_wrap_chars"])) if hook else ""
        if wrapped:
            tf = Path(td) / "hook.txt"
            tf.write_text(wrapped, encoding="utf-8")
            textfile = str(tf)
        ass = None
        if caption_words:
            from .captions import build_ass
            af = Path(td) / "caps.ass"
            af.write_text(build_ass(caption_words, cfg), encoding="utf-8")
            ass = str(af)
        fc = build_filter(media, cfg, textfile, fill, ass)

        cmd = [ff, "-y", "-v", "error", "-ss", f"{start:.3f}", "-t", f"{dur:.3f}", "-i", str(media.path)]
        if not media.has_audio:
            cmd += ["-f", "lavfi", "-t", f"{dur:.3f}", "-i", "anullsrc=r=48000:cl=stereo"]
        cmd += ["-filter_complex", fc, "-map", "[v]", "-map", "0:a:0" if media.has_audio else "1:a:0"]
        if media.has_audio:
            cmd += ["-af", f"loudnorm=I={r['loudness_lufs']}:TP=-1.5:LRA=11"]
        cmd += ["-c:v", "libx264", "-crf", str(r["crf"]), "-preset", "medium", "-pix_fmt", "yuv420p",
                "-r", str(r["fps"]), "-c:a", "aac", "-b:a", str(r["audio_bitrate"]), "-ar", "48000",
                "-ac", "2", "-movflags", "+faststart", "-t", f"{dur:.3f}", str(out_path)]
        try:
            subprocess.run(cmd, capture_output=True, text=True, check=True)
        except subprocess.CalledProcessError as e:
            raise ClipperError(f"ffmpeg failed while rendering {out_path.name}:\n{e.stderr.strip()[-600:]}") from None
