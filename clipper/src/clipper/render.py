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


def _esc_path(p: str) -> str:
    """Escape a filesystem path for use inside an ffmpeg filter option value."""
    p = p.replace("\\", "/")
    return p.replace(":", r"\:").replace("'", r"\'")


def build_filter(media: Media, cfg: dict, textfile: str | None, fill: bool) -> str:
    r = cfg["render"]
    W, H, fps = int(r["width"]), int(r["height"]), int(r["fps"])
    portrait = media.height > media.width
    if portrait or fill:
        chain = f"[0:v]scale={W}:{H}:force_original_aspect_ratio=increase,crop={W}:{H},setsar=1"
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
    if textfile:
        font = find_bold_font()
        dt = [f"textfile='{_esc_path(textfile)}'"]
        if font:
            dt.append(f"fontfile='{_esc_path(font)}'")
        size = int(r["hook_fontsize"])
        dt += [f"fontsize={size}", "fontcolor=white", "borderw=7", "bordercolor=black",
               "line_spacing=12", "x=(w-text_w)/2", f"y=h*0.10",
               f"enable='between(t,0,{float(r['hook_seconds'])})'"]
        chain += ",drawtext=" + ":".join(dt)
    chain += ",format=yuv420p[v]"
    return chain


def render_clip(media: Media, start_abs: float, end_abs: float, out_path: Path, hook: str,
                cfg: dict, fill: bool = False) -> None:
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
        wrapped = wrap_hook(hook, int(r["hook_wrap_chars"])) if hook else ""
        if wrapped:
            tf = Path(td) / "hook.txt"
            tf.write_text(wrapped, encoding="utf-8")
            textfile = str(tf)
        fc = build_filter(media, cfg, textfile, fill)

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
