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


def _rect(cfg: dict, key: str = "facecam") -> tuple[float, float, float, float]:
    fc = cfg["render"].get(key)
    try:
        x, y, w, h = (float(v) for v in fc)
    except (TypeError, ValueError):
        raise ClipperError(f"{key} must be four numbers x,y,w,h (fractions of the frame), e.g. 0.72,0.55,0.28,0.45") from None
    if not (0 <= x < 1 and 0 <= y < 1 and 0.05 <= w <= 1 and 0.05 <= h <= 1 and x + w <= 1.001 and y + h <= 1.001):
        raise ClipperError(f"{key} x,y,w,h must be fractions between 0 and 1 and stay inside the frame.")
    return x, y, w, h


def title_band(cfg: dict, layout: str) -> tuple[float, float]:
    """Vertical band (fractions of the height) where the headline sits; used by QA."""
    if cfg["render"].get("title_style", "box") == "box" and layout == "split":
        t = int(cfg["render"]["split_top_px"]) / int(cfg["render"]["height"])
        return max(0.0, t - 0.12), min(1.0, t + 0.10)
    return 0.06, 0.32



def _measure(ff: str, font: str | None, size: int, line_file: str) -> tuple[int, int, int, int] | None:
    """Ink bounding box (x1, x2, y1, y2) of one line drawn at (100, 100): measured with ffmpeg's bbox filter so
    it is exact for whatever font/ffmpeg is installed."""
    import re
    dt = f"drawtext=textfile='{_esc_path(line_file)}':fontsize={size}:fontcolor=white:x=100:y=100"
    if font:
        dt += f":fontfile='{_esc_path(font)}'"
    r = subprocess.run([ff, "-v", "info", "-f", "lavfi", "-i", "color=c=black:s=1900x320:d=1", "-vf", f"{dt},bbox",
                        "-frames:v", "1", "-f", "null", "-"], capture_output=True, text=True)
    m = re.search(r"x1:(-?\d+) x2:(-?\d+) y1:(-?\d+) y2:(-?\d+)", r.stderr)
    return tuple(int(v) for v in m.groups()) if m else None


def plan_title(ff: str, cfg: dict, text: str, layout: str, td: Path) -> dict | None:
    """Layout of the headline: one white rounded box (a PNG) with every line of black bold text centred in it.
    The box is built from measured text sizes, so it does not depend on how a given ffmpeg draws text boxes."""
    r = cfg["render"]
    W, H = int(r["width"]), int(r["height"])
    fs = int(r["title_fontsize"])
    lines = wrap_title(text, int(r["title_wrap_chars"])).split("\n") if text else []
    lines = [l for l in lines if l.strip()]
    if not lines:
        return None
    font = find_bold_font(prefer_title=True)
    pitch, padx, pady, rad = int(fs * 1.28), 38, 26, 26
    ms = []
    for i, ln in enumerate(lines):
        lf = td / f"line{i}.txt"
        lf.write_text(ln, encoding="utf-8")
        m = _measure(ff, font, fs, str(lf))
        if m is None:   # could not measure: estimate (still centred, never crashes)
            m = (100, 100 + int(len(ln) * fs * 0.55), 100, 100 + int(fs * 0.95))
        ms.append((str(lf), m))
    inkw = [m[1] - m[0] + 1 for _, m in ms]
    bw = min(max(inkw) + 2 * padx, W - 40)
    top_ink, bottom_ink = ms[0][1][2] - 100, ms[-1][1][3] - 100          # first/last line ink offsets from its origin
    bh = (len(lines) - 1) * pitch + (bottom_ink - top_ink + 1) + 2 * pady
    xc = W // 2
    yc = int(r["split_top_px"]) if layout == "split" else int(H * 0.16)
    bx, by = xc - bw // 2, max(10, min(yc - bh // 2, H - bh - 10))
    y0 = by + pady - top_ink                                            # y origin of the first line's text
    out_lines = []
    for i, (lf, m) in enumerate(ms):
        out_lines.append({"file": lf, "x": int(round(xc - inkw[i] / 2 - (m[0] - 100))), "y": y0 + i * pitch})
    png = td / "titlebox.png"
    ex = (f"255*0.96*clip({rad}+0.5-sqrt(pow(max(max({rad}-X,X-({bw}-1-{rad})),0),2)+"
          f"pow(max(max({rad}-Y,Y-({bh}-1-{rad})),0),2)),0,1)")
    rr = subprocess.run([ff, "-y", "-v", "error", "-f", "lavfi", "-i", f"color=c=white:s={bw}x{bh}:d=1", "-vf",
                         f"format=rgba,geq=r=255:g=255:b=255:a='{ex}'", "-frames:v", "1", str(png)],
                        capture_output=True, text=True)
    if rr.returncode != 0 or not png.is_file():
        raise ClipperError(f"Could not build the title box: {rr.stderr.strip()[-200:]}")
    return {"png": str(png), "bx": bx, "by": by, "bw": bw, "bh": bh, "lines": out_lines, "font": font, "fs": fs}


def build_filter(media: Media, cfg: dict, textfile: str | None, fill: bool, ass: str | None = None,
                 title_plan: dict | None = None, box_input: int = 1) -> str:
    r = cfg["render"]
    W, H, fps = int(r["width"]), int(r["height"]), int(r["fps"])
    layout = effective_layout(media, cfg, fill)
    top_h = int(r["split_top_px"])
    if layout == "fill":
        chain = f"[0:v]scale={W}:{H}:force_original_aspect_ratio=increase,crop={W}:{H},setsar=1"
    elif layout == "split":
        x, y, w, h = _rect(cfg, "facecam")
        bot_h = H - top_h
        if r.get("gameplay"):
            gx, gy, gw, gh = _rect(cfg, "gameplay")
            game_crop = f"crop=w=trunc(iw*{gw}/2)*2:h=trunc(ih*{gh}/2)*2:x=iw*{gx}:y=ih*{gy},"
        else:
            game_crop = ""
        chain = (
            f"[0:v]split=2[camsrc][gamesrc];"
            f"[camsrc]crop=w=trunc(iw*{w}/2)*2:h=trunc(ih*{h}/2)*2:x=iw*{x}:y=ih*{y},"
            f"scale={W}:{top_h}:force_original_aspect_ratio=increase,crop={W}:{top_h},setsar=1[cam];"
            f"[gamesrc]{game_crop}scale={W}:{bot_h}:force_original_aspect_ratio=increase,crop={W}:{bot_h},setsar=1[game];"
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
    if title_plan:
        secs = float(r.get("title_seconds", 0) or 0)
        en = f":enable='between(t,0,{secs})'" if secs > 0 else ""
        chain += f"[b0];[b0][{box_input}:v]overlay=x={title_plan['bx']}:y={title_plan['by']}:format=auto{en}"
        for ln in title_plan["lines"]:
            dt = [f"textfile='{_esc_path(ln['file'])}'"]
            if title_plan["font"]:
                dt.append(f"fontfile='{_esc_path(title_plan['font'])}'")
            dt += [f"fontsize={title_plan['fs']}", "fontcolor=black", f"x={ln['x']}", f"y={ln['y']}"]
            chain += ",drawtext=" + ":".join(dt) + en
    elif textfile:   # legacy outline style
        font = find_bold_font()
        dt = [f"textfile='{_esc_path(textfile)}'"]
        if font:
            dt.append(f"fontfile='{_esc_path(font)}'")
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
        plan = None
        if r.get("title_style", "box") == "box":
            if hook:
                plan = plan_title(ff, cfg, hook, effective_layout(media, cfg, fill), Path(td))
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
        box_input = 1 if media.has_audio else 2
        fc = build_filter(media, cfg, textfile, fill, ass, plan, box_input)

        cmd = [ff, "-y", "-v", "error", "-ss", f"{start:.3f}", "-t", f"{dur:.3f}", "-i", str(media.path)]
        if not media.has_audio:
            cmd += ["-f", "lavfi", "-t", f"{dur:.3f}", "-i", "anullsrc=r=48000:cl=stereo"]
        if plan:
            cmd += ["-loop", "1", "-framerate", str(r["fps"]), "-t", f"{dur:.3f}", "-i", plan["png"]]
        cmd += ["-filter_complex", fc, "-map", "[v]", "-map", "0:a:0" if media.has_audio else "1:a:0"]
        if media.has_audio:
            cmd += ["-af", f"loudnorm=I={r['loudness_lufs']}:TP=-1.5:LRA=11"]
        cmd += ["-c:v", "libx264", "-crf", str(r["crf"]), "-preset", str(r.get("preset", "veryfast")), "-pix_fmt", "yuv420p",
                "-r", str(r["fps"]), "-c:a", "aac", "-b:a", str(r["audio_bitrate"]), "-ar", "48000",
                "-ac", "2", "-movflags", "+faststart", "-t", f"{dur:.3f}", str(out_path)]
        try:
            subprocess.run(cmd, capture_output=True, text=True, check=True)
        except subprocess.CalledProcessError as e:
            raise ClipperError(f"ffmpeg failed while rendering {out_path.name}:\n{e.stderr.strip()[-600:]}") from None
