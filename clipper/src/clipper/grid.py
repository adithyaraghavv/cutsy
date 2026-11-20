"""--grid-frame: one frame with a labelled 10% grid, to read off facecam coordinates."""
from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

from .errors import ClipperError
from .fonts import find_bold_font
from .ingest import is_url, probe
from .render import _esc_path


def write(source: str, t: float, out: Path) -> Path:
    if is_url(source):
        raise ClipperError("--grid-frame needs a LOCAL video file. After a run, use a .mp4 from the cache folder "
                           "(.clipper-cache\\<video id>\\), or any video file.")
    p = Path(source).expanduser()
    if not p.is_file():
        raise ClipperError(f"File not found: {source}")
    ff = shutil.which("ffmpeg")
    if not ff:
        raise ClipperError("ffmpeg was not found on PATH.")
    out.mkdir(parents=True, exist_ok=True)
    font = find_bold_font()
    ff_font = f"fontfile='{_esc_path(font)}':" if font else ""
    size = max(14, probe(p)["height"] // 28)   # drawtext does not accept `ih` in fontsize; no % signs (drawtext expands them)
    vf = ["drawgrid=w=iw/10:h=ih/10:t=2:c=yellow@0.8"]
    for i in range(10):
        vf.append(f"drawtext={ff_font}text='{i * 10}':x={i}*w/10+6:y=6:fontsize={size}:fontcolor=yellow:borderw=3:bordercolor=black")
    for j in range(1, 10):
        vf.append(f"drawtext={ff_font}text='{j * 10}':x=6:y={j}*h/10+6:fontsize={size}:fontcolor=cyan:borderw=3:bordercolor=black")
    dest = out / "grid.png"
    r = subprocess.run([ff, "-y", "-v", "error", "-ss", f"{t}", "-i", str(p), "-frames:v", "1", "-vf", ",".join(vf), str(dest)],
                       capture_output=True, text=True)
    if r.returncode != 0 or not dest.is_file():
        raise ClipperError(f"Could not make the grid frame: {r.stderr.strip()[-200:]}")
    return dest
