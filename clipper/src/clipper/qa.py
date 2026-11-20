"""Quality check of rendered clips.

heuristic - pixel checks on 4 frames (no LLM): no black bars, hook visible, not blank.
            Honest limits: it cannot judge cropping or whether the subject is visible.
chat      - writes a 2x2 contact sheet per clip + qa_prompt.md; Claude in the chat answers
            qa.json (hook readable? cropped? subject visible? black bars?) with an optional fix.
Failures are re-rendered once with the suggested fix ({"fill": bool, "hook": str}).
"""
from __future__ import annotations

import json
import shutil
import subprocess
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from .errors import ClipperError

W, H = 270, 480


@dataclass
class QAResult:
    passed: bool
    backend: str
    checks: dict = field(default_factory=dict)
    notes: list = field(default_factory=list)
    fix: dict | None = None

    def to_dict(self):
        return asdict(self)


def qa_times(duration: float, hook_s: float) -> list[float]:
    d = max(duration, 1.0)
    ts = [0.5, min(hook_s + 0.8, d - 0.2), d * 0.5, max(d - 0.7, 0.3)]
    return [round(min(max(t, 0.0), d - 0.1), 2) for t in ts]


def grab_gray(path: Path, t: float) -> np.ndarray:
    ff = shutil.which("ffmpeg")
    r = subprocess.run([ff, "-v", "error", "-ss", f"{t}", "-i", str(path), "-frames:v", "1",
                        "-vf", f"scale={W}:{H}", "-pix_fmt", "gray", "-f", "rawvideo", "-"],
                       capture_output=True)
    if len(r.stdout) < W * H:
        raise ClipperError(f"QA could not read a frame at {t}s from {Path(path).name}.")
    return np.frombuffer(r.stdout[: W * H], dtype=np.uint8).reshape(H, W).astype(np.float32)


def _bar(strip: np.ndarray) -> bool:
    # Only truly pure-black strips count. A dark game scene's blurred background is dark but never this flat/black.
    return strip.mean() < 1.5 and strip.std() < 1.0


def heuristic_qa(clip: Path, duration: float, hook_s: float, has_hook: bool, band: tuple = (0.06, 0.30),
                 persistent: bool = False) -> QAResult:
    ts = qa_times(duration, hook_s)
    fr = [grab_gray(clip, t) for t in ts]
    rb, cb = int(H * 0.06), int(W * 0.03)
    bars = 0
    for f in fr:
        centre = f[H // 4: 3 * H // 4, W // 4: 3 * W // 4]
        if centre.mean() > 15 and (_bar(f[:rb]) or _bar(f[-rb:]) or _bar(f[:, :cb]) or _bar(f[:, -cb:])):
            bars += 1
    no_bars = bars < 3
    sl = slice(int(H * band[0]), int(H * band[1]))
    early, late = (fr[0][sl] > 235).mean(), (fr[1][sl] > 235).mean()
    if persistent:   # headline box stays up the whole clip: bright box pixels must be there in every frame
        hook_ok = (not has_hook) or all((f[sl] > 235).mean() > 0.02 for f in fr)
    else:
        hook_ok = (not has_hook) or (early - late > 0.002)
    not_blank = max(f[H // 4: 3 * H // 4].mean() for f in fr) > 4
    checks = {"no_black_bars": bool(no_bars), "hook_visible": bool(hook_ok), "not_blank": bool(not_blank)}
    notes = ["cropping / subject visibility are not assessed by the heuristic QA (needs visual review)"]
    fix = {"fill": True} if not no_bars else None
    return QAResult(passed=all(checks.values()), backend="heuristic", checks=checks, notes=notes, fix=fix)


def make_sheet(clip: Path, dest: Path, duration: float, hook_s: float) -> Path:
    """4 frames of a rendered clip side by side (during the title, just after, mid, near the end)."""
    ff = shutil.which("ffmpeg")
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.parent / f"{dest.stem}_f"
    tmp.mkdir(exist_ok=True)
    for i, t in enumerate(qa_times(duration, hook_s), 1):
        subprocess.run([ff, "-y", "-v", "error", "-ss", f"{t}", "-i", str(clip), "-frames:v", "1",
                        "-vf", "scale=360:-2", "-q:v", "4", str(tmp / f"f{i}.jpg")], capture_output=True)
    subprocess.run([ff, "-y", "-v", "error", "-framerate", "1", "-i", str(tmp / "f%d.jpg"),
                    "-vf", "tile=4x1", "-frames:v", "1", "-q:v", "4", str(dest)], capture_output=True)
    shutil.rmtree(tmp, ignore_errors=True)
    return dest


def write_chat_pack(clips: list[dict], out: Path, hook_s: float) -> Path | None:
    ff = shutil.which("ffmpeg")
    qa = out / "qa"
    qa.mkdir(parents=True, exist_ok=True)
    rows = []
    for c in clips:
        clip = out / c["file"]
        dur = c["end_s"] - c["start_s"]
        tmp = qa / f"{clip.stem}_f"
        tmp.mkdir(exist_ok=True)
        for i, t in enumerate(qa_times(dur, hook_s), 1):
            subprocess.run([ff, "-y", "-v", "error", "-ss", f"{t}", "-i", str(clip), "-frames:v", "1",
                            "-vf", "scale=360:-2", "-q:v", "4", str(tmp / f"f{i}.jpg")], capture_output=True)
        subprocess.run([ff, "-y", "-v", "error", "-framerate", "1", "-i", str(tmp / "f%d.jpg"),
                        "-vf", "tile=4x1", "-frames:v", "1", "-q:v", "4", str(qa / f"{clip.stem}_sheet.jpg")],
                       capture_output=True)
        shutil.rmtree(tmp, ignore_errors=True)
        rows.append(f"- `{c['file']}`  title: \"{c.get('title') or c['hook']}\"  sheet: `qa/{clip.stem}_sheet.jpg`")
    p = out / "qa_prompt.md"
    p.write_text(f"""# Clip QA task

For each rendered clip below, look at its sheet (4 frames left-to-right: during the hook text,
just after it, mid-clip, near the end) and check:
1. the hook text is readable (big, white, not cut off, not covered)
2. nothing important is cropped
3. the subject / action is visible
4. no black bars

{chr(10).join(rows)}

Write **{out / 'qa.json'}** as a JSON list, one object per clip:

```json
[{{"file": "clip_1.mp4", "passed": true, "issues": [], "fix": null}}]
```

If it fails, give a `fix`: `{{"fill": true}}` (crop to fill 9:16) and/or `{{"hook": "shorter hook, max 6 words"}}`.
Clipper re-renders each failing clip once with your fix.
""", encoding="utf-8")
    return p


def load_chat_results(path: str | Path) -> dict[str, QAResult]:
    try:
        rows = json.loads(Path(path).read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise ClipperError(f"QA results file not found: {path}") from None
    except json.JSONDecodeError as e:
        raise ClipperError(f"{path} is not valid JSON ({e.msg}).") from None
    if isinstance(rows, dict):
        rows = rows.get("results", [])
    out = {}
    for r in rows:
        if not isinstance(r, dict) or "file" not in r or "passed" not in r:
            raise ClipperError(f"{path}: each entry needs 'file' and 'passed'.")
        fix = r.get("fix") if isinstance(r.get("fix"), dict) else None
        if fix:
            fix = {k: v for k, v in fix.items() if k in ("fill", "hook")}
            if "hook" in fix and (not isinstance(fix["hook"], str) or not 1 <= len(fix["hook"].split()) <= 30):
                raise ClipperError(f"{path}: fix.hook must be 1-30 words.")
        out[r["file"]] = QAResult(passed=bool(r["passed"]), backend="chat",
                                  notes=[str(i) for i in r.get("issues", [])], fix=fix or None)
    return out
