"""Per-0.5s signals: audio RMS + rise, motion, replay heatmap, chat spikes."""
from __future__ import annotations

import math
import shutil
import subprocess
from dataclasses import dataclass

import numpy as np

from .errors import ClipperError
from .ingest import Media

BIN = 0.5
AUDIO_SR = 16000
MW, MH = 64, 36


@dataclass
class Signals:
    t0: float                      # absolute source time of bin 0
    n: int
    raw: dict                      # name -> raw array (length n)
    z: dict                        # name -> z-scored array
    combined: np.ndarray


def zscore(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    s = x.std()
    if s < 1e-12:
        return np.zeros_like(x)
    return (x - x.mean()) / s


def _fit(x: np.ndarray, n: int) -> np.ndarray:
    if len(x) >= n:
        return x[:n]
    return np.concatenate([x, np.full(n - len(x), x[-1] if len(x) else 0.0)])


def _ffmpeg() -> str:
    p = shutil.which("ffmpeg")
    if not p:
        raise ClipperError("ffmpeg was not found on PATH.")
    return p


def audio_rms(path, start: float, dur: float, n: int) -> np.ndarray:
    spb = int(AUDIO_SR * BIN)
    cmd = [_ffmpeg(), "-v", "error", "-ss", f"{start:.3f}", "-t", f"{dur:.3f}", "-i", str(path),
           "-vn", "-ac", "1", "-ar", str(AUDIO_SR), "-f", "s16le", "-"]
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    out: list[np.ndarray] = []
    buf = b""
    chunk = spb * 2 * 256
    while True:
        data = p.stdout.read(chunk)
        if not data:
            break
        buf += data
        nb = len(buf) // (spb * 2)
        if nb:
            a = np.frombuffer(buf[: nb * spb * 2], dtype="<i2").astype(np.float32) / 32768.0
            out.append(np.sqrt((a.reshape(nb, spb) ** 2).mean(axis=1)))
            buf = buf[nb * spb * 2:]
    p.wait()
    if len(buf) >= 4:
        a = np.frombuffer(buf[: len(buf) // 2 * 2], dtype="<i2").astype(np.float32) / 32768.0
        out.append(np.array([math.sqrt(float((a ** 2).mean()))]))
    rms = np.concatenate(out) if out else np.zeros(1)
    return _fit(rms, n).astype(np.float64)


def rise(x: np.ndarray, lookback_bins: int) -> np.ndarray:
    """x minus the mean of the previous `lookback_bins` bins, clipped at 0."""
    lb = max(1, lookback_bins)
    c = np.concatenate([[0.0], np.cumsum(x)])
    idx = np.arange(len(x))
    lo = np.maximum(idx - lb, 0)
    cnt = np.maximum(idx - lo, 1)
    prev = (c[idx] - c[lo]) / cnt
    prev[0] = x[0]
    return np.maximum(0.0, x - prev)


def motion(path, start: float, dur: float, n: int) -> np.ndarray:
    fb = MW * MH
    cmd = [_ffmpeg(), "-v", "error", "-ss", f"{start:.3f}", "-t", f"{dur:.3f}", "-i", str(path),
           "-an", "-vf", f"fps=2,scale={MW}:{MH}:flags=area,format=gray", "-f", "rawvideo", "-"]
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    diffs = [0.0]
    prev = None
    while True:
        raw = p.stdout.read(fb)
        if len(raw) < fb:
            break
        f = np.frombuffer(raw, dtype=np.uint8).astype(np.float32)
        if prev is not None:
            diffs.append(float(np.abs(f - prev).mean()))
        prev = f
    p.wait()
    return _fit(np.array(diffs), n).astype(np.float64)


def heatmap_signal(heatmap: list, t0: float, n: int) -> np.ndarray | None:
    pts = [((h["start_time"] + h["end_time"]) / 2, float(h["value"])) for h in heatmap
           if "start_time" in h and "end_time" in h and "value" in h]
    if len(pts) < 2:
        return None
    pts.sort()
    xs, ys = zip(*pts)
    centers = t0 + (np.arange(n) + 0.5) * BIN
    return np.interp(centers, xs, ys)


def chat_signal(chat_times: list, t0: float, n: int, baseline_bins: int) -> np.ndarray | None:
    if not chat_times:
        return None
    t = np.asarray(chat_times, dtype=np.float64) - t0
    t = t[(t >= 0) & (t < n * BIN)]
    if len(t) == 0:
        return None
    counts = np.bincount((t / BIN).astype(int), minlength=n)[:n].astype(np.float64)
    smooth = np.convolve(counts, np.ones(4) / 4, mode="same")   # ~2 s
    return rise(smooth, baseline_bins)


def analyze(media: Media, cfg: dict, span: tuple[float, float] | None = None) -> Signals:
    """span is in *absolute source* seconds; defaults to the whole file."""
    abs0 = media.offset
    abs1 = media.offset + media.duration
    if span:
        abs0, abs1 = max(abs0, span[0]), min(abs1, span[1])
    if abs1 - abs0 < 1.0:
        raise ClipperError("The selected range is outside the video or shorter than 1 second.")
    start_in_file = abs0 - media.offset
    dur = abs1 - abs0
    n = int(math.ceil(dur / BIN))
    sc = cfg["signals"]

    raw: dict[str, np.ndarray] = {}
    if media.has_audio:
        rms = audio_rms(media.path, start_in_file, dur, n)
        raw["audio_rms"] = rms
        raw["audio_rise"] = rise(rms, int(sc["rise_lookback_s"] / BIN))
    raw["motion"] = motion(media.path, start_in_file, dur, n)
    hm = heatmap_signal(media.heatmap, abs0, n)
    if hm is not None:
        raw["heatmap"] = hm
    ch = chat_signal(media.chat_times, abs0, n, int(sc["chat_baseline_s"] / BIN))
    if ch is not None:
        raw["chat"] = ch

    z = {k: zscore(v) for k, v in raw.items()}
    w = cfg["weights"]
    present = {k: float(w.get(k, 0.0)) for k in z if float(w.get(k, 0.0)) > 0}
    if not present:
        raise ClipperError("All signal weights are zero or unavailable; check config.yaml.")
    total = sum(present.values())
    combined = sum(present[k] * z[k] for k in present) / total
    return Signals(t0=abs0, n=n, raw=raw, z=z, combined=combined)
