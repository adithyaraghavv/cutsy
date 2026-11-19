"""Sliding-window scoring and greedy non-overlapping selection."""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

from .signals import BIN, Signals


@dataclass
class Candidate:
    start_s: float          # absolute source seconds
    end_s: float
    score: float
    breakdown: dict         # signal -> mean z in window


def window_scores(c: np.ndarray, wb: int, stride: int, top_frac: float, mean_w: float):
    """Return (start_bin_indices, scores) for every window of wb bins."""
    n = len(c)
    if n <= wb:
        k = max(1, math.ceil(top_frac * n))
        top = np.partition(c, n - k)[n - k:].mean()
        return np.array([0]), np.array([top + mean_w * c.mean()])
    starts = np.arange(0, n - wb + 1, stride)
    k = max(1, math.ceil(top_frac * wb))
    view = sliding_window_view(c, wb)[::stride]
    scores = np.empty(len(starts))
    step = 4096
    for i in range(0, len(starts), step):
        blk = view[i:i + step]
        top = np.partition(blk, wb - k, axis=1)[:, wb - k:].mean(axis=1)
        scores[i:i + step] = top + mean_w * blk.mean(axis=1)
    return starts, scores


def pick(sig: Signals, window_s: float, cfg: dict, count: int | None = None) -> list[Candidate]:
    cc = cfg["candidates"]
    count = count or int(cc["count"])
    wb = max(1, int(round(window_s / BIN)))
    stride = max(1, int(round(float(cc["stride_s"]) / BIN)))
    starts, scores = window_scores(sig.combined, wb, stride, float(cc["top_fraction"]), float(cc["mean_weight"]))
    eff_wb = min(wb, sig.n)

    cum = {k: np.concatenate([[0.0], np.cumsum(v)]) for k, v in sig.z.items()}
    chosen: list[Candidate] = []
    taken = np.zeros(sig.n, dtype=bool)
    for idx in np.argsort(-scores, kind="stable"):
        s = int(starts[idx])
        e = s + eff_wb
        if taken[s:e].any():
            continue
        taken[s:e] = True
        chosen.append(Candidate(
            start_s=sig.t0 + s * BIN,
            end_s=sig.t0 + e * BIN,
            score=float(scores[idx]),
            breakdown={k: float((v[e] - v[s]) / (e - s)) for k, v in cum.items()},
        ))
        if len(chosen) >= count:
            break
    return chosen
