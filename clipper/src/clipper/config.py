from __future__ import annotations

import copy
from pathlib import Path

from . import minyaml
from .errors import ClipperError

DEFAULTS: dict = {
    "weights": {"audio_rms": 1.0, "audio_rise": 1.0, "motion": 1.0, "heatmap": 1.5, "chat": 1.5},
    "candidates": {"count": 15, "top_fraction": 0.2, "mean_weight": 0.5, "stride_s": 1.0},
    "signals": {"rise_lookback_s": 4.0, "chat_baseline_s": 30.0},
    "render": {
        "width": 1080, "height": 1920, "fps": 30, "crf": 21, "audio_bitrate": "160k",
        "loudness_lufs": -14, "hook_seconds": 2.6, "hook_fontsize": 88,
        "hook_wrap_chars": 16, "max_height_download": 1080,
        "layout": "blur", "facecam": None, "gameplay": None, "split_top_px": 900,
        "title_style": "box", "title_fontsize": 54, "title_wrap_chars": 28, "title_seconds": 0,
    },
    "captions": {"model": "base.en", "language": "en", "chunk_words": 2, "fontsize": 96,
                 "margin_v": 430, "highlight_bgr": "00D7FF"},
    "cache_dir": ".clipper-cache",
}


def _merge(base: dict, over: dict) -> dict:
    for k, v in over.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            _merge(base[k], v)
        else:
            base[k] = v
    return base


def load(path: str | Path | None = None) -> dict:
    cfg = copy.deepcopy(DEFAULTS)
    candidate = Path(path) if path else Path("config.yaml")
    if path and not candidate.is_file():
        raise ClipperError(f"Config file not found: {candidate}")
    if candidate.is_file():
        _merge(cfg, minyaml.loads(candidate.read_text(encoding="utf-8")))
    return cfg
