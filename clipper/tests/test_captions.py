import builtins
import json

import numpy as np
import pytest

from clipper import captions, pipeline
from clipper.errors import ClipperError
from conftest import probe_json
from test_phase2 import QUIET, cfg2  # noqa: F401  (fixture reuse)


def W(w, s, e):
    return {"w": w, "start": s, "end": e}


FAKE = [W("This", 30.2, 30.5), W("is", 30.5, 30.7), W("INSANE", 30.8, 31.4), W("wow", 31.5, 31.8),
        W("Look", 33.0, 33.3), W("at", 33.3, 33.4), W("that", 33.4, 33.6), W("12", 33.7, 34.0)]


def test_chunking_two_words_and_single_highlight():
    ch = captions.chunk_words(FAKE)
    assert [len(c["words"]) for c in ch] == [2, 2, 2, 2]
    assert [t for t, _ in ch[0]["words"]] == ["This", "is"] and not any(h for _, h in ch[0]["words"])
    assert [h for _, h in ch[1]["words"]] == [True, False]            # INSANE highlighted, not "wow"
    assert all(sum(h for _, h in c["words"]) <= 1 for c in ch)
    assert ch[3]["words"][1] == ("12", True)                           # numbers are key words
    assert all(c["end"] > c["start"] for c in ch)


def test_ass_contents(cfg):
    words = captions.slice_words(FAKE, 30.0, 40.0)
    assert words[0]["start"] == pytest.approx(0.2)
    ass = captions.build_ass(words, cfg)
    assert "PlayResX: 1080" in ass and "PlayResY: 1920" in ass
    assert ass.count("Dialogue:") == 4 and "THIS IS" in ass
    assert "{\\c&H00D7FF&}INSANE{\\c&HFFFFFF&}" in ass


def test_missing_faster_whisper_is_friendly(monkeypatch):
    real = builtins.__import__

    def fake(name, *a, **k):
        if name.startswith("faster_whisper"):
            raise ImportError(name)
        return real(name, *a, **k)
    monkeypatch.setattr(builtins, "__import__", fake)
    with pytest.raises(ClipperError, match="pip install faster-whisper"):
        captions.check_available()


def _px(path, t):
    import subprocess
    r = subprocess.run(["ffmpeg", "-v", "error", "-ss", str(t), "-i", str(path), "-frames:v", "1",
                        "-vf", "crop=1080:700:0:1000,scale=270:175", "-pix_fmt", "rgb24", "-f", "rawvideo", "-"],
                       capture_output=True)
    return np.frombuffer(r.stdout, np.uint8).reshape(175, 270, 3).astype(int)


def test_captions_burned_in_end_to_end(landscape, tmp_path, cfg2, monkeypatch):
    monkeypatch.setattr(captions, "check_available", lambda: None)
    calls = []

    def fake(wav, cfg):
        calls.append(1)
        return [W(w["w"], w["start"] - 30.0, w["end"] - 30.0) for w in FAKE]   # relative to window start (30.0)
    monkeypatch.setattr(captions, "_transcribe_words", fake)
    out = tmp_path / "o"
    r = pipeline.run(str(landscape), n=1, window=15, rng=None, fill=False, out=out, cfg=cfg2,
                     captions=True, log=QUIET)
    c = r["clips"][0]
    assert c["start_s"] == 30.0 and len(c["caption_words"]) == 8 and calls == [1]
    v = next(s for s in probe_json(out / "clip_1.mp4")["streams"] if s["codec_type"] == "video")
    assert (v["width"], v["height"]) == (1080, 1920)
    # INSANE at 0.8-1.4 s (highlighted gold) vs. a moment with no caption (7 s)
    on, off = _px(out / "clip_1.mp4", 1.0), _px(out / "clip_1.mp4", 7.0)
    white = lambda a: ((a > 235).all(axis=2)).sum()
    gold = lambda a: ((a[..., 0] > 200) & (a[..., 1] > 150) & (a[..., 2] < 90)).sum()
    assert white(on) > 20 + white(off) and gold(on) > 10 and gold(off) == 0
    assert "Captions:" in (out / "report.md").read_text()
    # cached transcript: a second render does not transcribe again
    pipeline.run(str(landscape), n=1, window=15, rng=None, fill=False, out=tmp_path / "o2", cfg=cfg2,
                 captions=True, log=QUIET)
    assert calls == [1]


def test_transcript_goes_into_evidence_pack(landscape, tmp_path, cfg2, monkeypatch):
    monkeypatch.setattr(captions, "check_available", lambda: None)
    monkeypatch.setattr(captions, "_transcribe_words", lambda wav, cfg: [W("hello", 1.0, 1.4), W("world", 1.5, 2.0)])
    out = tmp_path / "o"
    r = pipeline.run(str(landscape), n=1, window=15, rng=None, fill=False, out=out, cfg=cfg2,
                     judge_name="chat", shortlist=1, captions=True, log=QUIET)
    ev = json.loads((out / "evidence" / r["pending"][0] / "evidence.json").read_text())
    assert "hello world" in ev["transcript"]
