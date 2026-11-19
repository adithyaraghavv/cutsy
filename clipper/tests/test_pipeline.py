import json
from pathlib import Path

import pytest

from clipper import candidates, ingest, pipeline, signals
from conftest import probe_json


def _analyze(path, cfg):
    media = ingest.resolve(str(path), None, cfg)
    return media, signals.analyze(media, cfg)


@pytest.mark.parametrize("fx", ["landscape", "portrait"])
def test_loud_moving_section_is_picked(request, cfg, fx):
    path = request.getfixturevalue(fx)
    media, sig = _analyze(path, cfg)
    top = candidates.pick(sig, 12, cfg)[0]
    # burst is 30-45 s; a 12 s window must sit inside it (small tolerance)
    assert top.start_s >= 29 and top.end_s <= 46, (top.start_s, top.end_s)
    assert top.breakdown["audio_rms"] > 0.5 and top.breakdown["motion"] > 0.5


def test_candidates_do_not_overlap(landscape, cfg):
    _, sig = _analyze(landscape, cfg)
    cs = candidates.pick(sig, 10, cfg)
    assert 1 < len(cs) <= 15
    spans = sorted((c.start_s, c.end_s) for c in cs)
    assert all(a[1] <= b[0] + 1e-9 for a, b in zip(spans, spans[1:]))


def test_video_shorter_than_window(landscape, cfg):
    _, sig = _analyze(landscape, cfg)
    cs = candidates.pick(sig, 500, cfg)
    assert len(cs) == 1 and cs[0].start_s == 0 and cs[0].end_s == pytest.approx(60, abs=0.6)


def test_range_limits_analysis_with_absolute_times(landscape, cfg):
    media = ingest.resolve(str(landscape), None, cfg)
    sig = signals.analyze(media, cfg, (20, 50))
    assert sig.t0 == 20 and sig.n == 60
    top = candidates.pick(sig, 10, cfg)[0]
    assert 29 <= top.start_s and top.end_s <= 46


def test_heatmap_and_chat_signals_influence_score(landscape, cfg):
    media = ingest.resolve(str(landscape), None, cfg)
    # heatmap + chat both say 5-15 s is hot -> should beat nothing but must be present and shift picks
    media.heatmap = [{"start_time": s, "end_time": s + 5, "value": 1.0 if 5 <= s < 15 else 0.05}
                     for s in range(0, 60, 5)]
    media.chat_times = [5 + i * 0.1 for i in range(100)]
    sig = signals.analyze(media, cfg)
    assert {"heatmap", "chat"} <= set(sig.z)
    assert sig.z["heatmap"][20:30].mean() > sig.z["heatmap"][100:].mean()
    assert sig.z["chat"][10:30].mean() > 0


@pytest.mark.parametrize("fx,fill", [("landscape", False), ("landscape", True), ("portrait", False)])
def test_end_to_end_output_format(request, tmp_path, cfg, fx, fill):
    path = request.getfixturevalue(fx)
    out = tmp_path / "out"
    res = pipeline.run(str(path), n=2, window=10, rng=None, fill=fill, out=out, cfg=cfg, log=lambda *_: None)
    assert [c["file"] for c in res["clips"]] == ["clip_1.mp4", "clip_2.mp4"]
    data = json.loads((out / "clips.json").read_text())
    c0 = data["clips"][0]
    assert {"file", "start_s", "end_s", "score", "hook", "signals"} <= set(c0)
    for c in data["clips"]:
        info = probe_json(out / c["file"])
        v = next(s for s in info["streams"] if s["codec_type"] == "video")
        assert (v["width"], v["height"]) == (1080, 1920)
        assert v["pix_fmt"] == "yuv420p" and v["r_frame_rate"] == "30/1" and v["codec_name"] == "h264"
        assert any(s["codec_type"] == "audio" and s["codec_name"] == "aac" for s in info["streams"])


def test_no_audio_source_still_has_audio_track(tmp_path, cfg):
    import subprocess
    src = tmp_path / "silent.mp4"
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-f", "lavfi", "-i", "testsrc=s=320x180:r=25:d=8",
                    "-pix_fmt", "yuv420p", str(src)], check=True)
    out = tmp_path / "o"
    pipeline.run(str(src), n=1, window=5, rng=None, fill=False, out=out, cfg=cfg, log=lambda *_: None)
    info = probe_json(out / "clip_1.mp4")
    assert any(s["codec_type"] == "audio" for s in info["streams"])
