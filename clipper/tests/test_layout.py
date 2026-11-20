import json
import subprocess

import numpy as np
import pytest

from clipper import cli, judge, pipeline, render
from clipper.errors import ClipperError
from conftest import probe_json

QUIET = lambda *_: None
HEADLINE = "He is OFFICIALLY the FIRST person to COMPLETE the entire game and it only took him seven hours"


@pytest.fixture(scope="module")
def camvideo(tmp_path_factory):
    """16:9 source: blue 'gameplay', with a red 'facecam' rectangle at the bottom-right (x .70-1.0, y .55-1.0)."""
    p = tmp_path_factory.mktemp("cam") / "cam.mp4"
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-f", "lavfi", "-i",
                    "color=c=0x1030c0:s=640x360:r=25:d=14,drawbox=x=448:y=198:w=192:h=162:color=red:t=fill",
                    "-f", "lavfi", "-i", "sine=f=440:d=14", "-pix_fmt", "yuv420p", "-shortest", str(p)], check=True)
    return p


@pytest.fixture
def cfg2(cfg, tmp_path):
    cfg["cache_dir"] = str(tmp_path / "cache")
    return cfg


def frame(path, t):
    r = subprocess.run(["ffmpeg", "-v", "error", "-ss", str(t), "-i", str(path), "-frames:v", "1",
                        "-vf", "scale=270:480", "-pix_fmt", "rgb24", "-f", "rawvideo", "-"], capture_output=True)
    return np.frombuffer(r.stdout, np.uint8).reshape(480, 270, 3).astype(int)


def is_red(a):
    return ((a[..., 0] > 180) & (a[..., 1] < 80) & (a[..., 2] < 80)).mean()


def is_white(a):
    return ((a > 235).all(axis=2)).mean()


def test_split_layout_facecam_on_top_gameplay_below_headline_on_seam(camvideo, tmp_path, cfg2):
    cfg2["render"].update(layout="split", facecam=[0.70, 0.55, 0.30, 0.45])
    out = tmp_path / "o"
    r = pipeline.run(str(camvideo), n=1, window=10, rng=None, fill=False, out=out, cfg=cfg2, log=QUIET)
    clip = out / r["clips"][0]["file"]
    v = next(s for s in probe_json(clip)["streams"] if s["codec_type"] == "video")
    assert (v["width"], v["height"]) == (1080, 1920)
    f = frame(clip, 3)
    top, bottom = f[:int(480 * 0.40)], f[int(480 * 0.60):]
    assert is_red(top) > 0.8                      # top panel is the facecam crop (red)
    assert is_red(bottom) < 0.2 and ((bottom[..., 2] > 150) & (bottom[..., 0] < 60)).mean() > 0.7   # bottom = gameplay (blue)
    seam = f[int(480 * 0.40):int(480 * 0.60)]
    assert is_white(seam) > 0.03                  # white headline box sits on the seam
    for t in (0.5, 8.0):                          # and stays for the whole clip
        assert is_white(frame(clip, t)[int(480 * 0.40):int(480 * 0.60)]) > 0.03
    assert r["clips"][0]["qa"][0]["checks"]["hook_visible"] is True


def test_blur_layout_has_headline_box_near_top(camvideo, tmp_path, cfg2):
    out = tmp_path / "o"
    r = pipeline.run(str(camvideo), n=1, window=10, rng=None, fill=False, out=out, cfg=cfg2, log=QUIET)
    clip = out / r["clips"][0]["file"]
    assert is_white(frame(clip, 6)[int(480 * 0.06):int(480 * 0.32)]) > 0.02
    assert r["clips"][0]["qa"][0]["passed"]


def test_headline_title_is_drawn_and_reported(camvideo, tmp_path, cfg2, monkeypatch):
    def fake(self, items, ctx):
        out = {}
        for it in items:
            out[it.id] = judge.Judgment(id=it.id, score=8, reason="ok", best_start=it.cand.start_s,
                                        best_end=it.cand.end_s, hook="SHORT HOOK", caption="cap",
                                        hashtags=["#a", "#b", "#c", "#d", "#e"], risk_flags=[], backend="heuristic",
                                        title=HEADLINE)
        return out, []
    monkeypatch.setattr(judge.HeuristicJudge, "judge", fake)
    out = tmp_path / "o"
    r = pipeline.run(str(camvideo), n=1, window=10, rng=None, fill=False, out=out, cfg=cfg2, log=QUIET)
    c = r["clips"][0]
    assert c["title"] == HEADLINE and c["hook"] == "SHORT HOOK"
    assert HEADLINE in (out / "report.md").read_text() and HEADLINE in (out / "review.html").read_text()
    assert is_white(frame(out / c["file"], 4)[int(480 * 0.06):int(480 * 0.32)]) > 0.08   # multi-line box is taller


def test_title_validation():
    from clipper import evidence
    from clipper.candidates import Candidate
    c = Candidate(start_s=0, end_s=55, score=1, breakdown={})
    item = evidence.make_items(type("M", (), {"video_id": "v"})(), [c], 55, {})[0]
    base = {"id": item.id, "score": 8, "reason": "ok", "best_start": 0, "best_end": 50, "hook": "A B C",
            "caption": "cap", "hashtags": ["#a", "#b", "#c", "#d", "#e"], "risk_flags": []}
    assert judge.validate({**base, "title": HEADLINE}, item).title == HEADLINE
    assert judge.validate(base, item).title == ""
    with pytest.raises(ValueError, match="title"):
        judge.validate({**base, "title": " ".join(["word"] * 40)}, item)


def test_facecam_arg_validation(camvideo, tmp_path, capsys, monkeypatch):
    monkeypatch.chdir(tmp_path)
    assert cli.main([str(camvideo), "--facecam", "0.9,0.9,0.5,0.5", "--out", "o"]) == 2
    assert "facecam" in capsys.readouterr().err
    assert cli.main([str(camvideo), "--facecam", "a,b,c,d", "--out", "o"]) == 2
    assert cli.main([str(camvideo), "--layout", "split", "--out", "o"]) == 2
    assert "--facecam" in capsys.readouterr().err


def test_grid_frame_helper(camvideo, tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    assert cli.main([str(camvideo), "--grid-frame", "1", "--out", "g"]) == 0
    assert (tmp_path / "g" / "grid.png").stat().st_size > 1000
    assert cli.main(["https://youtu.be/x", "--grid-frame", "1", "--out", "g"]) == 2
    assert "LOCAL video" in capsys.readouterr().err


def test_outline_style_still_available(camvideo, tmp_path, cfg2):
    cfg2["render"]["title_style"] = "outline"
    out = tmp_path / "o"
    r = pipeline.run(str(camvideo), n=1, window=10, rng=None, fill=False, out=out, cfg=cfg2, log=QUIET)
    assert r["clips"][0]["qa"][0]["checks"]["hook_visible"] is True


def test_gameplay_region_excludes_the_facecam_corner(camvideo, tmp_path, cfg2):
    cfg2["render"].update(layout="split", facecam=[0.70, 0.55, 0.30, 0.45], gameplay=[0.0, 0.0, 0.69, 1.0])
    out = tmp_path / "o"
    r = pipeline.run(str(camvideo), n=1, window=10, rng=None, fill=False, out=out, cfg=cfg2, log=QUIET)
    f = frame(out / r["clips"][0]["file"], 3)
    assert is_red(f[:int(480 * 0.40)]) > 0.8                  # facecam still on top
    assert is_red(f[int(480 * 0.60):]) == 0                   # bottom panel: no facecam pixels at all


def test_gameplay_arg_validation(camvideo, tmp_path, capsys, monkeypatch):
    monkeypatch.chdir(tmp_path)
    assert cli.main([str(camvideo), "--facecam", "0.7,0.55,0.3,0.45", "--gameplay", "x,y,z,w", "--out", "o"]) == 2
    assert cli.main([str(camvideo), "--facecam", "0.7,0.55,0.3,0.45", "--gameplay", "0.5,0.5,0.9,0.9", "--out", "o"]) == 2
    assert "gameplay" in capsys.readouterr().err
