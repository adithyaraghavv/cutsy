import json
import subprocess
from pathlib import Path

import pytest

from clipper import cli, judge, pipeline, qa, trends
from clipper.errors import ClipperError
from conftest import probe_json

QUIET = lambda *_: None


@pytest.fixture
def cfg2(cfg, tmp_path):
    cfg["cache_dir"] = str(tmp_path / "cache")
    return cfg


def run_chat(src, out, cfg, **kw):
    kw.setdefault("judge_name", "chat")
    return pipeline.run(str(src), n=kw.pop("n", 2), window=kw.pop("window", 12), rng=None, fill=False,
                        out=out, cfg=cfg, log=QUIET, **kw)


def good_row(item_id, ev, **over):
    w = ev["window"]
    row = {"id": item_id, "score": 8, "reason": "Loud tone burst with fast movement.",
           "best_start": w["start_s"], "best_end": w["end_s"], "hook": "NO WAY THIS HAPPENED",
           "caption": "Wild moment", "hashtags": ["#a", "#b", "#c", "#d", "#e"], "risk_flags": ["profanity"]}
    row.update(over)
    return row


# ---------------------------------------------------------------- evidence ----

def test_chat_flow_awaits_then_renders_with_claude_choices(landscape, tmp_path, cfg2):
    out = tmp_path / "out"
    r = run_chat(landscape, out, cfg2, shortlist=3)
    assert r["status"] == "awaiting_judgments" and len(r["pending"]) == 3
    assert not list(out.glob("clip_*.mp4"))                       # nothing rendered yet
    assert (out / "judge_prompt.md").is_file()
    ids = r["pending"]
    for i in ids:
        d = out / "evidence" / i
        ev = json.loads((d / "evidence.json").read_text())
        frames = sorted(d.glob("frame_*.jpg"))
        assert len(frames) == 8 and (d / "contact.jpg").is_file()
        assert {"window", "signals_mean_z", "trim_length_s", "transcript", "peak_s"} <= set(ev)
        w = probe_json(frames[0])["streams"][0]["width"]
        assert w <= 512
    prompt = (out / "judge_prompt.md").read_text()
    assert all(i in prompt for i in ids) and "exactly 5" in prompt

    # Claude "writes" judgments: prefer the 2nd candidate, with a custom hook and trimmed range
    rows = []
    for k, i in enumerate(ids):
        ev = json.loads((out / "evidence" / i / "evidence.json").read_text())
        rows.append(good_row(i, ev, score=9 if k == 1 else 5,
                             hook="CUSTOM HOOK HERE" if k == 1 else "OTHER HOOK",
                             best_start=ev["window"]["start_s"] + 0.2, best_end=ev["window"]["end_s"] - 0.2))
    (out / "judgments.json").write_text(json.dumps(rows))
    r2 = run_chat(landscape, out, cfg2, shortlist=3, judgments=str(out / "judgments.json"))
    assert r2["status"] == "done"
    c1 = r2["clips"][0]
    assert c1["id"] == ids[1] and c1["hook"] == "CUSTOM HOOK HERE"
    assert c1["judge"]["risk_flags"] == ["profanity"]
    ev1 = json.loads((out / "evidence" / ids[1] / "evidence.json").read_text())
    assert c1["start_s"] == pytest.approx(ev1["window"]["start_s"] + 0.2, abs=0.01)
    v = next(s for s in probe_json(out / "clip_1.mp4")["streams"] if s["codec_type"] == "video")
    assert (v["width"], v["height"]) == (1080, 1920)
    rep = (out / "report.md").read_text()
    assert "CUSTOM HOOK HERE" in rep and "Judge score" in rep and "QA (heuristic)" in rep and "$0.00" in rep

    # cache: rerun without the judgments file does not ask again
    (out / "judgments.json").unlink()
    r3 = run_chat(landscape, out, cfg2, shortlist=3)
    assert r3["status"] == "done"


def test_invalid_judgments_are_rejected_with_reasons(landscape, tmp_path, cfg2):
    out = tmp_path / "out"
    r = run_chat(landscape, out, cfg2, shortlist=2)
    ids = r["pending"]
    ev = [json.loads((out / "evidence" / i / "evidence.json").read_text()) for i in ids]
    rows = [good_row(ids[0], ev[0], hook="ONE TWO THREE FOUR FIVE SIX SEVEN"),
            good_row(ids[1], ev[1], hashtags=["#a"], score=11, risk_flags=["nsfw"]),
            {"id": "c99-stale", "score": 5}]
    (out / "judgments.json").write_text(json.dumps(rows))
    r2 = run_chat(landscape, out, cfg2, shortlist=2, judgments=str(out / "judgments.json"))
    assert r2["status"] == "awaiting_judgments" and len(r2["pending"]) == 2
    text = " ".join(r2["problems"])
    assert "hook must be 1-6 words" in text and "exactly 5" in text and "score must be 1-10" in text
    assert "risk_flags" in text and "stale id" in text


def test_validate_trim_window_rules(landscape, cfg2):
    from clipper import evidence, ingest
    from clipper.candidates import Candidate
    media = ingest.resolve(str(landscape), None, cfg2)
    c = Candidate(start_s=0, end_s=55, score=1, breakdown={})
    item = evidence.make_items(media, [c], 55, {})[0]
    base = good_row(item.id, {"window": {"start_s": 0, "end_s": 55}})
    judge.validate(base, item)
    with pytest.raises(ValueError, match="trimmed length"):
        judge.validate({**base, "best_start": 10, "best_end": 30}, item)
    with pytest.raises(ValueError, match="inside the window"):
        judge.validate({**base, "best_start": 0, "best_end": 70}, item)


def test_bad_judgments_file_is_friendly(landscape, tmp_path, cfg2):
    out = tmp_path / "o"
    bad = tmp_path / "j.json"
    bad.write_text("{not json")
    with pytest.raises(ClipperError, match="not valid JSON"):
        run_chat(landscape, out, cfg2, judgments=str(bad))
    with pytest.raises(ClipperError, match="not found"):
        run_chat(landscape, out, cfg2, judgments=str(tmp_path / "missing.json"))


# --------------------------------------------------------------- heuristic ----

def test_heuristic_judge_valid_output(landscape, tmp_path, cfg2):
    out = tmp_path / "out"
    cfg2["cache_dir"] = str(tmp_path / "c")
    (tmp_path / "t.json").write_text(json.dumps({"keywords": ["gta"], "hashtags": ["#gta6", "#Rockstar"]}))
    monkey = None
    r = pipeline.run(str(landscape), n=2, window=12, rng=None, fill=False, out=out, cfg=cfg2,
                     niche="gta6", trends_path=str(tmp_path / "t.json"), log=QUIET)
    assert r["status"] == "done" and r["judge"] == "heuristic"
    for c in r["clips"]:
        j = c["judge"]
        assert len(j["hashtags"]) == 5 and all(t.startswith("#") for t in j["hashtags"])
        assert 1 <= len(j["hook"].split()) <= 6 and j["risk_assessed"] is False
    assert "#gta6" in r["clips"][0]["judge"]["hashtags"]
    assert "NOT assessed" in (out / "report.md").read_text()


# ------------------------------------------------------------------ trends ----

def test_trends_manual_and_youtube_mocked(monkeypatch, tmp_path):
    f = tmp_path / "trends.json"
    f.write_text(json.dumps({"keywords": ["speedrun"], "hashtags": ["#shorts"],
                             "niches": {"gta6": {"keywords": ["trailer 3"], "hashtags": ["#gta6"]}}}))
    monkeypatch.setenv("YOUTUBE_API_KEY", "SECRET-KEY-VALUE")
    calls = []

    def fake(url, headers=None, timeout=10):
        calls.append((url, headers))
        return {"items": [{"snippet": {"title": "GTA 6 insane chase #gta6 #viral", "tags": ["GTA VI", "rockstar"]}}]}
    monkeypatch.setattr(trends, "_http_get_json", fake)
    t = trends.load("gta6", f, log=QUIET)
    assert "speedrun" in t["keywords"] and "trailer 3" in t["keywords"] and "insane" in t["keywords"]
    assert t["hashtags"][0] == "#gta6" and "#viral" in t["hashtags"]
    assert len(calls) == 2 and all("SECRET-KEY-VALUE" not in u for u, _ in calls)  # key never in URL
    assert calls[0][1]["x-goog-api-key"] == "SECRET-KEY-VALUE"


def test_trends_fail_soft_and_never_leak_key(monkeypatch, tmp_path, capsys):
    monkeypatch.setenv("YOUTUBE_API_KEY", "SECRET-KEY-VALUE")
    def boom(*a, **k):
        raise RuntimeError("failed for key SECRET-KEY-VALUE")
    monkeypatch.setattr(trends, "_http_get_json", boom)
    logs = []
    t = trends.load("gta6", tmp_path / "none.json", log=logs.append)
    assert t["hashtags"] == ["#gta6"] and any("skipped" in n for n in t["notes"])
    assert "SECRET-KEY-VALUE" not in json.dumps(t) and "SECRET-KEY-VALUE" not in " ".join(logs)


def test_trends_without_key_or_file_is_empty(monkeypatch, tmp_path):
    monkeypatch.delenv("YOUTUBE_API_KEY", raising=False)
    monkeypatch.chdir(tmp_path)
    t = trends.load(None, None, log=QUIET)
    assert t["keywords"] == [] and t["hashtags"] == []


# ---------------------------------------------------------------------- QA ----

def _make_clip(path, vf):
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-f", "lavfi", "-i", vf, "-f", "lavfi", "-i", "sine=d=4",
                    "-pix_fmt", "yuv420p", "-shortest", str(path)], check=True)


def test_qa_detects_black_bars(tmp_path):
    p = tmp_path / "bars.mp4"
    _make_clip(p, "testsrc2=s=1080x1920:r=30:d=4,drawbox=x=0:y=0:w=1080:h=300:color=black:t=fill,"
                  "drawbox=x=0:y=1620:w=1080:h=300:color=black:t=fill")
    r = qa.heuristic_qa(p, 4.0, 2.6, has_hook=False)
    assert not r.passed and r.checks["no_black_bars"] is False and r.fix == {"fill": True}


def test_qa_does_not_flag_dark_scene_background(tmp_path):
    # dark-but-not-black bands (like a blurred background of a night scene) must NOT trigger the fill re-render
    p = tmp_path / "dark.mp4"
    _make_clip(p, "testsrc2=s=1080x1920:r=30:d=4,drawbox=x=0:y=0:w=1080:h=300:color=0x080808:t=fill,"
                  "drawbox=x=0:y=1620:w=1080:h=300:color=0x080808:t=fill")
    r = qa.heuristic_qa(p, 4.0, 2.6, has_hook=False)
    assert r.checks["no_black_bars"] is True and r.fix is None


def test_qa_passes_good_render_and_sees_hook(landscape, tmp_path, cfg2):
    out = tmp_path / "o"
    r = pipeline.run(str(landscape), n=1, window=10, rng=None, fill=False, out=out, cfg=cfg2, log=QUIET)
    q = r["clips"][0]["qa"][0]
    assert q["passed"] and q["checks"]["hook_visible"] and q["checks"]["no_black_bars"]


def test_qa_failure_rerenders_exactly_once(landscape, tmp_path, cfg2, monkeypatch):
    seen = []
    real = qa.heuristic_qa

    def failing(*a, **k):
        seen.append(1)
        r = real(*a, **k)
        if len(seen) == 1:
            return qa.QAResult(False, "heuristic", {"no_black_bars": False}, [], {"fill": True})
        return qa.QAResult(False, "heuristic", {"no_black_bars": False}, [], {"fill": True})  # still failing
    monkeypatch.setattr(qa, "heuristic_qa", failing)
    out = tmp_path / "o"
    r = pipeline.run(str(landscape), n=1, window=10, rng=None, fill=False, out=out, cfg=cfg2, log=QUIET)
    c = r["clips"][0]
    assert len(seen) == 2 and c["rerendered"] and c["fill"] is True and len(c["qa"]) == 2


def test_chat_qa_pack_and_results_apply_fix_once(landscape, tmp_path, cfg2):
    out = tmp_path / "o"
    r = pipeline.run(str(landscape), n=1, window=10, rng=None, fill=False, out=out, cfg=cfg2, qa_name="chat", log=QUIET)
    assert (out / "qa" / "clip_1_sheet.jpg").is_file() and (out / "qa_prompt.md").is_file()
    (out / "qa.json").write_text(json.dumps([{"file": "clip_1.mp4", "passed": False, "issues": ["hook too small"],
                                              "fix": {"fill": True, "hook": "SHORTER HOOK"}}]))
    res = pipeline.apply_qa_results(str(landscape), out=out, cfg=cfg2, qa_results=str(out / "qa.json"), log=QUIET)
    c = res["clips"][0]
    assert c["hook"] == "SHORTER HOOK" and c["fill"] is True and c["rerendered"]
    assert c["qa"][-1]["backend"] == "chat" and "hook too small" in (out / "report.md").read_text()
    (out / "qa.json").write_text(json.dumps([{"file": "clip_1.mp4", "passed": False, "fix": {"hook": "ONE TWO THREE FOUR FIVE SIX SEVEN"}}]))
    with pytest.raises(ClipperError, match="at most 6 words"):
        pipeline.apply_qa_results(str(landscape), out=out, cfg=cfg2, qa_results=str(out / "qa.json"), log=QUIET)


def test_cli_chat_flow_messages(landscape, tmp_path, capsys, monkeypatch):
    monkeypatch.chdir(tmp_path)
    rc = cli.main([str(landscape), "--judge", "chat", "--window", "12", "--shortlist", "2", "--out", "o"])
    out = capsys.readouterr().out
    assert rc == 0 and "judge_prompt.md" in out and "--judgments" in out and "Traceback" not in out
