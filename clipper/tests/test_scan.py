import json
import subprocess
from pathlib import Path

import numpy as np
import pytest

from clipper import candidates as cand
from clipper import captions, cli, ingest, pipeline, scan
from clipper.signals import Signals
from conftest import probe_json

QUIET = lambda *_: None


@pytest.fixture
def cfg2(cfg, tmp_path):
    cfg["cache_dir"] = str(tmp_path / "cache")
    return cfg


def fake_ytdlp(monkeypatch, src_video, calls, live=False):
    """Pretend to be yt-dlp using a local video: metadata, audio-only, section cuts and a chat replay."""
    info = {"id": "vidtest", "title": "9 HOUR STREAM", "is_live": live, "chapters": [], "heatmap": []}

    def run(args, check=True, show=False):
        calls.append(list(args))
        if "-J" in args:
            return subprocess.CompletedProcess(args, 0, json.dumps(info), "")
        o = Path(args[args.index("-o") + 1]) if "-o" in args else None
        if "--write-subs" in args:                       # chat replay: a burst of messages around 34-40 s
            lines = [json.dumps({"replayChatItemAction": {"videoOffsetTimeMsec": str(int((34 + i * 0.1) * 1000))}})
                     for i in range(60)]
            Path(str(o) + ".live_chat.json").write_text("\n".join(lines))
        elif "--download-sections" in args:              # one moment at "full quality"
            a, b = args[args.index("--download-sections") + 1].lstrip("*").split("-")
            subprocess.run(["ffmpeg", "-y", "-v", "error", "-ss", a, "-to", b, "-i", str(src_video), "-c:v", "libx264",
                            "-c:a", "aac", str(o).replace("%(ext)s", "mp4")], check=True)
        else:                                            # audio-only download of the whole stream
            subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", str(src_video), "-vn", "-c:a", "aac",
                            str(o).replace("%(ext)s", "m4a")], check=True)
        return subprocess.CompletedProcess(args, 0, "", "")
    monkeypatch.setattr(ingest, "_ytdlp", lambda: "yt-dlp")
    monkeypatch.setattr(ingest, "_run_ytdlp", run)


def test_url_scan_end_to_end(landscape, tmp_path, cfg2, monkeypatch):
    calls = []
    fake_ytdlp(monkeypatch, landscape, calls)
    out = tmp_path / "out"
    cfg2["scan"]["min_motion"] = 0          # this fixture is mostly a static picture; dead-screen check has its own tests
    r = pipeline.run("https://www.youtube.com/watch?v=vidtest", n=2, window=12, rng=None, fill=False, out=out,
                     cfg=cfg2, scan=True, log=QUIET)
    assert r["status"] == "done" and len(r["clips"]) == 2
    # whole stream audio-only first; the picture was fetched ONLY for the 2 chosen moments
    assert any("ba/b" in a for a in calls)
    sections = [c for c in calls if "--download-sections" in c]
    assert len(sections) == 2 and not any("--download-sections" in c and "ba/b" in c for c in calls)
    top = r["clips"][0]
    overlap = min(top["end_s"], 45) - max(top["start_s"], 30)             # the loud/hyped stretch is 30-45 s
    assert overlap >= 6 and top["start_s"] >= 24 and top["end_s"] <= 48   # starts a little before the payoff
    assert "chat" in top["signals"] and "audio_rms" in top["signals"] and "motion" not in top["signals"]
    assert top["link"] == "https://www.youtube.com/watch?v=vidtest&t=" + str(int(top["start_s"])) + "s"
    for c in r["clips"]:
        v = next(s for s in probe_json(out / c["file"])["streams"] if s["codec_type"] == "video")
        assert (v["width"], v["height"]) == (1080, 1920)
    html = (out / "review.html").read_text()
    assert html.count("<video") == 2 and "Copy caption" in html and "watch?v=vidtest" in html
    assert "Title on video" in (out / "report.md").read_text()


def test_scan_live_stream_refused(landscape, tmp_path, cfg2, monkeypatch):
    calls = []
    fake_ytdlp(monkeypatch, landscape, calls, live=True)
    rc = cli.main(["https://www.youtube.com/watch?v=vidtest", "--scan", "--out", str(tmp_path / "o"),
                   "--config", "config.yaml"])
    assert rc == 2 and len(calls) == 1                                    # metadata only, nothing downloaded


def test_scan_local_file_and_defaults(landscape, tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    rc = cli.main([str(landscape), "--scan", "--out", "o"])               # defaults: n=15, window=40
    assert rc == 0
    d = json.loads((tmp_path / "o" / "clips.json").read_text())
    assert d["clips"] and all(c["end_s"] - c["start_s"] <= 40.6 for c in d["clips"])
    assert (tmp_path / "o" / "review.html").is_file()


def test_transcribe_boosts_reactions_and_writes_title_from_speech(landscape, tmp_path, cfg2, monkeypatch):
    monkeypatch.setattr(captions, "check_available", lambda: None)

    def fake(wav, cfg):   # "no way" is said 3 s into every window, "bro" right after
        return [{"w": "this", "start": 1.0, "end": 1.2}, {"w": "is", "start": 1.2, "end": 1.4},
                {"w": "no", "start": 3.0, "end": 3.2}, {"w": "way", "start": 3.2, "end": 3.5},
                {"w": "bro", "start": 3.6, "end": 3.9}, {"w": "what", "start": 4.0, "end": 4.2}]
    monkeypatch.setattr(captions, "_transcribe_words", fake)
    calls = []
    cfg2["scan"]["min_motion"] = 0
    fake_ytdlp(monkeypatch, landscape, calls)
    r = pipeline.run("https://www.youtube.com/watch?v=vidtest", n=2, window=12, rng=None, fill=False,
                     out=tmp_path / "o", cfg=cfg2, scan=True, transcribe=True, log=QUIET)
    c = r["clips"][0]
    assert "NO WAY" in c["hook"] and "speech_hype" in c["signals"] and c["signals"]["speech_hype"] > 0
    assert "no way" in (c["judge"]["caption"] or "").lower()


def test_hook_and_hype_helpers():
    words = [{"w": w, "start": i * 0.3, "end": i * 0.3 + 0.2} for i, w in
             enumerate("so i walk in and OH MY GOD look at that".split())]
    assert scan.hype_score(words) >= 4
    assert scan.hook_from_words(words) == "OH MY GOD LOOK AT THAT"
    assert scan.hook_from_words([{"w": "hello", "start": 0, "end": 1}]) is None


def test_recenter_puts_peak_early_in_window_and_dedupes():
    n = 600
    comb = np.zeros(n)
    comb[300:306] = 5.0                       # a single hot spot at bin 300 (150 s)
    sig = Signals(t0=0.0, n=n, raw={"audio_rms": comb}, z={"audio_rms": comb}, combined=comb)
    cfg = {"candidates": {"top_fraction": 0.2, "mean_weight": 0.5, "stride_s": 1.0, "count": 15}}
    c = cand.Candidate(start_s=140.0, end_s=180.0, score=1.0, breakdown={}, peak_s=150.0)
    dup = cand.Candidate(start_s=150.0, end_s=190.0, score=0.5, breakdown={}, peak_s=152.0)
    out = cand.recenter([c, dup], sig, 40.0, cfg)
    assert len(out) == 1                      # duplicate of the same moment dropped
    assert out[0].start_s == pytest.approx(150.0 - 40 * 0.35, abs=0.5)
    assert out[0].start_s < 150.0 < out[0].end_s


def test_scan_audio_only_has_no_motion_signal(landscape, tmp_path, cfg2):
    a = tmp_path / "a.m4a"
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", str(landscape), "-vn", "-c:a", "aac", str(a)], check=True)
    m = ingest.resolve(str(a), None, cfg2) if False else ingest.Media(path=a, **ingest.probe(a, need_video=False))
    assert m.width == 0 and m.has_audio
    from clipper import signals
    sig = signals.analyze(m, cfg2)
    assert "motion" not in sig.raw and "audio_rms" in sig.raw


def test_diversify_keeps_moments_apart_and_backfills():
    mk = lambda s, sc: cand.Candidate(start_s=s, end_s=s + 40, score=sc, breakdown={}, peak_s=s + 10)
    cs = [mk(100, 5), mk(150, 4.5), mk(200, 4), mk(1000, 2), mk(2000, 1)]
    top3 = scan.diversify(cs, 3, 180)
    assert sorted(c.start_s for c in top3) == [100, 1000, 2000]              # the 150 and 200 near-duplicates are skipped
    top5 = scan.diversify(cs, 5, 180)                                         # not enough spread-out ones: backfill
    assert len(top5) == 5 and top5[0].start_s == 100


def test_heuristic_scores_separate_and_hooks_are_unique():
    from clipper import evidence, judge
    m = type("M", (), {"video_id": "v"})()
    cs = [cand.Candidate(start_s=i * 500.0, end_s=i * 500.0 + 40, score=s, breakdown={}, peak_s=i * 500.0 + 10)
          for i, s in enumerate([3.8, 3.2, 2.6, 2.0, 1.4])]
    items = evidence.make_items(m, cs, 40, {})
    out, _ = judge.HeuristicJudge().judge(items, {"niche": "gta6", "title": "X", "trends": {}})
    scores = [out[it.id].score for it in items]
    hooks = [out[it.id].hook for it in items]
    assert scores == sorted(scores, reverse=True) and len(set(scores)) == 5 and max(scores) < 10
    assert len(set(hooks)) == 5


def test_dead_screens_are_skipped_and_replaced(landscape, tmp_path, cfg2, monkeypatch):
    """A NO SIGNAL style (static) moment must be dropped and the next live one used instead."""
    calls = []
    fake_ytdlp(monkeypatch, landscape, calls)
    logs = []
    r = pipeline.run("https://www.youtube.com/watch?v=vidtest", n=1, window=12, rng=None, fill=False,
                     out=tmp_path / "o", cfg=cfg2, scan=True, log=logs.append)
    assert len(r["clips"]) == 1 and r["clips"][0]["start_s"] >= 24 and r["clips"][0]["end_s"] <= 48   # the moving part
    # with the filter on, only the live window survives even if more clips are asked for
    r2 = pipeline.run("https://www.youtube.com/watch?v=vidtest", n=4, window=12, rng=None, fill=False,
                      out=tmp_path / "o2", cfg=cfg2, scan=True, log=logs.append)
    assert 1 <= len(r2["clips"]) < 4
    assert any("skipped: frozen" in l for l in logs) and any("live moments were found" in l for l in logs)


def test_visual_liveliness_detects_static_and_black(tmp_path):
    from clipper import visual
    mk = lambda name, vf: subprocess.run(["ffmpeg", "-y", "-v", "error", "-f", "lavfi", "-i", vf, "-pix_fmt", "yuv420p",
                                          str(tmp_path / name)], check=True)
    mk("static.mp4", "color=c=0x303030:s=320x180:r=25:d=6,drawtext=text='NO SIGNAL':fontsize=30:fontcolor=white")
    mk("black.mp4", "color=c=black:s=320x180:r=25:d=6")
    mk("live.mp4", "testsrc2=s=320x180:r=25:d=6")
    assert visual.dead_reason(visual.liveliness(tmp_path / "static.mp4", 0, 6)).startswith("frozen")
    assert visual.dead_reason(visual.liveliness(tmp_path / "black.mp4", 0, 6)) == "black screen"
    assert visual.dead_reason(visual.liveliness(tmp_path / "live.mp4", 0, 6)) is None


def test_scan_without_speech_shows_no_placeholder_title(landscape, tmp_path, cfg2, monkeypatch):
    cfg2["scan"]["min_motion"] = 0
    calls = []
    fake_ytdlp(monkeypatch, landscape, calls)
    r = pipeline.run("https://www.youtube.com/watch?v=vidtest", n=2, window=12, rng=None, fill=False,
                     out=tmp_path / "o", cfg=cfg2, scan=True, log=QUIET)
    assert all(c["hook"] == "" and not c["title"] for c in r["clips"])
    assert "needs Claude" in (tmp_path / "o" / "review.html").read_text()


def test_chat_judge_scan_makes_labelled_sheets(landscape, tmp_path, cfg2, monkeypatch):
    cfg2["scan"]["min_motion"] = 0
    calls = []
    fake_ytdlp(monkeypatch, landscape, calls)
    out = tmp_path / "o"
    r = pipeline.run("https://www.youtube.com/watch?v=vidtest", n=3, window=12, rng=None, fill=False, out=out,
                     cfg=cfg2, scan=True, judge_name="chat", log=QUIET)
    assert r["status"] == "awaiting_judgments" and len(r["pending"]) >= 1
    sheets = sorted((out / "judge_sheets").glob("sheet_*.jpg"))
    assert sheets and probe_json(sheets[0])["streams"][0]["width"] <= 1600
    assert "NO SIGNAL" in (out / "judge_prompt.md").read_text()


def test_printed_resume_command_keeps_layout_options(landscape, tmp_path, cfg2, capsys):
    cfg2["render"].update(layout="split", facecam=[0.05, 0.0, 0.2, 0.3], gameplay=[0.3, 0.05, 0.7, 0.9])
    cfg2["scan"]["min_motion"] = 0
    logs = []
    pipeline.run(str(landscape), n=2, window=12, rng=None, fill=False, out=tmp_path / "o", cfg=cfg2, scan=True,
                 judge_name="chat", log=logs.append)
    text = "\n".join(logs)
    assert "--facecam 0.05,0,0.2,0.3" in text and "--gameplay 0.3,0.05,0.7,0.9" in text and "--judgments" in text
