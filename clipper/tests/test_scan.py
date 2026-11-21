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

    def run(args, check=True, show=False, timeout=None):
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
    assert any("wa/ba/b" in a for a in calls)             # smallest audio track: only loudness is measured
    sections = [c for c in calls if "--download-sections" in c]
    assert len(sections) == 2 and not any("--download-sections" in c and "wa/ba/b" in c for c in calls)
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
                      out=tmp_path / "o2", cfg=cfg2, scan=True, allow_repeats=True, log=logs.append)
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


def test_min_score_leaves_out_low_scored_candidates(landscape, tmp_path, cfg2):
    cfg2["scan"]["min_motion"] = 0
    out = tmp_path / "o"
    r = pipeline.run(str(landscape), n=3, window=12, rng=None, fill=False, out=out, cfg=cfg2, scan=True,
                     judge_name="chat", log=QUIET)
    ids = r["pending"]
    rows = []
    for k, i in enumerate(ids):
        ev = json.loads((out / "evidence" / i / "evidence.json").read_text())
        w = ev["window"]
        rows.append({"id": i, "score": 8 if k == 0 else 2, "reason": "r", "best_start": w["start_s"], "best_end": w["end_s"],
                     "hook": "KEEP ME" if k == 0 else "DROP ME", "caption": "c", "hashtags": ["#a", "#b", "#c", "#d", "#e"],
                     "risk_flags": []})
    (out / "judgments.json").write_text(json.dumps(rows))
    r2 = pipeline.run(str(landscape), n=3, window=12, rng=None, fill=False, out=out, cfg=cfg2, scan=True,
                      judge_name="chat", judgments=str(out / "judgments.json"), min_score=4, log=QUIET)
    assert [c["hook"] for c in r2["clips"]] == ["KEEP ME"]
    assert len(r2["not_selected"]) == len(ids) - 1


def test_failed_section_is_retried_then_succeeds(landscape, tmp_path, cfg2, monkeypatch):
    calls, boom = [], {"n": 0}
    fake_ytdlp(monkeypatch, landscape, calls)
    real = ingest._run_ytdlp

    def flaky(args, check=True, show=False, timeout=None):
        if "--download-sections" in args and boom["n"] < 2:
            boom["n"] += 1
            raise subprocess.CalledProcessError(1, args)
        return real(args, check=check, show=show, timeout=timeout)
    monkeypatch.setattr(ingest, "_run_ytdlp", flaky)
    logs = []
    m = ingest.download_section("https://www.youtube.com/watch?v=vidtest", "vidtest", 30, 42, cfg2, logs.append,
                                _sleep=lambda s: None)
    assert m.path.is_file() and boom["n"] == 2 and sum("retrying" in l for l in logs) == 2


def _failing_sections(monkeypatch, landscape, which):
    """Fake yt-dlp where the Nth distinct section (0-based) always fails, as a 403 would."""
    calls = []
    fake_ytdlp(monkeypatch, landscape, calls)
    real, seen = ingest._run_ytdlp, []

    def run(args, check=True, show=False, timeout=None):
        if "--download-sections" in args:
            sec = args[args.index("--download-sections") + 1]
            if sec not in seen:
                seen.append(sec)
            if which is None or seen.index(sec) in which:
                raise subprocess.CalledProcessError(1, args)
        return real(args, check=check, show=show, timeout=timeout)
    monkeypatch.setattr(ingest, "_run_ytdlp", run)
    monkeypatch.setattr(ingest.time, "sleep", lambda s: None)


def test_section_that_never_downloads_is_skipped_not_fatal(landscape, tmp_path, cfg2, monkeypatch):
    _failing_sections(monkeypatch, landscape, which={1})          # the 2nd moment is refused every time
    logs = []
    r = pipeline.run("https://www.youtube.com/watch?v=vidtest", n=4, window=12, rng=None, fill=False,
                     out=tmp_path / "o", cfg=cfg2, scan=True, log=logs.append)
    assert len(r["clips"]) >= 1                                   # the job carried on with what it had
    assert any("failed to download after" in l for l in logs) and any("could not be downloaded" in l for l in logs)


def test_all_sections_refused_gives_a_clear_error(landscape, tmp_path, cfg2, monkeypatch):
    _failing_sections(monkeypatch, landscape, which=None)
    with pytest.raises(Exception, match="None of the chosen moments could be downloaded"):
        pipeline.run("https://www.youtube.com/watch?v=vidtest", n=2, window=12, rng=None, fill=False,
                     out=tmp_path / "o", cfg=cfg2, scan=True, log=QUIET)


def test_worker_log_survives_unencodable_characters(tmp_path, monkeypatch, capsys):
    from clipper import worker

    class Cp1252Out:
        def __init__(self):
            self.buf = []
        def reconfigure(self, **kw):
            self.enc = kw
        def write(self, s):
            s.encode("cp1252")                                    # what the Windows console does
            self.buf.append(s)
        def flush(self):
            pass
    out = Cp1252Out()
    monkeypatch.setattr(worker.sys, "stdout", out)
    worker._safe_streams()
    assert out.enc["errors"] == "replace"


def test_presets_save_and_use(tmp_path, monkeypatch):
    from clipper import jobs, presets
    monkeypatch.setenv("CLIPPER_CONFIG_DIR", str(tmp_path / "cfg"))
    assert presets.save("IShowSpeed", "0,0.42,0.205,0.23", "0.21,0,0.79,1") == "ishowspeed"
    assert presets.names() == ["ishowspeed"]
    monkeypatch.setenv("CLIPPER_JOBS_DIR", str(tmp_path / "jobs"))
    f = tmp_path / "v.mp4"
    f.write_bytes(b"x")
    job = jobs.create(str(f), preset="ishowspeed")
    assert job["params"]["facecam"] == [0, 0.42, 0.205, 0.23] and job["params"]["gameplay"] == [0.21, 0, 0.79, 1]
    with pytest.raises(Exception, match="No preset named"):
        presets.get("nobody")
    with pytest.raises(Exception, match="facecam"):
        presets.save("bad", "1,2,3")


def test_cli_save_preset_and_use(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("CLIPPER_CONFIG_DIR", str(tmp_path / "cfg"))
    assert cli.main(["--save-preset", "speed", "--facecam", "0,0.42,0.205,0.23"]) == 0
    assert "Saved preset 'speed'" in capsys.readouterr().out
    assert cli.main(["x.mp4", "--preset", "nobody"]) != 0
    assert "No preset named" in capsys.readouterr().err


def _long_stream_signals(hours=6.0, hot_minutes=20.0, peaks=(1.5, 2.5, 3.5, 4.5, 5.5)):
    """A long stream whose first minutes are by far the loudest, plus a few smaller real highlights later."""
    n = int(hours * 3600 / 0.5)
    rng = np.random.default_rng(3)
    x = rng.normal(0, 0.3, n)
    hot = int(hot_minutes * 60 / 0.5)
    for i in range(0, hot, 80):                       # many strong spikes packed into the opening
        x[i:i + 6] += 6.0 + rng.random()
    for h in peaks:                                   # a real highlight roughly every hour
        i = int(h * 3600 / 0.5)
        x[i:i + 8] += 4.0
    return Signals(t0=0.0, n=n, raw={"audio_rms": x}, z={"audio_rms": x}, combined=x)


def test_scan_spreads_picks_over_a_long_stream(cfg):
    sig = _long_stream_signals()
    picks = scan.find_moments(None, sig, 40.0, 6, cfg, False, QUIET)[:6]
    hours = sorted({int(c.start_s // 3600) for c in picks})
    assert len(hours) >= 4                            # not all from the loud opening
    assert any(c.start_s > 3600 for c in picks) and sum(c.start_s < 1200 for c in picks) <= 2
    # gap between neighbours is kept
    st = sorted(c.start_s for c in picks)
    assert all(b - a >= 180 for a, b in zip(st, st[1:]))


def test_short_stream_keeps_the_old_behaviour(cfg):
    sig = _long_stream_signals(hours=0.2, hot_minutes=5, peaks=())
    assert scan.bucket_count(sig.n * 0.5, 6, cfg) == 1
    cands = scan.find_moments(None, sig, 40.0, 3, cfg, False, QUIET)
    assert cands == sorted(cands, key=lambda c: -c.score)


def test_quiet_stretches_are_not_forced_in(cfg):
    sig = _long_stream_signals(hours=6.0, hot_minutes=40, peaks=())   # only the first 40 min has anything
    picks = scan.find_moments(None, sig, 40.0, 4, cfg, False, QUIET)[:4]
    assert sum(c.start_s < 2400 + 40 for c in picks) >= 3             # dead hours did not displace the real action


def test_bucket_count_and_switch_off(cfg):
    assert scan.bucket_count(23.7 * 3600, 10, cfg) == 10
    assert scan.bucket_count(2 * 3600, 10, cfg) == 8
    cfg["scan"]["bucket_s"] = 0
    assert scan.bucket_count(23.7 * 3600, 10, cfg) == 1


def _chat_line(t_ms, *runs):
    return json.dumps({"replayChatItemAction": {"videoOffsetTimeMsec": str(t_ms), "actions": [{
        "addChatItemAction": {"item": {"liveChatTextMessageRenderer": {"message": {"runs": [
            ({"text": r} if isinstance(r, str) else {"emoji": {"emojiId": r[0]}}) for r in runs]}}}}}]}})


def test_funny_chat_messages_are_found(tmp_path):
    f = tmp_path / "chat.live_chat.json"
    f.write_text("\n".join([
        _chat_line(10_000, "nice play"), _chat_line(20_000, "LUL LUL"), _chat_line(21_000, "KEKW"),
        _chat_line(22_000, ("\U0001F480",)), _chat_line(23_000, "this is so cringe"), _chat_line(30_000, "gg"),
        _chat_line(31_000, "hahaha"), "not json", _chat_line(40_000, "skull issue"), ]))
    assert ingest.parse_chat_funny(f) == [20.0, 21.0, 22.0, 23.0, 31.0]


def test_funny_chat_becomes_a_signal_and_lifts_the_score(landscape, cfg):
    from clipper import signals
    media = ingest.Media(path=landscape, offset=0.0, **ingest.probe(landscape))
    base = signals.analyze(media, cfg)
    assert "chat_funny" not in base.raw
    media.chat_funny = [30 + i * 0.2 for i in range(40)]            # a burst of laughing at ~30-38 s
    sig = signals.analyze(media, cfg)
    assert "chat_funny" in sig.raw
    i = int(34 / 0.5)
    assert sig.z["chat_funny"][i] > 2 and sig.combined[i] > base.combined[i]


def test_ledger_skips_used_moments_and_allows_repeats(landscape, tmp_path, cfg2, monkeypatch):
    calls = []
    fake_ytdlp(monkeypatch, landscape, calls)
    url = "https://www.youtube.com/watch?v=vidtest"
    r1 = pipeline.run(url, n=1, window=12, rng=None, fill=False, out=tmp_path / "a", cfg=cfg2, scan=True, log=QUIET)
    assert len(r1["clips"]) == 1
    from clipper import ledger
    assert ledger.load(cfg2, "vidtest")                              # remembered
    logs = []
    with pytest.raises(Exception, match="No new moments"):
        pipeline.run(url, n=1, window=12, rng=None, fill=False, out=tmp_path / "b", cfg=cfg2, scan=True, log=logs.append)
    assert any("already cut into clips" in l for l in logs)
    r3 = pipeline.run(url, n=1, window=12, rng=None, fill=False, out=tmp_path / "c", cfg=cfg2, scan=True,
                      allow_repeats=True, log=QUIET)
    assert len(r3["clips"]) == 1


def test_ledger_learns_from_earlier_job_folders(tmp_path):
    from clipper import ledger
    jobs_root = tmp_path / "jobs"
    (jobs_root / "j1" / "out").mkdir(parents=True)
    (jobs_root / "j1" / "out" / "clips.json").write_text(json.dumps(
        {"video_id": "abc", "clips": [{"start_s": 100, "end_s": 140}, {"start_s": 500, "end_s": 540}]}))
    (jobs_root / "j2" / "out").mkdir(parents=True)
    (jobs_root / "j2" / "out" / "clips.json").write_text(json.dumps({"video_id": "other", "clips": [{"start_s": 1, "end_s": 2}]}))
    cfg = {"cache_dir": str(jobs_root / ".cache")}
    assert ledger.seed_from_jobs(cfg, "abc") == 2
    assert ledger.overlaps(120, 130, ledger.load(cfg, "abc")) and not ledger.overlaps(1000, 1040, ledger.load(cfg, "abc"))
    assert ledger.load(cfg, "other") == []


def test_clip_count_is_not_capped_at_ten(tmp_path, monkeypatch):
    from clipper import jobs
    monkeypatch.setenv("CLIPPER_JOBS_DIR", str(tmp_path / "jobs"))
    f = tmp_path / "v.mp4"
    f.write_bytes(b"x")
    assert jobs.create(str(f), n=30)["params"]["n"] == 30
    with pytest.raises(Exception, match="between 1 and 40"):
        jobs.create(str(f), n=41)


def test_chat_download_that_takes_too_long_is_abandoned(landscape, tmp_path, cfg2, monkeypatch):
    calls = []
    fake_ytdlp(monkeypatch, landscape, calls)
    real = ingest._run_ytdlp

    def slow_chat(args, check=True, show=False, timeout=None):
        if "--write-subs" in args:
            assert timeout == cfg2["scan"]["chat_timeout_s"]       # the configured limit is passed to yt-dlp
            raise subprocess.TimeoutExpired(args, timeout)
        return real(args, check=check, show=show, timeout=timeout)
    monkeypatch.setattr(ingest, "_run_ytdlp", slow_chat)
    logs = []
    r = pipeline.run("https://www.youtube.com/watch?v=vidtest", n=1, window=12, rng=None, fill=False,
                     out=tmp_path / "o", cfg=cfg2, scan=True, log=logs.append)
    assert len(r["clips"]) == 1 and any("taking too long" in l for l in logs)
    assert (Path(cfg2["cache_dir"]) / "vidtest" / "chat.skipped").is_file()
    n_chat = sum("--write-subs" in c for c in calls)
    pipeline.run("https://www.youtube.com/watch?v=vidtest", n=1, window=12, rng=None, fill=False,
                 out=tmp_path / "o2", cfg=cfg2, scan=True, allow_repeats=True, log=logs.append)
    assert sum("--write-subs" in c for c in calls) == n_chat         # not retried on the next run
    assert any("Skipping the chat replay" in l for l in logs)


def test_resuming_does_not_fetch_chat_again(landscape, tmp_path, cfg2, monkeypatch):
    calls = []
    fake_ytdlp(monkeypatch, landscape, calls)
    url = "https://www.youtube.com/watch?v=vidtest"
    out = tmp_path / "o"
    r = pipeline.run(url, n=1, window=12, rng=None, fill=False, out=out, cfg=cfg2, scan=True, judge_name="chat", log=QUIET)
    assert r["status"] == "awaiting_judgments"
    for f in (Path(cfg2["cache_dir"]) / "vidtest").glob("chat*"):      # as if the cache were somewhere else
        f.unlink()
    before = sum("--write-subs" in c for c in calls)
    logs = []
    pipeline.run(url, n=1, window=12, rng=None, fill=False, out=out, cfg=cfg2, scan=True, judge_name="chat", log=logs.append)
    assert sum("--write-subs" in c for c in calls) == before and any("not fetching the chat replay again" in l for l in logs)


def test_resume_command_keeps_the_cache_dir_and_cli_accepts_it(tmp_path, cfg2):
    cmd = pipeline._resume_cmd("https://youtu.be/x", 3, 40.0, None, False, None, tmp_path / "o", "chat", True, False,
                               False, True, cfg2, 0.0)
    assert "--cache-dir" in cmd
    assert cli.build_parser().parse_args(["x", "--cache-dir", "/tmp/c"]).cache_dir == "/tmp/c"


def test_moments_are_downloaded_in_parallel_but_judged_in_order(landscape, tmp_path, cfg2, monkeypatch):
    import threading, time as _t
    calls = []
    fake_ytdlp(monkeypatch, landscape, calls)
    real = ingest._run_ytdlp
    live, peak = {"n": 0}, {"n": 0}
    lock = threading.Lock()

    def slow(args, check=True, show=False, timeout=None):
        if "--download-sections" in args:
            with lock:
                live["n"] += 1
                peak["n"] = max(peak["n"], live["n"])
            _t.sleep(0.4)                                          # a download that mostly waits on the network
            try:
                return real(args, check=check, show=show, timeout=timeout)
            finally:
                with lock:
                    live["n"] -= 1
        return real(args, check=check, show=show, timeout=timeout)
    monkeypatch.setattr(ingest, "_run_ytdlp", slow)
    cfg2["scan"]["parallel_downloads"] = 3
    logs = []
    pipeline.run("https://www.youtube.com/watch?v=vidtest", n=4, window=12, rng=None, fill=False, out=tmp_path / "o",
                 cfg=cfg2, scan=True, allow_repeats=True, log=logs.append)
    assert peak["n"] >= 2                                          # overlapped
    order = [l for l in logs if l.startswith("[") and "moment" in l]
    assert order and order == sorted(order, key=lambda l: int(l.split("/")[0][1:]))   # progress still counts up in order


def test_audio_and_chat_download_overlap_and_audio_is_smallest_track(landscape, tmp_path, cfg2, monkeypatch):
    import time as _t
    calls = []
    fake_ytdlp(monkeypatch, landscape, calls)
    real = ingest._run_ytdlp
    span = {}

    def run(args, check=True, show=False, timeout=None):
        kind = "chat" if "--write-subs" in args else ("audio" if "audio.%(ext)s" in " ".join(args) else None)
        if kind == "chat":
            span["chat_start"] = _t.monotonic()
            _t.sleep(0.8)                                              # a slow chat download
        elif kind == "audio":
            span["audio_start"] = _t.monotonic()
            _t.sleep(0.8)                                              # a slow audio download
        out = real(args, check=check, show=show, timeout=timeout)
        if kind:
            span[kind + "_end"] = _t.monotonic()
        return out
    monkeypatch.setattr(ingest, "_run_ytdlp", run)
    m = ingest.resolve_scan("https://www.youtube.com/watch?v=vidtest", cfg2, QUIET, True)
    assert m.chat_times
    assert span["audio_start"] < span["chat_end"] and span["chat_start"] < span["audio_end"]   # they overlapped
    audio_call = next(c for c in calls if "-f" in c and "audio.%(ext)s" in " ".join(c))
    assert audio_call[audio_call.index("-f") + 1].startswith("wa")


def test_render_uses_the_fast_preset_by_default(cfg):
    assert cfg["render"]["preset"] == "veryfast"


def test_cli_prints_emoji_titles_on_a_cp1252_console(landscape, tmp_path, monkeypatch):
    """The 'red circle' in a stream title used to crash the command line on Windows."""
    import io, sys
    raw = io.BytesIO()
    out = io.TextIOWrapper(raw, encoding="cp1252", errors="strict", write_through=True)
    monkeypatch.setattr(sys, "stdout", out)
    from clipper.console import safe_streams
    safe_streams()
    print("\U0001F534 PLAYING GTA 6 RIGHT NOW")                  # would raise UnicodeEncodeError on cp1252
    out.flush()
    assert b"PLAYING GTA 6" in raw.getvalue()


def test_cli_main_switches_the_console_to_utf8(tmp_path, monkeypatch):
    from clipper import console
    seen = []
    monkeypatch.setattr(console, "safe_streams", lambda: seen.append(1))
    monkeypatch.setenv("CLIPPER_CONFIG_DIR", str(tmp_path / "cfg"))
    assert cli.main(["--save-preset", "x", "--facecam", "0,0,0.2,0.2"]) == 0
    assert seen
