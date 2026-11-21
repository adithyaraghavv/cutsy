import json
import os
import re
import stat
import subprocess
from pathlib import Path

import pytest

from clipper import cli, pipeline, uploader
from clipper.errors import ClipperError

SECRETS = {"YOUTUBE_OAUTH_CLIENT_ID": "cid-123", "YOUTUBE_OAUTH_CLIENT_SECRET": "SECRET-CLIENT-XYZ",
           "YOUTUBE_REFRESH_TOKEN": "REFRESH-TOKEN-ABC", "INSTAGRAM_ACCESS_TOKEN": "IG-TOKEN-QRS",
           "INSTAGRAM_USER_ID": "17841400000"}


@pytest.fixture
def env(monkeypatch, tmp_path):
    for k, v in SECRETS.items():
        monkeypatch.setenv(k, v)
    monkeypatch.setenv("CLIPPER_CONFIG_DIR", str(tmp_path / "cfg"))
    monkeypatch.setattr(uploader, "_is_interactive", lambda: True)
    monkeypatch.setattr(uploader.time, "sleep", lambda s: None)


class FakeNet:
    def __init__(self, ig_states=("IN_PROGRESS", "FINISHED")):
        self.calls, self.ig_states = [], list(ig_states)

    def __call__(self, method, url, *, headers=None, data=None, timeout=0):
        self.calls.append((method, url, headers or {}, data))
        if url == "https://oauth2.googleapis.com/token":
            return 200, {}, {"access_token": "ACCESS-TOKEN-1"}
        if url.startswith(uploader.YT_UPLOAD):
            return 200, {"location": "https://upload.example/yt/session1"}, {}
        if url == "https://upload.example/yt/session1":
            return 200, {}, {"id": "YT_VIDEO_ID"}
        if url.endswith("/media") and method == "POST":
            return 200, {}, {"id": "CONT1", "uri": "https://rupload.example/ig/CONT1"}
        if url == "https://rupload.example/ig/CONT1":
            return 200, {}, {"success": True}
        if "fields=status_code" in url:
            return 200, {}, {"status_code": self.ig_states.pop(0)}
        if url.endswith("/media_publish"):
            return 200, {}, {"id": "IG_MEDIA_ID"}
        return 404, {}, {"error": {"message": "nope"}}


@pytest.fixture
def clips(tmp_path):
    out = tmp_path / "out"
    out.mkdir()
    (out / "clip_1.mp4").write_bytes(b"VIDEO1" * 100)
    (out / "clip_2.mp4").write_bytes(b"VIDEO2" * 100)
    j = {"caption": "Wild moment", "hashtags": ["#a", "#b", "#c", "#d", "#e"], "risk_flags": ["spoilers"]}
    res = {"clips": [{"file": f"clip_{i}.mp4", "start_s": 0, "end_s": 50, "hook": "no way this happened",
                      "judge": j} for i in (1, 2)]}
    return res, out


def answers(*a):
    it = iter(a)
    return lambda prompt: next(it)


def test_refuses_without_interactive_terminal(monkeypatch, clips):
    monkeypatch.setattr(uploader, "_is_interactive", lambda: False)
    net = FakeNet(); monkeypatch.setattr(uploader, "_http", net)
    res, out = clips
    with pytest.raises(ClipperError, match="interactive terminal"):
        uploader.run_uploads(res, out, ["youtube"], "private", log=lambda *_: None, ask=answers("y", "y"))
    assert net.calls == []


def test_missing_credentials_are_friendly_and_make_no_calls(monkeypatch, clips, tmp_path):
    monkeypatch.setattr(uploader, "_is_interactive", lambda: True)
    for k in SECRETS:
        monkeypatch.delenv(k, raising=False)
    net = FakeNet(); monkeypatch.setattr(uploader, "_http", net)
    res, out = clips
    with pytest.raises(ClipperError, match="YOUTUBE_OAUTH_CLIENT_ID"):
        uploader.run_uploads(res, out, ["youtube"], "private", log=lambda *_: None, ask=answers("y"))
    with pytest.raises(ClipperError, match="INSTAGRAM_ACCESS_TOKEN"):
        uploader.run_uploads(res, out, ["instagram"], "private", log=lambda *_: None, ask=answers("y"))
    assert net.calls == []


def test_each_clip_confirmed_separately_youtube(env, monkeypatch, clips, capsys):
    net = FakeNet(); monkeypatch.setattr(uploader, "_http", net)
    res, out = clips
    logs = []
    done = uploader.run_uploads(res, out, ["youtube"], "private", log=logs.append, ask=answers("y", "n"))
    assert done == [{"clip": "clip_1.mp4", "platform": "youtube", "id": "YT_VIDEO_ID"}]   # clip_2 declined
    start = next(c for c in net.calls if c[1].startswith(uploader.YT_UPLOAD))
    meta = json.loads(start[3])
    assert meta["status"]["privacyStatus"] == "private" and meta["status"]["selfDeclaredMadeForKids"] is False
    assert meta["snippet"]["title"].endswith("#Shorts") and len(meta["snippet"]["title"]) <= 100
    assert start[2]["Authorization"] == "Bearer ACCESS-TOKEN-1"
    put = next(c for c in net.calls if c[0] == "PUT")
    assert put[3] == b"VIDEO1" * 100                       # the right file's bytes
    assert sum(1 for c in net.calls if c[0] == "PUT") == 1
    assert any("RISK FLAGS: spoilers" in l for l in logs)


@pytest.mark.parametrize("answer", ["", "n", "yes please", "Y!", "no"])
def test_anything_but_y_skips(env, monkeypatch, clips, answer):
    net = FakeNet(); monkeypatch.setattr(uploader, "_http", net)
    res, out = clips
    assert uploader.run_uploads(res, out, ["youtube"], "private", log=lambda *_: None, ask=answers(answer, answer)) == []
    assert not any(c[0] == "PUT" for c in net.calls)


def test_instagram_flow(env, monkeypatch, clips):
    net = FakeNet(); monkeypatch.setattr(uploader, "_http", net)
    res, out = clips
    done = uploader.run_uploads(res, out, ["instagram"], "private", log=lambda *_: None, ask=answers("y", "n"))
    assert done == [{"clip": "clip_1.mp4", "platform": "instagram", "id": "IG_MEDIA_ID"}]
    first = net.calls[0]
    form = dict(x.split("=", 1) for x in first[3].decode().split("&"))
    assert form["media_type"] == "REELS" and form["upload_type"] == "resumable"
    up = next(c for c in net.calls if c[1] == "https://rupload.example/ig/CONT1")
    assert up[2]["file_size"] == str(600) and up[2]["offset"] == "0" and up[3] == b"VIDEO1" * 100
    assert net.calls[-1][1].endswith("/media_publish")      # publish only after FINISHED


def test_instagram_processing_error_publishes_nothing(env, monkeypatch, clips):
    net = FakeNet(ig_states=("ERROR",)); monkeypatch.setattr(uploader, "_http", net)
    res, out = clips
    logs = []
    assert uploader.run_uploads(res, out, ["instagram"], "private", log=logs.append, ask=answers("y", "n")) == []
    assert not any(c[1].endswith("/media_publish") for c in net.calls)
    assert any("FAILED" in l for l in logs)


def test_api_error_is_friendly_and_scrubbed(env, monkeypatch, clips):
    def bad(method, url, **k):
        return 403, {}, {"error": {"message": "quota exceeded for ACCESS-TOKEN-1"}} if "token" not in url else (200, {}, None)
    def net(method, url, *, headers=None, data=None, timeout=0):
        if url == "https://oauth2.googleapis.com/token":
            return 200, {}, {"access_token": "ACCESS-TOKEN-1"}
        return 403, {}, {"error": {"message": "quota exceeded for ACCESS-TOKEN-1"}}
    monkeypatch.setattr(uploader, "_http", net)
    res, out = clips
    logs = []
    assert uploader.run_uploads(res, out, ["youtube"], "private", log=logs.append, ask=answers("y", "n")) == []
    text = " ".join(logs)
    assert "HTTP 403" in text and "ACCESS-TOKEN-1" not in text and "***" in text


def test_oauth_first_run_saves_owner_only_token_and_never_prints_secrets(env, monkeypatch, clips, tmp_path):
    monkeypatch.delenv("YOUTUBE_REFRESH_TOKEN")
    seen = []

    def net(method, url, *, headers=None, data=None, timeout=0):
        seen.append((url, data))
        if url == "https://oauth2.googleapis.com/token":
            return 200, {}, {"access_token": "ACCESS-TOKEN-1", "refresh_token": "NEW-REFRESH-TOKEN"}
        if url.startswith(uploader.YT_UPLOAD):
            return 200, {"location": "https://upload.example/yt/s"}, {}
        return 200, {}, {"id": "VID"}
    monkeypatch.setattr(uploader, "_http", net)
    monkeypatch.setattr(uploader, "_wait_for_code", lambda srv, state, timeout=0: "AUTH-CODE")
    res, out = clips
    logs = []
    uploader.run_uploads(res, out, ["youtube"], "private", log=logs.append, ask=answers("y", "n"))
    tf = tmp_path / "cfg" / "youtube_token.json"
    assert json.loads(tf.read_text())["refresh_token"] == "NEW-REFRESH-TOKEN"
    assert stat.S_IMODE(tf.stat().st_mode) == 0o600
    consent = next(l for l in logs if "accounts.google.com" in l)
    assert "code_challenge_method=S256" in consent and "youtube.upload" in consent and "SECRET-CLIENT-XYZ" not in consent
    blob = " ".join(logs)
    for s in ("SECRET-CLIENT-XYZ", "NEW-REFRESH-TOKEN", "ACCESS-TOKEN-1", "AUTH-CODE"):
        assert s not in blob
    assert not tf.resolve().is_relative_to(Path(__file__).resolve().parents[1])   # token never inside the repo


def test_cli_upload_end_to_end(env, monkeypatch, landscape, tmp_path, capsys):
    net = FakeNet(); monkeypatch.setattr(uploader, "_http", net)
    monkeypatch.setattr("builtins.input", answers("y", "n"))
    monkeypatch.chdir(tmp_path)
    rc = cli.main([str(landscape), "--n", "2", "--window", "12", "--upload", "youtube", "--out", "o"])
    out = capsys.readouterr().out
    assert rc == 0 and "uploaded (youtube id YT_VIDEO_ID)" in out and "skipped" in out
    assert sum(1 for c in net.calls if c[0] == "PUT") == 1
    for s in SECRETS.values():
        assert s not in out
    # --upload-only reuses the rendered clips
    monkeypatch.setattr("builtins.input", answers("n", "n"))
    assert cli.main(["x", "--upload", "youtube", "--upload-only", "--out", "o"]) == 0


def test_upload_is_only_reachable_through_the_approved_clip_path():
    root = Path(__file__).resolve().parents[1] / "src" / "clipper"
    for name in ("worker.py", "jobs.py", "pipeline.py", "mcp_server.py"):      # nothing imports the uploader directly
        assert "uploader" not in (root / name).read_text().replace("upload_worker", "").replace("launch_upload", ""), name
    assert "uploader" in (root / "upload_worker.py").read_text() and "uploader" in (root / "cli.py").read_text()
    mcp = (root / "mcp_server.py").read_text()
    assert "not been approved" in mcp and 'privacy: str = "private"' in mcp   # approval gate + private default


def test_no_default_upload(landscape, tmp_path, monkeypatch):
    called = []
    monkeypatch.setattr(uploader, "run_uploads", lambda *a, **k: called.append(1))
    monkeypatch.chdir(tmp_path)
    assert cli.main([str(landscape), "--window", "12", "--n", "1", "--out", "o"]) == 0
    assert called == []


def test_upload_limit_error_is_recognised_and_blocks_further_attempts(env, clips, monkeypatch):
    res, out = clips
    sent = []

    def net(method, url, *, headers=None, data=None, timeout=0):
        sent.append(url)
        if url == "https://oauth2.googleapis.com/token":
            return 200, {}, {"access_token": "ACCESS-TOKEN-1"}
        if url.startswith(uploader.YT_UPLOAD):
            return 400, {}, {"error": {"message": "The user has exceeded the number of videos they may upload.",
                                         "errors": [{"reason": "uploadLimitExceeded"}]}}
        return 404, {}, {}
    monkeypatch.setattr(uploader, "_http", net)
    with pytest.raises(uploader.UploadLimitError, match="daily upload limit"):
        uploader.youtube_upload(out / "clip_1.mp4", res["clips"][0], "private", lambda *_: None, interactive=False)
    assert uploader.limit_hours_left() > 5
    n = len(sent)
    with pytest.raises(uploader.UploadLimitError, match="nothing was uploaded"):
        uploader.youtube_upload(out / "clip_2.mp4", res["clips"][1], "private", lambda *_: None, interactive=False)
    assert len(sent) == n                                            # no network call while the cooldown is on
    # the user says it has reset: ignore_limit lets one attempt through again
    monkeypatch.setattr(uploader, "_http", FakeNet())
    vid = uploader.youtube_upload(out / "clip_2.mp4", res["clips"][1], "private", lambda *_: None, interactive=False,
                                  ignore_limit=True)
    assert vid == "YT_VIDEO_ID"


def test_upload_clip_tool_refuses_while_the_limit_cooldown_is_on(env, monkeypatch, tmp_path):
    from clipper import mcp_server
    from clipper import upload_limit
    upload_limit.mark_hit()
    monkeypatch.setenv("CLIPPER_JOBS_DIR", str(tmp_path / "jobs"))
    r = mcp_server.upload_clip("nojob", "clip_1.mp4")
    assert "daily upload limit" in r["error"] and r["retry_after_hours"] > 5
