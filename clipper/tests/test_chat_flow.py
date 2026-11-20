import json
import os
import stat
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pytest

from clipper import cli, jobs, mcp_server, upload_worker, uploader
from test_layout import camvideo  # noqa: F401  (fixture)
from test_mcp_jobs import SRC, judgments_for, jobs_env, wait  # noqa: F401
from test_uploader import FakeNet

CREDS = {"YOUTUBE_OAUTH_CLIENT_ID": "cid-123", "YOUTUBE_OAUTH_CLIENT_SECRET": "SECRET-CLIENT-XYZ"}


@pytest.fixture
def done_job(jobs_env, landscape):
    jid = mcp_server.start_clip_job(str(landscape), n=2, window=12, scan=True, min_score=4)["job_id"]
    st = wait(jid, {"awaiting_judgments", "failed"})
    assert st["status"] == "awaiting_judgments", st
    rows = judgments_for(jid, st["pending_candidates"])
    for k, r in enumerate(rows):
        r["title"] = f"Something wild happens in moment {k + 1} of the stream"
        r["score"] = 9 if k == 0 else 2
    mcp_server.submit_judgments(jid, rows)
    st = wait(jid, {"done", "failed"})
    assert st["status"] == "done", st
    return jid


def test_judge_sheets_preview_review_and_wait(done_job, jobs_env, landscape, monkeypatch):
    # new job to look at the judging stage
    jid = mcp_server.start_clip_job(str(landscape), n=3, window=11, scan=True)["job_id"]   # different window: no cached judgments
    wait(jid, {"awaiting_judgments"})
    out = mcp_server.get_judge_sheets(jid)
    assert type(out[0]).__name__ == "Image" and "judging" in out[-1].lower()
    info = json.loads(out[-2])
    assert info and {"id", "start_s", "end_s"} <= set(info[0])
    # finished job: preview, review page, bounded wait
    lc = mcp_server.list_clips(done_job)
    assert len(lc["clips"]) == 1 and lc["clips"][0]["hook"]                   # min_score dropped the score-2 candidate
    pv = mcp_server.get_clip_preview(done_job, lc["clips"][0]["clip"])
    assert type(pv[0]).__name__ == "Image" and Path(pv[0].path).is_file()
    assert "error" in json.loads(mcp_server.get_clip_preview(done_job, "clip_9.mp4")[0])
    opened = []
    monkeypatch.setattr(mcp_server.webbrowser, "open", lambda u: opened.append(u) or True)
    r = mcp_server.open_review(done_job)
    assert r["opened_in_browser"] and opened[0].endswith("review.html") and "Something wild" in Path(r["review_page"]).read_text()
    t0 = time.time()
    mcp_server.job_status(done_job, wait_s=999)                                # not active -> returns at once
    assert time.time() - t0 < 5


def test_upload_requires_approval_and_defaults_to_private(done_job, jobs_env, monkeypatch, tmp_path):
    clip = mcp_server.list_clips(done_job)["clips"][0]["clip"]
    for k, v in {**CREDS, "YOUTUBE_REFRESH_TOKEN": "RT-1"}.items():
        monkeypatch.setenv(k, v)
    net = FakeNet()
    monkeypatch.setattr(uploader, "_http", net)
    ran = []
    monkeypatch.setattr(jobs, "launch_upload", lambda jid, c, p, pr: ran.append(upload_worker.run(jobs.job_path(jid), c, p, pr, log=lambda *_: None)))
    # 1) not approved -> nothing happens
    assert "not been approved" in mcp_server.upload_clip(done_job, clip)["error"]
    assert ran == [] and net.calls == []
    # 2) bad arguments
    mcp_server.approve_clip(done_job, clip)
    assert "platform" in mcp_server.upload_clip(done_job, clip, platform="tiktok")["error"]
    assert "privacy" in mcp_server.upload_clip(done_job, clip, privacy="everyone")["error"]
    assert "Unknown clip" in mcp_server.upload_clip(done_job, "clip_9.mp4")["error"]
    # 3) approved -> uploads once, private by default
    r = mcp_server.upload_clip(done_job, clip)
    assert r["status"] == "queued" and r["privacy"] == "private"
    up = mcp_server.job_status(done_job)["uploads"][clip]["youtube"]
    assert up["status"] == "done" and up["id"] == "YT_VIDEO_ID" and up["url"] == "https://youtu.be/YT_VIDEO_ID"
    start = next(c for c in net.calls if c[1].startswith(uploader.YT_UPLOAD))
    meta = json.loads(start[3])
    assert meta["status"]["privacyStatus"] == "private" and "Something wild" in meta["snippet"]["title"]
    # 4) never twice
    assert "already" in mcp_server.upload_clip(done_job, clip)
    assert sum(1 for c in net.calls if c[0] == "PUT") == 1


def test_upload_without_one_time_signin_fails_friendly_without_network(done_job, jobs_env, monkeypatch, tmp_path):
    clip = mcp_server.list_clips(done_job)["clips"][0]["clip"]
    for k, v in CREDS.items():
        monkeypatch.setenv(k, v)
    monkeypatch.delenv("YOUTUBE_REFRESH_TOKEN", raising=False)
    monkeypatch.setenv("CLIPPER_CONFIG_DIR", str(tmp_path / "empty"))
    net = FakeNet()
    monkeypatch.setattr(uploader, "_http", net)
    mcp_server.approve_clip(done_job, clip)
    res = upload_worker.run(jobs.job_path(done_job), clip, "youtube", "private", log=lambda *_: None)
    assert res["status"] == "failed" and "clipper --auth youtube" in res["error"] and net.calls == []


def test_auth_command(monkeypatch, tmp_path, capsys):
    monkeypatch.delenv("YOUTUBE_OAUTH_CLIENT_ID", raising=False)
    monkeypatch.delenv("YOUTUBE_OAUTH_CLIENT_SECRET", raising=False)
    assert cli.main(["--auth", "youtube"]) == 2 and "YOUTUBE_OAUTH_CLIENT_ID" in capsys.readouterr().err
    assert cli.main([]) == 2 and "URL or file" in capsys.readouterr().err
    for k, v in CREDS.items():
        monkeypatch.setenv(k, v)
    monkeypatch.setenv("CLIPPER_CONFIG_DIR", str(tmp_path / "cfg"))
    monkeypatch.setattr(uploader, "_wait_for_code", lambda srv, state, timeout=0: "CODE")
    monkeypatch.setattr(uploader, "_http", lambda *a, **k: (200, {}, {"access_token": "AT", "refresh_token": "NEW-REFRESH"}))
    assert cli.main(["--auth", "youtube"]) == 0
    tf = tmp_path / "cfg" / "youtube_token.json"
    assert json.loads(tf.read_text())["refresh_token"] == "NEW-REFRESH" and stat.S_IMODE(tf.stat().st_mode) == 0o600
    out = capsys.readouterr()
    assert "NEW-REFRESH" not in out.out + out.err and "SECRET-CLIENT-XYZ" not in out.out + out.err


def test_start_job_layout_params_validation_and_env_defaults(jobs_env, landscape, monkeypatch):
    assert "facecam" in mcp_server.start_clip_job(str(landscape), facecam="0.9,0.9,0.5,0.5")["error"]
    assert "facecam" in mcp_server.start_clip_job(str(landscape), facecam="a,b,c,d")["error"]
    monkeypatch.setenv("CLIPPER_FACECAM", "0.05,0,0.20,0.30")
    monkeypatch.setenv("CLIPPER_GAMEPLAY", "0.30,0.05,0.70,0.90")
    r = mcp_server.start_clip_job(str(landscape), n=1, window=12)
    p = jobs.load(r["job_id"])["params"]
    assert p["facecam"] == [0.05, 0.0, 0.2, 0.3] and p["gameplay"] == [0.3, 0.05, 0.7, 0.9]


def test_worker_applies_facecam_split_layout(jobs_env, camvideo):
    jid = mcp_server.start_clip_job(str(camvideo), n=1, window=10, facecam="0.70,0.55,0.30,0.45",
                                    gameplay="0,0,0.69,1.0")["job_id"]
    st = wait(jid, {"awaiting_judgments", "failed"})
    assert st["status"] == "awaiting_judgments", st
    mcp_server.submit_judgments(jid, judgments_for(jid, st["pending_candidates"]))
    assert wait(jid, {"done", "failed"})["status"] == "done"
    clip = Path(mcp_server.list_clips(jid)["clips"][0]["path"])
    r = subprocess.run(["ffmpeg", "-v", "error", "-ss", "3", "-i", str(clip), "-frames:v", "1", "-vf", "scale=270:480",
                        "-pix_fmt", "rgb24", "-f", "rawvideo", "-"], capture_output=True)
    f = np.frombuffer(r.stdout, np.uint8).reshape(480, 270, 3).astype(int)
    red = lambda a: ((a[..., 0] > 180) & (a[..., 1] < 80) & (a[..., 2] < 80)).mean()
    assert red(f[:int(480 * 0.40)]) > 0.8 and red(f[int(480 * 0.60):]) == 0      # facecam top, game-only bottom


def test_stdio_exposes_all_tools(jobs_env):
    import asyncio
    from mcp.client.client import Client
    from mcp.client.stdio import StdioServerParameters
    env = {**os.environ, "PYTHONPATH": SRC, "CLIPPER_JOBS_DIR": str(jobs_env)}

    async def go():
        async with Client(StdioServerParameters(command=sys.executable, args=["-m", "clipper.mcp_server"], env=env)) as c:
            return {t.name for t in (await c.list_tools()).tools}
    names = asyncio.run(go())
    assert {"start_clip_job", "job_status", "get_judge_sheets", "get_evidence", "submit_judgments", "list_clips",
            "get_clip_preview", "open_review", "approve_clip", "upload_clip"} <= names
