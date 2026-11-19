import asyncio
import json
import os
import stat
import sys
import time
from pathlib import Path

import pytest

from clipper import jobs, mcp_server
from clipper.errors import ClipperError

SRC = str(Path(__file__).resolve().parents[1] / "src")


@pytest.fixture
def jobs_env(tmp_path, monkeypatch):
    d = tmp_path / "jobs"
    monkeypatch.setenv("CLIPPER_JOBS_DIR", str(d))
    monkeypatch.setenv("PYTHONPATH", SRC)
    monkeypatch.chdir(tmp_path)
    return d


def wait(job_id, statuses, timeout=150):
    end = time.time() + timeout
    while time.time() < end:
        st = mcp_server.job_status(job_id)
        if st.get("status") in statuses:
            return st
        time.sleep(1)
    raise AssertionError(f"timeout waiting for {statuses}; last={st}")


def judgments_for(job_id, pending, **over):
    rows = []
    for k, cid in enumerate(pending):
        ev = json.loads((jobs.job_path(job_id) / "out" / "evidence" / cid / "evidence.json").read_text())
        w = ev["window"]
        rows.append({"id": cid, "score": 9 - k, "reason": "Good action.", "best_start": w["start_s"],
                     "best_end": w["end_s"], "hook": f"HOOK NUMBER {k + 1}", "caption": "cap",
                     "hashtags": ["#a", "#b", "#c", "#d", "#e"], "risk_flags": [], **over})
    return rows


def test_full_job_lifecycle_in_background(jobs_env, landscape):
    r = mcp_server.start_clip_job(str(landscape), n=2, niche="gta6", window=12)
    jid = r["job_id"]
    assert r["status"] == "queued"                    # returned immediately, work is in a separate process
    st = wait(jid, {"awaiting_judgments", "failed"})
    assert st["status"] == "awaiting_judgments", st
    pending = st["pending_candidates"]
    assert len(pending) >= 2 and "get_evidence" in st["next"]

    ev = mcp_server.get_evidence(jid, pending[0])
    kinds = [type(x).__name__ for x in ev]
    assert kinds[0] == "Image" and "window" in ev[1] and "Clip judging task" in ev[2]
    assert "error" in mcp_server.get_evidence(jid, "c99-nope")[0]

    bad = judgments_for(jid, pending)
    bad[0]["hook"] = "WAY TOO MANY WORDS IN THIS HOOK OK"
    assert mcp_server.submit_judgments(jid, bad)["status"] == "queued"
    st = wait(jid, {"awaiting_judgments", "done", "failed"})
    assert st["status"] == "awaiting_judgments" and st["pending_candidates"] == [pending[0]]
    assert any("hook must be 1-6 words" in p for p in st["problems_with_previous_submission"])

    fixed = judgments_for(jid, [pending[0]])
    assert mcp_server.submit_judgments(jid, fixed)["status"] == "queued"
    st = wait(jid, {"done", "failed"})
    assert st["status"] == "done", st

    lc = mcp_server.list_clips(jid)
    assert len(lc["clips"]) == 2 and Path(lc["clips"][0]["path"]).is_file()
    assert lc["clips"][0]["approved"] is False and "report" in lc and "Hook" in lc["report"]
    assert (jobs.job_path(jid) / "out" / "report.md").is_file()

    name = lc["clips"][0]["clip"]
    ap = mcp_server.approve_clip(jid, name)
    assert ap["approved"] == name and ap["uploaded"] is False
    assert mcp_server.list_clips(jid)["clips"][0]["approved"] is True
    assert "error" in mcp_server.approve_clip(jid, "clip_99.mp4")
    # job state persists on disk (a restarted server sees the same thing)
    assert jobs.load(jid)["approvals"][name]


def test_worker_restart_resumes_stale_job(jobs_env, landscape):
    job = jobs.create(str(landscape), n=1, window=12)
    job.update(status="running", heartbeat=time.time() - 3600)      # worker "died" long ago
    jobs.write_job(job)
    st = mcp_server.job_status(job["id"])                            # refresh relaunches it
    assert jobs.load(job["id"])["resumes"] == 1
    st = wait(job["id"], {"awaiting_judgments", "failed"})
    assert st["status"] == "awaiting_judgments"


def test_live_stream_job_fails_with_friendly_message(jobs_env, tmp_path, monkeypatch):
    bindir = tmp_path / "bin"
    bindir.mkdir()
    fake = bindir / "yt-dlp"
    fake.write_text("#!/bin/sh\necho '{\"id\":\"abc\",\"is_live\":true,\"title\":\"x\"}'\n")
    fake.chmod(fake.stat().st_mode | stat.S_IEXEC)
    monkeypatch.setenv("PATH", f"{bindir}{os.pathsep}{os.environ['PATH']}")
    r = mcp_server.start_clip_job("https://www.youtube.com/watch?v=abc", n=1)
    st = wait(r["job_id"], {"failed", "awaiting_judgments", "done"})
    assert st["status"] == "failed" and "live" in st["message"] and "Traceback" not in st["message"]


def test_input_validation_and_safety(jobs_env):
    assert "not found" in mcp_server.start_clip_job("/no/such/file.mp4")["error"]
    assert "error" in mcp_server.start_clip_job("")
    assert "error" in mcp_server.job_status("../../etc/passwd")
    assert "error" in mcp_server.job_status("0" * 12)
    assert "error" in mcp_server.list_clips("0" * 12)
    with pytest.raises(ClipperError):
        jobs.job_path("../x")


def test_submit_requires_awaiting_status(jobs_env, landscape):
    job = jobs.create(str(landscape), n=1)
    assert "not awaiting" in mcp_server.submit_judgments(job["id"], [{"id": "x"}])["error"].replace("_", " ").replace("awaiting judgments", "awaiting")


def test_stdio_protocol_end_to_end(jobs_env, landscape):
    from mcp.client.client import Client
    from mcp.client.stdio import StdioServerParameters

    env = {**os.environ, "PYTHONPATH": SRC, "CLIPPER_JOBS_DIR": str(jobs_env)}
    params = StdioServerParameters(command=sys.executable, args=["-m", "clipper.mcp_server"], env=env)

    def payload(res):
        sc = getattr(res, "structured_content", None)
        if sc:
            return sc.get("result", sc)
        return json.loads(res.content[0].text)

    async def go():
        async with Client(params) as c:
            tools = {t.name for t in (await c.list_tools()).tools}
            assert {"start_clip_job", "job_status", "list_clips", "approve_clip",
                    "get_evidence", "submit_judgments"} <= tools
            r = payload(await c.call_tool("start_clip_job", {"url": str(landscape), "n": 1, "window": 12}))
            jid = r["job_id"]
            for _ in range(150):
                st = payload(await c.call_tool("job_status", {"job_id": jid}))
                if st["status"] in ("awaiting_judgments", "failed"):
                    break
                await asyncio.sleep(1)
            assert st["status"] == "awaiting_judgments", st
            ev = await c.call_tool("get_evidence", {"job_id": jid, "candidate_id": st["pending_candidates"][0]})
            assert any(getattr(x, "type", "") == "image" for x in ev.content)   # image comes over the protocol
            rows = judgments_for(jid, st["pending_candidates"])
            await c.call_tool("submit_judgments", {"job_id": jid, "judgments": rows})
            for _ in range(150):
                st = payload(await c.call_tool("job_status", {"job_id": jid}))
                if st["status"] in ("done", "failed"):
                    break
                await asyncio.sleep(1)
            assert st["status"] == "done", st
            lc = payload(await c.call_tool("list_clips", {"job_id": jid}))
            assert len(lc["clips"]) == 1
            return jid

    jid = asyncio.run(go())
    assert (jobs.job_path(jid) / "out" / "clip_1.mp4").is_file()
