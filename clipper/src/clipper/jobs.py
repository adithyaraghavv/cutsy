"""On-disk job store + detached background workers (jobs survive MCP server restarts).

Layout:  <jobs_dir>/<job_id>/job.json   state (status, params, approvals, heartbeat)
                            /log.txt    worker output
                            /out/       clipper output (clips, report.md, evidence/, state.json)
A job is run by `python -m clipper.worker <job_dir>` in its own process, so a tool call never
blocks and a crashed/restarted server does not lose work. If a worker's heartbeat goes stale,
the job is relaunched (the pipeline resumes from its cached download and analysis).
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

from . import ingest
from .errors import ClipperError

ID_RE = re.compile(r"^[0-9a-f]{12}$")
STALE_S = 45
MAX_RESUMES = 3
ACTIVE = {"queued", "running"}


def jobs_dir() -> Path:
    d = Path(os.environ.get("CLIPPER_JOBS_DIR", "jobs")).expanduser().resolve()
    d.mkdir(parents=True, exist_ok=True)
    return d


def job_path(job_id: str) -> Path:
    if not isinstance(job_id, str) or not ID_RE.match(job_id):
        raise ClipperError(f"Invalid job id {job_id!r}.")
    return jobs_dir() / job_id


def _now() -> float:
    return time.time()


WRITE_TRIES = 12
MAX_CLIPS = 40


def write_job(job: dict) -> None:
    d = job_path(job["id"])
    tmp = d / f"job.json.{os.getpid()}.tmp"
    job["updated"] = _now()
    tmp.write_text(json.dumps(job, indent=2), encoding="utf-8")
    for attempt in range(WRITE_TRIES):
        try:
            os.replace(tmp, d / "job.json")
            return
        except PermissionError:
            # Windows refuses the replace while another process (a worker, the MCP server) has job.json open.
            if attempt == WRITE_TRIES - 1:
                tmp.unlink(missing_ok=True)
                raise
            time.sleep(0.05 * (attempt + 1))


def load(job_id: str) -> dict:
    p = job_path(job_id) / "job.json"
    if not p.is_file():
        raise ClipperError(f"No job with id {job_id}.")
    for _ in range(5):          # tolerate a concurrent atomic replace
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            time.sleep(0.05)
    raise ClipperError(f"Job {job_id} state is unreadable.")


def create(source: str, *, n=3, niche=None, window=55.0, rng=None, fill=False, captions=False,
           judge="chat", shortlist=None, scan=False, transcribe=False,
           facecam=None, gameplay=None, min_score=0.0, preset=None, allow_repeats=False, use_chat=True) -> dict:
    source = (source or "").strip()
    if not source:
        raise ClipperError("Give a video URL or a local file path.")
    if not ingest.is_url(source) and not Path(source).expanduser().is_file():
        raise ClipperError(f"File not found: {source} (and it isn't an http(s) URL).")
    if not 1 <= int(n) <= MAX_CLIPS:
        raise ClipperError(f"n must be between 1 and {MAX_CLIPS}.")
    if not 5 <= float(window) <= 120:
        raise ClipperError("window must be between 5 and 120 seconds.")
    r = ingest.parse_range(rng) if rng else None
    if judge not in ("chat", "heuristic"):
        raise ClipperError("judge must be 'chat' or 'heuristic'.")
    trends = os.environ.get("CLIPPER_TRENDS") or ("trends.json" if Path("trends.json").is_file() else None)
    if preset:
        from . import presets
        pr = presets.get(preset)
        facecam, gameplay = facecam or pr["facecam"], gameplay or pr.get("gameplay")
    facecam = _rect(facecam or os.environ.get("CLIPPER_FACECAM"), "facecam")
    gameplay = _rect(gameplay or os.environ.get("CLIPPER_GAMEPLAY"), "gameplay")
    jid = uuid.uuid4().hex[:12]
    (jobs_dir() / jid / "out").mkdir(parents=True)
    job = {"id": jid, "source": ingest.is_url(source) and source or str(Path(source).expanduser().resolve()),
           "params": {"n": int(n), "niche": niche or None, "window": float(window),
                      "range": list(r) if r else None, "fill": bool(fill), "captions": bool(captions),
                      "judge": judge, "shortlist": shortlist, "scan": bool(scan), "transcribe": bool(transcribe), "facecam": facecam, "gameplay": gameplay,
                      "min_score": float(min_score), "allow_repeats": bool(allow_repeats), "chat": bool(use_chat), "trends": str(Path(trends).resolve()) if trends else None,
                      "judgments": None},
           "status": "queued", "message": "", "pending": [], "problems": [], "approvals": {},
           "resumes": 0, "created": _now(), "heartbeat": _now()}
    write_job(job)
    return job


def _rect(v, name: str):
    """'x,y,w,h' (fractions of the frame) -> [x, y, w, h] or None."""
    if not v:
        return None
    try:
        vals = [float(x) for x in (v.split(",") if isinstance(v, str) else v)]
        x, y, w, h = vals
    except (ValueError, TypeError):
        raise ClipperError(f"{name} must be four numbers x,y,w,h like 0.05,0,0.20,0.30") from None
    if not (0 <= x < 1 and 0 <= y < 1 and 0.05 <= w <= 1 and 0.05 <= h <= 1 and x + w <= 1.001 and y + h <= 1.001):
        raise ClipperError(f"{name} x,y,w,h must be fractions between 0 and 1 and stay inside the frame.")
    return vals


def set_upload(job_id: str, clip: str, platform: str, **fields) -> dict:
    job = load(job_id)
    job.setdefault("uploads", {}).setdefault(clip, {}).setdefault(platform, {}).update(fields)
    write_job(job)
    return job["uploads"][clip][platform]


def launch_upload(job_id: str, clip: str, platform: str, privacy: str) -> None:
    """Upload in its own detached process (so a tool call never waits on a big upload)."""
    d = job_path(job_id)
    env = dict(os.environ)
    env["PYTHONPATH"] = str(Path(__file__).resolve().parent.parent) + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    env["CLIPPER_JOBS_DIR"] = str(jobs_dir())
    kw: dict = {}
    if sys.platform.startswith("win"):
        kw["creationflags"] = 0x00000008 | 0x00000200
    else:
        kw["start_new_session"] = True
    with open(d / "upload_log.txt", "ab") as log:
        subprocess.Popen([sys.executable, "-m", "clipper.upload_worker", str(d), clip, platform, privacy], cwd=str(d),
                         env=env, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, **kw)


def launch(job: dict) -> None:
    d = job_path(job["id"])
    env = dict(os.environ)
    pkg_root = str(Path(__file__).resolve().parent.parent)
    env["PYTHONPATH"] = pkg_root + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    env["CLIPPER_JOBS_DIR"] = str(jobs_dir())
    kw: dict = {}
    if sys.platform.startswith("win"):
        kw["creationflags"] = 0x00000008 | 0x00000200   # DETACHED_PROCESS | CREATE_NEW_PROCESS_GROUP
    else:
        kw["start_new_session"] = True
    job["status"], job["heartbeat"] = "queued", _now()
    write_job(job)
    with open(d / "log.txt", "ab") as log:
        subprocess.Popen([sys.executable, "-m", "clipper.worker", str(d)], cwd=str(d), env=env,
                         stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, **kw)


def refresh(job_id: str) -> dict:
    """Load a job; relaunch it if its worker vanished (e.g. machine/server restart)."""
    job = load(job_id)
    if job["status"] in ACTIVE and _now() - job.get("heartbeat", 0) > STALE_S:
        if job.get("resumes", 0) < MAX_RESUMES:
            job["resumes"] = job.get("resumes", 0) + 1
            job["message"] = f"worker stopped; resuming (attempt {job['resumes']})"
            launch(job)
        else:
            job.update(status="failed", message="The worker kept stopping unexpectedly. See log.txt in the job folder.")
            write_job(job)
    return job


def log_tail(job_id: str, n: int = 6) -> list[str]:
    p = job_path(job_id) / "log.txt"
    if not p.is_file():
        return []
    lines = [l for l in p.read_text(encoding="utf-8", errors="replace").splitlines() if l.strip()]
    return lines[-n:]


def clips_of(job_id: str) -> dict | None:
    p = job_path(job_id) / "out" / "clips.json"
    if not p.is_file():
        return None
    return json.loads(p.read_text(encoding="utf-8"))
