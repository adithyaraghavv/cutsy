"""Background worker: python -m clipper.worker <job_dir>. Never prints user secrets; errors go to log.txt."""
from __future__ import annotations

import sys
import threading
import time
import traceback
from pathlib import Path

from . import config, jobs, pipeline
from .errors import ClipperError


def main(argv=None) -> int:
    d = Path((argv or sys.argv[1:])[0]).resolve()
    jid = d.name
    job = jobs.load(jid)
    job.update(status="running", message="", pending=[], problems=[])
    jobs.write_job(job)

    stop = threading.Event()

    def beat():
        while not stop.wait(5):
            try:
                j = jobs.load(jid)
                if j["status"] == "running":
                    j["heartbeat"] = time.time()
                    jobs.write_job(j)
            except Exception:  # noqa: BLE001 - heartbeat must never kill the worker
                pass
    threading.Thread(target=beat, daemon=True).start()

    def log(msg):
        print(msg, flush=True)

    p = job["params"]
    try:
        cfg = config.load(None)
        cfg["cache_dir"] = str(jobs.jobs_dir() / ".cache")
        res = pipeline.run(
            job["source"], n=p["n"], window=p["window"], rng=tuple(p["range"]) if p["range"] else None,
            fill=p["fill"], out=d / "out", cfg=cfg, judge_name=p["judge"], judgments=p.get("judgments"),
            shortlist=p.get("shortlist"), niche=p.get("niche"), trends_path=p.get("trends"),
            captions=p["captions"], scan=p.get("scan", False), transcribe=p.get("transcribe", False), log=log)
        job = jobs.load(jid)
        if res["status"] == "awaiting_judgments":
            job.update(status="awaiting_judgments", pending=res["pending"], problems=res.get("problems", []),
                       message=f"{len(res['pending'])} candidate(s) need judging.")
        else:
            job.update(status="done", pending=[], problems=[],
                       message=f"{len(res['clips'])} clip(s) ready.")
    except ClipperError as e:
        job = jobs.load(jid)
        job.update(status="failed", message=str(e))
    except Exception as e:  # noqa: BLE001
        print(traceback.format_exc(), flush=True)
        job = jobs.load(jid)
        job.update(status="failed", message=f"Unexpected error ({type(e).__name__}). Details are in log.txt in the job folder.")
    finally:
        stop.set()
    job["heartbeat"] = time.time()
    jobs.write_job(job)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
