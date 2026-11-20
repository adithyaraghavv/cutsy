"""clipper MCP server (stdio). Jobs run in the background; no tool call blocks on a long stream.

Tools:
  start_clip_job(url, n, niche)   start a background job (returns immediately)
  job_status(job_id)              progress / what to do next
  get_evidence(job_id, candidate) contact sheet + signals for a candidate (Claude judges these)
  submit_judgments(job_id, ...)   Claude's judgments (validated), then the job renders the clips
  list_clips(job_id)              finished clips + report
  approve_clip(job_id, clip)      record YOUR approval of a clip (never uploads anything)
stdout is the protocol channel: this module must never print.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from mcp.server.mcpserver import Image, MCPServer

from . import jobs
from .errors import ClipperError

server = MCPServer(
    "clipper",
    instructions=("Turns a long YouTube/Twitch VOD or local video into vertical 9:16 clips. Only for videos the user "
                  "has permission to clip. Start a job, poll job_status, judge candidates with get_evidence + "
                  "submit_judgments, then list_clips. Never approve a clip unless the user explicitly says so."),
)


def _err(e: ClipperError) -> dict:
    return {"error": str(e)}


@server.tool()
def start_clip_job(url: str, n: int = 3, niche: str | None = None, range: str | None = None,
                   window: float = 55.0, fill: bool = False, captions: bool = False, scan: bool = False,
                   transcribe: bool = False) -> dict:
    """Start a background clipping job for a video URL (or local file path). Returns a job_id immediately.

    n: clips wanted (1-10). niche: topic like "gta6" (steers hooks/hashtags). range: "START-END" seconds, to analyse
    only part of a long stream. fill: crop to fill 9:16 instead of blurred bars. captions: burn in speech captions.
    scan: analyse the WHOLE stream (audio, chat replay, most-replayed) and cut the best n moments (use n=10-15);
    transcribe: also read candidates' speech for reactions (needs faster-whisper).
    Only use for streams the user has the streamer's written permission to clip. Live (unfinished) streams fail.
    """
    try:
        job = jobs.create(url, n=n, niche=niche, window=window, rng=range, fill=fill, captions=captions, judge="chat",
                          scan=scan, transcribe=transcribe)
        jobs.launch(job)
    except ClipperError as e:
        return _err(e)
    return {"job_id": job["id"], "status": "queued",
            "next": "Poll job_status(job_id) every ~20-30 s. Long streams can take many minutes."}


@server.tool()
def job_status(job_id: str) -> dict:
    """Status of a job: queued | running | awaiting_judgments | done | failed, plus what to do next."""
    try:
        job = jobs.refresh(job_id)
    except ClipperError as e:
        return _err(e)
    out = {"job_id": job_id, "status": job["status"], "message": job.get("message", ""),
           "recent_log": jobs.log_tail(job_id)}
    if job["status"] == "awaiting_judgments":
        out["pending_candidates"] = job["pending"]
        out["problems_with_previous_submission"] = job.get("problems", [])
        out["next"] = ("For each pending candidate id call get_evidence(job_id, candidate_id), look at the contact "
                       "sheet, then call submit_judgments(job_id, judgments) with one entry per candidate.")
    elif job["status"] == "done":
        out["next"] = "Call list_clips(job_id) and summarise the report for the user."
    elif job["status"] == "failed":
        out["next"] = "Tell the user the message above; do not retry blindly."
    else:
        out["next"] = "Still working. Poll again in 20-30 s."
    return out


@server.tool()
def get_evidence(job_id: str, candidate_id: str) -> list:
    """Evidence for one pending candidate: a contact sheet image (8 frames in time order) and its signals JSON."""
    try:
        job = jobs.load(job_id)
        if candidate_id not in job.get("pending", []):
            return [json.dumps({"error": f"{candidate_id} is not a pending candidate of this job.",
                                "pending": job.get("pending", [])})]
        d = jobs.job_path(job_id) / "out" / "evidence" / candidate_id
        ev = json.loads((d / "evidence.json").read_text(encoding="utf-8"))
        prompt = (jobs.job_path(job_id) / "out" / "judge_prompt.md").read_text(encoding="utf-8")
    except (ClipperError, OSError, json.JSONDecodeError) as e:
        return [json.dumps({"error": str(e)})]
    return [Image(path=str(d / "contact.jpg")), json.dumps(ev, indent=2),
            "Judging rules (output schema and limits) are in this prompt:\n\n" + prompt]


@server.tool()
def submit_judgments(job_id: str, judgments: list[dict]) -> dict:
    """Submit judgments for pending candidates, then the job renders the clips.

    Each item: {id, score 1-10, reason (one line), best_start, best_end (absolute seconds, inside the candidate
    window, 45-60 s long or the whole window if shorter), hook (<=6 words), caption, hashtags (exactly 5, start
    with #), risk_flags (subset of spoilers/cutscenes/copyright/profanity)}. Invalid entries are reported by
    job_status and can be resubmitted individually.
    """
    try:
        job = jobs.refresh(job_id)
        if job["status"] != "awaiting_judgments":
            return {"error": f"Job is '{job['status']}', not awaiting_judgments."}
        if not isinstance(judgments, list) or not judgments:
            return {"error": "judgments must be a non-empty list."}
        f = jobs.job_path(job_id) / "out" / "judgments.json"
        have = {r.get("id"): r for r in json.loads(f.read_text(encoding="utf-8"))} if f.is_file() else {}
        for r in judgments:
            if isinstance(r, dict) and r.get("id"):
                have[r["id"]] = r
        f.write_text(json.dumps(list(have.values()), indent=2), encoding="utf-8")
        job["params"]["judgments"] = str(f)
        jobs.write_job(job)
        jobs.launch(job)
    except ClipperError as e:
        return _err(e)
    return {"status": "queued", "accepted_entries": len(judgments),
            "next": "Poll job_status; invalid entries (if any) come back as problems."}


@server.tool()
def list_clips(job_id: str) -> dict:
    """Finished clips (paths, times, hook, caption, hashtags, judge score/reason, QA) and the report text."""
    try:
        job = jobs.refresh(job_id)
        res = jobs.clips_of(job_id)
    except ClipperError as e:
        return _err(e)
    if not res or job["status"] != "done":
        return {"error": f"No finished clips yet (job is '{job['status']}')."}
    out_dir = jobs.job_path(job_id) / "out"
    clips = []
    for c in res["clips"]:
        j = c.get("judge") or {}
        clips.append({"clip": c["file"], "path": str(out_dir / c["file"]), "start_s": c["start_s"], "end_s": c["end_s"],
                      "hook": c["hook"], "caption": j.get("caption"), "hashtags": j.get("hashtags"),
                      "judge_score": j.get("score"), "reason": j.get("reason"), "risk_flags": j.get("risk_flags"),
                      "qa_passed": (c["qa"][-1]["passed"] if c.get("qa") else None),
                      "approved": c["file"] in job.get("approvals", {})})
    rep = out_dir / "report.md"
    return {"job_id": job_id, "title": res.get("title"), "clips": clips,
            "report_path": str(rep), "report": rep.read_text(encoding="utf-8")[:8000] if rep.is_file() else ""}


@server.tool()
def approve_clip(job_id: str, clip: str) -> dict:
    """Record that the USER explicitly approved a clip (e.g. "clip_1.mp4"). Call only after the user says so.
    This does not upload anything; uploading is a separate, opt-in module that is not part of this server."""
    try:
        job = jobs.refresh(job_id)
        res = jobs.clips_of(job_id)
        names = [c["file"] for c in (res or {}).get("clips", [])]
        if clip not in names:
            return {"error": f"Unknown clip {clip!r}. Available: {names}"}
        job.setdefault("approvals", {})[clip] = datetime.now(timezone.utc).isoformat()
        jobs.write_job(job)
    except ClipperError as e:
        return _err(e)
    return {"approved": clip, "uploaded": False,
            "note": "Approval recorded. Nothing has been uploaded; clipper never uploads on its own."}


def main() -> None:
    server.run("stdio")


if __name__ == "__main__":
    main()
