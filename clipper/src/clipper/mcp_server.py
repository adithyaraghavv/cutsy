"""clipper MCP server (stdio). Jobs run in the background; no tool call blocks on a long stream.

Tools:
  start_clip_job(url, n, niche, ...)  start a background job (returns immediately)
  job_status(job_id, wait_s)          progress / what to do next (optionally waits up to 45 s)
  get_judge_sheets(job_id)            picture sheets of all candidates (Claude judges these)
  get_evidence(job_id, candidate)     one candidate's contact sheet + signals
  submit_judgments(job_id, ...)       Claude's judgments (validated), then the job renders the clips
  list_clips(job_id)                  finished clips + report
  get_clip_preview(job_id, clip)      4 frames of a finished clip as an image
  open_review(job_id)                 open the review page (playable clips) in the user's browser
  approve_clip(job_id, clip)          record the USER's approval of a clip
  upload_clip(job_id, clip, ...)      upload an APPROVED clip (default private); needs one-time `clipper --auth youtube`
stdout is the protocol channel: this module must never print.
"""
from __future__ import annotations

import json
import time
import webbrowser
from datetime import datetime, timezone
from pathlib import Path

from mcp.server.mcpserver import Image, MCPServer

from . import jobs, qa
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
                   transcribe: bool = False, facecam: str | None = None, gameplay: str | None = None,
                   min_score: float = 0.0, preset: str | None = None, allow_repeats: bool = False, use_chat: bool = True) -> dict:
    """Start a background clipping job for a video URL (or local file path). Returns a job_id immediately.

    n: clips wanted (1-40; use what the user asked for). niche: topic like "gta6" (steers hooks/hashtags). range: "START-END" seconds, to analyse
    only part of a long stream. fill: crop to fill 9:16 instead of blurred bars. captions: burn in speech captions.
    scan: analyse the WHOLE stream (audio, chat replay, most-replayed) and cut the best n moments, spread over the whole stream;
    transcribe: also read candidates' speech for reactions (needs faster-whisper).
    facecam/gameplay: "x,y,w,h" fractions of the frame for the split layout (defaults come from CLIPPER_FACECAM /
    CLIPPER_GAMEPLAY). preset: name of a saved facecam/gameplay crop for a streamer (see clipper --save-preset). min_score: leave out candidates you scored below this (e.g. 4). use_chat: false skips the live-chat replay (it can take hours on a very long stream; the scan then uses audio, motion
    and the replay heatmap). allow_repeats: by default moments already clipped
    from this stream in an earlier job are skipped so every run brings new clips; set true to include them again.
    Only use for streams the user has the streamer's written permission to clip. Live (unfinished) streams fail.
    """
    try:
        job = jobs.create(url, n=n, niche=niche, window=window, rng=range, fill=fill, captions=captions, judge="chat",
                          scan=scan, transcribe=transcribe, facecam=facecam, gameplay=gameplay, min_score=min_score, preset=preset, allow_repeats=allow_repeats, use_chat=use_chat)
        jobs.launch(job)
    except ClipperError as e:
        return _err(e)
    return {"job_id": job["id"], "status": "queued",
            "next": "Poll job_status(job_id) every ~20-30 s. Long streams can take many minutes."}


@server.tool()
def job_status(job_id: str, wait_s: float = 0.0) -> dict:
    """Status of a job: queued | running | awaiting_judgments | done | failed, plus what to do next.
    wait_s (max 45): wait this long for a status change before answering, to save polling."""
    try:
        job = jobs.refresh(job_id)
        end = time.time() + min(max(float(wait_s), 0.0), 45.0)
        first = job["status"]
        while job["status"] == first and job["status"] in jobs.ACTIVE and time.time() < end:
            time.sleep(2)
            job = jobs.refresh(job_id)
    except ClipperError as e:
        return _err(e)
    out = {"job_id": job_id, "status": job["status"], "message": job.get("message", ""),
           "recent_log": jobs.log_tail(job_id)}
    if job["status"] == "awaiting_judgments":
        out["pending_candidates"] = job["pending"]
        out["problems_with_previous_submission"] = job.get("problems", [])
        out["next"] = ("Call get_judge_sheets(job_id) to see all candidates (4 per image, ids and times labelled), judge "
                       "them, then call submit_judgments(job_id, judgments) with one entry per candidate. "
                       "(get_evidence(job_id, candidate_id) shows one candidate in more detail.)")
    elif job["status"] == "done":
        out["next"] = "Call list_clips(job_id), open_review(job_id), and summarise the clips for the user."
    elif job["status"] == "failed":
        out["next"] = "Tell the user the message above; do not retry blindly."
    else:
        out["next"] = "Still working. Poll again in 20-30 s."
    if job.get("uploads"):
        out["uploads"] = job["uploads"]
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
def get_judge_sheets(job_id: str) -> list:
    """Picture sheets of ALL pending candidates (4 per image: 8 frames each, id and time window labelled above each).
    Judge from these, then call submit_judgments. Far cheaper than looking at candidates one by one."""
    try:
        job = jobs.load(job_id)
        if job["status"] != "awaiting_judgments":
            return [json.dumps({"error": f"Job is '{job['status']}', not awaiting_judgments."})]
        d = jobs.job_path(job_id) / "out"
        sheets = sorted((d / "judge_sheets").glob("sheet_*.jpg"))
        wins = []
        for cid in job.get("pending", []):
            ev = json.loads((d / "evidence" / cid / "evidence.json").read_text(encoding="utf-8"))
            w = ev["window"]
            wins.append({"id": cid, "start_s": w["start_s"], "end_s": w["end_s"], "trim_length_s": ev["trim_length_s"],
                         "signals": ev["signals_mean_z"], "transcript": ev.get("transcript")})
        prompt = (d / "judge_prompt.md").read_text(encoding="utf-8")
    except (ClipperError, OSError, json.JSONDecodeError, KeyError) as e:
        return [json.dumps({"error": str(e)})]
    return [*[Image(path=str(p)) for p in sheets], json.dumps(wins, indent=1),
            "Judging rules (output schema and limits):\n\n" + prompt]


@server.tool()
def get_clip_preview(job_id: str, clip: str) -> list:
    """4 frames of a finished clip as one image (during the headline, just after, mid-clip, near the end)."""
    try:
        res = jobs.clips_of(jobs.load(job_id)["id"])
        c = next((x for x in (res or {}).get("clips", []) if x["file"] == clip), None)
        if c is None:
            return [json.dumps({"error": f"Unknown clip {clip!r}."})]
        d = jobs.job_path(job_id) / "out"
        sheet = qa.make_sheet(d / clip, d / "preview" / f"{Path(clip).stem}.jpg", c["end_s"] - c["start_s"], 2.6)
    except (ClipperError, OSError) as e:
        return [json.dumps({"error": str(e)})]
    return [Image(path=str(sheet)), f"{clip}: {c.get('title') or c['hook']} ({c['end_s'] - c['start_s']:.0f} s)"]


@server.tool()
def open_review(job_id: str) -> dict:
    """Open the review page (every clip playable, with title/caption/hashtags and copy buttons) in the user's browser."""
    try:
        jobs.load(job_id)
        p = jobs.job_path(job_id) / "out" / "review.html"
        if not p.is_file():
            return {"error": "No review page yet; the job has not finished."}
        opened = webbrowser.open(p.as_uri())
    except ClipperError as e:
        return _err(e)
    return {"review_page": str(p), "opened_in_browser": bool(opened)}


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
    This alone uploads nothing; upload_clip does that."""
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
            "note": "Approval recorded. Nothing has been uploaded yet; upload_clip does that, only when the user asks."}


@server.tool()
def upload_clip(job_id: str, clip: str, platform: str = "youtube", privacy: str = "private",
                retry_after_limit: bool = False) -> dict:
    """Upload a clip the USER has approved (approve_clip first). Runs in the background; check job_status 'uploads'.

    Only call this after the user explicitly asked to post THIS clip to THIS platform. privacy: private (default) |
    unlisted | public; use public only if the user said so. YouTube needs the one-time sign-in `clipper --auth youtube`.
    Each clip is uploaded at most once per platform. If YouTube's daily upload limit was hit recently this refuses
    (nothing is queued) and says how many hours to wait; pass retry_after_limit=true only if the user says the limit reset."""
    try:
        if platform not in ("youtube", "instagram"):
            return {"error": "platform must be 'youtube' or 'instagram'."}
        if privacy not in ("private", "unlisted", "public"):
            return {"error": "privacy must be private, unlisted or public."}
        if platform == "youtube" and not retry_after_limit:
            from . import upload_limit
            left = upload_limit.hours_left()
            if left:
                return {"error": f"YouTube's daily upload limit was hit recently. Try again in about {left:.1f} hours "
                                 "(or pass retry_after_limit=true if the user says it has reset). Nothing was queued.",
                        "retry_after_hours": round(left, 1)}
        job = jobs.refresh(job_id)
        res = jobs.clips_of(job_id)
        names = [c["file"] for c in (res or {}).get("clips", [])]
        if clip not in names:
            return {"error": f"Unknown clip {clip!r}. Available: {names}"}
        if clip not in job.get("approvals", {}):
            return {"error": "This clip has not been approved. Ask the user which clips to post; call approve_clip only "
                             "after they say so, then upload_clip."}
        prev = job.get("uploads", {}).get(clip, {}).get(platform)
        if prev and prev.get("status") in ("queued", "running", "done"):
            return {"already": prev, "note": "This clip is already uploaded or uploading to that platform."}
        jobs.set_upload(job_id, clip, platform, status="queued", privacy=privacy,
                        approved_at=job["approvals"][clip], queued=time.time(), error="")
        jobs.launch_upload(job_id, clip, platform, privacy, **({"ignore_limit": True} if retry_after_limit else {}))
    except ClipperError as e:
        return _err(e)
    return {"status": "queued", "clip": clip, "platform": platform, "privacy": privacy,
            "next": "Upload runs in the background (usually under a minute). Poll job_status and read 'uploads'."}


def main() -> None:
    server.run("stdio")


if __name__ == "__main__":
    main()
