---
name: clipper
description: Turn a pasted YouTube or Twitch video link (or local video file) into upload-ready vertical 9:16 clips for Instagram Reels and YouTube Shorts using the clipper MCP server. Use when the user pastes a video URL and asks for clips, shorts, reels or highlights.
---

# clipper: video link -> vertical clips

Uses the `clipper` MCP server (tools: `start_clip_job`, `job_status`, `get_evidence`, `submit_judgments`,
`list_clips`, `approve_clip`). If those tools are missing, tell the user the clipper MCP server isn't
registered and point them to `docs/MCP-SETUP.md`; do not try to emulate it.

## Hard rules
- Only clip streams the user has the streamer's **written permission** to clip. If the link isn't the user's
  own and they haven't said they have permission, ask once before starting.
- **Never upload anything and never call `approve_clip` unless the user explicitly says to approve that clip.**
  `approve_clip` only records approval; there is no automatic upload.
- A live stream that hasn't ended can't be clipped. If the job fails with that message, relay it and stop.
- Never ask the user to paste API keys or passwords into the chat.

## Steps
1. **Start.** Call `start_clip_job(url, n, niche)`. Use `n=3` unless asked. Pass `niche` if the topic is clear
   (e.g. "gta6"). For streams longer than ~1 hour, ask which part matters and pass `range="START-END"`
   (seconds); analysing 8 hours at once is slow. Options: `fill=true` (crop to fill instead of blurred bars),
   `captions=true` (burn in speech captions; needs faster-whisper on the server).
2. **Poll.** Call `job_status(job_id)` every ~20-30 s. Tell the user it's running; long streams take minutes.
   Stop on `done`, `failed` or `awaiting_judgments`. On `failed`, relay `message` plainly.
3. **Judge (status `awaiting_judgments`).** For each id in `pending_candidates` call
   `get_evidence(job_id, candidate_id)`. You get a contact sheet (8 frames, left-to-right then top-to-bottom,
   in time order), signals, any transcript, and the judging rules. Look at the actual frames and judge only what
   they show; if something can't be told from the evidence, say so in `reason`. Then call
   `submit_judgments(job_id, judgments)` with one entry per candidate:
   `{id, score 1-10, reason (one line), best_start, best_end, hook (<=6 words), caption, hashtags (exactly 5,
   each starting with #), risk_flags}`. `best_start/best_end` are absolute seconds inside the candidate's
   window, 45-60 s long (or the whole window if it is shorter). `risk_flags` uses only
   spoilers / cutscenes / copyright / profanity (empty list if none seen). Be honest with scores: empty or
   static candidates should score low. If `job_status` later reports `problems_with_previous_submission`,
   fix just those entries and resubmit them.
4. **Present.** When `done`, call `list_clips(job_id)`. Summarise the report briefly (do not paste it whole):
   for each clip give the file path, time range, hook, caption, hashtags, judge score and reason, any risk flags
   (flag spoilers/copyright prominently) and the QA result. Mention anything the heuristic QA can't check
   (cropping, subject visibility) so the user previews the clip themselves. Offer to rerun with a different
   range, `n`, `fill` or `captions`.
5. **Approval.** Ask which clips (if any) the user approves. Only call `approve_clip` for clips they name.
   Make clear that nothing has been uploaded.

## Tone
Short, concrete updates. Say what you saw in the frames, not generic praise.
