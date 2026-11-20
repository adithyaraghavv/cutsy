---
name: clipper
description: Turn a pasted YouTube or Twitch video link (or local video file) into vertical 9:16 Shorts/Reels clips with real headlines and captions, show them for approval, and upload the approved ones to YouTube. Use when the user pastes a video URL and wants clips, shorts, reels or highlights.
---

# clipper: link -> best moments -> your approval -> upload

Uses the `clipper` MCP server (tools: `start_clip_job`, `job_status`, `get_judge_sheets`, `get_evidence`,
`submit_judgments`, `list_clips`, `get_clip_preview`, `open_review`, `approve_clip`, `upload_clip`).
If those tools are missing, tell the user the clipper MCP server isn't registered and point them to
`docs/MCP-SETUP.md`; do not try to emulate it.

## Hard rules
- Only clip streams the user has the streamer's **written permission** to clip. If the link isn't clearly the user's
  own and they haven't said they have permission, ask once before starting.
- **Never call `approve_clip` or `upload_clip` unless the user has explicitly said which clip(s) to post and where.**
  Approval is a clear statement such as "post clips 1 and 4 to YouTube". Silence, "looks good" or "nice" is not approval.
- Upload **private** unless the user explicitly asks for unlisted or public. Never upload a clip twice.
- A live stream that hasn't ended can't be clipped; if the job fails with that message, relay it and stop.
- Never ask the user to paste API keys, tokens or passwords into the chat.
- Judge only what the frames/transcript show. If you can't tell what is happening, say so and score low; never invent a story.

## Workflow
1. **Start.** Call `start_clip_job(url, n, scan=true, window, min_score=4, niche=<topic if clear, e.g. "gta6">)`.
   **Use exactly the number of clips and clip length the user asked for** (e.g. "6 clips of 45 seconds" -> `n=6, window=45`);
   only when they say nothing use the defaults `n=12, window=40`. Never replace a number the user gave.
   For one short video (under ~10 min) use `scan=false, n=3, window=55`. Layout (facecam/gameplay crop) comes from the
   server's defaults; only pass `facecam`/`gameplay` if the user gives numbers. Optional: `transcribe=true` (reads speech;
   needs faster-whisper on the user's PC). Tell the user it is running: a 9-hour stream can take an hour or more.
2. **Wait.** Call `job_status(job_id, wait_s=40)` repeatedly (each call waits up to 40 s). Keep the user posted in one
   short line every few polls. Stop on `awaiting_judgments`, `done` or `failed`. On `failed`, relay `message` plainly.
3. **Judge** (status `awaiting_judgments`). Call `get_judge_sheets(job_id)`: you get a few images (4 candidates each, id and
   time window labelled; 8 frames per candidate in time order) plus the rules. For EVERY candidate write one entry and call
   `submit_judgments(job_id, judgments)`:
   `{id, score 1-10, reason (one line), best_start, best_end, title, hook (<=6 words), caption, hashtags (exactly 5 with #),
   risk_flags}`.
   - Copy each `id` exactly. `best_start`/`best_end` = the candidate's window (when the window is 40 s that is the whole
     window; otherwise 45-60 s inside it).
   - `title` = the headline burned into the video: ONE sentence, 10-22 words, saying concretely what happens, 3-5 key words in
     CAPS, no emoji. `caption` may contain emoji. `hook` is a short fallback.
   - Score 1-3 and say why for: NO SIGNAL / pause menus / loading or settings screens / black or frozen pictures / nothing
     happening / adult or sexual content (not postable) / anything you can't make sense of. Real action, reactions, funny or
     dramatic moments score higher. `risk_flags` only from spoilers, cutscenes, copyright, profanity.
   If `job_status` later lists `problems_with_previous_submission`, fix those entries and resubmit just them.
4. **Present.** When `done`: `list_clips(job_id)`, then `open_review(job_id)` (opens the playable review page in the user's
   browser) and `get_clip_preview` for the best 3-4 clips so the user sees how they look. Summarise per clip: file, time in
   the source, on-video title, caption, hashtags, score and reason, risk flags (call out spoilers/copyright). Say honestly
   what you could not check (audio, exact cropping).
5. **Approval.** Ask which clips (if any) to post, to which platform (YouTube Shorts is set up) and the visibility
   (default private). Wait for an explicit answer.
6. **Upload.** For each clip the user named: `approve_clip(job_id, clip)` then `upload_clip(job_id, clip,
   platform="youtube", privacy=<what they chose, default "private">)`. Then poll `job_status(job_id, wait_s=30)` and read
   `uploads` until `done` (report the URL) or `failed` (relay `error`; if it says to run `clipper --auth youtube`, tell the
   user to run that once in a terminal). Tell the user exactly what was uploaded and where.

## Tone
Short, concrete updates. Say what you saw in the frames, not generic praise.
