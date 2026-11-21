from __future__ import annotations

import json
import shlex
from pathlib import Path

from . import candidates as cand
from . import ledger
from . import captions as cap_mod, evidence, ingest, judge as judge_mod, qa as qa_mod, report, review, signals
from . import scan as scan_mod
from . import trends as trends_mod
from .errors import ClipperError
from .render import effective_layout, render_clip, title_band


def _cand_to_dict(c: cand.Candidate) -> dict:
    return {"start_s": c.start_s, "end_s": c.end_s, "score": c.score, "peak_s": c.peak_s, "signals": c.breakdown}


def _cand_from_dict(d: dict) -> cand.Candidate:
    return cand.Candidate(start_s=d["start_s"], end_s=d["end_s"], score=d["score"],
                          breakdown=d["signals"], peak_s=d.get("peak_s", d["start_s"]))


def _load_state(out: Path) -> dict | None:
    p = out / "state.json"
    if p.is_file():
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None
    return None


def _save_state(out: Path, state: dict) -> None:
    (out / "state.json").write_text(json.dumps(state, indent=2), encoding="utf-8")


def _resume_cmd(source, n, window, rng, fill, niche, out, judge_name, scan=False, transcribe=False, captions=False, chat=True, cfg=None, min_score=0.0) -> str:
    parts = ["clipper", shlex.quote(source), "--n", str(n), "--window", f"{window:g}"]
    if rng:
        parts += ["--range", f"{rng[0]:g}-{rng[1]:g}"]
    if niche:
        parts += ["--niche", shlex.quote(niche)]
    if fill:
        parts.append("--fill")
    if scan:
        parts.append("--scan")
    if transcribe:
        parts.append("--transcribe")
    if captions:
        parts.append("--captions")
    if not chat:
        parts.append("--no-chat")
    if min_score:
        parts += ["--min-score", f"{min_score:g}"]
    r = (cfg or {}).get("render", {})
    if r.get("facecam"):
        parts += ["--facecam", ",".join(f"{v:g}" for v in r["facecam"])]
        if r.get("gameplay"):
            parts += ["--gameplay", ",".join(f"{v:g}" for v in r["gameplay"])]
    from .config import DEFAULTS
    cd = (cfg or {}).get("cache_dir")
    if cd and cd != DEFAULTS["cache_dir"]:
        parts += ["--cache-dir", shlex.quote(str(cd))]       # same downloads as the job that printed this command
    parts += ["--judge", judge_name, "--judgments", shlex.quote(str(Path(out) / "judgments.json")), "--out", shlex.quote(str(out))]
    return " ".join(parts)


def _chapter_at(media: ingest.Media, t: float) -> str:
    for ch in media.chapters:
        if ch.get("start_time", 0) <= t < ch.get("end_time", 1e18):
            return ch.get("title", "")
    return ""


def _render_one(media, j, out, name, hook, fill, cfg, hook_s, caption_words=None):
    end = min(j.best_end, media.offset + media.duration)
    render_clip(media, j.best_start, end, out / name, hook, cfg, fill=fill, caption_words=caption_words)
    return end


def _words_for(media, it, cfg):
    key = f"{media.video_id}-{it.hash}-{cfg['captions']['model']}"
    return cap_mod.transcribe_window(media, it.cand.start_s, it.cand.end_s, cfg, key)


def _drawn(c: dict) -> str:
    """The text burned into the video: the headline `title` if there is one, else the short hook."""
    return c.get("title") or c["hook"]


def _live_moments(source, media, cands, n, cfg, log) -> list:
    """Walk the candidates best-first, fetch each moment's picture, drop dead screens, keep n live ones."""
    from . import visual
    min_motion = float(cfg.get("scan", {}).get("min_motion", 0.25))
    valid: list = []
    failed = 0
    for c in cands:
        if len(valid) >= n:
            break
        log(f"[{len(valid) + 1}/{n}] moment {c.start_s:.0f}-{c.end_s:.0f}s")
        try:
            m = ingest.download_section(source, media.video_id, c.start_s, c.end_s, cfg, log) if ingest.is_url(source) else media
        except ClipperError as e:
            log(f"  skipped: {e}")
            failed += 1
            continue
        info = visual.liveliness(m.path, max(0.0, c.start_s - m.offset), c.end_s - c.start_s)
        why = visual.dead_reason(info, min_motion)
        if why:
            log(f"  skipped: {why}")
            continue
        valid.append((c, m))
    if len(valid) < n:
        log(f"Only {len(valid)} live moments were found (asked for {n})."
            + (f" {failed} moment(s) could not be downloaded and were skipped." if failed else ""))
    if not valid and failed:
        raise ClipperError("None of the chosen moments could be downloaded (YouTube kept refusing the links). "
                           "Try again later, or use --range on a shorter part of the stream.")
    if not valid:
        raise ClipperError("No new moments with a live picture are left in this stream: the rest were dead screens or "
                           "were already clipped earlier. Use --allow-repeats to include earlier clips, or try another "
                           "stream or --range.")
    return valid


def _heuristic_qa(media, clip, c, cfg):
    r = cfg["render"]
    layout = effective_layout(media, cfg, bool(c.get("fill")))
    persistent = r.get("title_style", "box") == "box" and not float(r.get("title_seconds", 0) or 0)
    return qa_mod.heuristic_qa(clip, c["end_s"] - c["start_s"], float(r["hook_seconds"]), bool(_drawn(c)),
                               band=title_band(cfg, layout), persistent=persistent)


def run(source: str, *, n: int, window: float, rng, fill: bool, out: Path, cfg: dict,
        judge_name: str = "heuristic", judgments: str | None = None, shortlist: int | None = None,
        niche: str | None = None, trends_path: str | None = None, qa_name: str = "heuristic",
        captions: bool = False, chat: bool = True, scan: bool = False, transcribe: bool = False,
        min_score: float = 0.0, allow_repeats: bool = False, log=print) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    if captions or (scan and transcribe):
        cap_mod.check_available()
    st0 = _load_state(out)
    if scan and chat and st0 and st0.get("source") == source and st0.get("scan") and st0.get("window") == window \
            and st0.get("range") == (list(rng) if rng else None) and st0.get("candidates"):
        chat = False        # resuming: the moments are already chosen, the chat replay is only needed to choose them
        log("Resuming from the saved analysis: not fetching the chat replay again.")
    media = ingest.resolve_scan(source, cfg, log, chat) if scan else ingest.resolve(source, rng, cfg, log=log, chat=chat)
    log(f"Source: {media.title or media.source}  ({media.duration:.0f}s, "
        f"{(str(media.width) + 'x' + str(media.height)) if media.width else 'audio only'}"
        f"{'' if media.has_audio else ', no audio'})")
    judge = judge_mod.get_judge(judge_name)

    state = _load_state(out)
    same = bool(state and state.get("video_id") == media.video_id and state.get("window") == window
                and state.get("range") == (list(rng) if rng else None) and state.get("candidates")
                and bool(state.get("scan")) == scan)
    if same:
        cands = [_cand_from_dict(d) for d in state["candidates"]]
        trends = state["trends"]
        log(f"Reusing analysis from {out}/state.json ({len(cands)} candidates)")
    else:
        extras = [k for k, ok in (("heatmap", media.heatmap), ("chat", media.chat_times),
                                  ("funny chat", getattr(media, "chat_funny", None))) if ok]
        log(f"Extra signals: {', '.join(extras) or 'none'}")
        sig = signals.analyze(media, cfg, rng)
        log(f"Analysed {sig.n * signals.BIN:.0f}s: {', '.join(sig.raw)}")
        if scan:
            cands = scan_mod.find_moments(media, sig, window, n, cfg, transcribe or captions, log)
        else:
            cands = cand.pick(sig, window, cfg)
        trends = trends_mod.load(niche, trends_path, log)
        if ingest.is_url(source) and not allow_repeats:
            cands, skipped = ledger.drop_used(cands, cfg, media.video_id)
            if skipped:
                log(f"Skipping {skipped} moment(s) already cut into clips from this stream earlier "
                    "(use --allow-repeats to include them).")
            if not cands:
                raise ClipperError("Every good moment found in this stream was already clipped earlier. "
                                   "Use --allow-repeats to clip them again, or try a different stream or --range.")

    sec: dict = {}
    if scan:
        valid = _live_moments(source, media, cands, n, cfg, log)
        items = evidence.make_items(media, [c for c, _ in valid], window, trends)
        if ingest.is_url(source):
            sec = {it.id: m for it, (_, m) in zip(items, valid)}
    else:
        k = max(n, shortlist or (5 if judge.external else len(cands)))
        items = evidence.make_items(media, cands[:k], window, trends)
    M = lambda it: sec.get(it.id, media)  # noqa: E731 - media that holds this item's picture
    W = (lambda it: scan_mod.cand_words(media, it.cand, cfg)) if scan else (lambda it: _words_for(media, it, cfg))
    have_words = bool(captions or (scan and transcribe))
    ctx = {"niche": niche, "title": media.title, "trends": trends, "judgments_path": judgments,
           "chapters_by_item": {it.id: _chapter_at(media, it.cand.peak_s) for it in items},
           "words_by_item": {it.id: W(it) for it in items} if scan and have_words else {},
           "generic_hooks": not scan}

    results: dict = {}
    for it in items:
        j = judge_mod.cache_get(cfg, judge.name, it)
        if j:
            results[it.id] = j
    pending = [it for it in items if it.id not in results]
    problems: list[str] = []
    if pending:
        found, problems = judge.judge(pending, ctx)
        for it in pending:
            if it.id in found:
                results[it.id] = found[it.id]
                judge_mod.cache_put(cfg, judge.name, it, found[it.id])
        pending = [it for it in items if it.id not in results]

    new_state = {"source": source, "video_id": media.video_id, "window": window,
                 "range": list(rng) if rng else None, "n": n, "fill": fill, "niche": niche, "scan": scan,
                 "judge": judge.name, "trends": trends, "candidates": [_cand_to_dict(c) for c in cands]}
    _save_state(out, new_state)
    (out / "candidates.json").write_text(json.dumps(
        [{"id": it.id, **_cand_to_dict(it.cand)} for it in items], indent=2), encoding="utf-8")

    if pending:
        if not judge.external:
            raise ClipperError("Internal error: judge left candidates unjudged.")
        for it in pending:
            if not (out / "evidence" / it.id / "contact.jpg").is_file():
                tr = {it.id: cap_mod.transcript_text(W(it))} if have_words else None
                evidence.write_pack(M(it), [it], out, window, tr)
        resume = _resume_cmd(source, n, window, rng, fill, niche, out, judge.name, scan, transcribe, captions, chat, cfg, min_score)
        prompt = judge_mod.write_prompt(out, pending, ctx, resume)
        sheets = evidence.write_sheets(out, pending)
        for pr in problems:
            log(f"  judgments problem: {pr}")
        if sheets:
            log(f"Made {len(sheets)} picture sheet(s) for Claude in {out / 'judge_sheets'} (4 candidates per image).")
        log(f"\nWaiting for Claude's judgments on {len(pending)} candidate(s).\n"
            f"  1. Give Claude the sheet images in {out / 'judge_sheets'} plus {prompt}\n"
            f"  2. It writes: {out / 'judgments.json'}\n"
            f"  3. Then run: {resume}")
        return {"status": "awaiting_judgments", "prompt": str(prompt), "pending": [i.id for i in pending],
                "problems": problems, "out": str(out)}

    ranked = sorted(items, key=lambda it: (-results[it.id].score, it.rank))
    keep = [it for it in ranked if results[it.id].score >= min_score]
    if len(keep) < len(ranked):
        log(f"Leaving out {len(ranked) - len(keep)} candidate(s) judged below --min-score {min_score:g}")
    chosen = keep[:n]
    rest = [it for it in ranked if it not in chosen]
    hook_s = float(cfg["render"]["hook_seconds"])
    clips = []
    for i, it in enumerate(chosen, 1):
        j = results[it.id]
        name = f"clip_{i}.mp4"
        log(f"[{i}/{len(chosen)}] {j.best_start:.1f}-{j.best_end:.1f}s judge {j.score:g}/10 hook \"{j.hook}\" -> {name}")
        cw = None
        if captions:
            cw = cap_mod.slice_words(W(it), j.best_start, j.best_end)
            log(f"    captions: {len(cw)} words")
        end = _render_one(M(it), j, out, name, j.title or j.hook, fill, cfg, hook_s, cw)
        c = {"file": name, "id": it.id, "start_s": round(j.best_start, 2), "end_s": round(end, 2),
             "score": round(it.cand.score, 4), "hook": j.hook, "title": j.title, "fill": fill,
             "signals": {a: round(b, 3) for a, b in it.cand.breakdown.items()},
             "judge": j.to_dict(), "qa": [], "rerendered": False, "caption_words": cw or [],
             "link": _link(media, j.best_start),
             "section": ({"path": str(sec[it.id].path), "offset": sec[it.id].offset} if it.id in sec else None)}
        if qa_name != "none":
            q = _heuristic_qa(M(it), out / name, c, cfg)
            c["qa"].append({**q.to_dict()})
            if not q.passed and q.fix:
                log(f"    QA failed {q.checks}; re-rendering once with fix {q.fix}")
                c["fill"] = q.fix.get("fill", c["fill"])
                if "hook" in q.fix:
                    c["title"] = q.fix["hook"]
                _render_one(M(it), j, out, name, _drawn(c), c["fill"], cfg, hook_s, c["caption_words"] or None)
                c["rerendered"] = True
                c["qa"].append(_heuristic_qa(M(it), out / name, c, cfg).to_dict())
        clips.append(c)

    res = {"source": media.source, "video_id": media.video_id, "title": media.title, "judge": judge.name,
           "qa_backend": qa_name, "cost_usd": round(judge.cost_usd, 4), "trend_notes": trends.get("notes", []),
           "clips": clips,
           "not_selected": [{"id": it.id, "score": results[it.id].score, "start_s": results[it.id].best_start,
                             "end_s": results[it.id].best_end, "reason": results[it.id].reason} for it in rest]}
    _finish(out, res)
    if ingest.is_url(source):
        ledger.record(cfg, media.video_id, [(c["start_s"], c["end_s"]) for c in clips])
    res["status"] = "done"
    if qa_name == "chat":
        p = qa_mod.write_chat_pack(clips, out, hook_s)
        res["qa_prompt"] = str(p)
        log(f"\nOptional Claude QA: ask Claude to follow {p}, then run:\n"
            f"  clipper {shlex.quote(source)} --out {shlex.quote(str(out))} --qa-results {out / 'qa.json'}")
    log(f"\nDone: {len(clips)} clip(s) in {out}/\n  Open {out / 'review.html'} in your browser to watch them and copy titles/captions (details: report.md)")
    return res


def _link(media, start_s: float) -> str | None:
    src = media.source or ""
    if ingest.is_url(src) and ("youtube.com" in src or "youtu.be" in src):
        return f"https://www.youtube.com/watch?v={media.video_id}&t={int(start_s)}s"
    return None


def _finish(out: Path, res: dict) -> None:
    (out / "clips.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
    report.write(out, res)
    review.write(out, res)


def apply_qa_results(source: str, *, out: Path, cfg: dict, qa_results: str, log=print) -> dict:
    """Load Claude's QA verdicts; re-render each failing clip once with its suggested fix."""
    p = out / "clips.json"
    if not p.is_file():
        raise ClipperError(f"No {p} found. Render clips first, then run QA.")
    res = json.loads(p.read_text(encoding="utf-8"))
    state = _load_state(out) or {}
    rng = tuple(state["range"]) if state.get("range") else None
    media = None
    verdicts = qa_mod.load_chat_results(qa_results)
    hook_s = float(cfg["render"]["hook_seconds"])
    for c in res["clips"]:
        v = verdicts.get(c["file"])
        if not v:
            continue
        c["qa"].append(v.to_dict())
        if not v.passed and v.fix:
            log(f"{c['file']}: Claude QA failed ({'; '.join(v.notes) or 'no detail'}); re-rendering once with {v.fix}")
            c["fill"] = v.fix.get("fill", c["fill"])
            if "hook" in v.fix:
                c["title"] = v.fix["hook"]
            j = judge_mod.Judgment(**{**c["judge"], "best_start": c["start_s"], "best_end": c["end_s"]})
            if c.get("section"):
                sp = Path(c["section"]["path"])
                cm = ingest.Media(path=sp, offset=c["section"]["offset"], video_id=res.get("video_id", ""), **ingest.probe(sp))
            else:
                media = media or ingest.resolve(source, rng, cfg)
                cm = media
            _render_one(cm, j, out, c["file"], _drawn(c), c["fill"], cfg, hook_s, c.get("caption_words") or None)
            c["rerendered"] = True
        elif not v.passed:
            log(f"{c['file']}: Claude QA failed but gave no fix; left as is. Issues: {'; '.join(v.notes)}")
    res["qa_backend"] = "chat"
    _finish(out, res)
    log(f"QA applied. See {out / 'report.md'}")
    return res
