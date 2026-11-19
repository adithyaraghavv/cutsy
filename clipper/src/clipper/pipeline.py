from __future__ import annotations

import json
import shlex
from pathlib import Path

from . import candidates as cand
from . import evidence, ingest, judge as judge_mod, qa as qa_mod, report, signals
from . import trends as trends_mod
from .errors import ClipperError
from .render import render_clip


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


def _resume_cmd(source, n, window, rng, fill, niche, out, judge_name) -> str:
    parts = ["clipper", shlex.quote(source), "--n", str(n), "--window", f"{window:g}"]
    if rng:
        parts += ["--range", f"{rng[0]:g}-{rng[1]:g}"]
    if niche:
        parts += ["--niche", shlex.quote(niche)]
    if fill:
        parts.append("--fill")
    parts += ["--judge", judge_name, "--judgments", shlex.quote(str(Path(out) / "judgments.json")), "--out", shlex.quote(str(out))]
    return " ".join(parts)


def _chapter_at(media: ingest.Media, t: float) -> str:
    for ch in media.chapters:
        if ch.get("start_time", 0) <= t < ch.get("end_time", 1e18):
            return ch.get("title", "")
    return ""


def _render_one(media, j, out, name, hook, fill, cfg, hook_s):
    end = min(j.best_end, media.offset + media.duration)
    render_clip(media, j.best_start, end, out / name, hook, cfg, fill=fill)
    return end


def _heuristic_qa(media, clip, c, cfg):
    return qa_mod.heuristic_qa(clip, c["end_s"] - c["start_s"], float(cfg["render"]["hook_seconds"]), bool(c["hook"]))


def run(source: str, *, n: int, window: float, rng, fill: bool, out: Path, cfg: dict,
        judge_name: str = "heuristic", judgments: str | None = None, shortlist: int | None = None,
        niche: str | None = None, trends_path: str | None = None, qa_name: str = "heuristic",
        log=print) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    media = ingest.resolve(source, rng, cfg)
    log(f"Source: {media.title or media.source}  ({media.duration:.0f}s, {media.width}x{media.height}"
        f"{'' if media.has_audio else ', no audio'})")
    judge = judge_mod.get_judge(judge_name)

    state = _load_state(out)
    same = bool(state and state.get("video_id") == media.video_id and state.get("window") == window
                and state.get("range") == (list(rng) if rng else None) and state.get("candidates"))
    if same:
        cands = [_cand_from_dict(d) for d in state["candidates"]]
        trends = state["trends"]
        log(f"Reusing analysis from {out}/state.json ({len(cands)} candidates)")
    else:
        extras = [k for k, ok in (("heatmap", media.heatmap), ("chat", media.chat_times)) if ok]
        log(f"Extra signals: {', '.join(extras) or 'none'}")
        sig = signals.analyze(media, cfg, rng)
        log(f"Analysed {sig.n * signals.BIN:.0f}s: {', '.join(sig.raw)}")
        cands = cand.pick(sig, window, cfg)
        trends = trends_mod.load(niche, trends_path, log)

    k = max(n, shortlist or (5 if judge.external else len(cands)))
    items = evidence.make_items(media, cands[:k], window, trends)
    ctx = {"niche": niche, "title": media.title, "trends": trends, "judgments_path": judgments,
           "chapters_by_item": {it.id: _chapter_at(media, it.cand.peak_s) for it in items}}

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
                 "range": list(rng) if rng else None, "n": n, "fill": fill, "niche": niche,
                 "judge": judge.name, "trends": trends, "candidates": [_cand_to_dict(c) for c in cands]}
    _save_state(out, new_state)
    (out / "candidates.json").write_text(json.dumps(
        [{"id": it.id, **_cand_to_dict(it.cand)} for it in items], indent=2), encoding="utf-8")

    if pending:
        if not judge.external:
            raise ClipperError("Internal error: judge left candidates unjudged.")
        for it in pending:
            if not (out / "evidence" / it.id / "contact.jpg").is_file():
                evidence.write_pack(media, [it], out, window)
        resume = _resume_cmd(source, n, window, rng, fill, niche, out, judge.name)
        prompt = judge_mod.write_prompt(out, pending, ctx, resume)
        for pr in problems:
            log(f"  judgments problem: {pr}")
        log(f"\nWaiting for Claude's judgments on {len(pending)} candidate(s).\n"
            f"  1. Ask Claude (in your chat) to follow: {prompt}\n"
            f"  2. It writes: {out / 'judgments.json'}\n"
            f"  3. Then run: {resume}")
        return {"status": "awaiting_judgments", "prompt": str(prompt), "pending": [i.id for i in pending],
                "problems": problems, "out": str(out)}

    ranked = sorted(items, key=lambda it: (-results[it.id].score, it.rank))
    chosen, rest = ranked[:n], ranked[n:]
    hook_s = float(cfg["render"]["hook_seconds"])
    clips = []
    for i, it in enumerate(chosen, 1):
        j = results[it.id]
        name = f"clip_{i}.mp4"
        log(f"[{i}/{len(chosen)}] {j.best_start:.1f}-{j.best_end:.1f}s judge {j.score:g}/10 hook \"{j.hook}\" -> {name}")
        end = _render_one(media, j, out, name, j.hook, fill, cfg, hook_s)
        c = {"file": name, "id": it.id, "start_s": round(j.best_start, 2), "end_s": round(end, 2),
             "score": round(it.cand.score, 4), "hook": j.hook, "fill": fill,
             "signals": {a: round(b, 3) for a, b in it.cand.breakdown.items()},
             "judge": j.to_dict(), "qa": [], "rerendered": False}
        if qa_name != "none":
            q = _heuristic_qa(media, out / name, c, cfg)
            c["qa"].append({**q.to_dict()})
            if not q.passed and q.fix:
                log(f"    QA failed {q.checks}; re-rendering once with fix {q.fix}")
                c["fill"] = q.fix.get("fill", c["fill"])
                c["hook"] = q.fix.get("hook", c["hook"])
                _render_one(media, j, out, name, c["hook"], c["fill"], cfg, hook_s)
                c["rerendered"] = True
                c["qa"].append(_heuristic_qa(media, out / name, c, cfg).to_dict())
        clips.append(c)

    res = {"source": media.source, "video_id": media.video_id, "title": media.title, "judge": judge.name,
           "qa_backend": qa_name, "cost_usd": round(judge.cost_usd, 4), "trend_notes": trends.get("notes", []),
           "clips": clips,
           "not_selected": [{"id": it.id, "score": results[it.id].score, "start_s": results[it.id].best_start,
                             "end_s": results[it.id].best_end, "reason": results[it.id].reason} for it in rest]}
    _finish(out, res)
    res["status"] = "done"
    if qa_name == "chat":
        p = qa_mod.write_chat_pack(clips, out, hook_s)
        res["qa_prompt"] = str(p)
        log(f"\nOptional Claude QA: ask Claude to follow {p}, then run:\n"
            f"  clipper {shlex.quote(source)} --out {shlex.quote(str(out))} --qa-results {out / 'qa.json'}")
    log(f"\nDone: {len(clips)} clip(s) in {out}/ (see clips.json and report.md)")
    return res


def _finish(out: Path, res: dict) -> None:
    (out / "clips.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
    report.write(out, res)


def apply_qa_results(source: str, *, out: Path, cfg: dict, qa_results: str, log=print) -> dict:
    """Load Claude's QA verdicts; re-render each failing clip once with its suggested fix."""
    p = out / "clips.json"
    if not p.is_file():
        raise ClipperError(f"No {p} found. Render clips first, then run QA.")
    res = json.loads(p.read_text(encoding="utf-8"))
    state = _load_state(out) or {}
    rng = tuple(state["range"]) if state.get("range") else None
    media = ingest.resolve(source, rng, cfg)
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
            c["hook"] = v.fix.get("hook", c["hook"])
            j = judge_mod.Judgment(**{**c["judge"], "best_start": c["start_s"], "best_end": c["end_s"]})
            _render_one(media, j, out, c["file"], c["hook"], c["fill"], cfg, hook_s)
            c["rerendered"] = True
        elif not v.passed:
            log(f"{c['file']}: Claude QA failed but gave no fix; left as is. Issues: {'; '.join(v.notes)}")
    res["qa_backend"] = "chat"
    _finish(out, res)
    log(f"QA applied. See {out / 'report.md'}")
    return res
