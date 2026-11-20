"""Judge layer with swappable backends.

A backend turns evidence Items into validated Judgments:
    heuristic - offline, free, no LLM: ranks by signals, hooks/hashtags from trends/title.
    chat      - "Claude in your chat": clipper writes an evidence pack + judge prompt; the Claude
                you are chatting with writes judgments.json; clipper validates and loads it.

To add a backend later (e.g. a paid API or a local model) subclass `Judge`, implement
`judge()`, and register it in `JUDGES`. Nothing else in the pipeline changes. Backends that
cost money should report spend through `Judge.cost_usd` so --budget-usd can stop the run.
"""
from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path

from .errors import ClipperError
from .evidence import Item, length_limits

RISK_FLAGS = ("spoilers", "cutscenes", "copyright", "profanity")
GENERIC_TAGS = ["#shorts", "#reels", "#fyp", "#viral", "#clips"]
TAG_STOP = set("playing first time this that with from your stream live full video game gameplay part episode reaction trailer official watch what when they them have been just like very into over under about after before again".split())
TOL = 0.6


@dataclass
class Judgment:
    id: str
    score: float
    reason: str
    best_start: float
    best_end: float
    hook: str
    caption: str
    hashtags: list
    risk_flags: list
    backend: str = ""
    risk_assessed: bool = True
    title: str = ""          # optional on-video headline (10-25 words); falls back to `hook`

    def to_dict(self) -> dict:
        return asdict(self)


def validate(raw: dict, item: Item) -> Judgment:
    """Strict check of one judgment against its candidate. Raises ValueError listing every problem."""
    p: list[str] = []
    lo, hi = length_limits(item.cand)
    c = item.cand

    def num(k):
        v = raw.get(k)
        if isinstance(v, bool) or not isinstance(v, (int, float)):
            p.append(f"{k} must be a number")
            return None
        return float(v)

    score, bs, be = num("score"), num("best_start"), num("best_end")
    if score is not None and not 1 <= score <= 10:
        p.append("score must be 1-10")
    if bs is not None and be is not None:
        if bs < c.start_s - TOL or be > c.end_s + TOL:
            p.append(f"best_start/best_end must lie inside the window {c.start_s:.1f}-{c.end_s:.1f}")
        if not lo - TOL <= be - bs <= hi + TOL:
            p.append(f"trimmed length must be {lo:.0f}-{hi:.0f} s (got {be - bs:.1f})")
    reason = raw.get("reason")
    if not isinstance(reason, str) or not reason.strip() or "\n" in reason.strip():
        p.append("reason must be a non-empty single line")
    hook = raw.get("hook")
    if not isinstance(hook, str) or not hook.strip() or len(hook.split()) > 6:
        p.append("hook must be 1-6 words")
    title = raw.get("title", "")
    if title not in (None, "") and (not isinstance(title, str) or not 2 <= len(title.split()) <= 30 or "\n" in title):
        p.append("title (optional) must be one line of 2-30 words")
    cap = raw.get("caption")
    if not isinstance(cap, str) or not cap.strip():
        p.append("caption must be non-empty")
    tags = raw.get("hashtags")
    if not (isinstance(tags, list) and len(tags) == 5 and all(isinstance(t, str) and re.fullmatch(r"#\w+", t) for t in tags)):
        p.append("hashtags must be exactly 5 strings like '#gta6'")
    flags = raw.get("risk_flags")
    if not (isinstance(flags, list) and all(f in RISK_FLAGS for f in flags)):
        p.append(f"risk_flags must be a list using only {list(RISK_FLAGS)}")
    if p:
        raise ValueError("; ".join(p))
    return Judgment(id=item.id, score=score, reason=reason.strip(), best_start=bs, best_end=be,
                    hook=hook.strip(), caption=cap.strip(), hashtags=tags, risk_flags=flags,
                    title=(title or "").strip())


class Judge:
    name = "base"
    external = False          # True: needs a human/Claude-in-chat step between runs
    cost_usd = 0.0

    def judge(self, items: list[Item], ctx: dict) -> tuple[dict, list[str]]:
        """Return ({item_id: Judgment}, problems). Items missing from the dict are 'pending'."""
        raise NotImplementedError


class HeuristicJudge(Judge):
    name = "heuristic"

    TEMPLATES = ["WAIT FOR THIS", "NO WAY THIS HAPPENED", "WATCH TILL THE END", "THIS GOT CRAZY", "YOU NEED TO SEE THIS"]

    def judge(self, items, ctx):
        out = {}
        niche, title = ctx.get("niche"), ctx.get("title") or ""
        trends = ctx["trends"]
        for it in items:
            c = it.cand
            lo, hi = length_limits(c)
            d = c.end_s - c.start_s
            L = min(d, 55.0 if d > 60 else d)
            bs = min(max(c.peak_s - L / 2, c.start_s), c.end_s - L)
            chap = (ctx.get("chapters_by_item") or {}).get(it.id)
            words = (ctx.get("words_by_item") or {}).get(it.id) or []
            from .scan import hook_from_words, quote_from_words
            spoken = hook_from_words(words) if words else None
            if spoken:
                hook = spoken
            elif chap:
                hook = " ".join(re.sub(r"[^\w' !?-]", " ", chap).split()[:6])
            else:
                hook = self.TEMPLATES[(it.rank - 1) % len(self.TEMPLATES)]
                if niche and len(niche.split()) <= 2 and it.rank % 2 == 1:
                    hook = f"{niche.upper()} GOES CRAZY"
            tags: list[str] = []
            for t in ([f"#{re.sub(r'[^0-9a-zA-Z]', '', niche).lower()}"] if niche else []) + trends.get("hashtags", []) \
                    + [f"#{w.lower()}" for w in re.findall(r"[A-Za-z0-9]{4,}", title) if w.lower() not in TAG_STOP] + GENERIC_TAGS:
                if len(t) > 1 and t.lower() not in [x.lower() for x in tags]:
                    tags.append(t)
            out[it.id] = Judgment(
                id=it.id, score=round(max(1.0, min(10.0, 5 + 1.5 * c.score)), 1),
                reason=f"Signal rank #{it.rank}: strongest audio/motion/crowd activity (score {c.score:.2f}). Not reviewed visually.",
                best_start=round(bs, 2), best_end=round(bs + L, 2), hook=hook,
                caption=(f'"{quote_from_words(words)}" - {title[:50]}' if words and quote_from_words(words)
                         else (f"{title[:60]} - {hook.capitalize()}" if title else hook.capitalize())),
                hashtags=tags[:5], risk_flags=[], backend=self.name, risk_assessed=False)
        return out, []


class ChatJudge(Judge):
    name = "chat"
    external = True

    def judge(self, items, ctx):
        path = ctx.get("judgments_path")
        found, problems = {}, []
        by_id = {it.id: it for it in items}
        if path:
            try:
                data = json.loads(Path(path).read_text(encoding="utf-8"))
            except FileNotFoundError:
                raise ClipperError(f"Judgments file not found: {path}") from None
            except json.JSONDecodeError as e:
                raise ClipperError(f"{path} is not valid JSON ({e.msg} at line {e.lineno}).") from None
            rows = data.get("judgments") if isinstance(data, dict) else data
            if not isinstance(rows, list):
                raise ClipperError(f"{path} must be a JSON list of judgments (or {{\"judgments\": [...]}}).")
            for row in rows:
                it = by_id.get(row.get("id")) if isinstance(row, dict) else None
                if it is None:
                    problems.append(f"unknown or stale id {row.get('id') if isinstance(row, dict) else row!r} (ignored)")
                    continue
                try:
                    j = validate(row, it)
                    j.backend = self.name
                    found[it.id] = j
                except ValueError as e:
                    problems.append(f"{it.id}: {e}")
        return found, problems


JUDGES = {"heuristic": HeuristicJudge, "chat": ChatJudge}


def get_judge(name: str) -> Judge:
    if name not in JUDGES:
        raise ClipperError(f"Unknown judge {name!r}. Available: {', '.join(JUDGES)}")
    return JUDGES[name]()


# ----------------------------------------------------------------- cache ----

def cache_path(cfg: dict, backend: str, h: str) -> Path:
    return Path(cfg["cache_dir"]) / "judgments" / f"{backend}-{h}.json"


def cache_get(cfg, backend, item) -> Judgment | None:
    p = cache_path(cfg, backend, item.hash)
    if p.is_file():
        try:
            d = json.loads(p.read_text(encoding="utf-8"))
            d["id"] = item.id
            return Judgment(**d)
        except (OSError, ValueError, TypeError):
            return None
    return None


def cache_put(cfg, backend, item, j: Judgment) -> None:
    p = cache_path(cfg, backend, item.hash)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(j.to_dict()), encoding="utf-8")


# ---------------------------------------------------------------- prompt ----

PROMPT = """# Clip judging task

You are the judge for `clipper`. Below are {n} candidate moments cut from "{title}". For each one,
look at `evidence/<id>/contact.jpg` (8 frames, left-to-right, top-to-bottom, in time order; the
individual `frame_N.jpg` files are there if you need detail) and `evidence/<id>/evidence.json`
(timestamps, signal strengths, transcript if any).

Pick the moments most likely to perform as a vertical Reel / Short and trim each to its best
45-60 seconds (or the whole window if it is shorter than 45 s). Judge only what the evidence shows;
if something cannot be told from it, say so in `reason` rather than guessing.

## Trends to match (optional context)
- Niche: {niche}
- Keywords: {keywords}
- Hashtags working now: {hashtags}

## Output
Write **{out}** containing ONLY a JSON list, one object per candidate (all {n}), exactly:

```json
[
  {{
    "id": "<copy the id>",
    "score": 7,
    "reason": "one line on why this moment works (or doesn't)",
    "best_start": 0.0,
    "best_end": 0.0,
    "title": "headline shown on the video",
    "hook": "max six words",
    "caption": "short caption",
    "hashtags": ["#one", "#two", "#three", "#four", "#five"],
    "risk_flags": []
  }}
]
```

Rules, all enforced by clipper (a bad entry is rejected and you will be asked to fix it):
- `score` is 1-10. `best_start`/`best_end` are ABSOLUTE source seconds inside the candidate's window
  (see `window` in evidence.json) and the length is within `trim_length_s`.
- `title` is the headline burned into the video (white box, black bold text, whole clip): ONE sentence of
  10-22 words that says what is happening and why it matters, like a news headline for the moment; put 3-5 key
  words in CAPS for emphasis (e.g. "He is OFFICIALLY the FIRST person to COMPLETE the whole game in 7 hours");
  no emoji (put emoji in `caption`); no spoilers you cannot see in the evidence.
- `hook` is at most 6 words (a short fallback). `hashtags` is exactly 5 strings, each starting with `#`.
- `risk_flags` uses only: spoilers, cutscenes, copyright, profanity (empty list if none seen).

## Candidates
{rows}

Then run: `{resume}`
"""


def write_prompt(out: Path, items: list[Item], ctx: dict, resume_cmd: str) -> Path:
    t = ctx["trends"]
    rows = "\n".join(f"- `{it.id}`  window {it.cand.start_s:.1f}-{it.cand.end_s:.1f}s  (signal rank #{it.rank})" for it in items)
    p = out / "judge_prompt.md"
    p.write_text(PROMPT.format(
        n=len(items), title=ctx.get("title") or "the video", niche=ctx.get("niche") or "(none)",
        keywords=", ".join(t.get("keywords", [])[:15]) or "(none)",
        hashtags=" ".join(t.get("hashtags", [])[:15]) or "(none)",
        out=out / "judgments.json", rows=rows, resume=resume_cmd), encoding="utf-8")
    return p
