"""Scan mode helpers: find the best moments across a whole stream.

1. audio (+ chat replay + YouTube "most replayed") signals over the entire stream -> many candidate windows
2. optionally transcribe the candidates (faster-whisper) and boost windows full of reactions
3. titles/captions suggested from what was actually said
"""
from __future__ import annotations

import math
import re

import numpy as np

from . import candidates as cand
from . import captions as cap_mod
from .signals import BIN, Signals

# Reaction phrases (lower-case) and how strongly they suggest a hype/funny moment.
HYPE = {
    "no way": 3, "oh my god": 3, "omg": 3, "what the": 3, "wtf": 3, "unbelievable": 3, "are you kidding": 3,
    "oh no": 2, "let's go": 2, "lets go": 2, "holy": 2, "insane": 2, "crazy": 2, "that's so": 1,
    "haha": 2, "hahaha": 2, "lol": 2, "laughing": 2, "dead": 1, "bro": 1, "dude": 1, "yo": 1, "damn": 1,
    "what is": 1, "look at": 1, "wait": 1, "scary": 1, "screaming": 1, "i can't": 2, "i cant": 2,
}


def _norm(w: str) -> str:
    return re.sub(r"[^\w']", "", w.lower())


def hype_hits(words: list[dict]) -> list[tuple[int, str, int]]:
    """(index, phrase, weight) for every reaction phrase found in the word list."""
    toks = [_norm(w["w"]) for w in words]
    hits = []
    for i in range(len(toks)):
        for ph, wt in HYPE.items():
            parts = ph.split()
            if toks[i:i + len(parts)] == [_norm(p) for p in parts]:
                hits.append((i, ph, wt))
    return hits


def hype_score(words: list[dict]) -> float:
    return float(sum(wt for _, _, wt in hype_hits(words)))


def hook_from_words(words: list[dict]) -> str | None:
    """A <=6-word on-video title taken from the strongest reaction in the speech."""
    hits = hype_hits(words)
    if not hits:
        return None
    i, ph, _ = max(hits, key=lambda h: h[2])
    lo = i   # start the title at the reaction itself
    toks = [re.sub(r"[^\w'!?-]", "", w["w"]) for w in words[lo:lo + 6]]
    toks = [t for t in toks if t]
    return " ".join(toks).upper() if toks else None


def quote_from_words(words: list[dict], limit: int = 14) -> str | None:
    hits = hype_hits(words)
    if not hits:
        return None
    i = max(hits, key=lambda h: h[2])[0]
    lo = max(0, i - 4)
    toks = [w["w"].strip() for w in words[lo:lo + limit] if w["w"].strip()]
    return " ".join(toks) if toks else None


def cand_words(media, c, cfg: dict) -> list[dict]:
    """Words (absolute source times) for a candidate window; cached on disk."""
    key = f"{media.video_id}-{c.start_s:.1f}-{c.end_s:.1f}-{cfg['captions']['model']}"
    return cap_mod.transcribe_window(media, c.start_s, c.end_s, cfg, key)


def diversify(cands: list, n: int, min_gap_s: float, buckets: int = 1, t0: float = 0.0, span_s: float = 0.0,
              floor_frac: float = 0.4) -> list:
    """Best-first selection that keeps chosen moments at least `min_gap_s` apart (a long hype stretch would
    otherwise fill the whole list with near-duplicates). If there are too few spread-out moments, the best of
    the skipped ones fill the remaining places.

    With buckets > 1 the stream is cut into that many equal time slices and the choice rotates through them
    (each slice's best moment first, then each slice's second best, ...), so the first picks are spread over the
    whole stream instead of piling up in its loudest stretch. Moments scoring under `floor_frac` of the overall
    best are only used if nothing better is left, so a dead or AFK stretch is not forced in."""
    if buckets <= 1 or span_s <= 0:
        chosen: list = []
        skipped: list = []
        for c in sorted(cands, key=lambda x: -x.score):
            if all(abs(c.start_s - k.start_s) >= min_gap_s for k in chosen):
                chosen.append(c)
            else:
                skipped.append(c)
            if len(chosen) >= n:
                break
        for c in skipped:
            if len(chosen) >= n:
                break
            chosen.append(c)
        return sorted(chosen, key=lambda x: -x.score)

    order = sorted(cands, key=lambda x: -x.score)
    best = order[0].score if order else 0.0
    floor = floor_frac * best if best > 0 else float("-inf")
    width = span_s / buckets
    lanes: dict[int, list] = {}
    weak: list = []
    for c in order:
        (lanes.setdefault(min(buckets - 1, max(0, int((c.start_s - t0) // width))), []) if c.score >= floor else weak).append(c)

    chosen = []

    def ok(c) -> bool:
        return all(abs(c.start_s - k.start_s) >= min_gap_s for k in chosen)

    while len(chosen) < n and any(lanes.values()):
        for b in sorted((b for b in lanes if lanes[b]), key=lambda b: -lanes[b][0].score):
            while lanes[b]:
                c = lanes[b].pop(0)
                if ok(c):
                    chosen.append(c)
                    break
            if len(chosen) >= n:
                break
    for c in weak + [c for ls in lanes.values() for c in ls]:
        if len(chosen) >= n:
            break
        if ok(c):
            chosen.append(c)
    for c in order:                                   # still short: allow close neighbours
        if len(chosen) >= n:
            break
        if c not in chosen:
            chosen.append(c)
    return chosen


def bucket_count(duration_s: float, n: int, cfg: dict) -> int:
    """How many time slices to spread the picks over: one per `scan.bucket_s` (default 15 min) of stream, at most n."""
    per = float(cfg.get("scan", {}).get("bucket_s", 900) or 0)
    if per <= 0:
        return 1
    return max(1, min(int(n), int(duration_s // per)))


def slice_signals(sig: Signals, a: int, b: int) -> Signals:
    return Signals(t0=sig.t0 + a * BIN, n=b - a, raw={k: v[a:b] for k, v in sig.raw.items()},
                   z={k: v[a:b] for k, v in sig.z.items()}, combined=sig.combined[a:b])


def find_moments(media, sig, window: float, n: int, cfg: dict, transcribe: bool, log) -> list:
    """Top moments across the whole stream. On a long stream the picks are spread over its whole length."""
    span = sig.n * BIN
    nb = bucket_count(span, n, cfg)
    if nb > 1:
        per = max(3, math.ceil(3 * n / nb) + 1)
        cands = []
        for b in range(nb):
            cands += cand.pick(slice_signals(sig, b * sig.n // nb, (b + 1) * sig.n // nb), window, cfg, count=per)
    else:
        cands = cand.pick(sig, window, cfg, count=min(max(3 * n, n + 10), 60))
    cands = cand.recenter(cands, sig, window, cfg)
    log(f"Scan: {len(cands)} candidate moments found across {span / 3600:.1f} h; signals used: {', '.join(sig.raw)}")
    if transcribe:
        log(f"Transcribing {len(cands)} candidates to find reactions (first run downloads the speech model)...")
        for i, c in enumerate(cands, 1):
            h = hype_score(cand_words(media, c, cfg))
            c.breakdown["speech_hype"] = round(h, 2)
            c.score += 0.5 * min(h / 6.0, 1.5)
            if i % 10 == 0:
                log(f"  transcribed {i}/{len(cands)}")
        cands.sort(key=lambda x: -x.score)
    gap = float(cfg.get("scan", {}).get("min_gap_s", 180))
    spread = diversify(cands, 3 * max(n, 1), gap, buckets=nb, t0=sig.t0, span_s=span)
    where = (f", spread over {nb} time slices of the stream" if nb > 1 else "")
    log(f"Scan: {len(spread)} spread-out candidate moments (at least {gap:.0f}s apart{where}); "
        f"the best {n} with a live picture will be cut")
    if spread:
        log("Scan: first picks at " + ", ".join(f"{c.start_s / 3600:.1f}h" for c in spread[:n]))
    return spread
