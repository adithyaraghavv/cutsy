"""Scan mode helpers: find the best moments across a whole stream.

1. audio (+ chat replay + YouTube "most replayed") signals over the entire stream -> many candidate windows
2. optionally transcribe the candidates (faster-whisper) and boost windows full of reactions
3. titles/captions suggested from what was actually said
"""
from __future__ import annotations

import re

from . import candidates as cand
from . import captions as cap_mod

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


def find_moments(media, sig, window: float, n: int, cfg: dict, transcribe: bool, log) -> list:
    """Top moments across the whole stream, best first."""
    k = min(max(3 * n, n + 10), 60)
    cands = cand.pick(sig, window, cfg, count=k)
    cands = cand.recenter(cands, sig, window, cfg)
    log(f"Scan: {len(cands)} candidate moments found across {sig.n * 0.5 / 3600:.1f} h; signals used: {', '.join(sig.raw)}")
    if transcribe:
        log(f"Transcribing {len(cands)} candidates to find reactions (first run downloads the speech model)...")
        for i, c in enumerate(cands, 1):
            h = hype_score(cand_words(media, c, cfg))
            c.breakdown["speech_hype"] = round(h, 2)
            c.score += 0.5 * min(h / 6.0, 1.5)
            if i % 10 == 0:
                log(f"  transcribed {i}/{len(cands)}")
        cands.sort(key=lambda x: -x.score)
    return cands
