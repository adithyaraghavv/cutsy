"""Pluggable, optional, fail-soft trend sources -> {"keywords": [...], "hashtags": [...]}.

Sources:
  * ManualTrends  - a trends.json you edit by hand
  * YouTubeTrends - official YouTube Data API (mostPopular + search for --niche),
                    key from the YOUTUBE_API_KEY environment variable only
No Instagram/TikTok scraping. A failing source never stops a run and never logs the key.
"""
from __future__ import annotations

import json
import os
import re
import urllib.error
import urllib.parse
import urllib.request
from collections import Counter
from pathlib import Path

YT = "https://www.googleapis.com/youtube/v3"
STOP = set("the a an and or of to in on for with is are was were this that it you i my we at by from "
           "how what why who vs part full new official video live stream gameplay trailer".split())


def _http_get_json(url: str, headers: dict | None = None, timeout: float = 10.0) -> dict:
    """Single network seam (tests replace this)."""
    req = urllib.request.Request(url, headers=headers or {})
    with urllib.request.urlopen(req, timeout=timeout) as r:  # noqa: S310 - fixed https host
        return json.loads(r.read().decode("utf-8"))


def _clean_tag(t: str) -> str:
    t = re.sub(r"[^0-9a-zA-Z]", "", t)
    return ("#" + t.lower()) if t else ""


def _extract(items: list[dict]) -> tuple[list[str], list[str]]:
    words: Counter = Counter()
    tags: Counter = Counter()
    for it in items:
        sn = it.get("snippet", {})
        for t in sn.get("tags", []) or []:
            if len(t) < 30:
                tags[_clean_tag(t)] += 1
        text = f"{sn.get('title', '')} {sn.get('description', '')[:300]}"
        for h in re.findall(r"#(\w{2,30})", text):
            tags["#" + h.lower()] += 1
        for w in re.findall(r"[A-Za-z0-9']{3,}", sn.get("title", "")):
            if w.lower() not in STOP:
                words[w.lower()] += 1
    tags.pop("", None)
    return [w for w, _ in words.most_common(15)], [t for t, _ in tags.most_common(15)]


def manual(path: str | Path | None, niche: str | None) -> dict:
    p = Path(path) if path else Path("trends.json")
    if not p.is_file():
        return {}
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        raise ValueError(f"{p} is not valid JSON")
    out = {"keywords": list(data.get("keywords", [])), "hashtags": list(data.get("hashtags", []))}
    nz = (data.get("niches") or {}).get((niche or "").lower())
    if nz:
        out["keywords"] += list(nz.get("keywords", []))
        out["hashtags"] += list(nz.get("hashtags", []))
    return out


def youtube(niche: str | None, region: str = "US") -> dict:
    key = os.environ.get("YOUTUBE_API_KEY")
    if not key:
        return {}
    headers = {"x-goog-api-key": key}   # header, not URL: keys never end up in logs/URLs
    items: list[dict] = []
    q = urllib.parse.urlencode({"part": "snippet", "chart": "mostPopular", "maxResults": 25,
                                "regionCode": region, "videoCategoryId": 20})
    items += _http_get_json(f"{YT}/videos?{q}", headers).get("items", [])
    if niche:
        q = urllib.parse.urlencode({"part": "snippet", "q": niche, "type": "video",
                                    "order": "viewCount", "maxResults": 15})
        items += _http_get_json(f"{YT}/search?{q}", headers).get("items", [])
    kw, ht = _extract(items)
    return {"keywords": kw, "hashtags": ht}


SOURCES = {"manual": manual, "youtube": lambda path, niche: youtube(niche)}


def load(niche: str | None = None, trends_path: str | Path | None = None, log=print) -> dict:
    """Merge all sources. Each source fails soft; `notes` records what happened (never secrets)."""
    keywords: list[str] = []
    hashtags: list[str] = []
    notes: list[str] = []
    for name, fn in SOURCES.items():
        try:
            r = fn(trends_path, niche)
        except urllib.error.HTTPError as e:
            notes.append(f"trends[{name}]: skipped (HTTP {e.code})")
            continue
        except Exception as e:  # noqa: BLE001 - fail soft by design
            notes.append(f"trends[{name}]: skipped ({type(e).__name__})")
            continue
        if r:
            keywords += r.get("keywords", [])
            hashtags += r.get("hashtags", [])
            notes.append(f"trends[{name}]: {len(r.get('keywords', []))} keywords, {len(r.get('hashtags', []))} hashtags")
    if niche:
        hashtags.insert(0, _clean_tag(niche))
    def dedupe(xs):
        seen, out = set(), []
        for x in xs:
            k = x.lower()
            if x and k not in seen:
                seen.add(k)
                out.append(x)
        return out
    for n in notes:
        log(n)
    return {"keywords": dedupe(keywords)[:25], "hashtags": dedupe(hashtags)[:25], "niche": niche, "notes": notes}
