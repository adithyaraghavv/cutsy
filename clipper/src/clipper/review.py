"""out/review.html: every clip playable in the browser with copy buttons for title/caption/hashtags."""
from __future__ import annotations

import html
from pathlib import Path

CSS = """body{font-family:system-ui,sans-serif;background:#111;color:#eee;margin:0;padding:24px}
h1{font-size:20px}.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(320px,1fr));gap:20px}
.card{background:#1c1c1c;border-radius:12px;padding:12px}video{width:100%;border-radius:8px;background:#000}
.t{font-weight:700;font-size:16px;margin:8px 0 4px}.m{color:#aaa;font-size:12px;margin:2px 0}
.c{font-size:14px;margin:6px 0;white-space:pre-wrap}button{background:#2d6cdf;color:#fff;border:0;border-radius:6px;
padding:6px 10px;cursor:pointer;margin:4px 6px 0 0}a{color:#7db3ff}.flag{color:#ffb84d}"""

JS = "function cp(id){navigator.clipboard.writeText(document.getElementById(id).innerText)}"


def write(out: Path, res: dict) -> Path:
    cards = []
    for i, c in enumerate(res["clips"]):
        j = c.get("judge") or {}
        tags = " ".join(j.get("hashtags", []))
        cap = f"{j.get('caption', '')}\n\n{tags}".strip()
        sig = ", ".join(f"{k} {v:+.1f}" for k, v in c["signals"].items())
        link = f'<a href="{html.escape(c["link"])}" target="_blank">open in source video</a>' if c.get("link") else ""
        flags = j.get("risk_flags") or []
        cards.append(f"""<div class="card"><video controls preload="metadata" src="{html.escape(c['file'])}"></video>
<div class="t" id="t{i}">{html.escape(c.get('title') or c['hook'])}</div>
<div class="c" id="c{i}">{html.escape(cap)}</div>
<div class="m">{html.escape(c['file'])} &middot; {c['start_s']:.0f}-{c['end_s']:.0f}s &middot; judge {j.get('score', '?')}/10 {link}</div>
<div class="m">{html.escape(j.get('reason', ''))}</div><div class="m">signals: {html.escape(sig)}</div>
{'<div class="m flag">risk: ' + html.escape(', '.join(flags)) + '</div>' if flags else ''}
<button onclick="cp('t{i}')">Copy title</button><button onclick="cp('c{i}')">Copy caption + hashtags</button></div>""")
    p = out / "review.html"
    p.write_text(f"<!doctype html><meta charset=utf-8><title>clipper review</title><style>{CSS}</style>"
                 f"<h1>{html.escape(res.get('title') or 'Clips')} &mdash; {len(res['clips'])} clips to review</h1>"
                 f"<div class=grid>{''.join(cards)}</div><script>{JS}</script>", encoding="utf-8")
    return p
