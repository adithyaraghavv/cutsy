from __future__ import annotations

from pathlib import Path


def _fmt(t: float) -> str:
    h, rem = divmod(int(t), 3600)
    m, s = divmod(rem, 60)
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m}:{s:02d}"


def write(out: Path, res: dict) -> Path:
    L = [f"# Clip report: {res.get('title') or res.get('source')}", "",
         f"- Source: `{res['source']}`", f"- Judge: **{res['judge']}**  |  QA: **{res['qa_backend']}**",
         f"- Claude API cost: **${res['cost_usd']:.2f}** (free backends; no API calls made)", ""]
    for n in res.get("trend_notes", []):
        L.append(f"- {n}")
    L.append("")
    for c in res["clips"]:
        j = c.get("judge") or {}
        L += [f"## {c['file']}  ({_fmt(c['start_s'])} - {_fmt(c['end_s'])}, {c['end_s'] - c['start_s']:.0f} s)", "",
              f"- **Title on video:** {c.get('title') or c['hook']}",
              *([f"- **Short hook:** {c['hook']}"] if c.get('title') else []),
              *([f"- **Source:** {c['link']}"] if c.get('link') else []), f"- **Caption:** {j.get('caption', '')}",
              f"- **Hashtags:** {' '.join(j.get('hashtags', []))}",
              f"- **Judge score:** {j.get('score', 'n/a')}/10 - {j.get('reason', '')}",
              f"- **Risk flags:** {', '.join(j.get('risk_flags', [])) or 'none'}"
              + ("" if j.get("risk_assessed", True) else " (NOT assessed: heuristic judge cannot see the video)"),
              f"- **Signal score:** {c['score']}  |  breakdown (mean z): "
              + ", ".join(f"{k} {v:+.2f}" for k, v in c["signals"].items())]
        if c.get("caption_words"):
            L.append(f"- **Captions:** {len(c['caption_words'])} words burned in (2-word chunks, key words highlighted)")
        for q in c.get("qa", []):
            checks = ", ".join(f"{k}={'ok' if v else 'FAIL'}" for k, v in q.get("checks", {}).items())
            L.append(f"- **QA ({q['backend']}):** {'PASS' if q['passed'] else 'FAIL'} {checks} "
                     + "; ".join(q.get("notes", [])) + (f" -> fix {q['fix']}" if q.get("fix") else ""))
        if c.get("rerendered"):
            L.append("- Re-rendered once after QA failure.")
        L.append("")
    if res.get("not_selected"):
        L += ["## Other judged candidates (not selected)", ""]
        L += [f"- {x['id']}: score {x['score']} ({_fmt(x['start_s'])}-{_fmt(x['end_s'])}) {x['reason']}" for x in res["not_selected"]]
    p = out / "report.md"
    p.write_text("\n".join(L) + "\n", encoding="utf-8")
    return p
