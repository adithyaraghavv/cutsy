"""Captions from speech (faster-whisper, offline, free) burned in as 2-word chunks with key words highlighted.

faster-whisper is optional: install it yourself (`pip install faster-whisper`). The model
downloads once from Hugging Face on first use and is cached under <cache_dir>/models.
"""
from __future__ import annotations

import hashlib
import json
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

from .errors import ClipperError
from .fonts import find_bold_font, font_family

INSTALL_HELP = (
    "Captions need faster-whisper, which isn't installed.\n"
    "  Install it yourself:  pip install faster-whisper\n"
    "  (The speech model, about 150 MB for the default, downloads once on first use.)\n"
    "  Or run without --captions."
)
STOP = set("the a an and or but of to in on at for with is are was were be been it its this that i you he she we they "
           "me my your his her our their so as if then than too very just do does did have has had not no yes oh um uh "
           "im ive dont cant wont like".split())


def check_available() -> None:
    try:
        import faster_whisper  # noqa: F401
    except ImportError:
        raise ClipperError(INSTALL_HELP) from None


def _transcribe_words(wav: Path, cfg: dict) -> list[dict]:
    """Seam for tests. Returns [{'w': str, 'start': s, 'end': s}] relative to the wav."""
    try:
        from faster_whisper import WhisperModel
    except ImportError:
        raise ClipperError(INSTALL_HELP) from None
    cc = cfg["captions"]
    try:
        model = WhisperModel(cc["model"], device="cpu", compute_type="int8",
                             download_root=str(Path(cfg["cache_dir"]) / "models"))
    except Exception as e:  # noqa: BLE001 - download/network problems -> friendly
        raise ClipperError(
            f"Could not load the speech model {cc['model']!r} ({type(e).__name__}). It downloads once from "
            "Hugging Face; check your connection, or run without --captions.") from None
    import wave

    import numpy as np
    with wave.open(str(wav), "rb") as wf:   # decode ourselves: avoids faster-whisper's `av` decoder
        audio = np.frombuffer(wf.readframes(wf.getnframes()), dtype="<i2").astype(np.float32) / 32768.0
    words = []
    try:
        segs, _ = model.transcribe(audio, language=cc.get("language") or None, word_timestamps=True,
                                   vad_filter=True, beam_size=1)
        for s in segs:
            for w in s.words or []:
                t = w.word.strip()
                if t:
                    words.append({"w": t, "start": float(w.start), "end": float(w.end)})
    except Exception as e:  # noqa: BLE001
        raise ClipperError(f"Speech recognition failed ({type(e).__name__}: {str(e)[:150]}). "
                           "Run without --captions to skip it.") from None
    return words


def transcribe_window(media, start_abs: float, end_abs: float, cfg: dict, key: str) -> list[dict]:
    """Words with ABSOLUTE source times for [start_abs, end_abs]; cached by key."""
    cp = Path(cfg["cache_dir"]) / "transcripts" / f"{key}.json"
    if cp.is_file():
        try:
            return json.loads(cp.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            pass
    if not media.has_audio:
        return []
    ff = shutil.which("ffmpeg")
    start = max(0.0, start_abs - media.offset)
    dur = max(0.5, min(end_abs - start_abs, media.duration - start))
    with tempfile.TemporaryDirectory(prefix="clipper_asr_") as td:
        wav = Path(td) / "a.wav"
        r = subprocess.run([ff, "-y", "-v", "error", "-ss", f"{start:.3f}", "-t", f"{dur:.3f}", "-i", str(media.path),
                            "-vn", "-ac", "1", "-ar", "16000", str(wav)], capture_output=True, text=True)
        if r.returncode != 0:
            raise ClipperError(f"Could not extract audio for transcription: {r.stderr.strip()[-200:]}")
        rel = _transcribe_words(wav, cfg)
    words = [{"w": w["w"], "start": round(start_abs + w["start"], 3), "end": round(start_abs + w["end"], 3)} for w in rel]
    cp.parent.mkdir(parents=True, exist_ok=True)
    cp.write_text(json.dumps(words), encoding="utf-8")
    return words


def transcript_text(words: list[dict]) -> str | None:
    if not words:
        return None
    out, last = [], -99.0
    for w in words:
        if w["start"] - last >= 5:
            out.append(f"[{w['start']:.1f}s]")
            last = w["start"]
        out.append(w["w"])
    return " ".join(out)


def slice_words(words: list[dict], start: float, end: float) -> list[dict]:
    """Words inside [start, end], re-timed relative to the clip start."""
    return [{"w": w["w"], "start": round(w["start"] - start, 3), "end": round(min(w["end"], end) - start, 3)}
            for w in words if w["start"] >= start - 0.05 and w["start"] < end]


def is_key(word: str) -> bool:
    core = re.sub(r"[^\w']", "", word).lower()
    return bool(core) and (core.isdigit() or word.endswith(("!", "?")) or (len(core) >= 5 and core not in STOP)
                           or word.isupper() and len(core) > 2)


def chunk_words(words: list[dict], size: int = 2) -> list[dict]:
    """2-word chunks: [{'start','end','words':[(text, highlighted)]}]. At most one highlighted word per chunk."""
    chunks = []
    for i in range(0, len(words), size):
        grp = words[i:i + size]
        keys = [j for j, w in enumerate(grp) if is_key(w["w"])]
        pick = max(keys, key=lambda j: len(grp[j]["w"])) if keys else None
        nxt = words[i + size]["start"] if i + size < len(words) else None
        end = grp[-1]["end"]
        end = min(nxt, end + 0.35) if nxt is not None else end + 0.35
        chunks.append({"start": grp[0]["start"], "end": max(end, grp[0]["start"] + 0.25),
                       "words": [(w["w"], j == pick) for j, w in enumerate(grp)]})
    return chunks


def _ts(t: float) -> str:
    t = max(t, 0.0)
    h, rem = divmod(t, 3600)
    m, s = divmod(rem, 60)
    return f"{int(h)}:{int(m):02d}:{s:05.2f}"


def build_ass(words: list[dict], cfg: dict) -> str:
    r, cc = cfg["render"], cfg["captions"]
    font = font_family(find_bold_font())
    hi = cc["highlight_bgr"]
    head = (
        "[Script Info]\nScriptType: v4.00+\n"
        f"PlayResX: {r['width']}\nPlayResY: {r['height']}\nWrapStyle: 2\nScaledBorderAndShadow: yes\n\n"
        "[V4+ Styles]\nFormat: Name,Fontname,Fontsize,PrimaryColour,SecondaryColour,OutlineColour,BackColour,Bold,"
        "Italic,Underline,StrikeOut,ScaleX,ScaleY,Spacing,Angle,BorderStyle,Outline,Shadow,Alignment,MarginL,MarginR,"
        f"MarginV,Encoding\nStyle: Cap,{font},{cc['fontsize']},&H00FFFFFF,&H00FFFFFF,&H00000000,&H64000000,1,0,0,0,"
        f"100,100,0,0,1,7,2,2,60,60,{cc['margin_v']},1\n\n"
        "[Events]\nFormat: Layer,Start,End,Style,Name,MarginL,MarginR,MarginV,Effect,Text\n")
    lines = []
    for ch in chunk_words(words, int(cc["chunk_words"])):
        txt = " ".join((f"{{\\c&H{hi}&}}{t.upper()}{{\\c&HFFFFFF&}}" if h else t.upper()).replace("\n", " ")
                       for t, h in ch["words"])
        lines.append(f"Dialogue: 0,{_ts(ch['start'])},{_ts(ch['end'])},Cap,,0,0,0,,{txt}")
    return head + "\n".join(lines) + "\n"
