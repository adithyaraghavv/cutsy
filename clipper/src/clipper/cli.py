from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from . import __version__, config, ingest, pipeline
from .errors import ClipperError


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="clipper",
        description="Turn a long YouTube/Twitch VOD (URL or local file) into vertical 9:16 clips.",
        epilog="Only clip streams you have the streamer's written permission to clip.",
    )
    p.add_argument("source", help="video URL or local file path")
    p.add_argument("--n", type=int, default=None, help="number of clips to render (default 3; 15 with --scan)")
    p.add_argument("--window", type=float, default=None, help="clip length in seconds (default 55; 40 with --scan)")
    p.add_argument("--range", dest="rng", metavar="START-END", help="analyse only this span, in seconds")
    p.add_argument("--niche", default=None, help='topic, e.g. "gta6": feeds trends, hooks and hashtags')
    p.add_argument("--budget-usd", type=float, default=2.0,
                   help="hard cap on paid-API spend; the free backends spend $0 (applies to future paid judges)")
    p.add_argument("--fill", action="store_true", help="crop to fill 9:16 instead of blurred side bars")
    p.add_argument("--out", default="out", help="output directory (default: out)")
    p.add_argument("--judge", default="heuristic", choices=["heuristic", "chat"],
                   help="heuristic: offline signal judge (default). chat: Claude in your chat judges an evidence pack")
    p.add_argument("--judgments", default=None, metavar="FILE", help="judgments.json written by Claude (chat judge)")
    p.add_argument("--shortlist", type=int, default=None, help="candidates to judge (default 5 for chat, all for heuristic)")
    p.add_argument("--trends", default=None, metavar="FILE", help="hand-edited trends.json (default: ./trends.json if present)")
    p.add_argument("--qa", default="heuristic", choices=["heuristic", "chat", "none"],
                   help="quality check of rendered clips (chat also writes a pack for Claude to review)")
    p.add_argument("--qa-results", default=None, metavar="FILE", help="qa.json written by Claude; re-renders failing clips once")
    p.add_argument("--captions", action="store_true",
                   help="burn in speech captions (2-word chunks, key words highlighted); needs faster-whisper")
    p.add_argument("--upload", choices=["youtube", "instagram", "both"], default=None,
                   help="OPTIONAL: after rendering, offer to upload each clip (asks y/n per clip; needs your own API credentials)")
    p.add_argument("--privacy", choices=["private", "unlisted", "public"], default="private",
                   help="YouTube visibility for uploads (default: private)")
    p.add_argument("--upload-only", action="store_true", help="skip rendering; offer to upload the clips already in --out")
    p.add_argument("--scan", action="store_true",
                   help="scan the WHOLE video/stream (audio + chat replay + most-replayed), then cut the best moments (default 15 clips)")
    p.add_argument("--transcribe", action="store_true",
                   help="with --scan: transcribe candidates to find reactions and write titles from speech (needs faster-whisper)")
    p.add_argument("--min-score", type=float, default=0.0,
                   help="leave out candidates the judge scored below this (1-10); e.g. 4 drops pause menus and dead scenes")
    p.add_argument("--no-chat", action="store_true", help="skip downloading the live-chat replay (much faster for very long streams)")
    p.add_argument("--layout", choices=["blur", "split"], default=None,
                   help="blur: footage centred on a blurred background (default). split: facecam on top, gameplay below (needs --facecam)")
    p.add_argument("--facecam", default=None, metavar="X,Y,W,H",
                   help="facecam rectangle as fractions of the frame, e.g. 0.72,0.55,0.28,0.45 (turns on --layout split)")
    p.add_argument("--gameplay", default=None, metavar="X,Y,W,H",
                   help="with --facecam: region of the frame used for the gameplay panel (skips the webcam corner / black bars)")
    p.add_argument("--grid-frame", type=float, default=None, metavar="SECONDS",
                   help="write out/grid.png (a frame with a % grid) from a LOCAL video file so you can read off --facecam numbers")
    p.add_argument("--config", default=None, help="path to config.yaml")
    p.add_argument("--version", action="version", version=f"clipper {__version__}")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        args.n = args.n if args.n is not None else (15 if args.scan else 3)
        args.window = args.window if args.window is not None else (40.0 if args.scan else 55.0)
        if args.n < 1 or args.window < 5:
            raise ClipperError("--n must be at least 1 and --window at least 5 seconds.")
        if args.budget_usd <= 0:
            raise ClipperError("--budget-usd must be greater than 0.")
        cfg = config.load(args.config)
        if args.facecam:
            try:
                cfg["render"]["facecam"] = [float(v) for v in args.facecam.split(",")]
            except ValueError:
                raise ClipperError("--facecam must be four numbers like 0.72,0.55,0.28,0.45") from None
            cfg["render"]["layout"] = args.layout or "split"
        elif args.layout == "split":
            raise ClipperError("--layout split needs --facecam X,Y,W,H (use --grid-frame to find the numbers).")
        elif args.layout:
            cfg["render"]["layout"] = args.layout
        if args.gameplay:
            try:
                cfg["render"]["gameplay"] = [float(v) for v in args.gameplay.split(",")]
            except ValueError:
                raise ClipperError("--gameplay must be four numbers like 0.30,0.05,0.70,0.90") from None
        if args.grid_frame is not None:
            from . import grid
            print(f"Wrote {grid.write(args.source, args.grid_frame, Path(args.out))}\nOpen it: the numbers are % of the width (top) and height (left).")
            return 0
        out = Path(args.out)
        if args.qa_results:
            pipeline.apply_qa_results(args.source, out=out, cfg=cfg, qa_results=args.qa_results)
            return 0
        if args.upload_only and not args.upload:
            raise ClipperError("--upload-only needs --upload youtube|instagram|both.")
        if args.upload_only:
            from . import uploader
            cj = out / "clips.json"
            if not cj.is_file():
                raise ClipperError(f"No {cj} found. Render clips first.")
            uploader.run_uploads(json.loads(cj.read_text(encoding="utf-8")), out,
                                 ["youtube", "instagram"] if args.upload == "both" else [args.upload], args.privacy)
            return 0
        rng = ingest.parse_range(args.rng)
        res = pipeline.run(args.source, n=args.n, window=args.window, rng=rng, fill=args.fill, out=out,
                           cfg=cfg, judge_name=args.judge, judgments=args.judgments, shortlist=args.shortlist,
                           niche=args.niche, trends_path=args.trends, qa_name=args.qa, captions=args.captions, chat=not args.no_chat,
                           scan=args.scan, transcribe=args.transcribe, min_score=args.min_score)
        if args.upload and res.get("status") == "done":
            from . import uploader
            uploader.run_uploads(res, out, ["youtube", "instagram"] if args.upload == "both" else [args.upload],
                                 args.privacy)
        elif args.upload:
            print("clipper: nothing to upload yet; finish judging first.", file=sys.stderr)
    except ClipperError as e:
        print(f"clipper: {e}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print("clipper: interrupted", file=sys.stderr)
        return 130
    return 0
