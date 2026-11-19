from __future__ import annotations

import argparse
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
    p.add_argument("--n", type=int, default=3, help="number of clips to render (default 3)")
    p.add_argument("--window", type=float, default=55, help="clip length in seconds (default 55)")
    p.add_argument("--range", dest="rng", metavar="START-END", help="analyse only this span, in seconds")
    p.add_argument("--niche", default=None, help='topic, e.g. "gta6" (used from Phase 2)')
    p.add_argument("--budget-usd", type=float, default=2.0, help="hard cap on Claude API spend (used from Phase 2)")
    p.add_argument("--fill", action="store_true", help="crop to fill 9:16 instead of blurred side bars")
    p.add_argument("--out", default="out", help="output directory (default: out)")
    p.add_argument("--config", default=None, help="path to config.yaml")
    p.add_argument("--version", action="version", version=f"clipper {__version__}")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.n < 1 or args.window < 5:
            raise ClipperError("--n must be at least 1 and --window at least 5 seconds.")
        if args.budget_usd <= 0:
            raise ClipperError("--budget-usd must be greater than 0.")
        cfg = config.load(args.config)
        rng = ingest.parse_range(args.rng)
        res = pipeline.run(args.source, n=args.n, window=args.window, rng=rng, fill=args.fill,
                           out=Path(args.out), cfg=cfg)
    except ClipperError as e:
        print(f"clipper: {e}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print("clipper: interrupted", file=sys.stderr)
        return 130
    print(f"\nDone: {len(res['clips'])} clip(s) in {args.out}/ (see clips.json)")
    return 0
