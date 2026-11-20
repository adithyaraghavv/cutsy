"""Detached upload process: python -m clipper.upload_worker <job_dir> <clip> <platform> <privacy>.
Only started by the MCP tool upload_clip, and only for a clip the user approved."""
from __future__ import annotations

import sys
import time
import traceback
from pathlib import Path

from . import jobs, uploader
from .errors import ClipperError


def run(job_dir, clip: str, platform: str, privacy: str, log=print) -> dict:
    d = Path(job_dir)
    jid = d.name
    res = jobs.clips_of(jid)
    c = next((x for x in (res or {}).get("clips", []) if x["file"] == clip), None)
    if c is None:
        return jobs.set_upload(jid, clip, platform, status="failed", error=f"Unknown clip {clip}")
    jobs.set_upload(jid, clip, platform, status="running", privacy=privacy, started=time.time(), error="")
    path = d / "out" / clip
    try:
        if platform == "youtube":
            vid = uploader.youtube_upload(path, c, privacy, log, interactive=False)
            return jobs.set_upload(jid, clip, platform, status="done", id=vid, url=f"https://youtu.be/{vid}" if vid else "")
        if platform == "instagram":
            mid = uploader.instagram_upload(path, c, log)
            return jobs.set_upload(jid, clip, platform, status="done", id=mid, url="")
        return jobs.set_upload(jid, clip, platform, status="failed", error=f"Unknown platform {platform}")
    except ClipperError as e:
        return jobs.set_upload(jid, clip, platform, status="failed", error=str(e))
    except Exception as e:  # noqa: BLE001
        print(traceback.format_exc(), flush=True)
        return jobs.set_upload(jid, clip, platform, status="failed",
                               error=f"Unexpected error ({type(e).__name__}); see upload_log.txt in the job folder.")


def main(argv=None) -> int:
    a = argv or sys.argv[1:]
    run(a[0], a[1], a[2], a[3])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
