"""OPTIONAL uploader: YouTube Shorts (YouTube Data API v3) and Instagram Reels (Instagram Graph API).

Safety rules enforced here:
  * runs only from the CLI with --upload; the MCP server and background worker never import this module
  * needs an interactive terminal and a typed 'y' for EACH clip; there is no "yes to all"
  * official APIs and OAuth/access tokens only: no scraping, no login automation, no passwords
  * credentials come from environment variables; tokens/secrets are never printed or logged
  * YouTube uploads default to PRIVATE; you must pass --privacy to make them unlisted/public
"""
from __future__ import annotations

import base64
import hashlib
import json
import os
import secrets
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

from .errors import ClipperError

GRAPH = "https://graph.facebook.com/v21.0"
YT_UPLOAD = "https://www.googleapis.com/upload/youtube/v3/videos"
YT_SCOPE = "https://www.googleapis.com/auth/youtube.upload"

YT_HELP = (
    "YouTube upload needs your own Google OAuth client (Desktop app type) with the YouTube Data API enabled.\n"
    "  Set these environment variables (never put them in files in the repo):\n"
    "    YOUTUBE_OAUTH_CLIENT_ID, YOUTUBE_OAUTH_CLIENT_SECRET\n"
    "  Setup steps: docs/UPLOADER.md"
)
IG_HELP = (
    "Instagram upload needs an Instagram Business/Creator account, a Meta developer app and an access token.\n"
    "  Set these environment variables: INSTAGRAM_ACCESS_TOKEN, INSTAGRAM_USER_ID\n"
    "  Setup steps: docs/UPLOADER.md"
)


# ------------------------------------------------------------------ helpers ----

def _is_interactive() -> bool:
    return sys.stdin.isatty() and sys.stdout.isatty()


def _http(method: str, url: str, *, headers: dict | None = None, data: bytes | None = None,
          timeout: float = 600.0) -> tuple[int, dict, dict | None]:
    """Single network seam (tests replace this). Returns (status, lowercase headers, json-or-None)."""
    req = urllib.request.Request(url, data=data, method=method, headers=headers or {})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:  # noqa: S310 - fixed https API hosts
            body = r.read()
            status, hdrs = r.status, {k.lower(): v for k, v in r.headers.items()}
    except urllib.error.HTTPError as e:
        body, status, hdrs = e.read(), e.code, {k.lower(): v for k, v in e.headers.items()}
    try:
        parsed = json.loads(body.decode("utf-8")) if body else None
    except (ValueError, UnicodeDecodeError):
        parsed = None
    return status, hdrs, parsed


def _scrub(text: str, secrets_: list[str]) -> str:
    for s in secrets_:
        if s:
            text = text.replace(s, "***")
    return text


def _check(status: int, body: dict | None, what: str, secrets_: list[str]) -> dict:
    if 200 <= status < 300:
        return body or {}
    err = ""
    if isinstance(body, dict):
        e = body.get("error")
        err = e.get("message", "") if isinstance(e, dict) else str(e or body.get("error_description", ""))
    raise ClipperError(_scrub(f"{what} failed (HTTP {status}): {err or 'no details'}", secrets_))


def _env(name: str) -> str | None:
    v = os.environ.get(name, "").strip()
    return v or None


def config_dir() -> Path:
    return Path(os.environ.get("CLIPPER_CONFIG_DIR") or Path.home() / ".config" / "clipper")


# ------------------------------------------------------------------ YouTube ----

def _token_file() -> Path:
    return config_dir() / "youtube_token.json"


def _save_refresh_token(tok: str) -> None:
    p = _token_file()
    p.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(p, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        json.dump({"refresh_token": tok}, f)
    try:
        os.chmod(p, 0o600)
    except OSError:
        pass


def _load_refresh_token() -> str | None:
    if _env("YOUTUBE_REFRESH_TOKEN"):
        return _env("YOUTUBE_REFRESH_TOKEN")
    p = _token_file()
    if p.is_file():
        try:
            return json.loads(p.read_text(encoding="utf-8")).get("refresh_token")
        except (OSError, ValueError):
            return None
    return None


def _wait_for_code(server: HTTPServer, state: str, timeout: float = 300.0) -> str:
    """Handle the single OAuth redirect on the loopback server and return the auth code."""
    result: dict = {}

    class H(BaseHTTPRequestHandler):
        def do_GET(self):  # noqa: N802
            q = urllib.parse.parse_qs(urllib.parse.urlparse(self.path).query)
            result["code"] = (q.get("code") or [None])[0]
            result["state"] = (q.get("state") or [None])[0]
            result["error"] = (q.get("error") or [None])[0]
            self.send_response(200)
            self.send_header("Content-Type", "text/plain; charset=utf-8")
            self.end_headers()
            self.wfile.write(b"clipper: authorisation received. You can close this tab.")

        def log_message(self, *a):  # silence
            pass

    server.timeout = timeout
    server.RequestHandlerClass = H
    server.handle_request()
    if result.get("error"):
        raise ClipperError(f"Google authorisation was refused ({result['error']}).")
    if not result.get("code") or result.get("state") != state:
        raise ClipperError("Google authorisation did not complete (timed out or state mismatch).")
    return result["code"]


def _yt_authorize(cid: str, secret: str, log) -> str:
    verifier = secrets.token_urlsafe(64)
    challenge = base64.urlsafe_b64encode(hashlib.sha256(verifier.encode()).digest()).rstrip(b"=").decode()
    state = secrets.token_urlsafe(16)
    srv = HTTPServer(("127.0.0.1", 0), BaseHTTPRequestHandler)
    redirect = f"http://127.0.0.1:{srv.server_port}"
    url = "https://accounts.google.com/o/oauth2/v2/auth?" + urllib.parse.urlencode({
        "client_id": cid, "redirect_uri": redirect, "response_type": "code", "scope": YT_SCOPE,
        "code_challenge": challenge, "code_challenge_method": "S256", "access_type": "offline",
        "prompt": "consent", "state": state})
    log("Open this URL in your browser and approve access for YOUR Google account:\n  " + url)
    try:
        code = _wait_for_code(srv, state)
    finally:
        srv.server_close()
    data = urllib.parse.urlencode({"code": code, "client_id": cid, "client_secret": secret, "redirect_uri": redirect,
                                   "grant_type": "authorization_code", "code_verifier": verifier}).encode()
    st, _, body = _http("POST", "https://oauth2.googleapis.com/token", data=data,
                        headers={"Content-Type": "application/x-www-form-urlencoded"})
    tok = _check(st, body, "Google token exchange", [secret, code, verifier])
    if not tok.get("refresh_token"):
        raise ClipperError("Google did not return a refresh token. Remove clipper's access at "
                           "myaccount.google.com/permissions and try again.")
    _save_refresh_token(tok["refresh_token"])
    log(f"Saved a refresh token to {_token_file()} (owner-only file). Remove it any time to revoke clipper's access locally.")
    return tok["access_token"]


def yt_access_token(log) -> str:
    cid, secret = _env("YOUTUBE_OAUTH_CLIENT_ID"), _env("YOUTUBE_OAUTH_CLIENT_SECRET")
    if not cid or not secret:
        raise ClipperError(YT_HELP)
    rt = _load_refresh_token()
    if not rt:
        return _yt_authorize(cid, secret, log)
    data = urllib.parse.urlencode({"client_id": cid, "client_secret": secret, "refresh_token": rt,
                                   "grant_type": "refresh_token"}).encode()
    st, _, body = _http("POST", "https://oauth2.googleapis.com/token", data=data,
                        headers={"Content-Type": "application/x-www-form-urlencoded"})
    return _check(st, body, "Google token refresh", [cid, secret, rt])["access_token"]


def yt_metadata(clip: dict, privacy: str) -> dict:
    hook = (clip.get("hook") or "").strip().title()
    title = (f"{hook} #Shorts" if hook else "Clip #Shorts")[:100]
    j = clip.get("judge") or {}
    tags = [t.lstrip("#") for t in j.get("hashtags", [])]
    desc = "\n\n".join(x for x in [j.get("caption", ""), " ".join(j.get("hashtags", [])) + " #Shorts"] if x)
    return {"snippet": {"title": title, "description": desc[:4900], "tags": tags, "categoryId": "20"},
            "status": {"privacyStatus": privacy, "selfDeclaredMadeForKids": False}}


def youtube_upload(path: Path, clip: dict, privacy: str, log) -> str:
    token = yt_access_token(log)
    size = path.stat().st_size
    meta = json.dumps(yt_metadata(clip, privacy)).encode()
    st, hdrs, body = _http("POST", f"{YT_UPLOAD}?uploadType=resumable&part=snippet,status", data=meta, headers={
        "Authorization": f"Bearer {token}", "Content-Type": "application/json; charset=UTF-8",
        "X-Upload-Content-Length": str(size), "X-Upload-Content-Type": "video/mp4"})
    _check(st, body, "YouTube upload start", [token])
    loc = hdrs.get("location")
    if not loc:
        raise ClipperError("YouTube did not return an upload location.")
    st, _, body = _http("PUT", loc, data=path.read_bytes(),
                        headers={"Content-Type": "video/mp4", "Content-Length": str(size)})
    res = _check(st, body, "YouTube upload", [token])
    return res.get("id", "")


# ---------------------------------------------------------------- Instagram ----

def ig_caption(clip: dict) -> str:
    j = clip.get("judge") or {}
    return "\n\n".join(x for x in [j.get("caption", "") or clip.get("hook", ""), " ".join(j.get("hashtags", []))] if x)[:2200]


def instagram_upload(path: Path, clip: dict, log, poll_s: float = 5.0, max_polls: int = 60) -> str:
    token, uid = _env("INSTAGRAM_ACCESS_TOKEN"), _env("INSTAGRAM_USER_ID")
    if not token or not uid:
        raise ClipperError(IG_HELP)
    size = path.stat().st_size
    auth = {"Authorization": f"Bearer {token}", "Content-Type": "application/x-www-form-urlencoded"}
    form = urllib.parse.urlencode({"media_type": "REELS", "upload_type": "resumable",
                                   "caption": ig_caption(clip)}).encode()
    st, _, body = _http("POST", f"{GRAPH}/{uid}/media", data=form, headers=auth)
    cont = _check(st, body, "Instagram container creation", [token])
    container, uri = cont.get("id"), cont.get("uri")
    if not container or not uri:
        raise ClipperError("Instagram did not return an upload container.")
    st, _, body = _http("POST", uri, data=path.read_bytes(), headers={
        "Authorization": f"OAuth {token}", "offset": "0", "file_size": str(size)})
    _check(st, body, "Instagram video upload", [token])
    for _ in range(max_polls):
        st, _, body = _http("GET", f"{GRAPH}/{container}?fields=status_code", headers={"Authorization": f"Bearer {token}"})
        code = _check(st, body, "Instagram status check", [token]).get("status_code")
        if code == "FINISHED":
            break
        if code in ("ERROR", "EXPIRED"):
            raise ClipperError(f"Instagram could not process the video (status {code}).")
        time.sleep(poll_s)
    else:
        raise ClipperError("Instagram is still processing the video; check the app later. Nothing was published.")
    st, _, body = _http("POST", f"{GRAPH}/{uid}/media_publish",
                        data=urllib.parse.urlencode({"creation_id": container}).encode(), headers=auth)
    return _check(st, body, "Instagram publish", [token]).get("id", "")


# ------------------------------------------------------------------- driver ----

def run_uploads(res: dict, out: Path, platforms: list[str], privacy: str, log=print, ask=None) -> list[dict]:
    """Ask before EVERY upload. Returns a list of {clip, platform, id} for what was actually uploaded."""
    ask = ask or input
    if not _is_interactive():
        raise ClipperError("Uploading needs an interactive terminal so you can confirm each clip. "
                           "Nothing was uploaded.")
    if "youtube" in platforms and not (_env("YOUTUBE_OAUTH_CLIENT_ID") and _env("YOUTUBE_OAUTH_CLIENT_SECRET")):
        raise ClipperError(YT_HELP)
    if "instagram" in platforms and not (_env("INSTAGRAM_ACCESS_TOKEN") and _env("INSTAGRAM_USER_ID")):
        raise ClipperError(IG_HELP)
    done: list[dict] = []
    for clip in res["clips"]:
        path = out / clip["file"]
        dur = clip["end_s"] - clip["start_s"]
        j = clip.get("judge") or {}
        for plat in platforms:
            log(f"\n--- {clip['file']} -> {plat} ({dur:.0f}s) ---\n  hook: {clip['hook']}\n  caption: {j.get('caption', '')}\n"
                f"  hashtags: {' '.join(j.get('hashtags', []))}"
                + (f"\n  visibility: {privacy}" if plat == "youtube" else ""))
            if j.get("risk_flags"):
                log(f"  RISK FLAGS: {', '.join(j['risk_flags'])}")
            if plat == "instagram" and dur > 90:
                log("  note: longer than 90 s; Instagram may reject or treat it differently.")
            if ask(f"Upload {clip['file']} to {plat}? Type 'y' to confirm, anything else skips: ").strip().lower() != "y":
                log("  skipped")
                continue
            try:
                vid = youtube_upload(path, clip, privacy, log) if plat == "youtube" else instagram_upload(path, clip, log)
            except ClipperError as e:
                log(f"  FAILED: {e}")
                continue
            log(f"  uploaded ({plat} id {vid})")
            done.append({"clip": clip["file"], "platform": plat, "id": vid})
    return done
