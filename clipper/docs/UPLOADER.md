# Optional uploader (YouTube Shorts, Instagram Reels)

Off by default. Nothing is uploaded unless you pass `--upload` **and** type `y` for each clip. It only runs from
the command line in an interactive terminal; the MCP server and background workers cannot upload (a test enforces
this). It uses the official APIs with your own accounts: no scraping, no login automation, no passwords.

```bash
# render, then be asked clip by clip:
python -m clipper video.mp4 --upload youtube              # uploaded as PRIVATE unless --privacy is given
python -m clipper video.mp4 --upload instagram
python -m clipper video.mp4 --upload both --privacy unlisted
# upload clips you already rendered into ./out:
python -m clipper x --upload youtube --upload-only --out out
```
Before each prompt clipper shows the file, hook, caption, hashtags, visibility and any risk flags. Anything other
than `y` skips that clip. Upload only content you have permission to post.

## Credentials (environment variables only)
| Variable | For |
|---|---|
| `YOUTUBE_OAUTH_CLIENT_ID`, `YOUTUBE_OAUTH_CLIENT_SECRET` | your Google OAuth client (type "Desktop app") |
| `YOUTUBE_REFRESH_TOKEN` | optional; otherwise stored after first consent, see below |
| `INSTAGRAM_ACCESS_TOKEN`, `INSTAGRAM_USER_ID` | your Instagram Business/Creator account |

Secrets are never printed or logged, error messages are scrubbed of them, and `.env` is git-ignored.

### YouTube setup (once)
1. In Google Cloud Console create a project, enable **YouTube Data API v3**, configure the OAuth consent screen
   (add yourself as a test user), and create an OAuth client of type **Desktop app**.
2. Export the client id and secret as above, run an upload, open the printed URL in your browser and approve.
   clipper receives the code on a one-time `127.0.0.1` loopback port (PKCE) and saves only a **refresh token** to
   `~/.config/clipper/youtube_token.json` (mode 600, outside the repo; set `CLIPPER_CONFIG_DIR` to move it).
   Delete that file, or revoke access at myaccount.google.com/permissions, to log clipper out.
3. Things to know (from Google's documented behaviour; verify in your console): videos uploaded through an API
   project that hasn't passed Google's audit are locked to private; each upload costs about 1,600 quota units
   against a default 10,000 per day (roughly 6 uploads/day).

### Instagram setup (once)
Needs an Instagram **Business or Creator** account linked to a Facebook Page, a Meta developer app with the
content-publishing permission, and a long-lived access token plus your Instagram user id. Reels are sent with the
resumable upload flow, polled until Instagram reports `FINISHED`, and only then published.

## Status
Both flows are covered by unit tests against mocked HTTP (request shapes, per-clip confirmation, privacy default,
error handling, secret scrubbing, token file permissions). They have **never been run against the real YouTube or
Instagram APIs**. Request details follow the APIs as documented, but treat your first uploads as a test:
use `--privacy private` and a throwaway clip.
