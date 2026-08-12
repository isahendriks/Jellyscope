# Jellyscope private dashboard -- frontend

This folder is the entire "frontend" of the private, password-protected live-stream
dashboard: `index.html`, `style.css`, `media.js`, `status.js`. No build step, no
framework, nothing to install -- open a file, edit it, save, refresh the browser tab.

## Mental model

Think of this as two separate programs talking to each other:

- **The backend** is `Monitor/analyse.py` (Python). It owns all the real data: it
  reads sensors, tracks the upload queues, decides what "backlog mode" means, and
  makes that data available as JSON at one URL (`/status`), plus two images
  (`/latest_frame.jpg`, `/latest_crops_strip.jpg`).
- **The frontend** (this folder) owns only what things look like. It has no idea
  what a camera or a queue is -- it just asks the backend "what's the latest data?"
  once a second, and turns whatever comes back into text and colors on the page.

The browser does the asking, repeatedly, forever (`setInterval` in `status.js` and
`media.js`) -- there are no page reloads, and no buttons or forms anywhere on this
page. It's a read-only dashboard.

**This isolation is a real guarantee, not just a description**: analyse.py serves
every file in this folder as a plain static file -- it never runs, imports, or
interprets anything in here. The worst a mistake in this folder can do is break
how the dashboard *looks*. It is not possible for an edit here to change what gets
recorded, analysed, or uploaded -- that all happens in a completely different part
of analyse.py that never reads anything from this folder.

## The password gate

The dashboard requires a password to view anything at all -- including this
folder's own files (a browser trying to load `style.css` without logging in first
gets blocked too). That check happens entirely in `analyse.py`, *before* any
request reaches this folder. Nothing here needs to know it exists, and there's
nothing to change here if the password itself needs to be updated -- that's a
`Monitor/dashboard_secrets.py` edit (a separate, gitignored file), not a frontend
one.

## How to run/view it

No build step. With `analyse.py` running, visit `http://<jetson-ip>:8080/` in a
browser, log in with the dashboard password when prompted, and you're looking at
these files being served directly.

To make a change: edit a file, save it, refresh the browser tab. If a CSS or JS
change doesn't seem to show up, try a hard refresh (Ctrl+Shift+R, or Cmd+Shift+R
on Mac) -- browsers sometimes cache static files more aggressively than you'd
expect during active editing.

## The `/status` data contract

Everything the page shows comes from one JSON response, at `/status`. The single
source of truth for exactly which fields exist is `_stream_status()` in
`Monitor/analyse.py` -- open that function and read the dictionary it returns (or
open the dashboard, open your browser's dev tools console, and type
`fetch('/status').then(r=>r.json()).then(console.log)` to see a live example).

**One rule that matters**: never rename a key in that Python dictionary without
also updating every place `status.js` reads it (search for `d.<the old name>`).
Nothing else checks that these match -- if they drift apart, that stat will just
silently stop updating, with no error anywhere.

## Two exercises to get started

**1. Change a color (CSS only, zero risk).** Open `style.css`, find
`.status-warn` (under `STATUS COLORS`), and change its color to something else.
Save, refresh. Next time a temperature reading crosses 90% of its max, it'll show
in your new color instead of amber. Nothing else on the page is affected.

**2. Add a stat that's already in the data (small JS change).** The `/status`
JSON actually includes a few fields the page doesn't show yet:
`backlog_mode_enter_threshold`, `backlog_mode_exit_threshold`, `last_upload_ok`,
`send_age_s`, `strobe_temp_c_mean`. `send_age_s` is the easiest to start with --
it's just a number of seconds, no color logic needed.

Walkthrough:
1. Open `status.js`, find `renderConnection(d)` (that's the CONNECTION section --
   `send_age_s` is about the send.py process, so that's the right section).
2. Add a new line to the array it returns, e.g.:
   ```js
   `Heartbeat age: ${fmtAge(d.send_age_s)}`,
   ```
3. Save, refresh the dashboard. You should see it appear in the CONNECTION
   section.

That's the whole loop this codebase uses everywhere: find the field in the JSON,
find the right renderer function, add one line with a `fmt*` helper, refresh.
