# Jellyscope public livestream

This folder is a complete, self-contained public website: `server.py` (the
backend) plus `index.html`, `style.css`, `media.js`, `crops.js` (the
frontend, no build step -- open a file, edit it, save, refresh the browser
tab). It runs on **server-lab** (this machine), not the Jetson.

## Mental model

- **The Jetson** does the actual monitoring at the research station, and
  uploads its results into this machine's `jellyscope_incoming/` folder via
  scp (a completely separate process from anything in this folder).
- **server.py** (this folder) reads that incoming folder and republishes a
  small, curated, read-only subset of it as JSON + images, on one port.
- **The frontend** (the other four files) owns only what things look like. It
  asks server.py "what's new?" every couple of seconds and turns the answer
  into a live image and a scrolling list of recent sightings.

This is a sibling of `../frontend_private/`, not a variant of it. That folder
is the password-gated ops dashboard the Jetson itself serves, for the people
running the station -- device temperatures, upload queues, leak detection.
This folder has **no login at all** (a deliberate, confirmed decision -- this
site is meant to be fully public) and only ever shows the live camera view
and recently identified sightings. See `server.py`'s module docstring for the
full reasoning on why public traffic lands here instead of on the Jetson, and
exactly what's left out on purpose.

## How to run it

```
pip install flask
python server.py
```

Visit `http://localhost:8090/` (or whatever `JELLYSCOPE_PUBLIC_PORT` is set
to). Two environment variables, both optional:

- `JELLYSCOPE_INCOMING` -- path to the folder the Jetson's uploads land in.
  Defaults to `C:\Users\jellyfish\jellyscope_incoming`, this machine's real
  path as of writing.
- `JELLYSCOPE_PUBLIC_PORT` -- defaults to `8090`.

To make a frontend change: edit a file, save it, refresh the browser tab (hard
refresh with Ctrl+Shift+R if a CSS/JS change doesn't seem to show up). To
change what data is served, edit `server.py` and restart it.

## Keeping it running (Task Scheduler)

This needs to run continuously, survive reboots, and restart itself if it
ever crashes. This machine doesn't have anything like `systemd` or `nssm`
installed, but Windows' built-in Task Scheduler covers all three:

```
schtasks /create /tn "JellyscopePublicSite" ^
  /tr "\"C:\Users\IsaH\AppData\Local\Programs\Python\Python311\python.exe\" \"C:\Users\IsaH\Documents\Github\Jellyscope\Jetson_monitoring\frontend_public\server.py\"" ^
  /sc onstart /ru SYSTEM /rl HIGHEST /f
```

This was deliberately **not** run automatically -- registering a startup task
is a persistent system change, worth doing with your eyes open rather than as
a side effect of a chat message. Run it yourself (as an Administrator, which
`IsaH` already is), or ask for it to be run for you.

After creating it, `schtasks /run /tn "JellyscopePublicSite"` starts it
immediately without waiting for a reboot; `schtasks /end /tn "JellyscopePublicSite"`
/ `schtasks /delete /tn "JellyscopePublicSite" /f` stop/remove it.

## Making it public (Tailscale Funnel)

server.py only listens locally/on the university network until this step --
Funnel is what actually exposes port 8090 to the public internet, without any
university firewall changes (it only needs the same outbound connection
Tailscale already has).

**This is currently blocked**: Tailscale on this machine is locked to the
`jellyfish` Windows account's session (confirmed via `tailscale status`
refusing to connect from the `IsaH` account, even elevated, with error
`Tailscale already in use by DESKTOP-UVCLFCF\jellyfish`). The agreed fix is
for someone on the `jellyfish` account to run, once:

```
tailscale set --operator=IsaH
```

After that, from either account:

```
tailscale funnel --bg 8090
```

`tailscale funnel status` shows the public URL once it's live. Funnel
persists across reboots on its own (it's a tailscaled feature, not tied to
the terminal session that enabled it) -- it does not need its own scheduled
task, only the JellyscopePublicSite one above needs to survive reboots.

## What's public, and why only that

`server.py`'s `PUBLIC_CROP_FIELDS` is the single source of truth for exactly
which fields get sent to a visitor's browser: species label, confidence,
size, and timestamp -- read out of each crop's metadata JSON one field at a
time, never the whole file. Device health, environmental sensor readings,
upload/queue diagnostics, and full training frames all exist in
`jellyscope_incoming/` but are never read by this server at all. If a future
change adds a new public field, add it to `PUBLIC_CROP_FIELDS` (or a new
endpoint, for a new folder) explicitly -- there's no path here that
accidentally forwards a whole file's contents.
