"""Provenance logbook -- appends human-readable, timestamped entries to
~/Documents/logbook.txt whenever a setting that affects the DATA this device
delivers changes: which AE/scorer/ViT checkpoint is active, background-subtraction
on/off or its alpha, frame sampling rate, SEGMENT/CLASSIFY tuning, etc. (the full
list lives in config.py's LOGBOOK_TRACKED_FIELDS). Aimed at people using data FROM
this device later, who need to know what settings were active when a given batch
was recorded -- not at developers (that's what git history/code comments are for).

config.py calls record_config_snapshot() once, at the bottom of the file, so EVERY
process that imports config.py (record.py, analyse.py, send.py, metadata.py, the
engines/ build tooling, update_segmentation_model.sh's sub-scripts, ad-hoc Test/
scripts -- anything) checks in automatically. Whichever one happens to import
config.py first after a real change (a hand-edit, or a script's sed) is the one
that logs it; every other process sees the state file already caught up and does
nothing -- no per-call-site hook needed anywhere else in the pipeline. A file lock
serializes the read-compare-write so supervisor.sh's four near-simultaneous
startups can't race each other into duplicate or lost entries.
"""

import fcntl
import json
import time
from pathlib import Path

LOGBOOK_PATH = Path.home() / "Documents" / "logbook.txt"
_STATE_DIR = Path(__file__).resolve().parent / "logs"
STATE_PATH = _STATE_DIR / "logbook_state.json"
LOCK_PATH = _STATE_DIR / "logbook.lock"


def _write_entry(text: str) -> None:
    """The actual write -- assumes the caller already holds LOCK_PATH's flock (either
    append_entry() below, or record_config_snapshot(), which is already inside its own
    locked section when it needs to write). Never call this directly from outside this
    module without holding that lock first."""
    LOGBOOK_PATH.parent.mkdir(parents=True, exist_ok=True)
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    with open(LOGBOOK_PATH, "a") as f:
        f.write(f"[{timestamp}] {text}\n\n")


def append_entry(text: str) -> None:
    """Appends one timestamped entry (blank line after, so entries stay visually
    separated in a plain text viewer). Locked for the same race-avoidance reason
    record_config_snapshot() is -- see module docstring. NOTE: flock is scoped to a
    single open() call, not reentrant within a process -- record_config_snapshot()
    below must NOT call this (it would deadlock against its own already-held lock);
    it calls _write_entry() directly instead, from inside its own locked section."""
    LOCK_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(LOCK_PATH, "w") as lock_file:
        fcntl.flock(lock_file, fcntl.LOCK_EX)  # released when lock_file closes below
        _write_entry(text)


def checkpoint_fingerprint(path: Path) -> dict:
    """(path, mtime, size), not just the path string -- catches a checkpoint file
    being overwritten in place under the same filename, not only a path/filename
    change."""
    if not path.exists():
        return {"path": str(path), "exists": False}
    st = path.stat()
    return {"path": str(path), "exists": True, "mtime": st.st_mtime, "size": st.st_size}


def _json_safe(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    return value


def record_config_snapshot(tracked: dict) -> None:
    """`tracked` is a flat {label: value} dict of every logbook-relevant setting (see
    config.py's LOGBOOK_TRACKED_FIELDS) -- values must be JSON-safe already (str, int,
    float, bool, None, dict, or a list/tuple of those) except Path, converted here.
    Compares against the last-recorded snapshot (STATE_PATH); appends a diff entry (or,
    the very first time this ever runs, a full baseline entry) only when something
    actually changed, then updates STATE_PATH to match -- so a process that finds
    nothing changed writes nothing, and restarting the same unchanged pipeline doesn't
    spam the logbook."""
    current = {k: _json_safe(v) for k, v in tracked.items()}

    STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    LOCK_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(LOCK_PATH, "w") as lock_file:
        fcntl.flock(lock_file, fcntl.LOCK_EX)  # holds the whole read-compare-write below

        previous = None
        if STATE_PATH.exists():
            try:
                previous = json.loads(STATE_PATH.read_text())
            except (OSError, json.JSONDecodeError):
                previous = None

        if previous is None:
            lines = [f"  {k}: {v}" for k, v in sorted(current.items())]
            _write_entry("Baseline configuration snapshot (first check-in since "
                         "logbook_state.json didn't exist or was unreadable):\n" + "\n".join(lines))
        else:
            changed = {k: (previous.get(k), v) for k, v in current.items() if previous.get(k) != v}
            if changed:
                lines = [f"  {k}: {old!r} -> {new!r}" for k, (old, new) in sorted(changed.items())]
                _write_entry("Configuration change detected:\n" + "\n".join(lines))

        STATE_PATH.write_text(json.dumps(current, indent=2, sort_keys=True))
