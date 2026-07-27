"""Shared scp-over-SSH upload helper, used by both send.py (crop batches) and
metadata.py (one small JSON per minute) so the retry logic isn't duplicated.

Originally rsync-over-SSH (resumable/checksummed transfer for free over a
flaky 5G/Tailscale link), but REMOTE_HOST (server-lab) turned out to be a
Windows box with no rsync binary -- rsync needs to be present on *both* ends
since it invokes a remote `rsync --server` process, and Windows doesn't ship
one. Switched to scp instead, since it's already there (bundled with the
Win32-OpenSSH client/server that's already working) and needs nothing extra
installed. The tradeoff is losing rsync's resumable/incremental-sync behavior,
which doesn't matter much here -- crops and metadata are small, individually-
named files (never partially re-sent), not one large file needing resume.

Unlike rsync, scp never creates missing destination directories -- and
send.py's crop destination changes daily (crops/<YYYYMMDD>/), so
ensure_remote_dir() below runs a best-effort `mkdir` over SSH first. The
remote command goes through Windows' default SSH shell (cmd.exe), so it uses
backslash-separated paths and ignores "already exists" failures -- cmd.exe's
mkdir isn't idempotent like `mkdir -p`, but that's fine, we don't care why it
failed, only that the directory exists by the time scp runs.

The pipeline's own queue contract (queue_io.py) is what decides "confirm
before delete", not any transport-level flag -- so failures here are always
handled the same way regardless of transport.
"""

import random
import subprocess
import time
from pathlib import Path

from config import REMOTE_HOST, REMOTE_BASE, SSH_CONNECT_TIMEOUT_S, UPLOAD_TIMEOUT_S

# Connection health, updated by every scp_upload() call (success or failure) -- read via
# get_transfer_stats() by send.py's heartbeat, and from there by analyse.py's live-stream
# "Connection" chapter. Module-level rather than threaded through every call site, same
# pattern as metro_m0.py/strobe.py's own persistent-connection state.
_last_attempt_unix: float | None = None
_last_attempt_ok: bool | None = None
_last_success_unix: float | None = None
_last_speed_bps: float | None = None  # bytes/sec of the most recent successful batch
_consecutive_failures = 0


def get_transfer_stats() -> dict:
    return {
        "last_attempt_unix": _last_attempt_unix,
        "last_attempt_ok": _last_attempt_ok,
        "last_success_unix": _last_success_unix,
        "last_speed_bps": _last_speed_bps,
        "consecutive_failures": _consecutive_failures,
    }

# ControlMaster/ControlPath/ControlPersist -- every ssh/scp call below used to open a brand
# new SSH connection from scratch (real handshake cost over Tailscale), and a single
# process_batch() cycle makes up to 4 of them (ensure_remote_dir + scp, x2 for images and
# json). With these options, whichever call happens first opens one real connection and
# leaves it open at ControlPath; every later ssh/scp invocation (from this process or any
# other on the same machine using the same REMOTE_HOST) reuses it near-instantly instead of
# re-handshaking. ControlPersist=600 keeps it alive for 10 minutes of idle time, well over
# send.py's 2s poll interval.
_CONTROL_PATH = f"/tmp/jellyscope-ssh-ctrl-{REMOTE_HOST}"
_SSH_OPTS = ["-o", "BatchMode=yes", "-o", f"ConnectTimeout={SSH_CONNECT_TIMEOUT_S}",
             "-o", "StrictHostKeyChecking=accept-new",
             "-o", "ControlMaster=auto", "-o", f"ControlPath={_CONTROL_PATH}", "-o", "ControlPersist=600"]


def ensure_remote_dir(remote_subdir: str, timeout_s: int = SSH_CONNECT_TIMEOUT_S + 5) -> None:
    """Best-effort remote `mkdir` over SSH -- ignores failure, since "already
    exists" is the expected/common case and cmd.exe's mkdir isn't idempotent
    the way `mkdir -p` is. Only matters that the directory exists afterward,
    not why the command itself succeeded or failed."""
    windows_path = f"{REMOTE_BASE}\\{remote_subdir}".replace("/", "\\")
    try:
        subprocess.run(
            ["ssh", *_SSH_OPTS, REMOTE_HOST, "mkdir", f'"{windows_path}"'],
            capture_output=True, text=True, timeout=timeout_s,
        )
    except subprocess.TimeoutExpired:
        pass


def scp_upload(local_paths: list[str], remote_subdir: str, timeout_s: int = UPLOAD_TIMEOUT_S) -> bool:
    """Upload a batch of local files to REMOTE_HOST:REMOTE_BASE/remote_subdir/ over
    SSH via scp. Returns True only if scp exits 0 (all files confirmed transferred)."""
    global _last_attempt_unix, _last_attempt_ok, _last_success_unix, _last_speed_bps, _consecutive_failures
    if not local_paths:
        return True
    ensure_remote_dir(remote_subdir, timeout_s=min(timeout_s, SSH_CONNECT_TIMEOUT_S + 5))
    remote_dest = f"{REMOTE_HOST}:{REMOTE_BASE}/{remote_subdir}/"
    cmd = ["scp", *_SSH_OPTS, *local_paths, remote_dest]
    # Measured before the subprocess call -- a file could in principle be gone by the time
    # scp reads it (queue_io guarantees these are stable, but stat() failing shouldn't ever
    # take the whole upload down over a metrics side-channel), hence the try/except.
    try:
        total_bytes = sum(Path(p).stat().st_size for p in local_paths)
    except OSError:
        total_bytes = None
    t_start = time.perf_counter()
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout_s + 30)
        ok = result.returncode == 0
        if not ok:
            print(f"[transfer] scp failed (exit {result.returncode}): {result.stderr.strip()[:500]}")
    except subprocess.TimeoutExpired:
        ok = False
    elapsed_s = time.perf_counter() - t_start

    _last_attempt_unix = time.time()
    _last_attempt_ok = ok
    if ok:
        _last_success_unix = _last_attempt_unix
        _consecutive_failures = 0
        if total_bytes is not None and elapsed_s > 0:
            _last_speed_bps = total_bytes / elapsed_s
    else:
        _consecutive_failures += 1
    return ok


def upload_with_retry(local_paths: list[str], remote_subdir: str,
                       max_retries: int = 6, base_delay: float = 2.0, max_delay: float = 120.0) -> bool:
    """Exponential backoff with jitter. Returns True on eventual success, False if
    all in-cycle retries are exhausted (caller decides whether to leave items
    queued for the next polling cycle, per the queue's crash-safe contract)."""
    for attempt in range(max_retries):
        if scp_upload(local_paths, remote_subdir):
            return True
        if attempt < max_retries - 1:
            delay = min(max_delay, base_delay * (2 ** attempt)) * (0.8 + 0.4 * random.random())
            time.sleep(delay)
    return False


def preflight_check() -> bool:
    """Confirm SSH key auth is working before a stage starts relying on it.
    Prints actionable output if not -- see monitoring setup notes."""
    try:
        whoami = subprocess.run(
            ["ssh", *_SSH_OPTS, REMOTE_HOST, "whoami"],
            capture_output=True, text=True, timeout=SSH_CONNECT_TIMEOUT_S + 5,
        )
    except subprocess.TimeoutExpired:
        print(f"[transfer] SSH to {REMOTE_HOST} timed out")
        return False
    if whoami.returncode != 0:
        print(f"[transfer] SSH key auth to {REMOTE_HOST} not working: {whoami.stderr.strip()}")
        print("[transfer] Run: ssh-keygen (if needed), then copy the public key into "
              f"{REMOTE_HOST}'s authorized_keys, then retest with "
              f"`ssh -o BatchMode=yes {REMOTE_HOST} whoami`.")
        return False
    return True
