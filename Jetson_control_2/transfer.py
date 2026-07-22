"""Shared rsync-over-SSH upload helper, used by both send.py (crop batches) and
metadata.py (one small JSON per minute) so the retry logic isn't duplicated.

Uses rsync via subprocess rather than paramiko: rsync is already present on the
Jetson, and gives resumable/checksummed transfer for free over a flaky 5G/
Tailscale link. The pipeline's own queue contract (queue_io.py) is what decides
"confirm before delete", not rsync's own --remove-source-files -- so failures
here are always handled the same way regardless of transport.
"""

import random
import subprocess
import time

from config import REMOTE_HOST, REMOTE_BASE, SSH_CONNECT_TIMEOUT_S, UPLOAD_TIMEOUT_S


def rsync_upload(local_paths: list[str], remote_subdir: str, timeout_s: int = UPLOAD_TIMEOUT_S) -> bool:
    """Upload a batch of local files to REMOTE_HOST:REMOTE_BASE/remote_subdir/ over
    SSH. Returns True only if rsync exits 0 (all files confirmed transferred and
    checksummed)."""
    if not local_paths:
        return True
    remote_dest = f"{REMOTE_HOST}:{REMOTE_BASE}/{remote_subdir}/"
    cmd = [
        "rsync", "-avz", "--timeout", str(timeout_s),
        "-e", f"ssh -o BatchMode=yes -o ConnectTimeout={SSH_CONNECT_TIMEOUT_S} -o StrictHostKeyChecking=accept-new",
        *local_paths, remote_dest,
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout_s + 30)
    except subprocess.TimeoutExpired:
        return False
    if result.returncode != 0:
        print(f"[transfer] rsync failed (exit {result.returncode}): {result.stderr.strip()[:500]}")
    return result.returncode == 0


def upload_with_retry(local_paths: list[str], remote_subdir: str,
                       max_retries: int = 6, base_delay: float = 2.0, max_delay: float = 120.0) -> bool:
    """Exponential backoff with jitter. Returns True on eventual success, False if
    all in-cycle retries are exhausted (caller decides whether to leave items
    queued for the next polling cycle, per the queue's crash-safe contract)."""
    for attempt in range(max_retries):
        if rsync_upload(local_paths, remote_subdir):
            return True
        if attempt < max_retries - 1:
            delay = min(max_delay, base_delay * (2 ** attempt)) * (0.8 + 0.4 * random.random())
            time.sleep(delay)
    return False


def preflight_check() -> bool:
    """Confirm SSH key auth + rsync are working before a stage starts relying on
    them. Prints actionable output if not -- see Jetson_control_2 setup notes."""
    try:
        whoami = subprocess.run(
            ["ssh", "-o", "BatchMode=yes", "-o", f"ConnectTimeout={SSH_CONNECT_TIMEOUT_S}", REMOTE_HOST, "whoami"],
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
