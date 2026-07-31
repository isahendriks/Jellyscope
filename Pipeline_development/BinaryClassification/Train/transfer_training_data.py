"""Incrementally copies the entire Binary_classifier training-data folder (every
monitoring effort's train_encoder/ + train_scorer/ -- everything the AE and
Mahalanobis scorer are trained on in train_AE.py / train_DNN.py) from the R:
network drive to the Jetson, so on-device retraining has the same data
available locally.

Usage: python transfer_training_data.py

Source:      R:\\LU24A1037-Jellyscope\\Jellyscope\\Training data new\\Binary_classifier\\
Destination: jellyfish@jellyscope:/mnt/sda1/TrainingData/Binary_classifier/

Incremental: no rsync client is available on this Windows machine (see
Jetson_monitoring/Monitor/transfer.py's docstring for the same tradeoff), so
this compares local vs. remote file size instead of a real rsync delta -- good
enough here since these are write-once capture images/tiles, never edited in
place. Only files that are missing or a different size on the remote get sent,
streamed as a single tar archive over one SSH connection (not one scp per
file), so re-running after adding a few new tiles is fast, and the very first
(full) run isn't paying per-file process/handshake overhead thousands of times
over.

One-way and additive only: never deletes anything on the remote, even if a
local file has since been removed or renamed.
"""

import shlex
import subprocess
import tempfile
from pathlib import Path

SOURCE_ROOT = Path(r"R:\LU24A1037-Jellyscope\Jellyscope\Training data new\Binary_classifier")
JETSON_HOST = "jellyfish@jellyscope"
DEST_ROOT = "/mnt/sda1/TrainingData/Binary_classifier"

# BatchMode=yes so a first-time host-key prompt or a password/passphrase request fails
# fast instead of hanging forever waiting for input nothing will ever supply (matches
# Jetson_monitoring/Monitor/transfer.py's own SSH options).
SSH_OPTS = ["-o", "BatchMode=yes", "-o", "ConnectTimeout=10", "-o", "StrictHostKeyChecking=accept-new"]


def ssh_run(*remote_argv, timeout=None):
    """Runs a command on JETSON_HOST. ssh joins every argv element after the host with a
    plain space and hands that one string to the remote shell -- so any argument containing
    a space (e.g. find's `-printf '%s %P\\n'` format string below) must be pre-quoted here,
    or the remote shell would word-split it wrong. Building the whole command with
    shlex.quote and passing it as a single argv element sidesteps that."""
    remote_cmd = " ".join(shlex.quote(a) for a in remote_argv)
    return subprocess.run(["ssh", *SSH_OPTS, JETSON_HOST, remote_cmd],
                           capture_output=True, text=True, timeout=timeout)


if not SOURCE_ROOT.exists():
    raise FileNotFoundError(f"Source folder not found: {SOURCE_ROOT}")

print(f"Source: {SOURCE_ROOT}")
print(f"Destination: {JETSON_HOST}:{DEST_ROOT}")

print("\nScanning local files...")
local_files = {}  # relative posix path -> size in bytes
for path in SOURCE_ROOT.rglob("*"):
    if path.is_file():
        local_files[path.relative_to(SOURCE_ROOT).as_posix()] = path.stat().st_size
print(f"  {len(local_files)} local files")

print("Listing files already on the Jetson...")
mkdir_result = ssh_run("mkdir", "-p", DEST_ROOT, timeout=30)
if mkdir_result.returncode != 0:
    raise RuntimeError(f"Failed to create remote directory: {mkdir_result.stderr.strip()}")

find_result = ssh_run("find", DEST_ROOT, "-type", "f", "-printf", r"%s %P\n", timeout=120)
if find_result.returncode != 0:
    raise RuntimeError(f"Failed to list remote files: {find_result.stderr.strip()}")

remote_files = {}  # relative path -> size in bytes
for line in find_result.stdout.splitlines():
    if not line.strip():
        continue
    size_str, rel = line.split(" ", 1)
    remote_files[rel] = int(size_str)
print(f"  {len(remote_files)} files already present remotely")

to_transfer = [rel for rel, size in local_files.items() if remote_files.get(rel) != size]

if not to_transfer:
    print("\nNothing to transfer -- Jetson copy is already up to date.")
else:
    print(f"\n{len(to_transfer)} new/changed file(s) to transfer "
          f"({len(local_files) - len(to_transfer)} already up to date, skipped).")

    # Stream exactly those files as a single tar archive over one SSH connection -- far
    # fewer round trips than one scp per file, and preserves relative directory structure
    # without needing per-directory mkdir/scp calls. The list is written to a temp file and
    # read back via tar's -T rather than passed as thousands of individual command-line
    # arguments, which could otherwise hit Windows' ~32K-character command-line limit on a
    # large first full sync.
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as filelist:
        filelist.write("\n".join(to_transfer))
        filelist_path = filelist.name

    try:
        tar_proc = subprocess.Popen(
            ["tar", "-cf", "-", "-C", str(SOURCE_ROOT), "-T", filelist_path],
            stdout=subprocess.PIPE,
        )
        remote_extract_cmd = " ".join(shlex.quote(a) for a in ("tar", "-xf", "-", "-C", DEST_ROOT))
        ssh_proc = subprocess.Popen(
            ["ssh", *SSH_OPTS, JETSON_HOST, remote_extract_cmd],
            stdin=tar_proc.stdout,
        )
        tar_proc.stdout.close()  # let tar_proc receive SIGPIPE if ssh_proc exits early
        ssh_proc.communicate()
        tar_proc.wait()
    finally:
        Path(filelist_path).unlink(missing_ok=True)

    if tar_proc.returncode == 0 and ssh_proc.returncode == 0:
        print(f"\n✓ Transferred {len(to_transfer)} file(s) to {JETSON_HOST}:{DEST_ROOT}")
    else:
        print(f"\nTransfer failed (tar exit {tar_proc.returncode}, ssh exit {ssh_proc.returncode})")
