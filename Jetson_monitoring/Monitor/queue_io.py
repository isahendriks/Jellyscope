"""Disk-backed producer/consumer queue primitives shared by every pipeline stage.

Each queue is a directory with three subdirectories:
  incoming/    -- producer writes finished items here (atomic write + rename)
  processing/  -- consumer claims items into here before working on them
  failed/      -- items that errored out past the retry ceiling, kept for debugging

There is exactly one producer and one consumer per queue (record.py -> analyse.py
for que_fullframes, analyse.py -> send.py for que_crops), so `processing/` exists
purely for crash recovery -- letting a restarted consumer tell "was mid-work when
it died" apart from "still queued" -- not for locking between peers.
"""

import json
import os
import shutil
import time
import uuid
from pathlib import Path


def ensure_queue_dirs(queue_root: Path) -> None:
    for sub in ("incoming", "processing", "failed"):
        (queue_root / sub).mkdir(parents=True, exist_ok=True)


def _atomic_write(final_path: Path, data: bytes) -> None:
    tmp_path = final_path.parent / f".tmp_{uuid.uuid4().hex}_{final_path.name}"
    tmp_path.write_bytes(data)
    os.replace(tmp_path, final_path)


def write_item(queue_root: Path, stem: str, image_bytes: bytes, image_ext: str, sidecar: dict,
               atomic_image: bool = True, atomic_json: bool = True) -> None:
    """Producer: write one queue item. Image is written before the JSON sidecar, so
    consumers can treat "sidecar exists" as "item is complete".

    The image is safe to write non-atomically (atomic_image=False): a crash mid-write
    just means the sidecar (written second) never gets created, so list_ready_stems()
    -- which globs *.json -- never surfaces the stem at all. The partial image is an
    orphaned file no consumer ever looks at, not a corruption risk.

    The JSON sidecar's presence is the completeness marker every consumer relies on, so
    atomic_json defaults to True -- a torn write there could leave a corrupt file a
    consumer's json.loads() chokes on (read_sidecar() has no error handling of its own).
    atomic_json=False trades that small, narrow-window risk for cutting this write's I/O
    further (que_crops uses both flags together -- see the 2026-08-10 investigation into
    exFAT's per-operation contention cost under concurrent record.py/analyse.py/
    send.py/metadata.py access); callers opting into it should make sure whatever reads
    the sidecar back handles a corrupt/truncated file gracefully (see send.py's
    read_sidecar call sites, hardened alongside this)."""
    incoming = queue_root / "incoming"
    image_path = incoming / f"{stem}{image_ext}"
    if atomic_image:
        _atomic_write(image_path, image_bytes)
    else:
        image_path.write_bytes(image_bytes)
    json_path = incoming / f"{stem}.json"
    json_bytes = json.dumps(sidecar, indent=2).encode("utf-8")
    if atomic_json:
        _atomic_write(json_path, json_bytes)
    else:
        json_path.write_bytes(json_bytes)


def list_ready_stems(queue_root: Path) -> list[str]:
    """Consumer: stems ready to claim, oldest first (filenames embed a timestamp,
    so lexical sort is chronological)."""
    incoming = queue_root / "incoming"
    return sorted(p.stem for p in incoming.glob("*.json") if not p.name.startswith(".tmp_"))


def move_between_queues(src_queue_root: Path, dst_queue_root: Path, stem: str, image_ext: str) -> bool:
    """Producer-side move of one complete item (image + json) from src's incoming/
    straight to dst's incoming/ -- for moving live-queue overflow into the backlog
    queue (analyse.py's overflow_live_queue_if_needed()), not a consumer claiming work,
    so unlike claim() this never touches processing/.

    Uses shutil.move rather than os.rename because src and dst can be on different
    physical disks (the live queue on sda1, backlog on sdb1) -- os.rename requires the
    same filesystem; shutil.move falls back to copy+delete when it isn't. Moves the
    image first, matching write_item()'s ordering, so a crash mid-move leaves at worst
    an orphaned image with no json at the destination (invisible to every consumer,
    same reasoning as write_item()'s atomic_image=False docstring) rather than a
    half-registered item any consumer could trip over."""
    src_incoming = src_queue_root / "incoming"
    dst_incoming = dst_queue_root / "incoming"
    src_img, src_json = src_incoming / f"{stem}{image_ext}", src_incoming / f"{stem}.json"
    if not src_img.exists() or not src_json.exists():
        return False
    try:
        shutil.move(str(src_img), str(dst_incoming / f"{stem}{image_ext}"))
        shutil.move(str(src_json), str(dst_incoming / f"{stem}.json"))
    except (FileNotFoundError, OSError):
        return False
    return True


def claim(queue_root: Path, stem: str, image_ext: str) -> tuple[Path, Path] | None:
    """Consumer: move one item from incoming/ to processing/. Returns the
    (image_path, json_path) now in processing/, or None if it's already gone."""
    incoming = queue_root / "incoming"
    processing = queue_root / "processing"
    src_img, src_json = incoming / f"{stem}{image_ext}", incoming / f"{stem}.json"
    dst_img, dst_json = processing / f"{stem}{image_ext}", processing / f"{stem}.json"
    try:
        os.rename(src_json, dst_json)
        os.rename(src_img, dst_img)
    except FileNotFoundError:
        return None
    return dst_img, dst_json

def ack_delete(queue_root: Path, stem: str, image_ext: str) -> None:
    """Consumer: processing succeeded, drop the item entirely (e.g. a
    que_fullframes item once analysed, or a que_crops item once its upload is
    confirmed)."""
    processing = queue_root / "processing"
    (processing / f"{stem}{image_ext}").unlink(missing_ok=True)
    (processing / f"{stem}.json").unlink(missing_ok=True)


def requeue(queue_root: Path, stem: str, image_ext: str, bump_field: str | None = None) -> None:
    """Consumer: processing failed but hasn't hit the retry ceiling -- move the
    item back to incoming/ so it's retried next cycle. Optionally bump a counter
    field in the sidecar JSON first (e.g. "upload_attempts")."""
    processing = queue_root / "processing"
    incoming = queue_root / "incoming"
    json_path = processing / f"{stem}.json"
    if bump_field and json_path.exists():
        sidecar = json.loads(json_path.read_text())
        sidecar[bump_field] = sidecar.get(bump_field, 0) + 1
        json_path.write_text(json.dumps(sidecar, indent=2))
    os.rename(processing / f"{stem}{image_ext}", incoming / f"{stem}{image_ext}")
    os.rename(json_path, incoming / f"{stem}.json")


def fail_item(queue_root: Path, stem: str, image_ext: str, error_text: str) -> None:
    """Consumer: give up on this item -- move it to failed/ with an error note."""
    processing = queue_root / "processing"
    failed = queue_root / "failed"
    for suffix in (image_ext, ".json"):
        src = processing / f"{stem}{suffix}"
        if src.exists():
            os.rename(src, failed / f"{stem}{suffix}")
    (failed / f"{stem}.error.txt").write_text(error_text)


def read_sidecar(json_path: Path) -> dict:
    return json.loads(json_path.read_text())


def read_sidecar_or_none(json_path: Path) -> dict | None:
    """Same as read_sidecar(), but returns None instead of raising on a corrupt/
    unparseable file -- for callers reading a sidecar that may have been written with
    write_item(..., atomic_json=False) (que_crops), where a crash at exactly the wrong
    moment could in principle leave a torn JSON file behind. Callers should treat None
    as "this item's metadata is unrecoverable" (e.g. fail_item it), not retry blindly --
    a torn file won't un-tear itself on the next attempt."""
    try:
        return read_sidecar(json_path)
    except (OSError, json.JSONDecodeError):
        return None


def recover_processing(queue_root: Path, image_ext: str, grace_seconds: float = 5.0) -> list[str]:
    """Consumer: on startup, sweep processing/ for items left over from a prior
    crash. Complete pairs are returned so the caller can re-run its normal
    per-item processing on them; incomplete pairs older than grace_seconds are
    moved to failed/ (a crash between the two renames in claim() is the one
    scenario a single-consumer design can't make atomic)."""
    processing = queue_root / "processing"
    failed = queue_root / "failed"
    now = time.time()
    recovered = []
    for json_path in sorted(processing.glob("*.json")):
        stem = json_path.stem
        img_path = processing / f"{stem}{image_ext}"
        if img_path.exists():
            recovered.append(stem)
        elif now - json_path.stat().st_mtime > grace_seconds:
            fail_item(queue_root, stem, image_ext, "orphaned .json in processing/ at startup, no matching image")
    for img_path in sorted(processing.glob(f"*{image_ext}")):
        stem = img_path.stem
        json_path = processing / f"{stem}.json"
        if not json_path.exists() and now - img_path.stat().st_mtime > grace_seconds:
            os.rename(img_path, failed / img_path.name)
    return recovered


def disk_free_bytes(path: Path) -> int:
    return shutil.disk_usage(path).free


def queue_depth(queue_root: Path) -> int:
    return len(list((queue_root / "incoming").glob("*.json")))


def failed_count(queue_root: Path) -> int:
    return len(list((queue_root / "failed").glob("*.json")))
