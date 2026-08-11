"""Stage 3: uploads crops + classification labels from que_crops to
server-lab via rsync-over-SSH. Keeps local copies and retries with
backoff if the server is unreachable -- only deletes local data after a
confirmed successful upload (see transfer.py / queue_io.py for the retry and
crash-safety contract).
"""

import json
import shutil
import time
from pathlib import Path

import config
import queue_io
import transfer

HEARTBEAT_PATH = config.HEARTBEAT_DIR / "send.heartbeat.json"
BATCH_SIZE = 20  # ceiling on crop_batch_size below, not a fixed size anymore -- see
# there for why (2026-08-11: fixed-size batches were routinely timing out on the
# relayed/slow link to server-lab, discarding the whole batch's progress each time
# since scp isn't resumable)
MIN_BATCH_SIZE = 1
# Training frames arrive at most once per config.TRAINING_FRAME_INTERVAL_S (analyse.py) --
# this cap is just a safety margin for catching up after send.py's been down a while, not
# a real per-cycle backlog like que_crops sees.
TRAINING_FRAME_BATCH_SIZE = 5
POLL_INTERVAL_S = 2

crops_sent_total = 0
crops_archived_oversized_total = 0
training_frames_sent_total = 0


def write_heartbeat() -> None:
    now = time.time()
    heartbeat = {
        "last_update": time.strftime("%H:%M:%S", time.localtime(now)),  # human-readable
        "last_update_unix": now,  # what metadata.py's staleness check actually uses
        "crops_sent_total": crops_sent_total,
        "crops_archived_oversized_total": crops_archived_oversized_total,
        "que_crops_failed": queue_io.failed_count(config.QUE_CROPS),
        "training_frames_sent_total": training_frames_sent_total,
        "que_training_frames_failed": queue_io.failed_count(config.QUE_TRAINING_FRAMES),
        "crop_batch_size": crop_batch_size,
        **transfer.get_transfer_stats(),
    }
    tmp = HEARTBEAT_PATH.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(heartbeat))
    tmp.replace(HEARTBEAT_PATH)


def archive_oversized(claimed: list[tuple[str, Path, Path]]) -> list[tuple[str, Path, Path]]:
    """Splits an oversized crop (see config.MAX_CROP_UPLOAD_BYTES's docstring) out of a
    just-claimed batch and moves it straight to OVERSIZED_CROPS_DIR instead of letting it
    anywhere near scp -- one giant file in a batch can make the whole batch too slow to
    transfer inside UPLOAD_TIMEOUT_S, and since scp uploads a batch in one shot, that
    stalls every normal crop queued behind it too (que_crops is strict FIFO). Returns
    whatever's left of `claimed` for the normal upload path below."""
    global crops_archived_oversized_total
    normal = []
    for item in claimed:
        stem, image_path, json_path = item
        size = image_path.stat().st_size
        if size <= config.MAX_CROP_UPLOAD_BYTES:
            normal.append(item)
            continue
        print(f"[send] {stem} is {size / 1e6:.1f}MB (over "
              f"{config.MAX_CROP_UPLOAD_BYTES / 1e6:.0f}MB) -- archiving to "
              f"{config.OVERSIZED_CROPS_DIR} instead of uploading")
        shutil.move(str(image_path), str(config.OVERSIZED_CROPS_DIR / image_path.name))
        shutil.move(str(json_path), str(config.OVERSIZED_CROPS_DIR / json_path.name))
        crops_archived_oversized_total += 1
    return normal


def process_batch(stems: list[str]) -> bool | None:
    """Returns True/False for an actual upload attempt's outcome, so the caller can
    grow/shrink crop_batch_size off it (see main loop below) -- None means nothing was
    actually attempted (everything already claimed elsewhere, or archived as oversized),
    which shouldn't move the batch-size estimate either way."""
    global crops_sent_total
    claimed = []
    for stem in stems:
        result = queue_io.claim(config.QUE_CROPS, stem, ".png")
        if result is not None:
            claimed.append((stem, *result))
    if not claimed:
        return None

    claimed = archive_oversized(claimed)
    if not claimed:
        return None

    image_paths = [str(image_path) for _, image_path, _ in claimed]
    json_paths = [str(json_path) for _, _, json_path in claimed]

    # Separate remote folders -- crops/ (images) and crop_metadata/ (per-crop JSON
    # sidecars) -- not metadata.py's own metadata/ folder, which holds unrelated
    # device-health reports and would get confusing to browse mixed in with these.
    date_str = time.strftime('%Y%m%d')
    success = (
        transfer.upload_with_retry(image_paths, f"crops/{date_str}", max_retries=config.MAX_UPLOAD_RETRIES_PER_CYCLE)
        and transfer.upload_with_retry(json_paths, f"crop_metadata/{date_str}", max_retries=config.MAX_UPLOAD_RETRIES_PER_CYCLE)
    )

    if success:
        for stem, _, _ in claimed:
            queue_io.ack_delete(config.QUE_CROPS, stem, ".png")
            crops_sent_total += 1
    else:
        for stem, _, json_path in claimed:
            sidecar = queue_io.read_sidecar_or_none(json_path)
            if sidecar is None:
                queue_io.fail_item(config.QUE_CROPS, stem, ".png", "unreadable/corrupt sidecar JSON")
                continue
            attempts = sidecar.get("upload_attempts", 0) + 1
            if attempts >= config.MAX_UPLOAD_ATTEMPTS_TOTAL:
                queue_io.fail_item(config.QUE_CROPS, stem, ".png",
                                    f"exceeded {config.MAX_UPLOAD_ATTEMPTS_TOTAL} upload attempts")
            else:
                queue_io.requeue(config.QUE_CROPS, stem, ".png", bump_field="upload_attempts")
    return success


def process_training_frame_batch(stems: list[str]) -> None:
    global training_frames_sent_total
    claimed = []
    for stem in stems:
        result = queue_io.claim(config.QUE_TRAINING_FRAMES, stem, ".png")
        if result is not None:
            claimed.append((stem, *result))
    if not claimed:
        return

    image_paths = [str(image_path) for _, image_path, _ in claimed]
    json_paths = [str(json_path) for _, _, json_path in claimed]

    # Separate remote folders, same crops/crop_metadata split as process_batch() above.
    date_str = time.strftime('%Y%m%d')
    success = (
        transfer.upload_with_retry(image_paths, f"training_frames/{date_str}",
                                    max_retries=config.MAX_UPLOAD_RETRIES_PER_CYCLE,
                                    timeout_s=config.TRAINING_FRAME_UPLOAD_TIMEOUT_S)
        and transfer.upload_with_retry(json_paths, f"training_frames_metadata/{date_str}",
                                        max_retries=config.MAX_UPLOAD_RETRIES_PER_CYCLE)
    )

    if success:
        for stem, _, _ in claimed:
            queue_io.ack_delete(config.QUE_TRAINING_FRAMES, stem, ".png")
            training_frames_sent_total += 1
    else:
        for stem, _, json_path in claimed:
            sidecar = queue_io.read_sidecar(json_path)
            attempts = sidecar.get("upload_attempts", 0) + 1
            if attempts >= config.MAX_UPLOAD_ATTEMPTS_TOTAL:
                queue_io.fail_item(config.QUE_TRAINING_FRAMES, stem, ".png",
                                    f"exceeded {config.MAX_UPLOAD_ATTEMPTS_TOTAL} upload attempts")
            else:
                queue_io.requeue(config.QUE_TRAINING_FRAMES, stem, ".png", bump_field="upload_attempts")


queue_io.ensure_queue_dirs(config.QUE_CROPS)
queue_io.ensure_queue_dirs(config.QUE_TRAINING_FRAMES)
config.HEARTBEAT_DIR.mkdir(parents=True, exist_ok=True)
config.OVERSIZED_CROPS_DIR.mkdir(parents=True, exist_ok=True)

if not transfer.preflight_check():
    print("Preflight SSH check failed -- send.py will keep retrying, but nothing "
          "will upload until SSH key auth to the remote host is set up.")

for stem in queue_io.recover_processing(config.QUE_CROPS, ".png"):
    print(f"Recovering {stem} from a previous crash...")
    queue_io.requeue(config.QUE_CROPS, stem, ".png")

for stem in queue_io.recover_processing(config.QUE_TRAINING_FRAMES, ".png"):
    print(f"Recovering {stem} from a previous crash...")
    queue_io.requeue(config.QUE_TRAINING_FRAMES, stem, ".png")

print("send.py ready, watching que_crops and que_training_frames...")

# Backlogs of stems already listed from disk but not yet claimed -- refilled by
# list_ready_stems() only once empty, rather than every cycle. incoming/ lives on
# exFAT (/mnt/sdb1), where directory listing is ~O(n) in total entries; re-listing
# the whole directory on every BATCH_SIZE-sized bite of a large backlog (e.g. after
# an upload outage lets tens of thousands of items pile up) made each cycle cost
# scale with the *entire* backlog instead of just the batch being processed, so
# nearly all of send.py's time went to re-scanning que_crops/incoming rather than
# uploading -- see the 2026-08-03 slow-pipeline incident.
crop_backlog: list[str] = []
training_backlog: list[str] = []

# Adapts within [MIN_BATCH_SIZE, BATCH_SIZE] -- additive-increase/multiplicative-decrease,
# same shape as TCP congestion control: grow by one crop per successful batch (cautious,
# since a fast link recovering from a slow patch shouldn't immediately get slammed with a
# full-size batch again), halve on any failed/timed-out batch (fast reaction to a link
# that's currently too slow for the size being tried). See transfer.py's estimate_timeout
# for the matching per-call timeout scaling.
crop_batch_size = BATCH_SIZE

while True:
    if not crop_backlog:
        crop_backlog = queue_io.list_ready_stems(config.QUE_CROPS)
    if not training_backlog:
        training_backlog = queue_io.list_ready_stems(config.QUE_TRAINING_FRAMES)

    if not crop_backlog and not training_backlog:
        # Still heartbeats while idle -- otherwise "nothing queued" and "send.py has
        # actually died" both look identical after HEARTBEAT_STALE_S (staleness is the
        # only signal the live-stream's Connection chapter and metadata.py's alive-check
        # have to go on).
        write_heartbeat()
        time.sleep(POLL_INTERVAL_S)
        continue

    if crop_backlog:
        batch, crop_backlog = crop_backlog[:crop_batch_size], crop_backlog[crop_batch_size:]
        result = process_batch(batch)
        if result is True:
            crop_batch_size = min(BATCH_SIZE, crop_batch_size + 1)
        elif result is False:
            crop_batch_size = max(MIN_BATCH_SIZE, crop_batch_size // 2)
    if training_backlog:
        batch, training_backlog = training_backlog[:TRAINING_FRAME_BATCH_SIZE], training_backlog[TRAINING_FRAME_BATCH_SIZE:]
        process_training_frame_batch(batch)
    write_heartbeat()
