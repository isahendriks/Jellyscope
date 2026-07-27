"""Stage 3: uploads crops + classification labels from que_crops to
server-lab via rsync-over-SSH. Keeps local copies and retries with
backoff if the server is unreachable -- only deletes local data after a
confirmed successful upload (see transfer.py / queue_io.py for the retry and
crash-safety contract).
"""

import json
import time

import config
import queue_io
import transfer

HEARTBEAT_PATH = config.HEARTBEAT_DIR / "send.heartbeat.json"
BATCH_SIZE = 20
POLL_INTERVAL_S = 2

crops_sent_total = 0


def write_heartbeat() -> None:
    now = time.time()
    heartbeat = {
        "last_update": time.strftime("%H:%M:%S", time.localtime(now)),  # human-readable
        "last_update_unix": now,  # what metadata.py's staleness check actually uses
        "crops_sent_total": crops_sent_total,
        "que_crops_failed": queue_io.failed_count(config.QUE_CROPS),
        **transfer.get_transfer_stats(),
    }
    tmp = HEARTBEAT_PATH.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(heartbeat))
    tmp.replace(HEARTBEAT_PATH)


def process_batch(stems: list[str]) -> None:
    global crops_sent_total
    claimed = []
    for stem in stems:
        result = queue_io.claim(config.QUE_CROPS, stem, ".png")
        if result is not None:
            claimed.append((stem, *result))
    if not claimed:
        return

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
            sidecar = queue_io.read_sidecar(json_path)
            attempts = sidecar.get("upload_attempts", 0) + 1
            if attempts >= config.MAX_UPLOAD_ATTEMPTS_TOTAL:
                queue_io.fail_item(config.QUE_CROPS, stem, ".png",
                                    f"exceeded {config.MAX_UPLOAD_ATTEMPTS_TOTAL} upload attempts")
            else:
                queue_io.requeue(config.QUE_CROPS, stem, ".png", bump_field="upload_attempts")


queue_io.ensure_queue_dirs(config.QUE_CROPS)
config.HEARTBEAT_DIR.mkdir(parents=True, exist_ok=True)

if not transfer.preflight_check():
    print("Preflight SSH check failed -- send.py will keep retrying, but nothing "
          "will upload until SSH key auth to the remote host is set up.")

for stem in queue_io.recover_processing(config.QUE_CROPS, ".png"):
    print(f"Recovering {stem} from a previous crash...")
    queue_io.requeue(config.QUE_CROPS, stem, ".png")

print("send.py ready, watching que_crops...")

while True:
    stems = queue_io.list_ready_stems(config.QUE_CROPS)
    if not stems:
        # Still heartbeats while idle -- otherwise "nothing queued" and "send.py has
        # actually died" both look identical after HEARTBEAT_STALE_S (staleness is the
        # only signal the live-stream's Connection chapter and metadata.py's alive-check
        # have to go on).
        write_heartbeat()
        time.sleep(POLL_INTERVAL_S)
        continue

    process_batch(stems[:BATCH_SIZE])
    write_heartbeat()
