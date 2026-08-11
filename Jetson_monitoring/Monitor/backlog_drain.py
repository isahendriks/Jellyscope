"""One-off maintenance script for the 2026-08-03 slow-pipeline incident: drains a
moved-aside que_crops-style backlog (already-processed crop PNG + JSON pairs, just
waiting on upload) to server-lab, independently of the live send.py loop.

Background: a server-lab connectivity outage let que_crops/incoming grow to ~60k
items; since it lives on exFAT (/mnt/sdb1), directory ops there degrade badly with
entry count, which was stalling the live pipeline (see send.py/analyse.py's own
backlog-caching fix from the same incident). The live incoming/ was moved aside and
replaced with a fresh empty one; this script drains the moved-aside copy separately,
so the historical backlog doesn't compete with real-time capture.

Not for que_fullframes backlogs -- those are raw, unanalysed camera captures (.tiff)
that need analyse.py's ML pipeline to become crops, not a plain upload.

Usage: python3 backlog_drain.py <queue_root>
where <queue_root> is laid out like any queue_io queue (incoming/, processing/,
failed/) -- e.g. que_crops_backlog_<stamp>/ created during the incident.

Safe to Ctrl-C and rerun -- same crash-recovery contract as send.py
(queue_io.recover_processing sweeps processing/ back to incoming/ on startup).
"""

import sys
import time
from collections import defaultdict
from pathlib import Path

import config
import queue_io
import transfer

BATCH_SIZE = 50


def stem_date(stem: str) -> str:
    """frame_YYYYMMDD_HHMMSS_..._crop_NNN_area_X.XX -- pull the original capture
    date out of the stem so backlog crops land in the dated remote folder they
    would have on time, not whatever today's date happens to be."""
    return stem.split("_")[1]


def process_batch(queue_root: Path, stems: list[str]) -> int:
    claimed = []
    for stem in stems:
        result = queue_io.claim(queue_root, stem, ".png")
        if result is not None:
            claimed.append((stem, *result))
    if not claimed:
        return 0

    by_date: dict[str, list[tuple[str, Path, Path]]] = defaultdict(list)
    for stem, image_path, json_path in claimed:
        if image_path.stat().st_size > config.MAX_CROP_UPLOAD_BYTES:
            print(f"[drain] {stem} over size limit -- archiving to {config.OVERSIZED_CROPS_DIR}")
            image_path.rename(config.OVERSIZED_CROPS_DIR / image_path.name)
            json_path.rename(config.OVERSIZED_CROPS_DIR / json_path.name)
            continue
        by_date[stem_date(stem)].append((stem, image_path, json_path))

    sent = 0
    for date_str, items in by_date.items():
        image_paths = [str(p) for _, p, _ in items]
        json_paths = [str(p) for _, _, p in items]
        success = (
            transfer.upload_with_retry(image_paths, f"crops/{date_str}",
                                        max_retries=config.MAX_UPLOAD_RETRIES_PER_CYCLE)
            and transfer.upload_with_retry(json_paths, f"crop_metadata/{date_str}",
                                            max_retries=config.MAX_UPLOAD_RETRIES_PER_CYCLE)
        )
        if success:
            for stem, _, _ in items:
                queue_io.ack_delete(queue_root, stem, ".png")
                sent += 1
        else:
            for stem, _, json_path in items:
                sidecar = queue_io.read_sidecar(json_path)
                attempts = sidecar.get("upload_attempts", 0) + 1
                if attempts >= config.MAX_UPLOAD_ATTEMPTS_TOTAL:
                    queue_io.fail_item(queue_root, stem, ".png",
                                        f"exceeded {config.MAX_UPLOAD_ATTEMPTS_TOTAL} upload attempts")
                else:
                    queue_io.requeue(queue_root, stem, ".png", bump_field="upload_attempts")
    return sent


def main(queue_root: Path) -> None:
    queue_io.ensure_queue_dirs(queue_root)
    for stem in queue_io.recover_processing(queue_root, ".png"):
        queue_io.requeue(queue_root, stem, ".png")

    if not transfer.preflight_check():
        print("[drain] SSH preflight check failed -- aborting.")
        sys.exit(1)

    print(f"[drain] draining {queue_root} ...")
    backlog: list[str] = []
    total_sent = 0
    last_report = 0
    t_start = time.time()
    while True:
        if not backlog:
            backlog = queue_io.list_ready_stems(queue_root)
        if not backlog:
            break
        batch, backlog = backlog[:BATCH_SIZE], backlog[BATCH_SIZE:]
        total_sent += process_batch(queue_root, batch)
        if total_sent - last_report >= 500:
            print(f"[drain] {total_sent} sent so far ({time.time() - t_start:.0f}s elapsed, "
                  f"~{len(backlog)} left in current listing)")
            last_report = total_sent
    print(f"[drain] done -- {total_sent} items sent from {queue_root} "
          f"({time.time() - t_start:.0f}s elapsed)")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python3 backlog_drain.py <queue_root>")
        sys.exit(1)
    main(Path(sys.argv[1]))
