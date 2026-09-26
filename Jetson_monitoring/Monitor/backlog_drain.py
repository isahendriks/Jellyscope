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

import json
import shutil
import sys
import time
from collections import defaultdict
from pathlib import Path

import config
import queue_io
import transfer

BATCH_SIZE = 50

# Caps this script's own share of the field link (scp -l, Kbit/s -- see transfer.py's
# scp_upload() docstring) so a drain run can't starve send.py's real-time uploads or
# the dashboard while it works through a big backlog. 800 Kbit/s (~100 KB/s) leaves
# most of a typical ~600KB/s link (see analyse.py's probed link_speed_bps) free for
# live traffic, at the cost of a slower drain -- tune this down further if the live
# pipeline/dashboard still feel starved with a drain running, or up if the link turns
# out to have more headroom than usual.
BANDWIDTH_LIMIT_KBPS = 800

# Progress snapshot for the live-stream dashboard (analyse.py's read_backlog_drain_status()
# reads this same path -- see its docstring). Lives in HEARTBEAT_DIR alongside the other
# stages' heartbeats even though this script isn't one of the supervised loops, purely so
# there's one shared, already-monitored place for the dashboard backend to look.
STATUS_PATH = config.HEARTBEAT_DIR / "backlog_drain.status.json"
config.HEARTBEAT_DIR.mkdir(parents=True, exist_ok=True)


def write_status(queue_root: Path, total_initial: int, sent: int, archived: int, failed: int,
                  t_start: float, done: bool) -> None:
    remaining = max(total_initial - sent - archived - failed, 0)
    elapsed_s = time.time() - t_start
    rate_per_hour = (sent + archived) / elapsed_s * 3600 if elapsed_s > 0 else None
    eta_hours = remaining / rate_per_hour if rate_per_hour else None
    status = {
        "queue_root": str(queue_root),
        "total_initial": total_initial,
        "sent": sent,
        "archived_oversized": archived,
        "failed": failed,
        "remaining": remaining,
        "started_unix": t_start,
        "updated_unix": time.time(),
        "rate_per_hour": rate_per_hour,
        "eta_hours": eta_hours,
        "done": done,
    }
    tmp = STATUS_PATH.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(status))
    tmp.replace(STATUS_PATH)


def stem_date(stem: str) -> str:
    """frame_YYYYMMDD_HHMMSS_..._crop_NNN_area_X.XX -- pull the original capture
    date out of the stem so backlog crops land in the dated remote folder they
    would have on time, not whatever today's date happens to be."""
    return stem.split("_")[1]


def process_batch(queue_root: Path, stems: list[str]) -> dict[str, int]:
    claimed = []
    for stem in stems:
        result = queue_io.claim(queue_root, stem, ".png")
        if result is not None:
            claimed.append((stem, *result))
    if not claimed:
        return {"sent": 0, "archived": 0, "failed": 0}

    archived = 0
    by_date: dict[str, list[tuple[str, Path, Path]]] = defaultdict(list)
    for stem, image_path, json_path in claimed:
        if image_path.stat().st_size > config.MAX_CROP_UPLOAD_BYTES:
            print(f"[drain] {stem} over size limit -- archiving to {config.OVERSIZED_CROPS_DIR}")
            # shutil.move, not Path.rename -- queue_root (this backlog, sdb1) and
            # OVERSIZED_CROPS_DIR (sda1 since the 2026-08-11 QUEUE_ROOT migration) can be
            # different physical disks, and os.rename can't cross filesystems. Same fix
            # queue_io.move_between_queues already uses for the same reason.
            shutil.move(str(image_path), str(config.OVERSIZED_CROPS_DIR / image_path.name))
            shutil.move(str(json_path), str(config.OVERSIZED_CROPS_DIR / json_path.name))
            archived += 1
            continue
        by_date[stem_date(stem)].append((stem, image_path, json_path))

    sent = 0
    failed = 0
    for date_str, items in by_date.items():
        image_paths = [str(p) for _, p, _ in items]
        json_paths = [str(p) for _, _, p in items]
        success = (
            transfer.upload_with_retry(image_paths, f"crops/{date_str}",
                                        max_retries=config.MAX_UPLOAD_RETRIES_PER_CYCLE,
                                        bandwidth_limit_kbps=BANDWIDTH_LIMIT_KBPS)
            and transfer.upload_with_retry(json_paths, f"crop_metadata/{date_str}",
                                            max_retries=config.MAX_UPLOAD_RETRIES_PER_CYCLE,
                                            bandwidth_limit_kbps=BANDWIDTH_LIMIT_KBPS)
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
                    failed += 1
                else:
                    queue_io.requeue(queue_root, stem, ".png", bump_field="upload_attempts")
    return {"sent": sent, "archived": archived, "failed": failed}


def main(queue_root: Path) -> None:
    queue_io.ensure_queue_dirs(queue_root)
    for stem in queue_io.recover_processing(queue_root, ".png"):
        queue_io.requeue(queue_root, stem, ".png")

    if not transfer.preflight_check():
        print("[drain] SSH preflight check failed -- aborting.")
        sys.exit(1)

    print(f"[drain] draining {queue_root} ...")
    backlog: list[str] = queue_io.list_ready_stems(queue_root)
    total_initial = len(backlog)
    total_sent = 0
    total_archived = 0
    total_failed = 0
    last_report = 0
    t_start = time.time()
    write_status(queue_root, total_initial, total_sent, total_archived, total_failed, t_start, done=False)
    while True:
        if not backlog:
            backlog = queue_io.list_ready_stems(queue_root)
        if not backlog:
            break
        batch, backlog = backlog[:BATCH_SIZE], backlog[BATCH_SIZE:]
        result = process_batch(queue_root, batch)
        total_sent += result["sent"]
        total_archived += result["archived"]
        total_failed += result["failed"]
        write_status(queue_root, total_initial, total_sent, total_archived, total_failed, t_start, done=False)
        if total_sent - last_report >= 500:
            print(f"[drain] {total_sent} sent so far ({time.time() - t_start:.0f}s elapsed, "
                  f"~{len(backlog)} left in current listing)")
            last_report = total_sent
    write_status(queue_root, total_initial, total_sent, total_archived, total_failed, t_start, done=True)
    print(f"[drain] done -- {total_sent} items sent from {queue_root} "
          f"({time.time() - t_start:.0f}s elapsed)")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python3 backlog_drain.py <queue_root>")
        sys.exit(1)
    main(Path(sys.argv[1]))
