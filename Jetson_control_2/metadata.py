"""Stage 4: fully independent device/pipeline health reporting. Collects a
sample every ~10s, aggregates and uploads once a minute via the same
transfer.py helper send.py uses. Only ever *reads* the other stages' heartbeat
files and queue directories -- never depends on the camera, GPU models, or any
other stage being alive, so it can't itself hang on a camera/GPU fault.
"""

import glob
import json
import time
from pathlib import Path

import psutil

import config
import queue_io
import transfer

METADATA_OUTBOX = config.PIPELINE_DIR / "metadata_outbox"
THERMAL_ZONE_GLOB = "/sys/class/thermal/thermal_zone*"


def read_thermal_zones() -> dict:
    """Reads whatever thermal zones this JetPack build exposes under /sys
    (temps are millidegrees C). Zones that are empty/unreadable are skipped
    rather than crashing the whole sample."""
    zones = {}
    for zone_dir in glob.glob(THERMAL_ZONE_GLOB):
        try:
            zone_type = (Path(zone_dir) / "type").read_text().strip()
            temp_raw = (Path(zone_dir) / "temp").read_text().strip()
            zones[zone_type] = int(temp_raw) / 1000.0
        except (OSError, ValueError):
            continue
    return zones


def heartbeat_status(stage_name: str) -> dict:
    path = config.HEARTBEAT_DIR / f"{stage_name}.heartbeat.json"
    if not path.exists():
        return {"alive": False, "last_update_age_s": None}
    try:
        heartbeat = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return {"alive": False, "last_update_age_s": None}
    age = time.time() - heartbeat.get("last_update", 0)
    heartbeat["alive"] = age < config.HEARTBEAT_STALE_S
    heartbeat["last_update_age_s"] = age
    return heartbeat


def collect_sample() -> dict:
    thermal = read_thermal_zones()
    mem = psutil.virtual_memory()
    return {
        "timestamp_unix": time.time(),
        "cpu_percent": psutil.cpu_percent(interval=None),
        "thermal_zones_c": thermal,
        "max_temp_c": max(thermal.values()) if thermal else None,
        "mem_used_mb": mem.used / 1e6,
        "mem_total_mb": mem.total / 1e6,
        "uptime_s": time.time() - psutil.boot_time(),
        "disk_free_gb": queue_io.disk_free_bytes(config.QUEUE_ROOT) / 1e9,
        "que_fullframes_depth": queue_io.queue_depth(config.QUE_FULLFRAMES),
        "que_crops_depth": queue_io.queue_depth(config.QUE_CROPS),
        "que_fullframes_failed": queue_io.failed_count(config.QUE_FULLFRAMES),
        "que_crops_failed": queue_io.failed_count(config.QUE_CROPS),
        "record": heartbeat_status("record"),
        "analyse": heartbeat_status("analyse"),
        "send": heartbeat_status("send"),
    }


def aggregate(samples: list[dict]) -> dict:
    temps = [s["max_temp_c"] for s in samples if s["max_temp_c"] is not None]
    return {
        "window_start_unix": samples[0]["timestamp_unix"],
        "window_end_unix": samples[-1]["timestamp_unix"],
        "sample_count": len(samples),
        "cpu_percent_mean": sum(s["cpu_percent"] for s in samples) / len(samples),
        "max_temp_c_mean": sum(temps) / len(temps) if temps else None,
        "max_temp_c_peak": max(temps) if temps else None,
        "disk_free_gb_min": min(s["disk_free_gb"] for s in samples),
        "mem_used_mb_mean": sum(s["mem_used_mb"] for s in samples) / len(samples),
        # latest snapshot for point-in-time fields, not worth averaging
        "que_fullframes_depth": samples[-1]["que_fullframes_depth"],
        "que_crops_depth": samples[-1]["que_crops_depth"],
        "que_fullframes_failed": samples[-1]["que_fullframes_failed"],
        "que_crops_failed": samples[-1]["que_crops_failed"],
        "record": samples[-1]["record"],
        "analyse": samples[-1]["analyse"],
        "send": samples[-1]["send"],
    }


queue_io.ensure_queue_dirs(config.QUE_FULLFRAMES)  # harmless if record.py hasn't started yet
queue_io.ensure_queue_dirs(config.QUE_CROPS)
METADATA_OUTBOX.mkdir(parents=True, exist_ok=True)

print("metadata.py ready, sampling every "
      f"{config.METADATA_SAMPLE_INTERVAL_S}s, uploading every {config.METADATA_UPLOAD_INTERVAL_S}s...")

# Any batch left over from a previous crash: try to send it before starting a new window.
for leftover in sorted(METADATA_OUTBOX.glob("*.json")):
    if transfer.upload_with_retry([str(leftover)], "metadata"):
        leftover.unlink()

samples = []
window_start = time.time()

while True:
    samples.append(collect_sample())
    if time.time() - window_start >= config.METADATA_UPLOAD_INTERVAL_S:
        batch = aggregate(samples)
        batch_path = METADATA_OUTBOX / f"metadata_{time.strftime('%Y%m%d_%H%M%S')}.json"
        batch_path.write_text(json.dumps(batch, indent=2))

        if transfer.upload_with_retry([str(batch_path)], "metadata"):
            batch_path.unlink()
        # else: leave it in metadata_outbox/, picked up on next loop or restart

        samples = []
        window_start = time.time()

    time.sleep(config.METADATA_SAMPLE_INTERVAL_S)
