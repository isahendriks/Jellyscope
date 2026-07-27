"""Stage 4: fully independent device/pipeline health reporting, plus a second,
separately-uploaded environmental reporting stream. Only ever *reads* the
other stages' heartbeat files and queue directories -- never depends on the
camera, GPU models, or any other stage being alive, so it can't itself hang
on a camera/GPU fault.

Two distinct reports, on independent sampling/upload cadences:
  device metadata      -- disk/cpu/thermal/queue health (as before), plus
                           camera temp (via record.py's heartbeat -- metadata.py
                           can't read the camera directly, that connection is
                           exclusive to record.py), strobe controller converter/
                           driver temp (Modbus/TCP, see strobe.py), and the
                           Metro M0's BME280 enclosure pressure/humidity/
                           temperature, leak, and DS18B20 enclosure temp (see
                           metro_m0.py) -- all "is the hardware healthy"
                           signals, not the environment.
  environmental metadata -- Bar3XT water pressure, depth + temperature (also
                           via the Metro M0) -- what the device is actually
                           sitting in.

The M0 reports all eight sensor values on one line, so it's read once per
ENVIRONMENTAL_SAMPLE_INTERVAL_S (independent of, and typically faster than,
METADATA_SAMPLE_INTERVAL_S) and the latest reading is reused by whichever
report is due next.
"""

import glob
import json
import time
from pathlib import Path

import psutil

import config
import queue_io
import transfer
import metro_m0
import strobe

METADATA_OUTBOX = config.PIPELINE_DIR / "metadata_outbox"
ENVIRONMENTAL_OUTBOX = config.PIPELINE_DIR / "environmental_outbox"
THERMAL_ZONE_GLOB = "/sys/class/thermal/thermal_zone*"

# Live snapshot (not aggregated, not uploaded) -- overwritten every tick so
# analyse.py's live-stream page can show current environmental/device metadata
# without ever touching the M0 serial port or Modbus client itself (metadata.py
# is the sole owner of both).
LIVE_SNAPSHOT_PATH = config.HEARTBEAT_DIR / "metadata_live.json"


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


def read_gpu_percent() -> float | None:
    """Standard Jetson/Tegra sysfs GPU load node -- reports 0-1000 (permille)."""
    try:
        return int(config.GPU_LOAD_PATH.read_text().strip()) / 10.0
    except (OSError, ValueError):
        return None


def read_disk_free_gb() -> dict:
    """Root partition + both external SSDs, by mount point -- the "device" and
    "both hard drives" storage numbers for the live-stream display."""
    free = {}
    for path in [config.ROOT_DISK_PATH, *config.EXTERNAL_SSD_PATHS]:
        try:
            free[str(path)] = queue_io.disk_free_bytes(path) / 1e9
        except OSError:
            free[str(path)] = None
    return free


def heartbeat_status(stage_name: str) -> dict:
    path = config.HEARTBEAT_DIR / f"{stage_name}.heartbeat.json"
    if not path.exists():
        return {"alive": False, "last_update_age_s": None}
    try:
        heartbeat = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return {"alive": False, "last_update_age_s": None}
    # last_update is the human-readable HH:MM:SS string; last_update_unix (a plain
    # epoch float) is what staleness actually gets computed from.
    age = time.time() - heartbeat.get("last_update_unix", 0)
    heartbeat["alive"] = age < config.HEARTBEAT_STALE_S
    heartbeat["last_update_age_s"] = age
    return heartbeat


def collect_sample(m0: dict | None, gpu_percent_mean: float | None, gpu_percent_max: float | None) -> dict:
    """gpu_percent_mean/gpu_percent_max come from the caller's fast rolling GPU
    sample (see main loop below) -- GPU load is bursty (peaks during analyse.py's
    segment/classify, idle between frames), so a single instantaneous read here
    would just be a coin-flip between ~0% and a peak depending on exactly when
    this tick landed relative to the last frame's GPU burst."""
    thermal = read_thermal_zones()
    mem = psutil.virtual_memory()
    record_status = heartbeat_status("record")
    strobe_converter_c, strobe_driver_c = strobe.read_temperatures()
    return {
        "timestamp_unix": time.time(),
        "cpu_percent": psutil.cpu_percent(interval=None),
        "gpu_percent_mean": gpu_percent_mean,
        "gpu_percent_max": gpu_percent_max,
        "thermal_zones_c": thermal,
        "max_temp_c": max(thermal.values()) if thermal else None,
        "jetson_temp_c_mean": _mean(list(thermal.values())),
        "mem_used_mb": mem.used / 1e6,
        "mem_total_mb": mem.total / 1e6,
        "uptime_s": time.time() - psutil.boot_time(),
        "disk_free_gb": queue_io.disk_free_bytes(config.QUEUE_ROOT) / 1e9,
        "disk_free_gb_by_path": read_disk_free_gb(),
        "que_fullframes_depth": queue_io.queue_depth(config.QUE_FULLFRAMES),
        "que_crops_depth": queue_io.queue_depth(config.QUE_CROPS),
        "que_fullframes_failed": queue_io.failed_count(config.QUE_FULLFRAMES),
        "que_crops_failed": queue_io.failed_count(config.QUE_CROPS),
        "record": record_status,
        "analyse": heartbeat_status("analyse"),
        "send": heartbeat_status("send"),
        # camera_temp_c comes from record.py's own heartbeat, not read here directly --
        # the camera connection is exclusive to whichever process holds it open.
        "camera_temp_c": record_status.get("camera_temp_c"),
        "strobe_converter_temp_c": strobe_converter_c,
        "strobe_driver_temp_c": strobe_driver_c,
        "strobe_temp_c_mean": _mean([strobe_converter_c, strobe_driver_c]),
        # Converted from the BME280's native Pa to mbar (1 mbar = 100 Pa) so device
        # (enclosure air) and environmental (Bar3XT water) pressure share one unit.
        "bme280_pressure_mbar": (m0["p"] / 100.0) if m0 else None,
        "bme280_humidity_pct": m0["h"] if m0 else None,
        "bme280_temp_c": m0["b"] if m0 else None,
        "leak_detected": bool(m0["l"]) if m0 else None,
        "ds18b20_temp_c": m0["t"] if m0 else None,
    }


def collect_environmental_sample(m0: dict | None) -> dict:
    return {
        "timestamp_unix": time.time(),
        "bar3xt_pressure_mbar": m0["P"] if m0 else None,
        "bar3xt_depth_m": m0["d"] if m0 else None,
        "bar3xt_temp_c": m0["T"] if m0 else None,
    }


def _mean(values: list) -> float | None:
    values = [v for v in values if v is not None]
    return sum(values) / len(values) if values else None


def aggregate(samples: list[dict]) -> dict:
    temps = [s["max_temp_c"] for s in samples if s["max_temp_c"] is not None]
    gpu_means = [s["gpu_percent_mean"] for s in samples if s["gpu_percent_mean"] is not None]
    gpu_maxes = [s["gpu_percent_max"] for s in samples if s["gpu_percent_max"] is not None]
    leaks = [s["leak_detected"] for s in samples if s["leak_detected"] is not None]
    return {
        "window_start_unix": samples[0]["timestamp_unix"],
        "window_end_unix": samples[-1]["timestamp_unix"],
        "sample_count": len(samples),
        "cpu_percent_mean": _mean([s["cpu_percent"] for s in samples]),
        "gpu_percent_mean": _mean(gpu_means),
        "gpu_percent_peak": max(gpu_maxes) if gpu_maxes else None,
        "max_temp_c_mean": sum(temps) / len(temps) if temps else None,
        "max_temp_c_peak": max(temps) if temps else None,
        "disk_free_gb_min": min(s["disk_free_gb"] for s in samples),
        "mem_used_mb_mean": _mean([s["mem_used_mb"] for s in samples]),
        "camera_temp_c_mean": _mean([s["camera_temp_c"] for s in samples]),
        "strobe_converter_temp_c_mean": _mean([s["strobe_converter_temp_c"] for s in samples]),
        "strobe_driver_temp_c_mean": _mean([s["strobe_driver_temp_c"] for s in samples]),
        "bme280_pressure_mbar_mean": _mean([s["bme280_pressure_mbar"] for s in samples]),
        "bme280_humidity_pct_mean": _mean([s["bme280_humidity_pct"] for s in samples]),
        "bme280_temp_c_mean": _mean([s["bme280_temp_c"] for s in samples]),
        "ds18b20_temp_c_mean": _mean([s["ds18b20_temp_c"] for s in samples]),
        # A leak is a "did this happen at all" signal, not something to average away.
        "leak_detected_any": any(leaks) if leaks else None,
        # latest snapshot for point-in-time fields, not worth averaging
        "que_fullframes_depth": samples[-1]["que_fullframes_depth"],
        "que_crops_depth": samples[-1]["que_crops_depth"],
        "que_fullframes_failed": samples[-1]["que_fullframes_failed"],
        "que_crops_failed": samples[-1]["que_crops_failed"],
        "record": samples[-1]["record"],
        "analyse": samples[-1]["analyse"],
        "send": samples[-1]["send"],
    }


def aggregate_environmental(samples: list[dict]) -> dict:
    depths = [s["bar3xt_depth_m"] for s in samples if s["bar3xt_depth_m"] is not None]
    return {
        "window_start_unix": samples[0]["timestamp_unix"],
        "window_end_unix": samples[-1]["timestamp_unix"],
        "sample_count": len(samples),
        "bar3xt_pressure_mbar_mean": _mean([s["bar3xt_pressure_mbar"] for s in samples]),
        "bar3xt_temp_c_mean": _mean([s["bar3xt_temp_c"] for s in samples]),
        "bar3xt_depth_m_mean": _mean(depths),
        "bar3xt_depth_m_min": min(depths) if depths else None,
        "bar3xt_depth_m_max": max(depths) if depths else None,
    }


def write_live_snapshot(device_sample: dict | None, env_sample: dict | None) -> None:
    """Overwrites metadata_live.json with whatever's currently known -- NOT an
    aggregate, NOT uploaded, just the latest device + environmental readings
    for analyse.py's live-stream page to display. Only overwrites the half
    that actually changed this tick; the other half keeps its last value."""
    snapshot = {}
    if LIVE_SNAPSHOT_PATH.exists():
        try:
            snapshot = json.loads(LIVE_SNAPSHOT_PATH.read_text())
        except (OSError, json.JSONDecodeError):
            snapshot = {}

    if device_sample is not None:
        snapshot.update({
            "cpu_percent": device_sample["cpu_percent"],
            "gpu_percent_mean": device_sample["gpu_percent_mean"],
            "gpu_percent_max": device_sample["gpu_percent_max"],
            "jetson_temp_c_mean": device_sample["jetson_temp_c_mean"],
            "camera_temp_c": device_sample["camera_temp_c"],
            "strobe_temp_c_mean": device_sample["strobe_temp_c_mean"],
            "strobe_converter_temp_c": device_sample["strobe_converter_temp_c"],
            "strobe_driver_temp_c": device_sample["strobe_driver_temp_c"],
            "bme280_pressure_mbar": device_sample["bme280_pressure_mbar"],
            "bme280_humidity_pct": device_sample["bme280_humidity_pct"],
            "bme280_temp_c": device_sample["bme280_temp_c"],
            "ds18b20_temp_c": device_sample["ds18b20_temp_c"],
            "leak_detected": device_sample["leak_detected"],
            "disk_free_gb_by_path": device_sample["disk_free_gb_by_path"],
        })
    if env_sample is not None:
        snapshot.update({
            "bar3xt_pressure_mbar": env_sample["bar3xt_pressure_mbar"],
            "bar3xt_depth_m": env_sample["bar3xt_depth_m"],
            "bar3xt_temp_c": env_sample["bar3xt_temp_c"],
        })
    snapshot["timestamp_unix"] = time.time()

    tmp = LIVE_SNAPSHOT_PATH.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(snapshot))
    tmp.replace(LIVE_SNAPSHOT_PATH)


queue_io.ensure_queue_dirs(config.QUE_FULLFRAMES)  # harmless if record.py hasn't started yet
queue_io.ensure_queue_dirs(config.QUE_CROPS)
METADATA_OUTBOX.mkdir(parents=True, exist_ok=True)
ENVIRONMENTAL_OUTBOX.mkdir(parents=True, exist_ok=True)

print("metadata.py ready -- device: sampling every "
      f"{config.METADATA_SAMPLE_INTERVAL_S}s, uploading every {config.METADATA_UPLOAD_INTERVAL_S}s; "
      f"environmental: sampling every {config.ENVIRONMENTAL_SAMPLE_INTERVAL_S}s, "
      f"uploading every {config.ENVIRONMENTAL_UPLOAD_INTERVAL_S}s...")

# Any batches left over from a previous crash: try to send them before starting new windows.
for leftover in sorted(METADATA_OUTBOX.glob("*.json")):
    if transfer.upload_with_retry([str(leftover)], "metadata"):
        leftover.unlink()
for leftover in sorted(ENVIRONMENTAL_OUTBOX.glob("*.json")):
    if transfer.upload_with_retry([str(leftover)], "environmental_metadata"):
        leftover.unlink()

FAST_TICK_S = min(config.METADATA_SAMPLE_INTERVAL_S, config.ENVIRONMENTAL_SAMPLE_INTERVAL_S)

samples = []
env_samples = []
device_window_start = time.time()
env_window_start = time.time()
last_device_sample_time = 0.0
last_env_sample_time = 0.0
latest_m0_reading = None
gpu_percent_window = []  # fast per-tick GPU reads, reset every METADATA_SAMPLE_INTERVAL_S

while True:
    now = time.time()

    new_env_sample = None
    new_device_sample = None

    # GPU load is cheap to read (one sysfs file) and bursty enough that sampling it only
    # once per METADATA_SAMPLE_INTERVAL_S would just catch whatever instant happened to
    # land on -- so it's sampled every tick (FAST_TICK_S) instead, independent of the
    # heavier collect_sample() cadence below.
    gpu_reading = read_gpu_percent()
    if gpu_reading is not None:
        gpu_percent_window.append(gpu_reading)

    if now - last_env_sample_time >= config.ENVIRONMENTAL_SAMPLE_INTERVAL_S:
        latest_m0_reading = metro_m0.read_sample()
        new_env_sample = collect_environmental_sample(latest_m0_reading)
        env_samples.append(new_env_sample)
        last_env_sample_time = now

    if now - last_device_sample_time >= config.METADATA_SAMPLE_INTERVAL_S:
        gpu_percent_mean = _mean(gpu_percent_window)
        gpu_percent_max = max(gpu_percent_window) if gpu_percent_window else None
        new_device_sample = collect_sample(latest_m0_reading, gpu_percent_mean, gpu_percent_max)
        samples.append(new_device_sample)
        gpu_percent_window = []
        last_device_sample_time = now

    if new_env_sample is not None or new_device_sample is not None:
        write_live_snapshot(new_device_sample, new_env_sample)

    if samples and time.time() - device_window_start >= config.METADATA_UPLOAD_INTERVAL_S:
        batch = aggregate(samples)
        batch_path = METADATA_OUTBOX / f"metadata_{time.strftime('%Y%m%d_%H%M%S')}.json"
        batch_path.write_text(json.dumps(batch, indent=2))

        if transfer.upload_with_retry([str(batch_path)], "metadata"):
            batch_path.unlink()
        # else: leave it in metadata_outbox/, picked up on next loop or restart

        samples = []
        device_window_start = time.time()

    if env_samples and time.time() - env_window_start >= config.ENVIRONMENTAL_UPLOAD_INTERVAL_S:
        env_batch = aggregate_environmental(env_samples)
        env_batch_path = ENVIRONMENTAL_OUTBOX / f"environmental_{time.strftime('%Y%m%d_%H%M%S')}.json"
        env_batch_path.write_text(json.dumps(env_batch, indent=2))

        if transfer.upload_with_retry([str(env_batch_path)], "environmental_metadata"):
            env_batch_path.unlink()
        # else: leave it in environmental_outbox/, picked up on next loop or restart

        env_samples = []
        env_window_start = time.time()

    time.sleep(FAST_TICK_S)
