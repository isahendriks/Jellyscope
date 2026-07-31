"""Stage 1: camera acquisition. Captures frames from the Itala camera and writes
each one to the que_fullframes disk queue for analyse.py to pick up. Adapted
from Jetson_control/acquire_images.py -- GPU enhancement, segmentation, and
classification all happen downstream in analyse.py, not here.
"""

from itala import itala
import ctypes
import cv2
import numpy as np
import os
import time
import json

import config
import queue_io

### ==========================
### Recording parameters
### ==========================
# FRAME_SKIP (keep every Nth triggered frame) and GAIN_DB live in config.py, not here --
# analyse.py's live-stream /status reports FRAME_SKIP too, and livestream.py also sets
# GAIN_DB on the camera, so all three need to agree on the same values.
HDR_MAX = 4094  # should not be changed!!!!

DISK_CHECK_EVERY_N_FRAMES = 20
HEARTBEAT_EVERY_N_FRAMES = 10

HEARTBEAT_PATH = config.HEARTBEAT_DIR / "record.heartbeat.json"

### ==========================
### Initialize camera
### ==========================
queue_io.ensure_queue_dirs(config.QUE_FULLFRAMES)
config.HEARTBEAT_DIR.mkdir(parents=True, exist_ok=True)

system = itala.create_system()
devices_info = system.enumerate_devices(500)

if len(devices_info) == 0:
    print("No devices found. Exiting.")
    exit(1)
if devices_info[0].access_status != itala.DeviceAccessStatus_AvailableReadWrite:
    print("Device not accessible in RW mode. Exiting.")
    exit(1)

device = system.create_device(devices_info[0])
camera_serial = devices_info[0].serial_number
print("Device initialized.")
nodemap = device.node_map

nodemap.TriggerSelector.from_string("FrameBurstStart")
nodemap.TriggerSource.from_string("Line0")
nodemap.LineSelector.from_string("Line0")
nodemap.LineMode.from_string("Input")
nodemap.GainAuto.from_string("Off")
nodemap.Gain.set_value(config.GAIN_DB)

target = 4000  # booster frame size (4000)
valid_val = target - ((target - 560) % 8)
nodemap.GevSCPSPacketSize.value = valid_val
nodemap.AcquisitionBurstFrameCount.set_value(1)
nodemap.ExposureMode.from_string("TriggerWidth")
nodemap.TriggerMode.from_string("On")
nodemap.PixelFormat.from_string("Mono12p")

device.start_acquisition()
print("Acquisition started.")

waiting_time = (config.FRAME_SKIP + 2) * 1000
frame_counter = 0
time_counter = 0
consecutive_missed = 0
camera_temp_supported = True  # some firmware/nodemaps don't expose DeviceTemperature


def read_camera_temp_c():
    """Reads the camera's own DeviceTemperature node (same technique
    livestream.py uses). metadata.py can't read this directly -- GigE camera
    access is exclusive to whichever process holds the device open (this one)
    -- so it's exposed here via the heartbeat file instead."""
    global camera_temp_supported
    if not camera_temp_supported:
        return None
    try:
        return nodemap.DeviceTemperature.value
    except Exception as exc:
        camera_temp_supported = False
        print(f"Camera temperature unavailable: {exc}")
        return None


def write_heartbeat(camera_connected: bool) -> None:
    now = time.time()
    heartbeat = {
        "pid": os.getpid(),
        "last_update": time.strftime("%H:%M:%S", time.localtime(now)),  # human-readable
        "last_update_unix": now,  # what metadata.py's staleness check actually uses
        "frames_recorded_total": frame_counter,
        "camera_connected": camera_connected,
        "camera_temp_c": read_camera_temp_c() if camera_connected else None,
    }
    tmp = HEARTBEAT_PATH.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(heartbeat))
    tmp.replace(HEARTBEAT_PATH)


try:
    while True:
        # Disk-space backpressure: pause producing rather than crash/fill the disk.
        if frame_counter % DISK_CHECK_EVERY_N_FRAMES == 0:
            free = queue_io.disk_free_bytes(config.QUEUE_ROOT)
            if free < config.MIN_FREE_BYTES:
                print(f"Low disk space ({free / 1e9:.2f} GB free), pausing acquisition...")
                write_heartbeat(camera_connected=True)
                time.sleep(5)
                continue

        image = device.get_next_image(waiting_time)

        if image is None:
            consecutive_missed += 1
            print("No image returned, trigger issue")
            write_heartbeat(camera_connected=consecutive_missed < 5)
            continue
        consecutive_missed = 0

        time_counter += 1
        if time_counter % config.FRAME_SKIP != 0:
            image.dispose()
            continue

        if image.is_incomplete:
            print(f"Image {image.frame_id} incomplete.")
            image.dispose()
            continue

        image12 = image.convert(itala.PfncFormat_Mono12)
        height, width = image12.height, image12.width
        size = width * height

        p = (ctypes.c_uint16 * size).from_address(int(image12.get_data()))
        img_array = np.ctypeslib.as_array(p).reshape((height, width)).astype(np.uint16)
        image12.dispose()
        del p, image12

        frame_counter += 1
        timestamp = time.time()
        stem = f"frame_{time.strftime('%Y%m%d_%H%M%S')}_{frame_counter:08d}"

        # Uncompressed TIFF: record.py is a real-time acquisition loop, so skipping
        # PNG's zlib compression keeps per-frame CPU cost minimal (see
        # monitoring's queue contract notes for the disk-space tradeoff).
        ok, encoded = cv2.imencode(".tiff", img_array, [cv2.IMWRITE_TIFF_COMPRESSION, 1])
        if not ok:
            print(f"Failed to encode frame {frame_counter}, skipping")
            image.dispose()
            continue

        sidecar = {
            "frame_id": frame_counter,
            "timestamp_unix": timestamp,
            "capture_iso": time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime(timestamp)) + "Z",
            "width": width,
            "height": height,
            "hdr_max": HDR_MAX,
            "gain_db": config.GAIN_DB,
            "camera_serial": camera_serial,
        }
        queue_io.write_item(config.QUE_FULLFRAMES, stem, encoded.tobytes(), ".tiff", sidecar)

        image.dispose()
        del image, img_array

        if frame_counter % HEARTBEAT_EVERY_N_FRAMES == 0:
            write_heartbeat(camera_connected=True)

finally:
    device.stop_acquisition()
    device.dispose()
    system.dispose()
    write_heartbeat(camera_connected=False)
    print(f"[Stopped at Frame {frame_counter}] Acquisition stopped, resources released.")
