### macOS version of windows_control/monitoring/record_dark_frame.py (camera read through Aravis).
### Records a new dark frame (bkg.png) for record.py with the connected camera.
### Cover the lens or switch the strobe/light off, but keep the trigger running:
### exposure is set by the trigger pulse width (ExposureMode TriggerWidth), so the
### dark frame must be taken with the same trigger settings as the real recording.
### Saved unrotated as a 16-bit PNG next to this script - record.py rotates it itself.

import numpy as np
import cv2
import os
import time
from aravis_camera import open_camera, get_next_frame, close_camera

### ==========================
### Dark frame parameters
### ==========================
N_FRAMES = 20 # number of dark frames to average
OUTPUT_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "bkg.png")
GAIN_DB = 20.0 # hardware gain, must match record.py
TRIGGER_TIMEOUT_S = 30 # how long to wait for a trigger pulse
MAX_DARK_MEAN = 200 # warn if the average frame is brighter than this (12-bit counts) - light probably not off

### ==========================
### Initialize camera (same setup as record.py)
### ==========================
camera, stream = open_camera(GAIN_DB)

input("Cover the lens / switch the light off (keep the trigger running), then press Enter...")

camera.start_acquisition()
print("Acquisition started.")

frame_sum = None
n_recorded = 0

try:
    while n_recorded < N_FRAMES:
        status, frame_id, img_raw = get_next_frame(stream, TRIGGER_TIMEOUT_S * 1000)
        if status == "timeout":
            print("No image returned, trigger issue")
            continue
        if status == "incomplete":
            print(f"Image {frame_id} incomplete.")
            continue

        img_array = img_raw.astype(np.float64)
        frame_sum = img_array if frame_sum is None else frame_sum + img_array
        n_recorded += 1
        print(f"dark frame {n_recorded}/{N_FRAMES}, mean {img_array.mean():.1f}")

finally:
    close_camera(camera)

dark_frame = np.round(frame_sum / n_recorded).astype(np.uint16)
print(f"Average dark frame: mean {dark_frame.mean():.1f}, max {dark_frame.max()} (12-bit counts)")
if dark_frame.mean() > MAX_DARK_MEAN:
    print(f"WARNING: mean is above {MAX_DARK_MEAN}, the light may not have been off.")

if os.path.exists(OUTPUT_FILE): # keep the previous one instead of overwriting it
    backup = os.path.join(os.path.dirname(OUTPUT_FILE), f"bkg_old_{time.strftime('%Y%m%d_%H%M%S')}.png")
    os.rename(OUTPUT_FILE, backup)
    print(f"Existing bkg.png renamed to {backup}")

cv2.imwrite(OUTPUT_FILE, dark_frame)
print(f"Saved {OUTPUT_FILE}")
