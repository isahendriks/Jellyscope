### macOS version of windows_control/monitoring/record.py - same settings and processing,
### but the camera is read through Aravis (see aravis_camera.py) instead of the Itala SDK.

import numpy as np
import cv2
import os
import time
import torch
import kornia
from collections import deque
import gc
import psutil
from aravis_camera import open_camera, get_next_frame, close_camera

### ==========================
### Recording parameters
### ==========================
ACQUIRE_COUNT = 5000 # for long recording, <10000, for short recording
FRAME_SKIP = 10 # Seconds/frame

SAVE_PATH = os.path.expanduser("~/Documents/Faro/Faro_260929/mixed_sample_2") # external SSD: "/Volumes/<name of the SSD>/..."

IMG_NAME_PREFIX = "img_"
GAIN_DB = 20.0 # hardware gain

ENABLE_SAVE = True

### Background subtraction settings - in progress, keep to False
BG_SUB = False
BACKGROUND_FRAMES = 50 # size of rolling median
BACKGROUND_SUBSAMPLE = 20 # subsample to compute rolling median from
DOWNSAMPLE_FACTOR = 8 # downsample when storing median
bg_file = "bg_buffer.npy"

### Image processing settings
MEDIAN_KERNEL_SIZE = 3 # Median calculation kernel, can be adjusted,for denoising
POST_GAIN = 1 # only for faints
GAMMA = 0.7 # enhance image, can be adjusted
ENABLE_CLAHE = True # False skips CLAHE (images then differ from the Jetson's)
CLAHE_CLIP = 0.01 * 400 # kornia clip limit, same as Jetson config.CLAHE_CLIP (skimage's 0.01 means something different in kornia)
CLAHE_KERNEL = 512 # CLAHE tile size in px -> grid = round(width / CLAHE_KERNEL) = 9x9, same as Jetson config.CLAHE_TILE_PX
HDR_MAX = 4094 # should not be changed!!!!
ROTATE_FRAME = 90 # degrees counter-clockwise (0, 90, 180 or 270), same as Jetson config.ROTATE_FRAME
ROTATE_CODES = {90: cv2.ROTATE_90_COUNTERCLOCKWISE, 180: cv2.ROTATE_180, 270: cv2.ROTATE_90_CLOCKWISE}

### Live display settings, false for long term monitoring
ENABLE_LIVE_STREAM = False

# shouldn't have to change but you can
disp_scale = 50 # Downscale for display [%]
font = cv2.FONT_HERSHEY_SIMPLEX
font_scale = 10
thickness = 10
font_scale = int(font_scale*(disp_scale/100))
thickness = int(thickness*(disp_scale/100))
color = (255, 255, 255)

dispx = int(80*(disp_scale/100))
dispy1 = int(300*(disp_scale/100))
dispy2 = int(600*(disp_scale/100))
dispy3 = int(900*(disp_scale/100))

### ==========================
### Initialize camera
### ==========================
camera, stream = open_camera(GAIN_DB)

script_dir = os.path.dirname(os.path.abspath(__file__))
dark_frame = cv2.imread(os.path.join(script_dir, "bkg.png"), cv2.IMREAD_UNCHANGED)
if dark_frame is None: # same as the Jetson: skip dark-frame subtraction instead of crashing
    print("No bkg.png found, skipping dark-frame subtraction (record one with record_dark_frame.py).")
else:
    print("dtype:", dark_frame.dtype)
    dark_frame = dark_frame.astype(np.float32)
    if ROTATE_FRAME:
        dark_frame = cv2.rotate(dark_frame, ROTATE_CODES[ROTATE_FRAME])

if BG_SUB:
    if os.path.exists(bg_file):
        bg_buffer = np.load(bg_file)
        bg_buffer = deque(bg_buffer, maxlen=BACKGROUND_FRAMES)
    else:
        bg_buffer = deque(maxlen=BACKGROUND_FRAMES)
        print('made empty buffer')

camera.start_acquisition()
print("Acquisition started.")

if ENABLE_LIVE_STREAM:
    cv2.namedWindow("Live Stream", cv2.WINDOW_NORMAL)
    print("Live stream enabled. Press ESC to stop early.")

time_counter = 0 # counts time (1s per trigger)
frame_counter = 0 # counts actually recorded frames
saved_counter = 0 # counts saved frames

background_median_full = None
waiting_time = (FRAME_SKIP + 1) * 1000

process = psutil.Process()

try:
    while saved_counter < ACQUIRE_COUNT or ACQUIRE_COUNT == None:
        status, frame_id, img_raw = get_next_frame(stream, waiting_time)
        if status == "timeout":
            print("No image returned, trigger issue")
            continue

        time_counter += 1
        if time_counter % FRAME_SKIP != 0:
            continue
        if status == "incomplete":
            print(f"Image {frame_id} incomplete.")
            continue

        frame_counter +=1

        height, width = img_raw.shape
        img_array = img_raw.astype(np.float32)
        if ROTATE_FRAME:
            img_array = cv2.rotate(img_array, ROTATE_CODES[ROTATE_FRAME])

        if dark_frame is not None:
            img_array = np.subtract(img_array, dark_frame, out=img_array)
        img_array = np.clip(img_array, 1.0, HDR_MAX)

        # Spatial median denoising
        img_denoised = cv2.medianBlur(img_array.astype(np.uint16), MEDIAN_KERNEL_SIZE).astype(np.float32)

        # Background subtraction
        if BG_SUB:
            img_small = cv2.resize(img_denoised, (width//DOWNSAMPLE_FACTOR, height//DOWNSAMPLE_FACTOR), interpolation=cv2.INTER_LINEAR)
            bg_buffer.append(img_small)

            if len(bg_buffer) == BACKGROUND_FRAMES:
                # Compute median on downsampled frames
                idx = np.random.choice(len(bg_buffer), size=BACKGROUND_SUBSAMPLE, replace=False)
                subset = [bg_buffer[i] for i in idx]
                bg_small_median = np.median(np.stack(subset, axis=0), axis=0)

                # Upsample to full resolution
                background_median_full = cv2.resize(bg_small_median, (width, height), interpolation=cv2.INTER_LINEAR)

                # delete for memory management
                gc.collect()
                del subset, bg_small_median
            if background_median_full is not None:
                img_bg_subtracted = img_denoised - background_median_full
                img_bg_subtracted = np.clip(img_bg_subtracted, 0, HDR_MAX)
            else:
                img_bg_subtracted = img_denoised
                print("Background calculating " + str(len(bg_buffer)) + "/" + str(BACKGROUND_FRAMES))
        else:
            img_bg_subtracted = img_denoised

        # Gamma / post-gain
        img_norm = img_bg_subtracted / HDR_MAX
        img_gamma = np.clip(np.power(img_norm, GAMMA) * POST_GAIN, 0.0, 1.0)

        # CLAHE (kornia, same as Jetson gpu_preprocess.py)
        if ENABLE_CLAHE:
            clahe_grid = max(1, round(width / CLAHE_KERNEL))
            img_clahe = kornia.enhance.equalize_clahe(
                torch.from_numpy(img_gamma.astype(np.float32))[None, None],
                clip_limit=CLAHE_CLIP,
                grid_size=(clahe_grid, clahe_grid)
            ).squeeze().numpy()
            img_clahe = np.clip(img_clahe, 0.0, 1.0)
        else:
            img_clahe = img_gamma

        # HDR -> LDR
        img_to_save = np.round(img_clahe * 255).astype(np.uint8)

        # Save frame
        if background_median_full is not None or BG_SUB is False:
            timestamp = time.strftime("%Y%m%d_%H%M%S") + f"_{int(time.time()*1000)%1000:03d}"
            saved_counter+=1

            if ENABLE_SAVE:
                filename = f"{IMG_NAME_PREFIX}{timestamp}.png"
                output_path = os.path.join(SAVE_PATH, filename)
                os.makedirs(os.path.dirname(output_path), exist_ok=True)
                cv2.imwrite(output_path, img_to_save)
                print(f"saved im {saved_counter}/{ACQUIRE_COUNT} to {filename}")
            else:
                print(f"recorded im {saved_counter}/{ACQUIRE_COUNT}")

        # Live display
        if ENABLE_LIVE_STREAM:
            display_frame = cv2.resize(img_to_save, (0,0), fx=disp_scale/100, fy=disp_scale/100, interpolation=cv2.INTER_AREA)
            if BG_SUB and background_median_full is None:
                cv2.putText(display_frame, f"BG: {len(bg_buffer)}/{BACKGROUND_FRAMES}", (dispx,dispy1), font, font_scale, color, thickness)
                cv2.putText(display_frame, f"Computing BG",(dispx, dispy2), font, font_scale, color, thickness)
            elif BG_SUB and background_median_full is not None:
                cv2.putText(display_frame, f"Frame: {frame_counter}/{ACQUIRE_COUNT}", (dispx,dispy1), font, font_scale, color, thickness)
                cv2.putText(display_frame, f"BG Ready",(dispx, dispy2), font, font_scale, color, thickness)
            else:
                cv2.putText(display_frame, f"Frame: {frame_counter}/{ACQUIRE_COUNT}", (dispx,dispy1), font, font_scale, color, thickness)
                cv2.putText(display_frame, "BG sub disabled",(dispx, dispy2), font, font_scale, color, thickness)

            cv2.putText(display_frame, time.strftime("%H:%M:%S"), (dispx, dispy3), font, font_scale, color, thickness)
            cv2.imshow("Live Stream", display_frame)
            if cv2.waitKey(1) == 27:
                break

        # Memory check & cleanup
        if frame_counter % 100 == 0:
            gc.collect()

        del img_raw, img_array, img_denoised, img_bg_subtracted, img_norm, img_gamma, img_clahe, img_to_save

finally:
    close_camera(camera)

    mem_mb = process.memory_info().rss / 1e6
    print(f"[Stopped at Frame {frame_counter}] With Memory usage: {mem_mb:.1f} MB")


    if BG_SUB:
        np.save(bg_file, bg_buffer)

    if ENABLE_LIVE_STREAM:
        cv2.destroyAllWindows()
    print("Acquisition stopped, resources released.")
