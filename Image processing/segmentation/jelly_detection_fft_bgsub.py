"""
Merged FFT + rolling-median-background script.

- Reads images from LOAD_PATH (PNG/ JPG / tif / etc.)
- Uses a deque of downsampled-by-5 frames to compute rolling median background.
- First N_SKIP frames (default 100) are only used to initialize background and FFT baseline.
- After that, each frame is background-subtracted, CLAHE-applied and analyzed by masked FFT.
- If fft_mean / median_fft > THRESHOLD_RATIO -> considered an observation (save to obs folder),
  otherwise save to no_obs folder.
- Live display shows current frame and the upscaled median background side-by-side at 10% resolution.
  A green rectangle is drawn around the current frame if a jelly is detected.
"""

#%%
import os
import glob
import time
from collections import deque

import numpy as np
import cv2
import skimage
from skimage import exposure, measure

#%%

# ----------------------------
# User / runtime configuration
# ----------------------------
LOAD_PATH = r"C:/Users/Admin/OneDrive - Lund University/Documents/Jellyscope Data/Monitoring_Data/Kristineberg_250915/gain10_nobgsub"
OUTPUT_BASE = r"C:/Users/Admin/OneDrive - Lund University/Documents/Jellyscope Data/Monitoring_Data/Kristineberg_250915/gain10_bgsub_sorted"
OBS_FOLDER = os.path.join(OUTPUT_BASE, "obs")
NOOBS_FOLDER = os.path.join(OUTPUT_BASE, "no_obs")
LOG_PATH = os.path.join(OUTPUT_BASE, "observationlog.txt")

os.makedirs(OBS_FOLDER, exist_ok=True)
os.makedirs(NOOBS_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_BASE, exist_ok=True)

# rolling background
ROLLING_WINDOW_LEN = 100             # last N frames used for median background
DOWNSAMPLE_FACTOR_BG = 5       # downsample by x5 before adding to deque (as requested)
UPSAMPLE_INTERP = cv2.INTER_LINEAR

# display & save
DISPLAY_SCALE = 0.10           # display at 10% resolution (as requested)
THRESHOLD_RATIO = 1.014        # detection threshold for fft_ratio
N_SKIP = ROLLING_WINDOW_LEN                  # first N_SKIP frames used only to populate background & baseline

# FFT mask radii (adjust if needed)
R_IN = 70
R_OUT = 1000

# CLAHE / pre-processing for display & analysis
CLAHE_CLIP = 0.01
CLAHE_KERNEL = None  # None => automatic

# supported image extensions
EXTS = ("*.png", "*.tif", "*.tiff", "*.jpg", "*.jpeg", "*.bmp")

# ----------------------------
# Helper functions
# ----------------------------
def hist_equalization(img):
    """
    Simple CLAHE-based histogram equalization for grayscale float or uint images.
    Returns uint8 image (0..255).
    """
    # Convert to float in [0,1]
    if img.dtype == np.uint8:
        img_f = img.astype(np.float32) / 255.0
    else:
        img_f = skimage.img_as_float(img)

    # apply adaptive histogram equalization
    eq = exposure.equalize_adapthist(img_f, clip_limit=CLAHE_CLIP, kernel_size=CLAHE_KERNEL)
    eq_u8 = (np.clip(eq, 0.0, 1.0) * 255.0).astype(np.uint8)
    return eq_u8

def calc_fft_mean(img, r_in=R_IN, r_out=R_OUT):
    """
    Calculate mean of masked FFT magnitude spectrum for the input image (grayscale 2D array).
    img should be float or uint — will be converted to float for FFT.
    """
    img_f = img.astype(np.float32)
    fft_img = np.fft.fft2(img_f)
    fft_shift = np.fft.fftshift(fft_img)
    magnitude_spectrum = np.abs(fft_shift)

    rows, cols = img.shape
    crow, ccol = rows // 2, cols // 2
    y, x = np.ogrid[:rows, :cols]
    dist_from_center = (x - ccol) ** 2 + (y - crow) ** 2
    mask = np.ones((rows, cols), np.uint8)
    mask[dist_from_center <= r_in**2] = 0
    if r_out is not None:
        mask[dist_from_center >= r_out**2] = 0

    magnitude_masked = magnitude_spectrum * mask
    non_zero = magnitude_masked[magnitude_masked != 0]
    if non_zero.size > 0:
        return float(np.mean(non_zero))
    else:
        return 0.0

def safe_ratio(a, b):
    """Return a/b but handle zero denominator."""
    if b == 0:
        return np.inf if a > 0 else 1.0
    return a / b

# ----------------------------
# Load list of images
# ----------------------------
image_files = []
for ext in EXTS:
    image_files.extend(sorted(glob.glob(os.path.join(LOAD_PATH, ext))))
image_files = sorted(image_files)
n_images = len(image_files)
print(f"Found {n_images} images in {LOAD_PATH}")

if n_images == 0:
    raise SystemExit("No images found - check LOAD_PATH and extensions.")

# ----------------------------
# Rolling background deque (downsampled frames)
# ----------------------------
bg_deque = deque(maxlen=ROLLING_WINDOW_LEN)

# For FFT baseline, maintain a deque of recent FFT means (for rolling median)
fft_deque = deque(maxlen=ROLLING_WINDOW_LEN)

# Variables to hold upscaled median background (full res)
background_median_full = None

# Open log file
log_f = open(LOG_PATH, "w", buffering=1)
log_f.write(f"Observation log started: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")

# Create display window
cv2.namedWindow("Current | Background", cv2.WINDOW_NORMAL)

# ----------------------------
# Main loop: iterate images
# ----------------------------
for idx, filepath in enumerate(image_files):
    try:
        img_raw = skimage.io.imread(filepath)
    except Exception as e:
        print(f"Skipping {filepath}: could not read ({e})")
        continue

    # Ensure grayscale 2D
    if img_raw.ndim == 3:
        # convert to grayscale
        img_gray = skimage.color.rgb2gray(img_raw)
        img_gray = skimage.img_as_float(img_gray)
        img_full = (img_gray * 255.0).astype(np.uint8)
    else:
        # if multi-bit, normalize to 0..255 for processing
        if img_raw.dtype == np.uint16:
            img_full = ((img_raw.astype(np.float32) / np.iinfo(np.uint16).max) * 255.0).astype(np.uint8)
        else:
            img_full = skimage.img_as_ubyte(img_raw)

    height, width = img_full.shape

    # --- 1) Create downsampled version for background deque ---
    small_h = max(1, height // DOWNSAMPLE_FACTOR_BG)
    small_w = max(1, width // DOWNSAMPLE_FACTOR_BG)
    img_small = cv2.resize(img_full, (small_w, small_h), interpolation=cv2.INTER_AREA)
    # store as float for median accuracy
    bg_deque.append(img_small.astype(np.float32))

    # --- 2) Update background median_full if we have enough frames in deque ---
    if len(bg_deque) == ROLLING_WINDOW_LEN:
        # compute median of stack (on downsampled scale)
        stack = np.stack(list(bg_deque), axis=0)  # shape (N, h, w)
        bg_small_median = np.median(stack, axis=0).astype(np.float32)

        # upscale to full resolution
        background_median_full = cv2.resize(bg_small_median, (width, height), interpolation=UPSAMPLE_INTERP)
        # clip to 0..255 and cast to float32
        background_median_full = np.clip(background_median_full, 0, 255).astype(np.float32)

    # --- 3) Prepare analysis image: subtract upscaled median background (when available) ---
    if background_median_full is not None:
        img_f = img_full.astype(np.float32)
        img_bgsub = img_f - background_median_full
        img_bgsub = np.clip(img_bgsub, 0, 255).astype(np.uint8)
    else:
        # until we have a median background, use the original image for analysis
        img_bgsub = img_full.copy()

    # --- 4) Compute FFT mean for this frame ---
    fft_mean = calc_fft_mean(img_bgsub)

    # Add current to deque after we compute comparison 
    if len(fft_deque) >= 1:
        median_fft_prior = float(np.median(list(fft_deque)))
    else:
        median_fft_prior = fft_mean if idx < N_SKIP else 1.0  # avoid zero division

    fft_ratio = safe_ratio(fft_mean, median_fft_prior)

    # If still in the initialization phase (first N_SKIP frames), do not analyze/save
    is_initializing = (idx < N_SKIP)

    # Decide observation: only after initialization compare ratio to threshold
    is_obs = False
    if not is_initializing:
        if fft_ratio > THRESHOLD_RATIO:
            is_obs = True

    # --- 6) Save image to appropriate folder ---
    base_name = os.path.basename(filepath)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    save_name = f"{idx:06d}_{base_name}"

    if is_initializing:
        # during init, we can save into a 'init' folder if desired or skip saving
        # Let's write them to NOOBS folder as initialization frames (optional)
        save_path = os.path.join(NOOBS_FOLDER, save_name)
        cv2.imwrite(save_path, img_bgsub)
    else:
        if is_obs:
            save_path = os.path.join(OBS_FOLDER, save_name)
            cv2.imwrite(save_path, img_bgsub)
        else:
            save_path = os.path.join(NOOBS_FOLDER, save_name)
            cv2.imwrite(save_path, img_bgsub)

    # log line
    log_line = (f"Idx {idx:06d} file {base_name} fft_mean {fft_mean:.3f} median_fft_prior {median_fft_prior:.3f} "
                f"ratio {fft_ratio:.5f} init:{is_initializing} obs:{is_obs}\n")
    print(log_line.strip())
    log_f.write(log_line)

    # --- 7) Live display: show current and median background side-by-side at 10% size ---
    # Prepare display images (RGB for rectangle/color)
    display_curr = cv2.cvtColor(img_bgsub, cv2.COLOR_GRAY2BGR)
    if background_median_full is not None:
        bg_for_display = np.clip(background_median_full, 0, 255).astype(np.uint8)
    else:
        # if no background yet, show a black image or blurred average
        bg_for_display = np.zeros_like(img_bgsub)

    display_bg = cv2.cvtColor(bg_for_display, cv2.COLOR_GRAY2BGR)

    # If a jelly is detected, draw green rectangle around the current image (entire image)
    if (not is_initializing) and is_obs:
        # rectangle parameters:
        cv2.rectangle(display_curr, (0, 0), (width - 1, height - 1), (0, 255, 0), thickness=10)
        status_text = f"JELLY DETECTED! ratio {fft_ratio:.3f}"
    else:
        status_text = f"ratio {fft_ratio:.3f}"

    # overlay text on current image (top-left)
    cv2.putText(display_curr, f"Idx:{idx}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
    cv2.putText(display_curr, status_text, (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)

    # Resize to display scale
    disp_w = max(1, int(width * DISPLAY_SCALE))
    disp_h = max(1, int(height * DISPLAY_SCALE))
    disp_curr_small = cv2.resize(display_curr, (disp_w, disp_h), interpolation=cv2.INTER_AREA)
    disp_bg_small = cv2.resize(display_bg, (disp_w, disp_h), interpolation=cv2.INTER_AREA)

    # Combine side-by-side
    combined = np.hstack([disp_curr_small, disp_bg_small])

    # Put an overarching timestamp
    cv2.putText(combined, time.strftime("%Y-%m-%d %H:%M:%S"), (10, disp_h - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

    cv2.imshow("Current | Background", combined)

    # handle key press (ESC to quit)
    k = cv2.waitKey(1) & 0xFF
    if k == 27:
        print("ESC pressed, stopping.")
        break

    # --- 8) After display and save, update fft_deque with current fft (so future frames compare to "previouss") ---
    fft_deque.append(float(fft_mean))

# cleanup
log_f.write("Observation log ended: " + time.strftime('%Y-%m-%d %H:%M:%S') + "\n")
log_f.close()
cv2.destroyAllWindows()
print("Done.")