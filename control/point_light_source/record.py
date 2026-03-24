from itala import itala
import ctypes
import numpy as np
import cv2
import os
import time
import skimage
from skimage.transform import downscale_local_mean
from collections import deque
import gc
import psutil

### ==========================
### Recording parameters
### ==========================
ACQUIRE_COUNT = 10000 # for long recording, <10000, for short recording 
FRAME_SKIP = 1 # Seconds/frame

SAVE_PATH = "/home/jellyfish/point_light_source/test_2201"
#SAVE_PATH = "/media/jellyfish/PortableSSD/Training_data/Luidia_sarsia"

IMG_NAME_PREFIX = "img_"
GAIN_DB = 20.0 # hardware gain

# set binning factors 
BINNING_HORIZONTAL = 2 # can only be 1, 2 or 4
BINNING_VERTICAL = BINNING_HORIZONTAL

ENABLE_SAVE = False

### Image processing settings
MEDIAN_KERNEL_SIZE = 3 # Median calculation kernel, can be adjusted,for denoising
POST_GAIN = 1 # only for faints
GAMMA = 0.7 # enhance image, can be adjusted
CLAHE_CLIP = 0.01 # clahe parameter, can be adjusted
TILE_GRID_SIZE = 4 # Clahe parameter, can be adjusted
HDR_MAX = 4094 # should not be changed!!!!

### Live display settings, false for long term monitoring
ENABLE_LIVE_STREAM = True
disp_scale = 50 # Downscale for display [%]

# Set display settings for live stream
font = cv2.FONT_HERSHEY_SIMPLEX
scale = (disp_scale / 100) /BINNING_HORIZONTAL  # scale by display size and multiply by binning
font_scale = max(1, int(10 * scale))  # ensure minimum of 1
thickness = max(1, int(10 * scale))  # ensure minimum of 1
color = (255, 255, 255)

print("font_scale:", font_scale, "thickness:", thickness)

# Precalculate display positions scaled by display size and binning
dispx = int(80 * scale)
dispy1 = int(300 * scale)
dispy2 = int(600 * scale)
dispy3 = int(900 * scale)

# Precalculate CLAHE object if using OpenCV CLAHE
clahe = cv2.createCLAHE(clipLimit=CLAHE_CLIP * HDR_MAX, tileGridSize=(TILE_GRID_SIZE,TILE_GRID_SIZE))

### ==========================
### Initialize camera
### ==========================
system = itala.create_system()
devices_info = system.enumerate_devices(500)

if len(devices_info) == 0:
    print("No devices found. Exiting.")
    exit(1)
if devices_info[0].access_status != itala.DeviceAccessStatus_AvailableReadWrite:
    print("Device not accessible in RW mode. Exiting.")
    exit(1)

device = system.create_device(devices_info[0])
# Configure buffer handling mode before starting acquisition
datastream_nodemap = device.datastream_node_map
datastream_nodemap.StreamBufferHandlingMode.from_string("NewestOnly")

print("Device initialized.")
nodemap = device.node_map

nodemap.TriggerSelector.from_string("FrameBurstStart")
nodemap.TriggerSource.from_string("Line0")
nodemap.LineSelector.from_string("Line0")
nodemap.LineMode.from_string("Input")
nodemap.GainAuto.from_string("Off")
nodemap.Gain.set_value(GAIN_DB)

target = 5000 # jumbo frame size (4000)
valid_val = target - ((target - 560) % 8)
nodemap.GevSCPSPacketSize.value = valid_val
nodemap.AcquisitionBurstFrameCount.set_value(1)
nodemap.ExposureMode.from_string("TriggerWidth")
nodemap.TriggerMode.from_string("On")
nodemap.PixelFormat.from_string("Mono12p")

# Set binning
binning_h_node = nodemap.BinningHorizontal
binning_v_node = nodemap.BinningVertical
binning_h_node.value = BINNING_HORIZONTAL
binning_v_node.value = BINNING_VERTICAL

dark_frame = cv2.imread("bkg.png", cv2.IMREAD_UNCHANGED)
if BINNING_VERTICAL != 1 or BINNING_HORIZONTAL !=1:
    dark_frame = downscale_local_mean(dark_frame, (BINNING_VERTICAL, BINNING_HORIZONTAL)).astype(dark_frame.dtype)
else:
    dark_frame = dark_frame.astype(np.float32)

device.start_acquisition(20,0)
print("Acquisition started.")

if ENABLE_LIVE_STREAM:
    cv2.namedWindow("Live Stream", cv2.WINDOW_NORMAL)
    print("Live stream enabled. Press ESC to stop early.")

time_counter = 0 # counts time (1s per trigger)
frame_counter = 0 # counts actually recorded frames
saved_counter = 0 # counts saved frames

waiting_time = (FRAME_SKIP + 1) * 1000

process = psutil.Process()

try:
    while saved_counter < ACQUIRE_COUNT or ACQUIRE_COUNT is None:
        image = device.get_next_image(waiting_time)
        if image is None:
            print("No image returned, trigger issue")
            continue

        time_counter += 1
        if time_counter % FRAME_SKIP != 0:
            image.dispose()
            continue
        if image.is_incomplete:
            print(f"Image {image.frame_id} incomplete.")
            image.dispose()
            continue

        frame_counter +=1

        image12 = image.convert(itala.PfncFormat_Mono12)
        height, width = image12.height, image12.width
        size = width * height

        p = (ctypes.c_uint16 * size).from_address(int(image12.get_data()))

        img_array = np.ctypeslib.as_array(p).reshape((height, width)).astype(np.float32)
        # Dark frame subtraction
        img_array = np.subtract(img_array, dark_frame, out=img_array)
        img_array = np.clip(img_array, 1.0, HDR_MAX)

        # Spatial median denoising
        img_denoised = cv2.medianBlur(img_array.astype(np.uint16), MEDIAN_KERNEL_SIZE).astype(np.float32)

        # Gamma / post-gain
        img_norm = img_denoised / HDR_MAX
        img_gamma = np.clip(np.power(img_norm, GAMMA) * POST_GAIN, 0.0, 1.0)

        # CLAHE
        #img_clahe = skimage.exposure.equalize_adapthist(
            #(img_gamma * HDR_MAX).astype(np.uint16),
            #clip_limit=CLAHE_CLIP,
            #nbins=4096,
            #kernel_size=CLAHE_KERNEL
        #)
        #img_clahe = np.clip(img_clahe, 0.0, 1.0)
        # CLAHE using OpenCV (much faster than skimage)

        img_clahe = clahe.apply((img_gamma * 65535).astype(np.uint16))
        img_clahe = img_clahe.astype(np.float32) / 65535.0

        # HDR -> LDR
        img_to_save = (img_clahe * 255).astype(np.uint8)

        # Save frame
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
            display_frame = cv2.resize(img_to_save, (0,0), fx=disp_scale/100, fy=disp_scale/100, interpolation=cv2.INTER_AREA)
            cv2.putText(display_frame, f"Frame: {frame_counter}/{ACQUIRE_COUNT}", (dispx,dispy1), font, font_scale, color, thickness)
            cv2.putText(display_frame, f"Binning: {binning_h_node.value}x{binning_v_node.value}", (dispx, dispy2), font, font_scale, color, thickness)
            cv2.putText(display_frame, time.strftime("%H:%M:%S"), (dispx, dispy3), font, font_scale, color, thickness)
            cv2.imshow("Live Stream", display_frame)
            if cv2.waitKey(1) == 27:
                image.dispose()
                break

        # Memory check & cleanup
        if frame_counter % 100 == 0:
            gc.collect()

        image.dispose()
        del image, img_array, img_denoised, img_norm, img_gamma, img_clahe, img_to_save

finally:
    device.stop_acquisition()
    device.dispose()
    system.dispose()

    mem_mb = process.memory_info().rss / 1e6
    print(f"[Stopped at Frame {frame_counter}] With Memory usage: {mem_mb:.1f} MB")

    if ENABLE_LIVE_STREAM:
        cv2.destroyAllWindows()
    print("Acquisition stopped, resources released.")
