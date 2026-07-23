from itala import itala
import ctypes
import numpy as np
import cv2
import os
import time
import skimage
from collections import deque
import gc
import psutil

### ==========================
### Recording parameters
### ==========================
ACQUIRE_COUNT = 800 # for long recording, <10000, for short recording 
FRAME_SKIP = 3 # Seconds/frame

SAVE_PATH = "/media/jellyfish/PortableSSD/test_251111"
#SAVE_PATH = "/media/jellyfish/PortableSSD/Training_data/Luidia_sarsia"

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
CLAHE_CLIP = 0.01 # clahe parameter, can be adjusted
CLAHE_KERNEL = 512 # Clahe parameter, can be adjusted
HDR_MAX = 4094 # should not be changed!!!!

### Live display settings, false for long term monitoring
ENABLE_LIVE_STREAM = True

# shouldn't have to change but you can
disp_scale = 10 # Downscale for display [%]
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
system = itala.create_system()
devices_info = system.enumerate_devices(500)

if len(devices_info) == 0:
    print("No devices found. Exiting.")
    exit(1)
if devices_info[0].access_status != itala.DeviceAccessStatus_AvailableReadWrite:
    print("Device not accessible in RW mode. Exiting.")
    exit(1)

device = system.create_device(devices_info[0])
print("Device initialized.")
nodemap = device.node_map

nodemap.TriggerSelector.from_string("FrameBurstStart")
nodemap.TriggerSource.from_string("Line0")
nodemap.LineSelector.from_string("Line0")
nodemap.LineMode.from_string("Input")
nodemap.GainAuto.from_string("Off")
nodemap.Gain.set_value(GAIN_DB)

target = 4000 # booster frame size (4000)
valid_val = target - ((target - 560) % 8)
nodemap.GevSCPSPacketSize.value = valid_val
nodemap.AcquisitionBurstFrameCount.set_value(1)
nodemap.ExposureMode.from_string("TriggerWidth")
nodemap.TriggerMode.from_string("On")
nodemap.PixelFormat.from_string("Mono12p")

dark_frame = cv2.imread("bkg.png", cv2.IMREAD_UNCHANGED)
print("dtype:", dark_frame.dtype)
dark_frame = dark_frame.astype(np.float32)

if BG_SUB:
    if os.path.exists(bg_file):
        bg_buffer = np.load(bg_file)
        bg_buffer = deque(bg_buffer, maxlen=BACKGROUND_FRAMES)
    else:
        bg_buffer = deque(maxlen=BACKGROUND_FRAMES)
        print('made empty buffer')

device.start_acquisition()
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

        # CLAHE
        img_clahe = skimage.exposure.equalize_adapthist(
            (img_gamma * HDR_MAX).astype(np.uint16),
            clip_limit=CLAHE_CLIP,
            nbins=4096,
            kernel_size=CLAHE_KERNEL
        )
        img_clahe = np.clip(img_clahe, 0.0, 1.0)

        # HDR -> LDR
        img_to_save = (img_clahe * 255).astype(np.uint8)

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
                image.dispose()
                break

        # Memory check & cleanup
        if frame_counter % 100 == 0:
            gc.collect()

        image.dispose()
        del image, img_array, img_denoised, img_bg_subtracted, img_norm, img_gamma, img_clahe, img_to_save

finally:
    device.stop_acquisition()
    device.dispose()
    system.dispose()

    mem_mb = process.memory_info().rss / 1e6
    print(f"[Stopped at Frame {frame_counter}] With Memory usage: {mem_mb:.1f} MB")
 
    
    if BG_SUB:
        np.save(bg_file, bg_buffer)
        
    if ENABLE_LIVE_STREAM:
        cv2.destroyAllWindows()
    print("Acquisition stopped, resources released.")
