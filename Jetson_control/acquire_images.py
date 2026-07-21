from itala import itala
import ctypes
import numpy as np
import cv2
import os
import time
# import skimage
import gc
import psutil

### ==========================
### Recording parameters
### ==========================
ACQUIRE_COUNT = 100 # for long recording, <10000, for short recording 
FRAME_SKIP = 1 # Seconds/frame

SAVE_PATH = "jetson_firstrecordingattempt_0721"
image_type = "png"
#SAVE_PATH = "/media/jellyfish/PortableSSD/Training_data/Luidia_sarsia"

IMG_NAME_PREFIX = "img_"
GAIN_DB = 20.0 # hardware gain

ENABLE_SAVE = False
if ENABLE_SAVE:
    # Check if the save path exists, if not create it
    if not os.path.exists(SAVE_PATH):
        os.makedirs(SAVE_PATH)
    print(f"Saving images to {SAVE_PATH}")

ENABLE_LIVE_STREAM = True
STREAM_PORT = 8080 # view at http://<jetson-ip>:8080/ (or VS Code's forwarded port)
### Image processing settings
HDR_MAX = 4094 # should not be changed!!!!

### Live display settings, false for long term monitoring

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

device.start_acquisition()

print("Acquisition started.")

stop_event = None
if ENABLE_LIVE_STREAM:
    import threading
    from flask import Flask, Response

    latest_frame_lock = threading.Lock()
    latest_frame_jpeg = None
    stop_event = threading.Event()

    flask_app = Flask(__name__)

    @flask_app.route("/")
    def _stream_index():
        return (
            '<html><body style="margin:0;background:#000;height:100vh;'
            'display:flex;align-items:center;justify-content:center">'
            '<img src="/stream" style="max-width:100vw;max-height:100vh;object-fit:contain">'
            '<button onclick="this.innerText=\'Stopping...\';this.disabled=true;'
            "fetch('/stop',{method:'POST'})\" "
            'style="position:fixed;top:16px;left:16px;padding:12px 20px;font-size:16px;'
            'background:#c0392b;color:#fff;border:none;border-radius:6px;cursor:pointer">'
            "Stop Recording</button>"
            "</body></html>"
        )

    @flask_app.route("/stream")
    def _stream_feed():
        def generate():
            while True:
                with latest_frame_lock:
                    frame = latest_frame_jpeg
                if frame is not None:
                    yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + frame + b"\r\n")
                time.sleep(0.05)
        return Response(generate(), mimetype="multipart/x-mixed-replace; boundary=frame")

    @flask_app.route("/stop", methods=["POST"])
    def _stream_stop():
        stop_event.set()
        return "", 204

    threading.Thread(
        target=lambda: flask_app.run(host="0.0.0.0", port=STREAM_PORT, debug=False, use_reloader=False),
        daemon=True,
    ).start()
    print(f"Live stream enabled at http://<jetson-ip>:{STREAM_PORT}/")

time_counter = 0 # counts time (1s per trigger)
frame_counter = 0 # counts actually recorded frames
saved_counter = 0 # counts saved frames

background_median_full = None
waiting_time = (FRAME_SKIP + 2) * 1000

process = psutil.Process()

try:
    while (saved_counter < ACQUIRE_COUNT or ACQUIRE_COUNT == None) and not (stop_event and stop_event.is_set()):
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
        image12.dispose()
        del p, image12

        # img_array = np.subtract(img_array, dark_frame, out=img_array)
        img_array = np.clip(img_array, 1.0, HDR_MAX)

        img_to_save = img_array.astype(np.uint16)

        # Live display
        if ENABLE_LIVE_STREAM:
            img_to_display = np.clip(img_array * (255.0 / HDR_MAX), 0, 255).astype(np.uint8)
            display_frame = cv2.resize(img_to_display, (0,0), fx=disp_scale/100, fy=disp_scale/100, interpolation=cv2.INTER_AREA)
            cv2.putText(display_frame, f"Frame: {frame_counter}/{ACQUIRE_COUNT}", (dispx,dispy1), font, font_scale, color, thickness)
            cv2.putText(display_frame, time.strftime("%H:%M:%S"), (dispx, dispy3), font, font_scale, color, thickness)
            ok, encoded = cv2.imencode(".jpg", display_frame)
            if ok:
                with latest_frame_lock:
                    latest_frame_jpeg = encoded.tobytes()


        if ENABLE_SAVE:
            filename = os.path.join(SAVE_PATH, f"{IMG_NAME_PREFIX}{saved_counter:05d}.{image_type}")
            cv2.imwrite(filename, img_to_save)
            saved_counter += 1

        # Memory check & cleanup
        if frame_counter % 100 == 0:
            gc.collect()

        image.dispose()
        print(f"[Frame {frame_counter}] Saved: {saved_counter}/{ACQUIRE_COUNT} | Time: {time.strftime('%H:%M:%S')} | Memory usage: {process.memory_info().rss / 1e6:.1f} MB", end = '\r', flush=True)
        del image, img_array

finally:
    device.stop_acquisition()
    device.dispose()
    system.dispose()

    mem_mb = process.memory_info().rss / 1e6
    print(f"[Stopped at Frame {frame_counter}] With Memory usage: {mem_mb:.1f} MB")

    print("Acquisition stopped, resources released.")
