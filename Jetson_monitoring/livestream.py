from itala import itala
import ctypes
import json
import numpy as np
import cv2
import os
import time
# import skimage
import gc
import psutil

from Monitor import config

### ==========================
### Recording parameters -- livestream-only knobs (dev/live-view capture, not the
### production record.py/analyse.py pipeline) stay local; anything shared with that
### pipeline (gain, frame skip, strobe, HDR/PREPROCESS tuning, live-stream port/scale)
### is imported from Monitor/config.py instead of duplicated here, so both stay in sync.
### ==========================
ACQUIRE_COUNT = 60*60*2 # For 24-hour recording

SAVE_PATH = "/mnt/sdb1/Kristineberg_260730_continuous_preprocessed"
image_type = "png"
#SAVE_PATH = "/media/jellyfish/PortableSSD/Training_data/Luidia_sarsia"

IMG_NAME_PREFIX = "img_"

ENABLE_SAVE = True
if ENABLE_SAVE:
    # Check if the save path exists, if not create it
    if not os.path.exists(SAVE_PATH):
        os.makedirs(SAVE_PATH)
    print(f"Saving images to {SAVE_PATH}")

### ==========================
### Preprocessing pipeline (dark-subtract -> median-blur -> gamma -> CLAHE), mirrors
### Monitor/config.py's PREPROCESS knobs and runs through gpu_preprocess.py's GPU chain.
### Off by default since this is a quick dev/live-view capture script, not the production
### pipeline -- turn on to preview what analyse.py's PREPROCESS step would actually produce
### (applies to both the live stream image and any saved frames).
### ==========================
PREPROCESS = True

if PREPROCESS:
    import torch
    import gpu_preprocess

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
nodemap.Gain.set_value(config.GAIN_DB)

target = 4000 # booster frame size (4000)
valid_val = target - ((target - 560) % 8)
nodemap.GevSCPSPacketSize.value = valid_val
nodemap.AcquisitionBurstFrameCount.set_value(1)
nodemap.ExposureMode.from_string("TriggerWidth")
nodemap.TriggerMode.from_string("On")
nodemap.PixelFormat.from_string("Mono12p")

preprocess_device = None
dark_frame = None
if PREPROCESS:
    preprocess_device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    print(f"PREPROCESS enabled, using device: {preprocess_device}")
    if config.DARK_FRAME_PATH.exists():
        dark_frame = gpu_preprocess.load_dark_frame(
            str(config.DARK_FRAME_PATH), preprocess_device, rotate_angle_deg=config.ROTATE_FRAME,
        )
        print(f"Loaded dark frame from {config.DARK_FRAME_PATH}")
    else:
        print(f"Dark frame path {config.DARK_FRAME_PATH} not found, skipping dark-frame subtraction.")

device.start_acquisition()

print("Acquisition started.")

stop_event = None
if config.ENABLE_LIVE_STREAM:
    import threading
    from flask import Flask, Response, jsonify

    latest_frame_lock = threading.Lock()
    latest_frame_jpeg = None
    stop_event = threading.Event()

    status_lock = threading.Lock()
    status = {
        "frame": 0, "total": ACQUIRE_COUNT, "saved": 0,
        "camera_c": None, "proc_time_ms": None,
    }
    camera_temp_supported = True

    def _read_live_metadata() -> dict:
        # Reads metadata.py's live snapshot (metadata_live.json), same as analyse.py's own
        # live-stream page -- gives ENVIRONMENTAL/LEAK/DEVICE readings here too, if metadata.py
        # happens to be running alongside this capture. Returns {} (all N/A on the page)
        # if it isn't, same as analyse.py's fallback.
        #
        # Strobe converter/driver temps come from here too (not a direct Modbus poll of our
        # own) -- the strobe controller only accepts one Modbus/TCP connection at a time, and
        # running livestream.py's own client alongside metadata.py's meant the two fought over
        # that single slot, so the temperature readout would intermittently fail or sit at N/A.
        # metadata.py is the sole owner of that connection now, same as it already is in the
        # full record/analyse/send pipeline.
        path = config.HEARTBEAT_DIR / "metadata_live.json"
        if not path.exists():
            return {}
        try:
            return json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            return {}

    flask_app = Flask(__name__)

    @flask_app.route("/")
    def _stream_index():
        return (
            '<html><body style="margin:0;background:#000;height:100vh;'
            'display:flex;align-items:center;justify-content:center">'
            '<img id="videoFrame" style="max-width:100vw;max-height:100vh;object-fit:contain;'
            'border:2px solid #00ff00;box-sizing:border-box">'
            '<pre id="status" style="position:fixed;top:76px;left:16px;margin:0;'
            'padding:14px 18px;font-size:12px;line-height:1.6;color:#fff;'
            'background:rgba(10,10,10,0.88);border-radius:10px;white-space:pre;'
            'font-family:-apple-system,Menlo,Consolas,monospace"></pre>'
            "<script>"
            # Polls a single always-fresh frame instead of holding open a multipart MJPEG
            # stream -- see /latest_frame.jpg's comment for why (matches analyse.py's
            # live-stream pattern): a persistent stream can only fall further behind if the
            # browser/network ever briefly lags the production rate, since it must display
            # every buffered frame in order before reaching "now". A fresh request each tick
            # always gets whatever's current, nothing accumulates.
            "function updateFrame(){"
            "document.getElementById('videoFrame').src='/latest_frame.jpg?t='+Date.now();"
            "}"
            "updateFrame();setInterval(updateFrame,200);"
            # Same metadata panel layout/helpers as analyse.py's live-stream page (ENVIRONMENTAL/
            # LEAK DETECTION/DEVICE) -- QUEUE and CONNECTION are omitted since this standalone
            # capture script has no queues and no send.py uploading anything.
            "async function updateStatus(){"
            "try{"
            "const d=await (await fetch('/status')).json();"
            "const fmt=(v,u)=>(v===null||v===undefined)?'N/A':v.toFixed(1)+u;"
            "const overMax=(v,m)=>v!==null&&v!==undefined&&m!==null&&m!==undefined&&v>=m;"
            "const nearMax=(v,m)=>v!==null&&v!==undefined&&m!==null&&m!==undefined&&v>=0.9*m;"
            "const flag=(v,m)=>overMax(v,m)?' [DANGER]':nearMax(v,m)?' [WARN]':'';"
            "const vsMax=(v,m,u)=>`${fmt(v,u)} / max ${fmt(m,u)}${flag(v,m)}`;"
            "const hdr=(label)=>`<b style=\"font-size:14px;letter-spacing:0.4px\">${label}</b>\\n`;"
            "document.getElementById('status').innerHTML="
            "`<b style=\"font-size:15px\">${d.time}</b>\\n`+"
            "`Frame: ${d.frame}/${d.total}  Saved: ${d.saved}\\n`+"
            "`Processing time/frame: ${fmt(d.proc_time_ms,' ms')}\\n \\n`+"
            "hdr('LEAK DETECTION')+"
            "`Leak: ${d.leak_detected===true?'!!! DETECTED !!!':d.leak_detected===false?'OK':'N/A'}  "
            "Enclosure humidity: ${fmt(d.bme280_humidity_pct,'%')}  "
            "Dew point: ${fmt(d.dew_point_c,' C')}\\n \\n`+"
            "hdr('ENVIRONMENTAL')+"
            "`Bar3XT pressure: ${fmt(d.bar3xt_pressure_mbar,' mbar')}  "
            "depth: ${fmt(d.bar3xt_depth_m,' m')}\\n`+"
            "`Bar3XT temp: ${fmt(d.bar3xt_temp_c,' C')}  "
            "DS18B20 temp: ${fmt(d.ds18b20_temp_c,' C')}\\n \\n`+"
            "hdr('DEVICE')+"
            "`Enclosure pressure: ${fmt(d.bme280_pressure_mbar,' mbar')}  "
            "Enclosure temp (BME280): ${fmt(d.bme280_temp_c,' C')}\\n`+"
            "`Camera: ${vsMax(d.camera_c,d.camera_max_temp_c,' C')}\\n`+"
            "`Strobe converter: ${vsMax(d.strobe_converter_c,d.strobe_max_temp_c,' C')}\\n`+"
            "`Strobe driver: ${vsMax(d.strobe_driver_c,d.strobe_max_temp_c,' C')}\\n`+"
            "`Jetson: ${vsMax(d.jetson_temp_c_mean,d.jetson_max_temp_c,' C')}\\n`+"
            "`CPU: ${fmt(d.cpu_percent,'%')}  GPU: ${fmt(d.gpu_percent_mean,'%')} (max ${fmt(d.gpu_percent_max,'%')})\\n`+"
            "`Disk free: device: ${fmt(d.disk_root,' GB')} `+"
            "`ssd1: ${fmt(d.disk_ssd1,' GB')}  ssd2: ${fmt(d.disk_ssd2,' GB')}`;"
            "const danger=d.leak_detected===true||"
            "overMax(d.camera_c,d.camera_max_temp_c)||"
            "overMax(d.strobe_converter_c,d.strobe_max_temp_c)||"
            "overMax(d.strobe_driver_c,d.strobe_max_temp_c)||"
            "overMax(d.jetson_temp_c_mean,d.jetson_max_temp_c);"
            "document.getElementById('status').style.background="
            "danger?'rgba(192,57,43,0.85)':'rgba(10,10,10,0.88)';"
            "}catch(e){}"
            "}"
            "updateStatus();setInterval(updateStatus,1000);"
            "</script>"
            "</body></html>"
        )

    @flask_app.route("/status")
    def _stream_status():
        with status_lock:
            payload = dict(status)
        payload["time"] = time.strftime("%H:%M:%S")
        live = _read_live_metadata()
        disk = live.get("disk_free_gb_by_path", {})
        payload.update({
            "strobe_converter_c": live.get("strobe_converter_temp_c"),
            "strobe_driver_c": live.get("strobe_driver_temp_c"),
            "bar3xt_pressure_mbar": live.get("bar3xt_pressure_mbar"),
            "bar3xt_depth_m": live.get("bar3xt_depth_m"),
            "bar3xt_temp_c": live.get("bar3xt_temp_c"),
            "bme280_pressure_mbar": live.get("bme280_pressure_mbar"),
            "bme280_humidity_pct": live.get("bme280_humidity_pct"),
            "bme280_temp_c": live.get("bme280_temp_c"),
            "dew_point_c": live.get("dew_point_c"),
            "ds18b20_temp_c": live.get("ds18b20_temp_c"),
            "leak_detected": live.get("leak_detected"),
            "jetson_temp_c_mean": live.get("jetson_temp_c_mean"),
            "cpu_percent": live.get("cpu_percent"),
            "gpu_percent_mean": live.get("gpu_percent_mean"),
            "gpu_percent_max": live.get("gpu_percent_max"),
            "camera_max_temp_c": config.CAMERA_MAX_TEMP_C,
            "strobe_max_temp_c": config.STROBE_MAX_TEMP_C,
            "jetson_max_temp_c": config.JETSON_MAX_TEMP_C,
            "disk_root": disk.get(str(config.ROOT_DISK_PATH)),
            "disk_ssd1": disk.get(str(config.EXTERNAL_SSD_PATHS[0])),
            "disk_ssd2": disk.get(str(config.EXTERNAL_SSD_PATHS[1])),
        })
        return jsonify(payload)

    @flask_app.route("/latest_frame.jpg")
    def _latest_frame():
        # Single-shot fetch of whatever's currently latest, not a persistent multipart
        # stream -- see updateFrame()'s comment above for why.
        with latest_frame_lock:
            frame = latest_frame_jpeg
        if frame is None:
            return Response(status=404)
        resp = Response(frame, mimetype="image/jpeg")
        resp.headers["Cache-Control"] = "no-store"
        return resp

    @flask_app.route("/stop", methods=["POST"])
    def _stream_stop():
        stop_event.set()
        return "", 204

    threading.Thread(
        target=lambda: flask_app.run(host="0.0.0.0", port=config.LIVESTREAM_PORT, debug=False, use_reloader=False),
        daemon=True,
    ).start()
    print(f"Live stream enabled at http://<jetson-ip>:{config.LIVESTREAM_PORT}/")

time_counter = 0 # counts time (1s per trigger)
frame_counter = 0 # counts actually recorded frames
saved_counter = 0 # counts saved frames

background_median_full = None
waiting_time = (config.FRAME_SKIP + 2) * 1000

process = psutil.Process()

try:
    while (saved_counter < ACQUIRE_COUNT or ACQUIRE_COUNT == None) and not (stop_event and stop_event.is_set()):
        image = device.get_next_image(waiting_time)

        if image is None:
            print("No image returned, trigger issue")
            continue

        time_counter += 1

        if time_counter % config.FRAME_SKIP != 0:
            image.dispose()
            continue

        if image.is_incomplete:
            print(f"Image {image.frame_id} incomplete.")
            image.dispose()
            continue

        frame_counter +=1
        frame_start = time.perf_counter()

        image12 = image.convert(itala.PfncFormat_Mono12)
        height, width = image12.height, image12.width
        size = width * height

        p = (ctypes.c_uint16 * size).from_address(int(image12.get_data()))
        
        img_array = np.ctypeslib.as_array(p).reshape((height, width)).astype(np.float32)
        image12.dispose()
        del p, image12

        # img_array = np.subtract(img_array, dark_frame, out=img_array)
        img_array = np.clip(img_array, 1.0, config.HDR_MAX)

        img_to_save = img_array.astype(np.uint16)

        enhanced = None
        if PREPROCESS:
            clahe_grid_n = max(1, round(width / config.CLAHE_TILE_PX))
            enhanced = gpu_preprocess.gpu_preprocess_frame(
                img_to_save, dark_frame, preprocess_device, config.HDR_MAX, config.MEDIAN_KERNEL_SIZE,
                config.GAMMA, config.POST_GAIN, config.CLAHE_CLIP, (clahe_grid_n, clahe_grid_n),
                rotate_angle_deg=config.ROTATE_FRAME,
            )

        # Live display
        if config.ENABLE_LIVE_STREAM:
            if camera_temp_supported:
                try:
                    with status_lock:
                        status["camera_c"] = nodemap.DeviceTemperature.value
                except Exception as exc:
                    camera_temp_supported = False
                    print(f"Camera temperature unavailable: {exc}")

            with status_lock:
                status["frame"] = frame_counter
                status["saved"] = saved_counter

            if PREPROCESS:
                img_to_display = enhanced
            else:
                img_to_display = np.clip(img_array * (255.0 / config.HDR_MAX), 0, 255).astype(np.uint8)
            display_frame = cv2.resize(
                img_to_display, (0,0), fx=config.LIVESTREAM_DISP_SCALE/100, fy=config.LIVESTREAM_DISP_SCALE/100,
                interpolation=cv2.INTER_AREA,
            )
            ok, encoded = cv2.imencode(".jpg", display_frame)
            if ok:
                with latest_frame_lock:
                    latest_frame_jpeg = encoded.tobytes()


        if ENABLE_SAVE:
            filename = os.path.join(SAVE_PATH, f"{IMG_NAME_PREFIX}{time.strftime('%Y%m%d_%H%M%S')}.{image_type}")
            cv2.imwrite(filename, enhanced if PREPROCESS else img_to_save)
            saved_counter += 1

        proc_time_ms = (time.perf_counter() - frame_start) * 1000
        if config.ENABLE_LIVE_STREAM:
            with status_lock:
                status["proc_time_ms"] = proc_time_ms

        # Memory check & cleanup -- matches analyse.py's own periodic gc.collect()
        # + torch.cuda.empty_cache() (see config.CUDA_CACHE_CLEANUP_EVERY_N_FRAMES's
        # comment: without the empty_cache() half, fragmentation from repeated
        # gpu_preprocess_frame() calls -- the same GPU chain analyse.py uses -- builds
        # up until a native CUDA allocation blocks/crashes. gc.collect() alone only
        # reclaims CPU-side Python objects, not PyTorch's cached-but-unused GPU blocks.
        if frame_counter % config.CUDA_CACHE_CLEANUP_EVERY_N_FRAMES == 0:
            gc.collect()
            if preprocess_device is not None and preprocess_device.type == "cuda":
                torch.cuda.empty_cache()

        image.dispose()
        print(f"[Frame {frame_counter}] Saved: {saved_counter}/{ACQUIRE_COUNT} | Time: {time.strftime('%H:%M:%S')} | Proc: {proc_time_ms:.1f} ms | Memory usage: {process.memory_info().rss / 1e6:.1f} MB", end = '\r', flush=True)
        del image, img_array

finally:
    device.stop_acquisition()
    device.dispose()
    system.dispose()

    mem_mb = process.memory_info().rss / 1e6
    print(f"[Stopped at Frame {frame_counter}] With Memory usage: {mem_mb:.1f} MB")

    print("Acquisition stopped, resources released.")
