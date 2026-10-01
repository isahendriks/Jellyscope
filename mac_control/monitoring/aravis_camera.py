### Camera access for macOS. The Itala SDK only exists for Windows/Linux, so on the Mac the
### camera (a standard GigE Vision device) is driven with Aravis instead (brew install aravis).
### Same camera settings as windows_control/monitoring/record.py.

import gi
gi.require_version("Aravis", "0.8")
from gi.repository import Aravis
import numpy as np

N_BUFFERS = 10 # frames Aravis can queue before we read them


def open_camera(gain_db, packet_size_target=1400):
    """Finds the first camera, applies the trigger/gain settings and returns (camera, stream)."""
    Aravis.update_device_list()
    n_devices = Aravis.get_n_devices()
    if n_devices == 0:
        print("No devices found. Exiting.")
        exit(1)
    print(f"Found {Aravis.get_device_id(0)} at {Aravis.get_device_address(0)}")

    try:
        camera = Aravis.Camera.new(Aravis.get_device_id(0))
    except Exception as e: # Aravis raises when another program holds the camera (no RW access)
        print(f"Device not accessible: {e}. Exiting.")
        exit(1)
    print("Device initialized.")

    camera.set_string("TriggerSelector", "FrameBurstStart")
    camera.set_string("TriggerSource", "Line0")
    camera.set_string("LineSelector", "Line0")
    camera.set_string("LineMode", "Input")
    camera.set_string("GainAuto", "Off")
    camera.set_float("Gain", gain_db)

    # 1400 works without jumbo frames. Raise only if jumbo frames (MTU 9000) are enabled on the Mac's adapter
    valid_val = packet_size_target - ((packet_size_target - 560) % 8)
    camera.gv_set_packet_size(valid_val)
    camera.set_integer("AcquisitionBurstFrameCount", 1)
    camera.set_string("ExposureMode", "TriggerWidth")
    camera.set_string("TriggerMode", "On")
    camera.set_pixel_format_from_string("Mono12p")

    stream = camera.create_stream(None, None)
    payload = camera.get_payload()
    for _ in range(N_BUFFERS):
        stream.push_buffer(Aravis.Buffer.new_allocate(payload))
    return camera, stream


def unpack_mono12p(data, width, height):
    """GenICam Mono12p: 2 pixels in 3 bytes, LSB first -> uint16 array with values 0..4095."""
    raw = data[: width * height * 3 // 2].reshape(-1, 3).astype(np.uint16)
    out = np.empty(raw.shape[0] * 2, dtype=np.uint16)
    out[0::2] = raw[:, 0] | ((raw[:, 1] & 0x0F) << 8)
    out[1::2] = (raw[:, 1] >> 4) | (raw[:, 2] << 4)
    return out.reshape(height, width)


def get_next_frame(stream, timeout_ms):
    """Returns (status, frame_id, img) with status "ok", "timeout" or "incomplete".
    img is a 12-bit uint16 array, like image.convert(itala.PfncFormat_Mono12) on Windows."""
    buffer = stream.timeout_pop_buffer(int(timeout_ms * 1000)) # Aravis wants microseconds
    if buffer is None:
        return "timeout", None, None
    frame_id = buffer.get_frame_id()
    if buffer.get_status() != Aravis.BufferStatus.SUCCESS:
        stream.push_buffer(buffer)
        return "incomplete", frame_id, None

    width, height = buffer.get_image_width(), buffer.get_image_height()
    data = np.frombuffer(buffer.get_data(), dtype=np.uint8) # get_data() copies, so the buffer can go back right away
    stream.push_buffer(buffer)

    if data.size >= width * height * 2: # camera fell back to an unpacked format (Mono12/Mono16)
        img = data[: width * height * 2].view("<u2").reshape(height, width)
    else:
        img = unpack_mono12p(data, width, height)
    return "ok", frame_id, img


def close_camera(camera):
    camera.stop_acquisition()
