"""GPU-accelerated port of control/monitoring/record.py's CPU preprocessing chain
(dark-frame subtract -> median blur -> gamma -> CLAHE), via kornia + torch.

Dark-frame subtract/clip and the median blur run on CPU (plain numpy + cv2) --
measured kornia.filters.median_blur at ~1.7s for a 3x3 kernel on a real
4512x4512 frame (it's unfold+sort based, not a native optimized median filter),
against ~10-20ms for cv2.medianBlur doing the exact same operation on CPU. Only
gamma + CLAHE (both legitimately cheap and correctly GPU-accelerated, ~70ms
total measured) still round-trip through CUDA -- one host->device upload after
the CPU steps, one device->host copy at the very end.

Note: kornia's CLAHE (equalize_clahe) and skimage's equalize_adapthist (used by
the original CPU chain) are different implementations of adaptive histogram
equalization. Validate the GPU output visually against a known-good skimage
result during testing (see monitoring's verification plan) -- don't
assume bit-identical output.
"""

import cv2
import kornia
import numpy as np
import torch


def rotate_frame(img: np.ndarray, angle_deg: float) -> np.ndarray:
    """Rotates a square frame around its center by angle_deg (positive = counter-
    clockwise, matching cv2.getRotationMatrix2D's convention). Multiples of 90 use
    cv2.rotate (exact -- no interpolation, no border artifacts, no dtype/precision
    loss); other angles fall back to a warpAffine, which does interpolate and
    replicate-pads whatever corners the rotation reveals."""
    angle_deg = angle_deg % 360
    if angle_deg == 0:
        return img
    if angle_deg == 90:
        return cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
    if angle_deg == 180:
        return cv2.rotate(img, cv2.ROTATE_180)
    if angle_deg == 270:
        return cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)

    h, w = img.shape[:2]
    rotation_matrix = cv2.getRotationMatrix2D((w / 2, h / 2), angle_deg, 1.0)
    return cv2.warpAffine(img, rotation_matrix, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)


def load_dark_frame(path: str, device: torch.device, rotate_angle_deg: float = 0.0) -> np.ndarray:
    """Load the dark/background frame once at process startup and keep it
    resident as a plain CPU array -- dark-frame subtract now happens on CPU
    (see gpu_preprocess_frame), so it no longer needs to live on the GPU.
    `device` is accepted but unused; kept so call sites don't need to change.
    Rotated once here by the same angle gpu_preprocess_frame applies to each live
    frame, so it stays aligned with what it gets subtracted from."""
    dark_frame = cv2.imread(path, cv2.IMREAD_UNCHANGED).astype(np.float32)
    return rotate_frame(dark_frame, rotate_angle_deg)


def gpu_preprocess_frame(
    img_array_u16: np.ndarray,
    dark_frame: np.ndarray | None,
    device: torch.device,
    hdr_max: int,
    median_kernel: int,
    gamma: float,
    post_gain: float,
    clahe_clip: float,
    clahe_grid: tuple[int, int],
    rotate_angle_deg: float = 0.0,
) -> np.ndarray:
    """Mirrors control/monitoring/record.py lines ~144-190:
      np.subtract + np.clip        -> stays on CPU (numpy)
      cv2.medianBlur               -> stays on CPU (cv2.medianBlur) -- see module docstring
      np.power(..., GAMMA)*POST_GAIN -> torch.pow (trivial, on GPU)
      skimage.equalize_adapthist   -> kornia.enhance.equalize_clahe (on GPU)
    Returns an 8-bit numpy array ready for cv2.imwrite / segmentation.
    """
    img = img_array_u16.astype(np.float32)
    if rotate_angle_deg:
        img = rotate_frame(img, rotate_angle_deg)
    if dark_frame is not None:
        img = img - dark_frame
    img = np.clip(img, 1.0, hdr_max)
    img = cv2.medianBlur(img, median_kernel)

    x = torch.from_numpy(img).unsqueeze(0).unsqueeze(0).to(device, non_blocking=True)  # (1,1,H,W)

    x = x / hdr_max
    x = torch.clamp(torch.pow(x, gamma) * post_gain, 0.0, 1.0)

    x = kornia.enhance.equalize_clahe(x, clip_limit=clahe_clip, grid_size=clahe_grid)
    x = torch.clamp(x, 0.0, 1.0)

    x_u8 = (x * 255).round().clamp(0, 255).to(torch.uint8)
    return x_u8.squeeze(0).squeeze(0).cpu().numpy()  # single device->host copy
