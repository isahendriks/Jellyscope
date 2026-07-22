"""GPU-accelerated port of control/monitoring/record.py's CPU preprocessing chain
(dark-frame subtract -> median blur -> gamma -> CLAHE), via kornia + torch.

Everything after the initial host->device copy stays on the CUDA tensor; the
only device->host copy happens once, at the very end -- unlike the CPU version,
there's no per-step round trip.

Note: kornia's CLAHE (equalize_clahe) and skimage's equalize_adapthist (used by
the original CPU chain) are different implementations of adaptive histogram
equalization. Validate the GPU output visually against a known-good skimage
result during testing (see Jetson_control_2's verification plan) -- don't
assume bit-identical output.
"""

import kornia
import numpy as np
import torch


def load_dark_frame(path: str, device: torch.device) -> torch.Tensor:
    """Load the dark/background frame once at process startup and keep it
    resident on the GPU, so it isn't re-uploaded every frame."""
    import cv2
    dark_frame = cv2.imread(path, cv2.IMREAD_UNCHANGED).astype(np.float32)
    return torch.from_numpy(dark_frame).unsqueeze(0).unsqueeze(0).to(device)


def gpu_preprocess_frame(
    img_array_u16: np.ndarray,
    dark_frame_gpu: torch.Tensor | None,
    device: torch.device,
    hdr_max: int,
    median_kernel: int,
    gamma: float,
    post_gain: float,
    clahe_clip: float,
    clahe_grid: tuple[int, int],
) -> np.ndarray:
    """Mirrors control/monitoring/record.py lines ~144-190:
      np.subtract + np.clip        -> torch.clamp (after subtracting dark_frame_gpu)
      cv2.medianBlur               -> kornia.filters.median_blur
      np.power(..., GAMMA)*POST_GAIN -> torch.pow (trivial, stays on GPU)
      skimage.equalize_adapthist   -> kornia.enhance.equalize_clahe
    Returns an 8-bit numpy array ready for cv2.imwrite / segmentation.
    """
    x = torch.from_numpy(img_array_u16.astype(np.float32)).unsqueeze(0).unsqueeze(0).to(device, non_blocking=True)  # (1,1,H,W)

    if dark_frame_gpu is not None:
        x = x - dark_frame_gpu
    x = torch.clamp(x, 1.0, hdr_max)

    x = kornia.filters.median_blur(x, (median_kernel, median_kernel))

    x = x / hdr_max
    x = torch.clamp(torch.pow(x, gamma) * post_gain, 0.0, 1.0)

    x = kornia.enhance.equalize_clahe(x, clip_limit=clahe_clip, grid_size=clahe_grid)
    x = torch.clamp(x, 0.0, 1.0)

    x_u8 = (x * 255).round().clamp(0, 255).to(torch.uint8)
    return x_u8.squeeze(0).squeeze(0).cpu().numpy()  # single device->host copy
