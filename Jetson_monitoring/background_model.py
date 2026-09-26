"""EMA background model for biofouling suppression.

Owns a single float32 buffer of the slowly-adapting per-pixel background estimate, as
one process-lifetime module-level global (same idiom record.py's own globals like
frame_counter use) -- there's only ever one buffer per process, no need for the
ceremony of a class/instance.

Runs on the RAW (pre-PREPROCESS) 12-bit-in-16-bit frame, upstream of
gpu_preprocess.gpu_preprocess_frame's dark-frame-subtract/median-blur/gamma/CLAHE
chain -- see analyse.py's process_frame() call site. Complementary to, not a
replacement for, gpu_preprocess.load_dark_frame's static dark_frame: that corrects
fixed sensor pattern noise; this corrects slowly-accumulating biofouling on the
camera window, which a static reference frame can't track since it changes gradually
between the ~twice-weekly window cleanings.

EMA chosen over a windowed running mean (needs O(N) frame memory) or a rolling
median (best "snap" recovery after a cleaning, but no cheap O(1) per-pixel update --
real cost on a Jetson): O(1) memory (one frame-sized buffer, not a deque of frames),
O(1) update per frame, and the "ghosting" after a cleaning event (background still
carrying old fouling values) decays fast enough at typical alphas not to matter for
this use case -- no explicit cleaning-event detection/reset needed.
"""

import numpy as np

_bg = None  # float32 buffer, lazily seeded from the first frame this process sees


def update_and_subtract(frame: np.ndarray, alpha: float, max_val: float) -> np.ndarray:
    """Updates the EMA background estimate with `frame` (bg = alpha*frame +
    (1-alpha)*bg) and returns frame - bg clipped to [0, max_val] and cast back to
    frame's own dtype -- the exact same "subtract, clip negatives to 0" pattern
    gpu_preprocess.gpu_preprocess_frame's own static dark_frame subtraction already
    uses (max_val is meant to be config.HDR_MAX, same reference that clip uses).

    2026-08-14: an earlier version re-centered the diff around a mid-grey `offset`
    (frame - bg + offset, clipped to [0, 2*offset]) instead of clipping straight to
    0. That shipped, then immediately made SEGMENT flag entire live frames as one
    giant detection. Root-caused empirically (see background_model prototyping
    session, not alpha or spatial-frequency as first suspected -- both were tested
    and *didn't* fix it): the AE was trained on raw camera frames, which are
    naturally dark/near-zero almost everywhere with sparse bright specks. Recentering
    to mid-grey shifts *every* pixel's typical value away from that distribution
    regardless of alpha or how bg is computed, which is what CLAHE + the AE choked
    on. Plain clip-to-0 preserves the same "mostly near-zero, sparse bright content"
    shape the raw frames already have -- frame≈bg (the common case) still lands near
    0, same as it always did, so CLAHE and the AE see something close to their native
    input distribution. Confirmed against the real INT8 TensorRT encoder/decoder +
    Mahalanobis scorer on real captured frames: recentered ⇒ 100% frame coverage
    every time regardless of alpha/blur; clip-to-0 ⇒ normal small-coverage detections,
    consistently across a multi-frame run."""
    global _bg
    frame_f = frame.astype(np.float32)
    if _bg is None:
        _bg = frame_f.copy()  # seed from the first frame, not zeros -- avoids a fake
        # "cleaning event" ramp-up (the whole frame reading as one giant
        # fouling-removal difference) on every process startup/restart
    else:
        _bg = alpha * frame_f + (1 - alpha) * _bg

    corrected = np.clip(frame_f - _bg, 0, max_val)
    return corrected.astype(frame.dtype)


def get_background() -> np.ndarray | None:
    """Current EMA background estimate (float32), or None before the first frame --
    exposed for the live-stream's biofouling-model debug thumbnails."""
    return _bg