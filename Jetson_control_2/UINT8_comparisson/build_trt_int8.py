"""Build TensorRT engines (FP16 and INT8) from the ONNX files produced by
export_onnx.py, calibrated on the real tiles/crops collected by
collect_calibration_data.py.

Uses the TensorRT Python API directly (Builder + IInt8EntropyCalibrator2) --
no pycuda dependency. The calibrator hands TensorRT raw CUDA pointers via
torch tensors' .data_ptr(), since torch already owns a CUDA context here;
this avoids adding a second CUDA memory manager just for calibration.

Produces, under Jetson_control_2/trt/engines/:
  seg_encoder_fp16.engine      seg_encoder_int8.engine
  seg_decoder_fp16.engine      seg_decoder_int8.engine
  seg_scorer_fp16.engine       seg_scorer_int8.engine
  vit_classifier_fp16.engine   vit_classifier_int8.engine   (only if
                                                              vit_crops.npy
                                                              exists)

Not yet run against a real TensorRT install -- the TensorRT Python API has
moved around across versions (execute_v2 vs execute_async_v3, implicit vs
explicit batch, set_memory_pool_limit vs max_workspace_size). Check
`python -c "import tensorrt as trt; print(trt.__version__)"` on the Jetson
first; this was written against the TensorRT 10.x API, which is what
JetPack 7.2 (per Jetson_control/verify_stack.py) should ship.

Usage: python build_trt_int8.py [--skip-fp16] [--skip-int8]
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # Jetson_control_2/

import numpy as np
import tensorrt as trt
import torch

import config

ONNX_DIR = config.PIPELINE_DIR / "trt" / "onnx"
CAL_DIR = config.PIPELINE_DIR / "trt" / "calibration"
ENGINE_DIR = config.PIPELINE_DIR / "trt" / "engines"
ENGINE_DIR.mkdir(parents=True, exist_ok=True)

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

# Fixed (not dynamic-range) batch shapes -- a dynamic profile (e.g. 1..8912)
# makes TensorRT tactic-search across the whole range, which is what made the
# original build take 15+ minutes without finishing even the first engine.
# Fixing min==opt==max cuts that to a single shape's worth of search.
#
# Both pulled from config.py so analyse.py, segmentation_trt.py/vit_classifier_trt.py,
# and this build script all agree on the same fixed shapes.
# SEG_BATCH=1280 is analyse.py's *actual* per-image tile count (5 offsets x
# 16x16 grid) -- since 1280 < the old dynamic-range's 8912 split threshold, that
# threshold never actually triggers in production, so 1280 is both the
# realistic shape to build for and much faster than 8912 would be.
# VIT_BATCH=16 is analyse.py's per-image crop batch (padded up from the real,
# variable crop count -- observed 0-9 in sample data) -- NOT 146 (the
# calibration-set size), which was the original build's mismatch: it made the
# engine unusable for real per-image crop counts. See models/vit_classifier_trt.py.
SEG_BATCH = config.SEG_ENGINE_BATCH
VIT_BATCH = config.VIT_ENGINE_BATCH

# Calibration needs whole batches at exactly the fixed shape above -- a
# calibration set of, say, 4096 samples can't feed a single 8912-shaped batch
# at all, so subsampling has to stay a clean multiple of the batch size, not
# an arbitrary smaller number.
#
# Originally 4 (5120 of 38400 tiles) to hit a ~30min build budget. Bumped to use
# ALL 38400 collected tiles (30 batches) after test_analyse_on_folder.py showed a
# real accuracy regression at the smaller calibration size -- INT8 SEGMENT produced
# 55 crops vs FP32's 226 across the same 48 sample images, not just noise near the
# scorer_threshold=0.3004 decision boundary. More calibration data is the direct fix
# for exactly this kind of range-estimation error.
SEG_CALIBRATION_BATCHES = 30


class NpyCalibrator(trt.IInt8EntropyCalibrator2):
    """Feeds batches of real (not synthetic) arrays to TensorRT for INT8
    range calibration. `input_arrays` is a list of np.ndarray, one per ONNX
    input in order, all sharing dim-0 length (the calibration set size)."""

    def __init__(self, input_arrays, cache_path: Path, batch_size: int = 512):
        super().__init__()
        self.arrays = [np.ascontiguousarray(a, dtype=np.float32) for a in input_arrays]
        self.n = self.arrays[0].shape[0]
        self.batch_size = min(batch_size, self.n)
        self.cache_path = cache_path
        self.idx = 0
        # Keep device buffers alive for the calibrator's lifetime -- TensorRT
        # only holds onto the pointers, not the tensors.
        self.device_buffers = [
            torch.empty((self.batch_size,) + a.shape[1:], dtype=torch.float32, device="cuda")
            for a in self.arrays
        ]

    def get_batch_size(self):
        return self.batch_size

    def get_batch(self, names):
        if self.idx + self.batch_size > self.n:
            return None
        ptrs = []
        for arr, buf in zip(self.arrays, self.device_buffers):
            chunk = arr[self.idx:self.idx + self.batch_size]
            buf.copy_(torch.from_numpy(chunk))
            ptrs.append(int(buf.data_ptr()))
        self.idx += self.batch_size
        return ptrs

    def read_calibration_cache(self):
        if self.cache_path.exists():
            return self.cache_path.read_bytes()
        return None

    def write_calibration_cache(self, cache):
        self.cache_path.write_bytes(cache)


def build_engine(onnx_path: Path, engine_path: Path, precision: str, calibrator=None, max_batch: int = SEG_BATCH):
    builder = trt.Builder(TRT_LOGGER)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, TRT_LOGGER)

    # parse_from_file (not parse(bytes)) -- these ONNX files store large weights
    # in a companion .onnx.data file, referenced by a relative path. parse()
    # only gets raw bytes with no directory context, so it can't resolve that
    # reference; parse_from_file() knows the file's actual location and
    # resolves the external data relative to it.
    if not parser.parse_from_file(str(onnx_path)):
        for i in range(parser.num_errors):
            print(parser.get_error(i))
        raise RuntimeError(f"Failed to parse {onnx_path}")

    build_config = builder.create_builder_config()
    build_config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 2 << 30)  # 2 GiB

    profile = builder.create_optimization_profile()
    for i in range(network.num_inputs):
        inp = network.get_input(i)
        shape = list(inp.shape)
        # Fixed shape (min == opt == max) -- see SEG_BATCH/VIT_BATCH comment above.
        fixed_shape = [max_batch if d == -1 else d for d in shape]
        profile.set_shape(inp.name, fixed_shape, fixed_shape, fixed_shape)
    build_config.add_optimization_profile(profile)

    if precision == "fp16":
        build_config.set_flag(trt.BuilderFlag.FP16)
    elif precision == "int8":
        build_config.set_flag(trt.BuilderFlag.FP16)  # let layers INT8 handles poorly fall back to FP16 instead of FP32
        build_config.set_flag(trt.BuilderFlag.INT8)
        build_config.int8_calibrator = calibrator

    serialized = builder.build_serialized_network(network, build_config)
    if serialized is None:
        raise RuntimeError(f"Engine build failed for {onnx_path} ({precision})")
    engine_path.write_bytes(serialized)
    print(f"  -> wrote {engine_path} ({serialized.nbytes / 1e6:.1f} MB)")


def main(skip_fp16: bool, skip_int8: bool):
    seg_tiles = np.load(CAL_DIR / "seg_tiles.npy")
    seg_rows = np.load(CAL_DIR / "seg_rows.npy")
    seg_cols = np.load(CAL_DIR / "seg_cols.npy")
    seg_mu = np.load(CAL_DIR / "seg_mu.npy")
    seg_recon_err = np.load(CAL_DIR / "seg_recon_err.npy")

    # Subsample to a clean multiple of SEG_BATCH -- calibration feeds whole
    # fixed-shape batches, so this must divide evenly, not just be "smaller."
    n_cal = min(SEG_CALIBRATION_BATCHES * SEG_BATCH, seg_tiles.shape[0])
    n_cal -= n_cal % SEG_BATCH
    seg_tiles, seg_rows, seg_cols = seg_tiles[:n_cal], seg_rows[:n_cal], seg_cols[:n_cal]
    seg_mu, seg_recon_err = seg_mu[:n_cal], seg_recon_err[:n_cal]
    print(f"Using {n_cal} of the collected segment tiles for calibration ({n_cal // SEG_BATCH} batch(es) of {SEG_BATCH})")

    jobs = [
        ("seg_encoder", [seg_tiles, seg_rows, seg_cols], SEG_BATCH),
        ("seg_decoder", [seg_mu], SEG_BATCH),
        ("seg_scorer", [seg_mu, seg_recon_err, seg_rows, seg_cols], SEG_BATCH),
    ]

    vit_crops_path = CAL_DIR / "vit_crops.npy"
    if vit_crops_path.exists():
        vit_crops = np.load(vit_crops_path)
        vit_sizes = np.load(CAL_DIR / "vit_sizes.npy")
        jobs.append(("vit_classifier", [vit_crops, vit_sizes], VIT_BATCH))
    else:
        print(f"Skipping vit_classifier (no {vit_crops_path} -- see collect_calibration_data.py)")

    for name, cal_arrays, batch in jobs:
        onnx_path = ONNX_DIR / f"{name}.onnx"
        if not onnx_path.exists():
            print(f"Skipping {name}: {onnx_path} not found (run export_onnx.py first)")
            continue

        fp16_path = ENGINE_DIR / f"{name}_fp16.engine"
        if not skip_fp16:
            if fp16_path.exists():
                print(f"Skipping {name} FP16 engine, already built ({fp16_path})")
            else:
                print(f"Building {name} FP16 engine (fixed batch={batch})...")
                build_engine(onnx_path, fp16_path, "fp16", max_batch=batch)

        int8_path = ENGINE_DIR / f"{name}_int8.engine"
        if not skip_int8:
            if int8_path.exists():
                print(f"Skipping {name} INT8 engine, already built ({int8_path})")
            else:
                print(f"Building {name} INT8 engine (fixed batch={batch}, calibrating on {cal_arrays[0].shape[0]} real samples)...")
                # Calibrator batch size MUST equal the engine's fixed shape --
                # the network's optimization profile only accepts exactly `batch`.
                calibrator = NpyCalibrator(cal_arrays, CAL_DIR / f"{name}_calibration.cache", batch_size=batch)
                build_engine(onnx_path, int8_path, "int8", calibrator=calibrator, max_batch=batch)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-fp16", action="store_true")
    parser.add_argument("--skip-int8", action="store_true")
    args = parser.parse_args()
    main(args.skip_fp16, args.skip_int8)
