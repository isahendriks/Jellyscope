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
from pathlib import Path

import numpy as np
import tensorrt as trt
import torch

import config

ONNX_DIR = config.PIPELINE_DIR / "trt" / "onnx"
CAL_DIR = config.PIPELINE_DIR / "trt" / "calibration"
ENGINE_DIR = config.PIPELINE_DIR / "trt" / "engines"
ENGINE_DIR.mkdir(parents=True, exist_ok=True)

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

# analyse.py's actual per-batch tile count -- engines are built with an
# optimization profile spanning 1..this so one engine covers every batch
# size analyse.py will ever hand it (including the final, smaller batch).
MAX_BATCH = 8912


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


def build_engine(onnx_path: Path, engine_path: Path, precision: str, calibrator=None, max_batch: int = MAX_BATCH):
    builder = trt.Builder(TRT_LOGGER)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, TRT_LOGGER)

    with open(onnx_path, "rb") as f:
        if not parser.parse(f.read()):
            for i in range(parser.num_errors):
                print(parser.get_error(i))
            raise RuntimeError(f"Failed to parse {onnx_path}")

    build_config = builder.create_builder_config()
    build_config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 2 << 30)  # 2 GiB

    profile = builder.create_optimization_profile()
    for i in range(network.num_inputs):
        inp = network.get_input(i)
        shape = list(inp.shape)
        min_shape = [1 if d == -1 else d for d in shape]
        opt_shape = [max(1, max_batch // 4) if d == -1 else d for d in shape]
        max_shape = [max_batch if d == -1 else d for d in shape]
        profile.set_shape(inp.name, min_shape, opt_shape, max_shape)
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
    print(f"  -> wrote {engine_path} ({len(serialized) / 1e6:.1f} MB)")


def main(skip_fp16: bool, skip_int8: bool):
    seg_tiles = np.load(CAL_DIR / "seg_tiles.npy")
    seg_rows = np.load(CAL_DIR / "seg_rows.npy")
    seg_cols = np.load(CAL_DIR / "seg_cols.npy")
    seg_mu = np.load(CAL_DIR / "seg_mu.npy")
    seg_recon_err = np.load(CAL_DIR / "seg_recon_err.npy")

    jobs = [
        ("seg_encoder", [seg_tiles, seg_rows, seg_cols]),
        ("seg_decoder", [seg_mu]),
        ("seg_scorer", [seg_mu, seg_recon_err, seg_rows, seg_cols]),
    ]

    vit_crops_path = CAL_DIR / "vit_crops.npy"
    if vit_crops_path.exists():
        vit_crops = np.load(vit_crops_path)
        vit_sizes = np.load(CAL_DIR / "vit_sizes.npy")
        jobs.append(("vit_classifier", [vit_crops, vit_sizes]))
    else:
        print(f"Skipping vit_classifier (no {vit_crops_path} -- see collect_calibration_data.py)")

    for name, cal_arrays in jobs:
        onnx_path = ONNX_DIR / f"{name}.onnx"
        if not onnx_path.exists():
            print(f"Skipping {name}: {onnx_path} not found (run export_onnx.py first)")
            continue

        if not skip_fp16:
            print(f"Building {name} FP16 engine...")
            build_engine(onnx_path, ENGINE_DIR / f"{name}_fp16.engine", "fp16")

        if not skip_int8:
            print(f"Building {name} INT8 engine (calibrating on {cal_arrays[0].shape[0]} real samples)...")
            calibrator = NpyCalibrator(cal_arrays, CAL_DIR / f"{name}_calibration.cache")
            build_engine(onnx_path, ENGINE_DIR / f"{name}_int8.engine", "int8", calibrator=calibrator)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-fp16", action="store_true")
    parser.add_argument("--skip-int8", action="store_true")
    args = parser.parse_args()
    main(args.skip_fp16, args.skip_int8)
