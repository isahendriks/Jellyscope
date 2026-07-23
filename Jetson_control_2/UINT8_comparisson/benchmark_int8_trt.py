"""Benchmark PyTorch FP32/FP16 vs TensorRT FP16/INT8 for both SEGMENT (AE
encode -> decode -> scorer) and CLASSIFY (ViT) -- extends
benchmark_segment_inference.py's FP16-only comparison with the TensorRT
engines build_trt_int8.py produces.

Two things get measured, not just one:
  1. Speed -- per-batch wall time for each backend/precision.
  2. Accuracy drift -- INT8 can be fast and silently wrong for a small
     anomaly-scoring network like this one. Compares TensorRT INT8 output
     against the PyTorch FP32 reference on the SAME real calibration data
     (not synthetic random tiles, so this reflects the actual input
     distribution) via max-abs-diff, and for the classifier, top-1 class
     agreement.

Uses the real tiles/crops from collect_calibration_data.py for both parts --
random tiles would make the accuracy comparison meaningless, and would also
not exercise INT8 the way it was actually calibrated.

Not yet run against a real TensorRT install -- see build_trt_int8.py's
docstring for the same API-version caveat; TRTModule below uses the
TensorRT 10.x execute_async_v3 + set_tensor_address API.

Usage: python benchmark_int8_trt.py
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # Jetson_control_2/

import numpy as np
import tensorrt as trt
import torch

import config
from models import segmentation as seg_models
from models import vit_classifier

CAL_DIR = config.PIPELINE_DIR / "trt" / "calibration"
ENGINE_DIR = config.PIPELINE_DIR / "trt" / "engines"

device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

N_WARMUP = 3
N_TIMED = 10


class TRTModule:
    """Minimal TensorRT engine runner using torch CUDA tensors for I/O (no
    pycuda) -- same zero-copy-via-torch approach as build_trt_int8.py's
    calibrator."""

    def __init__(self, engine_path: Path):
        runtime = trt.Runtime(TRT_LOGGER)
        self.engine = runtime.deserialize_cuda_engine(engine_path.read_bytes())
        self.context = self.engine.create_execution_context()
        self.stream = torch.cuda.Stream()
        names = [self.engine.get_tensor_name(i) for i in range(self.engine.num_io_tensors)]
        self.input_names = [n for n in names if self.engine.get_tensor_mode(n) == trt.TensorIOMode.INPUT]
        self.output_names = [n for n in names if self.engine.get_tensor_mode(n) == trt.TensorIOMode.OUTPUT]

    def __call__(self, *inputs):
        assert len(inputs) == len(self.input_names), (
            f"expected {len(self.input_names)} inputs ({self.input_names}), got {len(inputs)}"
        )
        keep_alive = []
        for name, tensor in zip(self.input_names, inputs):
            tensor = tensor.contiguous().to(device=device, dtype=torch.float32)
            keep_alive.append(tensor)
            self.context.set_input_shape(name, tuple(tensor.shape))
            self.context.set_tensor_address(name, int(tensor.data_ptr()))

        outputs = []
        for name in self.output_names:
            shape = tuple(self.context.get_tensor_shape(name))
            out = torch.empty(shape, dtype=torch.float32, device=device)
            keep_alive.append(out)
            self.context.set_tensor_address(name, int(out.data_ptr()))
            outputs.append(out)

        with torch.cuda.stream(self.stream):
            self.context.execute_async_v3(self.stream.cuda_stream)
        self.stream.synchronize()
        return outputs[0] if len(outputs) == 1 else tuple(outputs)


def timeit(fn) -> float:
    for _ in range(N_WARMUP):
        fn()
    if device.type == "cuda":
        torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(N_TIMED):
        fn()
    if device.type == "cuda":
        torch.cuda.synchronize()
    return (time.perf_counter() - start) / N_TIMED


def max_abs_diff(a: torch.Tensor, b: torch.Tensor) -> float:
    return (a.float() - b.float()).abs().max().item()


def benchmark_segment():
    print("=" * 70)
    print("SEGMENT: AE encode -> decode -> scorer")
    print("=" * 70)

    seg_tiles = torch.from_numpy(np.load(CAL_DIR / "seg_tiles.npy")).to(device)
    seg_rows = torch.from_numpy(np.load(CAL_DIR / "seg_rows.npy")).to(device)
    seg_cols = torch.from_numpy(np.load(CAL_DIR / "seg_cols.npy")).to(device)

    # Must match build_trt_int8.py's SEG_BATCH exactly -- the TRT engines were
    # built with a fixed (not dynamic) input shape, so they only accept this
    # one batch size. 1280 is analyse.py's actual per-image tile count (5
    # offsets x 16x16 grid), not the 8912 split-threshold (which never
    # actually triggers in production since 1280 < 8912).
    batch_size = min(1280, seg_tiles.shape[0])
    tiles, rows, cols = seg_tiles[:batch_size], seg_rows[:batch_size], seg_cols[:batch_size]
    print(f"Using a batch of {tiles.shape[0]} real tiles (analyse.py's actual per-image tile count)")

    print("\nLoading PyTorch segmentation models...")
    seg_model, scorer, scorer_threshold, seg_grid_size, seg_image_size = seg_models.load_segmentation_models(
        config.SEGMENTATION_AE_MODEL_PATH, config.SEGMENTATION_SCORER_MODEL_PATH,
        config.SEGMENTATION_ENCODER_TYPE, device,
    )

    def torch_segment(t, r, c, model, scorer, dtype):
        with torch.inference_mode():
            t, r, c = t.to(dtype), r.to(dtype), c.to(dtype)
            mu = seg_models.encode(model, config.SEGMENTATION_ENCODER_TYPE, t, r, c)
            x_hat = model.decode(mu)
            recon_err = ((t - x_hat) ** 2).flatten(1).mean(dim=1)
            return scorer.predict_batch(mu, recon_err, r, c)

    print("Timing PyTorch FP32...")
    t_fp32 = timeit(lambda: torch_segment(tiles, rows, cols, seg_model, scorer, torch.float32))
    scores_fp32 = torch_segment(tiles, rows, cols, seg_model, scorer, torch.float32)

    print("Timing PyTorch FP16...")
    # NOTE: .half() mutates seg_model/scorer's weights in place (matches
    # benchmark_segment_inference.py) -- scores_fp32 above was already
    # captured, so this is safe, but don't reuse seg_model as "the fp32
    # model" after this line.
    seg_model.half()
    scorer.half()
    t_fp16 = timeit(lambda: torch_segment(tiles, rows, cols, seg_model, scorer, torch.float16))

    results = [("pytorch_fp32", t_fp32, 0.0), ("pytorch_fp16", t_fp16, None)]

    engine_names = ["seg_encoder", "seg_decoder", "seg_scorer"]
    for label, suffix in [("trt_fp16", "fp16"), ("trt_int8", "int8")]:
        paths = [ENGINE_DIR / f"{n}_{suffix}.engine" for n in engine_names]
        if not all(p.exists() for p in paths):
            print(f"Skipping {label}: engines not found under {ENGINE_DIR} (run build_trt_int8.py)")
            continue

        print(f"Loading {label} engines...")
        trt_encoder, trt_decoder, trt_scorer = (TRTModule(p) for p in paths)

        def trt_segment():
            mu = trt_encoder(tiles, rows, cols)
            x_hat = trt_decoder(mu)
            recon_err = ((tiles - x_hat) ** 2).flatten(1).mean(dim=1)
            return trt_scorer(mu, recon_err, rows, cols)

        print(f"Timing {label}...")
        t = timeit(trt_segment)
        scores_trt = trt_segment()
        diff = max_abs_diff(scores_trt, scores_fp32)
        results.append((label, t, diff))

    print(f"\n{'backend':<16} {'per-batch (s)':>14} {'speedup vs fp32':>18} {'max|Δscore| vs fp32':>22}")
    baseline = results[0][1]
    for label, t, diff in results:
        diff_str = f"{diff:.4f}" if diff is not None else "n/a"
        print(f"{label:<16} {t:>14.4f} {baseline / t:>17.2f}x {diff_str:>22}")


def benchmark_classify():
    print("\n" + "=" * 70)
    print("CLASSIFY: ViT")
    print("=" * 70)

    vit_crops_path = CAL_DIR / "vit_crops.npy"
    if not vit_crops_path.exists():
        print(f"Skipping: {vit_crops_path} not found (see collect_calibration_data.py)")
        return

    vit_crops = torch.from_numpy(np.load(vit_crops_path)).to(device)
    vit_sizes = torch.from_numpy(np.load(CAL_DIR / "vit_sizes.npy")).to(device)
    print(f"Using {vit_crops.shape[0]} real calibration crops")

    print("Loading PyTorch ViT classifier...")
    classifier, idx_to_class = vit_classifier.load_classifier(
        config.VIT_CHECKPOINT_PATH, device, class_names_fallback=config.VIT_CLASS_NAMES_FALLBACK,
    )

    def torch_classify(img, size, model, dtype):
        with torch.inference_mode():
            return model(img.to(dtype), size.to(dtype))

    print("Timing PyTorch FP32...")
    t_fp32 = timeit(lambda: torch_classify(vit_crops, vit_sizes, classifier, torch.float32))
    logits_fp32 = torch_classify(vit_crops, vit_sizes, classifier, torch.float32)
    preds_fp32 = logits_fp32.argmax(dim=1)

    print("Timing PyTorch FP16...")
    classifier.half()  # mutates in place, same caveat as benchmark_segment()
    t_fp16 = timeit(lambda: torch_classify(vit_crops, vit_sizes, classifier, torch.float16))

    results = [("pytorch_fp32", t_fp32, 0.0, 1.0), ("pytorch_fp16", t_fp16, None, None)]

    for label, suffix in [("trt_fp16", "fp16"), ("trt_int8", "int8")]:
        engine_path = ENGINE_DIR / f"vit_classifier_{suffix}.engine"
        if not engine_path.exists():
            print(f"Skipping {label}: {engine_path} not found")
            continue
        trt_classifier = TRTModule(engine_path)
        t = timeit(lambda: trt_classifier(vit_crops, vit_sizes))
        logits_trt = trt_classifier(vit_crops, vit_sizes)
        diff = max_abs_diff(logits_trt, logits_fp32)
        agreement = (logits_trt.argmax(dim=1) == preds_fp32).float().mean().item()
        results.append((label, t, diff, agreement))

    print(f"\n{'backend':<16} {'per-batch (s)':>14} {'speedup vs fp32':>18} {'max|Δlogit|':>14} {'top1 agreement':>16}")
    baseline = results[0][1]
    for label, t, diff, agree in results:
        diff_str = f"{diff:.4f}" if diff is not None else "n/a"
        agree_str = f"{agree * 100:.1f}%" if agree is not None else "n/a"
        print(f"{label:<16} {t:>14.4f} {baseline / t:>17.2f}x {diff_str:>14} {agree_str:>16}")


if __name__ == "__main__":
    benchmark_segment()
    benchmark_classify()
