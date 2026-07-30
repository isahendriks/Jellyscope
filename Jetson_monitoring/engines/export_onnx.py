"""Export the SEGMENT (AE encoder / decoder + scorer) and CLASSIFY (ViT)
models to ONNX -- the first step toward building TensorRT INT8 engines.

Part of a 4-script chain for testing whether TensorRT INT8 speeds up the
pipeline (see also collect_calibration_data.py, build_trt_int8.py,
benchmark_int8_trt.py). Run this one first, on the Jetson, with the real
checkpoints loaded via config.py.

seg_models.encode() is a free function (it dispatches AE vs VAE), not a
method on seg_model, so it's wrapped in a tiny nn.Module below purely so
torch.onnx.export has something with a forward() to trace -- encode/decode/
predict_batch themselves are untouched.

Produces, under monitoring/trt/onnx/:
  seg_encoder.onnx     (tiles, rows, cols)             -> mu
  seg_decoder.onnx     (mu)                            -> x_hat
  seg_scorer.onnx      (mu, recon_err, rows, cols)     -> scores
  vit_classifier.onnx  (img_batch, size_batch)         -> logits

Each is exported with a dynamic batch/tile-count axis (name "n") so one
engine can serve any batch size -- analyse.py's batch_size is 8912 tiles,
but an image's last batch is usually smaller.

Not yet run/verified against real checkpoints -- do that on the Jetson and
expect to iterate (ONNX opset vs installed TensorRT version is the most
likely first snag; check `python -c "import tensorrt as trt; print(trt.__version__)"`
and adjust OPSET below if the parser in build_trt_int8.py rejects it).

Usage: python export_onnx.py
"""

import sys
from pathlib import Path

_PIPELINE_DIR = Path(__file__).resolve().parent.parent  # Jetson_monitoring/
for _p in (_PIPELINE_DIR / "Monitor", _PIPELINE_DIR):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import torch
import torch.nn as nn

import config
from models import segmentation as seg_models
from models import vit_classifier

ONNX_DIR = config.PIPELINE_DIR / "trt" / "onnx"
ONNX_DIR.mkdir(parents=True, exist_ok=True)

OPSET = 18  # 17 failed to downgrade for vit_classifier's Resize op (torch.onnx version
            # converter has no opset-17 adapter for it); 18 exports cleanly for all four
            # models, and TensorRT 10.16 supports it fine.
CLASSIFIER_IMAGE_SIZE = 256  # matches analyse.py
N_EXAMPLE = 8  # only fixes dtypes for tracing; exported batch dim is dynamic

device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
print(f"Using device: {device}")

print("Loading segmentation models (AE + BinaryScorer)...")
seg_model, scorer, scorer_threshold, seg_grid_size, seg_image_size = seg_models.load_segmentation_models(
    config.SEGMENTATION_AE_MODEL_PATH, config.SEGMENTATION_SCORER_MODEL_PATH,
    config.SEGMENTATION_ENCODER_TYPE, device,
)
seg_model.eval()
scorer.eval()

print("Loading ViT classifier...")
classifier, idx_to_class = vit_classifier.load_classifier(
    config.VIT_CHECKPOINT_PATH, device, class_names_fallback=config.VIT_CLASS_NAMES_FALLBACK,
)
classifier.eval()


class EncoderWrapper(nn.Module):
    def __init__(self, seg_model, encoder_type):
        super().__init__()
        self.seg_model = seg_model
        self.encoder_type = encoder_type

    def forward(self, tiles, rows, cols):
        return seg_models.encode(self.seg_model, self.encoder_type, tiles, rows, cols)


class DecoderWrapper(nn.Module):
    def __init__(self, seg_model):
        super().__init__()
        self.seg_model = seg_model

    def forward(self, mu):
        return self.seg_model.decode(mu)


class ScorerWrapper(nn.Module):
    def __init__(self, scorer):
        super().__init__()
        self.scorer = scorer

    def forward(self, mu, recon_err, rows, cols):
        return self.scorer.predict_batch(mu, recon_err, rows, cols)


def export(module, args, path, input_names, output_names, dynamic_axes):
    module.eval()
    with torch.inference_mode():
        torch.onnx.export(
            module, args, str(path),
            input_names=input_names, output_names=output_names,
            dynamic_axes=dynamic_axes, opset_version=OPSET, do_constant_folding=True,
        )
    print(f"  -> wrote {path}")


print("\nExporting SEGMENT encoder...")
tiles = torch.rand(N_EXAMPLE, 1, seg_image_size, seg_image_size, device=device)
rows = torch.rand(N_EXAMPLE, device=device) * (seg_grid_size - 1)
cols = torch.rand(N_EXAMPLE, device=device) * (seg_grid_size - 1)
encoder_wrap = EncoderWrapper(seg_model, config.SEGMENTATION_ENCODER_TYPE).to(device)
with torch.inference_mode():
    mu_example = encoder_wrap(tiles, rows, cols)
export(
    encoder_wrap, (tiles, rows, cols), ONNX_DIR / "seg_encoder.onnx",
    ["tiles", "rows", "cols"], ["mu"],
    {"tiles": {0: "n"}, "rows": {0: "n"}, "cols": {0: "n"}, "mu": {0: "n"}},
)

print("Exporting SEGMENT decoder...")
decoder_wrap = DecoderWrapper(seg_model).to(device)
with torch.inference_mode():
    x_hat_example = decoder_wrap(mu_example)
export(
    decoder_wrap, (mu_example,), ONNX_DIR / "seg_decoder.onnx",
    ["mu"], ["x_hat"],
    {"mu": {0: "n"}, "x_hat": {0: "n"}},
)

print("Exporting SEGMENT scorer...")
recon_err_example = ((tiles - x_hat_example) ** 2).flatten(1).mean(dim=1)
scorer_wrap = ScorerWrapper(scorer).to(device)
export(
    scorer_wrap, (mu_example, recon_err_example, rows, cols), ONNX_DIR / "seg_scorer.onnx",
    ["mu", "recon_err", "rows", "cols"], ["scores"],
    {"mu": {0: "n"}, "recon_err": {0: "n"}, "rows": {0: "n"}, "cols": {0: "n"}, "scores": {0: "n"}},
)

print("Exporting CLASSIFY (ViT)...")
img_batch = torch.rand(N_EXAMPLE, 1, CLASSIFIER_IMAGE_SIZE, CLASSIFIER_IMAGE_SIZE, device=device)
size_batch = torch.rand(N_EXAMPLE, 1, device=device)
export(
    classifier, (img_batch, size_batch), ONNX_DIR / "vit_classifier.onnx",
    ["img_batch", "size_batch"], ["logits"],
    {"img_batch": {0: "n"}, "size_batch": {0: "n"}, "logits": {0: "n"}},
)

print(f"\nDone. ONNX files in {ONNX_DIR}")
print("Next: python collect_calibration_data.py, then python build_trt_int8.py")
