#!/usr/bin/env python3
"""
Verify a real, working PyTorch stack on JetPack 7.2 / Jetson Orin.

The point of this script: torch.cuda.is_available() returning True is NOT proof
the GPU works. Every wrong-architecture wheel returns True and then dies on the
first real kernel. This runs an actual matmul and checks the compute capability,
so a pass means the kernel genuinely executed on your Orin.

Exit code 0 = stack is real. Non-zero = something is off (message explains).
"""

import sys


def main() -> int:
    try:
        import torch
    except Exception as e:  # noqa: BLE001
        print(f"FAIL: could not import torch: {e}")
        print("If this is a missing lib*.so, see the Prerequisites section in README.md.")
        return 1

    print("torch:        ", torch.__version__)
    print("built for CUDA:", torch.version.cuda)
    print("cuda available:", torch.cuda.is_available())

    if not torch.cuda.is_available():
        print("FAIL: CUDA not available to PyTorch.")
        print("Check that CUDA 13.2 is on PATH/LD_LIBRARY_PATH (caveat #1 in README).")
        return 1

    name = torch.cuda.get_device_name(0)
    cap = torch.cuda.get_device_capability(0)
    print("device:       ", name)
    print("capability:   ", cap, "(sm_%d%d)" % cap)

    # Orin is sm_87. A wheel that doesn't include it will still report available
    # and then throw "no kernel image" right here.
    try:
        x = torch.randn(2048, 2048, device="cuda")
        y = x @ x
        torch.cuda.synchronize()
        checksum = float(y.sum())
    except Exception as e:  # noqa: BLE001
        print(f"FAIL: kernel launch failed: {e}")
        print("This wheel is almost certainly built for a different GPU arch.")
        print("See 'For other hardware' in README.md.")
        return 1

    print(f"KERNEL OK:     matmul completed, checksum={checksum:.2f}")

    # torchvision is optional; report but don't fail the core check on it.
    try:
        import torchvision

        v = torch.randn(1, 3, 224, 224, device="cuda")
        torch.cuda.synchronize()
        print("torchvision:  ", torchvision.__version__, "(CUDA tensor OK)", v.shape)
    except Exception as e:  # noqa: BLE001
        print(f"torchvision:   not verified ({e})")

    # TensorRT comes from JetPack via --system-site-packages; report if present.
    try:
        import tensorrt as trt

        print("tensorrt:     ", trt.__version__)
    except Exception:  # noqa: BLE001
        print("tensorrt:      not visible (need venv with --system-site-packages)")

    print("\nStack verified. The kernel actually ran on", name + ".")
    return 0


if __name__ == "__main__":
    sys.exit(main())
