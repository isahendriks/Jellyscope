#!/usr/bin/env bash
#
# This is a modified version of the NVIDIA Jetson PyTorch install script foud on:
# It is aimed at installing PyTorch on JetPack 7.2 for Jetson Orin NX 16GB (sm_87), CUDA 13.2, Python 3.12.
# it was last used on 21/07/2026 
# PyTorch on JetPack 7.2 for Jetson Orin NX 16GB (sm_87), CUDA 13.2, Python 3.12.
# The working, undocumented path: prerelease cu132 aarch64 wheels.
#
# Verified on: Seeed reComputer Super, Jetson Orin NX 16GB, JP7.2 (L4T r39.2),
# Ubuntu 24.04, CUDA 13.2, Python 3.12.
#
# Read the four caveats in README.md before running. This is field notes, not
# official guidance. Check your own hardware/CUDA/Python before trusting it.

set -euo pipefail

# NOTE: this creates ~/.venvs/venv-jellyscope. The venv actually used by the
# Jetson_control_2 pipeline (record.py/analyse.py/etc) is a separate, older one
# at Jetson_control/.venv-jellyscope, which predates this script and was never
# recreated with --system-site-packages. Rather than recreate it here (which
# would mean re-downloading torch/torchvision/kornia/itala/etc from scratch),
# that venv was patched directly with a site-packages/*.pth file pointing at
# /usr/lib/python3.12/dist-packages, so it can see system TensorRT too. If you
# ever do rebuild Jetson_control/.venv-jellyscope from scratch, just point VENV
# below at it instead and this script's --system-site-packages handles it
# properly from the start.
VENV="${HOME}/.venvs/venv-jellyscope"
CU_INDEX="https://download.pytorch.org/whl/cu132"

echo "==> Step 0: locate CUDA 13.2"

# Caveat #1: CUDA must be ON PATH, not just installed. Find the right dir.
CUDA_DIR="$(ls -d /usr/local/cuda-13.2 2>/dev/null || true)"
if [ -z "${CUDA_DIR}" ]; then
  echo "Could not find /usr/local/cuda-13.2."
  echo "Available CUDA installs:"
  ls -d /usr/local/cuda* 2>/dev/null || echo "  (none found)"
  echo "Edit CUDA_HOME below to match the 13.2 install, then re-run."
  exit 1
fi

export CUDA_HOME="${CUDA_DIR}"
export PATH="${CUDA_HOME}/bin:${PATH}"
export LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${LD_LIBRARY_PATH:-}"

echo "==> nvcc check"
which nvcc
nvcc --version

echo "==> Step 1: persist CUDA on PATH for future terminals"
if ! grep -q "CUDA 13.2 for JetPack 7.2" "${HOME}/.bashrc" 2>/dev/null; then
  cat <<EOF >> "${HOME}/.bashrc"

# CUDA 13.2 for JetPack 7.2
export CUDA_HOME=${CUDA_HOME}
export PATH=\$CUDA_HOME/bin:\$PATH
export LD_LIBRARY_PATH=\$CUDA_HOME/lib64:\${LD_LIBRARY_PATH:-}
EOF
  echo "    added CUDA exports to ~/.bashrc"
fi

echo "==> Step 2: system prerequisites"
# Build basics + the system libs the wheel links (NVPL, cuDSS handled below).
sudo apt-get update
sudo apt-get install -y \
  python3.12 python3.12-venv python3.12-dev python3-pip \
  build-essential cmake git pkg-config \
  libopenblas-dev libjpeg-dev zlib1g-dev

echo "==> Step 3: NVPL + cuDSS (needed on a fresh flash for torch to import)"
# Only needed if not already present. Safe to re-run.
if ! ldconfig -p | grep -q libnvpl_lapack; then
  if [ ! -f /usr/share/keyrings/cuda-archive-keyring.gpg ]; then
    TMP_DEB="$(mktemp --suffix=.deb)"
    wget -qO "${TMP_DEB}" \
      https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/sbsa/cuda-keyring_1.1-1_all.deb
    sudo dpkg -i "${TMP_DEB}"
    sudo apt-get update
  fi
  sudo apt-get install -y nvpl
fi

if ! ldconfig -p | grep -q libcudss; then
  sudo apt-get install -y libcudss0-cuda-13
  echo "/usr/lib/aarch64-linux-gnu/libcudss/13" | sudo tee /etc/ld.so.conf.d/cudss.conf
  sudo ldconfig
fi

echo "==> Step 4: fresh Python 3.12 venv (with system site packages for TensorRT)"
rm -rf "${VENV}"
python3.12 -m venv "${VENV}" --system-site-packages
# shellcheck disable=SC1091
source "${VENV}/bin/activate"
python -m pip install --upgrade pip setuptools wheel packaging

if ! grep -q "alias jp72=" "${HOME}/.bashrc" 2>/dev/null; then
  cat <<EOF >> "${HOME}/.bashrc"

# JetPack 7.2 Python environment
alias jp72='source ${VENV}/bin/activate'
EOF
fi

echo "==> Step 5: clear any wrong torch, purge cache"
python -m pip uninstall -y torch torchvision torchaudio torchdata torchtext torch-tensorrt 2>/dev/null || true
python -m pip cache purge || true

echo "==> Step 6: install torch (prerelease cu132) -- the line that actually works"
# --pre is REQUIRED (wheels are prerelease). --extra-index-url (not --index-url)
# so numpy and friends still resolve from PyPI.
python -m pip install --pre torch --extra-index-url "${CU_INDEX}"
# pip install --pre torch --index-url "${CU_INDEX}" 

echo "==> Step 7: verify torch on the GPU BEFORE adding torchvision"
python - <<'PY'
import torch
print("torch:", torch.__version__)
print("torch cuda:", torch.version.cuda)
print("cuda available:", torch.cuda.is_available())
assert torch.cuda.is_available(), "CUDA not available in PyTorch"
print("device:", torch.cuda.get_device_name(0))
print("capability:", torch.cuda.get_device_capability(0))
x = torch.randn(2048, 2048, device="cuda")
y = x @ x
torch.cuda.synchronize()
print("KERNEL OK:", float(y.sum()))
PY

echo "==> Step 8: torchvision (only after torch passes)"
python -m pip install --pre torchvision --extra-index-url "${CU_INDEX}"


echo "==> Step 9: common packages + Jupyter kernel"
python -m pip install numpy pillow matplotlib jupyterlab ipykernel
python -m ipykernel install --user --name jetson-jp72 --display-name "Jetson JP7.2 Python 3.12"

echo "==> Step 10: onnx + confirm TensorRT visibility (for the INT8/TensorRT export pipeline)"
# onnx is a normal portable package -- plain pip install works here.
python -m pip install onnx
# TensorRT itself is NOT pip-installable on this platform -- it's the apt-installed
# system package (tensorrt/libnvinfer, from JetPack), made visible to this venv via
# --system-site-packages above. Just verify that actually worked rather than assume.
python -c "import tensorrt as trt; print('tensorrt visible in venv:', trt.__version__)"

echo
echo "Done. Run 'python verify_stack.py' for the full stack check."
echo "New terminals: type 'jp72' to activate the environment."
echo
echo "NOTE: torchaudio is intentionally NOT installed (common conflict source)."
echo "      If you need it, pin it to the exact installed torch version."
