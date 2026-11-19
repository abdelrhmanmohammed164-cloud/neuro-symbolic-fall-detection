#!/bin/bash
# One-time setup of the Python environment for the fall detection project (macOS, Apple Silicon).
# Creates "University project/.venv" and installs everything. Log: setup_log.txt
P="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"   # repo root (override with PROJECT_ROOT)
cd "$P" || { echo "folder not found: $P"; exit 1; }
mkdir -p logs; exec > >(tee logs/setup_log.txt) 2>&1
set -e

PY=$(command -v python3 || true)
if [ -z "$PY" ]; then
  echo "python3 not found. Install Python 3.11 from https://www.python.org/downloads/macos/ and run again."
  exit 1
fi
echo "python: $PY $($PY --version)"
$PY -c 'import sys; assert sys.version_info >= (3, 9), "need Python >= 3.9"'

[ -d .venv ] || $PY -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install torch torchvision onnxruntime opencv-python rtmlib tqdm matplotlib scikit-learn numpy

python - <<'EOF'
import platform, torch, onnxruntime, cv2, sklearn
print("macOS", platform.mac_ver()[0], platform.machine())
print("torch", torch.__version__, "| MPS (Mac GPU) available:", torch.backends.mps.is_available())
print("onnxruntime", onnxruntime.__version__, "| providers:", onnxruntime.get_available_providers())
x = torch.randn(256, 256, device="mps" if torch.backends.mps.is_available() else "cpu")
print("GPU test matmul OK:", float((x @ x).sum()) == float((x @ x).sum()))
EOF
echo "SETUP_OK"
