#!/bin/bash
# External test on GMDCSA-24 (never used for training, validation or tuning).
#   1. preprocessing of the 160 videos with the same pipeline as URFD (resumable: done videos are skipped)
#   2. evaluation of the 6 center models and the 6 baseline_v2 models; rules fixed in external_test.py
# Needs the dataset in external_data/GMDCSA24 (Subject 1..4 with Fall/ADL folders and CSVs).
# Usage: bash scripts/run_external_test.sh        (stop with Ctrl+C and run again to continue)
P="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"   # repo root (override with PROJECT_ROOT)
command -v caffeinate >/dev/null 2>&1 || caffeinate() { [ "$1" = "-i" ] && shift; "$@"; }   # macOS keep-awake; no-op elsewhere
cd "$P" || exit 1
source .venv/bin/activate
CODE="$P/fall_detection_models/code/stgcn_urfd"
SRC="$P/external_data/GMDCSA24"; NPY="$P/external_data/gmdcsa24_npy"; REP="$P/reports/external_gmdcsa24"
[ -d "$SRC/Subject 1" ] || { echo "dataset not found in $SRC"; exit 1; }
DEV=cpu; python -c "import torch,sys; sys.exit(0 if torch.backends.mps.is_available() else 1)" && DEV=mps
mkdir -p "$P/logs" "$REP"
cd "$CODE" || exit 1
exec > >(tee -a "$P/logs/external_test.log") 2>&1
set -o pipefail
echo "=== $(date) | external test GMDCSA-24 | device $DEV"

caffeinate -i python -u preprocess_external.py --data_dir "$SRC" --out_dir "$NPY" \
  --rtmo "$P/fall_detection_models/weights/rtmo-l_onnx/end2end.onnx" --device "$DEV" 2>&1 \
  | grep --line-buffered -v -i -E "warn|onnxruntime|CoreML|session_state" || { echo "PREPROCESSING FAILED"; exit 1; }
n=$(ls "$NPY"/*_meta.json 2>/dev/null | wc -l | tr -d ' ')
[ "$n" = "160" ] || { echo "only $n of 160 videos preprocessed; run again to continue"; exit 1; }

caffeinate -i python -u external_test.py --data_dir "$NPY" --runs_root "$P/runs/exp_mcc" --out "$REP" 2>&1 \
  | grep --line-buffered -v -i -E "warn|detach" || { echo "EVALUATION FAILED"; exit 1; }
echo "EXTERNAL_TEST_DONE -> reports/external_gmdcsa24/external_test.md"
