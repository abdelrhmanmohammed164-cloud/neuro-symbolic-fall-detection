#!/bin/bash
# Run the full preprocessing + sanity check on ONE sequence on the Mac GPU.
# Usage: bash scripts/run_test_one_video.sh [fall-01]      Log: logs/test_log.txt
SEQ="${1:-fall-01}"
P="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"   # repo root (override with PROJECT_ROOT)
cd "$P" || exit 1
mkdir -p logs; exec > >(tee logs/test_log.txt) 2>&1
set -e
source .venv/bin/activate

KIND=$([[ $SEQ == fall* ]] && echo falls || echo adls)
CODE="$P/fall_detection_models/code/stgcn_urfd"
W="$P/fall_detection_models/weights"
FR="$P/tests/work/frames"; OUT="$P/tests/work/urfd_npy"
mkdir -p "$FR" "$OUT"
[ -d "$FR/$SEQ-cam0-rgb" ] || unzip -q "$P/URFD/$KIND/$SEQ-cam0-rgb.zip" -d "$FR"

cd "$CODE"
START=$(date +%s)
python preprocess_sequence.py --rgb_dir "$FR/$SEQ-cam0-rgb" --seq "$SEQ" \
  --rtmo "$W/rtmo-l_onnx/end2end.onnx" --da_weights "$W/depth_anything_v2_vits.pth" \
  --da_repo "$P/fall_detection_models/code/Depth-Anything-V2" --out_dir "$OUT" --device auto
echo "preprocess time: $(( $(date +%s) - START )) s"
python make_frame_labels.py --features_dir "$P/URFD/features" --npy_dir "$OUT"
python check_sequence.py --seq "$SEQ" --rgb_dir "$FR/$SEQ-cam0-rgb" --npy_dir "$OUT" \
  --features "$P/URFD/features/urfall-cam0-$KIND.csv" --out "$P/tests/work/check_$SEQ.png"
echo "TEST_OK"
