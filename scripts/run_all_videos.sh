#!/bin/bash
# Preprocess all 70 URFD videos on the Mac GPU with chunk-level checkpoints (about 1 hour).
# Safe to stop at any time (Ctrl+C, closing the lid, power loss): just run it again and it
# continues from the last saved chunk. Finished videos are verified and skipped.
# Usage: bash scripts/run_all_videos.sh
P="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"   # repo root (override with PROJECT_ROOT)
command -v caffeinate >/dev/null 2>&1 || caffeinate() { [ "$1" = "-i" ] && shift; "$@"; }   # macOS keep-awake; no-op elsewhere
cd "$P" || exit 1
source .venv/bin/activate
cd "$P/fall_detection_models/code/stgcn_urfd"
pre() { python preprocess.py --root "$P" "$@"; }

# caffeinate keeps the Mac awake while processing runs
caffeinate -i python preprocess.py --root "$P" --resume 2>&1 | grep -v -E "onnxruntime|VerifyEachNode|GetCapability|xFormers"
pre --validate-checkpoints
OUT=$(pre --status | grep '^output dir:' | cut -d' ' -f3-)
python make_frame_labels.py --features_dir "$P/URFD/features" --npy_dir "$OUT" | tail -2
python summarize_preprocessing.py --npy_dir "$OUT" --features_dir "$P/URFD/features" \
  --out "$P/reports/preprocessing_summary.csv"
pre --status | tail -3
N=$(pre --status | grep -cE '^(fall|adl)-[0-9]+ +COMPLETED')
[ "$N" = "70" ] && echo "ALL_OK  (outputs in: $OUT)" || echo "NOT FINISHED: $N/70 completed. Run this script again to continue."
