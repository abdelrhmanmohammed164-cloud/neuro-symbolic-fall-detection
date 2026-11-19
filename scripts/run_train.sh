#!/bin/bash
# ST-GCN neural stream: splits -> audit -> 15 pre-training tests -> training.
# Training does NOT start if any test fails. Safe to stop at any time: run again to resume
# from the last completed epoch of the current fold (finished folds are loaded, not retrained).
# Usage: bash scripts/run_train.sh
P="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"   # repo root (override with PROJECT_ROOT)
command -v caffeinate >/dev/null 2>&1 || caffeinate() { [ "$1" = "-i" ] && shift; "$@"; }   # macOS keep-awake; no-op elsewhere
cd "$P" || exit 1
source .venv/bin/activate
CODE="$P/fall_detection_models/code/stgcn_urfd"
DATA=$(ls -d "$P"/processed_data/cfg_*/urfd_npy | head -1)
mkdir -p "$P/splits" "$P/reports" "$P/runs" "$P/logs"
cd "$CODE" || exit 1
exec > >(tee -a "$P/logs/train_pipeline.log") 2>&1
set -o pipefail   # a failing python step stops the script even when piped through grep
echo "=== $(date) | data: $DATA"

# 1. scene groups (from the preprocessing checkpoints) and persistent splits (never silently regenerated)
[ -f "$P/reports/scene_groups.csv" ] || python scene_groups.py --checkpoints "$(ls -d "$P"/checkpoints/cfg_* | head -1)" \
  --out "$P/reports/scene_groups.csv" --thr 0.9 || exit 1
python splits.py --data_dir "$DATA" --scene_csv "$P/reports/scene_groups.csv" --mode scene \
  --out "$P/splits/split_scene.json" || exit 1
python splits.py --data_dir "$DATA" --mode video --out "$P/splits/split_video.json" || exit 1

# 2. dataset audit (read-only)
python audit_dataset.py --npy_dir "$DATA" --features_dir "$P/URFD/features" --out_dir "$P/reports" > /dev/null || exit 1
echo "audit written to reports/dataset_validation_report.csv and reports/dataset_audit_summary.json"

# 3. pre-training tests: stop here if anything fails
python tests_pretraining.py --data_dir "$DATA" --features_dir "$P/URFD/features" \
  --split_scene "$P/splits/split_scene.json" --split_video "$P/splits/split_video.json" \
  --out "$P/reports/pretraining_tests.json" 2>&1 | grep -E "^\[|PASS ->"
if [ "${PIPESTATUS[0]}" != "0" ]; then echo "TESTS FAILED -> training not started"; exit 1; fi

# 4a. PRIMARY: scene-disjoint 2-fold CV (test rooms never seen in training)
caffeinate -i python train.py --data_dir "$DATA" --split_file "$P/splits/split_scene.json" \
  --out_dir "$P/runs/stgcn_scene" 2>&1 | grep -v -i "warn" || exit 1
# 4b. SECONDARY (optimistic, scenes shared between train and test): 5-fold by video
caffeinate -i python train.py --data_dir "$DATA" --split_file "$P/splits/split_video.json" \
  --out_dir "$P/runs/stgcn_video" 2>&1 | grep -v -i "warn" || exit 1
echo "TRAINING_DONE  results: runs/stgcn_scene/summary.json (primary), runs/stgcn_video/summary.json (secondary)"
