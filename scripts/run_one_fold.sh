#!/bin/bash
# Train ONE fold (scene split, fold 1) and check whether the model learns or memorises.
# Same checks as run_train.sh first (splits, audit, 15 tests); training only starts if all pass.
# The fold is saved in runs/stgcn_scene with the final settings, so run_train.sh later reuses it.
# Usage: bash scripts/run_one_fold.sh
P="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"   # repo root (override with PROJECT_ROOT)
command -v caffeinate >/dev/null 2>&1 || caffeinate() { [ "$1" = "-i" ] && shift; "$@"; }   # macOS keep-awake; no-op elsewhere
cd "$P" || exit 1
source .venv/bin/activate
CODE="$P/fall_detection_models/code/stgcn_urfd"
DATA=$(ls -d "$P"/processed_data/cfg_*/urfd_npy | head -1)
mkdir -p "$P/splits" "$P/reports" "$P/runs" "$P/logs"
cd "$CODE" || exit 1
exec > >(tee -a "$P/logs/train_one_fold.log") 2>&1
set -o pipefail
echo "=== $(date) | data: $DATA"

[ -f "$P/reports/scene_groups.csv" ] || python scene_groups.py --checkpoints "$(ls -d "$P"/checkpoints/cfg_* | head -1)" \
  --out "$P/reports/scene_groups.csv" --thr 0.9 || exit 1
python splits.py --data_dir "$DATA" --scene_csv "$P/reports/scene_groups.csv" --mode scene \
  --out "$P/splits/split_scene.json" || exit 1
python splits.py --data_dir "$DATA" --mode video --out "$P/splits/split_video.json" || exit 1
python audit_dataset.py --npy_dir "$DATA" --features_dir "$P/URFD/features" --out_dir "$P/reports" > /dev/null || exit 1
python tests_pretraining.py --data_dir "$DATA" --features_dir "$P/URFD/features" \
  --split_scene "$P/splits/split_scene.json" --split_video "$P/splits/split_video.json" \
  --out "$P/reports/pretraining_tests.json" 2>&1 | grep -E "^\[|PASS ->"
if [ "${PIPESTATUS[0]}" != "0" ]; then echo "TESTS FAILED -> training not started"; exit 1; fi

caffeinate -i python train.py --data_dir "$DATA" --split_file "$P/splits/split_scene.json" \
  --out_dir "$P/runs/stgcn_scene" --only_folds 1 2>&1 | grep -v -i "warn" || exit 1
python diagnose_fold.py --data_dir "$DATA" --split_file "$P/splits/split_scene.json" \
  --run_dir "$P/runs/stgcn_scene" --fold 1 --out "$P/reports/fold1_diagnosis.png" 2>&1 | grep -v -i "warn" || exit 1
echo "ONE_FOLD_DONE  -> reports/fold1_diagnosis.png and reports/fold1_diagnosis.json"
