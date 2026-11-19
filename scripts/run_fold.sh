#!/bin/bash
# Train and diagnose ONE fold of the scene split with the FROZEN baseline settings.
# Usage: bash scripts/run_fold.sh 0
# - same checks as before (splits verified unchanged, 15 tests); training starts only if all pass
# - writes only runs/stgcn_scene/fold<k>/ and reports/fold<k>_diagnosis.{png,json}
# - refuses to overwrite an existing diagnosis; never touches other folds
K="$1"
[[ "$K" =~ ^[0-9]+$ ]] || { echo "usage: bash run_fold.sh <fold number>"; exit 1; }
P="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"   # repo root (override with PROJECT_ROOT)
command -v caffeinate >/dev/null 2>&1 || caffeinate() { [ "$1" = "-i" ] && shift; "$@"; }   # macOS keep-awake; no-op elsewhere
cd "$P" || exit 1
source .venv/bin/activate
CODE="$P/fall_detection_models/code/stgcn_urfd"
DATA=$(ls -d "$P"/processed_data/cfg_*/urfd_npy | head -1)
DIAG="$P/reports/fold${K}_diagnosis.png"
[ -e "$DIAG" ] && { echo "$DIAG already exists -> not overwriting. Move it aside on purpose first."; exit 1; }
mkdir -p "$P/logs"
cd "$CODE" || exit 1
exec > >(tee -a "$P/logs/train_fold${K}.log") 2>&1
set -o pipefail
echo "=== $(date) | fold $K | data: $DATA"
echo "fold 1 checkpoints before:"; shasum -a 256 "$P"/runs/stgcn_scene/fold1/*.pt 2>/dev/null | cut -c1-16

python splits.py --data_dir "$DATA" --scene_csv "$P/reports/scene_groups.csv" --mode scene \
  --out "$P/splits/split_scene.json" || exit 1
python tests_pretraining.py --data_dir "$DATA" --features_dir "$P/URFD/features" \
  --split_scene "$P/splits/split_scene.json" --split_video "$P/splits/split_video.json" \
  --out "$P/reports/pretraining_tests_fold${K}.json" 2>&1 | grep -E "^\[|PASS ->"
if [ "${PIPESTATUS[0]}" != "0" ]; then echo "TESTS FAILED -> training not started"; exit 1; fi

caffeinate -i python train.py --data_dir "$DATA" --split_file "$P/splits/split_scene.json" \
  --out_dir "$P/runs/stgcn_scene" --only_folds "$K" 2>&1 | grep -v -i "warn" || exit 1
python diagnose_fold.py --data_dir "$DATA" --split_file "$P/splits/split_scene.json" \
  --run_dir "$P/runs/stgcn_scene" --fold "$K" --out "$DIAG" 2>&1 | grep -v -i "warn" || exit 1
echo "fold 1 checkpoints after (must be identical):"; shasum -a 256 "$P"/runs/stgcn_scene/fold1/*.pt 2>/dev/null | cut -c1-16
echo "FOLD_${K}_DONE -> reports/fold${K}_diagnosis.png and .json"
