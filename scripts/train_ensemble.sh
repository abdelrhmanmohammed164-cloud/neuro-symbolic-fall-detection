#!/bin/bash
# Trains the 3 neck-centred ST-GCN models (exp04_center, seeds 42, 43, 44) on both folds of split_scene_v2,
# then writes the out-of-fold ensemble predictions on URFD. Test data are never used for training or checkpoint
# selection (train.py --no_test --select val_mcc). Safe to stop and run again: finished runs are skipped and an
# interrupted run resumes from its last completed epoch.
# Usage: bash scripts/train_ensemble.sh
P="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
command -v caffeinate >/dev/null 2>&1 || caffeinate() { [ "$1" = "-i" ] && shift; "$@"; }
cd "$P" || exit 1
source .venv/bin/activate
CODE="$P/fall_detection_models/code/stgcn_urfd"
DATA=$(ls -d "$P"/processed_data/cfg_*/urfd_npy | head -1)
SPLIT="$P/splits/split_scene_v2.json"
ROOT="$P/runs/exp_mcc"
mkdir -p "$P/logs" "$ROOT"
cd "$CODE" || exit 1
exec > >(tee -a "$P/logs/train_ensemble.log") 2>&1
set -o pipefail
echo "=== $(date) | data: $DATA"

python tests_pretraining.py --data_dir "$DATA" --features_dir "$P/URFD/features" \
  --split_scene "$SPLIT" --split_video "$P/splits/split_video_v2.json" \
  --out "$P/logs/pretraining_tests.json" 2>&1 | grep -E "^\[|FAIL ->"
if [ "${PIPESTATUS[0]}" != "0" ]; then echo "TESTS FAILED -> training not started"; exit 1; fi

run() {   # run <name> [extra train.py args]
  local name="$1"; shift
  echo "--- $name $*"
  caffeinate -i python train.py --data_dir "$DATA" --split_file "$SPLIT" --out_dir "$ROOT/$name" \
    --no_test --select val_mcc --norm_mode window_center "$@" 2>&1 | grep -v -i -E "warn|detach\(\) first" || { echo "RUN FAILED: $name"; exit 1; }
}
run exp04_center
run exp04_center_s43 --seed 43
run exp04_center_s44 --seed 44

python neural_ensemble.py --data_dir "$DATA" --runs_root "$ROOT" --mode oof --split_file "$SPLIT" \
  --out "$P/reports/symbolic_v3_precision/neural_test_oof.csv" || exit 1
echo "ENSEMBLE_DONE -> reports/symbolic_v3_precision/neural_test_oof.csv"
