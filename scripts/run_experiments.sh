#!/bin/bash
# Controlled experiments on the NEW validation design (split_scene_v2). No test data is ever evaluated
# (train.py --no_test). Checkpoint selection by validation MCC (--select val_mcc): the 10-epoch trial
# (runs/exp_v2_quick, kept) showed that the lowest val loss on an unseen scene is a do-nothing model. One change per experiment, all compared with the pre-registered rule in
# compare_experiments.py. Safe to stop (Ctrl+C) and run again: finished runs are skipped and an
# interrupted run resumes from its last completed epoch.
# Usage: bash scripts/run_experiments.sh          (full: 7 runs x 2 folds x 50 epochs)
#        bash scripts/run_experiments.sh quick    (small trial: same steps, 10 epochs, 2 runs, separate folders)
#        bash scripts/run_experiments.sh confirm  (seed confirmation of exp04_center: 3 more runs, same folders)
P="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"   # repo root (override with PROJECT_ROOT)
command -v caffeinate >/dev/null 2>&1 || caffeinate() { [ "$1" = "-i" ] && shift; "$@"; }   # macOS keep-awake; no-op elsewhere
QUICK=0; [ "$1" = "quick" ] && QUICK=1
CONFIRM=0; [ "$1" = "confirm" ] && CONFIRM=1
cd "$P" || exit 1
source .venv/bin/activate
CODE="$P/fall_detection_models/code/stgcn_urfd"
DATA=$(ls -d "$P"/processed_data/cfg_*/urfd_npy | head -1)
SPLIT="$P/splits/split_scene_v2.json"
if [ $QUICK = 1 ]; then
  ROOT="$P/runs/exp_mcc_quick"; REP="$P/reports/experiments_mcc_quick"; LOG="$P/logs/experiments_mcc_quick.log"
  EXTRA="--epochs 10"
else
  ROOT="$P/runs/exp_mcc"; REP="$P/reports/experiments_mcc"; LOG="$P/logs/experiments_mcc.log"; EXTRA=""
fi
mkdir -p "$P/logs" "$ROOT" "$REP"
cd "$CODE" || exit 1
exec > >(tee -a "$LOG") 2>&1
set -o pipefail
echo "=== $(date) | data: $DATA | quick=$QUICK"

# new splits (saved once, never overwritten; the original split files stay untouched)
python splits.py --data_dir "$DATA" --scene_csv "$P/reports/scene_groups.csv" --mode scene --val_mode scene \
  --out "$SPLIT" || exit 1
python splits.py --data_dir "$DATA" --mode video --val_mode random_grouped \
  --out "$P/splits/split_video_v2.json" || exit 1
python tests_pretraining.py --data_dir "$DATA" --features_dir "$P/URFD/features" \
  --split_scene "$SPLIT" --split_video "$P/splits/split_video_v2.json" \
  --out "$REP/pretraining_tests_v2.json" 2>&1 | grep -E "^\[|FAIL ->"
if [ "${PIPESTATUS[0]}" != "0" ]; then echo "TESTS FAILED -> experiments not started"; exit 1; fi

run() {   # run <name> [extra train.py args]
  local name="$1"; shift
  echo "--- $name $*"
  caffeinate -i python train.py --data_dir "$DATA" --split_file "$SPLIT" --out_dir "$ROOT/$name" \
    --no_test --select val_mcc $EXTRA "$@" 2>&1 | grep -v -i -E "warn|detach\(\) first" || { echo "RUN FAILED: $name"; exit 1; }
}
if [ $CONFIRM = 1 ]; then
  # pre-registered rule: compare_seeds.py (written before these runs started)
  run baseline_v2_s44 --seed 44
  run exp04_center_s43 --seed 43 --norm_mode window_center
  run exp04_center_s44 --seed 44 --norm_mode window_center
  python compare_seeds.py --runs_root "$ROOT" --out "$REP" || exit 1
  echo "CONFIRM_DONE -> reports/experiments_mcc/seed_confirmation.md"
  exit 0
fi

run baseline_v2_s42
if [ $QUICK = 1 ]; then
  run exp05_padbalance --pad_balance
else
  run baseline_v2_s43 --seed 43
  run exp01_gradclip   --grad_clip 1.0
  run exp02_leftpad    --pad_side left
  run exp03_useconf    --use_conf
  run exp04_center     --norm_mode window_center
  run exp05_padbalance --pad_balance
fi

python compare_experiments.py --runs_root "$ROOT" --out "$REP" || exit 1
if [ $QUICK = 1 ]; then
  echo "QUICK_DONE -> $(basename "$REP")/comparison.md (10 epochs only: a pipeline and learning check, NOT a basis for any decision)"
else
  echo "EXPERIMENTS_DONE -> reports/experiments_mcc/comparison.md"
fi
