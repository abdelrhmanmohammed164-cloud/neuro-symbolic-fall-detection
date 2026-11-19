#!/bin/bash
# ONE-TIME test evaluation of the configuration chosen on validation (exp04_center, 3 seeds) plus the
# baseline_v2 reference (3 seeds). No training: each run resumes from its finished checkpoints and only
# evaluates the test fold. Reporting rules are fixed in final_test_report.py.
# Usage: bash scripts/run_final_test.sh
P="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"   # repo root (override with PROJECT_ROOT)
command -v caffeinate >/dev/null 2>&1 || caffeinate() { [ "$1" = "-i" ] && shift; "$@"; }   # macOS keep-awake; no-op elsewhere
cd "$P" || exit 1
source .venv/bin/activate
CODE="$P/fall_detection_models/code/stgcn_urfd"
DATA=$(ls -d "$P"/processed_data/cfg_*/urfd_npy | head -1)
SPLIT="$P/splits/split_scene_v2.json"
ROOT="$P/runs/exp_mcc"; REP="$P/reports/final_test"
mkdir -p "$P/logs" "$REP"
cd "$CODE" || exit 1
exec > >(tee -a "$P/logs/final_test.log") 2>&1
set -o pipefail
echo "=== $(date) | final test evaluation"

evaluate() {   # evaluate <run name> [the same train.py args the run was trained with]
  local name="$1"; shift
  [ -f "$ROOT/$name/fold1/last_checkpoint.pt" ] || { echo "missing finished run: $name"; exit 1; }
  echo "--- $name $*"
  caffeinate -i python train.py --data_dir "$DATA" --split_file "$SPLIT" --out_dir "$ROOT/$name" \
    --select val_mcc "$@" 2>&1 | grep -E "TEST|RESUMED|compatible|refus|Error" || { echo "EVALUATION FAILED: $name"; exit 1; }
}
evaluate exp04_center                     --norm_mode window_center
evaluate exp04_center_s43  --seed 43      --norm_mode window_center
evaluate exp04_center_s44  --seed 44      --norm_mode window_center
evaluate baseline_v2_s42
evaluate baseline_v2_s43   --seed 43
evaluate baseline_v2_s44   --seed 44

python final_test_report.py --runs_root "$ROOT" --out "$REP" || exit 1
echo "FINAL_TEST_DONE -> reports/final_test/final_test.md"
