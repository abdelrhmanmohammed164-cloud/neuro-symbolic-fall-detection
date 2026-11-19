# Lightweight ST-GCN for URFD fall detection

| File | What it does |
|---|---|
| `graph.py` | 14-joint skeleton, spatial partitioning (self / centripetal / centrifugal), COCO-17 -> 14 mapping |
| `model.py` | ST-GCN, 6 blocks, 32 -> 64 -> 128 channels, dropout 0.25 per block, 0.5 before the classifier (~0.5 M parameters) |
| `dataset.py` | Loads one `.npy` per sequence, 80-frame windows, flip with L/R swap, scale, shift, jitter |
| `train.py` | 5-fold stratified CV by sequence, early stopping and temperature scaling on an inner validation split, metrics, probability export |
| `make_frame_labels.py` | Per-frame fall labels from the URFD features CSVs |

## Input format

One file per sequence in `--data_dir`:

- `fall-01.npy`, `adl-07.npy`, ...: float32, shape `(C, T_full, V=14, M=1)`, channels `x/W, y/H, conf` (+ optional `z`). `T_full` is the full clip length; do not cut it to 80 yourself.
- `fall-01_frame_labels.npy` (optional, recommended): int `(T_full,)`, 1 = fall frame.
- Missing joints: all channels 0 (confidence 0).

`in_channels` is read from the data, so adding a 4th channel needs no code change.

## Run

```bash
pip install torch scikit-learn numpy
python make_frame_labels.py --features_dir ~/Downloads/URFD/features --npy_dir ../urfd_npy
python train.py --data_dir ../urfd_npy --out_dir runs/stgcn
```

## Outputs (for late fusion)

- `oof_sequence_probs.csv`: sequence, label, fold, prob (max calibrated window probability)
- `oof_frame_probs.csv`: sequence, frame (1-based), fold, prob. The probability at frame t comes from the window ending at t, which matches N_t in the paper (Eq. 6). Feed it to the fusion step with the symbolic S_t.
- `metrics.json`: per-fold and pooled accuracy, precision, recall, F1, confusion matrix `[[TN, FP], [FN, TP]]`.

## Limits

- URFD has 5 actors and no subject IDs, so the same person appears in training and test folds. Sequence-level CV avoids video leakage, not subject leakage.
- Temperature is fitted on about 11 validation sequences per fold, so calibration is approximate.
- Verified only on synthetic data so far; real numbers come after RTMO preprocessing.

## Preprocessing one sequence (tested on fall-01)

```bash
python preprocess_sequence.py --rgb_dir fall-01-cam0-rgb --seq fall-01 \
    --rtmo ../../weights/rtmo-l_onnx/end2end.onnx \
    --da_weights ../../weights/depth_anything_v2_vits.pth --da_repo ../Depth-Anything-V2 \
    --out_dir ../../urfd_npy
python make_frame_labels.py --features_dir ../../../URFD/features --npy_dir ../../urfd_npy
python check_sequence.py --seq fall-01 --rgb_dir fall-01-cam0-rgb --npy_dir ../../urfd_npy \
    --features ../../../URFD/features/urfall-cam0-falls.csv --out check_fall-01.png
```

Needs: `torch torchvision onnxruntime opencv-python rtmlib tqdm matplotlib scikit-learn`.
Runtime on CPU for 160 frames: about 4 min pose (RTMO-l) + 4 min depth (Depth-Anything Small).

fall-01 result: person found in 152/160 frames; corr(neck height in image, URFD D_mm) = 0.97,
corr(Depth-Anything neck value, D_mm) = -0.97 (negative because the model outputs inverse depth).

Neck while lying (fixed): RTMO's shoulder confidence drops to ~0.05 when the person lies on
the floor, so the paper's neck (shoulder midpoint) disappears. preprocess_sequence.py now writes
`<seq>_anchor.npy` (u, v, source): source 1 = neck (both shoulders >= 0.3, as in the paper),
source 2 = mid-hip fallback (both hips >= 0.3), 0 = none. Lying flat, hip and neck heights are
nearly equal; standing, the hips (~0.9 m) are far above the 0.19 m threshold, so the fallback cannot
fake "near the floor". Neck depth is sampled at this anchor. Person detection threshold lowered
from 0.3 to 0.1 (joint threshold stays 0.3). fall-01 after the fix: person in 160/160 frames,
anchor neck 117 / hip 23 / none 20 (frames 131-145, person barely visible).

GPU: `--device auto` uses MPS (PyTorch) for Depth-Anything and CoreML (onnxruntime) for RTMO on
Apple Silicon; if CoreML cannot run RTMO, it falls back to CPU automatically.

## Checkpointed preprocessing of all 70 videos (preprocess.py)

```bash
python preprocess.py --resume                 # process / continue where it stopped
python preprocess.py --status                 # progress per video from the SQLite manifest
python preprocess.py --validate-checkpoints   # re-verify all checkpoints + outputs -> reports/validation_report.csv
python preprocess.py --cleanup-checkpoints --yes   # optional: delete chunk files of verified videos
```

- Reads frames straight from `URFD/<kind>/<seq>-cam0-rgb.zip` (inputs are never modified).
- Chunks of 60 frames: `checkpoints/cfg_<fp>/<seq>/chunk_XXXX.npz`, manifest `checkpoints/processing_manifest.sqlite`.
- Final outputs: `processed_data/cfg_<fp>/urfd_npy/` (`<seq>.npy`, `_anchor`, `_neck_depth`, `_raw_coco17`, `_timestamps`, `_meta.json`).
- Every write: temp file -> fsync -> reload/validate -> atomic rename -> manifest commit.
- Config fingerprint = code version, chunk size, thresholds, depth settings, SHA-256 of both model files.
  A different config gets its own folders; old checkpoints are kept and never reused.
- Depth is stored per frame at 160x120 (float16), so the neck depth and later the floor fit can be
  recomputed without rerunning Depth-Anything. The neck depth is the 3x3 median at that resolution.
- Timestamps come from `<seq>-data.csv`; if its row count does not match the frames (adl-37) the
  frame index / 30 fps is used and the source is recorded in `_meta.json`.
- `test_checkpoints.sh` runs the 8 recovery tests on disposable data (results: reports/checkpoint_test_results.txt).

## Neural stream training (after the pre-training audit)

```bash
bash ~/Downloads/University\ project/scripts/run_train.sh
```
Runs: scene groups -> persistent splits (`splits/`) -> dataset audit -> 15 pre-training tests ->
training (only if all tests pass). Model: `TwoStreamSTGCN` (paper Eq. 2-6, coordinate + motion streams).
Primary result: `runs/stgcn_scene/summary.json` (leave-scene-out CV). Secondary, optimistic:
`runs/stgcn_video/summary.json`. Per fold: `last_checkpoint.pt`, `best_checkpoint.pt`,
`training_history.csv`, `result.json`; `oof_frame_probs.csv` = per-frame fall probabilities for fusion.
Full audit: `reports/ST-GCN_audit_report.md`.
One fold first (learning vs memorising check): `bash scripts/run_one_fold.sh`
-> `reports/fold1_diagnosis.png` / `.json` (train vs val vs test, learning curves, baseline).

## After the scene-CV audit (`reports/scene_cv_audit/ST-GCN_scene_CV_audit_report.md`)

Validation redesign (model unchanged, original splits and their fingerprints untouched):

- `splits.py --val_mode scene` -> `splits/split_scene_v2.json`: validation = one whole ADL scene that is
  not in training (pre-registered: the smallest one; fold 0 -> A7, fold 1 -> A8) + a random subset of the
  falls. Test folds identical to the original split.
- `splits.py --mode video --val_mode random_grouped` -> `splits/split_video_v2.json`: same draw, but the
  near-duplicate group adl-25/27/29 is never split across train / val / test.
- Tests: 07 now checks the near-duplicate group across train / val / test (the original splits FAIL it,
  which is the bug the audit found); 15 = whole dataset (gate); 16 = padding vs label per training fold
  (WARN, reported, does not block); 17 = `--no_test` never evaluates the test fold; 18 = `--select val_mcc` picks the right epoch and resumes exactly.

New `train.py` options (defaults reproduce the original runs exactly, same config fingerprint):

| Option | Meaning |
|---|---|
| `--no_test` | model-selection run: writes `fold<k>/val_result.json` (every validation frame), never loads a test prediction |
| `--grad_clip X` | clip gradient norm |
| `--pad_side left` | pad before the real frames (current frame always last) instead of after |
| `--pad_balance` | per-class padding probability so padded fractions are equal inside each fold |
| `--select val_mcc` | checkpoint with the highest validation MCC instead of the lowest val loss |

Checkpoint selection for the experiments: `--select val_mcc` (highest validation MCC, ties -> lower val loss).
A 10-epoch trial (`runs/exp_v2_quick`, kept) showed that the lowest val loss on an unseen scene is a model that
predicts nothing, and F1 rewards a model that predicts fall everywhere; MCC is 0 for both. Test 18 checks it.

Experiments: `bash scripts/run_experiments.sh` (baseline with 2 seeds + 5 single-change experiments, all
`--no_test --select val_mcc`), then `compare_experiments.py` applies the pre-registered rule written at the top
of that file. Results: `reports/experiments_mcc/comparison.md`. Small trial first: `... run_experiments.sh quick`.

Diagnostics: `analyze_fold.py` (per-video errors, events, latency, bootstrap CIs), `context_probe.py`,
`scale_probe.py`, `missing_probe.py` (counterfactual probes; diagnostic only).

Legacy scripts (`run_one_fold.sh`, `run_fold.sh`, `run_train.sh`) use the original splits and now stop at
test 07 on purpose. They are kept only to document how the baseline was produced.
