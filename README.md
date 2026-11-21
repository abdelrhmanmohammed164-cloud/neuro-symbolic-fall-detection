# Neuro-Symbolic Fall Detection on URFD

Adaptation of the neural stream (two-stream ST-GCN) from
*A Neuro-Symbolic Approach to Fall Detection via Monocular Depth Estimation*,
Applied Sciences 2026, 16, 1895 ([doi:10.3390/app16041895](https://doi.org/10.3390/app16041895)),
to the public [UR Fall Detection dataset (URFD)](https://fenix.ur.edu.pl/mkepski/ds/uf.html).

The paper's own data are not public, so this is an **adaptation, not a reproduction**: numbers here cannot be
compared one to one with the paper's Table 4.

## Active models

| Part | Result |
|---|---|
| Neural stream: ensemble of 3 neck-centred ST-GCN seeds (`neural_ensemble.py`, `scripts/train_ensemble.sh`) | URFD out-of-fold: MCC 0.547, F1 0.619, precision 0.570. GMDCSA-24 (160 unseen videos, all 6 models): MCC 0.530, precision 0.829; `reports/external_gmdcsa24/external_test.md`. The ensemble was chosen after both test sets had been seen, so these numbers are post-hoc |
| Symbolic stream: improved V3 (`symbolic_v3_improved.py`, `evaluate_symbolic_v3.py`) | Neck height in metres from the Kinect, V3 rules with a 1 s descent window, a minimum drop and V3's latch, chosen per fold by inner group CV on train+val. URFD test: precision 0.796, recall 0.396, F1 0.528, MCC 0.513, 30/30 falls, 28/40 ADL videos without alarm; `reports/symbolic_v3_precision/` |
| **Neuro-symbolic fusion** (`fusion_neurosymbolic.py`, applied with `fusion.py --rule neurosymbolic`) | Neural alarm vetoed while the metric neck height says the person is upright, OR the V3 alarm with weak neural support. Thresholds chosen per fold on train+val without test labels. URFD test (post-hoc): MCC 0.650, F1 0.703, accuracy 0.909, recall 0.711, precision 0.695, 30/30 falls; +0.10 MCC over the neural ensemble (bootstrap 95% CI +0.02 to +0.18); `reports/fusion_neurosymbolic/report.md` |
| External test of the improved V3 on FUKinect-Fall (21 new people, Kinect v1, no refitting) | fold 0 config: 132/167 falls, 18% of ADL clips with an alarm; fold 1 config: 17/167 falls. Neither passed the pre-registered rule; `reports/symbolic_v3_external_fukinect/report.md` |

Earlier versions (the original V3, v2, v4, the earlier fusion experiments, V3-norm, the audits and diagnostics) are only in the git history.

Check what this checkout can run with `python preflight.py`, and the split integrity with `data_audit.py` (`reports/data_integrity_audit.json`).

## Repository layout

```
README.md
requirements.txt
scripts/                       setup, preprocessing, ensemble training, external test
fall_detection_models/
  code/stgcn_urfd/             project code (see its README.md)
  weights/                     NOT in git: model weights (see weights/README.md)
URFD/                          NOT in git: raw dataset (see URFD/README.md); the feature CSVs are included
processed_data/                preprocessed skeletons and metric Kinect neck heights for all 70 videos
splits/                        the saved splits (split_scene_v2.json, split_video_v2.json)
reports/                       results of the active models
```

## Setup

Tested on macOS (Apple Silicon, MPS). Linux and Windows (WSL) should work on CPU or CUDA.

```bash
git clone https://github.com/abdelrhmanmohammed164-cloud/neuro-symbolic-fall-detection.git
cd neuro-symbolic-fall-detection
bash scripts/setup_mac.sh            # creates .venv and installs requirements (works on Linux too)
```

### Training the neural ensemble (no download needed)

The preprocessed skeletons are in `processed_data/`, so the models can be trained right away:

```bash
bash scripts/train_ensemble.sh       # 3 seeds x 2 folds, then the out-of-fold ensemble predictions
```

Every run is resumable: stop it and run the same command again. Training with batch 256 needs about 6 GB of RAM.

### Redoing the preprocessing (needs the raw data and the models)

1. Download URFD camera 0 into `URFD/` as described in [`URFD/README.md`](URFD/README.md).
2. Download the weights and the Depth-Anything-V2 code as described in
   [`fall_detection_models/weights/README.md`](fall_detection_models/weights/README.md).
3. `bash scripts/run_all_videos.sh`

## Rules for working on this project

These come from the audits and keep the results honest:

- **Never choose anything using test results.** Experiments run with `train.py --no_test`; the test folds are
  evaluated once, at the end, for the chosen configuration only. Fold 0 and Fold 1 test sets have already been
  seen once, so any later number on them is post-hoc and must be reported as such.
- **Never overwrite a split file.** `splits.py` refuses to; create a new file instead.
- **New experiment, new folder.** `train.py` refuses to resume or overwrite a run made with different settings.
- **Run the tests first.** `tests_pretraining.py` (18 tests) runs automatically before training, and training does
  not start if any test fails. `tests_symbolic_v3_improved.py` checks the symbolic stream and
  `python -m unittest tests_fusion_audit.py` the fusion, the export and the split audit.
- Report event-level results (falls detected, ADL videos with false alarms) with confidence intervals over videos,
  not only frame-level accuracy.

## Data and licences

- URFD is licensed [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/) by its authors
  (Kwolek and Kepski, University of Rzeszow). The preprocessed arrays in `processed_data/` and the frame images in
  the reports are derived from URFD and are shared under the same licence, for non-commercial research only.
- RTMO (OpenMMLab) and Depth-Anything-V2 have their own licences; weights are not redistributed here.
- No licence has been chosen yet for the project code.

## Reference

Paper: Applied Sciences 2026, 16, 1895, doi:10.3390/app16041895.
Dataset: B. Kwolek and M. Kepski, "Human fall detection on embedded platform using depth maps and wireless
accelerometer", Computer Methods and Programs in Biomedicine, 2014.
