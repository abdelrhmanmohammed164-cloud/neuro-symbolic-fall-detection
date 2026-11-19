# Neuro-Symbolic Fall Detection on URFD

Adaptation of the neural stream (two-stream ST-GCN) from
*A Neuro-Symbolic Approach to Fall Detection via Monocular Depth Estimation*,
Applied Sciences 2026, 16, 1895 ([doi:10.3390/app16041895](https://doi.org/10.3390/app16041895)),
to the public [UR Fall Detection dataset (URFD)](https://fenix.ur.edu.pl/mkepski/ds/uf.html).

The paper's own data are not public, so this is an **adaptation, not a reproduction**: numbers here cannot be
compared one to one with the paper's Table 4.

**Start here:** [`reports/URFD_project_record.pdf`](reports/URFD_project_record.pdf) documents every step, from the
first dataset audit to the current experiments.

## Status

| Part | State |
|---|---|
| URFD download (camera 0, 70 videos) | Done |
| Preprocessing: RTMO-l pose + Depth-Anything-V2 depth, checkpointed | Done, 70/70 videos, 0 failures |
| Two-stream ST-GCN (paper Eq. 2 to 6) | Done |
| Audit before training (leakage, shortcuts) | Done: `reports/ST-GCN_audit_report.md` |
| Scene-disjoint baseline, both folds | Done: `reports/scene_cv_audit/` |
| Validation redesign + 18 pre-training tests | Done |
| Controlled experiments (`scripts/run_experiments.sh`) | Running; results will go to `reports/experiments_mcc/` |
| Symbolic stream (neck height in metres, Eq. 1, 7 to 10) | Not started |
| Fusion (Eq. 11 to 13) | Not started |
| External test dataset | Not started |

Baseline on the two held-out scenes (pooled): 28 of 30 falls detected, but only 11 of 40 ADL videos free of false
alarms; frame-level F1 0.555. Details and the reasons (camera angle, fall direction, lighting) are in the PDF.

## Repository layout

```
README.md
requirements.txt
scripts/                       one-command entry points (run from anywhere; they find the repo root)
fall_detection_models/
  code/stgcn_urfd/             all project code (see its README.md)
  weights/                     NOT in git: model weights (see weights/README.md)
URFD/                          NOT in git: raw dataset (see URFD/README.md); features/README_features.md is included
processed_data/                preprocessed skeletons for all 70 videos (about 6 MB), ready for training
splits/                        saved train/val/test splits; never regenerated silently
runs/stgcn_scene/              baseline results (no model checkpoints)
reports/                       audits, tests, diagnoses, the project record PDF
tests/test_fall-01/            the single-video check figure
```

## Setup

Tested on macOS (Apple Silicon, MPS). Linux and Windows (WSL) should work on CPU or CUDA.

```bash
git clone https://github.com/abdelrhmanmohammed164-cloud/neuro-symbolic-fall-detection.git
cd neuro-symbolic-fall-detection
bash scripts/setup_mac.sh            # creates .venv and installs requirements (works on Linux too)
```

### Training only (no download needed)

The preprocessed skeletons are in `processed_data/`, so the ST-GCN can be trained right away:

```bash
bash scripts/run_experiments.sh quick    # 10-epoch trial of the whole pipeline, about 15 min on a Mac GPU
bash scripts/run_experiments.sh          # full experiments: 7 runs x 2 folds x 50 epochs, about 3 hours
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
- **Run the tests first.** `tests_pretraining.py` (18 tests) runs automatically before every training script;
  training does not start if any test fails.
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
