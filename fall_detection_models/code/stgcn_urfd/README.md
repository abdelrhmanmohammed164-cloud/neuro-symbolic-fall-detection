# Code

Only the files the two active models need. Earlier versions are in the git history.

## Preprocessing (URFD videos to skeletons)

| File | What it does |
|---|---|
| `preprocess.py` | Checkpointed preprocessing of all 70 URFD videos: RTMO-l pose and Depth-Anything-V2 depth, read straight from the URFD zips. `--resume`, `--status`, `--validate-checkpoints` |
| `preprocess_sequence.py` | Pose and depth for one sequence (used by `preprocess.py`), including the hip fallback when the neck is not visible |
| `make_frame_labels.py` | Per-frame fall labels from the URFD feature CSVs |
| `summarize_preprocessing.py` | Summary table after preprocessing |
| `preprocess_external.py` | The same preprocessing for GMDCSA-24 (external test of the neural stream) |

Outputs go to `processed_data/cfg_<fingerprint>/urfd_npy/`. A different configuration gets its own folder, and old outputs are never reused.

## Neural stream: ST-GCN ensemble

| File | What it does |
|---|---|
| `graph.py` | 14-joint skeleton and spatial partitioning |
| `model.py` | Two-stream ST-GCN (paper Eq. 2 to 6) |
| `dataset.py` | 80-frame windows, augmentation, neck-centred coordinates (`--norm_mode window_center`) |
| `train.py` | Training per fold, temperature scaling on validation, checkpoint by validation MCC (`--select val_mcc`), `--no_test` for runs that must not touch test data |
| `scene_groups.py`, `splits.py` | Scene groups and the saved splits (`splits/split_scene_v2.json`). `splits.py` refuses to overwrite a split file |
| `tests_pretraining.py` | 18 checks run before every training (leakage, near-duplicates, `--no_test`, checkpoint selection) |
| `neural_ensemble.py` | Mean of the calibrated probabilities of the 3 seeds. `--mode oof` for URFD (each video scored only by the models of the fold where it is test), `--mode all` for other datasets |
| `external_test.py` | GMDCSA-24 evaluation |

Train the three models and write the ensemble predictions: `bash scripts/train_ensemble.sh`.

## Symbolic stream: improved V3

| File | What it does |
|---|---|
| `kinect_neck_height.py` | Metric neck height on URFD from the Kinect depth: RANSAC floor plane on the background, neck pixel from RTMO, back-projection |
| `symbolic_v3_improved.py` | V3 rules (Eq. 8 to 10) with a 1 s descent window, a minimum drop and V3's latch (variants A, P, B, C, D) |
| `evaluate_symbolic_v3.py` | Inner group CV on train+val per fold, the pre-registered selection rule, one score on test |
| `plot_symbolic_v3.py` | Figures for `reports/symbolic_v3_precision/` |
| `tests_symbolic_v3_improved.py` | 30 checks (rules, causality, split and leakage checks on the real data) |
| `fukinect_neck_height.py`, `evaluate_v3_fukinect.py` | External test of the frozen V3 configs on FUKinect-Fall |

```bash
python evaluate_symbolic_v3.py              # reports/symbolic_v3_precision/
python plot_symbolic_v3.py
python tests_symbolic_v3_improved.py
```

`PROJECT_ROOT` overrides the project root for all scripts.
