# V3-norm: smoothed speed and relative height (results)

Pre-registration: `PREREGISTRATION.md` (commit ca19ac0), written before any run. Selection used only TRAIN+VAL in each fold (inner group CV, same rule as the V3 precision study), and TEST was scored once. No amendments.

## URFD TEST (pooled over both folds, frame level)

| Model | Precision | Recall | F1 | Accuracy | Bal. acc. | MCC | Falls detected | Clean ADL |
|---|---|---|---|---|---|---|---|---|
| Improved V3 (reference) | 0.796 | 0.396 | 0.528 | 0.893 | 0.689 | 0.513 | 30/30 | 28/40 |
| **V3-norm** (smoothing + relative height) | 0.710 | 0.347 | 0.466 | 0.880 | 0.661 | 0.441 | 30/30 | 28/40 |
| Ablation: smoothing only | 0.667 | 0.377 | 0.482 | 0.877 | 0.672 | 0.441 | 30/30 | 25/40 |
| Ablation: relative height only | 0.782 | 0.391 | 0.521 | 0.891 | 0.686 | 0.503 | 30/30 | 30/40 |

Chosen configs. Selection picked NC with delta 0.50 m in both folds.

| Fold | tau_h (relative) | tau_h (m at 1.45 m) | tau_v (relative/s) |
|---|---|---|---|
| 0 | 0.414 | 0.60 | 0.17 |
| 1 | 0.310 | 0.45 | 0.50 |

## Decision (pre-registered)

1. MCC 0.441 is below 0.513 - 0.02 = 0.493: **fails**.
2. The tau_h gap between folds is 0.15 m, against the old 0.20 m: passes.

**V3-norm does not replace the improved V3.** The improved V3 stays the active symbolic model.

The relative-height-only ablation would have passed the MCC rule (0.503). It was not a candidate under the pre-registration, and choosing it now, after seeing TEST, would be selection on test. It is recorded here as a lead only.

## FUKinect-Fall (not blind, as stated in the pre-registration)

| Config | Falls detected | ADL alarms | MCC | Lying alarms |
|---|---|---|---|---|
| fold 0, improved V3 | 132/167 (0.79) | 149/839 (0.18) | 0.508 | 0.79 |
| fold 0, V3-norm | 124/167 (0.74) | 150/839 (0.18) | 0.471 | 0.77 |
| fold 1, improved V3 | 17/167 (0.10) | 3/839 (0.004) | 0.262 | 0.02 |
| fold 1, V3-norm | 29/167 (0.17) | 17/839 (0.02) | 0.273 | 0.10 |

Neither config passes the 0.80 / 0.45 rule. The changes did not fix the problem found there.

## Why it did not work (post-hoc reading)

- **Relative height changes almost nothing between these two datasets.** Standing neck height is about 1.45 m on URFD and 1.47 m on FUKinect. The real difference is how high the Kinect v1 skeleton puts the neck of a person on the floor (median 0.38 m), not how tall people are. Normalising by standing height cannot move that.
- **The fold disagreement is still there.** Fold 1's selection data still pushes tau_h down, to 0.45 m against 0.40 m before. The gap narrowed but did not close.
- **The median filter cost precision on URFD.** Precision dropped from 0.80 to 0.71 and recall from 0.40 to 0.35. The causal filter adds about 3 frames of delay at onset and keeps the neck low a little longer afterwards. On FUKinect it did not change the false alarms on lying (0.79 to 0.77), so the speed condition still does not separate lying down from falling there.

The lying-down ADL remains the hard case for any height and speed rule. What would separate it from a fall is how the body reaches the floor (a controlled descent through sitting or kneeling versus an uncontrolled one), and V3 does not encode that.

## Files

- `results.json`: inner CV for every candidate, chosen models, TEST metrics, ablations, FUKinect.
- `final_config.json`: V3-norm configs per fold (kept for the record, not active).
- Code: `symbolic_v3_norm.py`, `evaluate_v3_norm.py`.
