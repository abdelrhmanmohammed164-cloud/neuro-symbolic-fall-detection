# V3-norm: improved V3 with smoothed speed and relative height (pre-registration)

Written before any V3-norm output was computed on URFD. Changes after this point go in a dated amendment at the end, with the reason.

## Why

The external test on FUKinect-Fall (`reports/symbolic_v3_external_fukinect/report.md`) found two weak points in the improved V3:

1. The absolute height threshold is tied to one sensor's scale. The two URFD folds chose tau_h = 0.60 m and 0.40 m, and that choice alone moved FUKinect fall detection from 0.79 to 0.10.
2. Descent speed is an unfiltered 0.2 s difference, so skeleton jitter (2 to 6 m/s on FUKinect) keeps Eq. 9 on almost all the time.

## Changes (everything stays causal)

V3-norm keeps variants C and D exactly as they are. Only the signal they read changes.

- **Smoothing.** After V3's forward fill, the neck height goes through a causal running median over the last 7 frames (0.23 s), ignoring NaNs. Speed, drop and height are all computed on this smoothed signal.
- **Relative height.** Each frame's height is divided by the person's standing neck height, estimated causally.
  - The estimate is the 90th percentile of all valid smoothed heights from frame 0 up to t, clipped to [1.20, 1.70] m.
  - Until 10 valid frames exist, it is 1.45 m.
  - Height, drop and speed are all expressed in these relative units. Missing frames are still treated as high (2.0).
- **Grids.** V3's grids are divided by 1.45 m, the typical standing neck height on URFD, so the candidates cover the same physical range as before:
  - tau_h: HEIGHT_GRID / 1.45, capped at 0.60 / 1.45;
  - tau_v: VELOCITY_GRID / 1.45;
  - delta in {0.30, 0.40, 0.50} / 1.45.
- **tau_s.** As before, the 95th percentile of the training lying frames, now in relative units.

## Candidates and selection (same protocol as the V3 precision study)

- Pool: NC and ND, each with three delta values, so 6 candidates.
- Each outer fold k of `split_scene_v2.json` gets inner group CV on TRAIN(k) + VAL(k), with the same inner folds as `evaluate_symbolic_v3.py`.
- The selection rule is unchanged:
  - the recall floor is 90% of variant A's inner recall;
  - among candidates above the floor, the best inner F1 wins;
  - ties within 0.005 F1 go to higher precision, then to fewer conditions.
- The chosen candidate is refitted on TRAIN(k) + VAL(k) and scored once on TEST(k).

## Primary comparison

Pooled TEST over both folds, frame level, same `metrics()` as before: precision, recall, F1, accuracy, balanced accuracy and MCC, plus falls detected and clean ADL videos. The reference is the improved V3 (selected) from `reports/symbolic_v3_precision/results.json`: P 0.796, R 0.396, F1 0.528, MCC 0.513, 30/30 falls, 28/40 clean ADL.

V3-norm is reported as a replacement only if both hold:

1. TEST MCC is at least the improved V3's MCC minus 0.02;
2. the two folds' chosen tau_h, converted to metres at 1.45 m, differ by less than the improved V3's 0.20 m.

Otherwise the improved V3 stays the active symbolic model.

## Secondary (descriptive)

- Ablations on TEST using the same selection: smoothing only (absolute height) and relative height only (no smoothing).
- FUKinect-Fall with each fold's chosen config, frozen, using the same clip-level rule and the 0.80 / 0.45 thresholds as the external test.
  - This result is not blind: the changes were designed after seeing FUKinect results. It shows whether the fix addresses the problem found there. It is not a new independent test.
