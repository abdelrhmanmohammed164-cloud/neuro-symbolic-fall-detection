# Improved V3 on FUKinect-Fall: external test

Pre-registration: `PREREGISTRATION.md` (commit d54bc45, written and committed before any V3 output). Both frozen configs from `symbolic_v3_precision/final_config.json` were run once each with no refitting. The secondary analyses are as pre-registered. The section marked post-hoc was added after the results.

## Data and height check

- 1006 clips (167 falls, 839 ADL) from 21 subjects who are not in URFD.
- The floor was fitted per subject on the foot joints, using no labels. The camera height came out at 0.87 to 0.93 m (the tripod is 95 cm; the gap is the ankle joint sitting a few cm above the floor), with a tilt under 2 degrees.
- The standing neck height has a median of 1.47 m, so the metric scale matches URFD.

## Primary result (clip level, an alarm on any frame counts as a fall)

| | Falls detected | ADL with alarm | Precision | Recall | F1 | MCC | Bal. acc. | Pre-registered rule |
|---|---|---|---|---|---|---|---|---|
| URFD test (reference) | 30/30 (1.00) | 12/40 (0.30) | 0.714 | 1.000 | 0.833 | 0.707 | 0.850 | n/a |
| fold 0 config (D) | 132/167 (0.79) | 149/839 (0.18) | 0.470 | 0.790 | 0.589 | 0.508 | 0.806 | fail (detection 0.79 < 0.80) |
| fold 1 config (C) | 17/167 (0.10) | 3/839 (0.004) | 0.850 | 0.102 | 0.182 | 0.262 | 0.549 | fail (detection 0.10) |

The URFD clip-level figures come from the same any-alarm rule on the 70 URFD test videos. They are not the frame-level numbers in the V3 precision report.

Alarm rate per action for the fold 0 config:

| Action | walking | bending | sitting | squatting | lying | falling |
|---|---|---|---|---|---|---|
| Alarm rate | 0.00 | 0.00 | 0.00 | 0.10 | 0.79 | 0.79 |

The fold 1 config fires on 1.8% of lying clips and on 10% of falls.

**Verdict under the pre-registered rule:** neither config passes. The fold 0 config misses by two clips: 134 detected falls would have reached 0.80. Its false-alarm rate (0.18) is well inside the limit. The fold 1 config fails clearly.

## Secondary analyses (pre-registered)

| | fold 0: detection / FA / MCC | fold 1: detection / FA / MCC |
|---|---|---|
| Primary (30 fps, floor per subject) | 0.790 / 0.178 / 0.508 | 0.102 / 0.004 / 0.262 |
| Signal assumed 20 fps | 0.802 / 0.180 / 0.514 | 0.108 / 0.005 / 0.262 |
| One global floor | 0.790 / 0.176 / 0.510 | 0.114 / 0.004 / 0.280 |

The results do not depend on the frame-rate assumption or on how the floor is fitted.

Per subject, the fold 0 config detected all 8 falls for 9 subjects, but only 2 of 8 for subjects 14 and 20.

## What went wrong (post-hoc, descriptive)

1. **The height threshold does not transfer.** On this sensor, the lowest neck height in a fall clip has a median of 0.38 m (10th to 90th percentile: 0.16 to 0.54 m).
   - 93% of falls go below 0.60 m, but only 54% go below 0.40 m.
   - That one number explains the split between the configs. On URFD, the two folds chose tau_h = 0.60 and 0.40 almost interchangeably.
   - Here the choice decides whether V3 works at all. This is the clearest sign of overfitting in this test: the thresholds were fitted to URFD's height scale, not to any individual URFD video.
2. **The descent-speed condition is always on.** Peak descent speed from the Kinect v1 skeleton exceeds the fold 0 tau_v (0.89 m/s) in 100% of falling, lying and squatting clips. The medians are 4.2, 2.4 and 3.0 m/s, and spikes reach 6 m/s.
   - Those values reflect skeleton jitter and jumps, not real motion. The descent speed in V3 is a 0.2 s finite difference with no filtering.
   - So on this data V3 effectively reduces to "neck low and dropped by delta". That is why it cannot separate a deliberate lie-down (79% alarms) from a fall (79% detected).
3. **The rest of the ADLs are clean.** Walking, bending and sitting get no alarms at all, and squatting gets 10%. The rules that cut URFD's false positives still work wherever the body stays well above the floor.

## Conclusion

The improved V3 did not just memorize URFD's 70 videos. With the fold 0 config it scores almost the same MCC on new people and a new sensor (0.508 clip-level here, against 0.513 frame-level on URFD). It also keeps the false-alarm rate low on everything except lying down. The weak point is the absolute height and speed thresholds:

- they are fitted to one sensor's measurement scale;
- the two URFD folds disagree on tau_h;
- that disagreement becomes a 0.79 vs 0.10 gap in fall detection here.

Any next version should fix this, for example with speed computed on a smoothed signal and a height threshold relative to the person's own standing height. Such a version would need its own pre-registration and a dataset other than this one for the final check, because these results have now been seen.

## Files

- `results.json`: all metrics, per-subject counts and sensitivity runs.
- `clips_fold0_D.csv`, `clips_fold1_C.csv`: one row per clip (action, subject, repeat, label, alarm, time of first alarm).
- Code: `fukinect_neck_height.py`, `evaluate_v3_fukinect.py`.
