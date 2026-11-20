# Symbolic stream v4 and neural-symbolic fusion on URFD

## 1. Result in brief

Measured, under the protocol in `PREREGISTRATION.md`:

- The symbolic stream improved on its own. Pooled MCC, validation / test: original notebook (A) 0.098 / 0.248,
  v3 (B) 0.214 / 0.403, v4 (C) 0.316 / 0.498. The ordering is the same on validation and on test.
- The pre-registered fusion did not beat the neural ensemble. Each fold's validation chose logistic stacking
  (F6). On the test folds it reached MCC 0.489 against 0.547 for the ensemble alone (paired bootstrap over
  videos: -0.059, 95% interval -0.121 to -0.007). Under our own rule, the final system stays the neural ensemble.
- Post-hoc observation, not a result: a parameter-free gated veto (F4, drop a neural alarm when the Kinect
  neck height is at least 0.8 m) reached MCC 0.634 on the test folds (+0.086, interval +0.014 to +0.154),
  but it was slightly worse than the ensemble on both validation sets (0.600 vs 0.611 and 0.796 vs 0.816),
  so the protocol could not select it. It is a hypothesis to test on new data, not a claimed improvement.

All test numbers are post-hoc in any case: the URFD test folds were seen in earlier stages of the project.

## 2. Audit of the symbolic stream before v4

Ranked by impact. "Confirmed" means checked in code or data.

| # | Problem | Where | Status |
|---|---|---|---|
| 1 | Rule needs "low" and "fast" at the same frame; a falling neck is fast while high and slow once on the floor. The training grid then moved tau_h to 0.75 m, which is no longer "near the floor" | v3 trigger | Confirmed |
| 2 | Fall state needs the neck to be low already, so every frame of the falling phase is missed (600 of 600 early fall frames) | v3 state | Confirmed |
| 3 | Hard labels only, no confidence; thresholds from a 225-point MCC grid on training videos | v3 | Confirmed |
| 4 | Original notebook: height measured in the image (not metres), non-causal (whole-video percentiles, centred smoothing, interpolation from future frames), hip fallback placed the neck above the hip while lying, leave-one-room-out instead of our two folds, 25 instead of 30% confidence threshold | notebook | Confirmed (earlier review) |
| 5 | No explicit conflict resolution: overlapping conditions were combined ad hoc | all versions | Confirmed |
| 6 | Missing heights were treated as "not low" silently | v3 | Confirmed |

## 3. Leakage audit

| Check | Finding | Action |
|---|---|---|
| Train/val/test overlap inside a fold | None (test 4 in `tests_symbolic_v4.py`) | none |
| Room shared between train and test | None | none |
| Cross-fold overlap | Fold 1 VAL (room A8 + falls 13, 19, 20, 23) lies inside fold 0 TEST and fold 0 VAL inside fold 1 TEST. Any choice made on both folds' validation sees test videos. This also affected the earlier ST-GCN experiment selection (exp04_center was kept with a rule over both folds) | Confirmed. From v4 on, every choice for fold k uses fold k data only |
| Symbolic thresholds | Fitted on fold k TRAIN only | none |
| Neural predictions | VAL: fold k ensemble on fold k VAL (never trained on them). TEST: out-of-fold file. No in-sample prediction is used | none |
| Fusion parameters | F3 weight and F6 stacking fitted on fold k VAL only; F6's validation score is leave-one-video-out | none |
| Implementation bug | First v4 run read URFD 3-class labels from the ADL file too (0/1 there mark postures), which polluted the training "lying" set (tau_low 1.34 m in fold 1). Fixed to the registered definition; both runs are kept (`results_run1_bug.json`) | Confirmed and fixed (Amendment 1) |
| Test-set reuse | The URFD test folds were used in earlier stages | All test numbers labelled post-hoc |
| Subject leakage | URFD has 5 actors and no subject IDs | Cannot be checked |

## 4. Error analysis of the neural ensemble (test folds, out-of-fold)

- Calibration is reasonable: ECE 0.066 (validation) and 0.075 (test); ROC-AUC 0.931 / 0.841.
- False alarms: 923 frames, 733 of them in room A4 (the dark room), 173 in A7. 529 of the 923 happen while
  the Kinect neck height is at least 0.8 m, which is not compatible with lying on the floor.
- Missed fall frames: 581, of which 412 are in URFD's "falling" phase and 169 in "lying"; 267 happen while the
  neck is still at least 0.8 m high (the label starts before the body goes down). The symbolic stream cannot
  help there.
- Confident mistakes: 416 false-alarm frames with probability >= 0.9 and 290 missed frames with <= 0.1.
- Two fall videos are never detected from the onset on (fall-03, fall-11).

## 5. Symbolic v4

A causal state machine on the neck height in metres (Kinect floor plane + RTMO neck, validated against URFD's
own floor distance: correlation 0.988 on falls, 0.929 on ADL). States in priority order (the conflict rule):
UNKNOWN, FALL, DESCENDING, UPRIGHT, LOW_NOFALL, MID. Thresholds come from fold k training labels:

| | tau_low (m) | delta_fall (m) | delta_on (m) |
|---|---|---|---|
| fold 0 | 0.425 | 0.358 | 0.179 |
| fold 1 | 0.340 | 0.583 | 0.291 |

Confidence is the fall rate of training frames in each state (Laplace smoothed), for example FALL 0.998 and
0.995, UPRIGHT 0.061 and 0.051. On the test folds it is well calibrated (ECE 0.065).

## 6. Standalone results (pooled over both folds)

| Model | VAL MCC | TEST MCC | TEST precision | TEST recall | TEST falls | TEST ADL without alarm |
|---|---|---|---|---|---|---|
| F neural ensemble | 0.678 | 0.547 | 0.570 | 0.678 | 28/30 | 23/40 |
| A original notebook | 0.098 | 0.248 | 0.318 | 0.465 | 26/30 | 18/40 |
| B v3 | 0.214 | 0.403 | 0.735 | 0.280 | 30/30 | 23/40 |
| C v4 | 0.316 | 0.498 | 0.482 | 0.718 | 30/30 | 0/40 |
| D1 v4 without DESCENDING | 0.265 | 0.444 | 0.599 | 0.444 | 30/30 | 35/40 |
| D2 v4 without the unknown-height hold | 0.346 | 0.511 | 0.506 | 0.704 | 30/30 | 0/40 |
| E v4, calibrated p >= 0.5 | 0.213 | 0.383 | 0.481 | 0.468 | 30/30 | 27/40 |

Ablation: DESCENDING buys recall (falling phase) at the cost of many short alarms in ADL; dropping it gives
the most precise symbolic stream (1.0 false alarm per minute, 35 of 40 ADL videos clean).

## 7. Fusion (pooled; selection per fold on VAL)

| Option | VAL MCC | TEST MCC | TEST F1 | TEST precision | TEST recall | falls | ADL clean |
|---|---|---|---|---|---|---|---|
| F neural alone | 0.678 | 0.547 | 0.619 | 0.570 | 0.678 | 28 | 23 |
| G = F1 + A (paper rule, original symbolic) | 0.098 | 0.204 | 0.346 | 0.276 | 0.465 | 26 | 18 |
| F1 + B | 0.455 | 0.448 | 0.520 | 0.580 | 0.471 | 30 | 28 |
| F1 + C | 0.358 | 0.447 | 0.529 | 0.405 | 0.763 | 30 | 0 |
| F2 mean + E | 0.428 | 0.528 | 0.604 | 0.548 | 0.672 | 30 | 24 |
| F3 weighted (w=0.6) + E | 0.505 | 0.564 | 0.632 | 0.608 | 0.658 | 29 | 26 |
| F3 weighted (w=0.9) + E | 0.645 | 0.545 | 0.617 | 0.568 | 0.677 | 28 | 23 |
| F4 gated veto + C | 0.660 | 0.634 | 0.683 | 0.743 | 0.632 | 28 | 23 |
| F5 veto + rescue + D2 | 0.503 | 0.614 | 0.673 | 0.654 | 0.694 | 30 | 21 |
| **F6 stacking + E (selected in both folds)** | 0.844 (LOO) | **0.489** | 0.571 | 0.512 | 0.647 | 28 | 23 |

The complete table, per-fold values, macro/weighted F1, log loss and calibration are in `results.json`.

Why stacking failed: fitted on 8 or 9 validation videos, it learned a negative weight for the symbolic
probability in fold 0 (-4.36). Refitting without each validation video in turn keeps it between -3.65 and
-5.0, so it is a property of that validation room (A7, where people kneel and lie down on purpose and the
symbolic stream often says FALL), not of a single video. On the test rooms the relation does not hold.

## 8. Complementarity (test folds, neural vs symbolic hard output)

| Symbolic | both right | neural only | symbolic only | both wrong | double fault | error correlation | Q |
|---|---|---|---|---|---|---|---|
| A original | 8478 | 1954 | 697 | 807 | 0.068 | 0.275 | 0.668 |
| B v3 | 9444 | 988 | 1011 | 493 | 0.041 | 0.235 | 0.647 |
| C v4 | 9039 | 1393 | 998 | 506 | 0.042 | 0.184 | 0.534 |

v4 has the lowest error correlation with the network and catches 279 fall frames the network misses, against
91 for the original notebook.

## 9. What the evidence supports

- Supported: v4 is a better symbolic stream than v3 and the original (both validation and test agree), and
  its errors overlap least with the network's.
- Not supported: any claim that a fusion beats the ensemble. The pre-registered choice lost on test.
- Hypothesis for the next step: the gated veto F4. It has no fitted parameter and a physical justification
  (a neck at 0.8 m or higher is not on the floor), and it removed 529 of the network's 923 test false alarms.
  It needs a pre-registered test on data not used so far. GMDCSA-24 cannot serve because it has no depth
  sensor; a dataset with depth (for example CMDFall) or new recordings would.
- The validation sets (8 and 9 videos per fold) are too small to choose between fusion methods reliably.

## 10. Reproduce

From the project root:

```
python3 fall_detection_models/code/stgcn_urfd/kinect_neck_height.py          # once, metric neck height
python3 fall_detection_models/code/stgcn_urfd/tests_symbolic_v4.py           # 23 checks
python3 fall_detection_models/code/stgcn_urfd/fusion_v4_experiments.py       # all candidates, about 8 minutes
python3 fall_detection_models/code/stgcn_urfd/fusion_v4_plots.py             # figures and paired bootstrap
```

Neural inputs: `reports/symbolic_v4/neural_val.csv` (each fold's ensemble on its own validation videos) and
`neural_test_oof.csv` (`neural_ensemble.py --mode oof`). Files: `symbolic_v4.py` (rules), `configs.json`
(fitted thresholds and state probabilities per fold), `per_frame_errors.csv` (one row per test frame: labels,
neural probability and prediction, symbolic state, confidence and prediction, correctness of each).

Figures: `figures/confusion_test.png`, `figures/error_overlap.png`, `figures/timelines.png`.
