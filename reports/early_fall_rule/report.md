# Optional early-fall rule for the improved Symbolic V3

## Recommendation
Keep the improved V3 unchanged (flag off). The early-fall rule is implemented and can be switched on, but the
evidence does not support it: it helped in one development fold and hurt in the other, its thresholds differ
threefold between folds, it barely fires on the untouched GMDCSA-24 dataset, and on the URFD test folds it recovered
no fall frame that the neural network missed while adding false alarms.

## 1. Diagnosis
At fall onset the neck is still high, so V3's height rules cannot fire, and the network is not yet confident.
A second cause showed up during this study and is outside this phase (the fusion is unchanged): the veto in the
fusion removes neural alarms while the neck is at 0.8 m or higher, which is exactly the first part of a fall.
On the validation videos (never used for training the network) the veto removed 28 of the 337 fall frames the
network had caught and moved the median onset delay from 0.35 s to 0.58 s; on the test folds the network alone
catches 54% of the first second of each fall, the fusion 40%.

## 2-5. The rule, its definition and justification
Pose features from RTMO (COCO-17, confidence >= 0.3), causal, no pixel thresholds:
- theta(t): angle between hip midpoint -> shoulder midpoint and the image's upward vertical (degrees).
- B: inclination speed w(t) = (theta(t) - theta(t - 0.2 s)) / 0.2 s. A falling torso rotates fast toward horizontal
  before the neck is low.
- C: normalised hip descent u(t) = (P_y(t) - P_y(t - 0.2 s)) / Lref(t) / 0.2 s, in torso lengths per second, with
  Lref the median torso length over the past 2 s (removes camera distance and body size). The pelvis drops fast
  at fall onset.
- D: B and C within the same 0.3 s.
- Trigger after 3 consecutive frames above threshold (keypoint jitter), alarm held 1.0 s (URFD's falling phase),
  output = V3 OR alarm (it can only add positives). Thresholds: 99th percentile over ADL frames of the fitting videos.
- Activities that move the same way: sitting down fast, bending to pick something up, lying down on a bed. The
  99th-percentile threshold and the 3-frame requirement are the false-alarm controls.

## 6-7. Code and flag
`early_fall.py` (features, `EarlyRule`, `fit`, `V3WithEarly(v3_model, early=None)`; `early=None` is the off switch
and reproduces V3 exactly, checked by `tests_early_fall.py`, 14 tests). V3 files were not modified.
`evaluate_early_fall.py` runs the whole study.

## 8-9. Development results (inner group CV on each fold's train + validation videos only)

| Fold | Variant | Precision | Recall | F1 | ADL clean | Recall in the first second after onset |
|---|---|---|---|---|---|---|
| 0 | A (V3) | 0.613 | 0.620 | 0.617 | 13/17 | 0.139 |
| 0 | B | 0.598 | 0.682 | 0.637 | 13/17 | 0.292 |
| 0 | C | 0.578 | 0.632 | 0.604 | 13/17 | 0.164 |
| 0 | D | 0.614 | 0.622 | 0.618 | 13/17 | 0.139 |
| 1 | A (V3) | 0.666 | 0.350 | 0.459 | 13/23 | 0.044 |
| 1 | B | 0.400 | 0.388 | 0.394 | 8/23 | 0.093 |
| 1 | C | 0.451 | 0.359 | 0.400 | 9/23 | 0.056 |
| 1 | D | 0.560 | 0.354 | 0.434 | 12/23 | 0.050 |

Pre-registered decision: fold 0 accepts B (higher recall and F1, precision drop 0.015 within A's inner-fold
spread of 0.266); fold 1 accepts nothing (every variant lowers F1 and adds alarms to 1 to 5 ADL videos).
Thresholds: fold 0 w > 203 deg/s, u > 2.7 torso lengths/s; fold 1 w > 626 deg/s, u > 5.4. A threefold difference
from the same percentile rule means the ADL motion statistics depend heavily on the room (fold 1 includes the
dark room A4, where keypoints jitter), so the threshold would not transfer.

Fusion on the validation videos (veto+sym unchanged): fold 0 MCC 0.395 (A) vs 0.394 (B); fold 1 0.718 (A) vs
0.320 (B), 0.410 (C), 0.545 (D). No variant improved the fusion in either fold.

## 10. Complementarity on the URFD test folds (reported once; not independent for this change)

| | Symbolic P | Symbolic R | Symbolic FP | Neural FN caught by symbolic | Shared FP | Fusion MCC | Fusion FP |
|---|---|---|---|---|---|---|---|
| A (V3) | 0.796 | 0.396 | 183 | 107 | 146 | 0.651 | 435 |
| B | 0.519 | 0.423 | 706 | 107 | 160 | 0.536 | 944 |
| C | 0.558 | 0.402 | 574 | 107 | 179 | 0.560 | 821 |
| D | 0.656 | 0.399 | 377 | 107 | 146 | 0.602 | 629 |
| Per-fold choice (B in fold 0, A in fold 1) | 0.546 | 0.423 | 635 | 107 | 148 | 0.547 | 885 |

The new alarms fall on frames the network already flags, so not one extra neural miss is recovered (107 in every
row), while the false alarms rise. Event level: all variants detect 30/30 falls; fusion onset delay is unchanged
(median 0.55 s).

## 11. External check (GMDCSA-24, untouched, the rule alone, thresholds from the 70 URFD videos)

| Variant | Falls with an alarm from onset | ADL videos without alarm | Alarms / min |
|---|---|---|---|
| B | 5/79 | 79/81 | 0.24 |
| C | 9/79 | 81/81 | 0.00 |
| D | 2/79 | 81/81 | 0.00 |

The rule causes almost no false alarms on new homes and cameras, but it also misses almost every fall there:
URFD's 99th percentiles are too high for a laptop webcam at a different distance and frame rate. The feature
does not generalise with a fixed threshold.

## 12. Generalisation risks and what would help
- The thresholds depend on the room's normal motion and on keypoint jitter, which change with every camera.
- A pose-only early cue overlaps with what the ST-GCN already sees (it is trained on the same keypoints), so it
  adds little complementary information.
- The larger lever for early falls is in the fusion: exempting frames with a fast neck descent from the veto would
  keep the network's early alarms. That changes the fusion, so it belongs to a separate, pre-registered phase.
