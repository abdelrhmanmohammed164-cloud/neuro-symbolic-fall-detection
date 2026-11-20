# Veto exemption during a fast descent: result

Rule (pre-registered): the fusion's veto (drop a neural alarm when the neck is at 0.8 m or higher) is not applied
while the neck has descended faster than the fold's V3 tau_v within the last 1.0 s.

## Validation (decision data)
| Fold | Rule | MCC | Recall | FP | First-second recall | Median onset delay |
|---|---|---|---|---|---|---|
| 0 | F0 current | 0.395 | 0.852 | 426 | 0.583 | 0.42 s |
| 0 | F1 exemption | 0.425 | 0.905 | 426 | 0.733 | 0.27 s |
| 1 | F0 current | 0.718 | 0.658 | 24 | 0.367 | 0.62 s |
| 1 | F1 exemption | 0.770 | 0.731 | 24 | 0.525 | 0.40 s |

The pre-registered rule adopts F1 in both folds.

## Test folds (post-hoc, scored once)
| Rule | Precision | Recall | F1 | MCC | FP | FN | Falls | ADL clean | First-second recall | Median onset delay |
|---|---|---|---|---|---|---|---|---|---|---|
| Neural alone | 0.570 | 0.678 | 0.619 | 0.547 | 923 | 581 | 28/30 | 23/40 | 0.542 | 0.33 s |
| F0 current | 0.735 | 0.668 | 0.700 | 0.651 | 435 | 598 | 30/30 | 24/40 | 0.404 | 0.55 s |
| F1 exemption | 0.669 | 0.686 | 0.677 | 0.619 | 613 | 566 | 30/30 | 21/40 | 0.440 | 0.47 s |

Per fold (MCC): fold 0 F0 0.643, F1 0.590; fold 1 F0 0.659, F1 0.659.

## Reading
- The exemption does what it was designed for: earlier detection (median onset 0.55 s -> 0.47 s), more fall frames
  (FN 598 -> 566). The cost on test is 178 more false-alarm frames, all in fold 0, so MCC drops 0.651 -> 0.619.
- Validation and test disagree. Validation is small (6 falls, 11 ADL videos) and, on validation, V3 fitted on the
  training videos alone had tau_v at the grid floor (0.25 m/s), so almost any downward motion lifted the veto there;
  the final V3 (fitted on train + validation) has tau_v 0.89 and 0.73 m/s. The rule tested on validation is
  therefore looser than the one applied on test.
- Decision by the protocol: F1. Decision by the test numbers: F0. The test is not independent, so neither can be
  confirmed on URFD; it is a trade-off between earlier alarms (F1) and fewer false alarms (F0).
