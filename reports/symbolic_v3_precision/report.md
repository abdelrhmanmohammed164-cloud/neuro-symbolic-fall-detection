# Symbolic V3: precision study

## 1. Result in brief

Protocol: `PREREGISTRATION.md` (committed before any number). Selection used inner group cross-validation on each
fold's training and validation videos only; the test folds were scored once at the end (post-hoc, as all URFD
test numbers are, because the test folds were seen in earlier stages).

| Test folds, 70 videos | Precision | Recall | F1 | Accuracy | Balanced acc. | MCC | TP | FP | FN | TN | ADL without alarm | Alarms / min |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| A original V3 | 0.736 | 0.255 | 0.379 | 0.874 | 0.619 | 0.384 | 460 | 165 | 1343 | 9968 | 20/40 | 4.83 |
| **E improved V3** | **0.796** | **0.396** | **0.528** | **0.893** | **0.689** | **0.513** | 713 | 183 | 1090 | 9950 | **28/40** | **2.62** |

Every metric improved, in both folds (precision fold 0: 0.716 to 0.728, fold 1: 0.780 to 0.910; recall fold 0:
0.349 to 0.464, fold 1: 0.165 to 0.330). The number of false-positive frames stayed about the same (165 to 183)
while true positives rose by 55%, so precision rose without trading recall away. All 30 falls are still detected.

The selected configuration differs per fold, which is the honest outcome of selecting per fold:
fold 0 = D (delta 0.30 m: tau_h 0.60 m, tau_v 0.89 m/s, latch below 0.445 m);
fold 1 = C (delta 0.40 m: tau_h 0.40 m, tau_v 0.73 m/s). See `final_config.json`.

## 2. Why V3's precision was low (audit)

| # | Finding | Evidence | Fix |
|---|---|---|---|
| 1 | Eq. 8 and Eq. 9 were required at the same frame. A falling neck moves fast while still high and slowly once on the floor, so the training grid pushed tau_h to 0.75 to 0.80 m to catch any fall at all. At that height the "floor" rule also fires on crouching, kneeling and sitting on the floor | A's thresholds: tau_h 0.80 / 0.75 m. 138 of A's 236 inner-CV false-alarm frames have the neck between 0.5 and 0.8 m | Variant B: Eq. 9 over the last 1.0 s |
| 2 | No negative evidence. A fast partial lowering (crouch, kneel, sit down quickly) passes both rules | All 236 false-alarm frames are in ADL videos; 169 of them follow a drop of at least 0.4 m within 1.5 s; the most affected videos are adl-38, adl-30 and adl-10 | Variant C: the neck must have dropped at least delta within 1.5 s, and tau_h <= 0.60 m |
| 3 | V3's latch (state) was a separate output, not part of the decision, so recall stayed low once the body stopped moving | A recall 0.255; lying frames after the impact are missed | Variant D: the latch extends a trigger while the neck stays low |
| 4 | Missing heights: handled correctly (never "low") | tests_symbolic_v3_improved.py | none |
| 5 | Empty video crashed the new vectorised window function (found by the new tests) | tests_symbolic_v3_improved.py | fixed |
| 6 | Label mapping, causality, thresholds from training only | tests_symbolic_v3_improved.py, including equivalence with the original V3 (identical thresholds and predictions) | none needed |

The chosen variant removes 171 of A's 236 inner-CV false-alarm frames, mostly in the 0.5 to 0.8 m band
(135 of 138), which is exactly the crouch/kneel band. It removes fewer near the floor (7 of 32 below 0.3 m): there the neck really is at floor level, which a height rule
alone cannot separate from a fall.

## 3. Comparison with the paper

| | Accuracy | Precision | Recall | F1 | MCC |
|---|---|---|---|---|---|
| Paper, symbolic stream (Table 4) | 0.77 | 0.99 | 0.21 | 0.35 | 0.40 |
| P: paper rule on our data (tau_h 0.19 m, same frame) | 0.846 | 0.266 | 0.009 | 0.018 | 0.024 |
| A: original V3 | 0.874 | 0.736 | 0.255 | 0.379 | 0.384 |
| E: improved V3 | 0.893 | 0.796 | 0.396 | 0.528 | 0.513 |

What the paper specifies: neck height H in metres from a 4x4 matrix W fitted on 11 control points measured with a
3D scanner (Eq. 1, 7); H <= 0.19 m (Eq. 8); vertical speed <= -theta (Eq. 9); AND with temporal smoothing (Eq. 10).
It does not give theta, the speed interval or the smoothing window. Training data: 500 simulated clips; test: 50
private videos of 5 people (36 falls, 14 non-falls).

Why the paper's precision is higher, ranked by what we could check:
1. Prevalence. Precision depends on how many test frames are falls. Their test has 36 fall videos out of 50; ours
   has 30 out of 70 and 15.1% fall frames. With E's own error rates (TPR 0.396, FPR 0.018), precision would be
   0.904 at 30% fall frames and 0.956 at 50% (computed, not measured). Much of the gap is the test mix, not the rule.
2. The paper's rule does not transfer as written. With 0.19 m and the same-frame AND, it fires on 0.5% of frames
   and finds 4 of 30 falls (variant P), because the neck is slow by the time it is below 0.19 m. Their speed is
   probably measured over a longer interval or their height differs (W from control points vs our Kinect floor
   plane; neck definition), but the paper does not say.
3. Different negatives. Their 14 non-fall videos are surveillance scenes; URFD's ADL videos are dominated by
   crouching, kneeling and lying down on purpose, the hardest negatives for a height rule.
4. Labels. URFD counts the whole lying period after a fall as positive; the paper does not say how it labels it.

These datasets and protocols differ, so the numbers are not directly comparable.

## 4. Leakage report

| Check | Result |
|---|---|
| Train/validation/test overlap, room overlap | none (tests_symbolic_v3_improved.py) |
| Inner CV: ADL room in both inner train and held-out | none; near-duplicates (adl-25/27/29) never split |
| Thresholds and delta | fitted on inner-train (selection) or SELECTION(k) (final), never on held-out labels |
| Test folds in selection | not used; scored once after the choice |
| Cross-fold overlap (fold 1 VAL inside fold 0 TEST) | avoided: selection for fold k uses fold k data only |
| Neural predictions | not used for selection; only for the complementarity description (out-of-fold) |
| Earlier exposure of the test folds | yes, so test numbers are post-hoc |
| Subject leakage | not checkable (URFD has no subject IDs) |

## 5. All candidates

Inner CV (selection data only), pooled:

| Candidate | fold 0 P / R / F1 | fold 1 P / R / F1 |
|---|---|---|
| A | 0.567 / 0.258 / 0.355 | 0.823 / 0.285 / 0.423 |
| P | 0.096 / 0.005 / 0.010 | 1.000 / 0.014 / 0.027 |
| B | 1.000 / 0.414 / 0.585 | 0.441 / 0.365 / 0.400 |
| C 0.30 | 0.708 / 0.510 / 0.593 | 0.631 / 0.353 / 0.452 |
| C 0.40 | 0.784 / 0.478 / 0.594 | **0.666 / 0.350 / 0.459** |
| C 0.50 | 0.849 / 0.403 / 0.546 | 0.675 / 0.337 / 0.449 |
| D 0.30 | **0.613 / 0.620 / 0.617** | 0.489 / 0.359 / 0.414 |
| D 0.40 | 0.614 / 0.591 / 0.602 | 0.557 / 0.359 / 0.437 |
| D 0.50 | 0.710 / 0.531 / 0.608 | 0.628 / 0.342 / 0.443 |

Bold: the selected option (best F1 above the recall floor of 0.9 x A's recall). In fold 1 the chosen option has
lower inner-CV precision than A (0.666 vs 0.823) but higher recall and F1; on the test fold its precision was
higher (0.910 vs 0.780). With only 2 or 3 inner folds, fold-to-fold spread is large (for example D 0.30 in fold 1:
F1 0.471, 0.492 and 0.296), so small differences between candidates are not meaningful.

Test folds, every candidate refitted on its selection set (for transparency; not used to choose):

| Candidate | Precision | Recall | F1 | Accuracy | Bal. acc. | MCC | FP | ADL clean |
|---|---|---|---|---|---|---|---|---|
| A | 0.736 | 0.255 | 0.379 | 0.874 | 0.619 | 0.384 | 165 | 20 |
| P | 0.266 | 0.009 | 0.018 | 0.846 | 0.502 | 0.024 | 47 | 38 |
| B | 0.632 | 0.428 | 0.510 | 0.876 | 0.692 | 0.453 | 449 | 21 |
| C 0.30 | 0.709 | 0.393 | 0.505 | 0.884 | 0.682 | 0.471 | 291 | 26 |
| C 0.40 | 0.708 | 0.380 | 0.495 | 0.883 | 0.676 | 0.462 | 282 | 26 |
| C 0.50 | 0.717 | 0.356 | 0.476 | 0.882 | 0.666 | 0.450 | 253 | 27 |
| D 0.30 | 0.737 | 0.473 | 0.576 | 0.895 | 0.721 | 0.536 | 304 | 28 |
| D 0.40 | 0.745 | 0.471 | 0.577 | 0.896 | 0.721 | 0.539 | 291 | 28 |
| D 0.50 | 0.774 | 0.455 | 0.573 | 0.898 | 0.716 | 0.543 | 239 | 29 |
| **E selected** | **0.796** | 0.396 | 0.528 | 0.893 | 0.689 | 0.513 | 183 | 28 |

Ablation reading: the 1 s descent window (B) buys recall but alone lets many crouches through; the drop
requirement (C) brings precision back; the latch (D) adds recall at a small precision cost.

## 6. Overfitting and robustness

- Complexity: at most 4 conditions and 3 fitted numbers per fold; delta takes one of 3 preset values. No rule
  refers to a video, room or person.
- Stress tests on the test folds with the selected models fixed (`figures/stress_tests.png`):
  20% of frames missing, height scale x0.9 / x1.1, floor offset +/-5 cm change E's MCC by -0.04 to 0.00.
  Height noise hurts E more than A: with 3 cm noise precision falls to 0.658 (MCC 0.463), with 6 cm to 0.523
  (MCC 0.388), still above A's MCC at the same noise (0.375, 0.366). Reason: the 1 s descent window and the 1.5 s
  drop take maxima, which pick up noise spikes. A new sensor with noisier depth than the Kinect would need a
  short causal smoothing of the height before the rules (not tested here, so not part of E).
- Only one dataset with depth is available (GMDCSA-24 has no depth sensor), so no leave-one-dataset-out test was
  possible. The stress tests cannot prove performance on a new dataset.

## 7. Complementarity with the neural ensemble (test, description only)

| Symbolic | both right | neural only | symbolic only | both wrong | error correlation | fall frames neural missed, symbolic caught | symbolic false alarms where neural was right |
|---|---|---|---|---|---|---|---|
| A | 9448 | 984 | 980 | 524 | 0.254 | 69 | 153 |
| E | 9779 | 653 | 884 | 620 | 0.376 | 107 | 37 |

E is right more often on its own and makes far fewer false alarms the network does not make (37 vs 153), and it
catches more of the fall frames the network misses (107 vs 69). Its errors correlate more with the network's
(0.376 vs 0.254). Whether this helps
a fusion has to be tested with its own protocol.

## 8. Using the improved V3 on a new dataset

1. Produce the neck height above the floor in metres per frame at 30 fps: a depth sensor plus a floor plane
   (as `kinect_neck_height.py` does) or a calibrated camera (paper Eq. 1). Without metric height the rules do not
   apply; do not substitute image height.
2. Do not refit. Use the thresholds in `final_config.json`. For a single model across both folds, prefer fold 0's
   D configuration if the new data has long lying periods, fold 1's C if alarms are costly; report which.
3. Missing heights are never treated as low; videos at other frame rates must be resampled to 30 fps.
4. If the depth is noisier than a Kinect v1 (check the floor-plane RMS; ours is 0.5 to 0.9 cm), smooth the height
   causally first and report it as a deviation.

## 9. Reproduce

```
python3 fall_detection_models/code/stgcn_urfd/kinect_neck_height.py         # once
python3 fall_detection_models/code/stgcn_urfd/tests_symbolic_v3_improved.py  # 32 checks
python3 fall_detection_models/code/stgcn_urfd/evaluate_symbolic_v3.py        # whole study, a few minutes
python3 fall_detection_models/code/stgcn_urfd/plot_symbolic_v3.py            # figures
```
Files: `symbolic_v3_improved.py` (model), `symbolic_v3.py` (untouched original; byte-identical backup in `archive/old_symbolic_code/symbolic_v3_original.py`),
`results.json`, `final_config.json`, `false_positives_A_inner_cv.csv`, `figures/`.
Notebooks are no longer part of the active project (moved to `archive/notebooks`).
