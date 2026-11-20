# Symbolic V3 precision study: protocol fixed before any new number

Scope: Symbolic V3 only. V4 code, rules, thresholds and results are not used. No notebooks.
The original V3 stays untouched (`symbolic_v3.py`, byte-identical backup `symbolic_v3_original.py`).

## Data use
- URFD camera 0, 70 videos, neck height in metres from the Kinect (`kinect_neck_height.py`), binary frame labels
  (the ST-GCN labels), outer folds of `splits/split_scene_v2.json`.
- For outer fold k, SELECTION = TRAIN(k) + VAL(k). TEST(k) is used once, after the selection.
- Inner group cross-validation inside SELECTION(k): every ADL room is one group; the fall videos are split into
  as many groups as there are ADL rooms (sorted by video id, round robin). Inner fold i holds out ADL room i and
  fall group i. Near-duplicate videos (adl-25/27/29) share a room, so they never straddle inner folds.
- Thresholds are always fitted on the inner training part (or on SELECTION for the final model), never on
  held-out labels. No neural prediction is used for selection.
- Test numbers are post-hoc anyway (the URFD test folds were seen in earlier stages).

## Candidates (all keep the V3 structure: Eq. 8 height rule, Eq. 9 descent rule, Eq. 10 temporal smoothing)
- A  original V3, as is (grid tau_h 0.10-0.80 m x tau_v 0.25-2.5 m/s, best training MCC).
- P  paper reproduction: tau_h fixed at 0.19 m (paper Eq. 8), tau_v from the same grid, same-frame AND.
- B  bug fix: Eq. 9 is checked over the last 1.0 s instead of the same frame (a fall is fast while the body is
     still high and slow once it is on the floor; the same-frame AND forced tau_h up to 0.75 m).
- C  B + negative evidence: the neck must have dropped by at least delta (max over the last 1.5 s minus now),
     delta in {0.30, 0.40, 0.50} m; and the floor rule is restricted to tau_h <= 0.60 m (physical floor proximity).
- D  C + V3's own latch: once triggered, the output stays positive while the neck stays below tau_s
     (95th percentile of training 'lying' heights); a new positive needs a new trigger.
- E  the final choice among A, P, B, C(delta), D(delta), by the rule below.

## Selection rule (per outer fold, pooled inner-CV predictions)
- Recall floor: inner-CV recall of A (no candidate may lose more than 10% of A's recall, relative).
- Among candidates above the floor: highest inner-CV F1; ties within 0.005 -> higher precision, then fewer
  conditions. If no candidate passes the floor, A is kept.
- The chosen candidate is refitted on SELECTION(k) and scored once on TEST(k).

## Reported for every candidate
Precision, recall, F1, accuracy, balanced accuracy, MCC, TP, FP, TN, FN, positive prediction rate, rule coverage
(fraction of frames where each condition holds), events (falls detected, ADL videos without alarm),
inner-CV per-fold values (stability). Plus: PR trade-off over tau_h, per-rule false-positive analysis,
stress tests (height noise, missing frames, calibration bias, floor offset), and on TEST only, as description,
the error overlap with the out-of-fold neural ensemble.
