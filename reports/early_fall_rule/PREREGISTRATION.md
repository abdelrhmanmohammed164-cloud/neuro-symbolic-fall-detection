# Optional early-fall rule for the improved Symbolic V3: protocol fixed before any new number

Scope: the improved V3 (`symbolic_v3_improved.py`, configuration in `reports/symbolic_v3_precision/final_config.json`)
stays unchanged. The new rule is an optional addition behind a flag (`early_rule=None` reproduces V3 exactly).
No V4, no notebooks, no change to the neural network or to the fusion rule (veto+sym).

## Data use
- The 70 URFD test videos were used to find the weakness (shared missed frames at fall onset). Every URFD video is
  in exactly one test fold, so development is done per fold: for outer fold k only SELECTION(k) = TRAIN(k) + VAL(k)
  is used, with the same inner group cross-validation as the V3 precision study (ADL rooms as groups).
- TEST(k) is scored once at the end. It is reported as "not independent" for this change, because the pooled test
  diagnosis motivated it.
- Genuinely untouched data: GMDCSA-24 (160 videos, 4 people, 3 homes). It has no depth sensor, so the full V3 cannot
  run there, but the new rule uses image keypoints only. It is evaluated once on GMDCSA-24, on its own, with
  thresholds fitted on all 70 URFD videos, after the URFD decision is made.

## Features (causal, image keypoints from RTMO, COCO-17, confidence >= 0.3; no pixel thresholds)
- Shoulder midpoint S(t), hip midpoint P(t) (one side if the other is missing; missing if both are).
- Torso length L(t) = |S(t) - P(t)|; reference length Lref(t) = median of L over the past 2 s (frames <= t).
- Torso inclination theta(t) = angle between P->S and the image's upward vertical, degrees.
- B: inclination speed w(t) = (theta(t) - theta(t - 6)) / 0.2 s, deg/s.
- C: normalised hip descent u(t) = (P_y(t) - P_y(t - 6)) / Lref(t) / 0.2 s, torso lengths per second (image y down).
- D: B and C both above threshold within the same 0.3 s.

## Rule
- Trigger: the feature is above its threshold on 3 consecutive frames (0.1 s), to suppress keypoint jitter.
- Hold: once triggered, the early alarm stays on for 1.0 s (the length of URFD's 'falling' phase), then ends.
- Output: V3 OR early alarm. The early rule can only add positives; V3 itself is untouched.
- Thresholds: the 99th percentile of the feature over the ADL frames of the fitting videos (so at most about
  1% of ADL frames exceed it by construction). No MCC or F1 search.

## Acceptance (per outer fold, pooled inner-CV predictions on SELECTION(k))
Baseline A = improved V3. A variant is acceptable only if all hold:
1. it recovers fall frames A misses (recall higher than A's);
2. inner-CV F1 higher than A's;
3. precision not lower than A's precision minus the standard deviation of A's precision across the inner folds
   (the baseline's own variability, not an invented tolerance);
4. ADL videos without alarm not fewer than A's minus 1.
Among acceptable variants: highest F1; ties within 0.005 -> fewer features (B or C before D). None -> keep A.
Fusion (veto+sym, unchanged) is then reported on fold k's VAL videos (neural: fold k models, never trained on them).
