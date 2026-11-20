# Fusion of the neural ensemble with symbolic v2: rules fixed before any fused number was computed

Data use
- Symbolic thresholds (tau_h, tau_v, tau_s) are fitted per fold on that fold's TRAIN videos only.
- The decision uses the VALIDATION videos of split_scene_v2 (fold 0: A7 + 2 falls, fold 1: A8 + 4 falls),
  scored by the ensemble models of that fold (never trained on them).
- The test folds are scored once, after the decision, and reported as post-hoc (they were seen before).

Candidates
- N: neural ensemble alone, threshold 0.5.
- F: paper Eq. 11 to 13, fused(t) = (N(t) OR trigger(t)) AND fall-state support in the last 80 frames.

Rule
- Adopt F only if, on the validation videos pooled over both folds:
  MCC(F) >= MCC(N) + 0.02, falls detected(F) >= falls detected(N), ADL videos with no false alarm(F) >= N.
- Otherwise keep N and report F as rejected. No other variant is tried on the test folds.

## Amendment 1 (after the validation result above, before any test number)
F was rejected on validation: the OR adds the trigger's false alarms (the trigger alone leaves 1 of 11 ADL
validation videos clean; the paper's symbolic precision is 0.99, ours about 0.49), and the veto costs recall.
One more candidate is tried on validation only, with the same adoption rule against N:
- V: veto only, fused(t) = N(t) AND fall-state support in the last 80 frames (no OR).
If V also fails, N stays the final system and no fusion is reported on the test folds as a result.
