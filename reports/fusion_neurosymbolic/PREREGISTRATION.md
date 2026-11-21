# Neuro-symbolic fusion: neural ensemble + improved V3 (pre-registration)

Written before this fusion family was scored on any data. Changes after this point go in a dated amendment at the end, with the reason.

## What is already known (stated so the reader can judge the bias)

- The URFD test folds have been seen many times. An earlier fusion study (`fusion_v3_improved`, now in the git history) scored fixed rules on them.
  - "veto+sym" (neural at 0.5, vetoed where the neck is at 0.8 m or higher, OR the symbolic alarm) reached MCC 0.651 on test.
  - On the tiny per-fold validation sets, no rule beat the neural ensemble, so none was adopted.
- The rule family below was designed with that knowledge. Every URFD test number it produces is therefore post-hoc. URFD has no untouched videos left, and neither external set has both streams: GMDCSA-24 has no depth, and FUKinect-Fall has no RGB.

## Idea

The two streams make different mistakes:

- **Neural false alarms.** Most of them happen while the person is clearly upright, and the metric neck height says so. The rule ignores the neural alarm when the neck is high.
- **Neural misses.** Some falls the neural stream misses are caught by V3. The rule adds V3's alarm, optionally only when the neural stream gives at least weak support, to cut V3's own false alarms.

`alarm(t) = [ p(t) >= t_n  AND  h(t) < t_up ]  OR  [ s(t) = 1  AND  p(t) >= t_low ]`

| Term | Meaning |
|---|---|
| `p` | ensemble probability |
| `h` | metric neck height (missing height: no veto) |
| `s` | improved V3 output |

Everything is causal: each term uses frame t and earlier only.

Grid, 175 combinations:

| Parameter | Values |
|---|---|
| t_n | 0.3, 0.4, 0.5, 0.6, 0.7 |
| t_up (m) | 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2 |
| t_low | 0, 0.02, 0.05, 0.1, 0.2 |

## Selection (cross-fitted, no test-fold labels)

For outer fold k, the selection set is SEL(k) = TRAIN(k) + VAL(k). It is exactly the test set of the other fold.

- **Neural predictions on SEL(k).** These come from fold (1-k)'s ensemble, which never trained on SEL(k). This is the out-of-fold file `reports/symbolic_v3_precision/neural_test_oof.csv`.
  - **Known limitation:** those models were trained on TEST(k) videos. The fusion thresholds see no TEST(k) label, but they are tuned on predictions of models that did.
  - The alternative, fold k's own validation, has 4 to 9 videos, which is too few to choose 3 thresholds.
- **Symbolic predictions on SEL(k).** These come from inner group CV inside SEL(k), with the same inner folds as `evaluate_symbolic_v3.py`. V3 is refitted on the inner train part, using fold k's chosen variant and delta, and predicts the inner held-out part. No TEST(k) label is used.
- **Choice of thresholds.** The combination with the highest frame-level MCC on SEL(k) wins. Ties within 0.005 go first to the higher t_up (less veto), then to the lower t_low, then to the t_n closest to 0.5.
- **Adoption for fold k.**
  - Selection MCC must be at least the neural ensemble's selection MCC + 0.02.
  - The fusion must also detect at least as many falls as the neural ensemble on SEL(k).
  - Otherwise fold k keeps the neural ensemble.

## Test

Each fold's TEST is scored once with fold k's models (neural: fold k ensemble, symbolic: fold k's final improved V3) and fold k's chosen thresholds.

- **Metrics:** P, R, F1, accuracy, balanced accuracy, MCC, falls detected and clean ADL videos, pooled over both folds.
- **Uncertainty:** a paired bootstrap over test videos (2000 resamples) of the MCC difference against the neural ensemble and against V3.
- **References:** the neural ensemble alone, the improved V3 alone, and the old fixed veto+sym (t_n 0.5, t_up 0.8, t_low 0), all on the same frames.
