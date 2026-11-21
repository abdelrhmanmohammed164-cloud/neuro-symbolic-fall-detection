# Neuro-symbolic fusion: results

Pre-registration: `PREREGISTRATION.md` (commit 29e2761), written before this rule family was scored. The thresholds were chosen per fold on TRAIN+VAL, using no TEST label, and TEST was scored once. All URFD test numbers are post-hoc: these test folds had been seen before, and the rule was designed knowing that a fixed veto+sym rule did well on them.

## The rule

`alarm(t) = [ p(t) >= t_n AND neck height(t) < t_up ] OR [ V3(t) = 1 AND p(t) >= t_low ]`

- The neural alarm is ignored while the metric neck height says the person is upright.
- V3's alarm is added, gated by weak neural support.

Both streams stay frozen: the neural ensemble and the improved V3 are used exactly as reported.

## Selection (cross-fitted)

| Fold | Selection videos | Chosen t_n / t_up / t_low | Selection MCC | Neural on the same videos | Adopted |
|---|---|---|---|---|---|
| 0 | 29 | 0.6 / 1.2 m / 0.0 | 0.638 | 0.592 | yes |
| 1 | 41 | 0.4 / 1.2 m / 0.1 | 0.716 | 0.543 | yes |

In fold 0, selection MCC is flat for t_up between 0.9 and 1.2 m. The tie rule picks the weakest veto. In fold 1, t_up = 1.1 m was 0.001 higher, and the tie rule moved it to 1.2 m.

## URFD test (pooled over both folds, frame level)

| Model | Precision | Recall | F1 | Accuracy | Bal. acc. | MCC | Falls detected | Clean ADL |
|---|---|---|---|---|---|---|---|---|
| Neural ensemble | 0.570 | 0.678 | 0.619 | 0.874 | 0.793 | 0.547 | 28/30 | 23/40 |
| Improved V3 | 0.796 | 0.396 | 0.528 | 0.893 | 0.689 | 0.513 | 30/30 | 28/40 |
| **Fusion (this study)** | 0.695 | 0.711 | **0.703** | **0.909** | **0.828** | **0.650** | **30/30** | 23/40 |
| Old fixed veto+sym (reference) | 0.735 | 0.668 | 0.700 | 0.913 | 0.813 | 0.650 | 30/30 | 24/40 |

Paired bootstrap over test videos (2000 resamples), MCC difference:

| Comparison | Mean | 95% CI | Share of resamples where the fusion is better |
|---|---|---|---|
| Fusion minus neural | +0.102 | +0.025 to +0.180 | 99.6% |
| Fusion minus V3 | +0.139 | +0.022 to +0.249 | 99.0% |

Per fold:

| Fold | Neural MCC | V3 MCC | Fusion MCC | Fusion falls | Fusion clean ADL |
|---|---|---|---|---|---|
| 0 | 0.543 | 0.528 | 0.706 | 18/18 | 10/23 |
| 1 | 0.592 | 0.509 | 0.593 | 12/12 | 13/17 |

## Reading

- **The fusion beats both streams on every pooled measure except clean ADL videos.**
  - It catches every fall, including the 2 the neural ensemble missed.
  - It has the best recall and F1.
  - It cuts neural false-alarm frames from 923 to 562.
  - Clean ADL stays at 23/40: the false alarms that remain are shorter, but they still touch the same ADL videos.
- **The gain is not even across folds.**
  - Fold 0 accounts for most of it: +0.16 MCC over the neural ensemble.
  - In fold 1 the fusion only matches the neural ensemble on MCC, but it detects all 12 falls instead of 10.
- **The selected rule reached the same MCC as the old fixed rule.** The cross-fitted selection found thresholds without looking at test labels, and they land where the post-hoc rule was. That is consistent with the rule capturing a real complementarity, but it does not prove it.

## Against the paper (Table 4, fused stream)

| | Accuracy | Precision | Recall | F1 | MCC |
|---|---|---|---|---|---|
| Paper | 0.88 | 0.88 | 0.65 | 0.76 | 0.70 |
| This fusion (URFD, post-hoc) | 0.91 | 0.70 | 0.71 | 0.70 | 0.65 |

The comparison is not like for like: the paper uses its own data, and URFD is a different dataset.

## Limits

- **The selection data are not fully independent of the test videos.** The neural predictions on SEL(k) come from the other fold's models, and those models were trained on TEST(k) videos. The fusion thresholds never see a TEST(k) label, but they are tuned on outputs of models that did.
- **The URFD test folds were seen before this design.**
- **No external check is possible for the fusion.** GMDCSA-24 has no depth, so there is no metric neck height for V3 or for the veto. FUKinect-Fall has no RGB, so the neural stream cannot run on it.
- An honest final number needs a new dataset with RGB, depth and frame labels.

## Files

- `results.json`: selection grid (top 5 per fold), chosen thresholds, per-fold and pooled test metrics, bootstrap.
- `final_config.json`: frozen thresholds per fold. `fusion.py --rule neurosymbolic` applies them.
- `test_predictions.csv`: frame-level neural, V3 and fused alarms on the test folds.
- Code:
  - `fusion_neurosymbolic.py`: the study.
  - `fusion.py`: applies a frozen rule to prediction CSVs.
  - `export_symbolic_predictions.py`: exports the V3 alarms per frame.
