# Symbolic v4 and neural-symbolic fusion: protocol fixed before any v4 or fusion number

## Data and roles
- Folds: `splits/split_scene_v2.json`. For fold k: TRAIN = fold k train videos, VAL = fold k val videos,
  TEST = fold k test videos.
- Leakage found in the old protocol: fold 1 VAL (room A8 + falls 13, 19, 20, 23) lies inside fold 0 TEST,
  and fold 0 VAL (room A7 + falls 01, 10) inside fold 1 TEST. A choice made on both folds' validation
  therefore sees test videos. From now on every choice for fold k uses fold k TRAIN and fold k VAL only.
- Symbolic thresholds: fold k TRAIN labels only (URFD 3-class labels for the 'falling'/'lying' phases).
- Neural predictions: VAL = fold k ensemble (3 seeds) on fold k VAL (never trained on them);
  TEST = the out-of-fold file (fold k models on fold k TEST). No in-sample neural prediction is used.
- The test folds were seen in earlier stages, so all test numbers stay post-hoc. GMDCSA-24 has no
  depth sensor, so the metric-height symbolic stream cannot be tested on it.

## Candidates
Symbolic
- A  original notebook (teammate), mode urfd_state_aware, its own grids, fitted on TRAIN.
- B  v3: metric height, paper rule (low AND fast at the same frame), grid on TRAIN.
- C  v4: state machine on the metric height, thresholds from TRAIN label statistics, no MCC grid:
     tau_low = 95th pct of TRAIN lying heights; delta_fall = (smallest TRAIN drop into lying within 1.5 s) - 0.05 m;
     delta_on = delta_fall / 2; upright = neck >= 0.8 m (domain constant). States in priority order:
     UNKNOWN (no height > 5 frames), FALL (entered when low with a drop >= delta_fall in the last 1.5 s,
     held while low or unknown <= 1 s), DESCENDING (drop >= delta_on in the last 1 s and still moving down
     > 0.3 m/s), UPRIGHT, LOW_NOFALL (low without a fast drop: slow, intentional lowering), MID.
     Hard output: FALL or DESCENDING.
- D  ablations of C: D1 without DESCENDING; D2 without the 1 s carry-over for unknown height.
- E  C with calibrated confidence: p_s(state) = fall rate of TRAIN frames in that state (Laplace +1/+2);
     hard output p_s >= 0.5.

Neural: F = ensemble alone (threshold 0.5).

Fusion (for symbolic in {A, B, C, D1, D2, E} where defined)
- F1 paper Eq. 11-13: (neural OR symbolic) AND symbolic support in the last 80 frames.
- F2 mean of probabilities (E only), threshold 0.5.
- F3 weighted mean w*p_n + (1-w)*p_s, w in {0.6, 0.7, 0.8, 0.9} (E only).
- F4 gated veto: neural AND NOT UPRIGHT (C/E states).
- F5 gated veto + rescue: (neural AND NOT UPRIGHT) OR FALL.
- F6 logistic stacking on [logit p_n, p_s], fitted on fold k VAL; its VAL score is leave-one-video-out.
G = neural + A with F1. H = neural + improved symbolic (the selected option).

## Selection rule (per fold, on fold k VAL only)
Pick the option with the highest VAL MCC among all fused options; ties -> fewer parameters.
It is adopted for fold k only if VAL MCC >= neural VAL MCC + 0.02, VAL falls detected not lower and
VAL ADL videos without false alarm not lower. Otherwise fold k keeps the neural ensemble alone.
The TEST folds are then scored once with each fold's choice. All other options are also reported on TEST
for transparency, marked as not selected.
