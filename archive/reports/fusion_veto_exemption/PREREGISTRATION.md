# Veto exemption during a fast descent: protocol fixed before any number

Current fusion (baseline F0, `fusion_v3_improved.py`, rule veto+sym):
  fused(t) = (neural(t) AND NOT upright(t)) OR symbolic(t),   upright(t) = neck height >= 0.8 m.

Problem (validation evidence): at fall onset the neck is still >= 0.8 m, so the veto removes the network's early
alarms (validation: 28 of 337 network-detected fall frames removed, median onset delay 0.35 s -> 0.58 s).

Change (F1), one condition and no new fitted number:
  fused(t) = (neural(t) AND NOT (upright(t) AND NOT descending(t))) OR symbolic(t)
  descending(t) = the neck's downward speed exceeded tau_v at least once in the last 1.0 s, where tau_v is the
  Eq. 9 threshold of that fold's improved V3 (already fitted on training videos) and the 1.0 s window is V3's own
  descent window. A person standing or sitting still keeps the veto; a person whose neck is going down fast does not.

Data: per fold k, VAL(k) with the network's fold-k models (never trained on them) and the improved V3 fitted on
TRAIN(k). TEST(k) scored once afterwards with V3 fitted on TRAIN(k) + VAL(k) (post-hoc as all URFD test numbers).

Adoption per fold k (VAL only): F1 adopted if its VAL MCC >= F0's VAL MCC, its recall over the first second after
each fall onset is higher than F0's, and it detects no fewer falls and leaves no fewer ADL videos without alarm.
Otherwise fold k keeps F0. Reported: frame metrics, onset delay, first-second recall, FP/FN, on VAL and TEST.
