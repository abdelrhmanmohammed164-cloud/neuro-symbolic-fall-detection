# Inner-validation comparison (split_scene_v2, no test data)

Validation sets are tiny (fold 0: 2 falls + 6 ADL from scene A7; fold 1: 4 falls + 5 ADL from scene A8).
Baseline seed noise in F1: fold 0 0.000, fold 1 0.000.

| run | fold | F1 | recall | spec | falls det. | ADL clean | FA/min | ADL FP first 80 / after | best ep | decision |
|---|---|---|---|---|---|---|---|---|---|---|
| baseline_v2_s42 | 0 | 0.000 | 0.000 | 1.000 | 0/2 | 6/6 | 0.00 | 0.00 / 0.00 | 2 | reference |
|  | 1 | 0.000 | 0.000 | 1.000 | 0/4 | 5/5 | 0.00 | 0.00 / 0.00 | 1 |  |
| exp05_padbalance | 0 | 0.000 | 0.000 | 1.000 | 0/2 | 6/6 | 0.00 | 0.00 / 0.00 | 2 | REJECT |
|  | 1 | 0.000 | 0.000 | 1.000 | 0/4 | 5/5 | 0.00 | 0.00 / 0.00 | 0 |  |

## Decisions

- **exp05_padbalance: REJECT** (fold 0: F1 0.000 < required 0.020; fold 1: F1 0.000 < required 0.020)
