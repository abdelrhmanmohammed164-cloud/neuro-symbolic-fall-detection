# Inner-validation comparison (split_scene_v2, no test data)

Validation sets are tiny (fold 0: 2 falls + 6 ADL from scene A7; fold 1: 4 falls + 5 ADL from scene A8).
Baseline seed noise in MCC: fold 0 0.170, fold 1 0.089.

| run | fold | MCC | F1 | recall | spec | falls det. | ADL clean | FA/min | ADL FP first 80 / after | best ep | decision |
|---|---|---|---|---|---|---|---|---|---|---|---|
| baseline_v2_s42 | 0 | 0.368 | 0.353 | 0.899 | 0.749 | 2/2 | 1/6 | 7.96 | 0.00 / 0.35 | 35 | reference |
|  | 1 | 0.469 | 0.511 | 0.396 | 0.968 | 4/4 | 3/5 | 3.21 | 0.00 / 0.06 | 29 |  |
| baseline_v2_s43 | 0 | 0.538 | 0.554 | 0.793 | 0.916 | 2/2 | 4/6 | 1.77 | 0.00 / 0.12 | 11 | reference |
|  | 1 | 0.558 | 0.523 | 0.354 | 1.000 | 3/4 | 5/5 | 0.00 | 0.00 / 0.00 | 33 |  |
| exp01_gradclip | 0 | 0.407 | 0.388 | 0.917 | 0.780 | 2/2 | 1/6 | 8.85 | 0.00 / 0.30 | 42 | REJECT |
|  | 1 | 0.606 | 0.675 | 0.688 | 0.926 | 4/4 | 3/5 | 6.42 | 0.00 / 0.13 | 24 |  |
| exp02_leftpad | 0 | 0.529 | 0.507 | 0.964 | 0.856 | 2/2 | 1/6 | 6.19 | 0.00 / 0.20 | 22 | REJECT |
|  | 1 | 0.591 | 0.622 | 0.496 | 0.979 | 4/4 | 3/5 | 4.82 | 0.00 / 0.04 | 18 |  |
| exp03_useconf | 0 | 0.442 | 0.425 | 0.917 | 0.812 | 2/2 | 1/6 | 7.96 | 0.00 / 0.26 | 11 | REJECT |
|  | 1 | 0.760 | 0.798 | 0.762 | 0.969 | 4/4 | 2/5 | 12.85 | 0.00 / 0.05 | 14 |  |
| exp04_center | 0 | 0.577 | 0.576 | 0.893 | 0.905 | 2/2 | 1/6 | 13.27 | 0.00 / 0.13 | 45 | KEEP |
|  | 1 | 0.744 | 0.750 | 0.600 | 1.000 | 4/4 | 5/5 | 0.00 | 0.00 / 0.00 | 19 |  |
| exp05_padbalance | 0 | 0.361 | 0.333 | 0.959 | 0.702 | 2/2 | 0/6 | 6.19 | 0.00 / 0.41 | 23 | REJECT |
|  | 1 | 0.496 | 0.563 | 0.488 | 0.948 | 3/4 | 2/5 | 4.82 | 0.00 / 0.09 | 21 |  |

## Decisions

- **exp01_gradclip: REJECT** (fold 0: MCC 0.407 < required 0.538)
- **exp02_leftpad: REJECT** (fold 0: MCC 0.529 < required 0.538)
- **exp03_useconf: REJECT** (fold 0: MCC 0.442 < required 0.538; fold 1: fewer ADL videos without false alarm)
- **exp04_center: KEEP**
- **exp05_padbalance: REJECT** (fold 0: MCC 0.361 < required 0.538; fold 0: fewer ADL videos without false alarm; fold 1: MCC 0.496 < required 0.558; fold 1: fewer falls detected; fold 1: fewer ADL videos without false alarm)
