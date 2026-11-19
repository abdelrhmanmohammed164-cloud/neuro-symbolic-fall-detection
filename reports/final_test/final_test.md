# Final test evaluation (post-hoc: the test folds were seen once before)

Mean over seeds 42, 43, 44, with the range in brackets. Threshold 0.5.

## pooled

| configuration | MCC | F1 | recall | specificity | falls detected | ADL with no false alarm | false alarms / min |
|---|---|---|---|---|---|---|---|
| primary (exp04_center) | 0.503 (0.497 to 0.515) | 0.582 (0.577 to 0.591) | 0.664 (0.632 to 0.728) | 0.890 (0.869 to 0.901) | 28.0 (27.0 to 29.0) of 30 | 21.7 (18.0 to 24.0) of 40 | 5.84 (4.83 to 7.25) |
| reference (baseline_v2) | 0.343 (0.323 to 0.364) | 0.453 (0.437 to 0.470) | 0.553 (0.526 to 0.567) | 0.842 (0.832 to 0.850) | 25.7 (24.0 to 27.0) of 30 | 19.0 (18.0 to 21.0) of 40 | 6.51 (4.43 to 8.05) |

## fold0

| configuration | MCC | F1 | recall | specificity | falls detected | ADL with no false alarm | false alarms / min |
|---|---|---|---|---|---|---|---|
| primary (exp04_center) | 0.530 (0.489 to 0.601) | 0.595 (0.557 to 0.657) | 0.790 (0.769 to 0.802) | 0.851 (0.818 to 0.892) | 18.0 (18.0 to 18.0) of 18 | 9.3 (8.0 to 11.0) of 23 | 8.51 (8.00 to 9.14) |
| reference (baseline_v2) | 0.385 (0.349 to 0.403) | 0.468 (0.444 to 0.482) | 0.783 (0.739 to 0.813) | 0.735 (0.730 to 0.744) | 18.0 (18.0 to 18.0) of 18 | 7.0 (5.0 to 9.0) of 23 | 9.78 (6.09 to 11.81) |

## fold1

| configuration | MCC | F1 | recall | specificity | falls detected | ADL with no false alarm | false alarms / min |
|---|---|---|---|---|---|---|---|
| primary (exp04_center) | 0.524 (0.432 to 0.605) | 0.577 (0.529 to 0.616) | 0.544 (0.472 to 0.658) | 0.932 (0.845 to 0.989) | 10.0 (9.0 to 11.0) of 12 | 12.3 (10.0 to 14.0) of 17 | 2.85 (1.28 to 5.12) |
| reference (baseline_v2) | 0.368 (0.281 to 0.501) | 0.424 (0.346 to 0.526) | 0.333 (0.266 to 0.400) | 0.955 (0.938 to 0.977) | 7.7 (6.0 to 9.0) of 12 | 12.0 (11.0 to 13.0) of 17 | 2.85 (1.71 to 4.27) |

## Per seed (pooled over both folds)

| configuration | seed | MCC | F1 | falls | ADL clean | TP / FN / FP / TN |
|---|---|---|---|---|---|---|
| primary (exp04_center) | 42 | 0.515 | 0.591 | 29/30 | 18/40 | 1313 / 490 / 1328 / 8805 |
| primary (exp04_center) | 43 | 0.498 | 0.578 | 27/30 | 24/40 | 1140 / 663 / 1001 / 9132 |
| primary (exp04_center) | 44 | 0.497 | 0.577 | 28/30 | 23/40 | 1139 / 664 / 1005 / 9128 |
| reference (baseline_v2) | 42 | 0.364 | 0.470 | 27/30 | 18/40 | 1020 / 783 / 1515 / 8618 |
| reference (baseline_v2) | 43 | 0.341 | 0.452 | 26/30 | 21/40 | 1023 / 780 / 1704 / 8429 |
| reference (baseline_v2) | 44 | 0.323 | 0.437 | 24/30 | 18/40 | 948 / 855 / 1586 / 8547 |
