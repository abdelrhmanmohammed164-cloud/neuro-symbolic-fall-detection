# External test on GMDCSA-24 (never used for training, validation or tuning)

160 videos (79 falls, 81 ADL), 4 subjects, 3 homes, laptop webcam. Threshold 0.5. Mean over the 6 models of each configuration, range in brackets.

## all

| metric | center (6 models) | baseline_v2 (6 models) | center ensemble |
|---|---|---|---|
| MCC | 0.505 (0.300 to 0.664) | 0.430 (0.259 to 0.639) | 0.530 |
| F1 | 0.564 (0.332 to 0.748) | 0.477 (0.221 to 0.720) | 0.580 |
| Precision | 0.748 (0.685 to 0.875) | 0.722 (0.579 to 0.790) | 0.829 |
| Recall | 0.493 (0.219 to 0.789) | 0.418 (0.128 to 0.714) | 0.446 |
| Specificity | 0.944 (0.898 to 0.986) | 0.941 (0.865 to 0.989) | 0.971 |
| Accuracy | 0.835 (0.786 to 0.871) | 0.814 (0.780 to 0.872) | 0.843 |
| Falls detected | 64.5 (45.0 to 77.0) of 79 | 54.3 (32.0 to 73.0) of 79 | 65.0 of 79 |
| Median delay (s) | 0.76 (0.13 to 1.50) | 1.28 (0.17 to 2.53) | 0.63 |
| ADL with no false alarm | 59.7 (47.0 to 76.0) of 81 | 67.5 (60.0 to 76.0) of 81 | 71.0 of 81 |
| False alarms / min | 2.69 (0.65 to 4.68) | 1.71 (0.40 to 3.14) | 1.05 |

Accuracy of always predicting 'no fall' here: 0.758

## 30 fps source

| metric | center (6 models) | baseline_v2 (6 models) | center ensemble |
|---|---|---|---|
| MCC | 0.503 (0.303 to 0.670) | 0.439 (0.271 to 0.636) | 0.508 |
| F1 | 0.566 (0.341 to 0.757) | 0.481 (0.228 to 0.721) | 0.567 |
| Precision | 0.748 (0.691 to 0.863) | 0.752 (0.626 to 0.826) | 0.807 |
| Recall | 0.497 (0.227 to 0.803) | 0.412 (0.132 to 0.713) | 0.437 |
| Specificity | 0.940 (0.892 to 0.985) | 0.947 (0.883 to 0.991) | 0.965 |
| Accuracy | 0.829 (0.779 to 0.870) | 0.812 (0.774 to 0.868) | 0.832 |
| Falls detected | 55.7 (40.0 to 66.0) of 68 | 46.7 (27.0 to 63.0) of 68 | 55.0 of 68 |
| Median delay (s) | 0.71 (0.15 to 1.38) | 1.26 (0.17 to 2.63) | 0.60 |
| ADL with no false alarm | 46.3 (35.0 to 58.0) of 63 | 53.8 (48.0 to 61.0) of 63 | 54.0 of 63 |
| False alarms / min | 2.78 (0.86 to 4.84) | 1.50 (0.21 to 3.01) | 1.29 |

Accuracy of always predicting 'no fall' here: 0.748

## 15 fps source

| metric | center (6 models) | baseline_v2 (6 models) | center ensemble |
|---|---|---|---|
| MCC | 0.506 (0.263 to 0.680) | 0.403 (0.209 to 0.654) | 0.615 |
| F1 | 0.550 (0.293 to 0.745) | 0.463 (0.191 to 0.728) | 0.633 |
| Precision | 0.738 (0.579 to 0.916) | 0.623 (0.438 to 0.738) | 0.923 |
| Recall | 0.474 (0.188 to 0.728) | 0.442 (0.112 to 0.754) | 0.482 |
| Specificity | 0.958 (0.916 to 0.991) | 0.919 (0.803 to 0.984) | 0.989 |
| Accuracy | 0.857 (0.804 to 0.896) | 0.819 (0.757 to 0.883) | 0.884 |
| Falls detected | 8.8 (5.0 to 11.0) of 11 | 7.7 (5.0 to 10.0) of 11 | 10.0 of 11 |
| Median delay (s) | 1.07 (0.10 to 2.47) | 1.38 (0.15 to 2.83) | 1.05 |
| ADL with no false alarm | 13.3 (11.0 to 18.0) of 18 | 13.7 (12.0 to 16.0) of 18 | 17.0 of 18 |
| False alarms / min | 2.42 (0.00 to 4.20) | 2.31 (0.65 to 4.20) | 0.32 |

Accuracy of always predicting 'no fall' here: 0.792

## subject 1

| metric | center (6 models) | baseline_v2 (6 models) | center ensemble |
|---|---|---|---|
| MCC | 0.513 (0.251 to 0.702) | 0.522 (0.260 to 0.804) | 0.566 |
| F1 | 0.602 (0.334 to 0.764) | 0.550 (0.231 to 0.860) | 0.644 |
| Precision | 0.727 (0.601 to 0.889) | 0.808 (0.671 to 0.852) | 0.821 |
| Recall | 0.560 (0.231 to 0.853) | 0.497 (0.135 to 0.887) | 0.530 |
| Specificity | 0.908 (0.808 to 0.979) | 0.960 (0.932 to 0.989) | 0.955 |
| Accuracy | 0.811 (0.743 to 0.885) | 0.830 (0.748 to 0.919) | 0.836 |
| Falls detected | 13.7 (8.0 to 16.0) of 16 | 10.8 (4.0 to 16.0) of 16 | 13.0 of 16 |
| Median delay (s) | 0.52 (0.10 to 1.35) | 0.53 (0.13 to 1.70) | 0.40 |
| ADL with no false alarm | 11.0 (8.0 to 14.0) of 16 | 14.2 (13.0 to 15.0) of 16 | 13.0 of 16 |
| False alarms / min | 4.75 (2.15 to 11.30) | 1.35 (0.54 to 2.69) | 1.61 |

Accuracy of always predicting 'no fall' here: 0.721

## subject 2

| metric | center (6 models) | baseline_v2 (6 models) | center ensemble |
|---|---|---|---|
| MCC | 0.549 (0.341 to 0.718) | 0.438 (0.227 to 0.705) | 0.576 |
| F1 | 0.592 (0.361 to 0.785) | 0.482 (0.183 to 0.772) | 0.604 |
| Precision | 0.811 (0.705 to 0.916) | 0.732 (0.550 to 0.810) | 0.905 |
| Recall | 0.500 (0.238 to 0.782) | 0.423 (0.103 to 0.737) | 0.453 |
| Specificity | 0.964 (0.934 to 0.991) | 0.942 (0.853 to 0.990) | 0.985 |
| Accuracy | 0.852 (0.797 to 0.897) | 0.817 (0.777 to 0.895) | 0.857 |
| Falls detected | 21.7 (17.0 to 25.0) of 25 | 18.2 (9.0 to 25.0) of 25 | 23.0 of 25 |
| Median delay (s) | 1.03 (0.27 to 1.80) | 1.62 (0.43 to 2.87) | 1.20 |
| ADL with no false alarm | 18.5 (15.0 to 23.0) of 23 | 18.7 (15.0 to 22.0) of 23 | 22.0 of 23 |
| False alarms / min | 1.82 (0.00 to 4.21) | 1.73 (0.25 to 2.97) | 0.25 |

Accuracy of always predicting 'no fall' here: 0.759

## subject 3

| metric | center (6 models) | baseline_v2 (6 models) | center ensemble |
|---|---|---|---|
| MCC | 0.482 (0.208 to 0.741) | 0.401 (0.234 to 0.613) | 0.484 |
| F1 | 0.512 (0.219 to 0.806) | 0.459 (0.209 to 0.714) | 0.501 |
| Precision | 0.794 (0.642 to 0.926) | 0.696 (0.529 to 0.764) | 0.888 |
| Recall | 0.433 (0.132 to 0.794) | 0.411 (0.122 to 0.725) | 0.349 |
| Specificity | 0.964 (0.939 to 0.995) | 0.927 (0.802 to 0.986) | 0.985 |
| Accuracy | 0.827 (0.757 to 0.902) | 0.794 (0.760 to 0.850) | 0.821 |
| Falls detected | 16.2 (10.0 to 21.0) of 21 | 14.7 (10.0 to 19.0) of 21 | 17.0 of 21 |
| Median delay (s) | 1.05 (0.10 to 2.35) | 1.24 (0.10 to 2.37) | 1.30 |
| ADL with no false alarm | 17.0 (15.0 to 22.0) of 22 | 17.5 (15.0 to 20.0) of 22 | 21.0 of 22 |
| False alarms / min | 2.18 (0.00 to 3.35) | 2.03 (0.61 to 4.27) | 0.30 |

Accuracy of always predicting 'no fall' here: 0.742

## subject 4

| metric | center (6 models) | baseline_v2 (6 models) | center ensemble |
|---|---|---|---|
| MCC | 0.462 (0.386 to 0.527) | 0.368 (0.252 to 0.464) | 0.500 |
| F1 | 0.542 (0.427 to 0.628) | 0.417 (0.277 to 0.563) | 0.576 |
| Precision | 0.634 (0.541 to 0.789) | 0.659 (0.446 to 0.825) | 0.686 |
| Recall | 0.508 (0.308 to 0.733) | 0.343 (0.167 to 0.569) | 0.497 |
| Specificity | 0.916 (0.850 to 0.974) | 0.942 (0.886 to 0.991) | 0.943 |
| Accuracy | 0.835 (0.818 to 0.857) | 0.822 (0.784 to 0.838) | 0.854 |
| Falls detected | 13.0 (10.0 to 15.0) of 17 | 10.7 (9.0 to 13.0) of 17 | 12.0 of 17 |
| Median delay (s) | 0.23 (0.07 to 0.43) | 0.97 (0.15 to 2.03) | 0.15 |
| ADL with no false alarm | 13.2 (9.0 to 17.0) of 20 | 17.2 (15.0 to 20.0) of 20 | 15.0 of 20 |
| False alarms / min | 3.10 (1.24 to 5.89) | 1.55 (0.00 to 3.72) | 2.48 |

Accuracy of always predicting 'no fall' here: 0.800

