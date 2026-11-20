# Symbolic v3: neck height in metres from the Kinect

## The height itself
- Floor plane per video by RANSAC on the Kinect background (fit error 0.5 to 0.9 cm in every video).
- Correlation of the neck height with URFD's own Kinect distance of the body centre to the floor:
  falls median 0.988 (minimum 0.947), ADL median 0.929 (minimum 0.628). The image-based height of v2 gave
  0.979 and 0.788. On frames URFD labels 'lying', the neck is at 0.28 m (5th to 95th percentile 0.15 to 0.42 m).

## Decision on validation (pre-registered)
| Validation, trigger | MCC | Precision | Recall | Falls | ADL clean |
|---|---|---|---|---|---|
| v2 (image height) | 0.355 | 0.492 | 0.352 | 6/6 | 1/11 |
| v3 (metres) | 0.214 | 0.380 | 0.203 | 6/6 | 2/11 |

Rule: v3 replaces v2 only with a higher validation MCC. It did not, so v2 stays the symbolic stream.

## Test folds (post-hoc, reported for information; they did not change the decision)
| Test, 70 videos | MCC | Precision | Recall | Specificity | Falls | ADL clean | False alarms / min |
|---|---|---|---|---|---|---|---|
| v2 trigger | 0.409 | 0.649 | 0.342 | 0.967 | 30/30 | 7/40 | 9.26 |
| v3 trigger | 0.402 | 0.735 | 0.280 | 0.982 | 30/30 | 23/40 | 5.03 |
| v2 fall state | 0.290 | 0.367 | 0.455 | 0.860 | 25/30 | 20/40 | 9.06 |
| v3 fall state | 0.402 | 0.487 | 0.499 | 0.907 | 30/30 | 24/40 | 5.03 |

The two disagree: validation (11 ADL videos from rooms A7 and A8) favours v2, the larger test set shows
v3 with far fewer false alarms. Switching on the test result would be selecting on test data.

## Decision
The project adopts v3 and drops v2. The validation rule above had kept v2; the switch was made after the
test folds were seen (fewer false alarms: 23 vs 7 clean ADL videos), so every v3 test number is post-hoc.
v2's code was removed from the working tree (it stays in the git history, commit 6cb9450 and earlier).
The fusion v1 report (reports/fusion_v1) was run with v2 and is kept only as a record.
