# External test of the improved V3 on FUKinect-Fall (pre-registration)

Written before any V3 output on this dataset was computed. Nothing below is changed after seeing results; any change is added as a dated amendment at the end with the reason.

## Question

Does the improved V3, frozen exactly as selected on URFD, keep its behaviour on people, rooms and a sensor it never saw? If it only worked because its thresholds fit URFD's 70 videos, fall detection should drop or false alarms should rise sharply here.

## Data

FUKinect-Fall, skeleton part (`skeleton.rar` from github.com/MuzafferAslan23/Fall-Detection-Dataset). Kinect v1 on a tripod 95 cm above the floor, 21 subjects aged 19 to 72, six actions (walking, bending, sitting, squatting, lying, falling), 8 repeats each. The archive holds 1006 clips (2 of the nominal 1008 are missing): 167 falls and 839 ADL. Each clip is a T x 60 matrix, 20 joints in metres in camera coordinates, Kinect SDK v1 joint order. Labels are per clip only.

None of these subjects, rooms or recordings appear in URFD.

## Neck height (no labels used)

- Neck = ShoulderCenter (joint index 2).
- A frame whose 60 values are all zero counts as missing (NaN), and V3 treats it as 2.0 m as it does on URFD.
- Floor plane: one plane per subject, fitted with RANSAC (1000 iterations, 3 cm inlier distance, least-squares refit on the inliers) to both foot joints (15 and 19) from every frame of every clip of that subject. Action labels are not used.
- Height = signed distance from the neck to that plane, positive above it.
- Sanity checks, reported but not used to change anything: the camera origin's distance to the plane should be near 0.95 m, and the plane normal should be close to the camera y axis.

## Models (frozen, no refitting)

The two configs in `reports/symbolic_v3_precision/final_config.json`, each run once:

| Config | Variant | tau_h | tau_v | delta | tau_s |
|---|---|---|---|---|---|
| fold 0 | D | 0.60 | 0.8929 | 0.30 | 0.4453 |
| fold 1 | C | 0.40 | 0.7321 | 0.40 | 0.4359 |

Code: `symbolic_v3_improved.py` as committed in `ca3aefc`, FPS = 30.

## Metrics

A clip is called a fall when V3 outputs 1 on at least one frame.

- Fall detection rate = falls called / 167.
- ADL false-alarm rate = ADL clips called a fall / 839, also given per action.
- Clip-level precision, recall, F1, accuracy, balanced accuracy and MCC.
- URFD reference, same clip-level rule on its 70 test videos: 30/30 falls detected (1.00), 12/40 ADL with an alarm (0.30).

## Decision rule

For each config separately, V3 counts as generalizing if both hold:

1. fall detection rate >= 0.80;
2. ADL false-alarm rate <= 0.45 (URFD's 0.30 plus 15 points).

If one config passes and the other fails, the result is reported as mixed, with no choice between them after the fact.

## Secondary analyses (descriptive only)

- Per-subject fall detection and false-alarm rates.
- Frame-rate sensitivity: the skeleton clips are shorter than the 4-5 s the dataset paper states for the depth videos, so the true skeleton rate may be below 30 fps. Re-run with the height signal assumed to be 20 fps, linearly resampled to 30 fps.
- Floor sensitivity: one global plane fitted to all subjects instead of one per subject.
- Neck-height distributions per action, to check the metric scale against URFD.
