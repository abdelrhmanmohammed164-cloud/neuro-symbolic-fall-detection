# Fusion v1: decision

Result on validation (17 videos, the ensemble models of each fold never trained on them):

| Candidate | MCC | Recall | Precision | Falls | ADL clean |
|---|---|---|---|---|---|
| N: neural ensemble alone | 0.678 | 0.786 | 0.654 | 6/6 | 7/11 |
| F: (N OR trigger) AND support, paper Eq. 11 to 13 | 0.485 | 0.550 | 0.538 | 5/6 | 5/11 |
| V: N AND support (veto only) | 0.526 | 0.550 | 0.605 | 5/6 | 7/11 |

Decision under the pre-registered rule: keep N. Neither fusion is run on the test folds.

Why the symbolic stream does not complement the ensemble yet: its height is measured in the image, so it
fails on the same videos. The ensemble's validation false alarms (adl-30 kneeling close to the camera,
adl-39, adl-40) all bring the neck to image height below 0 with fast descents, so the veto keeps them; and in
fall-20 the image height never goes below 0.39, so the veto removes a fall the ensemble had caught. A
complementary signal needs height in metres (Kinect depth or a calibrated W, paper Eq. 1 and 7), which
does not depend on how close the person is to the camera.
