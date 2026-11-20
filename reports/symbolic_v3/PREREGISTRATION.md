# Symbolic v3 (height in metres from the Kinect): rule fixed before any v3 validation or test number

- v3 = symbolic v2 with the relative image height replaced by the neck height above the floor in metres
  (kinect_neck_height.py), gaps of up to 5 frames carried forward causally. Speed in m/s over 0.2 s.
- Grids (train only): tau_h 0.10 to 0.80 m (15 steps), tau_v 0.25 to 2.5 m/s (15 steps); tau_s = 95th
  percentile of the training 'lying' frames, as in v2.
- v3 replaces v2 if its trigger has a higher pooled validation MCC than v2's trigger (0.355) and detects no
  fewer validation falls. Then the test folds are scored once (post-hoc: they were seen before).
