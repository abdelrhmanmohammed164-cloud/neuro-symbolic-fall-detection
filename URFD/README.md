# URFD raw data (not in git)

Source: https://fenix.ur.edu.pl/mkepski/ds/uf.html (licence CC BY-NC-SA 4.0).

Only camera 0 is used. Download into this folder:

```
URFD/
  falls/   fall-01 .. fall-30:  <seq>-cam0-rgb.zip, <seq>-cam0-d.zip, <seq>-data.csv
  adls/    adl-01 .. adl-40:    <seq>-cam0-rgb.zip, <seq>-cam0-d.zip, <seq>-data.csv
  features/  urfall-cam0-falls.csv, urfall-cam0-adls.csv   (per-frame labels and Kinect features)
```

212 files, about 6.8 GB. The server is slow (about 0.4 MB/s per connection); downloading several files in
parallel helps. Check every archive with `unzip -tq` before preprocessing.

Training does NOT need the video archives: the preprocessed skeletons are already in `processed_data/`.
The two label files `features/urfall-cam0-falls.csv` and `features/urfall-cam0-adls.csv` ARE included in git
(small, CC BY-NC-SA 4.0), because `make_frame_labels.py` and the pre-training tests read them.
