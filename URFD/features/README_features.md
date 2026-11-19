# URFD features (urfall-cam0-*.csv)

Source: https://fenix.ur.edu.pl/mkepski/ds/uf.html (computed by the dataset authors from the Kinect depth maps, camera 0).
The original files have no header; `*_with_header.csv` are the same rows with column names added.

| # | Column | Meaning |
|---|---|---|
| 1 | sequence | Sequence name, e.g. fall-01, adl-12 |
| 2 | frame | Frame number (matches the PNG index in the cam0 zips) |
| 3 | label | -1 = not lying, 0 = temporary pose during the fall, 1 = lying on the ground. Authors exclude 0 from classification |
| 4 | HeightWidthRatio | Bounding-box height / width |
| 5 | MajorMinorRatio | Major / minor axis ratio of the segmented person blob |
| 6 | BoundingBoxOccupancy | Fraction of the bounding box covered by person pixels |
| 7 | MaxStdXZ | Std. dev. of the person's 3D point cloud along X and Z (depth) |
| 8 | HHmaxRatio | Current person height / height while standing |
| 9 | H_mm | Person height in mm |
| 10 | D_mm | Distance of the person's centre to the floor, in mm |
| 11 | P40 | Share of the person's points within 40 cm of the floor |

Notes
- Not every frame has a row: 18 ADL sequences have fewer rows than frames (frames without a detected person are missing).
- adl-37-data.csv on the server is identical to adl-40-data.csv (330 rows vs 350 frames), and adl-02-data.csv is identical to adl-03-data.csv.
- data.csv (per sequence): falls = frame, time_ms, SV_total (accelerometer, g); ADLs = frame, time_ms only.
