"""
summarize_preprocessing.py
Quality table for all preprocessed sequences -> preprocessing_summary.csv

Per sequence: frames, frames with a person, % missing joints, anchor source counts,
% frames with neck depth, and correlation of the anchor height in the image with the Kinect
D_mm from the URFD features (the closer to 1, the better the 2D signal follows the real height).

Usage:
  python summarize_preprocessing.py --npy_dir work/urfd_npy --features_dir URFD/features --out work/preprocessing_summary.csv
"""
import argparse
import csv
import os

import numpy as np

from dataset import is_sequence_file

p = argparse.ArgumentParser()
p.add_argument("--npy_dir", required=True)
p.add_argument("--features_dir", required=True)
p.add_argument("--out", required=True)
p.add_argument("--img_h", type=int, default=480)
a = p.parse_args()

D = {}
for fn in ("urfall-cam0-falls.csv", "urfall-cam0-adls.csv"):
    if not os.path.exists(os.path.join(a.features_dir, fn)):
        continue
    with open(os.path.join(a.features_dir, fn)) as f:
        for r in csv.reader(f):
            if r:
                D.setdefault(r[0], {})[int(r[1])] = float(r[9])

rows = []
for fn in sorted(os.listdir(a.npy_dir)):
    if not is_sequence_file(fn):
        continue
    s = fn[:-4]
    x = np.load(os.path.join(a.npy_dir, fn))
    anc = np.load(os.path.join(a.npy_dir, f"{s}_anchor.npy"))
    dep = np.load(os.path.join(a.npy_dir, f"{s}_neck_depth.npy"))
    T = x.shape[1]
    conf = x[2, :, :, 0]
    h = np.where(anc[:, 2] > 0, 1 - anc[:, 1] / a.img_h, np.nan)
    fr = [f for f in D.get(s, {}) if 1 <= f <= T and np.isfinite(h[f - 1])]
    corr = (np.corrcoef([h[f - 1] for f in fr], [D[s][f] for f in fr])[0, 1]
            if len(fr) > 10 and np.std([D[s][f] for f in fr]) > 0 else np.nan)
    rows.append(dict(
        sequence=s, frames=T, frames_with_person=int((conf > 0).any(1).sum()),
        missing_joints_pct=round(100 * float((conf <= 0).mean()), 1),
        anchor_neck=int((anc[:, 2] == 1).sum()), anchor_hip=int((anc[:, 2] == 2).sum()),
        anchor_none=int((anc[:, 2] == 0).sum()),
        depth_valid_pct=round(100 * float(np.isfinite(dep).mean()), 1),
        corr_height_vs_D=round(float(corr), 3) if np.isfinite(corr) else "",
    ))

with open(a.out, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)

c = np.array([r["corr_height_vs_D"] for r in rows if r["corr_height_vs_D"] != ""], float)
print(f"sequences: {len(rows)} | total frames: {sum(r['frames'] for r in rows)}")
print(f"frames with person: {sum(r['frames_with_person'] for r in rows) / sum(r['frames'] for r in rows):.1%}")
print(f"anchor none: {sum(r['anchor_none'] for r in rows) / sum(r['frames'] for r in rows):.1%}")
if len(c):
    print(f"corr(anchor height, D_mm): median {np.median(c):.2f}, min {c.min():.2f}; "
          f"sequences below 0.7: {[r['sequence'] for r in rows if r['corr_height_vs_D'] != '' and r['corr_height_vs_D'] < 0.7]}")
print(f"saved {a.out}")
