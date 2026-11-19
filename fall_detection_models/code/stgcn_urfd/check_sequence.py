"""
check_sequence.py
Sanity check for one preprocessed sequence:
  - draws the 14-joint skeleton on 4 frames
  - plots neck height in the image, Depth-Anything neck value and URFD D_mm (Kinect) over time
  - runs the (untrained) ST-GCN on the sequence to confirm shapes and per-frame output

Usage:
  python check_sequence.py --seq fall-01 --rgb_dir fall-01-cam0-rgb --npy_dir urfd_npy \
      --features urfall-cam0-falls.csv --out check_fall-01.png
"""
import argparse
import csv
import glob
import os

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from dataset import EvalWindows, load_sequences
from graph import CENTER, EDGES, JOINTS
from model import TwoStreamSTGCN

p = argparse.ArgumentParser()
p.add_argument("--seq", required=True)
p.add_argument("--rgb_dir", required=True)
p.add_argument("--npy_dir", required=True)
p.add_argument("--features", required=True)
p.add_argument("--out", required=True)
a = p.parse_args()

x = np.load(os.path.join(a.npy_dir, f"{a.seq}.npy"))           # (3, T, 14, 1)
depth = np.load(os.path.join(a.npy_dir, f"{a.seq}_neck_depth.npy"))
frames = sorted(glob.glob(os.path.join(a.rgb_dir, "*.png")),
                key=lambda s: int(os.path.splitext(s)[0].rsplit("-", 1)[-1]))
T = x.shape[1]
assert len(frames) == T == len(depth), (len(frames), T, len(depth))

feat = {}
with open(a.features) as f:
    for r in csv.reader(f):
        if r and r[0] == a.seq:
            feat[int(r[1])] = (int(r[2]), float(r[9]))           # label, D_mm

H, W = cv2.imread(frames[0]).shape[:2]
fall_frames = [fr for fr, (l, _) in feat.items() if l in (0, 1)]
first_fall = min(fall_frames) if fall_frames else T // 2
show = sorted({1, max(1, first_fall - 10), min(T, first_fall + 10), T})

fig = plt.figure(figsize=(16, 9))
for i, fr in enumerate(show):
    ax = fig.add_subplot(2, 4, i + 1)
    img = cv2.cvtColor(cv2.imread(frames[fr - 1]), cv2.COLOR_BGR2RGB)
    ax.imshow(img)
    pts = x[:2, fr - 1, :, 0].T * [W, H]
    ok = x[2, fr - 1, :, 0] > 0
    for i0, i1 in EDGES:
        if ok[i0] and ok[i1]:
            ax.plot(*zip(pts[i0], pts[i1]), color="lime", lw=2)
    ax.scatter(pts[ok, 0], pts[ok, 1], s=14, c="red")
    if ok[CENTER]:
        ax.scatter(*pts[CENTER], s=60, c="yellow", edgecolors="k", label="Neck")
    lab = feat.get(fr, (None,))[0]
    ax.set_title(f"frame {fr}  URFD label {lab}  joints {ok.sum()}/14")
    ax.axis("off")

t = np.arange(1, T + 1)
anc_path = os.path.join(a.npy_dir, f"{a.seq}_anchor.npy")
if os.path.exists(anc_path):
    anc = np.load(anc_path)
    neck_y = np.where(anc[:, 2] > 0, 1 - anc[:, 1] / H, np.nan)
    src = anc[:, 2]
else:
    neck_y = np.where(x[2, :, CENTER, 0] > 0, 1 - x[1, :, CENTER, 0], np.nan)
    src = np.where(np.isfinite(neck_y), 1, 0)
ax = fig.add_subplot(2, 1, 2)
ax.plot(t, np.where(src == 1, neck_y, np.nan), label="neck height in image (1 - y/H)", color="tab:blue")
ax.plot(t, np.where(src == 2, neck_y, np.nan), label="hip fallback height in image", color="tab:purple", ls="--")
ax.plot(t, depth / np.nanmax(depth), label="Depth-Anything neck value (scaled to max)", color="tab:orange")
fr_d = sorted(feat)
ax.plot(fr_d, np.array([feat[f][1] for f in fr_d]) / 1000, label="URFD D_mm / 1000 (Kinect, body centre to floor)",
        color="tab:green")
lying = [f for f in fr_d if feat[f][0] in (0, 1)]
if lying:
    ax.axvspan(min(lying), max(lying), color="red", alpha=0.08, label="URFD fall frames (label 0/1)")
ax.set_xlabel("frame"); ax.set_ylim(0, 1.3); ax.legend(loc="upper right", fontsize=8)
ax.set_title(f"{a.seq}: neck signals vs Kinect ground truth")
fig.tight_layout()
fig.savefig(a.out, dpi=110)

# Untrained ST-GCN forward pass on all sliding windows of this sequence.
seq = [s for s in load_sequences(a.npy_dir) if s["name"] == a.seq]
ds = EvalWindows(seq, window=80, stride=1)
model = TwoStreamSTGCN().eval()
with torch.no_grad():
    xb = torch.stack([ds[i][0] for i in range(len(ds))])
    prob = torch.sigmoid(model(xb))
print(f"saved {a.out}")
print(f"anchor source: neck {int((src == 1).sum())}, hip fallback {int((src == 2).sum())}, none {int((src == 0).sum())}")
print(f"windows {len(ds)}  batch {tuple(xb.shape)}  probs {tuple(prob.shape)} "
      f"range [{prob.min():.3f}, {prob.max():.3f}] (untrained, values have no meaning yet)")
valid_d = np.isfinite(depth) & np.isfinite(neck_y)
fr_with_d = [f for f in range(1, T + 1) if f in feat and valid_d[f - 1]]
if len(fr_with_d) > 5:
    c1 = np.corrcoef([neck_y[f - 1] for f in fr_with_d], [feat[f][1] for f in fr_with_d])[0, 1]
    c2 = np.corrcoef([depth[f - 1] for f in fr_with_d], [feat[f][1] for f in fr_with_d])[0, 1]
    print(f"corr(neck height in image, D_mm) = {c1:.2f} | corr(Depth-Anything value, D_mm) = {c2:.2f}")
