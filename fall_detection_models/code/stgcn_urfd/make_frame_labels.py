"""
make_frame_labels.py
Build <seq>_frame_labels.npy (1 = fall frame) from the URFD features CSVs.

URFD labels per depth frame: -1 not lying, 0 falling (transition), 1 lying on the ground.
Mapping used here (a choice, not from the paper):
    fall-XX : label 0 or 1 -> 1 (falling / fallen), -1 -> 0
    adl-XX  : always 0  (ADL "lying" = lying on a bed or sofa on purpose, not a fall)
Frames that have no row in the CSV get 0.

Usage:
    python make_frame_labels.py --features_dir ~/Downloads/URFD/features --npy_dir ../urfd_npy
"""
import argparse
import csv
import os
import numpy as np

p = argparse.ArgumentParser()
p.add_argument("--features_dir", required=True)
p.add_argument("--npy_dir", required=True, help="folder with <seq>.npy, shape (C, T, V, M)")
a = p.parse_args()

rows = {}
for fn in ("urfall-cam0-falls.csv", "urfall-cam0-adls.csv"):
    path = os.path.join(a.features_dir, fn)
    if not os.path.exists(path):
        print(f"note: {fn} not found, skipped")
        continue
    with open(path) as f:
        for r in csv.reader(f):
            if r:
                rows.setdefault(r[0], {})[int(r[1])] = int(r[2])

AUX = ("_frame_labels", "_neck_depth", "_raw_coco17", "_anchor", "_timestamps")
for fn in sorted(os.listdir(a.npy_dir)):
    if not fn.endswith(".npy") or fn[:-4].endswith(AUX):
        continue
    seq = fn[:-4]
    T = np.load(os.path.join(a.npy_dir, fn), mmap_mode="r").shape[1]
    lab = np.zeros(T, np.int64)
    if seq.startswith("fall"):
        for frame, l in rows.get(seq, {}).items():      # frame numbers are 1-based
            if 1 <= frame <= T and l in (0, 1):
                lab[frame - 1] = 1
    np.save(os.path.join(a.npy_dir, seq + "_frame_labels.npy"), lab)
    print(f"{seq}: T={T} fall frames={int(lab.sum())}")
