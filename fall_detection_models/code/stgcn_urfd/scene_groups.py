"""
scene_groups.py
Assigns every sequence to a recording scene (room / camera placement) using the Depth-Anything
maps of its first 5 frames, stored in the preprocessing checkpoints. Needed because in URFD the
scenes are class-pure (falls and ADLs were filmed in different places), so a split that ignores
scenes lets a model recognise the room instead of the fall.

Method: z-scored mean depth map of frames 0-4 per sequence -> Pearson correlation between
sequences -> connected components of the graph with edges corr > threshold. Reports how stable
the grouping is across thresholds and writes the chosen one.

Usage (numpy only):
  python scene_groups.py --checkpoints checkpoints/cfg_x --out reports/scene_groups.csv --thr 0.9
"""
import argparse
import csv
import os

import numpy as np


def components(S, thr):
    n = len(S); lab = -np.ones(n, int); g = 0
    for i in range(n):
        if lab[i] >= 0:
            continue
        stack = [i]; lab[i] = g
        while stack:
            u = stack.pop()
            for v in np.nonzero(S[u] > thr)[0]:
                if lab[v] < 0:
                    lab[v] = g; stack.append(v)
        g += 1
    return lab


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoints", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--thr", type=float, default=0.9)
    a = p.parse_args()
    names = sorted(d for d in os.listdir(a.checkpoints) if os.path.isdir(os.path.join(a.checkpoints, d)))
    X = []
    for s in names:
        with np.load(os.path.join(a.checkpoints, s, "chunk_0000.npz")) as z:
            d = z["depth_small"][:5].astype(np.float32).mean(0)
        X.append(((d - d.mean()) / (d.std() + 1e-6)).ravel())
    S = np.corrcoef(np.stack(X))
    y = np.array([n.startswith("fall") for n in names])
    for thr in (0.8, 0.85, 0.9, 0.93, 0.95):
        lab = components(S, thr)
        mixed = sum(1 for g in set(lab) if 0 < y[lab == g].sum() < (lab == g).sum())
        print(f"thr {thr}: {len(set(lab))} groups | sizes {sorted(np.bincount(lab).tolist(), reverse=True)} | mixed-class groups {mixed}")
    lab = components(S, a.thr)
    order = sorted(set(lab), key=lambda g: (not y[lab == g].any(), -np.sum(lab == g)))
    rename = {g: f"{'F' if y[lab == g].any() else 'A'}{i + 1}" for i, g in enumerate(order)}
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "w", newline="") as f:
        w = csv.writer(f); w.writerow(["sequence", "scene_group", "label"])
        for n, g, l in zip(names, lab, y):
            w.writerow([n, rename[g], int(l)])
    print(f"saved {a.out} with threshold {a.thr}")
    for g in sorted(set(rename.values())):
        mem = [n for n, l in zip(names, lab) if rename[l] == g]
        print(f"  {g}: {len(mem):2d} -> {', '.join(mem)}")


if __name__ == "__main__":
    main()
