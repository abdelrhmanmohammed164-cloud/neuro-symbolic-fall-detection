"""
fukinect_neck_height.py
Metric neck height for FUKinect-Fall skeleton clips (Kinect v1, 20 joints, metres, camera frame).

Neck = ShoulderCenter (joint 2). Floor = RANSAC plane per subject on the foot joints (15, 19) of all that
subject's clips, no action labels used. See reports/symbolic_v3_external_fukinect/PREREGISTRATION.md.

Usage: python fukinect_neck_height.py <skeleton_dir> <out_npz> [--global-floor]
"""
import sys, glob, os, re
import numpy as np
import scipy.io as sio

NECK, FEET = 2, (15, 19)
ACTIONS = ("walking", "bending", "sitting", "squatting", "lying", "falling")


def load_clips(root):
    clips = []
    for f in sorted(glob.glob(os.path.join(root, "*", "*", "*.mat"))):
        act = os.path.basename(os.path.dirname(os.path.dirname(f)))
        subj = int(os.path.basename(os.path.dirname(f)))
        rep = int(re.search(r"_(\d+)\.mat$", f).group(1))
        a = sio.loadmat(f)["iskelet"].astype(float).reshape(-1, 20, 3)
        a[np.all(a.reshape(len(a), -1) == 0, 1)] = np.nan
        clips.append(dict(action=act, subject=subj, rep=rep, joints=a))
    return clips


def fit_plane(pts, iters=1000, thr=0.03, seed=0):
    pts = pts[np.all(np.isfinite(pts), 1)]
    rng = np.random.default_rng(seed); best = None
    for _ in range(iters):
        s = pts[rng.choice(len(pts), 3, replace=False)]
        n = np.cross(s[1] - s[0], s[2] - s[0]); nn = np.linalg.norm(n)
        if nn < 1e-9:
            continue
        n /= nn; d = -n @ s[0]
        inl = np.abs(pts @ n + d) < thr
        if best is None or inl.sum() > best.sum():
            best = inl
    P = pts[best]; c = P.mean(0)
    n = np.linalg.svd(P - c, full_matrices=False)[2][-1]
    if n[1] < 0:                       # camera y points up, so the normal should too
        n = -n
    return n, -n @ c, best.mean()


def heights(clips, global_floor=False):
    planes = {}
    groups = {0: clips} if global_floor else {s: [c for c in clips if c["subject"] == s] for s in sorted({c["subject"] for c in clips})}
    for key, cs in groups.items():
        feet = np.concatenate([c["joints"][:, list(FEET)].reshape(-1, 3) for c in cs])
        planes[key] = fit_plane(feet)
    for c in clips:
        n, d, _ = planes[0 if global_floor else c["subject"]]
        c["h"] = c["joints"][:, NECK] @ n + d
    return planes


if __name__ == "__main__":
    root, out = sys.argv[1], sys.argv[2]
    g = "--global-floor" in sys.argv
    clips = load_clips(root); planes = heights(clips, g)
    for k, (n, d, frac) in planes.items():
        tilt = np.degrees(np.arccos(abs(n[1])))
        print(f"floor {k:>2}: camera height {d:.3f} m, tilt {tilt:4.1f} deg, inliers {frac:.2f}")
    np.savez_compressed(out, action=[c["action"] for c in clips], subject=[c["subject"] for c in clips],
                        rep=[c["rep"] for c in clips], h=np.array([c["h"] for c in clips], dtype=object))
    print("clips", len(clips), "->", out)
