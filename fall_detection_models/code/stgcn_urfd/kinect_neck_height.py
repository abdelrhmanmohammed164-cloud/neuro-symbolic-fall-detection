"""
kinect_neck_height.py
Neck height above the floor in metres, from the URFD Kinect depth of camera 0 (paper Eq. 1 and 7, with the
Kinect's own geometry in place of the paper's control-point matrix W).

Per video:
  1. Depth in mm: d = C * P / 65535, C = 6000 for falls and 7000 for ADL (URFD documentation).
  2. Floor plane: the per-pixel median of up to 40 depth frames removes the person; the pixels of the lower
     45% of that background image are back-projected with the Kinect v1 depth intrinsics and a plane is
     fitted with RANSAC (inliers within 2 cm, normal within 35 degrees of the camera's vertical axis),
     then refined by least squares on the inliers.
  3. Neck point: the shoulder midpoint from our RTMO keypoints (hip midpoint when the shoulders are lost),
     the same anchor as the ST-GCN preprocessing. Its depth is the 20th percentile of the valid depth in a
     9x9 patch (the person is the nearest surface; this also absorbs the small RGB/depth offset).
  4. Height = signed distance from the back-projected neck point to the floor plane, in metres.

Output: processed_data/kinect_height/<video>.npz with height_m (T,), neck_xyz (T, 3), plane (4,),
plane_inliers, floor_rms_m. Frames without a valid neck or depth get NaN.

Usage (from the project root; resumable, skips finished videos):
  python3 fall_detection_models/code/stgcn_urfd/kinect_neck_height.py [video ...]
"""
import io
import json
import os
import sys
import zipfile

import numpy as np
from PIL import Image

ROOT = os.environ.get("PROJECT_ROOT", os.getcwd())
NPY = os.path.join(ROOT, "processed_data", "cfg_d31bfed741ab", "urfd_npy")
OUT = os.path.join(ROOT, "processed_data", "kinect_height")
FX, FY, CX, CY = 594.21, 591.04, 339.5, 242.7      # Kinect v1 depth camera (Burrus calibration)
CONF = 0.3


class DepthZip:
    """Reads one depth frame at a time (the Mac VM has little memory)."""
    def __init__(self, vid):
        kind = "falls" if vid.startswith("fall") else "adls"
        self.C = 6000.0 if kind == "falls" else 7000.0
        self.z = zipfile.ZipFile(os.path.join(ROOT, "URFD", kind, f"{vid}-cam0-d.zip"))
        self.names = sorted(n for n in self.z.namelist() if n.endswith(".png"))

    def __len__(self):
        return len(self.names)

    def __getitem__(self, i):
        P = np.array(Image.open(io.BytesIO(self.z.read(self.names[i]))), dtype=np.float32)
        return self.C * P / 65535.0 / 1000.0             # metres; 0 = no reading


def backproject(u, v, z):
    return np.stack([(u - CX) * z / FX, (v - CY) * z / FY, z], -1)


def fit_floor(bg, rng):
    H, W = bg.shape
    vv, uu = np.mgrid[int(H * 0.55):H, 0:W]
    z = bg[int(H * 0.55):H, :]
    m = (z > 0.4) & (z < 6.0)
    pts = backproject(uu[m].astype(float), vv[m].astype(float), z[m])
    if len(pts) > 30000:
        pts = pts[rng.choice(len(pts), 30000, replace=False)]
    best, best_n = None, 0
    for _ in range(400):
        a, b, c = pts[rng.choice(len(pts), 3, replace=False)]
        n = np.cross(b - a, c - a); nn = np.linalg.norm(n)
        if nn < 1e-9:
            continue
        n /= nn
        if abs(n[1]) < np.cos(np.radians(35)):            # floor normal ~ camera y axis
            continue
        d = -n @ a
        inl = np.abs(pts @ n + d) < 0.02
        if inl.sum() > best_n:
            best, best_n = inl, inl.sum()
    P = pts[best]; c = P.mean(0)
    n = np.linalg.svd(P - c, full_matrices=False)[2][-1]
    if n[1] > 0:                                          # image y points down: make the normal point up
        n = -n
    d = -n @ c
    rms = float(np.sqrt(np.mean((P @ n + d) ** 2)))
    return np.append(n, d), int(best_n), rms


def neck_pixel(k):
    def mid(i, j):
        a, b = k[i], k[j]
        if a[2] >= CONF and b[2] >= CONF:
            return 0.5 * (a[:2] + b[:2])
        if a[2] >= CONF:
            return a[:2]
        if b[2] >= CONF:
            return b[:2]
        return None
    p = mid(5, 6)
    return p if p is not None else mid(11, 12)


def run(vid):
    out = os.path.join(OUT, f"{vid}.npz")
    if os.path.exists(out):
        return "skip"
    kp = np.load(os.path.join(NPY, f"{vid}_raw_coco17.npy"))
    frames = DepthZip(vid)
    T = min(len(frames), len(kp))
    rng = np.random.default_rng(0)
    idx = np.linspace(0, len(frames) - 1, min(25, len(frames))).astype(int)
    stack = np.stack([frames[i][::2, ::2] for i in idx]).astype(np.float16)    # half resolution for the floor
    stack[stack == 0] = np.nan
    with np.errstate(all="ignore"):
        bg_half = np.nanmedian(stack.astype(np.float32), 0)
    del stack
    bg = np.nan_to_num(np.repeat(np.repeat(bg_half, 2, 0), 2, 1), nan=0.0)
    plane, n_inl, rms = fit_floor(bg, rng)
    xyz = np.full((len(kp), 3), np.nan); h = np.full(len(kp), np.nan)
    for t in range(T):
        p = neck_pixel(kp[t])
        if p is None:
            continue
        u, v = int(round(p[0])), int(round(p[1]))
        if not (0 <= u < 640 and 0 <= v < 480):
            continue
        patch = frames[t][max(0, v - 4):v + 5, max(0, u - 4):u + 5]
        patch = patch[(patch > 0.4) & (patch < 6.0)]
        if len(patch) < 5:
            continue
        z = float(np.percentile(patch, 20))
        X = backproject(float(u), float(v), z)
        xyz[t] = X; h[t] = X @ plane[:3] + plane[3]
    os.makedirs(OUT, exist_ok=True)
    np.savez(out, height_m=h, neck_xyz=xyz, plane=plane, plane_inliers=n_inl, floor_rms_m=rms,
             n_depth_frames=len(frames), n_rgb_frames=len(kp))
    return f"T={len(kp)} depth={len(frames)} floor inliers={n_inl} rms={rms * 100:.1f}cm median h={np.nanmedian(h):.2f}m"


if __name__ == "__main__":
    vids = sys.argv[1:] or sorted(json.load(open(os.path.join(ROOT, "splits", "split_scene_v2.json")))["group_of"])
    for v in vids:
        print(v, run(v), flush=True)
