"""
preprocess_sequence.py
RGB frames of one URFD sequence -> ST-GCN input + neck depth.

Steps (paper Section 3.2 unless marked [E] = engineering choice):
  1. RTMO-l (ONNX, via rtmlib) on every frame -> COCO-17 keypoints + confidence.
  2. Keep one person per frame: highest mean score, or nearest to the previous neck [E].
  3. COCO-17 -> 14 joints (Neck = shoulder midpoint).
  4. Confidence < 0.3 -> missing; gaps <= 5 frames linearly interpolated; 5-frame moving average.
  5. Normalise x / W, y / H.
  6. Depth-Anything-V2 (relative, inverse depth) per frame; neck value = median of a 5x5 patch [E].

Outputs in --out_dir:
  <seq>.npy               float32 (3, T, 14, 1): x/W, y/H, conf (missing joints = 0)
  <seq>_neck_depth.npy    float32 (T,): Depth-Anything relative inverse depth at the neck (NaN if no neck)
  <seq>_raw_coco17.npy    float32 (T, 17, 3): raw RTMO output for the chosen person (NaN if none)

Usage:
  python preprocess_sequence.py --rgb_dir fall-01-cam0-rgb --seq fall-01 \
      --rtmo weights/rtmo-l_onnx/end2end.onnx \
      --da_weights weights/depth_anything_v2_vits.pth --da_repo code/Depth-Anything-V2 \
      --out_dir urfd_npy
"""
import argparse
import glob
import os
import sys

import cv2
import numpy as np

from graph import CENTER, coco17_to_urfd14

CONF_THR, MAX_GAP, SMOOTH = 0.3, 5, 5


def pick_person(kpts, scores, prev_neck):
    """kpts (N,17,2), scores (N,17) -> index of the person to keep, or None."""
    if len(kpts) == 0:
        return None
    if prev_neck is None or len(kpts) == 1:
        return int(np.argmax(scores.mean(1)))
    necks = (kpts[:, 5] + kpts[:, 6]) / 2
    return int(np.argmin(np.linalg.norm(necks - prev_neck, axis=1)))


def body_anchor(raw, thr=CONF_THR):
    """Point used for neck height / depth (u, v in pixels) + source per frame [E].

    source 1 = neck (both shoulders >= thr, as in the paper)
    source 2 = mid-hip fallback (both hips >= thr), used when the shoulders are not visible,
               e.g. lying on the floor with the head turned away. Lying flat, hip and neck are
               at almost the same height; standing, the hips are far above the 0.19 m floor
               threshold, so the fallback cannot create a false 'near the floor'.
    source 0 = none (NaN)
    """
    T = raw.shape[0]
    out = np.full((T, 3), np.nan, np.float32)
    out[:, 2] = 0
    for t in range(T):
        k = raw[t]
        if np.isnan(k[0, 0]):
            continue
        if k[5, 2] >= thr and k[6, 2] >= thr:
            out[t] = [*(k[5, :2] + k[6, :2]) / 2, 1]
        elif k[11, 2] >= thr and k[12, 2] >= thr:
            out[t] = [*(k[11, :2] + k[12, :2]) / 2, 2]
    # 5-frame moving average on each source run (paper smoothing), no mixing across sources.
    sm = out.copy()
    for t in range(T):
        if out[t, 2] == 0:
            continue
        w = [s for s in range(max(0, t - 2), min(T, t + 3)) if out[s, 2] == out[t, 2]]
        sm[t, :2] = out[w, :2].mean(0)
    return sm


def run_pose(frames, rtmo_path, device="cpu", det_thr=0.1):
    from rtmlib import RTMO
    try:                                              # mps -> CoreML on the Mac GPU / Neural Engine
        model = RTMO(rtmo_path, score_thr=det_thr, backend="onnxruntime", device=device)
        model(cv2.imread(frames[0]))                  # fail early if the provider can't run it
    except Exception as e:
        print(f"RTMO on '{device}' failed ({type(e).__name__}: {e}); falling back to cpu")
        model = RTMO(rtmo_path, score_thr=det_thr, backend="onnxruntime", device="cpu")
    out = np.full((len(frames), 17, 3), np.nan, np.float32)
    prev = None
    for t, f in enumerate(frames):
        img = cv2.imread(f)
        k, s = model(img)
        k, s = np.asarray(k), np.asarray(s)
        i = pick_person(k, s, prev) if k.size else None
        if i is not None:
            out[t, :, :2], out[t, :, 2] = k[i], s[i]
            prev = (k[i, 5] + k[i, 6]) / 2
    return out


def clean_joints(j14):
    """j14 (T, 14, 3) in pixels. Threshold, interpolate short gaps, smooth. Missing -> NaN."""
    x = j14.copy()
    bad = ~(x[..., 2] >= CONF_THR)                    # also catches NaN
    x[bad] = np.nan
    T = x.shape[0]
    for v in range(x.shape[1]):
        valid = ~np.isnan(x[:, v, 0])
        idx = np.nonzero(valid)[0]
        if len(idx) < 2:
            continue
        for a, b in zip(idx[:-1], idx[1:]):           # interpolate gaps of <= MAX_GAP frames
            if 1 < b - a <= MAX_GAP + 1:
                w = (np.arange(a + 1, b) - a) / (b - a)
                x[a + 1:b, v, :2] = x[a, v, :2] + w[:, None] * (x[b, v, :2] - x[a, v, :2])
                x[a + 1:b, v, 2] = np.minimum(x[a, v, 2], x[b, v, 2])
        # moving average over valid frames only (window SMOOTH, centred)
        xy = x[:, v, :2]
        sm = xy.copy()
        h = SMOOTH // 2
        for t in range(T):
            win = xy[max(0, t - h):t + h + 1]
            if not np.isnan(xy[t, 0]):
                sm[t] = np.nanmean(win, axis=0)
        x[:, v, :2] = sm
    return x


def run_depth(frames, necks_px, da_repo, da_weights, encoder="vits", device="cpu", patch=2):
    sys.path.insert(0, da_repo)
    import torch
    from depth_anything_v2.dpt import DepthAnythingV2
    cfg = {"vits": dict(features=64, out_channels=[48, 96, 192, 384]),
           "vitb": dict(features=128, out_channels=[96, 192, 384, 768])}[encoder]
    dev = device
    model = DepthAnythingV2(encoder=encoder, **cfg)
    model.load_state_dict(torch.load(da_weights, map_location="cpu"))
    model = model.to(dev).eval()
    out = np.full(len(frames), np.nan, np.float32)
    maps = {}
    for t, f in enumerate(frames):
        img = cv2.imread(f)
        d = model.infer_image(img, 518)               # (H, W) relative inverse depth
        if not np.isnan(necks_px[t, 0]):
            u, v = int(round(necks_px[t, 0])), int(round(necks_px[t, 1]))
            H, W = d.shape
            u, v = np.clip(u, 0, W - 1), np.clip(v, 0, H - 1)
            out[t] = np.median(d[max(0, v - patch):v + patch + 1, max(0, u - patch):u + patch + 1])
        if t in (0, len(frames) // 2, len(frames) - 1):
            maps[t] = d
    return out, maps


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--rgb_dir", required=True)
    p.add_argument("--seq", required=True)
    p.add_argument("--rtmo", required=True)
    p.add_argument("--da_weights", required=True)
    p.add_argument("--da_repo", required=True)
    p.add_argument("--encoder", default="vits", choices=["vits", "vitb"])
    p.add_argument("--out_dir", required=True)
    p.add_argument("--device", default="auto", choices=["auto", "mps", "cuda", "cpu"],
                   help="auto: mps on Apple Silicon, cuda on NVIDIA, else cpu")
    a = p.parse_args()
    if a.device == "auto":
        import torch
        a.device = "cuda" if torch.cuda.is_available() else (
            "mps" if torch.backends.mps.is_available() else "cpu")
    print("device:", a.device)
    os.makedirs(a.out_dir, exist_ok=True)

    frames = sorted(glob.glob(os.path.join(a.rgb_dir, "*.png")),
                    key=lambda s: int(os.path.splitext(s)[0].rsplit("-", 1)[-1]))
    H, W = cv2.imread(frames[0]).shape[:2]
    raw_path = os.path.join(a.out_dir, f"{a.seq}_raw_coco17.npy")
    if os.path.exists(raw_path):                      # reuse pose results if already computed
        raw = np.load(raw_path)
    else:
        raw = run_pose(frames, a.rtmo, a.device)
        np.save(raw_path, raw)

    j14 = clean_joints(coco17_to_urfd14(np.nan_to_num(raw, nan=0.0)))
    anchor = body_anchor(raw)                         # (T, 3): u, v, source
    np.save(os.path.join(a.out_dir, f"{a.seq}_anchor.npy"), anchor)
    depth, _ = run_depth(frames, anchor[:, :2], a.da_repo, a.da_weights, a.encoder, a.device)
    np.save(os.path.join(a.out_dir, f"{a.seq}_neck_depth.npy"), depth)

    miss = np.isnan(j14[..., 0])
    x = np.zeros((3, len(frames), 14, 1), np.float32)
    x[0, :, :, 0] = np.where(miss, 0, j14[..., 0] / W)
    x[1, :, :, 0] = np.where(miss, 0, j14[..., 1] / H)
    x[2, :, :, 0] = np.where(miss, 0, j14[..., 2])
    np.save(os.path.join(a.out_dir, f"{a.seq}.npy"), x)
    src = anchor[:, 2]
    print(f"{a.seq}: T={len(frames)} frames with person={int((~miss).any(1).sum())} "
          f"missing joints={miss.mean():.1%} | anchor: neck {int((src == 1).sum())}, "
          f"hip fallback {int((src == 2).sum())}, none {int((src == 0).sum())} | "
          f"depth valid {np.isfinite(depth).mean():.1%}")


if __name__ == "__main__":
    main()
