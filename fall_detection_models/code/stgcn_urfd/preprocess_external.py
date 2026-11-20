"""
preprocess_external.py
GMDCSA-24 videos -> the same skeleton tensors as URFD, for an external test the model never trained on.

Identical to the URFD pipeline (functions imported, not copied):
  RTMO-l (ONNX via rtmlib, det_thr 0.1), one person per frame (pick_person), COCO-17 -> 14 joints,
  confidence < 0.3 missing, gaps <= 5 frames interpolated, 5-frame moving average (clean_joints),
  x/W and y/H. Depth is not computed (the neural stream does not use it).

Decided BEFORE any result on this dataset existed:
  - Frame rate: the model was trained on 30 fps windows of 80 frames. Every video is resampled to
    30 fps on its raw keypoints before cleaning: each 30 fps time point takes the linear interpolation
    of the two nearest source frames (confidence = the smaller one); if either source frame has no
    person, the nearer frame is copied. 30 fps videos are unchanged; 29 and 15 fps videos are resampled.
  - Labels (from Subject N/Fall.csv, ADL.csv, times in seconds): a 30 fps frame is a fall frame if its
    time lies inside a "Falling" segment. Several Falling segments are joined. A missing "to"
    ("[2.3 6]") is read as an interval, and a stray colon after a bracket ("[1.8 to 5]:") is ignored. A Falling segment with no times (Subject 4 Fall 15) runs from
    the end of the previous segment to the end of the video. ADL videos have no fall frames.
  - Names: fall-s<subject>-<nn> and adl-s<subject>-<nn>, so dataset.load_sequences gives the video label.

Outputs (resumable: a finished video is skipped; files written atomically):
  <out_dir>/<name>.npy (3, T30, 14, 1), <name>_frame_labels.npy (T30,), <name>_raw_coco17.npy,
  <name>_meta.json

Usage:
  python preprocess_external.py --data_dir external_data/GMDCSA24 --out_dir external_data/gmdcsa24_npy \
      --rtmo fall_detection_models/weights/rtmo-l_onnx/end2end.onnx --device mps
"""
import argparse
import csv
import glob
import json
import os
import re
import time

import cv2
import numpy as np

from graph import coco17_to_urfd14
from preprocess_sequence import clean_joints, pick_person

TARGET_FPS = 30.0
SEG = re.compile(r"([A-Za-z][^\[\];:]*?)\s*(?:\[\s*([\d.]+)\s*(?:to)?\s*([\d.]+)\s*\])?\s*:?\s*(?:;|$)")


def parse_labels(cls, duration):
    """'Falling (SW)[3.4 to 6]; Sitting[0 to 3.4]' -> list of (start, end) seconds of falling."""
    segs = [(n.strip(), a, b) for n, a, b in SEG.findall(cls.strip()) if n.strip()]
    out, prev_end = [], 0.0
    for name, a, b in segs:
        if a and b:
            a, b = float(a), float(b)
        if name.lower().startswith("fall"):
            if a == "" or b == "":
                # no times given: from the end of the previous segment to the end of the video
                ends = [float(y) for _, _, y in segs if y]
                a, b = (max(ends) if ends else prev_end), duration
            out.append((float(a), float(b)))
        if b != "":
            prev_end = max(prev_end, float(b))
    return out


def read_annotations(data_dir):
    ann = {}
    for csv_path in sorted(glob.glob(os.path.join(data_dir, "Subject *", "*.csv"))):
        subj = re.search(r"Subject (\d+)", csv_path).group(1)
        kind = "fall" if os.path.basename(csv_path).lower().startswith("fall") else "adl"
        for r in csv.reader(open(csv_path, encoding="utf-8-sig")):
            if not r or r[0].startswith("File"):
                continue
            vid = r[0].strip()
            name = f"{kind}-s{subj}-{vid.split('.')[0]}"
            ann[name] = dict(path=os.path.join(os.path.dirname(csv_path), "Fall" if kind == "fall" else "ADL", vid),
                             kind=kind, subject=int(subj), classes=",".join(r[5:]).strip(),
                             description=r[4].strip() if len(r) > 4 else "", time=r[2].strip() if len(r) > 2 else "")
    return ann


def resample_raw(raw, fps):
    """raw (T, 17, 3) at fps -> (T30, 17, 3) at 30 fps (see module docstring)."""
    T = raw.shape[0]
    if abs(fps - TARGET_FPS) < 0.5:
        return raw.copy()
    T30 = int(round(T * TARGET_FPS / fps))
    out = np.full((T30, 17, 3), np.nan, np.float32)
    for k in range(T30):
        s = k * fps / TARGET_FPS
        i0 = min(int(np.floor(s)), T - 1); i1 = min(i0 + 1, T - 1); w = s - i0
        a, b = raw[i0], raw[i1]
        if np.isnan(a[0, 0]) or np.isnan(b[0, 0]):
            out[k] = a if (w < 0.5 or np.isnan(b[0, 0])) else b
        else:
            out[k, :, :2] = (1 - w) * a[:, :2] + w * b[:, :2]
            out[k, :, 2] = np.minimum(a[:, 2], b[:, 2])
    return out


def atomic_save(path, arr=None, text=None):
    tmp = path + f".tmp-{os.getpid()}"
    with open(tmp, "wb") as f:
        if arr is not None:
            np.save(f, arr)
        else:
            f.write(text.encode())
        f.flush(); os.fsync(f.fileno())
    os.replace(tmp, path)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", required=True); p.add_argument("--out_dir", required=True)
    p.add_argument("--rtmo", required=True); p.add_argument("--device", default="cpu")
    p.add_argument("--det_thr", type=float, default=0.1)
    p.add_argument("--only", nargs="*", default=None)
    a = p.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)
    ann = read_annotations(a.data_dir)
    names = sorted(n for n in ann if not a.only or n in a.only)
    from rtmlib import RTMO
    try:
        model = RTMO(a.rtmo, score_thr=a.det_thr, backend="onnxruntime", device=a.device)
        model(np.zeros((480, 640, 3), np.uint8)); dev = a.device
    except Exception as e:
        print(f"RTMO on {a.device} failed ({e}); using cpu")
        model = RTMO(a.rtmo, score_thr=a.det_thr, backend="onnxruntime", device="cpu"); dev = "cpu"
    t_all = time.time()
    for i, n in enumerate(names):
        meta_path = os.path.join(a.out_dir, n + "_meta.json")
        if os.path.exists(meta_path) and os.path.exists(os.path.join(a.out_dir, n + ".npy")):
            print(f"[{i + 1}/{len(names)}] {n}: done -> skip", flush=True)
            continue
        v = ann[n]; t0 = time.time()
        cap = cv2.VideoCapture(v["path"]); fps = cap.get(cv2.CAP_PROP_FPS)
        W, H = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        raw, prev = [], None
        while True:
            ok, img = cap.read()
            if not ok:
                break
            k, s = model(img); k, s = np.asarray(k), np.asarray(s)
            kp = np.full((17, 3), np.nan, np.float32)
            j = pick_person(k, s, prev) if k.size else None
            if j is not None:
                kp[:, :2], kp[:, 2] = k[j], s[j]; prev = (k[j, 5] + k[j, 6]) / 2
            raw.append(kp)
        cap.release()
        raw = np.stack(raw); T_src = len(raw); dur = T_src / fps
        raw30 = resample_raw(raw, fps); T = len(raw30)
        j14 = clean_joints(coco17_to_urfd14(np.nan_to_num(raw30, nan=0.0)))
        miss = np.isnan(j14[..., 0])
        x = np.zeros((3, T, 14, 1), np.float32)
        x[0, :, :, 0] = np.where(miss, 0, j14[..., 0] / W)
        x[1, :, :, 0] = np.where(miss, 0, j14[..., 1] / H)
        x[2, :, :, 0] = np.where(miss, 0, j14[..., 2])
        segs = parse_labels(v["classes"], dur) if v["kind"] == "fall" else []
        tt = np.arange(T) / TARGET_FPS
        lab = np.zeros(T, np.int64)
        for s0, s1 in segs:
            lab[(tt >= s0) & (tt <= s1)] = 1
        if v["kind"] == "fall" and lab.sum() == 0:
            raise SystemExit(f"{n}: fall video without fall frames ({v['classes']})")
        atomic_save(os.path.join(a.out_dir, n + ".npy"), x)
        atomic_save(os.path.join(a.out_dir, n + "_frame_labels.npy"), lab)
        atomic_save(os.path.join(a.out_dir, n + "_raw_coco17.npy"), raw30)
        meta = dict(name=n, source=os.path.relpath(v["path"], a.data_dir), subject=v["subject"], kind=v["kind"],
                    fps_source=fps, frames_source=T_src, frames_30fps=T, image_wh=[W, H], fall_segments_s=segs,
                    fall_frames=int(lab.sum()), classes=v["classes"], description=v["description"],
                    recording=v["time"], frames_with_person=int((~np.isnan(raw[:, 0, 0])).sum()),
                    missing_joint_pct=float(miss.mean() * 100), pose_device=dev, det_thr=a.det_thr)
        atomic_save(meta_path, text=json.dumps(meta, indent=1))
        print(f"[{i + 1}/{len(names)}] {n}: {T_src} frames @ {fps:.0f} fps -> {T} @ 30 | person {meta['frames_with_person']}"
              f" | fall frames {meta['fall_frames']} | {time.time() - t0:.0f} s", flush=True)
    print(f"ALL_DONE {len(names)} videos in {(time.time() - t_all) / 60:.1f} min")


if __name__ == "__main__":
    main()
