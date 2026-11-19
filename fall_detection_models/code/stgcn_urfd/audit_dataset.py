"""
audit_dataset.py
Pre-training audit of the preprocessed URFD skeleton data (read-only).

Checks: inventory, readability, shapes/dtypes, finiteness, normalisation range, empty / all-zero /
static / repeated sequences, missing-joint encoding, label traceability, joint ordering against the
raw RTMO COCO-17 output, anatomical plausibility, exact and near duplicates, and shortcut risks
(sequence length, padding, missing-pose pattern, start position in the image).

Usage:
  python audit_dataset.py --npy_dir processed_data/cfg_x/urfd_npy --features_dir URFD/features \
      --out_dir reports
Writes: dataset_validation_report.csv, dataset_audit_summary.json
"""
import argparse
import csv
import hashlib
import itertools
import json
import os

import numpy as np

from graph import COCO17_INDEX, J, JOINTS

EXPECTED = [f"fall-{i:02d}" for i in range(1, 31)] + [f"adl-{i:02d}" for i in range(1, 41)]
SUFFIXES = ["", "_anchor", "_neck_depth", "_raw_coco17", "_timestamps", "_frame_labels"]


def auc(score, y):
    """Rank AUC of a scalar score for label y (0.5 = no information; 0 or 1 = perfect separation)."""
    score, y = np.asarray(score, float), np.asarray(y, int)
    order = np.argsort(score, kind="mergesort")
    ranks = np.empty(len(score)); ranks[order] = np.arange(1, len(score) + 1)
    for v in np.unique(score):                                   # average ties
        m = score == v; ranks[m] = ranks[m].mean()
    n1, n0 = y.sum(), (1 - y).sum()
    return float((ranks[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--npy_dir", required=True)
    p.add_argument("--features_dir", required=True)
    p.add_argument("--out_dir", required=True)
    a = p.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)

    feat = {}
    for fn in ("urfall-cam0-falls.csv", "urfall-cam0-adls.csv"):
        for r in csv.reader(open(os.path.join(a.features_dir, fn))):
            if r:
                feat.setdefault(r[0], {})[int(r[1])] = int(r[2])

    rows, data, problems = [], {}, []
    present = {f[:-4] for f in os.listdir(a.npy_dir) if f.endswith(".npy")}
    missing_seq = [s for s in EXPECTED if s not in present]
    extra_seq = sorted(s for s in present if s in EXPECTED) and \
        [s for s in present if not s.endswith(tuple(x for x in SUFFIXES if x)) and s not in EXPECTED]
    if missing_seq:
        problems.append(f"missing sequences: {missing_seq}")
    if extra_seq:
        problems.append(f"unexpected sequences: {extra_seq}")

    for s in EXPECTED:
        if s in missing_seq:
            continue
        r = dict(sequence=s, label=int(s.startswith("fall")))
        issues = []
        try:
            arr = {suf: np.load(os.path.join(a.npy_dir, f"{s}{suf}.npy"), allow_pickle=False) for suf in SUFFIXES}
            meta = json.load(open(os.path.join(a.npy_dir, f"{s}_meta.json")))
        except Exception as ex:
            r["issues"] = f"unreadable: {ex}"; rows.append(r); problems.append(f"{s} unreadable"); continue
        x, raw, fl = arr[""], arr["_raw_coco17"], arr["_frame_labels"]
        T = x.shape[1] if x.ndim == 4 else -1
        if x.shape != (3, T, 14, 1) or x.dtype != np.float32:
            issues.append(f"skeleton shape/dtype {x.shape} {x.dtype}")
        for suf, shp in (("_anchor", (T, 3)), ("_neck_depth", (T,)), ("_raw_coco17", (T, 17, 3)),
                         ("_timestamps", (T,)), ("_frame_labels", (T,))):
            if arr[suf].shape != shp:
                issues.append(f"{suf} shape {arr[suf].shape}")
        if not np.isfinite(x).all():
            issues.append("non-finite skeleton values")
        valid = x[2, :, :, 0] > 0                                  # (T, 14)
        xy = x[:2, :, :, 0]
        if (xy[:, ~valid] != 0).any():
            issues.append("missing joints are not encoded as 0")
        if valid.any():
            v = xy[:, valid]
            if v.min() < -0.02 or v.max() > 1.02:
                issues.append(f"coords outside [0,1]: {v.min():.3f}..{v.max():.3f}")
            if x[2][x[2] > 0].max() > 1.0001:
                issues.append("confidence > 1")
        if not valid.any():
            issues.append("ALL-ZERO sequence")
        # static / repeated frames
        both = valid[1:] & valid[:-1]
        disp = np.abs(np.diff(xy, axis=1)).sum(0)[both] if both.any() else np.array([0.0])
        rep = sum(np.array_equal(x[:, t], x[:, t + 1]) and valid[t].any() for t in range(T - 1))
        # label traceability
        if meta.get("video_id") != s or meta.get("n_frames") != T:
            issues.append("meta.json does not match")
        if r["label"] == 0 and fl.any():
            issues.append("fall frames inside an ADL")
        if r["label"] == 1 and not fl.any():
            issues.append("fall sequence without fall frames")
        urfd_fall = sum(1 for f, l in feat.get(s, {}).items() if l in (0, 1)) if r["label"] else 0
        if r["label"] and int(fl.sum()) != urfd_fall:
            issues.append(f"frame labels {int(fl.sum())} != URFD rows with label 0/1 ({urfd_fall})")
        # joint ordering vs raw RTMO output (COCO-17 index map in graph.py)
        rv = ~np.isnan(raw[:, 0, 0])
        map_err = []
        for name, ci in COCO17_INDEX.items():
            jv = valid[:, J[name]] & rv
            if jv.sum() > 5:
                map_err.append(np.median(np.abs(xy[0, jv, J[name]] * 640 - raw[jv, ci, 0])))
        neck_v = valid[:, J["Neck"]] & rv
        neck_err = np.median(np.abs(xy[0, neck_v, J["Neck"]] * 640 - (raw[neck_v, 5, 0] + raw[neck_v, 6, 0]) / 2)) \
            if neck_v.sum() > 5 else np.nan
        # anatomy in upright frames (URFD label -1): head above neck above hips above knees above ankles
        up = [f - 1 for f, l in feat.get(s, {}).items() if l == -1 and f <= T]
        order_ok = []
        chain = [J["Head"], J["Neck"], J["R_Hip"], J["R_Knee"], J["R_Ankle"]]
        for t in up:
            if valid[t, chain].all():
                ys = xy[1, t, chain]
                order_ok.append(bool(np.all(np.diff(ys) > 0)))
        r.update(frames=T, valid_joint_pct=round(100 * valid.mean(), 1),
                 frames_no_person=int((~valid.any(1)).sum()),
                 mean_abs_motion=round(float(disp.mean()), 5), repeated_frames=int(rep),
                 joint_map_err_px=round(float(np.max(map_err)), 2) if map_err else "",
                 neck_midpoint_err_px=round(float(neck_err), 2) if np.isfinite(neck_err) else "",
                 upright_order_pct=round(100 * np.mean(order_ok), 1) if order_ok else "",
                 fall_frames=int(fl.sum()), timestamps=meta.get("timestamps_source"),
                 input_fp=meta.get("input_fp", "")[:16],
                 sha256=hashlib.sha256(x.tobytes()).hexdigest()[:16],
                 start_neck_x=round(float(np.nanmean(np.where(valid[:10, 1], xy[0, :10, 1], np.nan))), 3)
                 if valid[:10, 1].any() else "",
                 start_neck_y=round(float(np.nanmean(np.where(valid[:10, 1], xy[1, :10, 1], np.nan))), 3)
                 if valid[:10, 1].any() else "",
                 issues="; ".join(issues))
        rows.append(r); data[s] = x
        if issues:
            problems.append(f"{s}: {'; '.join(issues)}")

    with open(os.path.join(a.out_dir, "dataset_validation_report.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)

    # duplicates: exact (array hash, source video hash) and near (resampled trajectories)
    dup_arr = [k for k, g in itertools.groupby(sorted(r["sha256"] for r in rows)) if len(list(g)) > 1]
    dup_src = [k for k, g in itertools.groupby(sorted(r["input_fp"] for r in rows)) if len(list(g)) > 1]
    def resample(x, n=64):
        t = np.linspace(0, x.shape[1] - 1, n).round().astype(int)
        return x[:2, t, :, 0]
    seqs = list(data)
    R = {s: resample(data[s]) for s in seqs}
    near = []
    for i, j in itertools.combinations(seqs, 2):
        d = np.abs(R[i] - R[j]).mean()
        if d < 0.01:
            near.append((i, j, round(float(d), 4)))
    dists = sorted(np.abs(R[i] - R[j]).mean() for i, j in itertools.combinations(seqs, 2))

    # shortcut risks: how well does a nuisance variable alone separate falls from ADLs?
    y = np.array([r["label"] for r in rows])
    def col(k):
        return np.array([float(r[k]) if r[k] != "" else np.nan for r in rows])
    T = col("frames")
    shortcut = {}
    for k in ["frames", "valid_joint_pct", "frames_no_person", "start_neck_x", "start_neck_y"]:
        v = col(k); m = np.isfinite(v)
        shortcut[k] = dict(auc=round(auc(v[m], y[m]), 3),
                           fall_median=float(np.nanmedian(v[y == 1])), adl_median=float(np.nanmedian(v[y == 0])))
    short = {"fall": int(((T < 80) & (y == 1)).sum()), "adl": int(((T < 80) & (y == 0)).sum())}

    summary = dict(
        expected=len(EXPECTED), present=len(rows), missing=missing_seq, problems=problems,
        label_counts=dict(fall=int(y.sum()), adl=int((1 - y).sum())),
        total_frames=int(T.sum()), fall_frames=int(sum(r["fall_frames"] for r in rows)),
        exact_duplicate_skeletons=dup_arr, duplicate_source_videos=dup_src,
        near_duplicates_lt_0p01=near, nearest_pair_distance=round(float(dists[0]), 4),
        pair_distance_p1=round(float(np.percentile(dists, 1)), 4),
        joint_map_err_px_max=max(r["joint_map_err_px"] for r in rows if r["joint_map_err_px"] != ""),
        neck_midpoint_err_px_max=max(r["neck_midpoint_err_px"] for r in rows if r["neck_midpoint_err_px"] != ""),
        upright_order_pct_median=float(np.median([r["upright_order_pct"] for r in rows if r["upright_order_pct"] != ""])),
        upright_order_pct_min=min((r["upright_order_pct"], r["sequence"]) for r in rows if r["upright_order_pct"] != ""),
        static_lowest_motion=sorted((r["mean_abs_motion"], r["sequence"]) for r in rows)[:3],
        repeated_frames_total=int(sum(r["repeated_frames"] for r in rows)),
        sequences_shorter_than_80=short, shortcut_auc=shortcut,
        timestamp_fallbacks=[r["sequence"] for r in rows if r["timestamps"] != "data_csv"],
    )
    json.dump(summary, open(os.path.join(a.out_dir, "dataset_audit_summary.json"), "w"), indent=1, default=str)
    print(json.dumps(summary, indent=1, default=str))


if __name__ == "__main__":
    main()
