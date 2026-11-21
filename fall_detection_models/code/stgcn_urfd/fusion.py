"""
fusion.py
Apply a frozen fusion rule to frame-aligned neural and symbolic predictions. Nothing is fitted here.

Rules
  neurosymbolic  [p >= t_n AND neck height < t_up] OR [symbolic AND p >= t_low]  (thresholds from
                 reports/fusion_neurosymbolic/final_config.json, chosen per fold without test labels;
                 needs the metric neck heights)
  or             p >= 0.5 OR symbolic
  and            p >= 0.5 AND symbolic

Inputs must cover the same (sequence, 1-based frame) keys. Missing, duplicated or unaligned frames are refused,
never filled in.

URFD (each test video uses the thresholds of its own fold):
  python fusion.py --neural ../../../reports/symbolic_v3_precision/neural_test_oof.csv \
      --symbolic symbolic_oof.csv --rule neurosymbolic \
      --config ../../../reports/fusion_neurosymbolic/final_config.json \
      --height_dir ../../../processed_data/kinect_height \
      --split_file ../../../splits/split_scene_v2.json --allow_posthoc_test --out fused.csv
Other data: pass --fold 0 or --fold 1 instead of --split_file to pick one frozen config.
"""
import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np

from symbolic_v3_improved import ffill

RULES = ("neurosymbolic", "or", "and")


def read_predictions(path, field):
    result = {}
    with open(path, newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"sequence", "frame", field}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError(f"{path}: missing columns {sorted(required - set(reader.fieldnames or []))}")
        for row in reader:
            seq, frame = row["sequence"].strip(), int(row["frame"])
            if not seq or frame < 1:
                raise ValueError(f"{path}: invalid sequence/frame")
            key = (seq, frame)
            if key in result:
                raise ValueError(f"{path}: duplicate sequence/frame {key}")
            value = float(row[field])
            if not math.isfinite(value) or not 0 <= value <= 1:
                raise ValueError(f"{path}: invalid {field} at {key}: {value}")
            label = None
            if row.get("label", "") != "":
                label = int(row["label"])
                if label not in (0, 1):
                    raise ValueError(f"{path}: nonbinary label at {key}")
            result[key] = (value, label)
    if not result:
        raise ValueError(f"{path}: empty predictions")
    return result


def load_heights(height_dir, sequences):
    out = {}
    for seq in sequences:
        path = Path(height_dir) / f"{seq}.npz"
        if not path.exists():
            raise FileNotFoundError(f"missing metric neck heights: {path}")
        with np.load(path, allow_pickle=False) as z:
            out[seq] = ffill(np.asarray(z["height_m"], dtype=float))   # same gap filling as V3 (up to 5 frames)
    return out


def fuse(neural, symbolic, rule="neurosymbolic", threshold=0.5, heights=None, config_of=None):
    """config_of: sequence -> dict(t_n, t_up, t_low); heights: sequence -> array (frame 1 = index 0)."""
    if rule not in RULES or not 0 <= threshold <= 1:
        raise ValueError("invalid fusion rule or threshold")
    if neural.keys() != symbolic.keys():
        a, b = neural.keys() - symbolic.keys(), symbolic.keys() - neural.keys()
        raise ValueError(f"unaligned frame keys: neural-only {len(a)}, symbolic-only {len(b)}")
    if rule == "neurosymbolic" and (heights is None or config_of is None):
        raise ValueError("neurosymbolic needs heights and a frozen config")
    rows = []
    for seq, frame in sorted(neural):
        p, y = neural[seq, frame]
        s, sy = symbolic[seq, frame]
        if s not in (0.0, 1.0):
            raise ValueError(f"symbolic output must be binary at {(seq, frame)}")
        if y is not None and sy is not None and y != sy:
            raise ValueError(f"conflicting labels at {(seq, frame)}")
        row = dict(sequence=seq, frame=frame, label=y if y is not None else sy, neural_prob=p, symbolic_alarm=int(s))
        if rule == "or":
            alarm = p >= threshold or s == 1
        elif rule == "and":
            alarm = p >= threshold and s == 1
        else:
            c, h = config_of[seq], heights[seq]
            hh = h[frame - 1] if frame - 1 < len(h) else np.nan
            if not c.get("adopted", True):                          # fold kept the neural ensemble
                alarm = p >= threshold
            else:
                up = bool(np.isfinite(hh) and hh >= c["t_up"])       # missing height never vetoes
                alarm = (p >= c["t_n"] and not up) or (s == 1 and p >= c["t_low"])
            row["neck_height_m"] = None if not np.isfinite(hh) else round(float(hh), 4)
        row["fused_alarm"] = int(alarm)
        rows.append(row)
    return rows


def evaluate(rows):
    if any(r["label"] is None for r in rows):
        return {"labeled": False, "frames": len(rows)}
    y = np.array([r["label"] for r in rows]); p = np.array([r["fused_alarm"] for r in rows])
    tp, fp = int(((p == 1) & (y == 1)).sum()), int(((p == 1) & (y == 0)).sum())
    fn, tn = int(((p == 0) & (y == 1)).sum()), int(((p == 0) & (y == 0)).sum())
    pr = tp / (tp + fp) if tp + fp else 0.0; rc = tp / (tp + fn) if tp + fn else 0.0; sp = tn / (tn + fp) if tn + fp else 0.0
    d = math.sqrt(float((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)))
    by_seq = {}
    for r in rows:
        by_seq.setdefault(r["sequence"], []).append(r)
    falls = {s: rs for s, rs in by_seq.items() if any(r["label"] for r in rs)}
    adl = {s: rs for s, rs in by_seq.items() if not any(r["label"] for r in rs)}
    def detected(rs):                                           # any alarm from the first fall frame on
        onset = next(i for i, r in enumerate(rs) if r["label"])
        return any(r["fused_alarm"] for r in rs[onset:])
    return dict(labeled=True, frames=len(rows), TP=tp, FP=fp, FN=fn, TN=tn, precision=pr, recall=rc,
                f1=2 * pr * rc / (pr + rc) if pr + rc else 0.0, accuracy=(tp + tn) / len(rows),
                balanced_accuracy=(rc + sp) / 2, mcc=(tp * tn - fp * fn) / d if d else 0.0,
                fall_videos_detected=sum(detected(rs) for rs in falls.values()), fall_videos=len(falls),
                adl_videos_without_alarm=sum(not any(r["fused_alarm"] for r in rs) for rs in adl.values()),
                adl_videos=len(adl))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--neural", required=True, help="CSV: sequence, frame (1-based), prob[, label]")
    ap.add_argument("--symbolic", required=True, help="CSV: sequence, frame (1-based), alarm[, label]")
    ap.add_argument("--out", required=True)
    ap.add_argument("--rule", choices=RULES, default="neurosymbolic")
    ap.add_argument("--threshold", type=float, default=0.5, help="neural threshold for 'or' / 'and'; never tune on test")
    ap.add_argument("--config", help="frozen fusion config (reports/fusion_neurosymbolic/final_config.json)")
    ap.add_argument("--height_dir", help="folder with <sequence>.npz metric neck heights")
    ap.add_argument("--fold", type=int, help="which frozen config to use for non-URFD data")
    ap.add_argument("--split_file", help="URFD: check the predictions cover exactly the test folds; each video uses its fold's config")
    ap.add_argument("--allow_posthoc_test", action="store_true", help="acknowledge that the URFD test folds were already seen")
    a = ap.parse_args(argv)
    n, s = read_predictions(a.neural, "prob"), read_predictions(a.symbolic, "alarm")
    seqs = sorted({k[0] for k in n})
    fold_of = None
    if a.split_file:
        split = json.loads(Path(a.split_file).read_text())
        fold_of = {v: f["fold"] for f in split["folds"] for v in f["test"]}
        if len(fold_of) != sum(len(f["test"]) for f in split["folds"]):
            raise ValueError("overlapping test folds")
        if set(seqs) != set(fold_of):
            raise ValueError(f"URFD predictions must cover exactly the test sequences: missing {len(set(fold_of) - set(seqs))}, extra {len(set(seqs) - set(fold_of))}")
        if not a.allow_posthoc_test:
            raise ValueError("the URFD test folds were already seen: pass --allow_posthoc_test (results are post-hoc)")
    heights = config_of = None
    if a.rule == "neurosymbolic":
        if not (a.config and a.height_dir):
            ap.error("--rule neurosymbolic needs --config and --height_dir")
        cfgs = json.loads(Path(a.config).read_text())
        if fold_of is None and a.fold is None:
            ap.error("pass --split_file (URFD) or --fold")
        pick = lambda v: cfgs[str(fold_of[v] if fold_of is not None else a.fold)]
        config_of = {v: pick(v) for v in seqs}
        heights = load_heights(a.height_dir, seqs)
    rows = fuse(n, s, a.rule, a.threshold, heights, config_of)
    out = Path(a.out); out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    m = evaluate(rows)
    m.update(rule=a.rule, posthoc=bool(a.split_file))
    out.with_suffix(".metrics.json").write_text(json.dumps(m, indent=2) + "\n")
    print(json.dumps(m, indent=2))


if __name__ == "__main__":
    main()
