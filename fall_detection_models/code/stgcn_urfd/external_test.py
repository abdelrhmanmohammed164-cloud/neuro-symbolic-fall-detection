"""
external_test.py
External test on GMDCSA-24: a dataset none of the models was trained, validated or tuned on.

PRE-REGISTERED (written before any model was run on GMDCSA-24):
  models    : the chosen configuration exp04_center: 3 seeds x 2 folds = 6 trained models (no retraining,
              no fine-tuning, threshold 0.5, each model's own validation temperature).
              Reference: baseline_v2 with the same 3 seeds x 2 folds, run the same way.
  primary   : mean and range over the 6 center models, on all 160 videos, of
              frame level  : MCC, F1, precision, recall, specificity, accuracy
              event level  : falls detected (of 79), median delay after the labelled fall start,
                             ADL videos with no false alarm (of 81), false-alarm episodes per minute of ADL
  secondary : the average probability of the 6 center models (one ensemble), same metrics;
              the same metrics split by source frame rate (30 vs 15 fps) and by subject (descriptive only).
  nothing is tuned on these results.

Usage:
  python external_test.py --data_dir external_data/gmdcsa24_npy --runs_root runs/exp_mcc --out reports/external_gmdcsa24
"""
import argparse
import json
import os

import numpy as np
import torch

from dataset import EvalWindows, load_sequences
from model import TwoStreamSTGCN
from train import binary_metrics, mcc, pick_device, predict

FPS = 30.0
CENTER = [(n, k) for n in ("exp04_center", "exp04_center_s43", "exp04_center_s44") for k in (0, 1)]
BASE = [(n, k) for n in ("baseline_v2_s42", "baseline_v2_s43", "baseline_v2_s44") for k in (0, 1)]


def run_model(root, name, k, seqs, dev):
    fd = os.path.join(root, name, f"fold{k}")
    ck = torch.load(os.path.join(fd, "best_checkpoint.pt"), map_location="cpu", weights_only=False)
    cfg = ck["config"]
    T = json.load(open(os.path.join(fd, "result.json")))["temperature"]
    m = TwoStreamSTGCN(use_conf=cfg["use_conf"]).to(dev); m.load_state_dict(ck["model"]); m.eval()
    ds = EvalWindows(seqs, cfg["window"], stride=1, norm_mode=cfg["norm_mode"], pad_side=cfg.get("pad_side", "right"))
    lg, y, si, en = predict(m, ds, dev)
    p = torch.sigmoid(torch.tensor(lg / T)).numpy()
    return p, y, si, en


def evaluate(p, y, si, en, seqs, subset=None):
    keep = np.ones(len(p), bool) if subset is None else np.isin(si, subset)
    m = binary_metrics(y[keep], p[keep], 0.5)
    det = falls = clean = adl = eps = adl_frames = 0; lat = []
    for j, s in enumerate(seqs):
        if subset is not None and j not in subset:
            continue
        idx = np.nonzero(si == j)[0]; idx = idx[np.argsort(en[idx])]
        lab = y[idx].astype(int); pr = p[idx] >= 0.5
        if s["label"] == 1:
            falls += 1; on = int(np.argmax(lab)); d = np.nonzero(pr[on:])[0]
            if len(d):
                det += 1; lat.append(d[0] / FPS)
        else:
            adl += 1; clean += int(not pr.any()); adl_frames += len(idx)
            eps += int(pr[0]) + int(((pr[1:].astype(int) - pr[:-1].astype(int)) == 1).sum())
    return dict(mcc=mcc(m), f1=m["f1"], precision=m["precision"], recall=m["recall_sensitivity"],
                specificity=m["specificity"], accuracy=m["accuracy"], tp=m["tp"], fp=m["fp"], fn=m["fn"], tn=m["tn"],
                falls_detected=det, falls=falls, latency_s_median=float(np.median(lat)) if lat else None,
                adl_clean=clean, adl=adl, fa_per_min=eps / (adl_frames / FPS / 60) if adl_frames else None,
                always_no_fall_accuracy=float((y[keep] == 0).mean()))


def rng(vals, nd=3):
    v = [x for x in vals if x is not None and not (isinstance(x, float) and np.isnan(x))]
    return f"{np.mean(v):.{nd}f} ({min(v):.{nd}f} to {max(v):.{nd}f})" if v else "-"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True); ap.add_argument("--runs_root", required=True)
    ap.add_argument("--out", required=True); ap.add_argument("--device", default=None)
    a = ap.parse_args()
    dev = pick_device(a.device)
    seqs = load_sequences(a.data_dir)
    meta = {s["name"]: json.load(open(os.path.join(a.data_dir, s["name"] + "_meta.json"))) for s in seqs}
    print(f"{len(seqs)} videos: {sum(s['label'] for s in seqs)} falls, {sum(1 - s['label'] for s in seqs)} ADL | device {dev}")
    groups = {"all": None,
              "30 fps source": [j for j, s in enumerate(seqs) if meta[s["name"]]["fps_source"] >= 25],
              "15 fps source": [j for j, s in enumerate(seqs) if meta[s["name"]]["fps_source"] < 25]}
    for subj in sorted({meta[s["name"]]["subject"] for s in seqs}):
        groups[f"subject {subj}"] = [j for j, s in enumerate(seqs) if meta[s["name"]]["subject"] == subj]
    res, probs = {}, {}
    for label, models in (("center", CENTER), ("baseline_v2", BASE)):
        res[label] = {}; probs[label] = []
        for name, k in models:
            p, y, si, en = run_model(a.runs_root, name, k, seqs, dev)
            probs[label].append(p)
            res[label][f"{name}/fold{k}"] = {g: evaluate(p, y, si, en, seqs, sub) for g, sub in groups.items()}
            r = res[label][f"{name}/fold{k}"]["all"]
            print(f"{label:12s} {name}/fold{k}: MCC {r['mcc']:.3f} F1 {r['f1']:.3f} falls {r['falls_detected']}/{r['falls']} "
                  f"ADL clean {r['adl_clean']}/{r['adl']}", flush=True)
        pe = np.mean(probs[label], axis=0)
        res[label]["ensemble (mean probability of the 6 models)"] = {g: evaluate(pe, y, si, en, seqs, sub) for g, sub in groups.items()}
    os.makedirs(a.out, exist_ok=True)
    json.dump(dict(rule=__doc__.split("PRE-REGISTERED")[1].split("Usage:")[0].strip(), results=res,
                   n_videos=len(seqs)), open(os.path.join(a.out, "external_test.json"), "w"), indent=1, default=float)
    L = ["# External test on GMDCSA-24 (never used for training, validation or tuning)", "",
         f"{len(seqs)} videos ({sum(s['label'] for s in seqs)} falls, {sum(1 - s['label'] for s in seqs)} ADL), 4 subjects, 3 homes, "
         "laptop webcam. Threshold 0.5. Mean over the 6 models of each configuration, range in brackets.", ""]
    keys = [("MCC", "mcc", 3), ("F1", "f1", 3), ("Precision", "precision", 3), ("Recall", "recall", 3),
            ("Specificity", "specificity", 3), ("Accuracy", "accuracy", 3), ("Falls detected", "falls_detected", 1),
            ("Median delay (s)", "latency_s_median", 2), ("ADL with no false alarm", "adl_clean", 1),
            ("False alarms / min", "fa_per_min", 2)]
    for g in groups:
        L += [f"## {g}", "", "| metric | center (6 models) | baseline_v2 (6 models) | center ensemble |", "|---|---|---|---|"]
        for title, key, nd in keys:
            vals = {lab: [v[g][key] for n, v in res[lab].items() if not n.startswith("ensemble")] for lab in res}
            ens = res["center"]["ensemble (mean probability of the 6 models)"][g][key]
            tot = ""
            if key == "falls_detected":
                tot = f" of {res['center']['exp04_center/fold0'][g]['falls']}"
            if key == "adl_clean":
                tot = f" of {res['center']['exp04_center/fold0'][g]['adl']}"
            L.append(f"| {title} | {rng(vals['center'], nd)}{tot} | {rng(vals['baseline_v2'], nd)}{tot} | "
                     f"{'-' if ens is None else f'{ens:.{nd}f}'}{tot} |")
        L.append(f"\nAccuracy of always predicting 'no fall' here: "
                 f"{res['center']['exp04_center/fold0'][g]['always_no_fall_accuracy']:.3f}\n")
    open(os.path.join(a.out, "external_test.md"), "w").write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
