"""
final_test_report.py
One-time test evaluation report for the configuration chosen on validation.

PRE-REGISTERED REPORTING (written before any test number of these runs existed):
  primary   : exp04_center with seeds 42, 43, 44 (the configuration CONFIRMED by compare_seeds.py).
              Reported as mean and range over the three seeds, per fold and pooled over both folds.
              No seed is singled out.
  reference : baseline_v2 with seeds 42, 43, 44, evaluated the same way. It only shows whether the
              validation gain carries over to the test folds; it changes no decision.
  metrics   : frame level (MCC, F1, precision, recall, specificity) at threshold 0.5, and event level
              (falls detected, median delay, ADL videos with no false alarm, false-alarm episodes per
              minute of ADL video).
  caveat    : the test folds were already seen once (first baseline, runs/stgcn_scene), so these numbers
              are post-hoc, not an independent test. Nothing is tuned after this report.

Usage:
  python final_test_report.py --runs_root runs/exp_mcc --out reports/final_test
"""
import argparse
import json
import os

import numpy as np

from train import binary_metrics, mcc

FPS = 30.0
PRIMARY = {42: "exp04_center", 43: "exp04_center_s43", 44: "exp04_center_s44"}
REFERENCE = {42: "baseline_v2_s42", 43: "baseline_v2_s43", 44: "baseline_v2_s44"}


def evaluate(frames):
    y = np.array([f["label"] for f in frames]); p = np.array([f["prob"] for f in frames])
    m = binary_metrics(y, p, 0.5)
    by = {}
    for f in frames:
        by.setdefault(f["sequence"], []).append(f)
    det = falls = clean = adl = eps = adl_frames = 0; lat = []
    for s, L in by.items():
        L.sort(key=lambda f: f["frame"])
        lab = np.array([f["label"] for f in L]); pr = np.array([f["prob"] for f in L]) >= 0.5
        if s.startswith("fall"):
            falls += 1; on = int(np.argmax(lab)); d = np.nonzero(pr[on:])[0]
            if len(d):
                det += 1; lat.append(d[0] / FPS)
        else:
            adl += 1; clean += int(not pr.any()); adl_frames += len(L)
            eps += int(pr[0]) + int(((pr[1:].astype(int) - pr[:-1].astype(int)) == 1).sum())
    return dict(mcc=mcc(m), f1=m["f1"], precision=m["precision"], recall=m["recall_sensitivity"],
                specificity=m["specificity"], tp=m["tp"], fp=m["fp"], fn=m["fn"], tn=m["tn"],
                falls_detected=det, falls=falls, latency_s_median=float(np.median(lat)) if lat else None,
                adl_clean=clean, adl=adl, fa_per_min=eps / (adl_frames / FPS / 60) if adl_frames else None)


def load(root, name, k):
    r = json.load(open(os.path.join(root, name, f"fold{k}", "result.json")))
    return r["frames"]


def summarise(root, runs):
    out = {}
    for seed, name in runs.items():
        f0, f1 = load(root, name, 0), load(root, name, 1)
        out[seed] = {"fold0": evaluate(f0), "fold1": evaluate(f1), "pooled": evaluate(f0 + f1)}
    return out


def fmt_range(vals, nd=3):
    v = [x for x in vals if x is not None]
    return f"{np.mean(v):.{nd}f} ({min(v):.{nd}f} to {max(v):.{nd}f})" if v else "-"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--runs_root", required=True); p.add_argument("--out", required=True)
    a = p.parse_args()
    res = {"primary (exp04_center)": summarise(a.runs_root, PRIMARY),
           "reference (baseline_v2)": summarise(a.runs_root, REFERENCE)}
    lines = ["# Final test evaluation (post-hoc: the test folds were seen once before)", "",
             "Mean over seeds 42, 43, 44, with the range in brackets. Threshold 0.5.", ""]
    for part in ("pooled", "fold0", "fold1"):
        lines += [f"## {part}", "", "| configuration | MCC | F1 | recall | specificity | falls detected | ADL with no false alarm | false alarms / min |",
                  "|---|---|---|---|---|---|---|---|"]
        for cfg, r in res.items():
            g = lambda key: [r[s][part][key] for s in (42, 43, 44)]
            falls = f"{fmt_range(g('falls_detected'), 1)} of {r[42][part]['falls']}"
            adl = f"{fmt_range(g('adl_clean'), 1)} of {r[42][part]['adl']}"
            lines.append(f"| {cfg} | {fmt_range(g('mcc'))} | {fmt_range(g('f1'))} | {fmt_range(g('recall'))} | "
                         f"{fmt_range(g('specificity'))} | {falls} | {adl} | {fmt_range(g('fa_per_min'), 2)} |")
        lines.append("")
    lines += ["## Per seed (pooled over both folds)", "", "| configuration | seed | MCC | F1 | falls | ADL clean | TP / FN / FP / TN |",
              "|---|---|---|---|---|---|---|"]
    for cfg, r in res.items():
        for s in (42, 43, 44):
            q = r[s]["pooled"]
            lines.append(f"| {cfg} | {s} | {q['mcc']:.3f} | {q['f1']:.3f} | {q['falls_detected']}/{q['falls']} | "
                         f"{q['adl_clean']}/{q['adl']} | {q['tp']} / {q['fn']} / {q['fp']} / {q['tn']} |")
    os.makedirs(a.out, exist_ok=True)
    open(os.path.join(a.out, "final_test.md"), "w").write("\n".join(lines) + "\n")
    json.dump(dict(rule=__doc__.split("PRE-REGISTERED REPORTING")[1].split("Usage:")[0].strip(), results=res),
              open(os.path.join(a.out, "final_test.json"), "w"), indent=1, default=float)
    print("\n".join(lines))


if __name__ == "__main__":
    main()
