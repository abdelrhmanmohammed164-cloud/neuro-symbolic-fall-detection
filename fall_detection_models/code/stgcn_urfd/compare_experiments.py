"""
compare_experiments.py
Compares model-selection runs (train.py --no_test) on the INNER VALIDATION set of split_scene_v2
(one unseen ADL scene + some falls per fold). It never opens a test result.

PRE-REGISTERED DECISION RULE (fixed before any real experiment was run; the only earlier run was a
10-epoch pipeline trial whose selected models predicted nothing, which is why selection and this
rule use MCC: it is 0 for any constant predictor, while F1 rewards an all-fall model):
  every run : --select val_mcc (checkpoint with the highest validation MCC, ties -> lower val loss)
  reference : baseline_v2 (original settings) with seeds 42 and 43
  noise     : per fold, |MCC(seed 42) - MCC(seed 43)| of the baseline
  an experiment (seed 42) is KEPT only if, in BOTH folds:
      val frame MCC >= baseline(seed 42) MCC + max(0.02, noise of that fold)
      val fall events detected   >= baseline
      val ADL videos with no false alarm >= baseline
  otherwise it is REJECTED. Kept experiments are combined only in a new run that must pass the
  same rule. Caveat written into every output: the validation sets are tiny (fold 0: 2 falls,
  6 ADL; fold 1: 4 falls, 5 ADL), so this rule can only detect large effects.

Usage:
  python compare_experiments.py --runs_root runs/exp_mcc --out reports/experiments_mcc
"""
import argparse
import json
import os

import numpy as np

from train import binary_metrics, mcc

FPS = 30.0


def fold_metrics(path):
    r = json.load(open(path))
    if r.get("test_evaluated", True) is not False or "frames" not in r:
        raise SystemExit(f"{path} is not a --no_test validation result; refusing to read it")
    fr = r["frames"]
    y = np.array([f["label"] for f in fr]); p = np.array([f["prob"] for f in fr])
    m = binary_metrics(y, p, 0.5)
    by = {}
    for f in fr:
        by.setdefault(f["sequence"], []).append(f)
    ev, n_falls, clean, n_adl, lat, fa_eps, adl_frames = 0, 0, 0, 0, [], 0, 0
    early, late = [], []
    for s, L in by.items():
        L.sort(key=lambda f: f["frame"])
        lab = np.array([f["label"] for f in L]); pr = np.array([f["prob"] for f in L]) >= 0.5
        if s.startswith("fall"):
            n_falls += 1; on = int(np.argmax(lab)); d = np.nonzero(pr[on:])[0]
            if len(d):
                ev += 1; lat.append(d[0] / FPS)
        else:
            n_adl += 1; clean += int(not pr.any()); adl_frames += len(L)
            fa_eps += int(pr[0]) + int(((pr[1:].astype(int) - pr[:-1].astype(int)) == 1).sum())
            fr_idx = np.array([f["frame"] for f in L])
            early += pr[fr_idx < 80].tolist(); late += pr[fr_idx >= 80].tolist()
    return dict(best_epoch=r["best_epoch"], mcc=mcc(m), f1=m["f1"], recall=m["recall_sensitivity"],
                specificity=m["specificity"],
                falls_detected=ev, falls=n_falls, adl_clean=clean, adl=n_adl,
                latency_s_median=float(np.median(lat)) if lat else None,
                fa_per_min=fa_eps / (adl_frames / FPS / 60) if adl_frames else None,
                adl_fp_rate_first80=float(np.mean(early)) if early else None,
                adl_fp_rate_after80=float(np.mean(late)) if late else None)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--runs_root", required=True); p.add_argument("--out", required=True)
    p.add_argument("--baseline", default="baseline_v2_s42"); p.add_argument("--baseline_seed2", default="baseline_v2_s43")
    a = p.parse_args()
    runs = {}
    for name in sorted(os.listdir(a.runs_root)):
        folds = {}
        for k in (0, 1):
            fp = os.path.join(a.runs_root, name, f"fold{k}", "val_result.json")
            if os.path.exists(fp):
                folds[k] = fold_metrics(fp)
        if len(folds) == 2:
            runs[name] = folds
        else:
            print(f"skip {name}: not finished ({len(folds)}/2 folds)")
    if a.baseline not in runs:
        raise SystemExit(f"baseline {a.baseline} not finished")
    base = runs[a.baseline]; b2 = runs.get(a.baseline_seed2)
    noise = {k: (abs(base[k]["mcc"] - b2[k]["mcc"]) if b2 else 0.0) for k in (0, 1)}
    decisions = {}
    for name, folds in runs.items():
        if name.startswith("baseline"):
            continue
        ok, why = True, []
        for k in (0, 1):
            need = base[k]["mcc"] + max(0.02, noise[k])
            if folds[k]["mcc"] < need:
                ok = False; why.append(f"fold {k}: MCC {folds[k]['mcc']:.3f} < required {need:.3f}")
            if folds[k]["falls_detected"] < base[k]["falls_detected"]:
                ok = False; why.append(f"fold {k}: fewer falls detected")
            if folds[k]["adl_clean"] < base[k]["adl_clean"]:
                ok = False; why.append(f"fold {k}: fewer ADL videos without false alarm")
        decisions[name] = dict(decision="KEEP" if ok else "REJECT", reasons=why)
    os.makedirs(a.out, exist_ok=True)
    json.dump(dict(rule=__doc__.split("PRE-REGISTERED DECISION RULE")[1].split("Usage:")[0].strip(),
                   seed_noise_f1=noise, runs=runs, decisions=decisions),
              open(os.path.join(a.out, "comparison.json"), "w"), indent=1)
    lines = ["# Inner-validation comparison (split_scene_v2, no test data)", "",
             "Validation sets are tiny (fold 0: 2 falls + 6 ADL from scene A7; fold 1: 4 falls + 5 ADL from scene A8).",
             f"Baseline seed noise in MCC: fold 0 {noise[0]:.3f}, fold 1 {noise[1]:.3f}.", "",
             "| run | fold | MCC | F1 | recall | spec | falls det. | ADL clean | FA/min | ADL FP first 80 / after | best ep | decision |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    fmt = lambda v: "-" if v is None else f"{v:.2f}"
    for name, folds in runs.items():
        for k in (0, 1):
            f = folds[k]
            dec = decisions.get(name, {}).get("decision", "reference") if k == 0 else ""
            lines.append(f"| {name if k == 0 else ''} | {k} | {f['mcc']:.3f} | {f['f1']:.3f} | {f['recall']:.3f} | {f['specificity']:.3f} | "
                         f"{f['falls_detected']}/{f['falls']} | {f['adl_clean']}/{f['adl']} | {fmt(f['fa_per_min'])} | "
                         f"{fmt(f['adl_fp_rate_first80'])} / {fmt(f['adl_fp_rate_after80'])} | {f['best_epoch']} | {dec} |")
    lines += ["", "## Decisions", ""]
    for name, d in decisions.items():
        lines.append(f"- **{name}: {d['decision']}**" + (f" ({'; '.join(d['reasons'])})" if d["reasons"] else ""))
    open(os.path.join(a.out, "comparison.md"), "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
