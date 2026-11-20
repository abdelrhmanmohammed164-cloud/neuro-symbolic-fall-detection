"""
fusion_v3_improved.py
Fusion of the neural ensemble (3-seed neck-centred ST-GCN) with the improved Symbolic V3
(symbolic_v3_improved.py, variant chosen per fold in reports/symbolic_v3_precision).

Fusion rules (none has a fitted parameter, so nothing is tuned on the data):
  paper     (neural OR symbolic) AND symbolic positive at least once in the last 80 frames (paper Eq. 11-13)
  or        neural OR symbolic
  and       neural AND symbolic
  veto      neural, except where the neck is at 0.8 m or higher
  veto+sym  veto OR symbolic

Choice per fold on that fold's VAL videos (neural: that fold's models, never trained on them; symbolic: refitted
on TRAIN only for this score). A rule is adopted for fold k only if its VAL MCC beats the neural ensemble by at
least 0.02 without fewer falls detected or fewer clean ADL videos; otherwise fold k keeps the neural ensemble.
TEST: symbolic refitted on TRAIN + VAL (the final improved V3), every rule scored once. Test numbers are post-hoc.

Run: python3 fusion_v3_improved.py   ->  reports/fusion_v3_improved/{results.json, test_predictions.npz}
"""
import json
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import evaluate_symbolic_v3 as E  # noqa: E402
import symbolic_v3_improved as V  # noqa: E402

ROOT = E.ROOT
OUT = os.path.join(ROOT, "reports", "fusion_v3_improved")
NEURAL_VAL = os.path.join(OUT, "neural_val.csv")
NEURAL_TEST = os.path.join(ROOT, "reports", "symbolic_v3_precision", "neural_test_oof.csv")
RULES = ("paper", "or", "and", "veto", "veto+sym")


def probs(csv, vids):
    df = pd.read_csv(csv); out = {}
    for v, g in df.groupby("sequence"):
        p = np.zeros(len(vids[v].y)); p[g.frame.values - 1] = g.prob.values; out[v] = p
    return out


def support(x, win=80):
    c = np.concatenate([[0], np.cumsum(x)]); t = np.arange(len(x))
    return ((c[t + 1] - c[np.maximum(0, t - win + 1)]) > 0).astype(int)


def fuse(rule, pn, s, h):
    n = (pn >= 0.5).astype(int); up = (np.nan_to_num(h, nan=0.0) >= 0.8).astype(int)
    return {"neural": n, "symbolic": s,
            "paper": (n | s) & support(s), "or": n | s, "and": n & s,
            "veto": n & (1 - up), "veto+sym": (n & (1 - up)) | s}[rule].astype(int)


def main():
    split, vids, _ = E.load()
    choice_sym = json.load(open(os.path.join(ROOT, "reports", "symbolic_v3_precision", "results.json")))["choice"]
    NV, NT = probs(NEURAL_VAL, vids), probs(NEURAL_TEST, vids)
    res = dict(choice={}, val={}, test={}, per_fold_test={})
    test_rows = {r: [] for r in ("neural", "symbolic") + RULES}
    for f in split["folds"]:
        k = f["fold"]; name = choice_sym[str(k)]["chosen"]
        var, d = name[0], (float(name.split("=")[1][:-1]) if "=" in name else 0.0)
        m_val, _ = V.fit([vids[v] for v in f["train"]], var, d)                    # for the VAL score
        m_fin, _ = V.fit([vids[v] for v in f["train"] + f["val"]], var, d)         # final improved V3
        val = {r: E.metrics([(v, vids[v].y, fuse(r, NV[v], m_val.predict(vids[v]), vids[v].h)) for v in f["val"]])
               for r in ("neural", "symbolic") + RULES}
        base = val["neural"]
        best = max(RULES, key=lambda r: val[r]["MCC"]); b = val[best]
        ok = b["MCC"] >= base["MCC"] + 0.02 and b["falls_detected"] >= base["falls_detected"] and b["adl_clean"] >= base["adl_clean"]
        res["choice"][k] = dict(symbolic=name, best_on_val=best, val_mcc=round(b["MCC"], 4), neural_val_mcc=round(base["MCC"], 4),
                                adopted=ok, final=best if ok else "neural")
        res["val"][k] = {r: E.rnd(m) for r, m in val.items()}
        for r in test_rows:
            rows = [(v, vids[v].y, fuse(r, NT[v], m_fin.predict(vids[v]), vids[v].h)) for v in f["test"]]
            test_rows[r] += rows; res["per_fold_test"][f"{k}|{r}"] = E.rnd(E.metrics(rows))
        print(f"fold {k}: symbolic {name}; best rule on VAL {best} (MCC {b['MCC']:.3f} vs neural {base['MCC']:.3f}) -> "
              f"{'adopt' if ok else 'keep neural'}", flush=True)
    for r, rows in test_rows.items():
        res["test"][r] = E.rnd(E.metrics(rows))
    sel = []
    for f in split["folds"]:
        fin = res["choice"][f["fold"]]["final"]
        sel += [x for x in test_rows[fin] if x[0] in f["test"]]
    res["test"]["selected (per fold)"] = E.rnd(E.metrics(sel))
    # paired bootstrap over test videos: MCC difference vs neural
    rng = np.random.default_rng(0); idx_all = np.arange(len(test_rows["neural"])); boot = {}
    def mcc_of(rows, idx):
        y = np.concatenate([rows[i][1] for i in idx]); p = np.concatenate([rows[i][2] for i in idx])
        tp = int(((p == 1) & (y == 1)).sum()); fp = int(((p == 1) & (y == 0)).sum()); fn = int(((p == 0) & (y == 1)).sum()); tn = len(y) - tp - fp - fn
        return V.mcc_counts(tp, fp, fn, tn)
    for r in RULES:
        d = []
        for _ in range(2000):
            idx = rng.choice(idx_all, len(idx_all)); d.append(mcc_of(test_rows[r], idx) - mcc_of(test_rows["neural"], idx))
        boot[r] = dict(mean=round(float(np.mean(d)), 4), ci95=[round(float(np.percentile(d, 2.5)), 4), round(float(np.percentile(d, 97.5)), 4)],
                       p_better=round(float(np.mean(np.array(d) > 0)), 3))
    res["bootstrap_vs_neural"] = boot
    os.makedirs(OUT, exist_ok=True)
    json.dump(res, open(os.path.join(OUT, "results.json"), "w"), indent=1, default=str)
    np.savez(os.path.join(OUT, "test_predictions.npz"), **{f"{v}__{r}": p for r, rows in test_rows.items() for v, _, p in rows})
    print("wrote", OUT)


if __name__ == "__main__":
    main()
