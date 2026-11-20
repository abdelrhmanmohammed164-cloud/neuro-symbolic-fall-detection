"""
fusion_veto_exemption.py
F0 = current fusion (veto+sym, fusion_v3_improved.py, unchanged).
F1 = the same, but the veto does not apply while the neck is descending fast
     (downward speed > the fold's V3 tau_v at least once in the last 1.0 s).
Protocol: reports/fusion_veto_exemption/PREREGISTRATION.md. Run: python3 fusion_veto_exemption.py
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import evaluate_early_fall as EV  # noqa: E402  (onset_stats, parse)
import evaluate_symbolic_v3 as E  # noqa: E402
import fusion_v3_improved as FU  # noqa: E402
import symbolic_v3_improved as V  # noqa: E402

ROOT = E.ROOT
OUT = os.path.join(ROOT, "reports", "fusion_veto_exemption")


def fuse_f1(pn, s, video, tau_v):
    n = (pn >= 0.5).astype(int)
    up = np.nan_to_num(video.h, nan=0.0) >= 0.8
    descending = np.nan_to_num(video.vmax, nan=0.0) > tau_v
    veto = up & ~descending
    return ((n & ~veto) | s).astype(int)


def evaluate(rows):
    m = E.metrics(rows); o = EV.onset_stats(rows)
    return dict(**E.rnd(m), onset=o)


def main():
    split, vids, _ = E.load()
    chosen = json.load(open(os.path.join(ROOT, "reports", "symbolic_v3_precision", "results.json")))["choice"]
    NV = FU.probs(os.path.join(ROOT, "reports", "fusion_v3_improved", "neural_val.csv"), vids)
    NT = FU.probs(FU.NEURAL_TEST, vids)
    res = dict(val={}, decision={}, test={}, per_fold_test={})
    test = {"neural": [], "F0": [], "F1": []}
    for f in split["folds"]:
        k = f["fold"]; var, d = EV.parse(chosen[str(k)]["chosen"])
        m_tr, _ = V.fit([vids[v] for v in f["train"]], var, d)
        m_fin, _ = V.fit([vids[v] for v in f["train"] + f["val"]], var, d)
        rows = {"neural": [], "F0": [], "F1": []}
        for v in f["val"]:
            s = m_tr.predict(vids[v]); y = vids[v].y
            rows["neural"].append((v, y, (NV[v] >= 0.5).astype(int)))
            rows["F0"].append((v, y, FU.fuse("veto+sym", NV[v], s, vids[v].h)))
            rows["F1"].append((v, y, fuse_f1(NV[v], s, vids[v], m_tr.tau_v)))
        val = {n: evaluate(r) for n, r in rows.items()}
        a, b = val["F0"], val["F1"]
        ok = (b["MCC"] >= a["MCC"] and b["onset"]["first_second_recall"] > a["onset"]["first_second_recall"]
              and b["falls_detected"] >= a["falls_detected"] and b["adl_clean"] >= a["adl_clean"])
        res["val"][k] = val; res["decision"][k] = dict(tau_v_train=round(m_tr.tau_v, 4), adopt_F1=ok, final="F1" if ok else "F0")
        print(f"fold {k} VAL: F0 MCC {a['MCC']:.3f} first-s recall {a['onset']['first_second_recall']:.3f} | "
              f"F1 MCC {b['MCC']:.3f} first-s recall {b['onset']['first_second_recall']:.3f} -> {'F1' if ok else 'F0'}", flush=True)
        for v in f["test"]:
            s = m_fin.predict(vids[v]); y = vids[v].y
            test["neural"].append((v, y, (NT[v] >= 0.5).astype(int)))
            test["F0"].append((v, y, FU.fuse("veto+sym", NT[v], s, vids[v].h)))
            test["F1"].append((v, y, fuse_f1(NT[v], s, vids[v], m_fin.tau_v)))
        for n in test:
            res["per_fold_test"][f"{k}|{n}"] = evaluate([r for r in test[n] if r[0] in f["test"]])
    for n, r in test.items():
        res["test"][n] = evaluate(r)
    sel = []
    for f in split["folds"]:
        sel += [r for r in test[res["decision"][f["fold"]]["final"]] if r[0] in f["test"]]
    res["test"]["selected per fold"] = evaluate(sel)
    os.makedirs(OUT, exist_ok=True)
    json.dump(res, open(os.path.join(OUT, "results.json"), "w"), indent=1, default=str)
    print("wrote", OUT)


if __name__ == "__main__":
    main()
