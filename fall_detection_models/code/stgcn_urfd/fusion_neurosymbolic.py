"""
fusion_neurosymbolic.py
Neuro-symbolic fusion of the neural ensemble and the improved V3 (reports/fusion_neurosymbolic/PREREGISTRATION.md).

  alarm(t) = [p(t) >= t_n AND h(t) < t_up] OR [s(t) = 1 AND p(t) >= t_low]

p = ensemble probability, h = metric neck height (missing -> no veto), s = improved V3 output. Causal per frame.
Thresholds are chosen per outer fold on SEL(k) = TRAIN(k) + VAL(k), using out-of-fold neural predictions (the other
fold's ensemble) and inner-CV symbolic predictions, then TEST(k) is scored once.

Run: python3 fusion_neurosymbolic.py    (PROJECT_ROOT overrides the project root)
Out: reports/fusion_neurosymbolic/{results.json, final_config.json, test_predictions.csv}
"""
import itertools
import json
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import evaluate_symbolic_v3 as E   # noqa: E402
import symbolic_v3_improved as V   # noqa: E402

ROOT = E.ROOT
OUT = os.path.join(ROOT, "reports", "fusion_neurosymbolic")
NEURAL_OOF = os.path.join(ROOT, "reports", "symbolic_v3_precision", "neural_test_oof.csv")
SYM_RESULTS = os.path.join(ROOT, "reports", "symbolic_v3_precision", "results.json")
T_N = (0.3, 0.4, 0.5, 0.6, 0.7)
T_UP = (0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2)
T_LOW = (0.0, 0.02, 0.05, 0.1, 0.2)
OLD_VETO_SYM = dict(t_n=0.5, t_up=0.8, t_low=0.0)


def fuse(p, s, h, t_n, t_up, t_low):
    up = np.nan_to_num(h, nan=0.0) >= t_up                     # missing height never vetoes
    return (((p >= t_n) & ~up) | ((s == 1) & (p >= t_low))).astype(int)


def neural_probs(vids):
    df = pd.read_csv(NEURAL_OOF); out = {}
    for v, g in df.groupby("sequence"):
        p = np.zeros(len(vids[v].y)); p[g.frame.values - 1] = g.prob.values; out[v] = p
    return out


def parse(name):
    return name[0], (float(name.split("=")[1][:-1]) if "=" in name else 0.0)


def symbolic_inner_cv(vids, sel, room, var, d):
    """Out-of-sample V3 predictions on SEL(k): refit on each inner train part, predict the held-out part."""
    out = {}
    for tr, te in E.inner_folds(sel, room):
        m, _ = V.fit([vids[v] for v in tr], var, d)
        for v in te:
            out[v] = m.predict(vids[v])
    return out


def score(rows):
    return E.metrics(rows)


def choose(cands):
    """cands: list of (params, metrics). Highest MCC; ties within 0.005 -> higher t_up, lower t_low, t_n nearest 0.5."""
    best = max(m["MCC"] for _, m in cands)
    near = [(c, m) for c, m in cands if m["MCC"] >= best - 0.005]
    near.sort(key=lambda x: (-x[0]["t_up"], x[0]["t_low"], abs(x[0]["t_n"] - 0.5), -x[1]["MCC"]))
    return near[0]


def paired_bootstrap(a_rows, b_rows, n=2000, seed=0):
    """MCC(a) - MCC(b), resampling videos."""
    rng = np.random.default_rng(seed); idx = np.arange(len(a_rows)); d = []
    def mcc(rows, ii):
        y = np.concatenate([rows[i][1] for i in ii]); p = np.concatenate([rows[i][2] for i in ii])
        tp = int(((p == 1) & (y == 1)).sum()); fp = int(((p == 1) & (y == 0)).sum()); fn = int(((p == 0) & (y == 1)).sum())
        return V.mcc_counts(tp, fp, fn, len(y) - tp - fp - fn)
    for _ in range(n):
        ii = rng.choice(idx, len(idx)); d.append(mcc(a_rows, ii) - mcc(b_rows, ii))
    d = np.array(d)
    return dict(mean=round(float(d.mean()), 4), ci95=[round(float(np.percentile(d, 2.5)), 4), round(float(np.percentile(d, 97.5)), 4)],
                p_better=round(float((d > 0).mean()), 3))


def main():
    split, vids, _ = E.load(); room = split["group_of"]
    sym_choice = json.load(open(SYM_RESULTS))["choice"]
    P = neural_probs(vids)
    res = dict(selection={}, configs={}, per_fold_test={}, test={})
    test = {k: [] for k in ("neural", "symbolic", "old veto+sym", "fusion")}
    for f in split["folds"]:
        k = f["fold"]; sel = f["train"] + f["val"]; var, d = parse(sym_choice[str(k)]["chosen"])
        # ---- selection on SEL(k): no TEST(k) label
        s_sel = symbolic_inner_cv(vids, sel, room, var, d)
        cands = []
        for t_n, t_up, t_low in itertools.product(T_N, T_UP, T_LOW):
            c = dict(t_n=t_n, t_up=t_up, t_low=t_low)
            cands.append((c, score([(v, vids[v].y, fuse(P[v], s_sel[v], vids[v].h, **c)) for v in sel])))
        cfg, m = choose(cands)
        base = score([(v, vids[v].y, (P[v] >= 0.5).astype(int)) for v in sel])
        adopt = m["MCC"] >= base["MCC"] + 0.02 and m["falls_detected"] >= base["falls_detected"]
        res["selection"][k] = dict(symbolic=f"{var} delta={d}", chosen=cfg, sel_metrics=E.rnd(m), neural_sel_metrics=E.rnd(base),
                                   adopted=bool(adopt), n_videos=len(sel),
                                   top5=[dict(**c, MCC=round(x["MCC"], 4)) for c, x in sorted(cands, key=lambda z: -z[1]["MCC"])[:5]])
        res["configs"][k] = dict(cfg, adopted=bool(adopt))
        print(f"fold {k}: SEL {len(sel)} videos, chosen {cfg} (SEL MCC {m['MCC']:.3f} vs neural {base['MCC']:.3f}) -> "
              f"{'adopt' if adopt else 'keep neural'}", flush=True)
        # ---- TEST(k), once
        m_fin, _ = V.fit([vids[v] for v in sel], var, d)
        for v in f["test"]:
            p, y, h = P[v], vids[v].y, vids[v].h; s = m_fin.predict(vids[v])
            row = dict(neural=(p >= 0.5).astype(int), symbolic=s, **{"old veto+sym": fuse(p, s, h, **OLD_VETO_SYM)},
                       fusion=fuse(p, s, h, **cfg) if adopt else (p >= 0.5).astype(int))
            for name, pred in row.items():
                test[name].append((v, y, pred))
        for name in test:
            res["per_fold_test"][f"{k}|{name}"] = E.rnd(score([r for r in test[name] if r[0] in f["test"]]))
    for name, rows in test.items():
        res["test"][name] = E.rnd(score(rows))
    res["bootstrap"] = {"fusion - neural": paired_bootstrap(test["fusion"], test["neural"]),
                        "fusion - symbolic": paired_bootstrap(test["fusion"], test["symbolic"])}
    os.makedirs(OUT, exist_ok=True)
    json.dump(res, open(os.path.join(OUT, "results.json"), "w"), indent=1, default=str)
    json.dump(res["configs"], open(os.path.join(OUT, "final_config.json"), "w"), indent=1)
    pd.DataFrame([dict(sequence=v, frame=t + 1, label=int(y[t]), neural=int(test["neural"][i][2][t]),
                       symbolic=int(test["symbolic"][i][2][t]), fusion=int(test["fusion"][i][2][t]))
                  for i, (v, y, _) in enumerate(test["fusion"]) for t in range(len(y))]
                 ).to_csv(os.path.join(OUT, "test_predictions.csv"), index=False)
    for name, m in res["test"].items():
        print(f"TEST {name:13s} P {m['Precision']:.3f} R {m['Recall']:.3f} F1 {m['F1']:.3f} Acc {m['Accuracy']:.3f} "
              f"BalAcc {m['BalancedAccuracy']:.3f} MCC {m['MCC']:.3f} falls {m['falls_detected']}/{m['falls']} clean {m['adl_clean']}/{m['adl']}")
    print("bootstrap", res["bootstrap"])


if __name__ == "__main__":
    main()
