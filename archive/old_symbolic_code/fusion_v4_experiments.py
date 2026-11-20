"""
fusion_v4_experiments.py
Symbolic candidates A to E, the neural ensemble (F) and the fusions F1 to F6, evaluated under
reports/symbolic_v4/PREREGISTRATION.md: every choice for fold k is made on fold k TRAIN and VAL only,
then the TEST folds are scored once.

Inputs (paths relative to the project root, override with PROJECT_ROOT):
  processed_data/cfg_d31bfed741ab/urfd_npy   frame labels, RTMO keypoints, Depth-Anything neck depth
  processed_data/kinect_height               metric neck height (kinect_neck_height.py)
  URFD/features/urfall-cam0-*.csv            URFD 3-class labels (falling / lying)
  splits/split_scene_v2.json
  reports/symbolic_v4/neural_val.csv         ensemble on each fold's VAL videos (make_neural_val.py)
  reports/symbolic_v4/neural_test_oof.csv    out-of-fold ensemble on TEST (neural_ensemble.py --mode oof)
  notebooks/symbolic_fall_detector_urfd.ipynb  the original notebook (candidate A)

Outputs in reports/symbolic_v4/: results.json, per_frame_errors.csv, tables.md, configs.json, figures.
"""
import json
import os
import sys
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=RuntimeWarning)
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import symbolic_v3 as S3  # noqa: E402
import symbolic_v4 as S4  # noqa: E402

ROOT = os.environ.get("PROJECT_ROOT", os.path.abspath(os.path.join(HERE, "..", "..", "..")))
NPY = os.path.join(ROOT, "processed_data", "cfg_d31bfed741ab", "urfd_npy")
KIN = os.path.join(ROOT, "processed_data", "kinect_height")
OUT = os.path.join(ROOT, "reports", "symbolic_v4")
FPS = 30.0


# ---------------------------------------------------------------- data
def load_all():
    split = json.load(open(os.path.join(ROOT, "splits", "split_scene_v2.json")))
    cols = ["v", "f", "lab", "hw", "mm", "occ", "std", "hh", "H", "Dmm", "p40"]
    U = pd.concat([pd.read_csv(os.path.join(ROOT, "URFD", "features", f"urfall-cam0-{k}.csv"), header=None, names=cols)
                   for k in ("falls", "adls")])
    data = {}
    for v in sorted(split["group_of"]):
        y = np.load(os.path.join(NPY, f"{v}_frame_labels.npy")).astype(int)
        l3 = np.full(len(y), -1)
        if v.startswith("fall"):            # fall phases only; the ADL file's 0/1 codes mark postures, not falls
            d = U[U.v == v]; i = d.f.values - 1; ok = i < len(y); l3[i[ok]] = d.lab.values[ok]
        data[v] = dict(y=y, l3=l3, h=S3.load_height(KIN, v), room=split["group_of"][v])
    return split, data


def neural_probs(csv, data):
    df = pd.read_csv(csv); out = {}
    for v, g in df.groupby("sequence"):
        p = np.zeros(len(data[v]["y"])); p[g.frame.values - 1] = g.prob.values; out[v] = p
    return out


# ---------------------------------------------------------------- metrics
def confusion(y, p):
    return dict(tp=int(((y == 1) & (p == 1)).sum()), fp=int(((y == 0) & (p == 1)).sum()),
                fn=int(((y == 1) & (p == 0)).sum()), tn=int(((y == 0) & (p == 0)).sum()))


def auc(y, s):
    order = np.argsort(s); r = np.empty(len(s)); r[order] = np.arange(1, len(s) + 1)
    # average ranks for ties
    _, inv, cnt = np.unique(s, return_inverse=True, return_counts=True)
    sums = np.bincount(inv, r); r = (sums / cnt)[inv]
    n1, n0 = (y == 1).sum(), (y == 0).sum()
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)) if n1 and n0 else float("nan")


def pr_auc(y, s):
    o = np.argsort(-s); y = y[o]; tp = np.cumsum(y); prec = tp / np.arange(1, len(y) + 1)
    return float((prec * y).sum() / max(y.sum(), 1))


def metrics(rows, prob_rows=None):
    """rows: list of (video, y, hard prediction)."""
    y = np.concatenate([r[1] for r in rows]); p = np.concatenate([r[2] for r in rows]).astype(int)
    c = confusion(y, p); tp, fp, fn, tn = c["tp"], c["fp"], c["fn"], c["tn"]
    pr, rc = tp / max(tp + fp, 1), tp / max(tp + fn, 1)
    f1 = 2 * pr * rc / max(pr + rc, 1e-9)
    npv, spec = tn / max(tn + fn, 1), tn / max(tn + fp, 1)
    f1_0 = 2 * npv * spec / max(npv + spec, 1e-9)
    n1, n0 = tp + fn, tn + fp
    det = falls = clean = adl = eps = fr = 0
    for v, yy, pp in rows:                                    # ST-GCN event definitions
        pp = np.asarray(pp).astype(int)
        if v.startswith("fall"):
            falls += 1; det += int(pp[int(np.argmax(yy)):].any())
        else:
            adl += 1; clean += int(not pp.any()); fr += len(pp); eps += int(pp[0]) + int((np.diff(pp) == 1).sum())
    m = dict(MCC=S3.mcc(y, p), F1=f1, Precision=pr, Recall=rc, Specificity=spec, Accuracy=(tp + tn) / len(y),
             MacroF1=(f1 + f1_0) / 2, WeightedF1=(f1 * n1 + f1_0 * n0) / (n1 + n0),
             falls_detected=det, falls=falls, adl_clean=clean, adl=adl,
             fa_per_min=eps / max(fr / FPS / 60, 1e-9), **c)
    if prob_rows is not None:
        s = np.clip(np.concatenate(prob_rows), 1e-6, 1 - 1e-6)
        m.update(ROC_AUC=auc(y, s), PR_AUC=pr_auc(y, s), LogLoss=float(-np.mean(y * np.log(s) + (1 - y) * np.log(1 - s))),
                 Brier=float(np.mean((s - y) ** 2)), ECE=ece(y, s))
    return m


def ece(y, s, bins=10):
    e = 0.0
    for lo in np.linspace(0, 1, bins + 1)[:-1]:
        m = (s >= lo) & (s < lo + 1 / bins) if lo < 0.9 else (s >= lo)
        if m.any():
            e += m.mean() * abs(s[m].mean() - y[m].mean())
    return float(e)


def boot_mcc(rows, n=1000, seed=0):
    rng = np.random.default_rng(seed); vals = []
    for _ in range(n):
        pick = [rows[i] for i in rng.integers(0, len(rows), len(rows))]
        y = np.concatenate([r[1] for r in pick]); p = np.concatenate([r[2] for r in pick])
        vals.append(S3.mcc(y, p))
    return [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))]


def complementarity(y, a, b):
    """a = neural hard, b = symbolic hard (same frames)."""
    ca, cb = a == y, b == y
    n11, n00 = int((ca & cb).sum()), int((~ca & ~cb).sum())
    n10, n01 = int((ca & ~cb).sum()), int((~ca & cb).sum())
    q = (n11 * n00 - n10 * n01) / max(n11 * n00 + n10 * n01, 1)
    ea, eb = (~ca).astype(float), (~cb).astype(float)
    corr = float(np.corrcoef(ea, eb)[0, 1]) if ea.std() > 0 and eb.std() > 0 else float("nan")
    return dict(both_correct=n11, neural_only_correct=n10, symbolic_only_correct=n01, both_wrong=n00,
                double_fault_rate=n00 / len(y), disagreement=(n10 + n01) / len(y), Q_statistic=q, error_correlation=corr,
                fall_frames_neural_missed_symbolic_caught=int(((y == 1) & ~ca & cb).sum()),
                nonfall_frames_neural_alarm_symbolic_quiet=int(((y == 0) & ~ca & cb).sum()))


# ---------------------------------------------------------------- symbolic candidates
def candidate_A(split, data):
    """The original notebook, unchanged, fitted on TRAIN (its default mode and grids)."""
    nbp = os.path.join(ROOT, "notebooks", "symbolic_fall_detector_urfd.ipynb")
    nb = json.load(open(nbp)); ns = {}
    code = [''.join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code"]
    for c in code[:6]:                      # config, input, extractor, height, detector, metrics (not the adapter)
        exec(c, ns)
    out = {}
    for f in split["folds"]:
        def inp(v):
            k = np.load(os.path.join(NPY, f"{v}_raw_coco17.npy")); nd = np.load(os.path.join(NPY, f"{v}_neck_depth.npy"))
            return ns["SymbolicVideoInput"](v, data[v]["room"], k, joint_depths=np.repeat(nd[:, None], 17, 1),
                                            image_size=(480, 640), labels=data[v]["y"])
        m = ns["SymbolicFallDetector"](cfg=ns["SymbolicConfig"](mode="urfd_state_aware"))
        m.fit_thresholds([inp(v) for v in f["train"]])
        for part in ("val", "test"):
            for v in f[part]:
                ysym = m.predict_video(inp(v))["y_sym"]
                out[(f["fold"], part, v)] = dict(hard=ysym, support80=S3.support_window(ysym), p=None, upright=None)
    return out


def candidate_B(split, data):
    out = {}
    for f in split["folds"]:
        lying = np.concatenate([np.nan_to_num(data[v]["h"], nan=2.0)[data[v]["l3"] == 1] for v in f["train"] if v.startswith("fall")])
        m, _ = S3.fit([(data[v]["h"], data[v]["y"]) for v in f["train"]], lying)
        for part in ("val", "test"):
            for v in f[part]:
                o = m.predict(data[v]["h"])
                out[(f["fold"], part, v)] = dict(hard=o["trigger"], support80=o["support80"], p=None, upright=None)
    return out


def candidate_v4(split, data, **kw):
    out, cfg = {}, {}
    for f in split["folds"]:
        m = S4.fit([(data[v]["h"], data[v]["y"], data[v]["l3"]) for v in f["train"]], **kw)
        cfg[f["fold"]] = m.config()
        for part in ("val", "test"):
            for v in f[part]:
                o = m.predict(data[v]["h"])
                out[(f["fold"], part, v)] = dict(hard=o["hard"], support80=o["support80"], p=o["p"], upright=o["upright"],
                                                 state=o["state"])
    return out, cfg


# ---------------------------------------------------------------- fusion
def logit(p):
    p = np.clip(p, 1e-4, 1 - 1e-4); return np.log(p / (1 - p))


def fit_lr(X, y, l2=1.0, iters=500):
    """Plain L2 logistic regression by Newton steps (deterministic, no extra dependency)."""
    X = np.hstack([X, np.ones((len(X), 1))]); w = np.zeros(X.shape[1])
    for _ in range(iters):
        z = X @ w; p = 1 / (1 + np.exp(-z)); g = X.T @ (p - y) + l2 * np.r_[w[:-1], 0]
        H = (X * (p * (1 - p))[:, None]).T @ X + l2 * np.diag(np.r_[np.ones(len(w) - 1), 0]) + 1e-9 * np.eye(len(w))
        step = np.linalg.solve(H, g); w -= step
        if np.abs(step).max() < 1e-8:
            break
    return w


def lr_predict(w, X):
    return 1 / (1 + np.exp(-(np.hstack([X, np.ones((len(X), 1))]) @ w)))


def fuse(method, pn, s, w=None):
    n = (pn >= 0.5).astype(int)
    if method == "F1":
        return ((n | s["hard"]) & s["support80"]).astype(int)
    if method == "F2":
        return (((pn + s["p"]) / 2) >= 0.5).astype(int)
    if method == "F3":
        return ((w * pn + (1 - w) * s["p"]) >= 0.5).astype(int)
    if method == "F4":
        return (n & (1 - s["upright"])).astype(int)
    if method == "F5":
        return ((n & (1 - s["upright"])) | (s["state"] == S4.S["FALL"])).astype(int)
    raise ValueError(method)


def main():
    split, data = load_all()
    NV = neural_probs(os.path.join(OUT, "neural_val.csv"), data)
    NT = neural_probs(os.path.join(OUT, "neural_test_oof.csv"), data)
    neural = lambda part, v: NV[v] if part == "val" else NT[v]

    print("candidate A (original notebook) ...", flush=True)
    SYM = {"A": candidate_A(split, data), "B": candidate_B(split, data)}
    SYM["C"], cfgC = candidate_v4(split, data)
    SYM["D1"], cfgD1 = candidate_v4(split, data, use_descending=False)
    SYM["D2"], cfgD2 = candidate_v4(split, data, unknown_hold=0)
    SYM["E"] = {k: dict(v, hard=(v["p"] >= 0.5).astype(int)) for k, v in SYM["C"].items()}
    configs = {"C": cfgC, "D1": cfgD1, "D2": cfgD2}

    options = [("F", None, None)]                                  # (fusion, symbolic, weight)
    options += [("F1", k, None) for k in ("A", "B", "C", "D1", "D2", "E")]
    options += [("F2", "E", None)] + [("F3", "E", w) for w in (0.6, 0.7, 0.8, 0.9)]
    options += [("F4", k, None) for k in ("C", "D1", "D2")] + [("F5", k, None) for k in ("C", "D1", "D2")]
    options += [("F6", "E", None)]
    nparams = {"F": 0, "F1": 0, "F4": 0, "F5": 0, "F2": 0, "F3": 1, "F6": 3}

    def name(o):
        return "F neural alone" if o[0] == "F" else f"{o[0]}+{o[1]}" + (f"(w={o[2]})" if o[2] else "")

    results, choice, lr_w = {}, {}, {}
    for f in split["folds"]:
        k = f["fold"]
        # stacking: fitted on this fold's VAL; its VAL score is leave-one-video-out
        Xv = {v: np.c_[logit(neural("val", v)), SYM["E"][(k, "val", v)]["p"]] for v in f["val"]}
        lr_w[k] = fit_lr(np.vstack([Xv[v] for v in f["val"]]), np.concatenate([data[v]["y"] for v in f["val"]]))
        loo = {}
        for v in f["val"]:
            others = [u for u in f["val"] if u != v]
            w = fit_lr(np.vstack([Xv[u] for u in others]), np.concatenate([data[u]["y"] for u in others]))
            loo[v] = (lr_predict(w, Xv[v]) >= 0.5).astype(int)
        for part in ("val", "test"):
            for o in options:
                rows, probs = [], []
                for v in f[part]:
                    pn = neural(part, v); y = data[v]["y"]
                    if o[0] == "F":
                        pred = (pn >= 0.5).astype(int); probs.append(pn)
                    elif o[0] == "F6":
                        X = np.c_[logit(pn), SYM["E"][(k, part, v)]["p"]]
                        pred = loo[v] if part == "val" else (lr_predict(lr_w[k], X) >= 0.5).astype(int)
                    else:
                        pred = fuse(o[0], pn, SYM[o[1]][(k, part, v)], o[2])
                    rows.append((v, y, pred))
                results[(k, part, name(o))] = rows
        # selection on VAL of this fold only
        base = metrics(results[(k, "val", "F neural alone")])
        cands = sorted(((metrics(results[(k, "val", name(o))])["MCC"], -nparams[o[0]], name(o)) for o in options[1:]), reverse=True)
        best_mcc, _, best = cands[0]; bm = metrics(results[(k, "val", best)])
        ok = (best_mcc >= base["MCC"] + 0.02 and bm["falls_detected"] >= base["falls_detected"] and bm["adl_clean"] >= base["adl_clean"])
        choice[k] = dict(best_on_val=best, val_mcc=best_mcc, neural_val_mcc=base["MCC"], adopted=ok,
                         final=best if ok else "F neural alone")
        print(f"fold {k}: best on VAL {best} MCC {best_mcc:.3f} vs neural {base['MCC']:.3f} -> {'ADOPT' if ok else 'keep neural'}")

    # pooled tables
    names = [name(o) for o in options]
    table = {}
    for part in ("val", "test"):
        for n in names:
            rows = results[(0, part, n)] + results[(1, part, n)]
            table[(part, n)] = metrics(rows)
    sym_alone = {}
    for kname in ("A", "B", "C", "D1", "D2", "E"):
        for part in ("val", "test"):
            rows, pr = [], []
            for f in split["folds"]:
                for v in f[part]:
                    s = SYM[kname][(f["fold"], part, v)]; rows.append((v, data[v]["y"], s["hard"]))
                    if s["p"] is not None:
                        pr.append(s["p"])
            sym_alone[(part, kname)] = metrics(rows, pr if pr else None)
    # neural alone with probabilistic metrics
    for part in ("val", "test"):
        rows, pr = [], []
        for f in split["folds"]:
            for v in f[part]:
                pn = neural(part, v); rows.append((v, data[v]["y"], (pn >= 0.5).astype(int))); pr.append(pn)
        sym_alone[(part, "neural")] = metrics(rows, pr)
    # final selected system on TEST
    final_rows = results[(0, "test", choice[0]["final"])] + results[(1, "test", choice[1]["final"])]
    final = metrics(final_rows); final["MCC_CI95_videos"] = boot_mcc(final_rows)
    neural_test_rows = results[(0, "test", "F neural alone")] + results[(1, "test", "F neural alone")]
    nt = metrics(neural_test_rows); nt["MCC_CI95_videos"] = boot_mcc(neural_test_rows)

    # complementarity on TEST (neural vs each symbolic hard output)
    comp = {}
    for kname in ("A", "B", "C", "E"):
        y, a, b = [], [], []
        for f in split["folds"]:
            for v in f["test"]:
                y.append(data[v]["y"]); a.append((NT[v] >= 0.5).astype(int)); b.append(SYM[kname][(f["fold"], "test", v)]["hard"])
        comp[kname] = complementarity(np.concatenate(y), np.concatenate(a), np.concatenate(b))

    # per-frame error analysis dataset (TEST, out-of-fold everywhere)
    recs = []
    for f in split["folds"]:
        for v in f["test"]:
            s = SYM["E"][(f["fold"], "test", v)]; pn = NT[v]; y = data[v]["y"]
            for t in range(len(y)):
                nh, sh = int(pn[t] >= 0.5), int(s["hard"][t])
                recs.append(dict(video=v, frame=t + 1, fold=f["fold"], room=data[v]["room"], label=int(y[t]),
                                 urfd_label=int(data[v]["l3"][t]), neck_height_m=data[v]["h"][t],
                                 neural_prob=round(float(pn[t]), 4), neural_pred=nh,
                                 symbolic_state=S4.STATES[int(s["state"][t])], symbolic_conf=round(float(s["p"][t]), 4),
                                 symbolic_pred=sh, neural_correct=int(nh == y[t]), symbolic_correct=int(sh == y[t]),
                                 both_wrong=int(nh != y[t] and sh != y[t])))
    os.makedirs(OUT, exist_ok=True)
    pd.DataFrame(recs).to_csv(os.path.join(OUT, "per_frame_errors.csv"), index=False)

    def clean(d):
        return {k: (round(v, 4) if isinstance(v, float) else v) for k, v in d.items()}
    res = dict(choice=choice, final_test=clean(final), neural_test=clean(nt),
               fused={f"{p}|{n}": clean(m) for (p, n), m in table.items()},
               standalone={f"{p}|{n}": clean(m) for (p, n), m in sym_alone.items()},
               complementarity_test={k: clean(v) for k, v in comp.items()},
               per_fold_test={f"{k}|{n}": clean(metrics(results[(k, "test", n)])) for k in (0, 1) for n in names},
               per_fold_val={f"{k}|{n}": clean(metrics(results[(k, "val", n)])) for k in (0, 1) for n in names},
               stacking_weights={k: [round(float(x), 4) for x in w] for k, w in lr_w.items()})
    json.dump(res, open(os.path.join(OUT, "results.json"), "w"), indent=1, default=float)
    json.dump(configs, open(os.path.join(OUT, "configs.json"), "w"), indent=1)
    print("wrote", OUT)


if __name__ == "__main__":
    main()
