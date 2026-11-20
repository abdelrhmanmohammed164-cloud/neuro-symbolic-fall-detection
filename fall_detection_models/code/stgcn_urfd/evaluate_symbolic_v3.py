"""
evaluate_symbolic_v3.py
The V3 precision study (reports/symbolic_v3_precision/PREREGISTRATION.md).

  1. For each outer fold k: inner group cross-validation on SELECTION(k) = TRAIN(k) + VAL(k) for every
     candidate (A, P, B, C with delta 0.30/0.40/0.50, D with the same deltas).
  2. Selection rule on the pooled inner-CV predictions (recall floor = 90% of A's recall, then best F1).
  3. The chosen candidate (and, for the comparison tables, every candidate) is refitted on SELECTION(k) and
     scored once on TEST(k).
  4. Diagnostics: precision-recall trade-off over tau_h, per-rule false positives, stress tests, and on TEST the
     error overlap with the out-of-fold neural ensemble.

Run from anywhere: python3 evaluate_symbolic_v3.py   (PROJECT_ROOT overrides the project root)
Outputs: reports/symbolic_v3_precision/{results.json, tables.md, figures/*.png, final_config.json}
"""
import json
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import symbolic_v3_improved as V  # noqa: E402

ROOT = os.environ.get("PROJECT_ROOT", os.path.abspath(os.path.join(HERE, "..", "..", "..")))
NPY = os.path.join(ROOT, "processed_data", "cfg_d31bfed741ab", "urfd_npy")
KIN = os.path.join(ROOT, "processed_data", "kinect_height")
OUT = os.path.join(ROOT, "reports", "symbolic_v3_precision")
NEURAL_OOF = os.path.join(OUT, "neural_test_oof.csv")
CANDIDATES = [("A", 0.0), ("P", 0.0), ("B", 0.0)] + [(v, d) for v in ("C", "D") for d in (0.30, 0.40, 0.50)]
N_CONDITIONS = {"A": 2, "P": 2, "B": 2, "C": 3, "D": 4}


def cname(v, d):
    return v if v in ("A", "P", "B") else f"{v}(delta={d:.2f})"


# ------------------------------------------------------------------ data
def load():
    split = json.load(open(os.path.join(ROOT, "splits", "split_scene_v2.json")))
    cols = ["v", "f", "lab", "hw", "mm", "occ", "std", "hh", "H", "Dmm", "p40"]
    U = pd.read_csv(os.path.join(ROOT, "URFD", "features", "urfall-cam0-falls.csv"), header=None, names=cols)
    UA = pd.read_csv(os.path.join(ROOT, "URFD", "features", "urfall-cam0-adls.csv"), header=None, names=cols)
    vids, posture = {}, {}
    for v in sorted(split["group_of"]):
        y = np.load(os.path.join(NPY, f"{v}_frame_labels.npy")).astype(int)
        h = np.load(os.path.join(KIN, f"{v}.npz"))["height_m"]
        lab = np.full(len(y), -1)
        d = (U if v.startswith("fall") else UA)[lambda df: df.v == v]
        i = d.f.values - 1; ok = i < len(y); lab[i[ok]] = d.lab.values[ok]
        lying = (lab == 1) if v.startswith("fall") else None
        vids[v] = V.Video(h, y, lying)
        posture[v] = lab                 # fall videos: -1/0/1 phases; ADL videos: URFD posture codes (analysis only)
    return split, vids, posture


def inner_folds(selection, room):
    adl_rooms = sorted({room[v] for v in selection if v.startswith("adl")})
    falls = sorted(v for v in selection if v.startswith("fall"))
    n = len(adl_rooms)
    return [([v for v in selection if not ((v.startswith("adl") and room[v] == r) or (v in falls[i::n]))],
             [v for v in selection if (v.startswith("adl") and room[v] == r) or (v in falls[i::n])]) for i, r in enumerate(adl_rooms)]


# ------------------------------------------------------------------ metrics
def metrics(rows):
    """rows: list of (video id, labels, predictions)."""
    y = np.concatenate([r[1] for r in rows]); p = np.concatenate([r[2] for r in rows]).astype(int)
    tp = int(((p == 1) & (y == 1)).sum()); fp = int(((p == 1) & (y == 0)).sum())
    fn = int(((p == 0) & (y == 1)).sum()); tn = int(((p == 0) & (y == 0)).sum())
    pr, rc, sp = tp / max(tp + fp, 1), tp / max(tp + fn, 1), tn / max(tn + fp, 1)
    det = falls = clean = adl = eps = fr = 0
    for v, yy, pp in rows:
        pp = np.asarray(pp).astype(int)
        if v.startswith("fall"):
            falls += 1; det += int(pp[int(np.argmax(yy)):].any())
        else:
            adl += 1; clean += int(not pp.any()); fr += len(pp); eps += int(pp[0]) + int((np.diff(pp) == 1).sum())
    return dict(Precision=pr, Recall=rc, F1=2 * pr * rc / max(pr + rc, 1e-9), Accuracy=(tp + tn) / len(y),
                BalancedAccuracy=(rc + sp) / 2, Specificity=sp, MCC=V.mcc_counts(tp, fp, fn, tn), TP=tp, FP=fp, TN=tn, FN=fn,
                PositiveRate=(tp + fp) / len(y), falls_detected=det, falls=falls, adl_clean=clean, adl=adl,
                alarms_per_min=eps / max(fr / V.FPS / 60, 1e-9))


def coverage(model, videos):
    """Fraction of frames where each condition holds (and on how many of them the label is 'fall')."""
    tot = {}
    for vd in videos:
        c = model.conditions(vd)
        for k, m in c.items():
            a = tot.setdefault(k, [0, 0, 0]); a[0] += int(m.sum()); a[1] += int((m & (vd.y == 1)).sum()); a[2] += len(m)
    return {k: dict(frames=a[0], fraction=a[0] / a[2], fall_rate_when_true=a[1] / max(a[0], 1)) for k, a in tot.items()}


def rnd(d):
    return {k: (round(v, 4) if isinstance(v, float) else v) for k, v in d.items()}


# ------------------------------------------------------------------ main
def main():
    split, vids, posture = load()
    room = split["group_of"]
    results = dict(inner={}, choice={}, test={}, per_fold_test={}, inner_per_fold={}, configs={})

    # 1-2. inner group CV and selection, per outer fold
    for f in split["folds"]:
        k = f["fold"]; sel = f["train"] + f["val"]; folds = inner_folds(sel, room)
        results["configs"][k] = {"inner_folds": [sorted(set(room[v] for v in te if v.startswith('adl'))) + [v for v in te if v.startswith('fall')] for _, te in folds]}
        inner = {}
        for var, d in CANDIDATES:
            rows, per = [], []
            for tr, te in folds:
                m, _ = V.fit([vids[v] for v in tr], var, d)
                r = [(v, vids[v].y, m.predict(vids[v])) for v in te]; rows += r; per.append(metrics(r))
            inner[cname(var, d)] = metrics(rows)
            results["inner_per_fold"][f"{k}|{cname(var, d)}"] = [rnd({x: p[x] for x in ("Precision", "Recall", "F1")}) for p in per]
        floor = 0.9 * inner["A"]["Recall"]
        ok = [(n, m) for n, m in inner.items() if m["Recall"] >= floor]
        best_f1 = max(m["F1"] for _, m in ok)
        near = [(n, m) for n, m in ok if m["F1"] >= best_f1 - 0.005]
        near.sort(key=lambda x: (-x[1]["Precision"], N_CONDITIONS[x[0][0]]))
        chosen = near[0][0]
        results["choice"][k] = dict(chosen=chosen, recall_floor=round(floor, 4), A_inner=rnd(inner["A"]), chosen_inner=rnd(inner[chosen]))
        results["inner"][k] = {n: rnd(m) for n, m in inner.items()}
        print(f"fold {k}: inner folds {len(folds)}, recall floor {floor:.3f}, chosen {chosen} "
              f"(inner P {inner[chosen]['Precision']:.3f} R {inner[chosen]['Recall']:.3f} F1 {inner[chosen]['F1']:.3f}; "
              f"A: P {inner['A']['Precision']:.3f} R {inner['A']['Recall']:.3f} F1 {inner['A']['F1']:.3f})", flush=True)

    # 3. refit on SELECTION and score TEST once (all candidates, for the tables)
    test_pred, models = {}, {}
    for f in split["folds"]:
        k = f["fold"]; sel = [vids[v] for v in f["train"] + f["val"]]
        for var, d in CANDIDATES:
            m, _ = V.fit(sel, var, d); models[(k, cname(var, d))] = m
            for v in f["test"]:
                test_pred[(cname(var, d), v)] = m.predict(vids[v])
    names = [cname(v, d) for v, d in CANDIDATES] + ["E (selected)"]
    for f in split["folds"]:
        for v in f["test"]:
            test_pred[("E (selected)", v)] = test_pred[(results["choice"][f["fold"]]["chosen"], v)]
    all_test = [v for f in split["folds"] for v in f["test"]]
    for n in names:
        results["test"][n] = rnd(metrics([(v, vids[v].y, test_pred[(n, v)]) for v in all_test]))
        for f in split["folds"]:
            results["per_fold_test"][f"{f['fold']}|{n}"] = rnd(metrics([(v, vids[v].y, test_pred[(n, v)]) for v in f["test"]]))
    results["final_models"] = {k: models[(k, results["choice"][k]["chosen"])].config() for k in (0, 1)}
    results["A_models"] = {k: models[(k, "A")].config() for k in (0, 1)}
    results["coverage_test"] = {n: {k: {kk: (round(vv, 4) if isinstance(vv, float) else vv) for kk, vv in c.items()}
                                    for k, c in coverage(models[(0, n)], [vids[v] for v in split["folds"][0]["test"]]).items()}
                                for n in ("A", results["choice"][0]["chosen"])}

    # 4a. per-rule false positives of A on the inner-CV predictions (selection data only)
    fp_rows = []
    for f in split["folds"]:
        k = f["fold"]
        for tr, te in inner_folds(f["train"] + f["val"], room):
            m, _ = V.fit([vids[v] for v in tr], "A")
            mC, _ = V.fit([vids[v] for v in tr], results["choice"][k]["chosen"][0],
                          float(results["choice"][k]["chosen"].split("=")[1][:-1]) if "=" in results["choice"][k]["chosen"] else 0.0)
            for v in te:
                vd = vids[v]; p = m.predict(vd); pc = mC.predict(vd); c = m.conditions(vd)
                for t in np.where((p == 1) & (vd.y == 0))[0]:
                    hb = "<0.3" if vd.hn[t] < 0.3 else "0.3-0.5" if vd.hn[t] < 0.5 else "0.5-0.8" if vd.hn[t] < 0.8 else ">=0.8"
                    fp_rows.append(dict(fold=k, video=v, kind="fall video (before onset / after)" if v.startswith("fall") else "ADL",
                                        urfd_code=int(posture[v][t]), height_bin=hb, drop_ge_0_4=bool(vd.drop[t] >= 0.4),
                                        fast_same_frame=bool(c["fast_now"][t]), removed_by_chosen=bool(pc[t] == 0)))
    fp = pd.DataFrame(fp_rows); os.makedirs(OUT, exist_ok=True); fp.to_csv(os.path.join(OUT, "false_positives_A_inner_cv.csv"), index=False)
    results["fp_analysis"] = dict(
        total=len(fp), by_kind=fp.kind.value_counts().to_dict(), by_height=fp.height_bin.value_counts().to_dict(),
        by_adl_urfd_code=fp[fp.kind == "ADL"].urfd_code.value_counts().to_dict(),
        with_drop_ge_0_4=int(fp.drop_ge_0_4.sum()), removed_by_chosen=int(fp.removed_by_chosen.sum()),
        top_videos=fp.video.value_counts().head(8).to_dict())

    # 4b. precision-recall trade-off on the inner CV: sweep tau_h (tau_v fitted inside), A vs each fold's choice
    def parse(name):
        return name[0], (float(name.split("=")[1][:-1]) if "=" in name else 0.0)
    pr = {}
    for label in ("A", "chosen"):
        pts = []
        for th in V.HEIGHT_GRID:
            rows = []
            for f in split["folds"]:
                var, d = ("A", 0.0) if label == "A" else parse(results["choice"][f["fold"]]["chosen"])
                for tr, te in inner_folds(f["train"] + f["val"], room):
                    tr_v = [vids[v] for v in tr]
                    lying = np.concatenate([x.hn[x.lying] for x in tr_v if x.lying is not None and x.lying.any()])
                    tau_s = float(np.percentile(lying, 95)); best = (-2, None)
                    for tv in V.VELOCITY_GRID:
                        mm = V.V3Improved(var, th, tv, d, tau_s); sc = V._score(mm, tr_v)
                        if sc > best[0]:
                            best = (sc, mm)
                    rows += [(v, vids[v].y, best[1].predict(vids[v])) for v in te]
            m = metrics(rows); pts.append(dict(tau_h=round(float(th), 3), Precision=round(m["Precision"], 4), Recall=round(m["Recall"], 4), F1=round(m["F1"], 4)))
        pr[label] = pts
    results["pr_tradeoff_inner_cv"] = pr

    # 4c. stress tests on TEST (selected models fixed, inputs perturbed)
    rng = np.random.default_rng(0)
    def perturbed(vd, kind):
        h = vd.h.copy()
        if kind == "noise 3 cm": h = h + rng.normal(0, 0.03, len(h))
        if kind == "noise 6 cm": h = h + rng.normal(0, 0.06, len(h))
        if kind == "20% frames missing": h[rng.random(len(h)) < 0.2] = np.nan
        if kind == "height scale x0.9": h = h * 0.9
        if kind == "height scale x1.1": h = h * 1.1
        if kind == "floor offset +5 cm": h = h + 0.05
        if kind == "floor offset -5 cm": h = h - 0.05
        nv = V.Video(h, vd.y, vd.lying); return nv
    stress = {}
    for kind in ("none", "noise 3 cm", "noise 6 cm", "20% frames missing", "height scale x0.9", "height scale x1.1", "floor offset +5 cm", "floor offset -5 cm"):
        for n in ("A", "E (selected)"):
            rows = []
            for f in split["folds"]:
                mdl = models[(f["fold"], "A" if n == "A" else results["choice"][f["fold"]]["chosen"])]
                for v in f["test"]:
                    vd = vids[v] if kind == "none" else perturbed(vids[v], kind)
                    rows.append((v, vids[v].y, mdl.predict(vd)))
            m = metrics(rows); stress[f"{n}|{kind}"] = rnd({x: m[x] for x in ("Precision", "Recall", "F1", "MCC")})
    results["stress_test"] = stress

    # 4d. complementarity with the out-of-fold neural ensemble (TEST, description only)
    if os.path.exists(NEURAL_OOF):
        df = pd.read_csv(NEURAL_OOF); comp = {}
        for n in ("A", "E (selected)"):
            Y, N_, S_ = [], [], []
            for v in all_test:
                g = df[df.sequence == v]; pn = np.zeros(len(vids[v].y)); pn[g.frame.values - 1] = g.prob.values
                Y.append(vids[v].y); N_.append((pn >= 0.5).astype(int)); S_.append(test_pred[(n, v)])
            y, a, b = map(np.concatenate, (Y, N_, S_)); ca, cb = a == y, b == y
            comp[n] = dict(both_correct=int((ca & cb).sum()), neural_only=int((ca & ~cb).sum()), symbolic_only=int((~ca & cb).sum()),
                           both_wrong=int((~ca & ~cb).sum()), double_fault=round(float((~ca & ~cb).mean()), 4),
                           error_correlation=round(float(np.corrcoef(~ca, ~cb)[0, 1]), 4),
                           fall_frames_neural_missed_symbolic_caught=int(((y == 1) & ~ca & cb).sum()),
                           symbolic_false_alarm_frames_where_neural_correct=int(((y == 0) & ca & ~cb).sum()))
        results["complementarity_test"] = comp

    json.dump(results, open(os.path.join(OUT, "results.json"), "w"), indent=1, default=str)
    json.dump(results["final_models"], open(os.path.join(OUT, "final_config.json"), "w"), indent=1)
    print("wrote", OUT)


if __name__ == "__main__":
    main()
