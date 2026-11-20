"""
evaluate_early_fall.py
The early-fall rule study (reports/early_fall_rule/PREREGISTRATION.md).
  1. Per outer fold: inner group CV on SELECTION(k) for A (improved V3) and A+B, A+C, A+D; acceptance rule.
  2. Fold k VAL: V3 and the early rule fitted on TRAIN(k); unchanged fusion (veto+sym) with the neural ensemble.
  3. TEST(k), fitted on SELECTION(k), scored once (not independent for this change).
  4. GMDCSA-24 (untouched): the early rule alone, thresholds from the ADL frames of all 70 URFD videos.
Outputs: reports/early_fall_rule/results.json
"""
import json
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import early_fall as EF  # noqa: E402
import evaluate_symbolic_v3 as E  # noqa: E402
import fusion_v3_improved as FU  # noqa: E402
import symbolic_v3_improved as V  # noqa: E402

ROOT = E.ROOT
OUT = os.path.join(ROOT, "reports", "early_fall_rule")
GMD = os.path.join(ROOT, "external_data", "gmdcsa24_npy")
KINDS = ("B", "C", "D")


def parse(name):
    return name[0], (float(name.split("=")[1][:-1]) if "=" in name else 0.0)


def onset_stats(rows):
    """Per fall video: delay (s) from the labelled onset to the first positive at or after it; early-phase recall."""
    delays, early_hit, early_tot = [], 0, 0
    for v, y, p in rows:
        if not v.startswith("fall") or not y.any():
            continue
        on = int(np.argmax(y)); d = np.nonzero(p[on:])[0]
        delays.append(d[0] / EF.FPS if len(d) else None)
        seg = p[on:on + 30]; early_hit += int(seg.sum()); early_tot += len(seg)
    got = [d for d in delays if d is not None]
    return dict(falls=len(delays), detected=len(got), median_delay_s=round(float(np.median(got)), 3) if got else None,
                mean_delay_s=round(float(np.mean(got)), 3) if got else None,
                first_second_recall=round(early_hit / max(early_tot, 1), 4))


def comp(rows_sym, neural, vids):
    y = np.concatenate([r[1] for r in rows_sym]); s = np.concatenate([r[2] for r in rows_sym])
    n = np.concatenate([(neural[r[0]] >= 0.5).astype(int) for r in rows_sym])
    return dict(neural_FN=int(((y == 1) & (n == 0)).sum()), symbolic_catches_neural_FN=int(((y == 1) & (n == 0) & (s == 1)).sum()),
                shared_FN=int(((y == 1) & (n == 0) & (s == 0)).sum()), neural_FP=int(((y == 0) & (n == 1)).sum()),
                shared_FP=int(((y == 0) & (n == 1) & (s == 1)).sum()), symbolic_only_FP=int(((y == 0) & (n == 0) & (s == 1)).sum()))


def main():
    split, vids, _ = E.load(); room = split["group_of"]
    kp = {v: np.load(os.path.join(E.NPY, f"{v}_raw_coco17.npy")) for v in vids}
    feats = {v: EF.features(kp[v]) for v in vids}
    chosen = json.load(open(os.path.join(ROOT, "reports", "symbolic_v3_precision", "results.json")))["choice"]
    NV = FU.probs(FU.NEURAL_VAL if os.path.exists(FU.NEURAL_VAL) else os.path.join(OUT, "neural_val.csv"), vids)
    NT = FU.probs(FU.NEURAL_TEST, vids)
    res = dict(inner={}, decision={}, val={}, test={}, per_fold_test={}, thresholds={}, onset={}, complementarity={})

    def build(fit_vids, var, d, kind):
        m, _ = V.fit([vids[v] for v in fit_vids], var, d)
        er = None if kind == "A" else EF.fit([feats[v] for v in fit_vids if v.startswith("adl")], kind)
        return EF.V3WithEarly(m, er)

    test_rows = {x: [] for x in ("A",) + KINDS}
    for f in split["folds"]:
        k = f["fold"]; var, d = parse(chosen[str(k)]["chosen"]); sel = f["train"] + f["val"]
        inner = {x: [] for x in ("A",) + KINDS}; prec_A = []
        for tr, te in E.inner_folds(sel, room):
            for x in ("A",) + KINDS:
                mdl = build(tr, var, d, x)
                rows = [(v, vids[v].y, mdl.predict(vids[v], feats[v])) for v in te]
                inner[x] += rows
                if x == "A":
                    prec_A.append(E.metrics(rows)["Precision"])
        M = {x: E.metrics(r) for x, r in inner.items()}; sdA = float(np.std(prec_A))
        acc = []
        for x in KINDS:
            ok = (M[x]["Recall"] > M["A"]["Recall"] and M[x]["F1"] > M["A"]["F1"]
                  and M[x]["Precision"] >= M["A"]["Precision"] - sdA and M[x]["adl_clean"] >= M["A"]["adl_clean"] - 1)
            acc.append((x, ok))
        good = [x for x, ok in acc if ok]
        if good:
            best = max(M[x]["F1"] for x in good)
            final = sorted([x for x in good if M[x]["F1"] >= best - 0.005], key=lambda x: (x == "D", -M[x]["F1"]))[0]
        else:
            final = "A"
        res["inner"][k] = {x: E.rnd(m) for x, m in M.items()}
        res["inner"][k]["onset"] = {x: onset_stats(r) for x, r in inner.items()}
        res["decision"][k] = dict(v3=chosen[str(k)]["chosen"], sd_precision_A=round(sdA, 4), acceptable={x: ok for x, ok in acc}, final=final)
        print(f"fold {k}: V3 {chosen[str(k)]['chosen']}, sd(prec A)={sdA:.3f}, acceptable {dict(acc)} -> {final}", flush=True)
        for x in ("A",) + KINDS:
            print(f"   inner {x}: P {M[x]['Precision']:.3f} R {M[x]['Recall']:.3f} F1 {M[x]['F1']:.3f} ADL clean {M[x]['adl_clean']}/{M[x]['adl']} "
                  f"onset {res['inner'][k]['onset'][x]}", flush=True)
        # VAL: fit on TRAIN only; fusion unchanged
        res["val"][k] = {}
        for x in ("A",) + KINDS:
            mdl = build(f["train"], var, d, x)
            srows = [(v, vids[v].y, mdl.predict(vids[v], feats[v])) for v in f["val"]]
            frows = [(v, vids[v].y, FU.fuse("veto+sym", NV[v], p, vids[v].h)) for v, _, p in srows]
            res["val"][k][x] = dict(symbolic=E.rnd(E.metrics(srows)), fusion=E.rnd(E.metrics(frows)),
                                    complementarity=comp(srows, NV, vids))
        # TEST: fit on SELECTION
        for x in ("A",) + KINDS:
            mdl = build(sel, var, d, x)
            if x != "A":
                res["thresholds"][f"{k}|{x}"] = mdl.early.config()
            test_rows[x] += [(v, vids[v].y, mdl.predict(vids[v], feats[v])) for v in f["test"]]
    for x in ("A",) + KINDS:
        srows = test_rows[x]; frows = [(v, vids[v].y, FU.fuse("veto+sym", NT[v], p, vids[v].h)) for v, _, p in srows]
        res["test"][x] = dict(symbolic=E.rnd(E.metrics(srows)), fusion=E.rnd(E.metrics(frows)),
                              onset_symbolic=onset_stats(srows), onset_fusion=onset_stats(frows), complementarity=comp(srows, NT, vids))
    sel_rows = []
    for f in split["folds"]:
        fin = res["decision"][f["fold"]]["final"]; sel_rows += [r for r in test_rows[fin] if r[0] in f["test"]]
    frows = [(v, vids[v].y, FU.fuse("veto+sym", NT[v], p, vids[v].h)) for v, _, p in sel_rows]
    res["test"]["selected per fold"] = dict(symbolic=E.rnd(E.metrics(sel_rows)), fusion=E.rnd(E.metrics(frows)),
                                            onset_symbolic=onset_stats(sel_rows), onset_fusion=onset_stats(frows), complementarity=comp(sel_rows, NT, vids))
    neural_rows = [(v, vids[v].y, (NT[v] >= 0.5).astype(int)) for f in split["folds"] for v in f["test"]]
    res["test"]["neural alone"] = dict(fusion=E.rnd(E.metrics(neural_rows)), onset_fusion=onset_stats(neural_rows))

    # GMDCSA-24, untouched: early rule alone, thresholds from all 70 URFD videos' ADL frames
    if os.path.isdir(GMD):
        ext = {}
        names = sorted({f[:-len("_raw_coco17.npy")] for f in os.listdir(GMD) if f.endswith("_raw_coco17.npy")})
        G = {n: (np.load(os.path.join(GMD, f"{n}_raw_coco17.npy")), np.load(os.path.join(GMD, f"{n}_frame_labels.npy")).astype(int)) for n in names}
        Gf = {n: EF.features(G[n][0]) for n in names}
        for x in KINDS:
            er = EF.fit([feats[v] for v in vids if v.startswith("adl")], x)
            rows = [(n, G[n][1], er.alarm(Gf[n])) for n in names]
            m = E.metrics(rows); o = onset_stats(rows)
            ext[x] = dict(thresholds=er.config(), metrics=E.rnd(m), onset=o)
            print(f"GMDCSA {x}: falls with alarm from onset {m['falls_detected']}/{m['falls']}, ADL without alarm {m['adl_clean']}/{m['adl']}, "
                  f"alarms/min {m['alarms_per_min']:.2f}, P {m['Precision']:.3f} R {m['Recall']:.3f}, onset {o}", flush=True)
        res["external_gmdcsa24"] = ext
    os.makedirs(OUT, exist_ok=True)
    json.dump(res, open(os.path.join(OUT, "results.json"), "w"), indent=1, default=str)
    print("wrote", OUT)


if __name__ == "__main__":
    main()
