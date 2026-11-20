"""
evaluate_v3_norm.py
V3-norm study (reports/symbolic_v3_norm/PREREGISTRATION.md): inner group CV on TRAIN+VAL per outer fold, the
V3 precision study's selection rule, one score on TEST. Then ablations and the (non-blind) FUKinect check.

Usage: python3 evaluate_v3_norm.py [fukinect_h_subject.npz]     (PROJECT_ROOT overrides the project root)
"""
import json, os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import symbolic_v3_improved as V        # noqa: E402
import symbolic_v3_norm as N            # noqa: E402
import evaluate_symbolic_v3 as E        # noqa: E402

OUT = os.path.join(E.ROOT, "reports", "symbolic_v3_norm")
REF_RESULTS = os.path.join(E.ROOT, "reports", "symbolic_v3_precision", "results.json")
CANDS = [(var, d) for var in ("C", "D") for d in N.DELTAS]
NCOND = {"C": 3, "D": 4}
SETTINGS = {"V3-norm": dict(smooth=True, relative=True),
            "ablation: smoothing only": dict(smooth=True, relative=False),
            "ablation: relative height only": dict(smooth=False, relative=True)}


def name(var, d):
    return f"N{var}(delta={d * N.REF:.2f} m)"


def study(split, raw, floors, flags):
    vids = {v: N.NormVideo(h, y, ly, **flags) for v, (h, y, ly) in raw.items()}
    room = split["group_of"]; out = dict(choice={}, inner={}, models={}); test_rows = []
    for f in split["folds"]:
        k = f["fold"]; sel = f["train"] + f["val"]; folds = E.inner_folds(sel, room); inner = {}
        for var, d in CANDS:
            rows = []
            for tr, te in folds:
                m, _ = N.fit([vids[v] for v in tr], var, d)
                rows += [(v, vids[v].y, m.predict(vids[v])) for v in te]
            inner[name(var, d)] = E.metrics(rows)
        ok = [(n, m) for n, m in inner.items() if m["Recall"] >= floors[k]] or list(inner.items())
        best = max(m["F1"] for _, m in ok)
        near = sorted([(n, m) for n, m in ok if m["F1"] >= best - 0.005], key=lambda x: (-x[1]["Precision"], NCOND[x[0][1]]))
        chosen = near[0][0]; var, d = next((v_, d_) for v_, d_ in CANDS if name(v_, d_) == chosen)
        m, _ = N.fit([vids[v] for v in sel], var, d)
        out["choice"][k] = dict(chosen=chosen, recall_floor=round(floors[k], 4), above_floor=len(ok) < len(inner) or all(x["Recall"] >= floors[k] for x in inner.values()),
                                inner=E.rnd(inner[chosen]))
        out["inner"][k] = {n: E.rnd(x) for n, x in inner.items()}
        cfg = m.config(); cfg["tau_h_m_at_1.45"] = round(cfg["tau_h"] * N.REF, 3); out["models"][k] = cfg
        test_rows += [(v, vids[v].y, m.predict(vids[v])) for v in f["test"]]
    out["test"] = E.rnd(E.metrics(test_rows))
    return out


def fukinect(npz, cfgs):
    z = np.load(npz, allow_pickle=True); res = {}
    for k, cfg in cfgs.items():
        mdl = V.V3Improved(**{x: cfg[x] for x in ("variant", "tau_h", "tau_v", "delta", "tau_s")})
        y = np.array([a == "falling" for a in z["action"]]); p = np.array([mdl.predict(N.NormVideo(np.asarray(h, float))).any() for h in z["h"]])
        tp, fp, fn, tn = int((p & y).sum()), int((p & ~y).sum()), int((~p & y).sum()), int((~p & ~y).sum())
        pr = tp / max(tp + fp, 1); rc = tp / (tp + fn)
        per = {a: round(float(p[z["action"] == a].mean()), 4) for a in ("walking", "bending", "sitting", "squatting", "lying", "falling")}
        res[k] = dict(falls=f"{tp}/{tp + fn}", adl_alarms=f"{fp}/{fp + tn}", fall_detection_rate=round(rc, 4),
                      adl_false_alarm_rate=round(fp / (fp + tn), 4), precision=round(pr, 4), F1=round(2 * pr * rc / max(pr + rc, 1e-9), 4),
                      MCC=round(V.mcc_counts(tp, fp, fn, tn), 4), per_action_alarm_rate=per,
                      passes_0_80_0_45=bool(rc >= 0.80 and fp / (fp + tn) <= 0.45))
    return res


def main():
    split, vids, _ = E.load()
    raw = {v: (vd.h, vd.y, vd.lying) for v, vd in vids.items()}
    ref = json.load(open(REF_RESULTS))
    floors = {f["fold"]: ref["choice"][str(f["fold"])]["recall_floor"] for f in split["folds"]}
    results = {"reference_improved_V3_test": ref["test"]["E (selected)"],
               "reference_improved_V3_models": ref["final_models"]}
    for s, flags in SETTINGS.items():
        results[s] = study(split, raw, floors, flags)
        t = results[s]["test"]
        print(f"{s:32s} TEST P {t['Precision']:.3f} R {t['Recall']:.3f} F1 {t['F1']:.3f} Acc {t['Accuracy']:.3f} "
              f"BalAcc {t['BalancedAccuracy']:.3f} MCC {t['MCC']:.3f} falls {t['falls_detected']}/{t['falls']} clean {t['adl_clean']}/{t['adl']}")
        for k, c in results[s]["choice"].items():
            print("    fold", k, c["chosen"], results[s]["models"][k])
    r = results["reference_improved_V3_test"]; n = results["V3-norm"]
    gap_ref = abs(ref["final_models"]["0"]["tau_h"] - ref["final_models"]["1"]["tau_h"])
    gap_new = abs(n["models"][0]["tau_h_m_at_1.45"] - n["models"][1]["tau_h_m_at_1.45"])
    results["decision"] = dict(mcc_new=n["test"]["MCC"], mcc_ref=r["MCC"], rule1=n["test"]["MCC"] >= r["MCC"] - 0.02,
                               tau_h_gap_m_new=round(gap_new, 3), tau_h_gap_m_ref=round(gap_ref, 3), rule2=gap_new < gap_ref)
    results["decision"]["replace"] = bool(results["decision"]["rule1"] and results["decision"]["rule2"])
    print("decision", results["decision"])
    if len(sys.argv) > 1:
        results["fukinect_nonblind"] = fukinect(sys.argv[1], n["models"])
        for k, v in results["fukinect_nonblind"].items():
            print("FUKinect fold", k, v)
    os.makedirs(OUT, exist_ok=True)
    json.dump(results, open(os.path.join(OUT, "results.json"), "w"), indent=1, default=str)
    json.dump({k: v for k, v in n["models"].items()}, open(os.path.join(OUT, "final_config.json"), "w"), indent=1)
    print("wrote", OUT)


if __name__ == "__main__":
    main()
