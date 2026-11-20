"""
evaluate_v3_fukinect.py
Runs the frozen improved V3 configs (final_config.json) once on FUKinect-Fall neck heights. No refitting.
See reports/symbolic_v3_external_fukinect/PREREGISTRATION.md.

Usage: python evaluate_v3_fukinect.py <h_subject.npz> <h_global.npz> <final_config.json> <out_dir>
"""
import sys, os, json, csv
import numpy as np
from symbolic_v3_improved import Video, V3Improved, mcc_counts

ACTIONS = ("walking", "bending", "sitting", "squatting", "lying", "falling")


def resample(h, src_fps, dst_fps=30.0):
    if src_fps == dst_fps or len(h) < 2:
        return h
    t_src = np.arange(len(h)) / src_fps
    t_dst = np.arange(0, t_src[-1] + 1e-9, 1 / dst_fps)
    ok = np.isfinite(h)
    out = np.interp(t_dst, t_src[ok], h[ok])
    near = np.round(t_dst * src_fps).astype(int).clip(0, len(h) - 1)
    out[~ok[near]] = np.nan                       # keep gaps as gaps
    return out


def run(npz, cfg, src_fps=30.0):
    z = np.load(npz, allow_pickle=True)
    m = V3Improved(**cfg); rows = []
    for a, s, r, h in zip(z["action"], z["subject"], z["rep"], z["h"]):
        p = m.predict(Video(resample(np.asarray(h, float), src_fps)))
        rows.append(dict(action=str(a), subject=int(s), rep=int(r), y=int(a == "falling"), alarm=int(p.any()),
                         first_alarm_s=(round(float(np.argmax(p)) / 30.0, 2) if p.any() else None)))
    return rows


def summarize(rows):
    y = np.array([r["y"] for r in rows]); p = np.array([r["alarm"] for r in rows])
    tp, fp = int(((p == 1) & (y == 1)).sum()), int(((p == 1) & (y == 0)).sum())
    fn, tn = int(((p == 0) & (y == 1)).sum()), int(((p == 0) & (y == 0)).sum())
    P = tp / (tp + fp) if tp + fp else 0.0; R = tp / (tp + fn); spec = tn / (tn + fp)
    out = dict(TP=tp, FP=fp, FN=fn, TN=tn, fall_detection_rate=R, adl_false_alarm_rate=fp / (fp + tn),
               precision=P, recall=R, F1=2 * P * R / (P + R) if P + R else 0.0, accuracy=(tp + tn) / len(y),
               balanced_accuracy=(R + spec) / 2, MCC=mcc_counts(tp, fp, fn, tn))
    out["per_action_alarm_rate"] = {a: float(np.mean([r["alarm"] for r in rows if r["action"] == a])) for a in ACTIONS}
    out = {k: (round(v, 4) if isinstance(v, float) else v) for k, v in out.items()}
    out["per_action_alarm_rate"] = {k: round(v, 4) for k, v in out["per_action_alarm_rate"].items()}
    out["passes"] = bool(out["fall_detection_rate"] >= 0.80 and out["adl_false_alarm_rate"] <= 0.45)
    return out


def per_subject(rows):
    res = {}
    for s in sorted({r["subject"] for r in rows}):
        rs = [r for r in rows if r["subject"] == s]
        f = [r["alarm"] for r in rs if r["y"] == 1]; a = [r["alarm"] for r in rs if r["y"] == 0]
        res[s] = dict(falls=len(f), detected=int(sum(f)), adl=len(a), adl_alarms=int(sum(a)))
    return res


if __name__ == "__main__":
    h_subj, h_glob, cfg_path, out = sys.argv[1:5]
    os.makedirs(out, exist_ok=True)
    cfgs = json.load(open(cfg_path)); results = {}
    for fold, cfg in cfgs.items():
        name = f"fold{fold}_{cfg['variant']}"
        prim = run(h_subj, cfg)
        results[name] = dict(config=cfg, primary=summarize(prim), per_subject=per_subject(prim),
                             sensitivity_20fps=summarize(run(h_subj, cfg, src_fps=20.0)),
                             sensitivity_global_floor=summarize(run(h_glob, cfg)))
        with open(os.path.join(out, f"clips_{name}.csv"), "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(prim[0].keys())); w.writeheader(); w.writerows(prim)
    json.dump(results, open(os.path.join(out, "results.json"), "w"), indent=1)
    for k, v in results.items():
        p = v["primary"]
        print(k, "| falls", f"{p['TP']}/{p['TP'] + p['FN']}", "| ADL alarms", f"{p['FP']}/{p['FP'] + p['TN']}",
              "| P %.3f R %.3f F1 %.3f MCC %.3f BalAcc %.3f" % (p["precision"], p["recall"], p["F1"], p["MCC"], p["balanced_accuracy"]),
              "| pass", p["passes"])
        print("   per action", p["per_action_alarm_rate"])
        for s in ("sensitivity_20fps", "sensitivity_global_floor"):
            q = v[s]; print("  ", s, "det %.3f FA %.3f MCC %.3f" % (q["fall_detection_rate"], q["adl_false_alarm_rate"], q["MCC"]))
