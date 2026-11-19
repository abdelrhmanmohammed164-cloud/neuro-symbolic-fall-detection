"""
compare_seeds.py
Seed confirmation of the only experiment kept by compare_experiments.py (exp04_center, --norm_mode
window_center). Validation only: it refuses any file that is not a --no_test validation result.

PRE-REGISTERED CONFIRMATION RULE (written before the confirmation runs started):
  pairs     : seed 42 -> baseline_v2_s42 vs exp04_center
              seed 43 -> baseline_v2_s43 vs exp04_center_s43
              seed 44 -> baseline_v2_s44 vs exp04_center_s44
  CONFIRMED only if, for EVERY seed and in BOTH folds:
      center MCC > baseline MCC (same seed, same fold)
  and, summed over the three seeds, per fold:
      falls detected (center) >= falls detected (baseline)
      ADL validation videos with no false alarm (center) >= baseline
  Otherwise NOT CONFIRMED: the baseline configuration stays the reference.
  The mean MCC difference per fold and its smallest per-seed value are reported either way.

Usage:
  python compare_seeds.py --runs_root runs/exp_mcc --out reports/experiments_mcc
"""
import argparse
import json
import os

import numpy as np

from compare_experiments import fold_metrics

PAIRS = {42: ("baseline_v2_s42", "exp04_center"),
         43: ("baseline_v2_s43", "exp04_center_s43"),
         44: ("baseline_v2_s44", "exp04_center_s44")}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--runs_root", required=True); p.add_argument("--out", required=True)
    a = p.parse_args()
    m = {}
    for seed, names in PAIRS.items():
        for n in names:
            m[n] = {k: fold_metrics(os.path.join(a.runs_root, n, f"fold{k}", "val_result.json")) for k in (0, 1)}
    ok, why, rows, diffs = True, [], [], {0: [], 1: []}
    for k in (0, 1):
        fb = fc = ab = ac = 0
        for seed, (b, c) in PAIRS.items():
            mb, mc = m[b][k], m[c][k]
            d = mc["mcc"] - mb["mcc"]; diffs[k].append(d)
            rows.append((seed, k, mb, mc, d))
            if not d > 0:
                ok = False; why.append(f"seed {seed} fold {k}: center MCC {mc['mcc']:.3f} <= baseline {mb['mcc']:.3f}")
            fb += mb["falls_detected"]; fc += mc["falls_detected"]; ab += mb["adl_clean"]; ac += mc["adl_clean"]
        if fc < fb:
            ok = False; why.append(f"fold {k}: falls detected {fc} < {fb} (summed over seeds)")
        if ac < ab:
            ok = False; why.append(f"fold {k}: ADL videos without false alarm {ac} < {ab} (summed over seeds)")
    lines = ["# Seed confirmation of exp04_center (validation only, no test data)", "",
             "| seed | fold | baseline MCC | center MCC | difference | baseline falls / ADL clean | center falls / ADL clean |",
             "|---|---|---|---|---|---|---|"]
    for seed, k, mb, mc, d in rows:
        lines.append(f"| {seed} | {k} | {mb['mcc']:.3f} | {mc['mcc']:.3f} | {d:+.3f} | "
                     f"{mb['falls_detected']}/{mb['falls']} · {mb['adl_clean']}/{mb['adl']} | "
                     f"{mc['falls_detected']}/{mc['falls']} · {mc['adl_clean']}/{mc['adl']} |")
    lines += ["", f"Mean MCC difference: fold 0 {np.mean(diffs[0]):+.3f} (smallest {min(diffs[0]):+.3f}), "
                  f"fold 1 {np.mean(diffs[1]):+.3f} (smallest {min(diffs[1]):+.3f})", "",
              f"**Decision: {'CONFIRMED' if ok else 'NOT CONFIRMED'}**"]
    lines += [f"- {w}" for w in why]
    os.makedirs(a.out, exist_ok=True)
    open(os.path.join(a.out, "seed_confirmation.md"), "w").write("\n".join(lines) + "\n")
    json.dump(dict(rule=__doc__.split("PRE-REGISTERED CONFIRMATION RULE")[1].split("Usage:")[0].strip(),
                   confirmed=ok, reasons=why, metrics=m,
                   mean_mcc_diff={k: float(np.mean(v)) for k, v in diffs.items()}),
              open(os.path.join(a.out, "seed_confirmation.json"), "w"), indent=1)
    print("\n".join(lines))


if __name__ == "__main__":
    main()
