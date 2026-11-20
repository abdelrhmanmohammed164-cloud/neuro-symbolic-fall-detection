"""
tests_symbolic_v3_improved.py
Checks for the improved V3: equivalence of variant A with the original V3, rule edge cases, causality,
inner-CV and split integrity. Run: python3 tests_symbolic_v3_improved.py  (PROJECT_ROOT for the data checks)
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import symbolic_v3_improved as V  # noqa: E402

results = []


def check(name, cond, detail=""):
    results.append(bool(cond)); print(("PASS " if cond else "FAIL ") + name + (f"  ({detail})" if detail else ""))


def curve(kind, T=200):
    h = np.full(T, 1.35)
    if kind == "fall":       # fast drop to 0.55 m in 0.4 s, then the body settles slowly to 0.25 m (as real falls do)
        h[60:72] = np.linspace(1.35, 0.55, 12); h[72:102] = np.linspace(0.55, 0.25, 30); h[102:] = 0.25
    if kind == "slow":       h[30:180] = np.linspace(1.35, 0.25, 150); h[180:] = 0.25
    if kind == "sit":        h[60:75] = np.linspace(1.35, 0.95, 15); h[75:] = 0.95
    if kind == "crouch":     h[60:72] = np.linspace(1.35, 0.55, 12); h[72:] = 0.55
    return h


# 1. rules on synthetic signals
for var, d in (("B", 0.0), ("C", 0.40), ("D", 0.40)):
    m = V.V3Improved(var, tau_h=0.40, tau_v=1.0, delta=d, tau_s=0.45)
    check(f"{var}: fast fall is detected", m.predict(V.Video(curve("fall"))).any())
    check(f"{var}: slow lowering is not", not m.predict(V.Video(curve("slow"))).any())
    check(f"{var}: quick sit-down is not", not m.predict(V.Video(curve("sit"))).any())
m = V.V3Improved("A", tau_h=0.40, tau_v=1.0)
check("A: same-frame rule misses a fall that ends slowly at the floor (the V3 weakness)", not m.predict(V.Video(curve("fall"))).any())
mC = V.V3Improved("C", tau_h=0.60, tau_v=1.0, delta=0.90)
check("C: quick crouch to 0.55 m is blocked by the drop requirement (0.80 m < delta 0.90 m)", not mC.predict(V.Video(curve("crouch"))).any())
mD = V.V3Improved("D", tau_h=0.40, tau_v=1.0, delta=0.4, tau_s=0.45)
p = mD.predict(V.Video(curve("fall")))
check("D: latch keeps the fall on while lying", p[120:].all())
h = curve("fall"); h[150:] = np.linspace(0.25, 1.35, 50)
check("D: getting up ends the latch", mD.predict(V.Video(h))[-1] == 0)

# 2. missing data and edge cases
check("all-missing video -> no positives", not V.V3Improved("D", 0.4, 1.0, 0.4, 0.45).predict(V.Video(np.full(60, np.nan))).any())
check("empty video", len(V.V3Improved("C", 0.4, 1.0, 0.4).predict(V.Video(np.array([])))) == 0)
h = curve("fall"); h[100:200] = np.nan
check("missing heights never count as low", V.Video(h).hn[110] == V.MISSING_AS)

# 3. causality
for var in ("A", "B", "C", "D"):
    m = V.V3Improved(var, 0.40, 1.0, 0.4, 0.45); h = curve("fall"); h2 = h.copy(); h2[120:] = 1.35
    check(f"{var}: causal", (m.predict(V.Video(h))[:120] == m.predict(V.Video(h2))[:120]).all())

# 4. variant A equals the original V3, and data/split checks
root = os.environ.get("PROJECT_ROOT", os.path.abspath(os.path.join(HERE, "..", "..", "..")))
sp_path, kin = os.path.join(root, "splits", "split_scene_v2.json"), os.path.join(root, "processed_data", "kinect_height")
if os.path.exists(sp_path) and os.path.isdir(kin):
    import symbolic_v3 as S3
    import evaluate_symbolic_v3 as E
    E.ROOT, E.NPY, E.KIN = root, os.path.join(root, "processed_data", "cfg_d31bfed741ab", "urfd_npy"), kin
    split, vids, _ = E.load()
    f0 = split["folds"][0]
    tr = f0["train"]
    lying = np.concatenate([vids[v].hn[vids[v].lying] for v in tr if vids[v].lying is not None and vids[v].lying.any()])
    orig, _ = S3.fit([(S3.load_height(kin, v), vids[v].y) for v in tr], lying)
    new, _ = V.fit([vids[v] for v in tr], "A")
    check("A thresholds equal original V3", abs(orig.tau_h - new.tau_h) < 1e-9 and abs(orig.tau_v - new.tau_v) < 1e-9,
          f"orig {orig.tau_h:.3f}/{orig.tau_v:.3f} new {new.tau_h:.3f}/{new.tau_v:.3f}")
    same = all((orig.predict(S3.load_height(kin, v))["trigger"] == new.predict(vids[v])).all() for v in f0["test"])
    check("A predictions equal original V3 on every test video of fold 0", same)
    for f in split["folds"]:
        sel = f["train"] + f["val"]
        check(f"fold {f['fold']}: selection set and test disjoint", not (set(sel) & set(f["test"])))
        folds = E.inner_folds(sel, split["group_of"])
        cover = sorted(v for _, te in folds for v in te)
        check(f"fold {f['fold']}: inner folds partition the selection set", cover == sorted(sel))
        leak = any(split["group_of"][v] in {split["group_of"][u] for u in te if u.startswith("adl")} for tr_, te in folds for v in tr_ if v.startswith("adl"))
        check(f"fold {f['fold']}: no ADL room in both inner train and inner held-out", not leak)
        near = {"adl-25", "adl-27", "adl-29"}
        check(f"fold {f['fold']}: near-duplicates never split across inner folds", all(len(near & set(te)) in (0, len(near & set(sel))) for _, te in folds))
else:
    print("SKIP data checks (no project data)")

src = open(os.path.join(HERE, "symbolic_v3_improved.py")).read() + open(os.path.join(HERE, "evaluate_symbolic_v3.py")).read()
check("no V4 code used", "symbolic_v4" not in src and "fusion_v4" not in src)
check("no notebook used", ".ipynb" not in src)
n_fail = results.count(False)
print(f"\n{len(results) - n_fail} passed, {n_fail} failed"); sys.exit(1 if n_fail else 0)
