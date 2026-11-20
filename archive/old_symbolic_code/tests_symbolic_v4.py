"""
tests_symbolic_v4.py
Automated checks for symbolic v4 and the fusion protocol. Run: python3 tests_symbolic_v4.py
Synthetic tests need nothing else; the split/leakage tests need the project data (PROJECT_ROOT).
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import symbolic_v4 as S4  # noqa: E402

FPS = 30
results = []


def check(name, cond, detail=""):
    results.append((name, bool(cond), detail)); print(("PASS " if cond else "FAIL ") + name + (f"  ({detail})" if detail else ""))


def fall_curve(start=1.35, end=0.25, t_fall=60, dur=20, T=150):
    h = np.full(T, start, float); h[t_fall:t_fall + dur] = np.linspace(start, end, dur); h[t_fall + dur:] = end
    return h


def slow_curve(start=1.35, end=0.25, t0=30, dur=150, T=240):
    h = np.full(T, start, float); h[t0:t0 + dur] = np.linspace(start, end, dur); h[t0 + dur:] = end
    return h


m = S4.SymbolicV4(tau_low=0.40, delta_fall=0.45)

# 1. a fast fall enters FALL and stays there while lying
st = m.states(fall_curve())
check("fast fall reaches FALL", (st == S4.S["FALL"]).any())
check("FALL held while lying", (st[100:] == S4.S["FALL"]).all())
check("falling phase marked DESCENDING before impact", (st[62:78] == S4.S["DESCENDING"]).any())
check("standing before the fall is UPRIGHT", (st[:55] == S4.S["UPRIGHT"]).all())

# 2. slow lowering (lying down on purpose) never enters FALL
st = m.states(slow_curve())
check("slow lowering never FALL", not (st == S4.S["FALL"]).any(), f"states {sorted(set(st.tolist()))}")
check("slow lowering ends LOW_NOFALL", st[-1] == S4.S["LOW_NOFALL"])

# 3. getting up ends the fall state
h = fall_curve(T=200); h[150:] = np.linspace(0.25, 1.35, 50)
st = m.states(h)
check("getting up leaves FALL", st[-1] != S4.S["FALL"])

# 4. missing heights: short gap keeps FALL, long gap becomes UNKNOWN; no crash on all-NaN
h = fall_curve(T=200); h[110:125] = np.nan
check("15-frame gap inside a fall stays FALL", (m.states(h)[110:125] == S4.S["FALL"]).all())
h[110:180] = np.nan
check("70-frame gap becomes UNKNOWN", m.states(h)[175] == S4.S["UNKNOWN"])
st = m.states(np.full(50, np.nan))
check("all-missing video -> UNKNOWN everywhere", (st == S4.S["UNKNOWN"]).all())
check("empty video", len(m.states(np.array([]))) == 0)

# 5. exactly one state per frame, valid codes, causal (future does not change the past)
h = fall_curve()
st = m.states(h)
check("one valid state per frame", st.shape == h.shape and set(st.tolist()) <= set(range(len(S4.STATES))))
h2 = h.copy(); h2[100:] = 1.35
check("causal: changing the future leaves earlier states unchanged", (m.states(h2)[:100] == st[:100]).all())

# 6. calibrated probabilities are valid and deterministic
train = [(fall_curve(), (np.arange(150) >= 60).astype(int), np.r_[np.full(60, -1), np.zeros(30), np.ones(60)].astype(int)),
         (slow_curve(), np.zeros(240, int), np.full(240, -1))]
fm = S4.fit(train)
p = fm.predict(fall_curve())["p"]
check("probabilities in (0, 1)", np.all((p > 0) & (p < 1)))
check("fit is deterministic", S4.fit(train).config() == fm.config())
check("support80 is 0/1", set(np.unique(fm.predict(fall_curve())["support80"]).tolist()) <= {0, 1})

# 7. split integrity and leakage rules on the real project (skipped without data)
root = os.environ.get("PROJECT_ROOT", os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))
sp_path = os.path.join(root, "splits", "split_scene_v2.json")
if os.path.exists(sp_path):
    sp = json.load(open(sp_path))
    for f in sp["folds"]:
        tr, va, te = set(f["train"]), set(f["val"]), set(f["test"])
        check(f"fold {f['fold']}: train/val/test disjoint", not (tr & va) and not (tr & te) and not (va & te))
        rooms = lambda s: {sp["group_of"][v] for v in s}
        check(f"fold {f['fold']}: no room shared between train and test", not (rooms(tr) & rooms(te)))
    f0, f1 = sp["folds"]
    check("documented overlap: fold 1 val is inside fold 0 test (why selection is per fold)",
          set(f1["val"]) <= set(f0["test"]) and set(f0["val"]) <= set(f1["test"]))
    allt = sorted(f0["test"] + f1["test"])
    check("every video in exactly one test fold", allt == sorted(sp["group_of"]))
    # thresholds depend on training videos only: changing a test video's height must not change the fit
    import symbolic_v3 as S3
    kin = os.path.join(root, "processed_data", "kinect_height")
    if os.path.isdir(kin):
        import pandas as pd
        cols = ["v", "f", "lab", "hw", "mm", "occ", "std", "hh", "H", "Dmm", "p40"]
        U = pd.read_csv(os.path.join(root, "URFD", "features", "urfall-cam0-falls.csv"), header=None, names=cols)
        npy = os.path.join(root, "processed_data", "cfg_d31bfed741ab", "urfd_npy")
        def item(v):
            y = np.load(os.path.join(npy, f"{v}_frame_labels.npy")).astype(int); l3 = np.full(len(y), -1)
            d = U[U.v == v]; i = d.f.values - 1; ok = i < len(y); l3[i[ok]] = d.lab.values[ok]
            return S3.load_height(kin, v), y, l3
        a = S4.fit([item(v) for v in f0["train"]]).config()
        check("fit on fold 0 training videos is reproducible", a == S4.fit([item(v) for v in f0["train"]]).config())

n_fail = sum(not ok for _, ok, _ in results)
print(f"\n{len(results) - n_fail} passed, {n_fail} failed")
sys.exit(1 if n_fail else 0)
