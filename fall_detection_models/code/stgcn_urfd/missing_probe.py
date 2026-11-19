"""
missing_probe.py (DIAGNOSTIC ONLY: explains a failure mode, selects nothing)
Fold 0 false alarms in scene A4 start while 6-8 of 14 joints are missing (coded (0,0), conf 0) and end
~80 frames later. Counterfactual: fill each missing joint with the nearest valid value of the same
joint in time (conf set to a small positive value so it counts as present), run the SAME frozen
Fold 0 checkpoint, and compare ADL false-alarm rates. No setting is chosen from this.
"""
import json, sys, numpy as np, torch
ROOT, CODE, CKPT, OUT = sys.argv[1:5]
sys.path.insert(0, CODE)
from dataset import load_sequences, window_by_end
from model import TwoStreamSTGCN
torch.set_num_threads(4)
ck = torch.load(CKPT, map_location="cpu", weights_only=False)
T_ = json.load(open(f"{ROOT}/runs/stgcn_scene/fold0/result.json"))["temperature"]
m = TwoStreamSTGCN(); m.load_state_dict(ck["model"]); m.eval()
fold = json.load(open(f"{ROOT}/splits/split_scene.json"))["folds"][0]
seqs = {s["name"]: s for s in load_sequences(f"{ROOT}/processed_data/cfg_d31bfed741ab/urfd_npy")}

def fill(x):
    x = x.copy(); T, V = x.shape[1], x.shape[2]
    for v in range(V):
        ok = np.nonzero(x[2, :, v, 0] > 0)[0]
        if len(ok) == 0:
            continue
        for t in range(T):
            if x[2, t, v, 0] <= 0:
                j = ok[np.argmin(np.abs(ok - t))]
                x[:2, t, v, 0] = x[:2, j, v, 0]; x[2, t, v, 0] = 0.05
    return x

def probs(x):
    W = np.stack([window_by_end(x, t, 80)[0] for t in range(x.shape[1])])
    with torch.no_grad():
        return torch.sigmoid(torch.cat([m(torch.from_numpy(W[i:i + 256])) for i in range(0, len(W), 256)]) / T_).numpy()

out = {}
for n in fold["test"]:
    s = seqs[n]
    if s["label"]:
        continue
    a = probs(s["data"]); b = probs(fill(s["data"]))
    out[n] = dict(orig_fp=float((a >= .5).mean()), filled_fp=float((b >= .5).mean()),
                  orig_fp_first80=float((a[:80] >= .5).mean()), filled_fp_first80=float((b[:80] >= .5).mean()),
                  missing_frac=float((s["data"][2] <= 0).mean()))
    print(n, {k: round(v, 2) for k, v in out[n].items()}, flush=True)
tot = {k: float(np.mean([r[k] for r in out.values()])) for k in ("orig_fp", "filled_fp", "orig_fp_first80", "filled_fp_first80")}
tot["adl_videos_with_alarm_orig"] = sum(r["orig_fp"] > 0 for r in out.values())
tot["adl_videos_with_alarm_filled"] = sum(r["filled_fp"] > 0 for r in out.values())
out["_summary"] = tot
print("summary", tot)
json.dump(out, open(OUT, "w"), indent=1)
