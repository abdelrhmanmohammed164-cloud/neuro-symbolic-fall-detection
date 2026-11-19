"""
context_probe.py (diagnostic, uses TRAIN/VAL videos only, never the test fold)
Hypothesis: the Fold 1 model needs the fall to fill a large part of the 80-frame window, because
every training fall starts before frame 60. Probe: prepend k pre-fall frames (ping-pong of the
first 20 real frames, so the person keeps standing/moving) to train/val fall videos and measure
detection latency of the frozen best checkpoint. Also: P(fall) on ADL val videos is unchanged.
"""
import json, sys, numpy as np, torch
ROOT, CODE, OUT = sys.argv[1:4]
sys.path.insert(0, CODE)
from dataset import load_sequences, window_by_end, normalize
from model import TwoStreamSTGCN

ck = torch.load(f"{ROOT}/runs/stgcn_scene/fold1/best_checkpoint.pt", map_location="cpu", weights_only=False)
res = json.load(open(f"{ROOT}/runs/stgcn_scene/fold1/result.json"))
T_ = res["temperature"]
m = TwoStreamSTGCN(use_conf=ck["config"]["use_conf"]); m.load_state_dict(ck["model"]); m.eval()
fold = json.load(open(f"{ROOT}/splits/split_scene.json"))["folds"][1]
seqs = {s["name"]: s for s in load_sequences(f"{ROOT}/processed_data/cfg_d31bfed741ab/urfd_npy")}

def probs(x):
    T = x.shape[1]
    W = np.stack([window_by_end(x, t, 80)[0] for t in range(T)])
    with torch.no_grad():
        lg = torch.cat([m(torch.from_numpy(W[i:i + 256])) for i in range(0, T, 256)])
    return torch.sigmoid(lg / T_).numpy()

def prepend(x, k):
    head = x[:, :20]
    pp = np.concatenate([head, head[:, ::-1]], axis=1)
    reps = int(np.ceil(k / pp.shape[1])) + 1
    pre = np.concatenate([pp] * reps, axis=1)[:, :k]
    return np.concatenate([pre, x], axis=1)

out = {"fall": {}, "adl": {}}
for part in ("train", "val"):
    for n in fold[part]:
        s = seqs[n]; x = s["data"]; fl = s["frame_labels"]
        if s["label"] == 1:
            on = int(np.argmax(fl)); r = {}
            for k in (0, 40, 80, 120):
                p = probs(prepend(x, k)); o = on + k
                det = np.nonzero(p[o:] >= 0.5)[0]
                pre_alarm = bool((p[:o] >= 0.5).any())
                r[k] = dict(latency_frames=int(det[0]) if len(det) else None, recall=float((p[o:][fl[on:] == 1] >= .5).mean()),
                            alarm_before_onset=pre_alarm)
            out["fall"][f"{part}:{n}"] = dict(onset=on + 1, **{str(k): v for k, v in r.items()})
        else:
            r = {}
            for k in (0, 80):
                p = probs(prepend(x, k))[k:]
                r[str(k)] = float((p >= .5).mean())
            out["adl"][f"{part}:{n}"] = r
json.dump(out, open(OUT, "w"), indent=1)
print("video onset | latency (frames) / fall-frame recall for k = 0, 40, 80, 120 prepended frames")
for n, r in out["fall"].items():
    print(f"{n:16s} {r['onset']:4d} | " + "  ".join(
        f"k{k}: {r[k]['latency_frames'] if r[k]['latency_frames'] is not None else 'MISS':>4} / {r[k]['recall']:.2f}" for k in ("0", "40", "80", "120")))
summ = {k: [r[k]["recall"] for r in out["fall"].values()] for k in ("0", "40", "80", "120")}
lat = {k: [r[k]["latency_frames"] for r in out["fall"].values()] for k in ("0", "40", "80", "120")}
for k in summ:
    L = [v for v in lat[k] if v is not None]
    print(f"k={k:>3}: mean fall-frame recall {np.mean(summ[k]):.2f} | detected {len(L)}/{len(lat[k])} | median latency {np.median(L) if L else float('nan'):.0f} frames")
print("ADL positive-frame rate (k=0 vs k=80):", {n: r for n, r in out["adl"].items() if r["0"] > 0 or r["80"] > 0})
