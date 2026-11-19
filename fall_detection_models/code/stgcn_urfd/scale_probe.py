"""
scale_probe.py (diagnostic, TRAIN/VAL videos only)
Hypothesis: test falls whose body scale is above the training range are missed / detected late.
Probe: zoom each train/val video by factor s around its mean neck position (missing joints stay 0),
then measure fall detection (latency, fall-frame recall) and ADL positive-frame rate with the frozen
best Fold 1 checkpoint.
"""
import json, sys, numpy as np, torch
ROOT, CODE, OUT = sys.argv[1:4]
sys.path.insert(0, CODE)
from dataset import load_sequences, window_by_end
from model import TwoStreamSTGCN
torch.set_num_threads(4)
ck = torch.load(f"{ROOT}/runs/stgcn_scene/fold1/best_checkpoint.pt", map_location="cpu", weights_only=False)
T_ = json.load(open(f"{ROOT}/runs/stgcn_scene/fold1/result.json"))["temperature"]
m = TwoStreamSTGCN(); m.load_state_dict(ck["model"]); m.eval()
fold = json.load(open(f"{ROOT}/splits/split_scene.json"))["folds"][1]
seqs = {s["name"]: s for s in load_sequences(f"{ROOT}/processed_data/cfg_d31bfed741ab/urfd_npy")}

def zoom(x, s):
    x = x.copy(); miss = x[2] <= 0
    ok = ~miss[:, 1]
    for d in (0, 1):
        c = x[d, :, 1][ok].mean()
        x[d] = np.where(miss, 0.0, (x[d] - c) * s + c)
    return x

def probs(x, ends):
    W = np.stack([window_by_end(x, t, 80)[0] for t in ends])
    with torch.no_grad():
        return torch.sigmoid(torch.cat([m(torch.from_numpy(W[i:i + 256])) for i in range(0, len(W), 256)]) / T_).numpy()

out = {}
for s in (1.0, 1.25, 1.5, 1.8):
    lat, rec, det, adl = [], [], 0, []
    nf = 0
    for part in ("train", "val"):
        for n in fold[part]:
            q = seqs[n]; x = zoom(q["data"], s); fl = q["frame_labels"]; T = x.shape[1]
            if q["label"]:
                on = int(np.argmax(fl)); p = probs(x, range(on, T)); nf += 1
                d = np.nonzero(p >= .5)[0]
                if len(d): det += 1; lat.append(int(d[0]))
                rec.append(float((p[fl[on:] == 1] >= .5).mean()))
            else:
                p = probs(x, range(0, T, 3)); adl.append(float((p >= .5).mean()))
    out[s] = dict(falls_detected=f"{det}/{nf}", median_latency_frames=float(np.median(lat)) if lat else None,
                  mean_fall_frame_recall=float(np.mean(rec)), adl_positive_frame_rate=float(np.mean(adl)),
                  adl_videos_with_alarm=int(sum(a > 0 for a in adl)))
    print(s, out[s], flush=True)
json.dump(out, open(OUT, "w"), indent=1)
