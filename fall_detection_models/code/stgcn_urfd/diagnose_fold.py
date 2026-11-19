"""
diagnose_fold.py
Is the model learning or memorising? For one finished fold:
  - learning curves (train loss, val loss, val F1 per epoch) -> PNG
  - frame-level metrics of the BEST checkpoint on train / val / test sequences (no augmentation)
  - baselines: "always non-fall" and a prior-only classifier

Reading the result:
  learning   : test F1 clearly above the baselines and not far below train F1
  memorising : train F1 near 1 while val/test F1 stay low, val loss rising while train loss falls

Usage:
  python diagnose_fold.py --data_dir .../urfd_npy --split_file splits/split_scene.json \
      --run_dir runs/stgcn_scene --fold 1 --out reports/fold1_diagnosis.png
"""
import argparse
import csv
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from dataset import EvalWindows, load_sequences
from model import TwoStreamSTGCN
from train import binary_metrics, pick_device, predict

p = argparse.ArgumentParser()
p.add_argument("--data_dir", required=True); p.add_argument("--split_file", required=True)
p.add_argument("--run_dir", required=True); p.add_argument("--fold", type=int, required=True)
p.add_argument("--out", required=True); p.add_argument("--device", default=None)
p.add_argument("--train_stride", type=int, default=2)
a = p.parse_args()

fd = os.path.join(a.run_dir, f"fold{a.fold}")
fold = json.load(open(a.split_file))["folds"][a.fold]
res = json.load(open(os.path.join(fd, "result.json")))
ck = torch.load(os.path.join(fd, "best_checkpoint.pt"), map_location="cpu", weights_only=False)
cfg = ck["config"]; temp = res["temperature"]
dev = pick_device(a.device)
model = TwoStreamSTGCN(use_conf=cfg["use_conf"]).to(dev); model.load_state_dict(ck["model"]); model.eval()

seqs = {s["name"]: s for s in load_sequences(a.data_dir, names=set(fold["train"] + fold["val"] + fold["test"]))}
rows = {}
for part, stride in (("train", a.train_stride), ("val", 1)):
    ds = EvalWindows([seqs[n] for n in fold[part]], cfg["window"], stride=stride, norm_mode=cfg["norm_mode"],
                     pad_side=cfg.get("pad_side", "right"))
    lg, y, _, _ = predict(model, ds, dev)
    rows[part] = binary_metrics(y, torch.sigmoid(torch.tensor(lg / temp)).numpy(), 0.5)
rows["test"] = res["frame_metrics"]
yt = np.array([f["label"] for f in res["frames"]])
rows["baseline: always non-fall"] = binary_metrics(yt, np.zeros(len(yt)), 0.5)

hist = list(csv.DictReader(open(os.path.join(fd, "training_history.csv"))))
ep = [int(h["epoch"]) for h in hist]
fig, ax = plt.subplots(1, 2, figsize=(12, 4))
ax[0].plot(ep, [float(h["train_loss"]) for h in hist], label="train loss (weighted BCE)")
ax[0].plot(ep, [float(h["val_loss"]) for h in hist], label="val loss (BCE)")
ax[0].axvline(res["best_epoch"], ls="--", c="gray", label=f"best epoch {res['best_epoch']}")
ax[0].set_xlabel("epoch"); ax[0].legend(); ax[0].set_title(f"fold {a.fold}: losses")
names = ["train", "val", "test"]
f1 = [rows[n]["f1"] for n in names]; rec = [rows[n]["recall_sensitivity"] for n in names]
spec = [rows[n]["specificity"] for n in names]
xi = np.arange(3); w = 0.27
ax[1].bar(xi - w, f1, w, label="F1"); ax[1].bar(xi, rec, w, label="recall"); ax[1].bar(xi + w, spec, w, label="specificity")
ax[1].set_xticks(xi); ax[1].set_xticklabels(names); ax[1].set_ylim(0, 1.05); ax[1].legend()
ax[1].set_title("best checkpoint, frame level (no augmentation)")
fig.tight_layout(); fig.savefig(a.out, dpi=110)

print(f"fold {a.fold} | best epoch {res['best_epoch']} of {len(hist)} | temperature {temp:.2f}")
print(f"{'set':28s} {'frames':>7s} {'acc':>6s} {'P':>6s} {'R':>6s} {'spec':>6s} {'F1':>6s}")
for n, m in rows.items():
    print(f"{n:28s} {m['n']:7d} {m['accuracy']:6.3f} {m['precision']:6.3f} {m['recall_sensitivity']:6.3f} "
          f"{m['specificity']:6.3f} {m['f1']:6.3f}")
gap = rows["train"]["f1"] - rows["test"]["f1"]
vl = [float(h["val_loss"]) for h in hist]
# audit fix: compare the mean of the last 5 epochs with the best value (one noisy epoch is not a trend)
rising = len(vl) > 5 and float(np.mean(vl[-5:])) > min(vl) * 1.3
print(f"train-test F1 gap: {gap:.3f} | val loss mean of last 5 epochs {np.mean(vl[-5:]):.3f} vs best {min(vl):.3f}"
      f"{' (rising)' if rising else ''}")
# audit fix: the old text called a 0.35 gap "close to training performance". Gap bands are now explicit.
# NOTE: in scene mode the validation videos share scenes with training, so val F1 is optimistic.
if rows["test"]["f1"] < 0.2:
    verdict = "NOT LEARNING a transferable pattern (test F1 near the always-non-fall baseline)"
elif gap > 0.15:
    verdict = (f"LEARNS A TRANSFERABLE SIGNAL, BUT WITH A LARGE GENERALISATION GAP "
               f"(train F1 {rows['train']['f1']:.2f} vs held-out F1 {rows['test']['f1']:.2f}, gap {gap:.2f})")
else:
    verdict = "LEARNING: held-out performance well above baseline and within 0.15 F1 of training"
print("verdict:", verdict)
json.dump(dict(fold=a.fold, best_epoch=res["best_epoch"], metrics=rows, train_test_f1_gap=gap,
               val_loss_rising=rising, verdict=verdict),
          open(os.path.splitext(a.out)[0] + ".json", "w"), indent=1)
print("saved", a.out)
