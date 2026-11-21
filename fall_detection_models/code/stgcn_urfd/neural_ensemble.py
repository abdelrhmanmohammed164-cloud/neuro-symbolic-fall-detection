"""
neural_ensemble.py
The final neural stream: the average fall probability of the neck-centred ST-GCN models (exp04_center).

Decision: the ensemble was better than the single models on both test sets (URFD test, 3 seeds per
fold: MCC 0.547 vs 0.503; GMDCSA-24, 6 models: 0.530 vs 0.505), so it is the neural stream used from now
on.

Modes
  all : the 6 models (3 seeds x 2 folds), for data that is not URFD (as in the GMDCSA-24 external test).
  oof : out-of-fold, for URFD only. A URFD video only gets the 3 models whose training never saw its
        room (the models of the fold where the video is in the test set). Using all 6 on URFD would leak.

Every model applies its own validation temperature; the ensemble is the mean of the calibrated
probabilities; the decision threshold stays 0.5.

Usage (writes one CSV: sequence, frame (1-based), label, prob):
  python neural_ensemble.py --data_dir external_data/gmdcsa24_npy --runs_root runs/exp_mcc --mode all --out x.csv
  python neural_ensemble.py --data_dir processed_data/cfg_*/urfd_npy --runs_root runs/exp_mcc --mode oof \
      --split_file splits/split_scene_v2.json --out urfd_neural_oof.csv
"""
import argparse
import csv
import json
import os

import numpy as np
import torch

from dataset import EvalWindows, load_sequences
from model import TwoStreamSTGCN
from train import pick_device, predict

RUNS = ("exp04_center", "exp04_center_s43", "exp04_center_s44")


class NeuralEnsemble:
    def __init__(self, runs_root, folds=(0, 1), device=None):
        self.dev = pick_device(device)
        self.models = []
        for name in RUNS:
            for k in folds:
                fd = os.path.join(runs_root, name, f"fold{k}")
                ck = torch.load(os.path.join(fd, "best_checkpoint.pt"), map_location="cpu", weights_only=False)
                m = TwoStreamSTGCN(use_conf=ck["config"]["use_conf"]).to(self.dev)
                m.load_state_dict(ck["model"]); m.eval()
                T = json.load(open(os.path.join(fd, "result.json")))["temperature"]
                self.models.append(dict(name=f"{name}/fold{k}", model=m, T=T, cfg=ck["config"]))
        cfgs = {(c["cfg"]["window"], c["cfg"]["norm_mode"], c["cfg"].get("pad_side", "right")) for c in self.models}
        assert len(cfgs) == 1, f"models disagree on input settings: {cfgs}"
        self.window, self.norm_mode, self.pad_side = cfgs.pop()

    def predict(self, seqs):
        """-> per-window arrays (prob, label, sequence index, end frame index), causal windows, stride 1."""
        ds = EvalWindows(seqs, self.window, stride=1, norm_mode=self.norm_mode, pad_side=self.pad_side)
        ps = []
        for c in self.models:
            lg, y, si, en = predict(c["model"], ds, self.dev)
            ps.append(torch.sigmoid(torch.tensor(lg / c["T"])).numpy())
        return np.mean(ps, axis=0), y, si, en


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True); ap.add_argument("--runs_root", required=True)
    ap.add_argument("--mode", choices=["all", "oof"], required=True); ap.add_argument("--split_file", default=None)
    ap.add_argument("--out", required=True); ap.add_argument("--device", default=None)
    a = ap.parse_args()
    seqs = load_sequences(a.data_dir)
    rows = []
    if a.mode == "all":
        groups = [((0, 1), seqs)]
    else:
        assert a.split_file, "--mode oof needs --split_file"
        split = json.load(open(a.split_file))
        by = {s["name"]: s for s in seqs}
        groups = [((k,), [by[n] for n in split["folds"][k]["test"]]) for k in (0, 1)]
        covered = sorted(n for _, g in groups for n in (s["name"] for s in g))
        assert covered == sorted(by), "every URFD video must be in exactly one test fold"
    for folds, group in groups:
        ens = NeuralEnsemble(a.runs_root, folds=folds, device=a.device)
        p, y, si, en = ens.predict(group)
        print(f"folds {folds}: {len(ens.models)} models on {len(group)} videos")
        rows += [dict(sequence=group[s]["name"], frame=int(e) + 1, label=int(l), prob=float(q))
                 for q, l, s, e in zip(p, y, si, en)]
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["sequence", "frame", "label", "prob"]); w.writeheader(); w.writerows(rows)
    print(f"wrote {len(rows)} frame probabilities -> {a.out}")


if __name__ == "__main__":
    main()
