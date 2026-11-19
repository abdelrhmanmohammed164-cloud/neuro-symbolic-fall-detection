"""
train.py
Training of the two-stream ST-GCN neural stream with crash-safe checkpoints and exact resume.

Reads a split file made by splits.py (never builds splits itself). For each fold:
  train on `train`, pick the best epoch and fit the temperature on `val`, evaluate once on `test`.

Paper (Section 3.3.1 / 4) defaults: Adadelta, 50 epochs, batch 256, T = 80.
Engineering choices (paper silent): BCE with pos_weight from training windows, weight decay 1e-4,
best epoch by validation loss (no early stopping), temperature scaling on validation.

Per fold, in <out_dir>/fold<k>/:
  last_checkpoint.pt    every epoch: model, optimizer, epoch, step, best metric, history, RNG states,
                        config / data / split fingerprints, label map
  best_checkpoint.pt    only replaced when validation loss improves
  training_history.csv, training.log, result.json (written when the fold is complete)
All checkpoints: temp file -> flush -> fsync -> atomic rename.
Resume: the last completed epoch is restored if every fingerprint matches; otherwise the run stops
(no silent restart). Mid-epoch progress is not saved: training resumes from the last completed epoch.

Usage:
  python train.py --data_dir .../urfd_npy --split_file splits/split_scene.json --out_dir runs/stgcn_scene
"""
import argparse
import csv
import hashlib
import io
import json
import logging
import os
import random
import sys
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from dataset import EvalWindows, TrainWindows, load_sequences
from model import TwoStreamSTGCN

LABEL_MAP = {"0": "non-fall (ADL frame)", "1": "fall frame (URFD label 0 or 1 in a fall video)"}
CONFIG_KEYS = ["window", "windows_per_seq", "min_pos", "p_pad", "norm_mode", "use_conf", "optimizer",
               "lr", "weight_decay", "epochs", "batch_size", "val_stride", "seed"]


def get_args(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", required=True)
    p.add_argument("--split_file", required=True)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--window", type=int, default=80)
    p.add_argument("--windows_per_seq", type=int, default=64)
    p.add_argument("--min_pos", type=int, default=10)
    p.add_argument("--p_pad", type=float, default=0.35)
    p.add_argument("--norm_mode", choices=["image", "window_center"], default="image")
    p.add_argument("--use_conf", action="store_true")
    p.add_argument("--optimizer", choices=["adadelta", "adamw"], default="adadelta")
    p.add_argument("--lr", type=float, default=None, help="default 1.0 (adadelta) / 1e-3 (adamw)")
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--val_stride", type=int, default=4)
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default=None)
    p.add_argument("--only_folds", type=int, nargs="*", default=None)
    p.add_argument("--fresh", action="store_true", help="move an incompatible fold dir aside and restart it")
    p.add_argument("--stop_after_epoch", type=int, default=None, help="TEST hook: exit after this epoch")
    # audit additions (defaults reproduce the original behaviour and the original config fingerprint)
    p.add_argument("--grad_clip", type=float, default=None, help="max gradient norm (default: off)")
    p.add_argument("--pad_side", choices=["right", "left"], default="right")
    p.add_argument("--pad_balance", action="store_true", help="equal padded fraction per class inside each fold")
    p.add_argument("--select", choices=["val_loss", "val_mcc"], default="val_loss",
                   help="checkpoint selection: lowest val loss (original) or highest val MCC (ties: lower val loss). "
                        "MCC is 0 for any constant predictor (all fall or all non-fall), F1 is not")
    p.add_argument("--no_test", action="store_true",
                   help="model-selection run: never touches the test fold; writes fold<k>/val_result.json only")
    a = p.parse_args(argv)
    if a.lr is None:
        a.lr = 1.0 if a.optimizer == "adadelta" else 1e-3
    return a


EXTRA_DEFAULTS = {"grad_clip": None, "pad_side": "right", "pad_balance": False, "select": "val_loss"}    # added after the baseline runs


def config_of(a):
    """Config used for fingerprints. New options enter only when they differ from their default, so
    every run made before they existed keeps exactly the same fingerprint."""
    c = {k: getattr(a, k) for k in CONFIG_KEYS}
    c.update({k: getattr(a, k) for k, d in EXTRA_DEFAULTS.items() if getattr(a, k) != d})
    return c


# ---------------------------------------------------------------------------------------------
def sha(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True).encode()).hexdigest()


def data_fingerprint(data_dir, names):
    h = hashlib.sha256()
    for n in sorted(names):
        for suf in ("", "_frame_labels"):
            p = os.path.join(data_dir, f"{n}{suf}.npy")
            if os.path.exists(p):
                h.update(n.encode() + suf.encode()); h.update(open(p, "rb").read())
    return h.hexdigest()


def atomic_torch_save(obj, path):
    buf = io.BytesIO(); torch.save(obj, buf)
    tmp = f"{path}.tmp-{os.getpid()}"
    with open(tmp, "wb") as f:
        f.write(buf.getvalue()); f.flush(); os.fsync(f.fileno())
    torch.load(tmp, map_location="cpu", weights_only=False)        # reload = validate before rename
    os.replace(tmp, path)
    fd = os.open(os.path.dirname(path), os.O_RDONLY); os.fsync(fd); os.close(fd)


def atomic_json(obj, path):
    tmp = f"{path}.tmp-{os.getpid()}"
    with open(tmp, "w") as f:
        json.dump(obj, f); f.flush(); os.fsync(f.fileno())
    json.load(open(tmp)); os.replace(tmp, path)


def rng_state():
    st = dict(python=random.getstate(), numpy=np.random.get_state(), torch=torch.get_rng_state())
    if torch.cuda.is_available():
        st["cuda"] = torch.cuda.get_rng_state_all()
    return st


def set_rng_state(st):
    random.setstate(st["python"]); np.random.set_state(st["numpy"]); torch.set_rng_state(st["torch"])
    if "cuda" in st and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(st["cuda"])


def pick_device(name):
    if name:
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def binary_metrics(y, p, thr):
    """Dataset-level metrics from one confusion matrix (positive class = fall)."""
    y = np.asarray(y).astype(int); yhat = (np.asarray(p) >= thr).astype(int)
    tp = int(((yhat == 1) & (y == 1)).sum()); tn = int(((yhat == 0) & (y == 0)).sum())
    fp = int(((yhat == 1) & (y == 0)).sum()); fn = int(((yhat == 0) & (y == 1)).sum())
    div = lambda a, b: float(a / b) if b else float("nan")
    prec, rec = div(tp, tp + fp), div(tp, tp + fn)
    return dict(n=len(y), tp=tp, tn=tn, fp=fp, fn=fn, accuracy=div(tp + tn, len(y)), precision=prec,
                recall_sensitivity=rec, specificity=div(tn, tn + fp),
                f1=div(2 * prec * rec, prec + rec) if (tp > 0) else 0.0,
                false_positive_rate=div(fp, fp + tn), false_negative_rate=div(fn, fn + tp),
                confusion_matrix=[[tn, fp], [fn, tp]])


def mcc(m):
    """Matthews correlation from a binary_metrics dict; 0 when undefined (constant predictions)."""
    tp, tn, fp, fn = (float(m[k]) for k in ("tp", "tn", "fp", "fn"))
    d = ((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)) ** 0.5
    return (tp * tn - fp * fn) / d if d > 0 else 0.0


@torch.no_grad()
def predict(model, ds, device, bs=256):
    model.eval()
    lg, ys, si, en = [], [], [], []
    for x, y, s, e in DataLoader(ds, batch_size=bs, shuffle=False):
        lg.append(model(x.to(device)).float().cpu()); ys.append(y); si.append(s); en.append(e)
    return torch.cat(lg).numpy(), torch.cat(ys).numpy(), torch.cat(si).numpy(), torch.cat(en).numpy()


def fit_temperature(logits, labels):
    lg = torch.tensor(logits, dtype=torch.float32); y = torch.tensor(labels, dtype=torch.float32)
    log_t = torch.zeros(1, requires_grad=True)
    opt = torch.optim.LBFGS([log_t], lr=0.1, max_iter=100); bce = nn.BCEWithLogitsLoss()

    def closure():
        opt.zero_grad(); loss = bce(lg / log_t.exp(), y); loss.backward(); return loss
    opt.step(closure)
    return float(log_t.detach().exp().clamp(0.05, 20.0))


# ---------------------------------------------------------------------------------------------
class FoldRunner:
    def __init__(self, k, fold, seqs_by_name, args, fps, device, log):
        self.k, self.fold, self.args, self.fps, self.device, self.log = k, fold, args, fps, device, log
        self.dir = os.path.join(args.out_dir, f"fold{k}")
        os.makedirs(self.dir, exist_ok=True)
        self.tr = [seqs_by_name[n] for n in fold["train"]]
        self.va = [seqs_by_name[n] for n in fold["val"]]
        self.te = [seqs_by_name[n] for n in fold["test"]]

    def new_model(self):
        torch.manual_seed(self.args.seed * 1000 + self.k)                 # identical init per fold
        return TwoStreamSTGCN(use_conf=self.args.use_conf).to(self.device)

    def make_opt(self, model):
        a = self.args
        if a.optimizer == "adadelta":
            return torch.optim.Adadelta(model.parameters(), lr=a.lr, weight_decay=a.weight_decay)
        return torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=a.weight_decay)

    def compatible(self, ck):
        return all(ck.get(key) == val for key, val in self.fps.items())

    def run(self):
        a, log = self.args, self.log
        res_path = os.path.join(self.dir, "val_result.json" if a.no_test else "result.json")
        if os.path.exists(res_path):
            res = json.load(open(res_path))
            if all(res.get(k) == v for k, v in self.fps.items()):
                log.info("fold %d: complete and compatible -> loaded %s", self.k, os.path.basename(res_path))
                return res
            if a.fresh:
                os.replace(self.dir, f"{self.dir}.old-{int(time.time())}"); os.makedirs(self.dir)
                log.warning("fold %d: incompatible result moved aside, starting fresh", self.k)
            else:
                raise SystemExit(f"fold {self.k}: result.json was made with other settings/data/split. "
                             f"Use another --out_dir (or --fresh to move it aside).")
        train_ds = TrainWindows(self.tr, a.window, a.windows_per_seq, a.min_pos, a.norm_mode, augment=True,
                                p_pad=a.p_pad, seed=a.seed * 1000 + self.k, pad_side=a.pad_side,
                                pad_balance=a.pad_balance)
        val_ds = EvalWindows(self.va, a.window, stride=a.val_stride, min_pos=a.min_pos, norm_mode=a.norm_mode,
                             pad_side=a.pad_side)
        model = self.new_model(); opt = self.make_opt(model)
        # pos_weight from training windows of epoch 0 (labels only; no augmentation randomness needed)
        ys = np.array([y for y, w in train_ds.labels() if w > 0])
        pos_weight = float((ys == 0).sum() / max(1, (ys == 1).sum()))
        start, step, best, history = 0, 0, float("inf"), []
        best_score = (-2.0, float("-inf"))                # --select val_mcc: (val MCC, -val loss)
        last = os.path.join(self.dir, "last_checkpoint.pt"); bestp = os.path.join(self.dir, "best_checkpoint.pt")
        if os.path.exists(last):
            ck = torch.load(last, map_location="cpu", weights_only=False)
            if not self.compatible(ck):
                if not a.fresh:
                    raise SystemExit(f"fold {self.k}: last_checkpoint.pt is from different settings/data/split; "
                                     f"refusing to resume or overwrite. Use --fresh to move it aside.")
                os.replace(self.dir, f"{self.dir}.old-{int(time.time())}"); os.makedirs(self.dir)
                log.warning("fold %d: incompatible checkpoint moved aside, starting fresh", self.k)
            else:
                model.load_state_dict(ck["model"]); opt.load_state_dict(ck["optimizer"])
                start, step, best, history = ck["epoch"] + 1, ck["step"], ck["best_val_loss"], ck["history"]
                best_score = tuple(ck.get("best_score", best_score))
                pos_weight = ck["pos_weight"]; set_rng_state(ck["rng"])
                log.info("fold %d: RESUMED after epoch %d (best val loss %.4f)", self.k, ck["epoch"], best)
        crit = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight], device=self.device), reduction="none")
        log.info("fold %d: train %d seq (%d windows/epoch), val %d, test %d | pos_weight %.3f | padded windows "
                 "fall %.0f%% / non-fall %.0f%%", self.k, len(self.tr), len(train_ds), len(self.va), len(self.te),
                 pos_weight, *(100 * v for v in (train_ds.padded_fraction()[1], train_ds.padded_fraction()[0])))

        for epoch in range(start, a.epochs):
            torch.manual_seed(a.seed * 100003 + self.k * 1009 + epoch)       # dropout etc.: per-epoch seed
            train_ds.set_epoch(epoch)
            g = torch.Generator().manual_seed(a.seed * 7919 + self.k * 31 + epoch)
            model.train(); tot = wsum = 0.0
            for x, y, w in DataLoader(train_ds, batch_size=a.batch_size, shuffle=True, generator=g):
                x, y, w = x.to(self.device), y.to(self.device), w.to(self.device)
                loss_vec = crit(model(x), y)
                loss = (loss_vec * w).sum() / w.sum().clamp(min=1.0)
                opt.zero_grad(set_to_none=True); loss.backward()
                if a.grad_clip:
                    nn.utils.clip_grad_norm_(model.parameters(), a.grad_clip)
                opt.step(); step += 1
                tot += float((loss_vec * w).sum()); wsum += float(w.sum())
            lg, yv, _, _ = predict(model, val_ds, self.device)
            vloss = float(nn.functional.binary_cross_entropy_with_logits(torch.tensor(lg), torch.tensor(yv)))
            vm = binary_metrics(yv, 1 / (1 + np.exp(-np.clip(lg, -50, 50))), a.threshold)
            if a.select == "val_mcc":
                score = (round(mcc(vm), 6), -vloss)
                improved = score > best_score
                if improved:
                    best_score = score
            else:
                improved = vloss < best - 1e-6
            history.append(dict(epoch=epoch, step=step, train_loss=tot / max(wsum, 1), val_loss=vloss,
                                val_f1=vm["f1"], val_recall=vm["recall_sensitivity"], best=improved))
            if a.select == "val_mcc":
                history[-1]["val_mcc"] = mcc(vm)
            state = dict(model=model.state_dict(), optimizer=opt.state_dict(), epoch=epoch, step=step,
                         best_val_loss=min(best, vloss), best_score=list(best_score), history=history,
                         pos_weight=pos_weight, rng=rng_state(),
                         label_map=LABEL_MAP, config=config_of(a), **self.fps)
            if improved:
                best = min(best, vloss) if a.select == "val_mcc" else vloss
                atomic_torch_save(dict(state, best_epoch=epoch), bestp)
            atomic_torch_save(state, last)
            with open(os.path.join(self.dir, "training_history.csv"), "w", newline="") as f:
                wr = csv.DictWriter(f, fieldnames=list(history[0])); wr.writeheader(); wr.writerows(history)
            log.info("fold %d epoch %2d | train %.4f | val loss %.4f F1 %.3f R %.3f%s", self.k, epoch,
                     history[-1]["train_loss"], vloss, vm["f1"], vm["recall_sensitivity"], " *best" if improved else "")
            if a.stop_after_epoch is not None and epoch >= a.stop_after_epoch:
                log.warning("TEST: stopping after epoch %d", epoch); sys.exit(95)

        # ---- finalize: best weights, temperature on val, one evaluation on the test fold ----
        ck = torch.load(bestp, map_location="cpu", weights_only=False)
        model.load_state_dict(ck["model"])
        lg, yv, _, _ = predict(model, val_ds, self.device)
        temp = fit_temperature(lg, yv)
        if a.no_test:
            # model-selection run: evaluate every validation frame (stride 1); the test fold is never loaded
            vfull = EvalWindows(self.va, a.window, stride=1, min_pos=a.min_pos, norm_mode=a.norm_mode,
                                pad_side=a.pad_side)
            lg, yv, si, en = predict(model, vfull, self.device)
            p = torch.sigmoid(torch.tensor(lg / temp)).numpy()
            frames = [dict(sequence=self.va[s]["name"], frame=int(e) + 1, fold=self.k, label=int(yy), prob=float(pp))
                      for s, e, yy, pp in zip(si, en, yv, p)]
            res = dict(best_epoch=ck["best_epoch"], temperature=temp, frames=frames, test_evaluated=False,
                       frame_metrics=binary_metrics(yv, p, a.threshold), **self.fps)
            atomic_json(res, res_path)
            fm = res["frame_metrics"]
            log.info("fold %d VALIDATION (frame level, no test): P %.3f R %.3f spec %.3f F1 %.3f | best epoch %d",
                     self.k, fm["precision"], fm["recall_sensitivity"], fm["specificity"], fm["f1"], ck["best_epoch"])
            return res
        test_ds = EvalWindows(self.te, a.window, stride=1, min_pos=a.min_pos, norm_mode=a.norm_mode,
                              pad_side=a.pad_side)
        lg, yt, si, en = predict(model, test_ds, self.device)
        p = torch.sigmoid(torch.tensor(lg / temp)).numpy()
        frames = [dict(sequence=self.te[s]["name"], frame=int(e) + 1, fold=self.k, label=int(yy), prob=float(pp))
                  for s, e, yy, pp in zip(si, en, yt, p)]
        seqrows = []
        for j, s in enumerate(self.te):
            m = si == j
            seqrows.append(dict(sequence=s["name"], fold=self.k, label=s["label"], prob=float(p[m].max())))
        res = dict(best_epoch=ck["best_epoch"], temperature=temp, frames=frames, sequences=seqrows,
                   frame_metrics=binary_metrics(yt, p, a.threshold),
                   sequence_metrics=binary_metrics([r["label"] for r in seqrows], [r["prob"] for r in seqrows],
                                                   a.threshold), **self.fps)
        atomic_json(res, res_path)
        fm = res["frame_metrics"]
        log.info("fold %d TEST (frame level): acc %.3f P %.3f R %.3f spec %.3f F1 %.3f | best epoch %d, T=%.2f",
                 self.k, fm["accuracy"], fm["precision"], fm["recall_sensitivity"], fm["specificity"], fm["f1"],
                 ck["best_epoch"], temp)
        return res


def main(argv=None):
    a = get_args(argv)
    os.makedirs(a.out_dir, exist_ok=True)
    log = logging.getLogger(f"train-{a.out_dir}"); log.setLevel(logging.INFO); log.handlers.clear()
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    for h in (logging.FileHandler(os.path.join(a.out_dir, "training.log")), logging.StreamHandler(sys.stdout)):
        h.setFormatter(fmt); log.addHandler(h)
    split = json.load(open(a.split_file))
    names = sorted({n for f in split["folds"] for n in f["train"] + f["val"] + f["test"]})
    seqs = {s["name"]: s for s in load_sequences(a.data_dir, names=set(names))}
    missing = [n for n in names if n not in seqs]
    if missing:
        raise SystemExit(f"split mentions sequences not in data_dir: {missing}")
    fps = dict(config_fp=sha(config_of(a)), data_fp=data_fingerprint(a.data_dir, names),
               split_fp=split["fingerprint"])
    device = pick_device(a.device)
    log.info("=== train | device %s | split %s (%s) | config %s | data %s", device, split["mode"],
             split["fingerprint"][:12], fps["config_fp"][:12], fps["data_fp"][:12])
    results = []
    for k, fold in enumerate(split["folds"]):
        if a.only_folds is not None and k not in a.only_folds:
            continue
        results.append(FoldRunner(k, fold, seqs, a, fps, device, log).run())
    if a.no_test or (a.only_folds is not None and len(results) < len(split["folds"])):
        return 0
    frames = [r for res in results for r in res["frames"]]
    seqrows = [r for res in results for r in res["sequences"]]
    for fn, rows in (("oof_frame_probs.csv", frames), ("oof_sequence_probs.csv", seqrows)):
        with open(os.path.join(a.out_dir, fn), "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    summary = dict(split_mode=split["mode"], **fps,
                   pooled_frame=binary_metrics([r["label"] for r in frames], [r["prob"] for r in frames], a.threshold),
                   pooled_sequence=binary_metrics([r["label"] for r in seqrows], [r["prob"] for r in seqrows], a.threshold),
                   per_fold=[dict(fold=i, best_epoch=r["best_epoch"], temperature=r["temperature"],
                                  frame=r["frame_metrics"], sequence=r["sequence_metrics"]) for i, r in enumerate(results)])
    atomic_json(summary, os.path.join(a.out_dir, "summary.json"))
    pf = summary["pooled_frame"]
    log.info("=== POOLED out-of-fold TEST (frame level, %s split): acc %.3f P %.3f R(sens) %.3f spec %.3f F1 %.3f "
             "FPR %.3f FNR %.3f CM %s", split["mode"], pf["accuracy"], pf["precision"], pf["recall_sensitivity"],
             pf["specificity"], pf["f1"], pf["false_positive_rate"], pf["false_negative_rate"], pf["confusion_matrix"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
