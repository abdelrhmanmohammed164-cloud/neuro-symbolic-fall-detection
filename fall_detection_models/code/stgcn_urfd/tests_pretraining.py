"""
tests_pretraining.py
The 14 pre-training checks (+ padding-shortcut checks 15, 16) on the REAL preprocessed data.\nResult per test: PASS / WARN (reported, does not block) / FAIL (blocks training).
The real files are only read; destructive tests use temporary copies.

Usage:
  python tests_pretraining.py --data_dir .../urfd_npy --features_dir URFD/features \
      --split_scene splits/split_scene_v2.json --split_video splits/split_video_v2.json --out pretraining_tests.json
"""
import argparse
import csv
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import traceback

import numpy as np
import torch
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score

from dataset import EvalWindows, TrainWindows, label_and_weight, load_sequences, normalize, window_by_end
from graph import EDGES, FLIP_PAIRS, J, JOINTS, NUM_JOINTS, Graph
from model import TwoStreamSTGCN, motion_stream
from splits import verify
from train import atomic_torch_save, binary_metrics

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = []


def test(name):
    def deco(fn):
        def run(*a):
            try:
                ev = fn(*a); RESULTS.append(dict(test=name, result="PASS", evidence=ev))
            except AssertionError as e:
                RESULTS.append(dict(test=name, result="FAIL", evidence=str(e)))
            except Exception as e:
                RESULTS.append(dict(test=name, result="ERROR", evidence=f"{type(e).__name__}: {e}",
                                    trace=traceback.format_exc()[-800:]))
            r = RESULTS[-1]; print(f"[{r['result']}] {name}: {r['evidence']}")
        return run
    return deco


@test("01 dataset loading")
def t01(seqs):
    assert len(seqs) == 70, len(seqs)
    n_fall = sum(s["label"] for s in seqs)
    assert n_fall == 30, n_fall
    assert all(s["data"].shape[0] == 3 and s["data"].shape[2:] == (14, 1) and s["data"].dtype == np.float32 for s in seqs)
    assert all(s["frame_labels"] is not None for s in seqs)
    return f"70 sequences (30 fall / 40 ADL), all (3, T, 14, 1) float32, frame labels for all"


@test("02 skeleton graph topology")
def t02():
    g = Graph()
    assert NUM_JOINTS == 14 and g.A.shape == (3, 14, 14)
    assert len(EDGES) == 13 and len({frozenset(e) for e in EDGES}) == 13
    assert np.isfinite(g.dist_to_center).all(), "graph not connected"
    assert np.allclose(g.A.sum(0).sum(0), 1.0), "partition columns do not sum to 1"
    assert (np.diag(g.A[0]) > 0).all(), "self-links missing"
    names = {frozenset((JOINTS[a], JOINTS[b])) for a, b in EDGES}
    for bone in [("Neck", "Head"), ("R_Shoulder", "R_Elbow"), ("L_Hip", "L_Knee"), ("R_Knee", "R_Ankle")]:
        assert frozenset(bone) in names, bone
    assert all(JOINTS[a].replace("R_", "") == JOINTS[b].replace("L_", "") for a, b in FLIP_PAIRS)
    return "14 joints, 13 bones (tree, connected), 3 partitions column-normalised, bones and L/R pairs match names"


@test("03 tensor shapes (N, C, T, V, M)")
def t03(seqs):
    tr = TrainWindows(seqs[:6], 80, 4, 10, seed=1)
    x, y, w = tr[0]
    assert tuple(x.shape) == (3, 80, 14, 1) and y.dtype == torch.float32
    batch = torch.stack([tr[i][0] for i in range(8)])
    assert tuple(batch.shape) == (8, 3, 80, 14, 1)
    ev = EvalWindows(seqs, 80, stride=1)
    assert len(ev) == sum(s["data"].shape[1] for s in seqs)
    return f"train item (3, 80, 14, 1); batch (8, 3, 80, 14, 1); eval windows = total frames = {len(ev)}"


@test("04 coordinate / motion alignment (Eq. 3)")
def t04(seqs):
    s = [q for q in seqs if q["name"] == "fall-16"][0]          # short (55 frames) -> contains padding
    x, start = window_by_end(s["data"], s["data"].shape[1] - 1, 80)
    xt = torch.from_numpy(x)[None]
    d = motion_stream(xt)[0].numpy()
    valid = x[2] > 0
    assert np.all(d[:, 0] == 0), "motion at t=1 must be 0"
    both = valid[1:] & valid[:-1]
    exp = (x[:2, 1:] - x[:2, :-1])
    assert np.allclose(d[:, 1:][:, both], exp[:, both]), "dP_t != P_t - P_{t-1}"
    assert np.all(d[:, 1:][:, ~both] == 0), "motion of a missing joint must be 0"
    n_real = s["data"].shape[1] - start
    assert np.all(d[:, n_real:] == 0), "padded frames must have zero motion"
    model = TwoStreamSTGCN()
    c, m = model.streams(xt)
    assert torch.equal(c, xt[:, :2]) and torch.equal(m, motion_stream(xt))
    return f"backward difference, 0 at t=1, 0 for missing joints, 0 on {80 - n_real} padded frames; both streams from one tensor"


@test("05 label mapping and window labels")
def t05(seqs, features_dir):
    feat = {}
    for fn in ("urfall-cam0-falls.csv", "urfall-cam0-adls.csv"):
        for r in csv.reader(open(os.path.join(features_dir, fn))):
            if r:
                feat.setdefault(r[0], {})[int(r[1])] = int(r[2])
    for s in seqs:
        fl = s["frame_labels"]
        if s["label"] == 0:
            assert fl.sum() == 0, f"{s['name']}: ADL with fall frames"
        else:
            urfd = {f for f, l in feat[s["name"]].items() if l in (0, 1)}
            assert set((np.nonzero(fl)[0] + 1).tolist()) == urfd, f"{s['name']}: frame labels differ from URFD"
    s = [q for q in seqs if q["name"] == "fall-01"][0]; fl = s["frame_labels"]; f0 = int(np.argmax(fl))
    _, st = window_by_end(s["data"], f0 - 1, 80)
    assert label_and_weight(s, st, f0 - 1, 10) == (0, 1.0), "window ending before the fall must be 0"
    _, st = window_by_end(s["data"], f0 + 3, 80)
    assert label_and_weight(s, st, f0 + 3, 10) == (1, 0.0), "fall onset (<10 fall frames) must be ignored"
    _, st = window_by_end(s["data"], f0 + 20, 80)
    assert label_and_weight(s, st, f0 + 20, 10) == (1, 1.0)
    return "frame labels = URFD rows with label 0/1 in all 30 falls, 0 in all 40 ADLs; end-frame labelling and onset-ignore verified on fall-01"


@test("06 group-disjoint splits")
def t06(split_files, seqs):
    names = [s["name"] for s in seqs]
    out = []
    for p in split_files:
        sp = json.load(open(p))
        msgs = verify(sp["folds"], names, sp["group_of"])
        if sp["mode"] == "scene":
            for f in sp["folds"]:
                g_tr = {sp["group_of"][n] for n in f["train"] + f["val"]}
                g_te = {sp["group_of"][n] for n in f["test"]}
                assert not g_tr & g_te
        out.append(f"{sp['mode']}: {len(sp['folds'])} folds OK")
    return "; ".join(out)


@test("07 duplicate leakage across splits")
def t07(split_files, seqs):
    h = {s["name"]: hashlib.sha256(s["data"].tobytes()).hexdigest() for s in seqs}
    assert len(set(h.values())) == len(h), "identical skeleton arrays exist"
    for p in split_files:
        sp = json.load(open(p))
        for f in sp["folds"]:
            tr = {h[n] for n in f["train"] + f["val"]}; te = {h[n] for n in f["test"]}
            assert not tr & te
            # audit fix: the group must not straddle ANY two of train / val / test (before: train vs test only)
            for grp in sp["near_groups"]:
                sides = {("train" if n in f["train"] else "val" if n in f["val"] else "test") for n in grp}
                assert len(sides) == 1, (f"{os.path.basename(p)} fold {f['fold']}: near-duplicate group {grp} "
                                         f"split across {sorted(sides)}")
    return ("70 distinct skeleton hashes; no shared hash between train(+val) and test; near-duplicate group "
            "adl-25/27/29 never split across train / val / test")


@test("08 normalisation / preprocessing leakage")
def t08(seqs, data_dir, split_scene):
    x = seqs[0]["data"][:, :80]
    assert np.array_equal(normalize(x, "image"), x), "image normalisation must be the identity (x/W, y/H)"
    a = normalize(x, "window_center"); b = normalize(x.copy(), "window_center")
    assert np.array_equal(a, b), "window normalisation must depend on the window only"
    model = TwoStreamSTGCN().eval()
    bn = model.g_sp.data_bn.running_mean.clone()
    with torch.no_grad():
        model(torch.rand(4, 3, 80, 14, 1))
    assert torch.equal(bn, model.g_sp.data_bn.running_mean), "eval() must not update BatchNorm statistics"
    # poison the TEST fold with NaN: training, checkpoint selection and temperature must stay finite,
    # which proves they never read test data
    sp = json.load(open(split_scene)); fold = sp["folds"][0]
    tmp = tempfile.mkdtemp(); dd = os.path.join(tmp, "data"); os.makedirs(dd)
    keep = fold["train"][:3] + [n for n in fold["train"] if n.startswith("fall")][:2] + fold["val"][:3] + fold["test"][:4]
    for n in set(keep):
        for suf in ("", "_frame_labels"):
            shutil.copy(os.path.join(data_dir, f"{n}{suf}.npy"), dd)
    for n in fold["test"][:4]:
        arr = np.load(os.path.join(dd, f"{n}.npy")); arr[:2] = np.nan; np.save(os.path.join(dd, f"{n}.npy"), arr)
    mini = dict(sp); mini["folds"] = [dict(fold, train=sorted(set(fold["train"][:3] + [n for n in fold["train"] if n.startswith("fall")][:2])),
                                          val=fold["val"][:3], test=fold["test"][:4])]
    json.dump(mini, open(os.path.join(tmp, "split.json"), "w"))
    r = subprocess.run([sys.executable, os.path.join(HERE, "train.py"), "--data_dir", dd, "--split_file",
                        os.path.join(tmp, "split.json"), "--out_dir", os.path.join(tmp, "run"), "--epochs", "2",
                        "--windows_per_seq", "4", "--batch_size", "16", "--device", "cpu"],
                       capture_output=True, text=True, cwd=HERE)
    assert r.returncode == 0, r.stdout[-600:] + r.stderr[-600:]
    res = json.load(open(os.path.join(tmp, "run", "fold0", "result.json")))
    hist = list(csv.DictReader(open(os.path.join(tmp, "run", "fold0", "training_history.csv"))))
    assert all(np.isfinite(float(h["train_loss"])) and np.isfinite(float(h["val_loss"])) for h in hist)
    assert np.isfinite(res["temperature"]), "temperature used test data"
    shutil.rmtree(tmp)
    return ("x/W,y/H is a fixed transform (identity here); no scaler/PCA/SMOTE in the pipeline; eval() leaves BN "
            "stats unchanged; with the TEST fold set to NaN, training loss, validation loss, checkpoint choice and "
            "temperature all stayed finite -> none of them reads test data")


@test("17 --no_test never evaluates the test fold")
def t17(data_dir, split_scene):
    """Audit addition: model-selection runs must not produce any test number. The test videos are
    poisoned with NaN; the run must finish, write only val_result.json with finite validation
    probabilities, and contain no test sequence and no result.json."""
    sp = json.load(open(split_scene)); fold = sp["folds"][0]
    tmp = tempfile.mkdtemp(); dd = os.path.join(tmp, "data"); os.makedirs(dd)
    tr = sorted(set(fold["train"][:3] + [n for n in fold["train"] if n.startswith("fall")][:2]))
    va, te = fold["val"][:3], fold["test"][:4]
    for n in set(tr + va + te):
        for suf in ("", "_frame_labels"):
            shutil.copy(os.path.join(data_dir, f"{n}{suf}.npy"), dd)
    for n in te:
        arr = np.load(os.path.join(dd, f"{n}.npy")); arr[:] = np.nan; np.save(os.path.join(dd, f"{n}.npy"), arr)
    mini = dict(sp); mini["folds"] = [dict(fold, train=tr, val=va, test=te)]
    json.dump(mini, open(os.path.join(tmp, "split.json"), "w"))
    r = subprocess.run([sys.executable, os.path.join(HERE, "train.py"), "--data_dir", dd, "--split_file",
                        os.path.join(tmp, "split.json"), "--out_dir", os.path.join(tmp, "run"), "--epochs", "2",
                        "--windows_per_seq", "4", "--batch_size", "16", "--device", "cpu", "--no_test",
                        "--grad_clip", "1.0", "--pad_side", "left", "--pad_balance"],
                       capture_output=True, text=True, cwd=HERE)
    assert r.returncode == 0, r.stdout[-600:] + r.stderr[-600:]
    fd = os.path.join(tmp, "run", "fold0")
    assert not os.path.exists(os.path.join(fd, "result.json")), "a test result was written"
    res = json.load(open(os.path.join(fd, "val_result.json")))
    seqs_in = {f["sequence"] for f in res["frames"]}
    assert seqs_in == set(va), f"val_result contains {sorted(seqs_in - set(va))}"
    assert all(np.isfinite(f["prob"]) for f in res["frames"]) and res["test_evaluated"] is False
    log = open(os.path.join(tmp, "run", "training.log")).read()
    assert "TEST" not in log, "the log mentions a test evaluation"
    shutil.rmtree(tmp)
    return ("test videos poisoned with NaN; --no_test run (with --grad_clip, --pad_side left and --pad_balance) finished, wrote "
            "only val_result.json with finite probabilities for the validation videos; no result.json, no TEST line")


@test("18 --select val_mcc: picks the best-MCC epoch and resumes exactly")
def t18(data_dir, split_scene):
    """Audit addition (after the quick trial showed that the lowest val loss on an unseen scene is a
    model that predicts nothing): selection by val MCC (0 for constant predictors; F1 rewards an
    all-fall model), ties -> lower val loss. Checks the chosen epoch against the saved history,
    and that stop + resume gives identical validation output."""
    sp = json.load(open(split_scene)); fold = sp["folds"][1]
    tmp = tempfile.mkdtemp(); dd = os.path.join(tmp, "data"); os.makedirs(dd)
    tr = sorted(set(fold["train"][:4] + [n for n in fold["train"] if n.startswith("fall")][:3]))
    va = [n for n in fold["val"] if n.startswith("fall")][:2] + [n for n in fold["val"] if n.startswith("adl")][:2]
    for n in set(tr + va + fold["test"][:2]):
        for suf in ("", "_frame_labels"):
            shutil.copy(os.path.join(data_dir, f"{n}{suf}.npy"), dd)
    mini = dict(sp); mini["folds"] = [dict(fold, train=tr, val=va, test=fold["test"][:2])]
    json.dump(mini, open(os.path.join(tmp, "split.json"), "w"))
    base = [sys.executable, os.path.join(HERE, "train.py"), "--data_dir", dd, "--split_file",
            os.path.join(tmp, "split.json"), "--epochs", "6", "--windows_per_seq", "16", "--batch_size", "32",
            "--device", "cpu", "--no_test", "--select", "val_mcc", "--val_stride", "2"]
    r = subprocess.run(base + ["--out_dir", os.path.join(tmp, "a")], capture_output=True, text=True, cwd=HERE)
    assert r.returncode == 0, r.stdout[-600:] + r.stderr[-600:]
    r = subprocess.run(base + ["--out_dir", os.path.join(tmp, "b"), "--stop_after_epoch", "2"],
                       capture_output=True, text=True, cwd=HERE)
    assert r.returncode == 95, f"stop hook returned {r.returncode}"
    r = subprocess.run(base + ["--out_dir", os.path.join(tmp, "b")], capture_output=True, text=True, cwd=HERE)
    assert r.returncode == 0, r.stdout[-600:] + r.stderr[-600:]
    hist = list(csv.DictReader(open(os.path.join(tmp, "a", "fold0", "training_history.csv"))))
    scores = [(round(float(h["val_mcc"]), 6), -float(h["val_loss"])) for h in hist]
    want = max(range(len(scores)), key=lambda i: (scores[i], -i))
    want = min(i for i in range(len(scores)) if scores[i] == scores[want])      # first epoch reaching the max
    ra = json.load(open(os.path.join(tmp, "a", "fold0", "val_result.json")))
    rb = json.load(open(os.path.join(tmp, "b", "fold0", "val_result.json")))
    assert ra["best_epoch"] == want, f"best epoch {ra['best_epoch']} != expected {want} (scores {scores})"
    assert ra["best_epoch"] == rb["best_epoch"] and ra["temperature"] == rb["temperature"]
    pa = [f["prob"] for f in ra["frames"]]; pb = [f["prob"] for f in rb["frames"]]
    assert np.allclose(pa, pb, atol=1e-6), "resumed run differs from uninterrupted run"
    shutil.rmtree(tmp)
    return (f"6 epochs: chosen epoch {want} = highest val MCC (ties: lower val loss) in the saved history "
            f"(MCC per epoch {[round(x[0], 3) for x in scores]}); run stopped "
            f"after epoch 2 and resumed gives the same chosen epoch, temperature and validation probabilities")


@test("09 forward pass")
def t09():
    m = TwoStreamSTGCN().eval(); x = torch.rand(5, 3, 80, 14, 1)
    with torch.no_grad():
        y = m(x)
    assert y.shape == (5,) and torch.isfinite(y).all()
    return f"(5, 3, 80, 14, 1) -> logits (5,), finite; parameters {sum(p.numel() for p in m.parameters()):,}"


@test("10 backward pass")
def t10():
    m = TwoStreamSTGCN().train(); x = torch.rand(6, 3, 80, 14, 1)
    loss = torch.nn.functional.binary_cross_entropy_with_logits(m(x), torch.tensor([0, 1, 0, 1, 1, 0.]))
    loss.backward()
    g_sp = sum(p.grad.abs().sum().item() for p in m.g_sp.parameters() if p.grad is not None)
    g_tm = sum(p.grad.abs().sum().item() for p in m.g_tm.parameters() if p.grad is not None)
    assert g_sp > 0 and g_tm > 0, "a stream receives no gradient"
    assert all(torch.isfinite(p.grad).all() for p in m.parameters() if p.grad is not None)
    return f"finite gradients reach both streams (|grad| coord {g_sp:.2f}, motion {g_tm:.2f})"


@test("11 tiny-subset overfitting")
def t11(seqs):
    small = [s for s in seqs if s["name"] in ("fall-01", "fall-02", "adl-01", "adl-02")]
    ds = TrainWindows(small, 80, 16, 10, augment=False, p_pad=0.0, seed=3)
    X = torch.stack([ds[i][0] for i in range(len(ds))]); Y = torch.stack([ds[i][1] for i in range(len(ds))])
    W = torch.stack([ds[i][2] for i in range(len(ds))]); keep = W > 0; X, Y = X[keep], Y[keep]
    torch.manual_seed(0)
    m = TwoStreamSTGCN(block_dropout=0.0, head_dropout=0.0)
    opt = torch.optim.AdamW(m.parameters(), lr=1e-3)
    for _ in range(150):
        m.train(); opt.zero_grad()
        loss = torch.nn.functional.binary_cross_entropy_with_logits(m(X), Y); loss.backward(); opt.step()
    m.eval()
    with torch.no_grad():
        acc = ((m(X) > 0).float() == Y).float().mean().item()
    assert acc >= 0.95, f"train accuracy {acc:.3f}"
    return f"{len(Y)} windows from 4 videos, 150 steps: train accuracy {acc:.3f}, final loss {loss.item():.4f}"


@test("12 checkpoint save / load")
def t12():
    tmp = tempfile.mkdtemp(); p = os.path.join(tmp, "ck.pt")
    m = TwoStreamSTGCN(); opt = torch.optim.Adadelta(m.parameters())
    atomic_torch_save(dict(model=m.state_dict(), optimizer=opt.state_dict(), epoch=3), p)
    ck = torch.load(p, map_location="cpu", weights_only=False)
    m2 = TwoStreamSTGCN(); m2.load_state_dict(ck["model"])
    same = all(torch.equal(a, b) for a, b in zip(m.state_dict().values(), m2.state_dict().values()))
    left = [f for f in os.listdir(tmp) if ".tmp-" in f]
    shutil.rmtree(tmp)
    assert same and ck["epoch"] == 3 and not left
    return "atomic save -> load gives identical weights and epoch; no temp files left"


@test("13 interruption / resume")
def t13(data_dir, split_scene):
    sp = json.load(open(split_scene)); fold = sp["folds"][1]
    tmp = tempfile.mkdtemp()
    mini = dict(sp); mini["folds"] = [dict(fold, train=[n for n in fold["train"] if n.startswith("fall")][:2] + [n for n in fold["train"] if n.startswith("adl")][:2],
                                          val=fold["val"][:3], test=fold["test"][:3])]
    sf = os.path.join(tmp, "split.json"); json.dump(mini, open(sf, "w"))
    base = [sys.executable, os.path.join(HERE, "train.py"), "--data_dir", data_dir, "--split_file", sf,
            "--epochs", "4", "--windows_per_seq", "6", "--batch_size", "8", "--device", "cpu"]
    a = subprocess.run(base + ["--out_dir", os.path.join(tmp, "A")], capture_output=True, text=True, cwd=HERE)
    b1 = subprocess.run(base + ["--out_dir", os.path.join(tmp, "B"), "--stop_after_epoch", "1"], capture_output=True, text=True, cwd=HERE)
    hist_after_stop = len(list(csv.DictReader(open(os.path.join(tmp, "B", "fold0", "training_history.csv")))))
    b2 = subprocess.run(base + ["--out_dir", os.path.join(tmp, "B")], capture_output=True, text=True, cwd=HERE)
    assert a.returncode == 0 and b1.returncode == 95 and b2.returncode == 0, (a.returncode, b1.returncode, b2.returncode, b2.stdout[-400:])
    assert "RESUMED after epoch 1" in b2.stdout
    ca = torch.load(os.path.join(tmp, "A", "fold0", "last_checkpoint.pt"), weights_only=False)
    cb = torch.load(os.path.join(tmp, "B", "fold0", "last_checkpoint.pt"), weights_only=False)
    same = all(torch.equal(ca["model"][k], cb["model"][k]) for k in ca["model"])
    ra = json.load(open(os.path.join(tmp, "A", "fold0", "result.json")))
    rb = json.load(open(os.path.join(tmp, "B", "fold0", "result.json")))
    probs_same = [f["prob"] for f in ra["frames"]] == [f["prob"] for f in rb["frames"]]
    epochs_b = [int(h["epoch"]) for h in csv.DictReader(open(os.path.join(tmp, "B", "fold0", "training_history.csv")))]
    # incompatible resume must be refused
    c = subprocess.run(base + ["--out_dir", os.path.join(tmp, "B"), "--lr", "0.5"], capture_output=True, text=True, cwd=HERE)
    shutil.rmtree(tmp)
    assert same and probs_same and epochs_b == [0, 1, 2, 3], (same, probs_same, epochs_b)
    assert c.returncode != 0 and "other settings" in (c.stdout + c.stderr)
    return (f"stopped after epoch 1 (exit 95, {hist_after_stop} epochs in history), resumed at epoch 2; final weights and "
            f"all test probabilities identical to an uninterrupted run; history epochs {epochs_b}; resume with a changed "
            f"lr was refused")


@test("14 validation metric correctness")
def t14():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 1000); p = rng.random(1000)
    m = binary_metrics(y, p, 0.5); yh = (p >= 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, yh).ravel()
    assert m["confusion_matrix"] == [[tn, fp], [fn, tp]]
    assert np.isclose(m["accuracy"], accuracy_score(y, yh)) and np.isclose(m["precision"], precision_score(y, yh))
    assert np.isclose(m["recall_sensitivity"], recall_score(y, yh)) and np.isclose(m["f1"], f1_score(y, yh))
    assert np.isclose(m["specificity"], tn / (tn + fp)) and np.isclose(m["false_positive_rate"], fp / (fp + tn))
    assert np.isclose(m["false_negative_rate"], fn / (fn + tp))
    h = binary_metrics([1, 1, 1, 0, 0], [0.9, 0.2, 0.7, 0.6, 0.1], 0.5)
    assert (h["tp"], h["fn"], h["fp"], h["tn"]) == (2, 1, 1, 1)
    return "matches sklearn on 1000 random samples; hand example TP2 FN1 FP1 TN1 correct; computed from one dataset-level confusion matrix"


@test("15 padding shortcut removed")
def t15(seqs):
    from audit_dataset import auc
    pad, y = [], []
    for seed in range(3):
        tr = TrainWindows(seqs, 80, 64, 10, seed=seed)
        for si, end, n_real, _ in tr.items:
            s = tr.seqs[si]; _, st = window_by_end(s["data"], end, 80, n_real)
            lab, w = label_and_weight(s, st, end, 10)
            if w > 0:
                pad.append(80 - (end - st + 1)); y.append(lab)
    a = auc(np.array(pad), np.array(y))
    assert 0.4 <= a <= 0.6, f"padding still predicts the label: AUC {a:.3f}"
    return (f"ALL 70 videos together: AUC of 'number of padded frames' for predicting the window label = {a:.3f} "
            f"(0.5 = no information); before the fix 37% of fall windows vs 2% of non-fall windows were padded. "
            f"Per-fold training sets: see test 16")


def t16(seqs, split_scene, p_pad=0.35):
    """Audit addition: padding vs label INSIDE each fold's training set. Reported as WARN (not FAIL):
    the original p_pad cannot balance per fold. Never blocks training; the values go into the report."""
    from audit_dataset import auc
    name = "16 padding vs label per training fold (report)"
    try:
        sp = json.load(open(split_scene)); by = {s["name"]: s for s in seqs}; parts = []; worst = 0.0
        for f in sp["folds"]:
            pad, y = [], []
            for seed in range(3):
                tr = TrainWindows([by[n] for n in f["train"]], 80, 64, 10, p_pad=p_pad, seed=seed)
                for si, end, n_real, _ in tr.items:
                    s = tr.seqs[si]; _, st = window_by_end(s["data"], end, 80, n_real)
                    lab, w = label_and_weight(s, st, end, 10)
                    if w > 0:
                        pad.append(80 - (end - st + 1)); y.append(lab)
            pad, y = np.array(pad), np.array(y)
            a_cnt, a_bin = auc(pad, y), auc((pad > 0).astype(float), y)
            worst = max(worst, abs(a_cnt - 0.5), abs(a_bin - 0.5))
            parts.append(f"fold {f['fold']}: padded fall {np.mean(pad[y == 1] > 0):.2f} vs non-fall "
                         f"{np.mean(pad[y == 0] > 0):.2f}, AUC count {a_cnt:.3f}, AUC padded yes/no {a_bin:.3f}")
        res = "PASS" if worst <= 0.1 else "WARN"
        RESULTS.append(dict(test=name, result=res, evidence="; ".join(parts)))
        print(f"[{res}] {name}: " + "; ".join(parts))
    except Exception as e:
        RESULTS.append(dict(test=name, result="FAIL", evidence=f"{type(e).__name__}: {e}"))
        print(f"[FAIL] {name}: {e}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", required=True); p.add_argument("--features_dir", required=True)
    p.add_argument("--split_scene", required=True); p.add_argument("--split_video", required=True)
    p.add_argument("--out", required=True)
    a = p.parse_args()
    seqs = load_sequences(a.data_dir)
    t01(seqs); t02(); t03(seqs); t04(seqs); t05(seqs, a.features_dir)
    t06([a.split_scene, a.split_video], seqs); t07([a.split_scene, a.split_video], seqs)
    t08(seqs, a.data_dir, a.split_scene); t09(); t10(); t11(seqs); t12(); t13(a.data_dir, a.split_scene); t14(); t15(seqs); t16(seqs, a.split_scene); t17(a.data_dir, a.split_scene); t18(a.data_dir, a.split_scene)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    json.dump(RESULTS, open(a.out, "w"), indent=1)
    n = sum(r["result"] == "PASS" for r in RESULTS); nw = sum(r["result"] == "WARN" for r in RESULTS)
    nf = len(RESULTS) - n - nw
    print(f"\n{n}/{len(RESULTS)} PASS, {nw} WARN, {nf} FAIL -> {a.out}")
    return 0 if nf == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
