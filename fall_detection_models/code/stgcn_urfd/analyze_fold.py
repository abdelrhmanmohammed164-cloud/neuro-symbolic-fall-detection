"""
analyze_fold1.py  (diagnostic only, reads saved outputs; changes nothing)
Phase 1: recompute Fold 1 metrics from the saved per-frame probabilities.
Phase 2/3: class balance, window-label statistics, scene statistics, per-video error report,
           event-level metrics, latency, false alarms per minute, error grouping, bootstrap CIs.
Held-out predictions are used ONLY to describe failures, never to choose anything.
"""
import csv, json, os, sys, collections
import numpy as np

ROOT = sys.argv[1]
K = int(sys.argv[4]) if len(sys.argv) > 4 else 1
OUT = sys.argv[2]
CODE = sys.argv[3]
sys.path.insert(0, CODE)
from dataset import load_sequences, TrainWindows, EvalWindows, window_by_end, label_and_weight, missing_mask
from train import binary_metrics

os.makedirs(OUT, exist_ok=True)
DATA = os.path.join(ROOT, "processed_data/cfg_d31bfed741ab/urfd_npy")
split = json.load(open(os.path.join(ROOT, "splits/split_scene.json")))
fold = split["folds"][K]
res = json.load(open(os.path.join(ROOT, f"runs/stgcn_scene/fold{K}/result.json")))
diag = json.load(open(os.path.join(ROOT, f"reports/fold{K}_diagnosis.json")))
hist = list(csv.DictReader(open(os.path.join(ROOT, f"runs/stgcn_scene/fold{K}/training_history.csv"))))
scene = {r["sequence"]: r["scene_group"] for r in csv.DictReader(open(os.path.join(ROOT, "reports/scene_groups.csv")))}
scene = {k: split["scene_merge"].get(v, v) for k, v in scene.items()}
seqs = {s["name"]: s for s in load_sequences(DATA)}
raw = collections.defaultdict(dict)                       # URFD raw frame labels -1/0/1
for fn in ("urfall-cam0-falls.csv", "urfall-cam0-adls.csv"):
    for r in csv.reader(open(os.path.join(ROOT, "URFD/features", fn))):
        if r:
            raw[r[0]][int(r[1])] = int(r[2])
FPS = 30.0
report = {}

# ---------------- Phase 1: recompute -----------------------------------------------------------
fr = res["frames"]
y = np.array([f["label"] for f in fr]); p = np.array([f["prob"] for f in fr])
rec = binary_metrics(y, p, 0.5)
keys = ["tp", "tn", "fp", "fn", "accuracy", "precision", "recall_sensitivity", "specificity", "f1"]
cmp = {k: (rec[k], res["frame_metrics"][k], diag["metrics"]["test"][k]) for k in keys}
consistent = all(abs(a - b) < 1e-9 and abs(a - c) < 1e-9 for a, b, c in cmp.values())
# labels in result.json vs frame-label files
lab_ok = all(seqs[f["sequence"]]["frame_labels"][f["frame"] - 1] == f["label"] for f in fr)
cover = collections.Counter(f["sequence"] for f in fr)
cover_ok = all(cover[n] == seqs[n]["data"].shape[1] for n in fold["test"]) and set(cover) == set(fold["test"])
dup = len(fr) - len({(f["sequence"], f["frame"]) for f in fr})
best_hist = [h for h in hist if h["best"] == "True"]
report["phase1"] = dict(recomputed=rec, consistent_with_result_and_diagnosis=consistent,
                        labels_match_frame_label_files=bool(lab_ok), every_test_frame_once=bool(cover_ok),
                        duplicate_frame_rows=dup, n_test_frames=len(fr),
                        best_epoch_from_history=int(best_hist[-1]["epoch"]), best_epoch_reported=res["best_epoch"],
                        min_val_loss_epoch=int(np.argmin([float(h["val_loss"]) for h in hist])),
                        sequence_metrics_saved=res["sequence_metrics"], temperature=res["temperature"],
                        diagnosis_verdict=diag["verdict"], diagnosis_gap=diag["train_test_f1_gap"])

# ---------------- Phase 2D: class balance --------------------------------------------------------
def balance(names):
    v1 = sum(seqs[n]["label"] for n in names); fT = sum(seqs[n]["data"].shape[1] for n in names)
    f1 = sum(int(seqs[n]["frame_labels"].sum()) for n in names)
    return dict(videos_fall=v1, videos_adl=len(names) - v1, frames=fT, fall_frames=f1, fall_frame_frac=f1 / fT)
bal = {part: balance(fold[part]) for part in ("train", "val", "test")}
tw = TrainWindows([seqs[n] for n in fold["train"]], 80, 64, 10, "image", augment=False, p_pad=0.35, seed=42 * 1000 + K)
lw = np.array(tw.labels())
bal["train_windows_epoch0"] = dict(n=len(lw), pos_weight0=int(((lw[:, 0] == 1) & (lw[:, 1] > 0)).sum()),
                                   pos=int(((lw[:, 0] == 1) & (lw[:, 1] > 0)).sum()),
                                   neg=int((lw[:, 0] == 0).sum()), excluded_w0=int((lw[:, 1] == 0).sum()))
b = bal["train_windows_epoch0"]; b["pos_weight"] = b["neg"] / b["pos"]; del b["pos_weight0"]
# pos_weight over all 50 epochs (it is computed from epoch 0 only)
pw = []
for e in range(50):
    tw.set_epoch(e); l = np.array(tw.labels()); pw.append(((l[:, 0] == 0).sum()) / max(1, ((l[:, 0] == 1) & (l[:, 1] > 0)).sum()))
b["pos_weight_range_over_epochs"] = [float(min(pw)), float(max(pw))]
report["class_balance"] = bal

# ---------------- Phase 2C: window labelling -----------------------------------------------------
wl = dict()
fall_ends, excl_ends, pos_rel = 0, 0, []
for e in range(50):
    tw.set_epoch(e)
    for si, end, n_real, _ in tw.items:
        s = tw.seqs[si]; fl = s["frame_labels"]
        if fl[end] == 1:
            _, st = window_by_end(s["data"], end, 80, n_real)
            on = int(np.argmax(fl)); rel = end - on
            fall_ends += 1
            if fl[st:end + 1].sum() < 10:
                excl_ends += 1
            else:
                pos_rel.append(rel)
pos_rel = np.array(pos_rel)
# phase of each fall frame (URFD raw label 0 = falling, 1 = lying)
ph = collections.Counter()
for n in fold["train"]:
    if n.startswith("fall"):
        for t in range(seqs[n]["data"].shape[1]):
            if seqs[n]["frame_labels"][t]:
                ph[raw[n].get(t + 1, 99)] += 1
wl.update(train_windows_ending_on_fall_frame_50ep=fall_ends, excluded_weight0=excl_ends,
          excluded_frac_of_fall_windows=excl_ends / fall_ends,
          trained_pos_windows_frames_after_onset=dict(p10=float(np.percentile(pos_rel, 10)),
                                                      median=float(np.median(pos_rel)),
                                                      p90=float(np.percentile(pos_rel, 90))),
          train_fall_frames_by_urfd_phase={"falling(0)": ph[0], "lying(1)": ph[1]})
# test frames that training could never have labelled positive with weight 1 (first 9 fall frames)
onset9 = 0
for n in fold["test"]:
    fl = seqs[n]["frame_labels"]
    if fl.any():
        onset9 += min(9, int(fl.sum()))
wl["test_fall_frames_in_first_9_after_onset"] = onset9
report["window_labels"] = wl

# ---------------- Phase 2B: scene statistics -----------------------------------------------------
def seq_stats(n):
    x = seqs[n]["data"][..., 0]                  # (C, T, V)
    miss = x[2] <= 0
    T = x.shape[1]
    neck_y = np.where(~miss[:, 1], x[1, :, 1], np.nan)
    # body scale: neck to mid-hip distance (joints 8, 11 = hips)
    hip = (x[:2, :, 8] + x[:2, :, 11]) / 2
    ok = ~miss[:, 1] & ~miss[:, 8] & ~miss[:, 11]
    scale = np.where(ok, np.linalg.norm(x[:2, :, 1] - hip, axis=0), np.nan)
    xy = x[:2]; valid = ~miss
    both = valid[1:] & valid[:-1]
    mot = np.where(both, np.linalg.norm(xy[:, 1:] - xy[:, :-1], axis=0), np.nan)
    conf = np.where(valid, x[2], np.nan)
    ts_path = os.path.join(DATA, n + "_timestamps.npy")
    fps = np.nan
    if os.path.exists(ts_path):
        ts = np.load(ts_path).astype(float)
        d = np.diff(ts); d = d[d > 0]
        if len(d):
            med = np.median(d); fps = 1000 / med if med > 1 else 1 / med
    return dict(T=T, miss=float(miss.mean()), conf=float(np.nanmean(conf)), neck_y=float(np.nanmean(neck_y)),
                neck_y_last=float(np.nanmean(neck_y[-10:])), scale=float(np.nanmedian(scale)),
                motion=float(np.nanmean(mot)), fps=float(fps))
ss = {n: seq_stats(n) for n in seqs}
by_scene = collections.defaultdict(list)
for n in seqs:
    by_scene[scene[n]].append(n)
scene_tab = {}
for g, ns in sorted(by_scene.items()):
    role = "test" if g in fold["test_groups"] else "train/val"
    agg = {k: float(np.nanmedian([ss[n][k] for n in ns])) for k in ss[ns[0]]}
    scene_tab[g] = dict(role_fold1=role, n=len(ns), falls=sum(seqs[n]["label"] for n in ns), **agg)
report["scenes"] = scene_tab

# ---------------- Phase 3: per-video errors ------------------------------------------------------
P = collections.defaultdict(dict)
for f in fr:
    P[f["sequence"]][f["frame"] - 1] = f["prob"]
rows = []
fn_groups = collections.Counter(); fp_groups = collections.Counter()
fn_feat, fp_feat, tp_feat, tn_feat = [], [], [], []
for n in fold["test"]:
    s = seqs[n]; T = s["data"].shape[1]; fl = s["frame_labels"]
    pr = np.array([P[n][t] for t in range(T)]); yh = pr >= 0.5
    x = s["data"][..., 0]; miss = x[2] <= 0
    r = dict(sequence=n, scene=scene[n], label=s["label"], T=T, fall_frames=int(fl.sum()),
             max_prob=float(pr.max()), tp=int((yh & (fl == 1)).sum()), fn=int((~yh & (fl == 1)).sum()),
             fp=int((yh & (fl == 0)).sum()), tn=int((~yh & (fl == 0)).sum()))
    # false-alarm episodes: runs of predicted positives on negative frames
    neg_pos = yh & (fl == 0)
    eps = int(((neg_pos[1:].astype(int) - neg_pos[:-1].astype(int)) == 1).sum() + int(neg_pos[0]))
    r["fa_episodes"] = eps
    r["fa_episodes_per_min"] = eps / (T / FPS / 60)
    if fl.any():
        on = int(np.argmax(fl))
        det = np.nonzero(yh[on:])[0]
        r["onset_frame"] = on + 1
        r["event_detected"] = bool(len(det))
        r["latency_frames"] = int(det[0]) if len(det) else None
        r["latency_s"] = float(det[0] / FPS) if len(det) else None
        r["detected_before_onset"] = bool(yh[:on].any())
        r["fall_frames_after_onset_to_end"] = T - on
        # FN by phase
        for t in range(T):
            if fl[t] and not yh[t]:
                rel = t - on
                ph_ = raw[n].get(t + 1, 99)
                fn_groups["onset_first9" if rel < 9 else ("falling_phase" if ph_ == 0 else "lying_phase")] += 1
    else:
        r.update(onset_frame=None, event_detected=None, latency_frames=None, latency_s=None, detected_before_onset=None)
    # per-frame features for grouping
    for t in range(T):
        st = max(0, t - 79)
        f_ = dict(miss=float(miss[st:t + 1].mean()), conf=float(np.where(~miss[st:t+1], x[2, st:t+1], np.nan).__array__()[~miss[st:t+1]].mean()) if (~miss[st:t+1]).any() else 0.0,
                  padded=t < 79, neck_y=float(x[1, t, 1]) if not miss[t, 1] else np.nan)
        if fl[t] and not yh[t]: fn_feat.append(f_)
        elif fl[t] and yh[t]: tp_feat.append(f_)
        elif not fl[t] and yh[t]: fp_feat.append(f_)
        else: tn_feat.append(f_)
    rows.append(r)
with open(os.path.join(OUT, f"fold{K}_per_video_errors.csv"), "w", newline="") as f:
    keys_ = list(rows[-1].keys()) if rows[-1]["label"] else None
    allk = []
    for r in rows:
        for k in r:
            if k not in allk: allk.append(k)
    w = csv.DictWriter(f, fieldnames=allk); w.writeheader(); w.writerows(rows)

def summ(L):
    if not L: return {}
    return dict(n=len(L), mean_missing_frac=float(np.mean([d["miss"] for d in L])),
                mean_conf=float(np.mean([d["conf"] for d in L])), padded_frac=float(np.mean([d["padded"] for d in L])),
                neck_y_median=float(np.nanmedian([d["neck_y"] for d in L])))
groups = dict(FN=summ(fn_feat), TP=summ(tp_feat), FP=summ(fp_feat), TN=summ(tn_feat))
fallrows = [r for r in rows if r["label"] == 1]; adlrows = [r for r in rows if r["label"] == 0]
fn_by_video = sorted(((r["fn"], r["sequence"]) for r in fallrows), reverse=True)
fp_by_video = sorted(((r["fp"], r["sequence"]) for r in adlrows + fallrows if r["fp"]), reverse=True)
tot_fn = sum(r["fn"] for r in rows); tot_fp = sum(r["fp"] for r in rows)
lat = [r["latency_s"] for r in fallrows if r["latency_s"] is not None]
adl_minutes = sum(r["T"] for r in adlrows) / FPS / 60
report["errors"] = dict(
    fn_total=tot_fn, fp_total=tot_fp,
    fn_by_phase=dict(fn_groups),
    fn_top3_videos_share=sum(v for v, _ in fn_by_video[:3]) / tot_fn, fn_by_video=fn_by_video,
    fp_top3_videos_share=sum(v for v, _ in fp_by_video[:3]) / tot_fp, fp_by_video=fp_by_video,
    events=dict(falls=len(fallrows), detected=sum(r["event_detected"] for r in fallrows),
                missed=[r["sequence"] for r in fallrows if not r["event_detected"]],
                latency_s_median=float(np.median(lat)), latency_s_max=float(np.max(lat)),
                latency_s_all={r["sequence"]: r["latency_s"] for r in fallrows},
                fall_videos_with_alarm_before_onset=[r["sequence"] for r in fallrows if r["detected_before_onset"]]),
    adl=dict(videos=len(adlrows), with_any_false_alarm=[r["sequence"] for r in adlrows if r["fp"]],
             fa_episodes=sum(r["fa_episodes"] for r in adlrows), adl_minutes=adl_minutes,
             fa_episodes_per_min=sum(r["fa_episodes"] for r in adlrows) / adl_minutes),
    grouping=groups)

# ---------------- uncertainty: bootstrap over videos -------------------------------------------
rng = np.random.default_rng(0)
B = 5000; f1s, evr, spv = [], [], []
fallv = [r for r in rows if r["label"]]; adlv = [r for r in rows if not r["label"]]
for _ in range(B):
    fs = [fallv[i] for i in rng.integers(0, len(fallv), len(fallv))]
    ad = [adlv[i] for i in rng.integers(0, len(adlv), len(adlv))]
    tp = sum(r["tp"] for r in fs + ad); fp = sum(r["fp"] for r in fs + ad); fn_ = sum(r["fn"] for r in fs + ad)
    f1s.append(2 * tp / max(1, 2 * tp + fp + fn_))
    evr.append(np.mean([r["event_detected"] for r in fs]))
    spv.append(np.mean([r["fp"] == 0 for r in ad]))
ci = lambda a: [float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5))]
report["uncertainty_video_bootstrap"] = dict(B=B, frame_f1=[rec["f1"], *ci(f1s)],
    event_recall=[np.mean([r["event_detected"] for r in fallv]), *ci(evr)],
    adl_videos_without_false_alarm=[np.mean([r["fp"] == 0 for r in adlv]), *ci(spv)])

# ---------------- leakage-related checks from data ---------------------------------------------
lk = {}
lk["near_duplicate_group_positions"] = {n: ("train" if n in fold["train"] else "val" if n in fold["val"] else "test")
                                        for n in split["near_groups"][0]}
lk["scene_of_val"] = sorted({scene[n] for n in fold["val"]})
lk["scene_of_train"] = sorted({scene[n] for n in fold["train"]})
lk["scene_of_test"] = sorted({scene[n] for n in fold["test"]})
lk["scene_class_purity"] = {g: [sum(seqs[n]["label"] for n in ns), len(ns)] for g, ns in sorted(by_scene.items())}
report["leakage_data_checks"] = lk

json.dump(report, open(os.path.join(OUT, f"fold{K}_audit.json"), "w"), indent=1, default=float)

# ---------------- plots ------------------------------------------------------------------------
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
ep = [int(h["epoch"]) for h in hist]
fig, ax = plt.subplots(1, 2, figsize=(12, 4))
ax[0].plot(ep, [float(h["train_loss"]) for h in hist], label="train loss (weighted BCE, augmented)")
ax[0].plot(ep, [float(h["val_loss"]) for h in hist], label="val loss (plain BCE, same scenes as train)")
ax[0].set_yscale("log"); ax[0].axvline(res["best_epoch"], ls="--", c="gray", label=f"selected epoch {res['best_epoch']}")
ax[0].set_xlabel("epoch"); ax[0].legend(fontsize=8); ax[0].set_title(f"Fold {K} losses (log scale)")
ax[1].plot(ep, [float(h["val_f1"]) for h in hist], label="val F1")
ax[1].plot(ep, [float(h["val_recall"]) for h in hist], label="val recall")
ax[1].axhline(rec["f1"], c="r", ls=":", label=f"test F1 at selected epoch ({rec['f1']:.2f})")
ax[1].set_ylim(0, 1.05); ax[1].set_xlabel("epoch"); ax[1].legend(fontsize=8); ax[1].set_title(f"Fold {K} validation metrics")
fig.tight_layout(); fig.savefig(os.path.join(OUT, f"fold{K}_learning_curves.png"), dpi=120); plt.close(fig)

nr = int(np.ceil(len(fold["test"]) / 5)); fig, axs = plt.subplots(nr, 5, figsize=(18, 2.7 * nr)); axs = axs.ravel()
for i, n in enumerate(fold["test"]):
    s = seqs[n]; T = s["data"].shape[1]; pr = [P[n][t] for t in range(T)]
    a = axs[i]; a.plot(np.arange(1, T + 1), pr, lw=1)
    a.fill_between(np.arange(1, T + 1), 0, s["frame_labels"], color="orange", alpha=0.3, step="mid")
    a.axhline(0.5, c="gray", lw=0.5, ls="--"); a.set_ylim(-0.05, 1.05); a.set_title(f"{n} ({scene[n]})", fontsize=9)
    a.tick_params(labelsize=7)
for j in range(len(fold["test"]), len(axs)): axs[j].axis("off")
fig.suptitle(f"Fold {K} test videos: P(fall) per frame (blue) vs fall frames (orange)", fontsize=12)
fig.tight_layout(); fig.savefig(os.path.join(OUT, f"fold{K}_per_video_probs.png"), dpi=100); plt.close(fig)

fig, ax = plt.subplots(figsize=(9, 4))
names = [r["sequence"] for r in rows]; xs = np.arange(len(rows))
ax.bar(xs, [r["fn"] for r in rows], label="FN frames"); ax.bar(xs, [r["fp"] for r in rows], bottom=[r["fn"] for r in rows], label="FP frames")
ax.set_xticks(xs); ax.set_xticklabels(names, rotation=90, fontsize=7); ax.legend(); ax.set_title(f"Fold {K} test errors per video")
fig.tight_layout(); fig.savefig(os.path.join(OUT, f"fold{K}_errors_per_video.png"), dpi=120); plt.close(fig)
print(json.dumps(report, indent=1, default=float)[:12000])
