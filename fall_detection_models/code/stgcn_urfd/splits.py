"""
splits.py
Builds, verifies and saves the cross-validation splits. Training only READS the saved file.

mode "scene" (primary, default): leave-scene-out CV. URFD scenes are class-pure (audit), so a
    split that mixes scenes lets the model recognise the room. One fold per fall scene; ADL scenes
    are spread over the folds (greedy, deterministic) to keep a similar fall/ADL ratio.
mode "video" (secondary, optimistic): stratified group k-fold over videos; scenes are shared
    between train and test, so results can include scene shortcuts. Report it only as such.

Inside each fold, `val_frac` of the TRAINING sequences (stratified by label, fixed seed) form the
validation set used for checkpoint selection and temperature scaling. The test fold is never used
for any choice. In scene mode with --val_mode random (original), validation shares scenes with
training. Audit fix: --val_mode scene holds out one WHOLE ADL scene (the smallest, pre-registered)
plus a random subset of the falls, and keeps near-duplicate groups together. Saved as a new file;
the original split and its fingerprint are unchanged.

Usage:
  python splits.py --data_dir processed/urfd_npy --scene_csv reports/scene_groups.csv \
      --mode scene --out splits/split_scene_v3.json
"""
import argparse
import csv
import hashlib
import json
import os

import numpy as np
from sklearn.model_selection import StratifiedGroupKFold, train_test_split

from dataset import is_sequence_file

# fall-07 forms its own 1-video group at threshold 0.9; it lies inside the F2 id range (fall-01..12)
# and correlates with F2 just under the threshold, so it is merged into F2 (documented in the report).
SCENE_MERGE = {"F3": "F2"}
# near-identical (but distinct) sequences found by the audit: keep together in video mode
NEAR_GROUPS = [["adl-25", "adl-27", "adl-29"]]


def fingerprint(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True).encode()).hexdigest()


def build(names, labels, mode, scene, folds, val_frac, seed, val_mode="random"):
    names = list(names); y = np.array([labels[n] for n in names])
    if mode == "scene":
        g = {n: SCENE_MERGE.get(scene[n], scene[n]) for n in names}
        fall_groups = sorted({g[n] for n in names if labels[n] == 1})
        adl_groups = sorted({g[n] for n in names if labels[n] == 0},
                            key=lambda s: (-sum(g[n] == s for n in names), s))
        test = {k: [x for x in names if g[x] == fg] for k, fg in enumerate(fall_groups)}
        for ag in adl_groups:                       # give each ADL scene to the fold that needs ADLs most
            k = min(test, key=lambda k: (sum(labels[x] == 0 for x in test[k]) /
                                         max(1, sum(labels[x] == 1 for x in test[k])), k))
            test[k] += [x for x in names if g[x] == ag]
        group_of = g
        test_sets = [sorted(test[k]) for k in sorted(test)]
    else:
        group_of = {n: n for n in names}
        for grp in NEAR_GROUPS:
            for n in grp:
                if n in group_of:
                    group_of[n] = grp[0]
        sgkf = StratifiedGroupKFold(n_splits=folds, shuffle=True, random_state=seed)
        groups = [group_of[n] for n in names]
        test_sets = [sorted(names[i] for i in te) for _, te in sgkf.split(np.zeros(len(names)), y, groups)]
    out = []
    for k, te in enumerate(test_sets):
        tr_all = [n for n in names if n not in te]
        if val_mode == "scene":
            tr, va, vg = scene_val(tr_all, labels, group_of, val_frac, seed + k)
        else:
            tr, va = train_test_split(tr_all, test_size=val_frac, stratify=[labels[n] for n in tr_all],
                                      random_state=seed + k)
            if val_mode == "random_grouped":            # audit fix for video mode: same draw, groups kept
                tr, va = keep_near_groups_together(tr, va)
            vg = None
        out.append(dict(fold=k, train=sorted(tr), val=sorted(va), test=sorted(te),
                        test_groups=sorted({group_of[n] for n in te}),
                        train_groups=sorted({group_of[n] for n in tr_all})))
        if vg is not None:
            out[-1]["val_adl_group"] = vg
    return out, group_of


def keep_near_groups_together(tr, va):
    """A near-duplicate group never straddles train and val: the whole group goes to the side that
    already holds most of it (ties -> train)."""
    tr, va = set(tr), set(va)
    for grp in NEAR_GROUPS:
        inn = [n for n in grp if n in tr | va]
        if not inn:
            continue
        in_va = sum(n in va for n in inn)
        to_val = in_va * 2 > len(inn)
        for n in inn:
            (va if to_val else tr).add(n); (tr if to_val else va).discard(n)
    return sorted(tr), sorted(va)


def scene_val(tr_all, labels, group_of, val_frac, seed):
    """val_mode 'scene' (audit fix): the validation set contains one WHOLE ADL scene that is not in
    training, so checkpoint selection sees an unseen room. Pre-registered rule: the ADL scene with
    the fewest videos (ties -> name). There is only one fall scene per training fold, so fall
    validation videos are a fixed random subset of it (they share the room with training)."""
    adl_groups = sorted({group_of[n] for n in tr_all if labels[n] == 0},
                        key=lambda g: (sum(group_of[n] == g for n in tr_all), g))
    vg = adl_groups[0]
    falls = sorted(n for n in tr_all if labels[n] == 1)
    rng = np.random.default_rng(seed)
    n_fv = max(2, int(round(val_frac * len(falls))))
    fall_val = sorted(rng.choice(falls, size=n_fv, replace=False).tolist())
    va = [n for n in tr_all if group_of[n] == vg] + fall_val
    tr = [n for n in tr_all if n not in va]
    tr, va = keep_near_groups_together(tr, va)
    return tr, va, vg


def verify(folds, names, group_of, check_near=False):
    """Raises if any check fails. Returns a list of human-readable PASS lines."""
    names = set(names); msgs = []
    tests = [set(f["test"]) for f in folds]
    assert set().union(*tests) == names, "test folds do not cover all sequences"
    assert sum(len(t) for t in tests) == len(names), "a sequence is in more than one test fold"
    msgs.append("every sequence is in exactly one test fold")
    for f in folds:
        tr, va, te = set(f["train"]), set(f["val"]), set(f["test"])
        assert not (tr & va) and not (tr & te) and not (va & te), f"fold {f['fold']}: overlapping sets"
        assert tr | va | te == names, f"fold {f['fold']}: sets do not cover all sequences"
        gtr = {group_of[n] for n in tr | va}; gte = {group_of[n] for n in te}
        assert not (gtr & gte), f"fold {f['fold']}: group(s) {gtr & gte} in train and test"
    msgs.append("train / val / test disjoint in every fold; no group in both train(+val) and test")
    if check_near:
        for f in folds:
            for grp in NEAR_GROUPS:
                sides = {("train" if n in f["train"] else "val" if n in f["val"] else "test") for n in grp if n in names}
                assert len(sides) <= 1, f"fold {f['fold']}: near-duplicate group {grp} split across {sorted(sides)}"
            if f.get("val_adl_group"):
                assert f["val_adl_group"] not in {group_of[n] for n in f["train"]}, \
                    f"fold {f['fold']}: validation ADL scene {f['val_adl_group']} also in training"
        msgs.append("near-duplicate groups never split across train / val / test; validation ADL scene not in training")
    return msgs


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", required=True)
    p.add_argument("--scene_csv", default=None)
    p.add_argument("--mode", choices=["scene", "video"], default="scene")
    p.add_argument("--folds", type=int, default=5, help="video mode only")
    p.add_argument("--val_frac", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--val_mode", choices=["random", "random_grouped", "scene"], default="random",
                   help="random: 20%% of training videos (original). random_grouped: same, near-duplicate groups "
                        "kept on one side. scene: one whole unseen ADL scene + some falls")
    p.add_argument("--out", required=True)
    a = p.parse_args()
    names = sorted(f[:-4] for f in os.listdir(a.data_dir) if is_sequence_file(f))
    labels = {n: int(n.startswith("fall")) for n in names}
    scene = {}
    if a.mode == "scene":
        for r in csv.DictReader(open(a.scene_csv)):
            scene[r["sequence"]] = r["scene_group"]
        missing = [n for n in names if n not in scene]
        assert not missing, f"no scene group for {missing}"
    folds, group_of = build(names, labels, a.mode, scene, a.folds, a.val_frac, a.seed, a.val_mode)
    msgs = verify(folds, names, group_of, check_near=(a.val_mode != "random"))
    doc = dict(mode=a.mode, seed=a.seed, val_frac=a.val_frac, n_sequences=len(names),
               scene_merge=SCENE_MERGE if a.mode == "scene" else {}, near_groups=NEAR_GROUPS,
               group_of=group_of, folds=folds)
    keys = ("mode", "seed", "val_frac", "folds")
    if a.val_mode != "random":                      # keeps the fingerprint of the original split unchanged
        doc["val_mode"] = a.val_mode; keys += ("val_mode",)
    doc["fingerprint"] = fingerprint({k: doc[k] for k in keys})
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    if os.path.exists(a.out):
        old = json.load(open(a.out))
        if old.get("fingerprint") != doc["fingerprint"]:
            raise SystemExit(f"{a.out} exists with a DIFFERENT split; refusing to overwrite. "
                             f"Delete it on purpose if you really want a new split.")
        print(f"{a.out} already exists and is identical -> unchanged")
    else:
        json.dump(doc, open(a.out, "w"), indent=1)
        print(f"saved {a.out}")
    for m in msgs:
        print("PASS:", m)
    for f in folds:
        lab = lambda s: (sum(labels[n] for n in s), sum(1 - labels[n] for n in s))
        print(f"fold {f['fold']}: test groups {f['test_groups']} | falls/ADL train {lab(f['train'])} "
              f"val {lab(f['val'])} test {lab(f['test'])}"
              + (f" | val ADL scene {f['val_adl_group']}" if f.get("val_adl_group") else ""))


if __name__ == "__main__":
    main()
