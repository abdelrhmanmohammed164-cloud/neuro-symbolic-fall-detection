"""Conservative data/memorization audit: group disjointness and byte-identical arrays.
Passing this check is NOT proof that a trained model cannot overfit.
"""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np


def audit(split_path, data_dir):
    split = json.loads(Path(split_path).read_text())
    expected = set(split["group_of"])
    errors = []
    warnings = []
    digests = {}
    groups = split["group_of"]
    for name in sorted(expected):
        path = Path(data_dir) / f"{name}.npy"
        if not path.is_file():
            errors.append(f"missing skeleton: {name}")
            continue
        arr = np.load(path, allow_pickle=False)
        if arr.ndim != 4 or arr.shape[0] != 3 or arr.shape[2:] != (14, 1) or not np.isfinite(arr).all():
            errors.append(f"bad shape or nonfinite skeleton: {name} {arr.shape}")
        # Identical arrays are potentially duplicated sequences. This check
        # cannot detect near-duplicate poses or duplicated camera views.
        key = hashlib.sha256(str(arr.shape).encode() + arr.tobytes()).hexdigest()
        digests.setdefault(key, []).append(name)
    duplicates = [names for names in digests.values() if len(names) > 1]
    for k, fold in enumerate(split["folds"]):
        sets = {part: set(fold[part]) for part in ("train", "val", "test")}
        for part, names in sets.items():
            if not names <= expected:
                errors.append(f"fold{k} unknown sequences in {part}: {sorted(names - expected)}")
        if set.union(*sets.values()) != expected:
            errors.append(f"fold{k} does not partition the entire dataset")
        for a, b in (("train", "val"), ("train", "test"), ("val", "test")):
            if sets[a] & sets[b]:
                errors.append(f"fold{k} sequence leakage between {a}/{b}")
            ga, gb = {groups[x] for x in sets[a] if x in groups}, {groups[x] for x in sets[b] if x in groups}
            if ga & gb:
                msg = f"fold{k} scene overlap between {a}/{b}: {sorted(ga & gb)}"
                (errors if "test" in (a, b) else warnings).append(msg)
        for dup in duplicates:
            inparts = [p for p, ns in sets.items() if set(dup) & ns]
            if len(inparts) > 1:
                errors.append(f"fold{k} duplicate arrays across {inparts}: {dup}")
    # Cross-fold: a video used for selection (train/val) in one fold may be test in another. That is expected in
    # k-fold CV, but it means choices pooled across folds would see test videos. Warn when a fold's VAL sits
    # inside another fold's TEST, so selection is kept per fold.
    folds = split["folds"]
    for i, fa in enumerate(folds):
        for j, fb in enumerate(folds):
            if i != j:
                shared = set(fa["val"]) & set(fb["test"])
                if shared:
                    warnings.append(f"fold{i} val inside fold{j} test: {len(shared)}/{len(fa['val'])} videos; select per fold only")
    return {"passed": not errors, "sequences": len(expected), "folds": len(split["folds"]),
            "identical_skeleton_groups": duplicates, "errors": errors, "warnings": warnings,
            "interpretation": "PASS means test isolation and exact duplicate checks only. A train/val scene warning on the falls is by design in split_scene_v2 (validation = one ADL scene + some falls) and makes validation-based selection optimistic. A cross-fold warning means selection must stay per fold. Overfitting requires curves and independent held-out subjects/cameras."}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split_file", required=True)
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--out")
    a = parser.parse_args()
    report = audit(a.split_file, a.data_dir)
    if a.out:
        Path(a.out).write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    raise SystemExit(0 if report["passed"] else 1)
