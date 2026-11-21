"""
build_dashboard.py
Builds dashboard/dashboard.html from the project's result files, so the dashboard always shows the current
numbers. The page is self-contained (data embedded, no internet needed).

Usage: python scripts/build_dashboard.py          (run from anywhere; PROJECT_ROOT overrides the root)
Reads:  reports/symbolic_v3_precision/neural_test_oof.csv   neural ensemble, out-of-fold, per frame
        processed_data/kinect_height/<video>.npz             metric neck height per frame
        reports/symbolic_v3_precision/results.json           improved V3 test results
        reports/external_gmdcsa24/external_test.json         neural ensemble on GMDCSA-24
        reports/symbolic_v3_external_fukinect/results.json   improved V3 on FUKinect-Fall
        reports/data_integrity_audit.json                    split audit (data_audit.py)
"""
import csv
import json
import math
import os
from pathlib import Path

import numpy as np

ROOT = Path(os.environ.get("PROJECT_ROOT", Path(__file__).resolve().parents[1]))
TEMPLATE = ROOT / "dashboard" / "template.html"
OUT = ROOT / "dashboard" / "dashboard.html"


def load_json(rel):
    return json.loads((ROOT / rel).read_text())


def main():
    rows = {}
    with open(ROOT / "reports/symbolic_v3_precision/neural_test_oof.csv", newline="") as f:
        for r in csv.DictReader(f):
            rows.setdefault(r["sequence"], []).append((int(r["frame"]), int(r["label"]), float(r["prob"])))
    data, counts = [], dict(TP=0, FP=0, TN=0, FN=0)
    for name in sorted(rows):
        fr = sorted(rows[name])
        frames = [x[0] for x in fr]
        if frames != list(range(1, len(frames) + 1)):
            raise ValueError(f"{name}: frames are not 1..T without gaps")
        h = np.load(ROOT / "processed_data/kinect_height" / f"{name}.npz")["height_m"]
        if len(h) != len(frames):
            raise ValueError(f"{name}: {len(h)} heights for {len(frames)} frames")
        labels = [x[1] for x in fr]; probs = [x[2] for x in fr]
        for y, p in zip(labels, probs):
            k = ("T" if (p >= 0.5) == bool(y) else "F") + ("P" if p >= 0.5 else "N")
            counts[k] += 1
        data.append(dict(name=name, kind="Fall" if name.startswith("fall") else "ADL", frame=frames, label=labels,
                         prob=[round(p, 5) for p in probs],
                         height=[None if not math.isfinite(v) else round(float(v), 4) for v in h]))
    sym = load_json("reports/symbolic_v3_precision/results.json")
    fuk = load_json("reports/symbolic_v3_external_fukinect/results.json")
    gm = load_json("reports/external_gmdcsa24/external_test.json")["results"]["center"]["ensemble (mean probability of the 6 models)"]["all"]
    db = dict(data=data, symbolic=sym["test"]["E (selected)"], neural=counts,
              folds={k: v for k, v in sym["per_fold_test"].items() if k.endswith("|E (selected)")},
              external={k: v["primary"] for k, v in fuk.items()}, gmdcsa=gm,
              audit=load_json("reports/data_integrity_audit.json"),
              source="neural_test_oof.csv + kinect_height/*.npz + results.json files")
    html = TEMPLATE.read_text()
    if html.count("__DB__") != 1:
        raise ValueError("template must contain __DB__ exactly once")
    blob = json.dumps(db, separators=(",", ":")).replace("</", "<\\/")      # cannot close the <script> tag
    OUT.write_text(html.replace("__DB__", blob))
    print(f"wrote {OUT} ({OUT.stat().st_size // 1024} KB): {len(data)} videos, "
          f"{sum(len(d['frame']) for d in data)} frames, neural TP {counts['TP']} FP {counts['FP']}")


if __name__ == "__main__":
    main()
