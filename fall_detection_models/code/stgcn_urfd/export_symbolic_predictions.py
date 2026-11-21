"""Export per-frame frozen symbolic predictions, aligned with neural OOF CSV.
Requires separately generated Kinect neck height npz files; NEVER refits on tests.
"""
import argparse
import csv
import json
from pathlib import Path
import numpy as np
from symbolic_v3_improved import V3Improved, Video


def export(split_path, config_path, height_dir, out):
    split = json.loads(Path(split_path).read_text())
    configs = json.loads(Path(config_path).read_text())
    rows = []
    seen = set()
    for i, fold in enumerate(split["folds"]):
        k = fold.get("fold", i)                      # the config of the fold where the video is test
        m = V3Improved(**configs[str(k)])
        for name in fold["test"]:
            if name in seen:
                raise ValueError(f"duplicate held-out video: {name}")
            seen.add(name)
            path = Path(height_dir) / f"{name}.npz"
            if not path.exists():
                raise FileNotFoundError(f"missing measured neck heights: {path}. Run Kinect height extraction; never substitute skeleton y coordinates for metres")
            with np.load(path, allow_pickle=False) as z:
                h = np.asarray(z["height_m"], dtype=float)
            alarms = m.predict(Video(h))
            for frame, a in enumerate(alarms, 1):
                rows.append(dict(sequence=name, frame=frame, alarm=int(a)))
    dst = Path(out); dst.parent.mkdir(parents=True, exist_ok=True)
    with dst.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["sequence", "frame", "alarm"])
        w.writeheader(); w.writerows(rows)
    return len(rows)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--split_file", required=True)
    p.add_argument("--config", required=True)
    p.add_argument("--height_dir", required=True)
    p.add_argument("--out", required=True)
    a = p.parse_args()
    print(f"exported {export(a.split_file, a.config, a.height_dir, a.out)} frames")
