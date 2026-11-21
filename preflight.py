"""Show what is usable in this checkout without silently downloading proprietary weights."""
import argparse
import json
from pathlib import Path


def check(root):
    root = Path(root)
    data = root / "processed_data/cfg_d31bfed741ab/urfd_npy"
    npys = list(data.glob("*.npy"))
    ckpts = list(root.glob("runs/**/best_checkpoint.pt"))
    neck = list((root / "processed_data/kinect_height").glob("*.npz"))
    weights = root / "fall_detection_models/weights"
    weight_files = [p for p in weights.rglob("*") if p.is_file() and p.suffix.lower() in (".pth", ".pt", ".ckpt", ".safetensors")]
    raw = list((root / "URFD").glob("**/*.zip"))
    return {
        "processed_skeleton_sequences": len([p for p in npys if not any(p.stem.endswith(x) for x in ("_anchor", "_frame_labels", "_neck_depth", "_raw_coco17", "_timestamps"))]),
        "neural_checkpoints": len(ckpts), "metric_neck_height_files": len(neck),
        "pose_depth_weight_files": len(weight_files), "raw_URFD_zips": len(raw),
        "can_train_neural_from_skeletons": len(npys) > 0,
        "can_infer_pretrained_neural": bool(ckpts),
        "can_export_symbolic_URFD": len(neck) >= 70,
        "can_redo_video_preprocessing": bool(weight_files) and bool(raw),
        "note": "Missing licensed/external model files must be obtained from official sources; no substitute or fabricated weights are generated."
    }


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", default=Path(__file__).resolve().parent)
    args = p.parse_args()
    print(json.dumps(check(args.root), indent=2))
