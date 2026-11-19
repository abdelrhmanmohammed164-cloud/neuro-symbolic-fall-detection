# Preprocessed URFD skeletons

`cfg_d31bfed741ab/urfd_npy/` holds the output of `preprocess.py` for all 70 videos (the folder name is the
fingerprint of the preprocessing configuration). Per sequence:

| File | Content |
|---|---|
| `<seq>.npy` | float32 (3, T, 14, 1): x/W, y/H, confidence (0 = missing joint) |
| `<seq>_frame_labels.npy` | int (T,): 1 = fall frame (URFD label 0 or 1 inside a fall video) |
| `<seq>_raw_coco17.npy` | raw RTMO keypoints |
| `<seq>_anchor.npy` | anchor joint used for depth: neck, hip fallback or none |
| `<seq>_neck_depth.npy` | Depth-Anything-V2 relative inverse depth at the anchor |
| `<seq>_timestamps.npy` | frame timestamps from the URFD synchronisation file |
| `<seq>_meta.json` | settings and the SHA-256 of the source archive |

Derived from URFD (CC BY-NC-SA 4.0): non-commercial research use, same licence.
