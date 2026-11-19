# Model weights (not in git)

Put these files here; `preprocess.py` finds them by these names.

| File | Source |
|---|---|
| `rtmo-l_onnx/end2end.onnx` | Unzip `rtmo-l_16xb16-600e_body7-640x640-b37118ce_20231211.zip` from OpenMMLab (the RTMO-l body7 ONNX export used by rtmlib): https://download.openmmlab.com/mmpose/v1/projects/rtmo/onnx_sdk/rtmo-l_16xb16-600e_body7-640x640-b37118ce_20231211.zip |
| `depth_anything_v2_vits.pth` | Depth-Anything-V2 Small: https://huggingface.co/depth-anything/Depth-Anything-V2-Small (file `depth_anything_v2_vits.pth`) |
| `depth_anything_v2_vitb.pth` (optional) | Depth-Anything-V2 Base: https://huggingface.co/depth-anything/Depth-Anything-V2-Base |

The Depth-Anything-V2 model code is also needed:

```bash
git clone https://github.com/DepthAnything/Depth-Anything-V2 fall_detection_models/code/Depth-Anything-V2
```

The preprocessing run records the SHA-256 of both weight files in its manifest, so a different file is detected.
