"""tests_early_fall.py: checks for the optional early-fall rule. Run: python3 tests_early_fall.py"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import early_fall as EF  # noqa: E402
import symbolic_v3_improved as V  # noqa: E402

res = []


def check(n, c, d=""):
    res.append(bool(c)); print(("PASS " if c else "FAIL ") + n + (f"  ({d})" if d else ""))


def person(T=150, tilt_start=None, tilt_frames=15, drop=False, scale=1.0, x0=320):
    """Synthetic COCO-17 skeleton: shoulders above hips, torso length 100*scale px; optional fast tilt and hip drop."""
    k = np.zeros((T, 17, 3)); k[:, :, 2] = 0.9
    for t in range(T):
        ang = 0.0
        if tilt_start is not None and t >= tilt_start:
            ang = min(80.0, 80.0 * (t - tilt_start) / tilt_frames)
        hip_y = 300.0 + (min(120.0, 120.0 * (t - tilt_start) / tilt_frames) if (drop and tilt_start is not None and t >= tilt_start) else 0.0)
        hx, hy = x0, hip_y * scale
        sx, sy = hx + 100 * scale * np.sin(np.radians(ang)), hy - 100 * scale * np.cos(np.radians(ang))
        k[t, 11, :2] = k[t, 12, :2] = (hx, hy); k[t, 5, :2] = k[t, 6, :2] = (sx, sy)
    return k


f_stand = EF.features(person())
check("upright torso -> theta about 0", np.nanmax(np.abs(f_stand["theta"])) < 1e-6)
f_fall = EF.features(person(tilt_start=60, drop=True))
check("fast tilt (80 deg in 0.5 s) -> inclination speed about 160 deg/s", abs(np.nanmax(f_fall["w"]) - 160) < 1, f"{np.nanmax(f_fall['w']):.1f}")
check("hip drop -> positive normalised descent", np.nanmax(f_fall["u"]) > 1.0)
f_big = EF.features(person(tilt_start=60, drop=True, scale=2.0))
check("hip descent is scale-free (same in torso lengths for a 2x larger person)", abs(np.nanmax(f_big["u"]) - np.nanmax(f_fall["u"])) < 0.05,
      f"{np.nanmax(f_big['u']):.2f} vs {np.nanmax(f_fall['u']):.2f}")
r = EF.EarlyRule("C", thr_w=100, thr_u=1.0)
a = r.alarm(f_fall)
check("alarm fires during the fall", a[60:80].any())
check("alarm holds at most 1 s after the last trigger", a.sum() <= EF.HOLD + 20)
check("no alarm while standing", not r.alarm(f_stand).any())
k = person(); k[70:72, 11:13, 1] += 200        # 2-frame keypoint glitch
check("2-frame keypoint jump does not trigger (3 consecutive frames needed)", not EF.EarlyRule("C", 100, 1.0).alarm(EF.features(k)).any())
k = person(tilt_start=60, drop=True); k2 = k.copy(); k2[100:] = k2[50]
check("causal", (EF.EarlyRule("D", 100, 1.0).alarm(EF.features(k))[:100] == EF.EarlyRule("D", 100, 1.0).alarm(EF.features(k2))[:100]).all())
k = person(); k[:, :, 2] = 0.1
check("missing keypoints -> no alarm", not EF.EarlyRule("B", 10, 0.1).alarm(EF.features(k)).any())
check("empty video", len(EF.EarlyRule("B", 10, 0.1).alarm(EF.features(np.zeros((0, 17, 3))))) == 0)
# off flag reproduces V3 exactly
vid = V.Video(np.r_[np.full(60, 1.35), np.linspace(1.35, 0.3, 12), np.full(78, 0.3)])
m = V.V3Improved("C", 0.4, 0.7, 0.4, 0.44)
check("early=None reproduces V3", (EF.V3WithEarly(m, None).predict(vid) == m.predict(vid)).all())
check("early rule only adds positives", (EF.V3WithEarly(m, r).predict(vid, f_fall) >= m.predict(vid)).all())
src = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "early_fall.py")).read()
check("no V4 code and no notebook", "symbolic_v4" not in src and ".ipynb" not in src)
n = res.count(False); print(f"\n{len(res) - n} passed, {n} failed"); sys.exit(1 if n else 0)
