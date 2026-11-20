"""
symbolic_v3.py
The symbolic stream (paper Eq. 8 to 10) on the neck height above the floor IN METRES.

Height: kinect_neck_height.py writes processed_data/kinect_height/<video>.npz. The floor plane comes from the
URFD Kinect depth (RANSAC), the neck point from our RTMO keypoints back-projected with the Kinect intrinsics.
Its correlation with URFD's own Kinect floor distance is 0.988 on falls and 0.929 on ADL (median per video).

Everything is causal (only frames <= t are used), like the ST-GCN.

Outputs per frame
  trigger : neck below tau_h (m) AND descending faster than tau_v (m/s) (Eq. 8 and 9), trailing majority
            smoothing over 0.25 s (Eq. 10).
  state   : fall state. Starts when the neck is below tau_s within 1.5 s of a fast descent, stays on while the
            neck stays below tau_s. Meant as the veto signal of the fusion (Eq. 11 to 13).

Fitting (training videos only)
  tau_h, tau_v : grid search for the best trigger MCC (tau_h 0.10 to 0.80 m, tau_v 0.25 to 2.5 m/s).
  tau_s        : 95th percentile of the height on training frames URFD labels 'lying on the ground'.

History: v1 was the teammate's notebook, v2 used a relative height measured in the image. v3 replaced both
(project decision; see reports/symbolic_v3/results.md for the validation and test numbers of v2 and v3).
"""
import os

import numpy as np

FPS = 30.0
HEIGHT_GRID = np.linspace(0.10, 0.80, 15)
VELOCITY_GRID = np.linspace(0.25, 2.5, 15)


def ffill(x, max_gap=5):
    """Causal gap filling: carry the last valid value forward for at most max_gap frames."""
    x = np.asarray(x, dtype=float).copy(); last, gap = np.nan, 0
    for t in range(len(x)):
        if np.isnan(x[t]):
            gap += 1
            if gap <= max_gap:
                x[t] = last
        else:
            last, gap = x[t], 0
    return x


def load_height(kinect_dir, vid):
    return ffill(np.load(os.path.join(kinect_dir, f"{vid}.npz"))["height_m"])


def descent_speed(h, lag_s=0.2):
    """Downward speed in m/s over lag_s, using the past only."""
    lag = max(1, int(round(lag_s * FPS)))
    v = np.zeros_like(h)
    v[lag:] = (h[:-lag] - h[lag:]) / (lag / FPS)
    return np.nan_to_num(v)


def trailing_majority(x, win):
    x = np.asarray(x).astype(int); c = np.cumsum(np.concatenate([[0], x])); out = np.zeros(len(x), int)
    for t in range(len(x)):
        lo = max(0, t - win + 1)
        out[t] = (c[t + 1] - c[lo]) >= (min(win, t + 1) // 2 + 1)
    return out


def support_window(state, win=80):
    """1 where the fall state was on at least once in the last `win` frames (paper Eq. 12 and 13)."""
    c = np.cumsum(np.concatenate([[0], np.asarray(state).astype(int)]))
    return np.array([(c[t + 1] - c[max(0, t - win + 1)]) > 0 for t in range(len(state))], int)


class SymbolicV3:
    def __init__(self, tau_h=0.19, tau_v=1.0, tau_s=0.45, smooth_s=0.25, lookback_s=1.5):
        self.tau_h, self.tau_v, self.tau_s = tau_h, tau_v, tau_s
        self.smooth = max(1, int(round(smooth_s * FPS)))
        self.lookback = int(round(lookback_s * FPS))

    def predict(self, h):
        v = descent_speed(h)
        hn = np.nan_to_num(h, nan=2.0)                 # unknown height counts as not low
        trigger = trailing_majority((hn < self.tau_h) & (v > self.tau_v), self.smooth)
        low, fast = hn < self.tau_s, v > self.tau_v
        state = np.zeros(len(h), int); on = False
        for t in range(len(h)):
            on = bool(low[t] and (on or fast[max(0, t - self.lookback):t + 1].any()))
            state[t] = on
        return dict(trigger=trigger, state=state, support80=support_window(state), h=h, v=v)


def mcc(y, p):
    y, p = np.asarray(y), np.asarray(p)
    tp = int(((y == 1) & (p == 1)).sum()); tn = int(((y == 0) & (p == 0)).sum())
    fp = int(((y == 0) & (p == 1)).sum()); fn = int(((y == 1) & (p == 0)).sum())
    d = np.sqrt(float((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)))
    return (tp * tn - fp * fn) / d if d > 0 else 0.0


def fit(train, lying_heights, height_grid=HEIGHT_GRID, velocity_grid=VELOCITY_GRID):
    """train: list of (h, labels) for TRAINING videos only; lying_heights: heights of training frames URFD
    labels 'lying'. Returns (model, training trigger MCC)."""
    Y = np.concatenate([y for _, y in train])
    best = (-2.0, None, None)
    for th in height_grid:
        for tv in velocity_grid:
            m = SymbolicV3(th, tv)
            s = mcc(Y, np.concatenate([m.predict(h)["trigger"] for h, _ in train]))
            if s > best[0]:
                best = (s, float(th), float(tv))
    return SymbolicV3(best[1], best[2], float(np.percentile(lying_heights, 95))), best[0]


def fuse(neural_prob, sym, thr=0.5):
    """Paper Eq. 11 to 13: (neural OR trigger) AND fall-state support in the last 80 frames."""
    return (((np.asarray(neural_prob) >= thr) | (sym["trigger"] == 1)) & (sym["support80"] == 1)).astype(int)
