"""
symbolic_v2.py
Symbolic stream, version 2, built to work together with the neural ensemble (neural_ensemble.py).

What changed from the teammate's notebook (symbolic_fall_detector.ipynb) and why
  1. Causal everywhere. The floor reference, the body scale, the gap filling and the smoothing only use
     the current and past frames, like the ST-GCN. The notebook used percentiles over the whole video,
     centred smoothing and interpolation from future frames.
  2. Hip fallback without the upward offset. When the shoulders disappear on the floor, the notebook put
     the neck 0.35 torso lengths ABOVE the hip, which raises the height exactly while the person lies down.
     We use the hip midpoint itself, as our preprocessing does.
  3. Confidence threshold 0.3, as in the paper (the notebook used 0.25).
  4. Two outputs instead of one, because the two streams fail in different ways:
       trigger  = paper Eq. 8 AND Eq. 9 (neck low AND fast descent), smoothed causally (Eq. 10).
                  High precision, low recall: used in the OR of the fusion.
       support  = the person is in a fall state: it starts when the neck gets low within 1.5 s of a fast
                  descent and stays on while the neck stays low. Used as the veto of the fusion: the neural
                  stream raises many alarms while people stand, sit or lie down slowly; those frames have
                  no fall state and are vetoed.
  5. The support height threshold is not tuned for MCC. It is fixed by a rule on the training videos only:
     the 95th percentile of the height over training frames that URFD labels "lying on the ground" (label 1),
     so the fall state covers about 95% of lying frames by construction.

Height is still a relative body height (about 1 when standing), not metres: URFD has no control points
for the paper's matrix W. It follows the Kinect floor distance closely (median correlation 0.97 on falls).

Fusion (paper Eq. 11 to 13): fused(t) = (neural(t) OR trigger(t)) AND support anywhere in the last 80 frames.
"""
import numpy as np

FPS = 30.0
CONF = 0.3
L_SH, R_SH, L_HIP, R_HIP, L_ANK, R_ANK, NOSE = 5, 6, 11, 12, 15, 16, 0


def _mid(k, i, j):
    a, b = k[i], k[j]
    if a[2] >= CONF and b[2] >= CONF:
        return 0.5 * (a[:2] + b[:2])
    if a[2] >= CONF:
        return a[:2]
    if b[2] >= CONF:
        return b[:2]
    return None


def _ffill(x, max_gap=5):
    """Causal gap filling: carry the last valid value forward for at most max_gap frames."""
    x = x.copy(); last, gap = np.nan, 0
    for t in range(len(x)):
        if np.isnan(x[t]):
            gap += 1
            if gap <= max_gap:
                x[t] = last
        else:
            last, gap = x[t], 0
    return x


def causal_height(kpts, neck_depth, min_hist=15):
    """Relative neck height above the floor, using only frames <= t. ~1 standing, near 0 on the floor."""
    T = len(kpts)
    ny = np.full(T, np.nan); ay = np.full(T, np.nan)
    for t in range(T):
        p = _mid(kpts[t], L_SH, R_SH)
        if p is None:
            p = _mid(kpts[t], L_HIP, R_HIP)          # hip itself, no upward offset
        if p is None and kpts[t, NOSE, 2] >= CONF:
            p = kpts[t, NOSE, :2]
        if p is not None:
            ny[t] = p[1]
        a = _mid(kpts[t], L_ANK, R_ANK)
        if a is not None:
            ay[t] = a[1]
    ny = _ffill(ny); d = _ffill(np.asarray(neck_depth, dtype=float))
    h = np.full(T, np.nan)
    for t in range(T):
        if np.isnan(ny[t]):
            continue
        dpast = d[:t + 1][~np.isnan(d[:t + 1])]
        scale = np.clip(d[t] / np.median(dpast), 0.5, 2.0) if len(dpast) and not np.isnan(d[t]) and np.median(dpast) > 1e-6 else 1.0
        both = ~np.isnan(ay[:t + 1]) & ~np.isnan(ny[:t + 1])
        if both.sum() >= min_hist:
            span = np.percentile(ay[:t + 1][both] - ny[:t + 1][both], 85)
            floor = np.percentile(ay[:t + 1][both], 90)
        elif not np.isnan(ay[t]):                     # warm-up: the person is assumed upright at the start
            span = ay[t] - ny[t]; floor = ay[t]
        else:
            continue
        span = max(span, 40.0)
        h[t] = np.clip((floor - ny[t]) / (span * scale), -0.2, 1.5)
    return _ffill(h)


def descent_speed(h, lag_s=0.2):
    lag = max(1, int(round(lag_s * FPS)))
    v = np.zeros_like(h)
    v[lag:] = (h[:-lag] - h[lag:]) / (lag / FPS)
    return np.nan_to_num(v)


def trailing_majority(x, win):
    x = x.astype(int); c = np.cumsum(np.concatenate([[0], x])); out = np.zeros(len(x), int)
    for t in range(len(x)):
        lo = max(0, t - win + 1); out[t] = (c[t + 1] - c[lo]) >= (min(win, t + 1) // 2 + 1)
    return out


class SymbolicV2:
    def __init__(self, tau_h=0.35, tau_v=0.45, tau_s=0.5, smooth_s=0.25, lookback_s=1.5):
        self.tau_h, self.tau_v, self.tau_s = tau_h, tau_v, tau_s
        self.smooth, self.lookback = max(1, int(round(smooth_s * FPS))), int(round(lookback_s * FPS))

    def features(self, kpts, neck_depth):
        h = causal_height(kpts, neck_depth)
        return h, descent_speed(h)

    def predict(self, h, v):
        hn = np.nan_to_num(h, nan=1.0)                # unknown height counts as not low
        trigger = trailing_majority((hn < self.tau_h) & (v > self.tau_v), self.smooth)
        low = hn < self.tau_s; fast = v > self.tau_v
        state = np.zeros(len(h), int); on = False
        for t in range(len(h)):
            recent = fast[max(0, t - self.lookback):t + 1].any()
            on = low[t] and (on or recent)
            state[t] = on
        return dict(trigger=trigger, state=state, h=h, v=v)


def support_window(state, win=80):
    c = np.cumsum(np.concatenate([[0], state.astype(int)]))
    return np.array([(c[t + 1] - c[max(0, t - win + 1)]) > 0 for t in range(len(state))], int)


def fuse(neural_prob, sym, win=80, thr=0.5):
    """Paper Eq. 11 to 13: (neural OR trigger) AND symbolic support in the last `win` frames."""
    return (((neural_prob >= thr) | (sym["trigger"] == 1)) & (support_window(sym["state"], win) == 1)).astype(int)


def mcc(y, p):
    tp = int(((y == 1) & (p == 1)).sum()); tn = int(((y == 0) & (p == 0)).sum())
    fp = int(((y == 0) & (p == 1)).sum()); fn = int(((y == 1) & (p == 0)).sum())
    d = np.sqrt(float((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)))
    return (tp * tn - fp * fn) / d if d > 0 else 0.0


def fit(train_feats, lying_h, height_grid=np.linspace(0.10, 0.80, 15), velocity_grid=np.linspace(0.25, 2.0, 15)):
    """train_feats: list of (h, v, labels) for TRAINING videos only; lying_h: heights of the training frames
    URFD labels 'lying on the ground'. tau_h, tau_v: best trigger MCC (paper rule).
    tau_s: 95th percentile of lying_h."""
    Y = np.concatenate([y for _, _, y in train_feats])
    best = (-2, None, None)
    for th in height_grid:
        for tv in velocity_grid:
            m = SymbolicV2(th, tv)
            P = np.concatenate([m.predict(h, v)["trigger"] for h, v, _ in train_feats])
            s = mcc(Y, P)
            if s > best[0]:
                best = (s, th, tv)
    tau_s = float(np.percentile(lying_h, 95))
    return SymbolicV2(best[1], best[2], tau_s), best[0]
