"""
early_fall.py
Optional early-fall rule for the improved Symbolic V3 (reports/early_fall_rule/PREREGISTRATION.md).

V3 reacts once the neck is low. At fall onset the neck is still high, so frames at the start of a fall are missed
by V3 and by the neural ensemble alike. This rule looks at the torso in the image, which changes before the neck
gets low:
  B  torso inclination speed  w(t) = (theta(t) - theta(t-6)) / 0.2 s           [deg/s]
     theta = angle between hip-midpoint -> shoulder-midpoint and the image's upward vertical
  C  normalised hip descent   u(t) = (P_y(t) - P_y(t-6)) / Lref(t) / 0.2 s      [torso lengths/s]
     P = hip midpoint, Lref = median torso length over the past 2 s (camera- and person-scale free)
  D  B and C within the same 0.3 s
Trigger: feature above its threshold on 3 consecutive frames. Hold: the alarm stays on for 1.0 s.
Thresholds: 99th percentile of the feature over ADL frames of the fitting videos (no score search).

Only past and current frames are used. The rule uses image keypoints only, so it also runs on data without depth.
Use: V3WithEarly(v3_model, EarlyRule(...)) with early=None reproduces V3 exactly (the on/off flag).
"""
import numpy as np

FPS = 30.0
CONF = 0.3
LAG = 6                   # 0.2 s
CONSEC = 3                # 0.1 s
HOLD = 30                 # 1.0 s
PAIR_WIN = 9              # 0.3 s, for D
LREF_WIN = 60             # 2 s


def _mid(k, i, j):
    a, b = k[:, i], k[:, j]
    ok_a, ok_b = a[:, 2] >= CONF, b[:, 2] >= CONF
    out = np.full((len(k), 2), np.nan)
    both = ok_a & ok_b
    out[both] = 0.5 * (a[both, :2] + b[both, :2])
    out[ok_a & ~ok_b] = a[ok_a & ~ok_b, :2]
    out[~ok_a & ok_b] = b[~ok_a & ok_b, :2]
    return out


def features(kpts):
    """kpts: (T, 17, 3) COCO-17 [x, y, conf] in pixels. Returns dict of per-frame arrays (NaN where unknown)."""
    k = np.asarray(kpts, dtype=float)
    T = len(k)
    if T == 0:
        e = np.array([], dtype=float)
        return dict(theta=e, w=e, u=e)
    S, P = _mid(k, 5, 6), _mid(k, 11, 12)
    d = S - P
    L = np.hypot(d[:, 0], d[:, 1])
    theta = np.degrees(np.arctan2(np.abs(d[:, 0]), -d[:, 1]))     # 0 = upright, 90 = horizontal (y points down)
    lref = np.full(T, np.nan)
    for t in range(T):
        w = L[max(0, t - LREF_WIN + 1):t + 1]
        w = w[~np.isnan(w) & (w > 0)]
        if len(w):
            lref[t] = np.median(w)
    w_ = np.full(T, np.nan); u_ = np.full(T, np.nan)
    w_[LAG:] = (theta[LAG:] - theta[:-LAG]) / (LAG / FPS)
    u_[LAG:] = (P[LAG:, 1] - P[:-LAG, 1]) / lref[LAG:] / (LAG / FPS)
    return dict(theta=theta, w=w_, u=u_)


def _consecutive(mask, n=CONSEC):
    out = np.zeros(len(mask), bool); run = 0
    for t, m in enumerate(mask):
        run = run + 1 if m else 0
        out[t] = run >= n
    return out


def _hold(trig, n=HOLD):
    out = np.zeros(len(trig), int); left = 0
    for t, g in enumerate(trig):
        if g:
            left = n
        out[t] = 1 if left > 0 else 0
        left = max(0, left - 1)
    return out


def _recent(mask, n):
    c = np.concatenate([[0], np.cumsum(mask.astype(int))]); t = np.arange(len(mask))
    return (c[t + 1] - c[np.maximum(0, t - n + 1)]) > 0


class EarlyRule:
    def __init__(self, kind="C", thr_w=np.inf, thr_u=np.inf):
        assert kind in ("B", "C", "D")
        self.kind, self.thr_w, self.thr_u = kind, float(thr_w), float(thr_u)

    def config(self):
        return dict(kind=self.kind, thr_w_deg_per_s=round(self.thr_w, 3), thr_u_torso_per_s=round(self.thr_u, 3))

    def alarm(self, f):
        bw = _consecutive(np.nan_to_num(f["w"], nan=-np.inf) > self.thr_w)
        bu = _consecutive(np.nan_to_num(f["u"], nan=-np.inf) > self.thr_u)
        trig = bw if self.kind == "B" else bu if self.kind == "C" else (bw & _recent(bu, PAIR_WIN)) | (bu & _recent(bw, PAIR_WIN))
        return _hold(trig)


def fit(adl_features, kind):
    """Thresholds = 99th percentile of the feature over ADL frames of the fitting videos."""
    w = np.concatenate([f["w"] for f in adl_features]); u = np.concatenate([f["u"] for f in adl_features])
    w, u = w[np.isfinite(w)], u[np.isfinite(u)]
    return EarlyRule(kind, np.percentile(w, 99), np.percentile(u, 99))


class V3WithEarly:
    """The improved V3 with the optional early rule. early=None -> identical to V3."""
    def __init__(self, v3_model, early=None):
        self.v3, self.early = v3_model, early

    def predict(self, video, feats=None):
        p = self.v3.predict(video)
        if self.early is None:
            return p
        return (p | self.early.alarm(feats)).astype(int)
