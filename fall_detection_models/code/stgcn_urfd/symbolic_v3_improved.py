"""
symbolic_v3_improved.py
Symbolic V3 with better precision, built directly on V3 (symbolic_v3.py, kept untouched as the baseline;
symbolic_v3_original.py is a byte-identical backup).

V3 structure kept: Eq. 8 (neck below tau_h), Eq. 9 (descent faster than tau_v), Eq. 10 (trailing majority over
0.25 s), thresholds chosen on training videos by MCC over the same grids, neck height in metres.

Variants (see reports/symbolic_v3_precision/PREREGISTRATION.md)
  A  original V3: Eq. 8 AND Eq. 9 at the same frame.
  P  paper reproduction: tau_h fixed at 0.19 m.
  B  Eq. 9 checked over the last 1.0 s (a falling neck is fast while high, slow once on the floor).
  C  B + negative evidence: drop of at least `delta` m within the last 1.5 s, and tau_h <= 0.60 m.
  D  C + latch: after a trigger the output stays on while the neck stays below tau_s.

Everything is causal. Missing heights never count as low (they are treated as 2.0 m).
The functions are vectorised versions of V3's; tests_symbolic_v3_improved.py checks that variant A gives the
same thresholds and predictions as symbolic_v3.py.
"""
import numpy as np

FPS = 30.0
HEIGHT_GRID = np.linspace(0.10, 0.80, 15)          # V3's grids
VELOCITY_GRID = np.linspace(0.25, 2.5, 15)
SMOOTH = 8                                         # round(0.25 s * 30)
DESCENT_WINDOW = 30                                # 1.0 s, variants B to D
DROP_WINDOW = 45                                   # 1.5 s, variants C and D
FLOOR_CAP = 0.60                                   # largest tau_h considered "near the floor" (C, D)
MISSING_AS = 2.0


def ffill(x, max_gap=5):
    x = np.asarray(x, dtype=float).copy(); last, gap = np.nan, 0
    for t in range(len(x)):
        if np.isnan(x[t]):
            gap += 1
            if gap <= max_gap:
                x[t] = last
        else:
            last, gap = x[t], 0
    return x


def descent_speed(h, lag=6):
    v = np.zeros_like(h); v[lag:] = (h[:-lag] - h[lag:]) / (lag / FPS)
    return np.nan_to_num(v)


def trailing_max(x, win):
    """max over the last `win` values, current included (NaN ignored; all-NaN -> NaN)."""
    from numpy.lib.stride_tricks import sliding_window_view
    if len(x) == 0:
        return np.array([], dtype=float)
    pad = np.concatenate([np.full(win - 1, np.nan), x])
    with np.errstate(all="ignore"):
        w = sliding_window_view(pad, win)
        out = np.where(np.all(np.isnan(w), 1), np.nan, np.nanmax(np.where(np.isnan(w), -np.inf, w), 1))
    return out


def trailing_majority(x, win=SMOOTH):
    x = np.asarray(x).astype(int); c = np.concatenate([[0], np.cumsum(x)])
    t = np.arange(len(x)); lo = np.maximum(0, t - win + 1)
    return ((c[t + 1] - c[lo]) >= (np.minimum(win, t + 1) // 2 + 1)).astype(int)


class Video:
    """Per-video features, computed once."""
    def __init__(self, h_raw, y=None, lying=None):
        self.h = ffill(h_raw)
        self.hn = np.nan_to_num(self.h, nan=MISSING_AS)
        self.v = descent_speed(self.h)
        self.vmax = trailing_max(self.v, DESCENT_WINDOW)
        hm = trailing_max(self.h, DROP_WINDOW)
        self.drop = np.nan_to_num(hm - self.h, nan=0.0)
        self.y = None if y is None else np.asarray(y).astype(int)
        self.lying = lying                         # bool mask of frames URFD labels 'lying' (fall videos), or None


class V3Improved:
    def __init__(self, variant="A", tau_h=0.75, tau_v=0.41, delta=0.0, tau_s=0.45):
        self.variant, self.tau_h, self.tau_v, self.delta, self.tau_s = variant, float(tau_h), float(tau_v), float(delta), float(tau_s)

    def config(self):
        return dict(variant=self.variant, tau_h=round(self.tau_h, 4), tau_v=round(self.tau_v, 4),
                    delta=self.delta, tau_s=round(self.tau_s, 4))

    def conditions(self, vid):
        low = vid.hn < self.tau_h                                  # Eq. 8
        fast_now = vid.v > self.tau_v                              # Eq. 9, same frame (A, P)
        fast_recent = vid.vmax > self.tau_v                        # Eq. 9 over the last 1 s (B to D)
        dropped = vid.drop >= self.delta                           # negative evidence (C, D)
        return dict(low=low, fast_now=fast_now, fast_recent=np.nan_to_num(fast_recent).astype(bool), dropped=dropped)

    def predict(self, vid):
        c = self.conditions(vid)
        if self.variant in ("A", "P"):
            raw = c["low"] & c["fast_now"]
        elif self.variant == "B":
            raw = c["low"] & c["fast_recent"]
        else:
            raw = c["low"] & c["fast_recent"] & c["dropped"]
        trig = trailing_majority(raw)                              # Eq. 10
        if self.variant != "D":
            return trig
        out = trig.copy(); on = False; below = vid.hn < self.tau_s
        for t in range(len(out)):
            on = bool(trig[t] or (on and below[t]))
            out[t] = on
        return out


def mcc_counts(tp, fp, fn, tn):
    d = np.sqrt(float((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)))
    return (tp * tn - fp * fn) / d if d > 0 else 0.0


def _score(model, videos):
    tp = fp = fn = tn = 0
    for vd in videos:
        p = model.predict(vd); y = vd.y
        tp += int(((p == 1) & (y == 1)).sum()); fp += int(((p == 1) & (y == 0)).sum())
        fn += int(((p == 0) & (y == 1)).sum()); tn += int(((p == 0) & (y == 0)).sum())
    return mcc_counts(tp, fp, fn, tn)


def fit(videos, variant="A", delta=0.0):
    """Thresholds from TRAINING videos only, V3's way: best MCC over the grid (first maximum kept)."""
    lying = [vd.hn[vd.lying] for vd in videos if vd.lying is not None and vd.lying.any()]
    tau_s = float(np.percentile(np.concatenate(lying), 95)) if lying else 0.45
    hgrid = np.array([0.19]) if variant == "P" else (HEIGHT_GRID[HEIGHT_GRID <= FLOOR_CAP + 1e-9] if variant in ("C", "D") else HEIGHT_GRID)
    best = (-2.0, None, None)
    for th in hgrid:
        for tv in VELOCITY_GRID:
            s = _score(V3Improved(variant, th, tv, delta, tau_s), videos)
            if s > best[0]:
                best = (s, float(th), float(tv))
    return V3Improved(variant, best[1], best[2], delta, tau_s), best[0]
