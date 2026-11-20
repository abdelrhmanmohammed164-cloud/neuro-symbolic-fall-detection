"""
symbolic_v3_norm.py
V3-norm: the improved V3 (variants C and D, symbolic_v3_improved.py, unchanged) reading a different signal:
  - causal running median (7 frames) on the neck height before speed, drop and height are computed;
  - height divided by a causal estimate of the person's standing neck height (90th percentile of the valid
    smoothed heights so far, clipped to [1.20, 1.70] m, 1.45 m until 10 valid frames exist).
See reports/symbolic_v3_norm/PREREGISTRATION.md. Ablation flags: smooth=False or relative=False.
"""
import numpy as np
import symbolic_v3_improved as V

REF = 1.45                       # typical standing neck height on URFD (m), used only to rescale V3's grids
MEDIAN_WIN = 7
STAND_PCT, STAND_MIN_VALID, STAND_CLIP = 90, 10, (1.20, 1.70)


def causal_median(x, win=MEDIAN_WIN):
    from numpy.lib.stride_tricks import sliding_window_view
    if len(x) == 0:
        return np.array([], dtype=float)
    w = sliding_window_view(np.concatenate([np.full(win - 1, np.nan), x]), win)
    with np.errstate(all="ignore"):
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            out = np.nanmedian(w, 1)
    out[np.isnan(x)] = np.nan                      # a missing frame stays missing
    return out


def standing_height(h):
    out = np.full(len(h), REF); vals = []
    for t, x in enumerate(h):
        if np.isfinite(x):
            vals.append(x)
        if len(vals) >= STAND_MIN_VALID:
            out[t] = np.clip(np.percentile(vals, STAND_PCT), *STAND_CLIP)
    return out


class NormVideo:
    """Same attributes V3Improved.predict reads (h, hn, v, vmax, drop, y, lying)."""
    def __init__(self, h_raw, y=None, lying=None, smooth=True, relative=True):
        h = V.ffill(h_raw)
        if smooth:
            h = causal_median(h)
        if relative:
            h = h / standing_height(h)
        else:
            h = h / REF                            # same units as the relative version, absolute scale
        self.h = h
        self.hn = np.nan_to_num(h, nan=V.MISSING_AS)
        self.v = V.descent_speed(h)
        self.vmax = V.trailing_max(self.v, V.DESCENT_WINDOW)
        self.drop = np.nan_to_num(V.trailing_max(h, V.DROP_WINDOW) - h, nan=0.0)
        self.y = None if y is None else np.asarray(y).astype(int)
        self.lying = lying


HEIGHT_GRID = V.HEIGHT_GRID / REF
VELOCITY_GRID = V.VELOCITY_GRID / REF
FLOOR_CAP = V.FLOOR_CAP / REF
DELTAS = tuple(d / REF for d in (0.30, 0.40, 0.50))


def fit(videos, variant="C", delta=0.30 / REF):
    lying = [vd.hn[vd.lying] for vd in videos if vd.lying is not None and vd.lying.any()]
    tau_s = float(np.percentile(np.concatenate(lying), 95)) if lying else 0.45 / REF
    hgrid = HEIGHT_GRID[HEIGHT_GRID <= FLOOR_CAP + 1e-9]
    best = (-2.0, None, None)
    for th in hgrid:
        for tv in VELOCITY_GRID:
            s = V._score(V.V3Improved(variant, th, tv, delta, tau_s), videos)
            if s > best[0]:
                best = (s, float(th), float(tv))
    return V.V3Improved(variant, best[1], best[2], delta, tau_s), best[0]
