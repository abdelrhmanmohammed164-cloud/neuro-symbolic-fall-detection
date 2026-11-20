"""
symbolic_v4.py
Symbolic stream v4: a small, explicit state machine on the neck height above the floor in metres
(processed_data/kinect_height, made by kinect_neck_height.py), with confidence calibrated on training frames.

Why v4 (problems found in v3, see reports/symbolic_v4/report.md)
  - v3's rule needs "low" and "fast" at the SAME frame. A falling neck is fast while still high and slow once
    on the floor, so the grid search pushed tau_h to 0.75 m (not "near the floor" any more) and recall stayed low.
  - v3's fall state needed the neck to be low already, so it missed every 'falling' frame by construction.
  - v3 gave hard labels only and its thresholds came from a 225-point MCC grid on training videos.

Rules (all causal: frame t only uses frames <= t). States are checked in this priority order, which is the
conflict-resolution order: the first state whose condition holds is the state of the frame.
  UNKNOWN     the height has been missing for more than 5 frames (after the 5-frame carry-forward).
  FALL        entered when the neck is low (h < tau_low) and it dropped by at least delta_fall within the last
              1.5 s (a fast drop to the floor); held while the neck stays low, or is missing for at most
              `unknown_hold` frames. Slow lowering (lying down on purpose) never enters FALL because its drop
              within 1.5 s stays below delta_fall.
  DESCENDING  the neck dropped by at least delta_on within the last 1 s and is still moving down faster than
              0.3 m/s (covers the falling phase before the body reaches the floor).
  UPRIGHT     h >= 0.8 m: a neck this high cannot belong to someone lying on the floor (domain constant).
  LOW_NOFALL  low without a preceding fast drop.
  MID         everything else.

Thresholds come from TRAINING labels only (no MCC search):
  tau_low    = 95th percentile of the neck height on training frames URFD labels 'lying' (label 1);
  delta_fall = smallest drop into lying among training falls (max height over the 1.5 s before the first
               lying frame minus the height at that frame), minus a 0.05 m margin;
  delta_on   = delta_fall / 2.

Outputs per frame: state (int code), hard (FALL or DESCENDING), p (calibrated probability: fall rate of the
training frames in the same state, Laplace smoothed). `p` is a frequency on training rooms, not a promise
of calibration on new rooms; check it on validation before trusting it in a fusion.
"""
import numpy as np

from symbolic_v3 import FPS, descent_speed, ffill, load_height, mcc, support_window  # noqa: F401  (re-exported)

STATES = ("UNKNOWN", "FALL", "DESCENDING", "UPRIGHT", "LOW_NOFALL", "MID")
S = {n: i for i, n in enumerate(STATES)}
UPRIGHT_M = 0.8


def window_drop(h, win):
    """max(h over the last `win` frames, current included) - h(t), ignoring missing values."""
    out = np.full(len(h), np.nan)
    for t in range(len(h)):
        w = h[max(0, t - win + 1):t + 1]
        if not np.isnan(h[t]) and (~np.isnan(w)).any():
            out[t] = np.nanmax(w) - h[t]
    return out


class SymbolicV4:
    def __init__(self, tau_low=0.40, delta_fall=0.45, delta_on=None, use_descending=True, unknown_hold=30,
                 state_p=None):
        self.tau_low, self.delta_fall = float(tau_low), float(delta_fall)
        self.delta_on = float(delta_fall / 2 if delta_on is None else delta_on)
        self.use_descending, self.unknown_hold = use_descending, int(unknown_hold)
        self.state_p = state_p                     # dict state code -> calibrated probability

    def config(self):
        return dict(tau_low=self.tau_low, delta_fall=self.delta_fall, delta_on=self.delta_on,
                    use_descending=self.use_descending, unknown_hold=self.unknown_hold,
                    state_p=None if self.state_p is None else {STATES[k]: v for k, v in self.state_p.items()})

    def states(self, h):
        h = np.asarray(h, dtype=float)
        v = descent_speed(h)
        drop15, drop10 = window_drop(h, int(1.5 * FPS)), window_drop(h, int(1.0 * FPS))
        st = np.full(len(h), S["MID"]); in_fall, missing = False, 0
        for t in range(len(h)):
            if np.isnan(h[t]):
                missing += 1
                if in_fall and missing <= self.unknown_hold:
                    st[t] = S["FALL"]
                else:
                    in_fall = False; st[t] = S["UNKNOWN"]
                continue
            missing = 0
            low = h[t] < self.tau_low
            if low and (in_fall or (not np.isnan(drop15[t]) and drop15[t] >= self.delta_fall)):
                in_fall = True; st[t] = S["FALL"]; continue
            in_fall = False
            if self.use_descending and not np.isnan(drop10[t]) and drop10[t] >= self.delta_on and v[t] > 0.3:
                st[t] = S["DESCENDING"]
            elif h[t] >= UPRIGHT_M:
                st[t] = S["UPRIGHT"]
            elif low:
                st[t] = S["LOW_NOFALL"]
        return st

    def predict(self, h):
        st = self.states(h)
        hard = np.isin(st, [S["FALL"], S["DESCENDING"]]).astype(int)
        p = None if self.state_p is None else np.array([self.state_p[int(s)] for s in st])
        fall_like = hard
        return dict(state=st, hard=hard, p=p, support80=support_window(fall_like), upright=(st == S["UPRIGHT"]).astype(int))


def drop_into_lying(h, lab3):
    """Drop (m) into the first frame URFD labels 'lying' (label 1), from the max over the previous 1.5 s."""
    idx = np.where(lab3 == 1)[0]
    if not len(idx) or np.isnan(h[idx[0]]):
        return None
    t = idx[0]
    return float(np.nanmax(h[max(0, t - int(1.5 * FPS)):t + 1]) - h[t])


def fit(train, use_descending=True, unknown_hold=30):
    """train: list of (h, binary labels, URFD 3-class labels) for TRAINING videos only. The 3-class labels
    must be -1 for ADL videos (URFD's ADL file uses 0/1 for postures such as lying on a bed, not for falls)."""
    lying = np.concatenate([h[(l3 == 1) & ~np.isnan(h)] for h, _, l3 in train])
    drops = [d for d in (drop_into_lying(h, l3) for h, _, l3 in train) if d is not None]
    tau_low = float(np.percentile(lying, 95))
    delta_fall = max(0.10, float(min(drops)) - 0.05)
    m = SymbolicV4(tau_low, delta_fall, use_descending=use_descending, unknown_hold=unknown_hold)
    st = np.concatenate([m.states(h) for h, _, _ in train]); y = np.concatenate([y for _, y, _ in train])
    m.state_p = {k: float((y[st == k].sum() + 1) / ((st == k).sum() + 2)) for k in range(len(STATES))}
    return m
