"""
dataset.py
URFD skeleton windows for the ST-GCN neural stream.

Files in data_dir (one sequence per file):
    <seq>.npy                float32 (C, T_full, V, M): x/W, y/H, conf (conf = 0 -> joint missing)
    <seq>_frame_labels.npy   int (T_full,), 1 = fall frame (from URFD labels 0/1 in fall videos)
Sequence label: 1 if the name starts with "fall".

Window convention (causal, matches the per-frame score N_t of Eq. 6):
    the window for frame t covers frames [t-W+1, t]; it is right-padded with frame t when fewer
    than W real frames exist (paper: "padded with the last valid observation").
Window label = label of its last real frame. A window whose last frame is a fall frame but
which contains fewer than `min_pos` fall frames is ambiguous (fall onset): weight 0 in training.

Padding shortcut fix (audit finding): only short clips need padding, and 11 of the 12 clips
shorter than 80 frames are falls. Training therefore applies the same truncate-and-pad operation
to a fixed fraction of windows of BOTH classes (p_pad), so padding stops carrying label information.

All randomness of a training epoch comes from rng = default_rng([seed, epoch]), so resuming at
an epoch boundary reproduces exactly the same windows.
"""
import csv
import os

import numpy as np
import torch
from torch.utils.data import Dataset

from graph import CENTER, FLIP_PAIRS

AUX_SUFFIXES = ("_frame_labels", "_neck_depth", "_raw_coco17", "_anchor", "_timestamps")


def is_sequence_file(fn):
    return fn.endswith(".npy") and not fn[:-4].endswith(AUX_SUFFIXES)


def load_sequences(data_dir, labels_csv=None, names=None):
    override = {}
    if labels_csv:
        with open(labels_csv) as f:
            for row in csv.DictReader(f):
                override[row["sequence"]] = int(row["label"])
    seqs = []
    for fn in sorted(os.listdir(data_dir)):
        if not is_sequence_file(fn):
            continue
        name = fn[:-4]
        if names is not None and name not in names:
            continue
        data = np.load(os.path.join(data_dir, fn), allow_pickle=False).astype(np.float32)
        if data.ndim == 3:
            data = data[..., None]
        assert data.ndim == 4, f"{fn}: expected (C, T, V, M), got {data.shape}"
        fl_path = os.path.join(data_dir, name + "_frame_labels.npy")
        fl = np.load(fl_path).astype(np.int64) if os.path.exists(fl_path) else None
        if fl is not None:
            assert len(fl) == data.shape[1], f"{name}: frame labels length != T"
        label = override.get(name, int(name.lower().startswith("fall")))
        seqs.append(dict(name=name, data=data, label=label, frame_labels=fl))
    if not seqs:
        raise FileNotFoundError(f"no .npy sequences in {data_dir}")
    return seqs


def missing_mask(x):
    """x (C, T, V, M) -> bool (T, V, M), True where the joint is missing."""
    if x.shape[0] >= 3:
        return x[2] <= 0
    return (x[0] == 0) & (x[1] == 0)


def window_by_end(x, end, window, n_real=None, pad_side="right"):
    """Frames [end-n_real+1, end] (n_real defaults to window).
    pad_side "right" (original, paper wording): real frames first, then copies of frame `end`.
    pad_side "left" (audit experiment): copies of the FIRST real frame first, so the current frame
    is always the last column of the window."""
    n_real = window if n_real is None else n_real
    s = max(0, end - n_real + 1)
    seg = x[:, s:end + 1]
    if seg.shape[1] < window:
        k = window - seg.shape[1]
        if pad_side == "left":
            seg = np.concatenate([np.repeat(seg[:, :1], k, axis=1), seg], axis=1)
        else:
            seg = np.concatenate([seg, np.repeat(seg[:, -1:], k, axis=1)], axis=1)
    return seg, s


def crop_window(x, start, window):
    """Kept for check_sequence.py: frames [start, start+window), right-padded."""
    seg = x[:, start:start + window]
    if seg.shape[1] < window:
        seg = np.concatenate([seg, np.repeat(seg[:, -1:], window - seg.shape[1], axis=1)], axis=1)
    return seg


def label_and_weight(seq, start, end, min_pos):
    """Label of the last real frame; weight 0 for ambiguous fall onsets."""
    fl = seq["frame_labels"]
    if fl is None:
        return seq["label"], 1.0
    y = int(fl[end])
    if y == 1 and fl[start:end + 1].sum() < min_pos:
        return 1, 0.0
    return y, 1.0


def normalize(x, mode):
    """'image' (paper): keep x/W, y/H. 'window_center': subtract the window's mean Neck position
    (keeps the downward motion inside the window). Fixed transforms, nothing is fitted on data."""
    if mode == "image":
        return x
    x = x.copy()
    miss = missing_mask(x)
    neck_ok = ~miss[:, CENTER, :]
    for d in (0, 1):
        ref = x[d, :, CENTER, :][neck_ok].mean() if neck_ok.any() else 0.5
        x[d] = np.where(miss, 0.0, x[d] - ref)
    return x


class SkeletonAugment:
    """x/y only (conf untouched). Missing joints stay exactly 0."""

    def __init__(self, p_flip=0.5, scale=(0.9, 1.1), shift=0.05, jitter=0.005, mode="image"):
        self.p_flip, self.scale, self.shift, self.jitter, self.mode = p_flip, scale, shift, jitter, mode

    def __call__(self, x, rng):
        x = x.copy()
        cx = 0.5 if self.mode == "image" else 0.0
        if rng.random() < self.p_flip:
            miss = missing_mask(x)
            x[0] = np.where(miss, 0.0, 2 * cx - x[0])
            for a, b in FLIP_PAIRS:
                x[:, :, [a, b]] = x[:, :, [b, a]]
        miss = missing_mask(x)
        s = rng.uniform(*self.scale)
        for d in (0, 1):
            x[d] = (x[d] - cx) * s + cx + rng.uniform(-self.shift, self.shift)
        x[:2] += rng.normal(0, self.jitter, size=x[:2].shape).astype(np.float32)
        x[:2] = np.where(miss[None], 0.0, x[:2])
        return x


class TrainWindows(Dataset):
    """`windows_per_seq` random windows per training sequence per epoch. Call set_epoch(e) first."""

    def __init__(self, seqs, window=80, windows_per_seq=64, min_pos=10, norm_mode="image",
                 augment=True, p_pad=0.35, min_real=40, p_fall_focus=0.5, seed=0, pad_side="right",
                 pad_balance=False):
        self.seqs, self.window, self.wps, self.min_pos = seqs, window, windows_per_seq, min_pos
        self.pad_side, self.pad_balance = pad_side, pad_balance
        self.norm_mode, self.p_pad, self.min_real, self.p_focus, self.seed = norm_mode, p_pad, min_real, p_fall_focus, seed
        self.aug = SkeletonAugment(mode=norm_mode) if augment else None
        self.set_epoch(0)

    def set_epoch(self, epoch):
        """Pre-draws every window of the epoch from rng([seed, epoch]) -> exact resume."""
        if self.pad_balance:
            return self._set_epoch_balanced(epoch)
        rng = np.random.default_rng([self.seed, epoch])
        self.items = []
        for si, seq in enumerate(self.seqs):
            T = seq["data"].shape[1]
            fl = seq["frame_labels"]
            for _ in range(self.wps):
                if fl is not None and fl.any() and rng.random() < self.p_focus:
                    end = int(rng.choice(np.nonzero(fl)[0]))          # window ends inside the fall
                else:
                    end = int(rng.integers(0, T))
                n_real = self.window
                if rng.random() < self.p_pad:                           # padding for both classes
                    n_real = int(rng.integers(self.min_real, self.window))
                self.items.append((si, end, n_real, int(rng.integers(2 ** 31))))

    def _set_epoch_balanced(self, epoch):
        """Audit experiment (--pad_balance): same window ends as the sampler above, but the extra
        padding probability is chosen PER CLASS (class = label of the end frame) so that, inside
        this fold's training set, the fraction of padded windows is the same for fall and non-fall.
        Natural padding (end < window-1) cannot be removed, so the other class is padded up to it."""
        rng = np.random.default_rng([self.seed, epoch, 1])
        ends = []
        for si, seq in enumerate(self.seqs):
            T = seq["data"].shape[1]; fl = seq["frame_labels"]
            for _ in range(self.wps):
                if fl is not None and fl.any() and rng.random() < self.p_focus:
                    end = int(rng.choice(np.nonzero(fl)[0]))
                else:
                    end = int(rng.integers(0, T))
                y = int(fl[end]) if fl is not None else seq["label"]
                ends.append((si, end, y))
        nat = {c: np.mean([e < self.window - 1 for _, e, y in ends if y == c] or [0.0]) for c in (0, 1)}
        target = max(nat[c] + (1 - nat[c]) * self.p_pad for c in (0, 1))
        p_c = {c: (target - nat[c]) / (1 - nat[c]) if nat[c] < 1 else 0.0 for c in (0, 1)}
        self.items, self.p_class = [], p_c
        for si, end, y in ends:
            n_real = self.window
            if end >= self.window - 1 and rng.random() < p_c[y]:
                n_real = int(rng.integers(self.min_real, self.window))
            self.items.append((si, end, n_real, int(rng.integers(2 ** 31))))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        si, end, n_real, aug_seed = self.items[i]
        seq = self.seqs[si]
        x, start = window_by_end(seq["data"], end, self.window, n_real, self.pad_side)
        x = normalize(x, self.norm_mode)
        if self.aug:
            x = self.aug(x, np.random.default_rng(aug_seed))
        y, w = label_and_weight(seq, start, end, self.min_pos)
        return torch.from_numpy(np.ascontiguousarray(x)), torch.tensor(y, dtype=torch.float32), \
            torch.tensor(w, dtype=torch.float32)

    def labels(self):
        """(label, weight) of every window of the current epoch, without building tensors."""
        out = []
        for si, end, n_real, _ in self.items:
            seq = self.seqs[si]
            _, start = window_by_end(seq["data"], end, self.window, n_real)
            out.append(label_and_weight(seq, start, end, self.min_pos))
        return out

    def padded_fraction(self):
        out = {0: [0, 0], 1: [0, 0]}
        for si, end, n_real, _ in self.items:
            seq = self.seqs[si]
            _, start = window_by_end(seq["data"], end, self.window, n_real)
            y, w = label_and_weight(seq, start, end, self.min_pos)
            if w > 0:
                out[y][0] += 1; out[y][1] += int(end - start + 1 < self.window)
        return {k: (v[1] / v[0] if v[0] else float("nan")) for k, v in out.items()}


class EvalWindows(Dataset):
    """One causal window per evaluated frame t (every `stride`-th frame and the last frame)."""

    def __init__(self, seqs, window=80, stride=1, min_pos=10, norm_mode="image", pad_side="right"):
        self.seqs, self.window, self.min_pos, self.norm_mode = seqs, window, min_pos, norm_mode
        self.pad_side = pad_side
        self.index = []
        for si, s in enumerate(seqs):
            T = s["data"].shape[1]
            ends = sorted(set(range(0, T, stride)) | {T - 1})
            self.index += [(si, t) for t in ends]

    def __len__(self):
        return len(self.index)

    def __getitem__(self, i):
        si, end = self.index[i]
        seq = self.seqs[si]
        x, start = window_by_end(seq["data"], end, self.window, pad_side=self.pad_side)
        x = normalize(x, self.norm_mode)
        fl = seq["frame_labels"]
        y = int(fl[end]) if fl is not None else seq["label"]
        return torch.from_numpy(np.ascontiguousarray(x)), torch.tensor(y, dtype=torch.float32), si, end
