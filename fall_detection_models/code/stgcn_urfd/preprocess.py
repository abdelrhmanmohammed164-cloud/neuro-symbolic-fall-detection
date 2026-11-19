"""
preprocess.py
Crash-safe, resumable preprocessing of all URFD sequences with chunk-level checkpoints.

    python preprocess.py --resume                 process everything, continuing where it stopped
    python preprocess.py --status                 show progress from the manifest
    python preprocess.py --validate-checkpoints   re-verify every checkpoint and output, write a report
    python preprocess.py --cleanup-checkpoints --yes   delete chunk files of verified videos (optional)

Layout under --root (default: the repository root, or $PROJECT_ROOT):
    URFD/{falls,adls}/<seq>-cam0-rgb.zip, <seq>-data.csv       inputs (never modified)
    checkpoints/processing_manifest.sqlite                      persistent manifest
    checkpoints/cfg_<fp12>/<seq>/chunk_0000.npz ...             chunk checkpoints, one dir per config
    processed_data/cfg_<fp12>/urfd_npy/<seq>*.npy, <seq>_meta.json   final outputs
    logs/processing.log, logs/errors.log
    reports/validation_report.csv

What a chunk holds (all per frame, no dependency on later frames):
    raw RTMO COCO-17 keypoints of the tracked person, detection mask, number of detections,
    Depth-Anything map at 160x120 (float16), frame indices, timestamps, tracking state in/out.
The only state carried between chunks is the tracked neck position (which person to follow).
Each chunk stores it at its start and end; a chunk whose stored start state does not match the
previous chunk's end state is treated as invalid and recomputed.
Interpolation, smoothing, the neck/hip anchor and the neck depth need the whole sequence, so they
run in a cheap finalize step that reads the chunks.

Every file is written as: temp file -> flush -> fsync -> reload + validate -> atomic rename ->
fsync dir -> manifest commit. Checkpoints made with another configuration are never reused.
"""
import argparse
import csv
import fcntl
import hashlib
import io
import json
import logging
import math
import os
import sqlite3
import sys
import time
import traceback
import zipfile

import cv2
import numpy as np

from graph import CENTER, coco17_to_urfd14
from preprocess_sequence import CONF_THR, MAX_GAP, SMOOTH, body_anchor, clean_joints, pick_person

CODE_VERSION = "preprocess-v1"
DEPTH_SMALL = (120, 160)            # stored depth map size (H, W) = 1/4 of 480x640
DEPTH_INPUT = 518                   # Depth-Anything input size
FPS_FALLBACK = 30.0                 # URFD cam0 frame rate, used when data.csv is unusable
OUT_SUFFIXES = ("", "_anchor", "_neck_depth", "_raw_coco17", "_timestamps")

log = logging.getLogger("preprocess")
_TEST_COUNTER = {"chunks": 0, "writes": 0}


# ----------------------------------------------------------------------------------------------
# small utilities
# ----------------------------------------------------------------------------------------------
def sha256_file(path, buf=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(buf), b""):
            h.update(b)
    return h.hexdigest()


def fsync_dir(d):
    fd = os.open(d, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def test_hook(name):
    """Fault injection for the recovery tests. Never active unless the env variable is set."""
    n = os.environ.get(name)
    if n is None:
        return False
    key = "writes" if name == "PREPROC_TEST_CRASH_IN_WRITE" else "chunks"
    return _TEST_COUNTER[key] >= int(n)


def atomic_write(final_path, data: bytes, validate):
    """temp -> flush -> fsync -> reload/validate -> rename -> fsync dir. Returns sha256."""
    d = os.path.dirname(final_path)
    os.makedirs(d, exist_ok=True)
    tmp = f"{final_path}.tmp-{os.getpid()}"
    _TEST_COUNTER["writes"] += 1
    with open(tmp, "wb") as f:
        if test_hook("PREPROC_TEST_CRASH_IN_WRITE"):        # simulate a crash halfway through a write
            f.write(data[: len(data) // 2]); f.flush(); os.fsync(f.fileno())
            log.warning("TEST: crash while writing %s", os.path.basename(final_path))
            os._exit(91)
        f.write(data)
        f.flush()
        os.fsync(f.fileno())
    validate(tmp)                                            # reload from disk and check
    os.replace(tmp, final_path)
    fsync_dir(d)
    return hashlib.sha256(data).hexdigest()


def npz_bytes(**arrays):
    b = io.BytesIO()
    np.savez_compressed(b, **arrays)
    return b.getvalue()


def npy_bytes(arr):
    b = io.BytesIO()
    np.save(b, arr)
    return b.getvalue()


# ----------------------------------------------------------------------------------------------
# manifest
# ----------------------------------------------------------------------------------------------
SCHEMA = """
CREATE TABLE IF NOT EXISTS configs(config_fp TEXT PRIMARY KEY, config_json TEXT, created_at REAL);
CREATE TABLE IF NOT EXISTS videos(video_id TEXT, config_fp TEXT, status TEXT, input_fp TEXT,
    n_frames INT, n_chunks INT, outputs_json TEXT, error TEXT, updated_at REAL,
    PRIMARY KEY(video_id, config_fp));
CREATE TABLE IF NOT EXISTS chunks(video_id TEXT, config_fp TEXT, chunk_idx INT, start_frame INT,
    end_frame INT, path TEXT, sha256 TEXT, status TEXT, device TEXT, created_at REAL,
    PRIMARY KEY(video_id, config_fp, chunk_idx));
CREATE TABLE IF NOT EXISTS file_hashes(path TEXT PRIMARY KEY, size INT, mtime_ns INT, sha256 TEXT);
"""


class Manifest:
    def __init__(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.con = sqlite3.connect(path, timeout=30)
        self.con.execute("PRAGMA journal_mode=WAL")
        self.con.execute("PRAGMA synchronous=FULL")
        self.con.executescript(SCHEMA)

    def file_hash(self, path):
        """sha256 of a large input file, cached by (size, mtime)."""
        st = os.stat(path)
        row = self.con.execute("SELECT size, mtime_ns, sha256 FROM file_hashes WHERE path=?", (path,)).fetchone()
        if row and row[0] == st.st_size and row[1] == st.st_mtime_ns:
            return row[2]
        h = sha256_file(path)
        with self.con:
            self.con.execute("INSERT OR REPLACE INTO file_hashes VALUES (?,?,?,?)",
                             (path, st.st_size, st.st_mtime_ns, h))
        return h

    def chunk(self, vid, cfg, k):
        return self.con.execute("SELECT * FROM chunks WHERE video_id=? AND config_fp=? AND chunk_idx=?",
                                (vid, cfg, k)).fetchone()

    def set_chunk(self, vid, cfg, k, s, e, path, sha, status, device):
        with self.con:
            self.con.execute("INSERT OR REPLACE INTO chunks VALUES (?,?,?,?,?,?,?,?,?,?)",
                             (vid, cfg, k, s, e, path, sha, status, device, time.time()))

    def mark_chunk(self, vid, cfg, k, status):
        with self.con:
            self.con.execute("UPDATE chunks SET status=? WHERE video_id=? AND config_fp=? AND chunk_idx=?",
                             (status, vid, cfg, k))

    def video(self, vid, cfg):
        return self.con.execute("SELECT * FROM videos WHERE video_id=? AND config_fp=?", (vid, cfg)).fetchone()

    def set_video(self, vid, cfg, status, input_fp=None, n_frames=None, n_chunks=None, outputs=None, error=None):
        old = self.video(vid, cfg)
        vals = dict(input_fp=input_fp, n_frames=n_frames, n_chunks=n_chunks,
                    outputs_json=json.dumps(outputs) if outputs is not None else None, error=error)
        if old:
            keep = dict(zip(["video_id", "config_fp", "status", "input_fp", "n_frames", "n_chunks",
                             "outputs_json", "error", "updated_at"], old))
            for k, v in vals.items():
                if v is None and k != "error":
                    vals[k] = keep[k]
        with self.con:
            self.con.execute("INSERT OR REPLACE INTO videos VALUES (?,?,?,?,?,?,?,?,?)",
                             (vid, cfg, status, vals["input_fp"], vals["n_frames"], vals["n_chunks"],
                              vals["outputs_json"], vals["error"], time.time()))


# ----------------------------------------------------------------------------------------------
# inputs
# ----------------------------------------------------------------------------------------------
def discover_videos(root, only=None):
    vids = []
    for kind in ("falls", "adls"):
        d = os.path.join(root, "URFD", kind)
        if not os.path.isdir(d):
            continue
        for fn in sorted(os.listdir(d)):
            if fn.endswith("-cam0-rgb.zip"):
                seq = fn[: -len("-cam0-rgb.zip")]
                if only and seq not in only:
                    continue
                vids.append(dict(id=seq, kind=kind, zip=os.path.join(d, fn),
                                 csv=os.path.join(d, f"{seq}-data.csv")))
    return vids


def frame_list(zip_path):
    with zipfile.ZipFile(zip_path) as z:
        names = [n for n in z.namelist() if n.endswith(".png")]
    return sorted(names, key=lambda s: int(os.path.splitext(s)[0].rsplit("-", 1)[-1]))


def load_timestamps(csv_path, T):
    """Per-frame time in ms from <seq>-data.csv (frame, time_ms, ...). Falls back to index / 30 fps
    when the file is missing, has a different number of rows, or is not strictly increasing."""
    fallback = np.arange(T, dtype=np.float64) * 1000.0 / FPS_FALLBACK
    if not os.path.exists(csv_path):
        return fallback, "index_fallback:no_csv"
    rows = [r for r in csv.reader(open(csv_path)) if r]
    if len(rows) != T:
        return fallback, f"index_fallback:csv_rows_{len(rows)}_vs_frames_{T}"
    ts = np.array([float(r[1]) for r in rows], np.float64)
    if not np.all(np.diff(ts) > 0):
        return fallback, "index_fallback:not_increasing"
    return ts, "data_csv"


# ----------------------------------------------------------------------------------------------
# models (loaded lazily, only when a chunk actually has to be computed)
# ----------------------------------------------------------------------------------------------
class Models:
    def __init__(self, args):
        self.args, self.pose, self.depth, self.torch = args, None, None, None

    def load(self):
        if self.pose is not None:
            return
        import torch
        self.torch = torch
        from rtmlib import RTMO
        dev = self.args.device
        probe = np.zeros((480, 640, 3), np.uint8)
        try:
            self.pose = RTMO(self.args.rtmo, score_thr=self.args.det_thr, backend="onnxruntime", device=dev)
            self.pose(probe)
            self.pose_device = dev
        except Exception as e:
            log.warning("RTMO on %s failed (%s); using cpu", dev, e)
            self.pose = RTMO(self.args.rtmo, score_thr=self.args.det_thr, backend="onnxruntime", device="cpu")
            self.pose_device = "cpu"
        sys.path.insert(0, self.args.da_repo)
        from depth_anything_v2.dpt import DepthAnythingV2
        cfg = {"vits": dict(features=64, out_channels=[48, 96, 192, 384]),
               "vitb": dict(features=128, out_channels=[96, 192, 384, 768])}[self.args.encoder]
        m = DepthAnythingV2(encoder=self.args.encoder, **cfg)
        m.load_state_dict(torch.load(self.args.da_weights, map_location="cpu"))
        self.depth = m.to(self.args.device).eval()
        log.info("models loaded: RTMO on %s, Depth-Anything-%s on %s",
                 self.pose_device, self.args.encoder, self.args.device)

    def infer(self, img, prev_neck):
        k, s = self.pose(img)
        k, s = np.asarray(k), np.asarray(s)
        n = int(len(k)) if k.size else 0
        i = pick_person(k, s, prev_neck) if n else None
        kp = np.full((17, 3), np.nan, np.float32)
        if i is not None:
            kp[:, :2], kp[:, 2] = k[i], s[i]
            prev_neck = ((k[i, 5] + k[i, 6]) / 2).astype(np.float32)
        d = self.depth.infer_image(img, DEPTH_INPUT)
        d_small = cv2.resize(d, (DEPTH_SMALL[1], DEPTH_SMALL[0]), interpolation=cv2.INTER_AREA)
        return kp, n, d_small.astype(np.float16), prev_neck

    def release(self):
        if self.torch is not None and self.args.device == "mps":
            self.torch.mps.empty_cache()


# ----------------------------------------------------------------------------------------------
# chunk checkpoints
# ----------------------------------------------------------------------------------------------
CHUNK_KEYS = ["video_id", "chunk_idx", "start", "end", "frame_idx", "frame_names", "timestamps_ms",
              "ts_source", "raw_coco17", "det_valid", "n_persons", "depth_small", "img_hw",
              "track_in", "track_out", "config_fp", "input_fp", "device", "code_version"]


def validate_chunk(path, vid, k, start, end, cfg_fp, input_fp):
    """Reload a chunk and check content. Returns the loaded dict or raises ValueError."""
    with np.load(path, allow_pickle=False) as z:
        c = {key: z[key] for key in z.files}
    miss = [key for key in CHUNK_KEYS if key not in c]
    if miss:
        raise ValueError(f"missing keys {miss}")
    if str(c["config_fp"]) != cfg_fp:
        raise ValueError("config fingerprint differs")
    if str(c["input_fp"]) != input_fp:
        raise ValueError("input video fingerprint differs")
    if str(c["video_id"]) != vid or int(c["chunk_idx"]) != k:
        raise ValueError("video id / chunk index differ")
    if int(c["start"]) != start or int(c["end"]) != end:
        raise ValueError("frame range differs")
    n = end - start
    if not np.array_equal(c["frame_idx"], np.arange(start, end)):
        raise ValueError("frame indices not contiguous")
    if c["raw_coco17"].shape != (n, 17, 3) or c["det_valid"].shape != (n,) or \
            c["depth_small"].shape != (n, *DEPTH_SMALL) or c["timestamps_ms"].shape != (n,):
        raise ValueError("bad shapes")
    if not (np.all(np.isfinite(c["timestamps_ms"])) and np.all(np.diff(c["timestamps_ms"]) > 0)):
        raise ValueError("timestamps not finite / increasing")
    raw, ok = c["raw_coco17"], c["det_valid"].astype(bool)
    if not np.all(np.isfinite(raw[ok])) or not np.all(np.isnan(raw[~ok])):
        raise ValueError("detection mask inconsistent with keypoints")
    if ok.any() and not ((raw[ok][..., 2] >= 0).all() and (raw[ok][..., 2] <= 1.0001).all()):
        raise ValueError("confidence outside [0, 1]")
    if not np.all(np.isfinite(c["depth_small"].astype(np.float32))):
        raise ValueError("non-finite depth")
    return c


class VideoJob:
    def __init__(self, v, ctx):
        self.v, self.ctx = v, ctx
        self.vid, self.cfg = v["id"], ctx.cfg_fp
        self.dir = os.path.join(ctx.ckpt_dir, self.vid)

    def chunk_path(self, k):
        return os.path.join(self.dir, f"chunk_{k:04d}.npz")

    def cleanup_temp(self):
        """Temp files from a crash mid-write are incomplete by definition: remove them."""
        if not os.path.isdir(self.dir):
            return
        for fn in os.listdir(self.dir):
            if ".tmp-" in fn:
                os.remove(os.path.join(self.dir, fn))
                log.warning("%s: removed incomplete temp file %s (crash during write)", self.vid, fn)

    def quarantine(self, path, why):
        dst = f"{path}.corrupt-{int(time.time())}"
        os.replace(path, dst)
        log.warning("%s: %s -> kept as %s (%s)", self.vid, os.path.basename(path), os.path.basename(dst), why)

    def check_chunk(self, k, s, e, input_fp, prev_track):
        """Return the verified chunk dict, or None if it must be (re)computed."""
        m, path = self.ctx.db, self.chunk_path(k)
        rec = m.chunk(self.vid, self.cfg, k)
        if not os.path.exists(path):
            if rec and rec[7] == "COMPLETED":
                log.warning("%s chunk %d: in manifest but file missing -> recompute", self.vid, k)
                m.mark_chunk(self.vid, self.cfg, k, "INVALID")
            return None
        try:
            c = validate_chunk(path, self.vid, k, s, e, self.cfg, input_fp)
            sha = sha256_file(path)
        except Exception as ex:
            self.quarantine(path, f"invalid: {ex}")
            if rec:
                m.mark_chunk(self.vid, self.cfg, k, "INVALID")
            return None
        if rec and rec[6] != sha:
            self.quarantine(path, "checksum differs from manifest")
            m.mark_chunk(self.vid, self.cfg, k, "INVALID")
            return None
        if not self._track_matches(c["track_in"], prev_track):
            log.warning("%s chunk %d: tracking state differs from previous chunk -> recompute", self.vid, k)
            self.quarantine(path, "stale tracking state")
            if rec:
                m.mark_chunk(self.vid, self.cfg, k, "INVALID")
            return None
        if not rec or rec[7] != "COMPLETED":
            # crash happened after the atomic rename but before the manifest commit
            m.set_chunk(self.vid, self.cfg, k, s, e, path, sha, "COMPLETED", str(c["device"]))
            log.info("%s chunk %d: valid file without manifest entry -> adopted", self.vid, k)
        return c

    @staticmethod
    def _track_matches(a, b, tol=0.5):
        a, b = np.asarray(a, np.float32), np.asarray(b, np.float32)
        if np.isnan(a).all() and np.isnan(b).all():
            return True
        return bool(np.all(np.isfinite(a)) and np.all(np.isfinite(b)) and np.abs(a - b).max() <= tol)

    def compute_chunk(self, k, s, e, names, ts, ts_src, input_fp, prev_track):
        ctx = self.ctx
        ctx.models.load()
        n = e - s
        raw = np.full((n, 17, 3), np.nan, np.float32)
        dsm = np.zeros((n, *DEPTH_SMALL), np.float16)
        npers = np.zeros(n, np.int16)
        track = None if np.isnan(prev_track).all() else prev_track.copy()
        hw = None
        with zipfile.ZipFile(self.v["zip"]) as z:
            for i, name in enumerate(names[s:e]):
                img = cv2.imdecode(np.frombuffer(z.read(name), np.uint8), cv2.IMREAD_COLOR)
                if img is None:
                    raise IOError(f"cannot decode {name}")
                hw = img.shape[:2]
                raw[i], npers[i], dsm[i], track = ctx.models.infer(img, track)
        track_out = np.full(2, np.nan, np.float32) if track is None else np.asarray(track, np.float32)
        arrays = dict(
            video_id=np.array(self.vid), chunk_idx=np.array(k), start=np.array(s), end=np.array(e),
            frame_idx=np.arange(s, e), frame_names=np.array(names[s:e]),
            timestamps_ms=ts[s:e], ts_source=np.array(ts_src), raw_coco17=raw,
            det_valid=~np.isnan(raw[:, 0, 0]), n_persons=npers, depth_small=dsm,
            img_hw=np.array(hw), track_in=prev_track.astype(np.float32), track_out=track_out,
            config_fp=np.array(self.cfg), input_fp=np.array(input_fp),
            device=np.array(ctx.models.pose_device + "/" + ctx.args.device), code_version=np.array(CODE_VERSION))
        path = self.chunk_path(k)
        sha = atomic_write(path, npz_bytes(**arrays),
                           lambda p: validate_chunk(p, self.vid, k, s, e, self.cfg, input_fp))
        if test_hook("PREPROC_TEST_CRASH_BEFORE_COMMIT"):
            log.warning("TEST: crash after rename, before manifest commit (chunk %d)", k)
            os._exit(92)
        ctx.db.set_chunk(self.vid, self.cfg, k, s, e, path, sha, "COMPLETED", str(arrays["device"]))
        _TEST_COUNTER["chunks"] += 1
        log.info("%s chunk %d [%d, %d) saved (%d/%d frames with person)", self.vid, k, s, e,
                 int(arrays["det_valid"].sum()), n)
        if test_hook("PREPROC_TEST_KILL_AFTER_CHUNK"):
            log.warning("TEST: process killed after committing chunk %d", k)
            os._exit(93)
        return validate_chunk(path, self.vid, k, s, e, self.cfg, input_fp)

    # ------------------------------------------------------------------------------------------
    def outputs_valid(self):
        row = self.ctx.db.video(self.vid, self.cfg)
        if not row or row[2] != "COMPLETED" or not row[6]:
            return False, "not completed"
        outs = json.loads(row[6])
        for fn, sha in outs.items():
            p = os.path.join(self.ctx.out_dir, fn)
            if not os.path.exists(p):
                return False, f"missing {fn}"
            if sha256_file(p) != sha:
                return False, f"checksum {fn}"
        return True, "ok"

    def run(self):
        ctx, m = self.ctx, self.ctx.db
        ok, why = self.outputs_valid()
        if ok:
            log.info("%s: completed and verified -> skip", self.vid)
            return "skipped"
        input_fp = m.file_hash(self.v["zip"])
        names = frame_list(self.v["zip"])
        T = len(names)
        ts, ts_src = load_timestamps(self.v["csv"], T)
        cs = ctx.args.chunk_size
        n_chunks = math.ceil(T / cs)
        row = m.video(self.vid, self.cfg)
        if row and row[3] and row[3] != input_fp:
            log.warning("%s: input video changed since last run -> its checkpoints are rejected", self.vid)
        m.set_video(self.vid, self.cfg, "IN_PROGRESS", input_fp, T, n_chunks)
        os.makedirs(self.dir, exist_ok=True)
        self.cleanup_temp()

        chunks, prev_track, reused = [], np.full(2, np.nan, np.float32), 0
        for k in range(n_chunks):
            s, e = k * cs, min(T, (k + 1) * cs)
            c = self.check_chunk(k, s, e, input_fp, prev_track)
            if c is None:
                c = self.compute_chunk(k, s, e, names, ts, ts_src, input_fp, prev_track)
            else:
                reused += 1
            chunks.append(c)
            prev_track = c["track_out"]
        log.info("%s: %d chunks (%d reused from checkpoints, %d computed)", self.vid, n_chunks, reused,
                 n_chunks - reused)
        self.finalize(chunks, T, input_fp, ts_src)
        ctx.models.release()
        return "processed"

    def finalize(self, chunks, T, input_fp, ts_src):
        """Sequence-level steps from the verified chunks; atomic multi-file commit."""
        fidx = np.concatenate([c["frame_idx"] for c in chunks])
        if not np.array_equal(fidx, np.arange(T)):
            raise RuntimeError("frame coverage error: duplicated or missing frames")
        raw = np.concatenate([c["raw_coco17"] for c in chunks])
        dsm = np.concatenate([c["depth_small"] for c in chunks])
        ts = np.concatenate([c["timestamps_ms"] for c in chunks])
        H, W = (int(x) for x in chunks[0]["img_hw"])

        j14 = clean_joints(coco17_to_urfd14(np.nan_to_num(raw, nan=0.0)))
        anchor = body_anchor(raw)
        depth = np.full(T, np.nan, np.float32)
        sy, sx = DEPTH_SMALL[0] / H, DEPTH_SMALL[1] / W
        for t in range(T):
            if anchor[t, 2] > 0:
                u = int(np.clip(round(anchor[t, 0] * sx), 0, DEPTH_SMALL[1] - 1))
                v = int(np.clip(round(anchor[t, 1] * sy), 0, DEPTH_SMALL[0] - 1))
                depth[t] = np.median(dsm[t, max(0, v - 1):v + 2, max(0, u - 1):u + 2].astype(np.float32))
        miss = np.isnan(j14[..., 0])
        x = np.zeros((3, T, 14, 1), np.float32)
        x[0, :, :, 0] = np.where(miss, 0, j14[..., 0] / W)
        x[1, :, :, 0] = np.where(miss, 0, j14[..., 1] / H)
        x[2, :, :, 0] = np.where(miss, 0, j14[..., 2])

        files = {f"{self.vid}.npy": x, f"{self.vid}_anchor.npy": anchor,
                 f"{self.vid}_neck_depth.npy": depth, f"{self.vid}_raw_coco17.npy": raw,
                 f"{self.vid}_timestamps.npy": ts}
        expected = {f"{self.vid}.npy": (3, T, 14, 1), f"{self.vid}_anchor.npy": (T, 3),
                    f"{self.vid}_neck_depth.npy": (T,), f"{self.vid}_raw_coco17.npy": (T, 17, 3),
                    f"{self.vid}_timestamps.npy": (T,)}
        outs = {}
        for fn, arr in files.items():
            def val(p, shape=expected[fn]):
                a = np.load(p, allow_pickle=False)
                if a.shape != shape:
                    raise ValueError(f"{fn}: shape {a.shape} != {shape}")
            outs[fn] = atomic_write(os.path.join(self.ctx.out_dir, fn), npy_bytes(arr), val)
        meta = dict(video_id=self.vid, kind=self.v["kind"], n_frames=T, image_hw=[H, W],
                    timestamps_source=ts_src, input_fp=input_fp, config_fp=self.cfg,
                    config=self.ctx.config, chunks=[dict(idx=int(c["chunk_idx"]), start=int(c["start"]),
                                                         end=int(c["end"]), device=str(c["device"]))
                                                    for c in chunks],
                    anchor_counts=dict(neck=int((anchor[:, 2] == 1).sum()),
                                       hip=int((anchor[:, 2] == 2).sum()),
                                       none=int((anchor[:, 2] == 0).sum())),
                    frames_with_person=int((~np.isnan(raw[:, 0, 0])).sum()))
        outs[f"{self.vid}_meta.json"] = atomic_write(
            os.path.join(self.ctx.out_dir, f"{self.vid}_meta.json"),
            json.dumps(meta, indent=1).encode(), lambda p: json.load(open(p)))
        self.ctx.db.set_video(self.vid, self.cfg, "COMPLETED", input_fp, T, len(chunks), outs)
        log.info("%s: COMPLETED (%d frames, person in %d, anchor neck/hip/none %d/%d/%d, timestamps %s)",
                 self.vid, T, meta["frames_with_person"], *meta["anchor_counts"].values(), ts_src)


# ----------------------------------------------------------------------------------------------
# context, commands
# ----------------------------------------------------------------------------------------------
class Ctx:
    def __init__(self, args):
        self.args = args
        self.db = Manifest(os.path.join(args.root, "checkpoints", "processing_manifest.sqlite"))
        self.config = dict(
            code_version=CODE_VERSION, chunk_size=args.chunk_size, det_thr=args.det_thr,
            joint_conf_thr=CONF_THR, max_interp_gap=MAX_GAP, smooth_window=SMOOTH,
            anchor_rule="neck if both shoulders>=conf_thr else mid-hip if both hips>=conf_thr",
            depth_encoder=args.encoder, depth_input=DEPTH_INPUT, depth_small=list(DEPTH_SMALL),
            depth_patch="3x3 median on depth_small", camera="cam0",
            rtmo_sha256=self.db.file_hash(args.rtmo), depth_weights_sha256=self.db.file_hash(args.da_weights))
        self.cfg_fp = hashlib.sha256(json.dumps(self.config, sort_keys=True).encode()).hexdigest()
        with self.db.con:
            self.db.con.execute("INSERT OR IGNORE INTO configs VALUES (?,?,?)",
                                (self.cfg_fp, json.dumps(self.config, sort_keys=True), time.time()))
        tag = f"cfg_{self.cfg_fp[:12]}"
        self.ckpt_dir = os.path.join(args.root, "checkpoints", tag)
        self.out_dir = os.path.join(args.root, "processed_data", tag, "urfd_npy")
        os.makedirs(self.ckpt_dir, exist_ok=True)
        os.makedirs(self.out_dir, exist_ok=True)
        self.models = Models(args)


def setup_logging(root):
    d = os.path.join(root, "logs")
    os.makedirs(d, exist_ok=True)
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    log.setLevel(logging.INFO)
    for h in (logging.FileHandler(os.path.join(d, "processing.log")), logging.StreamHandler(sys.stdout)):
        h.setFormatter(fmt); log.addHandler(h)
    eh = logging.FileHandler(os.path.join(d, "errors.log"))
    eh.setLevel(logging.ERROR); eh.setFormatter(fmt); log.addHandler(eh)


def cmd_resume(ctx, videos):
    log.info("=== resume | config %s | %d videos | chunk size %d", ctx.cfg_fp[:12], len(videos),
             ctx.args.chunk_size)
    res = {"processed": [], "skipped": [], "failed": []}
    for v in videos:
        try:
            res[VideoJob(v, ctx).run()].append(v["id"])
        except KeyboardInterrupt:
            log.warning("interrupted by user during %s; committed chunks are kept", v["id"])
            raise
        except Exception as ex:
            log.error("%s FAILED: %s\n%s", v["id"], ex, traceback.format_exc())
            ctx.db.set_video(v["id"], ctx.cfg_fp, "FAILED", error=str(ex)[:500])
            res["failed"].append(v["id"])
            ctx.models.release()
    log.info("=== done | processed %d | skipped (verified) %d | failed %d %s", len(res["processed"]),
             len(res["skipped"]), len(res["failed"]), res["failed"] or "")
    return 1 if res["failed"] else 0


def cmd_status(ctx, videos):
    rows = []
    for v in videos:
        row = ctx.db.video(v["id"], ctx.cfg_fp)
        done = ctx.db.con.execute("SELECT COUNT(*) FROM chunks WHERE video_id=? AND config_fp=? AND status='COMPLETED'",
                                  (v["id"], ctx.cfg_fp)).fetchone()[0]
        rows.append((v["id"], row[2] if row else "PENDING", done, row[5] if row else None,
                     row[4] if row else None))
    print(f"config {ctx.cfg_fp[:12]}  (chunk size {ctx.args.chunk_size})")
    print(f"output dir: {ctx.out_dir}")
    print(f"{'video':10s} {'status':12s} {'chunks':>9s} {'frames':>7s}")
    for vid, st, done, tot, T in rows:
        print(f"{vid:10s} {st:12s} {done:>4}/{tot if tot else '?':<4} {T if T else '':>7}")
    cnt = {}
    for r in rows:
        cnt[r[1]] = cnt.get(r[1], 0) + 1
    print("summary:", cnt)
    other = ctx.db.con.execute("SELECT config_fp, COUNT(*) FROM chunks WHERE config_fp<>? GROUP BY config_fp",
                               (ctx.cfg_fp,)).fetchall()
    for fp, n in other:
        print(f"note: {n} checkpoints from another config {fp[:12]} are kept but NOT used")
    return 0


def cmd_validate(ctx, videos):
    os.makedirs(os.path.join(ctx.args.root, "reports"), exist_ok=True)
    rep, bad = [], 0
    for v in videos:
        job = VideoJob(v, ctx)
        row = ctx.db.video(v["id"], ctx.cfg_fp)
        recs = ctx.db.con.execute("SELECT chunk_idx, start_frame, end_frame, sha256, status FROM chunks "
                                  "WHERE video_id=? AND config_fp=? ORDER BY chunk_idx",
                                  (v["id"], ctx.cfg_fp)).fetchall()
        issues, valid = [], 0
        input_fp = ctx.db.file_hash(v["zip"]) if recs or row else None
        for k, s, e, sha, st in recs:
            p = job.chunk_path(k)
            if st != "COMPLETED":
                issues.append(f"chunk {k} {st}"); continue
            if not os.path.exists(p):
                issues.append(f"chunk {k} missing"); ctx.db.mark_chunk(v["id"], ctx.cfg_fp, k, "INVALID"); continue
            if sha256_file(p) != sha:
                issues.append(f"chunk {k} checksum"); ctx.db.mark_chunk(v["id"], ctx.cfg_fp, k, "INVALID"); continue
            try:
                validate_chunk(p, v["id"], k, s, e, ctx.cfg_fp, input_fp)
                valid += 1
            except Exception as ex:
                issues.append(f"chunk {k} {ex}"); ctx.db.mark_chunk(v["id"], ctx.cfg_fp, k, "INVALID")
        stray = [f for f in (os.listdir(job.dir) if os.path.isdir(job.dir) else []) if ".tmp-" in f]
        if stray:
            issues.append(f"{len(stray)} incomplete temp files")
        out_ok, why = job.outputs_valid()
        status = row[2] if row else "PENDING"
        if status == "COMPLETED" and not out_ok:
            issues.append(f"outputs: {why}")
            ctx.db.set_video(v["id"], ctx.cfg_fp, "INVALID", error=why)
            status = "INVALID"
        if issues:
            bad += 1
        rep.append(dict(video_id=v["id"], status=status, chunks_total=row[5] if row else "",
                        chunks_valid=valid, chunks_recorded=len(recs), outputs_ok=out_ok,
                        issues="; ".join(issues)))
    path = os.path.join(ctx.args.root, "reports", "validation_report.csv")
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rep[0])); w.writeheader(); w.writerows(rep)
    print(f"validated {len(rep)} videos | with issues: {bad} | report: {path}")
    for r in rep:
        if r["issues"]:
            print(f"  {r['video_id']}: {r['issues']}")
    return 1 if bad else 0


def cmd_cleanup(ctx, videos, yes):
    if not yes:
        print("cleanup deletes chunk files of verified videos only. Re-run with --yes to confirm.")
        return 1
    removed = 0
    for v in videos:
        job = VideoJob(v, ctx)
        ok, _ = job.outputs_valid()
        if not ok or not os.path.isdir(job.dir):
            continue
        for fn in os.listdir(job.dir):
            if fn.startswith("chunk_") and fn.endswith(".npz"):
                os.remove(os.path.join(job.dir, fn)); removed += 1
        with ctx.db.con:
            ctx.db.con.execute("UPDATE chunks SET status='CLEANED' WHERE video_id=? AND config_fp=?",
                               (v["id"], ctx.cfg_fp))
    print(f"removed {removed} chunk files (final outputs kept)")
    return 0


def main():
    # project root = three folders above this file (fall_detection_models/code/stgcn_urfd); override with PROJECT_ROOT
    root_default = os.environ.get("PROJECT_ROOT") or os.path.abspath(
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".."))
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    g = p.add_mutually_exclusive_group()
    g.add_argument("--resume", action="store_true", help="process / continue (default action)")
    g.add_argument("--status", action="store_true")
    g.add_argument("--validate-checkpoints", action="store_true")
    g.add_argument("--cleanup-checkpoints", action="store_true")
    p.add_argument("--yes", action="store_true")
    p.add_argument("--root", default=root_default)
    p.add_argument("--videos", nargs="*", help="only these sequences, e.g. fall-01 adl-05")
    p.add_argument("--chunk_size", type=int, default=60)
    p.add_argument("--det_thr", type=float, default=0.1)
    p.add_argument("--encoder", default="vits", choices=["vits", "vitb"])
    p.add_argument("--device", default="auto", choices=["auto", "mps", "cuda", "cpu"])
    p.add_argument("--rtmo", default=None)
    p.add_argument("--da_weights", default=None)
    p.add_argument("--da_repo", default=None)
    a = p.parse_args()
    a.root = os.path.abspath(os.path.expanduser(a.root))
    mdl = os.path.join(a.root, "fall_detection_models")
    a.rtmo = a.rtmo or os.path.join(mdl, "weights", "rtmo-l_onnx", "end2end.onnx")
    a.da_weights = a.da_weights or os.path.join(mdl, "weights", f"depth_anything_v2_{a.encoder}.pth")
    a.da_repo = a.da_repo or os.path.join(mdl, "code", "Depth-Anything-V2")
    if a.device == "auto":
        import torch
        a.device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")

    setup_logging(a.root)
    os.makedirs(os.path.join(a.root, "checkpoints"), exist_ok=True)
    lock = open(os.path.join(a.root, "checkpoints", ".lock"), "w")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        print("another preprocess.py run is active on this folder; stop it first")
        return 2
    ctx = Ctx(a)
    videos = discover_videos(a.root, set(a.videos) if a.videos else None)
    if not videos:
        print("no input videos found under", os.path.join(a.root, "URFD"))
        return 1
    if a.status:
        return cmd_status(ctx, videos)
    if a.validate_checkpoints:
        return cmd_validate(ctx, videos)
    if a.cleanup_checkpoints:
        return cmd_cleanup(ctx, videos, a.yes)
    try:
        return cmd_resume(ctx, videos)
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    sys.exit(main())
