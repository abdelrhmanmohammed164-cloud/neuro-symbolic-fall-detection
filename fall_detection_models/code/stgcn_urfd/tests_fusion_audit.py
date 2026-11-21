"""Tests for fusion.py, fusion_neurosymbolic.py, data_audit.py and export_symbolic_predictions.py.
Run: python -m unittest tests_fusion_audit.py   (PROJECT_ROOT overrides the project root for the data tests)
"""
import json
import os
import tempfile
import unittest
from pathlib import Path

import numpy as np

from data_audit import audit
from export_symbolic_predictions import export
from fusion import evaluate, fuse, read_predictions
import fusion_neurosymbolic as FN

HERE = Path(__file__).resolve().parent
ROOT = Path(os.environ.get("PROJECT_ROOT", HERE.parents[2]))
CFG = dict(t_n=0.5, t_up=0.8, t_low=0.1, adopted=True)


def streams(ps, ss, seq="s", labels=None):
    labels = labels or [0] * len(ps)
    n = {(seq, i): (p, y) for i, (p, y) in enumerate(zip(ps, labels), 1)}
    s = {(seq, i): (float(a), y) for i, (a, y) in enumerate(zip(ss, labels), 1)}
    return n, s


class TestFixedRules(unittest.TestCase):
    def setUp(self):
        self.n, self.s = streams([0.8, 0.1], [1, 1], labels=[1, 0])

    def test_and(self):
        p = fuse(self.n, self.s, "and")
        self.assertEqual([x["fused_alarm"] for x in p], [1, 0])
        self.assertEqual(evaluate(p)["f1"], 1)

    def test_or(self):
        self.assertEqual([x["fused_alarm"] for x in fuse(self.n, self.s, "or")], [1, 1])

    def test_no_confirm_duplicate(self):
        with self.assertRaises(ValueError):
            fuse(self.n, self.s, "confirm")

    def test_reject_alignment(self):
        with self.assertRaisesRegex(ValueError, "unaligned"):
            fuse(self.n, {("s", 1): (1., 1)}, "or")

    def test_reject_label_disagreement(self):
        with self.assertRaisesRegex(ValueError, "conflicting"):
            fuse(self.n, {("s", 1): (1., 0), ("s", 2): (0., 0)}, "or")

    def test_reject_nonbinary_symbolic(self):
        with self.assertRaisesRegex(ValueError, "binary"):
            fuse(self.n, {("s", 1): (0.7, 1), ("s", 2): (0., 0)}, "or")

    def test_duplicate_csv(self):
        with tempfile.TemporaryDirectory() as d:
            f = Path(d) / "p.csv"
            f.write_text("sequence,frame,prob\ns,1,0.1\ns,1,0.9\n")
            with self.assertRaisesRegex(ValueError, "duplicate"):
                read_predictions(f, "prob")

    def test_no_labels(self):
        self.assertFalse(evaluate(fuse({("s", 1): (0.2, None)}, {("s", 1): (0., None)}, "or"))["labeled"])

    def test_mcc_and_detection_after_onset(self):
        n, s = streams([0.9, 0.1, 0.1], [0, 0, 0], labels=[0, 1, 1])   # alarm only before the fall
        m = evaluate(fuse(n, s, "or"))
        self.assertEqual(m["fall_videos_detected"], 0)
        self.assertLess(m["mcc"], 0)


class TestNeuroSymbolic(unittest.TestCase):
    def run_rule(self, ps, ss, hs, cfg=CFG):
        n, s = streams(ps, ss)
        return [r["fused_alarm"] for r in fuse(n, s, "neurosymbolic", heights={"s": np.array(hs, float)}, config_of={"s": cfg})]

    def test_veto_when_upright(self):
        self.assertEqual(self.run_rule([0.9, 0.9], [0, 0], [1.4, 0.3]), [0, 1])

    def test_missing_height_never_vetoes(self):
        self.assertEqual(self.run_rule([0.9], [0], [np.nan]), [1])

    def test_symbolic_needs_weak_neural_support(self):
        self.assertEqual(self.run_rule([0.05, 0.2], [1, 1], [0.3, 0.3]), [0, 1])

    def test_not_adopted_falls_back_to_neural(self):
        self.assertEqual(self.run_rule([0.9, 0.2], [1, 1], [1.4, 0.3], dict(CFG, adopted=False)), [1, 0])

    def test_needs_heights(self):
        n, s = streams([0.9], [0])
        with self.assertRaises(ValueError):
            fuse(n, s, "neurosymbolic")

    def test_study_rule_matches_apply_rule(self):
        rng = np.random.default_rng(0)
        p, sy, h = rng.random(200), (rng.random(200) > 0.7).astype(int), rng.uniform(0, 1.6, 200)
        h[::17] = np.nan
        a = FN.fuse(p, sy, h, CFG["t_n"], CFG["t_up"], CFG["t_low"])
        b = self.run_rule(list(p), list(sy), list(h))
        self.assertEqual(list(a), b)

    def test_causal(self):
        rng = np.random.default_rng(1)
        p, sy, h = rng.random(50), (rng.random(50) > 0.5).astype(int), rng.uniform(0, 1.6, 50)
        full = FN.fuse(p, sy, h, 0.5, 0.8, 0.1)
        self.assertTrue((FN.fuse(p[:30], sy[:30], h[:30], 0.5, 0.8, 0.1) == full[:30]).all())

    def test_tie_break_prefers_less_veto(self):
        c = FN.choose([(dict(t_n=0.5, t_up=0.8, t_low=0.0), {"MCC": 0.60}), (dict(t_n=0.5, t_up=1.0, t_low=0.0), {"MCC": 0.598})])
        self.assertEqual(c[0]["t_up"], 1.0)


class TestAuditAndExport(unittest.TestCase):
    def test_audit_catches_test_scene_leakage(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            np.save(root / "a.npy", np.zeros((3, 5, 14, 1), np.float32))
            np.save(root / "b.npy", np.ones((3, 5, 14, 1), np.float32))
            sp = root / "split.json"
            sp.write_text(json.dumps({"group_of": {"a": "X", "b": "X"}, "folds": [{"train": ["a"], "val": [], "test": ["b"]}]}))
            r = audit(sp, root)
            self.assertFalse(r["passed"])
            self.assertEqual(r["errors"], ["fold0 scene overlap between train/test: ['X']"])

    def test_audit_warns_cross_fold(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            for i, v in enumerate("abcd"):
                np.save(root / f"{v}.npy", np.full((3, 5, 14, 1), i, np.float32))
            sp = root / "split.json"
            sp.write_text(json.dumps({"group_of": {"a": "A", "b": "B", "c": "C", "d": "D"}, "folds": [
                {"train": ["a"], "val": ["b"], "test": ["c", "d"]}, {"train": ["c"], "val": ["d"], "test": ["a", "b"]}]}))
            r = audit(sp, root)
            self.assertTrue(r["passed"])
            self.assertTrue(any("val inside" in w for w in r["warnings"]))

    def test_missing_kinect_fails_cleanly(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            (root / "split.json").write_text(json.dumps({"folds": [{"fold": 0, "test": ["fall-01"]}]}))
            (root / "cfg.json").write_text(json.dumps({"0": {"variant": "D"}}))
            with self.assertRaises(FileNotFoundError):
                export(root / "split.json", root / "cfg.json", root, root / "out.csv")

    @unittest.skipUnless((ROOT / "processed_data" / "kinect_height" / "fall-01.npz").exists(), "no project data")
    def test_export_reproduces_v3_test_counts(self):
        """The exported URFD test alarms give the improved V3's reported TP/FP."""
        with tempfile.TemporaryDirectory() as d:
            out = Path(d) / "sym.csv"
            export(ROOT / "splits" / "split_scene_v2.json", ROOT / "reports" / "symbolic_v3_precision" / "final_config.json",
                   ROOT / "processed_data" / "kinect_height", out)
            s = read_predictions(out, "alarm")
            y = {}
            for v in {k[0] for k in s}:
                lab = np.load(ROOT / "processed_data" / "cfg_d31bfed741ab" / "urfd_npy" / f"{v}_frame_labels.npy")
                for t, l in enumerate(lab, 1):
                    y[v, t] = int(l)
            tp = sum(1 for k, (a, _) in s.items() if a == 1 and y[k] == 1)
            fp = sum(1 for k, (a, _) in s.items() if a == 1 and y[k] == 0)
            ref = json.load(open(ROOT / "reports" / "symbolic_v3_precision" / "results.json"))["test"]["E (selected)"]
            self.assertEqual((tp, fp), (ref["TP"], ref["FP"]))


if __name__ == "__main__":
    unittest.main()
