"""Tests for data_audit.py (split integrity). Run: python -m unittest tests_data_audit.py"""
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from data_audit import audit


class TestAudit(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
