"""Pins for the two-consumer engine build stamp."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import engine_build_fingerprint as fingerprint  # noqa: E402


class EngineBuildStampTests(unittest.TestCase):
    def test_write_stamp_records_both_installed_consumers(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            stamp = Path(tmp) / "stamp.json"
            artifacts = {
                "poke_engine": {"module_path": "/tmp/engine.py", "module_sha256": "a" * 64, "extensions": []},
                "pokezero_search": {"module_path": "/tmp/search.py", "module_sha256": "b" * 64, "extensions": []},
            }
            with (
                patch.object(fingerprint, "_stamp_path", return_value=stamp),
                patch.object(fingerprint, "compute_fingerprint", return_value={"fingerprint": "c" * 64}),
                patch.object(fingerprint, "_installed_artifacts", return_value=artifacts),
            ):
                fingerprint.write_stamp()
            payload = json.loads(stamp.read_text(encoding="utf-8"))
        self.assertEqual(payload["schema"], "pokezero-engine-build/2")
        self.assertEqual(payload["artifacts"], artifacts)

    def test_check_rejects_legacy_input_only_stamp(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            stamp = Path(tmp) / "stamp.json"
            stamp.write_text(json.dumps({"fingerprint": "c" * 64, "count": 1}), encoding="utf-8")
            with (
                patch.object(fingerprint, "_stamp_path", return_value=stamp),
                patch.object(fingerprint, "compute_fingerprint", return_value={"fingerprint": "c" * 64, "count": 1}),
                patch.object(fingerprint, "_installed_artifacts", return_value={}),
            ):
                problems = fingerprint.check(strict_mtime=False)
        self.assertTrue(any("does not attest both installed consumers" in problem for problem in problems))
        self.assertTrue(any("artifacts do not match" in problem for problem in problems))


if __name__ == "__main__":
    unittest.main()
