"""Tests for source-bound, non-historical deadline-profile receipts."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS))
SPEC = importlib.util.spec_from_file_location(
    "create_mcts_source_bound_deadline_profile_receipt_under_test",
    SCRIPTS / "create_mcts_source_bound_deadline_profile_receipt.py",
)
assert SPEC is not None and SPEC.loader is not None
receipt_tool = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = receipt_tool
SPEC.loader.exec_module(receipt_tool)


class SourceBoundDeadlineProfileReceiptTest(unittest.TestCase):
    def test_receipt_binds_the_current_engine_hash_not_the_historical_pin(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary) / "source"
            engine = source / "src/pokezero/engine_search.py"
            engine.parent.mkdir(parents=True)
            engine.write_text("current engine\n", encoding="utf-8")
            for relative in (
                "scripts/run_mcts_deadline_qualification.py",
                receipt_tool.PROFILE_RUNNER,
                receipt_tool.FIXED_WORK_PARITY_RUNNER,
                receipt_tool.LEAF_ABLATION_RUNNER,
                receipt_tool.LEAF_CONTINUATION_RUNNER,
                "scripts/engine_build_fingerprint.py",
                "src/pokezero/mcts_eval/deadline_qualification.py",
                "src/pokezero/mcts_eval/lattice.py",
            ):
                path = source / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("# source\n", encoding="utf-8")
            b2 = {
                "model_runtime": {"engine_fingerprint": "b" * 64},
                "immutable_image": "registry.example/pokezero@sha256:" + "c" * 64,
            }
            with (
                mock.patch.object(receipt_tool.deadline_receipt, "_clean_detached_commit", return_value="a" * 40),
                mock.patch.object(receipt_tool.deadline_receipt, "_read_canonical_json", return_value=b2),
                mock.patch.object(receipt_tool.deadline_receipt, "_validate_b2_receipt"),
                mock.patch.object(
                    receipt_tool.deadline_receipt,
                    "_execution_source_files",
                    return_value=[engine],
                ),
                mock.patch.object(
                    receipt_tool.deadline_receipt,
                    "_assignment_literals",
                    return_value={
                        "REQUIRED_RECEIPT_FILES": (
                            "src/pokezero/engine_search.py",
                            "scripts/run_mcts_deadline_qualification.py",
                        )
                    },
                ),
            ):
                receipt = receipt_tool.build_receipt(source_root=source, b2_receipt_path=source / "b2.json")
            self.assertEqual(receipt["engine_fingerprint"], "b" * 64)
            self.assertEqual(
                receipt["source_files_sha256"]["src/pokezero/engine_search.py"],
                receipt_tool.deadline_receipt._sha256(engine),
            )
            self.assertIn(receipt_tool.PROFILE_RUNNER, receipt["source_files_sha256"])
            self.assertIn(receipt_tool.FIXED_WORK_PARITY_RUNNER, receipt["source_files_sha256"])
            self.assertIn(receipt_tool.LEAF_ABLATION_RUNNER, receipt["source_files_sha256"])
            self.assertIn(receipt_tool.LEAF_CONTINUATION_RUNNER, receipt["source_files_sha256"])


if __name__ == "__main__":
    unittest.main()
