from __future__ import annotations

import importlib.util
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "run_q_root_selector_mutation_matrix.py"


def _module():
    spec = importlib.util.spec_from_file_location("q_root_selector_mutation_matrix", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class QRootSelectorMutationMatrixTest(unittest.TestCase):
    def test_accepts_an_explicit_receipt_bound_commit_without_git_metadata(self) -> None:
        module = _module()
        checked_out_commit = "b" * 40
        clean_head = subprocess.CompletedProcess(
            args=["git", "rev-parse", "HEAD"], returncode=0, stdout=checked_out_commit + "\n"
        )
        clean_status = subprocess.CompletedProcess(
            args=["git", "status", "--porcelain"], returncode=0, stdout=""
        )
        with patch.object(module.subprocess, "run", side_effect=[clean_head, clean_status]):
            self.assertEqual(module._source_commit(None), checked_out_commit)
        with patch.object(module.subprocess, "run", side_effect=[clean_head, clean_status]):
            self.assertEqual(module._source_commit(checked_out_commit), checked_out_commit)
        with patch.object(module.subprocess, "run", side_effect=[clean_head, clean_status]):
            with self.assertRaisesRegex(module.MutationError, "does not match"):
                module._source_commit("a" * 40)
        with patch.object(
            module.subprocess,
            "run",
            side_effect=subprocess.CalledProcessError(128, ["git", "rev-parse", "HEAD"]),
        ):
            self.assertEqual(module._source_commit("a" * 40), "a" * 40)
            with self.assertRaisesRegex(module.MutationError, "--source-commit"):
                module._source_commit("not-a-commit")

    def test_refuses_to_attribute_evidence_to_a_dirty_checkout(self) -> None:
        module = _module()
        clean_head = subprocess.CompletedProcess(
            args=["git", "rev-parse", "HEAD"], returncode=0, stdout="a" * 40 + "\n"
        )
        dirty_status = subprocess.CompletedProcess(
            args=["git", "status", "--porcelain"], returncode=0, stdout=" M source.py\n"
        )
        with patch.object(module.subprocess, "run", side_effect=[clean_head, dirty_status]):
            with self.assertRaisesRegex(module.MutationError, "not clean"):
                module._source_commit("a" * 40)

    def test_targeted_mutations_are_killed_against_the_mutated_src_copy(self) -> None:
        module = _module()
        # This in-process regression deliberately runs while the test checkout
        # is dirty.  Production CLI invocation does not patch `_source_commit`
        # and therefore refuses such a receipt; this test isolates mutation
        # execution from the separately tested provenance fence.
        with patch.object(module, "_source_commit", return_value="a" * 40):
            result = module.run(source_commit="a" * 40)
        self.assertTrue(result["complete"])
        self.assertTrue(result["all_killed"])
        self.assertEqual(result["baseline"]["status"], "CLEAN")
        self.assertEqual(result["baseline"]["tests_run"], module.EXPECTED_TEST_COUNT)
        self.assertEqual(
            {item["name"] for item in result["mutations"]},
            {item.name for item in module.MUTATIONS},
        )
        self.assertTrue(all(item["status"] == "KILLED" for item in result["mutations"]))
        self.assertTrue(all(item["exit_code"] != 0 for item in result["mutations"]))
        self.assertTrue(
            all(item["tests_run"] == module.EXPECTED_TEST_COUNT for item in result["mutations"])
        )

    def test_import_failure_cannot_be_counted_as_a_killed_mutation(self) -> None:
        module = _module()
        failed_import = subprocess.CompletedProcess(
            args=["python", "-m", "unittest"],
            returncode=1,
            stdout="ImportError: missing dependency\\nRan 0 tests in 0.001s\\nFAILED (errors=1)\\n",
        )
        with self.assertRaisesRegex(module.MutationError, "exactly"):
            module._validated_test_run(failed_import, clean_baseline=False)

    def test_runtime_errors_cannot_be_counted_as_a_killed_mutation(self) -> None:
        module = _module()
        failed_fixture = subprocess.CompletedProcess(
            args=["python", "-m", "unittest"],
            returncode=1,
            stdout=(
                "Ran 10 tests in 0.001s\n\n"
                "FAILED (errors=1)\n"
            ),
        )
        mixed_failure_and_error = subprocess.CompletedProcess(
            args=["python", "-m", "unittest"],
            returncode=1,
            stdout=(
                "Ran 10 tests in 0.001s\n\n"
                "FAILED (failures=1, errors=1)\n"
            ),
        )
        for result in (failed_fixture, mixed_failure_and_error):
            with self.assertRaisesRegex(module.MutationError, "assertion-only"):
                module._validated_test_run(result, clean_baseline=False)

    def test_write_once_is_idempotent_and_refuses_a_conflicting_receipt(self) -> None:
        module = _module()
        with tempfile.TemporaryDirectory() as directory:
            receipt = Path(directory) / "receipt.json"
            module._write_once(receipt, {"value": 1})
            module._write_once(receipt, {"value": 1})
            with self.assertRaisesRegex(module.MutationError, "refusing to replace"):
                module._write_once(receipt, {"value": 2})


if __name__ == "__main__":
    unittest.main()
