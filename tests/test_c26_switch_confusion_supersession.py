"""Fail-closed unit coverage for C26 supersession verifier evidence parsing."""

from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path
from unittest.mock import patch


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "verify_c26_switch_confusion_supersession.py"
SPEC = importlib.util.spec_from_file_location("c26_switch_verifier", SCRIPT)
assert SPEC and SPEC.loader
VERIFIER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(VERIFIER)

TEST_NAME = VERIFIER.CURRENT_REGRESSION


def cargo_output(*, test_line: str, count: int = 22, ignored: int = 0, filtered: int = 0) -> str:
    passed = count - ignored
    return (
        f"running {count} tests\n"
        f"{test_line}\n\n"
        "test result: ok. "
        f"{passed} passed; 0 failed; {ignored} ignored; 0 measured; "
        f"{filtered} filtered out; finished in 0.00s\n"
    )


class CargoEvidenceTests(unittest.TestCase):
    def test_accepts_exact_named_pass_with_complete_unfiltered_suite(self) -> None:
        evidence = VERIFIER.require_cargo_regression_evidence(
            cargo_output(test_line=f"test {TEST_NAME} ... ok"), TEST_NAME
        )

        self.assertEqual(evidence["test"], TEST_NAME)
        self.assertEqual(evidence["expected_count"], VERIFIER.EXPECTED_RENDERER_TEST_COUNT)
        self.assertEqual(evidence["passed"], VERIFIER.EXPECTED_RENDERER_TEST_COUNT)

    def test_21_test_suite_fails_closed(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "expected 22, got 21"):
            VERIFIER.require_cargo_regression_evidence(
                cargo_output(test_line=f"test {TEST_NAME} ... ok", count=21), TEST_NAME
            )

    def test_missing_named_test_fails_closed(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "does not prove required regression"):
            VERIFIER.require_cargo_regression_evidence(
                cargo_output(test_line="test another_renderer_test ... ok"), TEST_NAME
            )

    def test_filtered_suite_fails_closed(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "filtered tests"):
            VERIFIER.require_cargo_regression_evidence(
                cargo_output(test_line=f"test {TEST_NAME} ... ok", count=1, filtered=21), TEST_NAME
            )

    def test_named_ignored_test_fails_closed(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "does not prove required regression"):
            VERIFIER.require_cargo_regression_evidence(
                cargo_output(test_line=f"test {TEST_NAME} ... ignored", ignored=1), TEST_NAME
            )

    def test_any_ignored_test_in_suite_fails_closed(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "ignored tests"):
            VERIFIER.require_cargo_regression_evidence(
                cargo_output(test_line=f"test {TEST_NAME} ... ok", ignored=1), TEST_NAME
            )

    def test_zero_runnable_tests_fails_closed(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "no runnable tests"):
            VERIFIER.require_cargo_regression_evidence(
                cargo_output(test_line=f"test {TEST_NAME} ... ok", count=0), TEST_NAME
            )


class OriginMainRefreshTests(unittest.TestCase):
    def test_refreshes_origin_main_before_returning_claim_base(self) -> None:
        with patch.object(VERIFIER, "command", side_effect=["", "abc123\n"]) as command:
            authoritative_main = VERIFIER.refresh_authoritative_origin_main(REPO_ROOT)

        self.assertEqual(authoritative_main, "abc123")
        self.assertEqual(command.call_count, 2)
        self.assertEqual(
            command.call_args_list[0].args[2],
            [
                "git",
                "-C",
                str(REPO_ROOT),
                "fetch",
                "--no-tags",
                "origin",
                "+refs/heads/main:refs/remotes/origin/main",
            ],
        )

    def test_refresh_failure_prevents_any_tracking_ref_claim(self) -> None:
        with patch.object(VERIFIER, "command", side_effect=RuntimeError("fetch failed")) as command:
            with self.assertRaisesRegex(RuntimeError, "fetch failed"):
                VERIFIER.refresh_authoritative_origin_main(REPO_ROOT)

        self.assertEqual(command.call_count, 1)


if __name__ == "__main__":
    unittest.main()
