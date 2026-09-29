"""Regression coverage for the current-source deadline-profile bridge."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import unittest
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS))
SPEC = importlib.util.spec_from_file_location(
    "mcts_source_bound_deadline_profile_under_test",
    SCRIPTS / "run_mcts_source_bound_deadline_profile.py",
)
assert SPEC is not None and SPEC.loader is not None
profile = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = profile
SPEC.loader.exec_module(profile)


class SourceBoundDeadlineProfileTest(unittest.TestCase):
    def _receipt(self) -> dict[str, object]:
        return {
            "source_commit": "a" * 40,
            "engine_fingerprint": "b" * 64,
            "source_files_sha256": {
                "src/pokezero/engine_search.py": "c" * 64,
                profile.SELF_RELATIVE_PATH: "d" * 64,
            },
        }

    def test_profile_requires_its_own_file_in_the_source_receipt(self) -> None:
        with mock.patch.object(
            profile.legacy,
            "_source_receipt",
            return_value={
                **self._receipt(),
                "source_files_sha256": {"src/pokezero/engine_search.py": "c" * 64},
            },
        ):
            with self.assertRaisesRegex(profile.legacy.DeadlineQualificationError, "executing wrapper"):
                profile._source_bound_receipt("/receipt.json")

    def test_profile_uses_the_receipt_identity_and_never_calls_it_reviewed(self) -> None:
        receipt = self._receipt()
        original = profile.legacy._deadline_mechanics_evidence
        original_commit = profile.legacy.REVIEWED_DEADLINE_SOURCE_COMMIT
        original_hash = profile.legacy.REVIEWED_ENGINE_SEARCH_SHA256
        original_fingerprint = profile.legacy.REVIEWED_ENGINE_FINGERPRINT
        original_files = profile.legacy.REQUIRED_RECEIPT_FILES
        try:
            with mock.patch.object(
                profile.legacy,
                "_deadline_mechanics_evidence",
                return_value={"reviewed_source_commit": "historical", "other": 1},
            ):
                profile._configure_legacy(receipt)
                evidence = profile.legacy._deadline_mechanics_evidence(receipt)
            self.assertEqual(profile.legacy.REVIEWED_DEADLINE_SOURCE_COMMIT, "a" * 40)
            self.assertEqual(profile.legacy.REVIEWED_ENGINE_SEARCH_SHA256, "c" * 64)
            self.assertEqual(profile.legacy.REVIEWED_ENGINE_FINGERPRINT, "b" * 64)
            self.assertIn(profile.SELF_RELATIVE_PATH, profile.legacy.REQUIRED_RECEIPT_FILES)
            self.assertNotIn("reviewed_source_commit", evidence)
            self.assertEqual(evidence["source_bound_execution_commit"], "a" * 40)
            self.assertEqual(evidence["qualification_kind"], "source_bound_deadline_profile")
        finally:
            profile.legacy._deadline_mechanics_evidence = original
            profile.legacy.REVIEWED_DEADLINE_SOURCE_COMMIT = original_commit
            profile.legacy.REVIEWED_ENGINE_SEARCH_SHA256 = original_hash
            profile.legacy.REVIEWED_ENGINE_FINGERPRINT = original_fingerprint
            profile.legacy.REQUIRED_RECEIPT_FILES = original_files


if __name__ == "__main__":
    unittest.main()
