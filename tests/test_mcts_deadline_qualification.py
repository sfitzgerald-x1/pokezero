"""Independent evidence-contract tests for the fixed-deadline qualification."""

from __future__ import annotations

import copy
import unittest

from pokezero.mcts_eval.deadline_qualification import (
    DeadlineQualificationError,
    DeadlineQualificationRequirements,
    validate_deadline_qualification,
)


def _record(index: int, *, prefix: bool) -> dict:
    completed = 128 if prefix else 256
    return {
        "decision_id": f"decision-{index:02d}",
        "corpus_record_sha256": f"{index:x}" * 64,
        "outer_wall_ms": 1_010.0 + index,
        "invalid_actions": 0,
        "engine_mcts": {
            "leaf_eval": "model",
            "worlds_constructed": 4,
            "worlds_searched": 4,
            "time_budget": {
                "scope": "whole_model_decision",
                "requested_ms": 1000,
                "deadline_elapsed_ms": 1005.0 if prefix else 900.0,
                "deadline_overshoot_ms": 5.0 if prefix else 0.0,
                "exhausted": prefix,
                "worlds_budget_skipped": 0,
                "native_invocations": [
                    {
                        "status": "completed",
                        "multiplicity": 1,
                        "requested_iterations": 256,
                        "completed_iterations": completed,
                        "remaining_iterations": 256 - completed,
                        "time_budget_ms": 900 if prefix else 1000,
                        "time_budget_elapsed_ms": 901.0 if prefix else 900.0,
                        "time_budget_batch_overshoot_ms": 1.0 if prefix else 0.0,
                        "time_budget_exhausted": prefix,
                        "root_visits": {"side_one": completed, "side_two": completed},
                    },
                    {
                        "status": "completed",
                        "multiplicity": 1,
                        "requested_iterations": 256,
                        "completed_iterations": 256,
                        "remaining_iterations": 0,
                        "time_budget_ms": 900 if prefix else 1000,
                        "time_budget_elapsed_ms": 900.0,
                        "time_budget_batch_overshoot_ms": 0.0,
                        "time_budget_exhausted": False,
                        "root_visits": {"side_one": 256, "side_two": 256},
                    },
                ],
            },
        },
    }


class DeadlineQualificationTest(unittest.TestCase):
    def setUp(self) -> None:
        self.requirements = DeadlineQualificationRequirements(expected_decisions=2)

    def test_complete_evidence_needs_one_nonzero_prefix(self) -> None:
        summary = validate_deadline_qualification(
            [_record(0, prefix=True), _record(1, prefix=False)], requirements=self.requirements
        )
        self.assertEqual(summary["decision_count"], 2)
        self.assertEqual(summary["native_prefix_count"], 1)
        self.assertEqual(summary["deadline_overshoot_ms"]["max"], 5.0)

    def test_refused_native_witness_cannot_be_hidden_by_another_world(self) -> None:
        record = _record(0, prefix=True)
        record["engine_mcts"]["time_budget"]["native_invocations"][1] = {
            "status": "refused",
            "multiplicity": 1,
            "requested_iterations": 256,
            "time_budget_ms": 900,
            "refusal": "native_time_budget_unsupported",
        }
        with self.assertRaisesRegex(DeadlineQualificationError, "not completed"):
            validate_deadline_qualification(
                [record, _record(1, prefix=False)], requirements=self.requirements
            )

    def test_root_visit_mismatch_is_not_a_valid_prefix(self) -> None:
        record = _record(0, prefix=True)
        record["engine_mcts"]["time_budget"]["native_invocations"][0]["root_visits"]["side_two"] = 127
        with self.assertRaisesRegex(DeadlineQualificationError, "root visits"):
            validate_deadline_qualification(
                [record, _record(1, prefix=False)], requirements=self.requirements
            )

    def test_no_prefix_is_a_nonpass_not_a_successful_full_work_run(self) -> None:
        with self.assertRaisesRegex(DeadlineQualificationError, "nonzero, completed native deadline prefix"):
            validate_deadline_qualification(
                [_record(0, prefix=False), _record(1, prefix=False)], requirements=self.requirements
            )

    def test_fallback_is_not_recast_as_a_deadline_record(self) -> None:
        record = copy.deepcopy(_record(0, prefix=True))
        record["engine_mcts"]["fallback"] = "model_time_budget_no_completed_worlds"
        with self.assertRaisesRegex(DeadlineQualificationError, "fallback"):
            validate_deadline_qualification(
                [record, _record(1, prefix=False)], requirements=self.requirements
            )


if __name__ == "__main__":
    unittest.main()
