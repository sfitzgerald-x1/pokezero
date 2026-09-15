"""Independent evidence-contract tests for the fixed-deadline qualification."""

from __future__ import annotations

import copy
import unittest

from pokezero.mcts_eval.deadline_qualification import (
    DEADLINE_QUALIFICATION_SCHEMA_VERSION,
    DeadlineQualificationError,
    DeadlineQualificationRequirements,
    validate_deadline_qualification,
)


def _record(index: int, *, prefix: bool) -> dict:
    completed = 128 if prefix else 256
    return {
        "decision_id": f"decision-{index:02d}",
        "corpus_record_sha256": f"{index:x}" * 64,
        "root_action": "move 1",
        "outer_wall_ms": 1_010.0 + index,
        "invalid_actions": 0,
        "engine_mcts": {
            "leaf_eval": "model",
            "worlds_constructed": 4,
            "worlds_searched": 4,
            "time_budget": {
                "scope": "whole_model_decision",
                "requested_ms": 1000,
                "native_batch_guard_ms": 0,
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
        self.assertEqual(summary["schema_version"], DEADLINE_QUALIFICATION_SCHEMA_VERSION)
        self.assertEqual(summary["native_prefix_count"], 1)
        self.assertEqual(summary["deadline_overshoot_ms"]["max"], 5.0)
        self.assertEqual(summary["zero_completed_world_refusals"], 0)
        self.assertEqual(summary["world_coverage"]["constructed_total"], 8)
        self.assertEqual(summary["world_coverage"]["searched_total"], 8)

    def test_native_batch_guard_must_match_the_frozen_contract(self) -> None:
        requirements = DeadlineQualificationRequirements(
            expected_decisions=2,
            native_batch_guard_ms=64,
        )
        records = [_record(0, prefix=True), _record(1, prefix=False)]
        for record in records:
            record["engine_mcts"]["time_budget"]["native_batch_guard_ms"] = 64
        summary = validate_deadline_qualification(records, requirements=requirements)
        self.assertEqual(summary["requirements"]["native_batch_guard_ms"], 64)

        records[1]["engine_mcts"]["time_budget"]["native_batch_guard_ms"] = 63
        with self.assertRaisesRegex(DeadlineQualificationError, "native batch guard"):
            validate_deadline_qualification(records, requirements=requirements)

    def test_parallel_deadline_qualification_requires_remaining_budget_dispatch(self) -> None:
        records = [_record(0, prefix=True), _record(1, prefix=False)]
        for record in records:
            record["engine_mcts"]["world_parallelism"] = {
                "workers": 2,
                "native_invocations": 2,
                "independent_native_models": 2,
                "mode": "deadline_remaining_budget",
            }
        requirements = DeadlineQualificationRequirements(
            expected_decisions=2, model_world_workers=2
        )
        summary = validate_deadline_qualification(records, requirements=requirements)
        self.assertEqual(summary["decision_count"], 2)

        records[0]["engine_mcts"]["world_parallelism"]["mode"] = "fixed_work"
        with self.assertRaisesRegex(DeadlineQualificationError, "remaining-budget"):
            validate_deadline_qualification(records, requirements=requirements)

    def test_parallel_deadline_qualification_requires_independent_native_models(self) -> None:
        records = [_record(0, prefix=True), _record(1, prefix=False)]
        for record in records:
            record["engine_mcts"]["world_parallelism"] = {
                "workers": 2,
                "native_invocations": 2,
                "independent_native_models": 2,
                "mode": "deadline_remaining_budget",
            }
        requirements = DeadlineQualificationRequirements(
            expected_decisions=2, model_world_workers=2
        )

        for invalid_value in (None, 1, 3):
            with self.subTest(independent_native_models=invalid_value):
                invalid = copy.deepcopy(records)
                if invalid_value is None:
                    del invalid[0]["engine_mcts"]["world_parallelism"][
                        "independent_native_models"
                    ]
                else:
                    invalid[0]["engine_mcts"]["world_parallelism"][
                        "independent_native_models"
                    ] = invalid_value
                with self.assertRaisesRegex(
                    DeadlineQualificationError, "independent_native_models|independent native model"
                ):
                    validate_deadline_qualification(invalid, requirements=requirements)

    def test_missing_serialized_root_action_cannot_resume_as_valid_evidence(self) -> None:
        record = _record(0, prefix=True)
        del record["root_action"]
        with self.assertRaisesRegex(DeadlineQualificationError, "root_action"):
            validate_deadline_qualification(
                [record, _record(1, prefix=False)], requirements=self.requirements
            )

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
