from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest import mock

from pokezero.mcts_eval.deadline_qualification import DeadlineQualificationError


def _runner():
    path = Path(__file__).resolve().parents[1] / "scripts" / "run_mcts_own_policy_prior_replay.py"
    spec = importlib.util.spec_from_file_location("own_prior_replay_test_runner", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _payload(*, allocation):
    return {
        "decision_id": "public-decision-1",
        "engine_mcts": {"override": {"root_allocation": allocation}},
    }


class RootAllocationTest(unittest.TestCase):
    def test_guided_root_requires_authoritative_normalized_model_priors(self) -> None:
        runner = _runner()
        allocation = runner._allocation(
            _payload(
                allocation={
                    "worlds": 4,
                    "prior_authority": True,
                    "prior_cause": None,
                    "arms": [
                        {"move": "a", "visit_share": .6, "q": .3, "reported_prior": .2, "model_prior": .2},
                        {"move": "b", "visit_share": .4, "q": .2, "reported_prior": .8, "model_prior": .8},
                    ],
                }
            ),
            arm="guided",
            worlds=4,
        )
        self.assertTrue(allocation["prior_authority"])
        self.assertEqual([arm["model_prior"] for arm in allocation["arms"]], [.2, .8])

    def test_uniform_root_is_not_allowed_to_manufacture_model_preference(self) -> None:
        runner = _runner()
        payload = _payload(
            allocation={
                "worlds": 4,
                "prior_authority": False,
                "prior_cause": "no_root_priors",
                "arms": [
                    {"move": "a", "visit_share": .5, "q": .3, "reported_prior": .5, "model_prior": None},
                    {"move": "b", "visit_share": .5, "q": .2, "reported_prior": .5, "model_prior": None},
                ],
            }
        )
        self.assertFalse(runner._allocation(payload, arm="uniform", worlds=4)["prior_authority"])
        payload["engine_mcts"]["override"]["root_allocation"]["arms"][0]["model_prior"] = .5
        with self.assertRaisesRegex(DeadlineQualificationError, "manufactured a model prior"):
            runner._allocation(payload, arm="uniform", worlds=4)

    def test_allocation_difference_requires_the_same_mapped_moves(self) -> None:
        runner = _runner()
        guided = {"root_allocation": {"arms": [{"move": "a", "visit_share": .7}, {"move": "b", "visit_share": .3}]}}
        uniform = {"root_allocation": {"arms": [{"move": "a", "visit_share": .5}, {"move": "b", "visit_share": .5}]}}
        self.assertTrue(runner._allocation_changed(guided, uniform))
        uniform["root_allocation"]["arms"][1]["move"] = "c"
        self.assertFalse(runner._allocation_changed(guided, uniform))

    def test_runner_explicitly_enables_existing_override_telemetry(self) -> None:
        runner = _runner()
        captured = {}

        def fake_decider(*args, **kwargs):
            captured.update(kwargs)
            return object()

        with mock.patch.object(runner, "_LiveEngineTimingDecider", fake_decider):
            runner._new_decider(
                object(),
                SimpleNamespace(
                    showdown_root="/showdown",
                    deadline_ms=1000,
                    native_batch_guard_ms=64,
                    model_world_workers=1,
                ),
                guided=True,
            )
        self.assertTrue(captured["override_telemetry"])
        self.assertTrue(captured["model_priors"])
        self.assertFalse(captured["use_opponent_priors"])

    def test_runner_refuses_the_old_deadline_specific_receipt_schema(self) -> None:
        runner = _runner()
        with mock.patch.object(
            runner.common,
            "_read_json",
            return_value={"schema_version": "pokezero.mcts-deadline-source-receipt.v1"},
        ):
            with self.assertRaisesRegex(DeadlineQualificationError, "B2 image receipt"):
                runner._receipt_and_source("/does/not/matter.json")

    def test_runner_binds_active_source_and_native_fingerprint_to_b2_receipt(self) -> None:
        runner = _runner()
        commit = "a" * 40
        fingerprint = "b" * 64
        receipt = {
            "schema_version": "pokezero.b2-source-image-receipt.v7",
            "complete": True,
            "immutable_image": "registry.example/pokezero@sha256:" + "c" * 64,
            "image_digest": "sha256:" + "c" * 64,
            "source_commit": commit,
            "model_runtime": {
                "source": {"commit": commit, "tree_status": "clean_tracked_checkout"},
                "engine_fingerprint": fingerprint,
            },
        }
        active = {"commit": commit, "execution_tree_sha256": "d" * 64}
        with (
            mock.patch.object(runner.common, "_read_json", return_value=receipt),
            mock.patch.object(runner.common, "_active_source_provenance", return_value=active),
            mock.patch.object(runner.common, "assert_fresh"),
            mock.patch.object(runner.common, "compute_fingerprint", return_value={"fingerprint": fingerprint}),
        ):
            actual_receipt, actual_active = runner._receipt_and_source("/does/not/matter.json")
        self.assertEqual(actual_receipt, receipt)
        self.assertEqual(actual_active, active)


if __name__ == "__main__":
    unittest.main()
