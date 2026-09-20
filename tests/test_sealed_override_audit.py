from __future__ import annotations

from types import MappingProxyType
import unittest
from unittest.mock import patch

from pokezero.policy import PolicyDecision
from pokezero.rollout import RolloutSealedPreStepBoundary
from pokezero.mcts_eval.sealed_override_audit import (
    SealedOverrideAuditError,
    evaluate_measured_override_boundary,
)


def _boundary(*, override: bool | None, cause: str | None = None) -> RolloutSealedPreStepBoundary:
    override_metadata = {
        "model_override": override,
        "unmeasured_cause": cause,
        "model_argmax": 2,
        "search_argmax": 4,
        "root_q_gap": 0.125,
        "root_visit_gap": 0.5,
        "root_gap_action_indices": [4, 2],
        "root_allocation": {
            "worlds": 4,
            "prior_authority": True,
            "prior_cause": None,
            "arms": [
                {"move": "private-move-a", "action_index": 2, "visit_share": 0.25, "q": 0.25, "reported_prior": 0.6, "model_prior": 0.6},
                {"move": "private-move-b", "action_index": 4, "visit_share": 0.75, "q": 0.375, "reported_prior": 0.4, "model_prior": 0.4},
            ],
        },
    }
    metadata = {"engine_mcts": {"override": override_metadata}}
    return RolloutSealedPreStepBoundary(
        seed=20_260_920,
        battle_id="audit-battle",
        decision_round_index=3,
        requested_players=("p1", "p2"),
        snapshot=object(),
        decisions=MappingProxyType(
            {
                "p1": PolicyDecision(action_index=4, policy_id="mcts", metadata=metadata),
                "p2": PolicyDecision(action_index=7, policy_id="raw"),
            }
        ),
    )


class SealedOverrideAuditTest(unittest.TestCase):
    def test_rejects_flat_telemetry_envelope(self) -> None:
        boundary = _boundary(override=True)
        decisions = dict(boundary.decisions)
        decisions["p1"] = PolicyDecision(
            action_index=4,
            policy_id="mcts",
            metadata=boundary.decisions["p1"].metadata["engine_mcts"]["override"],
        )
        malformed = RolloutSealedPreStepBoundary(
            seed=boundary.seed,
            battle_id=boundary.battle_id,
            decision_round_index=boundary.decision_round_index,
            requested_players=boundary.requested_players,
            snapshot=boundary.snapshot,
            decisions=MappingProxyType(decisions),
        )
        with self.assertRaisesRegex(SealedOverrideAuditError, "engine MCTS metadata"):
            evaluate_measured_override_boundary(
                boundary=malformed,
                candidate_seat="p1",
                env_factory=lambda: self.fail("must not allocate environment"),
                continuation_policy_factory=lambda: self.fail("must not allocate policies"),
                rollout_config=object(),
            )

    def test_filters_clean_and_unmeasured_roots_from_override_denominator(self) -> None:
        for boundary in (_boundary(override=False), _boundary(override=None, cause="no-prior")):
            with self.subTest(boundary=boundary.decisions["p1"].metadata):
                self.assertIsNone(
                    evaluate_measured_override_boundary(
                        boundary=boundary,
                        candidate_seat="p1",
                        env_factory=lambda: self.fail("must not allocate environment"),
                        continuation_policy_factory=lambda: self.fail("must not allocate policies"),
                        rollout_config=object(),
                    )
                )

    def test_binds_actual_actions_and_sanitized_root_evidence(self) -> None:
        boundary = _boundary(override=True)
        expected = {"paired": "readout"}
        with patch(
            "pokezero.mcts_eval.sealed_override_audit.evaluate_sealed_override_pair",
            return_value=expected,
        ) as evaluate:
            readout = evaluate_measured_override_boundary(
                boundary=boundary,
                candidate_seat="p1",
                env_factory=lambda: object(),
                continuation_policy_factory=lambda: {"p1": object(), "p2": object()},
                rollout_config=object(),
                max_continuation_decision_rounds=20,
            )

        kwargs = evaluate.call_args.kwargs
        self.assertIs(kwargs["snapshot"], boundary.snapshot)
        self.assertEqual(kwargs["source_battle_id"], boundary.battle_id)
        self.assertEqual(kwargs["mcts_action"], 4)
        self.assertEqual(kwargs["raw_action"], 2)
        self.assertEqual(kwargs["opponent_action"], 7)
        self.assertEqual(kwargs["search_evidence"]["root_q_gap"], 0.125)
        self.assertNotIn("move", kwargs["search_evidence"]["root_allocation"]["arms"][0])
        self.assertEqual(readout["audit"], expected)
        self.assertNotIn("snapshot", readout)

    def test_records_one_sided_boundary_without_allocating_a_continuation(self) -> None:
        boundary = _boundary(override=True)
        one_sided = RolloutSealedPreStepBoundary(
            seed=boundary.seed,
            battle_id=boundary.battle_id,
            decision_round_index=boundary.decision_round_index,
            requested_players=("p1",),
            snapshot=boundary.snapshot,
            decisions=MappingProxyType({"p1": boundary.decisions["p1"]}),
        )
        readout = evaluate_measured_override_boundary(
            boundary=one_sided,
            candidate_seat="p1",
            env_factory=lambda: self.fail("must not allocate environment"),
            continuation_policy_factory=lambda: self.fail("must not allocate policies"),
            rollout_config=object(),
        )
        self.assertEqual(readout["audit_status"], "INAPPLICABLE_NON_SIMULTANEOUS")
        self.assertEqual(readout["requested_players"], ["p1"])
        self.assertEqual(readout["search_evidence"]["search_argmax"], 4)

    def test_ignores_opponent_only_forced_phase(self) -> None:
        boundary = _boundary(override=True)
        opponent_only = RolloutSealedPreStepBoundary(
            seed=boundary.seed,
            battle_id=boundary.battle_id,
            decision_round_index=boundary.decision_round_index,
            requested_players=("p2",),
            snapshot=boundary.snapshot,
            decisions=MappingProxyType({"p2": boundary.decisions["p2"]}),
        )
        self.assertIsNone(
            evaluate_measured_override_boundary(
                boundary=opponent_only,
                candidate_seat="p1",
                env_factory=lambda: self.fail("must not allocate environment"),
                continuation_policy_factory=lambda: self.fail("must not allocate policies"),
                rollout_config=object(),
            )
        )

    def test_rejects_measured_override_when_committed_action_disagrees_with_metadata(self) -> None:
        boundary = _boundary(override=True)
        decisions = dict(boundary.decisions)
        decisions["p1"] = PolicyDecision(
            action_index=3, policy_id="mcts", metadata=decisions["p1"].metadata
        )
        malformed = RolloutSealedPreStepBoundary(
            seed=boundary.seed,
            battle_id=boundary.battle_id,
            decision_round_index=boundary.decision_round_index,
            requested_players=boundary.requested_players,
            snapshot=boundary.snapshot,
            decisions=MappingProxyType(decisions),
        )
        with self.assertRaisesRegex(SealedOverrideAuditError, "committed MCTS action"):
            evaluate_measured_override_boundary(
                boundary=malformed,
                candidate_seat="p1",
                env_factory=lambda: self.fail("must not allocate environment"),
                continuation_policy_factory=lambda: self.fail("must not allocate policies"),
                rollout_config=object(),
            )
