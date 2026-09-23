from __future__ import annotations

from types import MappingProxyType
import unittest
from unittest.mock import patch

from pokezero.policy import PolicyDecision
from pokezero.rollout import RolloutSealedPreStepBoundary
from pokezero.mcts_eval.sealed_root_action_audit import (
    SealedRootActionAuditError,
    evaluate_root_action_boundary,
    root_action_candidates,
)


def _override(*, measured: bool | None = True) -> dict[str, object]:
    return {
        "model_override": measured,
        "unmeasured_cause": None if measured is not None else "no-prior",
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
                {"move": "private-a", "action_index": 2, "visit_share": 0.25, "q": 0.25,
                 "reported_prior": 0.6, "model_prior": 0.6},
                {"move": "private-b", "action_index": 4, "visit_share": 0.5, "q": 0.375,
                 "reported_prior": 0.3, "model_prior": 0.3},
                {"move": "private-c", "action_index": 7, "visit_share": 0.25, "q": 0.1,
                 "reported_prior": 0.1, "model_prior": 0.1},
            ],
        },
    }


def _boundary(*, measured: bool | None = True) -> RolloutSealedPreStepBoundary:
    metadata = {"engine_mcts": {"override": _override(measured=measured)}}
    return RolloutSealedPreStepBoundary(
        seed=20_260_923,
        battle_id="root-audit",
        decision_round_index=5,
        requested_players=("p1", "p2"),
        snapshot=object(),
        decisions=MappingProxyType({
            "p1": PolicyDecision(action_index=4, policy_id="mcts", metadata=metadata),
            "p2": PolicyDecision(action_index=1, policy_id="raw"),
        }),
        policy_observation_histories=MappingProxyType({
            "p1": (object(),),
            "p2": (object(),),
        }),
    )


class SealedRootActionAuditTest(unittest.TestCase):
    def test_candidate_set_keeps_raw_mcts_and_next_visited_action(self) -> None:
        evidence, actions = root_action_candidates(_override())
        self.assertEqual(actions, {"raw_policy": 2, "mcts_selected": 4, "visit_alternative": 7})
        self.assertNotIn("move", evidence["root_allocation"]["arms"][0])

    def test_candidate_set_does_not_promote_an_unvisited_action(self) -> None:
        override = _override()
        override["root_allocation"]["arms"][2]["visit_share"] = 0.0
        override["root_allocation"]["arms"][1]["visit_share"] = 0.75
        evidence, actions = root_action_candidates(override)
        self.assertEqual(actions, {"raw_policy": 2, "mcts_selected": 4})
        self.assertEqual(evidence["root_allocation"]["arms"][2]["visit_share"], 0.0)

    def test_non_override_is_not_in_action_audit_stratum(self) -> None:
        boundary = _boundary(measured=False)
        self.assertIsNone(
            evaluate_root_action_boundary(
                boundary=boundary,
                candidate_seat="p1",
                env_factory=lambda: self.fail("must not allocate environment"),
                continuation_policy_factory_builder=lambda _: {"policy_consistent": lambda: {}},
                continuation_rng_seeds=[1],
                rollout_config=object(),
            )
        )

    def test_binds_actual_source_actions_without_retaining_opponent_action(self) -> None:
        boundary = _boundary()
        grid = {"complete": "grid"}
        received_histories = []
        with patch(
            "pokezero.mcts_eval.sealed_root_action_audit.evaluate_sealed_root_action_grid",
            return_value=grid,
        ) as evaluate:
            readout = evaluate_root_action_boundary(
                boundary=boundary,
                candidate_seat="p1",
                env_factory=lambda: object(),
                continuation_policy_factory_builder=lambda histories: (
                    received_histories.append(histories)
                    or {
                        "deployed_raw": lambda: {"p1": object(), "p2": object()},
                        "policy_consistent": lambda: {"p1": object(), "p2": object()},
                        "uniform_own": lambda: {"p1": object(), "p2": object()},
                    }
                ),
                continuation_rng_seeds={
                    "deployed_raw": [101],
                    "policy_consistent": [101, 102],
                    "uniform_own": [101, 102],
                },
                rollout_config=object(),
                max_continuation_decision_rounds=20,
            )

        kwargs = evaluate.call_args.kwargs
        self.assertIs(kwargs["snapshot"], boundary.snapshot)
        self.assertEqual(kwargs["actions"], {"raw_policy": 2, "mcts_selected": 4, "visit_alternative": 7})
        self.assertEqual(kwargs["opponent_action"], 1)
        self.assertEqual(
            kwargs["continuation_rng_seeds"],
            {
                "deployed_raw": [101],
                "policy_consistent": [101, 102],
                "uniform_own": [101, 102],
            },
        )
        self.assertEqual(received_histories, [boundary.policy_observation_histories])
        self.assertEqual(readout["audit"], grid)
        self.assertNotIn("snapshot", readout)
        self.assertNotIn("opponent_action", readout)

    def test_root_candidates_require_a_real_override(self) -> None:
        with self.assertRaisesRegex(SealedRootActionAuditError, "measured MCTS override"):
            root_action_candidates(_override(measured=False))
