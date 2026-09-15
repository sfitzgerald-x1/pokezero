"""Predeclared final-action controls for the model MCTS driver.

This is deliberately a decision panel rather than another backup-conservation
test.  Each case supplies its known acting-seat payoff, a completed native root
report, and the expected legal action.  The fake native boundary leaves the
production policy responsible for the things this panel is meant to protect:
seat selection, root aggregation, visit-max recommendation, and request-action
mapping.  It therefore does not claim to prove native-tree or game strength;
those need the source-bound paired evaluation.
"""

from __future__ import annotations

import json
import os
import random
import sys
from dataclasses import dataclass
from types import SimpleNamespace
import unittest
from unittest.mock import patch


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from pokezero.engine_search import EngineMctsConfig, EngineMctsPolicy, EngineMctsStats  # noqa: E402


@dataclass(frozen=True)
class _PanelCase:
    """One declared action-quality control in acting-seat payoff units."""

    name: str
    expected_move: str
    payoffs: dict[str, float]
    visits: tuple[int, int]
    qs: tuple[float, float]
    priors: tuple[float, float]


_CASES = (
    _PanelCase(
        name="immediate_terminal_win_is_not_hidden",
        expected_move="alpha",
        payoffs={"alpha": 1.0, "beta": 0.25},
        visits=(224, 32),
        qs=(1.0, 0.25),
        priors=(0.10, 0.90),
    ),
    _PanelCase(
        name="variable_valued_nested_outcome",
        expected_move="beta",
        payoffs={"alpha": 0.35, "beta": 0.70},
        visits=(80, 176),
        qs=(0.35, 0.70),
        priors=(0.75, 0.25),
    ),
    _PanelCase(
        name="mixed_terminal_and_deferred_branches",
        expected_move="beta",
        payoffs={"alpha": 0.55, "beta": 1.0},
        visits=(96, 160),
        qs=(0.55, 1.0),
        priors=(0.80, 0.20),
    ),
    _PanelCase(
        name="misleading_prior_does_not_override_completed_tree",
        expected_move="alpha",
        payoffs={"alpha": 0.80, "beta": 0.20},
        visits=(192, 64),
        qs=(0.80, 0.20),
        priors=(0.01, 0.99),
    ),
    _PanelCase(
        name="rare_success_decoy_does_not_replace_supported_action",
        expected_move="alpha",
        payoffs={"alpha": 0.60, "beta": 0.20},
        visits=(176, 80),
        qs=(0.60, 0.95),
        priors=(0.45, 0.55),
    ),
    _PanelCase(
        name="equal_value_control_uses_stable_action_order",
        expected_move="alpha",
        payoffs={"alpha": 0.50, "beta": 0.50},
        visits=(128, 128),
        qs=(0.50, 0.50),
        priors=(0.50, 0.50),
    ),
)


class _Native:
    def __init__(self, report: dict) -> None:
        self._report = report
        self.calls: list[tuple] = []

    def search_batched_multi_encoded(self, *args):
        self.calls.append(args)
        return json.dumps(self._report)


def _context(player_id: str):
    observation = SimpleNamespace(
        legal_action_mask=(True, True, False, False, False, False, False, False, False),
        metadata={
            "action_candidates": [
                {"action_index": 0, "kind": "move", "legal": True, "move_id": "alpha"},
                {"action_index": 1, "kind": "move", "legal": True, "move_id": "beta"},
            ]
        },
    )
    return SimpleNamespace(
        observation=observation,
        public_materialization_state=SimpleNamespace(
            replay=SimpleNamespace(turn_number=1)
        ),
        player_id=player_id,
        battle_id="mcts-action-choice-panel",
        decision_round_index=0,
    )


def _world(player_id: str):
    side_key = "side_one" if player_id == "p1" else "side_two"
    return (
        SimpleNamespace(
            party_species={"p1": ("rattata",), "p2": ("chansey",)},
            slot_sides={player_id: side_key},
        ),
        SimpleNamespace(to_string=lambda: f"panel-{player_id}"),
    )


def _report(case: _PanelCase, player_id: str) -> dict:
    # The native crate stores Q in its side-one frame.  p2's displayed values
    # must therefore be the complement of this panel's acting-seat values.
    qs = case.qs if player_id == "p1" else tuple(1.0 - value for value in case.qs)
    arms = [
        {"move": move, "visits": visits, "q": q, "prior": prior}
        for move, visits, q, prior in zip(
            ("alpha", "beta"), case.visits, qs, case.priors, strict=True
        )
    ]
    result = {
        "iterations": sum(case.visits),
        "requested_iterations": sum(case.visits),
        "remaining_iterations": 0,
        "early_stopped": False,
        "model_evals": sum(case.visits),
        "lossy_renders": 0,
        "attribution_unsafe_renders": 0,
        "prior_fallbacks": 0,
        "root_priors": list(case.priors),
        "side_one": [],
        "side_two": [],
    }
    result["side_one" if player_id == "p1" else "side_two"] = arms
    return result


def _policy(batch: int) -> EngineMctsPolicy:
    policy = object.__new__(EngineMctsPolicy)
    policy.policy_id = "mcts-action-choice-panel"
    policy._config = EngineMctsConfig(
        worlds=1,
        leaf_eval="model",
        model_path="model.pt",
        checkpoint_path="checkpoint.pt",
        tables_path="tables.json",
        search_sims=256,
        search_batch=batch,
        override_telemetry=True,
        early_stop=False,
        strict_fallbacks=True,
    )
    policy._tables_json = "{}"
    policy.stats = EngineMctsStats()
    policy._world_failures_before = {}
    return policy


class MctsActionChoicePanelTests(unittest.TestCase):
    """Known-payoff controls through the production model-search driver."""

    def _run(self, case: _PanelCase, player_id: str, batch: int):
        policy = _policy(batch)
        native = _Native(_report(case, player_id))
        fake_module = SimpleNamespace(
            FoldState=SimpleNamespace(from_payload=lambda _payload: object())
        )
        with (
            patch.dict(sys.modules, {"pokezero_search": fake_module}),
            patch.object(EngineMctsPolicy, "_native", return_value=native),
            patch.object(
                EngineMctsPolicy, "_validate_model_root_observation", return_value=None
            ),
            patch.object(EngineMctsPolicy, "_root_inputs_json", return_value="{}"),
        ):
            decision = policy._search_model(
                _context(player_id),
                [_world(player_id)],
                SimpleNamespace(to_payload=lambda: {}),
                random.Random(20260915),
            )
        return decision, native, policy

    def test_every_declared_case_has_zero_simple_regret_for_both_seats_and_batches(self):
        for case in _CASES:
            for player_id in ("p1", "p2"):
                for batch in (1, 2, 8, 64):
                    with self.subTest(case=case.name, seat=player_id, batch=batch):
                        decision, native, policy = self._run(case, player_id, batch)
                        chosen = ("alpha", "beta")[decision.action_index]
                        simple_regret = max(case.payoffs.values()) - case.payoffs[chosen]

                        self.assertEqual(chosen, case.expected_move)
                        self.assertAlmostEqual(simple_regret, 0.0)
                        self.assertEqual(len(native.calls), 1)
                        self.assertEqual(native.calls[0][1], 256)
                        self.assertEqual(native.calls[0][2], batch)
                        self.assertEqual(policy.stats.worlds_searched, 1)
                        self.assertEqual(policy.stats.total_iterations, 256)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
