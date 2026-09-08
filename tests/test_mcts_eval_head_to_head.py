"""Focused contract tests for the small MCTS-vs-MCTS game runner."""

from __future__ import annotations

from dataclasses import dataclass, replace
import importlib.util
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from pokezero.mcts_eval.head_to_head import (
    HeadToHeadGame,
    HeadToHeadError,
    MctsPolicySpec,
    PublicOnlyMctsPolicy,
    complete_pair,
    load_pair,
    play_mirrored_pair,
    public_only_context,
    summarize_complete_pairs,
    write_game_immutable,
)
from pokezero.policy import PolicyContext


REPO_ROOT = Path(__file__).resolve().parents[1]


def _runner_module():
    spec = importlib.util.spec_from_file_location(
        "mcts_mcts_h2h_test", REPO_ROOT / "scripts" / "mcts_mcts_h2h.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@dataclass
class _Stats:
    decisions: int = 0
    searched_decisions: int = 0
    fallback_decisions: int = 0
    model_evals: int = 0
    total_iterations: int = 0
    worlds_constructed: int = 0
    worlds_searched: int = 0
    decision_wall_seconds: float = 0.0


class _Policy:
    def __init__(self, policy_id: str) -> None:
        self.policy_id = policy_id
        self.stats = _Stats()
        self.received_context = None

    def select_action(self, observation, *, rng):  # pragma: no cover - protocol spare path
        return SimpleNamespace(action_index=0)

    def select_action_with_context(self, context, *, rng):
        self.received_context = context
        return SimpleNamespace(action_index=0)


class _IsolatedPolicy(_Policy):
    """Test double for a policy whose decisions came from another source tree."""

    is_source_isolated = True


def _spec(config_id: str, **overrides) -> MctsPolicySpec:
    values = {
        "config_id": config_id,
        "policy_id": config_id,
        "source_commit": "a" * 40,
        "source_tree_sha256": "f" * 64,
        "engine_fingerprint": "b" * 16,
        "checkpoint_sha256": "c" * 64,
        "showdown_source_sha256": "d" * 64,
        "config": {"leaf_eval": "model", "search_sims": 32, "search_batch": 1},
    }
    values.update(overrides)
    return MctsPolicySpec(**values)


def _context() -> PolicyContext:
    p1_observation = SimpleNamespace(legal_action_mask=(True, False), private="p1")
    p2_observation = SimpleNamespace(legal_action_mask=(False, True), private="p2")
    trajectory = SimpleNamespace(
        battle_id="battle",
        format_id="gen3randombattle",
        seed=17,
        terminal=None,
        metadata={
            "private": "must not reach search",
            "public_resolved_action_rounds": [{"turn_index": 0, "actions": {}}],
        },
        steps=(
            SimpleNamespace(player_id="p1", turn_index=0, action_index=0, observation=p1_observation),
            SimpleNamespace(player_id="p2", turn_index=0, action_index=1, observation=p2_observation),
        ),
    )
    return PolicyContext(
        player_id="p1",
        decision_round_index=1,
        battle_id="battle",
        format_id="gen3randombattle",
        seed=17,
        observation=p1_observation,
        requested_players=("p1", "p2"),
        trajectory=trajectory,
        requested_legal_action_masks={"p1": (True, False), "p2": (False, True)},
        requested_observations={"p1": p1_observation, "p2": p2_observation},
        public_materialization_state=object(),
    )


class PublicContextTest(unittest.TestCase):
    def test_opponent_request_and_historic_observation_are_removed(self) -> None:
        sanitized = public_only_context(_context())

        self.assertEqual(set(sanitized.requested_observations), {"p1"})
        self.assertEqual(set(sanitized.requested_legal_action_masks), {"p1"})
        self.assertEqual(sanitized.trajectory.steps[0].observation.private, "p1")
        self.assertIsNone(sanitized.trajectory.steps[1].observation)
        self.assertNotIn("private", sanitized.trajectory.metadata)
        self.assertIn("public_resolved_action_rounds", sanitized.trajectory.metadata)

    def test_wrapper_forwards_only_the_sanitized_context(self) -> None:
        underlying = _Policy("candidate")
        PublicOnlyMctsPolicy(underlying).select_action_with_context(_context(), rng=None)

        self.assertIsNotNone(underlying.received_context)
        self.assertEqual(set(underlying.received_context.requested_observations), {"p1"})
        self.assertIsNone(underlying.received_context.trajectory.steps[1].observation)


class _Driver:
    def __init__(self, candidate_seat: str, candidate: PublicOnlyMctsPolicy, incumbent: PublicOnlyMctsPolicy, *, fallback: bool = False) -> None:
        self.candidate_seat = candidate_seat
        self.candidate = candidate
        self.incumbent = incumbent
        self.fallback = fallback

    def run(self, *, seed: int, battle_id: str):
        for policy in (self.candidate, self.incumbent):
            policy.stats.decisions += 2
            policy.stats.searched_decisions += 2
            policy.stats.model_evals += 8
            policy.stats.total_iterations += 16
            policy.stats.worlds_constructed += 4
            policy.stats.worlds_searched += 4
            policy.stats.decision_wall_seconds += 0.02
        if self.fallback:
            self.incumbent.stats.fallback_decisions += 1
        other = "p2" if self.candidate_seat == "p1" else "p1"
        winner = self.candidate_seat if self.candidate_seat == "p1" else other
        trajectory = SimpleNamespace(
            steps=(
                SimpleNamespace(
                    player_id=self.candidate_seat,
                    action_index=3,
                    metadata={"policy_elapsed_seconds": 0.01},
                ),
                SimpleNamespace(player_id=other, action_index=4, metadata={"policy_elapsed_seconds": 0.02}),
            )
        )
        return SimpleNamespace(
            terminal=SimpleNamespace(winner=winner, capped=False, turn_count=9),
            trajectory=trajectory,
        )


def _pair(*, fallback: bool = False):
    candidate = _spec("candidate")
    incumbent = _spec("incumbent", policy_id="incumbent")
    return play_mirrored_pair(
        seed=101,
        candidate=candidate,
        incumbent=incumbent,
        candidate_factory=lambda: _Policy("candidate"),
        incumbent_factory=lambda: _Policy("incumbent"),
        driver_factory=lambda _seed, seat, c, i: _Driver(seat, c, i, fallback=fallback),
    )


class MirroredPairTest(unittest.TestCase):
    def test_both_candidate_seats_are_required_and_score_as_one_pair(self) -> None:
        candidate = _spec("candidate")
        incumbent = _spec("incumbent", policy_id="incumbent")
        games = _pair()

        self.assertEqual([game.candidate_seat for game in games], ["p1", "p2"])
        self.assertEqual([game.result.outcome for game in games], ["win", "loss"])
        summary = summarize_complete_pairs(
            games,
            seeds=[101],
            candidate=candidate,
            incumbent=incumbent,
            bootstrap_resamples=20,
            bootstrap_seed=7,
        )
        self.assertEqual(summary["pair_scores"], [0.5])
        self.assertEqual(summary["candidate_score"]["point"], 0.5)
        self.assertEqual(summary["candidate_model_evals"], 16)

    def test_any_mcts_fallback_invalidates_the_game(self) -> None:
        with self.assertRaisesRegex(HeadToHeadError, "fallback"):
            _pair(fallback=True)

    def test_two_source_identities_are_refused_in_process(self) -> None:
        candidate = _spec("candidate")
        incumbent = _spec("incumbent", source_commit="z" * 40)
        with self.assertRaisesRegex(HeadToHeadError, "isolated-build"):
            play_mirrored_pair(
                seed=101,
                candidate=candidate,
                incumbent=incumbent,
                candidate_factory=lambda: _Policy("candidate"),
                incumbent_factory=lambda: _Policy("incumbent"),
                driver_factory=lambda _seed, seat, c, i: _Driver(seat, c, i),
            )

    def test_two_showdown_source_identities_are_refused_in_process(self) -> None:
        candidate = _spec("candidate")
        incumbent = _spec("incumbent", showdown_source_sha256="e" * 64)
        with self.assertRaisesRegex(HeadToHeadError, "Showdown source"):
            play_mirrored_pair(
                seed=101,
                candidate=candidate,
                incumbent=incumbent,
                candidate_factory=lambda: _Policy("candidate"),
                incumbent_factory=lambda: _Policy("incumbent"),
                driver_factory=lambda _seed, seat, c, i: _Driver(seat, c, i),
            )

    def test_source_different_builds_require_and_accept_a_marked_isolated_session(self) -> None:
        candidate = _spec("candidate")
        incumbent = _spec(
            "incumbent",
            policy_id="incumbent",
            source_commit="z" * 40,
            source_tree_sha256="e" * 64,
            engine_fingerprint="f" * 16,
        )

        def session_factory(_seed, seat):
            candidate_policy = PublicOnlyMctsPolicy(_Policy("candidate"))
            incumbent_policy = PublicOnlyMctsPolicy(_IsolatedPolicy("incumbent"))
            return _Driver(seat, candidate_policy, incumbent_policy), candidate_policy, incumbent_policy

        games = play_mirrored_pair(
            seed=101,
            candidate=candidate,
            incumbent=incumbent,
            candidate_factory=lambda: self.fail("session factory must be used"),
            incumbent_factory=lambda: self.fail("session factory must be used"),
            driver_factory=lambda *_args: self.fail("session factory must be used"),
            session_factory=session_factory,
            execution_mode="isolated_build",
        )

        self.assertEqual([game.candidate_seat for game in games], ["p1", "p2"])

    def test_isolated_build_refuses_a_session_without_a_source_isolated_policy(self) -> None:
        candidate = _spec("candidate")
        incumbent = _spec("incumbent", source_commit="z" * 40)

        def session_factory(_seed, seat):
            candidate_policy = PublicOnlyMctsPolicy(_Policy("candidate"))
            incumbent_policy = PublicOnlyMctsPolicy(_Policy("incumbent"))
            return _Driver(seat, candidate_policy, incumbent_policy), candidate_policy, incumbent_policy

        with self.assertRaisesRegex(HeadToHeadError, "no source-isolated policy"):
            play_mirrored_pair(
                seed=101,
                candidate=candidate,
                incumbent=incumbent,
                candidate_factory=lambda: self.fail("session factory must be used"),
                incumbent_factory=lambda: self.fail("session factory must be used"),
                driver_factory=lambda *_args: self.fail("session factory must be used"),
                session_factory=session_factory,
                execution_mode="isolated_build",
            )

    def test_isolated_build_still_requires_one_checkpoint_and_simulator(self) -> None:
        candidate = _spec("candidate")
        source_different = {"source_commit": "z" * 40}
        for override, expected in (
            ({**source_different, "checkpoint_sha256": "x" * 64}, "frozen checkpoint"),
            ({**source_different, "showdown_source_sha256": "x" * 64}, "frozen simulator"),
        ):
            incumbent = _spec("incumbent", **override)
            with self.assertRaisesRegex(HeadToHeadError, expected):
                play_mirrored_pair(
                    seed=101,
                    candidate=candidate,
                    incumbent=incumbent,
                    candidate_factory=lambda: self.fail("not reached"),
                    incumbent_factory=lambda: self.fail("not reached"),
                    driver_factory=lambda *_args: self.fail("not reached"),
                    session_factory=lambda *_args: self.fail("not reached"),
                    execution_mode="isolated_build",
                )


class DurableGameTest(unittest.TestCase):
    def test_partial_pair_is_durable_but_never_scoreable(self) -> None:
        candidate = _spec("candidate")
        incumbent = _spec("incumbent", policy_id="incumbent")
        games = _pair()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = write_game_immutable(root, games[0])
            self.assertTrue(path.is_file())
            loaded = load_pair(root, seed=101, candidate=candidate, incumbent=incumbent)
            self.assertEqual(set(loaded), {"p1"})
            with self.assertRaisesRegex(HeadToHeadError, "incomplete mirrored pair"):
                complete_pair(
                    list(loaded.values()), seed=101, candidate=candidate, incumbent=incumbent
                )
            write_game_immutable(root, games[1])
            self.assertEqual(
                [game.candidate_seat for game in complete_pair(
                    list(load_pair(root, seed=101, candidate=candidate, incumbent=incumbent).values()),
                    seed=101,
                    candidate=candidate,
                    incumbent=incumbent,
                )],
                ["p1", "p2"],
            )

    def test_existing_game_is_never_replaced(self) -> None:
        game = _pair()[0]
        with tempfile.TemporaryDirectory() as directory:
            write_game_immutable(directory, game)
            changed = replace(
                game,
                result=replace(game.result, outcome="loss"),
                terminal_winner="p2",
            )
            with self.assertRaisesRegex(HeadToHeadError, "refusing to replace"):
                write_game_immutable(directory, changed)

    def test_resumed_record_revalidates_terminal_outcome_and_fallbacks(self) -> None:
        # A record is only eligible for reuse if its candidate-scored result can
        # be recomputed from its terminal facts and it proves no fallback work.
        game = _pair()[1]
        bad_outcome = game.to_payload()
        bad_outcome["result"]["outcome"] = "win"
        with self.assertRaisesRegex(HeadToHeadError, "outcome"):
            HeadToHeadGame.from_payload(bad_outcome)

        bad_fallback = game.to_payload()
        bad_fallback["candidate_telemetry"]["fallback_decisions"] = 1
        with self.assertRaisesRegex(HeadToHeadError, "fallback"):
            HeadToHeadGame.from_payload(bad_fallback)


class SourceReceiptTest(unittest.TestCase):
    def test_durable_output_cannot_dirty_the_executing_source_tree(self) -> None:
        module = _runner_module()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with patch.object(module, "REPO_ROOT", root):
                with self.assertRaisesRegex(HeadToHeadError, "outside the PokeZero source"):
                    module._durable_output_root(root / "evidence")
                self.assertEqual(
                    module._durable_output_root(root.parent / "durable-evidence"),
                    (root.parent / "durable-evidence").resolve(),
                )

    def test_image_receipt_covers_the_executed_battle_bridge(self) -> None:
        module = _runner_module()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "src" / "pokezero").mkdir(parents=True)
            scripts = root / "scripts"
            scripts.mkdir()
            (root / "src" / "pokezero" / "runtime.py").write_text(
                "VALUE = 1\n", encoding="utf-8"
            )
            bridge = scripts / "battle_bridge.mjs"
            bridge.write_text("export const bridge = 1;\n", encoding="utf-8")
            (scripts / "battle_bridge_boundary_requests.mjs").write_text(
                "export const requests = 1;\n", encoding="utf-8"
            )
            with (
                patch.object(module, "REPO_ROOT", root),
                patch.dict(os.environ, {"POKEZERO_COMMIT": "a" * 40}, clear=False),
            ):
                first = module._source_provenance()
                bridge.write_text("export const bridge = 2;\n", encoding="utf-8")
                changed = module._source_provenance()

        self.assertEqual(first["tree_status"], "explicit_hash_without_git_python_and_bridge")
        self.assertNotEqual(first["tree_sha256"], changed["tree_sha256"])

    def test_isolated_worker_receipt_must_bind_the_exact_clean_incumbent(self) -> None:
        module = _runner_module()
        incumbent = _spec("incumbent", policy_id="incumbent")
        receipt = {
            "policy": incumbent.to_payload(),
            "commit": incumbent.source_commit,
            "tree_sha256": incumbent.source_tree_sha256,
            "tree_status": "clean_tracked_checkout",
            "engine_fingerprint": incumbent.engine_fingerprint,
            "worker_bootstrap_sha256": "a" * 64,
        }
        self.assertEqual(
            module._validate_isolated_receipt(
                receipt, incumbent=incumbent, bootstrap_sha256="a" * 64
            )["commit"],
            incumbent.source_commit,
        )
        receipt["tree_status"] = "dirty"
        with self.assertRaisesRegex(HeadToHeadError, "clean source checkout"):
            module._validate_isolated_receipt(
                receipt, incumbent=incumbent, bootstrap_sha256="a" * 64
            )


if __name__ == "__main__":
    unittest.main()
