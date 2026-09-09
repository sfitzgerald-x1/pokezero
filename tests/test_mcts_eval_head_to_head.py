"""Focused contract tests for the small MCTS-vs-MCTS game runner."""

from __future__ import annotations

from dataclasses import dataclass, replace
import fcntl
import importlib.util
import os
from pathlib import Path
import subprocess
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
from pokezero.mcts_eval.scoring import bootstrap_mean
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


class IsolatedRunnerCliTest(unittest.TestCase):
    def test_default_response_deadline_covers_observed_search_tail_budget(self) -> None:
        module = _runner_module()
        args = module.build_parser().parse_args(
            [
                "--checkpoint", "/checkpoint.pt",
                "--showdown-root", "/showdown",
                "--manifest", "/manifest.json",
                "--out-dir", "/out",
            ]
        )

        self.assertEqual(args.isolated_worker_timeout_seconds, 180.0)


@dataclass(frozen=True)
class _PilotConfig:
    search_sims: int = 256
    search_batch: int = 16


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
        self.assertEqual(summary["candidate_decision_walls_s"], [0.01, 0.01])
        self.assertEqual(summary["incumbent_decision_walls_s"], [0.02, 0.02])
        self.assertEqual(
            summary["candidate_decision_wall_summary"],
            {
                "count": 2,
                "total_s": 0.02,
                "min_s": 0.01,
                "p50_s": 0.01,
                "p95_s": 0.01,
                "max_s": 0.01,
            },
        )
        self.assertEqual(
            summary["incumbent_decision_wall_summary"],
            {
                "count": 2,
                "total_s": 0.04,
                "min_s": 0.02,
                "p50_s": 0.02,
                "p95_s": 0.02,
                "max_s": 0.02,
            },
        )

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
            candidate_policy = PublicOnlyMctsPolicy(_IsolatedPolicy("candidate"))
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

    def test_isolated_build_refuses_a_session_when_only_one_policy_is_isolated(self) -> None:
        candidate = _spec("candidate")
        incumbent = _spec("incumbent", source_commit="z" * 40)

        def session_factory(_seed, seat):
            candidate_policy = PublicOnlyMctsPolicy(_IsolatedPolicy("candidate"))
            incumbent_policy = PublicOnlyMctsPolicy(_Policy("incumbent"))
            return _Driver(seat, candidate_policy, incumbent_policy), candidate_policy, incumbent_policy

        with self.assertRaisesRegex(HeadToHeadError, "must source-isolate both"):
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


class BackupRepairPilotContractTest(unittest.TestCase):
    def _contract_inputs(self):
        module = _runner_module()
        pilot_seeds = tuple(range(2026091201, 2026091213))
        confirmation_seeds = list(range(2026091301, 2026091351))
        manifest = {
            "study": {
                "schema_version": module.BACKUP_REPAIR_PILOT_SCHEMA_VERSION,
                "stage": "pilot",
                "minimum_effect_delta": 0.05,
                "reserved_confirmation_seeds": confirmation_seeds,
                "replaces_nonbankable_study": module.BACKUP_REPAIR_SUPERSEDED_STUDY,
                "failure_retry_policy": module.BACKUP_REPAIR_FAILURE_RETRY_POLICY,
                "candidate_repair_baseline": module.BACKUP_REPAIR_CANDIDATE_BASELINE_COMMIT,
                "incumbent_repair_baseline": module.BACKUP_REPAIR_INCUMBENT_BASELINE_COMMIT,
            }
        }
        bootstrap = {"resamples": 10_000, "seed": 20260908, "confidence_level": 0.80}
        candidate = {
            "source_commit": "a" * 40,
            "config_id": "corrected-backups",
        }
        incumbent = {
            "source_commit": "b" * 40,
            "config_id": "pre-repair-backups",
        }
        return module, manifest, pilot_seeds, bootstrap, candidate, incumbent

    def test_contract_freezes_the_exact_contrast_and_reserved_roster(self) -> None:
        module, manifest, seeds, bootstrap, candidate, incumbent = self._contract_inputs()

        contract = module._backup_repair_pilot_contract(
            manifest,
            seeds=seeds,
            bootstrap=bootstrap,
            candidate_raw=candidate,
            incumbent_raw=incumbent,
            candidate_config=_PilotConfig(),
            incumbent_config=_PilotConfig(),
        )

        self.assertEqual(contract["pilot_seeds"], list(seeds))
        self.assertEqual(contract["reserved_confirmation_seeds"], manifest["study"]["reserved_confirmation_seeds"])
        self.assertEqual(contract["confidence_level"], 0.80)
        self.assertEqual(
            contract["replaces_nonbankable_study"], module.BACKUP_REPAIR_SUPERSEDED_STUDY
        )
        self.assertEqual(
            contract["failure_retry_policy"], module.BACKUP_REPAIR_FAILURE_RETRY_POLICY
        )
        self.assertEqual(contract["candidate_runtime_commit"], candidate["source_commit"])
        self.assertEqual(
            contract["candidate_repair_baseline"],
            module.BACKUP_REPAIR_CANDIDATE_BASELINE_COMMIT,
        )

    def test_contract_refuses_configuration_or_roster_drift(self) -> None:
        module, manifest, seeds, bootstrap, candidate, incumbent = self._contract_inputs()
        with self.assertRaisesRegex(HeadToHeadError, "identical EngineMctsConfig"):
            module._backup_repair_pilot_contract(
                manifest,
                seeds=seeds,
                bootstrap=bootstrap,
                candidate_raw=candidate,
                incumbent_raw=incumbent,
                candidate_config=_PilotConfig(),
                incumbent_config=_PilotConfig(search_sims=512),
            )

        manifest["study"]["reserved_confirmation_seeds"][0] = seeds[0]
        with self.assertRaisesRegex(HeadToHeadError, "overlap"):
            module._backup_repair_pilot_contract(
                manifest,
                seeds=seeds,
                bootstrap=bootstrap,
                candidate_raw=candidate,
                incumbent_raw=incumbent,
                candidate_config=_PilotConfig(),
                incumbent_config=_PilotConfig(),
            )

    def test_contract_refuses_superseded_rosters_or_failure_policy_drift(self) -> None:
        module, manifest, seeds, bootstrap, candidate, incumbent = self._contract_inputs()
        manifest["study"]["schema_version"] = "pokezero.mcts-h2h-backup-repair-pilot.v1"
        with self.assertRaisesRegex(HeadToHeadError, "registered backup-repair"):
            module._backup_repair_pilot_contract(
                manifest,
                seeds=seeds,
                bootstrap=bootstrap,
                candidate_raw=candidate,
                incumbent_raw=incumbent,
                candidate_config=_PilotConfig(),
                incumbent_config=_PilotConfig(),
            )

        manifest["study"]["schema_version"] = module.BACKUP_REPAIR_PILOT_SCHEMA_VERSION
        manifest["study"]["failure_retry_policy"] = {
            **module.BACKUP_REPAIR_FAILURE_RETRY_POLICY,
            "nonzero_runner_exit": "retry",
        }
        with self.assertRaisesRegex(HeadToHeadError, "failure/retry"):
            module._backup_repair_pilot_contract(
                manifest,
                seeds=seeds,
                bootstrap=bootstrap,
                candidate_raw=candidate,
                incumbent_raw=incumbent,
                candidate_config=_PilotConfig(),
                incumbent_config=_PilotConfig(),
            )

        manifest["study"]["failure_retry_policy"] = module.BACKUP_REPAIR_FAILURE_RETRY_POLICY
        reused = (*seeds[:-1], module.BACKUP_REPAIR_SUPERSEDED_PILOT_SEEDS[0])
        with self.assertRaisesRegex(HeadToHeadError, "non-bankable study"):
            module._backup_repair_pilot_contract(
                manifest,
                seeds=reused,
                bootstrap=bootstrap,
                candidate_raw=candidate,
                incumbent_raw=incumbent,
                candidate_config=_PilotConfig(),
                incumbent_config=_PilotConfig(),
            )

        manifest["study"]["candidate_repair_baseline"] = "0" * 40
        with self.assertRaisesRegex(HeadToHeadError, "corrected-backup baseline"):
            module._backup_repair_pilot_contract(
                manifest,
                seeds=seeds,
                bootstrap=bootstrap,
                candidate_raw=candidate,
                incumbent_raw=incumbent,
                candidate_config=_PilotConfig(),
                incumbent_config=_PilotConfig(),
            )

    def test_runtime_source_must_be_a_clean_descendant_of_the_mechanics_baseline(self) -> None:
        module = _runner_module()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)

            def git(*arguments: str) -> subprocess.CompletedProcess[str]:
                return subprocess.run(
                    ["git", "-C", str(root), *arguments],
                    check=True,
                    capture_output=True,
                    text=True,
                )

            git("init")
            git("config", "user.email", "tests@example.invalid")
            git("config", "user.name", "MCTS contract test")
            tracked = root / "tracked.txt"
            tracked.write_text("baseline\n", encoding="utf-8")
            git("add", "tracked.txt")
            git("commit", "-m", "baseline")
            baseline = git("rev-parse", "HEAD").stdout.strip()
            tracked.write_text("runtime compatibility\n", encoding="utf-8")
            git("commit", "-am", "runtime compatibility")
            head = git("rev-parse", "HEAD").stdout.strip()

            proof = module._require_clean_git_ancestor(
                root,
                expected_head=head,
                required_ancestor=baseline,
                role="candidate",
            )
            self.assertEqual(proof["head"], head)
            self.assertEqual(proof["required_ancestor"], baseline)
            self.assertTrue(proof["ancestor_verified"])

            with self.assertRaisesRegex(HeadToHeadError, "declared runtime commit"):
                module._require_clean_git_ancestor(
                    root,
                    expected_head=baseline,
                    required_ancestor=baseline,
                    role="candidate",
                )

            with self.assertRaisesRegex(HeadToHeadError, "does not descend"):
                module._require_clean_git_ancestor(
                    root,
                    expected_head=head,
                    required_ancestor="0" * 40,
                    role="candidate",
                )

            tracked.write_text("dirty\n", encoding="utf-8")
            with self.assertRaisesRegex(HeadToHeadError, "not a clean"):
                module._require_clean_git_ancestor(
                    root,
                    expected_head=head,
                    required_ancestor=baseline,
                    role="candidate",
                )

    def test_readout_uses_the_predeclared_delta_and_never_hides_prior_fallbacks(self) -> None:
        module, manifest, seeds, bootstrap, candidate, incumbent = self._contract_inputs()
        contract = module._backup_repair_pilot_contract(
            manifest,
            seeds=seeds,
            bootstrap=bootstrap,
            candidate_raw=candidate,
            incumbent_raw=incumbent,
            candidate_config=_PilotConfig(),
            incumbent_config=_PilotConfig(),
        )
        games = [
            SimpleNamespace(
                candidate_telemetry=SimpleNamespace(fallback_decisions=0, prior_fallbacks=0),
                incumbent_telemetry=SimpleNamespace(fallback_decisions=0, prior_fallbacks=0),
            )
            for _ in range(24)
        ]
        readout = module._backup_repair_pilot_readout(
            contract=contract,
            summary={"pair_scores": [1.0] * len(seeds)},
            games=games,
        )
        self.assertEqual(readout["candidate_score_delta_from_neutral"], {
            "point": 0.5,
            "low": 0.5,
            "high": 0.5,
        })
        self.assertEqual(readout["decision"], "ELIGIBLE_FOR_RESERVED_CONFIRMATION")

        games[0].candidate_telemetry.prior_fallbacks = 1
        fallback_readout = module._backup_repair_pilot_readout(
            contract=contract,
            summary={"pair_scores": [1.0] * len(seeds)},
            games=games,
        )
        self.assertEqual(fallback_readout["decision"], "INCONCLUSIVE_OR_NOT_PROMOTED")
        self.assertFalse(fallback_readout["promotion_checks"]["no_fallbacks_or_refusals"])

    def test_replacement_cli_refuses_direct_or_terminalled_roots_before_work(self) -> None:
        module = _runner_module()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            out_dir = root / "out"
            out_dir.mkdir()
            manifest = root / "manifest.json"
            manifest.write_text(
                "{"
                f"\"schema_version\":\"{module.MANIFEST_SCHEMA_VERSION}\","
                "\"study\":{\"schema_version\":\"pokezero.mcts-h2h-backup-repair-pilot.v3\"}"
                "}\n",
                encoding="utf-8",
            )
            argv = [
                "--checkpoint",
                str(root / "unread-checkpoint.pt"),
                "--showdown-root",
                str(root / "unread-showdown"),
                "--manifest",
                str(manifest),
                "--out-dir",
                str(out_dir),
            ]
            with patch.object(module, "_source_provenance", side_effect=AssertionError):
                with self.assertRaisesRegex(HeadToHeadError, "durable-launcher attempt id"):
                    module.main(argv)

            (out_dir / module.RUNNER_TERMINAL_NAME).write_text("not read\n", encoding="utf-8")
            with patch.object(module, "_source_provenance", side_effect=AssertionError):
                with self.assertRaisesRegex(HeadToHeadError, "terminal handoff"):
                    module.main(argv)

    def test_replacement_requires_a_matching_immutable_launcher_receipt(self) -> None:
        module = _runner_module()
        with tempfile.TemporaryDirectory() as directory:
            out_dir = Path(directory) / "out"
            attempt_id = "attempt-a"
            receipt = out_dir / "launcher-attempts" / f"{attempt_id}.json"
            receipt.parent.mkdir(parents=True)
            runner_script = Path(module.__file__).resolve()
            module._write_immutable_json(
                receipt,
                {
                    "schema_version": module.DURABLE_LAUNCHER_ATTEMPT_SCHEMA_VERSION,
                    "attempt_id": attempt_id,
                    "runner_script": str(runner_script),
                    "runner_script_sha256": module._sha256_file(runner_script),
                    "writer_lock": str(out_dir / "runner-writer.lock"),
                },
            )
            environment = {
                module.DURABLE_LAUNCHER_ATTEMPT_ID_ENV: attempt_id,
                module.DURABLE_LAUNCHER_OUT_DIR_ENV: str(out_dir),
                module.DURABLE_LAUNCHER_ATTEMPT_RECEIPT_ENV: str(receipt),
            }
            with patch.dict(os.environ, environment, clear=False):
                with self.assertRaisesRegex(HeadToHeadError, "writer-lock fd"):
                    module._require_durable_launcher_handoff(out_dir)

            lock_path = out_dir / "runner-writer.lock"
            lock_fd = os.open(lock_path, os.O_RDWR | os.O_CREAT, 0o600)
            try:
                fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                environment[module.DURABLE_LAUNCHER_WRITER_LOCK_FD_ENV] = str(lock_fd)
                with patch.dict(os.environ, environment, clear=False):
                    module._require_durable_launcher_handoff(out_dir)
            finally:
                fcntl.flock(lock_fd, fcntl.LOCK_UN)
                os.close(lock_fd)

            with patch.dict(
                os.environ,
                {**environment, module.DURABLE_LAUNCHER_ATTEMPT_ID_ENV: "wrong"},
                clear=False,
            ):
                with self.assertRaisesRegex(HeadToHeadError, "receipt does not match"):
                    module._require_durable_launcher_handoff(out_dir)


class BootstrapConfidenceTest(unittest.TestCase):
    def test_confidence_level_controls_percentile_width_and_rejects_invalid_values(self) -> None:
        values = [0.0, 0.5, 1.0]
        draws = [[0, 0, 0], [0, 1, 2], [2, 2, 2]]
        narrow = bootstrap_mean(values, draws, confidence_level=0.50)
        wide = bootstrap_mean(values, draws, confidence_level=0.95)
        self.assertGreaterEqual(narrow.low, wide.low)
        self.assertLessEqual(narrow.high, wide.high)
        with self.assertRaisesRegex(ValueError, "strictly between"):
            bootstrap_mean(values, draws, confidence_level=1.0)


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
        incumbent = _spec(
            "incumbent",
            policy_id="incumbent",
            config={
                "leaf_eval": "model",
                "search_sims": 32,
                "search_batch": 1,
                "model_decision_time_ms": None,
            },
        )
        receipt = {
            "policy": incumbent.to_payload(),
            "commit": incumbent.source_commit,
            "tree_sha256": incumbent.source_tree_sha256,
            "tree_status": "clean_tracked_checkout",
            "engine_fingerprint": incumbent.engine_fingerprint,
            "worker_bootstrap_sha256": "a" * 64,
            "reset_protocol": "policy_method_or_fresh_source_policy.v1",
            "config_compatibility": {
                "protocol": "disabled-diagnostic-omission.v1",
                "omitted_disabled_fields": ["model_decision_time_ms"],
            },
        }
        self.assertEqual(
            module._validate_isolated_receipt(
                receipt, policy=incumbent, role="incumbent", bootstrap_sha256="a" * 64
            )["commit"],
            incumbent.source_commit,
        )
        receipt["tree_status"] = "dirty"
        with self.assertRaisesRegex(HeadToHeadError, "clean source checkout"):
            module._validate_isolated_receipt(
                receipt, policy=incumbent, role="incumbent", bootstrap_sha256="a" * 64
            )
        receipt["tree_status"] = "clean_tracked_checkout"
        receipt.pop("reset_protocol")
        with self.assertRaisesRegex(HeadToHeadError, "source-safe reset protocol"):
            module._validate_isolated_receipt(
                receipt, policy=incumbent, role="incumbent", bootstrap_sha256="a" * 64
            )

    def test_isolated_worker_receipt_refuses_behavior_bearing_compatibility_omission(self) -> None:
        module = _runner_module()
        incumbent = _spec(
            "incumbent",
            policy_id="incumbent",
            config={
                "leaf_eval": "model",
                "search_sims": 32,
                "search_batch": 1,
                "root_selector_shadow": True,
            },
        )
        receipt = {
            "policy": incumbent.to_payload(),
            "commit": incumbent.source_commit,
            "tree_sha256": incumbent.source_tree_sha256,
            "tree_status": "clean_tracked_checkout",
            "engine_fingerprint": incumbent.engine_fingerprint,
            "worker_bootstrap_sha256": "a" * 64,
            "reset_protocol": "policy_method_or_fresh_source_policy.v1",
            "config_compatibility": {
                "protocol": "disabled-diagnostic-omission.v1",
                "omitted_disabled_fields": ["root_selector_shadow"],
            },
        }

        with self.assertRaisesRegex(HeadToHeadError, "not explicitly disabled"):
            module._validate_isolated_receipt(
                receipt, policy=incumbent, role="incumbent", bootstrap_sha256="a" * 64
            )

        missing_field_policy = _spec("incumbent", policy_id="incumbent")
        receipt["policy"] = missing_field_policy.to_payload()
        receipt["config_compatibility"] = {
            "protocol": "disabled-diagnostic-omission.v1",
            "omitted_disabled_fields": ["root_selector_shadow"],
        }
        with self.assertRaisesRegex(HeadToHeadError, "not explicitly disabled"):
            module._validate_isolated_receipt(
                receipt,
                policy=missing_field_policy,
                role="incumbent",
                bootstrap_sha256="a" * 64,
            )

    def test_isolated_worker_stderr_keeps_prior_attempts_and_gives_a_retry_a_fresh_path(self) -> None:
        module = _runner_module()
        with tempfile.TemporaryDirectory() as directory:
            out_root = Path(directory)
            first = module._isolated_worker_stderr_path(
                out_root,
                attempt_id="a" * 32,
                seed=19,
                candidate_seat="p1",
                role="candidate",
            )
            retry = module._isolated_worker_stderr_path(
                out_root,
                attempt_id="b" * 32,
                seed=19,
                candidate_seat="p1",
                role="candidate",
            )
            first.parent.mkdir(parents=True)
            first.write_text("interrupted child stderr\n", encoding="utf-8")

            self.assertNotEqual(first, retry)
            self.assertTrue(first.is_file())
            self.assertEqual(
                retry.relative_to(out_root).as_posix(),
                "worker-stderr/attempt-" + "b" * 32 + "/seed-19-p1-candidate.log",
            )


if __name__ == "__main__":
    unittest.main()
