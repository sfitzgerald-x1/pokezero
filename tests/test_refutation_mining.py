import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from pokezero.actions import ACTION_COUNT
from pokezero.collection import RolloutRecord, read_rollout_records, write_rollout_record
from pokezero.dataset import iter_training_cache_batches
from pokezero.env import StepResult, TerminalState
from pokezero.observation import PokeZeroObservationV0
from pokezero.policy import PolicyDecision
from pokezero.refutation_cli import main as refutation_cli_main
from pokezero.refutation_curriculum import (
    REFUTATION_CURRICULUM_METADATA_SCHEMA_VERSION,
    RefutationCurriculumConfig,
    collect_refutation_curriculum_rollouts,
    refutation_curriculum_start_count,
)
from pokezero.refutation_mining import (
    BranchTerminalResult,
    FRAGILE_STATE_SCHEMA_VERSION,
    RefutationCandidate,
    RefutationMiningConfig,
    ReplayTerminalBranchEvaluator,
    candidate_count_for_records,
    iter_fragile_states,
    mine_refutations,
    reproduce_refutation_archive,
    validate_refutation_report_payload,
    write_refutation_report,
)
from pokezero.refutation_population import (
    REFUTATION_BEHAVIOR_SEED_MANIFEST_SCHEMA_VERSION,
    RefutationBehaviorSeedConfig,
    build_refutation_behavior_seed_manifest,
)
from pokezero.refutation_progress import (
    REFUTATION_CYCLE_REPORT_SCHEMA_VERSION,
    RefutationCycleReportInput,
    build_refutation_cycle_report,
)
from pokezero.refutation_training import (
    RefutationTrainingConfig,
    refutation_training_examples,
    write_refutation_training_cache,
)
from pokezero.rollout import RolloutConfig
from pokezero.trajectory import BattleTrajectory, TrajectoryStep


def _observation(legal_actions: tuple[int, ...]) -> PokeZeroObservationV0:
    legal_action_mask = tuple(index in legal_actions for index in range(ACTION_COUNT))
    return PokeZeroObservationV0(
        categorical_ids=((0,),),
        numeric_features=((0.0,),),
        token_type_ids=(0,),
        attention_mask=(True,),
        legal_action_mask=legal_action_mask,
    )


def _step(player_id: str, turn_index: int, action_index: int, legal_actions: tuple[int, ...]) -> TrajectoryStep:
    observation = _observation(legal_actions)
    return TrajectoryStep(
        player_id=player_id,
        turn_index=turn_index,
        observation=observation,
        legal_action_mask=observation.legal_action_mask,
        action_index=action_index,
    )


def _record(
    *,
    battle_id: str = "battle-1",
    winner: str = "p1",
    policy_ids: dict[str, str] | None = None,
) -> RolloutRecord:
    trajectory = BattleTrajectory(battle_id=battle_id, format_id="gen3randombattle", seed=100)
    trajectory.append(_step("p1", 0, 0, (0,)))
    trajectory.append(_step("p2", 0, 1, (1, 2, 3)))
    trajectory.append(_step("p1", 1, 0, (0,)))
    trajectory.append(_step("p2", 1, 4, (4, 5)))
    terminal = TerminalState(winner=winner, turn_count=2, capped=False)
    trajectory.record_terminal(terminal)
    return RolloutRecord(
        battle_id=battle_id,
        seed=100,
        format_id="gen3randombattle",
        policy_ids=policy_ids or {"p1": "champion", "p2": "challenger"},
        decision_round_count=2,
        elapsed_seconds=1.0,
        terminal=terminal,
        trajectory=trajectory,
    )


def _refutation_report_payload(
    *,
    mode: str,
    sampled_win_count: int,
    refuted_game_count: int,
    certified_refutation_count: int | None = None,
) -> dict:
    certified = certified_refutation_count if certified_refutation_count is not None else refuted_game_count
    return {
        "schema_version": "pokezero.refutation_report.v1",
        "config": {
            "champion_policy_id": "champion",
            "champion_player_id": None,
            "max_wins": sampled_win_count,
            "max_decision_points_per_game": 1,
            "max_deviations_per_state": 1,
            "max_line_depth": 1,
            "certification_seed_count": 20,
            "min_flip_rate": 0.6,
            "mode": mode,
        },
        "source_record_count": sampled_win_count,
        "sampled_win_count": sampled_win_count,
        "scanned_decision_count": sampled_win_count,
        "candidate_deviation_count": sampled_win_count,
        "evaluated_deviation_count": sampled_win_count,
        "skipped_candidate_error_count": 0,
        "candidate_error_examples": [],
        "certified_refutation_count": certified,
        "distinct_certified_root_count": certified,
        "refuted_game_count": refuted_game_count,
        "refutation_rate": refuted_game_count / sampled_win_count,
        "certified_refutations_per_sampled_win": certified / sampled_win_count,
        "archive_path": "fragile.jsonl",
        "examples": [],
    }


class FakeTerminalEvaluator:
    evaluation_source = "terminal_rollout"
    value_head_used = False
    reseed_scope = "simulator_rng"

    def __init__(
        self,
        *,
        loser_winning_actions: set[int],
        loser_win_count: int,
        loser_winning_depths: set[int] | None = None,
    ) -> None:
        self.loser_winning_actions = loser_winning_actions
        self.loser_win_count = loser_win_count
        self.loser_winning_depths = loser_winning_depths
        self.calls = []

    def evaluate(self, *, record, candidate, certification_seed: int) -> BranchTerminalResult:
        self.calls.append(
            (
                record.battle_id,
                candidate.deviation_action_index,
                candidate.line_depth,
                certification_seed,
                dict(candidate.branch_actions),
            )
        )
        count_for_candidate = sum(
            1
            for _, action, depth, _, _ in self.calls
            if action == candidate.deviation_action_index and depth == candidate.line_depth
        )
        loser_wins = (
            candidate.deviation_action_index in self.loser_winning_actions
            and (self.loser_winning_depths is None or candidate.line_depth in self.loser_winning_depths)
            and count_for_candidate <= self.loser_win_count
        )
        return BranchTerminalResult(
            certification_seed=certification_seed,
            winner=candidate.loser_player_id if loser_wins else candidate.champion_player_id,
            capped=False,
            turn_count=3,
        )


class ValueHeadEvaluator(FakeTerminalEvaluator):
    value_head_used = True


class NonTerminalEvaluator(FakeTerminalEvaluator):
    evaluation_source = "value_head"


class ContinuationOnlyEvaluator(FakeTerminalEvaluator):
    reseed_scope = "continuation_policy_rng"


class FirstLegalPolicy:
    policy_id = "first-legal"

    def select_action(self, observation, *, rng) -> PolicyDecision:
        action_index = next(index for index, legal in enumerate(observation.legal_action_mask) if legal)
        return PolicyDecision(action_index=action_index, policy_id=self.policy_id)


class BranchReplayEnv:
    def __init__(self) -> None:
        self.reset_calls = []
        self.step_calls = []
        self.reseed_calls = []
        self.events = []
        self.round_index = 0
        self._terminal = None

    def reset(self, *, seed: int, format_id: str = "gen3randombattle") -> None:
        self.reset_calls.append((seed, format_id))
        self.step_calls.clear()
        self.round_index = 0
        self._terminal = None

    def requested_players(self):
        if self._terminal is not None:
            return ()
        return (("p1", "p2"), ("p1",))[min(self.round_index, 1)]

    def observe(self, player):
        return _observation((0,))

    def terminal(self):
        return self._terminal

    def reseed_simulator_rng(self, seed: int) -> None:
        self.reseed_calls.append(seed)
        self.events.append(("reseed", seed))

    def step(self, actions):
        self.step_calls.append(dict(actions))
        self.events.append(("step", dict(actions)))
        self.round_index += 1
        if len(self.step_calls) == 1:
            return StepResult(
                observations={"p1": _observation((0,))},
                rewards={"p1": 0.0, "p2": 0.0},
                terminal=None,
                requested_players=("p1",),
            )
        self._terminal = TerminalState(winner="p2", turn_count=2, capped=False)
        return StepResult(
            observations={},
            rewards={"p1": -1.0, "p2": 1.0},
            terminal=self._terminal,
            requested_players=(),
        )

    def close(self):
        self.closed = True


class PrefixReplayEnv(BranchReplayEnv):
    def requested_players(self):
        if self._terminal is not None:
            return ()
        return ("p1", "p2")

    def step(self, actions):
        self.step_calls.append(dict(actions))
        self.round_index += 1
        if len(self.step_calls) == 1:
            return StepResult(
                observations={"p1": _observation((0,)), "p2": _observation((0,))},
                rewards={"p1": 0.0, "p2": 0.0},
                terminal=None,
                requested_players=("p1", "p2"),
            )
        self._terminal = TerminalState(winner="p2", turn_count=2, capped=False)
        return StepResult(
            observations={},
            rewards={"p1": -1.0, "p2": 1.0},
            terminal=self._terminal,
            requested_players=(),
        )


class RefutationMiningTest(unittest.TestCase):
    def test_plan_counts_only_champion_wins_and_loser_deviations(self) -> None:
        config = RefutationMiningConfig(champion_policy_id="champion", max_wins=10)

        counts = candidate_count_for_records(
            records=(
                _record(battle_id="won", winner="p1"),
                _record(battle_id="lost", winner="p2"),
            ),
            config=config,
        )

        self.assertEqual(counts["source_record_count"], 2)
        self.assertEqual(counts["sampled_win_count"], 1)
        self.assertEqual(counts["scanned_decision_count"], 2)
        self.assertEqual(counts["candidate_deviation_count"], 3)

    def test_plan_counts_requested_depth_ladder_candidates(self) -> None:
        config = RefutationMiningConfig(champion_policy_id="champion", max_wins=10, max_line_depth=2)

        counts = candidate_count_for_records(records=(_record(),), config=config)

        self.assertEqual(counts["sampled_win_count"], 1)
        self.assertEqual(counts["scanned_decision_count"], 2)
        self.assertEqual(counts["candidate_deviation_count"], 5)

    def test_config_rejects_line_depth_above_r0_bound(self) -> None:
        with self.assertRaisesRegex(ValueError, "max_line_depth"):
            RefutationMiningConfig(champion_policy_id="champion", max_line_depth=4)

    def test_miner_certifies_only_terminal_rollout_flips_above_threshold(self) -> None:
        config = RefutationMiningConfig(
            champion_policy_id="champion",
            max_wins=10,
            certification_seed_count=20,
            min_flip_rate=0.60,
            max_decision_points_per_game=1,
        )
        evaluator = FakeTerminalEvaluator(loser_winning_actions={2}, loser_win_count=13)
        with tempfile.TemporaryDirectory() as temp_dir:
            archive_path = Path(temp_dir) / "fragile.jsonl"

            report = mine_refutations(
                records=(_record(),),
                config=config,
                evaluator=evaluator,
                archive_path=archive_path,
            )

            self.assertEqual(report.sampled_win_count, 1)
            self.assertEqual(report.scanned_decision_count, 1)
            self.assertEqual(report.candidate_deviation_count, 2)
            self.assertEqual(report.evaluated_deviation_count, 2)
            self.assertEqual(len(report.certified_refutations), 1)
            refutation = report.certified_refutations[0]
            self.assertEqual(refutation.evaluation_source, "terminal_rollout")
            self.assertEqual(refutation.search_stats["value_head_used"], False)
            self.assertEqual(refutation.search_stats["reseed_scope"], "simulator_rng")
            self.assertEqual(refutation.search_stats["simulator_rng_reseeded"], True)
            self.assertEqual(refutation.flip_rate, 13 / 20)
            self.assertEqual(refutation.candidate.branch_actions, {"p1": 0, "p2": 2})
            self.assertEqual(report.refutation_rate, 1.0)
            self.assertEqual(report.certified_refutations_per_sampled_win, 1.0)

            archive_rows = tuple(iter_fragile_states(archive_path))
            self.assertEqual(len(archive_rows), 1)
            self.assertEqual(archive_rows[0]["schema_version"], FRAGILE_STATE_SCHEMA_VERSION)
            self.assertEqual(archive_rows[0]["candidate"]["deviation_action_index"], 2)
            self.assertEqual(archive_rows[0]["certification"]["seed_count"], 20)

    def test_miner_can_certify_recorded_continuation_line_depth(self) -> None:
        config = RefutationMiningConfig(
            champion_policy_id="champion",
            max_wins=10,
            certification_seed_count=20,
            min_flip_rate=0.60,
            max_decision_points_per_game=1,
            max_deviations_per_state=1,
            max_line_depth=2,
        )
        evaluator = FakeTerminalEvaluator(
            loser_winning_actions={2},
            loser_winning_depths={2},
            loser_win_count=13,
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            archive_path = Path(temp_dir) / "fragile.jsonl"

            report = mine_refutations(
                records=(_record(),),
                config=config,
                evaluator=evaluator,
                archive_path=archive_path,
            )

            self.assertEqual(report.candidate_deviation_count, 2)
            self.assertEqual(report.evaluated_deviation_count, 2)
            self.assertEqual(len(report.certified_refutations), 1)
            refutation = report.certified_refutations[0]
            self.assertEqual(refutation.candidate.line_depth, 2)
            self.assertEqual(refutation.search_stats["depth"], 2)
            self.assertEqual(refutation.search_stats["max_line_depth"], 2)
            self.assertEqual(
                refutation.candidate.branch_action_sequence,
                ({"p1": 0, "p2": 2}, {"p1": 0, "p2": 4}),
            )

            archive_row = next(iter_fragile_states(archive_path))
            self.assertEqual(archive_row["candidate"]["line_depth"], 2)
            self.assertEqual(
                archive_row["candidate"]["branch_action_sequence"],
                [{"p1": 0, "p2": 2}, {"p1": 0, "p2": 4}],
            )

    def test_depth_variants_do_not_inflate_distinct_certified_roots(self) -> None:
        config = RefutationMiningConfig(
            champion_policy_id="champion",
            max_wins=10,
            certification_seed_count=20,
            min_flip_rate=0.60,
            max_decision_points_per_game=1,
            max_deviations_per_state=1,
            max_line_depth=2,
        )
        evaluator = FakeTerminalEvaluator(loser_winning_actions={2}, loser_win_count=40)
        with tempfile.TemporaryDirectory() as temp_dir:
            archive_path = Path(temp_dir) / "fragile.jsonl"
            report = mine_refutations(
                records=(_record(),),
                config=config,
                evaluator=evaluator,
                archive_path=archive_path,
            )
            validation = validate_refutation_report_payload(
                report=report.to_dict(),
                fragile_states=tuple(iter_fragile_states(archive_path)),
                min_sampled_wins=1,
                min_certified_refutations=2,
            )

        self.assertEqual(len(report.certified_refutations), 2)
        self.assertEqual(report.distinct_certified_root_count, 1)
        self.assertEqual(report.to_dict()["distinct_certified_root_count"], 1)
        failed = {check["name"] for check in validation["checks"] if not check["passed"]}
        self.assertIn("distinct_certified_root_count", failed)

    def test_miner_rejects_rng_luck_at_or_below_threshold(self) -> None:
        config = RefutationMiningConfig(
            champion_policy_id="champion",
            certification_seed_count=20,
            min_flip_rate=0.60,
            max_decision_points_per_game=1,
            max_deviations_per_state=1,
        )
        evaluator = FakeTerminalEvaluator(loser_winning_actions={2}, loser_win_count=12)
        with tempfile.TemporaryDirectory() as temp_dir:
            report = mine_refutations(
                records=(_record(),),
                config=config,
                evaluator=evaluator,
                archive_path=Path(temp_dir) / "fragile.jsonl",
            )

            self.assertEqual(len(report.certified_refutations), 0)

    def test_certification_rejects_value_head_evaluator(self) -> None:
        config = RefutationMiningConfig(
            champion_policy_id="champion",
            certification_seed_count=20,
            max_decision_points_per_game=1,
            max_deviations_per_state=1,
        )
        evaluator = ValueHeadEvaluator(loser_winning_actions={2}, loser_win_count=20)
        with tempfile.TemporaryDirectory() as temp_dir:
            with self.assertRaisesRegex(ValueError, "value-head"):
                mine_refutations(
                    records=(_record(),),
                    config=config,
                    evaluator=evaluator,
                    archive_path=Path(temp_dir) / "fragile.jsonl",
                )

    def test_certification_rejects_non_terminal_evaluator(self) -> None:
        config = RefutationMiningConfig(
            champion_policy_id="champion",
            certification_seed_count=20,
            max_decision_points_per_game=1,
            max_deviations_per_state=1,
        )
        evaluator = NonTerminalEvaluator(loser_winning_actions={2}, loser_win_count=20)
        with tempfile.TemporaryDirectory() as temp_dir:
            with self.assertRaisesRegex(ValueError, "terminal-rollout"):
                mine_refutations(
                    records=(_record(),),
                    config=config,
                    evaluator=evaluator,
                    archive_path=Path(temp_dir) / "fragile.jsonl",
                )

    def test_replay_terminal_evaluator_branches_then_continues_to_terminal(self) -> None:
        record = _record()
        config = RefutationMiningConfig(
            champion_policy_id="champion",
            certification_seed_count=20,
            max_decision_points_per_game=1,
            max_deviations_per_state=1,
        )
        env = BranchReplayEnv()
        evaluator = ReplayTerminalBranchEvaluator(
            env_factory=lambda: env,
            policies={"p1": FirstLegalPolicy(), "p2": FirstLegalPolicy()},
            rollout_config=RolloutConfig(max_decision_rounds=5),
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            candidate = next(
                ref.candidate
                for ref in mine_refutations(
                    records=(record,),
                    config=config,
                    evaluator=FakeTerminalEvaluator(loser_winning_actions={2}, loser_win_count=20),
                    archive_path=Path(temp_dir) / "fragile.jsonl",
                ).certified_refutations
            )

        result = evaluator.evaluate(record=record, candidate=candidate, certification_seed=777)

        self.assertEqual(result.winner, "p2")
        self.assertEqual(env.reset_calls, [(100, "gen3randombattle")])
        self.assertEqual(env.step_calls[0], {"p1": 0, "p2": 2})
        self.assertEqual(env.step_calls[1], {"p1": 0})
        self.assertEqual(evaluator.value_head_used, False)
        self.assertEqual(evaluator.reseed_scope, "continuation_policy_rng")

    def test_replay_terminal_evaluator_can_reseed_simulator_before_branch(self) -> None:
        record = _record()
        env = BranchReplayEnv()
        evaluator = ReplayTerminalBranchEvaluator(
            env_factory=lambda: env,
            policies={"p1": FirstLegalPolicy(), "p2": FirstLegalPolicy()},
            rollout_config=RolloutConfig(max_decision_rounds=5),
            reseed_simulator_rng=True,
        )
        candidate = RefutationCandidate(
            battle_id=record.battle_id,
            source_record_index=0,
            seed=record.seed,
            format_id=record.format_id,
            champion_player_id="p1",
            loser_player_id="p2",
            decision_round_index=0,
            step_index=1,
            recorded_action_index=1,
            deviation_action_index=2,
            branch_actions={"p1": 0, "p2": 2},
        )

        result = evaluator.evaluate(record=record, candidate=candidate, certification_seed=777)

        self.assertEqual(result.winner, "p2")
        self.assertEqual(env.reseed_calls, [777])
        self.assertEqual(env.step_calls[0], {"p1": 0, "p2": 2})
        self.assertEqual(
            env.events[:2],
            [("reseed", 777), ("step", {"p1": 0, "p2": 2})],
        )
        self.assertEqual(evaluator.reseed_scope, "simulator_rng")

    def test_replay_terminal_evaluator_can_force_bounded_branch_line_before_rollout(self) -> None:
        record = _record()
        env = BranchReplayEnv()
        evaluator = ReplayTerminalBranchEvaluator(
            env_factory=lambda: env,
            policies={"p1": FirstLegalPolicy(), "p2": FirstLegalPolicy()},
            rollout_config=RolloutConfig(max_decision_rounds=5),
        )
        candidate = RefutationCandidate(
            battle_id=record.battle_id,
            source_record_index=0,
            seed=record.seed,
            format_id=record.format_id,
            champion_player_id="p1",
            loser_player_id="p2",
            decision_round_index=0,
            step_index=1,
            recorded_action_index=1,
            deviation_action_index=2,
            branch_actions={"p1": 0, "p2": 2},
            branch_action_sequence=({"p1": 0, "p2": 2}, {"p1": 0}),
        )

        result = evaluator.evaluate(record=record, candidate=candidate, certification_seed=777)

        self.assertEqual(result.winner, "p2")
        self.assertEqual(env.step_calls, [{"p1": 0, "p2": 2}, {"p1": 0}])

    def test_miner_skips_infeasible_depth_line_without_aborting_run(self) -> None:
        record = _record()
        config = RefutationMiningConfig(
            champion_policy_id="champion",
            certification_seed_count=20,
            max_decision_points_per_game=1,
            max_deviations_per_state=1,
            max_line_depth=2,
        )
        evaluator = ReplayTerminalBranchEvaluator(
            env_factory=BranchReplayEnv,
            policies={"p1": FirstLegalPolicy(), "p2": FirstLegalPolicy()},
            rollout_config=RolloutConfig(max_decision_rounds=5),
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            report = mine_refutations(
                records=(record,),
                config=config,
                evaluator=evaluator,
                archive_path=Path(temp_dir) / "fragile.jsonl",
            )

        self.assertEqual(report.candidate_deviation_count, 2)
        self.assertEqual(report.evaluated_deviation_count, 2)
        self.assertEqual(report.skipped_candidate_error_count, 1)
        self.assertEqual(len(report.candidate_error_examples), 1)
        self.assertIn("does not match environment request", report.candidate_error_examples[0]["error"])
        self.assertEqual(len(report.certified_refutations), 1)

    def test_cli_plan_reads_rollout_jsonl(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "records.jsonl"
            with path.open("w", encoding="utf-8") as handle:
                write_rollout_record(handle, _record())

            stdout = io.StringIO()
            with contextlib.redirect_stdout(stdout):
                exit_code = refutation_cli_main(
                    [
                        "plan",
                        "--records",
                        str(path),
                        "--champion-policy-id",
                        "champion",
                    ]
                )

            self.assertEqual(exit_code, 0)
            self.assertIn('"candidate_deviation_count": 3', stdout.getvalue())

    def test_cli_mine_uses_simulator_rng_reseeded_evaluator(self) -> None:
        class FakeReport:
            def to_dict(self):
                return {
                    "schema_version": "pokezero.refutation_report.v1",
                    "archive_path": "fragile-states.jsonl",
                }

        captured = {}

        def fake_mine_refutations(**kwargs):
            captured["evaluator"] = kwargs["evaluator"]
            return FakeReport()

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            records_path = temp_path / "records.jsonl"
            with records_path.open("w", encoding="utf-8") as handle:
                write_rollout_record(handle, _record())

            stdout = io.StringIO()
            with (
                patch("pokezero.refutation_cli.env_config_with_policy_spec_masks", side_effect=lambda config, *_args, **_kwargs: config),
                patch("pokezero.refutation_cli.policy_spec_with_showdown_root", side_effect=lambda spec, _root: spec),
                patch("pokezero.refutation_cli.policy_from_spec", return_value=FirstLegalPolicy()),
                patch("pokezero.refutation_cli.mine_refutations", side_effect=fake_mine_refutations),
                contextlib.redirect_stdout(stdout),
            ):
                exit_code = refutation_cli_main(
                    [
                        "mine",
                        "--records",
                        str(records_path),
                        "--out-dir",
                        str(temp_path / "out"),
                        "--champion-policy-id",
                        "champion",
                        "--p1-policy",
                        "random-legal",
                        "--p2-policy",
                        "random-legal",
                    ]
                )

        self.assertEqual(exit_code, 0)
        self.assertTrue(captured["evaluator"].reseed_simulator_rng)
        self.assertEqual(captured["evaluator"].reseed_scope, "simulator_rng")

    def test_reproduce_refutation_archive_reruns_terminal_results(self) -> None:
        config = RefutationMiningConfig(
            champion_policy_id="champion",
            certification_seed_count=20,
            max_decision_points_per_game=1,
            max_deviations_per_state=1,
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            archive_path = Path(temp_dir) / "fragile.jsonl"
            report = mine_refutations(
                records=(_record(),),
                config=config,
                evaluator=FakeTerminalEvaluator(loser_winning_actions={2}, loser_win_count=20),
                archive_path=archive_path,
            )
            payload = reproduce_refutation_archive(
                records=(_record(),),
                fragile_states=tuple(iter_fragile_states(archive_path)),
                evaluator=FakeTerminalEvaluator(loser_winning_actions={2}, loser_win_count=20),
            )

        self.assertEqual(len(report.certified_refutations), 1)
        self.assertTrue(payload["passed"])
        self.assertEqual(payload["row_count"], 1)
        self.assertEqual(payload["mismatch_count"], 0)
        self.assertEqual(payload["rows"][0]["observed_summary"]["flip_rate"], 1.0)

    def test_reproduce_refutation_archive_reports_mismatch(self) -> None:
        config = RefutationMiningConfig(
            champion_policy_id="champion",
            certification_seed_count=20,
            max_decision_points_per_game=1,
            max_deviations_per_state=1,
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            archive_path = Path(temp_dir) / "fragile.jsonl"
            mine_refutations(
                records=(_record(),),
                config=config,
                evaluator=FakeTerminalEvaluator(loser_winning_actions={2}, loser_win_count=20),
                archive_path=archive_path,
            )
            payload = reproduce_refutation_archive(
                records=(_record(),),
                fragile_states=tuple(iter_fragile_states(archive_path)),
                evaluator=FakeTerminalEvaluator(loser_winning_actions=set(), loser_win_count=0),
            )

        self.assertFalse(payload["passed"])
        self.assertEqual(payload["mismatch_count"], 1)
        self.assertEqual(payload["rows"][0]["terminal_mismatch_count"], 20)

    def test_reproduce_refutation_archive_rejects_wrong_source_record_identity(self) -> None:
        config = RefutationMiningConfig(
            champion_policy_id="champion",
            certification_seed_count=20,
            max_decision_points_per_game=1,
            max_deviations_per_state=1,
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            archive_path = Path(temp_dir) / "fragile.jsonl"
            mine_refutations(
                records=(_record(battle_id="battle-1"),),
                config=config,
                evaluator=FakeTerminalEvaluator(loser_winning_actions={2}, loser_win_count=20),
                archive_path=archive_path,
            )
            with self.assertRaisesRegex(ValueError, "source record does not match"):
                reproduce_refutation_archive(
                    records=(_record(battle_id="battle-2"),),
                    fragile_states=tuple(iter_fragile_states(archive_path)),
                    evaluator=FakeTerminalEvaluator(loser_winning_actions={2}, loser_win_count=20),
                )

    def test_reproduce_refutation_archive_requires_canonical_seed_protocol(self) -> None:
        config = RefutationMiningConfig(
            champion_policy_id="champion",
            certification_seed_count=20,
            max_decision_points_per_game=1,
            max_deviations_per_state=1,
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            archive_path = Path(temp_dir) / "fragile.jsonl"
            mine_refutations(
                records=(_record(),),
                config=config,
                evaluator=FakeTerminalEvaluator(loser_winning_actions={2}, loser_win_count=20),
                archive_path=archive_path,
            )
            rows = [dict(row) for row in iter_fragile_states(archive_path)]
            for result in rows[0]["terminal_results"]:
                result["certification_seed"] += 1000
            payload = reproduce_refutation_archive(
                records=(_record(),),
                fragile_states=rows,
                evaluator=FakeTerminalEvaluator(loser_winning_actions={2}, loser_win_count=20),
            )

        self.assertFalse(payload["passed"])
        self.assertEqual(payload["mismatch_count"], 1)
        self.assertFalse(payload["rows"][0]["seed_protocol"]["passed"])
        self.assertEqual(payload["rows"][0]["terminal_mismatch_count"], 0)

    def test_reproduce_refutation_archive_requires_archived_threshold_pass(self) -> None:
        config = RefutationMiningConfig(
            champion_policy_id="champion",
            certification_seed_count=20,
            max_decision_points_per_game=1,
            max_deviations_per_state=1,
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            archive_path = Path(temp_dir) / "fragile.jsonl"
            mine_refutations(
                records=(_record(),),
                config=config,
                evaluator=FakeTerminalEvaluator(loser_winning_actions={2}, loser_win_count=20),
                archive_path=archive_path,
            )
            rows = [dict(row) for row in iter_fragile_states(archive_path)]
            rows[0]["certification"]["min_flip_rate"] = 1.0
            payload = reproduce_refutation_archive(
                records=(_record(),),
                fragile_states=rows,
                evaluator=FakeTerminalEvaluator(loser_winning_actions={2}, loser_win_count=20),
            )

        self.assertFalse(payload["passed"])
        self.assertEqual(payload["mismatch_count"], 1)
        self.assertFalse(payload["rows"][0]["threshold_passes"])

    def test_reproduce_refutation_archive_fails_empty_archive(self) -> None:
        payload = reproduce_refutation_archive(
            records=(_record(),),
            fragile_states=(),
            evaluator=FakeTerminalEvaluator(loser_winning_actions={2}, loser_win_count=20),
        )

        self.assertFalse(payload["passed"])
        self.assertEqual(payload["row_count"], 0)

    def test_reproduce_refutation_archive_requires_simulator_rng_evaluator(self) -> None:
        with self.assertRaisesRegex(ValueError, "simulator-RNG-reseeded"):
            reproduce_refutation_archive(
                records=(_record(),),
                fragile_states=(),
                evaluator=ContinuationOnlyEvaluator(loser_winning_actions={2}, loser_win_count=20),
            )

    def test_reproduce_refutation_archive_requires_archived_simulator_rng_scope(self) -> None:
        config = RefutationMiningConfig(
            champion_policy_id="champion",
            certification_seed_count=20,
            max_decision_points_per_game=1,
            max_deviations_per_state=1,
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            archive_path = Path(temp_dir) / "fragile.jsonl"
            mine_refutations(
                records=(_record(),),
                config=config,
                evaluator=FakeTerminalEvaluator(loser_winning_actions={2}, loser_win_count=20),
                archive_path=archive_path,
            )
            rows = [dict(row) for row in iter_fragile_states(archive_path)]
            rows[0]["certification"]["reseed_scope"] = "continuation_policy_rng"
            rows[0]["certification"]["simulator_rng_reseeded"] = False
            payload = reproduce_refutation_archive(
                records=(_record(),),
                fragile_states=rows,
                evaluator=FakeTerminalEvaluator(loser_winning_actions={2}, loser_win_count=20),
            )

        self.assertFalse(payload["passed"])
        self.assertEqual(payload["mismatch_count"], 1)
        self.assertFalse(payload["rows"][0]["archive_reseed_scope_passes"])

    def test_reproduce_refutation_archive_rejects_label_swap_against_record_winner(self) -> None:
        config = RefutationMiningConfig(
            champion_policy_id="champion",
            certification_seed_count=20,
            max_decision_points_per_game=1,
            max_deviations_per_state=1,
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            archive_path = Path(temp_dir) / "fragile.jsonl"
            mine_refutations(
                records=(_record(),),
                config=config,
                evaluator=FakeTerminalEvaluator(loser_winning_actions={2}, loser_win_count=20),
                archive_path=archive_path,
            )
            rows = [dict(row) for row in iter_fragile_states(archive_path)]
            rows[0]["candidate"]["champion_player_id"] = "p2"
            rows[0]["candidate"]["loser_player_id"] = "p1"
            with self.assertRaisesRegex(ValueError, "winner record"):
                reproduce_refutation_archive(
                    records=(_record(),),
                    fragile_states=rows,
                    evaluator=FakeTerminalEvaluator(loser_winning_actions={2}, loser_win_count=20),
                )

    def test_cli_reproduce_uses_simulator_rng_reseeded_evaluator(self) -> None:
        captured = {}

        def fake_reproduce_refutation_archive(**kwargs):
            captured["evaluator"] = kwargs["evaluator"]
            return {
                "schema_version": "pokezero.refutation_reproduction.v1",
                "passed": True,
                "row_count": 1,
                "mismatch_count": 0,
                "rows": [],
            }

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            records_path = temp_path / "records.jsonl"
            archive_path = temp_path / "fragile.jsonl"
            with records_path.open("w", encoding="utf-8") as handle:
                write_rollout_record(handle, _record())
            archive_path.write_text("{}\n", encoding="utf-8")

            stdout = io.StringIO()
            with (
                patch("pokezero.refutation_cli.env_config_with_policy_spec_masks", side_effect=lambda config, *_args, **_kwargs: config),
                patch("pokezero.refutation_cli.policy_spec_with_showdown_root", side_effect=lambda spec, _root: spec),
                patch("pokezero.refutation_cli.policy_from_spec", return_value=FirstLegalPolicy()),
                patch("pokezero.refutation_cli.reproduce_refutation_archive", side_effect=fake_reproduce_refutation_archive),
                contextlib.redirect_stdout(stdout),
            ):
                exit_code = refutation_cli_main(
                    [
                        "reproduce",
                        "--records",
                        str(records_path),
                        "--archive",
                        str(archive_path),
                        "--p1-policy",
                        "random-legal",
                        "--p2-policy",
                        "random-legal",
                    ]
                )

        self.assertEqual(exit_code, 0)
        self.assertTrue(captured["evaluator"].reseed_simulator_rng)
        self.assertEqual(captured["evaluator"].reseed_scope, "simulator_rng")

    def test_validate_refutation_report_payload_accepts_complete_terminal_archive(self) -> None:
        config = RefutationMiningConfig(
            champion_policy_id="champion",
            certification_seed_count=20,
            max_decision_points_per_game=1,
            max_deviations_per_state=1,
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            archive_path = Path(temp_dir) / "fragile.jsonl"
            report = mine_refutations(
                records=(_record(),),
                config=config,
                evaluator=FakeTerminalEvaluator(loser_winning_actions={2}, loser_win_count=20),
                archive_path=archive_path,
            )
            payload = validate_refutation_report_payload(
                report=report.to_dict(),
                fragile_states=tuple(iter_fragile_states(archive_path)),
                min_sampled_wins=1,
                min_certified_refutations=1,
            )

        self.assertTrue(payload["passed"])
        self.assertTrue(all(check["passed"] for check in payload["checks"]))

    def test_validate_refutation_report_payload_recomputes_certification_from_terminal_results(self) -> None:
        config = RefutationMiningConfig(
            champion_policy_id="champion",
            certification_seed_count=20,
            max_decision_points_per_game=1,
            max_deviations_per_state=1,
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            archive_path = Path(temp_dir) / "fragile.jsonl"
            report = mine_refutations(
                records=(_record(),),
                config=config,
                evaluator=FakeTerminalEvaluator(loser_winning_actions={2}, loser_win_count=20),
                archive_path=archive_path,
            )
            rows = [dict(row) for row in iter_fragile_states(archive_path)]
            for result in rows[0]["terminal_results"]:
                result["winner"] = rows[0]["candidate"]["champion_player_id"]
            payload = validate_refutation_report_payload(
                report=report.to_dict(),
                fragile_states=rows,
                min_sampled_wins=1,
                min_certified_refutations=1,
            )

        self.assertFalse(payload["passed"])
        failed = {check["name"] for check in payload["checks"] if not check["passed"]}
        self.assertIn("certification_consistency_rows", failed)
        self.assertIn("flip_rate_rows", failed)

    def test_validate_refutation_report_payload_requires_distinct_certification_seeds(self) -> None:
        config = RefutationMiningConfig(
            champion_policy_id="champion",
            certification_seed_count=20,
            max_decision_points_per_game=1,
            max_deviations_per_state=1,
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            archive_path = Path(temp_dir) / "fragile.jsonl"
            report = mine_refutations(
                records=(_record(),),
                config=config,
                evaluator=FakeTerminalEvaluator(loser_winning_actions={2}, loser_win_count=20),
                archive_path=archive_path,
            )
            rows = [dict(row) for row in iter_fragile_states(archive_path)]
            for result in rows[0]["terminal_results"]:
                result["certification_seed"] = 101
            payload = validate_refutation_report_payload(
                report=report.to_dict(),
                fragile_states=rows,
                min_sampled_wins=1,
                min_certified_refutations=1,
            )

        self.assertFalse(payload["passed"])
        failed = {check["name"] for check in payload["checks"] if not check["passed"]}
        self.assertIn("distinct_certification_seed_rows", failed)

    def test_validate_refutation_report_payload_rejects_continuation_only_reseeding_by_default(self) -> None:
        config = RefutationMiningConfig(
            champion_policy_id="champion",
            certification_seed_count=20,
            max_decision_points_per_game=1,
            max_deviations_per_state=1,
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            archive_path = Path(temp_dir) / "fragile.jsonl"
            report = mine_refutations(
                records=(_record(),),
                config=config,
                evaluator=FakeTerminalEvaluator(loser_winning_actions={2}, loser_win_count=20),
                archive_path=archive_path,
            )
            rows = [dict(row) for row in iter_fragile_states(archive_path)]
            rows[0]["certification"]["reseed_scope"] = "continuation_policy_rng"
            rows[0]["certification"]["simulator_rng_reseeded"] = False
            rows[0]["search_stats"]["reseed_scope"] = "continuation_policy_rng"
            rows[0]["search_stats"]["simulator_rng_reseeded"] = False
            payload = validate_refutation_report_payload(
                report=report.to_dict(),
                fragile_states=rows,
                min_sampled_wins=1,
                min_certified_refutations=1,
            )

        self.assertFalse(payload["passed"])
        failed = {check["name"] for check in payload["checks"] if not check["passed"]}
        self.assertIn("simulator_rng_reseeded_rows", failed)
        self.assertEqual(payload["summary"]["simulator_rng_reseeded_count"], 0)
        self.assertEqual(payload["summary"]["reseed_scope_counts"], {"continuation_policy_rng": 1})
        self.assertEqual(payload["warnings"][0]["name"], "simulator_rng_not_fully_reseeded")

    def test_validate_refutation_report_payload_can_waive_simulator_rng_for_exploratory_reports(self) -> None:
        config = RefutationMiningConfig(
            champion_policy_id="champion",
            certification_seed_count=20,
            max_decision_points_per_game=1,
            max_deviations_per_state=1,
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            archive_path = Path(temp_dir) / "fragile.jsonl"
            report = mine_refutations(
                records=(_record(),),
                config=config,
                evaluator=FakeTerminalEvaluator(loser_winning_actions={2}, loser_win_count=20),
                archive_path=archive_path,
            )
            rows = [dict(row) for row in iter_fragile_states(archive_path)]
            rows[0]["certification"]["reseed_scope"] = "continuation_policy_rng"
            rows[0]["certification"]["simulator_rng_reseeded"] = False
            rows[0]["search_stats"]["reseed_scope"] = "continuation_policy_rng"
            rows[0]["search_stats"]["simulator_rng_reseeded"] = False
            payload = validate_refutation_report_payload(
                report=report.to_dict(),
                fragile_states=rows,
                min_sampled_wins=1,
                min_certified_refutations=1,
                require_simulator_rng_reseed=False,
            )

        self.assertTrue(payload["passed"])
        self.assertFalse(payload["r0_acceptance_eligible"])
        self.assertEqual(payload["waivers"][0]["name"], "continuation_only_reseeds")
        self.assertTrue(payload["summary"]["simulator_rng_reseed_waived"])
        self.assertEqual(payload["summary"]["simulator_rng_reseeded_count"], 0)
        self.assertEqual(payload["warnings"][0]["name"], "simulator_rng_not_fully_reseeded")

    def test_validate_refutation_report_payload_rejects_inconsistent_simulator_rng_scope(self) -> None:
        config = RefutationMiningConfig(
            champion_policy_id="champion",
            certification_seed_count=20,
            max_decision_points_per_game=1,
            max_deviations_per_state=1,
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            archive_path = Path(temp_dir) / "fragile.jsonl"
            report = mine_refutations(
                records=(_record(),),
                config=config,
                evaluator=FakeTerminalEvaluator(loser_winning_actions={2}, loser_win_count=20),
                archive_path=archive_path,
            )
            rows = [dict(row) for row in iter_fragile_states(archive_path)]
            rows[0]["certification"]["simulator_rng_reseeded"] = True
            rows[0]["certification"]["reseed_scope"] = "continuation_policy_rng"
            payload = validate_refutation_report_payload(
                report=report.to_dict(),
                fragile_states=rows,
                min_sampled_wins=1,
                min_certified_refutations=1,
            )

        self.assertFalse(payload["passed"])
        failed = {check["name"] for check in payload["checks"] if not check["passed"]}
        self.assertIn("simulator_rng_reseed_scope_consistency_rows", failed)

    def test_validate_refutation_report_payload_enforces_fixed_r0_flip_rate(self) -> None:
        config = RefutationMiningConfig(
            champion_policy_id="champion",
            certification_seed_count=20,
            min_flip_rate=0.05,
            max_decision_points_per_game=1,
            max_deviations_per_state=1,
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            archive_path = Path(temp_dir) / "fragile.jsonl"
            report = mine_refutations(
                records=(_record(),),
                config=config,
                evaluator=FakeTerminalEvaluator(loser_winning_actions={2}, loser_win_count=12),
                archive_path=archive_path,
            )
            rows = [dict(row) for row in iter_fragile_states(archive_path)]
            payload = validate_refutation_report_payload(
                report=report.to_dict(),
                fragile_states=rows,
                min_sampled_wins=1,
                min_certified_refutations=1,
            )

        self.assertFalse(payload["passed"])
        failed = {check["name"] for check in payload["checks"] if not check["passed"]}
        self.assertIn("flip_rate_rows", failed)
        self.assertEqual(payload["summary"]["min_flip_rate"], 0.60)

    def test_cli_validate_enforces_default_r0_gate(self) -> None:
        config = RefutationMiningConfig(
            champion_policy_id="champion",
            certification_seed_count=20,
            max_decision_points_per_game=1,
            max_deviations_per_state=1,
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            archive_path = temp_path / "fragile.jsonl"
            report_path = temp_path / "report.json"
            report = mine_refutations(
                records=(_record(),),
                config=config,
                evaluator=FakeTerminalEvaluator(loser_winning_actions={2}, loser_win_count=20),
                archive_path=archive_path,
            )
            write_refutation_report(report_path, report)

            stdout = io.StringIO()
            with contextlib.redirect_stdout(stdout):
                exit_code = refutation_cli_main(
                    [
                        "validate",
                        "--report",
                        str(report_path),
                        "--archive",
                        str(archive_path),
                    ]
                )

        self.assertEqual(exit_code, 2)
        payload = json.loads(stdout.getvalue())
        self.assertFalse(payload["passed"])
        failed = {check["name"] for check in payload["checks"] if not check["passed"]}
        self.assertIn("sampled_win_count", failed)
        self.assertIn("distinct_certified_root_count", failed)

    def test_cli_validate_marks_smoke_thresholds_as_non_r0_eligible(self) -> None:
        config = RefutationMiningConfig(
            champion_policy_id="champion",
            certification_seed_count=20,
            max_decision_points_per_game=1,
            max_deviations_per_state=1,
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            archive_path = temp_path / "fragile.jsonl"
            report_path = temp_path / "report.json"
            report = mine_refutations(
                records=(_record(),),
                config=config,
                evaluator=FakeTerminalEvaluator(loser_winning_actions={2}, loser_win_count=20),
                archive_path=archive_path,
            )
            write_refutation_report(report_path, report)

            stdout = io.StringIO()
            with contextlib.redirect_stdout(stdout):
                exit_code = refutation_cli_main(
                    [
                        "validate",
                        "--report",
                        str(report_path),
                        "--archive",
                        str(archive_path),
                        "--min-sampled-wins",
                        "1",
                        "--min-certified-refutations",
                        "1",
                    ]
                )

        self.assertEqual(exit_code, 3)
        payload = json.loads(stdout.getvalue())
        self.assertTrue(payload["passed"])
        self.assertFalse(payload["r0_acceptance_eligible"])
        self.assertEqual(payload["waivers"][0]["name"], "relaxed_r0_thresholds")
        self.assertTrue(payload["summary"]["r0_thresholds_relaxed"])

    def test_cli_validate_can_waive_continuation_only_reseeds_for_dev_reports(self) -> None:
        config = RefutationMiningConfig(
            champion_policy_id="champion",
            certification_seed_count=20,
            max_decision_points_per_game=1,
            max_deviations_per_state=1,
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            archive_path = temp_path / "fragile.jsonl"
            report_path = temp_path / "report.json"
            report = mine_refutations(
                records=(_record(),),
                config=config,
                evaluator=FakeTerminalEvaluator(loser_winning_actions={2}, loser_win_count=20),
                archive_path=archive_path,
            )
            rows = [dict(row) for row in iter_fragile_states(archive_path)]
            rows[0]["certification"]["reseed_scope"] = "continuation_policy_rng"
            rows[0]["certification"]["simulator_rng_reseeded"] = False
            rows[0]["search_stats"]["reseed_scope"] = "continuation_policy_rng"
            rows[0]["search_stats"]["simulator_rng_reseeded"] = False
            archive_path.write_text(
                "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
                encoding="utf-8",
            )
            write_refutation_report(report_path, report)

            stdout = io.StringIO()
            with contextlib.redirect_stdout(stdout):
                exit_code = refutation_cli_main(
                    [
                        "validate",
                        "--report",
                        str(report_path),
                        "--archive",
                        str(archive_path),
                        "--min-sampled-wins",
                        "1",
                        "--min-certified-refutations",
                        "1",
                        "--allow-continuation-only-reseeds",
                    ]
                )

        payload = json.loads(stdout.getvalue())
        self.assertEqual(exit_code, 3)
        self.assertTrue(payload["passed"])
        self.assertFalse(payload["r0_acceptance_eligible"])
        self.assertEqual(payload["waivers"][0]["name"], "continuation_only_reseeds")
        self.assertEqual(payload["summary"]["simulator_rng_reseeded_count"], 0)

    def test_refutation_training_examples_retarget_loser_value_and_action(self) -> None:
        config = RefutationMiningConfig(
            champion_policy_id="champion",
            max_wins=10,
            certification_seed_count=20,
            min_flip_rate=0.60,
            max_decision_points_per_game=1,
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            report = mine_refutations(
                records=(_record(),),
                config=config,
                evaluator=FakeTerminalEvaluator(loser_winning_actions={2}, loser_win_count=13),
                archive_path=Path(temp_dir) / "fragile.jsonl",
            )
        row = report.certified_refutations[0].to_dict()

        examples = refutation_training_examples(
            records=(_record(),),
            fragile_states=(row,),
            config=RefutationTrainingConfig(target_mode="policy-value"),
        )

        self.assertEqual(len(examples), 1)
        example = examples[0]
        self.assertEqual(example.player_id, "p2")
        self.assertEqual(example.action_index, 2)
        self.assertAlmostEqual(example.return_value, 0.3)
        self.assertAlmostEqual(example.ppo_value_target, 0.3)
        self.assertEqual(example.training_weight, 1.0)
        self.assertIsNone(example.action_probability)
        self.assertEqual(example.step_metadata["refutation_training"]["deviation_action_index"], 2)

    def test_refutation_training_value_mode_keeps_recorded_action(self) -> None:
        config = RefutationMiningConfig(
            champion_policy_id="champion",
            max_wins=10,
            certification_seed_count=20,
            min_flip_rate=0.60,
            max_decision_points_per_game=1,
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            report = mine_refutations(
                records=(_record(),),
                config=config,
                evaluator=FakeTerminalEvaluator(loser_winning_actions={2}, loser_win_count=13),
                archive_path=Path(temp_dir) / "fragile.jsonl",
            )
        row = report.certified_refutations[0].to_dict()

        examples = refutation_training_examples(
            records=(_record(),),
            fragile_states=(row,),
            config=RefutationTrainingConfig(target_mode="value"),
        )

        self.assertEqual(len(examples), 1)
        self.assertEqual(examples[0].action_index, 1)
        self.assertAlmostEqual(examples[0].return_value, 0.3)
        self.assertAlmostEqual(examples[0].ppo_value_target, 0.3)
        self.assertIsNone(examples[0].action_probability)

    def test_refutation_training_policy_distribution_mode_emits_weighted_action_targets(self) -> None:
        config = RefutationMiningConfig(
            champion_policy_id="champion",
            max_wins=10,
            certification_seed_count=20,
            min_flip_rate=0.60,
            max_decision_points_per_game=1,
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            report = mine_refutations(
                records=(_record(),),
                config=config,
                evaluator=FakeTerminalEvaluator(loser_winning_actions={2}, loser_win_count=13),
                archive_path=Path(temp_dir) / "fragile.jsonl",
            )
        row = report.certified_refutations[0].to_dict()
        row["search_policy_distribution"] = {"2": 3.0, "3": 1.0}

        examples = refutation_training_examples(
            records=(_record(),),
            fragile_states=(row,),
            config=RefutationTrainingConfig(target_mode="policy-distribution-value"),
        )

        self.assertEqual([example.action_index for example in examples], [2, 3])
        self.assertAlmostEqual(examples[0].training_weight, 0.75)
        self.assertAlmostEqual(examples[1].training_weight, 0.25)
        self.assertAlmostEqual(
            examples[0].step_metadata["refutation_training"]["policy_target_probability"],
            0.75,
        )
        self.assertAlmostEqual(examples[0].return_value, 0.3)
        self.assertIsNone(examples[0].action_probability)

    def test_refutation_training_policy_distribution_mode_accepts_sequence_targets(self) -> None:
        config = RefutationMiningConfig(
            champion_policy_id="champion",
            max_wins=10,
            certification_seed_count=20,
            min_flip_rate=0.60,
            max_decision_points_per_game=1,
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            report = mine_refutations(
                records=(_record(),),
                config=config,
                evaluator=FakeTerminalEvaluator(loser_winning_actions={2}, loser_win_count=13),
                archive_path=Path(temp_dir) / "fragile.jsonl",
            )
        row = report.certified_refutations[0].to_dict()
        distribution = [0.0] * ACTION_COUNT
        distribution[2] = 3.0
        distribution[3] = 1.0
        row["search_policy_distribution"] = distribution

        examples = refutation_training_examples(
            records=(_record(),),
            fragile_states=(row,),
            config=RefutationTrainingConfig(target_mode="policy-distribution-value"),
        )

        self.assertEqual([example.action_index for example in examples], [2, 3])
        self.assertAlmostEqual(examples[0].training_weight, 0.75)
        self.assertAlmostEqual(examples[1].training_weight, 0.25)

    def test_refutation_training_policy_distribution_mode_requires_distribution(self) -> None:
        config = RefutationMiningConfig(
            champion_policy_id="champion",
            max_wins=10,
            certification_seed_count=20,
            min_flip_rate=0.60,
            max_decision_points_per_game=1,
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            report = mine_refutations(
                records=(_record(),),
                config=config,
                evaluator=FakeTerminalEvaluator(loser_winning_actions={2}, loser_win_count=13),
                archive_path=Path(temp_dir) / "fragile.jsonl",
            )
        row = report.certified_refutations[0].to_dict()

        with self.assertRaisesRegex(ValueError, "requires search_policy_distribution"):
            refutation_training_examples(
                records=(_record(),),
                fragile_states=(row,),
                config=RefutationTrainingConfig(target_mode="policy-distribution-value"),
            )

    def test_refutation_training_policy_distribution_mode_rejects_illegal_action_mass(self) -> None:
        config = RefutationMiningConfig(
            champion_policy_id="champion",
            max_wins=10,
            certification_seed_count=20,
            min_flip_rate=0.60,
            max_decision_points_per_game=1,
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            report = mine_refutations(
                records=(_record(),),
                config=config,
                evaluator=FakeTerminalEvaluator(loser_winning_actions={2}, loser_win_count=13),
                archive_path=Path(temp_dir) / "fragile.jsonl",
            )
        row = report.certified_refutations[0].to_dict()
        row["search_policy_distribution"] = {"4": 1.0}

        with self.assertRaisesRegex(ValueError, "illegal action 4"):
            refutation_training_examples(
                records=(_record(),),
                fragile_states=(row,),
                config=RefutationTrainingConfig(target_mode="policy-distribution-value"),
            )

    def test_refutation_training_policy_distribution_mode_caps_without_partial_rows(self) -> None:
        config = RefutationMiningConfig(
            champion_policy_id="champion",
            max_wins=10,
            certification_seed_count=20,
            min_flip_rate=0.60,
            max_decision_points_per_game=1,
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            report = mine_refutations(
                records=(_record(),),
                config=config,
                evaluator=FakeTerminalEvaluator(loser_winning_actions={2}, loser_win_count=13),
                archive_path=Path(temp_dir) / "fragile.jsonl",
            )
        two_target_row = report.certified_refutations[0].to_dict()
        two_target_row["search_policy_distribution"] = {"2": 3.0, "3": 1.0}
        one_target_row = report.certified_refutations[0].to_dict()
        one_target_row["search_policy_distribution"] = {"2": 1.0}

        examples = refutation_training_examples(
            records=(_record(),),
            fragile_states=(two_target_row, one_target_row),
            config=RefutationTrainingConfig(target_mode="policy-distribution-value", max_examples=1),
        )

        self.assertEqual(len(examples), 1)
        self.assertEqual(examples[0].action_index, 2)
        self.assertAlmostEqual(examples[0].training_weight, 1.0)

    def test_refutation_training_policy_distribution_summary_counts_rows_not_targets(self) -> None:
        config = RefutationMiningConfig(
            champion_policy_id="champion",
            max_wins=10,
            certification_seed_count=20,
            min_flip_rate=0.60,
            max_decision_points_per_game=1,
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            report = mine_refutations(
                records=(_record(),),
                config=config,
                evaluator=FakeTerminalEvaluator(loser_winning_actions={2}, loser_win_count=13),
                archive_path=root / "fragile.jsonl",
            )
            row = report.certified_refutations[0].to_dict()
            row["search_policy_distribution"] = {"2": 3.0, "3": 1.0}

            summary = write_refutation_training_cache(
                records=(_record(),),
                fragile_states=(row,),
                output_path=root / "refutation-cache",
                config=RefutationTrainingConfig(target_mode="policy-distribution-value"),
            )

        self.assertEqual(summary.fragile_state_count, 1)
        self.assertEqual(summary.example_count, 2)
        self.assertEqual(summary.skipped_count, 0)

    def test_refutation_training_prioritizes_higher_surprise_weights_before_cap(self) -> None:
        config = RefutationMiningConfig(
            champion_policy_id="champion",
            max_wins=10,
            certification_seed_count=20,
            min_flip_rate=0.60,
            max_decision_points_per_game=1,
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            report = mine_refutations(
                records=(_record(),),
                config=config,
                evaluator=FakeTerminalEvaluator(loser_winning_actions={2}, loser_win_count=13),
                archive_path=Path(temp_dir) / "fragile.jsonl",
            )
        low = report.certified_refutations[0].to_dict()
        high = dict(low)
        high["certification"] = dict(low["certification"])
        high["certification"]["flip_rate"] = 0.95
        high["terminal_results"] = list(low["terminal_results"])

        examples = refutation_training_examples(
            records=(_record(),),
            fragile_states=(low, high),
            config=RefutationTrainingConfig(
                target_mode="policy-value",
                max_examples=1,
                surprise_weight_scale=2.0,
            ),
        )

        self.assertEqual(len(examples), 1)
        self.assertGreater(examples[0].training_weight, 2.0)
        self.assertAlmostEqual(examples[0].step_metadata["refutation_training"]["flip_rate"], 0.95)

    def test_refutation_training_skips_non_certified_rows(self) -> None:
        config = RefutationMiningConfig(
            champion_policy_id="champion",
            max_wins=10,
            certification_seed_count=20,
            min_flip_rate=0.60,
            max_decision_points_per_game=1,
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            report = mine_refutations(
                records=(_record(),),
                config=config,
                evaluator=FakeTerminalEvaluator(loser_winning_actions={2}, loser_win_count=13),
                archive_path=Path(temp_dir) / "fragile.jsonl",
            )
        row = report.certified_refutations[0].to_dict()
        row["certification"]["passed"] = False

        examples = refutation_training_examples(
            records=(_record(),),
            fragile_states=(row,),
            config=RefutationTrainingConfig(target_mode="policy-value"),
        )

        self.assertEqual(examples, ())

    def test_refutation_training_cache_cli_writes_corrected_cache(self) -> None:
        config = RefutationMiningConfig(
            champion_policy_id="champion",
            max_wins=10,
            certification_seed_count=20,
            min_flip_rate=0.60,
            max_decision_points_per_game=1,
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            records_path = root / "records.jsonl"
            archive_path = root / "fragile.jsonl"
            cache_path = root / "refutation-cache"
            with records_path.open("w", encoding="utf-8") as handle:
                write_rollout_record(handle, _record())
            mine_refutations(
                records=(_record(),),
                config=config,
                evaluator=FakeTerminalEvaluator(loser_winning_actions={2}, loser_win_count=13),
                archive_path=archive_path,
            )
            stdout = io.StringIO()
            with contextlib.redirect_stdout(stdout):
                exit_code = refutation_cli_main(
                    [
                        "training-cache",
                        "--records",
                        str(records_path),
                        "--archive",
                        str(archive_path),
                        "--out",
                        str(cache_path),
                        "--surprise-weight-scale",
                        "2.0",
                    ]
                )
            payload = json.loads(stdout.getvalue())
            metadata = json.loads((cache_path / "metadata.json").read_text(encoding="utf-8"))
            (batch,) = tuple(iter_training_cache_batches(cache_path, batch_size=10))

        self.assertEqual(exit_code, 0)
        self.assertEqual(payload["example_count"], 1)
        self.assertEqual(payload["compatible_objectives"], ["behavior-cloning", "ppo", "reward-weighted"])
        self.assertEqual(metadata["example_count"], 1)
        self.assertEqual(metadata["refutation_training"]["target_mode"], "policy-value")
        self.assertEqual(
            metadata["refutation_training"]["surprise_weighting"],
            {
                "field": "training_weights",
                "max": 4.0,
                "mode": "certification-flip-rate",
                "scale": 2.0,
            },
        )
        self.assertAlmostEqual(metadata["refutation_training"]["training_weight_stats"]["mean"], 1.25)
        self.assertAlmostEqual(payload["training_weight_mean"], 1.25)
        self.assertEqual(
            metadata["refutation_training"]["compatible_objectives"],
            ["behavior-cloning", "ppo", "reward-weighted"],
        )
        self.assertEqual(batch.action_indices, (2,))
        self.assertAlmostEqual(batch.training_weights[0], 1.25, places=6)
        self.assertAlmostEqual(batch.returns[0], 0.3, places=6)
        self.assertAlmostEqual(batch.ppo_value_targets[0], 0.3, places=6)
        self.assertEqual(batch.ppo_value_target_mask, (True,))
        self.assertEqual(batch.action_probability_mask, (False,))

    def test_refutation_behavior_seed_manifest_uses_certified_rows_only(self) -> None:
        config = RefutationMiningConfig(
            champion_policy_id="champion",
            max_wins=10,
            certification_seed_count=20,
            min_flip_rate=0.60,
            max_decision_points_per_game=1,
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            report = mine_refutations(
                records=(_record(),),
                config=config,
                evaluator=FakeTerminalEvaluator(loser_winning_actions={2}, loser_win_count=13),
                archive_path=Path(temp_dir) / "fragile.jsonl",
            )
        row = report.certified_refutations[0].to_dict()
        failed = dict(row)
        failed["certification"] = dict(row["certification"])
        failed["certification"]["passed"] = False

        manifest = build_refutation_behavior_seed_manifest(
            (failed, row),
            config=RefutationBehaviorSeedConfig(min_flip_rate=0.60, mode="oracle"),
        )
        payload = manifest.to_dict()

        self.assertEqual(payload["schema_version"], REFUTATION_BEHAVIOR_SEED_MANIFEST_SCHEMA_VERSION)
        self.assertRegex(payload["source_digest"], r"^sha256:[0-9a-f]{64}$")
        self.assertEqual(payload["source_row_count"], 2)
        self.assertEqual(payload["seed_count"], 1)
        self.assertEqual(payload["skipped_count"], 1)
        seed = payload["seeds"][0]
        self.assertEqual(seed["seed_id"], "battle-1:round-0:step-1:action-2")
        self.assertEqual(seed["population_use"]["kind"], "refutation_behavior_seed")
        self.assertIn("legacy_checkpoint_strength_eval", seed["population_use"]["not_for"])

    def test_refutation_behavior_seed_manifest_cli_writes_payload(self) -> None:
        config = RefutationMiningConfig(
            champion_policy_id="champion",
            max_wins=10,
            certification_seed_count=20,
            min_flip_rate=0.60,
            max_decision_points_per_game=1,
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            archive_path = root / "fragile.jsonl"
            output_path = root / "behavior-seeds.json"
            mine_refutations(
                records=(_record(),),
                config=config,
                evaluator=FakeTerminalEvaluator(loser_winning_actions={2}, loser_win_count=13),
                archive_path=archive_path,
            )
            stdout = io.StringIO()
            with contextlib.redirect_stdout(stdout):
                exit_code = refutation_cli_main(
                    [
                        "behavior-seeds",
                        "--archive",
                        str(archive_path),
                        "--out",
                        str(output_path),
                        "--min-flip-rate",
                        "0.6",
                    ]
                )
            printed = json.loads(stdout.getvalue())
            written = json.loads(output_path.read_text(encoding="utf-8"))

        self.assertEqual(exit_code, 0)
        self.assertEqual(printed, written)
        self.assertEqual(written["seed_count"], 1)
        self.assertEqual(written["seeds"][0]["deviation_action_index"], 2)

    def test_refutation_behavior_seed_manifest_filters_and_caps_rows(self) -> None:
        config = RefutationMiningConfig(
            champion_policy_id="champion",
            max_wins=10,
            certification_seed_count=20,
            min_flip_rate=0.60,
            max_decision_points_per_game=1,
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            report = mine_refutations(
                records=(_record(),),
                config=config,
                evaluator=FakeTerminalEvaluator(loser_winning_actions={2}, loser_win_count=13),
                archive_path=Path(temp_dir) / "fragile.jsonl",
            )
        oracle = report.certified_refutations[0].to_dict()
        fair = report.certified_refutations[0].to_dict()
        fair["mode"] = "fair"

        fair_only = build_refutation_behavior_seed_manifest(
            (oracle, fair),
            config=RefutationBehaviorSeedConfig(mode="fair"),
        ).to_dict()
        capped = build_refutation_behavior_seed_manifest(
            (oracle, fair),
            config=RefutationBehaviorSeedConfig(max_seeds=1),
        ).to_dict()
        filtered = build_refutation_behavior_seed_manifest(
            (oracle, fair),
            config=RefutationBehaviorSeedConfig(min_flip_rate=0.99),
        ).to_dict()

        self.assertEqual(fair_only["seed_count"], 1)
        self.assertEqual(fair_only["seeds"][0]["mode"], "fair")
        self.assertEqual(capped["seed_count"], 1)
        self.assertEqual(capped["skipped_count"], 1)
        self.assertEqual(filtered["seed_count"], 0)
        self.assertEqual(filtered["skipped_count"], 2)

    def test_refutation_behavior_seed_manifest_rejects_non_terminal_rollout_rows(self) -> None:
        config = RefutationMiningConfig(
            champion_policy_id="champion",
            max_wins=10,
            certification_seed_count=20,
            min_flip_rate=0.60,
            max_decision_points_per_game=1,
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            report = mine_refutations(
                records=(_record(),),
                config=config,
                evaluator=FakeTerminalEvaluator(loser_winning_actions={2}, loser_win_count=13),
                archive_path=Path(temp_dir) / "fragile.jsonl",
            )
        row = report.certified_refutations[0].to_dict()
        row["evaluation_source"] = "value_head"

        with self.assertRaisesRegex(ValueError, "terminal-rollout"):
            build_refutation_behavior_seed_manifest((row,))

    def test_refutation_cycle_report_tracks_mode_trends_and_oracle_fair_gap(self) -> None:
        report = build_refutation_cycle_report(
            (
                RefutationCycleReportInput(
                    cycle_id="cycle-a",
                    report_path=Path("oracle-a.json"),
                    report=_refutation_report_payload(mode="oracle", sampled_win_count=200, refuted_game_count=80),
                ),
                RefutationCycleReportInput(
                    cycle_id="cycle-a",
                    report_path=Path("fair-a.json"),
                    report=_refutation_report_payload(mode="fair", sampled_win_count=200, refuted_game_count=50),
                ),
                RefutationCycleReportInput(
                    cycle_id="cycle-b",
                    report_path=Path("oracle-b.json"),
                    report=_refutation_report_payload(mode="oracle", sampled_win_count=200, refuted_game_count=40),
                ),
                RefutationCycleReportInput(
                    cycle_id="cycle-b",
                    report_path=Path("fair-b.json"),
                    report=_refutation_report_payload(mode="fair", sampled_win_count=200, refuted_game_count=20),
                ),
            )
        ).to_dict()

        self.assertEqual(report["schema_version"], REFUTATION_CYCLE_REPORT_SCHEMA_VERSION)
        self.assertEqual(report["cycle_count"], 2)
        self.assertTrue(report["mode_trends"]["oracle"]["declining"])
        self.assertEqual(report["mode_trends"]["oracle"]["delta"], -0.2)
        self.assertTrue(report["mode_trends"]["fair"]["declining"])
        self.assertEqual(len(report["oracle_fair_gaps"]), 2)
        self.assertEqual(report["oracle_fair_gaps"][0]["cycle_id"], "cycle-a")
        self.assertAlmostEqual(report["oracle_fair_gaps"][0]["oracle_minus_fair_refutation_rate"], 0.15)

    def test_refutation_cycle_report_sorts_cycles_before_trend(self) -> None:
        report = build_refutation_cycle_report(
            (
                RefutationCycleReportInput(
                    cycle_id="cycle-2",
                    report_path=Path("oracle-2.json"),
                    report=_refutation_report_payload(mode="oracle", sampled_win_count=200, refuted_game_count=40),
                ),
                RefutationCycleReportInput(
                    cycle_id="cycle-1",
                    report_path=Path("oracle-1.json"),
                    report=_refutation_report_payload(mode="oracle", sampled_win_count=200, refuted_game_count=80),
                ),
            )
        ).to_dict()

        self.assertEqual([row["cycle_id"] for row in report["mode_trends"]["oracle"]["rows"]], ["cycle-1", "cycle-2"])
        self.assertTrue(report["mode_trends"]["oracle"]["declining"])
        self.assertEqual(report["mode_trends"]["oracle"]["delta"], -0.2)

    def test_refutation_cycle_report_rejects_duplicate_cycle_mode(self) -> None:
        with self.assertRaisesRegex(ValueError, "duplicate refutation report"):
            build_refutation_cycle_report(
                (
                    RefutationCycleReportInput(
                        cycle_id="cycle-a",
                        report_path=Path("oracle-a-1.json"),
                        report=_refutation_report_payload(mode="oracle", sampled_win_count=200, refuted_game_count=80),
                    ),
                    RefutationCycleReportInput(
                        cycle_id="cycle-a",
                        report_path=Path("oracle-a-2.json"),
                        report=_refutation_report_payload(mode="oracle", sampled_win_count=200, refuted_game_count=70),
                    ),
                )
            )

    def test_refutation_cycle_report_rejects_zero_sample_reports(self) -> None:
        payload = _refutation_report_payload(mode="oracle", sampled_win_count=1, refuted_game_count=0)
        payload["sampled_win_count"] = 0
        payload["refutation_rate"] = 0.0

        with self.assertRaisesRegex(ValueError, "sampled_win_count must be positive"):
            build_refutation_cycle_report(
                (
                    RefutationCycleReportInput(
                        cycle_id="cycle-a",
                        report_path=Path("oracle-a.json"),
                        report=payload,
                    ),
                )
            )

    def test_refutation_cycle_report_cli_writes_payload(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            oracle_a = root / "oracle-a.json"
            fair_a = root / "fair-a.json"
            out = root / "cycle-report.json"
            oracle_a.write_text(
                json.dumps(_refutation_report_payload(mode="oracle", sampled_win_count=200, refuted_game_count=80)),
                encoding="utf-8",
            )
            fair_a.write_text(
                json.dumps(_refutation_report_payload(mode="fair", sampled_win_count=200, refuted_game_count=50)),
                encoding="utf-8",
            )
            stdout = io.StringIO()
            with contextlib.redirect_stdout(stdout):
                exit_code = refutation_cli_main(
                    [
                        "cycle-report",
                        "--report",
                        f"cycle-a={oracle_a}",
                        "--report",
                        f"cycle-a={fair_a}",
                        "--out",
                        str(out),
                    ]
                )
            printed = json.loads(stdout.getvalue())
            written = json.loads(out.read_text(encoding="utf-8"))

        self.assertEqual(exit_code, 0)
        self.assertEqual(printed, written)
        self.assertEqual(written["oracle_fair_gaps"][0]["cycle_id"], "cycle-a")

    def test_refutation_cycle_report_rejects_schema_mismatch(self) -> None:
        with self.assertRaisesRegex(ValueError, "unsupported refutation report schema"):
            build_refutation_cycle_report(
                (
                    RefutationCycleReportInput(
                        cycle_id="cycle-a",
                        report_path=Path("bad.json"),
                        report={"schema_version": "wrong"},
                    ),
                )
            )

    def test_refutation_curriculum_start_count_uses_ceiling_and_cap(self) -> None:
        self.assertEqual(
            refutation_curriculum_start_count(
                RefutationCurriculumConfig(total_games=101, curriculum_fraction=0.01)
            ),
            2,
        )
        self.assertEqual(
            refutation_curriculum_start_count(
                RefutationCurriculumConfig(total_games=1000, curriculum_fraction=0.10, max_starts=7)
            ),
            7,
        )

    def test_refutation_curriculum_collects_from_fragile_decision_boundary(self) -> None:
        mining_config = RefutationMiningConfig(
            champion_policy_id="champion",
            max_wins=10,
            certification_seed_count=20,
            min_flip_rate=0.60,
            max_decision_points_per_game=1,
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            archive_path = root / "fragile.jsonl"
            output_path = root / "curriculum.jsonl"
            report = mine_refutations(
                records=(_record(),),
                config=mining_config,
                evaluator=FakeTerminalEvaluator(loser_winning_actions={2}, loser_win_count=13),
                archive_path=archive_path,
            )
            envs = []

            def env_factory():
                env = BranchReplayEnv()
                envs.append(env)
                return env

            summary = collect_refutation_curriculum_rollouts(
                records=(_record(),),
                fragile_states=(report.certified_refutations[0].to_dict(),),
                env_factory=env_factory,
                policies={"p1": FirstLegalPolicy(), "p2": FirstLegalPolicy()},
                rollout_config=RolloutConfig(max_decision_rounds=5),
                output_path=output_path,
                config=RefutationCurriculumConfig(
                    total_games=10,
                    curriculum_fraction=0.1,
                    seed_start=500,
                ),
            )
            (record,) = read_rollout_records(output_path)

        self.assertEqual(summary.requested_start_count, 1)
        self.assertEqual(summary.emitted_count, 1)
        self.assertEqual(record.seed, 500)
        self.assertEqual(record.policy_ids, {"p1": "first-legal", "p2": "first-legal"})
        self.assertEqual(record.trajectory.metadata["starting_decision_round_index"], 0)
        curriculum = record.trajectory.metadata["refutation_curriculum"]
        self.assertEqual(curriculum["schema_version"], REFUTATION_CURRICULUM_METADATA_SCHEMA_VERSION)
        self.assertEqual(curriculum["source_schema_version"], FRAGILE_STATE_SCHEMA_VERSION)
        self.assertEqual(curriculum["source_battle_id"], "battle-1")
        self.assertEqual(curriculum["source_seed"], 100)
        self.assertEqual(curriculum["decision_round_index"], 0)
        self.assertEqual(curriculum["loser_player_id"], "p2")
        self.assertEqual(curriculum["deviation_action_index"], 2)
        self.assertAlmostEqual(curriculum["flip_rate"], 13 / 20)
        self.assertEqual(envs[0].reset_calls, [(100, "gen3randombattle")])
        self.assertTrue(getattr(envs[0], "closed"))

    def test_refutation_curriculum_replays_nonzero_prefix_before_policy_control(self) -> None:
        mining_config = RefutationMiningConfig(
            champion_policy_id="champion",
            max_wins=10,
            certification_seed_count=20,
            min_flip_rate=0.60,
            max_decision_points_per_game=2,
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            report = mine_refutations(
                records=(_record(),),
                config=mining_config,
                evaluator=FakeTerminalEvaluator(loser_winning_actions={5}, loser_win_count=13),
                archive_path=root / "fragile.jsonl",
            )
            row = report.certified_refutations[0].to_dict()
            self.assertEqual(row["candidate"]["decision_round_index"], 1)
            envs = []

            def env_factory():
                env = PrefixReplayEnv()
                envs.append(env)
                return env

            collect_refutation_curriculum_rollouts(
                records=(_record(),),
                fragile_states=(row,),
                env_factory=env_factory,
                policies={"p1": FirstLegalPolicy(), "p2": FirstLegalPolicy()},
                rollout_config=RolloutConfig(max_decision_rounds=5),
                output_path=root / "curriculum.jsonl",
                config=RefutationCurriculumConfig(total_games=1, curriculum_fraction=1.0),
            )

        self.assertEqual(envs[0].step_calls[0], {"p1": 0, "p2": 1})
        self.assertEqual(envs[0].step_calls[1], {"p1": 0, "p2": 0})

    def test_refutation_curriculum_cycles_rows_with_repeat_metadata(self) -> None:
        mining_config = RefutationMiningConfig(
            champion_policy_id="champion",
            max_wins=10,
            certification_seed_count=20,
            min_flip_rate=0.60,
            max_decision_points_per_game=1,
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            report = mine_refutations(
                records=(_record(),),
                config=mining_config,
                evaluator=FakeTerminalEvaluator(loser_winning_actions={2}, loser_win_count=13),
                archive_path=root / "fragile.jsonl",
            )
            output_path = root / "curriculum.jsonl"
            collect_refutation_curriculum_rollouts(
                records=(_record(),),
                fragile_states=(report.certified_refutations[0].to_dict(),),
                env_factory=BranchReplayEnv,
                policies={"p1": FirstLegalPolicy(), "p2": FirstLegalPolicy()},
                rollout_config=RolloutConfig(max_decision_rounds=5),
                output_path=output_path,
                config=RefutationCurriculumConfig(total_games=3, curriculum_fraction=1.0),
            )
            records = read_rollout_records(output_path)

        self.assertEqual(len(records), 3)
        self.assertEqual(
            [record.trajectory.metadata["refutation_curriculum"]["repeat_index"] for record in records],
            [0, 1, 2],
        )
        self.assertEqual([record.seed for record in records], [1, 2, 3])

    def test_refutation_curriculum_rejects_mismatched_source_coordinates(self) -> None:
        mining_config = RefutationMiningConfig(
            champion_policy_id="champion",
            max_wins=10,
            certification_seed_count=20,
            min_flip_rate=0.60,
            max_decision_points_per_game=1,
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            report = mine_refutations(
                records=(_record(),),
                config=mining_config,
                evaluator=FakeTerminalEvaluator(loser_winning_actions={2}, loser_win_count=13),
                archive_path=root / "fragile.jsonl",
            )
            row = report.certified_refutations[0].to_dict()
            row["candidate"] = dict(row["candidate"])
            row["candidate"]["seed"] = 999
            with self.assertRaisesRegex(ValueError, "seed does not match"):
                collect_refutation_curriculum_rollouts(
                    records=(_record(),),
                    fragile_states=(row,),
                    env_factory=BranchReplayEnv,
                    policies={"p1": FirstLegalPolicy(), "p2": FirstLegalPolicy()},
                    rollout_config=RolloutConfig(max_decision_rounds=5),
                    output_path=root / "curriculum.jsonl",
                    config=RefutationCurriculumConfig(total_games=1, curriculum_fraction=1.0),
                )


if __name__ == "__main__":
    unittest.main()
