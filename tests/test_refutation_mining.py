import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path

from pokezero.actions import ACTION_COUNT
from pokezero.collection import RolloutRecord, write_rollout_record
from pokezero.dataset import iter_training_cache_batches
from pokezero.env import StepResult, TerminalState
from pokezero.observation import PokeZeroObservationV0
from pokezero.policy import PolicyDecision
from pokezero.refutation_cli import main as refutation_cli_main
from pokezero.refutation_mining import (
    BranchTerminalResult,
    FRAGILE_STATE_SCHEMA_VERSION,
    RefutationMiningConfig,
    ReplayTerminalBranchEvaluator,
    candidate_count_for_records,
    iter_fragile_states,
    mine_refutations,
    validate_refutation_report_payload,
    write_refutation_report,
)
from pokezero.refutation_training import RefutationTrainingConfig, refutation_training_examples
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


class FakeTerminalEvaluator:
    evaluation_source = "terminal_rollout"
    value_head_used = False
    reseed_scope = "simulator_rng"

    def __init__(self, *, loser_winning_actions: set[int], loser_win_count: int) -> None:
        self.loser_winning_actions = loser_winning_actions
        self.loser_win_count = loser_win_count
        self.calls = []

    def evaluate(self, *, record, candidate, certification_seed: int) -> BranchTerminalResult:
        self.calls.append((record.battle_id, candidate.deviation_action_index, certification_seed, dict(candidate.branch_actions)))
        count_for_candidate = sum(1 for _, action, _, _ in self.calls if action == candidate.deviation_action_index)
        loser_wins = (
            candidate.deviation_action_index in self.loser_winning_actions
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


class FirstLegalPolicy:
    policy_id = "first-legal"

    def select_action(self, observation, *, rng) -> PolicyDecision:
        action_index = next(index for index, legal in enumerate(observation.legal_action_mask) if legal)
        return PolicyDecision(action_index=action_index, policy_id=self.policy_id)


class BranchReplayEnv:
    def __init__(self) -> None:
        self.reset_calls = []
        self.step_calls = []
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

    def step(self, actions):
        self.step_calls.append(dict(actions))
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

    def test_validate_refutation_report_payload_warns_on_continuation_only_reseeding(self) -> None:
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

        self.assertTrue(payload["passed"])
        self.assertEqual(payload["summary"]["simulator_rng_reseeded_count"], 0)
        self.assertEqual(payload["summary"]["reseed_scope_counts"], {"continuation_policy_rng": 1})
        self.assertEqual(payload["warnings"][0]["name"], "simulator_rng_not_fully_reseeded")

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
        self.assertIn("certified_refutation_count", failed)

    def test_cli_validate_passes_with_explicit_smoke_thresholds(self) -> None:
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

        self.assertEqual(exit_code, 0)
        self.assertTrue(json.loads(stdout.getvalue())["passed"])

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


if __name__ == "__main__":
    unittest.main()
