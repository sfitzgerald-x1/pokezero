import io
import os
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch

from pokezero.collection import (
    BenchmarkMatchup,
    BenchmarkMatchupResult,
    BenchmarkReport,
    CollectionMetrics,
    aggregate_benchmark_head_to_heads,
    ReusableEnvPool,
    benchmark_rollouts,
    collect_training_cache,
    collect_rollouts,
    current_peak_rss_mb,
    default_benchmark_matchups,
    iter_rollout_records,
    linear_policy_factory_from_model_spec,
    policy_benchmark_matchups,
    policy_factory_from_spec,
    policy_from_name,
    policy_from_spec,
    policy_spec_with_showdown_root,
    read_rollout_records,
    rollout_record_from_dict,
    rollout_record_to_dict,
    summarize_records,
)
from pokezero.dataset import TrajectoryDatasetConfig, TrainingCacheSummary, is_training_cache_path, iter_training_batches
from pokezero.env import StepResult, TerminalState
from pokezero.linear_policy import LinearPolicyModel, LinearSoftmaxPolicy, save_linear_model
from pokezero.local_showdown import DEFAULT_SHOWDOWN_ROOT, LocalShowdownConfig, LocalShowdownEnv
from pokezero.neural_policy import TorchUnavailableError, torch_available
from pokezero.observation import ObservationPerspective, ObservationSpec, PokeZeroObservationV0
from pokezero.policy import PolicyDecision, RandomLegalPolicy, ScriptedTeacherPolicy
from pokezero.replay_benchmark import (
    ReplayPrefixBenchmarkReport,
    ReplayPrefixTiming,
    benchmark_replay_prefixes,
    replay_prefix_counts,
)
from pokezero.rollout import RolloutConfig
from pokezero.rollout_cli import main as rollout_cli_main, print_benchmark_report
from pokezero.trajectory import BattleTrajectory, TrajectoryStep, trajectory_from_dict, trajectory_to_dict


def observation(mask: tuple[bool, ...]) -> PokeZeroObservationV0:
    spec = ObservationSpec(categorical_feature_count=1, numeric_feature_count=1)
    return PokeZeroObservationV0(
        categorical_ids=tuple((index,) for index in range(spec.token_count)),
        numeric_features=tuple((float(index),) for index in range(spec.token_count)),
        token_type_ids=tuple(0 for _ in range(spec.token_count)),
        attention_mask=tuple(True for _ in range(spec.token_count)),
        legal_action_mask=mask,
        perspective=ObservationPerspective.from_showdown_slot("p1", "p1"),
    )


def trajectory() -> BattleTrajectory:
    mask = (True, False, False, False, False, False, False, False, False)
    result = BattleTrajectory(
        battle_id="battle-1",
        format_id="gen3randombattle",
        seed=123,
        metadata={"max_decision_rounds": 250},
    )
    result.append(
        TrajectoryStep(
            player_id="p1",
            turn_index=0,
            observation=observation(mask),
            legal_action_mask=mask,
            action_index=0,
            reward=1.0,
            opponent_action_index=1,
            action_probability=0.5,
            metadata={"policy_id": "random-legal"},
        )
    )
    result.record_terminal(TerminalState(winner="p1", turn_count=12))
    return result


class OneTurnEnv:
    def __init__(self) -> None:
        self._observation = observation((True, False, False, False, False, False, False, False, False))
        self._requested = ("p1", "p2")
        self._terminal = None
        self.closed = False

    def reset(self, *, seed: int, format_id: str = "gen3randombattle") -> None:
        self._requested = ("p1", "p2")
        self._terminal = None

    def observe(self, player: str) -> PokeZeroObservationV0:
        return self._observation

    def legal_actions(self, player: str) -> tuple[bool, ...]:
        return self._observation.legal_action_mask

    def requested_players(self) -> tuple[str, ...]:
        return self._requested

    def step(self, actions: dict[str, int]) -> StepResult:
        self._requested = ()
        self._terminal = TerminalState(winner="p1", turn_count=1)
        return StepResult(
            observations={},
            rewards={"p1": 1.0, "p2": -1.0},
            terminal=self._terminal,
            requested_players=(),
        )

    def terminal(self) -> TerminalState | None:
        return self._terminal

    def close(self) -> None:
        self.closed = True


class ResetFailingEnv:
    def reset(self, *, seed: int, format_id: str = "gen3randombattle") -> None:
        raise RuntimeError("boom")


class SeedRecordingEnv(OneTurnEnv):
    def __init__(self, reset_seeds: list[int]) -> None:
        super().__init__()
        self.reset_seeds = reset_seeds

    def reset(self, *, seed: int, format_id: str = "gen3randombattle") -> None:
        self.reset_seeds.append(seed)
        super().reset(seed=seed, format_id=format_id)


class MetadataPolicy:
    def __init__(
        self,
        *,
        policy_id: str = "root-puct-diagnostic",
        fallback: bool = False,
        include_elapsed: bool = True,
        value_gate_used: bool | None = None,
        root_opponent_action_policy: str | None = None,
        root_opponent_action_scenario_count: int | None = None,
        root_total_visits: int | None = None,
        root_effective_total_visits: int | None = None,
        leaf_rollout_rounds: int | None = None,
        leaf_rollout_opponent_policy: str | None = None,
        leaf_actual_rounds: dict[str, int] | None = None,
        leaf_evaluations: dict[str, int] | None = None,
    ) -> None:
        self.policy_id = policy_id
        self.fallback = fallback
        self.include_elapsed = include_elapsed
        self.value_gate_used = value_gate_used
        self.root_opponent_action_policy = root_opponent_action_policy
        self.root_opponent_action_scenario_count = root_opponent_action_scenario_count
        self.root_total_visits = root_total_visits
        self.root_effective_total_visits = root_effective_total_visits
        self.leaf_rollout_rounds = leaf_rollout_rounds
        self.leaf_rollout_opponent_policy = leaf_rollout_opponent_policy
        self.leaf_actual_rounds = leaf_actual_rounds
        self.leaf_evaluations = leaf_evaluations

    def select_action(self, observation: PokeZeroObservationV0, *, rng) -> PolicyDecision:
        if self.fallback:
            return PolicyDecision(
                action_index=0,
                policy_id=self.policy_id,
                metadata={
                    "policy_family": "root-puct-search",
                    "root_puct_fallback": True,
                    "root_puct_fallback_reason": "search failed: boom",
                },
            )
        metadata = {
            "policy_family": "root-puct-search",
            "root_puct_fallback": False,
            "root_puct_selection_mode": "puct",
            "root_puct_candidate_count": 3,
            "root_puct_selected_value": 0.5,
            "root_puct_selected_score": 0.75,
        }
        if self.include_elapsed:
            metadata["root_puct_elapsed_seconds"] = 0.25
        if self.value_gate_used is not None:
            metadata["root_puct_value_gate_used"] = self.value_gate_used
        if self.root_opponent_action_policy is not None:
            metadata["root_puct_opponent_action_policy"] = self.root_opponent_action_policy
        if self.root_opponent_action_scenario_count is not None:
            metadata["root_puct_opponent_action_scenario_count"] = self.root_opponent_action_scenario_count
        if self.root_total_visits is not None:
            metadata["root_puct_total_visits"] = self.root_total_visits
        if self.root_effective_total_visits is not None:
            metadata["root_puct_effective_total_visits"] = self.root_effective_total_visits
        if self.leaf_rollout_rounds is not None:
            metadata["root_puct_leaf_rollout_rounds"] = self.leaf_rollout_rounds
        if self.leaf_rollout_opponent_policy is not None:
            metadata["root_puct_leaf_rollout_opponent_policy"] = self.leaf_rollout_opponent_policy
        if self.leaf_actual_rounds is not None:
            metadata["root_puct_leaf_actual_rollout_rounds"] = dict(self.leaf_actual_rounds)
        if self.leaf_evaluations is not None:
            metadata["root_puct_leaf_evaluations"] = dict(self.leaf_evaluations)
        return PolicyDecision(
            action_index=0,
            policy_id=self.policy_id,
            metadata=metadata,
        )


def integration_config() -> LocalShowdownConfig | None:
    root = Path(os.environ.get("POKEZERO_SHOWDOWN_ROOT") or DEFAULT_SHOWDOWN_ROOT)
    if not (root / "dist" / "sim" / "index.js").exists():
        return None
    if shutil.which("node") is None:
        return None
    return LocalShowdownConfig(showdown_root=root, read_timeout_seconds=10.0)


class CollectionTest(unittest.TestCase):
    def test_current_peak_rss_mb_normalizes_platform_units(self) -> None:
        usage = type("Usage", (), {"ru_maxrss": 2 * 1024 * 1024})()

        with patch("resource.getrusage", return_value=usage):
            with patch("pokezero.collection.sys.platform", "darwin"):
                self.assertEqual(current_peak_rss_mb(), 2.0)
            with patch("pokezero.collection.sys.platform", "linux"):
                self.assertEqual(current_peak_rss_mb(), 2048.0)

    def test_trajectory_dict_round_trip_preserves_observation_and_terminal(self) -> None:
        original = trajectory()

        restored = trajectory_from_dict(trajectory_to_dict(original))

        self.assertEqual(restored.battle_id, original.battle_id)
        self.assertEqual(restored.terminal, original.terminal)
        self.assertEqual(restored.steps[0].observation.perspective.showdown_slot, "p1")
        self.assertEqual(restored.steps[0].action_probability, 0.5)

    def test_rollout_record_dict_round_trip(self) -> None:
        metrics = summarize_records([], elapsed_seconds=1.0)
        self.assertEqual(metrics.games, 0)
        record = collect_one_record_for_test()

        restored = rollout_record_from_dict(rollout_record_to_dict(record))

        self.assertEqual(restored.battle_id, record.battle_id)
        self.assertEqual(restored.policy_ids, record.policy_ids)
        self.assertEqual(restored.terminal, TerminalState(winner="p1", turn_count=1))
        self.assertEqual(len(restored.trajectory.steps), 2)

    def test_rollout_record_belief_provenance_round_trips_and_tolerates_legacy(self) -> None:
        from dataclasses import replace as dc_replace

        from pokezero.collection import distinct_belief_set_source_hashes, write_rollout_record

        record = collect_one_record_for_test()
        self.assertIsNone(record.belief_set_source_hash)
        # legacy payloads (no key) read back as None
        legacy_payload = rollout_record_to_dict(record)
        self.assertNotIn("belief_set_source_hash", legacy_payload)
        self.assertIsNone(rollout_record_from_dict(legacy_payload).belief_set_source_hash)
        # provenance round-trips when set
        stamped = dc_replace(record, belief_set_source_hash="abc123")
        payload = rollout_record_to_dict(stamped)
        self.assertEqual(payload["belief_set_source_hash"], "abc123")
        self.assertEqual(rollout_record_from_dict(payload).belief_set_source_hash, "abc123")
        # distinct-hash helper peeks first records only
        with tempfile.TemporaryDirectory() as temp_dir:
            stamped_path = Path(temp_dir) / "stamped.jsonl"
            legacy_path = Path(temp_dir) / "legacy.jsonl"
            with stamped_path.open("w", encoding="utf-8") as handle:
                write_rollout_record(handle, stamped)
            with legacy_path.open("w", encoding="utf-8") as handle:
                write_rollout_record(handle, record)
            self.assertEqual(distinct_belief_set_source_hashes([stamped_path]), ("abc123",))
            self.assertEqual(
                distinct_belief_set_source_hashes([stamped_path, legacy_path]),
                ("abc123", None),
            )

    def test_training_cache_metadata_records_belief_provenance(self) -> None:
        import json
        from dataclasses import replace as dc_replace

        from pokezero.collection import distinct_belief_set_source_hashes
        from pokezero.dataset import TrainingCacheBuilder

        try:
            import numpy  # noqa: F401
        except ImportError:
            self.skipTest("numpy is not installed in this environment.")

        record = dc_replace(collect_one_record_for_test(), belief_set_source_hash="cachehash1")
        builder = TrainingCacheBuilder()
        builder.add_record(record)
        with tempfile.TemporaryDirectory() as temp_dir:
            cache_path = Path(temp_dir) / "cache"
            builder.write(cache_path, cache_root=Path(temp_dir))
            metadata = json.loads((cache_path / "metadata.json").read_text())
            self.assertEqual(metadata["belief_set_source_hash"], "cachehash1")
            self.assertFalse(metadata["belief_set_source_mixed"])
            # the provenance helper reads cache directories via their metadata
            self.assertEqual(distinct_belief_set_source_hashes([cache_path]), ("cachehash1",))

    def test_distinct_hashes_survive_malformed_lines_and_detect_in_file_mixes(self) -> None:
        from pokezero.collection import distinct_belief_set_source_hashes

        with tempfile.TemporaryDirectory() as temp_dir:
            junk = Path(temp_dir) / "junk.jsonl"
            junk.write_text("[1, 2, 3]\n")  # valid JSON, not an object — must not raise
            mixed = Path(temp_dir) / "mixed.jsonl"
            mixed.write_text('{"belief_set_source_hash": "h1"}\n{"battle_id": "x"}\n')
            self.assertEqual(distinct_belief_set_source_hashes([junk]), (None,))
            self.assertEqual(distinct_belief_set_source_hashes([mixed]), ("h1", None))

    def test_collect_rollouts_writes_jsonl_and_metrics(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "rollouts.jsonl"

            with patch("pokezero.collection.current_peak_rss_mb", return_value=123.5):
                metrics = collect_rollouts(
                    output_path=output_path,
                    games=2,
                    env_factory=OneTurnEnv,
                    policies={"p1": RandomLegalPolicy(), "p2": RandomLegalPolicy()},
                    rollout_config=RolloutConfig(max_decision_rounds=5),
                    seed_start=10,
                )

            records = read_rollout_records(output_path)
            streamed_records = list(iter_rollout_records(output_path))
        self.assertEqual(metrics.games, 2)
        self.assertEqual(metrics.p1_wins, 2)
        self.assertEqual(metrics.total_decision_rounds, 2)
        self.assertEqual(metrics.peak_rss_mb, 123.5)
        self.assertEqual(metrics.to_dict()["peak_rss_mb"], 123.5)
        self.assertEqual([record.seed for record in records], [10, 11])
        self.assertEqual([record.seed for record in streamed_records], [10, 11])
        self.assertEqual([record.battle_id for record in records], ["rollout-10", "rollout-11"])

    def test_collect_rollouts_reuses_one_warm_env_across_games(self) -> None:
        # Warm pool: the collector creates ONE env and resets it per game (reusing the bridge
        # process), instead of spawning a fresh env (node process) per game.
        instances: list[SeedRecordingEnv] = []

        def factory() -> SeedRecordingEnv:
            env = SeedRecordingEnv([])
            instances.append(env)
            return env

        with tempfile.TemporaryDirectory() as temp_dir:
            collect_rollouts(
                output_path=Path(temp_dir) / "rollouts.jsonl",
                games=3,
                env_factory=factory,
                policies={"p1": RandomLegalPolicy(), "p2": RandomLegalPolicy()},
                rollout_config=RolloutConfig(max_decision_rounds=5),
                seed_start=10,
            )

        self.assertEqual(len(instances), 1)  # one env reused, not three fresh spawns
        self.assertEqual(instances[0].reset_seeds, [10, 11, 12])  # reset once per game
        self.assertTrue(instances[0].closed)  # closed when collection finishes

    def test_collect_training_cache_writes_compact_cache_and_metrics(self) -> None:
        self._require_numpy()
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "cache"

            with patch("pokezero.collection.current_peak_rss_mb", return_value=123.5):
                metrics, cache = collect_training_cache(
                    output_path=output_path,
                    games=2,
                    env_factory=OneTurnEnv,
                    policies={"p1": RandomLegalPolicy(), "p2": RandomLegalPolicy()},
                    rollout_config=RolloutConfig(max_decision_rounds=5),
                    dataset_config=TrajectoryDatasetConfig(window_size=2),
                    seed_start=10,
                )
            batches = list(iter_training_batches(output_path, batch_size=2, config=TrajectoryDatasetConfig(window_size=2)))

            self.assertEqual(metrics.games, 2)
            self.assertEqual(metrics.p1_wins, 2)
            self.assertEqual(metrics.total_decision_rounds, 2)
            self.assertEqual(metrics.peak_rss_mb, 123.5)
            self.assertTrue(is_training_cache_path(cache.path))
            self.assertEqual(cache.record_count, 2)
            self.assertEqual(cache.example_count, 4)
            self.assertEqual([batch.batch_size for batch in batches], [2, 2])

    def test_collect_training_cache_rejects_storage_cap(self) -> None:
        self._require_numpy()
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "cache"

            with self.assertRaisesRegex(ValueError, "storage cap"):
                collect_training_cache(
                    output_path=output_path,
                    games=1,
                    env_factory=OneTurnEnv,
                    policies={"p1": RandomLegalPolicy(), "p2": RandomLegalPolicy()},
                    rollout_config=RolloutConfig(max_decision_rounds=5),
                    dataset_config=TrajectoryDatasetConfig(window_size=2),
                    seed_start=10,
                    max_cache_root_bytes=1,
                    cache_root=Path(temp_dir),
                )

            self.assertFalse(output_path.exists())

    def test_reusable_env_pool_reuses_per_thread_and_closes(self) -> None:
        instances: list[OneTurnEnv] = []

        def factory() -> OneTurnEnv:
            env = OneTurnEnv()
            instances.append(env)
            return env

        pool = ReusableEnvPool(factory)
        first, second = pool.get(), pool.get()
        self.assertIs(first, second)  # same thread -> same env reused
        self.assertEqual(len(instances), 1)
        pool.close_all()
        self.assertTrue(first.closed)

    def test_benchmark_rollouts_runs_default_matchups_without_writing_trajectories(self) -> None:
        with patch("pokezero.collection.current_peak_rss_mb", side_effect=(101.0, 102.0, 103.0, 104.0)):
            report = benchmark_rollouts(
                games=2,
                env_factory=OneTurnEnv,
                rollout_config=RolloutConfig(max_decision_rounds=5),
                seed_start=20,
            )

        self.assertEqual(report.games_per_matchup, 2)
        self.assertEqual(report.total_games, 8)
        self.assertEqual(report.average_decision_rounds, 1.0)
        self.assertEqual(report.to_dict()["average_decision_rounds"], 1.0)
        self.assertEqual(report.peak_rss_mb, 104.0)
        self.assertEqual(report.to_dict()["peak_rss_mb"], 104.0)
        self.assertEqual(
            [result.label for result in report.matchups],
            [matchup.label for matchup in default_benchmark_matchups()],
        )
        for result in report.matchups:
            self.assertEqual(result.seed_start, 20)
            self.assertEqual(result.metrics.games, 2)
            self.assertEqual(result.metrics.p1_wins, 2)
            self.assertEqual(result.metrics.total_decision_rounds, 2)

    def test_benchmark_rollouts_summarizes_policy_decision_metadata(self) -> None:
        report = benchmark_rollouts(
            games=2,
            env_factory=OneTurnEnv,
            rollout_config=RolloutConfig(max_decision_rounds=5),
            seed_start=20,
            matchups=(
                BenchmarkMatchup(
                    "root-puct vs random",
                    MetadataPolicy(
                        value_gate_used=True,
                        root_opponent_action_policy="benchmark",
                        root_opponent_action_scenario_count=2,
                        root_total_visits=11,
                        root_effective_total_visits=7,
                        leaf_rollout_rounds=2,
                        leaf_rollout_opponent_policy="benchmark",
                        leaf_actual_rounds={"0": 1, "2": 2},
                        leaf_evaluations={"rollout_terminal": 2, "rollout_value_fn": 1},
                    ),
                    RandomLegalPolicy(),
                ),
            ),
        )

        summary = report.to_dict()["matchups"][0]["metrics"]["policy_decision_summary"]
        self.assertEqual(summary["root-puct-diagnostic"]["decisions"], 2)
        self.assertEqual(summary["root-puct-diagnostic"]["root_puct_searches"], 2)
        self.assertEqual(summary["root-puct-diagnostic"]["root_puct_fallbacks"], 0)
        self.assertEqual(summary["root-puct-diagnostic"]["root_puct_total_visits"], 22)
        self.assertEqual(summary["root-puct-diagnostic"]["root_puct_effective_total_visits"], 14)
        self.assertEqual(summary["root-puct-diagnostic"]["root_puct_average_candidate_count"], 3.0)
        self.assertEqual(summary["root-puct-diagnostic"]["root_puct_average_elapsed_seconds"], 0.25)
        self.assertEqual(summary["root-puct-diagnostic"]["root_puct_average_selected_value"], 0.5)
        self.assertEqual(summary["root-puct-diagnostic"]["root_puct_average_selected_score"], 0.75)
        self.assertEqual(summary["root-puct-diagnostic"]["root_puct_value_gate_checks"], 2)
        self.assertEqual(summary["root-puct-diagnostic"]["root_puct_value_gate_uses"], 2)
        self.assertEqual(summary["root-puct-diagnostic"]["root_puct_selection_modes"], {"puct": 2})
        self.assertEqual(
            summary["root-puct-diagnostic"]["root_puct_opponent_action_policies"],
            {"benchmark": 2},
        )
        self.assertEqual(
            summary["root-puct-diagnostic"]["root_puct_opponent_action_scenario_counts"],
            {"2": 2},
        )
        self.assertEqual(summary["root-puct-diagnostic"]["root_puct_leaf_rollout_rounds"], {"2": 2})
        self.assertEqual(
            summary["root-puct-diagnostic"]["root_puct_leaf_rollout_opponent_policies"],
            {"benchmark": 2},
        )
        self.assertEqual(summary["root-puct-diagnostic"]["root_puct_leaf_actual_rollout_rounds"], {"0": 2, "2": 4})
        self.assertEqual(
            summary["root-puct-diagnostic"]["root_puct_leaf_evaluations"],
            {"rollout_terminal": 4, "rollout_value_fn": 2},
        )
        self.assertEqual(summary["random-legal"]["decisions"], 2)
        self.assertNotIn("root_puct_searches", summary["random-legal"])

    def test_benchmark_rollouts_summarizes_root_puct_fallback_metadata(self) -> None:
        report = benchmark_rollouts(
            games=2,
            env_factory=OneTurnEnv,
            rollout_config=RolloutConfig(max_decision_rounds=5),
            seed_start=20,
            matchups=(
                BenchmarkMatchup(
                    "fallback-root-puct vs random",
                    MetadataPolicy(policy_id="root-puct-fallback", fallback=True),
                    RandomLegalPolicy(),
                ),
            ),
        )

        summary = report.to_dict()["matchups"][0]["metrics"]["policy_decision_summary"]
        self.assertEqual(summary["root-puct-fallback"]["decisions"], 2)
        self.assertEqual(summary["root-puct-fallback"]["root_puct_searches"], 0)
        self.assertEqual(summary["root-puct-fallback"]["root_puct_fallbacks"], 2)
        self.assertEqual(
            summary["root-puct-fallback"]["root_puct_fallback_reasons"],
            {"search failed: boom": 2},
        )
        self.assertEqual(
            summary["root-puct-fallback"]["root_puct_fallback_categories"],
            {"search_failed": 2},
        )

    def test_benchmark_rollouts_summarizes_root_puct_value_gate_checks_without_uses(self) -> None:
        report = benchmark_rollouts(
            games=2,
            env_factory=OneTurnEnv,
            rollout_config=RolloutConfig(max_decision_rounds=5),
            seed_start=20,
            matchups=(
                BenchmarkMatchup(
                    "root-puct-gated vs random",
                    MetadataPolicy(policy_id="root-puct-gated", value_gate_used=False),
                    RandomLegalPolicy(),
                ),
            ),
        )

        summary = report.to_dict()["matchups"][0]["metrics"]["policy_decision_summary"]
        self.assertEqual(summary["root-puct-gated"]["root_puct_value_gate_checks"], 2)
        self.assertEqual(summary["root-puct-gated"]["root_puct_value_gate_uses"], 0)

    def test_benchmark_rollouts_omits_missing_root_puct_elapsed_average(self) -> None:
        report = benchmark_rollouts(
            games=2,
            env_factory=OneTurnEnv,
            rollout_config=RolloutConfig(max_decision_rounds=5),
            seed_start=20,
            matchups=(
                BenchmarkMatchup(
                    "root-puct-no-elapsed vs random",
                    MetadataPolicy(policy_id="root-puct-no-elapsed", include_elapsed=False),
                    RandomLegalPolicy(),
                ),
            ),
        )

        summary = report.to_dict()["matchups"][0]["metrics"]["policy_decision_summary"]
        self.assertEqual(summary["root-puct-no-elapsed"]["root_puct_searches"], 2)
        self.assertNotIn("root_puct_average_elapsed_seconds", summary["root-puct-no-elapsed"])

    def test_print_benchmark_report_includes_root_puct_diagnostics(self) -> None:
        report = benchmark_rollouts(
            games=1,
            env_factory=OneTurnEnv,
            rollout_config=RolloutConfig(max_decision_rounds=5),
            seed_start=20,
            matchups=(
                BenchmarkMatchup(
                    "root-puct vs random",
                    MetadataPolicy(
                        leaf_rollout_rounds=2,
                        root_opponent_action_policy="benchmark",
                        root_opponent_action_scenario_count=2,
                        leaf_rollout_opponent_policy="benchmark",
                        leaf_actual_rounds={"1": 3},
                        leaf_evaluations={"rollout_terminal": 2, "rollout_value_fn": 1},
                    ),
                    RandomLegalPolicy(),
                ),
                BenchmarkMatchup(
                    "fallback-root-puct vs random",
                    MetadataPolicy(policy_id="root-puct-fallback", fallback=True),
                    RandomLegalPolicy(),
                ),
            ),
        )

        stdout = io.StringIO()
        with patch("sys.stdout", stdout):
            print_benchmark_report(report)

        output = stdout.getvalue()
        self.assertIn("root-puct diagnostics:", output)
        self.assertIn("gate", output)
        self.assertIn("root-puct-fallback", output)
        self.assertIn("selection_modes:", output)
        self.assertIn("opponent_action_policies:", output)
        self.assertIn("benchmark=1", output)
        self.assertIn("opponent_action_scenario_counts:", output)
        self.assertIn("2=1", output)
        self.assertIn("leaf_rollouts_configured:", output)
        self.assertIn("leaf_rollout_opponents:", output)
        self.assertIn("benchmark=1", output)
        self.assertIn("leaf_rollouts_actual:", output)
        self.assertIn("leaf_evaluations:", output)
        self.assertIn("2=1", output)
        self.assertIn("1=3", output)
        self.assertIn("rollout_terminal=2", output)
        self.assertIn("fallback_reasons:", output)
        self.assertIn("search failed: boom=1", output)

    def test_benchmark_rollouts_reuses_seed_range_for_each_matchup(self) -> None:
        reset_seeds = []
        matchups = (
            BenchmarkMatchup("a", RandomLegalPolicy(), RandomLegalPolicy()),
            BenchmarkMatchup("b", RandomLegalPolicy(), RandomLegalPolicy()),
        )

        benchmark_rollouts(
            games=2,
            env_factory=lambda: SeedRecordingEnv(reset_seeds),
            rollout_config=RolloutConfig(max_decision_rounds=5),
            seed_start=7,
            matchups=matchups,
        )

        self.assertEqual(reset_seeds, [7, 8, 7, 8])

    def test_replay_prefix_counts_evenly_samples_valid_branch_prefixes(self) -> None:
        self.assertEqual(replay_prefix_counts(0, prefixes_per_game=5), ())
        self.assertEqual(replay_prefix_counts(1, prefixes_per_game=5), (0,))
        self.assertEqual(replay_prefix_counts(10, prefixes_per_game=5), (0, 2, 4, 7, 9))
        self.assertEqual(replay_prefix_counts(4, prefixes_per_game=10), (0, 1, 2, 3))

    def test_benchmark_replay_prefixes_replays_sampled_prefixes_on_warm_env(self) -> None:
        env = OneTurnEnv()

        report = benchmark_replay_prefixes(
            env_factory=lambda: env,
            policies={"p1": RandomLegalPolicy(), "p2": RandomLegalPolicy()},
            rollout_config=RolloutConfig(max_decision_rounds=3),
            games=2,
            prefixes_per_game=4,
            seed_start=20,
        )

        self.assertTrue(env.closed)
        self.assertEqual(report.games, 2)
        self.assertEqual(report.source_decision_rounds, (1, 1))
        self.assertEqual(report.total_prefixes, 2)
        self.assertEqual([timing.seed for timing in report.timings], [20, 21])
        self.assertEqual([timing.decision_round_count for timing in report.timings], [0, 0])
        self.assertGreaterEqual(report.average_replay_seconds, 0.0)

    def test_policy_benchmark_matchups_builds_mirrored_shared_opponent_matchups(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            first_checkpoint = Path(temp_dir) / "first.json"
            second_checkpoint = Path(temp_dir) / "second.json"
            save_linear_model(
                first_checkpoint,
                LinearPolicyModel.initialized(
                    feature_count=8,
                    window_size=1,
                    policy_id="first-linear",
                ),
            )
            save_linear_model(
                second_checkpoint,
                LinearPolicyModel.initialized(
                    feature_count=8,
                    window_size=1,
                    policy_id="second-linear",
                ),
            )

            matchups = policy_benchmark_matchups(
                policy_specs=(f"linear:{first_checkpoint}", f"linear:{second_checkpoint}"),
                opponent_policy_specs=("random-legal",),
                include_policy_head_to_head=True,
            )

        self.assertEqual(
            [matchup.label for matchup in matchups],
            [
                "first-linear vs random-legal",
                "random-legal vs first-linear",
                "second-linear vs random-legal",
                "random-legal vs second-linear",
                "first-linear vs second-linear",
                "second-linear vs first-linear",
            ],
        )
        self.assertEqual(
            [(matchup.p1_policy.policy_id, matchup.p2_policy.policy_id) for matchup in matchups],
            [
                ("first-linear", "random-legal"),
                ("random-legal", "first-linear"),
                ("second-linear", "random-legal"),
                ("random-legal", "second-linear"),
                ("first-linear", "second-linear"),
                ("second-linear", "first-linear"),
            ],
        )

    def test_policy_benchmark_matchups_rejects_duplicate_policy_ids(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            first_checkpoint = Path(temp_dir) / "first.json"
            second_checkpoint = Path(temp_dir) / "second.json"
            for checkpoint in (first_checkpoint, second_checkpoint):
                save_linear_model(
                    checkpoint,
                    LinearPolicyModel.initialized(
                        feature_count=8,
                        window_size=1,
                        policy_id="duplicate-linear",
                    ),
                )

            with self.assertRaisesRegex(ValueError, "duplicate candidate policy id: duplicate-linear"):
                policy_benchmark_matchups(
                    policy_specs=(f"linear:{first_checkpoint}", f"linear:{second_checkpoint}"),
                    opponent_policy_specs=("random-legal",),
                )

    def test_policy_benchmark_matchups_rejects_candidate_opponent_policy_id_collision(self) -> None:
        with self.assertRaisesRegex(ValueError, "candidate and opponent policy ids must be distinct"):
            policy_benchmark_matchups(
                policy_specs=("random-legal",),
                opponent_policy_specs=("random-legal", "simple-legal"),
            )

    def test_policy_benchmark_matchups_rejects_single_policy_head_to_head(self) -> None:
        with self.assertRaisesRegex(ValueError, "requires at least two distinct candidate policies"):
            policy_benchmark_matchups(
                policy_specs=("simple-legal",),
                opponent_policy_specs=("random-legal",),
                include_policy_head_to_head=True,
            )

    def test_benchmark_head_to_head_aggregates_mirror_pair(self) -> None:
        rows = (
            BenchmarkMatchupResult(
                label="simple-legal vs random-legal",
                p1_policy_id="simple-legal",
                p2_policy_id="random-legal",
                seed_start=1,
                metrics=CollectionMetrics(
                    games=10,
                    elapsed_seconds=1.0,
                    total_decision_rounds=20,
                    total_simulator_turns=18,
                    p1_wins=6,
                    p2_wins=3,
                    ties=1,
                    capped_games=0,
                ),
            ),
            BenchmarkMatchupResult(
                label="random-legal vs simple-legal",
                p1_policy_id="random-legal",
                p2_policy_id="simple-legal",
                seed_start=1,
                metrics=CollectionMetrics(
                    games=10,
                    elapsed_seconds=1.0,
                    total_decision_rounds=20,
                    total_simulator_turns=18,
                    p1_wins=4,
                    p2_wins=5,
                    ties=0,
                    capped_games=1,
                ),
            ),
            BenchmarkMatchupResult(
                label="random-legal vs random-legal",
                p1_policy_id="random-legal",
                p2_policy_id="random-legal",
                seed_start=1,
                metrics=CollectionMetrics(
                    games=10,
                    elapsed_seconds=1.0,
                    total_decision_rounds=20,
                    total_simulator_turns=18,
                    p1_wins=5,
                    p2_wins=5,
                    ties=0,
                    capped_games=0,
                ),
            ),
        )

        head_to_heads = aggregate_benchmark_head_to_heads(rows)

        self.assertEqual(len(head_to_heads), 1)
        result = head_to_heads[0]
        self.assertEqual(result.label, "simple-legal vs random-legal")
        self.assertEqual(result.games, 20)
        self.assertEqual(result.first_policy_wins, 11)
        self.assertEqual(result.second_policy_wins, 7)
        self.assertEqual(result.ties, 1)
        self.assertEqual(result.capped_games, 1)
        self.assertAlmostEqual(result.first_policy_win_rate, 0.55)

    def test_collect_rollouts_non_append_preserves_existing_file_on_failure(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "rollouts.jsonl"
            output_path.write_text("existing\n", encoding="utf-8")

            with self.assertRaisesRegex(RuntimeError, "boom"):
                collect_rollouts(
                    output_path=output_path,
                    games=1,
                    env_factory=ResetFailingEnv,
                    policies={"p1": RandomLegalPolicy(), "p2": RandomLegalPolicy()},
                    rollout_config=RolloutConfig(max_decision_rounds=5),
                )

            self.assertEqual(output_path.read_text(encoding="utf-8"), "existing\n")

    def test_summarize_records_requires_explicit_elapsed_seconds(self) -> None:
        records = [collect_one_record_for_test()]

        metrics = summarize_records(records, elapsed_seconds=2.0)

        self.assertEqual(metrics.games, 1)
        self.assertEqual(metrics.elapsed_seconds, 2.0)
        self.assertEqual(metrics.games_per_second, 0.5)

    def test_policy_from_name_rejects_unknown_policy(self) -> None:
        self.assertEqual(policy_from_name("random-legal").policy_id, "random-legal")
        self.assertEqual(policy_from_name("simple-legal").policy_id, "simple-legal")
        with self.assertRaisesRegex(ValueError, "Unsupported policy"):
            policy_from_name("unknown")

    def test_policy_from_spec_loads_scripted_teacher_options(self) -> None:
        policy = policy_from_spec(
            "scripted-teacher?showdown_root=/tmp/showdown&switch_margin=3&poor_move_threshold=20"
            "&team_status_cure_score=70&statused_switch_penalty=12&low_hp_switch_bonus=44"
            "&tie_breaker=first&allow_fallback=true&allow_unknown_moves=true"
        )

        self.assertIsInstance(policy, ScriptedTeacherPolicy)
        self.assertEqual(policy.showdown_root, Path("/tmp/showdown"))
        self.assertEqual(policy.switch_margin, 3.0)
        self.assertEqual(policy.poor_move_threshold, 20.0)
        self.assertEqual(policy.team_status_cure_score, 70.0)
        self.assertEqual(policy.statused_switch_penalty, 12.0)
        self.assertEqual(policy.low_hp_switch_bonus, 44.0)
        self.assertEqual(policy.tie_breaker, "first")
        self.assertTrue(policy.allow_fallback)
        self.assertTrue(policy.allow_unknown_moves)

    def test_policy_spec_with_showdown_root_injects_scripted_teacher_root(self) -> None:
        self.assertEqual(
            policy_spec_with_showdown_root("scripted-teacher", Path("/tmp/showdown")),
            "scripted-teacher?showdown_root=%2Ftmp%2Fshowdown",
        )
        self.assertEqual(
            policy_spec_with_showdown_root("random-legal", Path("/tmp/showdown")),
            "random-legal",
        )

    def test_policy_from_spec_loads_linear_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            checkpoint_path = Path(temp_dir) / "linear.json"
            save_linear_model(
                checkpoint_path,
                LinearPolicyModel.initialized(
                    feature_count=8,
                    window_size=1,
                    policy_id="linear-test",
                ),
            )

            policy = policy_from_spec(f"linear:{checkpoint_path}")

        self.assertIsInstance(policy, LinearSoftmaxPolicy)
        self.assertEqual(policy.policy_id, "linear-test")
        self.assertFalse(policy.deterministic)
        self.assertEqual(policy.exploration_epsilon, 0.0)
        self.assertEqual(policy.sampling_temperature, 1.0)

    def test_policy_from_spec_loads_linear_checkpoint_with_sampling_options(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            checkpoint_path = Path(temp_dir) / "linear.json"
            save_linear_model(
                checkpoint_path,
                LinearPolicyModel.initialized(
                    feature_count=8,
                    window_size=1,
                    policy_id="linear-test",
                ),
            )

            policy = policy_from_spec(f"linear:{checkpoint_path}?deterministic=true&epsilon=0.25&temperature=2.5")

        self.assertIsInstance(policy, LinearSoftmaxPolicy)
        self.assertTrue(policy.deterministic)
        self.assertEqual(policy.exploration_epsilon, 0.25)
        self.assertEqual(policy.sampling_temperature, 2.5)

    def test_policy_from_spec_accepts_neural_checkpoint_specs(self) -> None:
        if torch_available():
            self.skipTest("PyTorch checkpoint fixture is not available in this test module.")
        with self.assertRaisesRegex(TorchUnavailableError, "pip install -e"):
            policy_from_spec("neural:/tmp/model.pt?deterministic=true&epsilon=0.1&temperature=0.9&family_gated=true&device=cpu")

    def test_policy_from_spec_rejects_empty_neural_checkpoint_path(self) -> None:
        with self.assertRaisesRegex(ValueError, "after 'neural:'"):
            policy_from_spec("neural:")

    def test_policy_from_name_mentions_neural_policy_specs(self) -> None:
        with self.assertRaisesRegex(ValueError, "neural:/path/to/checkpoint.pt"):
            policy_from_name("unknown")

    def test_policy_factory_from_spec_reuses_immutable_linear_model(self) -> None:
        model = LinearPolicyModel.initialized(
            feature_count=8,
            window_size=1,
            policy_id="linear-test",
        )
        with patch("pokezero.linear_policy.load_linear_model", return_value=model) as load:
            factory = policy_factory_from_spec("linear:/tmp/linear.json?deterministic=true")
            first = factory()
            second = factory()

        self.assertEqual(load.call_count, 1)
        self.assertIsInstance(first, LinearSoftmaxPolicy)
        self.assertIsInstance(second, LinearSoftmaxPolicy)
        self.assertIs(first.model, model)
        self.assertIs(second.model, model)
        self.assertIsNot(first, second)
        self.assertTrue(first.deterministic)

    def test_linear_policy_factory_from_model_spec_preserves_options_without_loading(self) -> None:
        model = LinearPolicyModel.initialized(
            feature_count=8,
            window_size=1,
            policy_id="linear-test",
        )

        factory = linear_policy_factory_from_model_spec(
            "linear:/tmp/linear.json?sample=true&epsilon=0.25&temperature=2.5",
            model,
        )
        policy = factory()

        self.assertIsInstance(policy, LinearSoftmaxPolicy)
        self.assertIs(policy.model, model)
        self.assertFalse(policy.deterministic)
        self.assertEqual(policy.exploration_epsilon, 0.25)
        self.assertEqual(policy.sampling_temperature, 2.5)

    def test_linear_policy_factory_from_model_spec_rejects_empty_checkpoint_path(self) -> None:
        model = LinearPolicyModel.initialized(
            feature_count=8,
            window_size=1,
            policy_id="linear-test",
        )

        with self.assertRaisesRegex(ValueError, "checkpoint path"):
            linear_policy_factory_from_model_spec("linear:?sample=true", model)

    def test_policy_from_spec_rejects_empty_linear_checkpoint_path(self) -> None:
        with self.assertRaisesRegex(ValueError, "checkpoint path"):
            policy_from_spec("linear:")

    def test_policy_from_spec_rejects_conflicting_linear_sampling_options(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            checkpoint_path = Path(temp_dir) / "linear.json"
            save_linear_model(
                checkpoint_path,
                LinearPolicyModel.initialized(
                    feature_count=8,
                    window_size=1,
                    policy_id="linear-test",
                ),
            )

            with self.assertRaisesRegex(ValueError, "conflict"):
                policy_from_spec(f"linear:{checkpoint_path}?sample=true&deterministic=true")

    def test_rollout_cli_collect_wires_arguments_and_prints_metrics(self) -> None:
        fake_metrics = CollectionMetrics(
            games=1,
            elapsed_seconds=2.0,
            total_decision_rounds=4,
            total_simulator_turns=3,
            p1_wins=1,
            p2_wins=0,
            ties=0,
            capped_games=0,
            peak_rss_mb=55.5,
        )
        with patch("pokezero.rollout_cli.collect_rollouts", return_value=fake_metrics) as collect:
            with patch("sys.stdout", new_callable=io.StringIO) as stdout:
                exit_code = rollout_cli_main(
                    [
                        "collect",
                        "--games",
                        "1",
                        "--out",
                        "runs/test.jsonl",
                        "--seed-start",
                        "50",
                        "--max-decision-rounds",
                        "7",
                        "--p1-policy",
                        "simple-legal",
                        "--p2-policy",
                        "random-legal",
                    ]
                )

        self.assertEqual(exit_code, 0)
        kwargs = collect.call_args.kwargs
        self.assertEqual(kwargs["games"], 1)
        self.assertEqual(kwargs["seed_start"], 50)
        self.assertEqual(kwargs["rollout_config"].max_decision_rounds, 7)
        self.assertEqual(kwargs["policies"]["p1"].policy_id, "simple-legal")
        self.assertIn("games_per_second: 0.500", stdout.getvalue())
        self.assertIn("peak_rss_mb: 55.50", stdout.getvalue())

    def test_rollout_cli_collect_training_cache_wires_arguments_and_prints_cache_summary(self) -> None:
        fake_metrics = CollectionMetrics(
            games=1,
            elapsed_seconds=2.0,
            total_decision_rounds=4,
            total_simulator_turns=3,
            p1_wins=1,
            p2_wins=0,
            ties=0,
            capped_games=0,
            peak_rss_mb=55.5,
        )
        fake_cache = TrainingCacheSummary(path=Path("runs/cache"), record_count=1, example_count=4, byte_size=512)
        with patch("pokezero.rollout_cli.collect_training_cache", return_value=(fake_metrics, fake_cache)) as collect:
            with patch("sys.stdout", new_callable=io.StringIO) as stdout:
                exit_code = rollout_cli_main(
                    [
                        "collect-training-cache",
                        "--games",
                        "1",
                        "--out",
                        "runs/cache",
                        "--seed-start",
                        "50",
                        "--max-decision-rounds",
                        "7",
                        "--p1-policy",
                        "simple-legal",
                        "--p2-policy",
                        "random-legal",
                        "--window-size",
                        "3",
                        "--discount",
                        "0.9",
                        "--ppo-target-mode",
                        "gae",
                        "--gae-lambda",
                        "0.7",
                    ]
                )

        self.assertEqual(exit_code, 0)
        kwargs = collect.call_args.kwargs
        self.assertEqual(kwargs["games"], 1)
        self.assertEqual(kwargs["seed_start"], 50)
        self.assertEqual(kwargs["rollout_config"].max_decision_rounds, 7)
        self.assertEqual(kwargs["policies"]["p1"].policy_id, "simple-legal")
        self.assertEqual(kwargs["dataset_config"].window_size, 3)
        self.assertEqual(kwargs["dataset_config"].discount, 0.9)
        self.assertEqual(kwargs["dataset_config"].ppo_target_mode, "gae")
        self.assertEqual(kwargs["dataset_config"].gae_lambda, 0.7)
        self.assertEqual(kwargs["max_cache_root_bytes"], 50 * 1024 * 1024 * 1024)
        self.assertEqual(kwargs["cache_root"], Path("runs"))
        self.assertIn("training_cache: runs/cache", stdout.getvalue())
        self.assertIn("training_cache_examples: 4", stdout.getvalue())

    def test_rollout_cli_collect_selfplay_training_cache_wires_arguments_and_prints_cache_summary(self) -> None:
        fake_metrics = CollectionMetrics(
            games=2,
            elapsed_seconds=4.0,
            total_decision_rounds=8,
            total_simulator_turns=6,
            p1_wins=1,
            p2_wins=1,
            ties=0,
            capped_games=0,
            peak_rss_mb=66.0,
        )

        def fake_collect(**kwargs):
            kwargs["training_cache_paths_out"].extend([Path("runs/cache/cache-00001"), Path("runs/cache/cache-00002")])
            return fake_metrics

        with (
            patch("pokezero.rollout_cli.collect_selfplay_rollouts", side_effect=fake_collect) as collect,
            patch("pokezero.rollout_cli.training_cache_paths_byte_size", return_value=1024),
            patch("sys.stdout", new_callable=io.StringIO) as stdout,
        ):
            exit_code = rollout_cli_main(
                [
                    "collect-selfplay-training-cache",
                    "--games",
                    "2",
                    "--out",
                    "runs/cache",
                    "--seed-start",
                    "50",
                    "--max-decision-rounds",
                    "7",
                    "--current-policy",
                    "simple-legal",
                    "--opponent-policy",
                    "random-legal",
                    "--workers",
                    "3",
                    "--chunk-games",
                    "1",
                    "--window-size",
                    "3",
                    "--discount",
                    "0.9",
                    "--ppo-target-mode",
                    "gae",
                    "--gae-lambda",
                    "0.7",
                ]
            )

        self.assertEqual(exit_code, 0)
        kwargs = collect.call_args.kwargs
        self.assertIsNone(kwargs["output_path"])
        self.assertIsNone(kwargs["training_output_path"])
        self.assertEqual(kwargs["training_cache_output_path"], Path("runs/cache"))
        self.assertEqual(kwargs["training_cache_chunk_games"], 1)
        self.assertEqual(kwargs["training_cache_root"], Path("runs"))
        self.assertEqual(kwargs["games"], 2)
        self.assertEqual(kwargs["seed_start"], 50)
        self.assertEqual(kwargs["worker_count"], 3)
        self.assertEqual(kwargs["rollout_config"].max_decision_rounds, 7)
        self.assertEqual(kwargs["current_policy_spec"], "simple-legal")
        self.assertEqual(kwargs["opponent_policy_specs"], ("random-legal",))
        self.assertEqual(kwargs["training_cache_dataset_config"].window_size, 3)
        self.assertEqual(kwargs["training_cache_dataset_config"].discount, 0.9)
        self.assertEqual(kwargs["training_cache_dataset_config"].ppo_target_mode, "gae")
        self.assertEqual(kwargs["training_cache_dataset_config"].gae_lambda, 0.7)
        self.assertEqual(kwargs["training_cache_max_root_bytes"], 50 * 1024 * 1024 * 1024)
        self.assertIn("training_cache: runs/cache/cache-00001", stdout.getvalue())
        self.assertIn("training_cache_count: 2", stdout.getvalue())
        self.assertIn("training_cache_bytes: 1024", stdout.getvalue())

    def test_rollout_cli_collect_selfplay_training_cache_defaults_to_mirror_current_policy(self) -> None:
        fake_metrics = CollectionMetrics(
            games=1,
            elapsed_seconds=2.0,
            total_decision_rounds=4,
            total_simulator_turns=3,
            p1_wins=1,
            p2_wins=0,
            ties=0,
            capped_games=0,
        )
        with (
            patch("pokezero.rollout_cli.collect_selfplay_rollouts", return_value=fake_metrics) as collect,
            patch("sys.stdout", new_callable=io.StringIO),
        ):
            exit_code = rollout_cli_main(
                [
                    "collect-selfplay-training-cache",
                    "--games",
                    "1",
                    "--out",
                    "runs/cache",
                    "--current-policy",
                    "simple-legal",
                ]
            )

        self.assertEqual(exit_code, 0)
        self.assertEqual(collect.call_args.kwargs["current_policy_spec"], "simple-legal")
        self.assertEqual(collect.call_args.kwargs["opponent_policy_specs"], ("simple-legal",))

    def test_rollout_cli_collect_loads_linear_policy_spec(self) -> None:
        fake_metrics = CollectionMetrics(
            games=1,
            elapsed_seconds=2.0,
            total_decision_rounds=4,
            total_simulator_turns=3,
            p1_wins=1,
            p2_wins=0,
            ties=0,
            capped_games=0,
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            checkpoint_path = Path(temp_dir) / "linear.json"
            save_linear_model(
                checkpoint_path,
                LinearPolicyModel.initialized(
                    feature_count=8,
                    window_size=1,
                    policy_id="linear-cli-test",
                ),
            )
            with patch("pokezero.rollout_cli.collect_rollouts", return_value=fake_metrics) as collect:
                with patch("sys.stdout", new_callable=io.StringIO):
                    exit_code = rollout_cli_main(
                        [
                            "collect",
                            "--games",
                            "1",
                            "--out",
                            str(Path(temp_dir) / "rollouts.jsonl"),
                            "--p1-policy",
                            f"linear:{checkpoint_path}",
                            "--p2-policy",
                            "random-legal",
                        ]
                    )

        self.assertEqual(exit_code, 0)
        kwargs = collect.call_args.kwargs
        self.assertEqual(kwargs["policies"]["p1"].policy_id, "linear-cli-test")
        self.assertEqual(kwargs["policies"]["p2"].policy_id, "random-legal")

    def test_rollout_cli_benchmark_wires_arguments_and_prints_report(self) -> None:
        fake_report = BenchmarkReport(
            format_id="gen3randombattle",
            max_decision_rounds=7,
            games_per_matchup=3,
            matchups=(
                BenchmarkMatchupResult(
                    label="random-legal vs random-legal",
                    p1_policy_id="random-legal",
                    p2_policy_id="random-legal",
                    seed_start=50,
                    metrics=CollectionMetrics(
                        games=3,
                        elapsed_seconds=2.0,
                        total_decision_rounds=12,
                        total_simulator_turns=9,
                        p1_wins=1,
                        p2_wins=2,
                        ties=0,
                        capped_games=0,
                        peak_rss_mb=66.25,
                    ),
                ),
            ),
        )
        with patch("pokezero.rollout_cli.benchmark_rollouts", return_value=fake_report) as benchmark:
            with patch("sys.stdout", new_callable=io.StringIO) as stdout:
                exit_code = rollout_cli_main(
                    [
                        "benchmark",
                        "--games",
                        "3",
                        "--showdown-root",
                        "/tmp/showdown",
                        "--seed-start",
                        "50",
                        "--max-decision-rounds",
                        "7",
                    ]
                )

        self.assertEqual(exit_code, 0)
        kwargs = benchmark.call_args.kwargs
        self.assertEqual(kwargs["games"], 3)
        self.assertEqual(kwargs["seed_start"], 50)
        self.assertEqual(kwargs["rollout_config"].max_decision_rounds, 7)
        self.assertIn("total_games: 3", stdout.getvalue())
        self.assertIn("peak_rss_mb: 66.25", stdout.getvalue())
        self.assertIn("random-legal vs random-legal", stdout.getvalue())

    def test_rollout_cli_benchmark_wires_custom_policy_matchups(self) -> None:
        fake_report = BenchmarkReport(
            format_id="gen3randombattle",
            max_decision_rounds=9,
            games_per_matchup=2,
            matchups=(),
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            first_checkpoint = Path(temp_dir) / "first.json"
            second_checkpoint = Path(temp_dir) / "second.json"
            save_linear_model(
                first_checkpoint,
                LinearPolicyModel.initialized(
                    feature_count=8,
                    window_size=1,
                    policy_id="first-cli-linear",
                ),
            )
            save_linear_model(
                second_checkpoint,
                LinearPolicyModel.initialized(
                    feature_count=8,
                    window_size=1,
                    policy_id="second-cli-linear",
                ),
            )
            with patch("pokezero.rollout_cli.benchmark_rollouts", return_value=fake_report) as benchmark:
                with patch("sys.stdout", new_callable=io.StringIO):
                    exit_code = rollout_cli_main(
                        [
                            "benchmark",
                            "--games",
                            "2",
                            "--seed-start",
                            "60",
                            "--max-decision-rounds",
                            "9",
                            "--policy",
                            f"linear:{first_checkpoint}",
                            "--policy",
                            f"linear:{second_checkpoint}",
                            "--opponent-policy",
                            "random-legal",
                            "--include-policy-head-to-head",
                        ]
                    )

        self.assertEqual(exit_code, 0)
        kwargs = benchmark.call_args.kwargs
        self.assertEqual(kwargs["games"], 2)
        self.assertEqual(kwargs["seed_start"], 60)
        self.assertEqual(kwargs["rollout_config"].max_decision_rounds, 9)
        self.assertEqual(
            [matchup.label for matchup in kwargs["matchups"]],
            [
                "first-cli-linear vs random-legal",
                "random-legal vs first-cli-linear",
                "second-cli-linear vs random-legal",
                "random-legal vs second-cli-linear",
                "first-cli-linear vs second-cli-linear",
                "second-cli-linear vs first-cli-linear",
            ],
        )

    def test_rollout_cli_benchmark_rejects_opponents_without_custom_policy(self) -> None:
        with patch("sys.stderr", new_callable=io.StringIO) as stderr:
            exit_code = rollout_cli_main(
                [
                    "benchmark",
                    "--opponent-policy",
                    "random-legal",
                ]
            )

        self.assertEqual(exit_code, 1)
        self.assertIn("--opponent-policy requires at least one --policy", stderr.getvalue())

    def test_rollout_cli_benchmark_rejects_head_to_head_without_custom_policy(self) -> None:
        with patch("sys.stderr", new_callable=io.StringIO) as stderr:
            exit_code = rollout_cli_main(
                [
                    "benchmark",
                    "--include-policy-head-to-head",
                ]
            )

        self.assertEqual(exit_code, 1)
        self.assertIn("--include-policy-head-to-head requires at least two --policy values", stderr.getvalue())

    def test_rollout_cli_replay_benchmark_wires_arguments_and_prints_report(self) -> None:
        fake_report = ReplayPrefixBenchmarkReport(
            format_id="gen3randombattle",
            max_decision_rounds=9,
            games=2,
            prefixes_per_game=3,
            source_policy_ids={"p1": "random-legal", "p2": "random-legal"},
            source_decision_rounds=(10, 12),
            timings=(
                ReplayPrefixTiming(
                    seed=80,
                    decision_round_count=0,
                    elapsed_seconds=0.001,
                    requested_players=("p1", "p2"),
                    terminal=False,
                ),
                ReplayPrefixTiming(
                    seed=80,
                    decision_round_count=9,
                    elapsed_seconds=0.003,
                    requested_players=("p1", "p2"),
                    terminal=False,
                ),
            ),
        )
        with patch("pokezero.rollout_cli.benchmark_replay_prefixes", return_value=fake_report) as benchmark:
            with patch("sys.stdout", new_callable=io.StringIO) as stdout:
                exit_code = rollout_cli_main(
                    [
                        "replay-benchmark",
                        "--games",
                        "2",
                        "--prefixes-per-game",
                        "3",
                        "--showdown-root",
                        "/tmp/showdown",
                        "--seed-start",
                        "80",
                        "--max-decision-rounds",
                        "9",
                    ]
                )

        self.assertEqual(exit_code, 0)
        kwargs = benchmark.call_args.kwargs
        self.assertEqual(kwargs["games"], 2)
        self.assertEqual(kwargs["prefixes_per_game"], 3)
        self.assertEqual(kwargs["seed_start"], 80)
        self.assertEqual(kwargs["rollout_config"].max_decision_rounds, 9)
        self.assertEqual(kwargs["policies"]["p1"].policy_id, "random-legal")
        self.assertIn("p95_replay_ms: 3.00", stdout.getvalue())

    def _require_numpy(self) -> None:
        try:
            import numpy  # noqa: F401
        except ModuleNotFoundError:
            self.skipTest("NumPy is not installed in this environment.")

    @unittest.skipIf(integration_config() is None, "requires node and built Pokemon Showdown checkout")
    def test_collect_rollouts_smoke_with_local_showdown_env(self) -> None:
        config = integration_config()
        assert config is not None
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "showdown.jsonl"

            metrics = collect_rollouts(
                output_path=output_path,
                games=1,
                env_factory=lambda: LocalShowdownEnv(config),
                policies={"p1": RandomLegalPolicy(), "p2": RandomLegalPolicy()},
                rollout_config=RolloutConfig(max_decision_rounds=30),
                seed_start=3,
            )

            records = read_rollout_records(output_path)
        self.assertEqual(metrics.games, 1)
        self.assertEqual(len(records), 1)
        self.assertGreater(len(records[0].trajectory.steps), 0)
        self.assertIn(records[0].terminal.winner, {"p1", "p2", None})


def collect_one_record_for_test():
    with tempfile.TemporaryDirectory() as temp_dir:
        output_path = Path(temp_dir) / "rollouts.jsonl"
        collect_rollouts(
            output_path=output_path,
            games=1,
            env_factory=OneTurnEnv,
            policies={"p1": RandomLegalPolicy(), "p2": RandomLegalPolicy()},
            rollout_config=RolloutConfig(max_decision_rounds=5),
        )
        return read_rollout_records(output_path)[0]


if __name__ == "__main__":
    unittest.main()
