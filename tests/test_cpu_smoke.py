import io
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

from pokezero.cpu_smoke import CPU_SMOKE_RUN_SCHEMA_VERSION, run_cpu_smoke_experiment
from pokezero.cpu_smoke_cli import main as cpu_smoke_cli_main
from pokezero.env import StepResult, TerminalState
from pokezero.observation import ObservationPerspective, ObservationSpec, PokeZeroObservationV0
from pokezero.promotion import load_promotion_registry
from pokezero.rollout import RolloutConfig


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


class OneTurnEnv:
    def __init__(self) -> None:
        self._observation = observation((True, False, False, False, False, False, False, False, False))
        self._requested = ("p1", "p2")
        self._terminal = None

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
        pass


class CPUSmokeTest(unittest.TestCase):
    def test_run_cpu_smoke_experiment_writes_summary_and_exercises_promotion_audit(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            result = run_cpu_smoke_experiment(
                run_dir=Path(temp_dir) / "smoke",
                env_factory=OneTurnEnv,
                rollout_config=RolloutConfig(max_decision_rounds=5),
                audit_profile="smoke",
                train_games=2,
                validation_games=1,
                bootstrap_benchmark_games=1,
                preflight_games=0,
                selfplay_iterations=1,
                games_per_iteration=2,
                evaluation_games=1,
                teacher_policy_spec="simple-legal",
                bootstrap_opponent_policy_specs=("random-legal",),
                fixed_opponent_policy_specs=("random-legal",),
                feature_count=32,
                window_size=1,
            )
            summary = json.loads(result.summary_path.read_text(encoding="utf-8"))
            registry = load_promotion_registry(result.promotion_registry_path)
            artifact_dir_exists = result.promotion_artifact_dir.exists()

        self.assertTrue(result.passed)
        self.assertEqual(summary["schema_version"], CPU_SMOKE_RUN_SCHEMA_VERSION)
        self.assertEqual(summary["passed"], True)
        self.assertEqual(summary["bootstrap"]["train_games"], 2)
        self.assertEqual(summary["selfplay"]["iterations"], 1)
        self.assertEqual(summary["audit"]["passed"], True)
        self.assertEqual(len(registry.entries), 1)
        self.assertTrue(artifact_dir_exists)

    def test_cpu_smoke_cli_wires_arguments_and_prints_json(self) -> None:
        fake_result = SimpleNamespace(
            passed=True,
            to_dict=lambda: {"schema_version": CPU_SMOKE_RUN_SCHEMA_VERSION, "passed": True},
        )
        with patch("pokezero.cpu_smoke_cli.run_cpu_smoke_experiment", return_value=fake_result) as run:
            with patch("sys.stdout", new_callable=io.StringIO) as stdout:
                exit_code = cpu_smoke_cli_main(
                    [
                        "--run-dir",
                        "/tmp/pokezero-smoke",
                        "--showdown-root",
                        "/tmp/showdown",
                        "--teacher-policy",
                        "simple-legal",
                        "--bootstrap-opponent-policy",
                        "random-legal",
                        "--selfplay-opponent-policy",
                        "simple-legal",
                        "--audit-profile",
                        "smoke",
                        "--format",
                        "gen3randombattle",
                        "--max-decision-rounds",
                        "17",
                        "--workers",
                        "2",
                        "--train-games",
                        "3",
                        "--validation-games",
                        "2",
                        "--bootstrap-benchmark-games",
                        "7",
                        "--preflight-games",
                        "8",
                        "--selfplay-iterations",
                        "4",
                        "--games-per-iteration",
                        "5",
                        "--evaluation-games",
                        "6",
                        "--feature-count",
                        "64",
                        "--window-size",
                        "2",
                        "--epochs",
                        "3",
                        "--learning-rate",
                        "0.02",
                        "--json",
                    ]
                )
            payload = json.loads(stdout.getvalue())

        self.assertEqual(exit_code, 0)
        self.assertEqual(payload["schema_version"], CPU_SMOKE_RUN_SCHEMA_VERSION)
        kwargs = run.call_args.kwargs
        self.assertEqual(kwargs["run_dir"], Path("/tmp/pokezero-smoke"))
        self.assertEqual(kwargs["audit_profile"], "smoke")
        self.assertEqual(kwargs["train_games"], 3)
        self.assertEqual(kwargs["validation_games"], 2)
        self.assertEqual(kwargs["bootstrap_benchmark_games"], 7)
        self.assertEqual(kwargs["preflight_games"], 8)
        self.assertEqual(kwargs["selfplay_iterations"], 4)
        self.assertEqual(kwargs["games_per_iteration"], 5)
        self.assertEqual(kwargs["evaluation_games"], 6)
        self.assertEqual(kwargs["worker_count"], 2)
        self.assertEqual(kwargs["teacher_policy_spec"], "simple-legal")
        self.assertEqual(kwargs["bootstrap_opponent_policy_specs"], ("random-legal",))
        self.assertEqual(kwargs["fixed_opponent_policy_specs"], ("simple-legal",))
        self.assertEqual(kwargs["feature_count"], 64)
        self.assertEqual(kwargs["window_size"], 2)
        self.assertEqual(kwargs["epochs"], 3)
        self.assertEqual(kwargs["learning_rate"], 0.02)
        self.assertEqual(kwargs["rollout_config"].format_id, "gen3randombattle")
        self.assertEqual(kwargs["rollout_config"].max_decision_rounds, 17)

    def test_cpu_smoke_cli_prints_calibration_warning_for_tiny_runs(self) -> None:
        fake_result = SimpleNamespace(
            passed=True,
            run_dir=Path("/tmp/smoke"),
            summary_path=Path("/tmp/smoke/summary.json"),
            audit_profile="smoke",
            bootstrap=SimpleNamespace(
                manifest_path=Path("/tmp/smoke/bootstrap/manifest.json"),
                checkpoint_path=Path("/tmp/smoke/bootstrap/linear-bootstrap.json"),
            ),
            selfplay=SimpleNamespace(
                run_dir=Path("/tmp/smoke/selfplay"),
                latest_checkpoint_path=Path("/tmp/smoke/selfplay/iteration-1/linear-policy.json"),
            ),
            promotion_registry_path=Path("/tmp/smoke/promotions.json"),
            audit=SimpleNamespace(passed=True, checks=()),
            calibration=SimpleNamespace(
                notes=("Observed benchmark game counts are below the default audit minimum.",),
                min_latest_benchmark_games=1,
                suggested_cli_flags=lambda: ("--min-latest-benchmark-games", "1"),
            ),
        )
        with patch("pokezero.cpu_smoke_cli._run", return_value=fake_result):
            with patch("sys.stdout", new_callable=io.StringIO) as stdout:
                exit_code = cpu_smoke_cli_main(["--run-dir", "/tmp/pokezero-smoke"])

        output = stdout.getvalue()
        self.assertEqual(exit_code, 0)
        self.assertIn("calibration_notes:", output)
        self.assertIn("calibration_warning:", output)

    def test_run_cpu_smoke_experiment_rejects_partial_run_directory(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            run_dir = Path(temp_dir) / "smoke"
            (run_dir / "bootstrap").mkdir(parents=True)
            with self.assertRaisesRegex(ValueError, "partial artifacts"):
                run_cpu_smoke_experiment(
                    run_dir=run_dir,
                    env_factory=OneTurnEnv,
                    rollout_config=RolloutConfig(max_decision_rounds=5),
                    teacher_policy_spec="simple-legal",
                    bootstrap_opponent_policy_specs=("random-legal",),
                    fixed_opponent_policy_specs=("random-legal",),
                    feature_count=32,
                    window_size=1,
                )


if __name__ == "__main__":
    unittest.main()
