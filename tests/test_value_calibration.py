from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest

from pokezero.collection import RolloutRecord, write_rollout_record
from pokezero.env import TerminalState
from pokezero.neural_policy import TransformerTrainingConfig, require_torch, torch_available
from pokezero.observation import PokeZeroObservationV0
from pokezero.trajectory import BattleTrajectory, TrajectoryStep
from pokezero.value_calibration import evaluate_value_calibration
from pokezero.value_calibration import _ValueCalibrationTotals


def _observation() -> PokeZeroObservationV0:
    return PokeZeroObservationV0(
        categorical_ids=(),
        numeric_features=(),
        token_type_ids=(),
        attention_mask=(),
        legal_action_mask=(True, False, False, False, False, False, False, False, False),
    )


class ValueCalibrationTest(unittest.TestCase):
    def test_value_calibration_totals_compute_error_and_bins(self) -> None:
        totals = _ValueCalibrationTotals(bin_count=4)

        totals.add(
            predictions=(-0.75, -0.25, 0.25, 0.75),
            returns=(-1.0, 0.0, 1.0, 1.0),
        )
        report = totals.to_report()

        self.assertEqual(report.examples, 4)
        self.assertAlmostEqual(report.mse, (0.25**2 + 0.25**2 + 0.75**2 + 0.25**2) / 4)
        self.assertAlmostEqual(report.mae, (0.25 + 0.25 + 0.75 + 0.25) / 4)
        self.assertAlmostEqual(report.bias, (0.25 + -0.25 + -0.75 + -0.25) / 4)
        self.assertEqual(report.sign_accuracy, 0.75)
        self.assertEqual([bin_result.count for bin_result in report.bins], [1, 1, 1, 1])
        self.assertGreater(report.expected_calibration_error, 0.0)
        self.assertEqual(report.to_dict()["examples"], 4)

    def test_value_calibration_totals_rejects_empty_report(self) -> None:
        with self.assertRaisesRegex(ValueError, "no examples"):
            _ValueCalibrationTotals(bin_count=2).to_report()

    def test_value_calibration_totals_rejects_mismatched_lengths(self) -> None:
        with self.assertRaisesRegex(ValueError, "same length"):
            _ValueCalibrationTotals(bin_count=2).add(predictions=(0.0,), returns=(0.0, 1.0))

    def test_evaluate_value_calibration_runs_model_over_rollout_batches(self) -> None:
        if not torch_available():
            self.skipTest("PyTorch is not installed in this environment.")
        torch = require_torch()

        class FakeValueModel:
            def eval(self) -> None:
                pass

            def __call__(self, **kwargs):
                batch_size = int(kwargs["categorical_ids"].shape[0])
                return SimpleNamespace(value=torch.tensor((0.8, -0.6)[:batch_size]))

        trajectory = BattleTrajectory(battle_id="battle", format_id="gen3randombattle", seed=9)
        observation = _observation()
        trajectory.append(
            TrajectoryStep(
                player_id="p1",
                turn_index=0,
                observation=observation,
                legal_action_mask=observation.legal_action_mask,
                action_index=0,
            )
        )
        trajectory.append(
            TrajectoryStep(
                player_id="p2",
                turn_index=0,
                observation=observation,
                legal_action_mask=observation.legal_action_mask,
                action_index=0,
            )
        )
        trajectory.record_terminal(TerminalState(winner="p1", turn_count=1))
        record = RolloutRecord(
            battle_id="battle",
            seed=9,
            format_id="gen3randombattle",
            policy_ids={"p1": "fixture", "p2": "fixture"},
            decision_round_count=1,
            elapsed_seconds=0.1,
            terminal=trajectory.terminal,
            trajectory=trajectory,
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "rollouts.jsonl"
            with path.open("w", encoding="utf-8") as handle:
                write_rollout_record(handle, record)

            report = evaluate_value_calibration(
                model=FakeValueModel(),
                training_result=SimpleNamespace(training_config=TransformerTrainingConfig(window_size=1)),
                paths=path,
                batch_size=2,
                bins=4,
            )

        self.assertEqual(report.examples, 2)
        self.assertAlmostEqual(report.mae, (abs(0.8 - 1.0) + abs(-0.6 - -1.0)) / 2)
        self.assertEqual(report.sign_accuracy, 1.0)


if __name__ == "__main__":
    unittest.main()
