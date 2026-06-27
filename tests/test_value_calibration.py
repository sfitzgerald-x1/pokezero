import contextlib
import io
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

from pokezero.collection import RolloutRecord, write_rollout_record
from pokezero.env import TerminalState
from pokezero.neural_cli import print_value_calibration_report
from pokezero.neural_policy import (
    TransformerTrainingConfig,
    TransformerTrainingResult,
    ValueCalibrationTransform,
    require_torch,
    torch_available,
)
from pokezero.observation import PokeZeroObservationV0
from pokezero.trajectory import BattleTrajectory, TrajectoryStep
from pokezero.value_calibration import (
    ValueCalibrationReport,
    evaluate_value_calibration,
    fit_affine_value_calibration_transform,
    fit_value_calibration_transform,
    value_selection_metric_direction,
    value_selection_metric_value,
    value_selection_score,
)
from pokezero.value_calibration import _ValueCalibrationTotals, _trajectory_dataset_config_from_training_result


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

    def test_value_selection_metric_helpers_cover_min_and_max_metrics(self) -> None:
        report = ValueCalibrationReport(
            examples=4,
            mse=0.36,
            mae=0.4,
            bias=-0.25,
            sign_accuracy=0.75,
            expected_calibration_error=0.12,
            bins=(),
            slices=(),
        )

        self.assertEqual(value_selection_metric_direction("mae"), "min")
        self.assertEqual(value_selection_metric_direction("sign_accuracy"), "max")
        self.assertEqual(value_selection_metric_value(report, "abs_bias"), 0.25)
        self.assertEqual(value_selection_score(0.4, "mae"), -0.4)
        self.assertEqual(value_selection_score(0.75, "sign_accuracy"), 0.75)
        with self.assertRaisesRegex(ValueError, "unsupported value selection metric"):
            value_selection_metric_direction("not-a-metric")

    def test_fit_affine_value_calibration_transform_maps_predictions_to_returns(self) -> None:
        transform = fit_affine_value_calibration_transform(
            predictions=(-0.5, 0.0, 0.5),
            returns=(-1.0, 0.0, 1.0),
        )

        self.assertAlmostEqual(transform.scale, 2.0)
        self.assertAlmostEqual(transform.bias, 0.0)
        self.assertAlmostEqual(transform.apply(0.25), 0.5)
        self.assertEqual(transform.apply(2.0), 1.0)

    def test_fit_affine_value_calibration_transform_handles_constant_predictions(self) -> None:
        transform = fit_affine_value_calibration_transform(
            predictions=(0.25, 0.25, 0.25),
            returns=(-1.0, 0.0, 1.0),
        )

        self.assertEqual(transform.scale, 0.0)
        self.assertAlmostEqual(transform.bias, 0.0)

    def test_calibration_dataset_config_matches_training_target_config(self) -> None:
        training_config = TransformerTrainingConfig(
            window_size=3,
            discount=0.75,
            capped_terminal_value=-0.25,
            hp_delta_return_weight=0.2,
            faint_delta_return_weight=0.3,
            turn_penalty_after=20,
            turn_penalty=0.01,
            objective="ppo",
            ppo_target_mode="gae",
            gae_lambda=0.8,
        )

        dataset_config = _trajectory_dataset_config_from_training_result(
            TransformerTrainingResult(
                model_config=SimpleNamespace(),
                training_config=training_config,
                epochs=(),
            )
        )

        self.assertEqual(dataset_config.window_size, 3)
        self.assertEqual(dataset_config.discount, 0.75)
        self.assertEqual(dataset_config.capped_terminal_value, -0.25)
        self.assertEqual(dataset_config.hp_delta_return_weight, 0.2)
        self.assertEqual(dataset_config.faint_delta_return_weight, 0.3)
        self.assertEqual(dataset_config.turn_penalty_after, 20)
        self.assertEqual(dataset_config.turn_penalty, 0.01)
        self.assertEqual(dataset_config.ppo_target_mode, "gae")
        self.assertEqual(dataset_config.gae_lambda, 0.8)

    def test_evaluate_value_calibration_runs_model_over_rollout_batches(self) -> None:
        if not torch_available():
            self.skipTest("PyTorch is not installed in this environment.")
        torch = require_torch()

        class FakeValueModel:
            def __init__(self) -> None:
                self.training = True
                self.training_during_call: bool | None = None

            def eval(self) -> None:
                self.training = False

            def train(self, mode: bool = True) -> None:
                self.training = bool(mode)

            def __call__(self, **kwargs):
                self.training_during_call = self.training
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

            model = FakeValueModel()
            report = evaluate_value_calibration(
                model=model,
                training_result=SimpleNamespace(training_config=TransformerTrainingConfig(window_size=1)),
                paths=path,
                batch_size=2,
                bins=4,
            )

        self.assertEqual(report.examples, 2)
        self.assertAlmostEqual(report.mae, (abs(0.8 - 1.0) + abs(-0.6 - -1.0)) / 2)
        self.assertEqual(report.sign_accuracy, 1.0)
        self.assertFalse(model.training_during_call)
        self.assertTrue(model.training)

    def test_evaluate_value_calibration_applies_stored_transform(self) -> None:
        if not torch_available():
            self.skipTest("PyTorch is not installed in this environment.")
        torch = require_torch()

        class FakeValueModel:
            def eval(self) -> None:
                pass

            def __call__(self, **kwargs):
                batch_size = int(kwargs["categorical_ids"].shape[0])
                return SimpleNamespace(value=torch.tensor((0.4, -0.4)[:batch_size]))

        trajectory = BattleTrajectory(battle_id="battle", format_id="gen3randombattle", seed=9)
        observation = _observation()
        for player_id in ("p1", "p2"):
            trajectory.append(
                TrajectoryStep(
                    player_id=player_id,
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
                training_result=SimpleNamespace(
                    training_config=TransformerTrainingConfig(window_size=1),
                    value_calibration_transform=ValueCalibrationTransform(scale=2.0, bias=0.0),
                ),
                paths=path,
                batch_size=2,
                bins=4,
            )

        self.assertAlmostEqual(report.mae, (abs(0.8 - 1.0) + abs(-0.8 - -1.0)) / 2)
        self.assertEqual(report.sign_accuracy, 1.0)

    def test_value_calibration_uses_training_shaped_return_targets(self) -> None:
        if not torch_available():
            self.skipTest("PyTorch is not installed in this environment.")
        torch = require_torch()

        class ZeroValueModel:
            def eval(self) -> None:
                pass

            def __call__(self, **kwargs):
                batch_size = int(kwargs["categorical_ids"].shape[0])
                return SimpleNamespace(value=torch.zeros(batch_size))

        first_observation = PokeZeroObservationV0(
            categorical_ids=(),
            numeric_features=(),
            token_type_ids=(),
            attention_mask=(),
            legal_action_mask=(True, False, False, False, False, False, False, False, False),
            metadata={
                "self_team": [{"species": "Charizard", "hp_fraction": 1.0, "fainted": False}],
                "opponent_team": [{"species": "Xatu", "hp_fraction": 1.0, "fainted": False}],
            },
        )
        second_observation = PokeZeroObservationV0(
            categorical_ids=(),
            numeric_features=(),
            token_type_ids=(),
            attention_mask=(),
            legal_action_mask=(True, False, False, False, False, False, False, False, False),
            metadata={
                "self_team": [{"species": "Charizard", "hp_fraction": 1.0, "fainted": False}],
                "opponent_team": [{"species": "Xatu", "hp_fraction": 0.4, "fainted": False}],
            },
        )
        trajectory = BattleTrajectory(battle_id="battle", format_id="gen3randombattle", seed=9)
        for turn_index, observation in enumerate((first_observation, second_observation)):
            trajectory.append(
                TrajectoryStep(
                    player_id="p1",
                    turn_index=turn_index,
                    observation=observation,
                    legal_action_mask=observation.legal_action_mask,
                    action_index=0,
                )
            )
        trajectory.record_terminal(TerminalState(winner=None, turn_count=2))
        record = RolloutRecord(
            battle_id="battle",
            seed=9,
            format_id="gen3randombattle",
            policy_ids={"p1": "fixture"},
            decision_round_count=2,
            elapsed_seconds=0.1,
            terminal=trajectory.terminal,
            trajectory=trajectory,
        )
        training_result = TransformerTrainingResult(
            model_config=SimpleNamespace(),
            training_config=TransformerTrainingConfig(window_size=1, hp_delta_return_weight=3.0),
            epochs=(),
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "rollouts.jsonl"
            with path.open("w", encoding="utf-8") as handle:
                write_rollout_record(handle, record)

            report = evaluate_value_calibration(
                model=ZeroValueModel(),
                training_result=training_result,
                paths=path,
                batch_size=2,
                bins=4,
            )
            transform = fit_value_calibration_transform(
                model=ZeroValueModel(),
                training_result=training_result,
                paths=path,
                batch_size=2,
            )

        self.assertEqual(report.examples, 2)
        self.assertAlmostEqual(report.mae, 0.3)
        self.assertAlmostEqual(report.bias, -0.3)
        self.assertAlmostEqual(transform.bias, 0.3)

    def test_value_calibration_uses_ppo_gae_value_targets_when_available(self) -> None:
        if not torch_available():
            self.skipTest("PyTorch is not installed in this environment.")
        torch = require_torch()

        class ZeroValueModel:
            def eval(self) -> None:
                pass

            def __call__(self, **kwargs):
                batch_size = int(kwargs["categorical_ids"].shape[0])
                return SimpleNamespace(value=torch.zeros(batch_size))

        observation = _observation()
        trajectory = BattleTrajectory(battle_id="battle", format_id="gen3randombattle", seed=9)
        trajectory.append(
            TrajectoryStep(
                player_id="p1",
                turn_index=0,
                observation=observation,
                legal_action_mask=observation.legal_action_mask,
                action_index=0,
                value_estimate=0.2,
            )
        )
        trajectory.append(
            TrajectoryStep(
                player_id="p1",
                turn_index=1,
                observation=observation,
                legal_action_mask=observation.legal_action_mask,
                action_index=0,
                value_estimate=0.5,
            )
        )
        trajectory.record_terminal(TerminalState(winner="p1", turn_count=2))
        record = RolloutRecord(
            battle_id="battle",
            seed=9,
            format_id="gen3randombattle",
            policy_ids={"p1": "fixture"},
            decision_round_count=2,
            elapsed_seconds=0.1,
            terminal=trajectory.terminal,
            trajectory=trajectory,
        )
        training_result = TransformerTrainingResult(
            model_config=SimpleNamespace(),
            training_config=TransformerTrainingConfig(
                window_size=1,
                objective="ppo",
                ppo_target_mode="gae",
                gae_lambda=0.0,
            ),
            epochs=(),
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "rollouts.jsonl"
            with path.open("w", encoding="utf-8") as handle:
                write_rollout_record(handle, record)

            report = evaluate_value_calibration(
                model=ZeroValueModel(),
                training_result=training_result,
                paths=path,
                batch_size=2,
                bins=4,
            )
            transform = fit_value_calibration_transform(
                model=ZeroValueModel(),
                training_result=training_result,
                paths=path,
                batch_size=2,
            )

        self.assertEqual(report.examples, 2)
        self.assertAlmostEqual(report.mae, 0.75)
        self.assertAlmostEqual(report.bias, -0.75)
        self.assertAlmostEqual(transform.bias, 0.75)

    def test_evaluate_value_calibration_reports_stratified_slices(self) -> None:
        if not torch_available():
            self.skipTest("PyTorch is not installed in this environment.")
        torch = require_torch()

        class FakeValueModel:
            def eval(self) -> None:
                pass

            def __call__(self, **kwargs):
                batch_size = int(kwargs["categorical_ids"].shape[0])
                return SimpleNamespace(value=torch.tensor((0.8, -0.6, 0.1, -0.2)[:batch_size]))

        uncapped = BattleTrajectory(battle_id="uncapped", format_id="gen3randombattle", seed=9)
        uncapped.append(
            TrajectoryStep(
                player_id="p1",
                turn_index=0,
                observation=_observation(),
                legal_action_mask=_observation().legal_action_mask,
                action_index=0,
            )
        )
        uncapped.append(
            TrajectoryStep(
                player_id="p2",
                turn_index=15,
                observation=_observation(),
                legal_action_mask=_observation().legal_action_mask,
                action_index=0,
            )
        )
        uncapped.record_terminal(TerminalState(winner="p1", turn_count=16))
        capped = BattleTrajectory(battle_id="capped", format_id="gen3randombattle", seed=10)
        capped.append(
            TrajectoryStep(
                player_id="p1",
                turn_index=35,
                observation=_observation(),
                legal_action_mask=_observation().legal_action_mask,
                action_index=0,
            )
        )
        capped.append(
            TrajectoryStep(
                player_id="p2",
                turn_index=5,
                observation=_observation(),
                legal_action_mask=_observation().legal_action_mask,
                action_index=0,
            )
        )
        capped.record_terminal(TerminalState(winner=None, turn_count=250, capped=True))

        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "rollouts.jsonl"
            with path.open("w", encoding="utf-8") as handle:
                for trajectory in (uncapped, capped):
                    write_rollout_record(
                        handle,
                        RolloutRecord(
                            battle_id=trajectory.battle_id,
                            seed=trajectory.seed,
                            format_id=trajectory.format_id,
                            policy_ids={"p1": "fixture", "p2": "fixture"},
                            decision_round_count=trajectory.terminal.turn_count if trajectory.terminal else 0,
                            elapsed_seconds=0.1,
                            terminal=trajectory.terminal,
                            trajectory=trajectory,
                        ),
                    )

            report = evaluate_value_calibration(
                model=FakeValueModel(),
                training_result=SimpleNamespace(training_config=TransformerTrainingConfig(window_size=1)),
                paths=path,
                batch_size=4,
                bins=4,
            )

        slice_counts = {slice_result.name: slice_result.examples for slice_result in report.slices}
        slices = {slice_result.name: slice_result for slice_result in report.slices}

        self.assertEqual(slice_counts["return:positive"], 1)
        self.assertEqual(slice_counts["return:negative"], 1)
        self.assertEqual(slice_counts["return:zero"], 2)
        self.assertEqual(slice_counts["turn:early_0_9"], 2)
        self.assertEqual(slice_counts["turn:mid_10_29"], 1)
        self.assertEqual(slice_counts["turn:late_30_plus"], 1)
        self.assertEqual(slice_counts["terminal:uncapped"], 2)
        self.assertEqual(slice_counts["terminal:capped"], 2)
        self.assertAlmostEqual(slices["return:positive"].mae, 0.2)
        self.assertAlmostEqual(slices["return:negative"].mae, 0.4)
        self.assertAlmostEqual(slices["return:zero"].mae, 0.15)
        self.assertAlmostEqual(slices["return:zero"].bias, -0.05)
        self.assertFalse(slices["return:zero"].sign_accuracy_applicable)
        self.assertTrue(slices["return:positive"].sign_accuracy_applicable)
        payload = report.to_dict()
        self.assertIn("slices", payload)
        zero_payload = next(slice_payload for slice_payload in payload["slices"] if slice_payload["name"] == "return:zero")
        self.assertFalse(zero_payload["sign_accuracy_applicable"])

        stdout = io.StringIO()
        with contextlib.redirect_stdout(stdout):
            print_value_calibration_report(report)

        self.assertIn("return:zero", stdout.getvalue())
        self.assertIn("n/a", stdout.getvalue())


if __name__ == "__main__":
    unittest.main()
