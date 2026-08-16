import contextlib
import io
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

from pokezero.collection import RolloutRecord, write_rollout_record
from pokezero.dataset import TrajectoryDatasetConfig, write_training_cache_from_rollouts
from pokezero.env import TerminalState
from pokezero.neural_cli import print_value_calibration_report
from pokezero.neural_policy import (
    TransformerTrainingConfig,
    TransformerTrainingResult,
    ValueCalibrationTransform,
    require_torch,
    torch_available,
)
from pokezero.observation import ObservationSpec, PokeZeroObservationV0
from pokezero.trajectory import BattleTrajectory, TrajectoryStep
from pokezero.value_calibration import (
    ValueCalibrationReport,
    evaluate_value_calibration,
    fit_affine_value_calibration_transform,
    fit_isotonic_value_calibration_transform,
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


_CACHEABLE_MASK = (True, True, False, False, False, False, False, False, False)


def _cacheable_observation(value: int) -> PokeZeroObservationV0:
    """An observation with real feature rows, which a training cache requires."""
    spec = ObservationSpec(categorical_feature_count=1, numeric_feature_count=1)
    return PokeZeroObservationV0(
        categorical_ids=tuple((value,) for _ in range(spec.token_count)),
        numeric_features=tuple((float(value),) for _ in range(spec.token_count)),
        token_type_ids=tuple(0 for _ in range(spec.token_count)),
        attention_mask=tuple(True for _ in range(spec.token_count)),
        legal_action_mask=_CACHEABLE_MASK,
    )


def _cacheable_rollout_record() -> RolloutRecord:
    """A p1 win with recorded value estimates, so GAE-mode collection is well defined."""
    trajectory = BattleTrajectory(battle_id="cache", format_id="gen3randombattle", seed=9)
    for turn_index, value_estimate in ((0, 0.2), (1, 0.5)):
        trajectory.append(
            TrajectoryStep(
                player_id="p1",
                turn_index=turn_index,
                observation=_cacheable_observation(turn_index + 1),
                legal_action_mask=_CACHEABLE_MASK,
                action_index=0,
                value_estimate=value_estimate,
            )
        )
    trajectory.record_terminal(TerminalState(winner="p1", turn_count=2))
    return RolloutRecord(
        battle_id=trajectory.battle_id,
        seed=trajectory.seed,
        format_id=trajectory.format_id,
        policy_ids={"p1": "fixture"},
        decision_round_count=2,
        elapsed_seconds=0.1,
        terminal=trajectory.terminal,
        trajectory=trajectory,
    )


def _cache_calibration_report(
    *,
    model: object,
    cache_config: TrajectoryDatasetConfig,
    objective: str,
    ppo_target_mode: str,
    gae_lambda: float,
    freeze_non_value_parameters: bool = False,
) -> ValueCalibrationReport:
    """Collect one rollout into a training cache and calibrate a head against it."""
    with tempfile.TemporaryDirectory() as temp_dir:
        root = Path(temp_dir)
        jsonl = root / "rollouts.jsonl"
        with jsonl.open("w", encoding="utf-8") as handle:
            write_rollout_record(handle, _cacheable_rollout_record())
        cache = root / "cache"
        write_training_cache_from_rollouts(jsonl, cache, config=cache_config)
        training_result = TransformerTrainingResult(
            model_config=SimpleNamespace(),
            training_config=TransformerTrainingConfig(
                window_size=1,
                objective=objective,
                freeze_non_value_parameters=freeze_non_value_parameters,
                ppo_target_mode=ppo_target_mode,
                gae_lambda=gae_lambda,
            ),
            epochs=(),
        )
        return evaluate_value_calibration(
            model=model,
            training_result=training_result,
            paths=cache,
            batch_size=2,
            bins=4,
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
        self.assertAlmostEqual(report.pearson_correlation or 0.0, 7.0 / (55.0**0.5))
        self.assertEqual(report.to_dict()["examples"], 4)
        self.assertAlmostEqual(report.to_dict()["pearson_correlation"], 7.0 / (55.0**0.5))

    def test_value_calibration_correlation_is_absent_for_constant_targets(self) -> None:
        totals = _ValueCalibrationTotals(bin_count=2)

        totals.add(predictions=(-0.5, 0.0, 0.5), returns=(1.0, 1.0, 1.0))
        report = totals.to_report()

        self.assertIsNone(report.pearson_correlation)
        self.assertIsNone(report.to_dict()["pearson_correlation"])

    def test_value_calibration_correlation_is_absent_for_collapsed_prediction_head(self) -> None:
        totals = _ValueCalibrationTotals(bin_count=2)

        totals.add(
            predictions=tuple(0.1 for _ in range(10_000)),
            returns=tuple(1.0 if index % 2 else -1.0 for index in range(10_000)),
        )
        report = totals.to_report()

        self.assertIsNone(report.pearson_correlation)
        self.assertIsNone(report.to_dict()["pearson_correlation"])

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
            pearson_correlation=0.62,
            bins=(),
            slices=(),
        )

        self.assertEqual(value_selection_metric_direction("mae"), "min")
        self.assertEqual(value_selection_metric_direction("sign_accuracy"), "max")
        self.assertEqual(value_selection_metric_direction("pearson_correlation"), "max")
        self.assertEqual(value_selection_metric_value(report, "abs_bias"), 0.25)
        self.assertEqual(value_selection_metric_value(report, "pearson_correlation"), 0.62)
        self.assertEqual(value_selection_score(0.4, "mae"), -0.4)
        self.assertEqual(value_selection_score(0.75, "sign_accuracy"), 0.75)
        self.assertEqual(value_selection_score(0.62, "pearson_correlation"), 0.62)
        with self.assertRaisesRegex(ValueError, "unsupported value selection metric"):
            value_selection_metric_direction("not-a-metric")

    def test_value_selection_metric_rejects_unavailable_correlation(self) -> None:
        report = ValueCalibrationReport(
            examples=4,
            mse=0.36,
            mae=0.4,
            bias=-0.25,
            sign_accuracy=0.75,
            expected_calibration_error=0.12,
            pearson_correlation=None,
            bins=(),
            slices=(),
        )

        with self.assertRaisesRegex(ValueError, "pearson_correlation"):
            value_selection_metric_value(report, "pearson_correlation")

    def test_value_selection_metric_rejects_non_finite_correlation(self) -> None:
        report = ValueCalibrationReport(
            examples=4,
            mse=0.36,
            mae=0.4,
            bias=-0.25,
            sign_accuracy=0.75,
            expected_calibration_error=0.12,
            pearson_correlation=float("nan"),
            bins=(),
            slices=(),
        )

        with self.assertRaisesRegex(ValueError, "finite"):
            value_selection_metric_value(report, "pearson_correlation")

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

    def test_value_calibration_transform_round_trips_isotonic_points(self) -> None:
        transform = ValueCalibrationTransform(
            method="isotonic",
            points=((-1.0, -0.8), (0.0, 0.2), (1.0, 0.8)),
        )

        restored = ValueCalibrationTransform.from_dict(transform.to_dict())

        self.assertEqual(restored.method, "isotonic")
        self.assertEqual(restored.points, ((-1.0, -0.8), (0.0, 0.2), (1.0, 0.8)))
        self.assertAlmostEqual(restored.apply(0.5), 0.5)
        self.assertAlmostEqual(restored.apply(-2.0), -0.8)
        self.assertAlmostEqual(restored.apply(2.0), 0.8)

    def test_fit_isotonic_value_calibration_transform_pools_non_monotone_targets(self) -> None:
        transform = fit_isotonic_value_calibration_transform(
            predictions=(-0.8, -0.4, 0.0, 0.4, 0.8),
            returns=(-1.0, 1.0, -1.0, 1.0, 1.0),
        )

        self.assertEqual(transform.method, "isotonic")
        self.assertEqual(transform.points, ((-0.8, -1.0), (-0.4, 0.0), (0.0, 0.0), (0.4, 1.0), (0.8, 1.0)))
        self.assertAlmostEqual(transform.apply(-0.4), 0.0)
        self.assertAlmostEqual(transform.apply(-0.2), 0.0)
        self.assertAlmostEqual(transform.apply(0.6), 1.0)

    def test_fit_isotonic_value_calibration_transform_groups_duplicate_predictions(self) -> None:
        transform = fit_isotonic_value_calibration_transform(
            predictions=(0.2, 0.2, 0.8),
            returns=(-1.0, 1.0, 1.0),
        )

        self.assertEqual(transform.points, ((0.2, 0.0), (0.8, 1.0)))

    def test_calibration_dataset_config_matches_training_dataset_config(self) -> None:
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
        # ppo_target_mode/gae_lambda are part of a training cache's stamped identity, and the
        # cache reader refuses any field mismatch -- so calibration must request the same two
        # values the run trained with or it cannot open the cache at all. This does NOT change
        # the calibration target: GAE lands in ppo_value_targets, never in `returns`.
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

    def test_value_calibration_uses_outcome_returns_when_ppo_gae_is_configured(self) -> None:
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
        self.assertAlmostEqual(report.mae, 1.0)
        self.assertAlmostEqual(report.bias, -1.0)
        self.assertAlmostEqual(transform.bias, 1.0)

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
        self.assertIsNone(slices["return:positive"].pearson_correlation)
        self.assertIsNone(slices["return:negative"].pearson_correlation)
        self.assertIsNone(slices["return:zero"].pearson_correlation)
        payload = report.to_dict()
        self.assertIn("slices", payload)
        zero_payload = next(slice_payload for slice_payload in payload["slices"] if slice_payload["name"] == "return:zero")
        self.assertFalse(zero_payload["sign_accuracy_applicable"])
        self.assertIsNone(zero_payload["pearson_correlation"])

        stdout = io.StringIO()
        with contextlib.redirect_stdout(stdout):
            print_value_calibration_report(report)

        output = stdout.getvalue()
        self.assertIn("pearson_correlation:", output)
        self.assertIn("return:zero", output)
        self.assertIn("corr", output)
        self.assertIn("n/a", output)

    def test_value_calibration_reads_a_gae_collected_training_cache(self) -> None:
        """--value-calibration-data against a GAE-collected cache used to be unreachable.

        A training cache stamps its dataset config and `iter_training_cache_batches` refuses
        any field mismatch. Calibration built its request config without ppo_target_mode or
        gae_lambda, so every cache collected with ppo_target_mode='gae' -- i.e. every cache a
        PPO run produces -- raised "training cache dataset config does not match requested
        training config." and the report could not be produced at all.

        The second half of the assertion is the part that makes the fix safe: reading the GAE
        cache must return the SAME report as reading the same rollouts cached in returns mode,
        because `returns` is the outcome return in either mode and GAE targets live in
        separate columns calibration never reads.
        """
        if not torch_available():
            self.skipTest("PyTorch is not installed in this environment.")
        torch = require_torch()

        class ZeroValueModel:
            def eval(self) -> None:
                pass

            def __call__(self, **kwargs):
                return SimpleNamespace(value=torch.zeros(int(kwargs["history_mask"].shape[0])))

        gae_report = _cache_calibration_report(
            model=ZeroValueModel(),
            cache_config=TrajectoryDatasetConfig(window_size=1, ppo_target_mode="gae", gae_lambda=0.8),
            objective="ppo",
            ppo_target_mode="gae",
            gae_lambda=0.8,
        )
        returns_report = _cache_calibration_report(
            model=ZeroValueModel(),
            cache_config=TrajectoryDatasetConfig(window_size=1),
            objective="ppo",
            ppo_target_mode="returns",
            gae_lambda=0.95,
        )

        self.assertEqual(gae_report.examples, 2)
        self.assertEqual(gae_report.examples, returns_report.examples)
        self.assertAlmostEqual(gae_report.mae, returns_report.mae)
        self.assertAlmostEqual(gae_report.bias, returns_report.bias)
        self.assertAlmostEqual(gae_report.mse, returns_report.mse)
        # A zero-valued head against a p1 win: the target read is the outcome return, not a
        # bootstrapped GAE target built from the recorded value estimates (0.2, 0.5).
        self.assertAlmostEqual(gae_report.mae, 1.0)
        self.assertAlmostEqual(gae_report.bias, -1.0)

    def test_value_only_is_scored_against_returns_not_gae_targets(self) -> None:
        """The objective gate on `_value_targets`, pinned by the loss it changes.

        Widening the ppo_target_mode guard to admit `value-only` made this configuration
        constructible for the first time. Without the gate, a value-only run on a GAE cache
        trains toward `ppo_value_targets` (bootstrap targets anchored to the collecting
        checkpoint's own estimates) while `evaluate_value_calibration` selects the epoch on
        `returns`. Training on one target and selecting on another is silent, so it is pinned
        here on the VALUE OF THE LOSS -- asserting the returned tensor alone would pass against
        an implementation that ignored the objective for the ranking-loss path.
        """
        from pokezero import neural_policy as np_mod

        torch = __import__("torch")
        tensors = {
            "returns": torch.tensor([-1.0, -1.0, -1.0]),
            "ppo_value_targets": torch.tensor([-0.316, -0.620, -1.0]),
            "ppo_value_target_mask": torch.tensor([True, True, True]),
        }
        returns = tensors["returns"]

        value_only = np_mod._value_targets(tensors, "value-only")
        self.assertTrue(
            torch.equal(value_only, returns),
            f"value-only must be scored against outcome returns, got {value_only.tolist()}",
        )
        for other in ("behavior-cloning", "reward-weighted"):
            self.assertTrue(
                torch.equal(np_mod._value_targets(tensors, other), returns),
                f"{other} must be scored against outcome returns",
            )

        ppo = np_mod._value_targets(tensors, "ppo")
        self.assertTrue(
            torch.equal(ppo, tensors["ppo_value_targets"]),
            "ppo must still consume GAE targets -- a gate that starves PPO is not a fix",
        )
        # The two must actually DIFFER on this fixture, or the assertions above are vacuous:
        # a gate could be missing entirely and every equality would still hold.
        self.assertFalse(
            torch.equal(ppo, returns),
            "fixture is inert: GAE targets equal returns here, so the gate is untested",
        )

    def test_value_only_fine_tune_reads_the_gae_cache_it_is_pointed_at(self) -> None:
        """The value-tune shape of the same defect, isolated to the objective guard.

        `foundation-value-tune-run` fine-tunes with objective='value-only' on caches a PPO run
        collected. Naming the mode those caches were stamped with was rejected outright, so the
        request that reads them could not even be constructed -- a separate failure from the
        dropped fields above, and the one that fires first for value-tune.
        """
        if not torch_available():
            self.skipTest("PyTorch is not installed in this environment.")
        torch = require_torch()

        class ZeroValueModel:
            def eval(self) -> None:
                pass

            def __call__(self, **kwargs):
                return SimpleNamespace(value=torch.zeros(int(kwargs["history_mask"].shape[0])))

        report = _cache_calibration_report(
            model=ZeroValueModel(),
            cache_config=TrajectoryDatasetConfig(window_size=1, ppo_target_mode="gae", gae_lambda=0.8),
            objective="value-only",
            ppo_target_mode="gae",
            gae_lambda=0.8,
            freeze_non_value_parameters=True,
        )

        self.assertEqual(report.examples, 2)
        self.assertAlmostEqual(report.mae, 1.0)

    def test_value_only_objective_can_name_the_mode_its_cache_was_collected_with(self) -> None:
        config = TransformerTrainingConfig(
            window_size=1,
            objective="value-only",
            freeze_non_value_parameters=True,
            ppo_target_mode="gae",
            gae_lambda=0.8,
        )

        self.assertEqual(config.ppo_target_mode, "gae")
        self.assertEqual(config.gae_lambda, 0.8)
        # The guard still holds for every objective that cannot read such a cache.
        for objective in ("behavior-cloning", "reward-weighted"):
            with self.assertRaisesRegex(ValueError, "requires objective='ppo'"):
                TransformerTrainingConfig(window_size=1, objective=objective, ppo_target_mode="gae")


if __name__ == "__main__":
    unittest.main()
