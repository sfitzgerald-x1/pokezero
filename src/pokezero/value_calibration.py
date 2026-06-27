"""Value-head calibration metrics for transformer checkpoints."""

from __future__ import annotations

from dataclasses import dataclass
import math
from os import PathLike
from pathlib import Path
from typing import Iterable

from .dataset import TrajectoryDatasetConfig, iter_training_batches
from .neural_policy import (
    TransformerTrainingResult,
    ValueCalibrationTransform,
    require_torch,
    training_batch_to_torch,
)

PathInput = str | PathLike[str] | Path
VALUE_SELECTION_METRICS = (
    "mae",
    "mse",
    "expected_calibration_error",
    "sign_accuracy",
    "pearson_correlation",
    "abs_bias",
)


@dataclass(frozen=True)
class ValueCalibrationBin:
    lower: float
    upper: float
    count: int
    mean_prediction: float
    mean_return: float

    @property
    def calibration_error(self) -> float:
        return abs(self.mean_prediction - self.mean_return)

    def to_dict(self) -> dict[str, object]:
        return {
            "lower": self.lower,
            "upper": self.upper,
            "count": self.count,
            "mean_prediction": self.mean_prediction,
            "mean_return": self.mean_return,
            "calibration_error": self.calibration_error,
        }


@dataclass(frozen=True)
class ValueCalibrationReport:
    examples: int
    mse: float
    mae: float
    bias: float
    sign_accuracy: float
    expected_calibration_error: float
    bins: tuple[ValueCalibrationBin, ...]
    pearson_correlation: float | None = None
    slices: tuple["ValueCalibrationSlice", ...] = ()

    def to_dict(self) -> dict[str, object]:
        return {
            "examples": self.examples,
            "mse": self.mse,
            "mae": self.mae,
            "bias": self.bias,
            "sign_accuracy": self.sign_accuracy,
            "expected_calibration_error": self.expected_calibration_error,
            "pearson_correlation": self.pearson_correlation,
            "bins": [bin_result.to_dict() for bin_result in self.bins],
            "slices": [slice_result.to_dict() for slice_result in self.slices],
        }


@dataclass(frozen=True)
class ValueCalibrationSlice:
    name: str
    examples: int
    mse: float
    mae: float
    bias: float
    sign_accuracy: float
    expected_calibration_error: float
    pearson_correlation: float | None = None
    sign_accuracy_applicable: bool = True

    @classmethod
    def from_report(
        cls,
        *,
        name: str,
        report: ValueCalibrationReport,
        pearson_correlation_applicable: bool = True,
        sign_accuracy_applicable: bool = True,
    ) -> "ValueCalibrationSlice":
        return cls(
            name=name,
            examples=report.examples,
            mse=report.mse,
            mae=report.mae,
            bias=report.bias,
            sign_accuracy=report.sign_accuracy,
            expected_calibration_error=report.expected_calibration_error,
            pearson_correlation=report.pearson_correlation if pearson_correlation_applicable else None,
            sign_accuracy_applicable=sign_accuracy_applicable,
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "examples": self.examples,
            "mse": self.mse,
            "mae": self.mae,
            "bias": self.bias,
            "sign_accuracy": self.sign_accuracy,
            "expected_calibration_error": self.expected_calibration_error,
            "pearson_correlation": self.pearson_correlation,
            "sign_accuracy_applicable": self.sign_accuracy_applicable,
        }


def evaluate_value_calibration(
    *,
    model: object,
    training_result: TransformerTrainingResult,
    paths: PathInput | Iterable[PathInput],
    batch_size: int = 128,
    bins: int = 10,
    device: str | object | None = None,
) -> ValueCalibrationReport:
    if batch_size <= 0:
        raise ValueError("batch_size must be positive.")
    if bins <= 0:
        raise ValueError("bins must be positive.")
    torch_module = require_torch()
    was_training = getattr(model, "training", None)
    if hasattr(model, "eval"):
        model.eval()
    if device is not None and hasattr(model, "to"):
        model.to(device)
    dataset_config = _trajectory_dataset_config_from_training_result(training_result)
    # Calibration reports outcome-return quality; PPO GAE targets are bootstrapped training labels.
    totals = _ValueCalibrationTotals(bin_count=bins)
    slice_totals = _ValueCalibrationSliceTotals(bin_count=bins)
    try:
        with torch_module.no_grad():
            for batch in iter_training_batches(paths, batch_size=batch_size, config=dataset_config):
                tensors = training_batch_to_torch(batch, device=device)
                output = model(
                    categorical_ids=tensors["categorical_ids"],
                    numeric_features=tensors["numeric_features"],
                    token_type_ids=tensors["token_type_ids"],
                    attention_mask=tensors["attention_mask"],
                    history_mask=tensors["history_mask"],
                )
                transform = getattr(training_result, "value_calibration_transform", None)
                predictions = tuple(
                    _apply_value_calibration_transform(float(value), transform)
                    for value in output.value.detach().cpu().tolist()
                )
                returns = tuple(float(value) for value in tensors["returns"].detach().cpu().tolist())
                totals.add(predictions=predictions, returns=returns)
                slice_totals.add(
                    predictions=predictions,
                    returns=returns,
                    turn_indices=tuple(int(value) for value in batch.turn_indices),
                    terminal_capped=tuple(bool(value) for value in batch.terminal_capped),
                )
    finally:
        if was_training is not None and hasattr(model, "train"):
            model.train(bool(was_training))
    return totals.to_report(slices=slice_totals.to_slices())


def fit_value_calibration_transform(
    *,
    model: object,
    training_result: TransformerTrainingResult,
    paths: PathInput | Iterable[PathInput],
    batch_size: int = 128,
    device: str | object | None = None,
) -> ValueCalibrationTransform:
    if batch_size <= 0:
        raise ValueError("batch_size must be positive.")
    torch_module = require_torch()
    was_training = getattr(model, "training", None)
    if hasattr(model, "eval"):
        model.eval()
    if device is not None and hasattr(model, "to"):
        model.to(device)
    dataset_config = _trajectory_dataset_config_from_training_result(training_result)
    # Fit transforms against realized outcome returns, not bootstrapped PPO GAE targets.
    totals = _AffineFitTotals()
    try:
        with torch_module.no_grad():
            for batch in iter_training_batches(paths, batch_size=batch_size, config=dataset_config):
                tensors = training_batch_to_torch(batch, device=device)
                output = model(
                    categorical_ids=tensors["categorical_ids"],
                    numeric_features=tensors["numeric_features"],
                    token_type_ids=tensors["token_type_ids"],
                    attention_mask=tensors["attention_mask"],
                    history_mask=tensors["history_mask"],
                )
                predictions = tuple(float(value) for value in output.value.detach().cpu().tolist())
                returns = tuple(float(value) for value in tensors["returns"].detach().cpu().tolist())
                totals.add(predictions=predictions, returns=returns)
    finally:
        if was_training is not None and hasattr(model, "train"):
            model.train(bool(was_training))
    return totals.to_transform()


def fit_affine_value_calibration_transform(
    *,
    predictions: tuple[float, ...],
    returns: tuple[float, ...],
) -> ValueCalibrationTransform:
    totals = _AffineFitTotals()
    totals.add(predictions=predictions, returns=returns)
    return totals.to_transform()


def value_selection_metric_value(report: ValueCalibrationReport, metric: str) -> float:
    if metric == "mae":
        return _finite_value_selection_metric(float(report.mae), metric)
    if metric == "mse":
        return _finite_value_selection_metric(float(report.mse), metric)
    if metric == "expected_calibration_error":
        return _finite_value_selection_metric(float(report.expected_calibration_error), metric)
    if metric == "sign_accuracy":
        return _finite_value_selection_metric(float(report.sign_accuracy), metric)
    if metric == "pearson_correlation":
        if report.pearson_correlation is None:
            raise ValueError("pearson_correlation value selection requires non-constant predictions and returns.")
        return _finite_value_selection_metric(float(report.pearson_correlation), metric)
    if metric == "abs_bias":
        return _finite_value_selection_metric(abs(float(report.bias)), metric)
    raise ValueError(f"unsupported value selection metric: {metric!r}.")


def _finite_value_selection_metric(value: float, metric: str) -> float:
    if not math.isfinite(value):
        raise ValueError(f"{metric} value selection requires a finite metric value.")
    return value


def value_selection_metric_direction(metric: str) -> str:
    if metric not in VALUE_SELECTION_METRICS:
        raise ValueError(f"unsupported value selection metric: {metric!r}.")
    return "max" if metric in {"sign_accuracy", "pearson_correlation"} else "min"


def value_selection_score(metric_value: float, metric: str) -> float:
    return metric_value if value_selection_metric_direction(metric) == "max" else -metric_value


def _apply_value_calibration_transform(value: float, transform: object) -> float:
    if isinstance(transform, ValueCalibrationTransform):
        return transform.apply(value)
    return float(value)


def _trajectory_dataset_config_from_training_result(
    training_result: TransformerTrainingResult,
) -> TrajectoryDatasetConfig:
    training_config = training_result.training_config
    return TrajectoryDatasetConfig(
        window_size=training_config.window_size,
        discount=training_config.discount,
        capped_terminal_value=training_config.capped_terminal_value,
        hp_delta_return_weight=training_config.hp_delta_return_weight,
        faint_delta_return_weight=training_config.faint_delta_return_weight,
        turn_penalty_after=training_config.turn_penalty_after,
        turn_penalty=training_config.turn_penalty,
    )


@dataclass
class _AffineFitTotals:
    examples: int = 0
    prediction_sum: float = 0.0
    return_sum: float = 0.0
    prediction_square_sum: float = 0.0
    prediction_return_sum: float = 0.0

    def add(self, *, predictions: tuple[float, ...], returns: tuple[float, ...]) -> None:
        if len(predictions) != len(returns):
            raise ValueError("predictions and returns must have the same length.")
        for prediction, target in zip(predictions, returns, strict=True):
            self.examples += 1
            self.prediction_sum += prediction
            self.return_sum += target
            self.prediction_square_sum += prediction * prediction
            self.prediction_return_sum += prediction * target

    def to_transform(self) -> ValueCalibrationTransform:
        if self.examples == 0:
            raise ValueError("calibration data produced no examples.")
        denominator = (self.examples * self.prediction_square_sum) - (self.prediction_sum * self.prediction_sum)
        if abs(denominator) <= 1e-12:
            return ValueCalibrationTransform(scale=0.0, bias=self.return_sum / self.examples)
        scale = ((self.examples * self.prediction_return_sum) - (self.prediction_sum * self.return_sum)) / denominator
        bias = (self.return_sum - (scale * self.prediction_sum)) / self.examples
        return ValueCalibrationTransform(scale=scale, bias=bias)


@dataclass
class _ValueCalibrationTotals:
    bin_count: int
    examples: int = 0
    squared_error: float = 0.0
    absolute_error: float = 0.0
    signed_error: float = 0.0
    sign_correct: int = 0
    prediction_mean: float = 0.0
    return_mean: float = 0.0
    prediction_m2: float = 0.0
    return_m2: float = 0.0
    prediction_return_coproduct: float = 0.0

    def __post_init__(self) -> None:
        self._bin_totals: list[_BinTotals] = [_BinTotals() for _ in range(self.bin_count)]

    def add(self, *, predictions: tuple[float, ...], returns: tuple[float, ...]) -> None:
        if len(predictions) != len(returns):
            raise ValueError("predictions and returns must have the same length.")
        for prediction, target in zip(predictions, returns, strict=True):
            error = prediction - target
            self._add_correlation_sample(prediction=prediction, target=target)
            self.squared_error += error * error
            self.absolute_error += abs(error)
            self.signed_error += error
            if _sign(prediction) == _sign(target):
                self.sign_correct += 1
            self._bin_totals[self._bin_index(prediction)].add(prediction=prediction, target=target)

    def to_report(self, *, slices: tuple[ValueCalibrationSlice, ...] = ()) -> ValueCalibrationReport:
        if self.examples == 0:
            raise ValueError("calibration data produced no examples.")
        bins = tuple(
            bin_total.to_bin(
                lower=-1.0 + (2.0 * index / self.bin_count),
                upper=-1.0 + (2.0 * (index + 1) / self.bin_count),
            )
            for index, bin_total in enumerate(self._bin_totals)
        )
        expected_calibration_error = sum(
            (bin_result.count / self.examples) * bin_result.calibration_error
            for bin_result in bins
            if bin_result.count
        )
        return ValueCalibrationReport(
            examples=self.examples,
            mse=self.squared_error / self.examples,
            mae=self.absolute_error / self.examples,
            bias=self.signed_error / self.examples,
            sign_accuracy=self.sign_correct / self.examples,
            expected_calibration_error=expected_calibration_error,
            bins=bins,
            pearson_correlation=_pearson_correlation(
                count=self.examples,
                prediction_m2=self.prediction_m2,
                return_m2=self.return_m2,
                prediction_return_coproduct=self.prediction_return_coproduct,
            ),
            slices=slices,
        )

    def _add_correlation_sample(self, *, prediction: float, target: float) -> None:
        # Centered online moments avoid cancellation when a collapsed value head emits near-constant values.
        self.examples += 1
        prediction_delta = prediction - self.prediction_mean
        return_delta = target - self.return_mean
        self.prediction_mean += prediction_delta / self.examples
        self.return_mean += return_delta / self.examples
        self.prediction_m2 += prediction_delta * (prediction - self.prediction_mean)
        self.return_m2 += return_delta * (target - self.return_mean)
        self.prediction_return_coproduct += prediction_delta * (target - self.return_mean)

    def _bin_index(self, prediction: float) -> int:
        clipped = min(1.0, max(-1.0, prediction))
        if clipped == 1.0:
            return self.bin_count - 1
        return int(((clipped + 1.0) / 2.0) * self.bin_count)


@dataclass
class _BinTotals:
    count: int = 0
    prediction_sum: float = 0.0
    return_sum: float = 0.0

    def add(self, *, prediction: float, target: float) -> None:
        self.count += 1
        self.prediction_sum += prediction
        self.return_sum += target

    def to_bin(self, *, lower: float, upper: float) -> ValueCalibrationBin:
        if self.count == 0:
            return ValueCalibrationBin(
                lower=lower,
                upper=upper,
                count=0,
                mean_prediction=0.0,
                mean_return=0.0,
            )
        return ValueCalibrationBin(
            lower=lower,
            upper=upper,
            count=self.count,
            mean_prediction=self.prediction_sum / self.count,
            mean_return=self.return_sum / self.count,
        )


@dataclass
class _ValueCalibrationSliceTotals:
    bin_count: int

    def __post_init__(self) -> None:
        self._totals_by_name: dict[str, _ValueCalibrationTotals] = {}

    def add(
        self,
        *,
        predictions: tuple[float, ...],
        returns: tuple[float, ...],
        turn_indices: tuple[int, ...],
        terminal_capped: tuple[bool, ...],
    ) -> None:
        lengths = {len(predictions), len(returns), len(turn_indices), len(terminal_capped)}
        if len(lengths) != 1:
            raise ValueError("slice inputs must have the same length.")
        for prediction, target, turn_index, capped in zip(
            predictions,
            returns,
            turn_indices,
            terminal_capped,
            strict=True,
        ):
            for name in _slice_names(return_value=target, turn_index=turn_index, terminal_capped=capped):
                self._totals_for(name).add(predictions=(prediction,), returns=(target,))

    def to_slices(self) -> tuple[ValueCalibrationSlice, ...]:
        slices: list[ValueCalibrationSlice] = []
        for name in _SLICE_ORDER:
            totals = self._totals_by_name.get(name)
            if totals is not None and totals.examples:
                slices.append(
                    ValueCalibrationSlice.from_report(
                        name=name,
                        report=totals.to_report(),
                        pearson_correlation_applicable=not name.startswith("return:"),
                        sign_accuracy_applicable=(name != "return:zero"),
                    )
                )
        return tuple(slices)

    def _totals_for(self, name: str) -> _ValueCalibrationTotals:
        totals = self._totals_by_name.get(name)
        if totals is None:
            totals = _ValueCalibrationTotals(bin_count=self.bin_count)
            self._totals_by_name[name] = totals
        return totals


_SLICE_ORDER = (
    "return:positive",
    "return:negative",
    "return:zero",
    "turn:early_0_9",
    "turn:mid_10_29",
    "turn:late_30_plus",
    "terminal:uncapped",
    "terminal:capped",
)


def _slice_names(*, return_value: float, turn_index: int, terminal_capped: bool) -> tuple[str, ...]:
    if return_value > 0.0:
        return_name = "return:positive"
    elif return_value < 0.0:
        return_name = "return:negative"
    else:
        return_name = "return:zero"

    if turn_index < 10:
        turn_name = "turn:early_0_9"
    elif turn_index < 30:
        turn_name = "turn:mid_10_29"
    else:
        turn_name = "turn:late_30_plus"

    terminal_name = "terminal:capped" if terminal_capped else "terminal:uncapped"
    return (return_name, turn_name, terminal_name)


def _sign(value: float) -> int:
    if value > 0.0:
        return 1
    if value < 0.0:
        return -1
    return 0


def _pearson_correlation(
    *,
    count: int,
    prediction_m2: float,
    return_m2: float,
    prediction_return_coproduct: float,
) -> float | None:
    if count <= 1:
        return None
    prediction_variance = prediction_m2 / count
    return_variance = return_m2 / count
    if prediction_variance <= 1e-12 or return_variance <= 1e-12:
        return None
    correlation = prediction_return_coproduct / math.sqrt(prediction_m2 * return_m2)
    return max(-1.0, min(1.0, correlation))
