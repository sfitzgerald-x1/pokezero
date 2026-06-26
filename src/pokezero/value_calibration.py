"""Value-head calibration metrics for transformer checkpoints."""

from __future__ import annotations

from dataclasses import dataclass
from os import PathLike
from pathlib import Path
from typing import Iterable

from .dataset import TrajectoryDatasetConfig, iter_training_batches
from .neural_policy import TransformerTrainingResult, require_torch, training_batch_to_torch

PathInput = str | PathLike[str] | Path


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

    def to_dict(self) -> dict[str, object]:
        return {
            "examples": self.examples,
            "mse": self.mse,
            "mae": self.mae,
            "bias": self.bias,
            "sign_accuracy": self.sign_accuracy,
            "expected_calibration_error": self.expected_calibration_error,
            "bins": [bin_result.to_dict() for bin_result in self.bins],
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
    if hasattr(model, "eval"):
        model.eval()
    if device is not None and hasattr(model, "to"):
        model.to(device)
    dataset_config = TrajectoryDatasetConfig(
        window_size=training_result.training_config.window_size,
        discount=training_result.training_config.discount,
        capped_terminal_value=training_result.training_config.capped_terminal_value,
    )
    totals = _ValueCalibrationTotals(bin_count=bins)
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
    return totals.to_report()


@dataclass
class _ValueCalibrationTotals:
    bin_count: int
    examples: int = 0
    squared_error: float = 0.0
    absolute_error: float = 0.0
    signed_error: float = 0.0
    sign_correct: int = 0

    def __post_init__(self) -> None:
        self._bin_totals: list[_BinTotals] = [_BinTotals() for _ in range(self.bin_count)]

    def add(self, *, predictions: tuple[float, ...], returns: tuple[float, ...]) -> None:
        if len(predictions) != len(returns):
            raise ValueError("predictions and returns must have the same length.")
        for prediction, target in zip(predictions, returns, strict=True):
            error = prediction - target
            self.examples += 1
            self.squared_error += error * error
            self.absolute_error += abs(error)
            self.signed_error += error
            if _sign(prediction) == _sign(target):
                self.sign_correct += 1
            self._bin_totals[self._bin_index(prediction)].add(prediction=prediction, target=target)

    def to_report(self) -> ValueCalibrationReport:
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
        )

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


def _sign(value: float) -> int:
    if value > 0.0:
        return 1
    if value < 0.0:
        return -1
    return 0
