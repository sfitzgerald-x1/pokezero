"""Pure-Python masked linear policy baseline."""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
import hashlib
import inspect
import json
import math
from os import PathLike
from pathlib import Path
import random
from time import perf_counter
from typing import Any, Iterable, Literal, Mapping, Sequence

from .actions import ACTION_COUNT, ACTION_SCHEMA_VERSION
from .dataset import TrajectoryDatasetConfig, TrajectoryExample, iter_training_examples
from .observation import OBSERVATION_SCHEMA_VERSION, PokeZeroObservationV0
from .policy import PolicyDecision, legal_action_indices

LINEAR_POLICY_SCHEMA_VERSION = "pokezero.linear_policy.v4"
LINEAR_FEATURE_SCHEMA_VERSION = "pokezero.linear_features.v2"
LINEAR_FEATURE_FINGERPRINT_VERSION = "pokezero.linear_feature_fingerprint.v1"
LinearTrainingObjective = Literal["behavior-cloning", "reward-weighted"]
ALL_ACTIONS_LEGAL_MASK = tuple(True for _ in range(ACTION_COUNT))
_LINEAR_FEATURE_SOURCE_NAMES = (
    "features_from_example",
    "features_from_observation_window",
    "features_from_window",
    "_add_hashed_feature",
    "_hash_feature",
    "_sequence",
    "_zeros_like",
)


@lru_cache(maxsize=1)
def linear_feature_fingerprint() -> str:
    """Return a content-derived fingerprint for the linear feature extractor."""

    encoded = json.dumps(
        _linear_feature_fingerprint_payload(),
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _linear_feature_fingerprint_payload() -> dict[str, Any]:
    return {
        "fingerprint_version": LINEAR_FEATURE_FINGERPRINT_VERSION,
        "action_count": ACTION_COUNT,
        "action_schema_version": ACTION_SCHEMA_VERSION,
        "feature_schema_version": LINEAR_FEATURE_SCHEMA_VERSION,
        "observation_schema_version": OBSERVATION_SCHEMA_VERSION,
        "sources": {
            name: _callable_fingerprint_source(globals()[name])
            for name in _LINEAR_FEATURE_SOURCE_NAMES
        },
    }


def _callable_fingerprint_source(function: Any) -> str:
    try:
        return inspect.getsource(function)
    except OSError as exc:
        raise RuntimeError(
            "Linear feature fingerprint requires source files for the feature extractor."
        ) from exc


@dataclass(frozen=True)
class LinearPolicyModel:
    policy_id: str
    feature_count: int
    window_size: int
    weights: tuple[tuple[float, ...], ...]
    opponent_weights: tuple[tuple[float, ...], ...] = ()
    action_schema_version: str = ACTION_SCHEMA_VERSION
    observation_schema_version: str = OBSERVATION_SCHEMA_VERSION
    feature_schema_version: str = LINEAR_FEATURE_SCHEMA_VERSION
    feature_fingerprint: str = field(default_factory=linear_feature_fingerprint)

    def __post_init__(self) -> None:
        if self.action_schema_version != ACTION_SCHEMA_VERSION:
            raise ValueError(f"Unsupported action schema version: {self.action_schema_version!r}.")
        if self.observation_schema_version != OBSERVATION_SCHEMA_VERSION:
            raise ValueError(f"Unsupported observation schema version: {self.observation_schema_version!r}.")
        if self.feature_schema_version != LINEAR_FEATURE_SCHEMA_VERSION:
            raise ValueError(f"Unsupported linear feature schema version: {self.feature_schema_version!r}.")
        if self.feature_fingerprint != linear_feature_fingerprint():
            raise ValueError(f"Unsupported linear feature fingerprint: {self.feature_fingerprint!r}.")
        if self.feature_count <= 1:
            raise ValueError("feature_count must be greater than 1.")
        if self.window_size <= 0:
            raise ValueError("window_size must be positive.")
        if not self.opponent_weights:
            object.__setattr__(
                self,
                "opponent_weights",
                tuple(tuple(0.0 for _ in range(self.feature_count)) for _ in range(ACTION_COUNT)),
            )
        if len(self.weights) != ACTION_COUNT:
            raise ValueError(f"weights must contain {ACTION_COUNT} action rows.")
        if len(self.opponent_weights) != ACTION_COUNT:
            raise ValueError(f"opponent_weights must contain {ACTION_COUNT} action rows.")
        for row in (*self.weights, *self.opponent_weights):
            if len(row) != self.feature_count:
                raise ValueError("each weight row must match feature_count.")

    @classmethod
    def initialized(
        cls,
        *,
        feature_count: int,
        window_size: int,
        policy_id: str = "linear-softmax",
    ) -> "LinearPolicyModel":
        return cls(
            policy_id=policy_id,
            feature_count=feature_count,
            window_size=window_size,
            weights=tuple(tuple(0.0 for _ in range(feature_count)) for _ in range(ACTION_COUNT)),
            opponent_weights=tuple(tuple(0.0 for _ in range(feature_count)) for _ in range(ACTION_COUNT)),
        )

    def logits(self, features: Mapping[int, float]) -> tuple[float, ...]:
        return tuple(_dot(row, features) for row in self.weights)

    def action_probabilities(
        self,
        features: Mapping[int, float],
        legal_action_mask: Sequence[bool],
    ) -> tuple[float, ...]:
        legal = legal_action_indices(legal_action_mask)
        logits = self.logits(features)
        max_logit = max(logits[action_index] for action_index in legal)
        exp_by_action = {
            action_index: math.exp(logits[action_index] - max_logit)
            for action_index in legal
        }
        denominator = sum(exp_by_action.values())
        return tuple(exp_by_action.get(action_index, 0.0) / denominator for action_index in range(ACTION_COUNT))

    def predict_action(
        self,
        features: Mapping[int, float],
        legal_action_mask: Sequence[bool],
    ) -> int:
        probabilities = self.action_probabilities(features, legal_action_mask)
        legal = legal_action_indices(legal_action_mask)
        return max(legal, key=lambda action_index: (probabilities[action_index], -action_index))

    def opponent_action_probabilities(self, features: Mapping[int, float]) -> tuple[float, ...]:
        return _probabilities_from_weights(self.opponent_weights, features, ALL_ACTIONS_LEGAL_MASK)

    def predict_opponent_action(self, features: Mapping[int, float]) -> int:
        probabilities = self.opponent_action_probabilities(features)
        return max(range(ACTION_COUNT), key=lambda action_index: (probabilities[action_index], -action_index))

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": LINEAR_POLICY_SCHEMA_VERSION,
            "policy_id": self.policy_id,
            "action_schema_version": self.action_schema_version,
            "observation_schema_version": self.observation_schema_version,
            "feature_schema_version": self.feature_schema_version,
            "feature_fingerprint": self.feature_fingerprint,
            "feature_count": self.feature_count,
            "window_size": self.window_size,
            "weights": [list(row) for row in self.weights],
            "opponent_weights": [list(row) for row in self.opponent_weights],
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "LinearPolicyModel":
        if payload.get("schema_version") != LINEAR_POLICY_SCHEMA_VERSION:
            raise ValueError(f"Unsupported linear policy schema: {payload.get('schema_version')!r}.")
        return cls(
            policy_id=str(payload["policy_id"]),
            action_schema_version=_required_str(payload, "action_schema_version"),
            observation_schema_version=_required_str(payload, "observation_schema_version"),
            feature_schema_version=_required_str(payload, "feature_schema_version"),
            feature_fingerprint=_required_str(payload, "feature_fingerprint"),
            feature_count=int(payload["feature_count"]),
            window_size=int(payload["window_size"]),
            weights=tuple(tuple(float(value) for value in row) for row in _sequence(payload["weights"])),
            opponent_weights=tuple(
                tuple(float(value) for value in row)
                for row in _sequence(payload["opponent_weights"])
            ),
        )


@dataclass(frozen=True)
class LinearTrainingConfig:
    feature_count: int = 131_072
    window_size: int = 1
    discount: float = 1.0
    capped_terminal_value: float = 0.0
    objective: LinearTrainingObjective = "behavior-cloning"
    epochs: int = 1
    learning_rate: float = 0.05
    opponent_action_loss_weight: float = 0.0
    l2: float = 0.0
    shuffle_buffer_size: int = 1024
    shuffle_seed: int = 1
    max_examples: int | None = None
    policy_id: str = "linear-softmax"

    def __post_init__(self) -> None:
        if self.feature_count <= 1:
            raise ValueError("feature_count must be greater than 1.")
        if self.window_size <= 0:
            raise ValueError("window_size must be positive.")
        if not 0.0 <= self.discount <= 1.0:
            raise ValueError("discount must be between 0 and 1.")
        if not -1.0 <= self.capped_terminal_value <= 0.0:
            raise ValueError("capped_terminal_value must be between -1 and 0.")
        if self.objective not in ("behavior-cloning", "reward-weighted"):
            raise ValueError("objective must be behavior-cloning or reward-weighted.")
        if self.epochs <= 0:
            raise ValueError("epochs must be positive.")
        if self.learning_rate <= 0.0:
            raise ValueError("learning_rate must be positive.")
        if self.opponent_action_loss_weight < 0.0:
            raise ValueError("opponent_action_loss_weight must be non-negative.")
        if self.l2 < 0.0:
            raise ValueError("l2 must be non-negative.")
        if self.shuffle_buffer_size < 0:
            raise ValueError("shuffle_buffer_size must be non-negative.")
        if self.max_examples is not None and self.max_examples <= 0:
            raise ValueError("max_examples must be positive when set.")


@dataclass(frozen=True)
class LinearEpochMetrics:
    epoch: int
    examples: int
    loss: float
    accuracy: float
    elapsed_seconds: float
    opponent_examples: int = 0
    opponent_loss: float | None = None
    opponent_accuracy: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "epoch": self.epoch,
            "examples": self.examples,
            "loss": self.loss,
            "accuracy": self.accuracy,
            "elapsed_seconds": self.elapsed_seconds,
            "opponent_examples": self.opponent_examples,
            "opponent_loss": self.opponent_loss,
            "opponent_accuracy": self.opponent_accuracy,
        }


@dataclass(frozen=True)
class LinearTrainingResult:
    model: LinearPolicyModel
    config: LinearTrainingConfig
    epochs: tuple[LinearEpochMetrics, ...]
    validation_metrics: LinearEvaluationMetrics | None = None

    @property
    def final_metrics(self) -> LinearEpochMetrics:
        return self.epochs[-1]


@dataclass(frozen=True)
class LinearEvaluationMetrics:
    examples: int
    loss: float
    accuracy: float
    elapsed_seconds: float
    opponent_examples: int = 0
    opponent_loss: float | None = None
    opponent_accuracy: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "examples": self.examples,
            "loss": self.loss,
            "accuracy": self.accuracy,
            "elapsed_seconds": self.elapsed_seconds,
            "opponent_examples": self.opponent_examples,
            "opponent_loss": self.opponent_loss,
            "opponent_accuracy": self.opponent_accuracy,
        }


@dataclass
class LinearSoftmaxPolicy:
    model: LinearPolicyModel
    deterministic: bool = True
    exploration_epsilon: float = 0.0
    sampling_temperature: float = 1.0
    policy_id: str | None = None
    _history_by_player: dict[str, list[PokeZeroObservationV0]] = field(default_factory=dict, init=False)

    def __post_init__(self) -> None:
        if self.policy_id is None:
            self.policy_id = self.model.policy_id
        if not 0.0 <= self.exploration_epsilon <= 1.0:
            raise ValueError("exploration_epsilon must be between 0 and 1.")
        if self.sampling_temperature <= 0.0:
            raise ValueError("sampling_temperature must be positive.")

    def reset(self) -> None:
        self._history_by_player.clear()

    def select_action(
        self,
        observation: PokeZeroObservationV0,
        *,
        rng: random.Random,
    ) -> PolicyDecision:
        player_key = _observation_player_key(observation)
        history = self._history_by_player.setdefault(player_key, [])
        history.append(observation)
        features = features_from_observation_window(
            history[-self.model.window_size :],
            window_size=self.model.window_size,
            feature_count=self.model.feature_count,
        )
        probabilities = _probabilities_from_weights(
            self.model.weights,
            features,
            observation.legal_action_mask,
            temperature=self.sampling_temperature,
        )
        legal = legal_action_indices(observation.legal_action_mask)
        greedy_action = max(legal, key=lambda index: (probabilities[index], -index))
        random_exploration = self.exploration_epsilon and rng.random() < self.exploration_epsilon
        if random_exploration:
            action_index = rng.choice(legal)
        elif self.deterministic:
            action_index = greedy_action
        else:
            action_index = _sample_action(probabilities, legal, rng)
        action_probability = _behavior_probability(
            action_index=action_index,
            probabilities=probabilities,
            legal=legal,
            deterministic=self.deterministic,
            greedy_action=greedy_action,
            exploration_epsilon=self.exploration_epsilon,
        )
        return PolicyDecision(
            action_index=action_index,
            policy_id=str(self.policy_id),
            action_probability=action_probability,
            metadata={
                "policy_family": "linear-softmax",
                "deterministic": self.deterministic,
                "exploration_epsilon": self.exploration_epsilon,
                "sampling_temperature": self.sampling_temperature,
            },
        )


def train_linear_policy(
    paths: str | PathLike[str] | Path | Iterable[str | PathLike[str] | Path],
    *,
    config: LinearTrainingConfig | None = None,
    validation_paths: str | PathLike[str] | Path | Iterable[str | PathLike[str] | Path] | None = None,
    initial_model: LinearPolicyModel | None = None,
) -> LinearTrainingResult:
    training_config = config or LinearTrainingConfig()
    if initial_model is not None:
        if initial_model.feature_count != training_config.feature_count:
            raise ValueError("initial_model feature_count must match the training config.")
        if initial_model.window_size != training_config.window_size:
            raise ValueError("initial_model window_size must match the training config.")
        weights = [list(row) for row in initial_model.weights]
        opponent_weights = [list(row) for row in initial_model.opponent_weights]
    else:
        initialized = LinearPolicyModel.initialized(
            feature_count=training_config.feature_count,
            window_size=training_config.window_size,
            policy_id=training_config.policy_id,
        )
        weights = [list(row) for row in initialized.weights]
        opponent_weights = [list(row) for row in initialized.opponent_weights]
    epoch_metrics = []
    dataset_config = TrajectoryDatasetConfig(
        window_size=training_config.window_size,
        discount=training_config.discount,
        capped_terminal_value=training_config.capped_terminal_value,
    )

    for epoch in range(1, training_config.epochs + 1):
        start = perf_counter()
        total_loss = 0.0
        correct = 0
        examples = 0
        opponent_total_loss = 0.0
        opponent_correct = 0
        opponent_examples = 0
        for example in _iter_epoch_examples(
            paths,
            dataset_config=dataset_config,
            shuffle_buffer_size=training_config.shuffle_buffer_size,
            rng=random.Random(training_config.shuffle_seed + epoch - 1),
        ):
            features = features_from_example(example, feature_count=training_config.feature_count)
            probabilities = _probabilities_from_weights(weights, features, example.legal_action_mask)
            total_loss += -math.log(max(probabilities[example.action_index], 1e-12))
            if max(legal_action_indices(example.legal_action_mask), key=lambda index: (probabilities[index], -index)) == example.action_index:
                correct += 1
            _sgd_update(
                weights=weights,
                features=features,
                probabilities=probabilities,
                legal_action_mask=example.legal_action_mask,
                target_action=example.action_index,
                gradient_weight=_gradient_weight(example, training_config.objective),
                learning_rate=training_config.learning_rate,
                l2=training_config.l2,
            )
            if example.opponent_action_index is not None and training_config.opponent_action_loss_weight:
                opponent_probabilities = _probabilities_from_weights(
                    opponent_weights,
                    features,
                    ALL_ACTIONS_LEGAL_MASK,
                )
                opponent_target = int(example.opponent_action_index)
                opponent_total_loss += -math.log(max(opponent_probabilities[opponent_target], 1e-12))
                if max(range(ACTION_COUNT), key=lambda index: (opponent_probabilities[index], -index)) == opponent_target:
                    opponent_correct += 1
                _sgd_update(
                    weights=opponent_weights,
                    features=features,
                    probabilities=opponent_probabilities,
                    legal_action_mask=ALL_ACTIONS_LEGAL_MASK,
                    target_action=opponent_target,
                    gradient_weight=training_config.opponent_action_loss_weight,
                    learning_rate=training_config.learning_rate,
                    l2=training_config.l2,
                )
                opponent_examples += 1
            examples += 1
            if training_config.max_examples is not None and examples >= training_config.max_examples:
                break
        if examples == 0:
            raise ValueError("training data produced no examples.")
        epoch_metrics.append(
            LinearEpochMetrics(
                epoch=epoch,
                examples=examples,
                loss=total_loss / examples,
                accuracy=correct / examples,
                elapsed_seconds=perf_counter() - start,
                opponent_examples=opponent_examples,
                opponent_loss=(opponent_total_loss / opponent_examples) if opponent_examples else None,
                opponent_accuracy=(opponent_correct / opponent_examples) if opponent_examples else None,
            )
        )

    model = LinearPolicyModel(
        policy_id=training_config.policy_id,
        feature_count=training_config.feature_count,
        window_size=training_config.window_size,
        weights=tuple(tuple(row) for row in weights),
        opponent_weights=tuple(tuple(row) for row in opponent_weights),
    )
    validation_metrics = None
    if validation_paths is not None:
        validation_metrics = evaluate_linear_policy(
            validation_paths,
            model,
            discount=training_config.discount,
            capped_terminal_value=training_config.capped_terminal_value,
        )

    return LinearTrainingResult(
        model=model,
        config=training_config,
        epochs=tuple(epoch_metrics),
        validation_metrics=validation_metrics,
    )


def evaluate_linear_policy(
    paths: str | PathLike[str] | Path | Iterable[str | PathLike[str] | Path],
    model: LinearPolicyModel,
    *,
    discount: float = 1.0,
    capped_terminal_value: float = 0.0,
    max_examples: int | None = None,
) -> LinearEvaluationMetrics:
    if max_examples is not None and max_examples <= 0:
        raise ValueError("max_examples must be positive when set.")
    dataset_config = TrajectoryDatasetConfig(
        window_size=model.window_size,
        discount=discount,
        capped_terminal_value=capped_terminal_value,
    )
    start = perf_counter()
    total_loss = 0.0
    correct = 0
    examples = 0
    opponent_total_loss = 0.0
    opponent_correct = 0
    opponent_examples = 0

    for example in iter_training_examples(paths, config=dataset_config):
        features = features_from_example(example, feature_count=model.feature_count)
        probabilities = model.action_probabilities(features, example.legal_action_mask)
        total_loss += -math.log(max(probabilities[example.action_index], 1e-12))
        if model.predict_action(features, example.legal_action_mask) == example.action_index:
            correct += 1
        if example.opponent_action_index is not None:
            opponent_probabilities = model.opponent_action_probabilities(features)
            opponent_target = int(example.opponent_action_index)
            opponent_total_loss += -math.log(max(opponent_probabilities[opponent_target], 1e-12))
            if model.predict_opponent_action(features) == opponent_target:
                opponent_correct += 1
            opponent_examples += 1
        examples += 1
        if max_examples is not None and examples >= max_examples:
            break

    if examples == 0:
        raise ValueError("evaluation data produced no examples.")
    return LinearEvaluationMetrics(
        examples=examples,
        loss=total_loss / examples,
        accuracy=correct / examples,
        elapsed_seconds=perf_counter() - start,
        opponent_examples=opponent_examples,
        opponent_loss=(opponent_total_loss / opponent_examples) if opponent_examples else None,
        opponent_accuracy=(opponent_correct / opponent_examples) if opponent_examples else None,
    )


def features_from_example(
    example: TrajectoryExample,
    *,
    feature_count: int,
) -> dict[int, float]:
    return features_from_window(
        categorical_ids=example.categorical_ids,
        numeric_features=example.numeric_features,
        token_type_ids=example.token_type_ids,
        attention_mask=example.attention_mask,
        history_mask=example.history_mask,
        feature_count=feature_count,
    )


def features_from_observation_window(
    observations: Sequence[PokeZeroObservationV0],
    *,
    window_size: int,
    feature_count: int,
) -> dict[int, float]:
    if window_size <= 0:
        raise ValueError("window_size must be positive.")
    if not observations:
        raise ValueError("observations must contain at least one item.")
    observation = observations[-1]
    padding_count = max(0, window_size - len(observations))
    window = tuple(observations[-window_size:])
    categorical_padding = _zeros_like(observation.categorical_ids)
    numeric_padding = _zeros_like(observation.numeric_features)
    token_type_padding = _zeros_like(observation.token_type_ids)
    attention_padding = _zeros_like(observation.attention_mask)
    return features_from_window(
        categorical_ids=tuple([categorical_padding] * padding_count)
        + tuple(item.categorical_ids for item in window),
        numeric_features=tuple([numeric_padding] * padding_count)
        + tuple(item.numeric_features for item in window),
        token_type_ids=tuple([token_type_padding] * padding_count)
        + tuple(item.token_type_ids for item in window),
        attention_mask=tuple([attention_padding] * padding_count)
        + tuple(item.attention_mask for item in window),
        history_mask=tuple(False for _ in range(padding_count)) + tuple(True for _ in window),
        feature_count=feature_count,
    )


def features_from_window(
    *,
    categorical_ids: Sequence[Any],
    numeric_features: Sequence[Any],
    token_type_ids: Sequence[Any],
    attention_mask: Sequence[Any],
    history_mask: Sequence[bool],
    feature_count: int,
) -> dict[int, float]:
    if feature_count <= 1:
        raise ValueError("feature_count must be greater than 1.")
    features: dict[int, float] = {0: 1.0}
    for history_index, present in enumerate(history_mask):
        if not present:
            continue
        _add_hashed_feature(features, feature_count, f"h:{history_index}", 1.0)
        for token_index, row in enumerate(_sequence(categorical_ids[history_index])):
            for column_index, value in enumerate(_sequence(row)):
                categorical_value = int(value)
                if categorical_value:
                    _add_hashed_feature(
                        features,
                        feature_count,
                        f"c:{history_index}:{token_index}:{column_index}:{categorical_value}",
                        1.0,
                    )
        for token_index, row in enumerate(_sequence(numeric_features[history_index])):
            for column_index, value in enumerate(_sequence(row)):
                numeric_value = float(value)
                if numeric_value:
                    _add_hashed_feature(features, feature_count, f"n:{history_index}:{token_index}:{column_index}", numeric_value)
        for token_index, value in enumerate(_sequence(token_type_ids[history_index])):
            _add_hashed_feature(features, feature_count, f"t:{history_index}:{token_index}:{int(value)}", 1.0)
        for token_index, value in enumerate(_sequence(attention_mask[history_index])):
            if bool(value):
                _add_hashed_feature(features, feature_count, f"a:{history_index}:{token_index}", 1.0)
    return features


def save_linear_model(path: str | PathLike[str] | Path, model: LinearPolicyModel) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(model.to_dict(), indent=2, sort_keys=True), encoding="utf-8")


def load_linear_model(path: str | PathLike[str] | Path) -> LinearPolicyModel:
    return LinearPolicyModel.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


def _required_str(payload: Mapping[str, Any], key: str) -> str:
    try:
        return str(payload[key])
    except KeyError as exc:
        raise ValueError(f"linear policy checkpoint missing required field: {key}.") from exc


def _iter_epoch_examples(
    paths: str | PathLike[str] | Path | Iterable[str | PathLike[str] | Path],
    *,
    dataset_config: TrajectoryDatasetConfig,
    shuffle_buffer_size: int,
    rng: random.Random,
) -> Iterable[TrajectoryExample]:
    examples = iter_training_examples(paths, config=dataset_config)
    if shuffle_buffer_size == 0:
        yield from examples
        return

    buffer: list[TrajectoryExample] = []
    for example in examples:
        buffer.append(example)
        if len(buffer) < shuffle_buffer_size:
            continue
        index = rng.randrange(len(buffer))
        yield buffer.pop(index)
    while buffer:
        index = rng.randrange(len(buffer))
        yield buffer.pop(index)


def _sgd_update(
    *,
    weights: list[list[float]],
    features: Mapping[int, float],
    probabilities: Sequence[float],
    legal_action_mask: Sequence[bool],
    target_action: int,
    gradient_weight: float,
    learning_rate: float,
    l2: float,
) -> None:
    if gradient_weight == 0.0:
        return
    for action_index in legal_action_indices(legal_action_mask):
        error = gradient_weight * (probabilities[action_index] - (1.0 if action_index == target_action else 0.0))
        row = weights[action_index]
        for feature_index, feature_value in features.items():
            gradient = error * feature_value
            # Sparse approximation: shrink only active feature buckets.
            if l2:
                gradient += l2 * row[feature_index]
            row[feature_index] -= learning_rate * gradient


def _gradient_weight(
    example: TrajectoryExample,
    objective: LinearTrainingObjective,
) -> float:
    if objective == "behavior-cloning":
        return 1.0
    if objective == "reward-weighted":
        if example.terminal_capped:
            return float(example.return_value)
        return max(0.0, float(example.return_value))
    raise ValueError(f"Unsupported objective: {objective!r}.")


def _probabilities_from_weights(
    weights: Sequence[Sequence[float]],
    features: Mapping[int, float],
    legal_action_mask: Sequence[bool],
    *,
    temperature: float = 1.0,
) -> tuple[float, ...]:
    if temperature <= 0.0:
        raise ValueError("temperature must be positive.")
    legal = legal_action_indices(legal_action_mask)
    logits = tuple(_dot(row, features) / temperature for row in weights)
    max_logit = max(logits[action_index] for action_index in legal)
    exp_by_action = {
        action_index: math.exp(logits[action_index] - max_logit)
        for action_index in legal
    }
    denominator = sum(exp_by_action.values())
    return tuple(exp_by_action.get(action_index, 0.0) / denominator for action_index in range(ACTION_COUNT))


def _behavior_probability(
    *,
    action_index: int,
    probabilities: Sequence[float],
    legal: Sequence[int],
    deterministic: bool,
    greedy_action: int,
    exploration_epsilon: float,
) -> float:
    random_component = exploration_epsilon / len(legal) if legal else 0.0
    if deterministic:
        policy_mass = 1.0 if action_index == greedy_action else 0.0
    else:
        policy_mass = probabilities[action_index]
    return random_component + ((1.0 - exploration_epsilon) * policy_mass)


def _dot(row: Sequence[float], features: Mapping[int, float]) -> float:
    return sum(row[index] * value for index, value in features.items())


def _sample_action(probabilities: Sequence[float], legal: Sequence[int], rng: random.Random) -> int:
    threshold = rng.random()
    cumulative = 0.0
    for action_index in legal:
        cumulative += probabilities[action_index]
        if threshold <= cumulative:
            return action_index
    return legal[-1]


def _observation_player_key(observation: PokeZeroObservationV0) -> str:
    if observation.perspective is None:
        return "_default"
    return observation.perspective.player_id


def _add_hashed_feature(
    features: dict[int, float],
    feature_count: int,
    key: str,
    value: float,
) -> None:
    index = _hash_feature(key, feature_count)
    features[index] = features.get(index, 0.0) + value


def _hash_feature(key: str, feature_count: int) -> int:
    digest = hashlib.blake2b(key.encode("utf-8"), digest_size=8).digest()
    return 1 + (int.from_bytes(digest, "big") % (feature_count - 1))


def _sequence(value: Any) -> tuple[Any, ...]:
    if isinstance(value, list | tuple):
        return tuple(value)
    return tuple(value)


def _zeros_like(value: Any) -> Any:
    if isinstance(value, bool):
        return False
    if isinstance(value, int):
        return 0
    if isinstance(value, float):
        return 0.0
    return tuple(_zeros_like(item) for item in value)
