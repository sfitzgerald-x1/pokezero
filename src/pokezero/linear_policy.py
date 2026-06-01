"""Pure-Python masked linear policy baseline."""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
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

LINEAR_POLICY_SCHEMA_VERSION = "pokezero.linear_policy.v2"
LINEAR_FEATURE_SCHEMA_VERSION = "pokezero.linear_features.v1"
LinearTrainingObjective = Literal["behavior-cloning", "reward-weighted"]


@dataclass(frozen=True)
class LinearPolicyModel:
    policy_id: str
    feature_count: int
    window_size: int
    weights: tuple[tuple[float, ...], ...]
    action_schema_version: str = ACTION_SCHEMA_VERSION
    observation_schema_version: str = OBSERVATION_SCHEMA_VERSION
    feature_schema_version: str = LINEAR_FEATURE_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.action_schema_version != ACTION_SCHEMA_VERSION:
            raise ValueError(f"Unsupported action schema version: {self.action_schema_version!r}.")
        if self.observation_schema_version != OBSERVATION_SCHEMA_VERSION:
            raise ValueError(f"Unsupported observation schema version: {self.observation_schema_version!r}.")
        if self.feature_schema_version != LINEAR_FEATURE_SCHEMA_VERSION:
            raise ValueError(f"Unsupported linear feature schema version: {self.feature_schema_version!r}.")
        if self.feature_count <= 1:
            raise ValueError("feature_count must be greater than 1.")
        if self.window_size <= 0:
            raise ValueError("window_size must be positive.")
        if len(self.weights) != ACTION_COUNT:
            raise ValueError(f"weights must contain {ACTION_COUNT} action rows.")
        for row in self.weights:
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

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": LINEAR_POLICY_SCHEMA_VERSION,
            "policy_id": self.policy_id,
            "action_schema_version": self.action_schema_version,
            "observation_schema_version": self.observation_schema_version,
            "feature_schema_version": self.feature_schema_version,
            "feature_count": self.feature_count,
            "window_size": self.window_size,
            "weights": [list(row) for row in self.weights],
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
            feature_count=int(payload["feature_count"]),
            window_size=int(payload["window_size"]),
            weights=tuple(tuple(float(value) for value in row) for row in _sequence(payload["weights"])),
        )


@dataclass(frozen=True)
class LinearTrainingConfig:
    feature_count: int = 131_072
    window_size: int = 1
    discount: float = 1.0
    objective: LinearTrainingObjective = "behavior-cloning"
    epochs: int = 1
    learning_rate: float = 0.05
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
        if self.objective not in ("behavior-cloning", "reward-weighted"):
            raise ValueError("objective must be behavior-cloning or reward-weighted.")
        if self.epochs <= 0:
            raise ValueError("epochs must be positive.")
        if self.learning_rate <= 0.0:
            raise ValueError("learning_rate must be positive.")
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

    def to_dict(self) -> dict[str, Any]:
        return {
            "epoch": self.epoch,
            "examples": self.examples,
            "loss": self.loss,
            "accuracy": self.accuracy,
            "elapsed_seconds": self.elapsed_seconds,
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

    def to_dict(self) -> dict[str, Any]:
        return {
            "examples": self.examples,
            "loss": self.loss,
            "accuracy": self.accuracy,
            "elapsed_seconds": self.elapsed_seconds,
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
    else:
        weights = [list(row) for row in LinearPolicyModel.initialized(
            feature_count=training_config.feature_count,
            window_size=training_config.window_size,
            policy_id=training_config.policy_id,
        ).weights]
    epoch_metrics = []
    dataset_config = TrajectoryDatasetConfig(
        window_size=training_config.window_size,
        discount=training_config.discount,
    )

    for epoch in range(1, training_config.epochs + 1):
        start = perf_counter()
        total_loss = 0.0
        correct = 0
        examples = 0
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
            )
        )

    model = LinearPolicyModel(
        policy_id=training_config.policy_id,
        feature_count=training_config.feature_count,
        window_size=training_config.window_size,
        weights=tuple(tuple(row) for row in weights),
    )
    validation_metrics = None
    if validation_paths is not None:
        validation_metrics = evaluate_linear_policy(
            validation_paths,
            model,
            discount=training_config.discount,
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
    max_examples: int | None = None,
) -> LinearEvaluationMetrics:
    if max_examples is not None and max_examples <= 0:
        raise ValueError("max_examples must be positive when set.")
    dataset_config = TrajectoryDatasetConfig(window_size=model.window_size, discount=discount)
    start = perf_counter()
    total_loss = 0.0
    correct = 0
    examples = 0

    for example in iter_training_examples(paths, config=dataset_config):
        features = features_from_example(example, feature_count=model.feature_count)
        probabilities = model.action_probabilities(features, example.legal_action_mask)
        total_loss += -math.log(max(probabilities[example.action_index], 1e-12))
        if model.predict_action(features, example.legal_action_mask) == example.action_index:
            correct += 1
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
                _add_hashed_feature(features, feature_count, f"c:{history_index}:{token_index}:{column_index}:{int(value)}", 1.0)
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
