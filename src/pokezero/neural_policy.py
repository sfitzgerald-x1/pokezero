"""Optional PyTorch transformer policy scaffold.

The base PokeZero package deliberately stays dependency-light. This module is
safe to import without PyTorch installed; construction, tensor conversion, and
training helpers fail with a targeted install message until the `neural` extra
is available.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from os import PathLike
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from .actions import ACTION_COUNT, ACTION_SCHEMA_VERSION
from .dataset import TrajectoryDatasetConfig, TrainingBatch, iter_training_batches
from .observation import OBSERVATION_SCHEMA_VERSION, PokeZeroObservationV0
from .showdown import (
    ACTION_CANDIDATE_TOKEN_OFFSET,
    CATEGORY_ID_BUCKETS,
    DEFAULT_REPLAY_OBSERVATION_SPEC,
)

try:  # pragma: no cover - exercised only when the optional dependency exists.
    import torch
    from torch import nn
except ModuleNotFoundError:  # pragma: no cover - covered through require_torch.
    torch = None  # type: ignore[assignment]
    nn = None  # type: ignore[assignment]


NEURAL_POLICY_SCHEMA_VERSION = "pokezero.neural_policy.v0"
NEURAL_TRAINING_SCHEMA_VERSION = "pokezero.neural_training.v0"
NEURAL_INSTALL_MESSAGE = "PyTorch is required for neural policy support. Install with `pip install -e .[neural]`."


class TorchUnavailableError(RuntimeError):
    """Raised when optional neural functionality is used without PyTorch."""


@dataclass(frozen=True)
class TransformerPolicyConfig:
    """Entity-token transformer architecture for `PokeZeroObservationV0` batches."""

    policy_id: str = "entity-transformer"
    window_size: int = 4
    categorical_vocab_size: int = CATEGORY_ID_BUCKETS
    categorical_feature_count: int = DEFAULT_REPLAY_OBSERVATION_SPEC.categorical_feature_count
    numeric_feature_count: int = DEFAULT_REPLAY_OBSERVATION_SPEC.numeric_feature_count
    token_count: int = DEFAULT_REPLAY_OBSERVATION_SPEC.token_count
    embedding_dim: int = 128
    transformer_layers: int = 2
    attention_heads: int = 4
    feedforward_dim: int = 256
    dropout: float = 0.1
    action_schema_version: str = ACTION_SCHEMA_VERSION
    observation_schema_version: str = OBSERVATION_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.action_schema_version != ACTION_SCHEMA_VERSION:
            raise ValueError(f"Unsupported action schema version: {self.action_schema_version!r}.")
        if self.observation_schema_version != OBSERVATION_SCHEMA_VERSION:
            raise ValueError(f"Unsupported observation schema version: {self.observation_schema_version!r}.")
        if self.window_size <= 0:
            raise ValueError("window_size must be positive.")
        if self.categorical_vocab_size <= 1:
            raise ValueError("categorical_vocab_size must be greater than 1.")
        if self.categorical_feature_count <= 0:
            raise ValueError("categorical_feature_count must be positive.")
        if self.numeric_feature_count <= 0:
            raise ValueError("numeric_feature_count must be positive.")
        if self.token_count <= ACTION_CANDIDATE_TOKEN_OFFSET + ACTION_COUNT:
            raise ValueError("token_count must include action-candidate tokens.")
        if self.embedding_dim <= 0:
            raise ValueError("embedding_dim must be positive.")
        if self.transformer_layers <= 0:
            raise ValueError("transformer_layers must be positive.")
        if self.attention_heads <= 0:
            raise ValueError("attention_heads must be positive.")
        if self.embedding_dim % self.attention_heads != 0:
            raise ValueError("embedding_dim must be divisible by attention_heads.")
        if self.feedforward_dim <= 0:
            raise ValueError("feedforward_dim must be positive.")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in [0, 1).")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "TransformerPolicyConfig":
        return cls(
            policy_id=str(payload.get("policy_id") or "entity-transformer"),
            window_size=int(payload.get("window_size") or 4),
            categorical_vocab_size=int(payload.get("categorical_vocab_size") or CATEGORY_ID_BUCKETS),
            categorical_feature_count=int(
                payload.get("categorical_feature_count") or DEFAULT_REPLAY_OBSERVATION_SPEC.categorical_feature_count
            ),
            numeric_feature_count=int(payload.get("numeric_feature_count") or DEFAULT_REPLAY_OBSERVATION_SPEC.numeric_feature_count),
            token_count=int(payload.get("token_count") or DEFAULT_REPLAY_OBSERVATION_SPEC.token_count),
            embedding_dim=int(payload.get("embedding_dim") or 128),
            transformer_layers=int(payload.get("transformer_layers") or 2),
            attention_heads=int(payload.get("attention_heads") or 4),
            feedforward_dim=int(payload.get("feedforward_dim") or 256),
            dropout=float(payload.get("dropout", 0.1)),
            action_schema_version=str(payload.get("action_schema_version") or ACTION_SCHEMA_VERSION),
            observation_schema_version=str(payload.get("observation_schema_version") or OBSERVATION_SCHEMA_VERSION),
        )


@dataclass(frozen=True)
class TransformerTrainingConfig:
    batch_size: int = 64
    epochs: int = 1
    learning_rate: float = 3e-4
    weight_decay: float = 0.0
    window_size: int = 4
    discount: float = 1.0
    capped_terminal_value: float = 0.0
    value_loss_weight: float = 0.25
    opponent_action_loss_weight: float = 0.1
    max_batches: int | None = None
    device: str | None = None

    def __post_init__(self) -> None:
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive.")
        if self.epochs <= 0:
            raise ValueError("epochs must be positive.")
        if self.learning_rate <= 0.0:
            raise ValueError("learning_rate must be positive.")
        if self.weight_decay < 0.0:
            raise ValueError("weight_decay must be non-negative.")
        if self.window_size <= 0:
            raise ValueError("window_size must be positive.")
        if not 0.0 <= self.discount <= 1.0:
            raise ValueError("discount must be between 0 and 1.")
        if not -1.0 <= self.capped_terminal_value <= 0.0:
            raise ValueError("capped_terminal_value must be between -1 and 0.")
        if self.value_loss_weight < 0.0:
            raise ValueError("value_loss_weight must be non-negative.")
        if self.opponent_action_loss_weight < 0.0:
            raise ValueError("opponent_action_loss_weight must be non-negative.")
        if self.max_batches is not None and self.max_batches <= 0:
            raise ValueError("max_batches must be positive when set.")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class TransformerEpochMetrics:
    epoch: int
    examples: int
    loss: float
    policy_loss: float
    policy_accuracy: float
    value_loss: float | None = None
    opponent_loss: float | None = None
    opponent_accuracy: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class TransformerTrainingResult:
    model_config: TransformerPolicyConfig
    training_config: TransformerTrainingConfig
    epochs: tuple[TransformerEpochMetrics, ...]

    @property
    def final_metrics(self) -> TransformerEpochMetrics:
        return self.epochs[-1]


@dataclass(frozen=True)
class TransformerPolicyOutput:
    policy_logits: Any
    value: Any
    opponent_action_logits: Any


def torch_available() -> bool:
    return torch is not None and nn is not None


def require_torch() -> Any:
    if torch is None or nn is None:
        raise TorchUnavailableError(NEURAL_INSTALL_MESSAGE)
    return torch


if nn is not None:  # pragma: no cover - optional dependency path.

    class EntityTokenTransformerPolicy(nn.Module):  # type: ignore[misc]
        """Small transformer over history-token observations with action-token logits."""

        def __init__(self, config: TransformerPolicyConfig) -> None:
            super().__init__()
            self.config = config
            self.category_embedding = nn.Embedding(config.categorical_vocab_size, config.embedding_dim, padding_idx=0)
            self.token_type_embedding = nn.Embedding(config.categorical_vocab_size, config.embedding_dim, padding_idx=0)
            self.history_position_embedding = nn.Embedding(config.window_size, config.embedding_dim)
            self.numeric_projection = nn.Linear(config.numeric_feature_count, config.embedding_dim)
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=config.embedding_dim,
                nhead=config.attention_heads,
                dim_feedforward=config.feedforward_dim,
                dropout=config.dropout,
                batch_first=True,
                activation="gelu",
                norm_first=True,
            )
            self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=config.transformer_layers)
            self.policy_head = nn.Linear(config.embedding_dim, 1)
            self.value_head = nn.Linear(config.embedding_dim, 1)
            self.opponent_action_head = nn.Linear(config.embedding_dim, ACTION_COUNT)

        def forward(
            self,
            *,
            categorical_ids: Any,
            numeric_features: Any,
            token_type_ids: Any,
            attention_mask: Any,
            history_mask: Any,
        ) -> TransformerPolicyOutput:
            _validate_tensor_shapes(categorical_ids, numeric_features, token_type_ids, attention_mask, history_mask, self.config)
            batch_size, window_size, token_count, _ = categorical_ids.shape
            clipped_categories = categorical_ids.clamp(min=0, max=self.config.categorical_vocab_size - 1).long()
            category_embeddings = self.category_embedding(clipped_categories).sum(dim=3)
            token_embeddings = self.token_type_embedding(
                token_type_ids.clamp(min=0, max=self.config.categorical_vocab_size - 1).long()
            )
            numeric_embeddings = self.numeric_projection(numeric_features.float())
            history_positions = torch.arange(window_size, device=categorical_ids.device)
            history_embeddings = self.history_position_embedding(history_positions).view(1, window_size, 1, -1)
            x = category_embeddings + token_embeddings + numeric_embeddings + history_embeddings
            x = x.view(batch_size, window_size * token_count, self.config.embedding_dim)
            valid_tokens = (attention_mask.bool() & history_mask.bool().unsqueeze(-1)).view(batch_size, window_size * token_count)
            encoded = self.encoder(x, src_key_padding_mask=~valid_tokens)
            pooled = _masked_mean(encoded, valid_tokens)
            latest_action_start = ((window_size - 1) * token_count) + ACTION_CANDIDATE_TOKEN_OFFSET
            action_tokens = encoded[:, latest_action_start : latest_action_start + ACTION_COUNT, :]
            return TransformerPolicyOutput(
                policy_logits=self.policy_head(action_tokens).squeeze(-1),
                value=self.value_head(pooled).squeeze(-1),
                opponent_action_logits=self.opponent_action_head(pooled),
            )

else:

    class EntityTokenTransformerPolicy:  # type: ignore[no-redef]
        def __init__(self, config: TransformerPolicyConfig) -> None:
            raise TorchUnavailableError(NEURAL_INSTALL_MESSAGE)


def training_batch_to_torch(batch: TrainingBatch, *, device: str | Any | None = None) -> dict[str, Any]:
    torch_module = require_torch()
    tensors = {
        "categorical_ids": torch_module.tensor(batch.categorical_ids, dtype=torch_module.long, device=device),
        "numeric_features": torch_module.tensor(batch.numeric_features, dtype=torch_module.float32, device=device),
        "token_type_ids": torch_module.tensor(batch.token_type_ids, dtype=torch_module.long, device=device),
        "attention_mask": torch_module.tensor(batch.attention_mask, dtype=torch_module.bool, device=device),
        "history_mask": torch_module.tensor(batch.history_mask, dtype=torch_module.bool, device=device),
        "legal_action_mask": torch_module.tensor(batch.legal_action_mask, dtype=torch_module.bool, device=device),
        "action_indices": torch_module.tensor(batch.action_indices, dtype=torch_module.long, device=device),
        "returns": torch_module.tensor(batch.returns, dtype=torch_module.float32, device=device),
        "opponent_action_indices": torch_module.tensor(batch.opponent_action_indices, dtype=torch_module.long, device=device),
        "opponent_action_mask": torch_module.tensor(batch.opponent_action_mask, dtype=torch_module.bool, device=device),
    }
    return tensors


def observation_window_to_torch(
    observations: Sequence[PokeZeroObservationV0],
    *,
    window_size: int,
    device: str | Any | None = None,
) -> dict[str, Any]:
    if window_size <= 0:
        raise ValueError("window_size must be positive.")
    if not observations:
        raise ValueError("observations must contain at least one item.")
    torch_module = require_torch()
    observation = observations[-1]
    padding_count = max(0, window_size - len(observations))
    window = tuple(observations[-window_size:])
    categorical_padding = _zeros_like(observation.categorical_ids)
    numeric_padding = _zeros_like(observation.numeric_features)
    token_type_padding = _zeros_like(observation.token_type_ids)
    attention_padding = _zeros_like(observation.attention_mask)
    categorical_ids = tuple([categorical_padding] * padding_count) + tuple(item.categorical_ids for item in window)
    numeric_features = tuple([numeric_padding] * padding_count) + tuple(item.numeric_features for item in window)
    token_type_ids = tuple([token_type_padding] * padding_count) + tuple(item.token_type_ids for item in window)
    attention_mask = tuple([attention_padding] * padding_count) + tuple(item.attention_mask for item in window)
    history_mask = tuple(False for _ in range(padding_count)) + tuple(True for _ in window)
    return {
        "categorical_ids": torch_module.tensor((categorical_ids,), dtype=torch_module.long, device=device),
        "numeric_features": torch_module.tensor((numeric_features,), dtype=torch_module.float32, device=device),
        "token_type_ids": torch_module.tensor((token_type_ids,), dtype=torch_module.long, device=device),
        "attention_mask": torch_module.tensor((attention_mask,), dtype=torch_module.bool, device=device),
        "history_mask": torch_module.tensor((history_mask,), dtype=torch_module.bool, device=device),
    }


def train_transformer_policy(
    paths: str | PathLike[str] | Path | Iterable[str | PathLike[str] | Path],
    *,
    model_config: TransformerPolicyConfig | None = None,
    training_config: TransformerTrainingConfig | None = None,
) -> tuple[Any, TransformerTrainingResult]:
    torch_module = require_torch()
    resolved_training_config = training_config or TransformerTrainingConfig()
    resolved_model_config = model_config or TransformerPolicyConfig(window_size=resolved_training_config.window_size)
    device = resolved_training_config.device or ("cuda" if torch_module.cuda.is_available() else "cpu")
    model = EntityTokenTransformerPolicy(resolved_model_config).to(device)
    optimizer = torch_module.optim.AdamW(
        model.parameters(),
        lr=resolved_training_config.learning_rate,
        weight_decay=resolved_training_config.weight_decay,
    )
    dataset_config = TrajectoryDatasetConfig(
        window_size=resolved_training_config.window_size,
        discount=resolved_training_config.discount,
        capped_terminal_value=resolved_training_config.capped_terminal_value,
    )
    epoch_metrics: list[TransformerEpochMetrics] = []
    for epoch in range(1, resolved_training_config.epochs + 1):
        totals = _TorchMetricTotals()
        for batch_index, batch in enumerate(
            iter_training_batches(paths, batch_size=resolved_training_config.batch_size, config=dataset_config),
            start=1,
        ):
            tensors = training_batch_to_torch(batch, device=device)
            output = model(
                categorical_ids=tensors["categorical_ids"],
                numeric_features=tensors["numeric_features"],
                token_type_ids=tensors["token_type_ids"],
                attention_mask=tensors["attention_mask"],
                history_mask=tensors["history_mask"],
            )
            loss, pieces = _transformer_loss(output, tensors, resolved_training_config)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            totals.add(batch.batch_size, pieces)
            if resolved_training_config.max_batches is not None and batch_index >= resolved_training_config.max_batches:
                break
        if totals.examples == 0:
            raise ValueError("training data produced no examples.")
        epoch_metrics.append(totals.to_epoch_metrics(epoch))
    return model, TransformerTrainingResult(
        model_config=resolved_model_config,
        training_config=resolved_training_config,
        epochs=tuple(epoch_metrics),
    )


def save_transformer_checkpoint(
    path: str | PathLike[str] | Path,
    model: Any,
    *,
    result: TransformerTrainingResult,
) -> None:
    torch_module = require_torch()
    checkpoint_path = Path(path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch_module.save(
        {
            "schema_version": NEURAL_POLICY_SCHEMA_VERSION,
            "training_schema_version": NEURAL_TRAINING_SCHEMA_VERSION,
            "model_config": result.model_config.to_dict(),
            "training_config": result.training_config.to_dict(),
            "epochs": [metrics.to_dict() for metrics in result.epochs],
            "state_dict": model.state_dict(),
        },
        checkpoint_path,
    )


def load_transformer_checkpoint(path: str | PathLike[str] | Path, *, map_location: str | Any | None = None) -> tuple[Any, TransformerTrainingResult]:
    torch_module = require_torch()
    payload = torch_module.load(Path(path), map_location=map_location)
    if payload.get("schema_version") != NEURAL_POLICY_SCHEMA_VERSION:
        raise ValueError(f"Unsupported neural policy schema: {payload.get('schema_version')!r}.")
    model_config = TransformerPolicyConfig.from_dict(payload["model_config"])
    training_config = TransformerTrainingConfig(**dict(payload["training_config"]))
    model = EntityTokenTransformerPolicy(model_config)
    model.load_state_dict(payload["state_dict"])
    if map_location is not None:
        model.to(map_location)
    result = TransformerTrainingResult(
        model_config=model_config,
        training_config=training_config,
        epochs=tuple(
            TransformerEpochMetrics(
                epoch=int(metrics["epoch"]),
                examples=int(metrics["examples"]),
                loss=float(metrics["loss"]),
                policy_loss=float(metrics["policy_loss"]),
                policy_accuracy=float(metrics["policy_accuracy"]),
                value_loss=_optional_float(metrics.get("value_loss")),
                opponent_loss=_optional_float(metrics.get("opponent_loss")),
                opponent_accuracy=_optional_float(metrics.get("opponent_accuracy")),
            )
            for metrics in payload.get("epochs", ())
        ),
    )
    return model, result


@dataclass
class _TorchMetricTotals:
    examples: int = 0
    loss: float = 0.0
    policy_loss: float = 0.0
    policy_correct: int = 0
    value_loss: float = 0.0
    opponent_loss: float = 0.0
    opponent_correct: int = 0
    opponent_examples: int = 0

    def add(self, batch_size: int, pieces: Mapping[str, float | int]) -> None:
        self.examples += batch_size
        self.loss += float(pieces["loss"]) * batch_size
        self.policy_loss += float(pieces["policy_loss"]) * batch_size
        self.policy_correct += int(pieces["policy_correct"])
        self.value_loss += float(pieces["value_loss"]) * batch_size
        opponent_examples = int(pieces["opponent_examples"])
        if opponent_examples:
            self.opponent_examples += opponent_examples
            self.opponent_loss += float(pieces["opponent_loss"]) * opponent_examples
            self.opponent_correct += int(pieces["opponent_correct"])

    def to_epoch_metrics(self, epoch: int) -> TransformerEpochMetrics:
        return TransformerEpochMetrics(
            epoch=epoch,
            examples=self.examples,
            loss=self.loss / self.examples,
            policy_loss=self.policy_loss / self.examples,
            policy_accuracy=self.policy_correct / self.examples,
            value_loss=self.value_loss / self.examples,
            opponent_loss=(self.opponent_loss / self.opponent_examples) if self.opponent_examples else None,
            opponent_accuracy=(self.opponent_correct / self.opponent_examples) if self.opponent_examples else None,
        )


def _transformer_loss(output: TransformerPolicyOutput, tensors: Mapping[str, Any], config: TransformerTrainingConfig) -> tuple[Any, dict[str, float | int]]:
    torch_module = require_torch()
    masked_policy_logits = output.policy_logits.masked_fill(~tensors["legal_action_mask"], -1e9)
    policy_loss = torch_module.nn.functional.cross_entropy(masked_policy_logits, tensors["action_indices"])
    predictions = masked_policy_logits.argmax(dim=1)
    policy_correct = int((predictions == tensors["action_indices"]).sum().item())
    value_loss = torch_module.nn.functional.mse_loss(output.value, tensors["returns"])
    loss = policy_loss + (config.value_loss_weight * value_loss)
    opponent_loss_value = 0.0
    opponent_correct = 0
    opponent_examples = int(tensors["opponent_action_mask"].sum().item())
    if opponent_examples and config.opponent_action_loss_weight:
        opponent_logits = output.opponent_action_logits[tensors["opponent_action_mask"]]
        opponent_targets = tensors["opponent_action_indices"][tensors["opponent_action_mask"]]
        opponent_loss = torch_module.nn.functional.cross_entropy(opponent_logits, opponent_targets)
        opponent_loss_value = float(opponent_loss.detach().item())
        opponent_correct = int((opponent_logits.argmax(dim=1) == opponent_targets).sum().item())
        loss = loss + (config.opponent_action_loss_weight * opponent_loss)
    return loss, {
        "loss": float(loss.detach().item()),
        "policy_loss": float(policy_loss.detach().item()),
        "policy_correct": policy_correct,
        "value_loss": float(value_loss.detach().item()),
        "opponent_loss": opponent_loss_value,
        "opponent_correct": opponent_correct,
        "opponent_examples": opponent_examples,
    }


def _validate_tensor_shapes(
    categorical_ids: Any,
    numeric_features: Any,
    token_type_ids: Any,
    attention_mask: Any,
    history_mask: Any,
    config: TransformerPolicyConfig,
) -> None:
    if tuple(categorical_ids.shape[1:]) != (config.window_size, config.token_count, config.categorical_feature_count):
        raise ValueError("categorical_ids shape does not match TransformerPolicyConfig.")
    if tuple(numeric_features.shape[1:]) != (config.window_size, config.token_count, config.numeric_feature_count):
        raise ValueError("numeric_features shape does not match TransformerPolicyConfig.")
    if tuple(token_type_ids.shape[1:]) != (config.window_size, config.token_count):
        raise ValueError("token_type_ids shape does not match TransformerPolicyConfig.")
    if tuple(attention_mask.shape[1:]) != (config.window_size, config.token_count):
        raise ValueError("attention_mask shape does not match TransformerPolicyConfig.")
    if tuple(history_mask.shape[1:]) != (config.window_size,):
        raise ValueError("history_mask shape does not match TransformerPolicyConfig.")


def _masked_mean(values: Any, mask: Any) -> Any:
    torch_module = require_torch()
    weights = mask.float().unsqueeze(-1)
    denominator = weights.sum(dim=1).clamp(min=1.0)
    return (values * weights).sum(dim=1) / denominator


def _zeros_like(value: Any) -> Any:
    if isinstance(value, bool):
        return False
    if isinstance(value, int):
        return 0
    if isinstance(value, float):
        return 0.0
    return tuple(_zeros_like(item) for item in value)


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    return float(value)
