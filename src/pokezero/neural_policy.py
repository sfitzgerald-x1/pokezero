"""Optional PyTorch transformer policy scaffold.

The base PokeZero package deliberately stays dependency-light. This module is
safe to import without PyTorch installed; construction, tensor conversion, and
training helpers fail with a targeted install message until the `neural` extra
is available.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import json
from os import PathLike
from pathlib import Path
import random
from typing import Any, Iterable, Mapping, Sequence

from .actions import ACTION_COUNT, ACTION_SCHEMA_VERSION, MOVE_ACTION_COUNT
from .dataset import TrajectoryDatasetConfig, TrainingBatch, iter_training_batches
from .observation import OBSERVATION_SCHEMA_VERSION, PokeZeroObservationV0
from .policy import PolicyDecision, legal_action_indices
from .showdown import (
    ACTION_CANDIDATE_TOKEN_OFFSET,
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
DEFAULT_TOKEN_TYPE_VOCAB_SIZE = 16
# Small safety net for graceful degradation only. Gen 3 randbats are a closed universe and
# the lean encoding drops every dynamic/unactionable string (HP text, usernames, winner,
# free-form event payloads), so in practice nothing actionable should reach the OOV block;
# the spy audit found zero uncovered bounded categories. Sized for collision comfort, not as
# a real feature (was 4096, which dominated the embedding table with ~524K dead params).
DEFAULT_CATEGORY_OOV_BUCKETS = 16


def collect_categorical_ids(
    paths: str | PathLike[str] | Path | Iterable[str | PathLike[str] | Path],
) -> tuple[int, ...]:
    """Collect the distinct non-zero categorical ids that occur in rollout JSONL.

    These are the only embedding rows that ever carry trained signal. A compact category
    vocabulary built from them keeps a dedicated, collision-free row for every id the model
    can learn from the given data. Ids absent here are untrained (initialization) rows in
    either the full hash table or the compact table; in the compact table they fold into a
    shared out-of-vocabulary block and may collide, but since both schemes leave them
    untrained this does not change learned behavior.
    """
    if isinstance(paths, (str, Path, PathLike)):
        paths = [paths]
    ids: set[int] = set()
    for path in paths:
        with open(path, "r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)
                trajectory = record.get("trajectory") if isinstance(record, Mapping) else None
                steps = (trajectory or record).get("steps", []) if isinstance(trajectory or record, Mapping) else []
                for step in steps:
                    observation = step.get("observation") or {}
                    for row in observation.get("categorical_ids", []):
                        for value in row:
                            ivalue = int(value)
                            if ivalue > 0:
                                ids.add(ivalue)
    return tuple(sorted(ids))


class TorchUnavailableError(RuntimeError):
    """Raised when optional neural functionality is used without PyTorch."""


@dataclass(frozen=True)
class TransformerPolicyConfig:
    """Entity-token transformer architecture for `PokeZeroObservationV0` batches."""

    policy_id: str = "entity-transformer"
    window_size: int = 4
    categorical_vocab_size: int = 2
    token_type_vocab_size: int = DEFAULT_TOKEN_TYPE_VOCAB_SIZE
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
    category_vocab: tuple[str, ...] = ()
    category_oov_buckets: int = 0
    value_activation: str = "tanh"

    @classmethod
    def compact_category(
        cls,
        *,
        category_vocab: Iterable[str],
        category_oov_buckets: int = DEFAULT_CATEGORY_OOV_BUCKETS,
        **kwargs: Any,
    ) -> "TransformerPolicyConfig":
        """Build a config whose category embedding is a direct string→row vocabulary.

        ``category_vocab`` is the sorted closed-universe token strings (row = index+1); strings
        outside it fold into ``category_oov_buckets`` reserved rows. The full embedding has
        ``1 + len(vocab) + oov_buckets`` rows (row 0 is padding). The encoder pre-converts
        strings to rows via the matching CategoryVocabulary, so the model embeds rows directly
        (no remap). Stored here for reproducibility + embedding-size validation.
        """
        tokens = tuple(sorted({str(value).strip().lower() for value in category_vocab if str(value).strip()}))
        size = 1 + len(tokens) + int(category_oov_buckets)
        return cls(
            categorical_vocab_size=size,
            category_vocab=tokens,
            category_oov_buckets=int(category_oov_buckets),
            **kwargs,
        )

    def __post_init__(self) -> None:
        # Normalize to an immutable tuple of strings so a frozen config stays hashable and
        # to_dict()/from_dict() round-trips regardless of whether a list or tuple was passed.
        object.__setattr__(self, "category_vocab", tuple(str(value) for value in self.category_vocab))
        if self.action_schema_version != ACTION_SCHEMA_VERSION:
            raise ValueError(f"Unsupported action schema version: {self.action_schema_version!r}.")
        if self.observation_schema_version != OBSERVATION_SCHEMA_VERSION:
            raise ValueError(f"Unsupported observation schema version: {self.observation_schema_version!r}.")
        if self.window_size <= 0:
            raise ValueError("window_size must be positive.")
        if self.categorical_vocab_size <= 1:
            raise ValueError("categorical_vocab_size must be greater than 1.")
        if self.token_type_vocab_size <= 1:
            raise ValueError("token_type_vocab_size must be greater than 1.")
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
        if self.value_activation not in {"linear", "tanh"}:
            raise ValueError("value_activation must be 'linear' or 'tanh'.")
        # The legacy hash-bucket embedding is retired: a compact category vocabulary is
        # required. Build configs via TransformerPolicyConfig.compact_category(...).
        if not self.category_vocab:
            raise ValueError("category_vocab is required; build configs with compact_category(...).")
        if self.category_oov_buckets < 1:
            raise ValueError("category_oov_buckets must be >= 1 when category_vocab is set.")
        tokens = self.category_vocab
        if any(not str(token).strip() for token in tokens):
            raise ValueError("category_vocab tokens must be non-empty.")
        if any(earlier >= later for earlier, later in zip(tokens, tokens[1:])):
            raise ValueError("category_vocab must be sorted and unique (normalized).")
        expected = 1 + len(tokens) + self.category_oov_buckets
        if self.categorical_vocab_size != expected:
            raise ValueError(
                "categorical_vocab_size must equal 1 + len(category_vocab) + category_oov_buckets."
            )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "TransformerPolicyConfig":
        return cls(
            policy_id=_str_field(payload, "policy_id", "entity-transformer"),
            window_size=_int_field(payload, "window_size", 4),
            categorical_vocab_size=_int_field(payload, "categorical_vocab_size", 2),
            token_type_vocab_size=_int_field(payload, "token_type_vocab_size", DEFAULT_TOKEN_TYPE_VOCAB_SIZE),
            categorical_feature_count=_int_field(
                payload,
                "categorical_feature_count",
                DEFAULT_REPLAY_OBSERVATION_SPEC.categorical_feature_count,
            ),
            numeric_feature_count=_int_field(
                payload,
                "numeric_feature_count",
                DEFAULT_REPLAY_OBSERVATION_SPEC.numeric_feature_count,
            ),
            token_count=_int_field(payload, "token_count", DEFAULT_REPLAY_OBSERVATION_SPEC.token_count),
            embedding_dim=_int_field(payload, "embedding_dim", 128),
            transformer_layers=_int_field(payload, "transformer_layers", 2),
            attention_heads=_int_field(payload, "attention_heads", 4),
            feedforward_dim=_int_field(payload, "feedforward_dim", 256),
            dropout=_float_field(payload, "dropout", 0.1),
            action_schema_version=_str_field(payload, "action_schema_version", ACTION_SCHEMA_VERSION),
            observation_schema_version=_str_field(payload, "observation_schema_version", OBSERVATION_SCHEMA_VERSION),
            category_vocab=tuple(str(value) for value in (payload.get("category_vocab") or ())),
            category_oov_buckets=_int_field(payload, "category_oov_buckets", 0),
            # Historical checkpoints were trained with an unbounded linear value head. Keep that
            # behavior when loading configs that predate the explicit value activation field.
            value_activation=_str_field(payload, "value_activation", "linear"),
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
    switch_action_loss_weight: float = 1.0
    action_family_loss_weight: float = 0.0
    switch_target_loss_weight: float = 0.0
    max_batches: int | None = None
    device: str | None = None
    # Training objective: "behavior-cloning" (supervised cross-entropy to the chosen action),
    # "reward-weighted" (same CE, but only positive-return examples contribute to the policy
    # term), or "ppo" (clipped policy-gradient using recorded behavior-policy probabilities
    # and the value head as a baseline). PPO is the self-play RL operator.
    objective: str = "behavior-cloning"
    clip_epsilon: float = 0.2
    entropy_coef: float = 0.0
    normalize_advantage: bool = True

    def __post_init__(self) -> None:
        if self.objective not in ("behavior-cloning", "reward-weighted", "ppo"):
            raise ValueError("objective must be 'behavior-cloning', 'reward-weighted', or 'ppo'.")
        if self.clip_epsilon <= 0.0:
            raise ValueError("clip_epsilon must be positive.")
        if self.entropy_coef < 0.0:
            raise ValueError("entropy_coef must be non-negative.")
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
        if self.switch_action_loss_weight <= 0.0:
            raise ValueError("switch_action_loss_weight must be positive.")
        if self.action_family_loss_weight < 0.0:
            raise ValueError("action_family_loss_weight must be non-negative.")
        if self.switch_target_loss_weight < 0.0:
            raise ValueError("switch_target_loss_weight must be non-negative.")
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
    action_family_loss: float | None = None
    action_family_accuracy: float | None = None
    switch_target_loss: float | None = None
    switch_target_accuracy: float | None = None

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
            self.token_type_embedding = nn.Embedding(config.token_type_vocab_size, config.embedding_dim)
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
            # The observation already stores compact embedding rows (the encoder converts token
            # strings to rows via the matching CategoryVocabulary), so the embedding is indexed
            # directly — no in-model hash→row remap.

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
            clipped_categories = categorical_ids.long().clamp(min=0, max=self.config.categorical_vocab_size - 1)
            category_embeddings = self.category_embedding(clipped_categories).sum(dim=3)
            token_embeddings = self.token_type_embedding(
                token_type_ids.clamp(min=0, max=self.config.token_type_vocab_size - 1).long()
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
            raw_value = self.value_head(pooled).squeeze(-1)
            value = torch.tanh(raw_value) if self.config.value_activation == "tanh" else raw_value
            return TransformerPolicyOutput(
                policy_logits=self.policy_head(action_tokens).squeeze(-1),
                value=value,
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
        "action_probabilities": torch_module.tensor(batch.action_probabilities, dtype=torch_module.float32, device=device),
        "action_probability_mask": torch_module.tensor(batch.action_probability_mask, dtype=torch_module.bool, device=device),
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
        "legal_action_mask": torch_module.tensor((tuple(observation.legal_action_mask),), dtype=torch_module.bool, device=device),
    }


def evaluate_transformer_observation_value(
    *,
    model: Any,
    result: TransformerTrainingResult,
    observations: Sequence[PokeZeroObservationV0],
    device: str | Any | None = None,
) -> float:
    """Evaluate the transformer's value head for a player-relative observation history."""

    if not observations:
        raise ValueError("observations must contain at least one item.")
    torch_module = require_torch()
    if hasattr(model, "eval"):
        model.eval()
    if device is not None and hasattr(model, "to"):
        model.to(device)
    tensors = observation_window_to_torch(
        observations[-result.model_config.window_size :],
        window_size=result.model_config.window_size,
        device=device,
    )
    with torch_module.no_grad():
        output = model(
            categorical_ids=tensors["categorical_ids"],
            numeric_features=tensors["numeric_features"],
            token_type_ids=tensors["token_type_ids"],
            attention_mask=tensors["attention_mask"],
            history_mask=tensors["history_mask"],
        )
    return float(output.value[0].detach().cpu().item())


def evaluate_transformer_action_priors(
    *,
    model: Any,
    result: TransformerTrainingResult,
    observations: Sequence[PokeZeroObservationV0],
    temperature: float = 1.0,
    device: str | Any | None = None,
) -> tuple[float, ...]:
    """Evaluate masked legal-action priors from the transformer's policy head."""

    if not observations:
        raise ValueError("observations must contain at least one item.")
    if temperature <= 0.0:
        raise ValueError("temperature must be positive.")
    torch_module = require_torch()
    if hasattr(model, "eval"):
        model.eval()
    if device is not None and hasattr(model, "to"):
        model.to(device)
    tensors = observation_window_to_torch(
        observations[-result.model_config.window_size :],
        window_size=result.model_config.window_size,
        device=device,
    )
    with torch_module.no_grad():
        output = model(
            categorical_ids=tensors["categorical_ids"],
            numeric_features=tensors["numeric_features"],
            token_type_ids=tensors["token_type_ids"],
            attention_mask=tensors["attention_mask"],
            history_mask=tensors["history_mask"],
        )
        probabilities = _masked_action_probabilities(
            output.policy_logits[0],
            tensors["legal_action_mask"][0],
            temperature=temperature,
        )
    return tuple(float(probabilities[index].detach().cpu().item()) for index in range(ACTION_COUNT))


def evaluate_transformer_opponent_action_priors(
    *,
    model: Any,
    result: TransformerTrainingResult,
    observations: Sequence[PokeZeroObservationV0],
    temperature: float = 1.0,
    device: str | Any | None = None,
) -> tuple[float, ...]:
    """Evaluate unmasked opponent-action priors from the auxiliary opponent head."""

    if not observations:
        raise ValueError("observations must contain at least one item.")
    if temperature <= 0.0:
        raise ValueError("temperature must be positive.")
    torch_module = require_torch()
    if hasattr(model, "eval"):
        model.eval()
    if device is not None and hasattr(model, "to"):
        model.to(device)
    tensors = observation_window_to_torch(
        observations[-result.model_config.window_size :],
        window_size=result.model_config.window_size,
        device=device,
    )
    with torch_module.no_grad():
        output = model(
            categorical_ids=tensors["categorical_ids"],
            numeric_features=tensors["numeric_features"],
            token_type_ids=tensors["token_type_ids"],
            attention_mask=tensors["attention_mask"],
            history_mask=tensors["history_mask"],
        )
        probabilities = _action_probabilities(
            output.opponent_action_logits[0],
            temperature=temperature,
        )
    return tuple(float(probabilities[index].detach().cpu().item()) for index in range(ACTION_COUNT))


@dataclass
class TransformerSoftmaxPolicy:
    """Policy adapter that makes a transformer checkpoint playable in rollouts."""

    model: Any
    result: TransformerTrainingResult
    deterministic: bool = True
    exploration_epsilon: float = 0.0
    sampling_temperature: float = 1.0
    family_gated_selection: bool = False
    device: str | Any | None = None
    policy_id: str | None = None
    _history_by_player: dict[str, list[PokeZeroObservationV0]] | None = None

    def __post_init__(self) -> None:
        require_torch()
        if not 0.0 <= self.exploration_epsilon <= 1.0:
            raise ValueError("exploration_epsilon must be between 0 and 1.")
        if self.sampling_temperature <= 0.0:
            raise ValueError("sampling_temperature must be positive.")
        if self.family_gated_selection and not self.deterministic:
            raise ValueError("family_gated_selection currently requires deterministic selection.")
        if self.policy_id is None:
            self.policy_id = self.result.model_config.policy_id
        if self._history_by_player is None:
            self._history_by_player = {}
        if hasattr(self.model, "eval"):
            self.model.eval()
        if self.device is not None and hasattr(self.model, "to"):
            self.model.to(self.device)

    def reset(self) -> None:
        if self._history_by_player is not None:
            self._history_by_player.clear()

    def select_action(
        self,
        observation: PokeZeroObservationV0,
        *,
        rng: random.Random,
    ) -> PolicyDecision:
        torch_module = require_torch()
        player_key = _observation_player_key(observation)
        history_by_player = self._history_by_player if self._history_by_player is not None else {}
        history = history_by_player.setdefault(player_key, [])
        history.append(observation)
        tensors = observation_window_to_torch(
            history[-self.result.model_config.window_size :],
            window_size=self.result.model_config.window_size,
            device=self.device,
        )
        with torch_module.no_grad():
            output = self.model(
                categorical_ids=tensors["categorical_ids"],
                numeric_features=tensors["numeric_features"],
                token_type_ids=tensors["token_type_ids"],
                attention_mask=tensors["attention_mask"],
                history_mask=tensors["history_mask"],
            )
            probabilities = _masked_action_probabilities(
                output.policy_logits[0],
                tensors["legal_action_mask"][0],
                temperature=self.sampling_temperature,
            )
        legal = legal_action_indices(observation.legal_action_mask)
        greedy_action = _greedy_action_index(
            probabilities=tuple(float(probabilities[index].item()) for index in range(ACTION_COUNT)),
            legal=legal,
            family_gated=self.family_gated_selection,
        )
        random_exploration = self.exploration_epsilon and rng.random() < self.exploration_epsilon
        if random_exploration:
            action_index = rng.choice(legal)
        elif self.deterministic:
            action_index = greedy_action
        else:
            action_index = _sample_action(tuple(float(probabilities[index].item()) for index in range(ACTION_COUNT)), legal, rng)
        return PolicyDecision(
            action_index=action_index,
            policy_id=str(self.policy_id),
            action_probability=_behavior_probability(
                action_index=action_index,
                probabilities=tuple(float(probabilities[index].item()) for index in range(ACTION_COUNT)),
                legal=legal,
                deterministic=self.deterministic,
                greedy_action=greedy_action,
                exploration_epsilon=self.exploration_epsilon,
            ),
            metadata={
                "policy_family": "transformer-softmax",
                "deterministic": self.deterministic,
                "exploration_epsilon": self.exploration_epsilon,
                "sampling_temperature": self.sampling_temperature,
                "family_gated_selection": self.family_gated_selection,
            },
        )


def load_transformer_policy(
    path: str | PathLike[str] | Path,
    *,
    deterministic: bool = True,
    exploration_epsilon: float = 0.0,
    sampling_temperature: float = 1.0,
    family_gated_selection: bool = False,
    device: str | Any | None = None,
) -> TransformerSoftmaxPolicy:
    model, result = load_transformer_checkpoint(path, map_location=device)
    return TransformerSoftmaxPolicy(
        model=model,
        result=result,
        deterministic=deterministic,
        exploration_epsilon=exploration_epsilon,
        sampling_temperature=sampling_temperature,
        family_gated_selection=family_gated_selection,
        device=device,
    )


def train_transformer_policy(
    paths: str | PathLike[str] | Path | Iterable[str | PathLike[str] | Path],
    *,
    model_config: TransformerPolicyConfig | None = None,
    training_config: TransformerTrainingConfig | None = None,
    initial_model: Any | None = None,
) -> tuple[Any, TransformerTrainingResult]:
    torch_module = require_torch()
    resolved_training_config = training_config or TransformerTrainingConfig()
    if model_config is None:
        raise ValueError("model_config is required (build it with TransformerPolicyConfig.compact_category).")
    resolved_model_config = model_config
    device = resolved_training_config.device or ("cuda" if torch_module.cuda.is_available() else "cpu")
    if initial_model is None:
        model = EntityTokenTransformerPolicy(resolved_model_config).to(device)
    else:
        _validate_initial_model_config(initial_model, resolved_model_config)
        model = initial_model.to(device) if hasattr(initial_model, "to") else initial_model
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


def _validate_initial_model_config(model: Any, expected: TransformerPolicyConfig) -> None:
    initial_config = getattr(model, "config", None)
    if initial_config is None:
        return
    comparable_expected = replace(expected, policy_id=getattr(initial_config, "policy_id", expected.policy_id))
    if initial_config != comparable_expected:
        raise ValueError("initial_model config must match model_config except for policy_id.")


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
    payload = torch_module.load(Path(path), map_location=map_location, weights_only=True)
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
                action_family_loss=_optional_float(metrics.get("action_family_loss")),
                action_family_accuracy=_optional_float(metrics.get("action_family_accuracy")),
                switch_target_loss=_optional_float(metrics.get("switch_target_loss")),
                switch_target_accuracy=_optional_float(metrics.get("switch_target_accuracy")),
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
    action_family_loss: float = 0.0
    action_family_correct: int = 0
    action_family_examples: int = 0
    switch_target_loss: float = 0.0
    switch_target_correct: int = 0
    switch_target_examples: int = 0

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
        action_family_examples = int(pieces["action_family_examples"])
        if action_family_examples:
            self.action_family_examples += action_family_examples
            self.action_family_loss += float(pieces["action_family_loss"]) * action_family_examples
            self.action_family_correct += int(pieces["action_family_correct"])
        switch_target_examples = int(pieces["switch_target_examples"])
        if switch_target_examples:
            self.switch_target_examples += switch_target_examples
            self.switch_target_loss += float(pieces["switch_target_loss"]) * switch_target_examples
            self.switch_target_correct += int(pieces["switch_target_correct"])

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
            action_family_loss=(self.action_family_loss / self.action_family_examples) if self.action_family_examples else None,
            action_family_accuracy=(self.action_family_correct / self.action_family_examples) if self.action_family_examples else None,
            switch_target_loss=(self.switch_target_loss / self.switch_target_examples) if self.switch_target_examples else None,
            switch_target_accuracy=(self.switch_target_correct / self.switch_target_examples) if self.switch_target_examples else None,
        )


def _opponent_loss_terms(output: TransformerPolicyOutput, tensors: Mapping[str, Any], config: TransformerTrainingConfig):
    """Auxiliary opponent-action cross-entropy over examples with a recorded opponent label."""
    torch_module = require_torch()
    examples = int(tensors["opponent_action_mask"].sum().item())
    if not examples or not config.opponent_action_loss_weight:
        return None, 0.0, 0, examples
    logits = output.opponent_action_logits[tensors["opponent_action_mask"]]
    targets = tensors["opponent_action_indices"][tensors["opponent_action_mask"]]
    loss = torch_module.nn.functional.cross_entropy(logits, targets)
    correct = int((logits.argmax(dim=1) == targets).sum().item())
    return loss, float(loss.detach().item()), correct, examples


def _transformer_loss(output: TransformerPolicyOutput, tensors: Mapping[str, Any], config: TransformerTrainingConfig) -> tuple[Any, dict[str, float | int]]:
    torch_module = require_torch()
    functional = torch_module.nn.functional
    masked_policy_logits = output.policy_logits.masked_fill(~tensors["legal_action_mask"], -1e9)
    policy_correct = int((masked_policy_logits.argmax(dim=1) == tensors["action_indices"]).sum().item())
    value_loss = functional.mse_loss(output.value, tensors["returns"])

    if config.objective == "ppo":
        # Clipped policy-gradient (PPO): importance-weight the chosen action's log-prob by a
        # value-baselined advantage, using the recorded behavior-policy probability. Only
        # examples with a recorded action probability contribute to the policy term.
        log_probs = functional.log_softmax(masked_policy_logits, dim=1)
        chosen_log_prob = log_probs.gather(1, tensors["action_indices"].unsqueeze(1)).squeeze(1)
        # Only examples with a recorded, strictly-positive behavior probability are valid for
        # importance sampling; a zero/missing behavior prob has an undefined ratio, so exclude it.
        mask = (tensors["action_probability_mask"] & (tensors["action_probabilities"] > 0)).float()
        behavior_log_prob = tensors["action_probabilities"].clamp(min=1e-6).log()
        denom = mask.sum().clamp(min=1.0)
        advantage = tensors["returns"] - output.value.detach()
        if config.normalize_advantage and float(denom.item()) > 1.0:
            masked_mean = (advantage * mask).sum() / denom
            masked_var = (((advantage - masked_mean) ** 2) * mask).sum() / denom
            advantage = (advantage - masked_mean) / (masked_var.sqrt() + 1e-8)
        ratio = (chosen_log_prob - behavior_log_prob).exp()
        surrogate = torch_module.min(
            ratio * advantage,
            ratio.clamp(1.0 - config.clip_epsilon, 1.0 + config.clip_epsilon) * advantage,
        )
        policy_loss = -(surrogate * mask).sum() / denom
        entropy_mean = (-(log_probs.exp() * log_probs).sum(dim=1) * mask).sum() / denom
        loss = policy_loss + (config.value_loss_weight * value_loss) - (config.entropy_coef * entropy_mean)
    elif config.objective == "reward-weighted":
        per_example_policy_loss = functional.cross_entropy(
            masked_policy_logits,
            tensors["action_indices"],
            reduction="none",
        )
        weights = tensors["returns"].clamp(min=0.0) * _action_family_loss_weights(tensors, config)
        denom = weights.sum().clamp(min=1.0)
        policy_loss = (per_example_policy_loss * weights).sum() / denom
        loss = policy_loss + (config.value_loss_weight * value_loss)
    else:
        per_example_policy_loss = functional.cross_entropy(
            masked_policy_logits,
            tensors["action_indices"],
            reduction="none",
        )
        weights = _action_family_loss_weights(tensors, config)
        policy_loss = (per_example_policy_loss * weights).sum() / weights.sum().clamp(min=1.0)
        loss = policy_loss + (config.value_loss_weight * value_loss)

    opponent_loss, opponent_loss_value, opponent_correct, opponent_examples = _opponent_loss_terms(output, tensors, config)
    if opponent_loss is not None:
        loss = loss + (config.opponent_action_loss_weight * opponent_loss)
    family_loss, family_loss_value, family_correct, family_examples = _action_family_loss_terms(masked_policy_logits, tensors, config)
    if family_loss is not None:
        loss = loss + (config.action_family_loss_weight * family_loss)
    switch_loss, switch_loss_value, switch_correct, switch_examples = _switch_target_loss_terms(masked_policy_logits, tensors, config)
    if switch_loss is not None:
        loss = loss + (config.switch_target_loss_weight * switch_loss)
    return loss, {
        "loss": float(loss.detach().item()),
        "policy_loss": float(policy_loss.detach().item()),
        "policy_correct": policy_correct,
        "value_loss": float(value_loss.detach().item()),
        "opponent_loss": opponent_loss_value,
        "opponent_correct": opponent_correct,
        "opponent_examples": opponent_examples,
        "action_family_loss": family_loss_value,
        "action_family_correct": family_correct,
        "action_family_examples": family_examples,
        "switch_target_loss": switch_loss_value,
        "switch_target_correct": switch_correct,
        "switch_target_examples": switch_examples,
    }


def _action_family_loss_terms(masked_policy_logits: Any, tensors: Mapping[str, Any], config: TransformerTrainingConfig):
    """Auxiliary move-vs-switch loss over the model's legal action logits."""
    if not config.action_family_loss_weight:
        return None, 0.0, 0, 0
    torch_module = require_torch()
    functional = torch_module.nn.functional
    move_family_logits = torch_module.logsumexp(masked_policy_logits[:, :MOVE_ACTION_COUNT], dim=1)
    switch_family_logits = torch_module.logsumexp(masked_policy_logits[:, MOVE_ACTION_COUNT:], dim=1)
    family_logits = torch_module.stack((move_family_logits, switch_family_logits), dim=1)
    family_targets = (tensors["action_indices"] >= MOVE_ACTION_COUNT).long()
    loss = functional.cross_entropy(family_logits, family_targets)
    correct = int((family_logits.argmax(dim=1) == family_targets).sum().item())
    return loss, float(loss.detach().item()), correct, int(family_targets.numel())


def _switch_target_loss_terms(masked_policy_logits: Any, tensors: Mapping[str, Any], config: TransformerTrainingConfig):
    """Auxiliary conditional switch-target loss over examples whose teacher action switches."""
    if not config.switch_target_loss_weight:
        return None, 0.0, 0, 0
    torch_module = require_torch()
    functional = torch_module.nn.functional
    switch_mask = tensors["action_indices"] >= MOVE_ACTION_COUNT
    examples = int(switch_mask.sum().item())
    if not examples:
        return None, 0.0, 0, 0
    logits = masked_policy_logits[switch_mask, MOVE_ACTION_COUNT:]
    targets = tensors["action_indices"][switch_mask] - MOVE_ACTION_COUNT
    loss = functional.cross_entropy(logits, targets)
    correct = int((logits.argmax(dim=1) == targets).sum().item())
    return loss, float(loss.detach().item()), correct, examples


def _action_family_loss_weights(tensors: Mapping[str, Any], config: TransformerTrainingConfig):
    torch_module = require_torch()
    weights = torch_module.ones_like(tensors["returns"])
    if config.switch_action_loss_weight == 1.0:
        return weights
    return torch_module.where(
        tensors["action_indices"] >= MOVE_ACTION_COUNT,
        weights * float(config.switch_action_loss_weight),
        weights,
    )


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


def _masked_action_probabilities(logits: Any, legal_action_mask: Any, *, temperature: float) -> Any:
    torch_module = require_torch()
    masked_logits = logits.masked_fill(~legal_action_mask.bool(), -1e9)
    return torch_module.nn.functional.softmax(masked_logits / temperature, dim=0)


def _action_probabilities(logits: Any, *, temperature: float) -> Any:
    torch_module = require_torch()
    return torch_module.nn.functional.softmax(logits / temperature, dim=0)


def _sample_action(probabilities: Sequence[float], legal: Sequence[int], rng: random.Random) -> int:
    threshold = rng.random()
    cumulative = 0.0
    for action_index in legal:
        cumulative += probabilities[action_index]
        if threshold <= cumulative:
            return action_index
    return legal[-1]


def _greedy_action_index(*, probabilities: Sequence[float], legal: Sequence[int], family_gated: bool) -> int:
    if not family_gated:
        return max(legal, key=lambda index: (float(probabilities[index]), -index))
    legal_moves = tuple(index for index in legal if index < MOVE_ACTION_COUNT)
    legal_switches = tuple(index for index in legal if index >= MOVE_ACTION_COUNT)
    if not legal_moves or not legal_switches:
        return max(legal, key=lambda index: (float(probabilities[index]), -index))
    move_mass = sum(float(probabilities[index]) for index in legal_moves)
    switch_mass = sum(float(probabilities[index]) for index in legal_switches)
    family_legal = legal_switches if switch_mass > move_mass else legal_moves
    return max(family_legal, key=lambda index: (float(probabilities[index]), -index))


def _behavior_probability(
    *,
    action_index: int,
    probabilities: Sequence[float],
    legal: Sequence[int],
    deterministic: bool,
    greedy_action: int,
    exploration_epsilon: float,
) -> float:
    if deterministic:
        exploit_probability = 1.0 - exploration_epsilon if action_index == greedy_action else 0.0
        explore_probability = exploration_epsilon / len(legal)
        return exploit_probability + explore_probability
    # Sampling branch: behavior policy mixes softmax sampling with epsilon-uniform exploration,
    # so the true probability is (1 - epsilon) * pi(a) + epsilon / |legal| (reduces to pi(a) when
    # epsilon == 0). PPO importance ratios rely on this being the actual behavior probability.
    return (1.0 - exploration_epsilon) * probabilities[action_index] + (exploration_epsilon / len(legal))


def _observation_player_key(observation: PokeZeroObservationV0) -> str:
    if observation.perspective is None:
        return "default"
    return observation.perspective.player_id or observation.perspective.showdown_slot


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


def _int_field(payload: Mapping[str, Any], key: str, default: int) -> int:
    if key not in payload:
        return default
    return int(payload[key])


def _float_field(payload: Mapping[str, Any], key: str, default: float) -> float:
    if key not in payload:
        return default
    return float(payload[key])


def _str_field(payload: Mapping[str, Any], key: str, default: str) -> str:
    if key not in payload:
        return default
    return str(payload[key])
