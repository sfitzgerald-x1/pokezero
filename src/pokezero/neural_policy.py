"""Optional PyTorch transformer policy scaffold.

The base PokeZero package deliberately stays dependency-light. This module is
safe to import without PyTorch installed; construction, tensor conversion, and
training helpers fail with a targeted install message until the `neural` extra
is available.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import json
import math
from os import PathLike
from pathlib import Path
import random
from typing import Any, Callable, Iterable, Mapping, Sequence

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
CONSTANT_LEARNING_RATE_SCHEDULE = "constant"
MIT_THESIS_LEARNING_RATE_SCHEDULE = "mit-thesis"
LEARNING_RATE_SCHEDULES = (CONSTANT_LEARNING_RATE_SCHEDULE, MIT_THESIS_LEARNING_RATE_SCHEDULE)
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
    temporal_aggregator: str = "mean"

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
        if self.transformer_layers < 0:
            raise ValueError("transformer_layers must be non-negative.")
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
        if self.temporal_aggregator not in {"mean", "gru"}:
            raise ValueError("temporal_aggregator must be 'mean' or 'gru'.")
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
            temporal_aggregator=_str_field(payload, "temporal_aggregator", "mean"),
        )


@dataclass(frozen=True)
class TransformerTrainingConfig:
    batch_size: int = 64
    epochs: int = 1
    learning_rate: float = 3e-4
    learning_rate_schedule: str = CONSTANT_LEARNING_RATE_SCHEDULE
    learning_rate_schedule_total_games: int | None = None
    learning_rate_progress_start: float = 0.0
    learning_rate_progress_end: float = 0.0
    weight_decay: float = 0.0
    window_size: int = 4
    discount: float = 1.0
    capped_terminal_value: float = 0.0
    hp_delta_return_weight: float = 0.0
    faint_delta_return_weight: float = 0.0
    turn_penalty_after: int | None = None
    turn_penalty: float = 0.0
    value_loss_weight: float = 0.25
    value_ranking_loss_weight: float = 0.0
    value_ranking_margin: float = 0.0
    opponent_action_loss_weight: float = 0.1
    switch_action_loss_weight: float = 1.0
    action_family_loss_weight: float = 0.0
    switch_target_loss_weight: float = 0.0
    max_batches: int | None = None
    device: str | None = None
    # Training objective: "behavior-cloning" (supervised cross-entropy to the chosen action),
    # "reward-weighted" (same CE, but only positive-return examples contribute to the policy
    # term), or "ppo" (clipped policy-gradient using recorded behavior-policy probabilities
    # and the value head as a baseline). "value-only" optimizes only return prediction and is
    # intended for value-head calibration/fine-tuning. PPO is the self-play RL operator.
    objective: str = "behavior-cloning"
    clip_epsilon: float = 0.2
    entropy_coef: float = 0.0
    normalize_advantage: bool = True
    ppo_target_mode: str = "returns"
    gae_lambda: float = 0.95
    # Optional global gradient-norm clip (torch.nn.utils.clip_grad_norm_) applied before each
    # optimizer step. None disables clipping (legacy behavior). The MIT thesis recipe uses 0.5430.
    max_grad_norm: float | None = None
    freeze_non_value_parameters: bool = False

    def __post_init__(self) -> None:
        if self.objective not in ("behavior-cloning", "reward-weighted", "ppo", "value-only"):
            raise ValueError("objective must be 'behavior-cloning', 'reward-weighted', 'ppo', or 'value-only'.")
        if self.ppo_target_mode not in {"returns", "gae"}:
            raise ValueError("ppo_target_mode must be 'returns' or 'gae'.")
        if self.objective != "ppo" and self.ppo_target_mode != "returns":
            raise ValueError("ppo_target_mode='gae' requires objective='ppo'.")
        if not 0.0 <= self.gae_lambda <= 1.0:
            raise ValueError("gae_lambda must be between 0 and 1.")
        if self.objective == "value-only" and not self.freeze_non_value_parameters:
            raise ValueError("objective='value-only' requires freeze_non_value_parameters=True.")
        if self.freeze_non_value_parameters and self.objective != "value-only":
            raise ValueError("freeze_non_value_parameters requires objective='value-only'.")
        if self.clip_epsilon <= 0.0:
            raise ValueError("clip_epsilon must be positive.")
        if self.entropy_coef < 0.0:
            raise ValueError("entropy_coef must be non-negative.")
        if self.max_grad_norm is not None and self.max_grad_norm <= 0.0:
            raise ValueError("max_grad_norm must be positive when set.")
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive.")
        if self.epochs <= 0:
            raise ValueError("epochs must be positive.")
        if self.learning_rate <= 0.0:
            raise ValueError("learning_rate must be positive.")
        if self.learning_rate_schedule not in LEARNING_RATE_SCHEDULES:
            raise ValueError(f"learning_rate_schedule must be one of: {', '.join(LEARNING_RATE_SCHEDULES)}.")
        if self.learning_rate_schedule_total_games is not None and self.learning_rate_schedule_total_games <= 0:
            raise ValueError("learning_rate_schedule_total_games must be positive when set.")
        if not math.isfinite(self.learning_rate_progress_start):
            raise ValueError("learning_rate_progress_start must be finite.")
        if not math.isfinite(self.learning_rate_progress_end):
            raise ValueError("learning_rate_progress_end must be finite.")
        if not 0.0 <= self.learning_rate_progress_start <= 1.0:
            raise ValueError("learning_rate_progress_start must be between 0 and 1.")
        if not 0.0 <= self.learning_rate_progress_end <= 1.0:
            raise ValueError("learning_rate_progress_end must be between 0 and 1.")
        if self.learning_rate_progress_end < self.learning_rate_progress_start:
            raise ValueError("learning_rate_progress_end must be greater than or equal to learning_rate_progress_start.")
        if self.weight_decay < 0.0:
            raise ValueError("weight_decay must be non-negative.")
        if self.window_size <= 0:
            raise ValueError("window_size must be positive.")
        if not 0.0 <= self.discount <= 1.0:
            raise ValueError("discount must be between 0 and 1.")
        if not -1.0 <= self.capped_terminal_value <= 0.0:
            raise ValueError("capped_terminal_value must be between -1 and 0.")
        if self.hp_delta_return_weight < 0.0:
            raise ValueError("hp_delta_return_weight must be non-negative.")
        if self.faint_delta_return_weight < 0.0:
            raise ValueError("faint_delta_return_weight must be non-negative.")
        if self.turn_penalty_after is not None and self.turn_penalty_after < 0:
            raise ValueError("turn_penalty_after must be non-negative when set.")
        if self.turn_penalty < 0.0:
            raise ValueError("turn_penalty must be non-negative.")
        if self.turn_penalty > 0.0 and self.turn_penalty_after is None:
            raise ValueError("turn_penalty_after must be set when turn_penalty is positive.")
        if self.value_loss_weight < 0.0:
            raise ValueError("value_loss_weight must be non-negative.")
        if self.value_ranking_loss_weight < 0.0:
            raise ValueError("value_ranking_loss_weight must be non-negative.")
        if self.value_ranking_margin < 0.0:
            raise ValueError("value_ranking_margin must be non-negative.")
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
    learning_rate: float | None = None
    value_loss: float | None = None
    value_ranking_loss: float | None = None
    value_ranking_pairs: int | None = None
    opponent_loss: float | None = None
    opponent_accuracy: float | None = None
    action_family_loss: float | None = None
    action_family_accuracy: float | None = None
    switch_target_loss: float | None = None
    switch_target_accuracy: float | None = None
    ppo_valid_examples: int | None = None
    ppo_valid_fraction: float | None = None
    ppo_advantage_mean: float | None = None
    ppo_advantage_std: float | None = None
    ppo_ratio_mean: float | None = None
    ppo_clip_fraction: float | None = None
    ppo_entropy: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ValueCalibrationTransform:
    scale: float = 1.0
    bias: float = 0.0
    clip_min: float = -1.0
    clip_max: float = 1.0
    method: str = "affine"
    points: tuple[tuple[float, float], ...] = ()

    def __post_init__(self) -> None:
        if self.method not in {"affine", "isotonic"}:
            raise ValueError("method must be 'affine' or 'isotonic'.")
        if self.clip_min >= self.clip_max:
            raise ValueError("clip_min must be less than clip_max.")
        points = tuple((float(raw), float(calibrated)) for raw, calibrated in self.points)
        object.__setattr__(self, "points", points)
        if self.method == "isotonic":
            if not points:
                raise ValueError("isotonic value calibration requires at least one point.")
            for (left_raw, left_value), (right_raw, right_value) in zip(points, points[1:], strict=False):
                if right_raw <= left_raw:
                    raise ValueError("isotonic calibration points must have strictly increasing raw values.")
                if right_value < left_value:
                    raise ValueError("isotonic calibration points must have non-decreasing calibrated values.")

    def apply(self, value: float) -> float:
        if self.method == "isotonic":
            calibrated = self._apply_isotonic(float(value))
        else:
            calibrated = (self.scale * float(value)) + self.bias
        return min(self.clip_max, max(self.clip_min, calibrated))

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "method": self.method,
            "scale": float(self.scale),
            "bias": float(self.bias),
            "clip_min": float(self.clip_min),
            "clip_max": float(self.clip_max),
        }
        if self.method == "isotonic":
            payload["points"] = [[float(raw), float(calibrated)] for raw, calibrated in self.points]
        return payload

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ValueCalibrationTransform":
        method = str(payload.get("method", "affine"))
        return cls(
            scale=float(payload.get("scale", 1.0)),
            bias=float(payload.get("bias", 0.0)),
            clip_min=float(payload.get("clip_min", -1.0)),
            clip_max=float(payload.get("clip_max", 1.0)),
            method=method,
            points=tuple(tuple(point) for point in payload.get("points", ())),
        )

    def _apply_isotonic(self, value: float) -> float:
        if len(self.points) == 1:
            return self.points[0][1]
        first_raw, first_value = self.points[0]
        if value <= first_raw:
            return first_value
        last_raw, last_value = self.points[-1]
        if value >= last_raw:
            return last_value
        for (left_raw, left_value), (right_raw, right_value) in zip(self.points, self.points[1:], strict=True):
            if value <= right_raw:
                ratio = (value - left_raw) / (right_raw - left_raw)
                return left_value + (ratio * (right_value - left_value))
        return last_value


@dataclass(frozen=True)
class TransformerTrainingResult:
    model_config: TransformerPolicyConfig
    training_config: TransformerTrainingConfig
    epochs: tuple[TransformerEpochMetrics, ...]
    value_calibration_transform: ValueCalibrationTransform | None = None

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
            if config.transformer_layers > 0:
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
                policy_input_dim = config.embedding_dim
            else:
                self.encoder = None
                # Zero-layer CPU mode has no attention path to mix board context into action tokens.
                # Score each candidate from its own token features plus the pooled position.
                policy_input_dim = config.embedding_dim * 2
            self.temporal_gru = (
                nn.GRU(config.embedding_dim, config.embedding_dim, batch_first=True)
                if config.temporal_aggregator == "gru"
                else None
            )
            self.policy_head = nn.Linear(policy_input_dim, 1)
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
            if self.encoder is None:
                encoded = x
            else:
                encoded = self.encoder(x, src_key_padding_mask=~valid_tokens)
            pooled = self._pool_encoded_history(
                encoded=encoded,
                attention_mask=attention_mask.bool(),
                history_mask=history_mask.bool(),
            )
            latest_action_start = ((window_size - 1) * token_count) + ACTION_CANDIDATE_TOKEN_OFFSET
            action_tokens = encoded[:, latest_action_start : latest_action_start + ACTION_COUNT, :]
            if self.encoder is None:
                pooled_actions = pooled.unsqueeze(1).expand(batch_size, ACTION_COUNT, self.config.embedding_dim)
                action_tokens = torch.cat((action_tokens, pooled_actions), dim=-1)
            raw_value = self.value_head(pooled).squeeze(-1)
            value = torch.tanh(raw_value) if self.config.value_activation == "tanh" else raw_value
            return TransformerPolicyOutput(
                policy_logits=self.policy_head(action_tokens).squeeze(-1),
                value=value,
                opponent_action_logits=self.opponent_action_head(pooled),
            )

        def _pool_encoded_history(self, *, encoded: Any, attention_mask: Any, history_mask: Any) -> Any:
            batch_size, window_size = history_mask.shape
            token_count = self.config.token_count
            embedding_dim = self.config.embedding_dim
            if self.temporal_gru is None:
                valid_tokens = (attention_mask & history_mask.unsqueeze(-1)).view(batch_size, window_size * token_count)
                return _masked_mean(encoded, valid_tokens)

            encoded_by_turn = encoded.view(batch_size, window_size, token_count, embedding_dim)
            turn_embeddings = _masked_mean(
                encoded_by_turn.reshape(batch_size * window_size, token_count, embedding_dim),
                (attention_mask & history_mask.unsqueeze(-1)).reshape(batch_size * window_size, token_count),
            ).view(batch_size, window_size, embedding_dim)
            raw_valid_lengths = history_mask.long().sum(dim=1)
            valid_lengths = raw_valid_lengths.clamp(min=1)
            # Observation windows are left-padded: invalid turns form a prefix and valid turns form
            # a chronological suffix. Compact that suffix before packing so the GRU sees only real
            # history steps in oldest-to-newest order.
            start_offsets = (window_size - valid_lengths).unsqueeze(1)
            time_offsets = torch.arange(window_size, device=encoded.device).unsqueeze(0)
            source_indices = (start_offsets + time_offsets).clamp(max=window_size - 1)
            gather_indices = source_indices.unsqueeze(-1).expand(batch_size, window_size, embedding_dim)
            compacted_turns = turn_embeddings.gather(dim=1, index=gather_indices)
            packed_turns = nn.utils.rnn.pack_padded_sequence(
                compacted_turns,
                valid_lengths.cpu(),
                batch_first=True,
                enforce_sorted=False,
            )
            _, hidden = self.temporal_gru(packed_turns)
            return torch.where(raw_valid_lengths.unsqueeze(-1) > 0, hidden[-1], torch.zeros_like(hidden[-1]))

else:

    class EntityTokenTransformerPolicy:  # type: ignore[no-redef]
        def __init__(self, config: TransformerPolicyConfig) -> None:
            raise TorchUnavailableError(NEURAL_INSTALL_MESSAGE)


def training_batch_to_torch(batch: TrainingBatch, *, device: str | Any | None = None) -> dict[str, Any]:
    torch_module = require_torch()
    tensor = torch_module.as_tensor
    tensors = {
        "categorical_ids": tensor(batch.categorical_ids, dtype=torch_module.long, device=device),
        "numeric_features": tensor(batch.numeric_features, dtype=torch_module.float32, device=device),
        "token_type_ids": tensor(batch.token_type_ids, dtype=torch_module.long, device=device),
        "attention_mask": tensor(batch.attention_mask, dtype=torch_module.bool, device=device),
        "history_mask": tensor(batch.history_mask, dtype=torch_module.bool, device=device),
        "legal_action_mask": tensor(batch.legal_action_mask, dtype=torch_module.bool, device=device),
        "action_indices": tensor(batch.action_indices, dtype=torch_module.long, device=device),
        "returns": tensor(batch.returns, dtype=torch_module.float32, device=device),
        "ppo_advantages": tensor(batch.ppo_advantages, dtype=torch_module.float32, device=device),
        "ppo_advantage_mask": tensor(batch.ppo_advantage_mask, dtype=torch_module.bool, device=device),
        "ppo_value_targets": tensor(batch.ppo_value_targets, dtype=torch_module.float32, device=device),
        "ppo_value_target_mask": tensor(batch.ppo_value_target_mask, dtype=torch_module.bool, device=device),
        "opponent_action_indices": tensor(batch.opponent_action_indices, dtype=torch_module.long, device=device),
        "opponent_action_mask": tensor(batch.opponent_action_mask, dtype=torch_module.bool, device=device),
        "action_probabilities": tensor(batch.action_probabilities, dtype=torch_module.float32, device=device),
        "action_probability_mask": tensor(batch.action_probability_mask, dtype=torch_module.bool, device=device),
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
    value = float(output.value[0].detach().cpu().item())
    transform = getattr(result, "value_calibration_transform", None)
    if isinstance(transform, ValueCalibrationTransform):
        return transform.apply(value)
    return value


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
        raw_value_estimate = float(output.value[0].detach().cpu().item())
        value_estimate = raw_value_estimate if math.isfinite(raw_value_estimate) else None
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
            value_estimate=value_estimate,
            metadata={
                "policy_family": "transformer-softmax",
                "deterministic": self.deterministic,
                "exploration_epsilon": self.exploration_epsilon,
                "sampling_temperature": self.sampling_temperature,
                "family_gated_selection": self.family_gated_selection,
                **(
                    {"value_estimate_dropped": "non_finite"}
                    if value_estimate is None
                    else {}
                ),
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
    epoch_callback: Callable[[Any, TransformerTrainingResult], None] | None = None,
    consumed_cache_callback: Callable[[Path], None] | None = None,
) -> tuple[Any, TransformerTrainingResult]:
    torch_module = require_torch()
    resolved_training_config = training_config or TransformerTrainingConfig()
    if model_config is None:
        raise ValueError("model_config is required (build it with TransformerPolicyConfig.compact_category).")
    resolved_model_config = model_config
    if resolved_training_config.window_size != resolved_model_config.window_size:
        raise ValueError("training_config.window_size must match model_config.window_size.")
    device = resolve_torch_device(resolved_training_config.device)
    if initial_model is None:
        model = EntityTokenTransformerPolicy(resolved_model_config).to(device)
    else:
        _validate_initial_model_config(initial_model, resolved_model_config)
        model = initial_model.to(device) if hasattr(initial_model, "to") else initial_model
    trainable_parameters = _configure_trainable_parameters(
        model,
        freeze_non_value_parameters=resolved_training_config.freeze_non_value_parameters,
    )
    if resolved_training_config.freeze_non_value_parameters and hasattr(model, "eval"):
        model.eval()
    elif hasattr(model, "train"):
        model.train()
    optimizer = torch_module.optim.AdamW(
        trainable_parameters,
        lr=resolved_training_config.learning_rate,
        weight_decay=resolved_training_config.weight_decay,
    )
    dataset_config = TrajectoryDatasetConfig(
        window_size=resolved_training_config.window_size,
        discount=resolved_training_config.discount,
        capped_terminal_value=resolved_training_config.capped_terminal_value,
        hp_delta_return_weight=resolved_training_config.hp_delta_return_weight,
        faint_delta_return_weight=resolved_training_config.faint_delta_return_weight,
        turn_penalty_after=resolved_training_config.turn_penalty_after,
        turn_penalty=resolved_training_config.turn_penalty,
        ppo_target_mode=resolved_training_config.ppo_target_mode,
        gae_lambda=resolved_training_config.gae_lambda,
    )
    epoch_metrics: list[TransformerEpochMetrics] = []
    for epoch in range(1, resolved_training_config.epochs + 1):
        epoch_learning_rate = _learning_rate_for_epoch(resolved_training_config, epoch)
        _set_optimizer_learning_rate(optimizer, epoch_learning_rate)
        totals = _TorchMetricTotals()
        cache_callback_for_epoch = (
            consumed_cache_callback
            if consumed_cache_callback is not None and epoch == resolved_training_config.epochs
            else None
        )
        for batch_index, batch in enumerate(
            iter_training_batches(
                paths,
                batch_size=resolved_training_config.batch_size,
                config=dataset_config,
                consumed_cache_callback=cache_callback_for_epoch,
            ),
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
            if resolved_training_config.max_grad_norm is not None:
                torch_module.nn.utils.clip_grad_norm_(
                    trainable_parameters,
                    resolved_training_config.max_grad_norm,
                )
            optimizer.step()
            totals.add(batch.batch_size, pieces)
            if resolved_training_config.max_batches is not None and batch_index >= resolved_training_config.max_batches:
                break
        if totals.examples == 0:
            raise ValueError("training data produced no examples.")
        epoch_metrics.append(totals.to_epoch_metrics(epoch, learning_rate=epoch_learning_rate))
        if epoch_callback is not None:
            epoch_callback(
                model,
                TransformerTrainingResult(
                    model_config=resolved_model_config,
                    training_config=resolved_training_config,
                    epochs=tuple(epoch_metrics),
                ),
            )
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


def _configure_trainable_parameters(model: Any, *, freeze_non_value_parameters: bool) -> list[Any]:
    if not hasattr(model, "named_parameters"):
        return list(model.parameters())
    trainable_parameters = []
    for name, parameter in model.named_parameters():
        trainable = not freeze_non_value_parameters or name.startswith("value_head.")
        parameter.requires_grad = trainable
        if trainable:
            trainable_parameters.append(parameter)
    if not trainable_parameters:
        raise ValueError("training configuration produced no trainable model parameters.")
    return trainable_parameters


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
            "value_calibration_transform": (
                result.value_calibration_transform.to_dict() if result.value_calibration_transform is not None else None
            ),
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
    value_calibration_payload = payload.get("value_calibration_transform")
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
                value_ranking_loss=_optional_float(metrics.get("value_ranking_loss")),
                value_ranking_pairs=_optional_int(metrics.get("value_ranking_pairs")),
                opponent_loss=_optional_float(metrics.get("opponent_loss")),
                opponent_accuracy=_optional_float(metrics.get("opponent_accuracy")),
                action_family_loss=_optional_float(metrics.get("action_family_loss")),
                action_family_accuracy=_optional_float(metrics.get("action_family_accuracy")),
                switch_target_loss=_optional_float(metrics.get("switch_target_loss")),
                switch_target_accuracy=_optional_float(metrics.get("switch_target_accuracy")),
                ppo_valid_examples=_optional_int(metrics.get("ppo_valid_examples")),
                ppo_valid_fraction=_optional_float(metrics.get("ppo_valid_fraction")),
                ppo_advantage_mean=_optional_float(metrics.get("ppo_advantage_mean")),
                ppo_advantage_std=_optional_float(metrics.get("ppo_advantage_std")),
                ppo_ratio_mean=_optional_float(metrics.get("ppo_ratio_mean")),
                ppo_clip_fraction=_optional_float(metrics.get("ppo_clip_fraction")),
                ppo_entropy=_optional_float(metrics.get("ppo_entropy")),
                learning_rate=_optional_float(metrics.get("learning_rate")),
            )
            for metrics in payload.get("epochs", ())
        ),
        value_calibration_transform=(
            ValueCalibrationTransform.from_dict(value_calibration_payload)
            if isinstance(value_calibration_payload, Mapping)
            else None
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
    value_ranking_loss: float = 0.0
    value_ranking_pairs: int = 0
    opponent_loss: float = 0.0
    opponent_correct: int = 0
    opponent_examples: int = 0
    action_family_loss: float = 0.0
    action_family_correct: int = 0
    action_family_examples: int = 0
    switch_target_loss: float = 0.0
    switch_target_correct: int = 0
    switch_target_examples: int = 0
    ppo_objective_examples: int = 0
    ppo_valid_examples: int = 0
    ppo_advantage_sum: float = 0.0
    ppo_advantage_square_sum: float = 0.0
    ppo_ratio_sum: float = 0.0
    ppo_clip_count: int = 0
    ppo_entropy_sum: float = 0.0

    def add(self, batch_size: int, pieces: Mapping[str, float | int]) -> None:
        self.examples += batch_size
        self.loss += float(pieces["loss"]) * batch_size
        self.policy_loss += float(pieces["policy_loss"]) * batch_size
        self.policy_correct += int(pieces["policy_correct"])
        self.value_loss += float(pieces["value_loss"]) * batch_size
        value_ranking_pairs = int(pieces.get("value_ranking_pairs", 0))
        if value_ranking_pairs:
            self.value_ranking_pairs += value_ranking_pairs
            self.value_ranking_loss += float(pieces.get("value_ranking_loss", 0.0)) * value_ranking_pairs
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
        self.ppo_objective_examples += int(pieces["ppo_objective_examples"])
        ppo_valid_examples = int(pieces["ppo_valid_examples"])
        if ppo_valid_examples:
            self.ppo_valid_examples += ppo_valid_examples
            self.ppo_advantage_sum += float(pieces["ppo_advantage_sum"])
            self.ppo_advantage_square_sum += float(pieces["ppo_advantage_square_sum"])
            self.ppo_ratio_sum += float(pieces["ppo_ratio_sum"])
            self.ppo_clip_count += int(pieces["ppo_clip_count"])
            self.ppo_entropy_sum += float(pieces["ppo_entropy_sum"])

    def to_epoch_metrics(self, epoch: int, *, learning_rate: float | None = None) -> TransformerEpochMetrics:
        ppo_advantage_mean = None
        ppo_advantage_std = None
        ppo_ratio_mean = None
        ppo_clip_fraction = None
        ppo_entropy = None
        if self.ppo_valid_examples:
            ppo_advantage_mean = self.ppo_advantage_sum / self.ppo_valid_examples
            ppo_advantage_variance = max(
                0.0,
                (self.ppo_advantage_square_sum / self.ppo_valid_examples) - (ppo_advantage_mean**2),
            )
            ppo_advantage_std = math.sqrt(ppo_advantage_variance)
            ppo_ratio_mean = self.ppo_ratio_sum / self.ppo_valid_examples
            ppo_clip_fraction = self.ppo_clip_count / self.ppo_valid_examples
            ppo_entropy = self.ppo_entropy_sum / self.ppo_valid_examples
        return TransformerEpochMetrics(
            epoch=epoch,
            examples=self.examples,
            loss=self.loss / self.examples,
            policy_loss=self.policy_loss / self.examples,
            policy_accuracy=self.policy_correct / self.examples,
            learning_rate=learning_rate,
            value_loss=self.value_loss / self.examples,
            value_ranking_loss=(
                self.value_ranking_loss / self.value_ranking_pairs if self.value_ranking_pairs else None
            ),
            value_ranking_pairs=self.value_ranking_pairs if self.value_ranking_pairs else None,
            opponent_loss=(self.opponent_loss / self.opponent_examples) if self.opponent_examples else None,
            opponent_accuracy=(self.opponent_correct / self.opponent_examples) if self.opponent_examples else None,
            action_family_loss=(self.action_family_loss / self.action_family_examples) if self.action_family_examples else None,
            action_family_accuracy=(self.action_family_correct / self.action_family_examples) if self.action_family_examples else None,
            switch_target_loss=(self.switch_target_loss / self.switch_target_examples) if self.switch_target_examples else None,
            switch_target_accuracy=(self.switch_target_correct / self.switch_target_examples) if self.switch_target_examples else None,
            ppo_valid_examples=self.ppo_valid_examples if self.ppo_objective_examples else None,
            ppo_valid_fraction=(self.ppo_valid_examples / self.ppo_objective_examples) if self.ppo_objective_examples else None,
            ppo_advantage_mean=ppo_advantage_mean,
            ppo_advantage_std=ppo_advantage_std,
            ppo_ratio_mean=ppo_ratio_mean,
            ppo_clip_fraction=ppo_clip_fraction,
            ppo_entropy=ppo_entropy,
        )


def learning_rate_for_progress(*, base_learning_rate: float, schedule: str, progress: float) -> float:
    if base_learning_rate <= 0.0 or not math.isfinite(base_learning_rate):
        raise ValueError("base_learning_rate must be positive and finite.")
    if schedule not in LEARNING_RATE_SCHEDULES:
        raise ValueError(f"learning_rate_schedule must be one of: {', '.join(LEARNING_RATE_SCHEDULES)}.")
    if not math.isfinite(progress) or not 0.0 <= progress <= 1.0:
        raise ValueError("learning rate progress must be finite and between 0 and 1.")
    if schedule == CONSTANT_LEARNING_RATE_SCHEDULE:
        return float(base_learning_rate)
    return float(base_learning_rate) / (((8.0 * float(progress)) + 1.0) ** 1.5)


def _learning_rate_for_epoch(config: TransformerTrainingConfig, epoch: int) -> float:
    if epoch < 1 or epoch > config.epochs:
        raise ValueError("epoch is outside the configured training range.")
    if config.epochs == 1:
        progress = config.learning_rate_progress_start
    else:
        fraction = (epoch - 1) / (config.epochs - 1)
        progress = config.learning_rate_progress_start + (
            (config.learning_rate_progress_end - config.learning_rate_progress_start) * fraction
        )
    return learning_rate_for_progress(
        base_learning_rate=config.learning_rate,
        schedule=config.learning_rate_schedule,
        progress=progress,
    )


def _set_optimizer_learning_rate(optimizer: Any, learning_rate: float) -> None:
    for group in optimizer.param_groups:
        group["lr"] = learning_rate


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
    value_targets = _value_targets(tensors)
    value_loss = functional.mse_loss(output.value, value_targets)
    value_ranking_loss, value_ranking_loss_value, value_ranking_pairs = _value_ranking_loss_terms(
        output.value,
        value_targets,
        config,
    )
    ppo_objective_examples = 0
    ppo_valid_examples = 0
    ppo_advantage_sum = 0.0
    ppo_advantage_square_sum = 0.0
    ppo_ratio_sum = 0.0
    ppo_clip_count = 0
    ppo_entropy_sum = 0.0

    if config.objective == "value-only":
        policy_loss = value_loss * 0.0
        loss = value_loss + (config.value_ranking_loss_weight * value_ranking_loss)
        opponent_loss, opponent_loss_value, opponent_correct, opponent_examples = None, 0.0, 0, 0
        family_loss, family_loss_value, family_correct, family_examples = None, 0.0, 0, 0
        switch_loss, switch_loss_value, switch_correct, switch_examples = None, 0.0, 0, 0
    elif config.objective == "ppo":
        ppo_objective_examples = int(tensors["returns"].numel())
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
        raw_advantage = _ppo_advantages(output, tensors)
        advantage = raw_advantage
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
        entropy = -(log_probs.exp() * log_probs).sum(dim=1)
        entropy_mean = (entropy * mask).sum() / denom
        valid_mask = mask.bool()
        ppo_valid_examples = int(valid_mask.sum().item())
        if ppo_valid_examples:
            valid_advantage = raw_advantage[valid_mask]
            valid_ratio = ratio[valid_mask]
            ppo_advantage_sum = float(valid_advantage.sum().detach().item())
            ppo_advantage_square_sum = float((valid_advantage * valid_advantage).sum().detach().item())
            ppo_ratio_sum = float(valid_ratio.sum().detach().item())
            ppo_clip_count = int(
                (
                    (valid_ratio < (1.0 - config.clip_epsilon))
                    | (valid_ratio > (1.0 + config.clip_epsilon))
                ).sum().detach().item()
            )
            ppo_entropy_sum = float(entropy[valid_mask].sum().detach().item())
        loss = (
            policy_loss
            + (config.value_loss_weight * value_loss)
            + (config.value_ranking_loss_weight * value_ranking_loss)
            - (config.entropy_coef * entropy_mean)
        )
    elif config.objective == "reward-weighted":
        per_example_policy_loss = functional.cross_entropy(
            masked_policy_logits,
            tensors["action_indices"],
            reduction="none",
        )
        weights = tensors["returns"].clamp(min=0.0) * _action_family_loss_weights(tensors, config)
        denom = weights.sum().clamp(min=1.0)
        policy_loss = (per_example_policy_loss * weights).sum() / denom
        loss = (
            policy_loss
            + (config.value_loss_weight * value_loss)
            + (config.value_ranking_loss_weight * value_ranking_loss)
        )
    else:
        per_example_policy_loss = functional.cross_entropy(
            masked_policy_logits,
            tensors["action_indices"],
            reduction="none",
        )
        weights = _action_family_loss_weights(tensors, config)
        policy_loss = (per_example_policy_loss * weights).sum() / weights.sum().clamp(min=1.0)
        loss = (
            policy_loss
            + (config.value_loss_weight * value_loss)
            + (config.value_ranking_loss_weight * value_ranking_loss)
        )

    if config.objective != "value-only":
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
        "value_ranking_loss": value_ranking_loss_value,
        "value_ranking_pairs": value_ranking_pairs,
        "opponent_loss": opponent_loss_value,
        "opponent_correct": opponent_correct,
        "opponent_examples": opponent_examples,
        "action_family_loss": family_loss_value,
        "action_family_correct": family_correct,
        "action_family_examples": family_examples,
        "switch_target_loss": switch_loss_value,
        "switch_target_correct": switch_correct,
        "switch_target_examples": switch_examples,
        "ppo_objective_examples": ppo_objective_examples,
        "ppo_valid_examples": ppo_valid_examples,
        "ppo_advantage_sum": ppo_advantage_sum,
        "ppo_advantage_square_sum": ppo_advantage_square_sum,
        "ppo_ratio_sum": ppo_ratio_sum,
        "ppo_clip_count": ppo_clip_count,
        "ppo_entropy_sum": ppo_entropy_sum,
    }


def _value_targets(tensors: Mapping[str, Any]):
    torch_module = require_torch()
    if "ppo_value_target_mask" not in tensors or "ppo_value_targets" not in tensors:
        return tensors["returns"]
    return torch_module.where(
        tensors["ppo_value_target_mask"],
        tensors["ppo_value_targets"],
        tensors["returns"],
    )


def _value_ranking_loss_terms(values: Any, targets: Any, config: TransformerTrainingConfig) -> tuple[Any, float, int]:
    if config.value_ranking_loss_weight <= 0.0:
        return values.sum() * 0.0, 0.0, 0
    torch_module = require_torch()
    functional = torch_module.nn.functional
    target_delta = targets.unsqueeze(1) - targets.unsqueeze(0)
    pair_mask = (target_delta.abs() > 1e-6) & torch_module.triu(
        torch_module.ones_like(target_delta, dtype=torch_module.bool),
        diagonal=1,
    )
    pair_count = int(pair_mask.sum().item())
    if not pair_count:
        return values.sum() * 0.0, 0.0, 0
    direction = target_delta[pair_mask].sign()
    prediction_delta = values.unsqueeze(1) - values.unsqueeze(0)
    loss = functional.softplus(config.value_ranking_margin - (direction * prediction_delta[pair_mask])).mean()
    return loss, float(loss.detach().item()), pair_count


def _ppo_advantages(output: TransformerPolicyOutput, tensors: Mapping[str, Any]):
    torch_module = require_torch()
    fallback = tensors["returns"] - output.value.detach()
    if "ppo_advantage_mask" not in tensors or "ppo_advantages" not in tensors:
        return fallback
    return torch_module.where(
        tensors["ppo_advantage_mask"],
        tensors["ppo_advantages"],
        fallback,
    )


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


def resolve_torch_device(device: str | Any | None = None) -> str | Any:
    if device is not None and device != "":
        return device
    torch_module = require_torch()
    return "cuda" if torch_module.cuda.is_available() else "cpu"


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    return float(value)


def _optional_int(value: Any) -> int | None:
    if value is None:
        return None
    return int(value)


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
