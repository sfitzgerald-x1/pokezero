"""Explicit v5 Torch consumer; legacy models and defaults are untouched."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import math
import re
from numbers import Real

from .v5_contract import (
    BELIEF_TYPE, CATEGORICAL_FEATURES, LabelVocabulary, MAX_VARIANTS, NUMERIC_FEATURES,
    OBSERVATION_SCHEMA_VERSION_V5, TRAINING_CONTRACT_SCHEMA, V5_TOKENS,
)


MODEL_INPUTS = ('categorical_ids', 'numeric_features', 'token_type_ids', 'attention_mask', 'history_mask')
TOKEN_ROLES = (0,) + (1,) * 6 + (2,) * 6 + (3,) * 9 + (5,) + (BELIEF_TYPE,) * 6
EXPERT_ROLES = (2,) + (0,) * 12 + (1,) * 9 + (2,) * 7


@dataclass(frozen=True)
class V5ModelConfig:
    category_vocab_size: int
    category_vocabulary_sha256: str
    label_vocabulary: LabelVocabulary
    arm: str = 'B'
    replicate: int = 0
    width: int = 512
    layers: int = 3
    attention_heads: int = 8
    expert_width: int = 512
    value_hidden: tuple[int, int] = (512, 512)
    value_bins: int = 101
    value_sigma_ratio: float = 0.75
    norm_epsilon: float = 1e-6
    dropout: float = 0.0

    def __post_init__(self):
        if (type(self.label_vocabulary) is not LabelVocabulary or self.arm not in ('A', 'B', 'D')
                or type(self.replicate) is not int or self.replicate not in (0, 1)
                or self.replicate == 1 and self.arm != 'B'):
            raise ValueError('initial arms are A,B,D and B replicate1')
        if (type(self.category_vocabulary_sha256) is not str
                or re.fullmatch('[0-9a-f]{64}', self.category_vocabulary_sha256) is None):
            raise ValueError('original category vocabulary SHA256 required')
        object.__setattr__(self, 'value_hidden', tuple(self.value_hidden))
        if (any(type(x) is not int or x <= 0 for x in (self.category_vocab_size, self.width,
                self.layers, self.attention_heads, self.expert_width, self.value_bins, *self.value_hidden))
                or len(self.value_hidden) != 2 or self.width % self.attention_heads or self.value_bins < 3):
            raise ValueError('positive compatible model dimensions and two value MLP widths required')
        for name in ('norm_epsilon', 'value_sigma_ratio', 'dropout'):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, Real):
                raise ValueError('real model hyperparameters required')
            object.__setattr__(self, name, float(value))
        if (not math.isfinite(self.norm_epsilon) or self.norm_epsilon <= 0
                or not math.isfinite(self.value_sigma_ratio) or self.value_sigma_ratio <= 0
                or not math.isfinite(self.dropout) or not 0 <= self.dropout < 1):
            raise ValueError('finite normalization/smoothing/dropout settings required')

    @property
    def treatment(self):
        return self.arm != 'A'

    def metadata(self):
        return dict(schema='pokezero.model.v5', observation_schema=OBSERVATION_SCHEMA_VERSION_V5,
            arm=self.arm, replicate=self.replicate, initialization='scratch', width=self.width,
            layers=self.layers, attention_heads=self.attention_heads, expert_width=self.expert_width,
            value_hidden=self.value_hidden, value_bins=self.value_bins, value_sigma_ratio=self.value_sigma_ratio,
            norm='pre_rmsnorm', norm_epsilon=self.norm_epsilon, feedforward='three_token_role_experts', dropout=self.dropout,
            category_vocab_size=self.category_vocab_size, numeric_features=NUMERIC_FEATURES,
            categorical_features=CATEGORICAL_FEATURES, tokens=V5_TOKENS,
            category_vocabulary_sha256=self.category_vocabulary_sha256,
            label_vocabulary_sha256=self.label_vocabulary.sha256,
            label_vocabulary=asdict(self.label_vocabulary), privileged_critic=False,
            actor='RAW_POLICY', search_mode='LABEL_ONLY' if self.arm == 'D' else 'OFF')

    @classmethod
    def from_metadata(cls, payload):
        if (payload.get('schema') != 'pokezero.model.v5'
                or payload.get('observation_schema') != OBSERVATION_SCHEMA_VERSION_V5
                or payload.get('privileged_critic') is not False or payload.get('actor') != 'RAW_POLICY'
                or payload.get('norm') != 'pre_rmsnorm' or payload.get('feedforward') != 'three_token_role_experts'
                or payload.get('initialization') != 'scratch'
                or (payload.get('tokens'), payload.get('numeric_features'), payload.get('categorical_features')) != (29, 132, 41)):
            raise ValueError('explicit public v5 model metadata required')
        vocabulary = LabelVocabulary(**payload['label_vocabulary'])
        if vocabulary.sha256 != payload.get('label_vocabulary_sha256'):
            raise ValueError('model label namespace metadata drift')
        keys = ('category_vocab_size', 'category_vocabulary_sha256', 'arm', 'replicate', 'width', 'layers',
            'attention_heads', 'expert_width', 'value_hidden', 'value_bins', 'value_sigma_ratio', 'norm_epsilon', 'dropout')
        result = cls(label_vocabulary=vocabulary, **{key: payload[key] for key in keys})
        if payload.get('search_mode') != ('LABEL_ONLY' if result.arm == 'D' else 'OFF'):
            raise ValueError('model actor/search arm metadata differs')
        return result


@dataclass(frozen=True)
class V5Output:
    policy_logits: object
    value: object
    value_logits: object = None
    opponent_semantic_logits: object = None
    hidden_species_logits: object = None
    hidden_variant_logits: object = None
    survival_logits: object = None
    final_hp: object = None


def require_v5_bindings(columns, config):
    if (columns.get('schema') != TRAINING_CONTRACT_SCHEMA
            or columns.get('observation_schema') != OBSERVATION_SCHEMA_VERSION_V5
            or columns.get('category_vocabulary_sha256') != config.category_vocabulary_sha256
            or columns.get('label_vocabulary_sha256') != config.label_vocabulary.sha256):
        raise ValueError('explicit v5 batch and exact category/label namespaces required')


def build_v5_model(config, *, enable_v5=False):
    """Opt-in precedes Torch/model allocation. No checkpoint loader is involved.

    The production defaults are 512x3. Small explicit configs are useful for
    authored unit tests; this builder makes no runtime/scientific qualification.
    """
    if enable_v5 is not True or type(config) is not V5ModelConfig:
        raise ValueError('explicit v5 model configuration/enablement required')
    from .neural_policy import require_torch
    torch = require_torch()
    nn = torch.nn

    class RMSNorm(nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = nn.Parameter(torch.ones(config.width))

        def forward(self, x):
            full = x.float()
            normalized = full * torch.rsqrt(full.square().mean(-1, keepdim=True) + config.norm_epsilon)
            return normalized.to(x.dtype) * self.weight.to(x.dtype)

    class RoleExperts(nn.Module):
        def __init__(self):
            super().__init__()
            self.experts = nn.ModuleList(nn.Sequential(nn.Linear(config.width, config.expert_width),
                nn.GELU(), nn.Linear(config.expert_width, config.width)) for _ in range(3))
            self.register_buffer('roles', torch.tensor(EXPERT_ROLES, dtype=torch.long), persistent=False)

        def forward(self, x):
            result = torch.zeros_like(x)
            for role, expert in enumerate(self.experts):
                indices = (self.roles == role).nonzero().flatten()
                # Autocast may return BF16 while the residual remains FP32.
                values = expert(x.index_select(1, indices)).to(result.dtype)
                result = result.index_copy(1, indices, values)
            return result

    class Block(nn.Module):
        def __init__(self):
            super().__init__()
            # Preserve compatible v4 attention/norm weight names for ordinary
            # candidate warm starts; new expert weights initialize separately.
            self.norm1 = RMSNorm()
            self.self_attn = nn.MultiheadAttention(config.width, config.attention_heads,
                dropout=config.dropout, batch_first=True)
            self.norm2 = RMSNorm()
            self.role_experts = RoleExperts()
            self.dropout = nn.Dropout(config.dropout)

        def forward(self, x, mask):
            normalized = self.norm1(x)
            value = self.self_attn(normalized, normalized, normalized, key_padding_mask=~mask,
                need_weights=False)[0]
            x = x + self.dropout(value)
            x = x + self.dropout(self.role_experts(self.norm2(x)))
            return x.masked_fill(~mask[..., None], 0)

    class Encoder(nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = nn.ModuleList(Block() for _ in range(config.layers))

        def forward(self, x, mask):
            for block in self.layers:
                x = block(x, mask)
            return x

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.config = config
            self.category_embedding = nn.Embedding(config.category_vocab_size, config.width, padding_idx=0)
            self.token_type_embedding = nn.Embedding(16, config.width)
            self.history_position_embedding = nn.Embedding(1, config.width)
            self.numeric_projection = nn.Linear(NUMERIC_FEATURES, config.width)
            self.encoder = Encoder()
            self.final_norm = RMSNorm()
            self.policy_head = nn.Linear(config.width, 1)
            self.value_query = nn.Parameter(torch.zeros(config.width)) if config.treatment else None
            self.value_head = (nn.Sequential(nn.Linear(config.width, config.value_hidden[0]), nn.GELU(),
                nn.Linear(config.value_hidden[0], config.value_hidden[1]), nn.GELU(),
                nn.Linear(config.value_hidden[1], config.value_bins)) if config.treatment
                else nn.Linear(config.width, 1))
            edges = torch.linspace(-1, 1, config.value_bins + 1)
            self.register_buffer('value_centers', (edges[:-1] + edges[1:]) / 2)
            if config.treatment:
                self.opponent_semantic_head = nn.Linear(config.width, len(config.label_vocabulary.opponent_classes))
                self.hidden_species_head = nn.Linear(config.width, len(config.label_vocabulary.species))
                self.hidden_variant_head = nn.Linear(config.width, MAX_VARIANTS)
                self.survival_head = nn.Linear(config.width, 1)
                self.final_hp_head = nn.Linear(config.width, 1)

        def forward(self, *, categorical_ids, numeric_features, token_type_ids,
                    attention_mask, history_mask, include_auxiliary=False):
            if type(include_auxiliary) is not bool:
                raise ValueError('explicit auxiliary-head boolean required')
            batch = categorical_ids.shape[0]
            shapes = ((categorical_ids, (batch, 1, V5_TOKENS, CATEGORICAL_FEATURES), torch.long),
                (numeric_features, (batch, 1, V5_TOKENS, NUMERIC_FEATURES), torch.float32),
                (token_type_ids, (batch, 1, V5_TOKENS), torch.long),
                (attention_mask, (batch, 1, V5_TOKENS), torch.bool), (history_mask, (batch, 1), torch.bool))
            if batch <= 0 or any(tuple(t.shape) != shape or t.dtype != dtype for t, shape, dtype in shapes):
                raise ValueError('v5 requires typed public Bx1x29 tensors with v4 feature widths')
            if (not torch.isfinite(numeric_features).all() or (categorical_ids < 0).any()
                    or (categorical_ids >= config.category_vocab_size).any()
                    or not history_mask.all() or not attention_mask.any(-1).all()):
                raise ValueError('finite original public categories and active frame required')
            if not attention_mask[:, :, 23:29].all():
                raise ValueError('all six declared v5 belief tokens must be attended')
            expected_roles = torch.tensor(TOKEN_ROLES, dtype=torch.long, device=token_type_ids.device)
            if not (token_type_ids == expected_roles).all():
                raise ValueError('original v4 roles plus six belief role7 tokens required')
            mask = attention_mask[:, 0]
            x = (self.category_embedding(categorical_ids[:, 0]).sum(-2)
                + self.numeric_projection(numeric_features[:, 0])
                + self.token_type_embedding(token_type_ids[:, 0]) + self.history_position_embedding.weight[0])
            x = x.masked_fill(~mask[..., None], 0)
            x = self.encoder(x, mask)
            encoded = self.final_norm(x).masked_fill(~mask[..., None], 0)
            weights = mask.float() / mask.sum(-1, keepdim=True)
            mean = (encoded * weights[..., None]).sum(1)
            policy = self.policy_head(encoded[:, 13:22]).squeeze(-1)
            if config.treatment:
                attention = (encoded.float() @ self.value_query.float()).float() / math.sqrt(config.width)
                attention = attention.masked_fill(~mask, -torch.finfo(torch.float32).max).softmax(-1)
                pooled = (encoded * attention[..., None].to(encoded.dtype)).sum(1)
                logits = self.value_head(pooled)
                value = (logits.float().softmax(-1) * self.value_centers.float()).sum(-1)
            else:
                logits = None
                value = self.value_head(mean).squeeze(-1).tanh()
            auxiliary = {}
            if config.treatment and include_auxiliary:
                beliefs = encoded[:, 23:29]
                pokemon = encoded[:, 1:13]
                auxiliary = dict(opponent_semantic_logits=self.opponent_semantic_head(mean),
                    hidden_species_logits=self.hidden_species_head(beliefs),
                    hidden_variant_logits=self.hidden_variant_head(beliefs),
                    survival_logits=self.survival_head(pokemon).squeeze(-1),
                    final_hp=self.final_hp_head(pokemon).squeeze(-1).sigmoid())
            return V5Output(policy, value, logits, **auxiliary)

    return Model()


def forward_v5(model, tensors, *, include_auxiliary=False):
    """A named public-input boundary; sidecars/labels never become model kwargs."""
    require_v5_bindings(tensors, model.config)
    return model(**{name: tensors[name] for name in MODEL_INPUTS}, include_auxiliary=include_auxiliary)
