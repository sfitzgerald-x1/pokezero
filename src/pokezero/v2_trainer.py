"""V2-c device-resident v5 trainer: the v5 update math without per-batch host work.

The original cache trainer (``v5_training._update``) is unchanged. This module
trains the same model with the same loss and optimizer recipe from a row store
that already lives on the device, and removes the host work that dominated each
update there: per-batch ``validate_batch``, numpy-to-torch copies, the
per-parameter ``isfinite`` host syncs and single-tensor AdamW.

Public API
----------
``V2Config``            Order, execution mode, optimizer kernel and check interval.
``RowStore`` / ``row_store``  Device-resident row store (DESIGN.md interface), validated once.
``validate_store``      The batch checks of ``forward_v5``/``v5_loss``, vectorized over all rows.
``load_cache_store``    Row store from v5 caches via ``VerifiedCache`` + ``v5_batch_to_torch``.
``BatchOrder``          Deterministic ``ordered``/``shuffled`` batches with an (epoch, step) cursor.
``create``              Scratch runtime with the exact ``v5_training.create`` initialization.
``Trainer``             ``train(store, window=..., max_updates=...)`` -> per-update metrics.
``save_checkpoint`` / ``load_checkpoint``  v5 checkpoint format plus a ``v2`` resume section.
``export_actor``        ``gpu_engine.v5_model.export`` applied to a V2 checkpoint.
``NonFiniteUpdate``     Raised when the device-side finiteness flag trips.

Math. ``mode='reference'`` calls the original ``forward_v5`` and ``v5_loss``.
The other modes call ``forward_unchecked`` and ``static_v5_loss``. Those apply the
same modules and terms in the same order, with fixed shapes. Boolean row
selection becomes masked reductions, and data checks move to ``validate_store``.
Rows with ``training_weight == 0`` take part in no loss term. That lets the
compiled modes pad a short final batch to a static shape without changing the
update. The only changes are reduction order and the optimizer kernel; see
``tests/test_v2_trainer.py`` for the measured sizes.
"""
from __future__ import annotations

from contextlib import nullcontext
from dataclasses import asdict, dataclass, fields, replace
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile

from . import v5_contract as contract, v5_data, v5_training as v5
from .v5_contract import MAX_VARIANTS, OBSERVATION_SCHEMA_VERSION_V5, TRAINING_CONTRACT_SCHEMA
from .v5_loss import BASE_PPO_FIELDS, FLOAT_FIELDS, INTEGER_FIELDS, MASK_FIELDS, SEARCH_FIELDS, V5LossConfig, v5_loss
from .v5_model import MODEL_INPUTS, TOKEN_ROLES, V5ModelConfig, V5Output, forward_v5, require_v5_bindings

V2_SCHEMA = 'pokezero.v2_trainer.v1'
BINDINGS = ('schema', 'observation_schema', 'category_vocabulary_sha256', 'label_vocabulary_sha256')
ORDERS = ('ordered', 'shuffled')
MODES = ('reference', 'eager', 'compiled', 'cudagraphs')
OPTIMIZERS = ('fused', 'foreach', 'single')
AUXILIARY_TERMS = ('opponent_action_loss', 'hidden_species_loss', 'hidden_variant_loss', 'survival_loss', 'final_hp_loss')
_FALLBACK = dict(cudagraphs='compiled', compiled='eager')
_V5_GROUP_FLAGS = dict(foreach=False, fused=None, capturable=False, differentiable=False)
_OUTPUT_FIELDS = tuple(item.name for item in fields(V5Output))


def _torch():
    return v5._torch()


def _sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


class NonFiniteUpdate(RuntimeError):
    """A loss, gradient or parameter became non-finite inside a check window."""

    def __init__(self, message, *, first_step, window):
        super().__init__(message)
        self.first_step = first_step
        self.window = window


@dataclass(frozen=True)
class V2Config:
    """Execution choices; none of these is part of the v5 checkpoint math recipe.

    order: ``ordered`` reproduces the original cache reader: per-segment
        sequential slices with each segment's short last batch. ``shuffled``
        draws one seeded permutation of all rows per epoch.
    order_seed: permutation seed; ``None`` uses ``TrainConfig.seed``.
    mode: ``reference`` (original forward_v5/v5_loss, eager), ``eager``,
        ``compiled`` (torch.compile) or ``cudagraphs`` (torch.compile with
        reduce-overhead). A failing compile falls back one level, then to eager.
    optimizer: AdamW kernel: ``fused``, ``foreach`` or ``single`` (the v5 kernel).
    check_interval: updates between host reads of the device finiteness flag.
    pad_partial: pad a short batch to ``batch_size`` with zero-weight rows.
        ``None`` pads only in the compiled modes.
    emulate_precision_casts: have Inductor round fused bf16 intermediates as
        eager autocast does. With it on, compiled CPU updates are bitwise eager.
    """
    order: str = 'ordered'
    order_seed: int | None = None
    mode: str = 'eager'
    optimizer: str = 'fused'
    check_interval: int = 64
    pad_partial: bool | None = None
    emulate_precision_casts: bool = True

    def __post_init__(self):
        if (self.order not in ORDERS or self.mode not in MODES or self.optimizer not in OPTIMIZERS
                or type(self.check_interval) is not int or self.check_interval <= 0
                or self.order_seed is not None and (type(self.order_seed) is not int or not 0 <= self.order_seed < 2**63)
                or self.pad_partial not in (None, True, False) or type(self.emulate_precision_casts) is not bool):
            raise ValueError('explicit V2 order/mode/optimizer/check interval required')
        if self.mode == 'reference' and self.pad_partial:
            raise ValueError('reference mode runs the original unpadded batch')

    def pads(self, mode):
        return self.pad_partial if self.pad_partial is not None else mode in ('compiled', 'cudagraphs')


# ---------------------------------------------------------------------------
# Row store


@dataclass(eq=False)
class RowStore:
    """One row per trainable learner decision, tensors already on ``device``.

    ``tensors`` holds the names/dtypes/shapes of
    ``v5_batch_to_torch(..., training=True)``. ``segments`` are consecutive row
    counts (one per source cache) that ``ordered`` batching respects. ``store_id``
    names the content for resume checks; the cache loader derives it from the
    cache metadata digests.
    """
    tensors: dict
    bindings: dict
    terminal_status: tuple | None
    rows: int
    segments: tuple
    store_id: str | None
    device: object

    def fingerprint(self):
        return dict(rows=self.rows, segments=list(self.segments), store_id=self.store_id,
            fields={name: [list(value.shape[1:]), str(value.dtype)] for name, value in sorted(self.tensors.items())})


def _tensor_names():
    return set(FLOAT_FIELDS) | set(INTEGER_FIELDS) | set(MASK_FIELDS)


def row_store(store, model_config, loss_config, *, segments=None, store_id=None, validate=True):
    """Wrap a ``v5_batch_to_torch``-shaped dict; validate it once (all rows)."""
    torch = _torch()
    if type(model_config) is not V5ModelConfig or type(loss_config) is not V5LossConfig or model_config.arm != loss_config.arm:
        raise ValueError('same-arm explicit v5 model/loss configuration required')
    require_v5_bindings(store, model_config)
    tensors = {name: value for name, value in store.items() if name in _tensor_names() and torch.is_tensor(value)}
    if 'categorical_ids' not in tensors:
        raise ValueError('row store requires public model inputs')
    rows = int(tensors['categorical_ids'].shape[0])
    device = tensors['categorical_ids'].device
    if rows <= 0 or any(value.shape[0] != rows or value.device != device for value in tensors.values()):
        raise ValueError('one nonempty row axis on one device required')
    status = store.get('terminal_status')
    status = None if status is None else tuple(status)
    if status is not None and len(status) != rows:
        raise ValueError('one literal terminal status per row required')
    segments = (rows,) if segments is None else tuple(segments)
    if not segments or any(type(x) is not int or x <= 0 for x in segments) or sum(segments) != rows:
        raise ValueError('positive segments covering every row required')
    result = RowStore(tensors, {name: store[name] for name in BINDINGS}, status, rows, segments, store_id, device)
    if validate:
        validate_store(result, model_config, loss_config)
    return result


def validate_store(store, model_config, loss_config):
    """Every data check made by ``forward_v5`` and ``v5_loss``, once, over all rows.

    Each of those checks is row-wise, so a store that passes here passes them for
    every batch drawn from it. One host sync total.
    """
    torch = _torch()
    t = store.tensors; n = store.rows; mc = model_config; lc = loss_config
    require_v5_bindings(store.bindings, mc)

    def column(name, shape, dtype):
        value = t.get(name)
        if value is None or tuple(value.shape) != (n, *shape) or value.dtype != dtype:
            raise ValueError('missing or wrong-shape v5 row-store column: ' + name)
        return value

    cats = column('categorical_ids', (1, 29, 41), torch.long)
    numeric = column('numeric_features', (1, 29, 132), torch.float32)
    roles = column('token_type_ids', (1, 29), torch.long)
    attention = column('attention_mask', (1, 29), torch.bool)
    history = column('history_mask', (1,), torch.bool)
    legal = column('legal_action_mask', (9,), torch.bool)
    for name in BASE_PPO_FIELDS[:8]:
        column(name, (), torch.long if name == 'action_indices' else torch.bool if name in MASK_FIELDS else torch.float32)
    checks = []

    def require(condition, message):
        checks.append((condition.reshape(()), message))

    require(torch.isfinite(numeric).all() & (cats >= 0).all() & (cats < mc.category_vocab_size).all()
        & history.all() & attention.any(-1).all(), 'finite original public categories and active frame required')
    require(attention[:, :, 23:29].all(), 'all six declared v5 belief tokens must be attended')
    require((roles == torch.tensor(TOKEN_ROLES, dtype=torch.long, device=store.device)).all(),
        'original v4 roles plus six belief role7 tokens required')
    weights = t['training_weights']
    require(torch.isfinite(weights).all() & (weights >= 0).all(), 'finite nonnegative training weights required')
    active = weights > 0
    require((legal.any(-1) | ~active).all(), 'every active actor row needs a legal action')
    ppo = t['ppo_advantage_mask'] & active
    require(~(ppo & ~t['action_probability_mask']).any(), 'PPO advantage requires an original behavior probability')
    actions = t['action_indices']; behavior = t['action_probabilities']; advantage = t['ppo_advantages']
    in_range = (actions >= 0) & (actions < 9)
    chosen_legal = legal.gather(1, torch.where(in_range, actions, 0)[:, None]).squeeze(1)
    require(~(ppo & ~(in_range & chosen_legal & torch.isfinite(behavior) & (behavior > 0) & (behavior <= 1)
        & torch.isfinite(advantage))).any(), 'original legal action/behavior probability/finite PPO advantage required')
    zeros = torch.zeros_like(active)
    spm = t['search_policy_target_mask'] if 'search_policy_target_mask' in t else zeros
    svm = t['search_value_target_mask'] if 'search_value_target_mask' in t else zeros
    if lc.arm != 'D':
        require(~((spm | svm) & active).any(), 'D labels cannot contaminate control or B arms')
    sm = spm & active
    if 'search_policy_targets' in t:
        target = column('search_policy_targets', (9,), torch.float32)
        bad = (~torch.isfinite(target).all(-1) | (target < 0).any(-1)
            | ((target.sum(-1) - 1).abs() > 1e-6) | ((target != 0) & ~legal).any(-1))
        require(~(sm & bad).any(), 'completed legal teacher probability target required')
    elif lc.arm == 'D':
        require(~sm.any(), 'search policy targets missing')
    vm = (t['ppo_value_target_mask'] | svm) & active
    value_target = torch.where(vm & ~svm, t['ppo_value_targets'], 0.)
    if (svm & active).any().item():
        outcome_mask = column('outcome_target_mask', (), torch.bool)
        require(~(svm & active & ~outcome_mask).any(), 'D outcome/root value target needs a known literal terminal outcome')
        value_target = torch.where(svm & active, column('search_value_targets', (), torch.float32), value_target)
    require(~(vm & ~(torch.isfinite(value_target) & (value_target.abs() <= 1))).any(),
        'finite unclipped value targets in[-1,1] required')
    if not mc.treatment and lc.value_clip is not None:
        require(~(vm & ~column('value_estimate_mask', (), torch.bool)).any(), 'A value clipping requires original old values')
    if mc.treatment:
        vocabulary = mc.label_vocabulary
        om = column('opponent_semantic_action_target_mask', (), torch.bool) & active
        ot = column('opponent_semantic_action_targets', (), torch.long)
        require(~(om & ((ot < 0) | (ot >= len(vocabulary.opponent_classes)))).any(), 'bound semantic opponent-action class required')
        hm = column('hidden_set_target_mask', (6,), torch.bool) & active[:, None]
        species = column('hidden_set_species', (6,), torch.long)
        species_ok = (species >= 0) & (species < len(vocabulary.species))
        require(~(hm & ~species_ok).any(), 'bound hidden-set species required')
        hidden = column('hidden_set_targets', (6, MAX_VARIANTS), torch.float32)
        counts = _variant_counts(mc, store.device)[torch.where(species_ok, species, 0)]
        valid = torch.arange(MAX_VARIANTS, device=store.device) < counts[..., None]
        bad = (~torch.isfinite(hidden).all(-1) | (hidden < 0).any(-1) | ((hidden != 0) & ~valid).any(-1)
            | ((hidden.sum(-1) - 1).abs() > 1e-6))
        require(~(hm & bad).any(), 'full source-axis hidden-set probability target required')
        pm = column('pokemon_end_target_mask', (12,), torch.bool) & active[:, None]
        if store.terminal_status is None:
            require(~(pm.any() | (svm & active).any()), 'actual terminal status required for terminal labels')
        else:
            complete = torch.tensor([status == 'COMPLETE' for status in store.terminal_status], device=store.device)
            require(~((pm & ~complete[:, None]).any() | (svm & active & ~complete).any()),
                'terminal labels require actual complete terminal state')
        survived = column('survival_targets', (12,), torch.float32)
        hp = column('final_hp_targets', (12,), torch.float32)
        consistent = (torch.isfinite(survived) & ((survived == 0) | (survived == 1)) & torch.isfinite(hp)
            & (hp >= 0) & (hp <= 1) & ((hp > 0) == survived.bool()))
        require(~(pm & ~consistent).any(), 'literal consistent terminal survival/final HP targets required')
    results = torch.stack([condition for condition, _ in checks]).cpu().tolist()
    for passed, (_, message) in zip(results, checks):
        if not passed:
            raise ValueError(message)


def _variant_counts(model_config, device):
    torch = _torch()
    return torch.tensor([count for _, count in model_config.label_vocabulary.species_variants], dtype=torch.long, device=device)


def load_cache_store(paths, model_config, loss_config, *, device, validate=True):
    """Concatenate v5 caches, in order, through the original reader.

    Each cache is opened with ``v5_data.VerifiedCache`` (array SHA checks and
    ``validate_batch``), checked with ``validate_bootstraps`` and converted
    with ``v5_batch_to_torch(..., training=True)``. Each cache becomes one segment.
    """
    torch = _torch()
    paths = tuple(Path(path) for path in paths)
    if not paths:
        raise ValueError('at least one v5 cache required')
    counts = []; digests = []
    for path in paths:
        metadata = json.loads((path / 'metadata.json').read_text())
        counts.append(int(metadata['example_count'])); digests.append(metadata['metadata_sha256'])
    total = sum(counts); tensors = None; status = []; bindings = None; offset = 0
    for path, count in zip(paths, counts):
        batch = v5_data.VerifiedCache(path)._batch
        if batch.batch_size != count:
            raise ValueError('v5 cache row count differs from metadata')
        v5.validate_bootstraps(batch)
        part = v5._training_tensors(batch, model_config, device)
        names = sorted(name for name, value in part.items() if torch.is_tensor(value))
        if tensors is None:
            bindings = {name: part[name] for name in BINDINGS}
            tensors = {name: torch.empty((total, *part[name].shape[1:]), dtype=part[name].dtype, device=device)
                for name in names}
        elif names != sorted(tensors) or any(part[name] != bindings[name] for name in BINDINGS):
            raise ValueError('v5 caches differ in columns or namespace')
        for name in names:
            tensors[name][offset:offset + count].copy_(part[name])
        status.extend(part.get('terminal_status', ('UNKNOWN',) * count))
        offset += count
        del part, batch
    store = dict(tensors, **bindings, terminal_status=tuple(status))
    return row_store(store, model_config, loss_config, segments=tuple(counts), store_id=_sha(digests), validate=validate)


# ---------------------------------------------------------------------------
# Batch order


class BatchOrder:
    """Deterministic batch list per epoch; global step ``s`` uses batches ``s*W .. s*W+W-1``.

    ``ordered``: every segment in sequence, ``batch_size`` slices, each
    segment's short last batch included. This is the ``train_caches`` order.
    ``shuffled``: a CPU ``randperm`` of all rows seeded by (seed, epoch), sliced
    into batches with the short last batch included.
    """

    def __init__(self, rows, segments, batch_size, *, order, seed, world_size=1):
        if order not in ORDERS or type(batch_size) is not int or batch_size <= 0 or world_size < 1:
            raise ValueError('explicit batch order required')
        self.rows = rows; self.segments = tuple(segments); self.batch_size = batch_size
        self.order = order; self.seed = seed; self.world_size = world_size
        if order == 'ordered':
            self._slices = []
            start = 0
            for count in self.segments:
                for begin in range(start, start + count, batch_size):
                    self._slices.append((begin, min(begin + batch_size, start + count)))
                start += count
            self.batches_per_epoch = len(self._slices)
        else:
            self._slices = None
            self.batches_per_epoch = (rows + batch_size - 1) // batch_size
        self.steps_per_epoch = (self.batches_per_epoch + world_size - 1) // world_size
        self._epoch = None; self._permutation = None

    def describe(self):
        return dict(order=self.order, seed=self.seed, batch_size=self.batch_size, world_size=self.world_size,
            rows=self.rows, segments=list(self.segments), batches_per_epoch=self.batches_per_epoch)

    def permutation(self, epoch):
        torch = _torch()
        if self._epoch != epoch:
            digest = hashlib.sha256(f'{V2_SCHEMA}:shuffle:{self.seed}:{epoch}'.encode()).digest()
            generator = torch.Generator(device='cpu').manual_seed(int.from_bytes(digest[:8], 'little') & (2**63 - 1))
            self._permutation = torch.randperm(self.rows, generator=generator)
            self._epoch = epoch
        return self._permutation

    def bounds(self, index):
        """Row span of batch ``index``: a row range (ordered) or a permutation range."""
        if self._slices is not None:
            return self._slices[index]
        return index * self.batch_size, min((index + 1) * self.batch_size, self.rows)

    def batch_rows(self, index):
        start, stop = self.bounds(index)
        return stop - start

    def step_rows(self, step):
        return sum(self.batch_rows(i) for i in range(step * self.world_size,
            min((step + 1) * self.world_size, self.batches_per_epoch)))

    def rank_batch(self, step, rank):
        index = step * self.world_size + rank
        return index if index < self.batches_per_epoch else None


# ---------------------------------------------------------------------------
# Fixed-shape forward and loss


def expert_indices(model, device):
    """The token indices that ``RoleExperts`` derives with ``nonzero`` on every call."""
    torch = _torch()
    roles = model.encoder.layers[0].role_experts.roles.to(device)
    return tuple((roles == role).nonzero().flatten() for role in range(3))


def forward_unchecked(model, tensors, *, include_auxiliary, experts):
    """``Model.forward`` without its input checks (``validate_store`` made them).

    Same modules in the same order, so it consumes dropout RNG identically. The
    one change is in ``RoleExperts``: the per-call ``nonzero`` is replaced by
    precomputed ``experts`` indices.
    """
    torch = _torch()
    config = model.config
    categorical_ids = tensors['categorical_ids']; numeric_features = tensors['numeric_features']
    token_type_ids = tensors['token_type_ids']; attention_mask = tensors['attention_mask']
    mask = attention_mask[:, 0]
    x = (model.category_embedding(categorical_ids[:, 0]).sum(-2)
        + model.numeric_projection(numeric_features[:, 0])
        + model.token_type_embedding(token_type_ids[:, 0]) + model.history_position_embedding.weight[0])
    x = x.masked_fill(~mask[..., None], 0)
    for block in model.encoder.layers:
        normalized = block.norm1(x)
        value = block.self_attn(normalized, normalized, normalized, key_padding_mask=~mask, need_weights=False)[0]
        x = x + block.dropout(value)
        hidden = block.norm2(x)
        result = torch.zeros_like(hidden)
        for indices, expert in zip(experts, block.role_experts.experts):
            values = expert(hidden.index_select(1, indices)).to(result.dtype)
            result = result.index_copy(1, indices, values)
        x = x + block.dropout(result)
        x = x.masked_fill(~mask[..., None], 0)
    encoded = model.final_norm(x).masked_fill(~mask[..., None], 0)
    weights = mask.float() / mask.sum(-1, keepdim=True)
    mean = (encoded * weights[..., None]).sum(1)
    policy = model.policy_head(encoded[:, 13:22]).squeeze(-1)
    if config.treatment:
        attention = (encoded.float() @ model.value_query.float()).float() / math.sqrt(config.width)
        attention = attention.masked_fill(~mask, -torch.finfo(torch.float32).max).softmax(-1)
        pooled = (encoded * attention[..., None].to(encoded.dtype)).sum(1)
        logits = model.value_head(pooled)
        value = (logits.float().softmax(-1) * model.value_centers.float()).sum(-1)
    else:
        logits = None
        value = model.value_head(mean).squeeze(-1).tanh()
    auxiliary = {}
    if config.treatment and include_auxiliary:
        beliefs = encoded[:, 23:29]; pokemon = encoded[:, 1:13]
        auxiliary = dict(opponent_semantic_logits=model.opponent_semantic_head(mean),
            hidden_species_logits=model.hidden_species_head(beliefs),
            hidden_variant_logits=model.hidden_variant_head(beliefs),
            survival_logits=model.survival_head(pokemon).squeeze(-1),
            final_hp=model.final_hp_head(pokemon).squeeze(-1).sigmoid())
    return V5Output(policy, value, logits, **auxiliary)


def _hl_gauss(targets, bins, sigma_ratio):
    """``value_precision_probe.hl_gauss_targets`` without its host-side range check."""
    torch = _torch()
    edges = torch.linspace(-1, 1, bins + 1, dtype=torch.float64, device=targets.device)
    sigma = sigma_ratio * (2 / bins)
    cdf = 0.5 * (1 + torch.erf((edges - targets.double().unsqueeze(-1)) / (sigma * math.sqrt(2))))
    mass = torch.diff(cdf, dim=-1)
    return (mass / mass.sum(dim=-1, keepdim=True)).float()


def metric_names(model_config, loss_config):
    names = ['entropy', 'uniform_kl', 'policy_loss', 'search_policy_loss', 'value_loss']
    if model_config.treatment:
        names += list(AUXILIARY_TERMS)
    return tuple(name for base in names for name in (base, base + '_count')) + ('loss',)


def static_v5_loss(output, tensors, model_config, config, *, variant_counts):
    """``v5_loss`` with fixed shapes: a masked reduction replaces each boolean selection.

    Returns ``(total, metrics, logits_ok)``. Metrics are device tensors.
    ``logits_ok`` replaces the original's per-batch check that the active policy
    logits are finite. Inactive rows (weight 0) and unselected entries go through
    ``torch.where`` before any log/exp, so they cannot create NaN gradients.
    """
    torch = _torch(); functional = torch.nn.functional
    batch = output.policy_logits.shape[0]
    weights = tensors['training_weights'].detach()
    active = weights > 0
    zero = sum(value.reshape(-1)[:0].sum() for name in _OUTPUT_FIELDS
        if (value := getattr(output, name)) is not None)
    metrics = {}

    def average(name, values, mask, row_weights):
        count = mask.sum()
        w = torch.where(mask, row_weights, 0)
        denominator = torch.where(count > 0, w.sum(), 1)
        result = (torch.where(mask, values, 0) * w).sum() / denominator
        metrics[name] = result.detach(); metrics[name + '_count'] = count
        return result

    legal = tensors['legal_action_mask'] | ~active[:, None]
    logits = output.policy_logits.float()
    logits_ok = torch.isfinite(torch.where(active[:, None], logits, 0)).all()
    log_probs = logits.masked_fill(~legal, -torch.inf).log_softmax(-1)
    safe_log_probs = log_probs.masked_fill(~legal, 0)
    entropy = average('entropy', -(log_probs.exp() * safe_log_probs).sum(-1), active, weights)
    legal_count = legal.sum(-1)
    uniform_kl = average('uniform_kl', -safe_log_probs.sum(-1) / legal_count - legal_count.float().log(), active, weights)
    ppo = tensors['ppo_advantage_mask'] & active
    actions = torch.where(ppo, tensors['action_indices'], 0)
    behavior = torch.where(ppo, tensors['action_probabilities'].detach(), 1.)
    advantage = torch.where(ppo, tensors['ppo_advantages'].detach(), 0.)
    if config.normalize_advantage:
        count = ppo.sum()
        w = torch.where(ppo, weights, 0)
        total_weight = torch.where(count > 0, w.sum(), 1)
        mean = (advantage * w).sum() / total_weight
        variance = (torch.where(ppo, (advantage - mean).square(), 0) * w).sum() / total_weight
        advantage = torch.where(ppo & (count > 1), (advantage - mean) / (variance.sqrt() + 1e-8), advantage)
    chosen = torch.where(ppo, log_probs.gather(1, actions[:, None]).squeeze(-1), 0.)
    log_ratio = chosen - behavior.log()
    positive = ppo & (advantage > 0); negative = ppo & (advantage < 0)
    positive_branch = (torch.where(positive, log_ratio, 0.).clamp(max=math.log1p(config.clip_epsilon))
        + torch.where(positive, advantage, 1.).log()).exp()
    negative_branch = -(torch.where(negative, log_ratio, 0.).clamp(min=math.log1p(-config.clip_epsilon))
        + torch.where(negative, -advantage, 1.).log()).exp()
    surrogate = torch.where(positive, positive_branch, torch.where(negative, negative_branch, 0.))
    policy = average('policy_loss', -surrogate, ppo, weights)
    zeros = torch.zeros_like(active)
    search_policy_mask = tensors.get('search_policy_target_mask', zeros)
    search_value_mask = tensors.get('search_value_target_mask', zeros)
    sm = search_policy_mask & active
    if 'search_policy_targets' in tensors:
        target = torch.where(sm[:, None], tensors['search_policy_targets'].detach(), 0.)
        teacher = average('search_policy_loss', -(target * safe_log_probs).sum(-1), sm, weights)
    else:
        teacher = average('search_policy_loss', torch.zeros_like(weights), sm, weights)
    vm = (tensors['ppo_value_target_mask'] | search_value_mask) & active
    base_mask = vm & ~search_value_mask
    target = torch.where(base_mask, tensors['ppo_value_targets'].detach().float(), 0.)
    if 'search_value_targets' in tensors:
        searched = search_value_mask & active
        target = torch.where(searched, tensors['search_value_targets'].detach().float(), target)
    target = torch.where(vm, target, 0.)
    if model_config.treatment:
        labels = _hl_gauss(target, model_config.value_bins, model_config.value_sigma_ratio)
        value_terms = -(labels * output.value_logits.float().log_softmax(-1)).sum(-1)
    else:
        value_terms = (output.value - target).square()
        if config.value_clip is not None:
            old = torch.where(vm, tensors['value_estimates'].detach(), 0.)
            clipped = old + (output.value - old).clamp(-config.value_clip, config.value_clip)
            value_terms = torch.maximum(value_terms, (clipped - target).square())
    value = average('value_loss', value_terms, vm, weights)
    auxiliary = zero
    if model_config.treatment:
        om = tensors['opponent_semantic_action_target_mask'] & active
        ot = torch.where(om, tensors['opponent_semantic_action_targets'], 0)
        auxiliary = auxiliary + average('opponent_action_loss', functional.cross_entropy(
            output.opponent_semantic_logits.float(), ot, reduction='none'), om, weights)
        hm = tensors['hidden_set_target_mask'] & active[:, None]
        species = torch.where(hm, tensors['hidden_set_species'], 0)
        hw = weights[:, None].expand(batch, 6)
        species_logits = output.hidden_species_logits.float()
        species_terms = functional.cross_entropy(species_logits.reshape(batch * 6, species_logits.shape[-1]),
            species.reshape(-1), reduction='none').reshape(batch, 6)
        auxiliary = auxiliary + average('hidden_species_loss', species_terms, hm, hw)
        hidden_target = torch.where(hm[..., None], tensors['hidden_set_targets'].detach(), 0.)
        valid = torch.arange(MAX_VARIANTS, device=species.device) < variant_counts[species][..., None]
        valid = valid | ~hm[..., None]
        hlog = output.hidden_variant_logits.float().masked_fill(~valid, -torch.inf).log_softmax(-1).masked_fill(~valid, 0)
        auxiliary = auxiliary + average('hidden_variant_loss', -(hidden_target * hlog).sum(-1), hm, hw)
        pm = tensors['pokemon_end_target_mask'] & active[:, None]
        pw = weights[:, None].expand(batch, 12)
        survived = torch.where(pm, tensors['survival_targets'].detach(), 0.)
        hp = torch.where(pm, tensors['final_hp_targets'].detach(), 0.)
        auxiliary = auxiliary + average('survival_loss', functional.binary_cross_entropy_with_logits(
            output.survival_logits.float(), survived, reduction='none'), pm, pw)
        auxiliary = auxiliary + average('final_hp_loss', (output.final_hp.float() - hp).square(), pm, pw)
    total = (policy + config.value_weight * value - config.entropy_coefficient * entropy
        + config.uniform_kl_coefficient * uniform_kl + (config.auxiliary_weight * auxiliary if model_config.treatment else zero)
        + (config.search_policy_weight * teacher if config.arm == 'D' else zero))
    metrics['loss'] = total.detach()
    return total, metrics, logits_ok


def _autocast(torch, device, precision):
    if precision != 'bfloat16':
        return nullcontext()
    return torch.autocast(device_type=torch.device(device).type, dtype=torch.bfloat16)


def _static_forward_loss(model, tensors, model_config, loss_config, precision, device, experts, variant_counts):
    """Forward + loss under the original autocast scope; returns (loss, metric vector, logits_ok)."""
    torch = _torch()
    with _autocast(torch, device, precision):
        output = forward_unchecked(model, tensors, include_auxiliary=model_config.treatment, experts=experts)
        loss, metrics, logits_ok = static_v5_loss(output, tensors, model_config, loss_config, variant_counts=variant_counts)
    vector = torch.stack([metrics[name].float() for name in metric_names(model_config, loss_config)])
    return loss, vector, logits_ok


# ---------------------------------------------------------------------------
# Runtime


@dataclass(eq=False)
class Runtime:
    model_config: V5ModelConfig
    loss_config: V5LossConfig
    train_config: v5.TrainConfig
    v2_config: V2Config
    model: object
    optimizer: object
    rng: dict
    source_contract: dict
    device: str
    world_size: int = 1
    rank: int = 0
    process_group: object = None
    steps: int = 0
    rows: int = 0
    cursor: dict | None = None
    order: dict | None = None
    window: v5.DeclaredGameWindow | None = None
    failed: str | None = None


def _distributed(process_group):
    if process_group is None:
        return 1, 0
    import torch.distributed as dist
    return dist.get_world_size(process_group), dist.get_rank(process_group)


def build_optimizer(parameters, train_config, kind):
    torch = _torch()
    options = dict(lr=train_config.learning_rate, weight_decay=train_config.weight_decay,
        betas=(train_config.beta1, train_config.beta2), eps=train_config.epsilon, amsgrad=False, maximize=False,
        capturable=False, differentiable=False)
    if kind == 'fused':
        return torch.optim.AdamW(parameters, foreach=None, fused=True, **options)
    if kind == 'foreach':
        return torch.optim.AdamW(parameters, foreach=True, fused=None, **options)
    return torch.optim.AdamW(parameters, foreach=False, fused=None, **options)


def create(model_config, loss_config, train_config, v2_config=None, *, enable_v5=False, device=None, process_group=None):
    """Scratch runtime. Model parameters and the RNG state come from ``v5_training.create``.

    ``device`` overrides ``train_config.device`` for placement only (for example,
    one CUDA device per DDP rank). Rank ``r > 0`` derives its own dropout stream
    from (seed, r). All ranks start from the same parameters.
    """
    if enable_v5 is not True:
        raise ValueError('explicit v5 trainer enablement required')
    v2_config = V2Config() if v2_config is None else v2_config
    if type(v2_config) is not V2Config:
        raise ValueError('explicit V2 configuration required')
    torch = _torch()
    device = train_config.device if device is None else device
    placed = replace(train_config, device=device)
    base = v5.create(model_config, loss_config, placed, enable_v5=True)
    world_size, rank = _distributed(process_group)
    rng = base.rng
    if rank:
        digest = hashlib.sha256(f'{V2_SCHEMA}:rank:{train_config.seed}:{rank}'.encode()).digest()
        with v5._rng_scope(torch, device, seed=int.from_bytes(digest[:4], 'little')):
            rng = v5._rng_state(torch, device.startswith('cuda'))
    optimizer = build_optimizer(base.model.parameters(), train_config, v2_config.optimizer)
    runtime = Runtime(model_config, loss_config, train_config, v2_config, base.model, optimizer, rng,
        contract.contract_descriptor(), device, world_size, rank, process_group)
    if world_size > 1:
        import torch.distributed as dist
        with torch.no_grad():
            for parameter in runtime.model.parameters():
                dist.broadcast(parameter.data, 0, group=process_group)
    return runtime


def learning_rate(train_config, window, epoch):
    """The rate ``train_caches`` applies during 1-based ``epoch``."""
    if train_config.learning_rate_schedule == 'constant':
        return train_config.learning_rate
    return v5._epoch_learning_rate(train_config, window, epoch)


# ---------------------------------------------------------------------------
# Trainer


class Trainer:
    """Owns the per-runtime execution state: compiled function, buffers, order cache."""

    def __init__(self, runtime):
        if type(runtime) is not Runtime:
            raise ValueError('V2 runtime required')
        torch = _torch()
        self.runtime = runtime
        self.requested_mode = runtime.v2_config.mode
        self.mode = self.requested_mode
        self.fallbacks = []
        self._compiled = None; self._warm = False
        self._experts = expert_indices(runtime.model, runtime.device)
        self._variant_counts = _variant_counts(runtime.model_config, runtime.device)
        self._names = metric_names(runtime.model_config, runtime.loss_config)
        self._parameters = [p for p in runtime.model.parameters()]
        self._buffer = torch.zeros((runtime.v2_config.check_interval, len(self._names) + 3),
            dtype=torch.float32, device=runtime.device)
        self._order = None; self._order_key = None; self._device_permutation = None
        self._build(self.mode)

    # -- execution mode -----------------------------------------------------
    def _build(self, mode):
        torch = _torch()
        self._compiled = None; self._warm = False
        if mode in ('compiled', 'cudagraphs'):
            # 'reduce-overhead' is exactly triton.cudagraphs; options and mode are exclusive.
            options = {'emulate_precision_casts': self.runtime.v2_config.emulate_precision_casts}
            if mode == 'cudagraphs':
                options['triton.cudagraphs'] = True
            try:
                self._compiled = torch.compile(_static_forward_loss, dynamic=False, options=options)
            except Exception as error:
                self._fallback(mode, error)

    def _fallback(self, mode, error):
        target = _FALLBACK[mode]
        self.fallbacks.append(dict(mode=mode, fallback=target, error=f'{type(error).__name__}: {error}'[:2000]))
        self.mode = target
        self._build(target)

    def mode_report(self):
        return dict(requested=self.requested_mode, active=self.mode, fallbacks=list(self.fallbacks),
            optimizer=self.runtime.v2_config.optimizer, pad_partial=self.runtime.v2_config.pads(self.mode))

    def _call(self, arguments):
        torch = _torch()
        if self._compiled is None:
            return _static_forward_loss(*arguments)
        if self.mode == 'cudagraphs':
            torch.compiler.cudagraph_mark_step_begin()
        return self._compiled(*arguments)

    def _forward_backward(self, batch):
        """Static forward+loss+backward; the first compiled call may fall back.

        Inductor compiles the backward lazily, so the fallback also covers
        ``loss.backward()``. RNG and gradients are reset before the retry.
        """
        rt = self.runtime; torch = _torch()
        arguments = (rt.model, batch, rt.model_config, rt.loss_config, rt.train_config.precision, rt.device,
            self._experts, self._variant_counts)
        while self._compiled is not None and not self._warm:
            rng = v5._rng_state(torch, rt.device.startswith('cuda'))
            try:
                loss, vector, logits_ok = self._call(arguments)
                loss.backward()
                self._warm = True
                return loss, vector, logits_ok
            except Exception as error:
                v5._restore_rng(torch, rng)
                rt.optimizer.zero_grad(set_to_none=True)
                self._fallback(self.mode, error)
        loss, vector, logits_ok = self._call(arguments)
        loss.backward()
        return loss, vector, logits_ok

    # -- data ---------------------------------------------------------------
    def _bind(self, store, window):
        rt = self.runtime; config = rt.train_config
        if type(store) is not RowStore or not _same_device(store.device, rt.device):
            raise ValueError('row store on the runtime device required')
        scheduled = config.learning_rate_schedule != 'constant'
        if scheduled and type(window) is not v5.DeclaredGameWindow:
            raise ValueError('explicit immutable declared-game learning-rate window required')
        if not scheduled and window is not None:
            raise ValueError('constant learning rate does not accept a scheduled window')
        seed = config.seed if rt.v2_config.order_seed is None else rt.v2_config.order_seed
        order = BatchOrder(store.rows, store.segments, config.batch_size, order=rt.v2_config.order,
            seed=seed, world_size=rt.world_size)
        description = dict(order.describe(), store=_sha(store.fingerprint()))
        if rt.cursor is not None and not rt.cursor['complete']:
            if description != rt.order or window != rt.window:
                raise ValueError('resume requires the original row store, order and declared window')
        elif rt.cursor is not None and description == rt.order and window == rt.window:
            pass  # Completed: the same store and window is a no-op, as in train_caches.
        else:
            if scheduled:
                previous = rt.window
                if previous is None:
                    if window.start_game != 0:
                        raise ValueError('fresh schedule must start at declared game zero')
                elif window != previous and (previous.end_game != window.start_game or previous.total_games != window.total_games):
                    raise ValueError('scheduled dataset transition requires contiguous declared windows')
                elif window != previous and rt.order is not None and rt.order['store'] == description['store']:
                    raise ValueError('new declared window cannot replay the previous row store')
                elif window == previous and rt.cursor is not None:
                    raise ValueError('a new row store requires its own declared window')
            rt.cursor = dict(epoch=0, step=0, epochs=config.epochs, steps_per_epoch=order.steps_per_epoch, complete=False)
            rt.order = description; rt.window = window
        key = (id(store), _sha(description))
        if self._order_key != key:
            self._order = order; self._order_key = key; self._device_permutation = None
        return self._order

    def _batch(self, store, order, epoch, index, pad):
        """Batch tensors for order batch ``index`` (or None), and its real row count."""
        torch = _torch()
        if index is None:
            return None, 0, None
        start, stop = order.bounds(index); real = stop - start
        width = order.batch_size if pad else real
        if order.order == 'ordered':
            host = None
            if width == real:
                batch = {name: value[start:stop] for name, value in store.tensors.items()}
                host = slice(start, stop)
            else:
                indices = start + torch.arange(width, device=store.device) % real
                batch = {name: value.index_select(0, indices) for name, value in store.tensors.items()}
        else:
            if self._device_permutation is None or self._device_permutation[0] != epoch:
                permutation = order.permutation(epoch)
                self._device_permutation = (epoch, permutation.to(store.device))
            permutation = self._device_permutation[1]
            indices = permutation[start:stop]
            host = order.permutation(epoch)[start:stop].tolist() if self.mode == 'reference' else None
            if width != real:
                indices = indices[torch.arange(width, device=store.device) % real]
            batch = {name: value.index_select(0, indices) for name, value in store.tensors.items()}
        if width != real:
            keep = torch.arange(width, device=store.device) < real
            batch['training_weights'] = torch.where(keep, batch['training_weights'], 0.)
        return batch, real, host

    # -- one optimizer step -------------------------------------------------
    def _reference_forward_loss(self, batch, store, host):
        torch = _torch(); rt = self.runtime
        tensors = dict(batch, **store.bindings)
        if store.terminal_status is not None:
            tensors['terminal_status'] = (store.terminal_status[host] if isinstance(host, slice)
                else tuple(store.terminal_status[i] for i in host))
        with _autocast(torch, rt.device, rt.train_config.precision):
            output = forward_v5(rt.model, tensors, include_auxiliary=rt.model_config.treatment)
            loss, metrics = v5_loss(output, tensors, rt.model_config, rt.loss_config)
        vector = torch.tensor([float(metrics.get(name, 0.)) for name in self._names], device=rt.device)
        logits_ok = torch.isfinite(output.policy_logits[batch['training_weights'] > 0].float()).all()
        return loss, vector, logits_ok

    def _step(self, batch, store, host):
        torch = _torch(); rt = self.runtime
        rt.optimizer.zero_grad(set_to_none=True)
        if batch is not None:
            if self.mode == 'reference':
                loss, vector, logits_ok = self._reference_forward_loss(batch, store, host)
                loss.backward()
            else:
                loss, vector, logits_ok = self._forward_backward(batch)
            ok = torch.isfinite(loss) & logits_ok
        else:
            vector = torch.zeros(len(self._names), device=rt.device)
            ok = torch.ones((), dtype=torch.bool, device=rt.device)
        if rt.world_size > 1:
            self._all_reduce_gradients(batch is not None)
        gradients = [p.grad for p in self._parameters if p.grad is not None]
        if rt.train_config.max_grad_norm is not None:
            norm = torch.nn.utils.clip_grad_norm_(self._parameters, rt.train_config.max_grad_norm, error_if_nonfinite=False)
        else:
            norm = torch.linalg.vector_norm(torch.stack([torch.linalg.vector_norm(g.float()) for g in gradients]))
        rt.optimizer.step()
        finite_parameters = torch.isfinite(torch.stack(torch._foreach_norm(self._parameters))).all()
        ok = ok & torch.isfinite(norm) & finite_parameters
        return torch.cat([vector, norm.detach().float().reshape(1), ok.float().reshape(1),
            torch.ones(1, device=rt.device) if batch is not None else torch.zeros(1, device=rt.device)])

    def _all_reduce_gradients(self, participating):
        """Sum gradients across ranks, then divide by the number of ranks with a batch."""
        torch = _torch(); rt = self.runtime
        import torch.distributed as dist
        gradients = [torch.zeros_like(p) if p.grad is None else p.grad for p in self._parameters]
        flag = torch.full((1,), float(participating), dtype=gradients[0].dtype, device=gradients[0].device)
        flat = torch.cat([g.reshape(-1) for g in gradients] + [flag])
        dist.all_reduce(flat, group=rt.process_group)
        count = flat[-1].clamp(min=1)
        offset = 0
        for parameter, gradient in zip(self._parameters, gradients):
            size = gradient.numel()
            parameter.grad = (flat[offset:offset + size] / count).view_as(gradient)
            offset += size

    # -- loop ---------------------------------------------------------------
    def train(self, store, *, window=None, max_updates=None):
        """Continue the runtime's cursor over ``store``; returns per-update metrics.

        The device finiteness flag is read every ``check_interval`` updates, at
        each epoch end and on return. A failure raises ``NonFiniteUpdate`` naming
        the update window. The runtime is then marked failed; reload the last
        checkpoint.
        """
        rt = self.runtime; torch = _torch(); config = rt.train_config
        if rt.failed:
            raise RuntimeError('runtime failed a finiteness check; reload the last checkpoint: ' + rt.failed)
        if max_updates is not None and (type(max_updates) is not int or max_updates < 0):
            raise ValueError('nonnegative update budget required')
        order = self._bind(store, window)
        cursor = rt.cursor; pad = rt.v2_config.pads(self.mode)
        updates = []; pending = []; current_rate = None
        with v5._rng_scope(torch, rt.device, state=rt.rng):
            rt.model.train()
            while not cursor['complete'] and (max_updates is None or len(updates) + len(pending) < max_updates):
                epoch, step = cursor['epoch'], cursor['step']
                rate = learning_rate(config, rt.window, epoch + 1)
                if rate != current_rate:
                    for group in rt.optimizer.param_groups:
                        group['lr'] = rate
                    current_rate = rate
                batch, real, host = self._batch(store, order, epoch, order.rank_batch(step, rt.rank), pad and self.mode != 'reference')
                self._buffer[len(pending)].copy_(self._step(batch, store, host))
                rt.steps += 1; rt.rows += order.step_rows(step)
                pending.append(dict(optimizer_step=rt.steps, owner_rows=real, learning_rate=rate,
                    learning_rate_epoch=epoch + 1, epoch=epoch, epoch_step=step))
                cursor['step'] += 1
                epoch_end = cursor['step'] == cursor['steps_per_epoch']
                if epoch_end:
                    cursor['step'] = 0; cursor['epoch'] += 1
                    cursor['complete'] = cursor['epoch'] == cursor['epochs']
                if len(pending) == len(self._buffer) or epoch_end:
                    updates.extend(self._flush(pending)); pending = []
            if pending:
                updates.extend(self._flush(pending))
            rt.rng = v5._rng_state(torch, rt.device.startswith('cuda'))
        return dict(schema=V2_SCHEMA, steps=rt.steps, rows=rt.rows, complete=cursor['complete'],
            cursor=dict(cursor), updates=updates, mode=self.mode_report())

    def _flush(self, pending):
        torch = _torch(); rt = self.runtime
        values = self._buffer[:len(pending)]
        if rt.world_size > 1:
            import torch.distributed as dist
            flags = values[:, -2].clone()
            dist.all_reduce(flags, op=dist.ReduceOp.MIN, group=rt.process_group)
            values = torch.cat([values[:, :-2], flags[:, None], values[:, -1:]], 1)
        host = values.cpu().tolist()
        results = []
        for record, row in zip(pending, host):
            if row[-2] != 1.0:
                first = record['optimizer_step']; window = (pending[0]['optimizer_step'], pending[-1]['optimizer_step'])
                rt.failed = (f'non-finite loss, gradient or parameter in update window steps {window[0]}..{window[1]} '
                    f'(first failing step {first}, epoch {record["epoch"] + 1}, epoch step {record["epoch_step"]})')
                raise NonFiniteUpdate(rt.failed, first_step=first, window=window)
            metrics = dict(record)
            if row[-1] == 1.0:
                for name, value in zip(self._names, row):
                    metrics[name] = int(value) if name.endswith('_count') else value
            metrics['gradient_norm'] = row[len(self._names)]
            results.append(metrics)
        return results


def _same_device(actual, requested):
    torch = _torch()
    requested = torch.device(requested)
    return actual.type == requested.type and (requested.index is None or actual.index == requested.index)


# ---------------------------------------------------------------------------
# Checkpoint


def _canonical_optimizer_state(optimizer):
    """The v5 AdamW state_dict form: v5 group flags, CPU float32 step counters."""
    torch = _torch()
    state = optimizer.state_dict()
    groups = []
    for group in state['param_groups']:
        groups.append(dict(group, **_V5_GROUP_FLAGS))
    values = {}
    for key, entry in state['state'].items():
        values[key] = {name: (value.detach().to('cpu', torch.float32).clone() if name == 'step'
            else value.detach().cpu().clone()) for name, value in entry.items()}
    return dict(state=values, param_groups=groups)


def _load_optimizer_state(optimizer, saved):
    current = optimizer.param_groups[0]
    groups = [dict(group, **{name: current[name] for name in _V5_GROUP_FLAGS}) for group in saved['param_groups']]
    optimizer.load_state_dict(dict(state=saved['state'], param_groups=groups))


def _v5_metadata(runtime):
    shim = v5.Runtime(runtime.model_config, runtime.loss_config, runtime.train_config, runtime.model,
        runtime.optimizer, runtime.rng, runtime.source_contract, steps=runtime.steps, rows=runtime.rows)
    return v5._metadata(shim)


def _v2_section(runtime, rng_by_rank):
    return dict(schema=V2_SCHEMA, config=asdict(runtime.v2_config), world_size=runtime.world_size,
        steps=runtime.steps, rows=runtime.rows, cursor=runtime.cursor, order=runtime.order,
        window=None if runtime.window is None else asdict(runtime.window),
        optimizer_flags={name: runtime.optimizer.param_groups[0][name] for name in _V5_GROUP_FLAGS},
        rng_by_rank=rng_by_rank)


def _v2_digest(section):
    return _sha({key: value for key, value in section.items() if key != 'rng_by_rank'})


def save_checkpoint(path, runtime):
    """Write a v5-format checkpoint with a ``v2`` resume section; never overwrite.

    ``gpu_engine.v5_model.export`` reads it directly. ``metadata.progress`` is
    None because the V2 cursor lives in ``v2``. Under DDP every rank calls this;
    rank 0 writes the file, with each rank's RNG state.
    """
    if type(runtime) is not Runtime:
        raise ValueError('V2 runtime required')
    if runtime.failed:
        raise RuntimeError('refusing to checkpoint a runtime that failed a finiteness check')
    torch = _torch()
    rng_by_rank = [runtime.rng]
    if runtime.world_size > 1:
        import torch.distributed as dist
        rng_by_rank = [None] * runtime.world_size
        dist.all_gather_object(rng_by_rank, runtime.rng, group=runtime.process_group)
    path = Path(path); receipt = None
    if runtime.rank == 0:
        if path.exists():
            raise FileExistsError(path)
        metadata = _v5_metadata(runtime)
        section = _v2_section(runtime, rng_by_rank)
        payload = dict(metadata=metadata, metadata_sha256=_sha(metadata),
            state_dict={name: value.detach().cpu().clone() for name, value in runtime.model.state_dict().items()},
            optimizer_state=_canonical_optimizer_state(runtime.optimizer), rng=rng_by_rank[0],
            v2=section, v2_sha256=_v2_digest(section))
        with tempfile.NamedTemporaryFile(dir=path.parent, prefix='.' + path.name + '.', suffix='.tmp', delete=False) as stream:
            temporary = Path(stream.name); torch.save(payload, stream); stream.flush(); os.fsync(stream.fileno())
        os.link(temporary, path); temporary.unlink()
        receipt = dict(path=str(path), sha256=v5_data._file_sha(path), steps=runtime.steps, rows=runtime.rows,
            schema=v5.CHECKPOINT_SCHEMA, v2_schema=V2_SCHEMA)
    if runtime.world_size > 1:
        import torch.distributed as dist
        dist.barrier(group=runtime.process_group)
    return receipt


def load_checkpoint(path, *, enable_v5=False, v2_config=None, expected_model_config=None, expected_loss_config=None,
                    expected_train_config=None, device=None, process_group=None):
    """Resume a V2 checkpoint exactly: model, optimizer, RNG, cursor and counters.

    ``v2_config`` may change execution-only choices (mode, check interval,
    optimizer kernel). ``order``/``order_seed`` must match an unfinished cursor.
    """
    if enable_v5 is not True:
        raise ValueError('explicit v5 trainer enablement required')
    torch = _torch()
    payload = torch.load(Path(path), map_location='cpu', weights_only=True)
    metadata = payload.get('metadata', {}); section = payload.get('v2')
    if (metadata.get('schema') != v5.CHECKPOINT_SCHEMA or payload.get('metadata_sha256') != _sha(metadata)
            or metadata.get('training_schema') != TRAINING_CONTRACT_SCHEMA
            or metadata.get('observation_schema') != OBSERVATION_SCHEMA_VERSION_V5
            or metadata.get('source_contract_sha256') != _sha(contract.contract_descriptor())
            or not isinstance(section, dict) or section.get('schema') != V2_SCHEMA
            or payload.get('v2_sha256') != _v2_digest(section)):
        raise ValueError('explicit intact same-source V2 checkpoint required')
    model = V5ModelConfig.from_metadata(metadata['model'])
    loss = V5LossConfig(**{field.name: metadata['loss'][field.name] for field in fields(V5LossConfig)})
    training = v5.TrainConfig(**metadata['training'])
    for expected, actual in ((expected_model_config, model), (expected_loss_config, loss), (expected_train_config, training)):
        if expected is not None and expected != actual:
            raise ValueError('explicit resume configuration differs')
    saved = V2Config(**section['config'])
    v2_config = saved if v2_config is None else v2_config
    if section['cursor'] is not None and not section['cursor']['complete'] and (
            (v2_config.order, v2_config.order_seed) != (saved.order, saved.order_seed)):
        raise ValueError('an unfinished cursor resumes only with its original batch order')
    runtime = create(model, loss, training, v2_config, enable_v5=True, device=device, process_group=process_group)
    if runtime.world_size != section['world_size']:
        raise ValueError('resume requires the original world size')
    runtime.model.load_state_dict(payload['state_dict'], strict=True)
    _load_optimizer_state(runtime.optimizer, payload['optimizer_state'])
    if any(not torch.isfinite(parameter).all() for parameter in runtime.model.parameters()):
        raise ValueError('finite V2 checkpoint parameters required')
    runtime.steps = section['steps']; runtime.rows = section['rows']
    runtime.cursor = section['cursor']; runtime.order = section['order']
    runtime.window = None if section['window'] is None else v5.DeclaredGameWindow(**section['window'])
    runtime.rng = section['rng_by_rank'][runtime.rank]
    with v5._rng_scope(torch, runtime.device, state=runtime.rng):
        pass
    runtime.model.train(metadata['model_training'])
    return runtime


def export_actor(checkpoint, out):
    """JAX actor archive from a V2 checkpoint, through ``gpu_engine.v5_model.export``."""
    from gpu_engine import v5_model as actor
    return actor.export(checkpoint, out, expected_checkpoint_sha256=v5_data._file_sha(checkpoint), enable_v5=True)
