"""Opt-in v5 tensor/loss consumers, without a trainer or collection loop."""
from __future__ import annotations

from dataclasses import asdict, dataclass, fields
import math
from numbers import Real
from collections.abc import Mapping

from .v5_contract import MAX_VARIANTS
from .v5_model import MODEL_INPUTS, V5ModelConfig, V5Output, require_v5_bindings


BASE_PPO_FIELDS = ('action_indices', 'action_probabilities', 'action_probability_mask',
    'ppo_advantages', 'ppo_advantage_mask', 'ppo_value_targets', 'ppo_value_target_mask',
    'training_weights', 'value_estimates', 'value_estimate_mask')
SEARCH_FIELDS = ('search_policy_targets', 'search_policy_target_mask', 'search_value_targets', 'search_value_target_mask')
FLOAT_FIELDS = ('numeric_features', 'action_probabilities', 'ppo_advantages', 'ppo_value_targets',
    'training_weights', 'value_estimates', 'search_policy_targets', 'search_value_targets',
    'hidden_set_targets', 'survival_targets', 'final_hp_targets', 'outcome_targets')
INTEGER_FIELDS = ('categorical_ids', 'token_type_ids', 'action_indices',
    'opponent_semantic_action_targets', 'hidden_set_species')
MASK_FIELDS = ('attention_mask', 'history_mask', 'legal_action_mask', 'action_probability_mask',
    'ppo_advantage_mask', 'ppo_value_target_mask', 'value_estimate_mask',
    'search_policy_target_mask', 'search_value_target_mask', 'opponent_semantic_action_target_mask',
    'hidden_set_target_mask', 'pokemon_end_target_mask', 'outcome_target_mask')


@dataclass(frozen=True)
class V5LossConfig:
    arm: str = 'B'
    clip_epsilon: float = 0.2
    value_weight: float = 0.25
    entropy_coefficient: float = 0.01
    uniform_kl_coefficient: float | None = None
    auxiliary_weight: float = 0.1
    search_policy_weight: float = 1.0
    value_lambda: float = 0.95
    policy_lambda: float = 0.95
    value_clip: float | None = None
    normalize_advantage: bool = True
    historical_opponent_pool: bool | None = None

    def __post_init__(self):
        treatment = self.arm != 'A'
        if self.arm not in ('A', 'B', 'D'):
            raise ValueError('initial v5 loss arms are A,B,D')
        if self.uniform_kl_coefficient is None:
            object.__setattr__(self, 'uniform_kl_coefficient', 0.01 if treatment else 0.0)
        if self.historical_opponent_pool is None:
            object.__setattr__(self, 'historical_opponent_pool', treatment)
        for name in ('clip_epsilon', 'value_weight', 'entropy_coefficient', 'uniform_kl_coefficient',
                'auxiliary_weight', 'search_policy_weight', 'value_lambda', 'policy_lambda'):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, Real):
                raise ValueError('real loss hyperparameters required')
            value = float(value)
            object.__setattr__(self, name, value)
            if not math.isfinite(value) or value < 0:
                raise ValueError('finite nonnegative v5 loss settings required')
        if (not 0 < self.clip_epsilon < 1 or not 0 <= self.policy_lambda <= 1
                or not (0.95 if treatment else 0) <= self.value_lambda <= 1
                or type(self.normalize_advantage) is not bool
                or type(self.historical_opponent_pool) is not bool
                or self.historical_opponent_pool != treatment):
            raise ValueError('independent value lambda>=.95 and explicit arm opponent-pool setting required')
        if self.value_clip is not None:
            if (treatment or isinstance(self.value_clip, bool) or not isinstance(self.value_clip, Real)
                    or not math.isfinite(self.value_clip) or self.value_clip <= 0):
                raise ValueError('B/D value clipping is removed; optional A clipping must be positive')
            object.__setattr__(self, 'value_clip', float(self.value_clip))
        if self.arm == 'A' and self.uniform_kl_coefficient:
            raise ValueError('uniform KL belongs to the B treatment')

    def metadata(self):
        return dict(schema='pokezero.loss.v5', **asdict(self),
            uniform_kl_direction='UNIFORM_TO_POLICY_LEGAL_ACTIONS',
            search_value_target='ALREADY_OUTCOME_ROOT_MIX_NO_SECOND_BLEND',
            historical_opponent_pool_scope='COLLECTOR_MUST_IMPLEMENT_CONFIGURED_POOL')


def v5_batch_to_torch(batch, model_config, *, device='cpu', training=False):
    """Consume explicit v5_columns plus untouched existing PPO/search columns.

    The data adapter owns cache/provenance validation. Only named public model
    tensors and named targets are tensorized; privileged sidecars stay outside.
    """
    from .neural_policy import require_torch
    torch = require_torch()
    columns = batch if isinstance(batch, Mapping) else getattr(batch, 'v5_columns', None)
    if not isinstance(columns, Mapping):
        raise ValueError('explicit v5 columns required; a legacy batch is not v5')
    require_v5_bindings(columns, model_config)
    resolved = dict(columns)
    if not isinstance(batch, Mapping):
        for name in MODEL_INPUTS + ('legal_action_mask',) + BASE_PPO_FIELDS + SEARCH_FIELDS:
            value = getattr(batch, name, None)
            if value is not None:
                resolved[name] = value
    required = set(MODEL_INPUTS) | {'legal_action_mask'}
    if training:
        required |= set(BASE_PPO_FIELDS[:8])
    if required - set(resolved):
        raise ValueError('required public/PPO tensor columns missing: ' + repr(sorted(required - set(resolved))))
    result = {key: columns[key] for key in ('schema', 'observation_schema',
        'category_vocabulary_sha256', 'label_vocabulary_sha256')}
    if training and 'terminal_status' in columns:
        result['terminal_status'] = tuple(columns['terminal_status'])
    names = required | ((set(FLOAT_FIELDS) | set(INTEGER_FIELDS) | set(MASK_FIELDS)) & set(resolved) if training else set())
    integer_types = (torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64)
    for name in names:
        raw = torch.as_tensor(resolved[name], device=device)
        if name in MASK_FIELDS:
            if raw.dtype != torch.bool:
                raise ValueError('boolean target/input mask required: ' + name)
            result[name] = raw
        elif name in INTEGER_FIELDS:
            if raw.dtype not in integer_types:
                raise ValueError('integer source target/category required: ' + name)
            result[name] = raw.long()
        else:
            if raw.dtype == torch.bool or raw.is_complex():
                raise ValueError('real numeric tensor required: ' + name)
            result[name] = raw.float()
    return result


def gae_targets(rewards, values, next_values, terminated, trace_continues, *,
                discount=1.0, policy_lambda=0.95, value_lambda=0.95):
    """Separate GAE traces on caller-owned ordered trajectory arrays [time,...].

    A nonterminal truncation keeps its supplied next-value bootstrap and ends
    the trace. True terminals have zero bootstrap; no unknown outcome is zeroed.
    """
    from .neural_policy import require_torch
    torch = require_torch()
    if (any(isinstance(x, bool) or not isinstance(x, Real) or not math.isfinite(x) or not 0 <= x <= 1
                for x in (discount, policy_lambda, value_lambda))
            or rewards.shape != values.shape or values.shape != next_values.shape
            or terminated.shape != values.shape or trace_continues.shape != values.shape
            or values.ndim < 1 or values.shape[0] == 0 or terminated.dtype != torch.bool
            or trace_continues.dtype != torch.bool or (terminated & trace_continues).any()
            or trace_continues[-1].any() or not values.dtype.is_floating_point
            or not all(torch.isfinite(x).all() for x in (rewards, values, next_values))):
        raise ValueError('finite aligned traces and explicit terminal/continuation masks required')
    with torch.no_grad():
        delta = rewards + discount * (~terminated) * next_values - values
        def trace(lam):
            running = torch.zeros_like(delta[0]); result = torch.empty_like(delta)
            for index in range(len(delta) - 1, -1, -1):
                running = delta[index] + discount * lam * trace_continues[index] * running
                result[index] = running
            return result
        return trace(policy_lambda), trace(value_lambda) + values


def v5_loss(output, tensors, model_config, config):
    """Weighted PPO plus separately masked value, auxiliary and D label terms."""
    from .neural_policy import require_torch
    from .value_precision_probe import hl_gauss_targets
    torch = require_torch(); functional = torch.nn.functional
    if type(output) is not V5Output or type(model_config) is not V5ModelConfig or type(config) is not V5LossConfig:
        raise ValueError('typed v5 model output/config/loss config required')
    require_v5_bindings(tensors, model_config)
    if model_config.arm != config.arm:
        raise ValueError('model/loss arm mismatch')
    batch = output.policy_logits.shape[0]
    if tuple(output.policy_logits.shape) != (batch, 9) or tuple(output.value.shape) != (batch,):
        raise ValueError('original v5 policy/value output shapes required')
    def column(name, shape, *, mask=False):
        if name not in tensors or tuple(tensors[name].shape) != shape:
            raise ValueError('missing or wrong-shape v5 target: ' + name)
        value = tensors[name]
        if mask and value.dtype != torch.bool:
            raise ValueError('boolean v5 mask required: ' + name)
        return value
    weights = column('training_weights', (batch,)).detach()
    if not torch.isfinite(weights).all() or (weights < 0).any():
        raise ValueError('finite nonnegative training weights required')
    active = weights > 0
    zero = sum(value.reshape(-1)[:0].sum() for item in fields(output)
        if (value := getattr(output, item.name)) is not None)
    metrics = {}
    def average(name, values, row_weights):
        if values.numel() == 0:
            result = zero
        else:
            if not torch.isfinite(values).all():
                raise ValueError('nonfinite active loss term: ' + name)
            result = (values * row_weights).sum() / row_weights.sum()
        metrics[name] = float(result.detach())
        metrics[name + '_count'] = values.numel()
        return result
    legal = column('legal_action_mask', (batch, 9), mask=True)
    if not legal[active].any(-1).all():
        raise ValueError('every active actor row needs a legal action')
    logits = output.policy_logits[active].float()
    if not torch.isfinite(logits).all():
        raise ValueError('finite active public policy logits required')
    log_probs = logits.masked_fill(~legal[active], -torch.inf).log_softmax(-1)
    safe_log_probs = log_probs.masked_fill(~legal[active], 0)
    local_weights = weights[active]
    entropy = average('entropy', -(log_probs.exp() * safe_log_probs).sum(-1), local_weights)
    legal_count = legal[active].sum(-1)
    uniform_kl = average('uniform_kl', -safe_log_probs.sum(-1) / legal_count - legal_count.float().log(), local_weights)
    ppo_mask = column('ppo_advantage_mask', (batch,), mask=True)[active]
    behavior_mask = column('action_probability_mask', (batch,), mask=True)[active]
    if (ppo_mask & ~behavior_mask).any():
        raise ValueError('PPO advantage requires an original behavior probability')
    actions = column('action_indices', (batch,))[active][ppo_mask]
    behavior = column('action_probabilities', (batch,))[active][ppo_mask].detach()
    advantage = column('ppo_advantages', (batch,))[active][ppo_mask].detach()
    if (actions.dtype != torch.long or (actions < 0).any() or (actions >= 9).any()
            or not legal[active][ppo_mask].gather(1, actions[:, None]).all()
            or not torch.isfinite(behavior).all() or (behavior <= 0).any() or (behavior > 1).any()
            or not torch.isfinite(advantage).all()):
        raise ValueError('original legal action/behavior probability/finite PPO advantage required')
    objective_weights = local_weights[ppo_mask]
    if config.normalize_advantage and advantage.numel() > 1:
        mean = (advantage * objective_weights).sum() / objective_weights.sum()
        variance = ((advantage - mean).square() * objective_weights).sum() / objective_weights.sum()
        advantage = (advantage - mean) / (variance.sqrt() + 1e-8)
    chosen = log_probs[ppo_mask].gather(1, actions[:, None]).squeeze(-1)
    ratio = (chosen - behavior.log()).exp()
    surrogate = torch.minimum(ratio * advantage, ratio.clamp(1 - config.clip_epsilon, 1 + config.clip_epsilon) * advantage)
    policy = average('policy_loss', -surrogate, objective_weights)

    search_policy_mask = column('search_policy_target_mask', (batch,), mask=True) if 'search_policy_target_mask' in tensors else torch.zeros_like(active)
    search_value_mask = column('search_value_target_mask', (batch,), mask=True) if 'search_value_target_mask' in tensors else torch.zeros_like(active)
    if config.arm != 'D' and ((search_policy_mask | search_value_mask) & active).any():
        raise ValueError('D labels cannot contaminate control or B arms')
    sm = search_policy_mask[active]
    if sm.any():
        target = column('search_policy_targets', (batch, 9))[active][sm].detach()
        if (not torch.isfinite(target).all() or (target < 0).any()
                or not torch.allclose(target.sum(-1), torch.ones_like(target.sum(-1)), atol=1e-6, rtol=0)
                or (target.masked_select(~legal[active][sm]) != 0).any()):
            raise ValueError('completed legal teacher probability target required')
        teacher_values = -(target * safe_log_probs[sm]).sum(-1)
    else:
        teacher_values = output.value[:0]
    teacher = average('search_policy_loss', teacher_values, local_weights[sm])

    value_mask = column('ppo_value_target_mask', (batch,), mask=True) | search_value_mask
    vm = value_mask & active
    base_mask = vm & ~search_value_mask
    target = torch.zeros_like(output.value, dtype=torch.float32)
    target[base_mask] = column('ppo_value_targets', (batch,))[base_mask].detach().float()
    if (search_value_mask & active).any():
        if (search_value_mask & active & ~column('outcome_target_mask', (batch,), mask=True)).any():
            raise ValueError('D outcome/root value target needs a known literal terminal outcome')
        target[search_value_mask & active] = column('search_value_targets', (batch,))[search_value_mask & active].detach().float()
    selected = target[vm]
    if not torch.isfinite(selected).all() or (selected.abs() > 1).any():
        raise ValueError('finite unclipped value targets in[-1,1] required')
    if model_config.treatment:
        if output.value_logits is None or tuple(output.value_logits.shape) != (batch, model_config.value_bins):
            raise ValueError('B/D require their original HL-Gauss logits')
        labels = hl_gauss_targets(selected, bins=model_config.value_bins, sigma_ratio=model_config.value_sigma_ratio)
        value_terms = -(labels * output.value_logits[vm].float().log_softmax(-1)).sum(-1)
    else:
        value_terms = (output.value[vm] - selected).square()
        if config.value_clip is not None:
            old_mask = column('value_estimate_mask', (batch,), mask=True)
            if not old_mask[vm].all():
                raise ValueError('A value clipping requires original old values')
            old = column('value_estimates', (batch,))[vm].detach()
            clipped = old + (output.value[vm] - old).clamp(-config.value_clip, config.value_clip)
            value_terms = torch.maximum(value_terms, (clipped - selected).square())
    value = average('value_loss', value_terms, weights[vm])
    auxiliary = zero
    if model_config.treatment:
        heads = ('opponent_semantic_logits', 'hidden_species_logits', 'hidden_variant_logits', 'survival_logits', 'final_hp')
        if any(getattr(output, name) is None for name in heads):
            raise ValueError('training B/D requires explicitly enabled auxiliary heads')
        for name, shape in (('opponent_semantic_logits', (batch, len(model_config.label_vocabulary.opponent_classes))),
                ('hidden_species_logits', (batch, 6, len(model_config.label_vocabulary.species))),
                ('hidden_variant_logits', (batch, 6, MAX_VARIANTS)), ('survival_logits', (batch, 12)), ('final_hp', (batch, 12))):
            if tuple(getattr(output, name).shape) != shape:
                raise ValueError('wrong v5 auxiliary output shape: ' + name)
        om = column('opponent_semantic_action_target_mask', (batch,), mask=True) & active
        ot = column('opponent_semantic_action_targets', (batch,))[om].detach()
        if ot.dtype != torch.long or (ot < 0).any() or (ot >= len(model_config.label_vocabulary.opponent_classes)).any():
            raise ValueError('bound semantic opponent-action class required')
        auxiliary = auxiliary + average('opponent_action_loss', functional.cross_entropy(
            output.opponent_semantic_logits[om].float(), ot, reduction='none'), weights[om])
        hm = column('hidden_set_target_mask', (batch, 6), mask=True) & active[:, None]
        species = column('hidden_set_species', (batch, 6))[hm].detach()
        if species.dtype != torch.long or (species < 0).any() or (species >= len(model_config.label_vocabulary.species)).any():
            raise ValueError('bound hidden-set species required')
        hw = weights[:, None].expand(batch, 6)[hm]
        auxiliary = auxiliary + average('hidden_species_loss', functional.cross_entropy(
            output.hidden_species_logits[hm].float(), species, reduction='none'), hw)
        hidden_target = column('hidden_set_targets', (batch, 6, MAX_VARIANTS))[hm].detach()
        counts = torch.tensor([count for _, count in model_config.label_vocabulary.species_variants], device=species.device)[species]
        valid = torch.arange(MAX_VARIANTS, device=species.device)[None] < counts[:, None]
        if (not torch.isfinite(hidden_target).all() or (hidden_target < 0).any()
                or (hidden_target.masked_select(~valid) != 0).any()
                or not torch.allclose(hidden_target.sum(-1), torch.ones_like(hidden_target.sum(-1)), atol=1e-6, rtol=0)):
            raise ValueError('full source-axis hidden-set probability target required')
        hlog = output.hidden_variant_logits[hm].float().masked_fill(~valid, -torch.inf).log_softmax(-1).masked_fill(~valid, 0)
        auxiliary = auxiliary + average('hidden_variant_loss', -(hidden_target * hlog).sum(-1), hw)
        pm = column('pokemon_end_target_mask', (batch, 12), mask=True) & active[:, None]
        if pm.any() or (search_value_mask & active).any() or 'terminal_status' in tensors:
            if 'terminal_status' not in tensors:
                raise ValueError('actual terminal status required for terminal labels')
            if len(tensors['terminal_status']) != batch:
                raise ValueError('one literal terminal status per row required')
            complete = torch.tensor([status == 'COMPLETE' for status in tensors['terminal_status']], device=active.device)
            if (pm & ~complete[:, None]).any() or (search_value_mask & active & ~complete).any():
                raise ValueError('terminal labels require actual complete terminal state')
        pw = weights[:, None].expand(batch, 12)[pm]
        survived = column('survival_targets', (batch, 12))[pm].detach()
        hp = column('final_hp_targets', (batch, 12))[pm].detach()
        if (not torch.isfinite(survived).all() or not ((survived == 0) | (survived == 1)).all()
                or not torch.isfinite(hp).all() or (hp < 0).any() or (hp > 1).any()
                or not ((hp > 0) == survived.bool()).all()):
            raise ValueError('literal consistent terminal survival/final HP targets required')
        auxiliary = auxiliary + average('survival_loss', functional.binary_cross_entropy_with_logits(
            output.survival_logits[pm].float(), survived, reduction='none'), pw)
        auxiliary = auxiliary + average('final_hp_loss', (output.final_hp[pm].float() - hp).square(), pw)
    total = (policy + config.value_weight * value - config.entropy_coefficient * entropy
        + config.uniform_kl_coefficient * uniform_kl + (config.auxiliary_weight * auxiliary if model_config.treatment else zero)
        + (config.search_policy_weight * teacher if config.arm == 'D' else zero))
    metrics['loss'] = float(total.detach())
    return total, metrics
