"""Champion 9375 as the search pricers' leaf and root evaluator, through their forward= seam.

The depth-N and depth-2 pricers (device_depthn_pricer.DepthNPricer, device_depth2_pricer.Depth2Pricer), D3's FramePricer,
the exactness oracles (depth2_oracle.Oracle, depthn_oracle.DepthNOracle) and the tools' root forwards all evaluate through
one seam, forward(params, inputs). Its default is the V2 head (device_depth2_pricer.default_forward: v5_model.forward(
params, inputs, model, compute_dtype='bfloat16')). A ChampionEvaluator is a drop-in forward with the V2 forward's contract:

- inputs: the executor's V2 public29 encoder inputs, as the pricers hand them over: numeric_features [N,1,29,132] f32,
  categorical_ids [N,1,29,41] i32, token_type_ids [N,1,29] i32, attention_mask [N,1,29] bool, history_mask [N,1] bool.
  The executor stays in V2 layout (its root-input check, pre code 10, compares V2 keys); the champion's own layout is
  taken inside the evaluator (`view`).
- params: ONE actor's parameter tree (champion.load's dict). The pricers stack a multi-actor bank and vmap or index it
  (grouped_forward, actor_params); a champion bank is (champion,) with learner and opponent actors 0/0, as the ladder.
- returns v5_model.Output(policy_logits [N,9] f32, value [N] f32, value_logits None, valid [N] bool): the 9 policy
  logits over the same v4 request-order lanes (action tokens 13..21) and the champion's tanh value. A row that is not
  valid has NaN logits and value, as the V2 head's invalid rows; the pricers' usable rule (valid, finite, |v| <= 1)
  then refuses it with D's codes. The champion's opponent_action_head is not evaluated by the pricers (unused).
- valid (VALID_RULE): v5_model.forward's input-validity rule applied to what the champion reads. V2: every numeric
  finite, every category in [0, vocabulary), the token roles exactly TOKEN_ROLES, history_mask[:, 0] (the one frame is
  real) and the six belief tokens attended. Champion: the same over its 23 tokens (finite numerics, categories in
  [0, config.vocabulary), roles exactly TOKEN_ROLES[:23], history_mask[:, 0]) and at least one of its 23 tokens
  attended (its attention softmax needs one; V2's always-attended belief tokens guarantee it there). The champion's own
  forward clamps out-of-range categories and roles (champion.forward); this rule refuses them instead, as V2 does.
- value convention: tanh(linear(mean pool)) in (-1, 1) from the OBSERVER's side, the player whose view is encoded: the
  champion is a PPO critic whose target is +1 iff that player won, -1 if it lost, 0 on a draw
  (pokezero.dataset._terminal_value_for_player). The pricers observe the LEARNER's view at every valued node and leaf and
  read the value as the learner's, beside terminals that are +1 / -1 / 0 from the learner's (owner's) side; the V2 head
  is observer-relative in the same way, and Codex's ladder mixes champion values with owner-side terminals identically
  (phase_d_search: value = terminal from the owner, else the champion's value of the owner's view). So the value passes
  through UNCHANGED: no sign flip, no perspective swap, no rescale. tests/test_gpu_engine_champion_evaluator.py pins
  the adapter's value, sign included, to champion.forward on the independently sliced inputs.

The slice: the champion observes pokezero.observation.v4, 23 tokens with no history axis; V2's 29 tokens are that v4
observation with six belief tokens APPENDED (v5_contract.append_belief_tokens keeps v4's rows, roles and order, then
adds tokens 23..28), and the device encoder's own champion layout is the [:, :23] prefix of the same tensors
(engine_device_observation.champion_inputs). `view` takes [..., :23(, F)] of every token tensor (the 29-token axis is
required, so a champion-layout executor fails closed) and keeps history_mask; the forward reads view[k][:, 0].

Digests: the exactness gate hashes each leaf's evaluator inputs (depth2_digest). Through `digest_view(forward)` the
pricers and oracles hash `view(inputs)` for a champion run (the five V2 key names, champion slice), so a digest certifies
exactly what the champion read; the V2 default hashes the inputs unchanged (byte-identical digests).

Identity: ('v2',) for the default, ('champion', archive sha256, precision) for a champion. `program_suffix(forward)` is
what every compiled-program key gains: nothing for V2 (keys byte-identical to before), (('evaluator',) + identity,)
otherwise, so two evaluators in one process (or one shared program table) never share a compiled program.

Loading: champion.load checks schema, keys, shapes, fp32 and the parameter count but takes no expected sha; `load`
adds the caller checks of phase_d_search.Search (lines 89-93): the archive bytes' sha256 and the export's recorded
checkpoint sha256 against the caller's, and 23 tokens. `check_binding` adds the rest of that binding for a bank run:
the real 9375 checkpoint (ActorContract.champion()) and the bank's encoder catalog compiled for it (its checkpoint
sha256, category vocabulary sha256 and size, and feature masks: the JAX encoder contract of live_protocol_actor).

Precision is explicit: bf16 (the ladder's default; bf16 dot operands, fp32 accumulation and everything else) or f32
(HIGHEST). The bf16 value has not been gated against f32 on real device leaves (gap 2).

Module import is numpy-only (no JAX before a bank's memory policy): JAX and champion load lazily.
"""
from dataclasses import asdict, dataclass, field
import re

import numpy as np

V2 = ('v2',)
TOKENS = 23                      # the champion's v4 prefix of the V2 encoding
V2_TOKENS = 29
INPUT_KEYS = ('numeric_features', 'categorical_ids', 'token_type_ids', 'attention_mask', 'history_mask')   # = depth2_digest.INPUT_KEYS
PRECISIONS = {'bf16': 'bfloat16', 'f32': 'float32'}
VALID_RULE = ('v5_model.forward\'s input-validity rule over what the champion reads: every numeric of its 23 tokens finite, every category '
    'in [0, vocabulary), token roles exactly TOKEN_ROLES[:23], history_mask[:, 0], and at least one of its 23 tokens attended; an invalid '
    'row has NaN logits and value (as V2)')
VALUE_RULE = ('tanh(linear(mean pool)) in (-1, 1) from the observer\'s side (the learner\'s view at every valued node and leaf), passed '
    'through unchanged beside +1/-1/0 terminals from the learner\'s side, as the V2 head')
SLICE_RULE = ('the first 23 tokens of each V2 token tensor (v4 prefix: v5_contract.append_belief_tokens, engine_device_observation.'
    'champion_inputs) plus history_mask; the executor stays in V2 layout')


# Seam helpers: any forward= argument ------------------------------------------------------------------------------------------------
def identity(forward):
    """The evaluator identity of a pricer's forward= argument: ('v2',) for the default (None); an evaluator's own `identity`
    (a ChampionEvaluator's ('champion', archive sha256, precision)); for any other callable ('forward', that callable), keyed
    by the object itself, so a custom forward never shares a program with another forward."""
    if forward is None:return V2
    own = getattr(forward, 'identity', None)
    return tuple(own) if own is not None else ('forward', forward)


def program_suffix(forward):
    """What every compiled-program key of a pricer built with this forward= gains: () for V2, so the default's keys are
    byte-identical to the pre-evaluator keys; otherwise (('evaluator',) + identity,)."""
    key = identity(forward)
    return () if key == V2 else (('evaluator',) + key,)


def digest_view(forward):
    """The function a digest hashes encoder inputs through for this forward= argument: the evaluator's `view` (what it reads),
    or None (the inputs unchanged: the V2 default and plain callables)."""
    return None if forward is None else getattr(forward, 'view', None)


def describe(forward):
    """dict(kind, identity) of a forward= argument, for logs and reports."""
    key = identity(forward)
    return dict(kind=key[0] if key[0] != 'forward' else 'custom', identity=[x if isinstance(x, str) else repr(x) for x in key])


# The champion evaluator -------------------------------------------------------------------------------------------------------------
@dataclass(frozen=True, eq=False)
class ChampionEvaluator:
    """forward(params, inputs) for champion 9375 under the V2 forward's contract (module docstring). `params`, `config` and
    `meta` are champion.load's; the pricers pass the params back through price() as the bank (champion,)."""
    params: object = field(repr=False)
    config: object
    meta: dict = field(repr=False)
    precision: str = 'bfloat16'

    kind = 'champion'

    def __post_init__(self):
        if self.precision not in PRECISIONS.values():raise ValueError(f'champion precision in {tuple(PRECISIONS.values())} required')
        if self.config.tokens != TOKENS:raise ValueError(f'the champion reads the {TOKENS}-token v4 prefix (config.tokens {self.config.tokens})')
        for name in ('weights_archive_sha256', 'checkpoint_sha256'):
            if not _hex64(self.meta.get(name)):raise ValueError(f'champion metadata {name} must be a 64-hex sha256')

    @property
    def archive_sha256(self):return self.meta['weights_archive_sha256']

    @property
    def checkpoint_sha256(self):return self.meta['checkpoint_sha256']

    @property
    def identity(self):return ('champion', self.archive_sha256, self.precision)

    @property
    def input_keys(self):
        """The digest's key set: the five V2 key names (token tensors sliced to the champion's 23)."""
        return INPUT_KEYS

    def view(self, inputs):
        """What the champion reads of V2 encoder inputs with any leading axes: each token tensor's first 23 tokens and the history
        mask, under the V2 key names ([..., 23, F] / [..., 23] / history unchanged). The 29-token axis is required."""
        if set(inputs) != set(INPUT_KEYS):raise ValueError(f'the five V2 encoder inputs {INPUT_KEYS} required, got {sorted(inputs)}')
        for key, axis in (('numeric_features', -2), ('categorical_ids', -2), ('token_type_ids', -1), ('attention_mask', -1)):
            if inputs[key].shape[axis] != V2_TOKENS:raise ValueError(f'{key}: the V2 {V2_TOKENS}-token layout required (got {inputs[key].shape})')
        t = TOKENS
        return dict(numeric_features=inputs['numeric_features'][..., :t, :], categorical_ids=inputs['categorical_ids'][..., :t, :],
            token_type_ids=inputs['token_type_ids'][..., :t], attention_mask=inputs['attention_mask'][..., :t], history_mask=inputs['history_mask'])

    def valid(self, view):
        """VALID_RULE on a view [N,1,23,...] -> bool [N]."""
        import jax.numpy as j
        from pokezero.v5_model import TOKEN_ROLES
        c = self.config;numeric, cats, types, attention, history = (view[k] for k in INPUT_KEYS)
        roles = j.asarray(TOKEN_ROLES[:TOKENS], types.dtype)
        return (j.all(j.isfinite(numeric), axis=(1, 2, 3)) & j.all((cats >= 0) & (cats < c.vocabulary), axis=(1, 2, 3))
            & j.all(types == roles, axis=(1, 2)) & history[:, 0] & j.any(attention, axis=(1, 2)))

    def core(self, params, numeric, categorical, token_types, attention):
        """champion.forward on champion-layout rows [N,23,...] at this evaluator's precision (the reference the adapter wraps)."""
        from . import champion
        return champion.forward(params, numeric, categorical, token_types, attention, self.config, compute_dtype=self.precision)

    def __call__(self, params, inputs):
        import jax.numpy as j
        from . import v5_model
        v = self.view(inputs);c = self.config;b = v['numeric_features'].shape[0]
        expected = dict(numeric_features=(b, 1, TOKENS, c.numeric), categorical_ids=(b, 1, TOKENS, c.categorical), token_type_ids=(b, 1, TOKENS),
            attention_mask=(b, 1, TOKENS), history_mask=(b, 1))
        if any(tuple(v[k].shape) != s for k, s in expected.items()):
            raise ValueError(f'champion inputs [N,1,29,...] with {c.numeric} numeric and {c.categorical} categorical features required, got '
                + str({k: tuple(x.shape) for k, x in v.items()}))
        valid = self.valid(v)
        out = self.core(params, v['numeric_features'][:, 0], v['categorical_ids'][:, 0], v['token_type_ids'][:, 0], v['attention_mask'][:, 0])
        return v5_model.Output(j.where(valid[:, None], out.policy_logits.astype(j.float32), j.nan), j.where(valid, out.value.astype(j.float32), j.nan),
            None, valid)

    def reference(self, params, inputs):
        """champion.forward on the champion layout sliced INDEPENDENTLY of `view`, on the host ([:, 0, :23] of numpy copies, as
        engine_device_observation.champion_inputs slices [:, :23]), op by op: the other side of `check`."""
        import jax.numpy as j
        host = {k: np.asarray(x) for k, x in inputs.items()}
        return self.core(params, *[j.asarray(host[k][:, 0, :TOKENS]) for k in ('numeric_features', 'categorical_ids', 'token_type_ids', 'attention_mask')])

    def check(self, params, inputs):
        """The champion's arm check on one root (gate.bind_roots, the first request): the adapter on the V2 root inputs must be valid
        and give EXACTLY the policy logits and value of champion.forward on the independently sliced champion layout (sign
        included). Both sides run op by op (not jitted): the same champion ops on the same values give the same bits on any
        backend, whereas two separately jitted programs may fuse differently and differ in the last bit (in bf16, a flipped
        operand rounding then moves a value by far more). It certifies the slice and the wiring; the pricers' own programs are
        checked against the oracle by the gates. Raises ValueError otherwise."""
        import jax
        import jax.numpy as j
        output = jax.device_get(self(params, {k: j.asarray(np.asarray(v)) for k, v in inputs.items()}))
        if not (bool(np.asarray(output.valid).reshape(-1)[0]) and np.isfinite(np.asarray(output.value)).all()
                and np.isfinite(np.asarray(output.policy_logits)).all()):
            raise ValueError('champion check: the adapter\'s output on the first root is invalid or non-finite (check the root inputs)')
        direct = jax.device_get(self.reference(params, inputs))
        for name in ('policy_logits', 'value'):
            a, b = np.asarray(getattr(output, name), np.float32), np.asarray(getattr(direct, name), np.float32).reshape(np.shape(getattr(output, name)))
            if not np.array_equal(a, b):
                raise ValueError(f'champion check: the adapter\'s {name} differs from champion.forward on the sliced inputs (max |diff| '
                    f'{float(np.max(np.abs(a - b)))})')
        return True

    def header(self):
        """The report's evaluator fields for this champion."""
        return dict(kind=self.kind, identity=list(self.identity), archive_sha256=self.archive_sha256, checkpoint_sha256=self.checkpoint_sha256,
            precision=self.precision, parameter_count=self.meta.get('parameter_count'), config=asdict(self.config), schema=self.meta.get('schema'),
            tokens=TOKENS, slice=SLICE_RULE, valid=VALID_RULE, value=VALUE_RULE, digest='depth2_digest over view(inputs) (the champion slice)')


def _hex64(value):
    return isinstance(value, str) and re.fullmatch('[0-9a-f]{64}', value) is not None


def load(path, *, expected_archive_sha256, expected_checkpoint_sha256, precision='bfloat16'):
    """champion.load(path) plus the caller checks phase_d_search.Search applies (lines 89-93) that champion.load leaves to its
    caller: the sha256 of the archive bytes actually deserialized (champion.snapshot) and the export's recorded checkpoint
    sha256 must equal the expected ones, and the config must read 23 tokens. Returns a ChampionEvaluator; ValueError otherwise."""
    for name, value in (('archive', expected_archive_sha256), ('checkpoint', expected_checkpoint_sha256)):
        if not _hex64(value):raise ValueError(f'explicit 64-hex champion {name} sha256 required')
    if precision not in PRECISIONS.values():raise ValueError(f'champion precision in {tuple(PRECISIONS.values())} required')
    from . import champion
    params, config, meta = champion.load(path)
    if meta['weights_archive_sha256'] != expected_archive_sha256:
        raise ValueError(f'champion archive sha256 {meta["weights_archive_sha256"]} differs from the expected {expected_archive_sha256}')
    if meta.get('checkpoint_sha256') != expected_checkpoint_sha256:
        raise ValueError(f'champion archive\'s checkpoint sha256 {meta.get("checkpoint_sha256")} differs from the expected {expected_checkpoint_sha256}')
    if config.tokens != TOKENS:raise ValueError(f'champion archive config reads {config.tokens} tokens, not the {TOKENS}-token v4 prefix')
    return ChampionEvaluator(params, config, meta, precision)


def check_binding(evaluator, catalog):
    """The rest of phase_d_search.Search's binding on a bank run: the archive's checkpoint is the real champion 9375
    (ActorContract.champion()), and the bank's encoder catalog was compiled for it: its checkpoint sha256, its category
    vocabulary (the contract's sha256 and the archive's size) and its feature masks are the champion contract's, the JAX
    encoder contract live_protocol_actor checks for B2. ValueError otherwise."""
    from pokezero.live_protocol_actor import ActorContract, MASK_NAMES, _vocab_sha
    contract = ActorContract.champion()
    if evaluator.checkpoint_sha256 != contract.checkpoint_sha256:
        raise ValueError(f'the real original 9375 champion checkpoint {contract.checkpoint_sha256} required, got {evaluator.checkpoint_sha256}')
    if catalog.checkpoint_sha256 != evaluator.checkpoint_sha256:
        raise ValueError(f'the bank encoder catalog was compiled for checkpoint {catalog.checkpoint_sha256}, not the champion\'s')
    if _vocab_sha(catalog.vocabulary) != contract.category_vocab_sha256 or catalog.vocabulary.size != evaluator.config.vocabulary:
        raise ValueError('the bank encoder catalog\'s category vocabulary is not the champion\'s (sha256 or size)')
    if tuple((k, getattr(catalog.masks, k)) for k in MASK_NAMES) != contract.feature_masks:
        raise ValueError('the bank encoder catalog\'s feature masks are not the champion contract\'s')
    return True


# Command line (numpy only: parse before a bank's memory policy) ---------------------------------------------------------------------
FLAGS = ('champion_archive', 'champion_archive_sha256', 'champion_checkpoint_sha256')


def add_arguments(parser):
    parser.add_argument('--champion-archive', help='price with champion 9375 (a gpu_engine.champion .npz export) as leaf and root evaluator, '
        'instead of the bank snapshot\'s V2 actor (exclusive with --head-archive)')
    parser.add_argument('--champion-archive-sha256', help='sha256 of --champion-archive\'s bytes (required with it)')
    parser.add_argument('--champion-checkpoint-sha256', help='the export\'s checkpoint sha256 (required with --champion-archive; must be 9375\'s)')
    parser.add_argument('--champion-precision', choices=tuple(PRECISIONS), help='champion forward precision (default bf16, as the ladder)')


def check_arguments(parser, args):
    """All three champion flags or none; --champion-precision only with them (default bf16); exclusive with --head-archive and
    its options. Sets args.champion_precision."""
    given = [getattr(args, name, None) for name in FLAGS]
    if any(given) and not all(given):parser.error('--champion-archive, --champion-archive-sha256 and --champion-checkpoint-sha256 go together')
    if not any(given):
        if getattr(args, 'champion_precision', None) is not None:parser.error('--champion-precision needs --champion-archive')
        return args
    head = [name for name in ('head_archive', 'head_archive_sha256', 'head_checkpoint_sha256', 'head_category_sha256', 'head_model_sha256')
        if getattr(args, name, None)]
    if head:parser.error('--champion-archive and --head-archive are mutually exclusive (given: ' + ', '.join('--' + h.replace('_', '-') for h in head) + ')')
    if getattr(args, 'raw_head_check', False):parser.error('--raw-head-check needs --head-archive (the champion has no V2 arm)')
    for name in FLAGS[1:]:
        if not _hex64(getattr(args, name)):parser.error(f'--{name.replace("_", "-")} must be a 64-hex sha256')
    if args.champion_precision is None:args.champion_precision = 'bf16'
    return args


def load_from_arguments(args):
    """The evaluator the parsed flags name (load's checks), or None without --champion-archive."""
    if not getattr(args, 'champion_archive', None):return None
    return load(args.champion_archive, expected_archive_sha256=args.champion_archive_sha256,
        expected_checkpoint_sha256=args.champion_checkpoint_sha256, precision=PRECISIONS[args.champion_precision or 'bf16'])


def log_fields(args):
    """The champion flags for a tool's EVALUATOR start line ({} without --champion-archive)."""
    if not getattr(args, 'champion_archive', None):return {}
    return dict(champion_archive=args.champion_archive, champion_archive_sha256=args.champion_archive_sha256,
        champion_checkpoint_sha256=args.champion_checkpoint_sha256, champion_precision=args.champion_precision)
