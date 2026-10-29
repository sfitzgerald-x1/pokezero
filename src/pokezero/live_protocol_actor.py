"""Isolated owning-stream actor; no bridge, game launcher or qualification.

Only literal lines from ONE getPlayerStreams owner enter this journal. Python
and JAX backends consume the same immutable ordered boundary. Provider factories
are trusted source dependencies, not runtime/model attestations. Even a complete
actor game leaves player-wire, full-encoder and B3 qualification false.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import threading
from types import MappingProxyType, MethodType
from typing import Any, Callable
import weakref

import numpy as np

from .observation import ObservationPerspective, PokeZeroObservationV0
from .policy import PolicyDecision


SHAPES = dict(categorical_ids=(23, 41), numeric_features=(23, 132),
              token_type_ids=(23,), attention_mask=(23,), legal_action_mask=(9,))
MASK_NAMES = ('exact_state', 'feature_pack_last_move', 'investment_belief_narrowing',
              'item_belief_narrowing', 'stats_block', 'tier2_investment',
              'tier2_residuals', 'transition_token_budget')
UNQUALIFIED = MappingProxyType(dict(public_wire_qualified=False,
    captured_player_wire=False, full_encoder_admitted=False, B3_qualified=False))


# Lease only mutable policy HISTORY, never shared model weights/dex tables.
# Strong container references prevent id reuse; weak owners release the ledger
# when a wrapper dies, rather than retaining every completed game's history.
_HISTORY_LEASES = {}
_HISTORY_LOCK = threading.RLock()


def _release_history(owner):
    with _HISTORY_LOCK:
        for identity, (_, lease) in tuple(_HISTORY_LEASES.items()):
            if lease is owner:
                del _HISTORY_LEASES[identity]


def _mutable_history(root):
    pending, seen = [root], set()
    while pending:
        value = pending.pop()
        if id(value) in seen:
            continue
        seen.add(id(value))
        if type(value) is dict or isinstance(value, MappingProxyType):
            if type(value) is dict:
                yield value
            pending.extend(value.values())
        elif type(value) in (list, set):
            yield value
            pending.extend(value)
        elif type(value) is tuple:
            pending.extend(value)
        elif isinstance(value, PokeZeroObservationV0):
            pending.extend(getattr(value, name) for name in value.__dataclass_fields__)
        elif isinstance(value, np.ndarray):
            yield value
            if value.base is not None:
                pending.append(value.base)
        elif isinstance(value, memoryview):
            yield value
            pending.append(value.obj)
        elif type(value) is bytearray:
            yield value


def _terminal(line):
    return line.startswith('|win|') or line == '|tie' or line.startswith('|tie|')


def _same_callable(actual, expected):
    # Bound method lookup creates a new object. Compare its actual receiver and
    # function identities, never an arbitrary callable's overloaded equality.
    return actual is expected or (type(actual) is MethodType and type(expected) is MethodType
        and actual.__self__ is expected.__self__ and actual.__func__ is expected.__func__)


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


def _sha(value):
    return hashlib.sha256(value).hexdigest()


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _hex(value, size=64):
    return type(value) is str and len(value) == size and all(c in '0123456789abcdef' for c in value)


def _packet(line, owner):
    def pairs(items):
        result = {}
        for key, value in items:
            _require(key not in result, 'duplicate request key')
            result[key] = value
        return result
    raw = json.loads(line[len('|request|'):], object_pairs_hook=pairs,
                     parse_constant=lambda _: (_ for _ in ()).throw(ValueError('nonfinite request')))
    _require(type(raw) is dict and type(raw.get('side')) is dict
             and raw['side'].get('id') == owner, 'owning request identity required')
    _require('wait' not in raw or type(raw['wait']) is bool, 'boolean wait required')
    _require('rqid' not in raw or (type(raw['rqid']) is int and 0 <= raw['rqid'] < 2**31),
             'canonical request ID required')
    _require('forceSwitch' not in raw or (type(raw['forceSwitch']) is list
             and len(raw['forceSwitch']) <= 1 and all(type(v) is bool for v in raw['forceSwitch'])),
             'singles boolean force-switch required')
    _canonical(raw)  # Refuse overflow-to-infinity too; retain the original line.
    return raw


def _actionable(packet):
    from .local_showdown import requested_players_from_requests
    return bool(requested_players_from_requests({packet['side']['id']: packet}))


@dataclass(frozen=True)
class ActorContract:
    """Exact declared namespace; champion() is the production9375 contract.

    Explicit alternate contracts support authored source fixtures. They do not
    qualify a different model, source, runtime or live player stream.
    """
    checkpoint_sha256: str
    category_vocab_sha256: str
    belief_set_source_hash: str
    model_config_sha256: str
    feature_masks: tuple[tuple[str, Any], ...]
    showdown_revision: str
    categorical_vocab_size: int = 916
    token_type_vocab_size: int = 16

    def __post_init__(self):
        _require(all(_hex(v) for v in (self.checkpoint_sha256,
            self.category_vocab_sha256, self.model_config_sha256)), 'exact SHA256 bindings required')
        # The production factory pins f5a5265143d423af. Source unit catalogs
        # also use explicit version labels; equality is exact, never normalized.
        _require(type(self.belief_set_source_hash) is str and 0 < len(self.belief_set_source_hash) <= 128
            and self.belief_set_source_hash.isascii()
            and all(c.isalnum() or c in '-_:' for c in self.belief_set_source_hash)
            and _hex(self.showdown_revision, 40),
                 'source and Showdown bindings required')
        _require(type(self.categorical_vocab_size) is int and self.categorical_vocab_size > 0
            and type(self.token_type_vocab_size) is int and self.token_type_vocab_size > 0,
            'positive exact vocabulary sizes required')
        _require(type(self.feature_masks) is tuple and tuple(k for k, _ in self.feature_masks) == MASK_NAMES,
                 'ordered complete immutable mask contract required')
        for key, value in self.feature_masks:
            _require(type(value) is (int if key == 'transition_token_budget' else bool), 'mask type')
        masks = dict(self.feature_masks)
        _require(masks['transition_token_budget'] == 0 and not masks['investment_belief_narrowing'],
                 'installed one-frame V4 masks required')

    @classmethod
    def champion(cls):
        from .recorded_mirror_roots import CHAMPION_CONTRACT as c
        return cls(c['checkpoint_sha256'], c['category_vocab_sha256'], c['belief_set_source_hash'],
            c['model_config_sha256'], tuple((k, c['feature_masks'][k]) for k in MASK_NAMES),
            'f76228a1354b5d0f307ca2d16101294ad3a2308b')

    @property
    def namespace(self):
        return _sha(_canonical(self.__dict__))


def _vocab_sha(vocab):
    return _sha(_canonical(tuple(vocab.tokens)))


def _mask_values(masks):
    return tuple((k, getattr(masks, 'opponent_tendency_stats_block' if k == 'stats_block' else k))
                 for k in MASK_NAMES)


class OwnedProvider:
    """A backend or policy instance belongs to at most one game, ever."""
    def __init__(self, contract):
        _require(type(contract) is ActorContract, 'actor contract required')
        self.contract = contract
        self._claimed = False

    def claim(self, contract):
        _require(self.contract == contract and not self._claimed, 'provider namespace or ownership alias')
        self._claimed = True

    def close(self):
        pass


class RawPolicy(OwnedProvider):
    """Common raw selection callback, also usable with authored fake policies."""
    def __init__(self, contract, select_action: Callable, reset: Callable):
        super().__init__(contract)
        _require(callable(select_action) and callable(reset), 'bound raw callbacks required')
        self._select_action, self.reset = select_action, reset
        self._transformer_policy = None
        self._history = None
        self._history_owner = weakref.ref(self, _release_history)
        self._validate = lambda: None

    def select_action(self, observation, *, rng):
        self._validate()
        decision = self._select_action(observation, rng=rng)
        self._validate()
        return decision

    def claim(self, contract):
        super().claim(contract)
        if self._transformer_policy is not None:
            _require(not getattr(self._transformer_policy, '_live_protocol_claimed', False),
                     'raw policy ownership alias')
            self._validate()
            self._history = self._transformer_policy._history_by_player
            _require(type(self._history) is dict, 'owned mutable history dictionary required')
            self._claim_history()
            self._transformer_policy._live_protocol_claimed = True

    def _claim_history(self):
        containers = tuple(_mutable_history(self._history))
        with _HISTORY_LOCK:
            # Validate the whole roster first, so refusal takes no partial lease.
            for value in containers:
                previous = _HISTORY_LEASES.get(id(value))
                _require(previous is None or previous[1]() is self,
                         'mutable raw history ownership alias')
            for value in containers:
                _HISTORY_LEASES[id(value)] = (value, self._history_owner)

    @classmethod
    def from_transformer(cls, contract, policy):
        """Bind an already-loaded model; never reads a checkpoint or changes precision."""
        from .neural_policy import TransformerSoftmaxPolicy, require_torch
        _require(type(policy) is TransformerSoftmaxPolicy, 'ordinary raw transformer policy required')
        config = policy.result.model_config
        _require(policy.weights_sha256 == contract.checkpoint_sha256
            and _sha(_canonical(config.to_dict())) == contract.model_config_sha256
            and _sha(_canonical(config.category_vocab)) == contract.category_vocab_sha256,
            'raw checkpoint/config/vocabulary contract mismatch')
        mask_values = dict(exact_state=config.exact_state_enabled,
            feature_pack_last_move=config.feature_pack_last_move,
            investment_belief_narrowing=config.investment_belief_narrowing,
            item_belief_narrowing=config.item_belief_narrowing, stats_block=config.stats_block_enabled,
            tier2_investment=config.tier2_investment, tier2_residuals=config.tier2_residuals,
            transition_token_budget=config.transition_token_budget)
        _require(tuple((k, mask_values[k]) for k in MASK_NAMES) == contract.feature_masks,
                 'raw feature masks differ')
        _require(config.categorical_vocab_size == contract.categorical_vocab_size
            and config.token_type_vocab_size == contract.token_type_vocab_size
            and config.observation_schema_version == 'pokezero.observation.v4'
            and config.window_size == 1 and config.temporal_aggregator == 'mean'
            and (config.token_count, config.numeric_feature_count, config.categorical_feature_count) == (23,132,41),
            'one-frame V4 model required')
        _require(policy.deterministic is True and policy.exploration_epsilon == 0
            and policy.sampling_temperature == 1 and policy.family_gated_selection is False
            and policy.history_mask_k is None and not policy.model.training
            and _same_callable(policy.forward_fn, policy._default_forward),
            'unchanged deterministic raw selection required')
        torch = require_torch()
        _require(not torch.backends.mha.get_fastpath_enabled()
            and torch.get_float32_matmul_precision() == 'highest'
            and not torch.backends.cuda.matmul.allow_tf32 and not torch.backends.cudnn.allow_tf32,
            'matched MHAOFF/TF32OFF/highest required')
        tensors = list(policy.model.parameters()) + list(policy.model.buffers())
        _require(all(t.device.type == 'cpu' and (not t.is_floating_point() or t.dtype == torch.float32)
                     for t in tensors), 'CPU FP32 model required')
        bound = cls(contract, policy.select_action, policy.reset)
        bound._transformer_policy = policy
        result, model = policy.result, policy.model
        forward, default_forward = policy.forward_fn, policy._default_forward
        select, reset = policy.select_action, policy.reset
        fields = ('deterministic', 'exploration_epsilon', 'sampling_temperature',
                  'family_gated_selection', 'history_mask_k', 'device', 'policy_id',
                  'checkpoint_path', 'record_policy_distribution')
        selection = tuple((type(getattr(policy,k)),getattr(policy,k)) for k in fields)
        timing = policy.inference_timing
        def validate_runtime():
            _require(policy.result is result and policy.result.model_config is config
                and policy.model is model and policy.weights_sha256 == contract.checkpoint_sha256
                and _sha(_canonical(config.to_dict())) == contract.model_config_sha256
                and _sha(_canonical(config.category_vocab)) == contract.category_vocab_sha256,
                'raw model/config/vocabulary binding drift')
            _require(tuple((k, getattr(config, {'exact_state':'exact_state_enabled',
                'stats_block':'stats_block_enabled'}.get(k,k))) for k in MASK_NAMES) == contract.feature_masks,
                'raw feature mask binding drift')
            _require(tuple((type(getattr(policy,k)),getattr(policy,k)) for k in fields) == selection
                and _same_callable(policy.forward_fn, forward)
                and _same_callable(policy._default_forward, default_forward)
                and _same_callable(policy.select_action, select) and _same_callable(policy.reset, reset)
                and _same_callable(bound._select_action, select) and _same_callable(bound.reset, reset)
                and policy.inference_timing is timing, 'raw selection/function binding drift')
            _require(not policy.model.training
                and not torch.backends.mha.get_fastpath_enabled()
                and torch.get_float32_matmul_precision() == 'highest'
                and not torch.backends.cuda.matmul.allow_tf32 and not torch.backends.cudnn.allow_tf32,
                'raw runtime precision/binding drift')
            _require(all(t.device.type == 'cpu' and (not t.is_floating_point() or t.dtype == torch.float32)
                for t in list(policy.model.parameters())+list(policy.model.buffers())), 'raw model dtype/device drift')
            if bound._history is not None:
                _require(policy._history_by_player is bound._history, 'raw history binding drift')
                bound._claim_history()
        bound._validate = validate_runtime
        return bound


class PythonBackend(OwnedProvider):
    """Production incremental parser/encoder over one actual owning history."""
    def __init__(self, contract, *, config, battle_id, owner, stream_id):
        super().__init__(contract)
        from .local_showdown import LocalShowdownEnv
        from .showdown import _ReplayParser
        _require(owner in ('p1','p2') and config.category_vocab is not None
            and config.observation_spec.schema_version == 'pokezero.observation.v4'
            and _vocab_sha(config.category_vocab) == contract.category_vocab_sha256
            and config.category_vocab.size == contract.categorical_vocab_size
            and _mask_values(config.feature_masks) == contract.feature_masks
            and config.set_belief_source is True, 'Python encoder contract differs')
        self.owner, self.env = owner, LocalShowdownEnv(config)
        _require(self.env._process is None and self.env._battle_token is None
            and self.env._belief_set_source.metadata.source_hash == contract.belief_set_source_hash,
            'fresh process-free bound Python source required')
        self.env._battle_id = battle_id
        self.env._parser = _ReplayParser(battle_id, complete_prefix=True,
            hp_visibility={owner:'exact', 'p2' if owner=='p1' else 'p1':'percentage'})

    def ingest(self, lines):
        self._check_namespace()
        for line in lines:
            # This internal seam names public ingestion "omniscient". Only the
            # already-isolated literal OWNER line reaches it, never another stream.
            stream = self.owner if line.startswith('|request|') else 'omniscient'
            self.env._apply_event(dict(type='stream', stream=stream, lines=[line]))
            self.env._sync_incremental_state()
        _require(set(self.env._latest_requests) <= {self.owner}
            and set(self.env._parser.requests) <= {self.owner}, 'foreign Python request state')

    def encode(self, snapshot):
        packet = _packet(snapshot.own_request_line, self.owner)
        self._check_namespace()
        _require(self.env._latest_requests.get(self.owner) == packet, 'Python owning request drift')
        observation = self.env.observe(self.owner)
        targets = _request_targets(packet)
        return observation, targets

    def _check_namespace(self):
        _require(self.env._process is None and self.env._battle_token is None
            and self.env._belief_set_source.metadata.source_hash == self.contract.belief_set_source_hash
            and _vocab_sha(self.env.config.category_vocab) == self.contract.category_vocab_sha256
            and _mask_values(self.env.config.feature_masks) == self.contract.feature_masks
            and self.env._parser.hp_visibility == {self.owner:'exact',
                'p2' if self.owner=='p1' else 'p1':'percentage'}, 'Python namespace drift')

    def close(self):
        _require(self.env._process is None and self.env._battle_token is None, 'unexpected simulator ownership')


class JaxBackend(OwnedProvider):
    """Existing atomic public fold plus own-request encoding and target checks."""
    def __init__(self, contract, *, catalog, battle_id, owner, stream_id):
        super().__init__(contract)
        from .recorded_encoder_runner import JaxRecordedAdapter
        _require(catalog.checkpoint_sha256 == contract.checkpoint_sha256
            and catalog.source.source_hash == contract.belief_set_source_hash
            and _vocab_sha(catalog.vocabulary) == contract.category_vocab_sha256
            and catalog.vocabulary.size == contract.categorical_vocab_size
            and tuple((k,getattr(catalog.masks,k)) for k in MASK_NAMES) == contract.feature_masks,
            'JAX encoder contract differs')
        self.catalog, self.owner = catalog, owner
        self.adapter = JaxRecordedAdapter(catalog)
        _require(not self.adapter.jax.config.jax_enable_x64
            and self.adapter.jax.config.jax_default_matmul_precision == 'highest', 'JAX precision differs')
        self.adapter.reset(stream_id=stream_id, battle_id=battle_id)
        self._catalog_binding = np.asarray(catalog.binding).copy()

    def _check_namespace(self):
        _require(self.catalog.checkpoint_sha256 == self.contract.checkpoint_sha256
            and self.catalog.source.source_hash == self.contract.belief_set_source_hash
            and _vocab_sha(self.catalog.vocabulary) == self.contract.category_vocab_sha256
            and tuple((k,getattr(self.catalog.masks,k)) for k in MASK_NAMES) == self.contract.feature_masks
            and np.array_equal(np.asarray(self.catalog.binding),self._catalog_binding), 'JAX namespace drift')
        _require(not self.adapter.jax.config.jax_enable_x64
            and self.adapter.jax.config.jax_default_matmul_precision == 'highest', 'JAX precision drift')

    def ingest(self, lines):
        self._check_namespace()
        # Flush segments at EVERY request: never move a request across later
        # public evidence, even though JAX private features are boundary-local.
        segment = []
        for line in lines:
            if line.startswith('|request|'):
                if segment:
                    self.adapter.feed(tuple(segment)); segment = []
                self.adapter.encoder.parse_own_requests(self.catalog,
                    [_packet(line,self.owner)], owner_sides=[int(self.owner[1])-1])
            else:
                segment.append(line)
        if segment:
            self.adapter.feed(tuple(segment))

    def encode(self, snapshot):
        self._check_namespace()
        side = int(self.owner[1])-1
        packet = _packet(snapshot.own_request_line, self.owner)
        arrays = self.adapter.encode(packet, owner_side=side)
        own = self.adapter.encoder.parse_own_requests(self.catalog,[packet],owner_sides=[side])
        from gpu_engine.private_requests import action_inputs
        actions = action_inputs(own.actions.request, observer_side=side)
        _require(np.array_equal(arrays['legal_action_mask'], np.asarray(actions.legal)[0]),
                 'encoded/request legal mask differs')
        return arrays, tuple(int(x) for x in np.asarray(actions.switch_targets)[0])


def _request_targets(packet):
    # Identical source canonical switch ordering, derived ONLY from own party.
    from .actions import canonical_switch_action_map
    rows = packet['side'].get('pokemon')
    _require(type(rows) is list and 1 <= len(rows) <= 6
        and all(type(p) is dict for p in rows), 'own singles team required')
    active = [i for i,p in enumerate(rows) if p.get('active') is True]
    _require(len(active) == 1, 'exactly one own active required')
    targets = tuple(canonical_switch_action_map(active[0], team_size=len(rows)))
    return targets + (-1,)*(5-len(targets))


@dataclass(frozen=True)
class OwnerSnapshot:
    namespace: str
    battle_id: str
    owner: str
    seed: int
    stream_id: int
    generation: int
    lines: tuple[str, ...]
    prefix_sha256: str
    request_count: int
    own_request_line: str | None


class NoAction(ValueError):
    """A legitimate absent/waiting own request; no policy call is permitted."""


class LiveProtocolActor:
    def __init__(self, contract, *, backend_factory, policy_factory):
        _require(type(contract) is ActorContract and callable(backend_factory)
            and callable(policy_factory), 'contract and fresh providers required')
        self.contract, self.backend_factory, self.policy_factory = contract, backend_factory, policy_factory
        self.status, self._seen_battles, self._seen_streams = 'IDLE', set(), set()
        self._backend = self._policy = None

    def begin_game(self, *, battle_id, owner, seed, stream_id):
        _require(self.status in ('IDLE','CLOSED') and type(battle_id) is str and battle_id.strip()
            and battle_id not in self._seen_battles and owner in ('p1','p2') and type(owner) is str
            and type(seed) is int and 0 <= seed < 2**32 and type(stream_id) is int
            and 0 < stream_id < 2**31 and stream_id not in self._seen_streams,
            'fresh canonical battle/owner/seed/stream registration required')
        self._seen_battles.add(battle_id); self._seen_streams.add(stream_id)
        self.battle_id, self.owner, self.seed, self.stream_id = battle_id, owner, seed, stream_id
        self._lines, self._request_line, self._request_count = [], None, 0
        self.generation, self._prefix, self._snapshot = 0, hashlib.sha256(), None
        self._decision_generation, self._choice, self.error, self.close_error = None, None, None, None
        self._decision_observation, self._decision_snapshot = None, None
        self._decision_request_count, self._terminal = None, False
        self._backend = self._policy = None
        self.status = 'ACTIVE'
        try:
            self._backend = self.backend_factory(self.contract,battle_id=battle_id,owner=owner,stream_id=stream_id)
            self._policy = self.policy_factory(self.contract)
            _require(isinstance(self._backend,OwnedProvider) and isinstance(self._policy,RawPolicy),
                     'owned backend and common raw policy required')
            self._backend.claim(self.contract); self._policy.claim(self.contract)
            self._policy.reset()
        except BaseException as error:
            self._poison(error); raise

    def _active(self):
        _require(self.status == 'ACTIVE', 'actor not active')
        if self._backend.contract != self.contract or self._policy.contract != self.contract:
            error = ValueError('provider namespace drift')
            self._poison(error)
            raise error

    def _poison(self, error):
        self.status, self.error, self._snapshot, self._choice = 'POISONED', type(error).__name__, None, None
        self._decision_observation, self._decision_snapshot = None, None

    @property
    def prefix_sha256(self):
        return self._prefix.hexdigest()

    def accept_owner_stream(self, *, battle_id, owner, generation, prefix_sha256, lines):
        self._active()
        try:
            _require(type(battle_id) is str and type(owner) is str
                and battle_id == self.battle_id and owner == self.owner and type(generation) is int
                and generation == self.generation and _hex(prefix_sha256)
                and prefix_sha256 == self.prefix_sha256,
                'stream battle/owner/generation/prefix mismatch')
            _require(not self._terminal, 'postterminal owner stream')
            _require(type(lines) is tuple and bool(lines) and all(type(s) is str and s.startswith('|')
                and '\n' not in s and '\r' not in s for s in lines), 'literal owning line tuple required')
            terminal_seen = False
            for line in lines:
                # A bare pipe is the existing protocol's empty separator.
                # It contains no event or private request. No other annotation
                # or gameplay after a terminal is admitted by this actor.
                _require(not terminal_seen or line == '|', 'postterminal line within owner chunk')
                terminal_seen |= _terminal(line)
            requests = [line for line in lines if line.startswith('|request|')]
            for line in requests:
                _packet(line,self.owner)
            self._backend.ingest(lines)
            for line in lines:
                self._prefix.update(line.encode()+b'\n')
            self._lines.extend(lines)
            self._request_count += len(requests)
            if requests:
                self._request_line = requests[-1]
            self._terminal = terminal_seen
            self.generation += 1
            self._snapshot, self._choice = None, None
            self._decision_observation, self._decision_snapshot = None, None
        except BaseException as error:
            self._poison(error); raise

    def snapshot_boundary(self):
        self._active()
        self._snapshot = OwnerSnapshot(self.contract.namespace,self.battle_id,self.owner,self.seed,
            self.stream_id,self.generation,tuple(self._lines),self.prefix_sha256,
            self._request_count,self._request_line)
        return self._snapshot

    def _current(self, snapshot):
        self._active()
        _require(snapshot is self._snapshot and type(snapshot) is OwnerSnapshot
            and snapshot.namespace == self.contract.namespace and snapshot.generation == self.generation
            and snapshot.prefix_sha256 == self.prefix_sha256, 'stale or foreign owning snapshot')

    def decide(self, snapshot, *, rng):
        self._current(snapshot)
        self._decision_observation, self._decision_snapshot = None, None
        if self._terminal or snapshot.own_request_line is None or not _actionable(_packet(snapshot.own_request_line,self.owner)):
            raise NoAction('own request is not actionable')
        _require(self._decision_request_count != self._request_count, 'own request already consumed')
        try:
            encoded, targets = self._backend.encode(snapshot)
            observation = self._observation(encoded,snapshot)
            from .showdown import _legal_action_mask
            _require(observation.legal_action_mask == _legal_action_mask(_packet(snapshot.own_request_line,self.owner)),
                     'encoded own legal mask differs')
            targets = tuple(targets)
            _require(targets == _request_targets(_packet(snapshot.own_request_line,self.owner)),
                     'encoded own switch targets differ')
            decision = self._policy.select_action(observation,rng=rng)
            self._active()
            _require(type(decision) is PolicyDecision and type(decision.action_index) is int
                and observation.legal_action_mask[decision.action_index], 'illegal raw decision')
            action = decision.action_index
            self._choice = f'move {action+1}' if action < 4 else f'switch {targets[action-4]+1}'
            _require(action < 4 or targets[action-4] >= 0, 'missing switch target')
            self._decision_generation = self.generation
            self._decision_request_count = self._request_count
            self._decision_observation, self._decision_snapshot = observation, snapshot
            return decision
        except BaseException as error:
            self._poison(error); raise

    def _observation(self, encoded, snapshot):
        arrays = encoded if type(encoded) is dict else {k:getattr(encoded,k) for k in SHAPES}
        _require(set(arrays) == set(SHAPES), 'exact five owning tensors required')
        result = {}
        for name, shape in SHAPES.items():
            a = np.asarray(arrays[name])
            kind = 'b' if name.endswith('mask') else 'iu' if name != 'numeric_features' else 'iuf'
            _require(a.shape == shape and a.dtype.kind in kind and np.isfinite(a).all(), 'invalid tensor '+name)
            if name == 'numeric_features':
                a = a.astype(np.float32)
                _require(np.isfinite(a).all(), 'nonfinite FP32 numeric tensor')
            elif not name.endswith('mask'):
                limit = self.contract.token_type_vocab_size if name=='token_type_ids' else self.contract.categorical_vocab_size
                _require((a >= 0).all() and (a < limit).all(), 'category/token index outside bound vocabulary')
            result[name] = tuple(tuple(row) for row in a.tolist()) if a.ndim==2 else tuple(a.tolist())
        _require(any(result['legal_action_mask']) and any(result['attention_mask']), 'no legal attended observation')
        if isinstance(encoded,PokeZeroObservationV0):
            _require(encoded.schema_version == 'pokezero.observation.v4' and encoded.perspective is not None
                and encoded.perspective.player_id == self.owner and encoded.metadata.get('battle_id') == self.battle_id,
                'Python observation owner/schema differs')
        return PokeZeroObservationV0(**result,schema_version='pokezero.observation.v4',
            perspective=ObservationPerspective.from_showdown_slot(self.owner,self.owner),
            metadata=MappingProxyType(dict(battle_id=self.battle_id,player_id=self.owner,
                owner_prefix_sha256=snapshot.prefix_sha256,actor_namespace=snapshot.namespace,
                generation=snapshot.generation,**UNQUALIFIED)))

    def choice(self, snapshot):
        self._current(snapshot)
        _require(self._choice is not None and self._decision_generation == self.generation, 'no bound decision')
        return self._choice

    def decision_observation(self, snapshot):
        """Exact immutable raw-policy input, after a completed current decision.

        This accessor performs no encoding, inference or cursor advance. Keep
        the snapshot passed to decide: an equal or newly issued snapshot is not
        the successful decision's identity.
        """
        self._current(snapshot)
        _require(not self._terminal and self._decision_snapshot is snapshot
            and self._decision_observation is not None and self._choice is not None
            and self._decision_generation == snapshot.generation
            and self._decision_request_count == snapshot.request_count,
            'no completed owning decision observation')
        return self._decision_observation

    def close_game(self):
        _require(self.status in ('ACTIVE','POISONED'), 'actor already closed or unstarted')
        was_poisoned = self.status == 'POISONED'
        for provider in (self._backend,self._policy):
            if isinstance(provider,OwnedProvider):
                try:
                    provider.close()
                except BaseException as error:
                    self.close_error = type(error).__name__; was_poisoned = True
        self.status, self._snapshot, self._choice = 'CLOSED', None, None
        self._decision_observation, self._decision_snapshot = None, None
        return dict(status='FAILED_CLOSED' if was_poisoned else 'TERMINAL_UNQUALIFIED' if self._terminal else 'ABORTED_UNQUALIFIED',
            battle_id=self.battle_id,owner=self.owner,generation=self.generation,
            terminal_seen=self._terminal,
            owner_lines=len(self._lines),owner_requests=self._request_count,
            prefix_sha256=self.prefix_sha256,error=self.error,close_error=self.close_error,**UNQUALIFIED)
