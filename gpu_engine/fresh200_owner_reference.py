"""Incremental process-free Python/B2 references for ONE literal Native owner.

Every actionable request is encoded in chronology by the existing production
backends. Waits invalidate endpoints; public tails require explicit re-encoding.
No device observations, policy/model, choices, chance draws or simulator setup
are accepted here. Runtime/source attestation and physical replay are separate
caller gates. Reference artifacts must be sealed before device observations.
"""
from dataclasses import dataclass
import hashlib
import os
import re
from types import MappingProxyType, SimpleNamespace

import numpy as np

from .fresh200_owner_history import OwnerHistory


def _runtime_bindings():
    # Lazy imports keep transport/ordering guards independent of JAX/Torch.
    from pokezero.live_protocol_actor import ActorContract, OwnerSnapshot, PythonBackend, JaxBackend, _terminal
    from pokezero.recorded_encoder_runner import literal_history
    from pokezero.recorded_encoder_parity import SHAPES
    from .owning_python_reference import _lines
    return SimpleNamespace(ActorContract=ActorContract, OwnerSnapshot=OwnerSnapshot,
        PythonBackend=PythonBackend, JaxBackend=JaxBackend, literal_history=literal_history,
        validate_lines=_lines, terminal=_terminal, shapes=SHAPES)


def _bindings(value):
    fields = dict(producer_commit=40, runtime_sha256=64, native_source_registry_hash=64)
    if (type(value) is not dict or set(value) != set(fields)
            or any(type(value[k]) is not str or re.fullmatch('[0-9a-f]{%d}' % n, value[k]) is None
                   for k, n in fields.items())):
        raise ValueError('explicit independently verified producer/runtime/Native source bindings required')
    return dict(value)


def _cpu_only():
    if (os.environ.get('JAX_PLATFORMS') != 'cpu' or os.environ.get('CUDA_VISIBLE_DEVICES') != ''
            or os.environ.get('NVIDIA_VISIBLE_DEVICES') not in ('void', 'none')):
        raise ValueError('explicit CPU-only reference environment required')


def compile_catalog(config, belief_source, dex):
    """Canonical B2 catalog only; no actor, model, scoring or policy is built."""
    _cpu_only()
    from pokezero.live_protocol_actor import ActorContract
    from . import public_belief, v4_encoder
    contract = ActorContract.champion()
    return v4_encoder.compile_encoder(public_belief.compile_source(belief_source), dex,
        config.category_vocab, checkpoint_sha256=contract.checkpoint_sha256,
        feature_masks=config.feature_masks)


def _freeze(value, *, device, contract, shapes):
    if type(value) is not dict or set(value) != set(shapes):
        raise ValueError('exact five production owner tensors required')
    result = {}
    for name, shape in shapes.items():
        array = np.asarray(value[name]);kind = 'b' if name.endswith('mask') else 'iu' if name != 'numeric_features' else 'iuf'
        if array.shape != shape or array.dtype.kind not in kind or not np.isfinite(array).all():
            raise ValueError('invalid production owner tensor: '+name)
        if device:
            dtype = np.dtype('float32' if name == 'numeric_features' else 'bool' if kind == 'b' else 'int32')
            if array.dtype != dtype:raise ValueError('canonical B2 tensor dtype required: '+name)
        if name in ('categorical_ids', 'token_type_ids'):
            size = contract.categorical_vocab_size if name == 'categorical_ids' else contract.token_type_vocab_size
            if np.any(array < 0) or np.any(array >= size):raise ValueError('owner tensor namespace differs: '+name)
        if name in ('attention_mask', 'legal_action_mask') and not array.any():
            raise ValueError('actionable owning tensors require attention and a legal action')
        # Immutable bytes also prevent callers from re-enabling writeability.
        # Python binary64 and canonical B2 float32 remain separate, uncast.
        result[name] = np.frombuffer(array.tobytes(order='C'), dtype=array.dtype).reshape(shape)
    return MappingProxyType(result)


@dataclass(frozen=True)
class OwnerReferences:
    cut: object
    python: object
    b2: object
    switch_targets: tuple

    def tensor_identity(self):
        return {kind:{name:dict(shape=list(value.shape), dtype=value.dtype.str,
                    sha256=hashlib.sha256(value.tobytes(order='C')).hexdigest())
                for name, value in tensors.items()} for kind, tensors in (('python', self.python), ('b2', self.b2))}


class OwnerReferenceSession:
    def __init__(self, config, catalog, *, battle_id, owner, seed, stream_id, bindings):
        _cpu_only()
        if (type(battle_id) is not str or not battle_id.strip() or '\n' in battle_id or '\r' in battle_id
                or owner not in ('p1', 'p2') or type(seed) is not int or not 0 <= seed < 2**32
                or type(stream_id) is not int or not 0 < stream_id < 2**31):
            raise ValueError('exact frozen battle/owner/seed/stream identity required')
        self.bindings = _bindings(bindings);self._runtime = runtime = _runtime_bindings()
        self.contract = runtime.ActorContract.champion()
        self.battle_id = battle_id;self.owner = owner;self.seed = seed;self.stream_id = stream_id
        self.history = OwnerHistory(owner);self._pending = [];self.records = [];self._sealed = {}
        self.status = 'ACTIVE';self.error = None;self.terminal = False
        self.python = runtime.PythonBackend(self.contract, config=config, battle_id=battle_id, owner=owner, stream_id=stream_id)
        try:
            self.b2 = runtime.JaxBackend(self.contract, catalog=catalog, battle_id=battle_id, owner=owner, stream_id=stream_id)
        except BaseException:
            self.python.close();raise
        if type(self.python) is not runtime.PythonBackend or type(self.b2) is not runtime.JaxBackend:
            raise ValueError('canonical production PythonBackend/JaxBackend required')

    def _active(self):
        if self.status != 'ACTIVE':raise ValueError('owner reference session is '+self.status)

    def _poison(self, error):
        self.status = 'POISONED';self.error = type(error).__name__;self.history.reference = None

    def _flush(self):
        if self._pending:
            lines = tuple(self._pending)
            self.python.ingest(lines);self.b2.ingest(lines);self._pending.clear()

    def ingest(self, lines):
        self._active()
        try:
            if type(lines) not in (list, tuple):raise ValueError('literal ordered Native owning lines required')
            if not lines:return ()
            # Validate the entire incoming batch before either backend sees
            # it. Before the first request, hold public lines without encoding.
            trial = OwnerHistory(self.owner);trial.append((*self.history.lines, *lines))
            if trial.current is not None:self._runtime.validate_lines(tuple(trial.lines), self.owner)
            created = []
            for line in lines:
                if self.terminal and line != '|':raise ValueError('postterminal literal owner input')
                cuts = self.history.append((line,))
                if cuts:
                    self._flush();self.python.ingest((line,));self.b2.ingest((line,))
                    if cuts[0].actionable:created.append(self._encode())
                else:
                    self._pending.append(line)
                    if self._runtime.terminal(line):
                        self.terminal = True;self.history.reference = None
            if self.history.current is not None:self._flush()
            return tuple(created)
        except BaseException as error:
            self._poison(error);raise

    def _encode(self):
        if self.terminal:raise ValueError('no reference endpoint after actual Native terminal lifecycle')
        cut = self.history.current_cut();runtime = self._runtime
        snapshot = runtime.OwnerSnapshot(self.contract.namespace, self.battle_id, self.owner, self.seed,
            self.stream_id, cut.generation, cut.lines, cut.prefix_sha256, cut.generation, cut.request_line)
        observation, python_targets = self.python.encode(snapshot)
        python = {name:getattr(observation, name) for name in runtime.shapes}
        with runtime.literal_history(self.b2.adapter, cut.lines):b2, b2_targets = self.b2.encode(snapshot)
        python_targets = tuple(python_targets);b2_targets = tuple(b2_targets)
        if (python_targets != b2_targets or len(python_targets) != 5
                or any(type(x) is not int or not -1 <= x < 6 for x in python_targets)):
            raise ValueError('canonical Python/B2 owning switch targets differ')
        result = OwnerReferences(cut,
            _freeze(python, device=False, contract=self.contract, shapes=runtime.shapes),
            _freeze(b2, device=True, contract=self.contract, shapes=runtime.shapes), python_targets)
        self.history.bind_reference(cut, result)
        self.records.append(dict(generation=cut.generation, prefix_sha256=cut.prefix_sha256,
            request_line=cut.request_line, tensors=result.tensor_identity()))
        return result

    def endpoint(self):
        self._active()
        if self.terminal:raise ValueError('no reference endpoint after actual Native terminal lifecycle')
        return self.history.endpoint()

    def reencode_current(self):
        self._active()
        try:
            if self.terminal:raise ValueError('no reference endpoint after actual Native terminal lifecycle')
            cut = self.history.current_cut()
            if self.history.reference is not None and self.history.reference[0] == cut:
                return self.history.reference[1]
            self._flush()
            return self._encode()
        except BaseException as error:
            self._poison(error);raise

    def seal_current(self, sealer):
        """A failed artifact write may retry without replaying or re-encoding."""
        pair = self.endpoint();key = pair.cut.prefix_sha256
        if key in self._sealed:return dict(self._sealed[key])
        ref = sealer(pair, self.identity())
        if (type(ref) is not dict or set(ref) != {'path', 'bytes', 'sha256'}
                or type(ref['path']) is not str or not ref['path'] or type(ref['bytes']) is not int or ref['bytes'] <= 0
                or type(ref['sha256']) is not str or re.fullmatch('[a-f0-9]{64}', ref['sha256']) is None):
            raise ValueError('sealed immutable current owner-reference artifact required')
        self._sealed[key] = dict(ref)
        return dict(ref)

    def identity(self):
        return dict(battle_id=self.battle_id, owner=self.owner, seed=self.seed, stream_id=self.stream_id,
            namespace=self.contract.namespace, bindings=dict(self.bindings), request_generation=self.history.generation,
            literal_owner_lines=len(self.history.lines), prefix_sha256=self.history.prefix_sha256,
            encoded_cuts=len(self.records), status=self.status, terminal=self.terminal,
            actual_device_observations_loaded=False, simulator_started=False, policy_or_scoring_run=False,
            qualification=False)

    def close(self):
        if self.status == 'CLOSED':return
        failures = []
        for backend in (self.python, self.b2):
            try:backend.close()
            except BaseException as error:failures.append(error)
        self.status = 'CLOSED';self.history.reference = None
        if failures:raise failures[0]
