"""Source-checked input access and freeze-gated physical boundary slices.

The caller first verifies the frozen capture DAG and its snapshot links. This
reader checks each linked descriptor against the independent f9 storage layout.
It hashes compressed artifacts without decoding expected state, then decodes
only explicitly permitted inputs or a sealed Native boundary's matching row.
There is no engine import, initialization, sampling or expected-state repair.
"""
from dataclasses import dataclass
import hashlib
import io
import json
import math
from pathlib import Path
import re
import zipfile

import numpy as np

from . import device_corpus_attestation as attestation
from .fresh200_boundary_gate import BoundaryGate
from .fresh200_storage import ArraySpec, INTEGRATION, SealedStorage

CONSTRUCTOR_FIELDS = ('species', 'level', 'types', 'stats', 'moves', 'pp', 'item',
    'ability', 'hidden_power_type', 'hidden_power_power', 'gender', 'happiness', 'keys')
MAX_BYTES = 512 * 1024**2


def _json(data):
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:raise ValueError('duplicate snapshot metadata key')
            result[key] = value
        return result
    def constant(value):raise ValueError('nonfinite snapshot metadata constant')
    return json.loads(data, object_pairs_hook=pairs, parse_constant=constant)


def _index(value, size):
    if type(value) is not int or not 0 <= value < size:
        raise ValueError('literal in-range sealed array coordinate required')
    return value


@dataclass(frozen=True)
class _Leaf:
    member: str
    spec: ArraySpec
    value_sha256: str
    offset: int


class _Snapshot:
    """A complete value-free descriptor and the exact verified NPZ bytes."""
    def __init__(self, root, reference, template, *, source):
        self.root = Path(root).resolve();self.reference = dict(reference)
        self.source = attestation._source(source)
        if self.source['engine_commit'] != INTEGRATION:
            raise ValueError('frozen f9 capture source required')
        document = _json(attestation.verify_artifact(root, reference))
        if (set(document) != {'schema', 'kind', 'source', 'tree', 'arrays'}
                or document['schema'] != attestation.SCHEMA or document['kind'] != 'snapshot'
                or document['source'] != self.source):
            raise ValueError('exact source-bound snapshot required')
        descriptor = document['tree']
        if attestation._structure(descriptor) != attestation._tree(template):
            raise ValueError('independent complete storage layout differs')
        leaves = []
        def visit(old, node, path):
            if isinstance(old, ArraySpec):
                key = 'a%06d' % len(leaves)
                if (node.get('array') != key or type(node.get('value_sha256')) is not str
                        or re.fullmatch('[0-9a-f]{64}', node['value_sha256']) is None):
                    raise ValueError('canonical source-layout leaf address/hash required')
                leaves.append((path, key+'.npy', old, node['value_sha256']))
            elif isinstance(old, tuple) and hasattr(old, '_fields'):
                for name, child in zip(old._fields, node['children']):
                    visit(getattr(old, name), child, (*path, name))
            elif isinstance(old, dict):
                for (name, value), child in zip(old.items(), node['children']):
                    visit(value, child, (*path, name))
            elif old is not None:
                raise ValueError('trusted ArraySpec/record/dict storage nodes required')
        visit(template, descriptor, ())
        self._data = attestation.verify_artifact(root, document['arrays'])
        self.arrays_reference = dict(document['arrays']);self._leaves = {}
        with zipfile.ZipFile(io.BytesIO(self._data)) as archive:
            members = archive.infolist()
            if (len(members) != len(leaves) or len({m.filename for m in members}) != len(members)
                    or {m.filename for m in members} != {x[1] for x in leaves}
                    or sum(m.file_size for m in members) > MAX_BYTES):
                raise ValueError('complete unique bounded NPZ member roster required')
            for path, name, spec, digest in leaves:
                member = archive.getinfo(name)
                if member.flag_bits & 1 or member.compress_type not in (zipfile.ZIP_STORED, zipfile.ZIP_DEFLATED):
                    raise ValueError('unsupported NPZ member encoding')
                with archive.open(member) as stream:
                    version = np.lib.format.read_magic(stream)
                    if version == (1, 0):reader = np.lib.format.read_array_header_1_0
                    elif version == (2, 0):reader = np.lib.format.read_array_header_2_0
                    else:raise ValueError('unsupported primitive NPY version')
                    shape, fortran, dtype = reader(stream, max_header_size=16384)
                    offset = stream.tell()
                if (shape != spec.shape or dtype.str != spec.dtype.str or fortran
                        or dtype.hasobject or dtype.kind not in 'biufc'
                        or member.file_size != offset+math.prod(spec.shape)*spec.dtype.itemsize):
                    raise ValueError('NPY header/length differs from trusted C-order storage')
                self._leaves[path] = _Leaf(name, spec, digest, offset)

    def _read(self, path, indices):
        """Hash full literal leaf bytes; decode only the requested C-order span."""
        leaf = self._leaves[path];shape = leaf.spec.shape
        if type(indices) is not tuple or len(indices) > len(shape):
            raise ValueError('literal leading storage coordinates required')
        ordinal = 0
        for value, size in zip(indices, shape):ordinal = ordinal*size+_index(value, size)
        tail = shape[len(indices):];length = math.prod(tail)*leaf.spec.dtype.itemsize
        start = ordinal*length;end = start+length;digest = hashlib.sha256();selected = bytearray();position = 0
        with zipfile.ZipFile(io.BytesIO(self._data)) as archive, archive.open(leaf.member) as stream:
            if len(stream.read(leaf.offset)) != leaf.offset:raise ValueError('truncated NPY header')
            for block in iter(lambda: stream.read(65536), b''):
                digest.update(block);stop = position+len(block)
                if position < end and stop > start:
                    selected.extend(block[max(0, start-position):min(len(block), end-position)])
                position = stop
        if (position != math.prod(shape)*leaf.spec.dtype.itemsize
                or digest.hexdigest() != leaf.value_sha256 or len(selected) != length):
            raise ValueError('sealed full leaf bits/span differ')
        value = np.frombuffer(selected, dtype=leaf.spec.dtype).reshape(tail).copy()
        value.flags.writeable = False
        return value

    def identity(self):
        return dict(snapshot=dict(self.reference), arrays=dict(self.arrays_reference),
            source=dict(self.source), leaf_count=len(self._leaves), no_value_decoding_on_open=True)


class SealedInputs:
    """Closed selectors: constructor operands or recorded replay inputs only."""
    def __init__(self, root, reference, storage, *, source, kind, start_step=0, steps=None):
        if not isinstance(storage, SealedStorage):raise ValueError('independent f9 SealedStorage required')
        if kind == 'constructor':
            if type(start_step) is not int or start_step != 0 or steps is not None:
                raise ValueError('constructor has no replay scan axis')
            template = storage.constructor_input()
        elif kind == 'replay':
            if (type(start_step) is not int or start_step < 0 or type(steps) is not int
                    or not 1 <= steps <= 4 or start_step+steps > 64):
                raise ValueError('exact linked frozen replay chunk coordinates required')
            template = storage.replay_inputs(steps)
        else:raise ValueError('constructor/replay input selector required')
        self.storage = storage;self.kind = kind;self.start_step = start_step;self.steps = steps
        self._snapshot = _Snapshot(root, reference, template, source=source)

    def _coordinates(self, row, step=None):
        _index(row, self.storage.rows)
        if self.kind == 'constructor':
            if step is not None:raise ValueError('constructor has no step coordinate')
            return (row,)
        return (_index(step, self.steps), row)

    def stream(self, row, *, step=None):
        value = self._snapshot._read(('streams',), self._coordinates(row, step))
        if value.shape != () or int(value) <= 0:raise ValueError('positive literal frozen stream identity required')
        return int(value)

    def constructor(self, row):
        if self.kind != 'constructor':raise ValueError('constructor input snapshot required')
        coordinates = self._coordinates(row)
        return dict(raw={name:self._snapshot._read(('raw', name), coordinates) for name in CONSTRUCTOR_FIELDS},
            initial_tie=self._snapshot._read(('initial_tie',), coordinates),
            streams=self._snapshot._read(('streams',), coordinates))

    def replay_inputs(self, step, row):
        if self.kind != 'replay':raise ValueError('replay input snapshot required')
        coordinates = self._coordinates(row, step)
        result = {name:self._snapshot._read((name,), coordinates)
            for name in ('actions', 'requested', 'streams', 'frames', 'status', 'actor_keys')}
        chance = self.storage._record('gpu_engine.transition.ChanceInputs')
        result['chance'] = {name:self._snapshot._read(('chance', name), coordinates) for name in chance._fields}
        return result

    def identity(self):return dict(self._snapshot.identity(), kind=self.kind,
        start_step=self.start_step, steps=self.steps, rows=self.storage.rows)


class ExpectedState:
    """Decode 73 expected physical fields only through a frozen BoundaryGate.

    A production Native sealer must validate and persist the typed projection
    before producing the gate's receipt. This class does not manufacture it.
    Constructor inputs and replay chunk links must come from the verified DAG.
    """
    def __init__(self, inputs, *, initialized_reference=None):
        if not isinstance(inputs, SealedInputs):raise ValueError('source-bound input reader required')
        self._inputs = inputs
        if inputs.kind == 'constructor':
            if initialized_reference is None:raise ValueError('linked initialized snapshot required')
            self._snapshot = _Snapshot(inputs._snapshot.root, initialized_reference,
                inputs.storage.initialized(), source=inputs._snapshot.source)
        else:
            if initialized_reference is not None:raise ValueError('replay already contains physical boundaries')
            self._snapshot = inputs._snapshot

    def read(self, gate, *, row):
        if not isinstance(gate, BoundaryGate):raise ValueError('typed Native boundary freeze gate required')
        def decode(address, receipt):
            if receipt['address'] != address or receipt['field_count'] != 73:
                raise ValueError('matching complete Native boundary receipt required')
            if self._inputs.kind == 'constructor':
                if address.role != 'INITIALIZED' or address.step != 0:
                    raise ValueError('initialized boundary address differs')
                coordinates = self._inputs._coordinates(row);prefix = 'state';local_step = None
            else:
                if address.role not in ('PRE', 'STEP', 'POST'):raise ValueError('replay boundary role required')
                local_step = address.step-self._inputs.start_step
                coordinates = self._inputs._coordinates(row, local_step)
                prefix = {'PRE':'pre_state', 'STEP':'step_state', 'POST':'post_state'}[address.role]
            if self._inputs.stream(row, step=local_step) != address.stream:
                raise ValueError('frozen stream/row address differs')
            fields = self._inputs.storage.state()._fields
            if len(fields) != 73:raise ValueError('complete 73-field trusted physical schema required')
            return {name:self._snapshot._read((prefix, name), coordinates) for name in fields}
        return gate.read_expected(decode)
