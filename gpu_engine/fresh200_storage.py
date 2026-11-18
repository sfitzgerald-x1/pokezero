"""Value-free, source-bound CPU schemas for the immutable f9 fresh200 capture.

No engine module, initializer, actor, sampler or PRNG is imported or evaluated.
Every restored value must come from the unchanged strict attestation loader.
GPU Trackers read with these schemas are capture-integrity evidence only; they
must never feed Native reconstruction, its witness, or reference encoding.
"""
import ast
from collections import namedtuple
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import zipfile

import numpy as np

INTEGRATION = 'f9b0c48ef0d14c15f017c3e4b97dddc6695c5b3d'
ORIGINAL_POOL = '859d07079b3955a3811993526216f23746c40158064c84f8528361922cefe9a3'
SCHEMA_SHA256 = '5d6d3bfc9b1c31b975b84dc8ad16b44882d7c44b3fc7d5a37a98b0fa1c074e0d'
TYPES = frozenset(('gpu_engine.state.BattleState', 'gpu_engine.transition.ChanceInputs',
    'gpu_engine.engine_device_public_hooks.Initialized', 'gpu_engine.engine_device_trackers.Trackers',
    'gpu_engine.engine_device_trackers.Tendency', 'gpu_engine.engine_device_damage_history.History',
    'gpu_engine.engine_device_negative_items.State', 'gpu_engine.engine_device_visible_belief.State',
    'gpu_engine.engine_device_public_port.State'))


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(65536), b''):digest.update(block)
    return digest.hexdigest()


@dataclass(frozen=True)
class ArraySpec:
    shape: tuple
    dtype: np.dtype

    def __post_init__(self):
        if not isinstance(self.shape, tuple) or any(type(x) is not int or x < 0 for x in self.shape):
            raise ValueError('literal nonnegative storage dimensions required')
        dtype = np.dtype(self.dtype)
        if dtype.kind not in 'biufc' or dtype.hasobject:raise ValueError('primitive storage dtype required')
        object.__setattr__(self, 'dtype', dtype)

    def __array__(self, *args, **kwargs):
        raise TypeError('storage specification has no battle values')


def scan_spec(value, steps):
    if type(steps) is not int or steps <= 0:raise ValueError('positive literal scan length required')
    if value is None:return None
    if isinstance(value, ArraySpec):return ArraySpec((steps, *value.shape), value.dtype)
    if isinstance(value, tuple) and hasattr(value, '_fields'):
        return type(value)(*(scan_spec(x, steps) for x in value))
    if isinstance(value, dict):return {k:scan_spec(v, steps) for k, v in value.items()}
    if isinstance(value, tuple):return tuple(scan_spec(x, steps) for x in value)
    if isinstance(value, list):return [scan_spec(x, steps) for x in value]
    raise TypeError('only trusted specification nodes may acquire a scan axis')


def pool_dimensions(path, expected_sha256):
    """Read only metadata_utf8 after binding the exact resident NPZ bytes."""
    path = Path(path)
    if sha256_file(path) != expected_sha256:raise ValueError('bound constructor pool SHA differs')
    with zipfile.ZipFile(path) as archive:
        members = [x for x in archive.infolist() if x.filename == 'metadata_utf8.npy']
        if len(members) != 1 or members[0].file_size > 4*1024**2:
            raise ValueError('one bounded literal pool metadata array required')
        with archive.open(members[0]) as stream:value = np.load(stream, allow_pickle=False)
    if value.dtype != np.dtype('uint8') or value.ndim != 1:
        raise ValueError('literal uint8 pool metadata required')
    blob = value.tobytes();metadata = json.loads(blob);vocabulary = metadata['vocabulary']
    names = {}
    for kind in ('abilities', 'items'):
        values = vocabulary[kind]
        if not isinstance(values, list) or not values or any(type(x) is not str for x in values):
            raise ValueError('complete bound '+kind+' vocabulary required')
        names[kind] = tuple(values)
    return dict(pool_path=str(path), pool_sha256=expected_sha256, pool_bytes=path.stat().st_size,
        metadata_sha256=hashlib.sha256(blob).hexdigest(),
        abilities=names['abilities'], items=names['items'],
        ability_words=(len(names['abilities'])+31)//32, item_words=(len(names['items'])+31)//32)


def _schema():
    path = Path(__file__).with_name('fresh200_storage_schema.json');blob = path.read_bytes()
    if hashlib.sha256(blob).hexdigest() != SCHEMA_SHA256:raise ValueError('trusted storage schema SHA differs')
    value = json.loads(blob)
    if (value['schema'] != 'pokezero.fresh200.trusted_storage.v1' or value['source_commit'] != INTEGRATION
            or set(value['records']) != TYPES or value['none_fields'] != []):
        raise ValueError('exact full initialized capture storage profile required')
    return value


def verify_source(source_root, schema):
    """Check original integration text; parse declarations without importing it."""
    root = Path(source_root);trees = {}
    for name, expected in schema['source_files'].items():
        path = root/name
        if sha256_file(path) != expected:raise ValueError('capture storage source differs: '+name)
        trees[name] = ast.parse(path.read_bytes(), filename=name)
    for identity, fields in schema['records'].items():
        module, name = identity.rsplit('.', 1);tree = trees[module.replace('.', '/')+'.py']
        definitions = [x for x in tree.body if isinstance(x, ast.ClassDef) and x.name == name]
        actual = [x.target.id for x in definitions[0].body if isinstance(x, ast.AnnAssign)] if len(definitions) == 1 else None
        expected = [x['name'] for x in fields]
        if actual != expected or len(set(expected)) != len(expected):
            raise ValueError('complete trusted source record fields differ: '+identity)
    return dict(schema['source_files'])


class SealedStorage:
    """Full typed storage layouts, fixed from source rather than NPZ descriptors."""
    def __init__(self, source_root, pool_path, *, rows=256, expected_pool_sha256=ORIGINAL_POOL):
        if type(rows) is not int or rows <= 0:raise ValueError('positive literal storage batch required')
        if expected_pool_sha256 != ORIGINAL_POOL:raise ValueError('the frozen original constructor pool is required')
        schema = _schema();self.source_files = verify_source(source_root, schema)
        self.binding = pool_dimensions(pool_path, expected_pool_sha256)
        self.rows = rows;self.schema = schema
        self.dimensions = dict(B=rows, V=len(schema['volatile_names']),
            A=self.binding['ability_words'], I=self.binding['item_words'])
        self.types = {}
        for identity, fields in schema['records'].items():
            module, name = identity.rsplit('.', 1)
            # Fixed reviewed registry only; never a class named by an artifact.
            self.types[identity] = namedtuple(name, [x['name'] for x in fields], module=module)

    def _record(self, identity):
        if identity not in TYPES:raise ValueError('untrusted storage record')
        return self.types[identity](*(self._spec(x['spec']) for x in self.schema['records'][identity]))

    def _spec(self, spec):
        kind = spec['kind']
        if kind == 'array':
            shape = tuple(self.dimensions[x] if isinstance(x, str) else x for x in spec['shape'])
            return ArraySpec(shape, np.dtype(spec['dtype']))
        if kind == 'record':return self._record(spec['type'])
        if kind == 'dict':return {x['name']:self._spec(x['spec']) for x in spec['entries']}
        if kind == 'none':return None
        raise ValueError('unsupported trusted storage node')

    def state(self):return self._record('gpu_engine.state.BattleState')

    def trackers(self):return self._record('gpu_engine.engine_device_trackers.Trackers')

    def constructor_input(self):
        # Producer device_get sorted the original constructor mapping.
        return dict(initial_tie=ArraySpec((self.rows,), np.int32), raw=self.state(),
            streams=ArraySpec((self.rows,), np.int32))

    def initialized(self):return self._record('gpu_engine.engine_device_public_hooks.Initialized')

    def replay_inputs(self, steps):
        b = self.rows;state = self.state();public = self.trackers()
        # Producer constructed this outer mapping after device_get.
        tree = dict(pre_state=state, pre_public=public, step_state=state, post_state=state, post_public=public,
            actions=ArraySpec((b, 2), np.int32), requested=ArraySpec((b, 2), np.bool_),
            streams=ArraySpec((b,), np.int32), frames=ArraySpec((b,), np.int32), status=ArraySpec((b,), np.int32),
            chance=self._record('gpu_engine.transition.ChanceInputs'), actor_keys=ArraySpec((b, 2, 2), np.uint32))
        return scan_spec(tree, steps)

    def identity(self):
        return dict(schema='pokezero.fresh200.storage_reader.v1', integration_commit=INTEGRATION,
            storage_schema_sha256=SCHEMA_SHA256, reader_sha256=sha256_file(__file__), source_files=self.source_files,
            pool_binding=self.binding, rows=self.rows, profile=self.schema['profile'],
            no_engine_import_or_initialization=True, no_random_inputs=True, qualification=False)
