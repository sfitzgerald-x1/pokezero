"""Immutable, template-checked transport for complete device-corpus trees.

This module neither constructs game state nor supplies missing history. Callers
must provide complete source identities and actual recorded arrays. Loading
uses a trusted runtime template; metadata never imports or instantiates types.
"""
import hashlib
import io
import json
from pathlib import Path
import re

import numpy as np

SCHEMA = 'pokezero.device_corpus_attestation.v1'
SOURCE_FIELDS = ('engine_commit', 'producer_commit', 'entry_sha256', 'runtime_sha256')


def _canonical(value):
    return (json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False) + '\n').encode()


def sha256_file(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):result.update(block)
    return result.hexdigest()


def _source(value):
    value = dict(value)
    for field in SOURCE_FIELDS:
        size = 40 if field.endswith('_commit') else 64
        if not isinstance(value.get(field), str) or not re.fullmatch('[0-9a-f]{%d}' % size, value[field]):
            raise ValueError('missing or invalid source binding: ' + field)
    _canonical(value)
    return value


def _name(value):
    if not isinstance(value, str) or not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]{0,119}', value):
        raise ValueError('invalid artifact name')
    return value


def _path(root, relative):
    if not isinstance(relative, str) or Path(relative).is_absolute():raise ValueError('artifact path must be relative')
    root = Path(root).resolve()
    target = (root / relative).resolve()
    if target == root or not target.is_relative_to(root):raise ValueError('artifact path escapes capsule')
    return target


def _reference(root, path):
    return dict(path=str(path.relative_to(Path(root).resolve())), sha256=sha256_file(path), bytes=path.stat().st_size)


def verify_artifact(root, reference, *, max_bytes=512 * 1024**2):
    """Return the exact bytes whose length and digest were verified."""
    if set(reference) != {'path', 'sha256', 'bytes'}:raise ValueError('incomplete artifact reference')
    expected = reference['bytes']
    if type(expected) is not int or expected < 0 or expected > max_bytes:raise ValueError('artifact size is out of bounds')
    if not isinstance(reference['sha256'], str) or not re.fullmatch(r'[0-9a-f]{64}', reference['sha256']):
        raise ValueError('invalid artifact digest')
    path = _path(root, reference['path'])
    if path.stat().st_size != expected:raise ValueError('artifact length changed')
    data = path.read_bytes()
    if len(data) != expected or hashlib.sha256(data).hexdigest() != reference['sha256']:
        raise ValueError('artifact digest changed')
    return data


def _tree(value, arrays=None):
    if value is None:return dict(kind='none')
    if isinstance(value, tuple) and hasattr(value, '_fields'):
        return dict(kind='namedtuple', type=type(value).__module__ + '.' + type(value).__qualname__,
                    fields=list(value._fields), children=[_tree(getattr(value, name), arrays) for name in value._fields])
    if isinstance(value, dict):
        if any(not isinstance(key, str) for key in value):raise TypeError('tree dictionary keys must be strings')
        return dict(kind='dict', keys=list(value), children=[_tree(child, arrays) for child in value.values()])
    if isinstance(value, (tuple, list)):
        return dict(kind='tuple' if isinstance(value, tuple) else 'list', children=[_tree(child, arrays) for child in value])
    scalar = type(value) in (bool, int, float, complex)
    if not scalar and not (hasattr(value, 'shape') and hasattr(value, 'dtype')):
        raise TypeError('unsupported tree leaf: ' + type(value).__name__)
    a = np.asarray(value) if scalar or arrays is not None else value
    dtype = np.dtype(a.dtype)
    if dtype.kind not in 'biufc' or dtype.hasobject:raise TypeError('unsupported primitive dtype: ' + str(dtype))
    result = dict(kind='scalar' if scalar else 'array', shape=list(a.shape), dtype=dtype.str)
    if scalar:result['python_type'] = type(value).__name__
    if arrays is not None:
        key = 'a%06d' % len(arrays)
        arrays[key] = np.ascontiguousarray(a) if a.ndim else a.copy()
        result.update(array=key, value_sha256=hashlib.sha256(arrays[key].tobytes(order='C')).hexdigest())
    return result


def _structure(value):
    """Descriptor without recorded values, for exact runtime-template checks."""
    return {key: [_structure(child) for child in item] if key == 'children' else item
            for key, item in value.items() if key not in ('array', 'value_sha256')}


def _restore(template, descriptor, arrays):
    kind = descriptor['kind']
    if kind == 'none':return None
    if kind == 'namedtuple':
        return type(template)(*(_restore(getattr(template, name), child, arrays)
                               for name, child in zip(descriptor['fields'], descriptor['children'])))
    if kind == 'dict':
        return {key: _restore(template[key], child, arrays) for key, child in zip(descriptor['keys'], descriptor['children'])}
    if kind in ('tuple', 'list'):
        children = [_restore(old, child, arrays) for old, child in zip(template, descriptor['children'])]
        return tuple(children) if kind == 'tuple' else children
    value = arrays[descriptor['array']]
    if value.dtype.str != descriptor['dtype'] or list(value.shape) != descriptor['shape']:
        raise ValueError('stored leaf shape or dtype differs from descriptor')
    if hashlib.sha256(value.tobytes(order='C')).hexdigest() != descriptor['value_sha256']:
        raise ValueError('stored leaf bits differ from descriptor')
    if kind == 'scalar':return dict(bool=bool, int=int, float=float, complex=complex)[descriptor['python_type']](value.item())
    return value.copy()


def _leaf_keys(descriptor):
    if descriptor['kind'] in ('array', 'scalar'):return [descriptor['array']]
    return [key for child in descriptor.get('children', ()) for key in _leaf_keys(child)]


def save_snapshot(root, name, tree, *, source):
    """Write every leaf and seal its complete typed descriptor; never overwrite."""
    root = Path(root).resolve();root.mkdir(parents=True, exist_ok=True)
    name = _name(name);source = _source(source)
    data_path, metadata_path = root / (name + '.npz'), root / (name + '.json')
    if data_path.exists() or metadata_path.exists():raise FileExistsError('snapshot already exists')
    arrays = {};descriptor = _tree(tree, arrays)
    with data_path.open('xb') as stream:np.savez_compressed(stream, **arrays)
    metadata = dict(schema=SCHEMA, kind='snapshot', source=source, tree=descriptor,
                    arrays=_reference(root, data_path))
    with metadata_path.open('xb') as stream:stream.write(_canonical(metadata))
    return _reference(root, metadata_path)


def load_snapshot(root, reference, template, *, expected_source):
    """Load all fields against the actual runtime template, with no zero fills."""
    metadata = json.loads(verify_artifact(root, reference))
    if metadata.get('schema') != SCHEMA or metadata.get('kind') != 'snapshot':raise ValueError('unsupported snapshot schema')
    if metadata.get('source') != _source(expected_source):raise ValueError('snapshot source binding differs')
    if _structure(metadata['tree']) != _tree(template):raise ValueError('runtime template differs from complete snapshot tree')
    data = verify_artifact(root, metadata['arrays'])
    keys = _leaf_keys(metadata['tree'])
    if len(keys) != len(set(keys)):raise ValueError('duplicate snapshot leaf address')
    with np.load(io.BytesIO(data), allow_pickle=False) as archive:
        if set(archive.files) != set(keys):raise ValueError('missing or extra snapshot leaves')
        arrays = {key: archive[key] for key in keys}
    return _restore(template, metadata['tree'], arrays)


def seal_manifest(root, name, payload, *, parents, source):
    """Seal a new DAG node only after every referenced parent verifies."""
    root = Path(root).resolve();root.mkdir(parents=True, exist_ok=True)
    source = _source(source);name = _name(name)
    parents = list(parents)
    for parent in parents:verify_artifact(root, parent)
    value = dict(schema=SCHEMA, kind='manifest', source=source, parents=parents, payload=payload)
    path = root / (name + '.json')
    with path.open('xb') as stream:stream.write(_canonical(value))
    return _reference(root, path)
