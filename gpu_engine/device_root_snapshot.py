"""Primitive-only transport of declared offline PRE roots; no model execution."""
import hashlib
import io
import json
from pathlib import Path
import numpy as np
import jax
import jax.numpy as j
from . import state, engine_device_trackers as trackers, engine_device_observation as observation
from . import engine_device_damage_history as damage, engine_device_negative_items as negative
from . import engine_device_visible_belief as visible_belief
from . import engine_device_public_port as public_port
from .v5_device_rollout import RootContext

SCHEMA = 'pokezero.device_heldout_pre_root.v1'
SOURCE_FIELDS = ('observations', 'requested', 'policy_logits', 'policy_finite', 'actions')
_TYPES = {c.__module__+'.'+c.__name__: c for c in
    (RootContext, state.BattleState, trackers.Trackers, trackers.Tendency,
     damage.History, negative.State, visible_belief.State, public_port.State, observation.Observation)}


def _json(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


def save(path, root, source_fields, *, namespace, selector):
    """Write one actual selected root exclusively, including both real views."""
    if type(root) is not RootContext or set(source_fields) != set(SOURCE_FIELDS):
        raise ValueError('actual PRE root and exact source fields required')
    arrays = {}
    def pack(value):
        if value is None: return None
        if hasattr(value, '_fields'):
            name = type(value).__module__+'.'+type(value).__name__
            if name not in _TYPES: raise ValueError('unknown typed root carrier')
            return dict(type=name, fields=list(value._fields), values=[pack(x) for x in value])
        if type(value) is dict:
            if any(type(k) is not str for k in value): raise ValueError('literal root field keys required')
            return dict(mapping=[[k, pack(v)] for k, v in value.items()])
        if type(value) is tuple: return dict(sequence=[pack(v) for v in value])
        a = np.asarray(value)
        if a.dtype not in (np.dtype('bool'), np.dtype('int32'), np.dtype('uint32'), np.dtype('float32')):
            raise ValueError('primitive device root array required')
        key = 'array_'+str(len(arrays)); arrays[key] = a
        return dict(array=key)
    body = dict(schema=SCHEMA, namespace=namespace, selector=selector,
        root=pack(root), source_fields=pack(source_fields))
    arrays['metadata_json'] = np.frombuffer(_json(body), np.uint8)
    buffer = io.BytesIO(); np.savez_compressed(buffer, **arrays); blob = buffer.getvalue()
    with Path(path).open('xb') as stream: stream.write(blob)
    return dict(path=str(Path(path).resolve()), sha256=hashlib.sha256(blob).hexdigest())


def load(entry, index):
    """Return typed root plus literal source fields; never rebuild missing history."""
    if entry['status'] != 'SAVED' or entry['artifact'] is None:
        raise ValueError('declared saved root unavailable: '+str(entry['reason']))
    blob = Path(entry['artifact']['path']).read_bytes()
    if hashlib.sha256(blob).hexdigest() != entry['artifact']['sha256']:
        raise ValueError('saved root bytes differ')
    with np.load(io.BytesIO(blob), allow_pickle=False) as archive:
        body = json.loads(archive['metadata_json'].tobytes())
        if (body['schema'] != SCHEMA or body['namespace'] != index['namespace']
                or body['selector'] != entry['selector']):
            raise ValueError('saved root source/address differs')
        used = {'metadata_json'}
        def unpack(value):
            if value is None: return None
            if set(value) == {'array'}:
                key = value['array']; used.add(key); a=archive[key]
                if a.dtype not in (np.dtype('bool'),np.dtype('int32'),np.dtype('uint32'),np.dtype('float32')):
                    raise ValueError('primitive saved root array required')
                return j.asarray(a)
            if set(value) == {'mapping'}:
                rows = value['mapping']
                if len({k for k, _ in rows}) != len(rows): raise ValueError('duplicate root key')
                return {k:unpack(v) for k, v in rows}
            if set(value) == {'sequence'}: return tuple(unpack(v) for v in value['sequence'])
            if set(value) != {'type', 'fields', 'values'} or value['type'] not in _TYPES:
                raise ValueError('unknown saved root structure')
            cls = _TYPES[value['type']]
            if value['fields'] != list(cls._fields) or len(value['values']) != len(cls._fields):
                raise ValueError('saved root schema changed; no zero-filled missing history')
            return cls(*(unpack(v) for v in value['values']))
        root = unpack(body['root']); fields = unpack(body['source_fields'])
        if used != set(archive.files): raise ValueError('extra root arrays')
    if (type(root) is not RootContext or set(fields) != set(SOURCE_FIELDS)
            or root.streams.shape != (1,) or root.frames.shape != (1,)
            or root.streams.dtype != j.int32 or root.frames.dtype != j.int32
            or int(root.streams[0]) != entry['selector']['stream']
            or int(root.frames[0]) != entry['selector']['frame']):
        raise ValueError('actual saved PRE root address differs')
    if any(a.ndim<1 or a.shape[0] != 1 for a in jax.tree.leaves(root)):
        raise ValueError('one actual root row required')
    for name,shape,dtype in (('requested',(1,2),j.bool_),('policy_logits',(1,2,9),j.float32),
            ('policy_finite',(1,2),j.bool_),('actions',(1,2),j.int32)):
        if getattr(fields[name],'shape',None)!=shape or fields[name].dtype!=j.dtype(dtype):
            raise ValueError('actual both-seat source field differs: '+name)
    if type(fields['observations']) is not tuple or len(fields['observations'])!=2:
        raise ValueError('both actual source views required')
    for view in fields['observations']:
        if type(view) is not observation.Observation:raise ValueError('actual typed source view required')
        for name,shape,dtype in (('numeric_features',(1,29,132),j.float32),('categorical_ids',(1,29,41),j.int32),
                ('token_type_ids',(1,29),j.int32),('attention_mask',(1,29),j.bool_),('legal_action_mask',(1,9),j.bool_)):
            if getattr(getattr(view,name),'shape',None)!=shape or getattr(view,name).dtype!=j.dtype(dtype):
                raise ValueError('actual public source view differs: '+name)
    return root, dict(fields, namespace=body['namespace'])
