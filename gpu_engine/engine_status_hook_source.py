"""Verified effective Gen3 callback identity inventory, NOT hook admission.

The ORIGINAL SourceRules loader owns root/node/source provenance. Compilation
rechecks its pinned Dex and inventories every effective on* callback (including
nested effects), exact types/body hashes and declared ordering metadata. Device
classes describe static ABSENCE over that finite vocabulary, never function-text
safety. Item0 is explicitly absent; an unknown ID is never absence.

NO_RELEVANT_HOOKS means only absence of the listed dispatch keys. It does not
prove absence of active volatile/side/field/slot carriers, dynamic duration or
custom Battle.events, nor constructor legality, hook history or owning lifetime.
Those obligations remain pending. Leftovers is only a transport CANDIDATE; even its exact source callback
identity requires actual original helper/span/both-owner drains and a separately
owned sticky hook-history record. Lum/update/cure histories are not qualified.
No admission, full encoder, B4 or caller-provided proof boolean API exists here.
"""
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import subprocess
from typing import NamedTuple
import weakref

import jax
import jax.numpy as jnp
import numpy as np

from . import engine_request_source as source, engine_status_residual_callbacks as status

SCHEMA = 'pokezero.gen3-status-hook-source.static.v1'
UNSUPPORTED, ABSENT, NO_RELEVANT_HOOKS, RELEVANT_UNREVIEWED, LEFTOVERS_TRANSPORT_CANDIDATE = range(5)
FAMILIES = ('Damage','Residual','Update','SetStatus','AfterSetStatus',
    'UseItem','TryEatItem','Eat','EatItem','AfterUseItem')
DAMAGE, RESIDUAL, UPDATE, SET_STATUS, AFTER_SET_STATUS, USE_ITEM, TRY_EAT_ITEM, EAT, EAT_ITEM, AFTER_USE_ITEM = (1<<n for n in range(10))
KINDS = ('abilities','items','species','moves')
DISPATCH = {name: (['onResidual','onSideResidual','onFieldResidual'] if name=='Residual' else
    ['on'+name] if name in ('Update','Eat') else
    ['on'+prefix+name for prefix in ('','Ally','Foe','Any','Source')]) for name in FAMILIES}
_COMPILED = weakref.WeakKeyDictionary()


class Rows(NamedTuple):
    manifest_known: object
    relevant_flags: object
    classification: object
    row_digest: object


class Data(NamedTuple):
    binding: object
    engine_binding: object
    rules_binding: object
    status_binding: object
    manifest_binding: object
    vocabulary_binding: object
    abilities: Rows
    items: Rows
    species: Rows
    moves: Rows


@dataclass(frozen=True, eq=False)
class Compiled:
    device: Data


def _host():
    if isinstance(jnp.zeros((),jnp.int32),jax.core.Tracer):
        raise ValueError('checked hook source host boundary cannot be traced')


def _json(value):
    return json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False)


def _digest(value):
    return hashlib.sha256(_json(value).encode()).hexdigest()


def _words(hex_digest):
    return jnp.asarray(np.frombuffer(bytes.fromhex(hex_digest),dtype='>u4').astype(np.uint32))


def _script():
    return Path(__file__).with_name('status_hook_source.cjs')


def _validate(payload,metadata,rules,origin):
    if (payload.get('schema')!=SCHEMA or payload.get('revision')!=source.SHOWDOWN
            or payload.get('format')!='gen3randombattle' or payload.get('dispatch')!=DISPATCH
            or payload.get('vocabulary')!=metadata['vocabulary']
            or payload.get('source_tree')!=json.loads(origin.source_tree_json)
            or payload.get('compiled_tree')!=json.loads(origin.compiled_tree_json)
            or payload.get('request_generator_sha256')!=origin.generator_sha256
            or payload.get('generator_sha256')!=hashlib.sha256(_script().read_bytes()).hexdigest()
            or payload.get('metadata_sha256')!=_digest(metadata)
            or payload.get('vocabulary_sha256')!=_digest(metadata['vocabulary'])
            or payload.get('rules_sha256')!=_digest(json.loads(source._rules(rules)[2]))
            or payload.get('scope')!='STATIC_IDENTITY_ONLY_NO_OWNING_HISTORY_NO_ADMISSION'):
        raise ValueError('exact verified source/descriptor/vocabulary manifest required')
    for kind in KINDS:
        rows=payload.get('rows',{}).get(kind,[])
        if len(rows)!=len(metadata['vocabulary'][kind]):
            raise ValueError('complete original callback vocabulary required')
        for i,(row,name) in enumerate(zip(rows,metadata['vocabulary'][kind])):
            if row.get('requested')!=name or row.get('index')!=i or row.get('kind')!=kind:
                raise ValueError('exact callback row namespace required')
            if i==0 and not (name=='_none' and row.get('absent') is True and row.get('id')==''):
                raise ValueError('explicit absent row required')
    if not isinstance(payload.get('conditions'),list) or not isinstance(payload.get('rules'),list):
        raise ValueError('static condition/format carrier inventory required')
    for row in [r for rows in payload['rows'].values() for r in rows]+payload['conditions']+payload['rules']+[payload.get('format_row',{})]:
        if not isinstance(row.get('entries'),list) or not isinstance(row.get('carriers'),list):
            raise ValueError('complete callback key inventory required')
        for entry in row['entries']+row['carriers']:
            if not isinstance(entry.get('path'),str) or len(entry.get('sha256',''))!=64:
                raise ValueError('callback identity hash required')
            try: bytes.fromhex(entry['sha256'])
            except ValueError: raise ValueError('callback identity hash required') from None
    return payload


def _rows(rows,kind):
    flags=[];classes=[];known=[];digests=[]
    for row in rows:
        ok=row.get('known') is True
        value=row.get('flags')
        if type(value) is not int or not 0<=value<1<<len(FAMILIES):ok=False
        computed=0
        for entry in row['entries']:
            key=entry.get('key');typ=entry.get('type')
            meta=entry.get('metadata')
            if (not isinstance(key,str) or type(meta) is not bool or
                    meta!=key.endswith(('Order','Priority','SubOrder'))):ok=False
            if meta:
                if typ not in ('undefined','number','boolean'):ok=False
            else:
                if typ not in ('undefined','function','number','boolean'):ok=False
                if any(k not in entry for k in ('Order','Priority','SubOrder')):ok=False
                if typ!='undefined':
                    for bit,keys in enumerate(DISPATCH.values()):
                        if key in keys:computed|=1<<bit
        if row['carriers']:computed|=RESIDUAL
        if computed!=value:ok=False
        # Unknown values remain present and unsupported; never infer absence.
        cls=UNSUPPORTED
        if ok:
            if row['absent']:cls=ABSENT
            elif not value:cls=NO_RELEVANT_HOOKS
            else:cls=RELEVANT_UNREVIEWED
            if (kind=='items' and row['id']=='leftovers' and value==RESIDUAL
                    and [x['key'] for x in row['entries'] if not x['metadata']
                         and x['type']!='undefined' and any(x['key'] in keys for keys in DISPATCH.values())]==['onResidual']
                    and any(x['key']=='onResidual' and x['type']=='function' and x['Order']==10 and x['SubOrder']==4
                            for x in row['entries'])):
                cls=LEFTOVERS_TRANSPORT_CANDIDATE
        known.append(ok);flags.append(value if ok else -1);classes.append(cls);digests.append(_words(_digest(row)))
    return Rows(jnp.array(known,jnp.bool_),jnp.array(flags,jnp.int32),
        jnp.array(classes,jnp.int32),jnp.stack(digests))


def compile_hook_source(rules,status_compiled,*,engine_catalog):
    """Host compilation from ORIGINAL loader origin and original status metadata."""
    _host()
    origin=source._rules_origin(rules)
    helper=status.check_compiled(status_compiled)
    record=status._COMPILED[status_compiled]
    metadata=record[2]
    if record[4] is not engine_catalog or not np.array_equal(
            np.asarray(rules.device.engine_binding),np.asarray(helper.engine_binding)):
        raise ValueError('same original status/catalog/source namespaces required')
    result=subprocess.run([origin.node,str(_script()),origin.root],input=_json(metadata),
        text=True,capture_output=True,check=True)
    payload=_validate(json.loads(result.stdout),metadata,rules,origin)
    manifest=_json(payload)
    binding=_digest(dict(schema=SCHEMA,manifest=hashlib.sha256(manifest.encode()).hexdigest(),
        status=np.asarray(helper.binding).tolist(),rules=np.asarray(rules.device.binding).tolist(),
        python=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()))
    device=Data(_words(binding),helper.engine_binding,rules.device.binding,helper.binding,
        _words(hashlib.sha256(manifest.encode()).hexdigest()),_words(payload['vocabulary_sha256']),
        *(_rows(payload['rows'][kind],kind) for kind in KINDS))
    compiled=Compiled(device)
    # Never retain the weak key holder in its own record.
    _COMPILED[compiled]=(device,tuple(jax.tree.leaves(device)),rules,status_compiled,
        engine_catalog,manifest,origin,hashlib.sha256(_script().read_bytes()).hexdigest())
    return compiled


def check_compiled(compiled):
    """Fresh original-holder/immutable-edge guards outside every owning JIT call."""
    _host()
    record=_COMPILED.get(compiled) if type(compiled) is Compiled else None
    if record is None or compiled.device is not record[0]:
        raise ValueError('original immutable hook source compiler required')
    leaves=tuple(jax.tree.leaves(compiled.device))
    if (any(type(getattr(compiled.device,kind)) is not Rows for kind in KINDS)
            or len(leaves)!=len(record[1]) or any(a is not b for a,b in zip(leaves,record[1]))
            or any(not isinstance(x,jax.Array) or isinstance(x,jax.core.Tracer) for x in leaves)
            or hashlib.sha256(_script().read_bytes()).hexdigest()!=record[7]):
        raise ValueError('original immutable hook source table edges required')
    if source._rules_origin(record[2]) is not record[6]:
        raise ValueError('original verified source origin required')
    status.check_compiled(record[3])
    if status._COMPILED[record[3]][4] is not record[4]:
        raise ValueError('original engine catalog required')
    return compiled.device


def manifest_json(compiled):
    """Canonical static evidence only; no runtime/owning-history certificate."""
    _host()
    check_compiled(compiled)
    return _COMPILED[compiled][5]
