"""Owner-only device packets for the existing v4 retained writers.

Only an original, registered actual request-source snapshot authorizes this
boundary. Engine, public semantic and checkpoint vocabulary IDs remain separate.
No request strings, host device reads or additional feature writer are used.
This partial source path does not establish full encoder or B4 qualification.
"""
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import NamedTuple
import weakref

import jax
import jax.numpy as jnp
import numpy as np

from . import v4_encoder, private_team, engine_private_team, engine_private_actions
from . import public_sleep, public_expected_stats, public_hazard_exposure, public_exact

SCHEMA = 'pokezero.engine_own_facts.v1'
SOURCE, OWNER, ORDER, PRIVATE = 1, 2, 4, 8
REASONS = {SOURCE:'actual-request-source-refusal', OWNER:'invalid-owner',
    ORDER:'request-order-or-engine-namespace', PRIVATE:'unsupported-own-semantic-or-status-facts'}


class Tables(NamedTuple):
    engine_binding: object
    binding: object
    outer_binding: object
    vocabulary_binding: object
    species_binding: object
    ability_binding: object
    action_binding: object
    species: object
    species_supported: object
    ability: object
    ability_supported: object
    status: object
    levitate: object
    source_stat_keys: object


@dataclass(frozen=True, eq=False)
class Compiled:
    device: Tables


class Result(NamedTuple):
    facts: v4_encoder.OwnFacts
    admitted: object
    reason: object


_COMPILED = weakref.WeakKeyDictionary()


def compile_own_facts(request_compiled, catalog):
    """Bind the actual reviewed request producer to a genuine v4 catalog."""
    from . import engine_request_source as source
    request = source._checked(request_compiled)
    v4_encoder.check_catalog(catalog)
    metadata = json.loads(request['metadata'])
    private = engine_private_team._checked_tables(request['private_team'])
    # These are the pinned Gen3 getSwitchRequestData exposures, not caller flags.
    if int(private.ability_mode) != 1 or not all(bool(x) for x in
            (private.stats_visible, private.item_visible, private.gender_visible)):
        raise ValueError('actual Gen3 baseAbility/stats/item/gender source required')
    # The source adapter's private-team binding incorporates the checkpoint.
    expected_private = engine_private_team.compile_engine_private_team(metadata,
        vocabulary=catalog.vocabulary, dex=catalog.dex,
        checkpoint_sha256=catalog.checkpoint_sha256,
        exposure=engine_private_team.RequestExposure('base','physical',True,True))
    if not np.array_equal(np.asarray(private.binding), np.asarray(expected_private.binding)):
        raise ValueError('request/encoder checkpoint, vocabulary or Dex differs')
    action = request['catalog']
    if action.names != catalog.actions.names:
        raise ValueError('request/encoder action namespace differs')
    for left, right in zip(action.device, catalog.actions.device):
        if not np.array_equal(np.asarray(left), np.asarray(right)):
            raise ValueError('request/encoder action tables differ')
    vb = private_team.vocabulary_binding(catalog.vocabulary)
    def equal(a, b):
        if not np.array_equal(np.asarray(a), np.asarray(b)):
            raise ValueError('own-fact static namespace differs')
    equal(request['lifecycle'].public_moves.device.binding, catalog.moves.device.binding)
    equal(request['lifecycle'].public_moves.device.source_binding, catalog.source.device.binding)
    for tables in (catalog.species.device, catalog.expected.device, catalog.hazard.device,
                   catalog.exact.device, catalog.sleep):
        equal(tables.vocabulary_binding, vb)
        equal(tables.species_binding if hasattr(tables,'species_binding') else tables.binding,
              catalog.species.device.binding)
    equal(catalog.exact.device.ability_binding, catalog.abilities.device.binding)
    equal(catalog.species.device.ability_binding, catalog.abilities.device.binding)
    if catalog.expected.names != catalog.species.names or dict(catalog.expected.ids) != dict(catalog.species.ids):
        raise ValueError('own target and status species identities differ')
    if catalog.hazard.species is not catalog.species or catalog.exact.abilities is not catalog.abilities:
        raise ValueError('original public semantic catalogs required')
    names = metadata['vocabulary']
    species = [catalog.species.ids.get(name, 0) if i else 0 for i,name in enumerate(names['species'])]
    abilities = [catalog.abilities.ids.get(name, 0) if i else 0 for i,name in enumerate(names['abilities'])]
    binding_payload = dict(schema=SCHEMA, request=np.asarray(request_compiled.binding).tolist(),
        producer_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        encoder=np.asarray(catalog.binding).tolist(),species=species,abilities=abilities,
        source_stats_keys=['atk','def','spa','spd','spe'],source_ability='baseAbility',
        source_faint_condition='0 fnt',decoded_target_hp='alive-condition-denominator')
    digest = hashlib.sha256(json.dumps(binding_payload,sort_keys=True,separators=(',',':')).encode()).digest()
    tables = Tables(private.engine_binding,jnp.asarray(np.frombuffer(digest,dtype='>u4').astype(np.uint32)),
        catalog.binding,vb,catalog.species.device.binding,catalog.abilities.device.binding,catalog.actions.device.binding,
        jnp.asarray(species,jnp.int32),jnp.asarray([bool(i) for i in species],jnp.bool_),
        jnp.asarray(abilities,jnp.int32),jnp.asarray([bool(i) for i in abilities],jnp.bool_),
        jnp.asarray([0,2,3,5,6,1,4],jnp.int32),
        jnp.asarray(catalog.abilities.ids.get('levitate',0),jnp.int32),
        jnp.array([False,True,True,True,True,True]))
    compiled = Compiled(tables)
    source_fields = tuple((key, request[key], v4_encoder._static_snapshot(request[key],set()))
        for key in ('rules','adapter','private_team','catalog','engine_table','lifecycle','metadata'))
    _COMPILED[compiled] = dict(tables=tables,edges=tuple(tables),catalog=catalog,
        source=source,request_compiled=request_compiled,request_binding=request_compiled.binding,
        request=request,source_fields=source_fields,
        fast=jax.jit(lambda state,orders,requests,owners: _device(state,orders,requests,owners,tables)),
        encode=jax.jit(lambda state,own,owners,streams:_encode(state,own,owners,streams,catalog)))
    return compiled


def _checked(compiled):
    record = _COMPILED.get(compiled) if type(compiled) is Compiled else None
    if (record is None or compiled.device is not record['tables']
            or any(a is not b for a,b in zip(compiled.device,record['edges']))):
        raise ValueError('original own-fact compiler holder and immutable edges required')
    engine_private_actions._immutable_arrays(compiled.device,'own-fact static tables')
    v4_encoder.check_catalog(record['catalog'])
    request = record['request']
    if record['source']._COMPILED.get(record['request_compiled']) is not request:
        raise ValueError('original actual request source lifetime required')
    if (record['request_compiled'].binding is not record['request_binding']
            or request['binding'] is not record['request_binding']):
        raise ValueError('original actual request source binding required')
    for key, original, snapshot in record['source_fields']:
        if request[key] is not original:
            raise ValueError('original actual request source edges required')
        v4_encoder._check_static(request[key],snapshot)
    return record


def _pick(values, owners):
    rows = jnp.arange(owners.shape[0]);seat = jnp.clip(owners,0,1)
    return jax.tree.map(lambda a,b:jnp.stack((a,b),axis=1)[rows,seat],*values)


def _device(state, orders, requests, owners, tables):
    """Trusted closed-over kernel: select owner BEFORE private validation."""
    b = owners.shape[0];rows = jnp.arange(b);seat = jnp.clip(owners,0,1)
    # All physical values are reduced to the selected owner before any checks.
    species,stats,hp,max_hp,status,base_ability,physical_active = (
        getattr(state,key)[rows,seat] for key in
        ('species','stats','hp','max_hp','status','base_ability','active'))
    ledger = _pick(orders,owners)
    # Only these source leaves have a battle axis and belong in OwnFacts.
    # Shared writer masks and internal server dispatch data are not inputs.
    facts, source_admitted, source_reason = _pick(tuple(
        (request.actions.facts,request.actions.admitted,request.reason) for request in requests),owners)
    mapping = ledger.request_to_physical
    safe = jnp.clip(mapping,0,5)
    gather = lambda a:jnp.take_along_axis(a,safe[(...,)+(None,)*(a.ndim-2)],axis=1)
    species,stats,hp,max_hp,status,base_ability = map(gather,(species,stats,hp,max_hp,status,base_ability))
    present = mapping>=0
    active = present&(mapping==physical_active[:,None])
    fainted = present&(hp==0)
    ss = jnp.clip(species,0,tables.species.shape[0]-1)
    aa = jnp.clip(base_ability,0,tables.ability.shape[0]-1)
    semantic_species = jnp.where(present,tables.species[ss],0)
    semantic_ability = jnp.where(present,tables.ability[aa],0)
    # Source literal 0 fnt omits prior status. Raw stats omits HP, but the
    # retained Python decoder derives target HP from an alive condition. Match
    # that decoded leaf while source_stat_keys separately preserves raw keys.
    source_status = jnp.where(present&~fainted,tables.status[jnp.clip(status,0,6)],0)
    target_stats = jnp.where(present[...,None],stats,0).astype(jnp.float32)
    target_stats = target_stats.at[...,0].set(jnp.where(present&~fainted,max_hp,0))
    known = present[...,None]&tables.source_stat_keys
    known = known.at[...,0].set(present&~fainted)
    owner_ok = (owners>=0)&(owners<=1)
    order_ok = (ledger.owner_side==owners)&ledger.admitted
    order_ok &= jnp.all(ledger.engine_binding==tables.engine_binding,axis=-1)
    order_ok &= jnp.all((mapping>=-1)&(mapping<6),axis=-1)
    order_ok &= jnp.all((~present)|(species>0),axis=-1)
    order_ok &= jnp.all(facts.team.active==active,axis=-1)
    source_ok = source_admitted&(source_reason==0)
    source_ok &= facts.team.owner_side==owners
    source_ok &= facts.request.owner_side==owners
    source_ok &= jnp.all(facts.team.vocabulary_binding==tables.vocabulary_binding,axis=-1)
    source_ok &= jnp.all(facts.catalog_binding==tables.action_binding,axis=-1)
    duplicate=(semantic_species[:,:,None]==semantic_species[:,None,:])&present[:,:,None]&present[:,None,:]&~jnp.eye(6,dtype=bool)
    valid = ((species>0)&(species<tables.species.shape[0])&tables.species_supported[ss]
        &(base_ability>0)&(base_ability<tables.ability.shape[0])&tables.ability_supported[aa]
        &(hp>=0)&((hp==0)|((status>=0)&(status<=6)))&jnp.all(stats[...,1:]>=0,axis=-1))
    private_ok = jnp.all(~present|valid,axis=-1)&~jnp.any(duplicate,axis=(-1,-2))
    ok=owner_ok&order_ok&source_ok&private_ok
    reason=(jnp.where(owner_ok,0,OWNER)|jnp.where(order_ok,0,ORDER)
        |jnp.where(source_ok,0,SOURCE)|jnp.where(private_ok,0,PRIVATE)).astype(jnp.int32)
    words=lambda x:jnp.broadcast_to(x,(b,8))
    own=v4_encoder.OwnFacts(facts,
        public_sleep.OwnStatusFacts(owners,words(tables.vocabulary_binding),words(tables.species_binding),
            semantic_species,present,source_status,active),
        public_expected_stats.OwnTargets(owners,words(tables.species_binding),words(tables.vocabulary_binding),
            semantic_species,target_stats,known),
        public_hazard_exposure.OwnHazardFacts(owners,words(tables.species_binding),words(tables.vocabulary_binding),
            semantic_species,present,active,fainted,present&(semantic_ability==tables.levitate)&(tables.levitate>0)),
        public_exact.OwnAbilities(owners,words(tables.ability_binding),words(tables.vocabulary_binding),
            present,semantic_ability),words(tables.outer_binding))
    # Zero every leaf including bindings on refusal. Even an ignored admission
    # result cannot turn this packet into admitted work in retained writers.
    own=jax.tree.map(lambda x:jnp.where(ok.reshape((b,)+(1,)*(x.ndim-1)),x,jnp.zeros_like(x)),own)
    return Result(own,ok,reason)


def produce_own_facts(constructed, compiled, *, owner_sides):
    """Actual registered constructor/request snapshot -> v4_encoder.OwnFacts."""
    record=_checked(compiled)
    snapshot=record['source']._constructed(constructed)
    b=snapshot.state.species.shape[0]
    engine_private_actions._shape(owner_sides,(b,),jnp.int32,'own-fact owner sides')
    return record['fast'](snapshot.state,snapshot.orders,snapshot.requests,owner_sides)


def encode_source(state, constructed, compiled, *, owner_sides):
    """Invoke the existing retained encoder with this actual source packet."""
    own=produce_own_facts(constructed,compiled,owner_sides=owner_sides)
    engine_private_actions._shape(state.cursor.stream_id,owner_sides.shape,jnp.int32,'own/public streams')
    return _checked(compiled)['encode'](state,own.facts,owner_sides,constructed.registry.stream_id)


def _encode(state, facts, owners, streams, catalog):
    # The existing OwnFacts format carries no battle ID. The actual callsite
    # must join registered stream identities before exposing private rows to a
    # public encoder state. A mismatched row receives no admitted observation.
    same=state.cursor.stream_id==streams
    facts=facts._replace(binding=jnp.where(same[:,None],facts.binding,jnp.zeros_like(facts.binding)))
    return v4_encoder.encode(state,facts,catalog,observer_sides=owners)
