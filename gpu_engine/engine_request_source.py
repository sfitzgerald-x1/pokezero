"""Trusted-server Gen3 own-request producer, with a real constructor callsite.

Server truth is used only inside source request construction. In particular,
physical ability trapping is not a request mask: the pinned source distinguishes
visible traps, hidden traps and possible public-species abilities. Only the
resulting own request fields enter the existing private action writer.

This remains a partial source path, not observation admission or B4 qualification.
Unrepresented locks, changed constructor data and missing public arrival/gender
provenance refuse. No CLI enables this path implicitly.
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

from . import engine_private_actions as actions
from . import engine_private_team as team
from . import engine_public_events as public
from . import engine_request_order as order_api
from . import private_actions
from . import source_provenance as provenance
from . import transition
from .state import BattleState, ROOTED

SCHEMA = 'pokezero.engine_request_source.v1'
SHOWDOWN = 'f76228a1354b5d0f307ca2d16101294ad3a2308b'
UNBOUND, PHASE, ARRIVAL, GENDER, CHANGED, LOCK, DISABLE, NAMESPACE = (1 << n for n in range(8))
TRAP_ORIGIN = 256
ADAPTER = 512
REASONS = {UNBOUND:'unbound-constructor', PHASE:'unproved-request-phase',
    ARRIVAL:'unproved-public-arrival', GENDER:'unresolved-constructor-gender',
    CHANGED:'changed-constructor-origin', LOCK:'unrepresented-lock-or-recharge-origin',
    DISABLE:'unrepresented-disable-visibility', NAMESPACE:'unsupported-source-namespace',
    TRAP_ORIGIN:'unrepresented-anonymous-trap-origin', ADAPTER:'existing-private-adapter-refusal'}


def _words(payload):
    raw = json.dumps(payload, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()
    return jnp.asarray(np.frombuffer(hashlib.sha256(raw).digest(), dtype='>u4').astype(np.uint32))


class RulesData(NamedTuple):
    binding: object
    engine_binding: object
    species_supported: object
    species_types: object
    possible_arena: object
    possible_magnet: object
    possible_shadow: object
    ability_supported: object
    arena: object
    magnet: object
    shadow: object
    levitate: object
    type_immune: object
    flying: object
    steel: object
    unsupported_history: object
    choice_band: object


@dataclass(frozen=True, eq=False)
class SourceRules:
    device: RulesData


_RULES = weakref.WeakKeyDictionary()
_RULE_ORIGINS = weakref.WeakKeyDictionary()
_COMPILED = weakref.WeakKeyDictionary()
_REGISTRATIONS = weakref.WeakKeyDictionary()
_CONSTRUCTED = weakref.WeakKeyDictionary()


def _compile_rules(metadata, payload):
    """Trusted loader primitive; public callers use load_source_rules."""
    if (payload.get('schema') != 'pokezero.gen3-request-source-rules.v1'
            or payload.get('revision') != SHOWDOWN or payload.get('format') != 'gen3randombattle'
            or payload.get('vocabulary') != metadata['vocabulary']
            or payload.get('compiled_tree') != metadata.get('compiled_tree')
            or payload.get('source_tree',{}).get('revision') != SHOWDOWN
            or len(payload.get('source_tree',{}).get('sha256','')) != 64
            or payload.get('constructor_known_type') is not True
            or payload.get('direct_simulator_request_id') != -1):
        raise ValueError('exact pinned request source rules required')
    sp, ab, ty, mv = (payload[k] for k in ('species','abilities','types','moves'))
    for rows, kind in ((sp,'species'),(ab,'abilities'),(ty,'types'),(mv,'moves')):
        if len(rows) != len(metadata['vocabulary'][kind]):
            raise ValueError('source rule namespace differs: '+kind)
    flag = lambda rows, key:jnp.asarray([row[key] for row in rows], jnp.bool_)
    possible = lambda ability:jnp.asarray([
        ability in row['alternate_maybe_trap_abilities'] for row in sp], jnp.bool_)
    type_ids = {name:i for i,name in enumerate(metadata['vocabulary']['types'])}
    species_types = [[type_ids.get(row['types'][i],-1) if i<len(row['types']) else 0
        for i in range(2)] for row in sp]
    device = RulesData(_words(payload), team.engine_metadata_binding(metadata),
        flag(sp,'supported'),jnp.asarray(species_types,jnp.int32),
        possible('arenatrap'), possible('magnetpull'), possible('shadowtag'),
        flag(ab,'supported'), *[flag(ab,key) for key in ('arena','magnet','shadow','levitate')],
        *[flag(ty,key) for key in ('immune','flying','steel')], flag(mv,'unsupported_history'),
        jnp.asarray([x == 'choiceband' for x in metadata['vocabulary']['items']],jnp.bool_))
    rules = SourceRules(device)
    _RULES[rules] = (device,tuple(device),json.dumps(payload,sort_keys=True))
    return rules


def load_source_rules(showdown_root, metadata, *, node='node'):
    """Read only the pinned static Dex; never construct a team or a Battle."""
    if isinstance(jnp.zeros((),jnp.int32),jax.core.Tracer):
        raise ValueError('verified source rule loader must remain outside outer JIT')
    script = Path(__file__).with_name('request_source_rules.cjs')
    result = subprocess.run([node,str(script),str(Path(showdown_root).resolve())],
        input=json.dumps(metadata,allow_nan=False),text=True,capture_output=True,check=True)
    payload = json.loads(result.stdout)
    if payload.get('generator_sha256') != hashlib.sha256(script.read_bytes()).hexdigest():
        raise ValueError('request rule generator changed')
    rules=_compile_rules(metadata,payload)
    # Only this successful pinned-source loader can mint an origin. Trusted
    # shape fixtures built through _compile_rules intentionally have none.
    origin=VerifiedRulesOrigin(str(Path(showdown_root).resolve()),str(node),
        json.dumps(payload['source_tree'],sort_keys=True),
        json.dumps(payload['compiled_tree'],sort_keys=True),payload['generator_sha256'])
    _RULE_ORIGINS[rules]=(origin,rules.device,_RULES[rules][2],
        (origin.root,origin.node,origin.source_tree_json,origin.compiled_tree_json,origin.generator_sha256))
    return rules


def _rules(rules):
    record = _RULES.get(rules) if type(rules) is SourceRules else None
    if (record is None or rules.device is not record[0]
            or any(a is not b for a,b in zip(rules.device,record[1]))):
        raise ValueError('original static source rules required')
    actions._immutable_arrays(rules.device,'source rules')
    return record


@dataclass(frozen=True,eq=False)
class VerifiedRulesOrigin:
    root: str
    node: str
    source_tree_json: str
    compiled_tree_json: str
    generator_sha256: str


def _rules_origin(rules):
    """Private original-loader edge, never a caller path or proof boolean."""
    if isinstance(jnp.zeros((),jnp.int32),jax.core.Tracer):
        raise ValueError('verified source rule origin must remain outside outer JIT')
    record=_rules(rules);origin=_RULE_ORIGINS.get(rules)
    if (origin is None or type(origin[0]) is not VerifiedRulesOrigin
            or origin[1] is not rules.device or origin[2]!=record[2]
            or (origin[0].root,origin[0].node,origin[0].source_tree_json,
                origin[0].compiled_tree_json,origin[0].generator_sha256)!=origin[3]
            or origin[0].generator_sha256!=hashlib.sha256(Path(__file__).with_name('request_source_rules.cjs').read_bytes()).hexdigest()):
        raise ValueError('original rules from verified pinned-source loader required')
    return origin[0]


@dataclass(frozen=True, eq=False)
class Compiled:
    binding: object


def compile_request_source(metadata, *, vocabulary, dex, checkpoint_sha256,
                           rules, lifecycle_catalog):
    """Own the actual source exposure contract; no caller-supplied request flags."""
    rule_record = _rules(rules)
    engine = team.engine_metadata_binding(metadata)
    if not np.array_equal(np.asarray(engine),np.asarray(rules.device.engine_binding)):
        raise ValueError('constructor/source policy namespace differs')
    public.check_catalog(lifecycle_catalog,lifecycle_catalog.engine_table,
        expected_binding=lifecycle_catalog.binding,expected_engine_binding=engine)
    private_team = team.compile_engine_private_team(metadata,vocabulary=vocabulary,dex=dex,
        checkpoint_sha256=checkpoint_sha256,exposure=team.RequestExposure('base','physical',True,True))
    catalog = private_actions.compile_actions(dex,vocabulary)
    producer_hash = hashlib.sha256(Path(__file__).read_bytes()+
        Path(__file__).with_name('request_source_rules.cjs').read_bytes()+rule_record[2].encode()).hexdigest()
    adapter = actions.compile_engine_actions(metadata,catalog=catalog,vocabulary=vocabulary,
        private_team=private_team,exposure=actions.Exposure(producer_hash,'retained-constructor'))
    binding = _words(dict(schema=SCHEMA,source=np.asarray(rules.device.binding).tolist(),
        adapter=np.asarray(adapter.device.binding).tolist(),producer=producer_hash,
        lifecycle=lifecycle_catalog.binding))
    compiled = Compiled(binding)
    _COMPILED[compiled] = dict(binding=binding,rules=rules,adapter=adapter,private_team=private_team,
        catalog=catalog,engine_table=lifecycle_catalog.engine_table,lifecycle=lifecycle_catalog,
        metadata=json.dumps(metadata,sort_keys=True),
        encoder=actions.prepare_engine_actions(adapter,expected_binding=adapter.device.binding),
        producer=jax.jit(_produce))
    return compiled


def _checked(compiled):
    record = _COMPILED.get(compiled) if type(compiled) is Compiled else None
    if record is None or compiled.binding is not record['binding']:
        raise ValueError('original compiled request producer required')
    _rules(record['rules']); actions._checked(record['adapter'])
    c = record['lifecycle']
    public.check_catalog(c,record['engine_table'],expected_binding=c.binding,
        expected_engine_binding=record['rules'].device.engine_binding)
    if isinstance(jnp.zeros((),jnp.int32),jax.core.Tracer):
        raise ValueError('checked request source host boundary cannot be traced')
    return record


class Original(NamedTuple):
    species: object
    level: object
    gender: object
    stats: object
    max_hp: object
    moves: object
    types: object
    hidden_power_type: object
    hidden_power_power: object
    base_ability: object


@dataclass(frozen=True, eq=False)
class Registration:
    constructor: actions.ConstructorPP
    original: Original
    unsafe_origin: object


def _registration(state, pool, indices, streams, owners):
    retained = actions.retain_constructor_pp(pool,indices,stream_ids=streams,owner_sides=owners)
    rows = jnp.arange(len(streams))
    original = Original(*(jnp.array(getattr(state,key)[rows,owners],copy=True) for key in Original._fields))
    registered = Registration(retained,original,jnp.zeros(len(streams),jnp.bool_))
    _REGISTRATIONS[registered] = (retained,original,tuple(original),registered.unsafe_origin)
    return registered


def _registered(value):
    record = _REGISTRATIONS.get(value) if type(value) is Registration else None
    if (record is None or value.constructor is not record[0] or value.original is not record[1]
            or any(a is not b for a,b in zip(value.original,record[2]))
            or value.unsafe_origin is not record[3]):
        raise ValueError('actual owning constructor callsite registration required')
    actions._constructor(value.constructor)
    actions._immutable_arrays(value.original,'original constructor fields')
    actions._immutable_arrays((value.unsafe_origin,),'source origin refusal')
    return record


class Produced(NamedTuple):
    # INTERNAL server packet: features alone are the policy action section.
    # In particular actions.engine_legal is a dispatch constraint, not a feature.
    actions: actions.Result
    reason: object


def _unsafe_history(state,rules):
    history=state.last_move
    valid=(history>=0)&(history<rules.unsupported_history.shape[0])
    return jnp.any((state.species>0)&(~valid|rules.unsupported_history[
        jnp.clip(history,0,rules.unsupported_history.shape[0]-1)]),axis=(1,2))


def _advance_registration(registration,state,rules):
    _registered(registration)
    unsafe=registration.unsafe_origin|_unsafe_history(state,rules)
    value=Registration(registration.constructor,registration.original,unsafe)
    _REGISTRATIONS[value]=(value.constructor,value.original,tuple(value.original),unsafe)
    return value


def _produce(state, ledger, arrival, original, constructor, unsafe_origin, rules, adapter_binding, engine_table):
    """Server-only primitive. Hidden foe cells never become feature fields."""
    b = ledger.owner_side.shape[0]
    rows = jnp.arange(b); own = jnp.clip(ledger.owner_side,0,1); foe = 1-own
    active = state.active[rows,own]; safe_active = jnp.clip(active,0,5)
    owned = {key:getattr(state,key)[rows,own] for key in Original._fields}
    same = jnp.ones(b,jnp.bool_)
    for key in Original._fields:
        value = owned[key]; expected = getattr(original,key)
        same &= jnp.all(value==expected,axis=tuple(range(1,value.ndim)))
    present = owned['species']>0
    safe_own_species = jnp.clip(owned['species'],0,rules.species_supported.shape[0]-1)
    own_species_ok = jnp.all(~present | ((owned['species']<rules.species_supported.shape[0])
        & rules.species_supported[safe_own_species]),axis=-1)
    type_origin = jnp.all(~present[...,None] | (original.types==rules.species_types[safe_own_species]),axis=(-1,-2))
    same &= type_origin
    gender_known = jnp.all(~present | ((owned['gender']>=0)&(owned['gender']<=2)),axis=-1)
    hp = state.hp[rows,own]
    active_hp = hp[rows,safe_active]
    can_switch = jnp.any((hp>0)&present&(jnp.arange(6)!=active[:,None]),axis=-1)
    all_active_hp = jnp.take_along_axis(state.hp,jnp.clip(state.active,0,5)[...,None],axis=2)[...,0]
    forced = state.forced_switch | (all_active_hp<=0)
    replacement = jnp.any(forced,axis=-1)
    own_forced = forced[rows,own]
    phase_ok = (~state.done & ~jnp.any(state.entry_pending,axis=-1)
        & jnp.all(jnp.any(state.hp>0,axis=-1),axis=-1)
        & (~own_forced | can_switch))
    mode = jnp.where(own_forced,actions.FORCE_SWITCH,
        jnp.where(replacement,actions.WAIT,actions.ORDINARY)).astype(jnp.int32)
    ordinary = mode==actions.ORDINARY
    arrival_ok = ((arrival.stream_id==ledger.stream_id)
        & jnp.all(arrival.engine_binding==ledger.engine_binding,axis=-1)
        & jnp.all(arrival.generation[rows,own,None]==ledger.generation[:,None],axis=-1)
        & jnp.all(arrival.generation>=0,axis=-1)
        & jnp.all(arrival.established,axis=-1)
        & jnp.all(arrival.physical_active==state.active,axis=-1)
        & jnp.all((state.active>=0)&(state.active<6),axis=-1)
        & jnp.all(arrival.public_slot>=0,axis=-1)&jnp.all(arrival.encounter>0,axis=-1))
    public_species = arrival.revealed_species_id[rows,foe]
    safe_species = jnp.clip(public_species,0,rules.species_supported.shape[0]-1)
    foe_active = jnp.clip(state.active[rows,foe],0,5)
    # A changed forme/Transform requires a genuine public update we do not have.
    arrival_ok &= public_species==state.species[rows,foe,foe_active]
    arrival_ok &= arrival.revealed_species_id[rows,own]==owned['species'][rows,safe_active]
    arrival_ok &= (public_species>0)&(public_species<rules.species_supported.shape[0])
    arrival_ok &= rules.species_supported[safe_species]
    moves = state.moves[rows,own,safe_active]
    pp = state.pp[rows,own,safe_active]
    slot_present = moves>0
    item = state.item[rows,own,safe_active]
    safe_item = jnp.clip(item,0,rules.choice_band.shape[0]-1)
    choice = state.choice_move[rows,own,safe_active]
    choice_active = ((item>=0)&(item<rules.choice_band.shape[0])&rules.choice_band[safe_item]
        & (choice>0)&jnp.any(moves==choice[:,None],axis=-1))
    disabled = slot_present & ((pp<=0)|(choice_active[:,None]&(moves!=choice[:,None])))
    source_disable_unknown = jnp.any(state.disabled[rows,own]&present[...,None],axis=(-1,-2))
    unsafe_history = unsafe_origin|_unsafe_history(state,rules)
    locked = (state.charge_move[rows,own,safe_active]!=0)|(state.encore_move[rows,own,safe_active]!=0)|(
        state.encore_counter[rows,own,safe_active]!=0)|unsafe_history
    own_ability = state.ability[rows,own,safe_active]
    foe_ability = state.ability[rows,foe,foe_active]
    good_ab = ((own_ability>0)&(own_ability<rules.ability_supported.shape[0])
        & (foe_ability>0)&(foe_ability<rules.ability_supported.shape[0]))
    oa = jnp.clip(own_ability,0,rules.ability_supported.shape[0]-1)
    fa = jnp.clip(foe_ability,0,rules.ability_supported.shape[0]-1)
    good_ab &= rules.ability_supported[oa]&rules.ability_supported[fa]
    types = state.types[rows,own,safe_active]
    good_type = jnp.all((types>=0)&(types<rules.type_immune.shape[0]),axis=-1)
    st = jnp.clip(types,0,rules.type_immune.shape[0]-1)
    immune = jnp.any(rules.type_immune[st],axis=-1)
    rooted = (state.volatile[rows,own,safe_active]&ROOTED)!=0
    grounded = rooted | (~jnp.any(rules.flying[st],axis=-1)&~rules.levitate[oa])
    steel = jnp.any(rules.steel[st],axis=-1)
    link = state.trap_source[rows,own,safe_active]
    linked = (link==state.active[rows,foe])&(link>=0)&(all_active_hp[rows,foe]>0)
    # The legacy anonymous bool has no visible/hidden source provenance.
    # Known direct move traps and Ingrain have independently represented causes.
    anonymous_trap = state.trapped[rows,own]&~(rooted|linked)
    visible_trap = (rooted|linked)&~immune
    # Gen3 Shadow Tag sets literal true; Arena/Magnet call tryTrap(true).
    visible_trap |= rules.shadow[fa]&(all_active_hp[rows,foe]>0)
    actual_hidden = ((rules.arena[fa]&grounded)|(rules.magnet[fa]&steel))&~immune
    # knownType=true is proved by unchanged original types/identity. Source
    # Actual runEvent includes Any callbacks. Public alternatives dispatch only
    # singleEvent('FoeMaybeTrapPokemon'); the static eligibility preserves that
    # distinction (Gen3 Magnet Pull has no FoeMaybeTrapPokemon callback).
    possible = ((rules.possible_arena[safe_species]&grounded)
        |(rules.possible_magnet[safe_species]&steel)
        |(rules.possible_shadow[safe_species]&~rules.shadow[oa]))&~immune
    actual_maybe = ((rules.arena[fa]&grounded)|(rules.magnet[fa]&steel)
        |(rules.shadow[fa]&~rules.shadow[oa]))&~immune
    trapped = can_switch & (visible_trap|actual_hidden|possible|actual_maybe)
    namespace_ok = good_ab&good_type
    reason = (jnp.where(phase_ok,0,PHASE)|jnp.where(arrival_ok,0,ARRIVAL)
        |jnp.where(gender_known,0,GENDER)|jnp.where(same,0,CHANGED)
        |jnp.where(unsafe_history|(ordinary&locked),LOCK,0)|jnp.where(ordinary&source_disable_unknown,DISABLE,0)
        |jnp.where(ordinary&~namespace_ok,NAMESPACE,0)
        |jnp.where(own_species_ok,0,NAMESPACE)
        |jnp.where(ordinary&anonymous_trap,TRAP_ORIGIN,0)).astype(jnp.int32)
    ok = reason==0
    struggle = ordinary&~jnp.any(slot_present&~disabled,axis=-1)
    mode = jnp.where(struggle,actions.STRUGGLE,mode)
    boundary = order_api.RequestBoundary(ledger.stream_id,ledger.owner_side,ledger.engine_binding,
        ledger.generation,active)
    packet = actions.SourceRequest(boundary,jnp.broadcast_to(adapter_binding,(b,8)),constructor.binding,
        jnp.full(b,-1,jnp.int32),jnp.where(ok,mode,actions.UNKNOWN),ok,same,
        jnp.where(ok[:,None]&ordinary[:,None],disabled,False),trapped&ok&ordinary,
        transition.action_mask(state,engine_table)[rows,own])
    return packet,reason


def produce_request(compiled, state, ledger, registration, arrival, *, lifecycle_admitted):
    """Actual source-own fields -> existing ActionFacts/project_actions/router.

    This is a server boundary, not a policy input API. The registry comes from
    actual initial/switch reveal callbacks; its species is never roster-matched.
    """
    record = _checked(compiled); _registered(registration)
    if type(state) is not BattleState:
        raise ValueError('actual BattleState required')
    b = ledger.owner_side.shape[0]
    for name in ('species','level','gender','hp','max_hp','status','item','ability','base_ability',
            'hidden_power_type','hidden_power_power','charge_move','encore_move','encore_counter',
            'choice_move','last_move','trap_source'):
        actions._shape(getattr(state,name),(b,2,6),jnp.int32,'server '+name)
    for name,tail,dtype in (('stats',(6,6),jnp.int32),('types',(6,2),jnp.int32),
            ('moves',(6,4),jnp.int32),('pp',(6,4),jnp.int32),('disabled',(6,4),jnp.bool_),
            ('volatile',(6,),jnp.uint32),('active',(),jnp.int32),('forced_switch',(),jnp.bool_),
            ('trapped',(),jnp.bool_),('entry_pending',(),jnp.bool_)):
        actions._shape(getattr(state,name),(b,2,*tail),dtype,'server '+name)
    actions._shape(state.done,(b,),jnp.bool_,'server done')
    for name,shape,dtype in (('stream_id',(b,),jnp.int32),('engine_binding',(b,8),jnp.uint32),
            ('generation',(b,2),jnp.int32),('physical_active',(b,2),jnp.int32),
            ('public_slot',(b,2),jnp.int32),('encounter',(b,2),jnp.int32),
            ('established',(b,2),jnp.bool_),('revealed_species_id',(b,2),jnp.int32)):
        actions._shape(getattr(arrival,name),shape,dtype,'public arrival '+name)
    actions._shape(lifecycle_admitted,(b,),jnp.bool_,'source lifecycle admission')
    retained = actions._constructor(registration.constructor)
    source,reason = record['producer'](state,ledger,arrival,registration.original,retained,registration.unsafe_origin,
        record['rules'].device,record['adapter'].device.binding,record['engine_table'])
    reason = reason | jnp.where(lifecycle_admitted,0,ARRIVAL)
    source = source._replace(complete=source.complete&lifecycle_admitted,
        mode=jnp.where(lifecycle_admitted,source.mode,actions.UNKNOWN))
    encoded = record['encoder'](state,ledger,source,registration.constructor,
        engine_binding=ledger.engine_binding)
    reason = reason | jnp.where((reason==0)&~encoded.admitted,ADAPTER,0)
    return Produced(encoded,reason)


@dataclass(frozen=True, eq=False)
class Constructed:
    state: BattleState
    cursor: object
    orders: tuple
    registrations: tuple
    requests: tuple
    callbacks: object
    registry: object
    birth_state: object = None  # actual pre-entry constructor output, never reconstructed later


def _constructed(value):
    record = _CONSTRUCTED.get(value) if type(value) is Constructed else None
    if record is None or any(getattr(value,name) is not original for name,original in record):
        raise ValueError('original actual constructor/transition result required')
    for registration in value.registrations:
        _registered(registration)
    return value


def _bind_constructed(*fields):
    value = Constructed(*fields)
    # Real constructor/transition outputs have immutable device leaves. A type
    # tag or copied holder cannot authorize a different physical lifetime.
    for part in (value.state,value.cursor,value.orders,value.requests,value.callbacks,value.registry,value.birth_state):
        actions._immutable_arrays(jax.tree.leaves(part),'actual source lifetime')
    _CONSTRUCTED[value] = tuple((name,getattr(value,name)) for name in value.__dataclass_fields__)
    return value


def _gender(state, pool, indices, initialization):
    if 'gender' in pool.arrays:
        values = pool.arrays['gender'][indices]
        if values.shape != state.gender.shape or not np.isin(values,[-1,0,1,2]).all():
            raise ValueError('actual retained constructor gender codes required')
        state = state._replace(gender=jnp.array(values,jnp.int32,copy=True))
    if initialization is not None:
        from .gender import initialize_genders
        if set(initialization) != {'policy','seed','domain','root_ids'}:
            raise ValueError('explicit existing prospective gender initialization required')
        policy = initialization['policy']; source = policy.get('provenance',{})
        metadata = pool.metadata(); receipt = provenance._pool(pool)
        if (source.get('showdown_commit') != SHOWDOWN
                or source.get('compiled_tree') != metadata.get('compiled_tree')
                or source.get('pool_sha256') != receipt['pool_sha256']
                or source.get('generator_sources') != {name:hashlib.sha256(
                    Path(__file__).with_name(name).read_bytes()).hexdigest()
                    for name in ('gender_policy.cjs','generate_teams.cjs')}):
            raise ValueError('verified gender policy/constructor source provenance required')
        explicit = state.gender
        state = state._replace(gender=jnp.full_like(state.gender,-1))
        state = initialize_genders(state,policy,metadata['vocabulary']['species'],
            seed=initialization['seed'],domain=initialization['domain'],
            root_ids=initialization['root_ids'],explicit_genders=explicit)
    return state


def construct_from_pool(pool, team_indices, *, compiled, stream_ids, seed=0,
                        gender_initialization=None, capacity=64, _execution_compiled=None):
    """Real opt-in constructor callsite; legacy state_from_pool stays unchanged.

    The same immutable pool and owned selection feed physical construction and
    both retained maxPP holders. Gender is assigned BEFORE actual initial public
    reveals/entry callbacks. Missing gender remains UNKNOWN, never absence.
    """
    if _execution_compiled is not None and isinstance(jnp.zeros((),jnp.int32),jax.core.Tracer):
        raise ValueError('checked accounting source constructor cannot be traced')
    from . import engine_lifecycle_callbacks as lifecycle
    record = _checked(compiled)
    if _execution_compiled is not None:
        from . import engine_execution_completeness as execution
        _,_,execution_adapter=execution._checked(_execution_compiled)
        if execution_adapter.causal.projector.catalog is not record['lifecycle']:
            raise ValueError('same original source lifecycle catalog required for accounting')
    provenance._pool(pool,rehash=True)
    if json.dumps(pool.metadata(),sort_keys=True) != record['metadata']:
        raise ValueError('actual constructor metadata differs from source producer')
    indices = np.asarray(team_indices)
    if (indices.ndim!=2 or indices.shape[1]!=2 or not len(indices)
            or indices.dtype.kind not in 'iu'):
        raise ValueError('actual constructor integer [B,2] selection required')
    indices = np.frombuffer(indices.tobytes(),dtype=indices.dtype).reshape(indices.shape)
    b = len(indices)
    actions._shape(stream_ids,(b,),jnp.int32,'actual constructor streams')
    streams = jnp.array(stream_ids,jnp.int32,copy=True)
    state = transition.state_from_pool(pool.arrays,indices,seed=seed,table=None)
    state = _gender(state,pool,indices,gender_initialization)
    registrations,orders = [],[]
    engine = jnp.broadcast_to(record['rules'].device.engine_binding,(b,8))
    for side in (0,1):
        owners = jnp.full(b,side,jnp.int32)
        registrations.append(_registration(state,pool,indices,streams,owners))
        size = jnp.sum(state.species[:,side]>0,axis=-1).astype(jnp.int32)
        orders.append(order_api.initialize_order(streams,owners,size,engine_binding=engine))
    catalog = record['lifecycle']
    cursor = lifecycle.register(streams,catalog=catalog,expected_binding=catalog.binding,
        expected_engine_binding=record['rules'].device.engine_binding)
    frame = lifecycle.initialize_with_lifecycle(state,record['engine_table'],cursor=cursor,
        catalog=catalog,expected_binding=catalog.binding,
        expected_engine_binding=record['rules'].device.engine_binding,capacity=capacity,
        execution_compiled=_execution_compiled)
    requests = tuple(produce_request(compiled,frame.state,o,r,frame.after_registry,
        lifecycle_admitted=frame.cursor.admitted) for o,r in zip(orders,registrations))
    return _bind_constructed(frame.state,frame.cursor,tuple(orders),tuple(registrations),requests,
        frame.callbacks,frame.after_registry,state)


def advance_source(constructed, engine_actions, *, compiled, chance_inputs=None, capacity=64):
    """Actual SAME transition + installed receipts + new own source requests."""
    from . import engine_lifecycle_callbacks as lifecycle
    record = _checked(compiled)
    _constructed(constructed)
    if constructed.callbacks.execution is not None:
        raise ValueError('accounted source snapshots require the owning two-owner consuming driver')
    catalog = record['lifecycle']
    frame = lifecycle.trace_step(constructed.state,engine_actions,record['engine_table'],
        cursor=constructed.cursor,catalog=catalog,expected_binding=catalog.binding,
        expected_engine_binding=record['rules'].device.engine_binding,
        chance_inputs=chance_inputs,capacity=capacity)
    orders = tuple(lifecycle.apply_installations(o,frame) for o in constructed.orders)
    registrations=tuple(_advance_registration(r,frame.state,record['rules'].device)
        for r in constructed.registrations)
    requests = tuple(produce_request(compiled,frame.state,o,r,frame.after_registry,
        lifecycle_admitted=frame.cursor.admitted) for o,r in zip(orders,registrations))
    return _bind_constructed(frame.state,frame.cursor,orders,registrations,requests,
        frame.callbacks,frame.after_registry,constructed.birth_state)


def route_choice(produced, ledger, choices):
    return actions.route_choice(produced.actions,ledger,choices)
