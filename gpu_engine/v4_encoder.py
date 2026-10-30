"""Compose protocol-fed public state and one owner's request into v4 rows.

This is the real writer composition, separate from observation.py's cost
placeholder. Remaining column and wire-family gaps stay explicit. Numerical
parity and model qualification are measured on recorded decisions separately.
"""
from dataclasses import dataclass, replace, fields, is_dataclass
from collections.abc import Mapping
import hashlib
import json
import weakref
from typing import NamedTuple
from types import MappingProxyType

import jax
import jax.numpy as jnp
import numpy as np

from pokezero.category_vocab import CategoryVocabulary
from pokezero.dex import ShowdownDex
from pokezero import showdown as reference
from . import private_actions,private_team,public_inputs,public_mon_state,public_context,public_wire
from . import public_belief,public_moves,public_abilities,public_items,public_team,public_field
from . import public_active,public_damage,public_types,public_sleep,public_toxic,public_tendency
from . import public_belief_refresh,public_belief_lifecycle,public_belief_features,public_belief_snapshot
from . import public_item_nonproc
from . import public_mudshot
from . import public_refresh_stream, public_sleep_clause, public_substitute, public_exact, public_switch_evidence
from . import public_expected_stats,public_hazard_exposure,v4_assembly

SCHEMA='pokezero.protocol_v4_encoder.v1'
PENDING_NUMERIC_COLUMNS=()


@dataclass(frozen=True)
class Masks:
    exact_state: bool
    feature_pack_last_move: bool
    investment_belief_narrowing: bool
    item_belief_narrowing: bool
    stats_block: bool
    tier2_investment: bool
    tier2_residuals: bool
    transition_token_budget: int

    @classmethod
    def from_checkpoint(cls,values):
        if not isinstance(values,dict) or set(values)!=set(cls.__dataclass_fields__):
            raise ValueError('all resolved champion mask fields required, without defaults')
        for key,value in values.items():
            if key=='transition_token_budget':
                if type(value) is not int or value!=0:raise ValueError('23-token v4 contract requires zero transition budget')
            elif type(value) is not bool:raise ValueError('resolved checkpoint masks must be booleans')
        if values['investment_belief_narrowing']:
            raise ValueError('investment evidence producer is not installed in this encoder')
        return cls(**values)


@dataclass(frozen=True, eq=False)
class Catalog:
    source: object
    vocabulary: object
    dex: object
    species: object
    moves: object
    abilities: object
    items: object
    field: object
    active: object
    substitute: object
    exact: object
    damage: object
    types: object
    sleep: object
    toxic: object
    tendency: object
    refresh: object
    lifecycle: object
    sleep_clause: object
    context: object
    features: object
    expected: object
    hazard: object
    actions: object
    masks: Masks
    binding: object
    checkpoint_sha256: str


_CATALOGS = weakref.WeakKeyDictionary()


def _static_snapshot(value, seen):
    """Retain actual immutable array edges, including nested writer tables."""
    if isinstance(value, np.ndarray):
        raise ValueError('encoder static arrays require immutable JAX storage')
    if isinstance(value, jax.Array):
        return value, ()
    if isinstance(value, (str, int, float, bool, type(None))):
        return value, ()
    if id(value) in seen:
        return value, ()
    seen.add(id(value))
    if is_dataclass(value):
        children = tuple((f.name, _static_snapshot(getattr(value, f.name), seen)) for f in fields(value))
        return value, ('attrs', children)
    if isinstance(value, Mapping):
        return value, ('mapping', tuple((k, _static_snapshot(v, seen)) for k, v in value.items()))
    if isinstance(value, (tuple, list)):
        return value, ('sequence', tuple(_static_snapshot(v, seen) for v in value))
    raise ValueError('unsupported encoder static holder: '+type(value).__name__)


def _check_static(value, snapshot):
    original, children = snapshot
    if value is not original:
        raise ValueError('original encoder catalog edges required')
    if not children:
        return
    kind, entries = children
    if kind == 'attrs':
        for key, child in entries:
            _check_static(getattr(value, key), child)
    elif kind == 'mapping':
        if tuple(value) != tuple(k for k, _ in entries):
            raise ValueError('original encoder catalog namespace required')
        for key, child in entries:
            _check_static(value[key], child)
    else:
        if len(value) != len(entries):
            raise ValueError('original encoder static sequence required')
        for current, child in zip(value, entries):
            _check_static(current, child)


def check_catalog(catalog):
    """Checked host boundary for the actual compiled encoder lifetime."""
    record = _CATALOGS.get(catalog) if type(catalog) is Catalog else None
    if record is None:
        raise ValueError('original compiled v4 encoder catalog required')
    for name, snapshot in record:
        _check_static(getattr(catalog, name), snapshot)
    if isinstance(jnp.zeros((), jnp.int32), jax.core.Tracer):
        raise ValueError('checked encoder catalog boundary cannot be traced')
    return catalog


def compile_encoder(source,dex,vocabulary,*,checkpoint_sha256,feature_masks):
    if not isinstance(checkpoint_sha256,str) or len(checkpoint_sha256)!=64 or any(c not in '0123456789abcdef' for c in checkpoint_sha256):
        raise ValueError('exact champion checkpoint SHA256 required')
    masks=Masks.from_checkpoint(feature_masks)
    # Request adapters also consult the dex on host. Freeze their source facts
    # along with device tables so a caller cannot mutate them after binding.
    dex=ShowdownDex(MappingProxyType({k:replace(v,boosts=MappingProxyType(dict(v.boosts))) for k,v in dex.moves.items()}),
        MappingProxyType({k:replace(v,base_stats=MappingProxyType(dict(v.base_stats))) for k,v in dex.species.items()}),
        MappingProxyType({k:MappingProxyType(dict(v)) for k,v in dex.type_chart.items()}))
    vocabulary=CategoryVocabulary(tuple(vocabulary.tokens),vocabulary.oov_buckets,dict(vocabulary.aliases))
    moves=public_moves.compile_moves(source,dex,vocabulary)
    abilities=public_abilities.compile_abilities(source,vocabulary,
        ability_names=('Natural Cure','Early Bird','Shed Skin','Trace','Pressure','Truant','Levitate','Arena Trap','Shadow Tag','Magnet Pull','Shield Dust'))
    items=public_items.compile_items(source,vocabulary)
    species=public_team.compile_species(source,dex,vocabulary,abilities)
    field=public_field.compile_field(vocabulary)
    active=public_active.compile_active(vocabulary,species_catalog=species)
    substitute=public_substitute.compile_substitute(active,species,vocabulary)
    damage=public_damage.compile_damage(vocabulary)
    types=public_types.compile_types(species,dex,vocabulary)
    sleep=public_sleep.compile_sleep(species,vocabulary)
    toxic=public_toxic.compile_toxic(species,vocabulary)
    tendency=public_tendency.compile_tendency(species,moves,vocabulary)
    refresh=public_belief_refresh.compile_refresh(source,moves)
    lifecycle=public_belief_lifecycle.compile_lifecycle(species,abilities,moves,sleep)
    clause=public_sleep_clause.compile_clause(sleep)
    switches=public_switch_evidence.compile_switch_evidence(source,species,abilities)
    nonproc=public_item_nonproc.compile_nonproc(source,species,items,sleep)
    mudshot=public_mudshot.compile_mudshot(source,species,abilities)
    context=public_context.compile_context(source,abilities,moves,items=items,
        item_belief_narrowing=masks.item_belief_narrowing,field_catalog=field,active_catalog=active,
        damage_tables=damage,type_catalog=types,sleep_tables=sleep,toxic_tables=toxic,
        tendency_tables=tendency.device,refresh_tables=refresh,lifecycle_tables=lifecycle,clause_tables=clause,switch_tables=switches,nonproc_tables=nonproc,
        mudshot_tables=mudshot,mudshot_move_id=moves.ids.get('mudshot',0))
    features=public_belief_features.compile_features(source,abilities,items,moves,vocabulary,species_catalog=species)
    exact=public_exact.compile_exact(abilities,features,sleep,vocabulary)
    expected=public_expected_stats.compile_expected(source,dex,species)
    hazard=public_hazard_exposure.compile_hazard(species,dex,vocabulary,feature_tables=features,
        field_tables=field.device,abilities=abilities)
    actions=private_actions.compile_actions(dex,vocabulary)
    bindings=dict(checkpoint=checkpoint_sha256,masks=feature_masks,schema=SCHEMA,
        context=np.asarray(context.binding).tolist(),features=np.asarray(features.binding).tolist(),
        expected=np.asarray(expected.device.binding).tolist(),hazard=np.asarray(hazard.device.binding).tolist(),
        actions=np.asarray(actions.device.binding).tolist(),substitute=np.asarray(substitute.binding).tolist(),
        exact=np.asarray(exact.device.binding).tolist())
    digest=hashlib.sha256(json.dumps(bindings,sort_keys=True,separators=(',',':')).encode()).digest()
    binding=jnp.asarray(np.frombuffer(digest,dtype='>u4').astype(np.uint32))
    catalog = Catalog(source,vocabulary,dex,species,moves,abilities,items,field,active,substitute,exact,damage,types,
        sleep,toxic,tendency,refresh,lifecycle,clause,context,features,expected,hazard,actions,masks,binding,checkpoint_sha256)
    seen = set()
    _CATALOGS[catalog] = tuple((f.name, _static_snapshot(getattr(catalog, f.name), seen)) for f in fields(catalog))
    return catalog


class State(NamedTuple):
    cursor: object
    beliefs: object
    mons: object
    context: object
    binding: object


class OwnFacts(NamedTuple):
    actions: object
    status: object
    targets: object
    hazard: object
    exact_abilities: object
    binding: object


class Encoded(NamedTuple):
    observation: v4_assembly.Assembly
    posterior_mask: object
    posterior_weights: object
    posterior_reason: object
    pending_numeric_columns: object
    full_encoder_admitted: object


def initialize(catalog,*,stream_ids,owner_sides=None):
    if getattr(stream_ids,'ndim',None)!=1 or stream_ids.dtype!=jnp.int32 or not stream_ids.shape[0]:
        raise ValueError('nonempty int32 stream identifiers required')
    b=stream_ids.shape[0]
    visibility = (public_toxic.UNKNOWN, public_toxic.UNKNOWN)
    if owner_sides is not None:
        owners = np.asarray(owner_sides)
        if owners.shape != (b,) or owners.dtype != np.int32 or np.any((owners < 0) | (owners > 1)):
            raise ValueError('owning stream requires B int32 absolute owner sides')
        # The wire owner fixes HP representation before the first event, as in
        # the Python owning parser. A denominator of 100 alone is ambiguous.
        visibility = np.where(np.arange(2)[None, :] == owners[:, None],
            public_toxic.EXACT, public_toxic.PERCENTAGE).astype(np.int32)
    beliefs=public_belief.initialize(jnp.zeros((b,2,6),jnp.int32),catalog.source.device)
    mons=public_mon_state.initialize(b,species_binding=catalog.species.device.binding)
    context=public_context.initialize(beliefs,catalog.context,species_binding=catalog.species.device.binding,
        toxic_complete_prefix=True,toxic_hp_visibility=visibility,tendency_complete_prefix=True,
        source_tables=catalog.source.device,clause_complete_prefix=True)
    cursor=public_inputs.initialize(stream_ids,source_binding=catalog.source.device.binding)
    return State(cursor,beliefs,mons,context,jnp.broadcast_to(catalog.binding,(b,8)))


def parse_own_requests(catalog,requests,*,owner_sides):
    """Read only the supplied owning packets, never the opponent request."""
    if len(requests)!=len(owner_sides) or not requests:raise ValueError('matching nonempty request/owner rows required')
    rows=[]
    for request,owner in zip(requests,owner_sides):
        if type(owner) is not int or owner not in (0,1):raise ValueError('absolute owning side required')
        rows.append(OwnFacts(private_actions.parse_action_facts(request,owner_side=owner,
                vocabulary=catalog.vocabulary,catalog=catalog.actions),
            public_sleep.parse_own_status(request,owner_side=owner,species_catalog=catalog.species,vocabulary=catalog.vocabulary),
            public_expected_stats.parse_own_targets(request,owner_side=owner,catalog=catalog.expected),
            public_hazard_exposure.parse_own_hazard(request,owner_side=owner,catalog=catalog.hazard,vocabulary=catalog.vocabulary),
            public_exact.parse_own_abilities(request,owner_side=owner,catalog=catalog.exact,vocabulary=catalog.vocabulary),
            catalog.binding))
    return jax.tree.map(lambda *values:jnp.stack(values),*rows)


def _outer_binding(value,batch):
    if getattr(value,'shape',None)!=(batch,8) or getattr(value,'dtype',None)!=jnp.dtype(jnp.uint32):
        raise ValueError('encoder namespace requires exact B by8 uint32 words')


def fold_chunk(state,chunk,catalog):
    _outer_binding(state.binding,state.mons.encounter.shape[0])
    result=public_inputs.fold_chunk(state.cursor,state.beliefs,catalog.source.device,chunk,
        mon_state=state.mons,context=state.context,context_tables=catalog.context)
    ok=jnp.all(state.binding==catalog.binding,axis=-1)
    error=jnp.where(ok,result.error,public_inputs.Error.SOURCE)
    # fold_chunk already owns every consumer's per-row commit, including its
    # global namespace words. A changed outer contract admits no work at all.
    choose=lambda old,new:jnp.where(ok.reshape((ok.shape[0],)+(1,)*(old.ndim-1)),new,old)
    refresh=public_refresh_stream.commit(ok,state.context.refresh,result.context.refresh)
    context=jax.tree.map(choose,state.context._replace(refresh=None),result.context._replace(refresh=None))._replace(refresh=refresh)
    candidate=State(jax.tree.map(choose,state.cursor,result.cursor),jax.tree.map(choose,state.beliefs,result.beliefs),
        jax.tree.map(choose,state.mons,result.mon_state),context,state.binding)
    return candidate,error


def wire_kwargs(catalog):
    return dict(species_catalog=catalog.species,source=catalog.source,moves=catalog.moves,abilities=catalog.abilities,
        items=catalog.items,context_tables=catalog.context,field_catalog=catalog.field,active_catalog=catalog.active,
        damage_tables=catalog.damage,type_catalog=catalog.types,sleep_tables=catalog.sleep,toxic_tables=catalog.toxic,
        tendency_catalog=catalog.tendency)


def _numeric_patch(patch,*,shape):
    return private_team.TeamFeaturePatch(patch.numeric,jnp.zeros(shape[:-1]+(41,),jnp.int32),
        patch.numeric_columns,jnp.zeros(41,bool),patch.admitted)


def _overlay_active(result,patch,mons,own_status,observer):
    """Public side values follow request-active own rows and public opponent rows.

    The Python source deliberately accepts a request-active pointer which differs
    from the replay's public active pointer. Own rows use the request flag, while
    the side's boost/volatile/type/last-move values remain public source facts.
    """
    b=mons.encounter.shape[0];rows=jnp.arange(b);obs=jnp.clip(observer,0,1)
    numeric=patch.numeric;cats=patch.categorical
    nm=jnp.broadcast_to(patch.numeric_columns,numeric.shape)
    cm=jnp.broadcast_to(patch.categorical_columns,cats.shape)
    own_active=own_status.present&own_status.active
    opp_active=(jnp.arange(6)[None,:]==mons.active[rows,1-obs,None])&(mons.encounter[rows,1-obs]>0)
    for section,side,active in (('own',obs,own_active),('opponent',1-obs,opp_active)):
        part=private_team.TeamFeaturePatch(
            jnp.broadcast_to(numeric[rows,side,None,:],(b,6,132)),
            jnp.broadcast_to(cats[rows,side,None,:],(b,6,41)),
            active[...,None]&nm[rows,side,None,:],active[...,None]&cm[rows,side,None,:],patch.admitted)
        result=v4_assembly.overlay(result,part,section=section)
    return result


def encode(state,own,catalog,*,observer_sides):
    """Run all installed retained writers, preserving their source order/masks."""
    b=state.mons.encounter.shape[0];rows=jnp.arange(b)
    _outer_binding(state.binding,b);_outer_binding(own.binding,b)
    if observer_sides.shape!=(b,) or observer_sides.dtype!=jnp.int32:raise ValueError('exact B int32 observer sides required')
    context=state.context;tables=catalog.context;mons=state.mons;observer=observer_sides
    valid=jnp.all(state.binding==catalog.binding,axis=-1)&jnp.all(own.binding==catalog.binding,axis=-1)
    valid &= ~state.cursor.needs_more&~context.tendency_raw_pending&~context.toxic_raw_pending&(context.refresh.family==0)
    visible=context.refresh.controller.visible
    summary,cache_ok=public_belief_snapshot.read(visible,state.beliefs,catalog.source.device)
    valid &= jnp.all(cache_ok,axis=(1,2))
    if context.nonproc is None or tables.nonproc is None:
        valid &= False
    else:valid &= public_context.validate_nonproc(context,mons,tables)
    if context.mudshot is None or tables.mudshot is None:
        valid &= False
    else:valid &= public_context.validate_mudshot(context,mons,catalog.source.device,tables,summary.effective)
    if context.switches is None or tables.switches is None:
        valid &= False
    else:
        valid &= public_switch_evidence.validate(context.switches,tables.switches,mons,context.abilities,
            catalog.source.device,summary.effective,context.sleep.live)
    context,view_beliefs,boundary_ok=public_context.observation_boundary_copy(context,state.beliefs,mons,
        catalog.source.device,tables,valid=valid)
    valid &= boundary_ok
    state=state._replace(context=context,beliefs=view_beliefs)
    visible=context.refresh.controller.visible
    summary,cache_ok=public_belief_snapshot.read(visible,state.beliefs,catalog.source.device)
    valid &= jnp.all(cache_ok,axis=(1,2))
    tendency=public_tendency.project_from_request(context.tendency,tables.tendency,mons,observer_side=observer,
        own_status=own.status,expected_vocabulary=catalog.species.device.vocabulary_binding,enabled=catalog.masks.stats_block)
    result=v4_assembly.project_base_sections(facts=own.actions,action_tables=catalog.actions.device,mons=mons,context=context,
        team_tables=catalog.species.device,field_tables=tables.field,tendency=tendency,observer_side=observer,
        vocabulary_binding=catalog.species.device.vocabulary_binding,category_count=catalog.vocabulary.size,exact_state=catalog.masks.exact_state)
    beliefs=public_belief_features.project_opponent(state.beliefs,summary,mons,context,catalog.features,
        observer_side=observer,expected_binding=catalog.features.binding,exact_state=catalog.masks.exact_state)
    result=v4_assembly.overlay(result,beliefs.patch,section='opponent')
    expected=public_expected_stats.project_opponent(state.beliefs,summary,mons,context.abilities,catalog.expected.device,
        observer_side=observer,expected_binding=catalog.expected.device.binding,own_targets=own.targets,exact_state=catalog.masks.exact_state)
    result=v4_assembly.overlay(result,expected,section='opponent')
    # The owning Transform changes public species/base types/stats, but does
    # not replace request HP, level, gender, actual stats, current item/ability.
    own_public=public_team.project_opponent(mons,context.abilities,catalog.species.device,
        observer_side=1-observer,expected_binding=catalog.species.device.binding)
    columns=jnp.array([reference.v4_numeric_index(getattr(reference,'NUMERIC_BASE_'+name))
        for name in ('HP','ATK','DEF','SPA','SPD','SPE')])
    cats=jnp.array([reference.CATEGORY_PRIMARY,reference.CATEGORY_TYPE_1,reference.CATEGORY_TYPE_2])
    transformed=context.abilities.transformed[rows,jnp.clip(observer,0,1)]
    nm=jnp.zeros((b,6,132),bool).at[:,:,columns].set(transformed[:,:,None])
    cm=jnp.zeros((b,6,41),bool).at[:,:,cats].set(transformed[:,:,None])
    full=private_team.TeamFeaturePatch(jnp.zeros((b,2,6,132),jnp.float32).at[rows,jnp.clip(observer,0,1)].set(own_public.numeric),
        jnp.zeros((b,2,6,41),jnp.int32).at[rows,jnp.clip(observer,0,1)].set(own_public.categorical),
        jnp.zeros((b,2,6,132),bool).at[rows,jnp.clip(observer,0,1)].set(nm),
        jnp.zeros((b,2,6,41),bool).at[rows,jnp.clip(observer,0,1)].set(cm),own_public.admitted)
    result=v4_assembly.overlay(result,v4_assembly.remap_public_own(full,mons,own.status,observer_side=observer),section='own')
    sleep=public_sleep.project(context.sleep,tables.sleep,mons,observer_side=observer,
        expected_vocabulary=catalog.species.device.vocabulary_binding,own_status=own.status,exact_state=catalog.masks.exact_state)
    result=v4_assembly.overlay_public_team(result,_numeric_patch(sleep,shape=(b,2,6,132)),mons,own.status,observer_side=observer)
    for patch in (v4_assembly.project_own_boosts(mons),
            public_active.project(context.active,tables.active,mons,expected_vocabulary=catalog.species.device.vocabulary_binding),
            public_damage.project_active(context.damage,tables.damage,mons,expected_vocabulary=catalog.species.device.vocabulary_binding),
            _numeric_patch(public_toxic.project_active(context.toxic,tables.toxic,expected_vocabulary=catalog.species.device.vocabulary_binding),shape=(b,2,132)),
            v4_assembly.project_active_categories(context,tables,mons,feature_pack_last_move=catalog.masks.feature_pack_last_move)):
        result=_overlay_active(result,patch,mons,own.status,observer)
    types=public_types.project(context.types,tables.types,mons,expected_vocabulary=catalog.species.device.vocabulary_binding)
    type_patch=private_team.TeamFeaturePatch(jnp.zeros((b,2,132),jnp.float32),types.categorical,
        jnp.zeros(132,bool),types.write_mask,types.admitted)
    result=_overlay_active(result,type_patch,mons,own.status,observer)
    sub=public_substitute.project(context.active,tables.active,mons,own.actions.team,catalog.substitute,
        observer_side=observer,expected_binding=catalog.substitute.binding,exact_state=catalog.masks.exact_state)
    result=v4_assembly.overlay(result,sub.own,section='own')
    result=v4_assembly.overlay(result,sub.opponent,section='opponent')
    ruled_out=jnp.zeros((b,2,6,len(catalog.abilities.names)),bool)
    if context.switches is not None and tables.switches is not None:
        ruled_out=context.switches.ruled_out[...,None]&(jnp.arange(len(catalog.abilities.names))==tables.switches.intimidate_id)
    exact=public_exact.project(own.exact_abilities,own.actions.team,own.status,mons,context,state.beliefs,summary,
        catalog.features,tables.sleep,catalog.exact.device,observer_side=observer,
        expected_binding=catalog.exact.device.binding,ruled_out_abilities=ruled_out,exact_state=catalog.masks.exact_state)
    result=v4_assembly.overlay(result,exact.own,section='own')
    result=v4_assembly.overlay(result,exact.opponent,section='opponent')
    result=v4_assembly.overlay(result,public_damage.project_credits(context.damage,tables.damage,mons,
        observer_side=observer,expected_vocabulary=catalog.species.device.vocabulary_binding),section='field')
    hazard=public_hazard_exposure.project(own.hazard,mons,context,state.beliefs,visible,catalog.source.device,
        context.field,catalog.hazard.device,catalog.features,observer_side=observer,expected_binding=catalog.hazard.device.binding)
    result=v4_assembly.overlay(result,_numeric_patch(hazard,shape=(b,132)),section='field')
    clause=public_sleep_clause.project(context.sleep_clause,tables.sleep_clause,observer_side=observer,
        expected_vocabulary=catalog.species.device.vocabulary_binding,exact_state=catalog.masks.exact_state)
    result=v4_assembly.overlay(result,clause,section='field')
    admitted=result.admitted&valid
    result=result._replace(admitted=admitted,numeric_features=jnp.where(admitted[:,None,None],result.numeric_features,0),
        categorical_ids=jnp.where(admitted[:,None,None],result.categorical_ids,0),attention_mask=result.attention_mask&admitted[:,None],
        legal_action_mask=result.legal_action_mask&admitted[:,None],switch_targets=jnp.where(admitted[:,None],result.switch_targets,-1))
    posterior=summary.effective[rows,jnp.clip(observer,0,1)]
    return Encoded(result,jnp.where(admitted[:,None,None],posterior,0),
        jnp.where(admitted[:,None,None],beliefs.posterior_weights,0),beliefs.reason,
        jnp.asarray(PENDING_NUMERIC_COLUMNS,jnp.int32),jnp.zeros(b,bool))
