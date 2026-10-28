"""Source-ordered belief buckets and PP columns for public opponent tokens.

Uses public encounter evidence and a previously materialized posterior. Original
observations, private opponent requests and sampled physical worlds are not inputs.
Expected stats, exact clocks and the other opponent columns are separate writers.
"""
import hashlib
import json
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from pokezero import showdown as reference
from pokezero.dex import normalize_id
from . import public_belief as belief, public_moves as move
from .private_team import TeamFeaturePatch, vocabulary_binding
from .public_mon_state import _int32
from .public_abilities import _bool

SCHEMA = 'pokezero.public_belief_features.v1'
CORE = ('REVEALED_MOVE_COUNT','CANDIDATE_SET_COUNT','UNCERTAINTY','POSSIBLE_ABILITY_COUNT',
        'POSSIBLE_ITEM_COUNT','POSSIBLE_MOVE_COUNT','REVEALED_ABILITY','REVEALED_ITEM')
NUMERIC_COLUMNS = tuple(reference.v4_numeric_index(getattr(reference,'NUMERIC_'+n)) for n in CORE) + tuple(
    reference.v4_numeric_index(offset+i) for offset in
    (reference.NUMERIC_OPP_MOVE_PP_OFFSET,reference.NUMERIC_OPP_MOVE_PP_VALID_OFFSET) for i in range(16))
CATEGORY_COLUMNS = tuple(range(reference.CATEGORY_BELIEF_ABILITY_OFFSET,
                               reference.CATEGORY_BELIEF_MOVE_OFFSET+reference.BELIEF_MOVE_BUCKET_COUNT))


class FeatureTables(NamedTuple):
    pool: jax.Array
    variant_valid: jax.Array
    variant_ability: jax.Array
    variant_item: jax.Array
    variant_move: jax.Array
    ability_category: jax.Array
    item_category: jax.Array
    move_category: jax.Array
    source_binding: jax.Array
    species_binding: jax.Array
    ability_binding: jax.Array
    item_binding: jax.Array
    move_tables: move.MoveTables
    vocabulary_binding: jax.Array
    binding: jax.Array


def compile_features(source,abilities,items,moves,vocabulary,*,species_catalog):
    for catalog in (abilities,items,moves):
        if not np.array_equal(np.asarray(catalog.device.source_binding),np.asarray(source.device.binding)):
            raise ValueError('belief feature catalogs require the same source')
    if (not np.array_equal(np.asarray(species_catalog.device.source_binding),np.asarray(source.device.binding))
            or not np.array_equal(np.asarray(species_catalog.device.ability_binding),np.asarray(abilities.device.binding))
            or not np.array_equal(np.asarray(species_catalog.device.vocabulary_binding),np.asarray(vocabulary_binding(vocabulary)))):
        raise ValueError('belief writer requires the same species/ability/source/vocabulary catalog')
    shape=tuple(source.device.variant_valid.shape)
    ability_ids,item_ids=np.zeros(shape,np.int32),np.zeros(shape,np.int32)
    move_ids=np.zeros((*shape,4),np.int32)
    for species,variants in enumerate(source.variants):
        for index,variant in enumerate(variants):
            if len(variant.moves)>4:raise ValueError('belief source variant exceeds four move slots')
            ability_ids[species,index]=abilities.ability_id(variant.ability) if variant.ability else 0
            item_ids[species,index]=items.item_id(variant.item) if variant.item else 0
            for slot,name in enumerate(variant.moves):move_ids[species,index,slot]=moves.move_id(name)
    categories=[]
    for kind,catalog in (('possible_ability',abilities),('possible_item',items),('possible_move',moves)):
        if catalog.names!=('',)+tuple(sorted(catalog.names[1:])):
            raise ValueError('feature bucket IDs must follow normalized lexical order')
        categories.append(np.array([move._static_category(vocabulary,'belief:'+kind+':'+normalize_id(n))
            if n else 0 for n in catalog.names],np.int32))
    vb=vocabulary_binding(vocabulary)
    payload=dict(schema=SCHEMA,variant_valid=np.asarray(source.device.variant_valid).tolist(),variant_ability=ability_ids.tolist(),variant_item=item_ids.tolist(),
        variant_move=move_ids.tolist(),categories=[x.tolist() for x in categories],
        source=np.asarray(source.device.binding).tolist(),species=np.asarray(species_catalog.device.binding).tolist(),ability=np.asarray(abilities.device.binding).tolist(),
        item=np.asarray(items.device.binding).tolist(),move=np.asarray(moves.device.binding).tolist(),
        vocabulary=np.asarray(vb).tolist())
    digest=hashlib.sha256(json.dumps(payload,sort_keys=True,separators=(',',':')).encode()).digest()
    binding=jnp.asarray(np.frombuffer(digest,dtype='>u4').astype(np.uint32))
    return FeatureTables(source.device.pool,source.device.variant_valid,jnp.asarray(ability_ids),jnp.asarray(item_ids),jnp.asarray(move_ids),
        *(jnp.asarray(x) for x in categories),source.device.binding,species_catalog.device.binding,abilities.device.binding,
        items.device.binding,moves.device,vb,binding)


def _bits(words,maximum):
    index=jnp.arange(maximum,dtype=jnp.uint32)
    return ((jnp.take(words,index//32,axis=-1)>>(index%32))&1).astype(bool)


def _first_occurrences(ids,selected,count):
    """Scatter source-order positions, avoiding a batch by fact by variant cube."""
    shape=ids.shape[:-1];length=ids.shape[-1]
    def one(row,keep):
        positions=jnp.where(keep&(row>0),jnp.arange(length,dtype=jnp.int32),length)
        return jnp.full(count,length,jnp.int32).at[row].min(positions).at[0].set(length)
    output=jax.vmap(one)(ids.reshape((-1,length)),selected.reshape((-1,length)))
    return output.reshape((*shape,count))


def _buckets(present,limit):
    count=present.shape[-1]
    ids=jnp.arange(count,dtype=jnp.int32)
    # Padding the namespace makes a small generated catalog follow the same
    # fixed bucket contract as the complete production catalog.
    values=jnp.where(present,ids,count)
    values=jnp.pad(values,[(0,0)]*(values.ndim-1)+[(0,max(0,limit-count))],constant_values=count)
    sorted_ids=jnp.sort(values,axis=-1)[...,:limit]
    return jnp.where(sorted_ids<count,sorted_ids,0)


class BeliefFeaturePatch(NamedTuple):
    patch: TeamFeaturePatch
    bucket_move_ids: jax.Array
    posterior_weights: jax.Array
    reason: jax.Array


def project_opponent(states,summary,mons,context,tables,*,observer_side,expected_binding,exact_state=True):
    """Reduce one observer's public posterior, keeping source reveal priority."""
    if context.items is None:raise ValueError('belief features require the item ledger')
    if mons.encounter.ndim!=3 or mons.encounter.shape[1:]!=(2,6) or not mons.encounter.shape[0]:
        raise ValueError('belief features require a nonempty public encounter batch')
    batch=mons.encounter.shape[0]
    def require(array,shape,dtype):
        if getattr(array,'shape',None)!=shape or getattr(array,'dtype',None)!=jnp.dtype(dtype):
            raise ValueError('belief feature state shape or dtype mismatch')
    shape=(batch,2,6)
    for array in (states.species,summary.count,summary.reason,mons.encounter,mons.source_species,
                  context.abilities.original,context.items.revealed):require(array,shape,jnp.int32)
    require(summary.uncertainty,shape,jnp.float32)
    require(context.items.removed,shape,jnp.bool_)
    for array in (states.hard,summary.effective):require(array,(*shape,2),jnp.uint32)
    for array in (context.moves.move_ids,context.moves.uses):require(array,(*shape,move.CAPACITY),jnp.int32)
    for array in (mons.species_binding,context.abilities.species_binding,context.items.species_binding,
                  context.abilities.binding,context.items.binding,context.moves.binding):require(array,(batch,8),jnp.uint32)
    observer=_int32(observer_side);exact=_bool(exact_state)
    if observer.shape not in ((),(batch,)) or exact.shape not in ((),(batch,)):
        raise ValueError('observer and mask require scalar or exact batch vectors')
    if getattr(expected_binding,'shape',None)!=(8,) or getattr(expected_binding,'dtype',None)!=jnp.dtype(jnp.uint32):
        raise ValueError('belief writer requires eight uint32 feature namespace words')
    if states.species.shape!=(batch,2,6) or summary.effective.shape!=(batch,2,6,2):
        raise ValueError('belief features require the public observer/encounter shape')
    rows=jnp.arange(batch);obs=jnp.clip(observer,0,1);side=1-obs
    species=states.species[rows,obs];species_safe=jnp.clip(species,0,tables.variant_move.shape[0]-1)
    ok=(observer>=0)&(observer<=1)&jnp.all(expected_binding==tables.binding)
    ok &= jnp.all(tables.move_tables.source_binding==tables.source_binding)
    for actual,expected in ((context.abilities.binding,tables.ability_binding),
                            (context.items.binding,tables.item_binding),
                            (context.moves.binding,tables.move_tables.binding)):
        ok &= jnp.all(actual==expected,axis=-1)
    ok &= jnp.all(mons.species_binding==tables.species_binding,axis=-1)
    ok &= jnp.all(context.abilities.species_binding==mons.species_binding,axis=-1)
    ok &= jnp.all(context.items.species_binding==mons.species_binding,axis=-1)
    ok &= jnp.all((species>=0)&(species<tables.variant_move.shape[0]),axis=-1)
    ok &= jnp.all(species==mons.source_species[rows,side],axis=-1)
    maximum=tables.variant_move.shape[1]
    pool=tables.pool[species_safe]
    ok &= ~jnp.any((summary.effective[rows,obs]&~pool)!=0,axis=(1,2))
    ok &= ~jnp.any((states.hard[rows,obs]&~pool)!=0,axis=(1,2))
    selected=_bits(summary.effective[rows,obs],maximum)
    ok &= ~jnp.any(selected&~tables.variant_valid[species_safe],axis=(1,2))
    ok &= jnp.all(jnp.isfinite(summary.uncertainty[rows,obs]) & (summary.uncertainty[rows,obs]>=0) &
        (summary.uncertainty[rows,obs]<=1)&(summary.reason[rows,obs]>=0)&(summary.reason[rows,obs]<=belief.NO_SOURCE),axis=-1)
    ok &= jnp.all(selected.sum(axis=-1)==summary.count[rows,obs],axis=-1)
    # Standing pins FILTER source summary lists; they do not reorder them from
    # the kept variant subset. Compute first appearances before applying pins.
    base=jnp.where((summary.reason[rows,obs]==belief.INCONSISTENT_REVEALS)[...,None],
        summary.effective[rows,obs],states.hard[rows,obs])
    base_selected=_bits(base,maximum)
    ability=tables.variant_ability[species_safe];item=tables.variant_item[species_safe]
    moves=tables.variant_move[species_safe].reshape((batch,6,-1))
    length=moves.shape[-1]
    move_selected=jnp.repeat(selected,4,axis=-1)
    move_base=jnp.repeat(base_selected,4,axis=-1)
    ability_first=_first_occurrences(ability,selected,tables.ability_category.shape[0])
    item_first=_first_occurrences(item,selected,tables.item_category.shape[0])
    possible_first=_first_occurrences(moves,move_selected,tables.move_category.shape[0])
    base_first=_first_occurrences(moves,move_base,tables.move_category.shape[0])
    possible_move=possible_first<length
    possible_ability=ability_first<maximum;possible_item=item_first<maximum
    known_ability=context.abilities.original[rows,side]
    known_item=context.items.revealed[rows,side]
    revealed=context.moves.move_ids[rows,side]
    for ids,count in ((known_ability,tables.ability_category.shape[0]),(known_item,tables.item_category.shape[0]),
                      (revealed,tables.move_category.shape[0])):
        ok &= jnp.all((ids>=0)&(ids<count),axis=tuple(range(1,ids.ndim)))
    ability_mask=jnp.where((known_ability>0)[...,None],
        jnp.arange(tables.ability_category.shape[0])==known_ability[...,None],possible_ability)
    item_mask=jnp.where((known_item>0)[...,None],
        jnp.arange(tables.item_category.shape[0])==known_item[...,None],possible_item)
    revealed_first=_first_occurrences(revealed,revealed>0,tables.move_category.shape[0])
    revealed_mask=revealed_first<revealed.shape[-1]
    revealed_count=jnp.sum(revealed>0,axis=-1)
    candidates=possible_move&~revealed_mask
    # Select additional possibilities in source-summary order, then sort the
    # bounded union lexically exactly as the Python categorical writer does.
    rank=jnp.argsort(jnp.argsort(jnp.where(candidates,base_first,length),axis=-1,stable=True),axis=-1,stable=True)
    extra=candidates&(rank<jnp.maximum(0,16-revealed_count)[...,None])
    move_buckets=_buckets(revealed_mask|extra,16)
    ability_buckets=_buckets(ability_mask,2);item_buckets=_buckets(item_mask,6)
    # Reuse the existing absolute-side PP consumer with just these public
    # bucket IDs. The unused observer seat stays padding.
    absolute=jnp.zeros((batch,2,6,16),jnp.int32).at[rows,side].set(move_buckets)
    fractions,valid_pp,pp_ok=move.bucket_pp(context.moves,tables.move_tables,absolute)
    ok &= pp_ok
    present=mons.encounter[rows,side]>0
    exact=jnp.broadcast_to(exact,(batch,))[:,None,None]
    core=jnp.stack((revealed_count,summary.count[rows,obs],summary.uncertainty[rows,obs],
        ability_mask.sum(axis=-1),item_mask.sum(axis=-1),possible_move.sum(axis=-1),
        known_ability>0,(known_item>0)&~context.items.removed[rows,side]),axis=-1).astype(jnp.float32)
    values=jnp.concatenate((core,jnp.where(exact,fractions[rows,side],0),
                            jnp.where(exact,valid_pp[rows,side],False).astype(jnp.float32)),axis=-1)
    cats=jnp.concatenate((tables.ability_category[ability_buckets],tables.item_category[item_buckets],
                          tables.move_category[move_buckets]),axis=-1)
    visible=present&ok[:,None]
    numeric=jnp.zeros((batch,6,132),jnp.float32).at[:,:,jnp.asarray(NUMERIC_COLUMNS)].set(jnp.where(visible[...,None],values,0))
    categorical=jnp.zeros((batch,6,41),jnp.int32).at[:,:,jnp.asarray(CATEGORY_COLUMNS)].set(jnp.where(visible[...,None],cats,0))
    patch=TeamFeaturePatch(numeric,categorical,jnp.zeros(132,bool).at[jnp.asarray(NUMERIC_COLUMNS)].set(True),
        jnp.zeros(41,bool).at[jnp.asarray(CATEGORY_COLUMNS)].set(True),ok)
    weights=selected.astype(jnp.float32)/jnp.maximum(summary.count[rows,obs][...,None],1)
    return BeliefFeaturePatch(patch,jnp.where(visible[...,None],move_buckets,0),
        jnp.where(visible[...,None],weights,0),summary.reason[rows,obs])
