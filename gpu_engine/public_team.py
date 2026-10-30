"""Device projection of public encounter facts into named opponent v4 columns.

This patch covers base species/types/stats, conditions, active boosts and gender.
Live retyping, belief condition overlays, exact clocks and the remaining belief
columns are separate writers. Coverage masks describe this patch, not a complete
observation. No private request or sampled physical state is an input.
"""
from dataclasses import dataclass
import hashlib
import json
from types import MappingProxyType
from typing import Mapping, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from pokezero import showdown as reference
from pokezero.dex import normalize_id
from .private_team import TeamFeaturePatch, vocabulary_binding
from .public_moves import _static_category


SCHEMA = 'pokezero.public_team_base.v1'
STATS = ('hp', 'atk', 'def', 'spa', 'spd', 'spe')
CORE = ('HP_FRACTION', 'ACTIVE', 'LEGAL', 'PRESENT', 'LEVEL',
        *('BASE_' + stat.upper() for stat in STATS),
        *('BOOST_' + stat.upper() for stat in STATS[1:]), 'GENDER_MALE', 'GENDER_FEMALE')
NUMERIC_COLUMNS = tuple(reference.v4_numeric_index(getattr(reference, 'NUMERIC_' + name)) for name in CORE)
CATEGORY_COLUMNS = (reference.CATEGORY_PRIMARY, reference.CATEGORY_SECONDARY, reference.CATEGORY_ROLE,
                    reference.CATEGORY_TYPE_1, reference.CATEGORY_TYPE_2)


class PublicTeamTables(NamedTuple):
    base_stats: jax.Array           # [species,6] raw facts, normalization on device
    original_hp: jax.Array          # exact dex lookup, Transform's source exception
    species_category: jax.Array
    type_categories: jax.Array
    status_categories: jax.Array
    opponent_category: jax.Array
    source_binding: jax.Array
    vocabulary_binding: jax.Array
    ability_binding: jax.Array
    binding: jax.Array


@dataclass(frozen=True)
class PublicSpeciesCatalog:
    names: tuple[str, ...]
    ids: Mapping[str, int]
    device: PublicTeamTables

    def species_id(self, name):
        key = normalize_id(name)
        if not key or key not in self.ids:
            raise ValueError('species outside frozen public dex namespace')
        return self.ids[key]


def compile_species(source, dex, vocabulary, abilities, *, extra_species=()):
    """Freeze the public species namespace and pure checkpoint category lookups."""
    if not np.array_equal(np.asarray(source.device.binding), np.asarray(abilities.device.source_binding)):
        raise ValueError('public species and ability catalogs require the same source')
    # Opaque IDs use dex normalization; checkpoint labels keep punctuation and
    # cosmetic-forme spelling. They are different namespaces (Unown-Z != unownz
    # for an unaliased checkpoint category).
    spelling = {normalize_id(name): name for name in source.species_names if name}
    spelling.update({normalize_id(name): info.name for name,info in dex.species.items()})
    if 'unown' in spelling:
        # These source-enumerated wire forms are absent as separate dex rows.
        # Keep the literal form for checkpoint categories; the existing source
        # fallback supplies base stats/types and belief identities separately.
        from pokezero.randbat_vocab import UNOWN_FORMES
        spelling.update({normalize_id(name): name for name in UNOWN_FORMES})
    spelling.update({normalize_id(name): name for name in extra_species if name})
    names = ('',) + tuple(spelling[key] for key in sorted(spelling) if key)
    bases, hp, cats, types = [], [], [], []
    for name in names:
        info = reference._species_info_base_fallback(dex, name) if name else None
        exact = dex.species_info(name) if name else None
        base = [float(info.base_stats.get(stat, 0)) if info else 0 for stat in STATS]
        if any(not np.isfinite(x) or abs(x) > np.finfo(np.float32).max for x in base):
            raise ValueError('public dex stats must fit finite float32')
        bases.append(base)
        hp.append(float(exact.base_stats.get('hp', 0)) if exact else 0)
        cats.append(_static_category(vocabulary, f'species:{name}') if name else 0)
        kind = info.types if info else ()
        types.append([_static_category(vocabulary, f'type:{kind[i]}') if i < len(kind) else 0 for i in (0, 1)])
    statuses = [_static_category(vocabulary, f'status:{name}') for name in ('none','slp','par','brn','frz','psn','tox')]
    opponent = _static_category(vocabulary, 'pokemon:opponent')
    vb = vocabulary_binding(vocabulary)
    contract = dict(schema=SCHEMA, names=names, bases=bases, original_hp=hp, categories=cats,
        types=types, statuses=statuses, opponent=opponent, source=np.asarray(source.device.binding).tolist(),
        vocabulary=np.asarray(vb).tolist(), abilities=np.asarray(abilities.device.binding).tolist())
    digest = hashlib.sha256(json.dumps(contract, sort_keys=True, separators=(',', ':')).encode()).digest()
    binding = jnp.asarray(np.frombuffer(digest, dtype='>u4').astype(np.uint32))
    return PublicSpeciesCatalog(names, MappingProxyType({normalize_id(name):i for i,name in enumerate(names) if i}),
        PublicTeamTables(jnp.asarray(bases, jnp.float32), jnp.asarray(hp, jnp.float32),
            jnp.asarray(cats,jnp.int32), jnp.asarray(types,jnp.int32), jnp.asarray(statuses,jnp.int32),
            jnp.asarray(opponent,jnp.int32), source.device.binding, vb, abilities.device.binding, binding))


def project_opponent(mons, abilities, tables, *, observer_side, expected_binding):
    """Write the public base patch for the opponent of each observer.

    Transform uses the publicly recorded donor identity; original HP, level and
    gender retain the reference writer's exceptions. The output is in public
    encounter order, including unfilled padded rows, never private party order.
    """
    if type(observer_side) is int:
        if observer_side not in (0,1):raise ValueError('observer side must be 0 or 1')
        observer_side = jnp.asarray(observer_side, jnp.int32)
    if getattr(observer_side,'dtype',None) != jnp.dtype(jnp.int32):
        raise ValueError('observer side requires int32 without coercion')
    if getattr(expected_binding,'dtype',None) != jnp.dtype(jnp.uint32) or expected_binding.shape != (8,):
        raise ValueError('public catalog binding requires eight uint32 words')
    batch = mons.encounter.shape[0]
    observer = jnp.broadcast_to(observer_side, (batch,))
    rows, side = jnp.arange(batch), jnp.clip(1-observer,0,1)
    address = rows,side
    species = mons.species[address]
    transformed = abilities.transformed[address] & (abilities.transform_species[address] > 0)
    effective = jnp.where(transformed, abilities.transform_species[address], species)
    present = mons.encounter[address] > 0
    valid_ids = (species > 0) & (species < len(tables.species_category)) & (effective > 0) & (effective < len(tables.species_category))
    statuses = mons.status[address]
    admitted = ((observer>=0) & (observer<=1) & jnp.all(tables.binding==expected_binding)
        & jnp.all(abilities.binding==tables.ability_binding,axis=-1)
        & jnp.all(mons.species_binding==tables.binding,axis=-1)
        & jnp.all(abilities.species_binding==tables.binding,axis=-1)
        & jnp.all(~present | valid_ids,axis=-1)
        & jnp.all(~present | ((statuses>=0) & (statuses<7)),axis=-1))
    visible = present & admitted[:,None]
    original, encoded = jnp.clip(species,0,len(tables.species_category)-1), jnp.clip(effective,0,len(tables.species_category)-1)
    base = tables.base_stats[encoded]
    original_hp = tables.original_hp[original]
    base = base.at[...,0].set(jnp.where(transformed & (original_hp!=0), original_hp, base[...,0]))
    active = jnp.arange(6)[None,:] == mons.active[address][:,None]
    hp = jnp.clip(mons.hp_numerator[address].astype(jnp.float32)/jnp.maximum(mons.hp_denominator[address],1),0,1)
    hp = jnp.where(mons.hp_known[address],hp,0)
    values = [hp, active, ~mons.fainted[address], present, jnp.minimum(1,mons.level[address]/100)]
    values.extend(jnp.minimum(1,base[...,i]/200) for i in range(6))
    boosts = jnp.clip(mons.boosts[address][...,:5]/6,-1,1)
    values.extend(jnp.where(active,boosts[:,i,None],0) for i in range(5))
    values.extend([mons.gender[address]==1,mons.gender[address]==2])
    compact = jnp.stack([jnp.asarray(x,jnp.float32) for x in values],axis=-1)
    numeric = jnp.zeros((batch,6,132),jnp.float32).at[...,jnp.asarray(NUMERIC_COLUMNS)].set(jnp.where(visible[...,None],compact,0))
    categories = jnp.stack([tables.species_category[encoded],tables.status_categories[jnp.clip(statuses,0,6)],
        jnp.full_like(species,tables.opponent_category),tables.type_categories[encoded,0],tables.type_categories[encoded,1]],axis=-1)
    categorical = jnp.zeros((batch,6,41),jnp.int32).at[...,jnp.asarray(CATEGORY_COLUMNS)].set(jnp.where(visible[...,None],categories,0))
    return TeamFeaturePatch(numeric,categorical,jnp.zeros(132,bool).at[jnp.asarray(NUMERIC_COLUMNS)].set(True),
        jnp.zeros(41,bool).at[jnp.asarray(CATEGORY_COLUMNS)].set(True),admitted)
