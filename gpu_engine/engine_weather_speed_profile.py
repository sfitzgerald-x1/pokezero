"""INTERNAL, conditional Speed profile of the actual physical birth arrays.

Neither a ConstructorPool hash nor the source's possible-item inventory proves
the values in a constructed BattleState. Inspect BOTH parties/ALL six slots.
This checks a bounded Speed function, not generator/whole-variant legality,
complete public fidelity, callback tie order, or B4 qualification.

The owning constructor/Session must capture qualify_birth BEFORE entry/item
callbacks can consume data, retain the original compiled profile, and carry the
returned record through every step. Record is trusted internal carry, NOT an
attestation token. A caller can author a NamedTuple or present a later state;
this module cannot prove birth time/history from that state. The Session's
existing original-carry identity guard must protect this boundary.

SourceRules binds the pinned original Gen3 random singles static loader/Dex.
The proof additionally assumes its original rules, no custom speed hooks or
Trick Room/twisted-dimension semantics, and faithful prior status/boost updates.
Those conditions cannot be established from a format label/current item alone.
No caller boolean is accepted as proof of them. Preserve all existing refusals.
"""
from dataclasses import dataclass
import hashlib
import json
from typing import NamedTuple
import weakref

import jax
import jax.numpy as jnp
import numpy as np

from pokezero.dex import normalize_id
from . import engine_private_team as team, engine_public_events as public
from . import engine_request_source as source, v4_encoder, mechanics
from .state import BattleState

SCHEMA = 'pokezero.engine_weather_birth_speed_profile.v1'
# Complete positive item union of pinned original gen3 randbat getItem, not
# the much larger metadata vocabulary. None is a separate explicit math scope.
SOURCE_ITEMS = ('choiceband', 'leftovers', 'liechiberry', 'lightball', 'lumberry',
    'petayaberry', 'salacberry', 'silkscarf', 'souldew', 'stick', 'thickclub',
    'twistedspoon', 'whiteherb')
# The71 ability IDs in all1682 original source variants. The only immediate
# ModifySpe hooks are Swift Swim/Chlorophyll (installed engine components).
SOURCE_ABILITIES = tuple('airlock arenatrap battlearmor blaze chlorophyll clearbody cloudnine colorchange '
    'compoundeyes cutecharm drizzle drought earlybird effectspore flamebody flashfire forecast guts '
    'hugepower hustle hypercutter immunity innerfocus insomnia intimidate keeneye levitate limber '
    'liquidooze magmaarmor magnetpull marvelscale minus naturalcure oblivious overgrow owntempo pickup '
    'plus poisonpoint pressure purepower rockhead roughskin runaway sandstream sandveil serenegrace '
    'shadowtag shedskin shellarmor shielddust soundproof speedboost static stickyhold sturdy suctioncups '
    'swarm swiftswim synchronize thickfat torrent trace truant vitalspirit voltabsorb waterabsorb '
    'waterveil whitesmoke wonderguard'.split())


class ProfileData(NamedTuple):
    binding: object
    engine_binding: object
    source_binding: object
    species: object
    birth_items: object
    current_items: object
    abilities: object


@dataclass(frozen=True, eq=False)
class CompiledProfile:
    device: ProfileData


class BirthRecord(NamedTuple):
    """All leaves lead with B, including bindings[B,8], safe for row carry."""
    qualified_birth: object
    stream_id: object
    engine_binding: object
    source_binding: object
    profile_binding: object
    birth_spe: object
    birth_ability: object


_PROFILES = weakref.WeakKeyDictionary()


def _json(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)


def compile_profile(metadata, *, catalog, engine_catalog, rules, allow_none_at_birth=False):
    """Checked immutable original edges; no generated roster or pool inference.

    allow_none_at_birth explicitly extends ONLY the mathematical item profile,
    never certifies that an itemless member belongs to the original generator.
    """
    if type(allow_none_at_birth) is not bool:
        raise ValueError('explicit boolean itemless birth math scope required')
    v4_encoder.check_catalog(catalog)
    public.check_catalog(engine_catalog, engine_catalog.engine_table,
        expected_binding=engine_catalog.binding, expected_engine_binding=team.engine_metadata_binding(metadata))
    rule_record = source._rules(rules)
    payload = json.loads(rule_record[2])
    if (engine_catalog.public_moves is not catalog.moves
            or not np.array_equal(np.asarray(catalog.source.device.binding), np.asarray(engine_catalog.source_binding))
            or not np.array_equal(np.asarray(rules.device.engine_binding), np.asarray(engine_catalog.engine_binding))
            or payload['vocabulary'] != metadata['vocabulary']
            or payload['compiled_tree'] != metadata.get('compiled_tree')
            or metadata.get('source_commit_label') != source.SHOWDOWN):
        raise ValueError('same original engine/source/rules/metadata edges required')
    vocab = metadata['vocabulary']
    for kind in ('species', 'items', 'abilities'):
        names = vocab[kind]
        if (type(names) is not list or not names or names[0] not in ('', '_none')
                or any(type(n) is not str or not n or normalize_id(n) != n for n in names[1:])
                or len(set(names)) != len(names)):
            raise ValueError('canonical ordered physical ' + kind + ' namespace required')
    variants = [v for rows in catalog.source.variants for v in rows]
    if not variants:
        raise ValueError('nonempty bound source variant inventory required')
    items = {normalize_id(v.item) for v in variants}
    abilities = {normalize_id(v.ability) for v in variants}
    if (not items <= set(SOURCE_ITEMS) | ({''} if allow_none_at_birth else set())
            or not abilities <= set(SOURCE_ABILITIES) or '' in abilities
            or not items - {''} <= set(vocab['items'][1:])
            or not abilities <= set(vocab['abilities'][1:])):
        raise ValueError('unsupported or unmapped bound source speed inventory')
    # Positive inventory support is necessary but NOT sufficient: birth below
    # also checks every actual item/ability ID against these tables.
    birth_items = jnp.asarray([allow_none_at_birth] + [n in items for n in vocab['items'][1:]], bool)
    current_items = birth_items.at[0].set(True)  # supported consumption produces0
    supported_abilities = jnp.asarray([False] + [n in abilities for n in vocab['abilities'][1:]], bool)
    supported_abilities &= rules.device.ability_supported
    species = jnp.asarray([False] + [catalog.source.species_id(n) > 0 for n in vocab['species'][1:]], bool)
    species &= rules.device.species_supported
    contract = dict(schema=SCHEMA, metadata=metadata, encoder=np.asarray(catalog.binding).tolist(),
        source=np.asarray(catalog.source.device.binding).tolist(), engine=engine_catalog.binding,
        rules=payload, items=sorted(items), abilities=sorted(abilities),
        allow_none_at_birth=allow_none_at_birth, physical='both_parties_all_six',
        bounds='stored_spe1..2500_stage-6..6_Q12par_current_equals_birth_ability',
        scope='conditional_speed_math_not_generator_or_complete_source_qualification')
    digest = hashlib.sha256(_json(contract).encode()).digest()
    device = ProfileData(jnp.asarray(np.frombuffer(digest, dtype='>u4').astype(np.uint32)),
        jnp.asarray(engine_catalog.engine_binding, jnp.uint32), catalog.source.device.binding,
        species, birth_items, current_items, supported_abilities)
    compiled = CompiledProfile(device)
    _PROFILES[compiled] = (device, tuple(device), metadata, _json(metadata), catalog, engine_catalog, rules)
    return compiled


def check_profile(profile):
    record = _PROFILES.get(profile) if type(profile) is CompiledProfile else None
    if (record is None or profile.device is not record[0]
            or any(a is not b for a, b in zip(profile.device, record[1]))
            or _json(record[2]) != record[3]):
        raise ValueError('original compiled speed profile/metadata/immutable edges required')
    _, _, _, _, catalog, engine_catalog, rules = record
    v4_encoder.check_catalog(catalog)
    source._rules(rules)
    public.check_catalog(engine_catalog, engine_catalog.engine_table,
        expected_binding=engine_catalog.binding, expected_engine_binding=profile.device.engine_binding)
    return profile.device


def _typed(value, shape, dtype):
    if (not isinstance(value, (jax.Array, jax.core.Tracer))
            or value.shape != shape or value.dtype != jnp.dtype(dtype)):
        raise ValueError('exact immutable physical batch shape/dtype required')


def _state_shape(state, stream_ids):
    if type(state) is not BattleState or getattr(stream_ids, 'ndim', None) != 1 or not stream_ids.shape[0]:
        raise ValueError('actual nonempty physical BattleState/stream batch required')
    b = stream_ids.shape[0]
    _typed(stream_ids, (b,), jnp.int32)
    for name in ('species', 'item', 'ability', 'base_ability', 'status', 'hp', 'max_hp'):
        _typed(getattr(state, name), (b, 2, 6), jnp.int32)
    _typed(state.stats, (b, 2, 6, 6), jnp.int32)
    _typed(state.boosts, (b, 2, 6, 7), jnp.int32)
    _typed(state.turn, (b,), jnp.int32)
    _typed(state.done, (b,), jnp.bool_)
    return b


def _members(state, data, *, birth):
    def allowed(value, table):
        return (value >= 0) & (value < table.shape[0]) & table[jnp.clip(value, 0, table.shape[0] - 1)]
    spe, stage = state.stats[..., 5], state.boosts[..., 4]
    members = (allowed(state.species, data.species)
        & allowed(state.item, data.birth_items if birth else data.current_items)
        & allowed(state.base_ability, data.abilities) & allowed(state.ability, data.abilities)
        & (state.ability == state.base_ability)
        & (spe >= 1) & (spe <= 2500) & (stage >= -6) & (stage <= 6)
        & (state.status >= mechanics.NONE) & (state.status <= mechanics.FRZ))
    return jnp.all(members, axis=(1, 2))


def qualify_birth_device(state, stream_ids, data):
    """Trusted device primitive; owning Session supplies ACTUAL pre-entry birth.

    Host callers use qualify_birth, which reruns existing compiler guards.
    turn0/not-done/all-full-HP are conservative checks, NOT proof of birth.
    Only the owning Session's original fresh constructor callsite establishes
    that precondition; resetting saved-root fields cannot establish it.
    """
    b = _state_shape(state, stream_ids)
    order = jnp.argsort(stream_ids)
    sorted_ids = stream_ids[order]
    equal = sorted_ids[1:] == sorted_ids[:-1]
    duplicates = jnp.concatenate((equal, jnp.array([False]))) | jnp.concatenate((jnp.array([False]), equal))
    duplicate_rows = jnp.zeros(b, bool).at[order].set(duplicates)
    fresh = (state.turn == 0) & ~state.done & jnp.all((state.hp > 0) & (state.hp == state.max_hp), axis=(1, 2))
    qualified = _members(state, data, birth=True) & fresh & (stream_ids > 0) & ~duplicate_rows
    broadcast = lambda v: jnp.broadcast_to(v, (b, 8))
    return BirthRecord(qualified, stream_ids, broadcast(data.engine_binding), broadcast(data.source_binding),
        broadcast(data.binding), state.stats[..., 5], state.base_ability)


def retain_device(record, state, stream_ids, data):
    """Trusted hotpath: False is sticky; current safe items never recreate birth.

    Supported old-item preservation/swap/consumption-to0 is closed over the
    profile. This does not attest unseen mutations, transient outside items,
    or skipped carries. Session ownership/fixed kernel identity is required.
    Transform/copied ability/stat changes conservatively latch False.
    """
    b = _state_shape(state, stream_ids)
    if type(record) is not BirthRecord:
        raise ValueError('original internal birth record carry required')
    for value, shape, dtype in ((record.qualified_birth, (b,), jnp.bool_),
            (record.stream_id, (b,), jnp.int32), (record.engine_binding, (b, 8), jnp.uint32),
            (record.source_binding, (b, 8), jnp.uint32), (record.profile_binding, (b, 8), jnp.uint32),
            (record.birth_spe, (b, 2, 6), jnp.int32), (record.birth_ability, (b, 2, 6), jnp.int32)):
        _typed(value, shape, dtype)
    bound = ((record.stream_id == stream_ids) & (stream_ids > 0)
        & jnp.all(record.engine_binding == data.engine_binding, axis=1)
        & jnp.all(record.source_binding == data.source_binding, axis=1)
        & jnp.all(record.profile_binding == data.binding, axis=1))
    unchanged = (jnp.all(state.stats[..., 5] == record.birth_spe, axis=(1, 2))
        & jnp.all(state.base_ability == record.birth_ability, axis=(1, 2)))
    return record._replace(qualified_birth=record.qualified_birth & bound & unchanged & _members(state, data, birth=False))


def qualify_birth(state, stream_ids, *, profile):
    return qualify_birth_device(state, stream_ids, check_profile(profile))


def retain(record, state, stream_ids, *, profile):
    return retain_device(record, state, stream_ids, check_profile(profile))
