"""Public source -> two declared Arm-D fixture plans, without a live-world donor.

This is the source-sampling boundary, not a current BattleState constructor or
teacher. It never reads a private opponent cell, generates a team, folds wire,
or runs a model/transition. A later consumer must map these original sampler
fixtures and the retained current PUBLIC facts into a hypothetical state. It
must not fill missing clocks/conditions from the real opponent. Until then the
current-world unknown mass is one and D remains unavailable.
"""
from dataclasses import dataclass, replace
import copy
import hashlib
import json
import random
from types import MappingProxyType

import jax.numpy as j
import numpy as np

from pokezero import v5_contract as contract
from pokezero.dex import normalize_id
from pokezero.determinization import (
    _fixture_from_variant_payload, _sample_gender, _sample_hidden_backline,
    _normalize_species_id,
)
from pokezero.randbat import Gen3RandbatSource
from pokezero.v5_gumbel import Recipe
from . import engine_device_trackers as public


SCHEMA = 'pokezero.device-public-source-world-plans.v1'
# Only these indexed owner cells leave the actual root. In particular no full
# state host copy, opposing request/constructor, shared queued actions or random
# draw capsules are retained. Global field facts come from the public readout.
OWN_MON_FIELDS = (
    'species', 'level', 'hidden_power_type', 'hidden_power_power', 'happiness',
    'gender', 'types', 'base_types', 'stats', 'hp', 'max_hp', 'status',
    'status_counter', 'sleep_skipped', 'rest_origin', 'moves', 'pp', 'disabled',
    'choice_move', 'last_move', 'encore_move', 'encore_counter', 'charge_move',
    'charge_counter', 'item', 'ability', 'base_ability', 'boosts', 'volatile',
    'confusion_counter', 'protect_counter', 'substitute_hp', 'trap_source',
    'truant_loaf',
)
OWN_SIDE_FIELDS = (
    'active', 'last_action_kind', 'last_action_index', 'entry_pending',
    'active_turns', 'trapped', 'forced_switch', 'baton_pass', 'wish_counter',
    'wish_amount',
)


def _freeze(value):
    if isinstance(value, dict):
        return MappingProxyType({key: _freeze(item) for key, item in value.items()})
    if isinstance(value, (tuple, list)):
        return tuple(_freeze(item) for item in value)
    if hasattr(value, 'shape'):
        result = np.array(value, copy=True)
        result.flags.writeable = False
        return result
    return value


@dataclass(frozen=True)
class Source:
    source_hash: str
    source_rows_sha256: str
    binding_sha256: str
    vocabulary: object
    original: Gen3RandbatSource


def compile_source(source, metadata, *, expected_source_hash,
                   expected_source_rows_sha256):
    """Freeze the genuine original source once, independently of any game pool.

    The existing source census binds the ordered variant axis; a copied source
    preserves the original species order used by the hidden-backline sampler.
    No pool argument exists and no sampled constructor rows can become a prior.
    """
    census = contract.source_variant_census(source)
    if (source.metadata.source_hash != expected_source_hash or
            census['source_rows_sha256'] != expected_source_rows_sha256 or
            census['species_over_capacity']):
        raise ValueError('original public source identity/ordered capacity differs')
    vocabulary = metadata['vocabulary']
    for name in ('species', 'moves', 'abilities', 'items', 'types'):
        values = vocabulary[name]
        if (not isinstance(values, (list, tuple)) or not values or
                any(type(item) is not str for item in values) or
                len(set(values)) != len(values)):
            raise ValueError('exact engine namespace required: ' + name)
    # No caller-owned mutable maps survive startup. Original frozen variant
    # objects and metadata are retained; nested source facts are copied/frozen.
    copied = Gen3RandbatSource.from_payload(copy.deepcopy(source.to_payload()))
    copied = replace(copied, universes=MappingProxyType(dict(copied.universes)),
        move_metadata=_freeze(dict(copied.move_metadata)),
        species_metadata=_freeze(dict(copied.species_metadata)))
    namespaces = {name: list(vocabulary[name]) for name in
        ('species', 'moves', 'abilities', 'items', 'types')}
    binding = hashlib.sha256(json.dumps((SCHEMA, expected_source_hash,
        expected_source_rows_sha256, namespaces), sort_keys=True,
        separators=(',', ':')).encode()).hexdigest()
    return Source(expected_source_hash, expected_source_rows_sha256, binding,
        _freeze(namespaces), copied)


@dataclass(frozen=True)
class Root:
    address: contract.DecisionAddress
    own: object                    # Only own indexed private cells, never foe.
    view: object                   # Host snapshot of safe public readout.
    public: object                 # Owner-safe REDACTED carry; rebase foe cursor.
    source_binding_sha256: str
    reply_private_history_available: bool = False


def _redact_peer_private(cloned, owner):
    """Preserve learner readout, without preserving another owner's knowledge.

    These are NOT generic zeros meaning known history. The unavailable halves
    cannot feed a reply policy until the explicit history-availability contract
    is met. The real foe's private coordinate assignment is never a world map.
    """
    foe = 1-owner
    history = cloned.damage_history
    if history is not None:
        # Shown losses are public. Exact victim losses belong only to that
        # victim's owner: learner dealt measures the peer's private HP, while
        # peer dealt measures the learner's own HP. The unavailable halves are
        # declared unknown-zero, not reconstructed past sampled-world history.
        history = history._replace(hp=history.hp.at[:, foe].set(0),
            maximum=history.maximum.at[:, foe].set(0),
            taken=history.taken.at[:, foe, 1].set(0),
            dealt=history.dealt.at[:, owner, 1].set(0))
    return cloned._replace(
        damage_history=history,
        physical_to_public=cloned.physical_to_public.at[:, foe].set(-1),
        active_physical=cloned.active_physical.at[:, foe].set(-1),
        own_order=cloned.own_order.at[:, foe].set(-1),
        damage_exact_taken=None if cloned.damage_exact_taken is None else
            cloned.damage_exact_taken.at[:, foe].set(0),
        damage_exact_dealt=None if cloned.damage_exact_dealt is None else
            cloned.damage_exact_dealt.at[:, owner].set(0))


def detach_root(context, source, *, row, owner, enable_device_worlds=False):
    """Detach an actual RootContext through the allowed owner/public boundary.

    The returned object contains no RootContext or live BattleState reference.
    Indexing happens BEFORE host conversion. Both later worlds share this
    root-public clone; neither may initialize trackers from sampled private data.
    """
    from .v5_device_rollout import RootContext
    if (enable_device_worlds is not True or type(context) is not RootContext
            or type(source) is not Source):
        raise ValueError('explicit actual device root context required')
    if (type(row) is not int or type(owner) is not int or owner not in (0, 1)
            or not 0 <= row < len(context.streams)):
        raise ValueError('actual root row and owner0/1 required')
    cloned = public.clone_for_worlds(context.public, j.asarray([row], j.int32))
    view = public.readout(cloned, j.asarray([owner], j.int32))
    own = {name: getattr(context.state, name)[row, owner]
        for name in OWN_MON_FIELDS + OWN_SIDE_FIELDS}
    address = contract.DecisionAddress(int(context.streams[row]),
        int(context.frames[row]), owner)
    snapshot = _freeze({**dict(view._asdict()),
        'hp_percent': cloned.hp_percent[:, 1-owner]})
    # Original opponent-own materialization admits the full-health birth only
    # when no prior private HP/status history is needed. An actual frame0 is
    # that boundary. Later unknown exact-damage history is not fabricated from
    # displayed fractions or read from the live foe's supplemental arrays.
    history_available = (address.frame == 0 and
        not np.any(np.asarray(view.error)) and
        np.all(np.asarray(view.hp_fraction)[np.asarray(view.present)] == 1) and
        not np.any(np.asarray(view.status)[np.asarray(view.present)]))
    return Root(address, _freeze(own), snapshot, _redact_peer_private(cloned, owner),
        source.binding_sha256, bool(history_available))


@dataclass(frozen=True)
class Start:
    world: int
    seed: int
    opponent_fixtures: tuple | None
    # Each revealed slot keeps the original full source variant index.
    revealed_source_indices: tuple
    refusals: tuple[str, ...]


@dataclass(frozen=True)
class Plans:
    root: Root
    selected: bool
    starts: tuple
    source_hash: str
    source_rows_sha256: str
    source_fixture_unknown_mass: float
    # Source fixture availability is deliberately not current-world readiness.
    current_world_unknown_mass: float = 1.0
    device_branch_ready: bool = False
    teacher_ready: bool = False
    next_consumer: str = ('Map source fixtures plus retained public current '
        'HP/status/PP/conditions and own exact cells to hypothetical BattleState; '
        'use original latent treatment or explicit refusal for unavailable '
        'clocks, then SAME device_public_leaf and completed-Q consumer.')


def _array(view, name, shape, dtype):
    value = view.get(name)
    if not isinstance(value, np.ndarray) or value.shape != shape or value.dtype != dtype:
        raise ValueError('public field unavailable/malformed: ' + name)
    return value


def _name(source, kind, value, *, absent=False):
    names = source.vocabulary[kind]
    if not 0 <= int(value) < len(names) or (not absent and int(value) == 0):
        raise ValueError('public namespace unavailable: ' + kind)
    return None if int(value) == 0 else names[int(value)]


def _revealed(root, source):
    """Join hard source bits to current public facts, never private set labels."""
    view = root.view
    if np.any(_array(view, 'error', (1,), np.dtype('int32'))):
        raise ValueError('public tracker input error')
    species = _array(view, 'species', (1, 6), np.dtype('int32'))[0]
    present = _array(view, 'present', (1, 6), np.dtype('bool'))[0]
    masks = _array(view, 'belief_mask', (1, 6, 2), np.dtype('uint32'))[0]
    moves = _array(view, 'revealed_moves', (1, 6, 4), np.dtype('int32'))[0]
    ability = _array(view, 'ability', (1, 6), np.dtype('int32'))[0]
    item = _array(view, 'item', (1, 6), np.dtype('int32'))[0]
    gender = _array(view, 'gender', (1, 6), np.dtype('int32'))[0]
    level = _array(view, 'level', (1, 6), np.dtype('int32'))[0]
    features = view.get('mon_features')
    swapped = None if features is None else features.get('item_swapped')
    if swapped is not None and (swapped.shape != (1, 6) or
            swapped.dtype not in (np.dtype('int32'), np.dtype('bool'))):
        raise ValueError('public item mutation facts malformed')
    result, seen = [], set()
    for slot in range(6):
        if not present[slot]:
            if species[slot] != 0 or np.any(masks[slot]):
                raise ValueError('unrevealed slot exposes source identity')
            continue
        name = _name(source, 'species', species[slot])
        universe = source.original.universe_for(name)
        if universe is None or not universe.variants:
            raise ValueError('revealed species has no original public source')
        key = universe.species
        if key in seen:
            raise ValueError('duplicate public species cannot fill distinct randbat slots')
        seen.add(key)
        count = len(universe.variants)
        raw = int(masks[slot, 0]) | int(masks[slot, 1]) << 32
        if count > 64 or raw >> count:
            raise ValueError('public variant bits outside original source axis')
        if not raw:
            raise ValueError('zero current public variant mass; no retry/fallback prior')
        shown_moves = tuple(_name(source, 'moves', value) for value in moves[slot] if value)
        # The tracker stores one opaque HP representative in the engine
        # namespace. Its alias spelling is NOT a publicly revealed HP type.
        # Original belief also excludes the forced pseudo-move Struggle.
        revealed_moves = tuple('hiddenpower' if normalize_id(name).startswith('hiddenpower')
            else name for name in shown_moves if normalize_id(name) != 'struggle')
        known_ability = _name(source, 'abilities', ability[slot], absent=True)
        known_item = _name(source, 'items', item[slot], absent=True)
        if known_item is not None and swapped is None:
            raise ValueError('current item is known but original-item mutation state unavailable')
        if swapped is not None and swapped[0, slot]:
            known_item = None  # Original sampler excludes mutated current items.
        eligible = tuple((index, variant) for index, variant in enumerate(universe.variants)
            if raw & (1 << index) and variant.matches(revealed_moves=revealed_moves,
                revealed_ability=known_ability, revealed_item=known_item)
            and variant.level == int(level[slot]))
        if not eligible:
            raise ValueError('current public facts have no source-consistent variant mass')
        if gender[slot] not in (-1, 0, 1, 2):
            raise ValueError('public gender namespace malformed')
        shown_gender = {-1: None, 0: 'N', 1: 'M', 2: 'F'}[int(gender[slot])]
        result.append((slot, universe, eligible, shown_gender, name))
    return tuple(result)


def plan_starts(root, source, recipe):
    """Original recipe address selection and two shared source fixture draws.

    The canonical sampler's revealed draw order is gender then uniform variant;
    hidden slots use its original without-replacement species/variant sampler.
    Source masks never become a synthetic generator-frequency prior. Refused
    slots are retained at weight1/2, never replaced or survivor-renormalized.
    """
    if type(root) is not Root or type(source) is not Source or type(recipe) is not Recipe:
        raise ValueError('typed detached root, bound public source and original recipe required')
    if root.source_binding_sha256 != source.binding_sha256:
        raise ValueError('root public source/engine namespace binding differs')
    if not recipe.selected(root.address):
        return Plans(root, False, (), source.source_hash, source.source_rows_sha256, 0.0)
    try:
        revealed = _revealed(root, source)
        pre_error = None
    except ValueError as error:
        revealed, pre_error = (), str(error)
    starts = []
    for world in (0, 1):
        seed = recipe.world_seed(root.address, world)
        fixtures, indices, failure = [], [], pre_error
        rng = random.Random(seed)
        if failure is None:
            for slot, universe, candidates, shown_gender, shown_species in revealed:
                gender = shown_gender or _sample_gender(shown_species,
                    set_source=source.original, rng=rng)
                index, variant = candidates[rng.randrange(len(candidates))]
                fixture = _fixture_from_variant_payload(variant.to_summary(),
                    fallback_species=shown_species, set_source=source.original, gender=gender)
                if fixture is None:
                    failure = 'original revealed fixture spread unavailable'
                    break
                fixtures.append(fixture)
                indices.append((slot, index))
            if failure is None:
                hidden = _sample_hidden_backline(source.original,
                    used_species={_normalize_species_id(row.species) for row in fixtures},
                    count=6-len(fixtures), rng=rng)
                if hidden is None:
                    failure = 'original hidden species/variant source unavailable'
                else:
                    fixtures.extend(hidden)
        starts.append(Start(world, seed, None if failure else tuple(fixtures),
            tuple(indices), () if failure is None else (failure,)))
    return Plans(root, True, tuple(starts), source.source_hash, source.source_rows_sha256,
        sum(start.opponent_fixtures is None for start in starts) / 2)
