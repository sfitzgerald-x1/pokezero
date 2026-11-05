"""Once-per-source device tables for the D3 device world preparation kernel (V2ePrepWorlds).

The bound public source (device_public_worlds.Source), its engine namespaces and move PP
(device_public_world_state.Tables) are flattened into int32 tables: universes in
set_source.universes.values() order, their variants, per-species fixed genders, the species
namespace -> universe join used by _revealed, and per (variant, hidden/revealed) the exact
materialization writes or refusal string that device_public_world_state._write_fixture produces
for that fixture. Nothing here samples or materializes a world; the oracle modules
(pokezero.determinization, pokezero.randbat) are called, never reimplemented. The layout mirrors
gpu_engine/cuda/v2e_prep.cuh (same enumerations, same order).
"""
from dataclasses import dataclass

import numpy as np

from pokezero.dex import normalize_id
from pokezero.determinization import (_fixture_from_variant, _fixture_from_variant_payload,
    _sample_gender)
from . import device_public_world_state as mapping
from .public_active import NAMES as VOLATILES

SLOTS, MOVE_SLOTS, BOOSTS, FEATURE_COUNT, SEED_WORDS, MAX_UNIVERSES = 6, 4, 7, 8, 8, 1024


def _enum(names):
    return {name: index for index, name in enumerate(names)}


UNI_COLUMNS = ('U_SPECIES', 'U_VAR_OFF', 'U_VAR_CNT', 'U_GENDER')
VAR_COLUMNS = ('V_LEVEL', 'V_MOVE0', 'V_MOVE1', 'V_MOVE2', 'V_MOVE3', 'V_HAS_HP', 'V_ABILITY', 'V_ITEM', 'V_SPECIES',
    'V_STAT0', 'V_STAT1', 'V_STAT2', 'V_STAT3', 'V_STAT4', 'V_STAT5', 'V_WMOVE0', 'V_WMOVE1', 'V_WMOVE2', 'V_WMOVE3',
    'V_WNMOVES', 'V_PP0', 'V_PP1', 'V_PP2', 'V_PP3', 'V_TYPE0', 'V_TYPE1', 'V_HP_TYPE', 'V_HP_POWER', 'V_CHOICEBAND',
    'V_REFUSAL', 'V_FIXTURE_NONE')
MISC_COLUMNS = ('MISC_STRUGGLE', 'MISC_TRUANT', 'MISC_UNIVERSES', 'MISC_VARIANTS', 'MISC_SPECIES', 'MISC_MOVES',
    'MISC_ABILITIES', 'MISC_ITEMS', 'MISC_VOLATILES')
FEATURES = ('toxic_stage', 'turns_active', 'stall_counter', 'confusion_turns', 'encore_turns', 'sub_present',
    'meanlook_trap', 'item_swapped')
FLAGS = ('F_BORROWED', 'F_REMOVED', 'F_PP_USED', 'F_LAST_MOVES', 'F_VOLATILES', 'F_MON_FEATURES', 'F_OWN_MEANLOOK',
    'F_FIELDS')
FEATURE_FLAG_BASE = 8
INPUTS = ('IN_MISC', 'IN_UNI', 'IN_VAR', 'IN_SPECIES_TO_UNIVERSE', 'IN_MOVE_IS_HP', 'IN_OWNER', 'IN_SPECIES',
    'IN_PRESENT', 'IN_ACTIVE', 'IN_BELIEF', 'IN_REVEALED_MOVES', 'IN_ABILITY', 'IN_ITEM', 'IN_GENDER', 'IN_LEVEL',
    'IN_ERROR', 'IN_HP_FRACTION', 'IN_HP_PERCENT', 'IN_STATUS', 'IN_BOOSTS', 'IN_TYPES', 'IN_BORROWED', 'IN_REMOVED',
    'IN_PP_USED', 'IN_LAST_MOVES', 'IN_VOLATILES', 'IN_WEATHER', 'IN_TURN', 'IN_SPIKES', 'IN_OWN_ORDER', 'IN_FEATURES',
    'IN_OWN_MEANLOOK', 'IN_FIELDS', 'IN_OWN_ACTIVE', 'IN_OWN_TRAP', 'IN_OWN_FLAGS', 'IN_SEEDS', 'IN_COUNT')
MON_COLUMNS = ('M_SPECIES', 'M_LEVEL', 'M_HP', 'M_MAX_HP', 'M_GENDER', 'M_ABILITY', 'M_BASE_ABILITY', 'M_ITEM',
    'M_STATUS', 'M_STATUS_COUNTER', 'M_HP_TYPE', 'M_HP_POWER', 'M_CHOICE_MOVE', 'M_LAST_MOVE', 'M_PROTECT_COUNTER',
    'M_TRAP_SOURCE', 'M_VOLATILE', 'M_TYPE0', 'M_TYPE1', 'M_BTYPE0', 'M_BTYPE1', 'M_MOVE0', 'M_MOVE1', 'M_MOVE2',
    'M_MOVE3', 'M_PP0', 'M_PP1', 'M_PP2', 'M_PP3', 'M_STAT0', 'M_STAT1', 'M_STAT2', 'M_STAT3', 'M_STAT4', 'M_STAT5',
    'M_BOOST0', 'M_BOOST1', 'M_BOOST2', 'M_BOOST3', 'M_BOOST4', 'M_BOOST5', 'M_BOOST6')
SIDE_COLUMNS = ('S_ACTIVE', 'S_ACTIVE_TURNS', 'S_LAST_KIND', 'S_LAST_INDEX', 'S_SPIKES0', 'S_SPIKES1', 'S_WEATHER',
    'S_WEATHER_COUNTER', 'S_TURN', 'S_OWN_TRAP0', 'S_OWN_TRAP1', 'S_OWN_TRAP2', 'S_OWN_TRAP3', 'S_OWN_TRAP4',
    'S_OWN_TRAP5')
FIX_COLUMNS = ('FX_UNIVERSE', 'FX_VARIANT', 'FX_GENDER', 'FX_SLOT')
CODE_COLUMNS = ('C_PLAN', 'C_MATERIALIZE')
# Static refusal strings by code (index 0: none); dynamic strings follow in each DeviceSource.
STATIC_REFUSALS = (None,
    'public tracker input error', 'unrevealed slot exposes source identity', 'public namespace unavailable: species',
    'revealed species has no original public source', 'duplicate public species cannot fill distinct randbat slots',
    'public variant bits outside original source axis', 'zero current public variant mass; no retry/fallback prior',
    'public namespace unavailable: moves', 'public namespace unavailable: abilities', 'public namespace unavailable: items',
    'current item is known but original-item mutation state unavailable',
    'current public facts have no source-consistent variant mass', 'public gender namespace malformed',
    'original revealed fixture spread unavailable', 'original hidden species/variant source unavailable',
    'suspended/replacement/entry boundary not materialized', 'source fixtures/current public encounter slots differ',
    'one currently known public opposing active required', 'owner cross-party trap link has no public current source',
    'literal public integer HP percentage unavailable', 'public current HP fraction malformed',
    'opposing sleep counter/provenance unavailable', 'public current item removal state unavailable',
    'public current PP-use counts unavailable', 'public PP-use/source move join unavailable',
    'public current clock not materialized: confusion_turns', 'public current clock not materialized: encore_turns',
    'public substitute HP/origin not materialized', 'public volatile bag unavailable', 'public caller identity unavailable',
    'public last caller/source slot unavailable', 'public Truant phase not materialized',
    'opposing faint/replacement boundary not materialized', 'public current field clocks unavailable',
    'public Wish caster amount unavailable', 'public weather/turn namespace malformed', 'public weather duration unavailable')
STATIC_REFUSAL_NAMES = ('R_NONE',
    'R_TRACKER_ERROR', 'R_UNREVEALED_EXPOSES', 'R_NS_SPECIES', 'R_NO_SOURCE', 'R_DUPLICATE_SPECIES', 'R_BITS_OUTSIDE',
    'R_ZERO_MASS', 'R_NS_MOVES', 'R_NS_ABILITIES', 'R_NS_ITEMS', 'R_ITEM_SWAP_UNAVAILABLE', 'R_NO_CONSISTENT',
    'R_GENDER_NS', 'R_REVEALED_FIXTURE', 'R_HIDDEN_SOURCE', 'R_BOUNDARY', 'R_ENCOUNTER_DIFFER', 'R_ONE_ACTIVE',
    'R_OWNER_TRAP', 'R_HP_PERCENT_UNAVAILABLE', 'R_HP_MALFORMED', 'R_SLEEP', 'R_ITEM_REMOVED_UNAVAILABLE',
    'R_PP_UNAVAILABLE', 'R_PP_JOIN', 'R_CLOCK_CONFUSION', 'R_CLOCK_ENCORE', 'R_SUBSTITUTE', 'R_VOLATILE_BAG',
    'R_LAST_MOVES_UNAVAILABLE', 'R_LAST_CALLER', 'R_TRUANT', 'R_FAINT', 'R_FIELDS', 'R_WISH', 'R_WEATHER_NS',
    'R_WEATHER_DURATION')
VOL_LEECH_SEED, VOL_INGRAIN, VOL_TRAPPED = -1, -2, -3
U, V, MISC, IN, M, S, FX, C = (_enum(x) for x in (UNI_COLUMNS, VAR_COLUMNS, MISC_COLUMNS, INPUTS, MON_COLUMNS,
    SIDE_COLUMNS, FIX_COLUMNS, CODE_COLUMNS))
R = _enum(STATIC_REFUSAL_NAMES)
F = {name: 1 << index for index, name in enumerate(FLAGS)}
PADDING = -1   # plan code of a padding root


def header_enumerations():
    """Every (name, value) the CUDA header must agree with (tests parse v2e_prep.cuh)."""
    values = {}
    for table in (U, V, MISC, IN, M, S, FX, C, R):
        values.update(table)
    values.update(U_COLUMNS=len(UNI_COLUMNS), V_COLUMNS=len(VAR_COLUMNS), MISC_HEADER=len(MISC_COLUMNS),
        IN_COUNT_ARGS=len(INPUTS), M_COLUMNS=len(MON_COLUMNS), S_COLUMNS=len(SIDE_COLUMNS), FX_COLUMNS=len(FIX_COLUMNS),
        C_COLUMNS=len(CODE_COLUMNS), R_STATIC_COUNT=len(STATIC_REFUSALS), kSlots=SLOTS, kMoveSlots=MOVE_SLOTS,
        kBoosts=BOOSTS, kFeatures=FEATURE_COUNT, kSeedWords=SEED_WORDS, kMaxUniverses=MAX_UNIVERSES,
        VOL_LEECH_SEED=VOL_LEECH_SEED, VOL_INGRAIN=VOL_INGRAIN, VOL_TRAPPED=VOL_TRAPPED, F_FEATURE_BASE=FEATURE_FLAG_BASE)
    values.update({name: 1 << index for index, name in enumerate(FLAGS)})
    values.update({'FT_' + name.upper(): index for index, name in enumerate(FEATURES)})
    return values


class _FixedDraw:
    """random.Random stand-in whose randrange always returns one value (fixed-gender probing)."""
    def __init__(self, value):
        self.value = value

    def randrange(self, n):
        return self.value


@dataclass(frozen=True)
class DeviceSource:
    tables: mapping.Tables
    universe_keys: tuple          # set_source.universes keys, in order
    variant_ids: tuple            # per universe: tuple of variant_id strings
    uni: np.ndarray               # [U, U_COLUMNS] int32
    var: np.ndarray               # [2, Vn, V_COLUMNS] int32 (case 0 hidden, case 1 revealed)
    species_to_universe: np.ndarray   # [S] int32, -1 when the vocabulary species has no universe
    move_is_hp: np.ndarray        # [M] int32
    misc: np.ndarray              # [MISC_HEADER + V + FEATURE_COUNT] int32
    strings: tuple                # refusal strings by code; index 0 is None
    volatile_count: int


def _gender_code(species, original):
    """-1 when _sample_gender draws, else the fixed code (0 N, 1 M, 2 F)."""
    first = _sample_gender(species, set_source=original, rng=_FixedDraw(0))
    second = _sample_gender(species, set_source=original, rng=_FixedDraw(1))
    if first != second:
        return -1
    return {'N': 0, 'M': 1, 'F': 2}[first]


def compile_device_source(tables):
    """Flatten the bound source and namespaces into the kernel's tables (host numpy)."""
    if type(tables) is not mapping.Tables:
        raise ValueError('original device materializer tables required')
    source = tables.source
    original = source.original
    names = tables.namespaces
    vocabulary = source.vocabulary
    strings = list(STATIC_REFUSALS)
    interned = {}

    def intern(text):
        if text not in interned:
            interned[text] = len(strings)
            strings.append(text)
        return interned[text]
    keys = tuple(original.universes)
    universes = tuple(original.universes.values())
    if len(universes) > MAX_UNIVERSES:
        raise ValueError('device source tables: more universes than the kernel bitmask capacity')
    index_of = {key: index for index, key in enumerate(keys)}
    species_to_universe = np.full(len(vocabulary['species']), -1, np.int32)
    representative = {}
    for species_id, name in enumerate(vocabulary['species']):
        if species_id == 0:
            continue
        universe = original.universe_for(name)
        if universe is None:
            continue
        key = next(k for k, u in original.universes.items() if u is universe)
        species_to_universe[species_id] = index_of[key]
        representative.setdefault(index_of[key], name)
    uni = np.zeros((len(universes), len(UNI_COLUMNS)), np.int32)
    rows = []
    variant_ids = []
    offset = 0
    for u, universe in enumerate(universes):
        uni[u, U['U_SPECIES']] = names['species'].get(normalize_id(universe.species), 0)
        uni[u, U['U_VAR_OFF']] = offset
        uni[u, U['U_VAR_CNT']] = len(universe.variants)
        uni[u, U['U_GENDER']] = _gender_code(universe.species, original)
        variant_ids.append(tuple(variant.variant_id for variant in universe.variants))
        offset += len(universe.variants)
        for variant in universe.variants:
            if len(variant.moves) > MOVE_SLOTS:
                raise ValueError('device source tables: variant with more than four moves: ' + variant.variant_id)
            row = np.zeros((2, len(VAR_COLUMNS)), np.int32)
            row[:, V['V_LEVEL']] = variant.level
            for k, move in enumerate(variant.moves):
                row[:, V['V_MOVE0'] + k] = names['moves'].get(normalize_id(move), 0)
            row[:, V['V_HAS_HP']] = int(any(normalize_id(move).startswith('hiddenpower') for move in variant.moves))
            row[:, V['V_ABILITY']] = names['abilities'].get(normalize_id(variant.ability), 0)
            row[:, V['V_ITEM']] = names['items'].get(normalize_id(variant.item), 0)
            row[:, V['V_HP_POWER']] = 70
            for case in (0, 1):
                if case == 0:
                    mon = _fixture_from_variant(variant, set_source=original, rng=_FixedDraw(0))
                else:
                    fallback = representative.get(u, universe.species)
                    mon = _fixture_from_variant_payload(variant.to_summary(), fallback_species=fallback,
                        set_source=original, gender='M')
                if mon is None:
                    row[case, V['V_FIXTURE_NONE']] = 1
                    continue
                arrays = mapping._empty_arrays()
                address = (0, 1, 0)
                try:
                    mapping._write_fixture(tables, mon, arrays, address)
                except (ValueError, KeyError, IndexError, TypeError) as error:
                    row[case, V['V_REFUSAL']] = intern(str(error))
                    continue
                row[case, V['V_SPECIES']] = 0 if case else int(arrays['species'][address])
                row[case, V['V_STAT0']:V['V_STAT5'] + 1] = arrays['stats'][address]
                row[case, V['V_WMOVE0']:V['V_WMOVE3'] + 1] = arrays['moves'][address]
                row[case, V['V_WNMOVES']] = len(mon.moves)
                row[case, V['V_PP0']:V['V_PP3'] + 1] = arrays['pp'][address]
                row[case, V['V_TYPE0']:V['V_TYPE1'] + 1] = arrays['types'][address]
                row[case, V['V_HP_TYPE']] = int(arrays['hidden_power_type'][address])
                row[case, V['V_HP_POWER']] = int(arrays['hidden_power_power'][address])
                row[case, V['V_CHOICEBAND']] = int(normalize_id(mon.item or '') == 'choiceband')
                if (int(arrays['ability'][address]) != row[case, V['V_ABILITY']]
                        or int(arrays['item'][address]) != row[case, V['V_ITEM']]
                        or int(arrays['level'][address]) != variant.level):
                    raise ValueError('device source tables: fixture namespace ids differ from variant matching ids')
            rows.append(row)
    var = np.stack(rows, axis=1) if rows else np.zeros((2, 0, len(VAR_COLUMNS)), np.int32)
    move_is_hp = np.asarray([int(normalize_id(name).startswith('hiddenpower')) for name in vocabulary['moves']], np.int32)
    volatile = []
    for name in VOLATILES:
        if name == 'leechseed':volatile.append(VOL_LEECH_SEED)
        elif name == 'ingrain':volatile.append(VOL_INGRAIN)
        elif name == 'trapped':volatile.append(VOL_TRAPPED)
        else:volatile.append(intern('public volatile/latent clock not materialized: ' + name))
    features = [intern('public current feature unavailable: ' + name) for name in FEATURES]
    header = [names['moves'].get('struggle', 0), names['abilities'].get('truant', 0), len(universes), int(var.shape[1]),
        len(vocabulary['species']), len(vocabulary['moves']), len(vocabulary['abilities']), len(vocabulary['items']),
        len(VOLATILES)]
    misc = np.asarray(header + volatile + features, np.int32)
    return DeviceSource(tables, keys, tuple(variant_ids), uni, var, species_to_universe, move_is_hp, misc,
        tuple(strings), len(VOLATILES))
