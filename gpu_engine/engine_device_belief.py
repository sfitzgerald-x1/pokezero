"""Source-only public belief algebra for the direct device encoder.

No engine/world argument exists here. Masks are original source predicates;
all changing facts are supplied by the public tracker. No fold or event ledger
is constructed. Tables are compiled once with the worker's immutable catalog.
"""
from typing import NamedTuple
import jax
import jax.numpy as j
import numpy as np

from pokezero.dex import normalize_id
from . import public_belief_features as original
from . import public_expected_stats as expected


class Tables(NamedTuple):
    # Directly usable by device-public reveal hooks; engine-ID namespaces.
    species_initial_mask: object
    move_mask: object
    ability_mask: object
    item_mask: object
    source_species: object
    variant_valid: object
    variant_ability: object
    variant_item: object
    variant_move: object
    ability_category: object
    item_category: object
    move_category: object
    engine_ability: object
    engine_item: object
    engine_move: object
    max_pp: object
    spreads: object
    spread_status: object
    base_stats: object
    info_present: object
    exact_hp: object
    flying: object
    trapper: object
    early_bird: object
    levitate: object
    species_types: object
    public_move: object
    source_move_evidence: object
    public_foe_targeted: object
    public_pressure: object
    standing_pins_enabled: object  # exact resolved checkpoint provenance bit


def compile_tables(metadata, catalog):
    """Join existing compiled source tables; never sample/materialize a world."""
    v = metadata['vocabulary']; source = catalog.source; feature = catalog.features
    sid = np.asarray([source.species_id(n) if i else 0 for i, n in enumerate(v['species'])], np.int32)
    def predicates(kind, names):
        ids = [source.predicate_id(kind, 'hiddenpower' if kind == 'move' and normalize_id(n).startswith('hiddenpower') else n)
            if i else 0 for i, n in enumerate(names)]
        result = np.asarray(source.device.predicates)[ids][:, sid].transpose(1, 0, 2).copy()
        # ID0 is absent evidence, not an unknown nonempty source predicate.
        result[:, 0] = np.asarray(source.device.pool)[sid]
        return result
    def ids(names, mapping):return np.asarray([mapping.get(normalize_id(n), 0) if i else 0 for i, n in enumerate(names)], np.int32)
    pub = np.asarray([catalog.species.ids.get(normalize_id(n), 0) if i else 0 for i, n in enumerate(v['species'])], np.int32)
    e = catalog.expected.device
    floating = {'base_stats'}
    from pokezero import showdown as reference
    type_ids = {normalize_id(n): i for i, n in enumerate(v['types'])}
    species_types = []
    for i, name in enumerate(v['species']):
        info = reference._species_info_base_fallback(catalog.dex, catalog.species.names[pub[i]]) if i else None
        kinds = () if info is None else info.types
        species_types.append([type_ids.get(normalize_id(kinds[k]), 0) if k < len(kinds) else 0 for k in (0, 1)])
    hp_ids = [i for i, name in enumerate(v['moves']) if normalize_id(name).startswith('hiddenpower')]
    generic_hp = next((i for i in hp_ids if normalize_id(v['moves'][i]) == 'hiddenpower'), hp_ids[0] if hp_ids else 0)
    public_move = [generic_hp if i in hp_ids else i for i in range(len(v['moves']))]
    engine_moves = ids(v['moves'], catalog.moves.ids)
    for i in hp_ids:engine_moves[i] = catalog.moves.ids.get('hiddenpower', 0)
    arrays = (np.asarray(source.device.pool)[sid], predicates('move', v['moves']),
        predicates('ability', v['abilities']), predicates('item', v['items']), sid,
        np.asarray(feature.variant_valid), np.asarray(feature.variant_ability), np.asarray(feature.variant_item),
        np.asarray(feature.variant_move), np.asarray(feature.ability_category), np.asarray(feature.item_category),
        np.asarray(feature.move_category), ids(v['abilities'], catalog.abilities.ids),
        ids(v['items'], catalog.items.ids), engine_moves, np.asarray(feature.move_tables.max_pp),
        np.asarray(e.spreads), np.asarray(e.spread_status), np.asarray(e.base)[pub], np.asarray(e.info_present)[pub],
        np.asarray(e.exact_hp)[pub], np.asarray(catalog.hazard.device.exact_flying)[pub],
        np.asarray(catalog.exact.device.trapper), np.asarray(catalog.exact.device.early_bird), np.asarray(catalog.hazard.device.levitate),
        np.asarray(species_types, np.int32), np.asarray(public_move, np.int32),
        np.asarray([bool(i) and normalize_id(n) != 'struggle' for i, n in enumerate(v['moves'])], bool),
        np.asarray(feature.move_tables.foe_targeted)[engine_moves],
        np.asarray([bool(i) and normalize_id(n) == 'pressure' for i, n in enumerate(v['abilities'])], bool),
        np.asarray(catalog.masks.investment_belief_narrowing,bool))
    # Preserve primitive dtype (especially uint32 bitmasks and compact spreads).
    return Tables(*(j.asarray(a, j.float32 if n in floating else None) for n, a in zip(Tables._fields, arrays)))


def _take(table, ids):return table[j.clip(ids, 0, table.shape[0] - 1)]


def _optional(view, name, default):
    value = getattr(view, name, None)
    return default if value is None else value


class Result(NamedTuple):
    numeric: object
    categorical: object
    weights: object
    effective: object
    count: object
    uncertainty: object
    reason: object
    pin_conflict: object
    certain_ability: object
    error: object


def project(view, tables):
    """Original source fallback, lexical buckets and exact revealed PP use."""
    species = view.species; b = species.shape[0]
    source_id = _take(tables.source_species, species)
    pool = _take(tables.species_initial_mask, species)
    hard = _optional(view, 'belief_mask', pool)
    if hard.shape != (b, 6, 2) or hard.dtype != j.uint32:
        raise ValueError('public hard mask requires Bx6x2 uint32')
    malformed = j.any((hard & ~pool) != 0, axis=(1, 2))
    hard = hard & pool
    known = j.any(pool != 0, axis=-1)
    inconsistent = known & ~j.any(hard != 0, axis=-1)
    base = j.where(inconsistent[..., None], pool, hard)
    pin = _optional(view, 'belief_pin', pool)
    pin_active = _optional(view, 'belief_pin_active', j.zeros((b, 6), bool))
    if pin.shape != (b, 6, 2) or pin.dtype != j.uint32 or pin_active.shape != (b, 6) or pin_active.dtype != j.bool_:
        raise ValueError('public pin requires Bx6x2 uint32 and Bx6 bool activity')
    enabled=tables.standing_pins_enabled
    if enabled.shape!=() or enabled.dtype!=j.bool_:raise ValueError('resolved standing-pin capability requires scalar bool')
    # The actual frozen B2/champion catalog rejects a true provenance mask.
    # Optional pin transport is not permission to enable either Tier2 producer.
    # Keep its typed/namespace checks, but disabled pins cannot change any view.
    pin_active=pin_active&enabled
    malformed |= j.any((pin & ~pool) != 0, axis=(1, 2))
    pin = pin & pool
    intersection = base & pin
    conflict = known & ~inconsistent & pin_active & ~j.any(intersection != 0, axis=-1)
    effective = j.where((known & ~inconsistent & pin_active & ~conflict)[..., None], intersection, base)
    maximum = tables.variant_valid.shape[-1]
    selected = original._bits(effective, maximum)
    count = j.sum(selected, axis=-1).astype(j.int32)
    total = j.sum(jax.lax.population_count(pool), axis=-1)
    uncertainty = j.where(known & ~inconsistent, count / j.maximum(total, 1), 1).astype(j.float32)
    reason = j.where(~known, 1, j.where(inconsistent, 2, 0)).astype(j.int32)
    ability = _take(tables.variant_ability, source_id)
    item = _take(tables.variant_item, source_id)
    moves = _take(tables.variant_move, source_id).reshape(b, 6, -1)
    length = moves.shape[-1]
    base_selected = original._bits(j.where(inconsistent[..., None], effective, hard), maximum)
    ability_first = original._first_occurrences(ability, selected, tables.ability_category.shape[0])
    item_first = original._first_occurrences(item, selected, tables.item_category.shape[0])
    possible_first = original._first_occurrences(moves, j.repeat(selected, 4, axis=-1), tables.move_category.shape[0])
    base_first = original._first_occurrences(moves, j.repeat(base_selected, 4, axis=-1), tables.move_category.shape[0])
    possible_move = possible_first < length
    possible_ability = ability_first < maximum; possible_item = item_first < maximum
    known_ability = _take(tables.engine_ability, view.ability)
    known_item = _take(tables.engine_item, view.item)
    revealed_engine = _optional(view, 'revealed_moves', j.zeros((b, 6, 4), j.int32))
    used = _optional(view, 'move_pp_used', j.zeros_like(revealed_engine))
    malformed |= j.any((revealed_engine < 0) | (revealed_engine >= tables.engine_move.shape[0]) | (used < 0), axis=(1, 2))
    revealed = _take(tables.engine_move, revealed_engine)
    ability_mask = j.where((known_ability > 0)[..., None], j.arange(tables.ability_category.shape[0]) == known_ability[..., None], possible_ability)
    item_mask = j.where((known_item > 0)[..., None], j.arange(tables.item_category.shape[0]) == known_item[..., None], possible_item)
    revealed_first = original._first_occurrences(revealed, revealed > 0, tables.move_category.shape[0])
    revealed_mask = revealed_first < revealed.shape[-1]
    revealed_count = j.sum(revealed > 0, axis=-1)
    candidates = possible_move & ~revealed_mask
    rank = j.argsort(j.argsort(j.where(candidates, base_first, length), axis=-1, stable=True), axis=-1, stable=True)
    extra = candidates & (rank < j.maximum(0, 16 - revealed_count)[..., None])
    move_buckets = original._buckets(revealed_mask | extra, 16)
    ability_buckets = original._buckets(ability_mask, 2); item_buckets = original._buckets(item_mask, 6)
    matches = (revealed[..., None, :] == move_buckets[..., :, None]) & (revealed[..., None, :] > 0)
    seen = j.any(matches, axis=-1) & (move_buckets > 0)
    uses = j.sum(j.where(matches, j.maximum(0, used[..., None, :]), 0), axis=-1)
    max_pp = _take(tables.max_pp, move_buckets)
    fractions = j.where(seen & (max_pp > 0), j.maximum(0, max_pp - uses) / j.maximum(max_pp, 1), 0)
    removed = _optional(view, 'item_removed', j.zeros((b, 6), bool))
    core = j.stack((revealed_count, count, uncertainty, ability_mask.sum(-1), item_mask.sum(-1),
        possible_move.sum(-1), known_ability > 0, (known_item > 0) & ~removed), axis=-1).astype(j.float32)
    values = j.concatenate((core, fractions, seen.astype(j.float32)), axis=-1)
    cats = j.concatenate((tables.ability_category[ability_buckets], tables.item_category[item_buckets], tables.move_category[move_buckets]), axis=-1)
    visible = view.present[..., None]
    numeric = j.zeros((b, 6, 132), j.float32).at[..., j.asarray(original.NUMERIC_COLUMNS)].set(j.where(visible, values, 0))
    categorical = j.zeros((b, 6, 41), j.int32).at[..., j.asarray(original.CATEGORY_COLUMNS)].set(j.where(visible, cats, 0))
    weights = selected.astype(j.float32) / j.maximum(count[..., None], 1)
    weights = j.pad(weights, ((0, 0), (0, 0), (0, 64 - maximum)))
    semantic_count = possible_ability.sum(axis=-1)
    certain = j.where(known_ability > 0, known_ability, j.where(semantic_count == 1, j.argmax(possible_ability, axis=-1), 0))
    return Result(numeric, categorical, j.where(visible, weights, 0), effective, count, uncertainty,
        reason, conflict, certain, malformed)


def expected_stats(view, result, tables, *, own_species=None, own_stats=None):
    """The same original baseline/spread bounds, never true opponent stats."""
    species = view.species; b = species.shape[0]; sid = _take(tables.source_species, species)
    level = j.clip(view.level, 1, 100)
    selected = original._bits(result.effective, tables.variant_valid.shape[-1])
    status = tables.spread_status[j.clip(sid, 0, tables.spreads.shape[0] - 1), level]
    maximum = selected.shape[-1]
    first = j.min(j.where(selected & (status != 1), j.arange(maximum), maximum), axis=-1)
    first_status = j.take_along_axis(j.pad(status, ((0, 0), (0, 0), (0, 1)), constant_values=1), first[..., None], axis=-1)[..., 0]
    use = (result.count > 0) & (first == maximum)
    spread = tables.spreads[sid, level].astype(j.int32)
    low = j.min(j.where(selected[..., None], spread, 65535), axis=-2)
    high = j.max(j.where(selected[..., None], spread, 0), axis=-2)
    base = _take(tables.base_stats, species).astype(j.int32)
    baseline = ((2 * base + 31 + 85 // 4) * level[..., None]) // 100 + 5
    baseline = baseline.at[..., 0].set(((2 * base[..., 0] + 31 + 85 // 4) * level) // 100 + level + 10)
    hp, atk = baseline[..., 0], baseline[..., 1]
    values = j.stack((hp, j.where(use, low[..., 0], hp), j.where(use, high[..., 0], hp),
        atk, j.where(use, low[..., 1], atk), j.where(use, high[..., 1], atk),
        *(j.where(base[..., n] > 0, j.where(use, high[..., n], baseline[..., n]), 0) for n in range(2, 6))), axis=-1)
    values = values.at[..., :6].set(j.where(((base[..., 0] > 0) & (base[..., 1] > 0))[..., None], values[..., :6], 0))
    values = j.where(_take(tables.info_present, species)[..., None], values, 0)
    transformed = _optional(view, 'transformed', j.zeros((b, 6), bool))
    copied = j.zeros_like(values)
    if own_species is not None and own_stats is not None:
        donor = _optional(view, 'transform_species', j.zeros((b, 6), j.int32))
        match = (donor[..., None] == own_species[:, None]) & (own_species[:, None] > 0)
        at = j.max(j.where(match, j.arange(6), -1), axis=-1)
        stats = own_stats[j.arange(b)[:, None], j.clip(at, 0, 5)]
        hp_base = _take(tables.exact_hp, species).astype(j.int32)
        trans_hp = j.where(hp_base > 0, ((2 * hp_base + 31 + 85 // 4) * level) // 100 + level + 10, 0)
        copied = j.stack((trans_hp, trans_hp, trans_hp, stats[..., 1], stats[..., 1], stats[..., 1],
            *(stats[..., n] for n in range(2, 6))), axis=-1)
        copied = j.where((at >= 0)[..., None], copied, 0)
    values = j.minimum(1., j.where(transformed[..., None], copied, values) / 714.)
    return j.where(view.present[..., None], values, 0).astype(j.float32), j.any(view.present & ~transformed & (first_status == 2), axis=-1)
