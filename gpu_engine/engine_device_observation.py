"""Direct device-state public29 inputs; independent of protocol capture/folds.

Startup binds immutable engine IDs to the actual checkpoint catalog. Decisions
read only the owning party and a separately maintained public opponent view.
Missing feature hooks write literal zero, listed by ``column_manifest``. Errors
are data: they never remove a row or turn a partial view into a capture abort.
This first cut is not an all-column parity or throughput qualification.
"""
from typing import NamedTuple

import jax
import jax.numpy as j
import numpy as np

from pokezero import showdown as reference
from pokezero.dex import normalize_id


SCHEMA = 'pokezero.engine_device_observation.v2'
TOKENS, NUMERIC, CATEGORICAL = 29, 132, 41
TOKEN_TYPES = (0,) + (1,) * 6 + (2,) * 6 + (3,) * 9 + (5,) + (7,) * 6
ERROR_KINDS = ('owner', 'own_ids', 'public_ids', 'order', 'public_tracker', 'posterior', 'nonfinite', 'belief', 'public_features')


class PublicView(NamedTuple):
    """One observer's public encounter order; never sampled hidden party facts.

    Species/type/ability/item IDs use the declared engine vocabulary. Own order
    is a request-slot -> physical-slot permutation. The optional posterior must
    come from the root-public tracker (also for sampled-world leaves).
    """
    own_order: object
    species: object
    level: object
    gender: object
    hp_fraction: object
    status: object
    active: object
    present: object
    fainted: object
    types: object
    boosts: object
    ability: object
    item: object
    error: object
    posterior_weights: object = None
    turn: object = None
    weather: object = None
    spikes: object = None
    belief_mask: object = None
    revealed_moves: object = None
    move_pp_used: object = None
    last_moves: object = None
    mon_features: object = None
    own_mon_features: object = None
    field_features: object = None
    stats_features: object = None
    own_last_moves: object = None
    borrowed_ability: object = None
    own_borrowed_ability: object = None
    volatiles: object = None
    own_volatiles: object = None
    item_removed: object = None
    belief_pin: object = None
    belief_pin_active: object = None
    transformed: object = None
    transform_species: object = None


class Tables(NamedTuple):
    species: object
    base_stats: object
    base_types: object
    types: object
    abilities: object
    items: object
    statuses: object
    move_primary: object
    move_numeric: object
    move_categories: object
    move_hp_kind: object
    hp_power: object
    max_pp: object
    fallback: object
    action_secondary: object
    action_role: object
    action_slots: object
    own_role: object
    foe_role: object
    field_role: object
    request_categories: object
    weather_categories: object
    prior: object
    prior_counts: object
    category_count: object
    ghost_type: object
    struggle_move: object
    belief: object = None
    features: object = None
    # Fidelity M1: the recharge request's one move row, {move: 'Recharge', id: 'recharge'}: the action
    # writer's primary category is move:recharge; 'recharge' is no dex move, so no mechanics or PP.
    recharge_primary: object = None


def compile_tables(metadata, *, catalog):
    """One startup compile from genuine constructor metadata + retained catalog.

    This reuses source dex/category algebra, not protocol state or admission.
    The returned primitive tables may be closed over by one cached JIT worker.
    """
    from . import v4_encoder
    from .public_moves import _static_category
    v4_encoder.check_catalog(catalog)
    vocab = metadata['vocabulary']
    def names(kind):
        values = vocab[kind]
        if (type(values) is not list or not values or values[0] not in ('', '_none')
                or any(type(x) is not str for x in values)
                or len({normalize_id(x) for x in values}) != len(values)):
            raise ValueError('ordered engine vocabulary required: ' + kind)
        return values
    species, types, abilities, items, moves = (names(k) for k in ('species', 'types', 'abilities', 'items', 'moves'))
    category = lambda text: _static_category(catalog.vocabulary, text) if text else 0
    species_ids = [catalog.species.ids.get(normalize_id(name), 0) if i else 0 for i, name in enumerate(species)]
    if any(i and not sid for i, sid in enumerate(species_ids)):
        raise ValueError('engine species missing from original source catalog')
    sd = catalog.species.device
    bases = np.asarray(sd.base_stats)[species_ids]
    base_types = np.asarray(sd.type_categories)[species_ids]
    action_ids = []
    for i, name in enumerate(moves):
        found = catalog.actions.ids.get(normalize_id(name), 0) if i else 0
        if i and not found:
            raise ValueError('engine move missing from original action catalog: ' + name)
        action_ids.append(found)
    ad = catalog.actions.device
    counts = []; prior = np.zeros((len(species), 64), np.float32)
    for i, name in enumerate(species):
        # Cosmetic public forms use the same original randbat base species.
        from pokezero.randbat import canonical_gen3_randbat_species_id
        key = canonical_gen3_randbat_species_id(name) if i else ''
        source_id = next((n for n, s in enumerate(catalog.source.species_names)
            if normalize_id(s) == normalize_id(key)), 0)
        count = len(catalog.source.variants[source_id]) if source_id else 0
        if count > 64:raise ValueError('v5 original source variant capacity64 exceeded')
        counts.append(count)
        if count:prior[i, :count] = 1. / count
    numeric = np.asarray(ad.numeric)[action_ids]
    move_categories = np.asarray(ad.categories)[action_ids]
    # Request id is generic Hidden Power; typed mechanics remain separate.
    primary = [category('move:' + ('hiddenpower' if normalize_id(n).startswith('hiddenpower') else n)) if i else 0
        for i, n in enumerate(moves)]
    pp = [catalog.dex.move_info(n).max_pp if i and catalog.dex.move_info(n) is not None else 0
        for i, n in enumerate(moves)]
    arrays = (np.asarray(sd.species_category)[species_ids], bases, base_types,
        [category('type:' + n) if i else 0 for i, n in enumerate(types)],
        [category('belief:possible_ability:' + normalize_id(n)) if i else 0 for i, n in enumerate(abilities)],
        [category('belief:possible_item:' + normalize_id(n)) if i else 0 for i, n in enumerate(items)],
        [category('status:' + n) for n in ('none', 'par', 'brn', 'psn', 'tox', 'slp', 'frz')],
        primary, numeric, move_categories, np.asarray(ad.hp_kind)[action_ids], np.asarray(ad.hp_power), pp,
        np.asarray(ad.fallback_primary), np.asarray(ad.secondary), np.asarray(ad.role), np.asarray(ad.slots),
        category('pokemon:self'), category('pokemon:opponent'), category('field'),
        [category('request_kind:' + n) for n in ('none', 'wait', 'force_switch', 'move')],
        [0] + [category('weather:' + n) for n in ('raindance', 'sunnyday', 'sandstorm', 'hail')], prior, counts,
        catalog.vocabulary.size, next((i for i, name in enumerate(types) if normalize_id(name) == 'ghost'), -1),
        next((i for i, name in enumerate(moves) if normalize_id(name) == 'struggle'), 0))
    floats = {'base_stats', 'move_numeric', 'prior'}
    from . import engine_device_belief, engine_device_features
    return Tables(*(j.asarray(value, j.float32 if name in floats else j.int32)
        for name, value in zip(Tables._fields, arrays)), engine_device_belief.compile_tables(metadata, catalog),
        engine_device_features.compile_tables(metadata, catalog), j.asarray(category('move:recharge'), j.int32))


class Observation(NamedTuple):
    numeric_features: object
    categorical_ids: object
    token_type_ids: object
    attention_mask: object
    legal_action_mask: object
    switch_targets: object          # physical targets for the five action lanes
    error: object                  # per-row bitfield, never a row-discard flag
    error_counts: object           # per ERROR_KINDS, current call only
    move_targets: object = None    # request slot -> original engine move slot


class _OwnRequest(NamedTuple):
    pp: object
    disabled: object
    moves: object
    choice_move: object
    active: object
    encore_move: object
    encore_counter: object
    last_action_kind: object
    last_action_index: object
    charge_move: object
    recharge_counter: object = None


def _index(name):
    return reference.v4_numeric_index(getattr(reference, 'NUMERIC_' + name))


def _put(values, names, tensors):
    columns = j.asarray([_index(n) for n in names], j.int32)
    return values.at[..., columns].set(j.stack([j.asarray(x, j.float32) for x in tensors], axis=-1))


def _lookup(table, ids):
    return table[j.clip(ids, 0, table.shape[0] - 1)]


def encode(state, public, tables, *, observer_sides, legal_action_mask):
    """Pure device view for one owner per batch row; no host/protocol calls.

    ``legal_action_mask`` is the actual engine request mask in physical bench
    order. We reorder its five switch lanes to the same own_order used by tokens.
    No opponent state is selected here. Public trackers are an explicit input so
    an Arm D sampled world cannot substitute its sampled hidden party for them.
    """
    b = state.hp.shape[0]
    if (observer_sides.shape != (b,) or observer_sides.dtype != j.int32
            or legal_action_mask.shape != (b, 9) or legal_action_mask.dtype != j.bool_):
        raise ValueError('B int32 owners and Bx9 bool actual request mask required')
    rows = j.arange(b); owner = j.clip(observer_sides, 0, 1)
    # Select the owner before inspecting any private contents.
    own = {name: getattr(state, name)[rows, owner] for name in
        ('species', 'level', 'gender', 'hp', 'max_hp', 'stats', 'status', 'types',
         'base_ability', 'item', 'active', 'boosts', 'moves', 'pp', 'disabled')}
    # Pinned Pokemon.getRequestData exposes baseAbility, not current borrowed
    # ability; the separate public borrowed-ability column remains deferred.
    own['ability'] = own.pop('base_ability')
    order = public.own_order
    if order.shape != (b, 6) or order.dtype != j.int32:
        raise ValueError('B by6 int32 request-to-physical order required')
    order_ok = j.all(j.sort(order, axis=-1) == j.arange(6), axis=-1)
    order = j.clip(order, 0, 5)
    def gather(x):return j.take_along_axis(x, order.reshape(order.shape + (1,) * (x.ndim - 2)), axis=1)
    num = j.zeros((b, TOKENS, NUMERIC), j.float32)
    cat = j.zeros((b, TOKENS, CATEGORICAL), j.int32)
    present = own['species'] > 0
    active = j.arange(6)[None] == own['active'][:, None]
    hp = j.clip(own['hp'] / j.maximum(1, own['max_hp']), 0, 1)
    alive = own['hp'] > 0
    own_num = j.zeros((b, 6, NUMERIC), j.float32)
    own_num = _put(own_num, ('HP_FRACTION', 'ACTIVE', 'LEGAL', 'PRESENT', 'LEVEL'),
        (hp, active, alive, present, j.minimum(1, own['level'] / 100)))
    stats = ('HP', 'ATK', 'DEF', 'SPA', 'SPD', 'SPE')
    own_num = _put(own_num, tuple('BASE_' + n for n in stats),
        tuple(j.minimum(1, _lookup(tables.base_stats, own['species'])[..., i] / 200) for i in range(6)))
    # Original _self_team_from_request fills stats.hp from exact condition maxHP.
    own_num = _put(own_num, tuple('ACTUAL_' + n for n in stats),
        (j.where(alive, j.minimum(1, own['max_hp'] / 714), 0),) +
        tuple(j.minimum(1, own['stats'][..., i] / 714) for i in range(1, 6)))
    own_num = _put(own_num, tuple('BOOST_' + n for n in stats[1:]),
        tuple(j.where(active, j.clip(own['boosts'][..., i] / 6, -1, 1), 0) for i in range(5)))
    own_num = _put(own_num, ('GENDER_MALE', 'GENDER_FEMALE', 'POSSIBLE_ABILITY_COUNT',
        'POSSIBLE_ITEM_COUNT', 'REVEALED_ABILITY', 'REVEALED_ITEM'),
        (own['gender'] == 1, own['gender'] == 2, own['ability'] > 0, own['item'] > 0,
         own['ability'] > 0, own['item'] > 0))
    own_cat = j.zeros((b, 6, CATEGORICAL), j.int32)
    for col, value in ((0, _lookup(tables.species, own['species'])), (1, _lookup(tables.statuses, j.where(alive, own['status'], 0))),
            (2, j.full_like(own['species'], tables.own_role)),
            (reference.CATEGORY_BELIEF_ABILITY_OFFSET, _lookup(tables.abilities, own['ability'])),
            (reference.CATEGORY_BELIEF_ITEM_OFFSET, _lookup(tables.items, own['item']))):
        own_cat = own_cat.at[..., col].set(value)
    own_cat = own_cat.at[..., 4:6].set(_lookup(tables.types, own['types']))
    own_num = j.where(present[..., None], own_num, 0); own_cat = j.where(present[..., None], own_cat, 0)
    num = num.at[:, 1:7].set(gather(own_num)); cat = cat.at[:, 1:7].set(gather(own_cat))
    # Public encounter slots, never physical opponent party slots.
    foe_num = j.zeros((b, 6, NUMERIC), j.float32)
    foe_num = _put(foe_num, ('HP_FRACTION', 'ACTIVE', 'LEGAL', 'PRESENT', 'LEVEL', 'GENDER_MALE', 'GENDER_FEMALE'),
        (public.hp_fraction, public.active, ~public.fainted, public.present,
         j.minimum(1, public.level / 100), public.gender == 1, public.gender == 2))
    foe_num = _put(foe_num, tuple('BASE_' + n for n in stats),
        tuple(j.minimum(1, _lookup(tables.base_stats, public.species)[..., i] / 200) for i in range(6)))
    foe_num = _put(foe_num, tuple('BOOST_' + n for n in stats[1:]),
        tuple(j.where(public.active, j.clip(public.boosts[..., i] / 6, -1, 1), 0) for i in range(5)))
    foe_num = _put(foe_num, ('REVEALED_ABILITY', 'REVEALED_ITEM'), (public.ability > 0, public.item > 0))
    foe_cat = j.zeros((b, 6, CATEGORICAL), j.int32)
    for col, value in ((0, _lookup(tables.species, public.species)), (1, _lookup(tables.statuses, public.status)),
            (2, j.full_like(public.species, tables.foe_role)),
            (reference.CATEGORY_BELIEF_ABILITY_OFFSET, _lookup(tables.abilities, public.ability)),
            (reference.CATEGORY_BELIEF_ITEM_OFFSET, _lookup(tables.items, public.item))):
        foe_cat = foe_cat.at[..., col].set(value)
    foe_cat = foe_cat.at[..., 4:6].set(_lookup(tables.types, public.types))
    foe_num = j.where(public.present[..., None], foe_num, 0)
    foe_cat = j.where(public.present[..., None], foe_cat, 0)
    num = num.at[:, 7:13].set(foe_num); cat = cat.at[:, 7:13].set(foe_cat)
    # Actual owning move slots; static mechanics share the original v4 algebra.
    actor = j.clip(own['active'], 0, 5)
    move = own['moves'][rows, actor]; pp = own['pp'][rows, actor]
    from .state import usable_moves
    # Run the unchanged request algebra on ONLY the selected owning side.
    request = _OwnRequest(*(getattr(state, name)[rows, owner][:, None] for name in _OwnRequest._fields
        if name != 'recharge_counter'))
    # Fidelity M1: a mustrecharge request (sim/pokemon.ts getMoves: [{move: 'Recharge', id: 'recharge'}],
    # trapped). Structs without the leaf (request fixtures) have no recharge.
    if getattr(state, 'recharge_counter', None) is not None:
        request = request._replace(recharge_counter=state.recharge_counter[rows, owner][:, None])
    usable = usable_moves(request)[:, 0]
    struggle = ~j.any(usable, axis=-1)
    charge_move = request.charge_move[:, 0][rows, actor]
    charged = charge_move > 0
    charge_slot = j.argmax(move == charge_move[:, None], axis=-1)
    move_targets = j.broadcast_to(j.arange(4, dtype=j.int32), (b, 4)).at[:, 0].set(j.where(charged, charge_slot, 0))
    # Original getMoves(lockedMove) returns one id/name row, with missing PP.
    move = j.where(charged[:, None], j.where(j.arange(4) == 0, charge_move[:, None], 0), move)
    usable = j.where(charged[:, None], j.arange(4) == 0, usable)
    forced = state.forced_switch[rows, owner] | ~alive[rows, actor]
    acting = ~forced & ~state.done & j.any(legal_action_mask, axis=-1)
    move = j.where(struggle[:, None], j.where(j.arange(4) == 0, tables.struggle_move, 0), move)
    # Fidelity M1: the recharge collapse, like the charge one: one present, usable row in lane 0 (engine
    # slot 0), named move:recharge, with no mechanics (no dex move) and the writer's PP fraction 1 (no PP
    # fields); lanes 1-3 absent; the engine mask already makes lane 0 the only legal action (trapped).
    recharging = j.zeros(b, bool)
    if getattr(state, 'recharge_counter', None) is not None and tables.recharge_primary is not None:
        recharging = state.recharge_counter[rows, owner][rows, actor] > 0
    recharge_row = recharging[:, None] & (j.arange(4) == 0) & acting[:, None]
    move = j.where(recharging[:, None], 0, move)
    moves_present = ((move > 0) & acting[:, None]) | recharge_row
    move = j.where(moves_present, move, 0)
    # Same original static Curse variants as private_actions, using own type.
    ghost = j.any(own['types'][rows, actor] == tables.ghost_type, axis=-1).astype(j.int32)
    safe_move = j.clip(move, 0, tables.move_primary.shape[0] - 1)
    values = tables.move_numeric[safe_move, ghost[:, None]]
    hp_kind = tables.move_hp_kind[safe_move]
    den = j.clip(own['max_hp'][rows, actor], 1, tables.hp_power.shape[1] - 1)
    numerator = j.clip(own['hp'][rows, actor], 0, den)
    dynamic = tables.hp_power[j.clip(hp_kind - 1, 0, 1), den[:, None], numerator[:, None]].astype(j.float32)
    power = j.where(hp_kind > 0, dynamic, values[..., 0])
    values = values.at[..., 0].set(j.minimum(1, power / 200))
    action_num = j.zeros((b, 9, NUMERIC), j.float32)
    action_num = action_num.at[:, :4, j.asarray([_index(n) for n in ('BASE_POWER', 'PRIORITY', 'ACCURACY', 'EFFECT_CHANCE', 'SELF_HP_COST')])].set(values)
    action_num = action_num.at[:, :4, _index('MOVE_PP_FRACTION')].set(j.where(moves_present,
        j.where((struggle | charged | recharging)[:, None], 1., j.clip(pp / j.maximum(1, tables.max_pp[safe_move]), 0, 1)), 0))
    action_num = action_num.at[:, :4, _index('PRESENT')].set(moves_present)
    action_num = action_num.at[:, :4, _index('ACTIVE')].set(moves_present & (usable | struggle[:, None]))
    request_active = j.argmax(gather(active), axis=-1)
    positions = j.arange(5)[None] + (j.arange(5)[None] >= request_active[:, None])
    targets = j.take_along_axis(order, positions, axis=1)
    physical_lanes = j.clip(targets - (targets > actor[:, None]), 0, 4)
    move_legal = j.take_along_axis(legal_action_mask[:, :4], move_targets, axis=1)
    move_legal &= j.where(charged[:, None], j.arange(4) == 0, True)
    legal = j.concatenate((move_legal, j.take_along_axis(legal_action_mask[:, 4:], physical_lanes, axis=1)), axis=-1)
    action_num = action_num.at[..., _index('LEGAL')].set(legal)
    switch_num = j.take_along_axis(own_num, targets[..., None], axis=1)
    switch_columns = [_index(n) for n in ('HP_FRACTION', 'ACTIVE', 'LEGAL', 'PRESENT', 'LEVEL') + tuple('BASE_' + n for n in stats) + tuple('ACTUAL_' + n for n in stats)]
    action_num = action_num.at[:, 4:, j.asarray(switch_columns)].set(switch_num[..., j.asarray(switch_columns)])
    action_num = action_num.at[:, 4:, _index('LEGAL')].set(legal[:, 4:])
    action_cat = j.zeros((b, 9, CATEGORICAL), j.int32)
    move_primary = j.where(recharge_row, 0 if tables.recharge_primary is None else tables.recharge_primary,
        tables.move_primary[safe_move])
    primary = j.concatenate((j.where(moves_present, move_primary, tables.fallback[:4]),
        j.where(j.take_along_axis(present, targets, axis=1),
            j.take_along_axis(own_cat[..., 0], targets, axis=1), tables.fallback[4:])), axis=-1)
    action_cat = action_cat.at[..., 0].set(primary).at[..., 1].set(tables.action_secondary[(j.arange(9) >= 4).astype(j.int32)])
    action_cat = action_cat.at[..., 2].set(tables.action_role).at[..., 3].set(tables.action_slots)
    action_cat = action_cat.at[:, :4, j.asarray((4, 6, 8, 7))].set(tables.move_categories[safe_move, ghost[:, None]])
    # Switch tokens use the species descriptor, not an active retyping overlay.
    switch_types = j.take_along_axis(_lookup(tables.base_types, own['species']), targets[..., None], axis=1)
    action_cat = action_cat.at[:, 4:, 4:6].set(j.where(j.take_along_axis(present, targets, axis=1)[..., None], switch_types, 0))
    num = num.at[:, 13:22].set(action_num); cat = cat.at[:, 13:22].set(action_cat)
    turn = state.turn if getattr(public, 'turn', None) is None else public.turn
    weather = state.weather if getattr(public, 'weather', None) is None else public.weather
    spikes = state.spikes if getattr(public, 'spikes', None) is None else public.spikes
    num = num.at[:, 0, _index('PRESENT')].set(1).at[:, 0, _index('TURN_COUNT')].set(j.minimum(1, turn / 1000))
    num = num.at[:, 0, _index('SELF_HAZARDS')].set(spikes[rows, owner] / 3)
    num = num.at[:, 0, _index('OPP_HAZARDS')].set(spikes[rows, 1 - owner] / 3)
    request_kind = j.where(state.done, 0, j.where(~j.any(legal_action_mask, axis=-1), 1, j.where(forced, 2, 3)))
    cat = cat.at[:, 0, 0].set(tables.request_categories[request_kind]).at[:, 0, 1].set(_lookup(tables.weather_categories, weather)).at[:, 0, 2].set(tables.field_role)
    prior = j.where(public.present[..., None], _lookup(tables.prior, public.species), 0)
    belief_error = j.zeros(b, bool); feature_error = j.zeros(b, bool)
    belief_result = None
    if tables.belief is not None:
        from . import engine_device_belief as belief
        from .public_belief_features import NUMERIC_COLUMNS, CATEGORY_COLUMNS
        from .public_expected_stats import NUMERIC_COLUMNS as EXPECTED_COLUMNS
        belief_result = belief.project(public, tables.belief)
        foe_num = foe_num.at[..., j.asarray(NUMERIC_COLUMNS)].set(belief_result.numeric[..., j.asarray(NUMERIC_COLUMNS)])
        foe_cat = foe_cat.at[..., j.asarray(CATEGORY_COLUMNS)].set(belief_result.categorical[..., j.asarray(CATEGORY_COLUMNS)])
        own_stats = own['stats'].at[..., 0].set(j.where(alive, own['max_hp'], 0))
        expected_values, expected_bad = belief.expected_stats(public, belief_result, tables.belief,
            own_species=gather(own['species']), own_stats=gather(own_stats))
        foe_num = foe_num.at[..., j.asarray(EXPECTED_COLUMNS)].set(expected_values)
        certain = belief_result.certain_ability
        trapper = _lookup(tables.belief.trapper, certain) & (certain > 0)
        foe_num = foe_num.at[..., _index('TRAPPER_ALIVE')].set(public.present & ~public.active & ~public.fainted & trapper)
        own_ability = _lookup(tables.belief.engine_ability, own['ability'])
        own_trapper = _lookup(tables.belief.trapper, own_ability) & (own_ability > 0)
        num = num.at[:, 1:7, _index('TRAPPER_ALIVE')].set(gather(present & ~active & alive & own_trapper))
        # Source hazard exposure counts known own bench + unseen public slots,
        # never the sampled opponent's actual party or ability.
        own_flying = _lookup(tables.belief.flying, own['species'])
        own_levitate = (own_ability == tables.belief.levitate) & (tables.belief.levitate > 0)
        own_count = j.sum(present & ~active & alive & ~own_flying & ~own_levitate, axis=-1)
        foe_flying = _lookup(tables.belief.flying, public.species)
        foe_levitate = (certain == tables.belief.levitate) & (tables.belief.levitate > 0)
        foe_count = j.maximum(0, 6 - public.present.sum(axis=-1)) + j.sum(public.present & ~public.active & ~public.fainted & ~foe_flying & ~foe_levitate, axis=-1)
        fractions = j.asarray((0., 1 / 8, 1 / 6, 1 / 4), j.float32)
        num = num.at[:, 0, _index('SELF_HAZARD_EXPECTED')].set(j.minimum(1., own_count * fractions[j.clip(spikes[rows, owner], 0, 3)] / 6))
        num = num.at[:, 0, _index('OPP_HAZARD_EXPECTED')].set(j.minimum(1., foe_count * fractions[j.clip(spikes[rows, 1 - owner], 0, 3)] / 6))
        posterior = belief_result.weights
        belief_error = belief_result.error | expected_bad
    else:
        posterior = (j.zeros_like(prior) if public.posterior_weights is None else public.posterior_weights)
    if posterior.shape != (b, 6, 64) or posterior.dtype != j.float32:
        raise ValueError('public posterior requires Bx6x64 float32 or None')
    posterior_ok = j.all(j.isfinite(posterior) & (posterior >= 0), axis=(1, 2))
    posterior = j.where(public.present[..., None] & j.isfinite(posterior), posterior, 0)
    tail = j.zeros((b, 6, 132), j.float32).at[..., :64].set(posterior).at[..., 64:128].set(prior)
    tail = tail.at[..., 128].set(j.where(public.present, _lookup(tables.prior_counts, public.species) / 64, 0))
    if belief_result is not None:
        tail = tail.at[..., 129].set(belief_result.count / 64).at[..., 130].set(belief_result.reason / 3).at[..., 131].set(belief_result.pin_conflict)
    if tables.features is not None:
        from . import engine_device_features as features
        own_facts = getattr(public, 'own_mon_features', None)
        foe_facts = getattr(public, 'mon_features', None)
        own_sub = None
        if hasattr(state, 'substitute_hp'):
            sub = getattr(state, 'substitute_hp')[rows, owner] > 0
            own_sub = gather(j.where(sub, j.floor(own['max_hp'] / 4) / j.maximum(1, own['max_hp']), 0))
        foe_sub = None if foe_facts is None or 'sub_present' not in foe_facts else j.where(foe_facts['sub_present'], .25, 0)
        on, oc, bad_own = features.project_mon(num[:, 1:7], cat[:, 1:7], own_facts,
            active=gather(active), present=gather(present), tables=tables.features,
            last_moves=getattr(public, 'own_last_moves', None), borrowed_ability=getattr(public, 'own_borrowed_ability', None),
            volatiles=getattr(public, 'own_volatiles', None), substitute_fraction=own_sub,
            status=gather(j.where(alive, own['status'], 0)), opponent_tendency=False)
        fn, fc, bad_foe = features.project_mon(foe_num, foe_cat, foe_facts,
            active=public.active, present=public.present, tables=tables.features,
            last_moves=getattr(public, 'last_moves', None), borrowed_ability=getattr(public, 'borrowed_ability', None),
            volatiles=getattr(public, 'volatiles', None), substitute_fraction=foe_sub, status=public.status)
        field, bad_field = features.project_field(num[:, 0], getattr(public, 'field_features', None))
        stat_num, stat_cat, stat_on, bad_stats = features.project_stats(getattr(public, 'stats_features', None), tables.features, b)
        if own_facts is not None and 'rest_sleep' in own_facts:
            on = on.at[..., _index('WAKE_KNOWN')].set((own_facts['rest_sleep'] != 0) & gather(present & alive & (own['status'] == 5)))
        if foe_facts is not None and 'rest_sleep' in foe_facts and belief_result is not None:
            from .public_belief_features import _bits
            selected = _bits(belief_result.effective, tables.belief.variant_valid.shape[-1])
            ability_ids = _lookup(tables.belief.variant_ability, _lookup(tables.belief.source_species, public.species))
            has_early = j.any(selected & (ability_ids == tables.belief.early_bird) & (tables.belief.early_bird > 0), axis=-1)
            known = (public.ability > 0) | ((belief_result.count > 0) & ~has_early)
            fn = fn.at[..., _index('WAKE_KNOWN')].set(public.present & (foe_facts['rest_sleep'] != 0) & (public.status == 5) & known)
        num = num.at[:, 1:7].set(on).at[:, 0].set(field).at[:, 22].set(stat_num)
        cat = cat.at[:, 1:7].set(oc).at[:, 22].set(stat_cat)
        foe_num, foe_cat = fn, fc
        feature_error = bad_own | bad_foe | bad_field | bad_stats
    else:stat_on = j.zeros(b, bool)
    num = num.at[:, 7:13].set(foe_num); cat = cat.at[:, 7:13].set(foe_cat)
    num = num.at[:, 23:].set(tail); cat = cat.at[:, 23:].set(foe_cat)
    attention = j.ones((b, TOKENS), bool).at[:, 1:7].set(gather(present)).at[:, 7:13].set(public.present).at[:, 22].set(stat_on)
    own_ids = (own['active'] >= 0) & (own['active'] < 6); public_ids = j.ones((b,), bool)
    for ids, table in ((own['species'], tables.species), (own['types'], tables.types), (own['ability'], tables.abilities),
            (own['item'], tables.items), (own['status'], tables.statuses), (move, tables.move_primary)):
        own_ids &= j.all((ids >= 0) & (ids < table.shape[0]), axis=tuple(range(1, ids.ndim)))
    for ids, table in ((public.species, tables.species), (public.types, tables.types), (public.ability, tables.abilities),
            (public.item, tables.items), (public.status, tables.statuses)):
        public_ids &= j.all((ids >= 0) & (ids < table.shape[0]), axis=tuple(range(1, ids.ndim)))
    finite = j.all(j.isfinite(num), axis=(1, 2))
    flags = j.stack(((observer_sides < 0) | (observer_sides > 1), ~own_ids, ~public_ids,
        ~order_ok, public.error != 0, ~posterior_ok, ~finite, belief_error, feature_error), axis=-1)
    error = j.sum(flags.astype(j.int32) * (1 << j.arange(len(ERROR_KINDS), dtype=j.int32)), axis=-1)
    num = j.nan_to_num(num, nan=0., posinf=0., neginf=0.)
    cat = j.where((cat >= 0) & (cat < tables.category_count), cat, 0)
    return Observation(num, cat, j.broadcast_to(j.asarray(TOKEN_TYPES, j.int32), (b, TOKENS)),
        attention, legal, targets, error, j.sum(flags.astype(j.int32), axis=0), move_targets)


def map_actions(observation, state, observer_sides, actions):
    """Translate selected v4 request lanes to existing engine dispatch lanes."""
    rows = j.arange(actions.shape[0]); owner = j.clip(observer_sides, 0, 1)
    move = j.clip(actions, 0, 3)
    if observation.move_targets is not None:move = observation.move_targets[rows, move]
    target = observation.switch_targets[rows, j.clip(actions - 4, 0, 4)]
    active = state.active[rows, owner]
    switch = 4 + target - (target > active).astype(j.int32)
    return j.where(actions < 4, move, switch).astype(j.int32)


def actor_inputs(observation):
    """Actual public29 tensors for v5; all rows execute, including error rows."""
    return dict(numeric_features=observation.numeric_features[:, None],
        categorical_ids=observation.categorical_ids[:, None], token_type_ids=observation.token_type_ids[:, None],
        attention_mask=observation.attention_mask[:, None], history_mask=j.ones((observation.error.shape[0], 1), bool))


def champion_inputs(observation):
    """Explicit champion layout selection from the SAME v5 representation.

    No fabricated scores, auxiliary forward, or error-row discard. The champion
    accepts its unchanged first23-token v4 prefix, without a history dimension.
    """
    return dict(numeric=observation.numeric_features[:, :23],
        categorical=observation.categorical_ids[:, :23],
        token_types=observation.token_type_ids[:, :23], attention=observation.attention_mask[:, :23])


def column_manifest(public=None, *, tracked_types=None, tracked_ability=None, tracked_item=None, posterior=True):
    """Complete meanings; optional real readout keys describe installed hooks.

    No array contents are copied to host: only the static dictionary keys and
    optional-tail presence are inspected. A missing hook remains ZERO, even
    though this module already implements its projection algebra.
    """
    from . import engine_device_features as features
    from .public_belief_features import NUMERIC_COLUMNS as BELIEF_COLUMNS, CATEGORY_COLUMNS as BELIEF_CATEGORIES
    from .public_expected_stats import NUMERIC_COLUMNS as EXPECTED_COLUMNS
    if tracked_types is None:tracked_types = public is not None and getattr(public, 'types', None) is not None
    if tracked_ability is None:tracked_ability = public is not None and getattr(public, 'ability', None) is not None
    if tracked_item is None:tracked_item = public is not None and getattr(public, 'item', None) is not None
    names = {}
    for name, value in vars(reference).items():
        if name.startswith('NUMERIC_') and isinstance(value, int):
            try:names.setdefault(reference.v4_numeric_index(value), []).append(name[8:])
            except (ValueError, KeyError):pass
    own = {'HP_FRACTION', 'ACTIVE', 'LEGAL', 'PRESENT', 'LEVEL', 'GENDER_MALE', 'GENDER_FEMALE',
        'POSSIBLE_ABILITY_COUNT', 'POSSIBLE_ITEM_COUNT', 'REVEALED_ABILITY', 'REVEALED_ITEM'}
    own |= {'BASE_' + n for n in ('HP', 'ATK', 'DEF', 'SPA', 'SPD', 'SPE')}
    own |= {'ACTUAL_' + n for n in ('HP', 'ATK', 'DEF', 'SPA', 'SPD', 'SPE')}
    own |= {'BOOST_' + n for n in ('ATK', 'DEF', 'SPA', 'SPD', 'SPE')}
    own |= {'TRAPPER_ALIVE', 'SUB_HP_FRACTION'}
    foe = own - {'POSSIBLE_ABILITY_COUNT', 'POSSIBLE_ITEM_COUNT'} - {n for n in own if n.startswith('ACTUAL_')}
    field = {'PRESENT', 'TURN_COUNT', 'SELF_HAZARDS', 'OPP_HAZARDS', 'SELF_HAZARD_EXPECTED', 'OPP_HAZARD_EXPECTED'}
    action = {'HP_FRACTION', 'ACTIVE', 'LEGAL', 'PRESENT', 'LEVEL', 'BASE_POWER', 'PRIORITY', 'ACCURACY',
        'EFFECT_CHANCE', 'SELF_HP_COST', 'MOVE_PP_FRACTION'} | {n for n in own if n.startswith(('BASE_', 'ACTUAL_'))}
    # Original B2 writers are sparse by token role. In particular, unnamed
    # positions inside the 16 PP/validity and six weather-reveal ranges are
    # still role-specific; absence of a NUMERIC_* name is not a missing hook.
    field_columns = {_index(n) for n in ('PRESENT', 'TURN_COUNT', 'SELF_HAZARDS', 'OPP_HAZARDS')}
    field_columns |= {_index(item[0]) for item in features.FIELD_FIELDS.values()}
    field_columns |= {_index('SELF_HAZARD_EXPECTED'), _index('OPP_HAZARD_EXPECTED')}
    stats_columns = {_index('PRESENT'), *(_index(item[0]) for item in features.STATS_FIELDS.values()),
        *(reference.v4_numeric_index(reference.NUMERIC_STAT_WEATHER_REVEAL_OFFSET + i) for i in range(6))}
    mon_columns = {_index(n) for n in own} | {_index(item[0]) for item in features.MON_FIELDS.values()}
    mon_columns |= {_index('WAKE_KNOWN'), _index('TOXIC_STAGE')}
    own_columns = mon_columns - {_index(features.MON_FIELDS[key][0]) for key in features.TENDENCY_FIELDS}
    opponent_columns = mon_columns | set(BELIEF_COLUMNS) | set(EXPECTED_COLUMNS)
    active_categories = {reference.CATEGORY_LAST_USED_MOVE, reference.CATEGORY_TRACED_ABILITY,
        *range(reference.CATEGORY_VOLATILE_OFFSET, reference.CATEGORY_VOLATILE_OFFSET + 6)}
    role_numeric = {'field': field_columns, 'own': own_columns, 'opponent': opponent_columns,
        'stats': stats_columns}
    role_categorical = {'field': {0, 1, 2},
        'own': {0, 1, 2, 4, 5, reference.CATEGORY_BELIEF_ABILITY_OFFSET,
            reference.CATEGORY_BELIEF_ITEM_OFFSET} | active_categories,
        'opponent': {0, 1, 2, 4, 5, *BELIEF_CATEGORIES} | active_categories,
        'stats': {reference.CATEGORY_ROLE}}
    result = []
    for section, produced, mode in (('field', field, 'DIRECT'), ('own', own, 'DIRECT'),
            ('opponent', foe, 'TRACKED'), ('actions', action, 'DIRECT'), ('stats', set(), 'ZERO'), ('belief', set(), 'ZERO')):
        for kind, width in (('numeric', NUMERIC), ('categorical', CATEGORICAL)):
            team_cats = {0, 1, 2, 4, 5, reference.CATEGORY_BELIEF_ABILITY_OFFSET, reference.CATEGORY_BELIEF_ITEM_OFFSET}
            public_cats = {0, 1, 2} | ({4, 5} if tracked_types else set())
            if tracked_ability:public_cats.add(reference.CATEGORY_BELIEF_ABILITY_OFFSET)
            if tracked_item:public_cats.add(reference.CATEGORY_BELIEF_ITEM_OFFSET)
            cats = {'field': {0, 1, 2}, 'own': team_cats, 'opponent': public_cats,
                'actions': set(range(9)), 'stats': set(), 'belief': public_cats}[section]
            for column in range(width):
                labels = names.get(column, []) if kind == 'numeric' else ['category_' + str(column)]
                done = bool(set(labels) & produced) if kind == 'numeric' else column in cats
                if section == 'opponent' and kind == 'numeric':
                    if 'REVEALED_ABILITY' in labels and not tracked_ability:done = False
                    if 'REVEALED_ITEM' in labels and not tracked_item:done = False
                status = mode if done else 'ZERO'
                reason = ('actual owning engine/field facts with original normalization' if mode == 'DIRECT' else
                    'public encounter tracker or public species catalog only') if done else 'deferred public hook; no inferred fact'
                if section == 'belief' and kind == 'categorical' and done:status = 'TRACKED'
                if section == 'belief' and kind == 'numeric' and 64 <= column <= 128:
                    status, reason = 'TRACKED', 'original source species prior; public species only'
                elif section == 'belief' and kind == 'numeric' and column < 64:
                    status, reason = ('TRACKED', 'public posterior input from tracker') if posterior else (
                        'ZERO', 'deferred public posterior narrowing hook; no sampled-world posterior')
                if section == 'stats':reason = 'deferred individual outcome/credit counter; token structure follows original compiled stats mask'
                if section == 'opponent' and kind == 'numeric' and column in (*BELIEF_COLUMNS, *EXPECTED_COLUMNS, _index('TRAPPER_ALIVE')):
                    status, reason = 'TRACKED', 'original source-ordered mask reduction/public reveal algebra; no sampled hidden facts'
                if section in ('opponent', 'belief') and kind == 'categorical' and column in BELIEF_CATEGORIES:
                    status, reason = 'TRACKED', 'lexically ordered source possibility/revealed categories'
                if section == 'field' and kind == 'numeric' and column in (_index('SELF_HAZARD_EXPECTED'), _index('OPP_HAZARD_EXPECTED')):
                    status, reason = 'TRACKED', 'own request facts + public seen/fainted/semantic-ability belief; unseen count remains6-seen'
                facts_name = {'own': 'own_mon_features', 'opponent': 'mon_features', 'field': 'field_features', 'stats': 'stats_features'}.get(section)
                facts = None if public is None or facts_name is None else getattr(public, facts_name, None)
                mapping = features.MON_FIELDS if section in ('own', 'opponent') else features.FIELD_FIELDS if section == 'field' else features.STATS_FIELDS
                for key, item in mapping.items():
                    if facts is not None and key in facts and kind == 'numeric' and column == _index(item[0]):
                        status, reason = 'TRACKED', 'actual public tracker raw ' + key + '; original divisor ' + str(item[1])
                if section == 'stats' and facts is not None and ((kind == 'numeric' and column == _index('PRESENT'))
                        or (kind == 'categorical' and column == reference.CATEGORY_ROLE)):
                    status, reason = 'TRACKED', 'original compiled stats_block controls stats token presence/role independently of individual counters'
                if section == 'own' and kind == 'numeric' and column in {
                        _index(features.MON_FIELDS[key][0]) for key in features.TENDENCY_FIELDS}:
                    status, reason = 'ZERO', 'structurally opponent-only original tendency columns; owning exact-state columns are separate'
                if facts is not None and section in ('own', 'opponent') and 'rest_sleep' in facts and kind == 'numeric' and column == _index('WAKE_KNOWN'):
                    status, reason = 'TRACKED', 'public Rest sleep + request-known/source-semantic EarlyBird certainty'
                if section in ('own', 'opponent') and kind == 'categorical':
                    prefix = 'own_' if section == 'own' else ''
                    for field_name, cols in ((prefix + 'last_moves', (reference.CATEGORY_LAST_USED_MOVE,)),
                            (prefix + 'borrowed_ability', (reference.CATEGORY_TRACED_ABILITY,)),
                            (prefix + 'volatiles', tuple(range(reference.CATEGORY_VOLATILE_OFFSET, reference.CATEGORY_VOLATILE_OFFSET + 6)))):
                        if public is not None and getattr(public, field_name, None) is not None and column in cols:
                            status, reason = 'TRACKED', 'actual public ' + field_name
                if section == 'opponent' and kind == 'numeric' and column == _index('SUB_HP_FRACTION') and facts is not None and 'sub_present' in facts:
                    status, reason = 'TRACKED', 'public Substitute presence; original opponent feature is .25, not hidden remaining HP'
                if section == 'stats' and facts is not None:
                    if (kind == 'numeric' and column == _index('PRESENT')) or (kind == 'categorical' and column == reference.CATEGORY_ROLE):
                        status, reason = 'TRACKED', 'actual public tendency tracker present'
                    for i in range(6):
                        if 'weather_reveal_' + str(i) in facts and kind == 'numeric' and column == reference.v4_numeric_index(reference.NUMERIC_STAT_WEATHER_REVEAL_OFFSET + i):
                            status, reason = 'TRACKED', 'actual public weather-reveal tendency'
                if section == 'belief' and kind == 'numeric':
                    labels = ([f'posterior_weight_{column}'] if column < 64 else [f'prior_weight_{column - 64}'] if column < 128 else
                        [('prior_count/64', 'posterior_count/64', 'reason/3', 'pin_conflict')[column - 128]])
                    status, reason = 'TRACKED', 'original source prior + public hard/pin mask fallback; uniform current sampler'
                    if column == 131:
                        status, reason = 'TRACKED', 'produced constant0: frozen investment_belief_narrowing=False disables both standing-pin producers'
                # Fidelity M1: Trick's -item lines (hooks.trick) and the holder's move lines (hooks.move)
                # now write ITEM_SWAPPED and CHOICE_LOCKED, so the readout keys are tracked like the others.
                if status == 'ZERO':
                    if section in ('actions', 'belief') or (section in ('own', 'opponent') and kind == 'numeric' and any(n.startswith(('STAT_', 'SELF_', 'OPP_', 'TT_', 'TM2_')) for n in labels)):
                        reason = 'structurally not applicable to this token section'
                    elif kind == 'numeric' and any('SCREEN' in n or 'REFLECT' in n or 'SAFEGUARD' in n or 'MIST' in n or 'FUTURE_SIGHT' in n for n in labels):
                        reason = 'unavailable: current physical engine state does not implement this field'
                    else:reason = 'deferred actual public observation hook; no private timer/final-state reconstruction'
                    if section == 'own' and (any(n.startswith('EXPECTED_') for n in labels) or
                            any(n in ('REVEALED_MOVE_COUNT', 'CANDIDATE_SET_COUNT', 'UNCERTAINTY', 'POSSIBLE_MOVE_COUNT') for n in labels)):
                        reason = 'structurally not applicable: owning party is request-known, not an opponent belief'
                    if section == 'opponent' and any(n.startswith('ACTUAL_') for n in labels):
                        reason = 'unavailable hidden opponent actual stats; only public/source expected bounds are permitted'
                    applicable = role_numeric if kind == 'numeric' else role_categorical
                    if section in applicable and column not in applicable[section] and not reason.startswith('unavailable hidden'):
                        if not reason.startswith('structurally not applicable: owning party'):
                            reason = 'structurally not applicable to this token section'
                    if section in ('own', 'opponent') and kind == 'numeric':
                        if column == _index('WRAP_TRAP_TURNS'):
                            reason = 'unavailable: current physical engine has no partial-trap duration handler'
                    if section == 'belief' and kind == 'numeric' and column == 131:
                        reason = 'deferred public damage-evidence pin producer; no fabricated pin conflict'
                start, stop = {'field': (0, 1), 'own': (1, 7), 'opponent': (7, 13),
                    'actions': (13, 22), 'stats': (22, 23), 'belief': (23, 29)}[section]
                result.append(dict(section=section, token_start=start, token_stop=stop,
                    kind=kind, column=column, names=labels, status=status, reason=reason))
    return result
