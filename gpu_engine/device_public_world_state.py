"""Direct tensor materialization of original public-source fixture plans.

No actual opponent state, history decoder, protocol adapter or model is an
input. Public percentages use the ORIGINAL materializer's documented rounded
scaling onto sampled maxHP. Unsupported latent clocks stay explicit refusals.
The real owner party is copied exactly; its public opponent remains source-
sampled, with root-public knowledge retained independently of that sample.
"""
from dataclasses import dataclass
from types import MappingProxyType

import jax.numpy as j
import numpy as np

from pokezero.dex import normalize_id
from pokezero.determinization import _normalize_species_id, _gen3_randbat_fixture_spread
from pokezero.engine_world import hidden_power_engine_id
from pokezero.gen3_damage import gen3_hp_stat, gen3_stat
from pokezero.v5_gumbel import Recipe
from . import device_public_worlds as plans
from .public_active import NAMES as VOLATILES
from .state import empty_state, BattleState, LEECH_SEED, ROOTED, ATTRACT


@dataclass(frozen=True)
class Tables:
    source: plans.Source
    namespaces: object
    max_pp: tuple
    copied_pp: tuple = ()   # Transform: a copied slot's pp, min(5, move.pp) (sim/pokemon.ts transformInto)


def compile_tables(source, metadata):
    """Startup-only names/PP from the same real engine metadata as the source."""
    if type(source) is not plans.Source:
        raise ValueError('original bound device public source required')
    vocabulary = metadata['vocabulary']
    if any(tuple(vocabulary[name]) != source.vocabulary[name] for name in source.vocabulary):
        raise ValueError('materializer engine/source namespace differs')
    names = {}
    for kind in ('species', 'moves', 'abilities', 'items', 'types'):
        values = vocabulary[kind]
        mapping = {normalize_id(name): i for i, name in enumerate(values)}
        if len(mapping) != len(values):
            raise ValueError('duplicate normalized engine namespace: ' + kind)
        names[kind] = MappingProxyType(mapping)
    rows = metadata['moves']
    if len(rows) != len(vocabulary['moves']):
        raise ValueError('same engine move rows required')
    pp = [];copied = []
    for index, row in enumerate(rows):
        if index == 0:
            pp.append(0);copied.append(0)
            continue
        value = row.get('pp') if isinstance(row, dict) else None
        if type(value) is not int or value <= 0:
            raise ValueError('original engine move maxPP unavailable')
        pp.append(value*8//5);copied.append(min(5, value))
    return Tables(source, MappingProxyType(names), tuple(pp), tuple(copied))


@dataclass(frozen=True)
class World:
    index: int
    state: object | None
    public: object | None
    refusals: tuple
    kernel_seed: int
    reply_private_history_available: bool


def _id(tables, kind, value, *, absent=False):
    name = normalize_id(value or '')
    if absent and not name:
        return 0
    value = tables.namespaces[kind].get(name, 0)
    if value <= 0:
        raise ValueError('sampled source namespace unavailable: '+kind+':'+name)
    return value


def _mon_feature(view, name, slot, *, required=True):
    features = view.get('mon_features')
    if features is None or name not in features:
        if required:
            raise ValueError('public current feature unavailable: '+name)
        return None
    value = features[name]
    if value.shape != (1, 6) or not np.isfinite(value).all():
        raise ValueError('public current feature malformed: '+name)
    return value[0, slot].item()


def _rebase(root, assignment, active, *, state):
    """Only sampled coordinates replace redacted peer identity bookkeeping.

    Encounter-slot PUBLIC arrays remain untouched. No sampled hidden species
    is pre-revealed; unseen physical members retain mapping-1 until actual
    hypothetical public install hooks reveal them.
    """
    t = root.public
    foe = 1-root.address.owner
    mapping = np.full((6,), -1, np.int32)
    for public_slot, physical in assignment.items():
        mapping[physical] = public_slot
    order = np.asarray([active, *[i for i in range(6) if i != active]], np.int32)
    history = t.damage_history
    if history is not None:
        # Only the future HP baseline comes from this materialized sample.
        # Keep all public shown and exact-owner accumulated sums unchanged;
        # never scale shown past losses into invented exact peer history.
        history = history._replace(
            hp=history.hp.at[0, foe].set(state.hp[0, foe, active]),
            maximum=history.maximum.at[0, foe].set(state.max_hp[0, foe, active]))
    return t._replace(
        damage_history=history,
        physical_to_public=t.physical_to_public.at[0, foe].set(j.asarray(mapping)),
        active_physical=t.active_physical.at[0, foe].set(active),
        own_order=t.own_order.at[0, foe].set(j.asarray(order)))


def _materialize_transform(root, tables, arrays, owner, foe, active, assignment):
    """Transform (gpu_engine/transform.py) in one world, after the owner cells and the sampled opponent are written.

    Our transformed active: the request shows its copied moves and PP and its copied Hidden Power type and power (Gen 3
    names both), but its copied Atk..Spe and ability are the TRUE opponent's (transformInto copies them; the request
    shows baseStoredStats). They are rebuilt from this world's sampled donor (the opposing member of the donor species),
    so no actual opposing stat reaches a sampled world; a sampled donor whose moves (or Hidden Power, when copied) differ
    from the copy refuses. The saves, flag, moves, PP, Hidden Power and types (public) stay the owner's own.
    The opponent's transformed active (the public Transform mark on its active slot): the sampled member is the base,
    so its sampled stats, moves, PP (source max minus the public uses before the Transform: the carry charges nothing
    while transformed) and Hidden Power become the saves, and the copy is our donor's (the member of the donor species):
    Atk..Spe, moves, base ability, species types and Hidden Power, the copied slots at min(5, pp) minus the carry's
    copied_pp_used. A donor that may have run another ability or type at the Transform (Trace, Color Change) refuses."""
    if 'transformed' not in arrays:
        return
    own = root.own
    if bool(np.asarray(own['transformed'])):
        mine = int(np.asarray(own['active']))
        donor = [k for k in range(6) if int(arrays['species'][0, foe, k]) == int(np.asarray(own['transform_species']))]
        if len(donor) != 1:
            raise ValueError('own Transform donor not in the sampled party')
        d = donor[0]
        # The copied moves are exact (our request lists them), so the sampled donor must hold that set; its Hidden Power
        # type and power are exact too (the request names them, gen < 6) and stay the owner's own.
        copied = arrays['moves'][0, owner, mine]
        if sorted(int(m) for m in arrays['moves'][0, foe, d] if m) != sorted(int(m) for m in copied if m):
            raise ValueError('own Transform donor set differs from the copied moves')
        names = getattr(getattr(tables, 'source', None), 'vocabulary', {}).get('moves', ())
        if any(0 < int(m) < len(names) and normalize_id(names[int(m)]).startswith('hiddenpower') for m in copied) and any(
                int(arrays[leaf][0, foe, d]) != int(arrays[leaf][0, owner, mine]) for leaf in ('hidden_power_type', 'hidden_power_power')):
            raise ValueError('own Transform donor Hidden Power differs from the copy')
        arrays['stats'][0, owner, mine, 1:] = arrays['stats'][0, foe, d, 1:]
        arrays['ability'][0, owner, mine] = arrays['ability'][0, foe, d]
    view = root.view
    marks, donors = view.get('transformed'), view.get('transform_species')
    slot = next((s for s, physical in assignment.items() if physical == active), None)
    if marks is None or donors is None or slot is None or not bool(np.asarray(marks)[0, slot]):
        return
    species = int(np.asarray(donors)[0, slot])
    donor = [k for k in range(6) if int(arrays['species'][0, owner, k]) == species]
    if len(donor) != 1:
        raise ValueError('public Transform donor not in the owner party')
    d = donor[0]
    names = tables.namespaces['abilities']
    base_ability = int(arrays['base_ability'][0, owner, d])
    if base_ability in (names.get('trace', -1), names.get('colorchange', -1)):
        raise ValueError('PUBLIC_TRANSFORM_DONOR_OVERRIDDEN')
    mine = bool(np.asarray(own['transformed'])) and d == int(np.asarray(own['active']))
    stats = np.asarray(own['transform_stats']) if mine else arrays['stats'][0, owner, d, 1:]
    moves = np.asarray(own['transform_moves']) if mine else arrays['moves'][0, owner, d]
    hp_type, hp_power = (np.asarray(own['transform_hidden_power']) if mine else
        (arrays['hidden_power_type'][0, owner, d], arrays['hidden_power_power'][0, owner, d]))
    address = (0, foe, active)
    arrays['transform_stats'][0, foe] = arrays['stats'][address][1:]
    arrays['transform_moves'][0, foe] = arrays['moves'][address]
    arrays['transform_pp'][0, foe] = arrays['pp'][address]
    arrays['transform_hidden_power'][0, foe] = (arrays['hidden_power_type'][address], arrays['hidden_power_power'][address])
    used = getattr(root.public, 'copied_pp_used', None)
    used = np.zeros(4, np.int32) if used is None else np.asarray(used)[0, foe]
    arrays['stats'][address][1:] = stats
    arrays['moves'][address] = moves
    arrays['pp'][address] = [max(0, tables.copied_pp[int(m)] - int(n)) if m else 0 for m, n in zip(moves, used)]
    arrays['disabled'][address] = False
    arrays['ability'][address] = base_ability
    arrays['types'][address] = arrays['base_types'][0, owner, d]
    arrays['hidden_power_type'][address], arrays['hidden_power_power'][address] = hp_type, hp_power
    arrays['transformed'][0, foe] = True
    arrays['transform_species'][0, foe] = species
    truant = tables.namespaces['abilities'].get('truant', 0)
    if truant and base_ability == truant:
        raise ValueError('public Truant phase not materialized')   # the copy's clock, as for any Truant foe active
    # Its last move (the public caller identity) names a copied slot, or Transform itself (failencore: history slot -1).
    last = view.get('last_moves')
    move = 0 if last is None else int(np.asarray(last)[0, slot])
    if move > 0:
        match = np.flatnonzero(arrays['moves'][address] == move)
        arrays['last_move'][address] = move
        arrays['last_action_kind'][0, foe] = 2
        arrays['last_action_index'][0, foe] = int(match[0]) if len(match) == 1 else -1
        held = int(arrays['item'][address])
        if held and held == tables.namespaces['items'].get('choiceband', -1):
            arrays['choice_move'][address] = move if len(match) == 1 else 0


def world_keys(seeds):
    """empty_state(1, seed=s).keys for many seeds in one call (fold_in(PRNGKey(s), 0))."""
    import jax
    seeds = np.asarray(seeds, np.uint32)
    if not len(seeds):
        return np.zeros((0, 1, 2), np.uint32)
    # Padded to a power-of-two bucket (repeating seed 0's slot) so frames of varying world counts
    # reuse the eager executables; keys are per seed, so padding changes none of the first len(seeds).
    count = len(seeds);bucket = max(64, 1 << (count-1).bit_length())
    padded = np.concatenate([seeds, np.full(bucket-count, seeds[0], np.uint32)])
    keys = jax.vmap(lambda s: jax.vmap(jax.random.fold_in, in_axes=(None, 0))(
        jax.random.PRNGKey(s), j.arange(1, dtype=j.uint32)))(j.asarray(padded))
    return np.asarray(jax.device_get(keys))[:count]


_EMPTY = {}


def _empty_arrays():
    """Host copy of empty_state(1)'s constant fields (keys are per world)."""
    if not _EMPTY:
        _EMPTY.update({name: np.asarray(value) for name, value in empty_state(1, seed=0)._asdict().items()})
    return {name: np.array(value, copy=True) for name, value in _EMPTY.items()}


def _rebase_host(root, assignment, active, *, state):
    """_rebase on host numpy carries (same writes, same dtypes)."""
    t = root.public
    foe = 1-root.address.owner
    mapping = np.full((6,), -1, np.int32)
    for public_slot, physical in assignment.items():
        mapping[physical] = public_slot
    order = np.asarray([active, *[i for i in range(6) if i != active]], np.int32)
    def put(array, index, value):
        result = np.array(array, copy=True);result[index] = value;return result
    history = t.damage_history
    if history is not None:
        history = history._replace(
            hp=put(history.hp, (0, foe), state.hp[0, foe, active]),
            maximum=put(history.maximum, (0, foe), state.max_hp[0, foe, active]))
    return t._replace(
        damage_history=history,
        physical_to_public=put(t.physical_to_public, (0, foe), mapping),
        active_physical=put(t.active_physical, (0, foe), active),
        own_order=put(t.own_order, (0, foe), order))


def kernel_seed(root, world, recipe):
    """The hypothetical world's kernel seed (empty_state seed), as materialize() derives it."""
    return int.from_bytes(recipe._digest(root.address, 'hypothetical-kernel', world)[:4], 'big')


def materialize_host(root, start, tables, recipe, *, keys):
    """materialize() for a detach_roots Root, entirely on host numpy (the D3 frame path).

    The same checks, writes and refusals as materialize(); only its device-array steps change:
    the empty state comes from a host template plus this world's key (keys maps kernel_seed to
    world_keys of it), arrays stay numpy, and the public rebase is done in numpy. Results equal
    materialize()'s byte for byte.
    """
    return _materialize(root, start, tables, recipe, host_keys=keys)


def materialize(root, start, tables, recipe, *, transform_mapper=None):
    return _materialize(root, start, tables, recipe, transform_mapper=transform_mapper)


def _materialize(root, start, tables, recipe, host_keys=None, transform_mapper=None):
    """One declared draw -> current tensor world or named refused world slot.

    Only ordinary boundaries are represented in this first current-state port.
    Source-fixture choices are unchanged and never retried on refusal. Private
    induced-sleep/temporary volatile durations without an original supported
    treatment are refused, rather than copied from the live opposing party.
    """
    if (type(root) is not plans.Root or type(start) is not plans.Start or
            type(tables) is not Tables or type(recipe) is not Recipe or
            root.source_binding_sha256 != tables.source.binding_sha256 or
            start.world not in (0, 1) or start.seed != recipe.world_seed(root.address, start.world)):
        raise ValueError('same declared source/root/world/recipe required')
    seed = kernel_seed(root, start.world, recipe)
    key = None if host_keys is None else np.array(host_keys[seed], np.uint32)   # a missing key is a bug, not a refusal
    try:
        if start.opponent_fixtures is None:
            raise ValueError('source draw refused: '+','.join(start.refusals))
        if len(start.opponent_fixtures) != 6:
            raise ValueError('complete original sampled source party required')
        owner, foe = root.address.owner, 1-root.address.owner
        own, view = root.own, root.view
        if any(bool(own[name]) for name in ('forced_switch', 'baton_pass', 'entry_pending')):
            raise ValueError('suspended/replacement/entry boundary not materialized')
        if host_keys is None:
            state = empty_state(1, seed=seed)
            arrays = {name: np.array(value, copy=True) for name, value in state._asdict().items()}
        else:
            arrays = _empty_arrays()
            arrays['keys'] = key.reshape(arrays['keys'].shape)
        for name in plans.OWN_MON_FIELDS + plans.OWN_SIDE_FIELDS:
            value = np.asarray(own[name])
            if value.shape != arrays[name][0, owner].shape or value.dtype != arrays[name].dtype:
                raise ValueError('exact owner current cell shape/dtype: '+name)
            arrays[name][0, owner] = value
        assignment = {slot: physical for physical, (slot, _) in enumerate(start.revealed_source_indices)}
        actual_present = tuple(np.flatnonzero(view['present'][0]))
        if tuple(assignment) != actual_present:
            raise ValueError('source fixtures/current public encounter slots differ')
        active_slots = np.flatnonzero(view['active'][0])
        if len(active_slots) != 1 or int(active_slots[0]) not in assignment:
            raise ValueError('one currently known public opposing active required')
        active = assignment[int(active_slots[0])]
        arrays['active'][0, foe] = active
        # A real owner's Mean Look link uses a physical opponent coordinate.
        # Rebase it using the PUBLIC current trap, never that donor coordinate.
        for physical in np.flatnonzero(arrays['trap_source'][0, owner] >= 0):
            features = view.get('own_mon_features')
            order = view['own_order'][0]
            slot = np.flatnonzero(order == physical)
            if (features is None or 'meanlook_trap' not in features or len(slot) != 1 or
                    not features['meanlook_trap'][0, int(slot[0])]):
                raise ValueError('owner cross-party trap link has no public current source')
            arrays['trap_source'][0, owner, physical] = active
        for physical, mon in enumerate(start.opponent_fixtures):
            address = (0, foe, physical)
            universe = tables.source.original.universe_for(mon.species)
            if universe is None or mon.nature != '' or not any(
                    (v.level, tuple(v.moves), normalize_id(v.ability), normalize_id(v.item)) ==
                    (mon.level, tuple(mon.moves), normalize_id(mon.ability), normalize_id(mon.item))
                    for v in universe.variants):
                raise ValueError('sampled fixture differs from original source row')
            spread = _gen3_randbat_fixture_spread({}, species=mon.species,
                moves=tuple(mon.moves), item=mon.item, level=mon.level,
                set_source=tables.source.original)
            if spread is None or dict(mon.evs or {}) != dict(spread['evs']) or dict(mon.ivs or {}) != dict(spread['ivs']):
                raise ValueError('sampled fixture differs from original source spread')
            info = tables.source.original.species_metadata.get(_normalize_species_id(mon.species), {})
            base = info.get('baseStats')
            types = info.get('types')
            if base is None or not types or not 1 <= len(types) <= 2:
                raise ValueError('original sampled species stats/types unavailable')
            stats = [gen3_stat(int(base[key]), int(mon.ivs[key]), int(mon.evs[key]), mon.level)
                for key in ('atk', 'def', 'spa', 'spd', 'spe')]
            maximum = 1 if base['hp'] == 1 else gen3_hp_stat(int(base['hp']), int(mon.ivs['hp']),
                int(mon.evs['hp']), mon.level)
            arrays['species'][address] = _id(tables, 'species', mon.species)
            arrays['level'][address] = mon.level
            arrays['stats'][address] = [maximum, *stats]
            arrays['hp'][address] = arrays['max_hp'][address] = maximum
            arrays['gender'][address] = {'N': 0, 'M': 1, 'F': 2}[mon.gender]
            arrays['ability'][address] = arrays['base_ability'][address] = _id(tables, 'abilities', mon.ability, absent=True)
            arrays['item'][address] = _id(tables, 'items', mon.item, absent=True)
            kinds = [_id(tables, 'types', name) for name in types]
            arrays['types'][address] = arrays['base_types'][address] = kinds+[0]*(2-len(kinds))
            if not 1 <= len(mon.moves) <= 4:
                raise ValueError('sampled source moves unavailable')
            for slot, name in enumerate(mon.moves):
                move = _id(tables, 'moves', name)
                arrays['moves'][address+(slot,)] = move
                arrays['pp'][address+(slot,)] = tables.max_pp[move]
                if normalize_id(name).startswith('hiddenpower'):
                    resolved = hidden_power_engine_id(normalize_id(name), mon.ivs)
                    typ = resolved[len('hiddenpower'):-2]
                    arrays['hidden_power_type'][address] = _id(tables, 'types', typ)
                    arrays['hidden_power_power'][address] = int(resolved[-2:])
        for public_slot, physical in assignment.items():
            address = (0, foe, physical)
            fraction = float(view['hp_fraction'][0, public_slot])
            percents = view.get('hp_percent')
            if percents is None or percents.shape != (1, 6) or percents.dtype != np.int32:
                raise ValueError('literal public integer HP percentage unavailable')
            shown = int(percents[0, public_slot])
            status = int(view['status'][0, public_slot])
            if (not np.isfinite(fraction) or not 0 <= shown <= 100 or
                    abs(fraction-shown/100) > 1e-6):
                raise ValueError('public current HP fraction malformed')
            # Exact original _hp_and_status scaling, including known faint0.
            maximum = int(arrays['max_hp'][address])
            arrays['hp'][address] = max(1, round(shown*maximum/100)) if shown else 0
            if not shown:
                status = 0  # Original materializer's known-faint treatment.
            if status not in (0, 1, 2, 3, 4, 6):
                raise ValueError('opposing sleep counter/provenance unavailable')
            arrays['status'][address] = status
            arrays['boosts'][address] = view['boosts'][0, public_slot]
            shown_types = np.asarray(view['types'][0, public_slot])
            if np.any(shown_types):
                arrays['types'][address] = shown_types
            copied = view.get('borrowed_ability')
            if copied is not None and copied[0, public_slot]:
                arrays['ability'][address] = copied[0, public_slot]
            removed = view.get('item_removed')
            if removed is None:
                raise ValueError('public current item removal state unavailable')
            if removed[0, public_slot]:
                arrays['item'][address] = 0
            elif view['item'][0, public_slot]:
                arrays['item'][address] = view['item'][0, public_slot]
            moves = view['revealed_moves'][0, public_slot]
            used = view.get('move_pp_used')
            if used is None:
                raise ValueError('public current PP-use counts unavailable')
            for shown, count in zip(moves, used[0, public_slot]):
                if not shown:
                    continue
                name = normalize_id(tables.source.vocabulary['moves'][int(shown)])
                matching = [k for k, value in enumerate(start.opponent_fixtures[physical].moves)
                    if normalize_id(value) == name or name.startswith('hiddenpower') and normalize_id(value).startswith('hiddenpower')]
                if name == 'struggle':
                    continue
                if len(matching) != 1 or int(count) < 0:
                    raise ValueError('public PP-use/source move join unavailable')
                slot = matching[0]
                arrays['pp'][address+(slot,)] = max(0, int(arrays['pp'][address+(slot,)])-int(count))
            if status == 4:
                arrays['status_counter'][address] = min(15, int(_mon_feature(view, 'toxic_stage', public_slot)))
            if physical == active:
                arrays['active_turns'][0, foe] = max(0, int(_mon_feature(view, 'turns_active', public_slot))-1)
                arrays['protect_counter'][address] = int(_mon_feature(view, 'stall_counter', public_slot))
                for clock in ('confusion_turns', 'encore_turns'):
                    if _mon_feature(view, clock, public_slot):
                        raise ValueError('public current clock not materialized: '+clock)
                if _mon_feature(view, 'sub_present', public_slot):
                    raise ValueError('public substitute HP/origin not materialized')
                if _mon_feature(view, 'meanlook_trap', public_slot):
                    arrays['trap_source'][address] = int(arrays['active'][0, owner])
                bag = view.get('volatiles')
                if bag is None:
                    raise ValueError('public volatile bag unavailable')
                for kind in np.flatnonzero(bag[0, public_slot]):
                    name = VOLATILES[int(kind)]
                    if name == 'leechseed':arrays['volatile'][address] |= LEECH_SEED
                    elif name == 'ingrain':arrays['volatile'][address] |= ROOTED
                    elif name == 'attract':arrays['volatile'][address] |= ATTRACT
                    elif name == 'trapped':arrays['trap_source'][address] = int(arrays['active'][0, owner])
                    else:raise ValueError('public volatile/latent clock not materialized: '+name)
                last = view.get('last_moves')
                if last is None:
                    raise ValueError('public caller identity unavailable')
                move = int(last[0, public_slot])
                marks = view.get('transformed')
                if marks is not None and bool(np.asarray(marks)[0, public_slot]):
                    move = 0   # Transform: its last move names the copy's slots (_materialize_transform)
                if move > 0:
                    match = np.flatnonzero(arrays['moves'][address] == move)
                    name = normalize_id(tables.source.vocabulary['moves'][move])
                    if name.startswith('hiddenpower'):
                        match = np.asarray([k for k, value in enumerate(start.opponent_fixtures[physical].moves)
                            if normalize_id(value).startswith('hiddenpower')])
                    if len(match) != 1:
                        raise ValueError('public last caller/source slot unavailable')
                    move = int(arrays['moves'][address+(int(match[0]),)])
                    arrays['last_move'][address] = move
                    arrays['last_action_kind'][0, foe] = 2
                    arrays['last_action_index'][0, foe] = int(match[0])
                    if normalize_id(start.opponent_fixtures[physical].item or '') == 'choiceband' and arrays['item'][address]:
                        arrays['choice_move'][address] = move
                elif move in (-1, -2):
                    arrays['last_action_kind'][0, foe] = 1
                    arrays['last_action_index'][0, foe] = active
                truant = tables.namespaces['abilities'].get('truant', 0)
                if truant and arrays['ability'][address] == truant:
                    raise ValueError('public Truant phase not materialized')
        (_materialize_transform if transform_mapper is None else transform_mapper)(root, tables, arrays, owner, foe, active, assignment)
        if arrays['hp'][0, foe, active] <= 0:
            raise ValueError('opposing faint/replacement boundary not materialized')
        fields = view.get('field_features')
        if fields is None:
            raise ValueError('public current field clocks unavailable')
        if bool(fields['opp_wish_pending'][0]):
            raise ValueError('public Wish caster amount unavailable')
        weather, turn = int(view['weather'][0]), int(view['turn'][0])
        if not 0 <= weather <= 4 or turn < 1:
            raise ValueError('public weather/turn namespace malformed')
        arrays['weather'][0] = weather
        if weather:
            remaining = int(fields['weather_turns'][0])
            if not bool(fields['weather_permanent'][0]) and not 1 <= remaining <= 5:
                raise ValueError('public weather duration unavailable')
            arrays['weather_counter'][0] = 0 if bool(fields['weather_permanent'][0]) else remaining
        arrays['spikes'][0] = view['spikes'][0]
        arrays['turn'][0] = turn-1
        arrays['entry_pending'][0] = False
        arrays['done'][0], arrays['winner'][0] = False, -1
        if host_keys is None:
            state = type(state)(**{name: j.asarray(value) for name, value in arrays.items()})
            carry = _rebase(root, assignment, active, state=state)
        else:
            state = BattleState(**arrays)
            carry = _rebase_host(root, assignment, active, state=state)
        return World(start.world, state, carry, (), seed, root.reply_private_history_available)
    except (ValueError, KeyError, IndexError, TypeError) as error:
        return World(start.world, None, None, (str(error),), seed, False)
