"""D3 device world preparation: batched detach, source sampling and materialization on device.

One jitted call per root batch runs detach_roots' primitive sequence over the batch's roots
(gathered by scan step and row), the V2ePrepWorlds CUDA kernel (plan_starts' _revealed and the
two CPython-seeded fixture draws, then materialize_host's checks and writes, per world) and the
expansion of its compact records into the pricer's world tensors: BattleState leaves and the
rebased public carries, [capacity, 2, ...]. The host keeps only what the pricer's _result needs:
world seeds, kernel seeds, refusal strings, revealed source indices and availability. Results
equal plan_starts + materialize_host on detach_roots roots byte for byte (device_d_frame_parity
compares them); the flag that selects this path is off by default.
"""
from dataclasses import dataclass
import time
from typing import NamedTuple

import jax
import jax.numpy as j
import numpy as np

from pokezero import v5_contract as contract
from pokezero.v5_gumbel import Recipe
from . import device_public_source_tables as tables_
from . import device_public_world_state as mapping
from . import device_public_worlds as plans
from . import engine_device_trackers as public
from . import v2e_cuda
from .state import BattleState

OWN_FIELDS = plans.OWN_MON_FIELDS + plans.OWN_SIDE_FIELDS
TARGET = 'v2e_V2ePrepWorlds'


class SampledFixture(NamedTuple):
    """One sampled source fixture: universe index (set_source order), variant index, gender code."""
    universe: int
    variant: int
    gender: int


class DeviceWorld(NamedTuple):
    """World.state/public marker of an available device-prepared world (its arrays stay on device)."""
    batch: int
    index: int


@dataclass(frozen=True)
class DeviceRoot:
    """Plans.root of the device path: the address and a handle into its prepared batch."""
    address: contract.DecisionAddress
    batch: object
    index: int
    source_binding_sha256: str
    reply_private_history_available: bool


class _Addressed:
    def __init__(self, address):
        self.address = address


def _pad(values, size, fill):
    return np.asarray(list(values) + [fill] * (size - len(values)), np.int32)


def _capacity(count):
    return 1 if count <= 1 else 1 << (count - 1).bit_length()


def _static_revealed_error(view):
    """_revealed's shape/dtype checks on the batched view: (before the per-root error check, after)."""
    def malformed(name, shape, dtype):
        value = getattr(view, name)
        return (value is None or tuple(value.shape[1:]) != shape or value.dtype != np.dtype(dtype))
    if malformed('error', (), 'int32'):
        return 'public field unavailable/malformed: error', None
    for name, shape, dtype in (('species', (6,), 'int32'), ('present', (6,), 'bool'), ('belief_mask', (6, 2), 'uint32'),
            ('revealed_moves', (6, 4), 'int32'), ('ability', (6,), 'int32'), ('item', (6,), 'int32'),
            ('gender', (6,), 'int32'), ('level', (6,), 'int32')):
        if malformed(name, shape, dtype):
            return None, 'public field unavailable/malformed: ' + name
    features = view.mon_features
    swapped = None if features is None else features.get('item_swapped')
    if swapped is not None and (tuple(swapped.shape[1:]) != (6,) or swapped.dtype not in (np.dtype('int32'), np.dtype('bool'))):
        return None, 'public item mutation facts malformed'
    return None, None


class Preparer:
    """Compiled source tables plus the per-shape jitted preparation functions."""

    def __init__(self, source, tables, recipe):
        if type(source) is not plans.Source or type(tables) is not mapping.Tables or type(recipe) is not Recipe:
            raise ValueError('bound public source, materializer tables and original recipe required')
        self.source, self.tables, self.recipe = source, tables, recipe
        self.device = tables_.compile_device_source(tables)
        v2e_cuda.load()
        d = self.device
        self.device_tables = jax.device_put(dict(misc=d.misc, uni=d.uni, var=d.var, s2u=d.species_to_universe,
            hp=d.move_is_hp))
        self.template = mapping._empty_arrays()
        self._prepare = jax.jit(self._prepare_impl)
        self._finalize = jax.jit(self._finalize_impl)
        self.batches = 0
        # Host-static _revealed refusal strings, interned at trace time after the source's own strings.
        self.extra_strings = []

    def _intern(self, text):
        if text not in self.extra_strings:
            self.extra_strings.append(text)
        return len(self.device.strings) + self.extra_strings.index(text)

    @property
    def strings(self):
        return self.device.strings + tuple(self.extra_strings)

    # ---- host: seeds ---------------------------------------------------------------------
    def _seeds(self, addresses, capacity):
        words = np.zeros((capacity, 2, tables_.SEED_WORDS), np.uint32)
        kernel = np.zeros((capacity, 2), np.uint32)
        world_seeds = []
        for i, address in enumerate(addresses):
            pair = []
            for world in (0, 1):
                seed = self.recipe.world_seed(address, world)
                pair.append(seed)
                words[i, world] = np.frombuffer(seed.to_bytes(4 * tables_.SEED_WORDS, 'little'), np.uint32)
                kernel[i, world] = mapping.kernel_seed(_Addressed(address), world, self.recipe)
            world_seeds.append(tuple(pair))
        return words, kernel, world_seeds

    # ---- device: detach + kernel + expansion ----------------------------------------------
    def _kernel_inputs(self, view, own, hp_percent, owners, seeds, count):
        d = self.device
        b = owners.shape[0]
        i32 = lambda x: j.asarray(x).astype(j.int32)
        zeros = lambda *shape: j.zeros((b,) + shape, j.int32)
        flags = 0
        optional = {}
        for flag, name, shape in (('F_BORROWED', 'borrowed_ability', (6,)), ('F_REMOVED', 'item_removed', (6,)),
                ('F_PP_USED', 'move_pp_used', (6, 4)), ('F_LAST_MOVES', 'last_moves', (6,))):
            value = getattr(view, name)
            if value is not None:
                flags |= tables_.F[flag]
                optional[name] = i32(value)
            else:
                optional[name] = zeros(*shape)
        volatile_count = d.volatile_count
        if view.volatiles is not None:
            if view.volatiles.shape[1:] != (6, volatile_count):
                raise NotImplementedError('device preparation: public volatile bag axis differs from the volatile names')
            flags |= tables_.F['F_VOLATILES']
            volatiles = i32(view.volatiles)
        else:
            volatiles = zeros(6, volatile_count)
        features = []
        if view.mon_features is not None:
            flags |= tables_.F['F_MON_FEATURES']
        for index, name in enumerate(tables_.FEATURES):
            value = None if view.mon_features is None else view.mon_features.get(name)
            if value is None:
                features.append(zeros(6))
                continue
            if value.dtype not in (np.dtype('int32'), np.dtype('bool')) or value.shape[1:] != (6,):
                raise NotImplementedError('device preparation requires int32/bool public mon feature: ' + name)
            flags |= 1 << (tables_.FEATURE_FLAG_BASE + index)
            features.append(i32(value))
        features = j.stack(features, axis=1)
        own_features = view.own_mon_features
        if own_features is not None and 'meanlook_trap' in own_features:
            flags |= tables_.F['F_OWN_MEANLOOK']
            own_meanlook = i32(own_features['meanlook_trap'])
        else:
            own_meanlook = zeros(6)
        fields = view.field_features
        if fields is not None:
            flags |= tables_.F['F_FIELDS']
            for key in ('opp_wish_pending', 'weather_turns', 'weather_permanent'):
                if key not in fields or fields[key].shape != (b,):
                    raise NotImplementedError('device preparation requires per-root public field feature: ' + key)
            field_values = j.stack([i32(fields['opp_wish_pending']), i32(fields['weather_turns']),
                i32(fields['weather_permanent'])], axis=1)
        else:
            field_values = zeros(3)
        belief = view.belief_mask if view.belief_mask is not None else j.zeros((b, 6, 2), j.uint32)
        before, after = _static_revealed_error(view)
        pre_before = 0 if before is None else self._intern(before)
        pre_after = 0 if after is None else self._intern(after)
        own_flags = j.stack([i32(own['forced_switch']), i32(own['baton_pass']), i32(own['entry_pending'])], axis=1)
        t = self.device_tables
        inputs = [t['misc'], t['uni'], t['var'], t['s2u'], t['hp'], i32(owners), i32(view.species), i32(view.present),
            i32(view.active), j.asarray(belief, j.uint32), i32(view.revealed_moves), i32(view.ability), i32(view.item),
            i32(view.gender), i32(view.level), i32(view.error), j.asarray(view.hp_fraction, j.float32), i32(hp_percent),
            i32(view.status), i32(view.boosts), i32(view.types), optional['borrowed_ability'], optional['item_removed'],
            optional['move_pp_used'], optional['last_moves'], volatiles, i32(view.weather), i32(view.turn),
            i32(view.spikes), i32(view.own_order), features, own_meanlook, field_values, i32(own['active']),
            i32(own['trap_source']), own_flags, j.asarray(seeds, j.uint32), i32(count).reshape(1)]
        if len(inputs) != len(tables_.INPUTS):
            raise ValueError('device preparation kernel input count differs from the layout')
        attributes = dict(flags=np.int64(flags), pre_error=np.int64(pre_before), pre_error_after=np.int64(pre_after),
            volatile_count=np.int64(volatile_count), universes=np.int64(len(d.uni)), variants=np.int64(d.var.shape[1]),
            species=np.int64(len(d.species_to_universe)), moves=np.int64(len(d.move_is_hp)))
        return inputs, attributes

    def _prepare_impl(self, context, steps, rows, owners, seeds, kernel_seeds, count):
        b = rows.shape[0]
        lane = j.arange(b, dtype=j.int32)
        state = jax.tree.map(lambda x: x[steps, rows], context.state)
        cloned = public.clone_for_worlds(jax.tree.map(lambda x: x[steps, rows], context.public), lane)
        view = public.readout(cloned, owners)
        own = {name: getattr(state, name)[lane, owners] for name in OWN_FIELDS}
        hp_percent = cloned.hp_percent[lane, 1 - owners]
        single = lambda row, owner: plans._redact_peer_private(jax.tree.map(lambda x: x[None], row), owner)
        redacted = jax.tree.map(lambda x: x[:, 0], jax.vmap(single)(cloned, owners))
        streams, frames = context.streams[steps, rows], context.frames[steps, rows]
        present = view.present
        history_available = ((frames == 0) & (view.error == 0)
            & j.all(j.where(present, view.hp_fraction == 1, True), axis=-1)
            & ~j.any(j.where(present, view.status != 0, False), axis=-1))
        inputs, attributes = self._kernel_inputs(view, own, hp_percent, owners, seeds, count)
        shape = lambda *dims: jax.ShapeDtypeStruct((b, 2) + dims, np.int32)
        code, mon, side, rev, fix = jax.ffi.ffi_call(TARGET, (shape(len(tables_.CODE_COLUMNS)),
            shape(6, len(tables_.MON_COLUMNS)), shape(len(tables_.SIDE_COLUMNS)), shape(6), shape(6, len(tables_.FIX_COLUMNS))))(
            *inputs, **attributes)
        keys = jax.vmap(lambda s: jax.vmap(jax.random.fold_in, in_axes=(None, 0))(
            jax.random.PRNGKey(s), j.arange(1, dtype=j.uint32)))(kernel_seeds.reshape(-1))
        states = self._expand(own, mon, side, keys.reshape(2 * b, 2), owners, b)
        publics = self._rebase(redacted, states, side, fix, owners, b)
        return dict(states=states, publics=publics, code=code, rev=rev, fix=fix, history_available=history_available,
            streams=streams, frames=frames)

    def _expand(self, own, mon, side, keys, owners, b):
        """materialize_host's state writes for every world: own cells on the owner's side, the kernel's
        foe records on the other, empty-state defaults elsewhere; leaves [b, 2, 1, ...] (the pricer stacks
        materialize()'s one-row arrays, so every world keeps a singleton batch axis)."""
        w = 2 * b
        M, S = tables_.M, tables_.S
        owner_w = j.repeat(owners, 2)
        mon = mon.reshape((w, 6, mon.shape[-1]))
        side = side.reshape((w, side.shape[-1]))
        template = self.template

        def own_w(name):
            return j.repeat(own[name], 2, axis=0)

        def both(name, own_value, foe_value):
            dtype = template[name].dtype
            own_value = j.asarray(own_value).astype(dtype)
            foe_value = j.broadcast_to(j.asarray(foe_value).astype(dtype), own_value.shape)
            mask = owner_w.reshape((w,) + (1,) * (own_value.ndim - 1)) == 0
            return j.stack([j.where(mask, own_value, foe_value), j.where(mask, foe_value, own_value)], axis=1)

        def column(name):
            return mon[..., M[name]]

        def columns(first, last):
            return mon[..., M[first]:M[last] + 1]

        def scol(name):
            return side[:, S[name]]
        leaves = {}
        mon_columns = dict(species='M_SPECIES', level='M_LEVEL', hidden_power_type='M_HP_TYPE', hidden_power_power='M_HP_POWER',
            gender='M_GENDER', hp='M_HP', max_hp='M_MAX_HP', status='M_STATUS', status_counter='M_STATUS_COUNTER',
            choice_move='M_CHOICE_MOVE', last_move='M_LAST_MOVE', item='M_ITEM', ability='M_ABILITY',
            base_ability='M_BASE_ABILITY', protect_counter='M_PROTECT_COUNTER', volatile='M_VOLATILE',
            trap_source='M_TRAP_SOURCE')
        for name, col in mon_columns.items():
            if name == 'trap_source':
                leaves[name] = both(name, side[:, S['S_OWN_TRAP0']:S['S_OWN_TRAP5'] + 1], column(col))
            else:
                leaves[name] = both(name, own_w(name), column(col))
        for name, first, last in (('types', 'M_TYPE0', 'M_TYPE1'), ('base_types', 'M_BTYPE0', 'M_BTYPE1'),
                ('stats', 'M_STAT0', 'M_STAT5'), ('moves', 'M_MOVE0', 'M_MOVE3'), ('pp', 'M_PP0', 'M_PP3'),
                ('boosts', 'M_BOOST0', 'M_BOOST6')):
            leaves[name] = both(name, own_w(name), columns(first, last))
        for name in ('happiness', 'sleep_skipped', 'rest_origin', 'disabled', 'encore_move', 'encore_counter',
                'charge_move', 'charge_counter', 'confusion_counter', 'substitute_hp', 'truant_loaf', 'trapped',
                'forced_switch', 'baton_pass', 'wish_counter', 'wish_amount'):
            leaves[name] = both(name, own_w(name), template[name][0, 1])
        for name, col in (('active', 'S_ACTIVE'), ('active_turns', 'S_ACTIVE_TURNS'), ('last_action_kind', 'S_LAST_KIND'),
                ('last_action_index', 'S_LAST_INDEX')):
            leaves[name] = both(name, own_w(name), scol(col))
        leaves['entry_pending'] = j.zeros((w, 2), bool)
        leaves['spikes'] = side[:, S['S_SPIKES0']:S['S_SPIKES1'] + 1]
        for name in ('queued_actions', 'queued_stall', 'phazing_draws', 'secondary_draws', 'sleep_talk_draws',
                'queued_actor', 'pending_residual', 'residual_draws', 'replacement_count'):
            leaves[name] = j.broadcast_to(j.asarray(template[name][0]), (w,) + template[name].shape[1:])
        leaves['weather'] = scol('S_WEATHER')
        leaves['weather_counter'] = scol('S_WEATHER_COUNTER')
        leaves['turn'] = scol('S_TURN')
        leaves['keys'] = j.asarray(keys, j.uint32)
        leaves['done'] = j.zeros((w,), bool)
        leaves['winner'] = j.full((w,), -1, j.int32)
        if set(leaves) != set(template):
            raise ValueError('device expansion covers different state fields than empty_state: '
                + str(sorted(set(leaves) ^ set(template))))
        for name, value in leaves.items():
            if value.dtype != template[name].dtype or value.shape[1:] != template[name].shape[1:]:
                raise ValueError(f'device expansion leaf {name}: {value.shape}{value.dtype} vs {template[name].shape}{template[name].dtype}')
        return BattleState(**{name: leaves[name].astype(template[name].dtype).reshape((b, 2, 1) + template[name].shape[1:])
            for name in template})

    def _rebase(self, redacted, states, side, fix, owners, b):
        """_rebase_host for every world: only sampled coordinates replace the redacted peer bookkeeping."""
        w = 2 * b
        S, FX = tables_.S, tables_.FX
        lane = j.arange(w)
        foe = 1 - j.repeat(owners, 2)
        side = side.reshape((w, side.shape[-1]))
        fix = fix.reshape((w, 6, fix.shape[-1]))
        active = side[:, S['S_ACTIVE']]
        mapping_ = fix[..., FX['FX_SLOT']]
        others = j.arange(5, dtype=j.int32)[None, :]
        order = j.concatenate([active[:, None], others + (others >= active[:, None]).astype(j.int32)], axis=1)
        flat = lambda x: x.reshape((w,) + x.shape[3:])
        hp, max_hp = flat(states.hp), flat(states.max_hp)
        t = jax.tree.map(lambda x: j.repeat(x, 2, axis=0), redacted)
        history = t.damage_history
        if history is not None:
            history = history._replace(hp=history.hp.at[lane, foe].set(hp[lane, foe, active]),
                maximum=history.maximum.at[lane, foe].set(max_hp[lane, foe, active]))
        t = t._replace(damage_history=history, physical_to_public=t.physical_to_public.at[lane, foe].set(mapping_),
            active_physical=t.active_physical.at[lane, foe].set(active), own_order=t.own_order.at[lane, foe].set(order))
        return jax.tree.map(lambda x: x.reshape((b, 2, 1) + x.shape[1:]), t)

    def _finalize_impl(self, states, publics, src):
        """FramePricer.price's world packing: refused worlds replaced by the first available one, padding by world 0."""
        capacity = src.shape[0] // 2
        gather = lambda x: j.take(x.reshape((-1,) + x.shape[2:]), src, axis=0).reshape((capacity, 2) + x.shape[2:])
        return jax.tree.map(gather, states), jax.tree.map(gather, publics)

    # ---- one root batch --------------------------------------------------------------------
    def prepare(self, context, *, steps, rows, owners, addresses):
        """Detach, sample and materialize every root (step, row, owner) of one device RootContext."""
        from .v5_device_rollout import RootContext
        if type(context) is not RootContext:
            raise ValueError('actual device root context required')
        steps, rows, owners, addresses = tuple(steps), tuple(rows), tuple(owners), tuple(addresses)
        n = len(rows)
        if not n or len({len(steps), len(rows), len(owners), len(addresses)}) != 1:
            raise ValueError('one address per root required')
        if any(type(s) is not int or type(r) is not int or type(o) is not int or o not in (0, 1)
                or not 0 <= s < context.streams.shape[0] or not 0 <= r < context.streams.shape[1]
                for s, r, o in zip(steps, rows, owners)):
            raise ValueError('actual root step/row and owner0/1 required')
        if any(type(a) is not contract.DecisionAddress or a.owner != o for a, o in zip(addresses, owners)):
            raise ValueError('typed decision address per root required')
        started = time.perf_counter()
        capacity = _capacity(n)
        words, kernel, world_seeds = self._seeds(addresses, capacity)
        seeded = time.perf_counter()
        out = self._prepare(context, _pad(steps, capacity, steps[0]), _pad(rows, capacity, rows[0]),
            _pad(owners, capacity, owners[0]), words, kernel, np.int32(n))
        dispatched = time.perf_counter()
        host = jax.device_get(dict(code=out['code'], rev=out['rev'], fix=out['fix'], history=out['history_available'],
            streams=out['streams'], frames=out['frames']))
        copied = time.perf_counter()
        for i, address in enumerate(addresses):
            if int(host['streams'][i]) != address.stream or int(host['frames'][i]) != address.frame:
                raise ValueError('PRE root context address differs')
        self.batches += 1
        batch = Batch(self, self.batches, capacity, n, out['states'], out['publics'], host['code'], host['rev'],
            host['fix'], kernel, world_seeds, self.strings)
        batch.plans = tuple(batch._plans(i, address, bool(host['history'][i])) for i, address in enumerate(addresses))
        batch.timing = dict(roots=n, capacity=capacity, seed_seconds=seeded - started, dispatch_seconds=dispatched - seeded,
            copy_seconds=copied - dispatched, decode_seconds=time.perf_counter() - copied,
            total_seconds=time.perf_counter() - started)
        return batch


class Batch:
    """One prepared root batch: device world tensors [capacity, 2, ...] plus host metadata per world."""

    def __init__(self, preparer, identity, capacity, count, states, publics, code, rev, fix, kernel_seeds, world_seeds, strings):
        self.preparer, self.identity, self.capacity, self.count = preparer, identity, capacity, count
        self.states, self.publics = states, publics
        self.code, self.rev, self.fix = code, rev, fix
        self.kernel_seeds, self.world_seeds, self.strings = kernel_seeds, world_seeds, strings
        self.plans = ();self.timing = None;self._host = None

    def refusals(self, index, world):
        """(plan refusal, materialization refusal) strings of one world, None when absent."""
        plan, materialize = (int(x) for x in self.code[index, world])
        if plan < 0:
            raise ValueError('padding world has no refusal')
        plan_text = self.strings[plan] if plan else None
        if plan_text is not None:
            return plan_text, 'source draw refused: ' + plan_text
        return None, (self.strings[materialize] if materialize else None)

    def available(self, index, world):
        return self.refusals(index, world) == (None, None)

    def _plans(self, index, address, history_available):
        source = self.preparer.source
        root = DeviceRoot(address, self, index, source.binding_sha256, history_available)
        starts = []
        for world in (0, 1):
            plan_text, _ = self.refusals(index, world)
            rev = self.rev[index, world]
            indices = tuple((slot, int(rev[slot])) for slot in range(6) if rev[slot] >= 0)
            fixtures = None
            if plan_text is None:
                fix = self.fix[index, world]
                fixtures = tuple(SampledFixture(int(u), int(k), int(g)) for u, k, g, _ in fix.tolist())
            starts.append(plans.Start(world, self.world_seeds[index][world], fixtures, indices,
                () if plan_text is None else (plan_text,)))
        return plans.Plans(root, True, tuple(starts), source.source_hash, source.source_rows_sha256,
            sum(start.opponent_fixtures is None for start in starts) / 2)

    def world(self, index, world, *, materialized=False):
        """mapping.World of one world; materialized=True slices host copies of its state and public arrays."""
        _, refusal = self.refusals(index, world)
        seed = int(self.kernel_seeds[index, world])
        root = self.plans[index].root
        if refusal is not None:
            return mapping.World(world, None, None, (refusal,), seed, False)
        if materialized:
            state, carry = self.host_world(2 * index + world)
        else:
            state = carry = DeviceWorld(self.identity, 2 * index + world)
        return mapping.World(world, state, carry, (), seed, root.reply_private_history_available)

    def host_world(self, flat):
        if self._host is None:
            flatten = lambda x: x.reshape((-1,) + x.shape[2:])
            self._host = jax.device_get((jax.tree.map(flatten, self.states), jax.tree.map(flatten, self.publics)))
        one = lambda x: np.ascontiguousarray(x[flat])   # [1, ...]: the world's own singleton batch axis
        return jax.tree.map(one, self._host[0]), jax.tree.map(one, self._host[1])

    def flat_worlds(self):
        flatten = lambda x: x.reshape((-1,) + x.shape[2:])
        return jax.tree.map(flatten, self.states), jax.tree.map(flatten, self.publics)


def price_worlds(requests, recipe, capacity, *, materialized=False):
    """FramePricer.price's device preparation: (worlds per request, packed states, packed publics).

    Every request's root must be a DeviceRoot. The packed tensors are None when no world is available.
    """
    roots = [r.plans.root for r in requests]
    if any(type(root) is not DeviceRoot for root in roots):
        raise ValueError('device-prepared roots required for device preparation pricing')
    worlds = []
    for request, root in zip(requests, roots):
        batch = root.batch
        if request.plans is not batch.plans[root.index]:
            raise ValueError('request plans differ from their prepared batch')
        worlds.append(tuple(batch.world(root.index, start.world, materialized=materialized) for start in request.plans.starts))
    count = len(requests)
    available = np.asarray([w.state is not None for pair in worlds for w in pair], bool)
    if not available.any():
        return worlds, None, None
    flat = np.arange(2 * count)
    first = int(np.flatnonzero(available)[0])
    src = np.full(2 * capacity, first, np.int32)
    src[:2 * count] = np.where(available, flat, first)
    src[2 * count:] = src[0]
    preparer = roots[0].batch.preparer
    same = (all(root.batch is roots[0].batch for root in roots) and roots[0].batch.capacity == capacity
        and all(root.index == i for i, root in enumerate(roots)))
    if same:
        states, publics = roots[0].batch.states, roots[0].batch.publics
    else:
        # General packing (saved-root gates, mixed batches): gather each world from its batch, pad to capacity.
        pieces = []
        for root in roots:
            s, p = root.batch.flat_worlds()
            pieces.append(jax.tree.map(lambda x: x[2 * root.index:2 * root.index + 2], (s, p)))
        pad = capacity - count
        def packed(*leaves):
            value = j.concatenate(leaves, axis=0)
            if pad:
                value = j.concatenate([value, j.broadcast_to(value[:1], (2 * pad,) + value.shape[1:])], axis=0)
            return value.reshape((capacity, 2) + value.shape[1:])
        states, publics = jax.tree.map(packed, *pieces)
    states, publics = preparer._finalize(states, publics, j.asarray(src))
    return worlds, states, publics
