"""V2-a refill collector against the production fixed-cohort collector.

Inert row-local encoder/actor/transition seams (as in the device rollout
tests); real sampling keys, refusal rules, Frames, training_batch and the
original Torch GAE. No real models, engine turns or games.
"""
from types import SimpleNamespace as NS
import math
import unittest
from unittest.mock import patch

import jax
import jax.numpy as j
import numpy as np

from gpu_engine import v2_collector as v2, v5_device_rollout as d, transition, engine_device_observation as e, v5_model
from gpu_engine import engine_device_public_hooks as h, benchmark_device_selfplay as sampling
from gpu_engine.state import empty_state
from pokezero.v5_model import TOKEN_ROLES
from pokezero.v5_loss import V5LossConfig
from tests.test_gpu_engine_v5_device_rollout import fixture

MAX_REQUESTS = 6
CHUNK = 4
ARCHIVES = ('2'*64, '3'*64, '6'*64)
# Per-game fake dynamics: foe HP pool, turn limit (tie), initial forced switch.
HP = (5, 9, 3, 30, 7, 4, 11, 6, 8)
LIMIT = (9, 9, 9, 9, 7, 4, 9, 9, 9)   # LIMIT 7 also poisons the actor at turn 2 -> REFUSED


def encode(state, view, tables, *, observer_sides, legal_action_mask):
    b = len(observer_sides)
    numeric = j.zeros((b, 29, 132), j.float32)
    numeric = numeric.at[:, 0, 0].set(state.turn.astype(j.float32)).at[:, 0, 1].set(state.hp[:, 1, 0].astype(j.float32))
    numeric = numeric.at[:, 0, 2].set(state.level[:, 0, 0].astype(j.float32)).at[:, 0, 3].set(observer_sides.astype(j.float32))
    categorical = j.zeros((b, 29, 41), j.int32).at[:, 0, 0].set(state.hp[:, 0, 0] % 7)
    return e.Observation(numeric, categorical, j.broadcast_to(j.array(TOKEN_ROLES, j.int32), (b, 29)),
        j.ones((b, 29), bool), legal_action_mask, j.broadcast_to(j.arange(1, 6, dtype=j.int32), (b, 5)),
        j.zeros(b, j.int32), j.zeros(len(e.ERROR_KINDS), j.int32), j.broadcast_to(j.arange(4, dtype=j.int32), (b, 4)))


def forward(params, inputs, config, *, compute_dtype):
    assert compute_dtype == 'bfloat16'
    x = inputs['numeric_features'][:, 0, 0];n = len(x)
    turn, hp, limit, side = x[:, 0], x[:, 1], x[:, 2], x[:, 3]
    a = params['actor'].astype(j.float32)
    logits = j.stack([((turn*(k+1) + hp*3 + a*7 + side*5 + k*k) % 11) / 4 for k in range(9)], axis=-1)
    value = j.clip((hp - turn - a) / 20, -.9, .9)
    value = j.where((limit == 7) & (turn == 2), j.nan, value)
    return v5_model.Output(logits, value, j.zeros((n, 101), j.float32), j.ones(n, bool))


def step(state, actions, table, public, *, belief_tables):
    b = state.done.shape[0];active = ~state.done
    turn = state.turn + active.astype(j.int32)
    hit = j.where(active, actions[:, 0] % 3 + 1, 0)
    foe = j.maximum(state.hp[:, 1, 0] - hit, 0)
    hp = state.hp.at[:, 1, 0].set(foe).at[:, 0, 0].add(-j.where(active, actions[:, 1] % 2, 0))
    knocked = active & (foe <= 0);tied = active & ~knocked & (turn >= state.level[:, 0, 0])
    ended = knocked | tied
    after = state._replace(turn=turn, hp=hp, done=state.done | ended,
        winner=j.where(knocked, 0, j.where(tied, 2, state.winner)), forced_switch=j.zeros_like(state.forced_switch))
    return h.PublicStepResult(transition.StepResult(after, j.zeros(b, bool), j.zeros(b, j.int32)), public)


def action_mask(state, table):
    return j.ones((state.done.shape[0], 2, 9), bool)


def seams():
    from contextlib import ExitStack
    stack = ExitStack()
    for target, name, value in ((e, 'encode', encode), (v5_model, 'forward', forward),
            (transition, 'step_with_public', step), (transition, 'action_mask', action_mask)):
        stack.enter_context(patch.object(target, name, value))
    return stack


def block(games, streams):
    """One construction block: fresh carry for the given fake game indices."""
    _, _, labels, config, _ = fixture()
    b = len(games);shape = (b, 2, 6)
    s = empty_state(b, seed=17 + games[0])
    hp = np.full(shape, 100, np.int32);hp[:, 1, 0] = [HP[g] for g in games];hp[:, 0, 0] = [40 + g for g in games]
    level = np.full(shape, 80, np.int32);level[:, 0, 0] = [LIMIT[g] for g in games]
    forced = np.zeros((b, 2), bool);forced[[i for i, g in enumerate(games) if g % 4 == 1], 1] = True
    s = s._replace(species=j.broadcast_to(j.array([1, 2], j.int32)[None, :, None], shape), level=j.asarray(level),
        gender=j.ones(shape, j.int32), hp=j.asarray(hp), max_hp=j.full(shape, 100, j.int32),
        moves=j.ones_like(s.moves), pp=j.full_like(s.pp, 10), done=j.zeros(b, bool), forced_switch=j.asarray(forced))
    public = h.initialize(s).trackers
    known = np.ones(shape, bool);known[:, :, 3:] = np.asarray([g % 2 == 0 for g in games])[:, None, None]
    truth = d.ConstructorTruth(j.zeros((b, 2, 6, 64), j.float32).at[..., 0].set(1),
        j.broadcast_to(j.array([0, 1], j.int32)[None, :, None], shape), j.asarray(known))
    return d.initialize((s, public), np.asarray(streams, np.int32), truth=truth), labels, config


def bank(bindings, streams, ordinals, opponents):
    """Original planner adapter: learner seat ordinal%2, opponent archive per game."""
    rows = [dict(stream_id=int(s), learner_side=o % 2, opponent=dict(archive_sha256=ARCHIVES[a]))
        for s, o, a in zip(streams, ordinals, opponents)]
    required = tuple(dict.fromkeys([ARCHIVES[0], *(ARCHIVES[a] for a in opponents)]))
    return d.actor_bank_from_assignments(bindings, required, current_archive=ARCHIVES[0], assignments=rows)


def params_for(archives):
    return tuple({'actor': j.array(ARCHIVES.index(a), j.int32)} for a in archives)


def declaration(opponents=None, split=(4, 9)):
    """Two original blocks with distinct sampling seeds; returns blocks and batches."""
    _, _, labels, config, bindings = fixture()
    bindings = d.Bindings(bindings.category_sha256, bindings.label_vocabulary, bindings.source_sha256,
        (ARCHIVES[0],)*2, (True, True), bindings.declaration_sha256)
    opponents = opponents or (0,)*split[1]
    out = [];seeds = (7, 2**31 + 9)
    for k, (lo, hi) in enumerate(((0, split[0]), split)):
        games = tuple(range(lo, hi));streams = tuple(1000 + 17*g for g in games)
        carry, labels, config = block(games, streams)
        out.append((carry, bank(bindings, streams, games, [opponents[g] for g in games]), seeds[k]))
    return out, labels, config, bindings


def run_original(blocks, labels, config, bindings):
    """Production fixed-cohort collector + training_batch per block, as the backend."""
    from pokezero.v5_loss import v5_batch_to_torch
    torch_rows = [];frames = []
    for carry, actor_bank, seed in blocks:
        collect = d.make_collector(NS(struggle=j.array(2, j.int32)), NS(belief=None), config, labels, bindings,
            sampling_seed=seed, chunk_steps=CHUNK, max_requests=MAX_REQUESTS, actor_bank=actor_bank)
        params = params_for(actor_bank.actor_archives);rolls = []
        for _ in range(math.ceil(MAX_REQUESTS / CHUNK)):
            device = collect(carry, params);carry = device.carry
            host = d.boundary(device._replace(carry=carry._replace(state=None, public=None, truth=None), roots=None))
            rolls.append(host)
            if np.all(host.carry.status != d.RUNNING):break
        streams = actor_bank.streams
        batch = d.training_batch(tuple(rolls), bindings, battle_ids=tuple(f'g{s}' for s in streams), seeds=tuple(range(len(streams))),
            loss_config=V5LossConfig(), actor_bank=actor_bank)
        tensors = v5_batch_to_torch(batch, config, training=True)
        keys = np.asarray([(a['stream'], a['frame'], a['owner']) for a in batch.v5_columns['addresses']])
        torch_rows.append((keys, tensors));frames.append((rolls, actor_bank))
    return torch_rows, frames


def run_v2(blocks, labels, config, *, lanes, rows_per_game=8, **options):
    roster = v2.roster_from_blocks(blocks)
    collector = v2.RefillCollector(NS(struggle=j.array(2, j.int32)), NS(belief=None), config, labels, roster,
        lanes=lanes, max_requests=MAX_REQUESTS, chunk_steps=CHUNK, rows_per_game=rows_per_game, **options)
    result = collector.run(params_for(roster.actor_archives))
    store = v2.finalize(result, loss_config=V5LossConfig())
    return roster, collector, result, store


def sorted_rows(keys, tensors):
    order = np.lexsort((keys[:, 2], keys[:, 1], keys[:, 0]))
    return keys[order], {name: (value[order] if hasattr(value, 'shape') and value.ndim and len(value) == len(keys)
        else tuple(np.asarray(value, dtype=object)[order]) if name == 'terminal_status' else value) for name, value in tensors.items()}


def live_frames(chunks, *, v2_frames):
    """(stream, frame) -> per-field values for every live lane-step."""
    out = {}
    for frames in chunks:
        req = np.asarray(frames.requested).any(axis=-1)
        for t, lane in zip(*np.nonzero(req)):
            key = (int(frames.streams[t, lane]), int(frames.frames[t, lane]))
            if key in out:raise AssertionError('duplicate live lane-step')
            out[key] = {name: np.asarray(getattr(frames, name)[t, lane]) for name in d.Frames._fields
                if getattr(frames, name) is not None and name not in ('actor_indices',)}
    return out


class RefillCollectorTests(unittest.TestCase):
    def compare_rows(self, original, store):
        import torch
        keys = np.concatenate([k for k, _ in original]);ours = store.to_torch()
        merged = {}
        for name in original[0][1]:
            values = [t[name] for _, t in original]
            if name == 'terminal_status':merged[name] = sum((tuple(v) for v in values), ())
            elif isinstance(values[0], str):merged[name] = values[0]
            else:merged[name] = torch.cat(values)
        want_keys, want = sorted_rows(keys, merged)
        got_keys, got = sorted_rows(store.keys(), {name: ours[name] for name in merged})
        np.testing.assert_array_equal(got_keys, want_keys)
        self.assertEqual(set(merged) - set(ours), set())
        for name, value in want.items():
            with self.subTest(field=name):
                if isinstance(value, str) or name == 'terminal_status':self.assertEqual(got[name], value)
                else:
                    self.assertEqual(got[name].dtype, value.dtype);self.assertEqual(tuple(got[name].shape), tuple(value.shape))
                    self.assertTrue(torch.equal(got[name], value), name)
        return want_keys, want

    def test_no_refill_row_store_equals_original_training_batch(self):
        blocks, labels, config, bindings = declaration()
        with seams():
            original, _ = run_original(blocks, labels, config, bindings)
            roster, collector, result, store = run_v2(blocks, labels, config, lanes=roster_games(blocks))
        self.assertEqual(len(collector.levels), 1);self.assertEqual(result.stats['compactions'], [])
        keys, want = self.compare_rows(original, store)
        statuses = set(want['terminal_status'])
        self.assertTrue({'COMPLETE', 'CAPPED', 'REFUSED'} <= statuses, statuses)
        self.assertTrue(bool(want['ppo_advantage_mask'].any()) and not bool(want['ppo_advantage_mask'].all()))
        self.assertEqual(set(map(int, keys[:, 2])), {0, 1})

    def test_refill_with_shrink_reproduces_every_trajectory_and_row(self):
        blocks, labels, config, bindings = declaration()
        with seams():
            original, original_frames = run_original(blocks, labels, config, bindings)
            _, _, fixed, fixed_store = run_v2(blocks, labels, config, lanes=roster_games(blocks), emit_frames=True)
            roster, collector, refill, store = run_v2(blocks, labels, config, lanes=3, emit_frames=True, shrink_factor=2, min_lanes=1)
        self.assertEqual([x.lanes for x in collector.levels], [3, 2, 1])
        self.assertTrue(refill.stats['compactions'], 'tail shrink exercised')
        self.assertGreater(refill.stats['occupancy'], fixed.stats['occupancy'])
        self.compare_rows(original, store)
        # Every live lane-step of every game equals the fixed-cohort Frames.
        old = {}
        for rolls, actor_bank in original_frames:
            old.update(live_frames([r.frames for r in rolls], v2_frames=False))
        for run in (fixed, refill):
            new = live_frames([f for f, _ in run.frames], v2_frames=True)
            self.assertEqual(set(new), set(old))
            for key, fields in old.items():
                for name, value in fields.items():
                    np.testing.assert_array_equal(new[key][name], value, err_msg=f'{key} {name}')
        # Same per-game terminal records as the original carries.
        records = v2.game_records(refill);row = {int(s): i for i, s in enumerate(records['streams'])}
        for rolls, actor_bank in original_frames:
            c = rolls[-1].carry
            for i, stream in enumerate(actor_bank.streams):
                g = row[stream]
                self.assertEqual(records['status'][g], c.status[i]);self.assertEqual(records['terminal_frame'][g], c.terminal_frame[i])
                self.assertEqual(records['winner'][g], c.winner[i])
                np.testing.assert_array_equal(records['terminal_hp'][g], c.terminal_hp[i])

    def test_static_actor_pairing_groups_refill_from_own_queues(self):
        # Self-play, learner-vs-historical 1 and 2 in both seats.
        opponents = (0, 1, 2, 0, 1, 1, 0, 2, 0)
        blocks, labels, config, bindings = declaration(opponents)
        with seams():
            original, _ = run_original(blocks, labels, config, bindings)
            roster, collector, result, store = run_v2(blocks, labels, config, lanes=5, shrink_factor=2, min_lanes=1)
        self.assertEqual(roster.actor_archives, ARCHIVES)
        self.assertEqual(sorted(roster.groups), sorted({(0, 0), (0, 1), (1, 0), (0, 2), (2, 0)}))
        self.assertEqual(sum(collector.levels[0].layout), 5);self.assertTrue(all(x >= 1 for x in collector.levels[0].layout))
        self.compare_rows(original, store)

    def test_canonical_layout_rebind_reuses_compiled_kernels_for_a_new_roster(self):
        # Two iterations with the same history size but different planner draws
        # share one layout; the rebound collector must equal the original pipeline.
        first, labels, config, bindings = declaration((0, 1, 2, 0, 1, 1, 0, 2, 0))
        second, _, _, _ = declaration((2, 0, 0, 1, 0, 2, 2, 0, 1))
        archives, groups = v2.canonical_groups(ARCHIVES[0], ARCHIVES[1:])
        self.assertEqual(groups, ((0, 0), (0, 1), (1, 0), (0, 2), (2, 0)))
        counts = v2.expected_counts(groups, 9, 0.5)
        with seams():
            original, _ = run_original(second, labels, config, bindings)
            build = lambda blocks: v2.roster_from_blocks(blocks, archives=archives, groups=groups)
            roster_a, roster_b = build(first), build(second)
            collector = v2.RefillCollector(NS(struggle=j.array(2, j.int32)), NS(belief=None), config, labels, roster_a,
                lanes=5, max_requests=MAX_REQUESTS, chunk_steps=CHUNK, rows_per_game=8, layout_counts=counts,
                shrink_factor=2, min_lanes=1)
            layout = collector.levels[0].layout
            collector.compile(params_for(archives))
            executables = dict(collector._exe)
            collector.run(params_for(archives))
            collector.rebind(roster_b)
            store = v2.finalize(collector.run(params_for(archives)), loss_config=V5LossConfig())
        self.assertEqual(collector._exe, executables)
        fresh = v2.RefillCollector(NS(struggle=j.array(2, j.int32)), NS(belief=None), config, labels, roster_b,
            lanes=5, max_requests=MAX_REQUESTS, chunk_steps=CHUNK, rows_per_game=8, layout_counts=counts)
        self.assertEqual(fresh.levels[0].layout, layout)
        self.compare_rows(original, store)
        with self.assertRaises(ValueError):
            collector.rebind(v2.roster_from_blocks(second))

    def test_row_buffer_overflow_segments_and_numpy_export_keep_every_row(self):
        blocks, labels, config, bindings = declaration()
        with seams():
            original, _ = run_original(blocks, labels, config, bindings)
            _, _, result, store = run_v2(blocks, labels, config, lanes=3, rows_per_game=1, shrink_factor=2, min_lanes=1)
        self.assertGreater(result.stats['row_segments'], 1)
        self.compare_rows(original, store)
        host = store.to_numpy();torch_rows = store.to_torch(exact_dtypes=False, status_strings=False)
        self.assertEqual(len(host['numeric_features']), store.n);self.assertEqual(host['categorical_ids'].dtype, np.int32)
        for name in v2.DIRECT_FIELDS + v2.TERMINAL_FIELDS:
            np.testing.assert_array_equal(host[name], torch_rows[name].numpy(), err_msg=name)

    def test_layouts_are_proportional_static_and_shrink_per_group(self):
        self.assertEqual(v2.lane_layout([10000, 10000], 2048), (1024, 1024))
        self.assertEqual(v2.lane_layout([3, 1], 8), (7, 1))
        self.assertEqual(sum(v2.lane_layout([9000, 600, 400, 3], 2048)), 2048)
        layout = v2.lane_layout([9000, 600, 400, 3], 2048);self.assertEqual(layout[-1], 1)
        self.assertEqual([sum(x) for x in v2.shrink_levels((2048,), factor=4, min_lanes=128)], [2048, 512, 128])
        self.assertEqual([sum(x) for x in v2.shrink_levels((8192,), factor=4, min_lanes=128)], [8192, 2048, 512, 128])
        level = v2.make_level((2, 3), ((0, 0), (0, 1)))
        self.assertEqual(level.group_of_lane.tolist(), [0, 0, 1, 1, 1]);self.assertEqual(level.group_first.tolist(), [0, 0, 2, 2, 2])
        self.assertEqual([a for a, _ in level.forward], [0, 1]);self.assertEqual(level.forward[1][1].tolist(), [5, 7, 9])
        with self.assertRaises(ValueError):v2.lane_layout([5, 5, 5], 2)

    def test_per_lane_seed_keys_equal_batch_seed_keys(self):
        streams = j.array([2120000000, 5, 2**31 - 1], j.int32);frames = j.array([0, 7, 1023], j.int32)
        for seed in (0, 7, 2120300009, 2**31, 2**32 - 1):
            want = sampling.sample_keys(streams, frames, sampling_seed=seed)
            got = jax.jit(v2.sample_keys)(streams, frames, j.full(3, seed, j.uint32))
            np.testing.assert_array_equal(np.asarray(got), np.asarray(want))

    def test_vectorized_gae_equals_original_torch_gae_bitwise(self):
        import torch
        from pokezero.v5_loss import gae_targets
        rng = np.random.default_rng(3);lengths = [1, 2, 5, 17, 64, 3]
        values = [rng.uniform(-1, 1, n).astype(np.float32) for n in lengths]
        outcome = np.asarray([1, -1, 0, 1, -1, 1], np.float32)
        for discount, lam in ((1., .95), (.97, .9), (1., 1.)):
            trajectory = np.concatenate([np.full(n, t) for t, n in enumerate(lengths)])
            position = np.concatenate([np.arange(n) for n in lengths])
            perm = rng.permutation(len(trajectory))
            got_a, got_t = v2.trajectory_gae(np.concatenate(values)[perm], trajectory[perm], position[perm], outcome,
                discount=discount, policy_lambda=lam, value_lambda=.95)
            inverse = np.argsort(perm);got_a, got_t = got_a[inverse], got_t[inverse]
            want_a, want_t = [], []
            for v, r in zip(values, outcome):
                v = torch.tensor(v);reward = torch.zeros_like(v);reward[-1] = float(r)
                terminated = torch.zeros(len(v), dtype=torch.bool);terminated[-1] = True
                a, t = gae_targets(reward, v, torch.cat((v[1:], torch.zeros(1))), terminated, ~terminated,
                    discount=discount, policy_lambda=lam, value_lambda=.95)
                want_a.append(a.numpy());want_t.append(t.numpy())
            np.testing.assert_array_equal(got_a, np.concatenate(want_a));np.testing.assert_array_equal(got_t, np.concatenate(want_t))


def roster_games(blocks):
    return sum(len(b.streams) for _, b, _ in blocks)


if __name__ == '__main__':unittest.main()
