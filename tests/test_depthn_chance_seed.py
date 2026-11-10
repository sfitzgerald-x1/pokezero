"""Chance seeds: the engine's chance draws apart from the constructed worlds (constructor_worlds' module docstring, Chance).

- flag off: the live search's responses and the re-pricer's rows equal, bit for bit, the golden captured before the change
  (tests/depthn_chance_golden.py at c109669b), with the defaults and with the flags spelled at their off values;
- the real engine: a chance seed re-keys only state.keys (and a due residual capsule), and two seeds give different outcomes
  of the same state on a move with accuracy < 100 and on a damage / critical-hit roll;
- the fake-engine live search with a chance seed, chance draws as extra world lanes, a world prefix, price_chance;
- the re-pricer's multi-seed mode (--chance-seeds), the server's and the bench's flags, the W x S bench driver;
- scripts/depthn_chance_noise.py on synthetic chance lines."""
from contextlib import ExitStack, redirect_stderr
import io
import json
import math
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch

import jax
import jax.numpy as jnp
import numpy as np

from gpu_engine import constructor_worlds as cw, depthn_backups as backups, depthn_live_search as live, search_hedge as hedge
from gpu_engine import transition as t
from gpu_engine.d3_search_latency import compose, reduce_block
from gpu_engine.device_public_world_state import World
from gpu_engine.state import empty_state
from pokezero import v5_gumbel as original
from scripts import depthn_chance_bench as B, depthn_chance_noise as noise, depthn_live_bench as bench, depthn_live_server as server
from scripts import depthn_rr_reprice as R
from tests import depthn_chance_golden as golden
from tests import test_depthn_rr_reprice as RT
from tests import test_gpu_engine_depthn_live_search as L

PINNED = {'JAX_DEFAULT_MATMUL_PRECISION': 'highest'}


def leaves_equal(a, b):
    left, right = jax.tree.leaves(a), jax.tree.leaves(b)
    return len(left) == len(right) and all(np.array_equal(np.asarray(x), np.asarray(y)) for x, y in zip(left, right))


# ---- flag off: the golden ------------------------------------------------------------------------------------------------------
class FlagOffGoldenTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.golden = json.loads(golden.GOLDEN.read_text())

    def test_live_responses_are_the_goldens_bit_for_bit(self):
        self.assertEqual(golden.live_responses(), self.golden['live'])
        self.assertEqual(golden.live_responses(chance_seed=0, chance_draws=1, worlds=None), self.golden['live'])   # the off spellings

    def test_repriced_rows_are_the_goldens_bit_for_bit(self):
        self.assertEqual(golden.reprice_rows(), self.golden['rows'])


# ---- the real engine -------------------------------------------------------------------------------------------------------------
def engine_fixture(accuracy=100):
    """A one-battle real BattleState (tests/test_gpu_engine_chance_inputs' fixture) whose every move is id 1 (power 50, accuracy
    `accuracy`), and its move table."""
    state = empty_state(1, seed=29)
    state = state._replace(stats=jnp.full_like(state.stats, 200).at[..., 0].set(100), level=jnp.full_like(state.level, 100),
        hp=jnp.full_like(state.hp, 100), max_hp=jnp.full_like(state.hp, 100), moves=jnp.ones_like(state.moves), pp=jnp.full_like(state.pp, 10),
        types=state.types.at[..., 0].set(1), done=jnp.zeros_like(state.done))
    table = t.MoveTable(jnp.array([0, 50, 0, 50]), jnp.array([0, 1, 1, 0]), jnp.array([False, True, False, True]), jnp.array([100, accuracy, 100, 100]),
        jnp.array([False] * 4), jnp.zeros(4, jnp.int32), jnp.ones((18, 18), jnp.int32), jnp.ones((18, 18), jnp.int32), jnp.array(3))
    return state, table


class EngineChanceTests(unittest.TestCase):
    def test_seed_zero_is_the_identity(self):
        state, _ = engine_fixture();w = World(3, state, dict(c=1), (), 77, False)
        self.assertIs(cw.rechance_state(state, 0), state);self.assertIs(cw.rechance_world(w, 0), w)
        none = World(4, None, None, ('refused',), 78, False);self.assertIs(cw.rechance_world(none, 5), none)
        batch = NS(roots=(), requests=(), worlds=None);self.assertIs(cw.rechance(batch, 0), batch);self.assertIs(cw.chance_lanes(batch, (0,)), batch)
        for bad in (-1, 2 ** 32, 1.5, True, '1'):
            with self.subTest(bad=bad), self.assertRaises(ValueError):cw.rechance_state(state, bad)

    def test_a_seed_changes_only_the_keys(self):
        state, _ = engine_fixture();out = {}
        for k in (1, 2, 3, 2 ** 32 - 1):
            new = cw.rechance_state(state, k);out[k] = np.asarray(new.keys)
            self.assertEqual((new.keys.shape, new.keys.dtype), (state.keys.shape, state.keys.dtype))
            want = jax.random.fold_in(jax.random.fold_in(state.keys[0], jnp.uint32(cw.CHANCE_DOMAIN)), jnp.uint32(k))
            np.testing.assert_array_equal(np.asarray(new.keys)[0], np.asarray(want))
            for field in type(state)._fields:
                if field != 'keys':self.assertIs(getattr(new, field), getattr(state, field), field)   # the very same arrays
            np.testing.assert_array_equal(np.asarray(cw.rechance_state(state, k).keys), out[k])   # deterministic
        self.assertEqual(len({x.tobytes() for x in out.values()} | {np.asarray(state.keys).tobytes()}), 5)   # all distinct
        w = World(3, state, dict(c=1), (), 77, False);new = cw.rechance_world(w, 2)
        self.assertEqual((new.index, new.public, new.refusals, new.kernel_seed, new.reply_private_history_available), (3, dict(c=1), (), 77, False))
        np.testing.assert_array_equal(np.asarray(new.state.keys), out[2])
        with self.assertRaisesRegex(ValueError, 'BattleState'):cw.rechance_state(NS(_replace=None, seed=1), 1)

    def test_a_due_residual_capsule_is_redrawn_from_the_new_keys(self):
        state, _ = engine_fixture();old = t.sample_chance_inputs(state)
        due = state._replace(pending_residual=jnp.ones(1, bool), residual_draws=old.draws, phazing_draws=old.phazing, secondary_draws=old.secondary,
            sleep_talk_draws=old.sleep_talk)   # phase_d_tail.restore's capsule, drawn from the constructed keys
        new = cw.rechance_state(due, 7);redrawn = t.sample_chance_inputs(new)
        for field, name in cw.CAPSULE:
            np.testing.assert_array_equal(np.asarray(getattr(new, field)), np.asarray(getattr(redrawn, name)))
            self.assertEqual(getattr(new, field).dtype, getattr(due, field).dtype)
        self.assertFalse(np.array_equal(np.asarray(new.residual_draws), np.asarray(due.residual_draws)))
        untouched = {f for f in type(state)._fields} - {'keys'} - {f for f, _ in cw.CAPSULE}
        self.assertTrue(all(getattr(new, f) is getattr(due, f) for f in untouched))
        plain = cw.rechance_state(state, 7)   # nothing due: the capsule is the state's own
        self.assertTrue(all(getattr(plain, f) is getattr(state, f) for f, _ in cw.CAPSULE))

    def outcomes(self, accuracy, seeds=range(0, 24)):
        state, table = engine_fixture(accuracy);actions = jnp.zeros((1, 2), jnp.int32);step = jax.jit(t.step)
        out = {}
        for k in seeds:
            after = step(cw.rechance_state(state, k), actions, table).state
            out[k] = tuple(int(x) for x in np.asarray(after.hp)[0, :, 0])
        return out

    def test_two_seeds_change_the_outcome_where_chance_matters_on_the_same_world(self):
        """The same constructed world (every leaf but the keys identical) steps to different HP under different chance seeds:
        a 50%-accurate move hits under some seeds and misses under others; at 100% accuracy the damage roll and the
        critical hit still move it."""
        state, _ = engine_fixture(50)
        a, b = cw.rechance_state(state, 1), cw.rechance_state(state, 2)
        self.assertTrue(all(leaves_equal(getattr(a, f), getattr(b, f)) for f in type(state)._fields if f != 'keys'))
        self.assertFalse(leaves_equal(a.keys, b.keys))
        inaccurate = self.outcomes(50);hits = {k: v for k, v in inaccurate.items() if v != (100, 100)}
        missed = [k for k, v in inaccurate.items() if v[1] == 100]   # p2's slot 0 untouched: p1's move missed
        self.assertTrue(missed and len(missed) < len(inaccurate), inaccurate)
        self.assertGreater(len(set(inaccurate.values())), 1)
        rolls = self.outcomes(100)
        self.assertGreater(len(set(rolls.values())), 1, rolls)   # damage roll / crit
        self.assertTrue(all(v[1] < 100 for v in rolls.values()))   # never a miss at 100% accuracy
        self.assertEqual(self.outcomes(50, (3,)), {3: inaccurate[3]})   # reproducible
        self.assertTrue(hits)


# ---- batches: lanes and prefixes --------------------------------------------------------------------------------------------
def fake_rechance(state, seed):
    """The fake engine's chance key is its `seed` (chance = seed % 17 at every node): re-keyed per chance seed."""
    if cw.chance_seed(seed) == 0:return state
    return state._replace(seed=(state.seed * jnp.uint32(2654435761) + jnp.uint32(seed)).astype(jnp.uint32))


class BatchTests(unittest.TestCase):
    def batch(self):
        engine, _ = engine_fixture()
        pairs = tuple((World(2 * r, engine, None, (), 10 + r, False), World(2 * r + 1, engine, None, (), 20 + r, False)) for r in range(3))
        roots = (dict(request_id='a', status='PRICED', rows=(0, 1)), dict(request_id='x', status='REFUSED', reason='no'), dict(request_id='b', status='PRICED', rows=(2,)))
        return cw.Batch(('q0', 'q1', 'q2'), NS(source='s', binding='b', pairs=pairs), roots, 4, 2)

    def test_chance_lanes_repeat_each_priced_roots_blocks_per_seed(self):
        from dataclasses import replace
        base = self.batch();base = replace(base, worlds=cw.dn.PrebuiltWorlds('s', None, base.worlds.pairs))
        out = cw.chance_lanes(base, (0, 5))
        self.assertEqual(out.requests, ('q0', 'q1', 'q0', 'q1', 'q2', 'q2'))
        self.assertEqual([r.get('rows') for r in out.roots], [(0, 1, 2, 3), None, (4, 5)])
        self.assertEqual((out.count, out.blocks, out.roots[1]), (8, 4, base.roots[1]))
        self.assertIs(out.worlds.pairs[0][0], base.worlds.pairs[0][0])   # seed 0: the worlds themselves
        lane = out.worlds.pairs[2][1];self.assertEqual((lane.index, lane.kernel_seed), (1, 20))
        np.testing.assert_array_equal(np.asarray(lane.state.keys), np.asarray(cw.rechance_state(base.worlds.pairs[0][1].state, 5).keys))
        single = cw.rechance(base, 5);self.assertEqual([r.get('rows') for r in single.roots], [(0, 1), None, (2,)])
        for bad in ((), (1, 1), (-1,)):
            with self.subTest(bad=bad), self.assertRaises(ValueError):cw.chance_lanes(base, bad)

    def test_world_prefix_keeps_the_first_worlds(self):
        from gpu_engine.phase_d_live_worlds import WorldSet
        engine, _ = engine_fixture();worlds = tuple(World(i, engine, None, (), 100 + i, False) for i in range(8))
        made = WorldSet('root', worlds, tuple(range(8)), 8, tuple(dict(world=i) for i in range(8)))
        item = cw.Prepared('r', 'detached', made, (True,) * 9, .5, 'binding')
        out = cw.world_prefix(item, 4)
        self.assertEqual((out.made.worlds, out.made.seeds, out.made.requested_worlds, out.made.mapper_diagnostics), (worlds[:4], (0, 1, 2, 3), 4,
            tuple(dict(world=i) for i in range(4))))
        self.assertTrue(out.made.fully_constructed);self.assertIs(cw.world_prefix(item, 8), item)
        refusal = cw.RootRefusal('r', 'K', 'd', 0., 'b');self.assertIs(cw.world_prefix(refusal, 4), refusal)
        for bad in (3, 0, 10):
            with self.subTest(bad=bad), self.assertRaises(ValueError):cw.world_prefix(item, bad)


# ---- the fake-engine live search --------------------------------------------------------------------------------------------
class LiveSearchChanceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        L.LiveSearchTests.setUpClass();cls.case = L.LiveSearchTests('test_constructor_refusals')

    def setUp(self):
        self.case.setUp();self.addCleanup(self.case.doCleanups)
        rekey = patch.object(cw, 'rechance_state', side_effect=fake_rechance);rekey.start();self.addCleanup(rekey.stop)

    def pricers(self, worlds, buckets=(1,)):
        from tests.test_gpu_engine_depthn_guard_inputs import fake_labels
        return live.make_pricers(self.case.executor, self.case.evaluator, L.PARAMS, worlds=worlds, buckets=buckets, block_lanes=81, blocks_per_call=4,
            labels=fake_labels)

    def run_one(self, search, name='c', i=0, W=8):
        rows, codex = self.case.run_batch(search, [L.request(name, 0)], {(name, 0): self.case.outcome(i, W)})
        return rows[0], codex

    def test_a_chance_seed_changes_the_prices_not_the_worlds(self):
        built = []
        original_build = cw.build
        with patch.object(cw, 'build', side_effect=lambda *a, **k: built.append(original_build(*a, **k)) or built[-1]):
            base, _ = self.run_one(self.case.search())
            seeded, _ = self.run_one(self.case.search(chance_seed=5))
            again, _ = self.run_one(self.case.search(chance_seed=5))
        self.assertEqual((base['status'], seeded['status']), ('COMPLETE', 'COMPLETE'))
        self.assertNotEqual(seeded['q_values'], base['q_values'])   # the fake engine's values carry chance / 1024 at both plies
        self.assertEqual(seeded['q_values'], again['q_values'])   # reproducible
        self.assertEqual((seeded['constructed_worlds'], seeded['requested_worlds'], seeded['pricer']['chance_seed']), (8, 8, 5))
        self.assertNotIn('chance_seed', base['pricer'])
        # The built worlds are the constructed ones, the same under seed 0 and seed 5 (the seed applies only after build).
        self.assertEqual(len(built), 3)
        for x, y in zip(built[0].worlds.pairs, built[1].worlds.pairs):
            for a, b in zip(x, y):self.assertTrue(leaves_equal((a.state, a.public), (b.state, b.public)))

    def test_chance_draws_price_extra_world_lanes_and_average_them(self):
        singles = [self.run_one(self.case.search(chance_seed=k))[0] for k in (3, 4)]
        s = self.case.search(pricers=self.pricers(16), chance_seed=3, chance_draws=2)
        self.assertEqual((s.blocks, s.chance_seeds, s.pricers[1].capacity), (8, (3, 4), 8))
        row, _ = self.run_one(s)
        self.assertEqual((row['status'], row['block_causes'], row['pricer']['chance_draws'], row['pricer']['chance_seeds'], row['pricer']['capacities']),
            ('COMPLETE', [None] * 8, 2, [3, 4], [8]))
        for a in range(9):
            if row['q_values'][a] is None:continue
            self.assertAlmostEqual(row['q_values'][a], (singles[0]['q_values'][a] + singles[1]['q_values'][a]) / 2, places=12)
        self.assertEqual(row['effect_guard']['turn2_cells']['p1_captures'], 1)
        with self.assertRaisesRegex(ValueError, 'capacity'):self.case.search(chance_draws=2)   # W x S lanes need the bigger pricers
        with L.Codex({('w', 0): self.case.outcome(2)}):report = self.case.search(pricers=self.pricers(16), chance_draws=2).warmup([L.request('w', 0)])
        self.assertEqual((report['buckets'][1]['complete'], report['labels']['root_rows'], report['labels']['p1_rows']), (1, 16, {1: 2 * 8 * 81}))

    def test_a_world_prefix_prices_the_first_worlds(self):
        s = self.case.search(pricers=self.pricers(4), worlds=4)
        row, _ = self.run_one(s)
        self.assertEqual((row['status'], row['rung'], row['requested_worlds'], row['constructed_worlds'], row['block_causes']), ('COMPLETE', 'DN2W8', 4, 4, [None] * 2))
        self.assertEqual((row['pricer']['W'], row['pricer']['K'], row['pricer']['constructor_worlds'], row['pricer']['capacities']), (4, 2, 8, [2]))
        part, _ = self.run_one(self.case.search(pricers=self.pricers(4), worlds=4), 'p', 1)
        self.assertEqual(part['status'], 'COMPLETE')
        refused = self.case.run_batch(self.case.search(pricers=self.pricers(4), worlds=4), [L.request('q', 0)], {('q', 0): self.case.outcome(1, refused=(6,))})[0][0]
        self.assertEqual((refused['status'], refused['constructed_worlds']), ('COMPLETE', 4))   # world 6 is outside the prefix
        with L.Codex({('w', 0): self.case.outcome(2)}):report = self.case.search(pricers=self.pricers(4), worlds=4).warmup([L.request('w', 0)])
        self.assertEqual(report['labels']['root_rows'], 4)
        for bad in (dict(worlds=3), dict(worlds=10), dict(worlds=0), dict(chance_draws=0), dict(chance_seed=-1), dict(chance_seed=2 ** 32 - 1, chance_draws=2)):
            with self.subTest(bad=bad), self.assertRaises(ValueError):self.case.search(**bad)

    def test_price_chance_reprices_the_built_batch_and_seed_0_is_the_live_decision(self):
        s = self.case.search();built = [];original_build = cw.build
        with patch.object(cw, 'build', side_effect=lambda *a, **k: built.append(original_build(*a, **k)) or built[-1]):row, _ = self.run_one(s)
        request = {k: L.request('c', 0)[k] for k in live.PROTOCOL_KEYS}
        zero = s.price_chance(built[0], [request], 0);result = zero['results'][0]
        self.assertEqual((zero['seed'], zero['bucket'], result['status']), (0, 1, 'COMPLETE'))
        for k in ('action_index', 'q_values', 'completed_q', 'policy_target', 'root_value', 'action_mean_q', 'action_policy_target', 'effect_guard'):
            self.assertEqual(json.loads(json.dumps(result[k])), json.loads(json.dumps(row[k])), k)
        self.assertEqual(len(result['blocks']), 4)
        seven = s.price_chance(built[0], [request], 7)['results'][0]
        self.assertNotEqual(seven['q_values'], result['q_values'])
        self.assertEqual(seven['q_values'], self.run_one(self.case.search(chance_seed=7), 'd')[0]['q_values'])   # the live path under seed 7


# ---- the re-pricer's multi-seed mode -----------------------------------------------------------------------------------------
def seeded_prices(w, seed):
    """prices(w) under chance seed `seed`: seed 0 the fixture's own; others move every priced cell by a seed-dependent amount."""
    base = RT.prices(w)
    if not seed:return base
    grid = [[None if v is None else v + seed * 1e-3 * ((a + b + w) % 3 - 1) for b, v in enumerate(row)] for a, row in enumerate(base.joint_values)]
    return original.WorldPrices(base.opponent_requested, base.opponent_legal, base.opponent_logits, grid)


ORIGINAL_PRICING = RT.pricing


def seeded_pricing(stack, still_clock=False):
    """The re-pricer fixture's fake pricing, where the chance seed of the price call moves the backed-up prices."""
    ORIGINAL_PRICING(stack, still_clock=still_clock)
    lanes = lambda made, seeds: made if tuple(seeds) == (0,) else NS(requests=made.requests, roots=made.roots, chance_seed=seeds[0])
    stack.enter_context(patch.object(cw, 'chance_lanes', side_effect=lanes))
    stack.enter_context(patch.object(cw, 'price', side_effect=lambda pricer, made, *a, **k: (NS(seed=getattr(made, 'chance_seed', 0)), None)))
    stack.enter_context(patch.object(backups, 'backup', side_effect=lambda frame, r, rule: (seeded_prices(2 * r, frame.seed), seeded_prices(2 * r + 1, frame.seed))))


class RepricerChanceTests(unittest.TestCase):
    def setUp(self):
        environment = patch.dict(os.environ, PINNED);environment.start();self.addCleanup(environment.stop)

    def replay(self, *extra):
        case = RT.MainTests('test_main_replays_records_and_joins');case.setUp();self.addCleanup(case.doCleanups)
        with tempfile.TemporaryDirectory() as tmp, patch.object(RT, 'pricing', seeded_pricing):
            code, rows, summary, header, *_ = case.replay(tmp, *extra)
            path = Path(tmp) / 'out' / 'chance-0.jsonl'
            lines = [json.loads(x) for x in path.read_text().splitlines()] if path.exists() else None
        return code, rows, summary, header, lines

    def test_the_multi_seed_mode_writes_per_seed_grids_and_leaves_the_rows_unchanged(self):
        code, rows, summary, header, lines = self.replay('--chance-seeds', '0,1,2')
        _, plain_rows, plain_summary, plain_header, none = self.replay()
        self.assertEqual(code, 0);self.assertIsNone(none);self.assertNotIn('chance', plain_header);self.assertNotIn('chance', plain_summary)
        self.assertEqual([golden.normalize(r) for r in rows], [golden.normalize(r) for r in plain_rows])   # the replay itself is unchanged
        self.assertEqual(header['chance'], dict(path='chance-0.jsonl', schema=R.CHANCE_SCHEMA, seeds=[0, 1, 2]))
        self.assertEqual([(x['ordinal'], x['decision_index']) for x in lines], [(1001, 0), (1001, 2), (1001, 3)])   # decision 4: refused before build
        first = lines[0];want = RT.expected_decision()
        self.assertEqual((first['schema'], first['chance_seeds'], first['worlds'], first['prior_invariant'], first['greedy_action_index'], first['greedy_source']),
            (R.CHANCE_SCHEMA, [0, 1, 2], 16, True, 0, 'live'))
        self.assertEqual(first['root'], dict(root_logits=list(RT.LOGITS), root_legal=list(RT.LEGAL), own_moves=['m0', 'm1', 'm2', 'm3']))
        self.assertEqual(first['priors'][3], dict(opponent_requested=True, kept=list(RT.KEPT), root_peer_logits=list(RT.prices(3).opponent_logits)))
        self.assertEqual([e['seed'] for e in first['seeds']], [0, 1, 2]);self.assertEqual({e['status'] for e in first['seeds']}, {'COMPLETE'})
        for e in first['seeds']:
            self.assertEqual(e['grids'], [[list(r) for r in seeded_prices(w, e['seed']).joint_values] for w in range(16)])
            decided = live.decide([(seeded_prices(2 * k, e['seed']), seeded_prices(2 * k + 1, e['seed'])) for k in range(8)], RT.LOGITS, RT.LEGAL)
            self.assertEqual((e['action_index'], json.loads(json.dumps(e['q_values']))), (decided['action_index'], json.loads(json.dumps(decided['q_values']))))
        self.assertEqual(first['seeds'][0]['q_values'], json.loads(json.dumps(want['q_values'])))
        self.assertNotEqual(first['seeds'][1]['grids'], first['seeds'][0]['grids'])
        self.assertEqual(first['control'], dict(equal=True, differ=[]))
        self.assertEqual(first['effect_guard'], R.guard_compact(rows[0]['effect_guard']))
        c = summary['chance']
        self.assertEqual((c['lines'], c['all_seeds_complete'], c['control']['passed'], c['control']['compared'], c['statuses']['1']), (3, 3, True, 3, dict(COMPLETE=3)))

    def test_the_analysis_reads_the_repricers_lines_and_reduces_to_the_live_decisions(self):
        """End to end: the re-pricer's chance lines through scripts/depthn_chance_noise.py; every seed's averaging action and q
        and its policy target, recomputed from the written grids, are the live code's own (consistency), at one draw."""
        lines = self.replay('--chance-seeds', '0,1,2')[4]
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'chance-0.jsonl';path.write_text(''.join(json.dumps(x) + '\n' for x in lines))
            s = noise.run([path], Path(tmp) / 'noise')
        self.assertEqual((s['seeds'], s['analysed'], s['excluded']), ([1, 2], 3, {}))
        self.assertEqual((s['consistency']['averaging'], s['consistency']['policy_target'], s['target_fallbacks']), (3, 3, 0))
        self.assertEqual(s['populations']['unguarded']['roots'], 0)   # the fixture's guard is the fail-closed error record: (b)

    def test_a_control_that_differs_is_reported(self):
        original_entry = R.seed_entry
        def skewed(search, made, body, seed):
            entry, priors = original_entry(search, made, body, seed)
            if seed == 0:entry['grids'][0][0][0] += 1e-9
            return entry, priors
        with patch.object(R, 'seed_entry', side_effect=skewed):_, _, summary, _, lines = self.replay('--chance-seeds', '0,1')
        self.assertEqual(lines[0]['control'], dict(equal=False, differ=['grids']));self.assertFalse(summary['chance']['control']['passed'])

    def test_a_seed_that_raises_is_recorded(self):
        original_price = live.DepthNLiveSearch.price_chance
        def broken(search, made, requests, seed):
            if seed == 2:raise RuntimeError('device lost')
            return original_price(search, made, requests, seed)
        with patch.object(live.DepthNLiveSearch, 'price_chance', broken):_, _, summary, _, lines = self.replay('--chance-seeds', '0,2')
        self.assertEqual([(e['seed'], e['status'], e.get('error')) for e in lines[0]['seeds']], [(0, 'COMPLETE', None), (2, 'ERROR', 'RuntimeError: device lost')])
        self.assertEqual((summary['chance']['all_seeds_complete'], summary['chance']['control']['passed']), (0, True))

    def test_options(self):
        run = ['--run-directory', 'r', '--out', 'o', '--cells', '1001']
        with patch.object(R.server.evaluators, 'check_arguments'):
            args = R.options(RT.BASE + RT.CHAMPION + run + ['--chance-seeds', '0,1,2,3,4,5,6,7,8'])
            self.assertEqual((args.chance_seeds, args.chance_seed, args.chance_draws, args.worlds), (tuple(range(9)), 0, 1, None))
            self.assertIsNone(R.options(RT.BASE + RT.CHAMPION + run).chance_seeds)
            self.assertEqual(R.options(RT.BASE + RT.CHAMPION + run + ['--chance-seed', '4']).chance_seed, 4)   # a single re-keyed replay
            for extra in (['--chance-seeds', '1,1'], ['--chance-seeds', '-1'], ['--chance-seeds', 'a'], ['--chance-seed', '1', '--chance-seeds', '0,1'],
                    ['--chance-draws', '2'], ['--worlds', '8'], ['--chance-seed', '-2']):
                with self.subTest(extra=extra), redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):R.options(RT.BASE + RT.CHAMPION + run + extra)

    def test_the_capture_wraps_build_and_restores_it(self):
        self.assertTrue(R.check_call_sites())
        build = cw.build;capture = R.Capture()
        with capture.installed():self.assertIsNot(cw.build, build)
        self.assertIs(cw.build, build)
        with self.assertRaises(KeyError):
            with capture.installed():raise KeyError('x')
        self.assertIs(cw.build, build)


# ---- the server and the bench --------------------------------------------------------------------------------------------------
BASE = ['--showdown-root', 's', '--node-binary', 'n', '--metadata', 'm', '--expected-metadata-sha256', 'a' * 64, '--vocabulary', 'v',
    '--expected-vocabulary-sha256', 'b' * 64, '--champion-archive', 'c.npz', '--champion-archive-sha256', 'c' * 64, '--champion-checkpoint-sha256', 'd' * 64]


class FlagTests(unittest.TestCase):
    def setUp(self):
        environment = patch.dict(os.environ, PINNED);environment.start();self.addCleanup(environment.stop)

    def test_server_flags_reach_the_search_and_the_header(self):
        import jax as jax_module
        from gpu_engine import device_public_teacher as teacher, engine_device_observation, phase_d_protocol_constructor as pc
        listen = ['--port', '9000', '--warmup-requests', 'w.jsonl']
        args = server.options(BASE + listen)
        self.assertEqual(server.chance_options(args), (0, 1, None));self.assertEqual(server.chance_header(args), {})
        args = server.options(BASE + listen + ['--chance-seed', '3', '--chance-draws', '2', '--worlds', '4'])
        self.assertEqual(server.chance_header(args), dict(chance_seed=3, chance_draws=2, worlds=4))
        for bad in (['--worlds', '5'], ['--worlds', '10'], ['--rung', 'DN2W16', '--worlds', '18'], ['--chance-draws', '0'], ['--chance-seed', '-1'],
                ['--chance-seed', str(2 ** 32 - 1), '--chance-draws', '2']):
            with self.subTest(bad=bad), redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):server.options(BASE + listen + bad)
        made, pricers = {}, {}
        stand_ins = ((server, 'matmul_precision', lambda: 'highest'), (jax_module, 'devices', lambda: [NS(platform='gpu')]),
            (pc, 'load_inputs', lambda a: ('catalog', 'source', 'metadata', 'config')), (server.evaluators, 'load_from_arguments', lambda a: NS(params='champion')),
            (pc, 'ProtocolConstructor', lambda **kw: NS(**kw)), (teacher, 'make_executor', lambda *a, **k: 'executor'),
            (t, 'tables_from_metadata', lambda m: 'tables'), (engine_device_observation, 'compile_tables', lambda m, catalog: 'observation'),
            (server.live, 'make_pricers', lambda *a, **k: pricers.update(k) or 'pricers'), (cw, 'binding', lambda *a: 'binding'),
            (server.live, 'DepthNLiveSearch', lambda **kw: made.update(kw) or 'search'))
        for argv, want, worlds in ((BASE + listen, {}, 8), (BASE + listen + ['--chance-seed', '3', '--chance-draws', '2', '--worlds', '4'],
                dict(chance_seed=3, chance_draws=2, worlds=4), 8)):
            made.clear();pricers.clear();args = server.options(argv)
            with ExitStack() as stack:
                for target, name, value in stand_ins:stack.enter_context(patch.object(target, name, side_effect=value))
                server.build(args)
            self.assertEqual({k: made[k] for k in ('chance_seed', 'chance_draws', 'worlds') if k in made}, want)
            self.assertEqual(pricers['worlds'], worlds)   # W x S lanes: 4 x 2 = 8, the live 8 x 1
        search = NS(tree_commit='t', rung='DN2W8', max_requests=4, min_price_seconds=1., session_cap=256, refuse_classes=(), pricer_identity={})
        with patch.object(server, 'matmul_precision', return_value='highest'):
            self.assertEqual(server.header(args, search)['chance_seed'], 3)
            self.assertNotIn('chance_seed', server.header(server.options(BASE + listen), search))

    def test_bench_options_take_the_chance_flags(self):
        roster = ['--roster', 'r.json', 'z', '--out', 'o']
        args = bench.options(BASE + roster + ['--rung', 'DN2W8', '--worlds', '4', '--chance-draws', '4'])
        self.assertEqual(server.chance_options(args), (0, 4, 4))
        with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):bench.options(BASE + roster + ['--worlds', '6', '--rung', 'DN2W16', '--chance-draws', '0'])
        with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):bench.options(BASE + roster + ['--worlds', '5'])

    def test_bench_replays_owner_captures_in_session_mode(self):
        from tests import test_depthn_live_bench as BT
        base = BASE + ['--out', 'o']
        args = bench.options(base + ['--captures', 'a.jsonl', 'b.jsonl', '--sessions', '1'])
        self.assertEqual((args.captures, args.roster, args.saved_root_admission), (['a.jsonl', 'b.jsonl'], None, False))
        for bad in ([], ['--captures', 'a.jsonl'], ['--captures', 'a.jsonl', '--roster', 'r', 'z', '--sessions', '1']):
            with self.subTest(bad=bad), redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):bench.options(base + bad)
        def capture(path):
            rows = [dict(BT.address_request(i), lines=[f'|turn|{i + 1}', f'|request|{i}']) for i in range(2)]
            if path == 'b.jsonl':rows[0]['cell'] = dict(ordinal=7)
            return rows
        with patch('pokezero.phase_d_owning_capture.read_capture', side_effect=capture):got = bench.capture_battles(['a.jsonl', 'b.jsonl', 'a.jsonl'])
        self.assertEqual(list(got), ['a.jsonl', 'b.jsonl'])
        self.assertEqual([(o, a, t) for o, a, _, _, t in got['a.jsonl']], [(0, ('b0', 'p1', 0, f'{0:064x}'), 1), (1, ('b1', 'p1', 1, f'{1:064x}'), 2)])
        self.assertEqual(got['b.jsonl'][0][1], (7, 'p1', 0, f'{0:064x}'))   # the saved-root address when the row carries its cell
        self.assertTrue(all(set(r) == set(bench.KEYS) for _, _, r, _, _ in got['a.jsonl']))
        with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):   # the re-pricer refuses the bench's roster flags
            R.options(RT.BASE + RT.CHAMPION + ['--run-directory', 'r', '--out', 'o', '--cells', '1', '--captures', 'a.jsonl'])


class ChanceBenchTests(unittest.TestCase):
    def test_configurations_rungs_and_commands(self):
        self.assertEqual(B.config_arg('16x1,8X2,4x4'), ((16, 1), (8, 2), (4, 4)))
        for bad in ('16', '3x1', '16x0', '32x1', '16x1,16x1'):
            with self.subTest(bad=bad), self.assertRaises(Exception):B.config_arg(bad)
        self.assertEqual([B.rung_for(w) for w in (4, 8, 16)], [('DN2W8', 4), ('DN2W8', None), ('DN2W16', None)])
        self.assertEqual(B.command(['--roster', 'r', 'z'], 4, 4, '/o', python='py'), ['py', str(B.BENCH), '--roster', 'r', 'z', '--rung', 'DN2W8', '--worlds', '4',
            '--chance-draws', '4', '--out', '/o/W4xS4'])
        self.assertEqual(B.command([], 16, 2, '/o', python='py')[2:], ['--rung', 'DN2W16', '--chance-draws', '2', '--out', '/o/W16xS2'])
        with patch('sys.stdout', new_callable=io.StringIO) as out:self.assertEqual(B.main(['--out', '/nowhere', '--dry-run', '--', '--roster', 'r', 'z']), 0)
        self.assertEqual(len(out.getvalue().splitlines()), 5)
        with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):B.main(['--out', '/nowhere', '--', '--rung', 'DN2W8'])

    def test_combine_tabulates_finished_configurations(self):
        stage = lambda x: dict(n=10, p50=x, p95=2 * x, p99=3 * x, max=3 * x)
        with tempfile.TemporaryDirectory() as tmp:
            for (w, s), x in zip(((16, 1), (16, 2), (8, 2)), (.5, .9, .6)):
                d = Path(tmp) / B.name(w, s);d.mkdir()
                pooled = dict(roots=12, priced=11, eligible=10, complete=10, causes={}, stages=dict(price_seconds=stage(x), decision_seconds=stage(x + .3),
                    wall_seconds=stage(x + .35), prepare_seconds=stage(.2)), latency=dict(within_ladder=.9), wall_eligible=stage(x + .35))
                (d / 'summary.json').write_text(json.dumps(dict(pooled=pooled, chance=dict(chance_draws=s))))
            summary = B.combine(tmp, B.CONFIGS)
            md = (Path(tmp) / 'summary.md').read_text()
        self.assertEqual(([r['lanes'] for r in summary['rows']], summary['missing']), ([16, 32, 16], ['W16xS4', 'W4xS4']))
        self.assertEqual((summary['rows'][1]['price_seconds']['p95'], summary['rows'][1]['within_ladder_eligible']), (1.8, .9))
        self.assertIn('| 16 x 2 | 32 | DN2W16 | 10 / 10 | 0.900 | 1.800 | 2.700 |', md);self.assertIn('Missing (not run or not finished): W16xS4, W4xS4', md)


# ---- the analysis on synthetic chance lines ----------------------------------------------------------------------------------
PRIOR = dict(opponent_requested=True, kept=[True, True] + [False] * 7, root_peer_logits=[0.] * 9)
LEGAL2 = [True, True] + [False] * 7
LOGITS2 = [1., 0.] + [0.] * 7   # greedy (argmax of the root logits) is action 0


def grid(q0, q1):
    """A grid whose action 0 / 1 is worth q0 / q1 against both kept replies; None elsewhere."""
    return [[q0, q0] + [None] * 7, [q1, q1] + [None] * 7] + [[None] * 9 for _ in range(7)]


def chance_line(per_seed, *, ordinal=1, index=0, guard=None, own_moves=None, statuses=None, worlds=2, wrong=None):
    """A chance line whose seed s prices every world at grid(*per_seed[s]); each seed's own decision is the live code's (so the
    consistency check holds), except seed `wrong`'s action."""
    seeds = []
    for s, (q0, q1) in per_seed.items():
        prices = tuple(original.WorldPrices(True, tuple(PRIOR['kept']), tuple(PRIOR['root_peer_logits']), grid(q0, q1)) for _ in range(worlds))
        choice = hedge.averaging(prices, tuple(LEGAL2))
        composed = compose([reduce_block(prices[2 * k:2 * k + 2], LOGITS2, LEGAL2, original) for k in range(worlds // 2)], LOGITS2, LEGAL2, original)
        action = hedge.top_action(choice.strategy)
        seeds.append(dict(seed=s, status=(statuses or {}).get(s, 'COMPLETE'), action_index=1 - action if s == wrong else action, q_values=list(choice.q),
            policy_target=list(composed['policy_target']), grids=[grid(q0, q1)] * worlds, effect_guard=None if guard is None else dict(b=guard.get('b'))))
    return dict(ordinal=ordinal, decision_index=index, battle_id='b', root=dict(root_logits=LOGITS2, root_legal=LEGAL2, own_moves=own_moves),
        greedy_action_index=0, greedy_source='live', priors=[PRIOR] * worlds, prior_invariant=True, effect_guard=guard, seeds=seeds)


FLIPPY = {0: (.1, .2), 1: (.1, .2), 2: (.1, .2), 3: (.1, .05), 4: (.1, .05)}   # seed 0: the control, outside the analysis
STEADY = {s: (.1, .2) for s in range(5)}


class AnalysisTests(unittest.TestCase):
    def test_one_flippy_root(self):
        r = noise.analyse(chance_line(FLIPPY), (1, 2, 3, 4))
        self.assertEqual((r['single']['averaging'], r['single']['target_argmax'], r['reference'], r['greedy']), ([1, 1, 0, 0], [0] * 4,
            dict(averaging=1, target_argmax=0), 0))   # the root prior holds target_argmax on greedy
        self.assertEqual(r['consistency'], dict(averaging=True, policy_target=True))
        self.assertEqual((r['design']['1']['averaging'], r['design']['2']['averaging'], r['halves']['averaging']), ([1, 1, 0, 0], [1, 0], [1, 0]))
        self.assertEqual((r['double_estimator'], r['double_estimator_swapped']), (0, 0))   # 1 wins seeds 1-2, loses 3-4: greedy kept
        g = r['gap'];sd = .075 * math.sqrt(4 / 3)
        self.assertEqual(g['best'], 1);self.assertAlmostEqual(g['gap'], .025);self.assertAlmostEqual(g['sd'], sd);self.assertAlmostEqual(g['z'], .025 / sd)
        self.assertAlmostEqual(g['se'], sd / 2)
        self.assertEqual([d['seed'] for d in r['single_draw_overrides']], [1, 2]);self.assertAlmostEqual(r['single_draw_overrides'][0]['z'], .1 / sd)
        self.assertEqual(r['lcb'], {'k=0': dict(in_sample=True, cross_fit=False), 'k=1': dict(in_sample=False, cross_fit=False),
            'k=2': dict(in_sample=False, cross_fit=False), 'k=3': dict(in_sample=False, cross_fit=False)})
        self.assertAlmostEqual(r['cross_fit']['gap'], -.05);self.assertEqual(r['cross_fit']['sd'], 0.)

    def test_the_mean_grid_at_one_seed_is_the_single_draw_and_two_seeds_average(self):
        line = chance_line(FLIPPY);by = {s: noise.worlds_of(line, e) for s, e in ((e['seed'], e) for e in line['seeds'])}
        one = noise.mean_worlds(by, (1,));self.assertEqual([w.joint_values for w in one], [w.joint_values for w in by[1]])
        two = noise.mean_worlds(by, (2, 3));self.assertAlmostEqual(two[0].joint_values[1][0], .125)
        self.assertIsNone(two[0].joint_values[2][0])

    def test_summary_populations_and_guards(self):
        lines = [chance_line(FLIPPY, ordinal=1, guard=dict(b=True, own_guarded_lanes=[None] * 4)),
            chance_line(STEADY, ordinal=2, guard=dict(b=False, own_guarded_lanes=[None, ['A', 'test'], None, None])),
            chance_line(STEADY, ordinal=3, guard=dict(b=False, own_guarded_lanes=[None] * 4), wrong=2),
            chance_line(FLIPPY, ordinal=4, statuses={3: 'REFUSED'}), dict(chance_line(FLIPPY, ordinal=5), seeds=chance_line(FLIPPY)['seeds'][:4]),
            dict(chance_line(STEADY, ordinal=6), greedy_action_index=5)]
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'chance-0.jsonl';path.write_text(''.join(json.dumps(x) + '\n' for x in lines + [lines[0]]))
            s = noise.run([path], Path(tmp) / 'out')
            roots = [json.loads(x) for x in (Path(tmp) / 'out' / 'roots.jsonl').read_text().splitlines()]
            md = (Path(tmp) / 'out' / 'summary.md').read_text()
            with patch('sys.stdout', new_callable=io.StringIO):self.assertEqual(noise.main(['--chance', str(path), '--out', str(Path(tmp) / 'again'), '--seeds', '1,2']), 0)
        self.assertEqual((s['seeds'], s['lines'], s['analysed'], s['excluded']), ([1, 2, 3, 4], 7, 3,
            dict(seed_not_complete=1, seed_missing=1, greedy_missing=1, duplicate=1)))
        self.assertEqual([r['ordinal'] for r in roots], [1, 2, 3]);self.assertNotIn('record', roots[0])
        self.assertEqual((s['consistency']['averaging'], s['consistency']['policy_target']), (2, 3))   # ordinal 3's seed 2 action is wrong
        a = s['populations']['all']['operators']['averaging']
        self.assertAlmostEqual(a['flip_rate'], (4 / 6) / 3);self.assertAlmostEqual(a['roots_with_any_flip'], 1 / 3)
        self.assertAlmostEqual(a['override_single'], (.5 + 1 + 1) / 3);self.assertAlmostEqual(a['override_reference'], 1.)
        self.assertAlmostEqual(a['convergence']['2']['agree_reference'], (.5 + 1 + 1) / 3);self.assertAlmostEqual(a['convergence']['2']['agree_disjoint'], 2 / 3)
        self.assertEqual(a['convergence']['4'], dict(groups=1, agree_reference=1., agree_disjoint=None, override=1.))
        gap = s['populations']['all']['gap']
        self.assertEqual((gap['override_roots'], gap['within_1sd'], gap['zero_sd'], gap['single_draw']['overrides']), (3, 1, 2, 10))
        rules = s['populations']['all']['rules']
        self.assertAlmostEqual(rules['double_estimator']['override'], 2 / 3);self.assertAlmostEqual(rules['averaging:single_draw']['override'], (.5 + 1 + 1) / 3)
        self.assertAlmostEqual(s['populations']['all']['lcb']['k=1']['in_sample'], 2 / 3)   # the steady roots: gap > 0 = k * se
        u = s['populations']['unguarded'];self.assertEqual(u['roots'], 2)   # (b) on ordinal 1: out
        self.assertAlmostEqual(u['operators']['averaging']['override_reference'], .5)   # ordinal 2's (c) on lane 1 plays greedy
        self.assertAlmostEqual(u['rules']['double_estimator']['override'], .5)
        self.assertEqual(s['guard'], dict(with_record=3, b=1, seed_b_unstable=0, seed_b_unknown=0))
        self.assertIn('| averaging |', md);self.assertIn('LCB rule', md)
        with self.assertRaisesRegex(ValueError, 'even number'):noise.run([], '/nowhere', (1, 2, 3))

    def test_target_argmax_falls_back_when_the_composition_refuses(self):
        with patch.object(noise, 'compose', side_effect=ValueError('finite in-range numerical value required')):
            r = noise.analyse(chance_line(FLIPPY), (1, 2, 3, 4))
        self.assertEqual(r['target_fallbacks'], 5);self.assertEqual(r['single']['target_argmax'], [0] * 4)   # argmax(logits + q): 1.1 > 0.2
        self.assertFalse(r['consistency']['policy_target'])


if __name__ == '__main__':
    unittest.main()
