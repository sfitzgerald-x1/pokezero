"""Pure helpers of the search-seat head-to-head and the cell comparison; no engine, model or device run."""
import json
from pathlib import Path
import tempfile
from typing import NamedTuple
import unittest

import jax.numpy as j
import numpy as np

from gpu_engine import v2_search_head_to_head as h2h, v2s_compare_cells as compare


class Toy(NamedTuple):
    own: object      # [B, 2, 3]: an own-boundary field
    other: object    # [B, 2]: a field the boundary never reads
    turn: object     # [B]: a global field


class HelperTests(unittest.TestCase):
    def test_world_block_seeds_prefix_and_determinism(self):
        seeds = h2h.world_block_seeds(17, 4)
        self.assertEqual(seeds[0], 17);self.assertEqual(len(set(seeds)), 4)
        self.assertEqual(h2h.world_block_seeds(17, 2), seeds[:2])
        with self.assertRaises(ValueError):h2h.world_block_seeds(17, 0)

    def test_turn_bands(self):
        self.assertEqual([h2h.band(t) for t in (0, 1, 3, 4, 9, 10, 19, 20, 500)],
            ['1-3', '1-3', '1-3', '4-9', '4-9', '10-19', '10-19', '20+', '20+'])
        with self.assertRaises(ValueError):h2h.band(-1)

    def test_counterfactual_keeps_only_own_cells(self):
        b = 4;state = Toy(j.arange(b * 2 * 3).reshape(b, 2, 3), j.arange(b * 2).reshape(b, 2) + 100, j.arange(b) + 1000)
        rows, owners, perm = [0, 2], [1, 0], np.roll(np.arange(b), 1)
        cf = h2h.counterfactual(state, rows, owners, perm, ('own',))
        own, swapped = np.asarray(state.own), np.asarray(state.own)[perm]
        for r, o in zip(rows, owners):
            np.testing.assert_array_equal(np.asarray(cf.own)[r, o], own[r, o])            # A's own cells kept
            np.testing.assert_array_equal(np.asarray(cf.own)[r, 1 - o], swapped[r, 1 - o])  # the foe's from another game
        np.testing.assert_array_equal(np.asarray(cf.other), np.asarray(state.other)[perm])
        np.testing.assert_array_equal(np.asarray(cf.turn), np.asarray(state.turn)[perm])
        self.assertFalse(h2h.leaves_equal(cf, state))
        self.assertTrue(h2h.leaves_equal(state, Toy(*state)))

    def test_leaves_equal_is_byte_and_dtype_exact(self):
        self.assertTrue(h2h.leaves_equal({'a': np.zeros(3, np.float32)}, {'a': np.zeros(3, np.float32)}))
        self.assertFalse(h2h.leaves_equal({'a': np.zeros(3, np.float32)}, {'a': np.zeros(3, np.float64)}))
        self.assertFalse(h2h.leaves_equal({'a': np.zeros(3, np.float32)}, {'a': -np.zeros(3, np.float32)}))   # -0.0 differs
        self.assertFalse(h2h.leaves_equal({'a': np.zeros(3)}, {'b': np.zeros(3)}))
        from types import MappingProxyType
        frozen = lambda: MappingProxyType({'x': np.arange(3), 'nested': MappingProxyType({'y': np.ones(2, np.int32)})})
        self.assertTrue(h2h.leaves_equal(frozen(), frozen()))   # nested read-only mappings compare by content
        changed = MappingProxyType({'x': np.arange(3), 'nested': MappingProxyType({'y': np.zeros(2, np.int32)})})
        self.assertFalse(h2h.leaves_equal(frozen(), changed))
        with self.assertRaises(TypeError):h2h.leaves_equal({'a': object()}, {'a': object()})


class RoundingEdgeTests(unittest.TestCase):
    def test_d_reduction_rounding_edge_counts_and_other_errors_raise(self):
        import collections, random
        from pokezero import v5_gumbel as g
        search = h2h.Search.__new__(h2h.Search)
        search.original = g;search._reduce = g.reduce_completed_q;search.events = collections.Counter()
        rng = random.Random(1);root = (True, True) + (False,) * 7;edges = 0
        for _ in range(2000):
            k = rng.randint(2, 9);legal = tuple(i < k for i in range(9))
            values = [[1. if i < 2 and j < k else None for j in range(9)] for i in range(9)]
            world = g.WorldPrices(True, legal, tuple(rng.gauss(0, 3) for _ in range(9)), values)
            try:g.reduce_completed_q(root_logits=(0.,) * 9, root_legal=root, worlds=(world, world))
            except ValueError:
                edges += 1
                result = search._reduce_tolerating_rounding(root_logits=(0.,) * 9, root_legal=root, worlds=(world, world))
                self.assertEqual(result.status, 'LABELED');self.assertIsNone(result.improvement)   # counted per real row in _price
        self.assertGreater(edges, 0)
        # A genuinely out-of-range or malformed input still raises.
        with self.assertRaises(ValueError):search._reduce_tolerating_rounding(root_logits=(0.,) * 9, root_legal=root, worlds=(world,))


class FakeWorlds:
    """detach_roots/plan_starts stand-ins: roots carry only an address; plans carry it through."""
    @staticmethod
    def detach_roots(context, source, *, rows, owners, enable_device_worlds):
        from types import SimpleNamespace
        from pokezero import v5_contract as c
        return tuple(SimpleNamespace(address=c.DecisionAddress(1000 + r, 7, o)) for r, o in zip(rows, owners))

    @staticmethod
    def plan_starts(root, source, recipe):
        return root


class FakePricer:
    """Depth-1 FramePricer stand-in: row r of the call gets grids[address.stream - 1000] (None refuses)."""
    def __init__(self, grids, edges=()):self.grids = grids;self.edges = set(edges);self.last_timing = {}

    def price(self, requests, materializer, recipe, params):
        from types import SimpleNamespace
        out = []
        for request in requests:
            grid = self.grids[request.plans.address.stream - 1000]
            if grid is None:
                out.append(SimpleNamespace(status='REFUSED', reason='unknown', world_reports=[dict(reason='physical-materialization:x')]));continue
            values = [[None] * 9 for _ in range(9)]
            for a, row in enumerate(grid):
                for b, v in enumerate(row):values[a][b] = v
            report = dict(opponent_requested=True, reply_legal=tuple(b < len(grid[0]) for b in range(9)), reply_logits=(0.,) * 9,
                joint_values=values, reason=None)
            edge = request.plans.address.stream - 1000 in self.edges   # D's rounding edge: LABELED with no improvement
            out.append(SimpleNamespace(status='LABELED', reason=None, reduction=SimpleNamespace(improvement=None if edge else object()),
                world_reports=[report, report]))
        return tuple(out)


class ChooseTests(unittest.TestCase):
    """Search.choose end to end on CPU fakes: surfaces -> reduction -> played action."""
    def search(self, reduction, grids, edges=(), prior_c=None):
        import collections
        from pokezero import v5_gumbel as g
        search = h2h.Search.__new__(h2h.Search)
        search.original, search.worlds, search.source, search.materializer = g, FakeWorlds, None, None
        search.capacities, search.kind, search.pricer, search.params = (4,), 'depth1', FakePricer(grids, edges), (None,)
        search.seed, search.workers, search.pool, search._reduce = 5, 1, None, g.reduce_completed_q
        search.evaluator, search.evaluator_checked = None, True
        search.configure(reduction=reduction, blend=.5, world_blocks=1, prior_c=prior_c)
        return search

    def choose(self, search, rows, raw):
        legal = np.zeros((len(rows), 9), bool);legal[:, :2] = True
        inputs = {'numeric_features': np.zeros((len(rows), 2), np.float32)}
        return search.choose(None, rows, [0] * len(rows), np.zeros((len(rows), 9), np.float32), legal, inputs, np.asarray(raw))

    def test_reductions_play_the_expected_action(self):
        from gpu_engine import search_hedge as hedge
        pennies = [[1., -.5], [-.5, .5]]      # hedge x = (.4, .6): top action 1; averaging Q = (.25, 0): argmax 0
        grids = {0: pennies, 1: None}
        # Deterministic hedge plays the top action of the maximin strategy, whatever the uniform would sample.
        out = self.choose(self.search('hedgedet', grids), [0, 1], [0, 1])
        self.assertEqual(out[0][0], 1);self.assertTrue(out[0][1]['searched'])
        self.assertAlmostEqual(out[0][1]['strategy'][1], .6, places=5)
        self.assertEqual(out[1], (1, out[1][1]));self.assertFalse(out[1][1]['searched'])     # refused: the raw action
        self.assertEqual(out[1][1]['refusal'], 'physical-materialization:x')
        # The sampled hedge follows search_uniform; averaging plays argmax Q; raw replays the sampled raw action.
        address = FakeWorlds.detach_roots(None, None, rows=[0], owners=[0], enable_device_worlds=True)[0].address
        expected = hedge.sample((.4, .6) + (0.,) * 7, hedge.search_uniform(5, address))
        self.assertEqual(self.choose(self.search('hedge', grids), [0], [1])[0][0], expected)
        self.assertEqual(self.choose(self.search('average', grids), [0], [1])[0][0], 0)
        self.assertEqual(self.choose(self.search('raw', grids), [0], [1])[0][0], 1)

    def test_prior_reduction_trades_q_against_the_root_policy(self):
        # Q = (.3, .2); A's root logits (0, 2) on the legal pair: greedy is action 1 and the switch point is c = .05.
        legal = np.zeros((1, 9), bool);legal[:, :2] = True
        logits = np.zeros((1, 9), np.float32);logits[0, 1] = 2.
        inputs = {'numeric_features': np.zeros((1, 2), np.float32)}
        def play(reduction, prior_c=None):
            return self.search(reduction, {0: [[.3, .3], [.2, .2]]}, prior_c=prior_c).choose(None, [0], [0], logits, legal, inputs, np.asarray([1]))[0]
        self.assertEqual(play('average')[0], 0);self.assertEqual(play('prior', 0.)[0], 0)
        action, record = play('prior', .04)
        self.assertEqual(action, 0);self.assertTrue(record['searched']);self.assertAlmostEqual(record['q_gain_vs_argmax'], .1)
        action, record = play('prior', .06)
        self.assertEqual(action, 1);self.assertEqual(record['q_gain_vs_argmax'], 0.)
        for reduction, prior_c in (('prior', None), ('prior', -.1), ('average', .1)):
            with self.assertRaises(ValueError):self.search(reduction, {0: [[0.]]}, prior_c=prior_c)

    def test_unknown_cell_refuses_every_reduction_and_edges_count_real_rows_only(self):
        holed = [[1., None], [-.5, .5]]
        for reduction, prior_c in (('raw', None), ('average', None), ('hedge', None), ('hedgedet', None), ('blend', None), ('prior', .1)):
            out = self.choose(self.search(reduction, {0: holed}, prior_c=prior_c), [0], [1])
            self.assertEqual(out[0][0], 1);self.assertEqual(out[0][1]['refusal'], 'reduction-unknown-cell')
        # One real row on a rounding edge, priced in a call padded to capacity 4 with copies of it: counted once.
        search = self.search('average', {0: [[1., -.5], [-.5, .5]]}, edges=(0,))
        self.choose(search, [0], [0])
        self.assertEqual(search.events['d_reduction_rounding_edge'], 1)

    def test_top_action_ties_go_to_the_lower_index(self):
        from gpu_engine import search_hedge as hedge
        self.assertEqual(hedge.top_action((0., .5, .5) + (0.,) * 6), 1)
        self.assertEqual(hedge.top_action((.2, .3, .5) + (0.,) * 6), 2)
        with self.assertRaises(ValueError):hedge.top_action((0.,) * 9)


class ChampionMirrorTests(unittest.TestCase):
    """The champion mirror's harness seams on CPU fakes: per-seat forwards, greedy fallback, the evaluator check, CLI guards."""
    def test_grouped_forward_routes_each_seat_to_its_actor(self):
        import jax
        from types import SimpleNamespace
        b = 3;a_side = j.asarray([0, 1, 0], j.int32)
        # Row r of the flat [2b] inputs is lane r // 2, seat r % 2; its feature encodes (lane, seat). Row 5 is NaN (invalid).
        features = [[10. * (r // 2) + r % 2] for r in range(2 * b)];features[5] = [float('nan')]
        inputs = dict(numeric_features=j.asarray(features, j.float32))
        def make(sign):   # two DIFFERENT forwards: A's adds params, B's subtracts it
            return lambda params, x: SimpleNamespace(policy_logits=j.repeat(x['numeric_features'] + sign * params, 9, axis=1),
                value=x['numeric_features'][:, 0] * 0 + sign * params, valid=j.isfinite(x['numeric_features'][:, 0]))
        run = jax.jit(lambda x, a: h2h.grouped_forward(x, (100., 200.), (make(1.), make(-1.)), a, lambda o: o.valid))
        logits, values, ok = run(inputs, a_side)
        for lane in range(b):
            for seat in (0, 1):
                r = 2 * lane + seat;mine = seat == int(a_side[lane]);shift = 100. if mine else -200.
                if r == 5:   # the invalid row: refused, NaN outputs (the champion adapter's convention)
                    self.assertFalse(bool(ok[lane, seat]));self.assertTrue(np.isnan(float(logits[lane, seat, 0])))
                    self.assertTrue(np.isnan(float(values[lane, seat])))
                else:
                    self.assertTrue(bool(ok[lane, seat]));self.assertEqual(float(values[lane, seat]), shift)
                    self.assertEqual(float(logits[lane, seat, 0]), 10. * lane + seat + shift)

    def test_fallback_actions(self):
        legal = np.asarray([[True, True, False] + [False] * 6, [False, True, True] + [False] * 6])
        logits = np.asarray([[1., 1., 9.] + [0.] * 6, [5., 2., 3.] + [0.] * 6], np.float32)
        np.testing.assert_array_equal(h2h.fallback_actions('argmax', [1, 2], logits, legal), [0, 2])   # legal only; ties low
        np.testing.assert_array_equal(h2h.fallback_actions('sample', [1, 2], logits, legal), [1, 2])
        with self.assertRaises(ValueError):h2h.fallback_actions('greedy', [1, 2], logits, legal)

    def test_champion_setup_binds_in_order(self):
        from types import SimpleNamespace
        from unittest import mock
        from gpu_engine import champion_evaluator
        calls = [];evaluator = SimpleNamespace(archive_sha256='a' * 64, checkpoint_sha256='c' * 64, params={'w': 1}, header=lambda: {'kind': 'champion'})
        with mock.patch.object(champion_evaluator, 'load_from_arguments', lambda args: calls.append('load') or evaluator), \
                mock.patch.object(champion_evaluator, 'check_binding', lambda e, catalog: calls.append(('bind', e is evaluator, catalog)) or True):
            got, record, params = h2h.champion_setup(SimpleNamespace(), 'CATALOG')
        self.assertEqual(calls, ['load', ('bind', True, 'CATALOG')])
        self.assertIs(got, evaluator);self.assertIs(params, evaluator.params)
        self.assertEqual((record.iteration, record.checkpoint_sha256, record.evaluator), (9375, 'c' * 64, {'kind': 'champion'}))

    def test_refused_decision_plays_the_greedy_fallback_and_raw_stays_raw(self):
        tests = ChooseTests();grids = {0: None, 1: [[1., -.5], [-.5, .5]]}
        legal = np.zeros((2, 9), bool);legal[:, :2] = True
        inputs = {'numeric_features': np.zeros((2, 2), np.float32)}
        out = tests.search('average', grids).choose(None, [0, 1], [0, 0], np.zeros((2, 9), np.float32), legal, inputs,
            np.asarray([1, 1]), fallback_actions=np.asarray([0, 1]))
        self.assertEqual(out[0][0], 0);self.assertFalse(out[0][1]['searched'])    # refused: the greedy fallback, not raw 1
        self.assertEqual(out[1][0], 0);self.assertTrue(out[1][1]['searched'])     # searched: averaging argmax, not its fallback 1
        raw = tests.search('raw', grids).choose(None, [1], [0], np.zeros((1, 9), np.float32), legal[:1], {'numeric_features': np.zeros((1, 2))},
            np.asarray([1]), fallback_actions=np.asarray([0]))
        self.assertEqual(raw[0][0], 1)                                              # the plumbing reduction replays the raw sample

    def test_evaluator_is_checked_once_on_the_first_request(self):
        seen = []
        class Evaluator:
            def check(self, params, inputs):seen.append({k: np.shape(v) for k, v in inputs.items()});return True
        tests = ChooseTests();search = tests.search('average', {0: [[1., -.5], [-.5, .5]]})
        search.evaluator, search.evaluator_checked = Evaluator(), False
        legal = np.zeros((1, 9), bool);legal[:, :2] = True
        for _ in range(2):
            search.choose(None, [0], [0], np.zeros((1, 9), np.float32), legal, {'numeric_features': np.zeros((1, 29, 132), np.float32)}, np.asarray([0]))
        self.assertEqual(seen, [{'numeric_features': (1, 1, 29, 132), 'history_mask': (1, 1)}])

    def test_champion_command_line_guards(self):
        sha = 'a' * 64
        with tempfile.TemporaryDirectory() as root:
            base = ['--config', 'unused.json', '--out', root, '--reduction', 'none', '--label', 'x', '--b-run', 'champion', '--b-iteration', '0']
            for extra in (['--champion-archive', 'c.npz'],                                                   # shas missing
                    ['--champion-archive', 'c.npz', '--champion-archive-sha256', sha, '--champion-checkpoint-sha256', sha, '--a-run', 'r', '--a-iteration', '1']):
                with self.assertRaises(SystemExit):h2h.main(base + extra)
            champion = ['--champion-archive', 'c.npz', '--champion-archive-sha256', sha, '--champion-checkpoint-sha256', sha]
            with self.assertRaises(SystemExit):h2h.main(base + champion + ['--collector-check', '3'])   # no v5 collector check
            with self.assertRaises(SystemExit):h2h.main(base + champion + ['--a-iteration', '9'])       # A is the champion
            cells = Path(root, 'cells.json')
            for cell, extra in ((dict(label='x', b_run='/some/run', b_iteration=99, reduction='none'), champion),   # V2 opponent in champion mode
                    (dict(label='x', b_run='champion', b_iteration=0, reduction='none'), ['--a-run', 'r', '--a-iteration', '1']),   # champion in V2 mode
                    (dict(label='x', b_run='/r', b_iteration=1, reduction='raw', fallback='argmax'), ['--a-run', 'r', '--a-iteration', '1']),
                    (dict(label='x', b_run='/r', b_iteration=1, reduction='none', fallback='argmax'), ['--a-run', 'r', '--a-iteration', '1']),
                    (dict(label='x', b_run='/r', b_iteration=1, reduction='prior'), ['--a-run', 'r', '--a-iteration', '1']),              # no prior_c
                    (dict(label='x', b_run='/r', b_iteration=1, reduction='prior', prior_c=-.1), ['--a-run', 'r', '--a-iteration', '1']),
                    (dict(label='x', b_run='/r', b_iteration=1, reduction='average', prior_c=.1), ['--a-run', 'r', '--a-iteration', '1'])):
                cells.write_text(json.dumps([cell]))
                with self.assertRaises(SystemExit):h2h.main(['--config', 'unused.json', '--out', root, '--cells', str(cells)] + extra)


class NearTerminalTests(unittest.TestCase):
    def write(self, root, games, decisions):
        prefix = Path(root, 'c');np.savez(str(prefix) + '-games.npz', **games)
        Path(str(prefix) + '-decisions.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in decisions));return prefix

    def test_verdicts_floor_and_exclusions(self):
        games = dict(streams=np.asarray([1, 2, 3, 4]), status=np.asarray([1, 1, 1, 2]), winner=np.asarray([0, 1, 2, -1]),
            terminal_frame=np.asarray([10, 10, 10, 1024]))
        def decisions(n_agree, n_disagree, v=.9):
            # stream 1: A (seat 0) won; stream 2: A (seat 0) lost; stream 3 a draw; stream 4 capped (excluded).
            win = [dict(frame=8, searched=True, root_value=v)] * n_agree + [dict(frame=8, searched=True, root_value=-v)] * n_disagree
            extra = [dict(frame=2, searched=True, root_value=v), dict(frame=9, searched=False, root_value=v), dict(frame=10, searched=True, root_value=v)]
            return [dict(stream=1, a_side=0, decisions=win + extra), dict(stream=2, a_side=0, decisions=[dict(frame=9, searched=True, root_value=-.2)]),
                dict(stream=3, a_side=0, decisions=[dict(frame=9, searched=True, root_value=.7)]), dict(stream=4, a_side=0, decisions=[dict(frame=1022, searched=True, root_value=.9)])]
        with tempfile.TemporaryDirectory() as root:
            r = compare.near_terminal(self.write(root, games, decisions(45, 5)))
            self.assertEqual(r['verdict'], 'PASS');self.assertEqual(r['confident']['decisions'], 50)
            self.assertAlmostEqual(r['confident']['agreement'], .9);self.assertEqual(r['draws'], 1)
            self.assertEqual(r['unconfident'], dict(decisions=1, agreement=1.0, mean_abs_value=.2))   # the loss, |v| .2, right sign
            self.assertEqual(compare.near_terminal(self.write(root, games, decisions(35, 15)))['verdict'], 'FAIL')
            self.assertEqual(compare.near_terminal(self.write(root, games, decisions(40, 0)))['verdict'], 'INCONCLUSIVE')

    def test_seat_one_and_window_edges(self):
        # A in seat 1 on both streams: stream 5 seat 1 won (A won), stream 6 seat 0 won (A lost). Terminal frame 20, so a
        # decision at frame 17 (3 frames out) counts and one at frame 16 (4 out) doesn't; the dropped ones carry the wrong sign.
        games = dict(streams=np.asarray([5, 6]), status=np.asarray([1, 1]), winner=np.asarray([1, 0]), terminal_frame=np.asarray([20, 20]))
        won = [dict(frame=17, searched=True, root_value=.9)] * 50 + [dict(frame=16, searched=True, root_value=-.9)] * 2
        lost = [dict(frame=19, searched=True, root_value=-.9), dict(frame=16, searched=True, root_value=.9)]
        with tempfile.TemporaryDirectory() as root:
            r = compare.near_terminal(self.write(root, games, [dict(stream=5, a_side=1, decisions=won), dict(stream=6, a_side=1, decisions=lost)]))
        self.assertEqual(r['verdict'], 'PASS');self.assertEqual(r['confident'], dict(decisions=51, agreement=1.0, mean_abs_value=.9))


class CompareTests(unittest.TestCase):
    def write(self, root, label, scores, games, decisions):
        prefix = Path(root, label)
        Path(str(prefix) + '-pairs.jsonl').write_text(''.join(json.dumps(dict(stream=s, swapped=s - 10, score=v)) + '\n' for s, v in scores.items()))
        np.savez(str(prefix) + '-games.npz', **games)
        Path(str(prefix) + '-decisions.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in decisions))
        return prefix

    def test_paired_difference_and_identity(self):
        games = dict(streams=np.arange(4), status=np.ones(4, np.int32), winner=np.zeros(4, np.int32), terminal_frame=np.full(4, 7))
        decisions = [dict(stream=1, a_side=0, decisions=[dict(frame=0, raw=2, chosen=2), dict(frame=1, raw=0, chosen=3)])]
        with tempfile.TemporaryDirectory() as root:
            x = self.write(root, 'x', {1: 1., 2: .5, 3: None, 4: 0.}, games, decisions)
            y = self.write(root, 'y', {1: .5, 2: .5, 3: 1., 4: 0.}, games, decisions)
            result = compare.paired_difference(compare.pairs(x), compare.pairs(y))
            self.assertEqual(result['pairs'], 3);self.assertEqual(result['excluded'], 1)
            self.assertAlmostEqual(result['difference_points'], 100 * .5 / 3)
            self.assertEqual((result['pairs_x_better'], result['pairs_y_better']), (1, 0))
            self.assertTrue(compare.identical(x, y)['all_identical'])
            z = self.write(root, 'z', {1: 1., 2: .5, 3: None, 4: 0.}, games,
                [dict(stream=1, a_side=0, decisions=[dict(frame=0, raw=2, chosen=2), dict(frame=1, raw=0, chosen=0)])])
            self.assertEqual(compare.identical(x, z)['differing_decisions'], 1)
            self.assertFalse(compare.identical(x, z)['all_identical'])


if __name__ == '__main__':
    unittest.main()
