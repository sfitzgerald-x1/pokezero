"""The gate-surfaces tool (gpu_engine.champion_gate_surfaces) on CPU.

- End to end on the champion leaf validation fixture (the depth-N fake engine on a zero-sum V2 layout, the readable tiny champion at both
  precisions sharing one parameter tree, the roots bound ONCE under bf16 as the gate binds them): every group's geometries priced by the
  real pricers at both precisions, bf16 really rounding at every quantity, no structural difference, (e)'s Q and the guard's spreads, the
  direct sites (R1 block invariance, R2's n2 floats, (e) and child blocks, R3's turn 1 identical to the depth-2 frame's), D3 at R3's
  bucket, the fixture's capacity flagged as a departure; and the same requests through the gate's own run_call_n: (c)'s and depth2_pass's
  compared counts, block invariance's arrays and the guard equal the tool's.
- Positions against the gate's own comparisons on the fake engine's frames: (c) at every ply (check_heads_n), the n2 floats (n2_compare),
  block invariance (run_call_n's invariance) and (e)'s Q (learner_max_root through decide_roots); the guard equals summarize_run's.
- Every site of the table maps to the bound field the gate applies: summarize_run / summarize_n2 under a sentinel declaration.
- The fold on synthetic measurements: constructions, the proposal and its rounding, withholding on non-finite differences, structural
  and premise flags, the direct raise, R1's block-invariance decision, pieces, the earlier reports' checks, the Tail accumulator; the
  command line, main (live and fold-only) and the empty exit.
"""
import contextlib
import copy
from dataclasses import replace
import io
import json
import math
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from gpu_engine import champion_gate_surfaces as gs, champion_leaf_validation as v, d3_depth2_gate as g2, d3_depthn_gate as gate, depth2_backups, v5_model
from tests import test_gpu_engine_champion_d3_geometry as D, test_gpu_engine_champion_evaluator as CE, test_gpu_engine_champion_leaf_validation as V
from tests import test_gpu_engine_depthn_oracle as N

THIS = sys.modules[__name__]
SMALL = {name: dict(capacity=32) for name in gs.RUN_ARGV}   # the fixture's capacity (27 roots): a departure from the pinned 512
SENTINEL = dict(content=dict(bounds=dict(gs.SENTINEL)), sha256='sentinel', path='sentinel')


def small_runs():
    return gs.run_arguments(SMALL)


# ---------------------------------------------------------------------------------------------------------------- end to end
@unittest.skipUnless(N.HAVE_PRICER, 'gpu_engine.device_depthn_pricer / depthn_backups are not on this tree')
class MeasureTests(N.FixtureN, unittest.TestCase):
    make = staticmethod(V.zero_sum_layout(N.executor_n))
    PRE_REFUSED = None

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        requests = list(cls.requests);r = requests[7]
        requests[7] = replace(r, root_inputs=dict(r.root_inputs, numeric_features=r.root_inputs['numeric_features'].at[..., 5, :].add(1.)))
        cls.v2_requests = tuple(requests);cls.pair = V.pair()
        cls.groups, _, cls.dropped, _ = g2.bind_roots([list(cls.v2_requests)], (cls.pair['bf16'].params,), None, evaluator=cls.pair['bf16'])
        cls.requests_bf16 = tuple(cls.groups[0]);cls.runs, cls.run_record = small_runs()
        with cls.patched():
            cls.measured = gs.measure(cls.groups, evaluators=cls.pair, executor=cls.ex, model=object(), materializer=cls.tables, recipe=cls.recipe,
                frame_pricer=D.gate_pricer(cls.ex), runs=cls.runs, split=32, r3_call_roots=8)

    @classmethod
    @contextlib.contextmanager
    def patched(cls):
        fixture = cls.__new__(cls)
        with patch.object(N.mapping, 'materialize_host', side_effect=fixture.materialize), patch.object(v5_model, 'forward', side_effect=CE.no_v2):yield

    def test_every_geometry_is_measured_and_bf16_rounds(self):
        g = self.measured['geometries'];self.assertEqual(self.dropped, [])
        self.assertEqual(set(g), {'R1.frame', 'R1.invariance_twin', 'R3.frame', 'R3.d3'} | {f'R2.{c}.{k}' for c in ('full_width', 'top3') for k in ('depthn', 'depth2_cb27', 'depth2_cb81')})
        expect = dict(depthn=('t1_value', 't1_own_logits', 't1_peer_logits', 'root_peer_logits', 'root_peer_logits.all_lanes', 'leaf_value'),
            n2_depthn=('t1_value', 't2_value', 't1_own_logits', 't1_peer_logits', 'root_peer_logits'), depth2=('t1_value', 't2_value', 'root_peer_logits'))
        for gid, m in g.items():
            with self.subTest(geometry=gid):
                self.assertEqual(m['calls'], [0] if m['kind'] != 'd3' else m['calls']);self.assertEqual(m['structural'], {}, m['structural'])
                for name in expect.get(m['kind'], ('d3.t1_value', 'd3.root_peer_logits.all_lanes')):
                    q = m['quantities'][name];self.assertGreater(q['count'], 0, name);self.assertEqual(q['nonfinite'], 0, name)
                    self.assertGreater(q['max'], 0., name);self.assertLess(q['max'], .05, name)   # bf16 really rounds, and only rounds
                    self.assertGreaterEqual(q['p99.99'], q['p99']);self.assertLessEqual(q['p99.99'], q['max'])
        self.assertGreater(g['R3.frame']['quantities']['ply2_value']['count'], 0);self.assertGreater(g['R3.frame']['quantities']['ply2_opp_logits']['count'], 0)
        for gid in ('R1.frame', 'R3.frame', 'R2.full_width.depthn', 'R2.full_width.depth2_cb81', 'R2.top3.depthn', 'R2.top3.depth2_cb81'):
            q = g[gid]['q'];self.assertGreater(q['count'], 0, gid);self.assertEqual(q['crosscheck_mismatches'], []);self.assertGreater(q['decide_roots']['roots_compared'], 0)
            self.assertAlmostEqual(q['max'], q['decide_roots']['max_abs_delta_q']);self.assertEqual(len(g[gid]['gaps']['bf16']), q['decide_roots']['roots_compared'])
        self.assertAlmostEqual(g['R3.frame']['q']['adapter'], gs.q_adapter(3))
        for gid in ('R1.frame', 'R1.invariance_twin', 'R2.top3.depth2_cb27', 'R2.top3.depth2_cb81'):
            self.assertEqual(set(g[gid]['whole']), set(gs.DEPTHN_FLOATS if gid.startswith('R1') else gs.DEPTH2_FLOATS), gid)
        for gid in ('R1.frame', 'R3.frame', 'R2.top3.depthn'):
            s = g[gid]['spreads'];self.assertIsNotNone(s['bf16']['guard'], gid);self.assertGreater(s['bf16']['p10'], 0.)
        self.assertEqual(g['R3.d3']['bucket'], 8);self.assertEqual(g['R3.d3']['skipped_roots'], 27 % 8)
        self.assertTrue(any('capacity 32 != pinned 512' in x for x in self.run_record['R1']['departures']))
        self.assertIn('call split 32 != every run\'s 512', self.measured['departures'])

    def test_the_direct_sites(self):
        d = self.measured['direct']
        for name in v.NAMES:
            inv = d['R1.block_invariance'][name];self.assertTrue(inv['discrete']['passed'], inv['discrete'])
            self.assertEqual(set(inv['floats']), set(gs.DEPTHN_FLOATS))
        ident = d['R3.turn1_identity']['bf16']['fields']   # item 3: depth 3's turn 1 is depth 2's, bit for bit
        for name in gs.TURN1_FIELDS:
            with self.subTest(field=name):self.assertGreater(ident[name]['compared'], 0);self.assertEqual((ident[name]['unequal'], ident[name]['max_abs_difference']), (0, 0.))
        for config in ('full_width', 'top3'):
            n2 = d[f'R2.{config}.n2']['bf16'];self.assertTrue(n2['discrete']['passed'], n2['discrete'])
            self.assertEqual(set(n2['floats']), {'t1_value', 't2_value', 'root_peer_logits', 't1_own_logits', 't1_peer_logits'})
            self.assertGreater(d[f'R2.{config}.e']['bf16']['decide']['roots_compared'], 0)
            self.assertTrue(d[f'R2.{config}.child_block_invariance']['bf16']['discrete']['passed'])

    def test_the_fold_on_the_fixture(self):
        earlier = dict(validation=D.earlier_report(self.pair), d3_geometry=None)
        body = gs.assemble(self.measured, earlier=earlier, runs=self.runs, run_record=self.run_record, header=dict(source=D.SOURCE, roots=27, champion={}))
        self.assertEqual((body['status'], body['exit']), ('MEASURED', 0));sites = {s['id']: s for s in body['fold']['sites']}
        for sid in ('R1.e.max_abs_delta_q', 'R3.e.max_abs_delta_q', 'R2.top3.n2.e.max_abs_delta_q', 'R2.full_width.n2.e_merged_depth2.max_abs_delta_q',
                'R1.block_invariance.leaf_value', 'R2.top3.n2.floats.t2_value', 'R2.full_width.n2.block_invariance.root_peer_logits', 'R3.b.t1_value'):
            with self.subTest(site=sid):self.assertIsNotNone(sites[sid]['tail'], sites[sid]);self.assertEqual(sites[sid]['flags'], [])
        e = sites['R1.e.max_abs_delta_q'];q = self.measured['geometries']['R1.frame']['q']
        self.assertAlmostEqual(e['tail'], 2 * (q['max'] + gs.q_adapter(2)));self.assertEqual(e['construction'], 'mirror')
        n2e = sites['R2.top3.n2.e.max_abs_delta_q'];g = self.measured['geometries']
        self.assertAlmostEqual(n2e['tail'], max(g['R2.top3.depth2_cb81']['q']['max'] + gs.q_adapter(2) + g['R2.top3.depthn']['q']['max'] + gs.q_adapter(2),
            n2e['direct_measured']['max']))
        self.assertTrue(body['fold']['r1_invariance']['discrete_equal']['bf16'])
        gaps = body['fold']['gaps']['R1.e.max_abs_delta_q'];self.assertEqual(gaps['roots'], q['decide_roots']['roots_compared'])
        guard = body['fold']['guard']['R1'];self.assertTrue(guard['measured']);self.assertEqual({r['site'] for r in guard['sites']},
            {s['id'] for s in body['fold']['sites'] if s['run'] == 'R1' and s['guarded']})

    def test_the_gates_own_call_agrees(self):
        """run_call_n on the same bound requests at R1's shapes (depth 2, k 9, fixed, INVARIANCE_LANES) under the bf16 champion: (c)'s and
        depth2_pass's compared counts equal the tool's masks on the bf16 frame (no structural difference: the f32 frame agrees everywhere),
        block invariance compares the arrays the tool measured whole, and the guard equals the tool's."""
        args = self.runs['R1_invariance'];bf16 = self.pair['bf16']
        pricers = gate.DepthNPricers(gate.pricer_module(), self.ex, object(), depth=2, k_own=args.k_own_ply, k_opp=args.k_opp_ply, selection='prior', capacity=32,
            actor_capacity=1, lanes=gate.INVARIANCE_LANES, fixed_capacity=True, forward=bf16)
        depth2 = gate.depth2_pass_pricer(pricers.module, self.ex, object(), capacity=32, actor_capacity=1, digest=True, fixed_capacity=True, forward=bf16)
        oracle = N.reference.DepthNOracle(self.ex, object(), chunk=64, forward=bf16)
        with self.patched():
            evaluation, _ = gate.run_call_n(requests=self.requests_bf16, recipe=self.recipe, materializer=self.tables, params=(bf16.params,), pricers=pricers, depth2_pricer=depth2,
                frame_pricer=g2.with_forward(D.gate_pricer(self.ex), bf16), oracle=oracle, sample=None, block=0, context=dict(block=0, call=0), stack_rows=64)
        self.assertEqual(evaluation['a']['mismatched'].get('t1_status', 0), 0)
        tool = self.measured['geometries']['R1.frame']['quantities'];c = evaluation['c']
        for gate_name, mine in (('t1_value_head', 't1_value'), ('t1_own_logits', 't1_own_logits'), ('t1_peer_logits', 't1_peer_logits'), ('root_peer_logits', 'root_peer_logits'),
                ('leaf_value', 'leaf_value'), ('depth2_pass_t1_value', 't1_value'), ('depth2_pass_t1_own_logits', 't1_own_logits'), ('depth2_pass_t1_peer_logits', 't1_peer_logits'),
                ('depth2_pass_root_peer_logits', 'root_peer_logits')):
            with self.subTest(quantity=gate_name):self.assertEqual(c[gate_name]['compared'], tool[mine]['count'])
        self.assertLessEqual(c['depth2_pass_t2_value']['compared'], tool['leaf_value']['count'])
        floats = evaluation['invariance'][0]['floats'];whole = self.measured['geometries']['R1.frame']['whole']
        self.assertEqual(set(floats), set(whole));self.assertTrue(all(floats[k]['compared'] == whole[k]['count'] + whole[k]['padding_nonfinite'] for k in floats))
        result = gate.summarize_run([evaluation], gate.INVARIANCE_LANES);spreads = self.measured['geometries']['R1.frame']['spreads']['bf16']
        self.assertEqual((result['tolerance']['p10_cell_a2_spread'], result['tolerance']['guard_bound']), (spreads['p10'], spreads['guard']))
        direct = self.measured['direct']['R1.block_invariance']['bf16']['floats']
        self.assertEqual({k: x['compared'] for k, x in direct.items()}, {k: x['compared'] for k, x in floats.items()})
        self.assertEqual({k: x['max_abs_difference'] for k, x in direct.items()}, {k: x['max_abs_difference'] for k, x in floats.items()})


# ---------------------------------------------------------------------------------------------------------------- positions on the fake engine
class PositionTests(N.FixtureN, unittest.TestCase):
    """The fake engine's oracle surface as the pricer (gate tests' stand-in): exact, so f32 = bf16 and every mask is the gate's on a clean pair."""
    def frame(self, depth, k):
        return self.cached(('frame', depth, k), lambda: self._price(depth, k))

    def _price(self, depth, k):
        with self.engine():return N.OracleNPricer(self.ex, object(), depth=depth, k_own=k, k_opp=k).price(self.requests, self.tables, self.recipe, N.PARAMS)

    def data(self, frame, depth, k):
        with self.engine():
            stack = N.reference.stack(self.requests, N.batch._numpy_worlds(self.requests, self.tables, self.recipe))
            return gate.oracle_levels(oracle=self.oracle, stack=stack, requests=self.requests, recipe=self.recipe, params=N.PARAMS, frame=frame, plan=gate.plan_of(depth, k, k))

    @staticmethod
    def spy():
        seen, numeric = {}, g2._numeric
        def record(cells, got, want, *a, **k):
            out = numeric(cells, got, want, *a, **k);seen.setdefault('calls', []).append(sorted(map(tuple, np.asarray(cells).tolist())));return out
        return seen, record

    def test_c_at_every_ply_is_check_heads_n(self):
        """depthn_masks on a clean pair against the cells check_heads_n hands each (c) quantity (g2._masked's cells): turn 1, ply 2 and leaves."""
        frame = self.frame(3, 2);data = self.data(frame, 3, 2);cells = {}
        masked = g2._masked
        def record(keys, got, want, mask):
            out = masked(keys, got, want, mask);cells.setdefault('order', []).append(sorted(map(tuple, out[0].tolist())));return out
        with patch.object(g2, '_masked', record):quantities = gate.check_heads_n(frame, data, self._d())
        names = [n for n in quantities];positions = dict(zip(names, cells['order']))
        masks, structural = gs.depthn_masks(frame, copy.deepcopy(frame));self.assertEqual({k: int(np.sum(m)) for k, (m, _) in structural.items() if np.sum(m)}, {})
        for gate_name, mine in (('t1_value_head', 't1_value'), ('t1_own_logits', 't1_own_logits'), ('t1_peer_logits', 't1_peer_logits'), ('root_peer_logits', 'root_peer_logits'),
                ('ply2_value_head', 'ply2_value'), ('ply2_own_logits', 'ply2_own_logits'), ('ply2_opp_logits', 'ply2_opp_logits'), ('leaf_value', 'leaf_value')):
            with self.subTest(quantity=gate_name):self.assertEqual(positions[gate_name], sorted(map(tuple, np.argwhere(masks[mine][1]).tolist())))

    def test_align_joins_by_action_path(self):
        """A frame whose turn-1 learner slots 0 and 1 are swapped at every expanded cell (and its ply-2 nodes and leaves with them: the same
        children in another slot order, as a bf16 near-tie orders them) aligns back exactly; compared slot by slot it would not. A map whose
        slot holds an action the other frame did not select leaves that subtree unfound and counted."""
        b = self.frame(3, 2);f = copy.deepcopy(b);k_opp = f.k_opp[0];K = f.k_own[0] * k_opp;level = f.nodes[0]
        cells = np.argwhere((np.asarray(b.t1_status) == 2) & (np.asarray(b.t1_own_actions)[..., 1] >= 0))
        for q, w, a1, r1 in cells.tolist():
            c = a1 * 9 + r1;f.t1_own_actions[q, w, a1, r1, [0, 1]] = f.t1_own_actions[q, w, a1, r1, [1, 0]]
            for x in range(k_opp):
                m, n = c * K + x, c * K + k_opp + x
                for name in gs.NODE_FIELDS:
                    arr = getattr(level, name);arr[q, w, [m, n]] = arr[q, w, [n, m]].copy()
                for name in gs.LEAF_FIELDS:
                    arr = getattr(f, name);arr[q, w, [m, n]] = arr[q, w, [n, m]].copy()
        self.assertGreater(len(cells), 10)
        aligned, found = gs.align(f, b);self.assertTrue(all(m.all() for m in found.values()))
        masks, structural = gs.depthn_masks(aligned, b, found);self.assertEqual({k: int(np.sum(m)) for k, (m, _) in structural.items() if np.sum(m)}, {})
        for name, (path, mask) in masks.items():
            with self.subTest(quantity=name):self.assertTrue(mask.any());self.assertEqual(float(gs.deltas(gs._get(aligned, path)[mask], gs._get(b, path)[mask]).max()), 0.)
        masks, _ = gs.depthn_masks(f, b);path, mask = masks['leaf_value'];self.assertGreater(float(gs.deltas(gs._get(f, path)[mask], gs._get(b, path)[mask]).max()), 0.)
        g = copy.deepcopy(b);(q, w, a1, r1), = cells[:1].tolist();held = set(np.asarray(b.t1_own_actions[q, w, a1, r1]).tolist())
        g.t1_own_actions[q, w, a1, r1, 0] = next(a for a in range(9) if a not in held)
        aligned, found = gs.align(g, b);masks, structural = gs.depthn_masks(aligned, b, found)
        self.assertEqual(int(np.count_nonzero(~found['ply2'])), k_opp);self.assertEqual(int(np.sum(structural['t1_selection_differs'][0])), 1)
        self.assertEqual(int(np.sum(structural['ply2_selection_differs'][0])), int(np.count_nonzero(np.asarray(b.nodes[0].status)[q, w, (a1 * 9 + r1) * K:(a1 * 9 + r1) * K + k_opp] != 0)))

    def _d(self):
        if 'd' not in self.cache:
            with self.engine():self.cache['d'] = self.frame_pricer().price(self.requests, self.tables, self.recipe, N.PARAMS)
        return self.cache['d']

    def test_the_n2_floats_are_n2_compares(self):
        theirs = N.fake_as_depth2(self.frame(2, 9));seen, record = self.spy()
        with patch.object(g2, '_numeric', record):_, floats = gate.n2_compare(theirs, copy.deepcopy(theirs))
        masks, structural = gs.turn1_masks(copy.deepcopy(theirs), theirs);t2, _ = gs.t2_masks(copy.deepcopy(theirs), theirs);masks.update(t2)
        for name, cells in zip(floats, seen['calls']):
            with self.subTest(quantity=name):self.assertEqual(cells, sorted(map(tuple, np.argwhere(masks[name][1]).tolist())))
        self.assertGreater(len(seen['calls'][1]), 0)

    def test_block_invariance_is_run_call_ns(self):
        """The arrays run_call_n's block invariance compares, element for element, are the tool's whole arrays."""
        pricers = self.pricers(depth=2, k_own=9, k_opp=9, lanes=gate.INVARIANCE_LANES)
        evaluation, _ = self.call(pricers);floats = evaluation['invariance'][0]['floats']
        frame = self.frame(2, 9);record = gs.Geometry('x', group='r1', kind='depthn', depth=2, label='x')
        record.add_whole(gate.frame_arrays(frame), gate.frame_arrays(copy.deepcopy(frame)), 0)
        self.assertEqual(set(floats), set(record.whole));self.assertEqual(set(floats), set(gs.DEPTHN_FLOATS))
        for name, m in floats.items():self.assertEqual(m['compared'], record.whole[name].count + record.padding[name], name)

    def test_es_q_goes_through_the_gates_decide_roots(self):
        """add_q: decide_roots is called (bf16 frame first), and every recorded Q equals learner_max_root(frame, q, root logits, root legal)."""
        a = self.frame(3, 2);b = copy.deepcopy(a);leaf = b.leaf_status == 4;b.leaf_value = np.where(leaf, b.leaf_value + np.float32(2 ** -9), b.leaf_value)
        calls, real = [], gate.decide_roots
        def spy(*args, **kw):calls.append(args);return real(*args, **kw)
        record = gs.Geometry('x', group='r3', kind='depthn', depth=3, label='x')
        with patch.object(gate, 'decide_roots', spy):result = record.add_q(b, a, self.requests, N.FakeBackupsN, 0)
        self.assertEqual(len(calls), 1);self.assertIs(calls[0][0], b);self.assertIs(calls[0][1], a)
        want = []
        for q, request in enumerate(self.requests):
            x, y = (N.FakeBackupsN.learner_max_root(f, q, request.root_logits, request.root_legal).improvement for f in (b, a))
            if x is None or y is None:continue
            want += [abs(p - r) for p, r, ok in zip(x.completed_q, y.completed_q, request.root_legal) if ok and p is not None and r is not None]
        tail = record.q.summary();self.assertEqual(tail['count'], len(want));self.assertEqual(tail['max'], max(want));self.assertEqual(result['max_abs_delta_q'], max(want))
        self.assertGreater(max(want), 0.);self.assertEqual(len(record.gaps['bf16']), result['roots_compared'])

    def test_the_guard_is_the_gates(self):
        """summarize_run's guard (p10 over the checked frames' non-zero spreads) equals the tool's on the same frame, saved by run_call_n."""
        with tempfile.TemporaryDirectory() as directory:
            evaluation, _ = self.call(self.pricers(depth=3, k_own=2, k_opp=2), save=(Path(directory), 'f'))
            frame, _, _ = gate.load_frame(Path(directory) / 'f-prior-k2.npz')
        result = gate.summarize_run([evaluation], (2187,));record = gs.Geometry('x', group='r3', kind='depthn', depth=3, label='x')
        record.add_spreads(dict(f32=frame, bf16=frame), 0);guard = record.report()['spreads']['bf16']
        self.assertEqual((guard['p10'], guard['guard']), (result['tolerance']['p10_cell_a2_spread'], result['tolerance']['guard_bound']));self.assertIsNotNone(guard['guard'])

    def test_every_site_maps_to_the_gates_field(self):
        """Each (check, quantity) row of summarize_run's tolerance table under a sentinel declaration, at R1's depth 2 (with block invariance)
        and R3's depth 3, and summarize_n2's per config, has the registry's field; (e) takes the near-tie bound and (f) the value bound."""
        runs, _ = small_runs();rows = {s['id']: s for s in gs.site_registry(runs)}
        for run, depth, k, lanes in (('R1', 2, 9, gate.INVARIANCE_LANES), ('R3', 3, 3, (2187,))):
            evaluation, _ = self.call(self.pricers(depth=depth, k_own=k, k_opp=k, lanes=lanes))
            result = gate.summarize_run([evaluation], lanes, declared=SENTINEL);table = result['tolerance']['table']
            gated = {(r['check'], r['quantity']) for r in table if r['quantity'] not in g2.FLOORS}
            mine = {(s['check'], s['quantity']) for s in rows.values() if s['run'] == run and s['check'] in ('b', 'c', 'block_invariance')}
            self.assertEqual(gated, mine, run)
            for r in table:
                if r['quantity'] in g2.FLOORS:continue
                with self.subTest(run=run, site=(r['check'], r['quantity'])):self.assertEqual(rows[f'{run}.{r["check"]}.{r["quantity"]}']['field'], gs.field_of(r['bound']))
            self.assertEqual(gs.field_of(result['checks']['e']['q_noise_bound']), rows[f'{run}.e.max_abs_delta_q']['field'])
            self.assertEqual(gs.field_of(result['checks']['f']['value_bound']), rows[f'{run}.f.max_abs_delta_row_q']['field'])
            self.assertEqual(rows[f'{run}.e.max_abs_delta_q']['field'], 'near_tie_q_bound');self.assertEqual(rows[f'{run}.f.max_abs_delta_child_value']['field'], 'value_bound')
        frame = N.fake_as_depth2(self.frame(2, 9));found, floats = gate.n2_compare(frame, copy.deepcopy(frame));e = gate.decide_roots(frame, frame, self.requests, depth2_backups)
        call = dict(a=found, floats=floats, c=dict(depth1_noise_floor=g2._numeric([[0]], [.001], [0.])), e=e, spreads=[.02] * 10, e_merged=copy.deepcopy(e),
            invariance=[g2.block_invariance(frame, copy.deepcopy(frame))])
        report = gate.summarize_n2({'top3': [call]}, declared=SENTINEL)['top3']
        for name, m in report['floats'].items():
            with self.subTest(n2=name):self.assertEqual(rows[f'R2.top3.n2.floats.{name}']['field'], gs.field_of(m['bound']))
        floats = {k: m for k, m in report['block_invariance']['floats'].items() if m['compared']}   # the fake frame's empty world_reports reads as a 0-size float
        for name, m in floats.items():
            with self.subTest(child_blocks=name):self.assertEqual(rows[f'R2.top3.n2.block_invariance.{name}']['field'], gs.field_of(m['bound']))
        self.assertEqual(set(floats), set(gs.DEPTH2_FLOATS))
        for check in ('e', 'e_merged_depth2'):self.assertEqual(gs.field_of(report[check]['q_noise_bound']), rows[f'R2.top3.n2.{check}.max_abs_delta_q']['field'])
        for name, m in report['turn1_heads_vs_oracle']['quantities'].items():
            if name not in g2.FLOORS:self.assertEqual(rows[f'R2.top3.n2.heads.{name}']['field'], gs.field_of(m['bound']), name)
        guarded = {n for n, m in report['floats'].items() if m.get('guard_bound') is not None}
        self.assertEqual(guarded, {s['quantity'] for s in rows.values() if s['check'] == 'n2.floats' and s['config'] == 'top3' and s['guarded']})


# ---------------------------------------------------------------------------------------------------------------- the fold on synthetic numbers
def tail(maximum, count=100, nonfinite=0):
    return dict(count=count, nonfinite=nonfinite, max=maximum)


def geometry(quantities=(), whole=(), q=None, structural=None, depth=2, spreads=None, gaps=None):
    return dict(quantities=dict(quantities), whole={k: dict(x, padding_nonfinite=0) for k, x in dict(whole).items()}, q=None if q is None else dict(q, adapter=gs.q_adapter(depth)),
        structural=structural or {}, depth=depth, spreads=spreads, gaps=gaps)


def full_measurements(value=.002, logit=.03, q=.004, deeper=.06):
    """Every geometry measured, every quantity at a round number (value / logit / deeper / Q)."""
    t1 = dict(t1_value=tail(value), t1_own_logits=tail(logit), t1_peer_logits=tail(logit), root_peer_logits=tail(logit), **{'root_peer_logits.all_lanes': tail(logit)})
    spread = lambda: dict(bf16=dict(p10=.2, guard=.05, parents=10, nonzero=10, zero_share=0.), f32=dict(p10=.2, guard=.05), per_call_bf16=[])
    g = {'R1.frame': geometry({**t1, 'leaf_value': tail(value)}, {k: tail(logit if 'logits' in k else value) for k in gs.DEPTHN_FLOATS}, tail(q), spreads=spread(), gaps=dict(bf16=[.001, .5, .2], f32=[.5])),
        'R1.invariance_twin': geometry({**t1, 'leaf_value': tail(value)}, {k: tail(logit if 'logits' in k else value) for k in gs.DEPTHN_FLOATS}),
        'R3.frame': geometry({**t1, 'ply2_value': tail(value), 'ply2_own_logits': tail(deeper), 'ply2_opp_logits': tail(deeper), 'leaf_value': tail(value)}, (), tail(q), depth=3, spreads=spread(),
            gaps=dict(bf16=[.1], f32=[.1])),
        'R3.d3': geometry({'d3.t1_value': tail(value), 'd3.root_peer_logits.all_lanes': tail(logit)})}
    for config in ('full_width', 'top3'):
        d2 = dict(t1_value=tail(value), t2_value=tail(value), t1_own_logits=tail(logit), t1_peer_logits=tail(logit), root_peer_logits=tail(logit))
        g[f'R2.{config}.depthn'] = geometry(d2, (), tail(q), spreads=spread(), gaps=dict(bf16=[.3], f32=[.3]))
        for b in (27, 81):g[f'R2.{config}.depth2_cb{b}'] = geometry(d2, {k: tail(logit if 'logits' in k else value) for k in gs.DEPTH2_FLOATS}, tail(q) if b == 81 else None,
            gaps=dict(bf16=[.3], f32=[.3]) if b == 81 else None)
    return g


def clean_direct(n=100):
    floats = lambda names: {k: dict(compared=n, exact_equal=n, nonfinite=0, max_abs_difference=0.) for k in names}
    item = lambda names: {p: dict(floats=floats(names), discrete=dict(passed=True, mismatched={}), calls=1) for p in v.NAMES}
    out = {'R1.block_invariance': item(gs.DEPTHN_FLOATS)}
    for config in ('full_width', 'top3'):
        out[f'R2.{config}.n2'] = item(('t1_value', 't2_value', 'root_peer_logits', 't1_own_logits', 't1_peer_logits'))
        out[f'R2.{config}.child_block_invariance'] = item(gs.DEPTH2_FLOATS)
        out[f'R2.{config}.e'] = {'bf16': dict(decide=dict(roots=10, roots_compared=10, status_differs=0, nonfinite_roots=0, argmax_q_differs=0, argmax_target_differs=0), max_abs_delta_q=0., gaps=[.3])}
    return out


class FoldTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.runs, cls.record = gs.run_arguments();entry = lambda x: dict(count=10, nonfinite=0, max=x)
        cls.earlier = dict(validation=D.earlier_report(V.pair()), d3_geometry=dict(p_run={'pricer.t1_value': entry(D.P_T1_VALUE), 'pricer.root_peer_logits.all_lanes': entry(D.P_ROOT_OPP)},
            d_all={'d3.t1_value': entry(D.P_T1_VALUE), 'd3.root_peer_logits.all_lanes': entry(D.P_ROOT_OPP)}))

    def fold(self, geometries, direct=None, earlier=None):
        return gs.fold(dict(geometries=geometries, direct=clean_direct() if direct is None else direct), self.earlier if earlier is None else earlier, runs=self.runs)

    def sites(self, body):
        return {s['id']: s for s in body['sites']}

    def test_round_up(self):
        for x, want in ((.0317265, .032), (.032, .032), (.6274, .63), (.2595, .26), (.862, .87), (.0012001, .0013), (1.25, 1.3), (0., 0.), (12.01, 13.)):
            with self.subTest(x=x):self.assertEqual(gs.round_up(x), want)
        self.assertIsNone(gs.round_up(math.inf));self.assertIsNone(gs.round_up(None))
        self.assertEqual(gs.round_up(1.25 * .02538112070083618), .032)   # the draft's value bound from the earlier tail

    def test_constructions_and_the_proposal(self):
        body = self.fold(full_measurements());s = self.sites(body);va, la = gs.ADAPTER['value'], gs.ADAPTER['logit']
        self.assertAlmostEqual(s['R1.c.depth2_pass_t1_value']['tail'], 2 * max(.002 + va, D.P_T1_VALUE))   # each side: the larger of this run's and P_all
        self.assertEqual([x['source'] for x in s['R1.c.depth2_pass_t1_value']['sides']], ['validation.json'] * 2)
        self.assertAlmostEqual(s['R1.c.leaf_value']['tail'], 2 * (.002 + va));self.assertEqual(s['R1.c.leaf_value']['sides'][0]['source'], 'this run')
        p_all = D.P_T1_VALUE;self.assertAlmostEqual(s['R1.c.t1_value_head']['tail'], max(p_all + .0047, 2 * p_all))   # validation P_all (larger) + O_sample
        self.assertEqual(s['R1.c.t1_value_head']['sides'][0]['source'], 'validation.json');self.assertEqual(s['R1.c.t1_value_head']['construction'], 'oracle_sample')
        self.assertAlmostEqual(s['R1.e.max_abs_delta_q']['tail'], 2 * (.004 + gs.q_adapter(2)));self.assertAlmostEqual(s['R3.e.max_abs_delta_q']['tail'], 2 * (.004 + gs.q_adapter(3)))
        self.assertAlmostEqual(s['R2.top3.n2.e.max_abs_delta_q']['tail'], 2 * (.004 + gs.q_adapter(2)))
        self.assertEqual(s['R1.f.max_abs_delta_row_q']['tail'], s['R1.c.leaf_value']['tail']);self.assertEqual(s['R3.f.max_abs_delta_child_value']['tail'], s['R3.c.leaf_value']['tail'])
        self.assertAlmostEqual(s['R1.block_invariance.t1_own_logits']['tail'], 2 * (.03 + la));self.assertAlmostEqual(s['R3.c.ply2_own_logits']['tail'], 2 * (.06 + la))
        p = body['proposal']['fields'];values = [x['tail'] for x in body['sites'] if x['field'] == 'value_bound' and not x['report_only'] and x['tail'] is not None]
        self.assertAlmostEqual(p['value_bound']['tail'], max(values));self.assertEqual(p['value_bound']['proposed'], gs.round_up(1.25 * max(values)))
        self.assertEqual(p['value_bound']['set_by'], max((x for x in body['sites'] if x['field'] == 'value_bound' and x['tail'] is not None and not x['report_only']), key=lambda x: x['tail'])['id'])
        dq = max(x['tail'] for x in body['sites'] if x['field'] == 'near_tie_q_bound');self.assertEqual(p['near_tie_q_bound']['proposed'], gs.round_up(1.25 * max(dq, max(values))))
        self.assertTrue(p['near_tie_q_bound']['set_by'].startswith('value tail' if max(values) > dq else 'dQ tail'))
        self.assertTrue(body['proposal']['r1_block_invariance']);self.assertEqual(body['unmeasured'], [])
        self.assertTrue(all(x['covered_by_draft'] for x in body['sites'] if x['tail'] is not None and not x['report_only']))
        gaps = body['gaps']['R1.e.max_abs_delta_q'];self.assertEqual((gaps['roots'], gaps['within']['draft_near_tie']['roots']), (3, 1))
        guard = body['guard']['R1'];self.assertTrue(guard['clean_run_passes']);self.assertTrue(all(r['within'] for r in guard['sites']))

    def test_non_finite_withholds_the_site_and_the_field(self):
        g = full_measurements();g['R3.frame']['quantities']['leaf_value'] = tail(.002, nonfinite=3)
        body = self.fold(g);s = self.sites(body);p = body['proposal']['fields']
        self.assertEqual((s['R3.c.leaf_value']['status'], s['R3.c.leaf_value']['tail']), ('withheld', None))
        self.assertEqual(s['R3.f.max_abs_delta_child_value']['status'], 'withheld')   # derived from it
        self.assertIsNone(p['value_bound']['proposed']);self.assertIn('R3.c.leaf_value', p['value_bound']['withheld'])
        self.assertIsNone(p['near_tie_q_bound']['proposed']);self.assertIsNotNone(p['turn1_logit_bound']['proposed'])
        self.assertTrue(any(x['site'] == 'R3.c.leaf_value' and x['status'] == 'withheld' and any('non-finite' in r for r in x['reasons']) for x in body['unmeasured']))
        g = full_measurements();g['R1.frame']['q'] = dict(g['R1.frame']['q'], nonfinite=1)
        p = self.fold(g)['proposal']['fields'];self.assertIsNone(p['near_tie_q_bound']['proposed']);self.assertIsNotNone(p['value_bound']['proposed'])

    def test_structural_premise_and_direct_flags(self):
        g = full_measurements();g['R2.top3.depth2_cb81']['structural'] = dict(t2_status_or_error=dict(count=4, affects=['t2_value', 'q'], first=[]))
        direct = clean_direct();direct['R2.top3.n2']['f32']['floats']['t1_value']['max_abs_difference'] = 1e-3   # > 2 x 1e-5
        direct['R2.full_width.n2']['bf16']['floats']['t2_value']['max_abs_difference'] = .5   # the site measured above its construction
        body = self.fold(g, direct);s = self.sites(body)
        self.assertTrue(any('structural R2.top3.depth2_cb81 t2_status_or_error 4' in f for f in s['R2.top3.n2.floats.t2_value']['flags']))
        self.assertTrue(any('structural' in f for f in s['R2.top3.n2.e.max_abs_delta_q']['flags']));self.assertEqual(s['R2.top3.n2.floats.t1_value']['flags'][0][:7], 'premise')
        self.assertFalse(any('structural' in f for f in s['R2.top3.n2.floats.root_peer_logits']['flags']))
        self.assertEqual(s['R2.full_width.n2.floats.t2_value']['tail'], .5);self.assertTrue(s['R2.full_width.n2.floats.t2_value']['flags'][0].startswith('direct'))
        self.assertEqual(body['proposal']['fields']['value_bound']['set_by'], 'R2.full_width.n2.floats.t2_value')

    def test_r1_block_invariance_is_dropped_when_it_raises_a_field(self):
        g = full_measurements();g['R1.frame']['whole']['leaf_value'] = dict(tail(.3), padding_nonfinite=0)   # padding garbage: far above every other value site
        body = self.fold(g);d = body['r1_invariance']
        self.assertFalse(d['block_invariance']);self.assertIn('value_bound', d['raises']);self.assertFalse(body['proposal']['r1_block_invariance'])
        self.assertEqual(body['proposal']['fields'], d['proposal_without']);self.assertLess(d['proposal_without']['value_bound']['proposed'], d['proposal_with']['value_bound']['proposed'])
        direct = clean_direct();direct['R1.block_invariance']['bf16']['discrete'] = dict(passed=False, mismatched={'leaf_status': 3})
        d = self.fold(full_measurements(), direct)['r1_invariance'];self.assertFalse(d['block_invariance']);self.assertIn('bf16 discrete', d['reasons'][0])

    def test_unmeasured_sites_are_listed_with_their_exact_coverage(self):
        g = full_measurements();del g['R2.top3.depthn'], g['R2.top3.depth2_cb27'], g['R2.top3.depth2_cb81'], g['R3.d3']
        direct = clean_direct();[direct.pop(k) for k in list(direct) if k.startswith('R2.top3')]
        body = self.fold(g, direct);s = self.sites(body);missing = {x['site'] for x in body['unmeasured']}
        self.assertTrue({'R2.top3.n2.floats.t1_value', 'R2.top3.n2.e.max_abs_delta_q', 'R2.top3.n2.block_invariance.t2_value', 'R2.top3.n2.e_merged_depth2.max_abs_delta_q'} <= missing)
        self.assertTrue(all(x['exact_coverage'] for x in body['unmeasured']));self.assertIn('R2.top3.n2.floats.t1_value', body['proposal']['fields']['value_bound']['unmeasured'])
        self.assertIsNotNone(body['proposal']['fields']['value_bound']['proposed'])   # unmeasured sites set nothing (report-only in the declaration)
        self.assertIsNotNone(s['R3.b.t1_value']['tail']);self.assertIn('D3 side not at R3\'s bucket (r3-d3 not measured)', s['R3.b.t1_value']['flags'])

    def test_pieces_and_earlier_checks(self):
        piece = dict(schema=gs.SCHEMA, header=dict(source=D.SOURCE, roots=27, champion={}, split=512, tree=dict(commit='a')), measurements=dict(geometries={'R1.frame': {}}, direct={}))
        merged, _ = gs.merge_pieces(dict(geometries={'R3.frame': {}}, direct={}), [('p', piece)]);self.assertEqual(set(merged['geometries']), {'R1.frame', 'R3.frame'})
        with self.assertRaisesRegex(SystemExit, 'measured twice'):gs.merge_pieces(dict(geometries={'R1.frame': {}}, direct={}), [('p', piece)])
        flags = gs.piece_checks(dict(source=D.SOURCE, roots=30, champion={}, split=512, tree=dict(commit='b')), [('p', piece)])
        self.assertTrue(any('roots 27 != 30' in f for f in flags));self.assertTrue(any('tree a != b' in f for f in flags))
        pair = V.pair();header = dict(source=D.SOURCE, roots=27, champion={k: pair['bf16'].header()[k] for k in ('archive_sha256', 'checkpoint_sha256')})
        self.assertEqual(gs.check_earlier(dict(validation=D.earlier_report(pair), d3_geometry=None), header)['validation'], [])
        self.assertTrue(any('roots 27' in f for f in gs.check_earlier(dict(validation=D.earlier_report(pair), d3_geometry=None), dict(header, roots=30))['validation']))
        self.assertTrue(any('capacity 256' in f for f in gs.check_earlier(dict(validation=D.earlier_report(pair, capacity=256)), header)['validation']))
        self.assertEqual(gs.check_earlier({}, header)['d3_geometry'], ['no d3_geometry.json'])

    def test_the_earlier_reports_alone_reproduce_the_draft(self):
        """Folded with no measurement, the earlier reports' all-positions numbers give the draft's five values, each set by the expected site."""
        earlier = dict(validation=D.earlier_report(V.pair(), root_opp=.10378537902832032, oracle_root_opp=.09199810028076172), d3_geometry=None)
        earlier['validation']['geometry']['levels'].update({'depth2.leaf': dict(pricer_all=dict(value=dict(count=9, nonfinite=0, max=.0102745605802536)),
            oracle_sample=dict(value=dict(count=9, nonfinite=0, max=.0046))), 'depth3.leaf': dict(pricer_all=dict(value=dict(count=9, nonfinite=0, max=.01269056035041809)),
            oracle_sample=dict(value=dict(count=9, nonfinite=0, max=.0044)))})
        p = gs.fold(dict(geometries={}, direct={}), earlier, runs=self.runs)['proposal']['fields']
        self.assertEqual((p['value_bound']['proposed'], p['value_bound']['set_by']), (.032, 'R3.c.leaf_value'))
        self.assertEqual(p['root_logit_bound']['proposed'], .26);self.assertEqual(p['near_tie_q_bound']['proposed'], .032)


class TailTests(unittest.TestCase):
    def test_against_spread(self):
        rng = np.random.default_rng(3);x = np.abs(rng.normal(0, .01, 200000));x[:500] = 0.;x[7] = np.inf;t = gs.Tail()
        for part in np.array_split(x, 3):t.add(part, call=0)
        s, want = t.summary(), v.spread(x)
        self.assertEqual((s['count'], s['nonfinite'], s['max']), (want['count'], want['nonfinite'], want['max']));self.assertAlmostEqual(s['exact'], want['exact'])
        finite = np.sort(x[np.isfinite(x)])
        for k, p in (('p99', 99), ('p99.9', 99.9), ('p99.99', 99.99)):   # the nearest-rank order statistic's bin edge: at or above it, within 0.1%
            rank = finite[math.ceil(p / 100 * finite.size) - 1]
            with self.subTest(q=k):self.assertGreaterEqual(s[k], rank);self.assertLessEqual(s[k], rank * 1.0011);self.assertLess(abs(s[k] - want[k]) / want[k], .01)
        self.assertEqual(s['largest'][0]['abs_difference'], 'non-finite');self.assertEqual(gs.Tail().summary(), dict(count=0, nonfinite=0, largest=[]))
        z = gs.Tail().add(np.zeros(10)).summary();self.assertEqual((z['max'], z['p99']), (0., 0.))

    def test_deltas_are_numerics(self):
        x = np.asarray([1., np.nan, np.inf, 2., 0.], np.float32);y = np.asarray([1.5, np.nan, 1., 2., -0.], np.float32)
        self.assertEqual(gs.deltas(x, y).tolist(), [.5, 0., math.inf, 0., 0.])   # NaN bits equal: exact; -0 vs 0: unequal bits, |diff| 0
        np.testing.assert_array_equal(np.sort(gs.deltas(x, y)[gs.deltas(x, y) > 0]), np.sort(g2._numeric(None, x, y, shape=x.shape)['_deltas']))
        d, keep, padding = gs.whole_deltas(np.asarray([np.nan, 1.], np.float32), np.asarray([-np.nan, 1.], np.float32))
        self.assertEqual((d.tolist(), keep.tolist(), padding), ([0.], [False, True], 1))


class ParseTests(unittest.TestCase):
    def refused(self, argv, message):
        stderr = io.StringIO()
        with patch.object(sys, 'stderr', stderr), self.assertRaises(SystemExit):gs.parse(argv)
        self.assertIn(message, stderr.getvalue())

    def test_the_command_line(self):
        base = ['--out', '/tmp/o', '--d3-root', '/d3', '--chunk', '0', '--validation-json', '/v.json', '--d3-geometry-json', '/d.json']
        args = gs.parse(base + CE.CHAMPION + ['--fixed-shapes', '--measure', 'r1,r1-invariance', '--calls', '0,2-4'])
        self.assertEqual((args.groups, args.call_set, args.fixed_shapes, args.r3_call_roots), (('r1', 'r1-invariance'), {0, 2, 3, 4}, True, 40))
        self.assertEqual(gs.parse(base + CE.CHAMPION).groups, gs.GROUPS)
        self.refused(base + CE.CHAMPION + ['--measure', 'r1-invariance'], 'r1-invariance needs r1');self.refused(base + CE.CHAMPION + ['--measure', 'r9'], 'unknown group')
        self.refused(base + CE.CHAMPION + ['--champion-precision', 'f32'], 'both precisions');self.refused(base + CE.CHAMPION[:4], 'are required')
        self.refused(base[:6] + CE.CHAMPION, '--validation-json');self.refused(['--out', '/o', '--validation-json', '/v', '--d3-geometry-json', '/d'], '--fold-from')
        fold = gs.parse(['--out', '/o', '--validation-json', '/v', '--d3-geometry-json', '/d', '--fold-from', '/a', '/b']);self.assertEqual(fold.fold_from, ['/a', '/b'])
        self.refused(['--out', '/o', '--validation-json', '/v', '--d3-geometry-json', '/d', '--fold-from', '/a'] + CE.CHAMPION, 'prices nothing')

    def test_the_runs_parse_with_the_gates_parser(self):
        runs, record = gs.run_arguments()
        self.assertEqual({k: record[k]['departures'] for k in gs.RUN_ARGV}, {k: [] for k in gs.RUN_ARGV});self.assertTrue(record['R2']['fixed_shapes_refused']['refused'])
        self.assertEqual((record['gate_cli_bare_defaults']['depth'], record['gate_cli_bare_defaults']['capacity'], record['gate_cli_bare_defaults']['n2_k']), (3, 512, 3))
        self.assertEqual((runs['R1'].depth, runs['R1'].k_own_ply, runs['R3'].k_own_ply, runs['R2'].n2_k, runs['R3'].mutation_test), (2, (9,), (3, 3), 3, True))
        runs, record = gs.run_arguments(dict(R1=dict(capacity=64)));self.assertEqual(record['R1']['departures'], ['R1: capacity 64 != pinned 512'])


@unittest.skipUnless(N.HAVE_PRICER, 'gpu_engine.device_depthn_pricer / depthn_backups are not on this tree')
class MainTests(N.FixtureN, unittest.TestCase):
    make = staticmethod(V.zero_sum_layout(N.executor_n))
    PRE_REFUSED = None

    def run_main(self, groups, argv=()):
        pair = V.pair();loaded = SimpleNamespace(provider=SimpleNamespace(executor=self.ex, materializer=self.tables, recipe=self.recipe), plan=SimpleNamespace(model=object(),
            historical_window=2), groups=groups, source=dict(D.SOURCE), frame_pricer=D.gate_pricer(self.ex))
        directory = Path(tempfile.mkdtemp(prefix='gate-surfaces-'));v_path, d_path = directory / 'validation.json', directory / 'd3_geometry.json'
        v_path.write_text(json.dumps(D.earlier_report(pair)));d_path.write_text(json.dumps(dict(schema='x')))
        real = gs.run_arguments
        with self.engine(), patch.object(v5_model, 'forward', side_effect=CE.no_v2), patch.object(g2, 'bank_chunk_requests', return_value=loaded), \
                patch.object(g2, 'load_champion', return_value=pair['bf16']), patch.object(gs, 'log', lambda *a, **k: None), \
                patch.object(gs, 'run_arguments', lambda overrides=None: real(SMALL)):
            out = directory / 'out'
            code = gs.main(['--out', str(out), '--d3-root', '/d3', '--chunk', '0', '--validation-json', str(v_path), '--d3-geometry-json', str(d_path), '--fixed-shapes',
                '--measure', ','.join(argv), '--r3-call-roots', '8'] + CE.CHAMPION)
            return code, json.loads((out / 'gate_surfaces.json').read_text()), (out / 'gate_surfaces.md').read_text(), directory

    def test_main_writes_a_piece_and_folds_it_offline(self):
        code, report, summary, directory = self.run_main([list(self.requests)], ('r3-d3',))
        self.assertEqual((code, report['exit'], report['status'], report['schema']), (0, 0, 'MEASURED', gs.SCHEMA))
        h = report['header'];self.assertEqual(h['groups'], ['r3-d3']);self.assertIn('commit', h['tree']);self.assertEqual(h['rules']['draft'], gs.DRAFT)
        self.assertEqual(h['estimate']['minutes'], {'r3-d3': gs.ESTIMATE_MINUTES['r3-d3']});self.assertEqual(set(h['gate_cli_defaults']), {'bare', *gs.RUN_ARGV})
        self.assertEqual(set(h['pricer_options']), {'R3.d3'});self.assertEqual(h['pricer_options']['R3.d3']['departures'], [])
        self.assertTrue(h['runs']['R2']['fixed_shapes_refused']['refused']);self.assertEqual(h['calls']['split'], 512)
        self.assertGreater(report['measurements']['geometries']['R3.d3']['quantities']['d3.t1_value']['count'], 0)
        self.assertTrue(any('schema' in f for f in report['flags']));self.assertIn('# Champion 9375: the gate\'s float surfaces', summary)
        self.assertIn('| R3.b.t1_value |', summary);self.assertTrue(any(x['site'] == 'R1.e.max_abs_delta_q' for x in report['fold']['unmeasured']))
        out = directory / 'fold'
        with patch.object(gs, 'log', lambda *a, **k: None):
            code = gs.main(['--out', str(out), '--validation-json', str(directory / 'validation.json'), '--d3-geometry-json', str(directory / 'd3_geometry.json'),
                '--fold-from', str(directory / 'out')])
        folded = json.loads((out / 'gate_surfaces.json').read_text());self.assertEqual((code, folded['header']['mode']), (0, 'FOLD_ONLY'))
        self.assertEqual(folded['measurements']['geometries']['R3.d3'], report['measurements']['geometries']['R3.d3'])
        self.assertEqual(folded['fold']['proposal'], report['fold']['proposal'])

    def test_an_empty_run_exits_2(self):
        code, report, summary, _ = self.run_main([], ('r3-d3',))
        self.assertEqual((code, report['exit'], report['status']), (2, 2, 'NOTHING_COMPARED'));self.assertIn('NOTHING_COMPARED, exit 2', summary)


if __name__ == '__main__':
    unittest.main(module=THIS)
