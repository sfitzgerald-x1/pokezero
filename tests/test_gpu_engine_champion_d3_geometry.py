"""The third geometry's tool (gpu_engine.champion_d3_geometry) on CPU.

The champion leaf validation tests' fixture: the depth-N fake engine with a zero-sum V2 layout, the readable tiny champion at both
precisions (one parameter tree), the independent champion-layout leaf and the roots bound by the validation tool's bind. D3's real
FramePricer prices at the gate's options (capacity 4096, block_rows 32, vectorized, host_prep) through g2.with_forward:
- bf16 vs f32 at every position (b) compares, over two calls: maxima, no structural difference, D_all = max + the f32 adapter bound, and the
  f32 anchor (the raw forward on independent walks) within its bounds; a mis-sliced champion exceeds the anchor's bound and is flagged;
- the comparison on synthetic D3 Results: non-finite counting, every structural difference listed by kind, positions compared only where
  both runs have them;
- D_all through the validation tool's own P_all construction; the proposal fold on a synthetic validation.json (the b sites, both bounds
  either way, a source / evaluator / roots / shape mismatch flagged, missing inputs leave no bound);
- the D3 options recorded and every departure flagged; an empty run exits 2; main writes d3_geometry.json and d3_geometry.md; the command
  line parses and refuses.
"""
import contextlib
from dataclasses import replace
import io
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from gpu_engine import champion_d3_geometry as d3g, champion_leaf_validation as v, d3_depth2_gate as g2, device_public_teacher_batch as batch, v5_model
from tests import test_gpu_engine_champion_evaluator as CE, test_gpu_engine_champion_leaf_validation as V, test_gpu_engine_depthn_oracle as N

THIS = sys.modules[__name__]
P_ROOT_OPP, O_ROOT_OPP, P_T1_VALUE = 0.10378537902832032, 0.09199810028076172, 0.009931856224536895   # the chunk-0 report's numbers


def gate_pricer(executor='ex', **kw):
    """The bank provider's FramePricer options (v5_device_teacher.Provider(vectorized=True, batched_prep=True)), historical window 2."""
    return batch.FramePricer(executor, object(), **dict(dict(capacity=4096, block_rows=32, actor_capacity=3, vectorized=True, host_prep=True), **kw))


def earlier_report(evaluators, *, roots=27, chunk=0, capacity=512, levels=True, **proposal):
    """A synthetic champion_leaf_validation report: the chunk-0 report's root and turn-1 numbers, its all_positions proposal."""
    m = lambda x, n=100, bad=0: dict(count=n, nonfinite=bad, max=x)
    level = lambda pv, po, ov, oo: dict(pricer_all=dict(value=m(pv), own_logits=m(.0888), opp_logits=m(po)), oracle_sample=dict(value=m(ov), own_logits=m(.0893), opp_logits=m(oo)))
    geo = dict(evaluator='bf16', shapes=dict(fixed_shapes=True, capacity=capacity, block_lanes=2187, depth3_k=3, oracle_chunk=512, matches_gate=capacity == 512,
        flags=[] if capacity == 512 else [f'capacity {capacity} != the gate\'s 512']),
        proposals=dict(all_positions=dict(dict(value_bound=.02538, value_bound_level='depth3.leaf', value_bound_without_root=.02538, value_bound_without_root_level='depth3.leaf',
            turn1_logit_bound=.50193, deeper_logit_bound=.68968, flags=[], matches_gate_shapes=True), **proposal)))
    if levels:geo['levels'] = {'root': level(.00345, P_ROOT_OPP, .00354, O_ROOT_OPP), 'depth2.turn1': level(P_T1_VALUE, .2062, .0047, .1057),
        'depth3.turn1': level(P_T1_VALUE, .2062, .0047, .1057)}
    header = dict(source=dict(d3_root='/d3', chunk=chunk, declaration_sha256='d' * 64), roots=roots, options=dict(max_roots=None, fixed_shapes=True, capacity=capacity),
        evaluators={name: {k: ev.header()[k] for k in ('archive_sha256', 'checkpoint_sha256', 'precision')} for name, ev in evaluators.items()})
    return dict(schema=v.SCHEMA, header=header, geometry=geo)


SOURCE = dict(d3_root='/d3', chunk=0, declaration_sha256='d' * 64)


@unittest.skipUnless(N.HAVE_PRICER, 'gpu_engine.device_depthn_pricer / depthn_backups are not on this tree')
class GeometryTests(N.FixtureN, unittest.TestCase):
    """The validation tests' requests (request 7's root inputs shifted: pre code 3) bound to the clean champion at both precisions; D3's real
    FramePricer at the gate's options prices them through g2.with_forward, the V2 forward patched to raise."""
    make = staticmethod(V.zero_sum_layout(N.executor_n))
    PRE_REFUSED = None

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        requests = list(cls.requests);r = requests[7]
        requests[7] = replace(r, root_inputs=dict(r.root_inputs, numeric_features=r.root_inputs['numeric_features'].at[..., 5, :].add(1.)))
        cls.v2_requests = tuple(requests);cls.observe = staticmethod(V.legacy_observer(N.executor_n()));cls.clean = V.pair()
        cls.bound, cls.dropped, _ = v.bind([list(cls.v2_requests)], cls.clean);cls.runs = {}

    @contextlib.contextmanager
    def engine(self):
        with super().engine(), patch.object(v5_model, 'forward', side_effect=CE.no_v2):yield

    def measured(self, evaluators, *, key=None, capacity=16, sample=48, **kw):
        if key is not None and key in self.runs:return self.runs[key]
        with self.engine():
            out = d3g.measure(self.bound, evaluators=evaluators, frame_pricer=gate_pricer(self.ex), executor=self.ex, materializer=self.tables, recipe=self.recipe,
                observe=self.observe, capacity=capacity, sample=sample, chunk=64, fixed_shapes=True, actor_capacity=3, **kw)
        if key is not None:self.runs[key] = out
        return out

    def test_bf16_against_f32_at_the_b_positions(self):
        measured = self.measured(self.clean, key='clean');calls = measured['calls']
        self.assertEqual([c['roots'] for c in calls], [16, len(self.v2_requests) - 16])   # two calls, merged
        for c in calls:
            for name in v.NAMES:self.assertEqual(c['costs'][name]['root_capacity'], 16);self.assertEqual(c['costs'][name]['expected_root_capacity'], 16)
        check = d3g.source_check(earlier_report(self.clean), source=SOURCE, evaluators=self.clean, roots=len(self.v2_requests))
        self.assertEqual((check['matches'], check['flags']), (True, []))
        report = d3g.assemble(measured, earlier=earlier_report(self.clean), check=check);c = report['comparison']
        self.assertEqual((report['status'], report['exit']), ('MEASURED', 0))
        values, logits = c['quantities'][d3g.VALUE], c['quantities'][d3g.LOGITS]
        self.assertEqual((c['structural']['count'], c['structural']['kinds']), (0, {}))
        self.assertEqual(values['count'], c['counts']['cells_compared']);self.assertEqual(c['counts']['cells_f32'], c['counts']['cells_bf16'])
        self.assertEqual(c['counts']['cells_compared'], c['counts']['cells_f32']);self.assertGreater(values['count'], 100);self.assertGreater(logits['count'], 100)
        self.assertEqual((values['nonfinite'], logits['nonfinite']), (0, 0));self.assertGreater(values['max'], 0.)   # bf16 really rounds at D3's geometry
        self.assertLess(values['max'], .05);self.assertGreaterEqual(values['max'], values['p99.99']);self.assertGreaterEqual(values['p99.99'], values['p99'])
        self.assertLess(c['counts']['priced_both'], c['counts']['worlds'])   # the missing world and request 7's pre code 3: refused by both runs alike
        self.assertGreater(c['informational'][d3g.ALL_LANES]['count'], logits['count'])
        d = report['d_all'];self.assertAlmostEqual(d[d3g.VALUE]['max'], values['max'] + v.ADAPTER_VALUE);self.assertAlmostEqual(d[d3g.LOGITS]['max'], logits['max'] + v.ADAPTER_LOGIT)
        a = report['anchor'];q = a['quantities']
        self.assertEqual(a['flags'], []);self.assertEqual(a['verified'], {d3g.VALUE: True, d3g.LOGITS: True})
        self.assertGreater(q[d3g.VALUE]['compared'], 20);self.assertLessEqual(q[d3g.VALUE]['max'], v.ADAPTER_VALUE);self.assertGreater(q[d3g.VALUE]['terminal'], 0)   # mode 20 ends at turn 1
        self.assertGreater(q[d3g.LOGITS]['f32']['count'], 20);self.assertLessEqual(q[d3g.LOGITS]['max'], v.ADAPTER_LOGIT)
        self.assertLessEqual(q[d3g.VALUE]['bf16']['max'], d[d3g.VALUE]['max'])   # the sample under D_all
        self.assertTrue(all(n == 0 for n in a['findings'].values()), a['findings'])
        self.assertTrue(all(x['anchor']['cells'] <= 48 and x['anchor']['roots'] <= 48 for x in report['calls']))
        opts = report['d3_options'];self.assertEqual(opts['departures'], ['call split 16 != the gate\'s 512'])   # the fixture's split only
        self.assertEqual({k: opts['pricers']['bf16'][k] for k in d3g.GATE_D3}, d3g.GATE_D3)
        self.assertEqual([opts['pricers'][n]['evaluator']['identity'][-1] for n in v.NAMES], ['float32', 'bfloat16'])
        p = report['proposal'];self.assertEqual((p['margin'], p['flags'], p['matches_earlier']), (None, [], True))
        self.assertAlmostEqual(p['sites']['b.t1_value']['tail'], P_T1_VALUE + d[d3g.VALUE]['max']);self.assertAlmostEqual(p['sites']['b.root_peer_logits']['tail'], P_ROOT_OPP + d[d3g.LOGITS]['max'])
        report = dict(report, header=dict(d3_options=report.pop('d3_options')));summary = d3g.markdown(v.plain(report))
        self.assertIn('| d3.t1_value |', summary);self.assertIn('| root_logit_all_geometries |', summary);self.assertIn('call split 16', summary)
        json.dumps(v.plain(report), allow_nan=False)

    def test_a_mis_sliced_champion_exceeds_the_anchor_bound(self):
        """D3 prices with a champion reading tokens 1..23 (both precisions): its f32 differs from the raw forward on the ladder's own layout, so the
        anchor flags the bound as exceeded and the layout as unequal to the evaluator's view."""
        measured = self.measured(V.pair(cls=CE.Shifted), capacity=32, sample=32);a = d3g.anchor_section([c['anchor'] for c in measured['calls']])
        self.assertGreater(a['quantities'][d3g.VALUE]['max'], v.ADAPTER_VALUE);self.assertFalse(a['verified'][d3g.VALUE])
        self.assertTrue(any('EXCEEDED' in x and d3g.VALUE in x for x in a['flags']), a['flags'])
        self.assertGreater(a['findings']['cells_inputs_unequal_f32'], 0);self.assertGreater(a['findings']['roots_inputs_unequal_bf16'], 0)

    def run_main(self, loaded, earlier, argv=('--fixed-shapes',)):
        with tempfile.TemporaryDirectory() as directory, self.engine(), patch.object(g2, 'bank_chunk_requests', return_value=loaded), \
                patch.object(g2, 'load_champion', return_value=self.clean['bf16']), patch.object(v, 'champion_observer', return_value=self.observe), \
                patch.object(d3g, 'log', lambda *a, **k: None), patch.object(d3g, 'CHUNK', 64):
            out, source = Path(directory) / 'out', Path(directory) / 'validation.json';source.write_text(json.dumps(earlier))
            code = d3g.main(['--out', str(out), '--d3-root', '/d3', '--chunk', '0', '--validation-json', str(source), '--sample', '32'] + list(argv) + CE.CHAMPION)
            return code, json.loads((out / 'd3_geometry.json').read_text()), (out / 'd3_geometry.md').read_text()

    def loaded(self, groups):
        return SimpleNamespace(provider=SimpleNamespace(executor=self.ex, materializer=self.tables, recipe=self.recipe), plan=SimpleNamespace(model=object(),
            historical_window=2), groups=groups, source=dict(SOURCE), frame_pricer=gate_pricer(self.ex))

    def test_main_writes_the_report_and_its_summary(self):
        code, report, summary = self.run_main(self.loaded([list(self.v2_requests)]), earlier_report(self.clean))
        self.assertEqual((code, report['exit'], report['status'], report['schema']), (0, 0, 'MEASURED', d3g.SCHEMA))
        h = report['header'];self.assertEqual(h['champion'], dict(archive=CE.CHAMPION[1], archive_sha256=CE.CHAMPION[3], checkpoint_sha256=CE.CHAMPION[5]))
        self.assertIn('commit', h['tree']);self.assertEqual(h['roots'], len(self.v2_requests));self.assertEqual(h['options']['split_capacity'], 512)
        self.assertEqual((h['d3_options']['departures'], h['d3_options']['matches_gate']), ([], True))   # the gate's options, split and shapes
        self.assertEqual(h['d3_options']['pricers']['f32']['actor_capacity'], 3);self.assertEqual(h['rules'], v.plain(d3g.RULES))
        self.assertEqual([h['evaluators'][n]['precision'] for n in v.NAMES], ['float32', 'bfloat16']);self.assertEqual(report['calls'][0]['roots'], len(self.v2_requests))
        self.assertEqual(report['source_check']['flags'], []);self.assertIsNotNone(report['proposal']['value_bound_all_geometries'])
        self.assertEqual(report['anchor']['flags'], []);self.assertIn('# Champion 9375: D3 FramePricer bf16 vs f32', summary);self.assertIn('## f32 anchor (sample)', summary)
        self.assertIn('- D3 options match the gate\'s: True', summary)

    def test_an_empty_run_exits_2(self):
        code, report, summary = self.run_main(self.loaded([]), earlier_report(self.clean, roots=0))
        self.assertEqual((code, report['exit'], report['status']), (2, 2, 'NOTHING_COMPARED'));self.assertIn('no D3 value', report['status_reasons'][0])
        self.assertEqual(report['comparison']['quantities'][d3g.VALUE]['count'], 0);self.assertIsNone(report['proposal']['value_bound_all_geometries'])
        self.assertIn('NOTHING_COMPARED, exit 2', summary);self.assertTrue(any('not verified' in x for x in report['anchor']['flags']))
        body = d3g.assemble(dict(calls=[], d3_options=d3g.d3_options({}), walk=None));self.assertEqual((body['status'], body['exit']), ('NOTHING_COMPARED', 2))
        d3g.markdown(v.plain(body))


# Synthetic D3 Results ------------------------------------------------------------------------------------------------------------------
def result(status, worlds):
    """A FramePricer-shaped Result: worlds is a list of a refusal reason (str) or dict(grid {(a, r): value}, requested, legal, logits, errors)."""
    reports = []
    for w in worlds:
        if isinstance(w, str):reports.append(dict(joint_values=None, reason=w));continue
        grid = [[w['grid'].get((a, r)) for r in range(9)] for a in range(9)]
        reports.append(dict(joint_values=grid, opponent_requested=w.get('requested', True), reply_legal=tuple(w.get('legal', (True, True) + (False,) * 7)),
            reply_logits=tuple(w.get('logits', (1., 2.) + (0.,) * 7)), errors=tuple(w.get('errors', ()))))
    return SimpleNamespace(status=status, world_reports=tuple(reports))


def request(q, legal=(True, True) + (False,) * 7):
    return SimpleNamespace(plans=SimpleNamespace(root=SimpleNamespace(address=SimpleNamespace(stream=100 + q, frame=4, owner=q % 2))), root_legal=legal, root_logits=(0.,) * 9)


class ComparisonTests(unittest.TestCase):
    def call(self):
        """Three roots: (0) both priced, small value and logit differences, one cell valued by f32 only; (1) world 1 refused by bf16 only, the
        root's status differs; (2) world 0's reply legal mask differs, world 1's opponent request differs, a NaN bf16 value."""
        g = {(0, 0): .5, (0, 1): -.25, (1, 0): .125, (1, 1): 1.}
        f32 = [result('COMPLETE', [dict(grid=g), dict(grid=g)]), result('COMPLETE', [dict(grid=g), dict(grid=g)]), result('COMPLETE', [dict(grid=g), dict(grid=g)])]
        moved = {**g, (0, 0): .5 + 2 ** -8, (1, 1): 1. - 2 ** -6};lesser = {k: x for k, x in moved.items() if k != (0, 1)}
        bf16 = [result('COMPLETE', [dict(grid=moved, logits=(1.0625, 2.) + (0.,) * 7), dict(grid=lesser, errors=((0, 1, 'ValueError: current public actor output refused/nonfinite'),))]),
            result('REFUSED', [dict(grid=g), 'ValueError: current public actor output refused/nonfinite']),
            result('COMPLETE', [dict(grid=g, legal=(True,) * 3 + (False,) * 6, logits=(9., 2.) + (0.,) * 7), dict(grid={**g, (1, 0): float('nan')}, requested=False)])]
        return d3g.compare_call(dict(f32=f32, bf16=bf16), dict(f32=[request(q) for q in range(3)], bf16=[request(q) for q in range(3)]), call=4)

    def test_maxima_non_finite_and_every_structural_difference(self):
        c = self.call();k = c['findings']
        self.assertEqual({name: e['count'] for name, e in k.items()}, dict(root_status=1, world_refused=1, reply_legal=1, opponent_requested=1,
            joint_value_none_pattern=1, joint_error=1))
        self.assertEqual(k['world_refused']['first'][0]['cell'], [1, 1]);self.assertEqual(k['world_refused']['first'][0]['f32'], 'priced')
        self.assertEqual(k['joint_value_none_pattern']['first'][0]['cell'], [0, 1, 0, 1]);self.assertEqual(k['root_status']['first'][0]['address'], [101, 4, 1])
        self.assertEqual(k['world_refused']['first'][0]['call'], 4)
        # values: every cell both runs value: 4 + 3 + 4 (root 1 world 0) + 4 + 4 (root 2) = 19, one of them NaN in bf16
        self.assertEqual((c['counts']['cells_compared'], c['counts']['priced_both'], c['counts']['requested_both']), (19, 5, 4))
        s = v.spread(c['values']);self.assertEqual((s['count'], s['nonfinite']), (19, 1));self.assertEqual(s['max'], 2 ** -6)
        # logits: legal in both on worlds both priced with the opponent requested in both (root 2 world 1 is not): 4 worlds x 2 lanes
        s = v.spread(c['logits']);self.assertEqual(s['count'], 8);self.assertEqual(s['max'], 8.);self.assertEqual(c['counts']['all_lanes_compared'], 5 * 9)
        merged = d3g.merge_comparison([dict(call=4, comparison=c), dict(call=5, comparison=self.call())])
        self.assertEqual(merged['structural']['count'], 12);self.assertEqual(merged['structural']['kinds']['joint_error']['count'], 2)
        self.assertEqual(merged['quantities'][d3g.VALUE]['count'], 38);self.assertEqual(merged['quantities'][d3g.VALUE]['nonfinite'], 2)
        self.assertEqual(merged['counts']['cells_compared'], 38);self.assertEqual(merged['pricing_order']['roots'], 0)

    def test_paired_requests_and_root_legal(self):
        g = {(0, 0): .5};same = [result('COMPLETE', [dict(grid=g), dict(grid=g)])]
        c = d3g.compare_call(dict(f32=same, bf16=same), dict(f32=[request(0)], bf16=[request(1, legal=(True,) * 3 + (False,) * 6)]))
        self.assertEqual({k: e['count'] for k, e in c['findings'].items()}, dict(request_address=1, request_root_legal=1))
        with self.assertRaisesRegex(ValueError, 'paired'):d3g.compare_call(dict(f32=same, bf16=same), dict(f32=[request(0)], bf16=[]))
        # each run's own root prior: a different pricing order is counted (informational), never a structural difference
        recipe = SimpleNamespace(pricing_order=lambda address, logits, legal: tuple(int(a) for a in np.argsort(-np.asarray(logits), kind='stable') if legal[a]))
        swapped = SimpleNamespace(**dict(vars(request(0)), root_logits=(0., 1.) + (0.,) * 7))
        c = d3g.compare_call(dict(f32=same, bf16=same), dict(f32=[request(0)], bf16=[swapped]), recipe=recipe)
        self.assertEqual((c['findings'], c['pricing_order_differs'], c['addresses']), ({}, [0], [[100, 4, 0]]))
        self.assertEqual(d3g.merge_comparison([dict(call=0, comparison=c)])['pricing_order']['roots'], 1)


class DAllAndProposalTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.evaluators = V.pair()

    def d(self, value, logits, bad=0):
        return d3g.d_all({d3g.VALUE: dict(count=50, nonfinite=bad, max=value), d3g.LOGITS: dict(count=90, nonfinite=0, max=logits)})

    def test_d_all_is_p_alls_construction(self):
        d = self.d(.004, .07);self.assertAlmostEqual(d[d3g.VALUE]['max'], .004 + v.ADAPTER_VALUE);self.assertAlmostEqual(d[d3g.LOGITS]['max'], .07 + v.ADAPTER_LOGIT)
        self.assertEqual(d[d3g.VALUE], v.pricer_all(dict(values=dict(x=dict(count=50, nonfinite=0, max=.004))), 'x', 'value'))
        self.assertEqual((d[d3g.VALUE]['precision_max'], d[d3g.VALUE]['f32_adapter_bound'], d[d3g.LOGITS]['f32_adapter_bound']), (.004, v.ADAPTER_VALUE, v.ADAPTER_LOGIT))
        self.assertEqual(d3g.d_all({}), {d3g.VALUE: dict(count=0, nonfinite=0), d3g.LOGITS: dict(count=0, nonfinite=0)})

    def check(self, earlier, **kw):
        return d3g.source_check(earlier, **dict(dict(source=SOURCE, evaluators=self.evaluators, roots=27), **kw))

    def test_the_fold_either_way(self):
        earlier = earlier_report(self.evaluators);root_c = max(P_ROOT_OPP + O_ROOT_OPP, 2 * P_ROOT_OPP)
        small = d3g.fold(earlier, self.d(.002, .05), self.check(earlier))   # D3 inside the earlier bounds
        self.assertEqual((small['flags'], small['margin'], small['matches_earlier']), ([], None, True))
        self.assertAlmostEqual(small['sites']['b.t1_value']['tail'], P_T1_VALUE + .002 + v.ADAPTER_VALUE);self.assertAlmostEqual(small['root_c']['tail'], root_c)
        self.assertAlmostEqual(small['sites']['b.root_peer_logits']['tail'], P_ROOT_OPP + .05 + v.ADAPTER_LOGIT)
        self.assertEqual((small['value_bound_all_geometries'], small['value_bound_all_geometries_without_root'], small['near_tie_q_bound_all_geometries']), (.02538,) * 3)
        self.assertAlmostEqual(small['root_logit_all_geometries'], root_c);self.assertEqual((small['turn1_logit_bound'], small['deeper_logit_bound']), (.50193, .68968))
        large = d3g.fold(earlier, self.d(.02, .15), self.check(earlier))   # D3 above them: the b sites set both
        self.assertAlmostEqual(large['value_bound_all_geometries'], P_T1_VALUE + .02 + v.ADAPTER_VALUE);self.assertAlmostEqual(large['near_tie_q_bound_all_geometries'], large['value_bound_all_geometries'])
        self.assertAlmostEqual(large['root_logit_all_geometries'], P_ROOT_OPP + .15 + v.ADAPTER_LOGIT);self.assertEqual(large['turn1_logit_bound'], .50193)

    def test_a_mismatched_or_incomplete_earlier_report_is_flagged(self):
        clean = earlier_report(self.evaluators)
        cases = dict(chunk=(earlier_report(self.evaluators, chunk=1), 'source chunk 1 != this run\'s 0'), roots=(clean, 'roots 27 != this run\'s 30'),
            capacity=(earlier_report(self.evaluators, capacity=256), 'capacity 256 != this run\'s 512'), shapes=(clean, 'fixed_shapes True != this run\'s False'),
            archive=(earlier_report(dict(self.evaluators, f32=replace(self.evaluators['f32'], meta=dict(self.evaluators['f32'].meta, weights_archive_sha256='c' * 64)))),
                'evaluator f32 archive_sha256'), precision=(earlier_report(dict(f32=self.evaluators['bf16'], bf16=self.evaluators['bf16'])), 'evaluator f32 precision'))
        for name, (earlier, message) in cases.items():
            with self.subTest(mismatch=name):
                kw = dict(roots=30) if name == 'roots' else dict(fixed_shapes=False) if name == 'shapes' else {}
                check = self.check(earlier, **kw);self.assertFalse(check['matches']);self.assertTrue(any(message in x for x in check['flags']), check['flags'])
                p = d3g.fold(earlier, self.d(.002, .05), check);self.assertFalse(p['matches_earlier']);self.assertTrue(any(x.startswith('source: ') and message in x for x in p['flags']))
                self.assertIsNotNone(p['value_bound_all_geometries'])   # still computed: flagged, not suppressed
        self.assertIn('the earlier pricer column does not describe the gate\'s shapes: capacity 256 != the gate\'s 512', self.check(earlier_report(self.evaluators, capacity=256))['flags'])
        bare = earlier_report(self.evaluators, levels=False);check = self.check(bare);p = d3g.fold(bare, self.d(.002, .05), check)
        self.assertIn('no geometry levels (P_all, O_sample) in the earlier report', check['flags'])
        self.assertIsNone(p['sites']['b.t1_value']['tail']);self.assertIsNone(p['value_bound_all_geometries']);self.assertIsNone(p['root_logit_all_geometries'])
        self.assertIn('b.t1_value: P_all unmeasured', p['flags']);self.assertIn('root (c) tail: no root level in the earlier report', p['flags'])
        p = d3g.fold(clean, self.d(.002, .05, bad=2), self.check(clean))   # non-finite D3 differences: no value tail
        self.assertIsNone(p['value_bound_all_geometries']);self.assertIn('b.t1_value: D_all has 2 non-finite differences', p['flags']);self.assertIsNotNone(p['root_logit_all_geometries'])
        p = d3g.fold(clean, d3g.d_all({}), self.check(clean));self.assertIn('b.root_peer_logits: D_all unmeasured', p['flags'])
        deeper = earlier_report(self.evaluators);deeper['geometry']['levels']['depth3.turn1']['pricer_all']['value']['max'] = .02
        self.assertTrue(any('P_all(depth3.turn1 value)' in x for x in d3g.fold(deeper, self.d(.002, .05), self.check(deeper))['flags']))
        self.assertEqual(d3g.source_check(None, source=SOURCE, evaluators=self.evaluators, roots=0)['flags'], ['no earlier validation report'])
        self.assertIsNone(d3g.fold(None, self.d(.002, .05))['value_bound_all_geometries'])


class OptionsTests(unittest.TestCase):
    def test_the_gates_options_are_recorded_and_departures_flagged(self):
        pair = V.pair();pricers = d3g.build_d3(gate_pricer(), pair)
        self.assertEqual([pricers[n].forward for n in v.NAMES], [pair['f32'], pair['bf16']])
        opts = d3g.d3_options(pricers, actor_capacity=3);self.assertEqual((opts['departures'], opts['matches_gate'], opts['notes']), ([], True, []))
        self.assertEqual(opts['pricers']['f32'], dict(d3g.GATE_D3, actor_capacity=3, evaluator=dict(kind='champion', identity=['champion', 'b' * 64, 'float32'])))
        self.assertEqual(opts['gate'], dict(d3g.GATE_D3, actor_capacity=3, split_capacity=512, fixed_shapes=True))
        for kw, message in ((dict(block_rows=16), 'block_rows 16 != the gate\'s 32'), (dict(host_prep=False), 'host_prep False != the gate\'s True'),
                (dict(capacity=2048), 'capacity 2048 != the gate\'s 4096'), (dict(actor_capacity=2), 'actor_capacity 2 != the gate\'s 3')):
            with self.subTest(**kw):
                opts = d3g.d3_options(d3g.build_d3(gate_pricer(**kw), pair), actor_capacity=3);self.assertFalse(opts['matches_gate'])
                self.assertEqual(opts['departures'], [f'f32: {message}', f'bf16: {message}'])
        opts = d3g.d3_options(pricers, actor_capacity=3, split_capacity=256, fixed_shapes=False)
        self.assertEqual(opts['departures'][0], 'call split 256 != the gate\'s 512');self.assertIn('--fixed-shapes not given', opts['departures'][1])
        swapped = {'f32': pricers['bf16'], 'bf16': pricers['f32']};opts = d3g.d3_options(swapped, actor_capacity=3)
        self.assertTrue(all('is not the champion at' in x for x in opts['departures']));self.assertEqual(len(opts['departures']), 2)
        opts = d3g.d3_options(pricers);self.assertEqual(opts['departures'], []);self.assertEqual(len(opts['notes']), 2)   # unknown window: noted, not passed off
        self.assertEqual((d3g.bucket(pricers['bf16'], 503), d3g.bucket(pricers['bf16'], 18), d3g.bucket(pricers['bf16'], 7)), (512, 32, 8))

    def test_tree_commit(self):
        with tempfile.TemporaryDirectory() as directory:
            self.assertIsNone(d3g.tree_commit(directory)['commit'])
            (Path(directory) / 'TREE_COMMIT').write_text('abc123\n');self.assertEqual(d3g.tree_commit(directory)['commit'], 'abc123')


class ParseTests(unittest.TestCase):
    def refused(self, argv, message):
        stderr = io.StringIO()
        with patch.object(sys, 'stderr', stderr), self.assertRaises(SystemExit):d3g.parse(argv)
        self.assertIn(message, stderr.getvalue())

    def test_the_command_line_parses_and_refuses(self):
        base = ['--out', '/tmp/o', '--d3-root', '/d3', '--chunk', '0', '--validation-json', '/v/validation.json']
        args = d3g.parse(base + CE.CHAMPION + ['--fixed-shapes'])
        self.assertEqual((args.d3_root, args.chunk, args.validation_json, args.sample, args.fixed_shapes, args.memory_policy, args.champion_precision),
            ('/d3', 0, '/v/validation.json', d3g.SAMPLE, True, False, None))
        self.assertFalse(d3g.parse(base + CE.CHAMPION).fixed_shapes);self.assertEqual(d3g.parse(base + CE.CHAMPION + ['--sample', '9']).sample, 9)
        self.refused(base[:6] + CE.CHAMPION, '--validation-json');self.refused(base[2:] + CE.CHAMPION, '--out');self.refused(base + CE.CHAMPION[:4], 'are required')
        self.refused(base + CE.CHAMPION[:3] + ['E' * 64] + CE.CHAMPION[4:], '64-hex');self.refused(base + CE.CHAMPION + ['--champion-precision', 'f32'], 'both precisions')
        self.refused(base + CE.CHAMPION + ['--sample', '0'], 'positive');self.refused(['--out', '/o', '--d3-root', '/d3', '--chunk', '-1', '--validation-json', '/v'] + CE.CHAMPION, '--chunk')
        self.refused(base + CE.CHAMPION + ['--saved-roots', '/x'], 'unrecognized')


if __name__ == '__main__':
    unittest.main(module=THIS)
