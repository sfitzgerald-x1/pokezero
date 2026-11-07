"""The third geometry's tool (gpu_engine.champion_d3_geometry) on CPU.

The champion leaf validation tests' fixture: the depth-N fake engine with a zero-sum V2 layout, the readable tiny champion at both
precisions (one parameter tree), the independent champion-layout leaf; the roots bound ONCE under the bf16 champion (g2.bind_roots, as the
gate) and priced at both precisions by D3's real FramePricer at the gate's options (capacity 4096, block_rows 32, vectorized, host_prep;
g2.with_forward) and by the gate's real full-width depth-2 pass (d3_depthn_gate.depth2_pass_pricer):
- bf16 vs f32 at every position (b) compares, over two calls, for both programs: maxima, all lanes >= legal lanes, no structural difference,
  D_all and P = max + the f32 adapter bound; the f32 anchor (the raw forward on independent walks) within its bounds on values, legal and
  illegal lanes; the fold names the term that sets each bound; a mis-sliced champion exceeds the anchor's bounds and is flagged;
- both comparisons on synthetic inputs: non-finite counting, every structural difference listed by kind, (b)'s positions exactly (all lanes
  of every D3-priced world, cells D3 priced that both frames enumerate);
- D_all through the validation tool's own P_all construction; the fold on a synthetic validation.json (the all-lanes term picked when it is
  larger, this run's P(t1) used and flagged when larger, the legal-lane cross-check, a source / evaluator / roots / shape mismatch flagged,
  missing inputs leave no bound);
- the D3 and depth-2 pass options recorded and every departure flagged; an empty run exits 2; main writes d3_geometry.json and
  d3_geometry.md; the command line parses and refuses.
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

import jax.numpy as j
import numpy as np

from gpu_engine import champion_d3_geometry as d3g, champion_evaluator as ce, champion_leaf_validation as v, d3_depth2_gate as g2, d3_depthn_gate as gate
from gpu_engine import device_public_teacher_batch as batch, v5_model
from tests import test_gpu_engine_champion_evaluator as CE, test_gpu_engine_champion_leaf_validation as V, test_gpu_engine_depthn_oracle as N

THIS = sys.modules[__name__]
P_ROOT_OPP, O_ROOT_OPP, P_T1_VALUE = 0.10378537902832032, 0.09199810028076172, 0.009931856224536895   # the chunk-0 report's numbers
SOURCE = dict(d3_root='/d3', chunk=0, declaration_sha256='d' * 64)
VALUE_BOUNDS = {'value_bound_all_geometries', 'value_bound_all_geometries_without_root', 'near_tie_q_bound_all_geometries'}   # literal: not d3g.BOUND_SITES
ROOT_BOUNDS = {'root_logit_all_geometries'}


def gate_pricer(executor='ex', **kw):
    """The bank provider's FramePricer options (v5_device_teacher.Provider(vectorized=True, batched_prep=True)), historical window 2."""
    return batch.FramePricer(executor, object(), **dict(dict(capacity=4096, block_rows=32, actor_capacity=3, vectorized=True, host_prep=True), **kw))


def earlier_report(evaluators, *, roots=27, chunk=0, capacity=512, levels=True, root_opp=P_ROOT_OPP, oracle_root_opp=O_ROOT_OPP, **proposal):
    """A synthetic champion_leaf_validation report: the chunk-0 report's root and turn-1 numbers (P_all with its precision max), its
    all_positions proposal."""
    m = lambda x, n=100: dict(count=n, nonfinite=0, max=x)
    pa = lambda x, bound: dict(count=100, nonfinite=0, precision_max=x - bound, f32_adapter_bound=bound, max=x)
    level = lambda pv, po, ov, oo: dict(pricer_all=dict(value=pa(pv, v.ADAPTER_VALUE), own_logits=pa(.0888, v.ADAPTER_LOGIT), opp_logits=pa(po, v.ADAPTER_LOGIT)),
        oracle_sample=dict(value=m(ov), own_logits=m(.0893), opp_logits=m(oo)))
    geo = dict(evaluator='bf16', shapes=dict(fixed_shapes=True, capacity=capacity, block_lanes=2187, depth3_k=3, oracle_chunk=512, matches_gate=capacity == 512,
        flags=[] if capacity == 512 else [f'capacity {capacity} != the gate\'s 512']),
        proposals=dict(all_positions=dict(dict(value_bound=.02538, value_bound_level='depth3.leaf', value_bound_without_root=.02538, value_bound_without_root_level='depth3.leaf',
            turn1_logit_bound=.50193, deeper_logit_bound=.68968, flags=[], matches_gate_shapes=True), **proposal)))
    if levels:geo['levels'] = {'root': level(.00345, root_opp, .00354, oracle_root_opp), 'depth2.turn1': level(P_T1_VALUE, .2062, .0047, .1057),
        'depth3.turn1': level(P_T1_VALUE, .2062, .0047, .1057)}
    header = dict(source=dict(d3_root='/d3', chunk=chunk, declaration_sha256='d' * 64), roots=roots, options=dict(max_roots=None, fixed_shapes=True, capacity=capacity),
        evaluators={name: {k: ev.header()[k] for k in ('archive_sha256', 'checkpoint_sha256', 'precision')} for name, ev in evaluators.items()})
    return dict(schema=v.SCHEMA, header=header, geometry=geo)


class RootPoisoned(ce.ChampionEvaluator):
    """bf16 only: NaN (invalid) on every turn-0 view of the root whose clock is `clock`: D3's bf16 refuses its worlds (pre 9: the opponent's
    prior is unusable); f32, and every other root, are untouched."""
    clock = None

    def __call__(self, params, inputs):
        out = super().__call__(params, inputs)
        if self.precision != 'bfloat16':return out
        x = inputs['numeric_features'][:, 0, 0];hit = (x[:, 4] == 0) & (x[:, 1] == type(self).clock)
        return out._replace(policy_logits=j.where(hit[:, None], j.nan, out.policy_logits), value=j.where(hit, j.nan, out.value), valid=out.valid & ~hit)


@unittest.skipUnless(N.HAVE_PRICER, 'gpu_engine.device_depthn_pricer / depthn_backups are not on this tree')
class GeometryTests(N.FixtureN, unittest.TestCase):
    """The validation tests' requests (request 7's root inputs shifted: pre code 3) bound once to the clean bf16 champion; D3's real FramePricer
    and the real depth-2 pass price them at both precisions, the V2 forward patched to raise."""
    make = staticmethod(V.zero_sum_layout(N.executor_n))
    PRE_REFUSED = None

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        requests = list(cls.requests);r = requests[7]
        requests[7] = replace(r, root_inputs=dict(r.root_inputs, numeric_features=r.root_inputs['numeric_features'].at[..., 5, :].add(1.)))
        cls.v2_requests = tuple(requests);cls.observe = staticmethod(V.legacy_observer(N.executor_n()));cls.clean = V.pair()
        cls.groups, _, cls.dropped, _ = g2.bind_roots([list(cls.v2_requests)], (cls.clean['bf16'].params,), None, evaluator=cls.clean['bf16']);cls.runs = {}

    @contextlib.contextmanager
    def engine(self):
        with super().engine(), patch.object(v5_model, 'forward', side_effect=CE.no_v2):yield

    def measured(self, evaluators, *, key=None, capacity=16, sample=48, **kw):
        if key is not None and key in self.runs:return self.runs[key]
        with self.engine():
            out = d3g.measure(self.groups, evaluators=evaluators, frame_pricer=gate_pricer(self.ex), executor=self.ex, model=object(), materializer=self.tables,
                recipe=self.recipe, observe=self.observe, capacity=capacity, sample=sample, chunk=64, fixed_shapes=True, actor_capacity=3, **kw)
        if key is not None:self.runs[key] = out
        return out

    def test_bf16_against_f32_at_the_b_positions(self):
        measured = self.measured(self.clean, key='clean');calls = measured['calls'];self.assertEqual(self.dropped, [])
        self.assertEqual([c['roots'] for c in calls], [16, len(self.v2_requests) - 16])   # two calls, merged
        for c in calls:
            for name in v.NAMES:self.assertEqual(c['costs'][f'd3.{name}']['root_capacity'], 16);self.assertEqual(c['costs'][f'd3.{name}']['expected_root_capacity'], 16)
        body = d3g.assemble(measured);c = body['comparison'];q, pq = c['quantities'], c['pricer']['quantities']
        self.assertEqual((body['status'], body['exit']), ('MEASURED', 0))
        self.assertEqual((c['structural']['count'], c['pricer']['structural']['count']), (0, 0))
        self.assertEqual(q[d3g.VALUE]['count'], c['counts']['cells_compared']);self.assertEqual(c['counts']['cells_f32'], c['counts']['cells_bf16'])
        self.assertEqual(c['counts']['cells_compared'], c['counts']['cells_f32']);self.assertGreater(q[d3g.VALUE]['count'], 100);self.assertGreater(q[d3g.LOGITS]['count'], 100)
        self.assertLess(c['counts']['priced_both'], c['counts']['worlds'])   # the missing world and request 7's pre code 3: refused by both runs alike
        for name in d3g.QUANTITIES + d3g.PRICER_QUANTITIES:
            m = {**q, **pq}[name]
            with self.subTest(quantity=name):self.assertEqual(m['nonfinite'], 0);self.assertGreater(m['max'], 0.);self.assertLess(m['max'], .05)   # bf16 really rounds
        self.assertGreaterEqual(q[d3g.VALUE]['max'], q[d3g.VALUE]['p99.99']);self.assertGreaterEqual(q[d3g.VALUE]['p99.99'], q[d3g.VALUE]['p99'])
        # (b)'s positions: all nine lanes of every world D3 priced, legal or not, requested or not; at least the legal lanes' maxima
        self.assertEqual(q[d3g.ALL_LANES]['count'], 9 * c['counts']['priced_both']);self.assertEqual(pq[d3g.P_ALL_LANES]['count'], 9 * c['counts']['priced_bf16'])
        self.assertGreater(c['counts']['priced_both'], c['counts']['requested_both'])   # unrequested worlds are in (b)'s positions
        self.assertGreaterEqual(q[d3g.ALL_LANES]['max'], q[d3g.LOGITS]['max']);self.assertGreaterEqual(pq[d3g.P_ALL_LANES]['max'], pq[d3g.P_LOGITS]['max'])
        self.assertEqual(pq[d3g.P_VALUE]['count'], c['counts']['cells_bf16'])   # full width enumerates every cell D3 priced
        d, p = body['d_all'], body['p_run']
        for name, m in {**q, **pq}.items():
            bound = v.ADAPTER_VALUE if name in (d3g.VALUE, d3g.P_VALUE) else v.ADAPTER_LOGIT
            with self.subTest(bounded=name):self.assertAlmostEqual({**d, **p}[name]['max'], m['max'] + bound)
        a = body['anchor'];aq = a['quantities']
        self.assertEqual(a['flags'], []);self.assertEqual(a['verified'], {d3g.VALUE: True, d3g.LOGITS: True, d3g.ILLEGAL: True})
        self.assertGreater(aq[d3g.VALUE]['compared'], 20);self.assertLessEqual(aq[d3g.VALUE]['max'], v.ADAPTER_VALUE);self.assertGreater(aq[d3g.VALUE]['terminal'], 0)   # mode 20
        for key in (d3g.LOGITS, d3g.ILLEGAL):
            with self.subTest(anchor=key):self.assertGreater(aq[key]['f32']['count'], 20);self.assertLessEqual(aq[key]['max'], v.ADAPTER_LOGIT)
        self.assertLessEqual(aq[d3g.VALUE]['bf16']['max'], d[d3g.VALUE]['max'])   # the sample under D_all
        self.assertTrue(all(n == 0 for n in a['findings'].values()), a['findings'])
        self.assertGreater(aq[d3g.ILLEGAL]['lanes'], 0)   # every sampled world's nine lanes: its legal ones and the rest
        self.assertEqual(aq[d3g.LOGITS]['lanes'] + aq[d3g.ILLEGAL]['lanes'], 9 * aq[d3g.LOGITS]['compared']);self.assertEqual(aq[d3g.LOGITS]['compared'], aq[d3g.ILLEGAL]['compared'])
        pa = body['pricer_anchor'];self.assertEqual(pa['flags'], [])   # the pass's f32 against D3's f32 at (b)'s positions, every kind of lane
        for k, bound in d3g.PRICER_ANCHOR.items():
            with self.subTest(pass_premise=k):self.assertGreater(pa['quantities'][k]['count'], 0);self.assertIs(pa['quantities'][k]['within'], True);self.assertEqual(pa['quantities'][k]['bound'], bound)
        self.assertEqual(pa['quantities']['t1_value']['count'], pq[d3g.P_VALUE]['count']);self.assertEqual(body['proposal']['flagged_bounds'], {})
        self.assertTrue(all(x['anchor']['cells'] <= 48 and x['anchor']['worlds'] <= 48 for x in body['calls']))
        opts, popts = body['d3_options'], body['pass_options']
        self.assertEqual(opts['departures'], ['call split 16 != the gate\'s 512']);self.assertEqual({k: opts['pricers']['bf16'][k] for k in d3g.GATE_D3}, d3g.GATE_D3)
        # the depth-2 pass: the gate's options but the fixture's capacity, which is flagged
        self.assertEqual(popts['departures'], ['f32: capacity 16 != the gate\'s 512', 'bf16: capacity 16 != the gate\'s 512'])
        self.assertEqual({k: popts['pricers']['f32'][k] for k in d3g.GATE_PASS if k != 'capacity'}, {k: x for k, x in d3g.GATE_PASS.items() if k != 'capacity'})
        self.assertEqual([popts['pricers'][n]['evaluator']['identity'][-1] for n in v.NAMES], ['float32', 'bfloat16'])
        # the fold, the earlier report's legal-lane root prior being this run's (the cross-check holds) and its oracle column 0: every term
        # is this run's, and the bound names the largest
        earlier = earlier_report(self.clean, root_opp=p[d3g.P_LOGITS]['max'], oracle_root_opp=0.);check = d3g.source_check(earlier, source=SOURCE, evaluators=self.clean, roots=27)
        self.assertEqual((check['matches'], check['flags']), (True, []))
        prop = d3g.fold(earlier, d, check, p);self.assertEqual((prop['flags'], prop['margin'], prop['matches_earlier']), ([], None, True))
        self.assertLess(prop['cross_checks']['root_peer_logits']['difference'], d3g.CROSS_CHECK)
        terms = {'root (c) tail': 2 * p[d3g.P_LOGITS]['max'], 'b.root_peer_logits': p[d3g.P_LOGITS]['max'] + d[d3g.LOGITS]['max'],
            'b.root_peer_logits_all_lanes': p[d3g.P_ALL_LANES]['max'] + d[d3g.ALL_LANES]['max']}
        for name, x in terms.items():
            with self.subTest(term=name):self.assertAlmostEqual(prop['root_c']['tail'] if name == 'root (c) tail' else prop['sites'][name]['tail'], x)
        self.assertAlmostEqual(prop['root_logit_all_geometries'], max(terms.values()));self.assertEqual(prop['root_logit_all_geometries_set_by'], max(terms, key=terms.get))
        self.assertEqual(prop['root_logit_all_geometries_set_by'], 'b.root_peer_logits_all_lanes')   # on the fixture the illegal lanes move most
        self.assertAlmostEqual(prop['sites']['b.t1_value']['tail'], P_T1_VALUE + d[d3g.VALUE]['max']);self.assertEqual(prop['cross_checks']['t1_value']['used'], 'validation.json')
        report = dict(body, header=dict(d3_options=body['d3_options'], pass_options=body['pass_options']), proposal=prop, source_check=check)
        summary = d3g.markdown(v.plain(report))
        for text in ('| d3.root_peer_logits.all_lanes |', '| pricer.root_peer_logits.all_lanes |', '| b.root_peer_logits_all_lanes |', 'set by ', 'capacity 16 != the gate'):
            self.assertIn(text, summary)
        json.dumps(v.plain(report), allow_nan=False)

    def test_bs_positions_are_the_bf16_d3s(self):
        """A root only the bf16 D3 refuses (its opponent's bf16 prior is NaN: pre 9): D3 lists both worlds as refused; they leave (b)'s positions,
        so the pass is compared on the bf16 D3's worlds only (no pricer difference, P_all_lanes over 9 x its worlds), and every site and bound
        is flagged with the refusal."""
        clocks = [float(r.plans.root.address.stream % 79) for r in self.groups[0]]
        q = next(i for i, x in enumerate(clocks) if i not in (3, 7) and self.MODES[i][0] % 100 != 6 and clocks.count(x) == 1)
        with patch.object(RootPoisoned, 'clock', clocks[q]):measured = self.measured(V.pair(cls=RootPoisoned), capacity=32, sample=16)
        body = d3g.assemble(measured);c = body['comparison'];pq = c['pricer']['quantities']
        refused = c['structural']['kinds']['world_refused'];self.assertEqual(refused['count'], 2);self.assertEqual([x['cell'] for x in refused['first']], [[q, 0], [q, 1]])
        self.assertEqual([x['bf16'] for x in refused['first']], ['ValueError: current public actor output refused/nonfinite'] * 2)
        self.assertEqual(c['counts']['priced_f32'] - c['counts']['priced_bf16'], 2);self.assertEqual(c['pricer']['counts']['worlds_d3_priced'], c['counts']['priced_bf16'])
        self.assertEqual(pq[d3g.P_ALL_LANES]['count'], 9 * c['counts']['priced_bf16']);self.assertEqual(c['pricer']['structural']['count'], 0)
        self.assertEqual(pq[d3g.P_ALL_LANES]['nonfinite'], 0);self.assertEqual(body['pricer_anchor']['flags'], [])
        prop = d3g.fold(earlier_report(self.clean), body['d_all'], None, body['p_run'], structural={k: e['count'] for k, e in c['structural']['kinds'].items()},
            pricer_anchor=body['pricer_anchor'])
        self.assertEqual(set(prop['flagged_bounds']), VALUE_BOUNDS | ROOT_BOUNDS);self.assertTrue(all(any('structural world_refused 2' in r for r in x) for x in prop['flagged_bounds'].values()))
        self.assertTrue(all('structural world_refused 2' in prop['sites'][site]['caveats'] for site in d3g.SITES));self.assertIsNotNone(prop['root_logit_all_geometries'])
        self.assertEqual(set(body['proposal']['flagged_bounds']), VALUE_BOUNDS | ROOT_BOUNDS)   # assemble's own fold, too

    def test_a_mis_sliced_champion_exceeds_the_anchor_bound(self):
        """D3 prices with a champion reading tokens 1..23 (both precisions): its f32 differs from the raw forward on the ladder's own layout, so the
        anchor flags the bounds as exceeded and the layout as unequal to the evaluator's view."""
        measured = self.measured(V.pair(cls=CE.Shifted), capacity=32, sample=32);a = d3g.anchor_section([c['anchor'] for c in measured['calls']])
        for key in (d3g.VALUE, d3g.LOGITS):
            with self.subTest(anchor=key):
                self.assertGreater(a['quantities'][key]['max'], v.ADAPTER_LOGIT);self.assertFalse(a['verified'][key])
                self.assertTrue(any('EXCEEDED' in x and x.startswith(key + ':') for x in a['flags']), a['flags'])
        self.assertGreater(a['findings']['cells_inputs_unequal_f32'], 0);self.assertGreater(a['findings']['roots_inputs_unequal_bf16'], 0)

    def run_main(self, loaded, earlier, argv=('--fixed-shapes',)):
        with tempfile.TemporaryDirectory() as directory, self.engine(), patch.object(g2, 'bank_chunk_requests', return_value=loaded), \
                patch.object(g2, 'load_champion', return_value=self.clean['bf16']), patch.object(v, 'champion_observer', return_value=self.observe), \
                patch.object(d3g, 'log', lambda *a, **k: None), patch.object(d3g, 'CHUNK', 64), patch.object(d3g, 'CAPACITY', 32):
            out, source = Path(directory) / 'out', Path(directory) / 'validation.json';source.write_text(json.dumps(earlier))
            code = d3g.main(['--out', str(out), '--d3-root', '/d3', '--chunk', '0', '--validation-json', str(source), '--sample', '32'] + list(argv) + CE.CHAMPION)
            return code, json.loads((out / 'd3_geometry.json').read_text()), (out / 'd3_geometry.md').read_text()

    def loaded(self, groups):
        return SimpleNamespace(provider=SimpleNamespace(executor=self.ex, materializer=self.tables, recipe=self.recipe), plan=SimpleNamespace(model=object(),
            historical_window=2), groups=groups, source=dict(SOURCE), frame_pricer=gate_pricer(self.ex))

    def test_main_writes_the_report_and_its_summary(self):
        """main at a 32-root split (the CPU fixture's): the run's capacity departs from the gate's 512 in the D3 split, the depth-2 pass and the
        earlier report's shapes, and each is flagged."""
        code, report, summary = self.run_main(self.loaded([list(self.v2_requests)]), earlier_report(self.clean))
        self.assertEqual((code, report['exit'], report['status'], report['schema']), (0, 0, 'MEASURED', d3g.SCHEMA))
        h = report['header'];self.assertEqual(h['champion'], dict(archive=CE.CHAMPION[1], archive_sha256=CE.CHAMPION[3], checkpoint_sha256=CE.CHAMPION[5]))
        self.assertIn('commit', h['tree']);self.assertEqual(h['roots'], len(self.v2_requests));self.assertEqual(h['options']['split_capacity'], 32)
        self.assertIn('bind_roots under the bf16 champion', h['binding']);self.assertEqual(h['rules'], v.plain(d3g.RULES))
        self.assertEqual(h['d3_options']['departures'], ['call split 32 != the gate\'s 512']);self.assertEqual(h['d3_options']['pricers']['f32']['actor_capacity'], 3)
        self.assertEqual(h['pass_options']['departures'], ['f32: capacity 32 != the gate\'s 512', 'bf16: capacity 32 != the gate\'s 512'])
        self.assertEqual([h['evaluators'][n]['precision'] for n in v.NAMES], ['float32', 'bfloat16']);self.assertEqual(report['calls'][0]['roots'], len(self.v2_requests))
        self.assertEqual(report['source_check']['flags'], ['capacity 512 != this run\'s 32']);self.assertIsNotNone(report['proposal']['root_logit_all_geometries'])
        self.assertIn(report['proposal']['root_logit_all_geometries_set_by'], ('root (c) tail', 'b.root_peer_logits', 'b.root_peer_logits_all_lanes'))
        self.assertEqual(report['anchor']['flags'], []);self.assertIn('# Champion 9375: D3 FramePricer and the depth-2 pass', summary)
        self.assertIn('## f32 anchor (sample)', summary);self.assertIn('- Options match the gate\'s: False (FLAGGED: call split 32', summary)

    def test_an_empty_run_exits_2(self):
        code, report, summary = self.run_main(self.loaded([]), earlier_report(self.clean, roots=0))
        self.assertEqual((code, report['exit'], report['status']), (2, 2, 'NOTHING_COMPARED'));self.assertIn('no D3 value', report['status_reasons'][0])
        self.assertEqual(report['comparison']['quantities'][d3g.VALUE]['count'], 0);self.assertIsNone(report['proposal']['value_bound_all_geometries'])
        self.assertIsNone(report['proposal']['root_logit_all_geometries']);self.assertIn('NOTHING_COMPARED, exit 2', summary)
        self.assertTrue(any('not verified' in x for x in report['anchor']['flags']))
        body = d3g.assemble(dict(calls=[], d3_options=d3g.d3_options({}), pass_options=d3g.pass_options({}), walk=None))
        self.assertEqual((body['status'], body['exit']), ('NOTHING_COMPARED', 2));d3g.markdown(v.plain(body))


# Synthetic D3 Results and frames ---------------------------------------------------------------------------------------------------------
def result(status, worlds):
    """A FramePricer-shaped Result: worlds is a list of a refusal reason (str) or dict(grid {(a, r): value}, requested, legal, logits, errors)."""
    reports = []
    for w in worlds:
        if isinstance(w, str):reports.append(dict(joint_values=None, reason=w));continue
        grid = [[w['grid'].get((a, r)) for r in range(9)] for a in range(9)]
        reports.append(dict(joint_values=grid, opponent_requested=w.get('requested', True), reply_legal=tuple(w.get('legal', (True, True) + (False,) * 7)),
            reply_logits=tuple(w.get('logits', (1., 2.) + (0.,) * 7)), errors=tuple(w.get('errors', ()))))
    return SimpleNamespace(status=status, world_reports=tuple(reports))


def request(q):
    return SimpleNamespace(plans=SimpleNamespace(root=SimpleNamespace(address=SimpleNamespace(stream=100 + q, frame=4, owner=q % 2))), root_legal=(True, True) + (False,) * 7)


class ComparisonTests(unittest.TestCase):
    def call(self):
        """Three roots: (0) both priced, small value and logit differences, one cell valued by f32 only; (1) world 1 refused by bf16 only, the
        root's status differs; (2) world 0's reply legal mask differs, world 1's opponent request differs (bf16's unrequested world keeps
        nonzero logits), a NaN bf16 value."""
        g = {(0, 0): .5, (0, 1): -.25, (1, 0): .125, (1, 1): 1.}
        f32 = [result('COMPLETE', [dict(grid=g), dict(grid=g)]), result('COMPLETE', [dict(grid=g), dict(grid=g)]), result('COMPLETE', [dict(grid=g), dict(grid=g)])]
        moved = {**g, (0, 0): .5 + 2 ** -8, (1, 1): 1. - 2 ** -6};lesser = {k: x for k, x in moved.items() if k != (0, 1)}
        bf16 = [result('COMPLETE', [dict(grid=moved, logits=(1.0625, 2.) + (0.,) * 7), dict(grid=lesser, errors=((0, 1, 'ValueError: current public actor output refused/nonfinite'),))]),
            result('REFUSED', [dict(grid=g), 'ValueError: current public actor output refused/nonfinite']),
            result('COMPLETE', [dict(grid=g, legal=(True,) * 3 + (False,) * 6, logits=(9., 2.) + (0.,) * 7), dict(grid={**g, (1, 0): float('nan')}, requested=False)])]
        return d3g.compare_call(dict(f32=f32, bf16=bf16), [request(q) for q in range(3)], call=4)

    def test_maxima_non_finite_and_every_structural_difference(self):
        c = self.call();k = c['findings']
        self.assertEqual({name: e['count'] for name, e in k.items()}, dict(root_status=1, world_refused=1, reply_legal=1, opponent_requested=1,
            unrequested_reply_logits_nonzero_bf16=1, joint_value_none_pattern=1, joint_error=1))
        self.assertEqual(k['world_refused']['first'][0]['cell'], [1, 1]);self.assertEqual(k['world_refused']['first'][0]['f32'], 'priced')
        self.assertEqual(k['joint_value_none_pattern']['first'][0]['cell'], [0, 1, 0, 1]);self.assertEqual(k['root_status']['first'][0]['address'], [101, 4, 1])
        self.assertEqual(k['world_refused']['first'][0]['call'], 4);self.assertEqual(k['unrequested_reply_logits_nonzero_bf16']['first'][0]['cell'], [2, 1])
        # values: every cell both runs value: 4 + 3 + 4 (root 1 world 0) + 4 + 4 (root 2) = 19, one of them NaN in bf16
        self.assertEqual((c['counts']['cells_compared'], c['counts']['priced_both'], c['counts']['requested_both']), (19, 5, 4))
        s = v.spread(c['values']);self.assertEqual((s['count'], s['nonfinite']), (19, 1));self.assertEqual(s['max'], 2 ** -6)
        # logits: legal in both on worlds both priced with the opponent requested in both (root 2 world 1 is not): 4 worlds x 2 lanes
        s = v.spread(c['logits']);self.assertEqual(s['count'], 8);self.assertEqual(s['max'], 8.)
        s = v.spread(c['all_lanes']);self.assertEqual((s['count'], s['max']), (5 * 9, 8.))   # every lane of every world both priced
        merged = d3g.merge_comparison([dict(call=4, comparison=c), dict(call=5, comparison=self.call())])
        self.assertEqual(merged['structural']['count'], 14);self.assertEqual(merged['structural']['kinds']['joint_error']['count'], 2)
        self.assertEqual(merged['quantities'][d3g.VALUE]['count'], 38);self.assertEqual(merged['quantities'][d3g.VALUE]['nonfinite'], 2)
        self.assertEqual(merged['quantities'][d3g.ALL_LANES]['count'], 90);self.assertEqual(merged['counts']['cells_compared'], 38)
        self.assertEqual(merged['pricer']['quantities'][d3g.P_VALUE]['count'], 0)   # no pricer part: an empty one
        with self.assertRaisesRegex(ValueError, 'one D3 Result per request'):d3g.compare_call(dict(f32=self.f32_one(), bf16=self.f32_one()), [request(0), request(1)])

    @staticmethod
    def f32_one():
        return [result('COMPLETE', [dict(grid={(0, 0): .5}), dict(grid={(0, 0): .5})])]

    def test_the_depth2_pass_at_bs_positions(self):
        """Two roots. (b)'s positions are the bf16 D3's: root 0's two worlds (requested, legal lanes 0-1) and root 1's world 0 (unrequested);
        root 1's world 1 (the bf16 D3 refused it, the f32 D3 did not) is outside, however far its frames differ. Cells: D3-priced and enumerated
        by both frames. The pass's f32 against D3's f32 at those positions, by kind of lane: an illegal lane 3e-4 apart is over 2e-4, flagged."""
        n = 2;shape = lambda *s: np.zeros((n, 2) + s, np.float32)
        def frame(lane5=0., lane0=0., cell=0., far=0.):
            logits = shape(9);logits[0, 0, 5] = lane5;logits[0, 0, 0] = lane0;logits[1, 1, 0] = far
            legal = np.zeros((n, 2, 9), bool);legal[:, :, :2] = True;requested = np.asarray([[True, True], [False, True]])
            status = np.zeros((n, 2, 9, 9), np.int8);status[0, 0, :3, :3] = 2;value = shape(9, 9);value[0, 0, 0, 0] = cell;value[0, 0, 2, 2] = far
            return SimpleNamespace(pre=np.asarray([[0, 0], [0, 1]], np.int32), root_peer_requested=requested, root_peer_legal=legal, root_peer_logits=logits,
                t1_status=status, t1_value=value)
        f32, bf16 = frame(), frame(lane5=.5, lane0=.01, cell=.002, far=9.);bf16.t1_status = bf16.t1_status.copy();bf16.t1_status[0, 0, 1, 0] = 0
        known = np.zeros((n, 2, 9, 9), bool);known[0, 0, :2, :2] = True   # (2, 2): a D3 error cell, no value
        def d3(priced, logits=None, value=None, known=known):
            legal = np.zeros((n, 2, 9), bool);legal[:, :, :2] = True
            return dict(d_priced=np.asarray(priced), d_known=known, d_value=np.where(known, 0. if value is None else value, np.nan).astype(np.float32),
                d_reply_logits=shape(9) if logits is None else logits, d_reply_legal=legal, d_opponent_requested=np.asarray([[True, True], [False, True]]))
        logits = shape(9);logits[0, 0, 5] = 3e-4;logits[0, 1, 0] = 1e-4;value = np.zeros((n, 2, 9, 9), np.float32);value[0, 0, 0, 0] = 1e-6
        lesser = known.copy();lesser[0, 0, 1, 1] = False   # a cell the f32 D3 does not value: outside the pass's anchor (no NaN difference)
        D3 = dict(bf16=d3([[True, True], [True, False]]), f32=d3([[True, True], [True, True]], logits, value, lesser))
        out = d3g.pricer_call(dict(f32=f32, bf16=bf16), D3, [request(0), request(1)], call=2)
        self.assertEqual({k: e['count'] for k, e in out['findings'].items()}, dict(pricer_t1_enumeration=1));self.assertEqual(out['findings']['pricer_t1_enumeration']['first'][0]['cell'], [0, 0, 1, 0])
        s = v.spread(out['values']);self.assertEqual((s['count'], s['max']), (3, np.float32(.002)))   # (1, 0) not enumerated by bf16; (2, 2) not D3-priced
        s = v.spread(out['all_lanes']);self.assertEqual((s['count'], s['max']), (3 * 9, .5))   # the illegal lane 5 counts; root 1 world 1 does not
        s = v.spread(out['logits']);self.assertEqual((s['count'], s['max']), (4, np.float32(.01)))   # precision_call's positions: pre 0 and requested, legal lanes
        self.assertEqual(out['counts'], dict(worlds_d3_priced=3, cells_d3_priced=4, cells_compared=3, cells_unenumerated_bf16=1, all_lanes_compared=27, legal_worlds=2,
            legal_lanes_compared=4, vs_d3_f32_t1_value=2, vs_d3_f32_legal_lanes=4, vs_d3_f32_illegal_lanes=14, vs_d3_f32_unrequested_worlds=9))
        vs = {k: v.spread(x) for k, x in out['vs_d3_f32'].items()};self.assertEqual(sum(x['nonfinite'] for x in vs.values()), 0)
        self.assertEqual({k: (x['count'], x.get('max')) for k, x in vs.items()}, dict(t1_value=(2, float(np.float32(1e-6))), legal_lanes=(4, float(np.float32(1e-4))),
            illegal_lanes=(14, float(np.float32(3e-4))), unrequested_worlds=(9, 0.)))
        pa = d3g.pricer_anchor_section(vs);self.assertEqual({k: x['within'] for k, x in pa['quantities'].items()}, dict(t1_value=True, legal_lanes=True, illegal_lanes=False,
            unrequested_worlds=True))
        self.assertEqual(len(pa['flags']), 1);self.assertTrue(pa['flags'][0].startswith('illegal_lanes: EXCEEDED: pass f32 vs D3 f32 max 0.0003 > 0.0002'))
        touched, neutral = d3g.caveats(None, pa);self.assertEqual(({k for k, x in touched.items() if x}, neutral), ({d3g.LANES_SITE}, []))

    def test_a_pass_kind_reaches_the_bounds_through_assemble(self):
        """A difference only the depth-2 pass shows (a cell the bf16 frame did not enumerate), D3's two runs and the pass premise clean: assemble
        hands the pass's kinds to the fold, and exactly the value bounds are flagged with it."""
        g = {(0, 0): .5, (0, 1): -.25, (1, 0): .125, (1, 1): 1.};same = [result('COMPLETE', [dict(grid=g), dict(grid=g)])]
        comparison = d3g.compare_call(dict(f32=same, bf16=same), [request(0)]);self.assertEqual(comparison['findings'], {})
        def frame():
            logits = np.zeros((1, 2, 9), np.float32);logits[..., :2] = (1., 2.);legal = np.zeros((1, 2, 9), bool);legal[..., :2] = True
            status, value = np.zeros((1, 2, 9, 9), np.int8), np.zeros((1, 2, 9, 9), np.float32)
            for (a, r), x in g.items():status[0, :, a, r] = 2;value[0, :, a, r] = x
            return SimpleNamespace(pre=np.zeros((1, 2), np.int32), root_peer_requested=np.ones((1, 2), bool), root_peer_legal=legal, root_peer_logits=logits,
                t1_status=status, t1_value=value)
        bf16 = frame();bf16.t1_status[0, 1, 1, 1] = 0
        pricer = d3g.pricer_call(dict(f32=frame(), bf16=bf16), comparison.pop('arrays'), [request(0)])
        self.assertEqual({k: e['count'] for k, e in pricer['findings'].items()}, dict(pricer_t1_enumeration=1))
        call = dict(call=0, roots=1, costs={f'{program}.{n}': {} for program in ('depth2_pass', 'd3') for n in v.NAMES}, comparison=comparison, pricer=pricer,
            anchor=dict(call=0, cells=dict(spec=np.zeros((0, 4), np.int64)), roots=dict(spec=np.zeros((0, 2), np.int64))))
        body = d3g.assemble(dict(calls=[call], d3_options=d3g.d3_options({}), pass_options=None, walk=None))
        self.assertEqual((body['comparison']['structural']['count'], body['pricer_anchor']['flags']), (0, []))
        self.assertEqual({k: e['count'] for k, e in body['comparison']['pricer']['structural']['kinds'].items()}, dict(pricer_t1_enumeration=1))
        self.assertEqual(set(body['proposal']['flagged_bounds']), VALUE_BOUNDS)
        self.assertTrue(all(x == ['b.t1_value: structural pricer_t1_enumeration 1'] for x in body['proposal']['flagged_bounds'].values()))

    def test_the_positions_are_check_ds(self):
        """g2.check_d's own numeric positions (the bf16 frame against the bf16 D3), cell for cell and lane for lane, on random Results and frames
        with refused worlds, D3 error cells, unrequested worlds and cells the frame did not enumerate (the reviewer's check)."""
        rng = np.random.default_rng(7);R = 40;worlds = []
        frame = dict(pre=np.full((R, 2), -1, np.int32), root_peer_requested=np.zeros((R, 2), bool), root_peer_legal=np.zeros((R, 2, 9), bool),
            root_peer_logits=np.zeros((R, 2, 9), np.float32), t1_status=np.zeros((R, 2, 9, 9), np.int8), t1_error=np.zeros((R, 2, 9, 9), np.int8), t1_value=np.zeros((R, 2, 9, 9), np.float32))
        for q in range(R):
            pair = []
            for w in range(2):
                if rng.random() < .2:pair.append('physical-materialization:none');continue
                req = rng.random() < .7;legal = rng.random(9) < .5;legal[0] |= not legal.any();logits = rng.normal(size=9).astype(np.float32) * req
                acts = rng.choice(9, size=rng.integers(1, 10), replace=False);replies = np.flatnonzero(legal) if req else [0];grid, errors = {}, []
                for a in acts:
                    for r in replies:
                        if rng.random() < .1:errors.append((int(a), int(r), 'ValueError: current public actor output refused/nonfinite'))
                        else:grid[(int(a), int(r))] = float(rng.normal())
                pair.append(dict(grid=grid, requested=req, legal=tuple(map(bool, legal)), logits=tuple(map(float, logits)), errors=tuple(errors)))
                frame['pre'][q, w] = 0;frame['root_peer_requested'][q, w] = req;frame['root_peer_legal'][q, w] = legal;frame['root_peer_logits'][q, w] = logits + 1e-3
                for (a, r), x in grid.items():
                    if rng.random() < .9:frame['t1_status'][q, w, a, r] = 2;frame['t1_value'][q, w, a, r] = x + 1e-3   # 10% not enumerated (k1 pruning)
                for a, r, _ in errors:frame['t1_status'][q, w, a, r] = 3;frame['t1_error'][q, w, a, r] = 4
            worlds.append(pair)
        bf16 = [result('COMPLETE', x) for x in worlds]
        f32 = [result('COMPLETE', [x if isinstance(x, str) else dict(x, grid={k: y + 5e-3 for k, y in x['grid'].items()}, logits=tuple(y + 5e-3 for y in x['logits'])) for x in pair])
            for pair in worlds]
        requests = [SimpleNamespace(**vars(request(q)), root_inputs={}) for q in range(R)]
        seen, numeric = [], g2._numeric
        def spy(cells, got, want, *a, **k):seen.append(sorted(map(tuple, cells)));return numeric(cells, got, want, *a, **k)
        with patch.object(g2, '_numeric', spy):found = g2.check_d(SimpleNamespace(**frame), bf16, requests, g2.Findings(), full_turn1=True)
        c = d3g.compare_call(dict(f32=f32, bf16=bf16), requests)
        frames = dict(bf16=SimpleNamespace(**frame), f32=SimpleNamespace(**{k: x + np.float32(2e-3) if x.dtype == np.float32 else x for k, x in frame.items()}))
        out = d3g.pricer_call(frames, c['arrays'], requests)
        self.assertGreater(found.values['t1_value']['compared'], 500);self.assertGreater(found.values['root_peer_logits']['compared'], 300)
        self.assertEqual(found.values['t1_value']['compared'], out['counts']['cells_compared'])
        self.assertEqual(found.values['root_peer_logits']['compared'], out['counts']['all_lanes_compared']);self.assertEqual(out['counts']['all_lanes_compared'], c['counts']['all_lanes_compared'])
        # the same cells and lanes, not only as many: every position check_d hands its float comparison, against the tool's masks
        G = c['arrays']['bf16'];cells = G['d_known'] & (frame['t1_status'] != 0)
        self.assertEqual(len(seen), 2);self.assertEqual(seen[0], sorted(map(tuple, np.argwhere(cells).tolist())))
        self.assertEqual(seen[1], sorted((q, w, lane) for q, w in np.argwhere(G['d_priced']).tolist() for lane in range(9)))


class DAllAndProposalTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.evaluators = V.pair()

    def d(self, value, logits, lanes=None, bad=0):
        return d3g.d_all({d3g.VALUE: dict(count=50, nonfinite=bad, max=value), d3g.LOGITS: dict(count=90, nonfinite=0, max=logits),
            d3g.ALL_LANES: dict(count=200, nonfinite=0, max=logits if lanes is None else lanes)})

    def p(self, value=.001, legal=P_ROOT_OPP - v.ADAPTER_LOGIT, lanes=None):
        return d3g.p_run({d3g.P_VALUE: dict(count=60, nonfinite=0, max=value), d3g.P_LOGITS: dict(count=80, nonfinite=0, max=legal),
            d3g.P_ALL_LANES: dict(count=300, nonfinite=0, max=legal if lanes is None else lanes)})

    def test_d_all_and_p_are_p_alls_construction(self):
        d = self.d(.004, .07, .09);self.assertAlmostEqual(d[d3g.VALUE]['max'], .004 + v.ADAPTER_VALUE);self.assertAlmostEqual(d[d3g.LOGITS]['max'], .07 + v.ADAPTER_LOGIT)
        self.assertAlmostEqual(d[d3g.ALL_LANES]['max'], .09 + v.ADAPTER_LOGIT);self.assertEqual(d[d3g.ALL_LANES]['f32_adapter_bound'], v.ADAPTER_LOGIT)
        self.assertEqual(d[d3g.VALUE], v.pricer_all(dict(values=dict(x=dict(count=50, nonfinite=0, max=.004))), 'x', 'value'))
        p = self.p(.002, .1, .3);self.assertAlmostEqual(p[d3g.P_VALUE]['max'], .002 + v.ADAPTER_VALUE);self.assertAlmostEqual(p[d3g.P_ALL_LANES]['max'], .3 + v.ADAPTER_LOGIT)
        self.assertEqual(d3g.d_all({}), {q: dict(count=0, nonfinite=0) for q in d3g.QUANTITIES});self.assertEqual(d3g.p_run(None), {q: dict(count=0, nonfinite=0) for q in d3g.PRICER_QUANTITIES})

    def check(self, earlier, **kw):
        return d3g.source_check(earlier, **dict(dict(source=SOURCE, evaluators=self.evaluators, roots=27), **kw))

    def test_the_fold_names_the_term_that_sets_each_bound(self):
        earlier = earlier_report(self.evaluators);root_c = max(P_ROOT_OPP + O_ROOT_OPP, 2 * P_ROOT_OPP);check = self.check(earlier)
        small = d3g.fold(earlier, self.d(.002, .05), check, self.p())   # D3 and the all-lane terms inside the earlier bounds
        self.assertEqual((small['flags'], small['margin'], small['matches_earlier']), ([], None, True))
        self.assertAlmostEqual(small['sites']['b.t1_value']['tail'], P_T1_VALUE + .002 + v.ADAPTER_VALUE);self.assertAlmostEqual(small['root_c']['tail'], root_c)
        self.assertAlmostEqual(small['sites']['b.root_peer_logits']['tail'], P_ROOT_OPP + .05 + v.ADAPTER_LOGIT)
        self.assertAlmostEqual(small['sites']['b.root_peer_logits_all_lanes']['tail'], P_ROOT_OPP + .05 + v.ADAPTER_LOGIT)
        self.assertEqual((small['value_bound_all_geometries'], small['value_bound_all_geometries_without_root'], small['near_tie_q_bound_all_geometries']), (.02538,) * 3)
        self.assertEqual(small['value_bound_all_geometries_set_by'], 'earlier all_positions value_bound')
        self.assertAlmostEqual(small['root_logit_all_geometries'], root_c);self.assertEqual(small['root_logit_all_geometries_set_by'], 'root (c) tail')
        self.assertEqual((small['turn1_logit_bound'], small['deeper_logit_bound']), (.50193, .68968))
        self.assertEqual(small['cross_checks']['t1_value']['used'], 'validation.json');self.assertLess(small['cross_checks']['root_peer_logits']['difference'], d3g.CROSS_CHECK)
        lanes = d3g.fold(earlier, self.d(.002, .05, lanes=.12), check, self.p(lanes=.15))   # the illegal lanes move more: the all-lane term sets it
        self.assertAlmostEqual(lanes['root_logit_all_geometries'], .15 + .12 + 2 * v.ADAPTER_LOGIT);self.assertEqual(lanes['root_logit_all_geometries_set_by'], 'b.root_peer_logits_all_lanes')
        legal = d3g.fold(earlier, self.d(.002, .15, lanes=.1), check, self.p())   # D3's legal lanes move most: the legal site sets it
        self.assertEqual(legal['root_logit_all_geometries_set_by'], 'b.root_peer_logits');self.assertAlmostEqual(legal['root_logit_all_geometries'], P_ROOT_OPP + .15 + v.ADAPTER_LOGIT)
        large = d3g.fold(earlier, self.d(.02, .05), check, self.p())   # D3's values above the earlier bound: b.t1_value sets it
        self.assertAlmostEqual(large['value_bound_all_geometries'], P_T1_VALUE + .02 + v.ADAPTER_VALUE);self.assertEqual(large['value_bound_all_geometries_set_by'], 'b.t1_value')
        self.assertAlmostEqual(large['near_tie_q_bound_all_geometries'], large['value_bound_all_geometries'])
        frame = d3g.fold(earlier, self.d(.002, .05), check, self.p(value=.015))   # this run's P(t1) at (b)'s cells is larger: used, flagged
        self.assertAlmostEqual(frame['sites']['b.t1_value']['tail'], .015 + v.ADAPTER_VALUE + .002 + v.ADAPTER_VALUE);self.assertEqual(frame['cross_checks']['t1_value']['used'], 'this run')
        self.assertTrue(any(x.startswith('P(t1 value): this run\'s pricer.t1_value') for x in frame['flags']), frame['flags'])
        moved = d3g.fold(earlier, self.d(.002, .05), check, self.p(legal=P_ROOT_OPP - v.ADAPTER_LOGIT + 1e-5))   # the legal-lane cross-check misses by 1e-5
        self.assertTrue(any(x.startswith('cross-check root opp_logits') for x in moved['flags']), moved['flags'])
        close = d3g.fold(earlier, self.d(.002, .05), check, self.p(legal=P_ROOT_OPP - v.ADAPTER_LOGIT + 5e-7));self.assertEqual(close['flags'], [])   # within 1e-6
        none = d3g.fold(earlier, self.d(.002, .05), check)   # no pricer measurement: no all-lane tail, no root bound, flagged
        self.assertIsNone(none['root_logit_all_geometries']);self.assertIn('b.root_peer_logits_all_lanes: P unmeasured', none['flags'])
        self.assertIsNotNone(none['value_bound_all_geometries']);self.assertIn('P(t1 value): this run\'s pricer.t1_value unmeasured: the earlier P_all only', none['flags'])

    def test_non_finite_p_t1_withholds_b_t1(self):
        """This run's pricer.t1_value with non-finite differences withholds b.t1_value, even when its finite max is under the earlier P_all
        (the reviewer's demo), and when every difference is non-finite."""
        earlier = earlier_report(self.evaluators);check = self.check(earlier)
        for label, m in (('some', dict(count=60, nonfinite=3, max=.001)), ('all', dict(count=60, nonfinite=60))):
            with self.subTest(non_finite=label):
                p = d3g.p_run({d3g.P_VALUE: m, d3g.P_LOGITS: dict(count=80, nonfinite=0, max=P_ROOT_OPP - v.ADAPTER_LOGIT), d3g.P_ALL_LANES: dict(count=300, nonfinite=0, max=.1)})
                prop = d3g.fold(earlier, self.d(.002, .05), check, p)
                self.assertIsNone(prop['sites']['b.t1_value']['tail']);self.assertIsNone(prop['value_bound_all_geometries']);self.assertIsNone(prop['near_tie_q_bound_all_geometries'])
                self.assertIn(f'P(t1 value): this run\'s pricer.t1_value has {m["nonfinite"]} non-finite differences at (b)\'s cells: no b.t1_value bound', prop['flags'])
                self.assertIn(f'b.t1_value: P has {m["nonfinite"]} non-finite differences', prop['flags']);self.assertEqual(prop['cross_checks']['t1_value']['used'], 'this run')
                self.assertIsNotNone(prop['root_logit_all_geometries'])

    def test_structural_differences_and_pass_premises_flag_their_sites_and_bounds(self):
        earlier = earlier_report(self.evaluators);check = self.check(earlier);d, p = self.d(.002, .05), self.p();T1, LEGAL, LANES = d3g.SITES
        every = VALUE_BOUNDS | ROOT_BOUNDS   # each case's sites and bounds are literal (a changed SITE_KINDS or BOUND_SITES must fail here)
        cases = dict(none_pattern=(dict(joint_value_none_pattern=2), {T1}, VALUE_BOUNDS), joint_error=(dict(joint_error=1), {T1}, VALUE_BOUNDS),
            enumeration=(dict(pricer_t1_enumeration=1), {T1}, VALUE_BOUNDS), refused=(dict(world_refused=3), {T1, LEGAL, LANES}, every),
            pre=(dict(pricer_pre=1), {T1, LEGAL, LANES}, every), requested=(dict(opponent_requested=1), {LEGAL, LANES}, ROOT_BOUNDS),
            legal=(dict(reply_legal=1), {LEGAL, LANES}, ROOT_BOUNDS), pricer_requested=(dict(pricer_root_peer_requested=2), {LEGAL, LANES}, ROOT_BOUNDS),
            unrequested=(dict(unrequested_reply_logits_nonzero_bf16=1), {LANES}, ROOT_BOUNDS), unknown=(dict(some_new_kind=4), {T1, LEGAL, LANES}, every))
        for name, (structural, sites, bounds) in cases.items():
            with self.subTest(kind=name):
                prop = d3g.fold(earlier, d, check, p, structural=structural);(kind, count), = structural.items()
                self.assertEqual({site for site, x in prop['sites'].items() if x['caveats']}, sites)
                self.assertEqual(set(prop['flagged_bounds']), bounds)
                self.assertTrue(all(any(f'structural {kind} {count}' in r for r in x) for x in prop['flagged_bounds'].values()))
                self.assertTrue(any(f.startswith(f'{site}: its (b) positions are not clean: structural {kind} {count}') for site in sites for f in prop['flags']))
                self.assertIsNotNone(prop['value_bound_all_geometries']);self.assertIsNotNone(prop['root_logit_all_geometries'])   # flagged, not withheld
        neutral = d3g.fold(earlier, d, check, p, structural=dict(root_status=1, refusal_reason=2, world_refused=0))
        self.assertEqual((neutral['flagged_bounds'], neutral['neutral_structural']), ({}, ['refusal_reason 2', 'root_status 1']))
        self.assertIn('structural differences touching no (b) position: refusal_reason 2, root_status 1', neutral['flags'])
        pa = d3g.pricer_anchor_section(dict(t1_value=v.spread(np.asarray([np.nan, 1e-6])), illegal_lanes=v.spread(np.asarray([1e-4]))))
        self.assertEqual({k: x['within'] for k, x in pa['quantities'].items()}, dict(t1_value=False, legal_lanes=None, illegal_lanes=True, unrequested_worlds=None))
        self.assertEqual(pa['flags'], ['t1_value: 1 non-finite pass f32 vs D3 f32 differences'])
        prop = d3g.fold(earlier, d, check, p, pricer_anchor=pa)
        self.assertEqual(set(prop['flagged_bounds']), VALUE_BOUNDS)
        self.assertTrue(prop['sites'][T1]['caveats'][0].startswith('pass premise t1_value'))

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
                p = d3g.fold(earlier, self.d(.002, .05), check, self.p());self.assertFalse(p['matches_earlier'])
                self.assertTrue(any(x.startswith('source: ') and message in x for x in p['flags']))
                self.assertIsNotNone(p['value_bound_all_geometries']);self.assertIsNotNone(p['root_logit_all_geometries'])   # still computed: flagged, not suppressed
        self.assertIn('the earlier pricer column does not describe the gate\'s shapes: capacity 256 != the gate\'s 512', self.check(earlier_report(self.evaluators, capacity=256))['flags'])
        bare = earlier_report(self.evaluators, levels=False);check = self.check(bare);p = d3g.fold(bare, self.d(.002, .05), check, self.p())
        self.assertIn('no geometry levels (P_all, O_sample) in the earlier report', check['flags'])
        self.assertIsNotNone(p['sites']['b.t1_value']['tail'])   # this run's P(t1) alone, flagged
        self.assertIsNone(p['sites']['b.root_peer_logits']['tail']);self.assertIsNone(p['root_logit_all_geometries'])
        self.assertIn('b.root_peer_logits: P unmeasured', p['flags']);self.assertIn('root (c) tail: no root level in the earlier report', p['flags'])
        self.assertIn('cross-check root opp_logits: unavailable (earlier -, this run 0.103685)', p['flags'])
        p = d3g.fold(clean, self.d(.002, .05, bad=2), self.check(clean), self.p())   # non-finite D3 differences: no value tail
        self.assertIsNone(p['value_bound_all_geometries']);self.assertIn('b.t1_value: D_all has 2 non-finite differences', p['flags']);self.assertIsNotNone(p['root_logit_all_geometries'])
        p = d3g.fold(clean, d3g.d_all({}), self.check(clean), self.p());self.assertIn('b.root_peer_logits_all_lanes: D_all unmeasured', p['flags'])
        deeper = earlier_report(self.evaluators);deeper['geometry']['levels']['depth3.turn1']['pricer_all']['value']['max'] = .02
        self.assertTrue(any('P_all(depth3.turn1 value)' in x for x in d3g.fold(deeper, self.d(.002, .05), self.check(deeper), self.p())['flags']))
        self.assertEqual(d3g.source_check(None, source=SOURCE, evaluators=self.evaluators, roots=0)['flags'], ['no earlier validation report'])
        self.assertIsNone(d3g.fold(None, self.d(.002, .05))['value_bound_all_geometries'])


class OptionsTests(unittest.TestCase):
    def test_the_d3_options_are_recorded_and_departures_flagged(self):
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

    @unittest.skipUnless(N.HAVE_PRICER, 'gpu_engine.device_depthn_pricer is not on this tree')
    def test_the_depth2_pass_options_are_the_gates_and_departures_flagged(self):
        pair = V.pair();passes = d3g.build_passes(gate.pricer_module(), 'ex', object(), pair)
        opts = d3g.pass_options(passes);self.assertEqual((opts['departures'], opts['matches_gate']), ([], True))
        self.assertEqual(opts['pricers']['bf16'], dict(d3g.GATE_PASS, evaluator=dict(kind='champion', identity=['champion', 'b' * 64, 'bfloat16'])))
        self.assertEqual([passes[n].forward for n in v.NAMES], [pair['f32'], pair['bf16']])
        self.assertEqual(d3g.pass_options(d3g.build_passes(gate.pricer_module(), 'ex', object(), pair, fixed_shapes=False))['departures'],
            ['f32: fixed_capacity False != the gate\'s True', 'bf16: fixed_capacity False != the gate\'s True'])
        module = gate.pricer_module()
        for kw, message in ((dict(block_lanes=729), 'block_lanes 729 != the gate\'s 2187'), (dict(k_own=3, k_opp=3), 'k_own [3] != the gate\'s [9]'),
                (dict(k1_opp=3), 'k1_opp 3 != the gate\'s 9'), (dict(actor_capacity=33), 'actor_capacity 33 != the gate\'s 1'), (dict(digest=False), 'digest False != the gate\'s True')):
            with self.subTest(**kw):
                args = dict(dict(capacity=512, depth=2, k_own=9, k_opp=9, block_lanes=2187, selection='prior', digest=True, actor_capacity=1, fixed_capacity=True), **kw)
                bad = {n: module.DepthNPricer('ex', object(), forward=pair[n], **args) for n in v.NAMES};opts = d3g.pass_options(bad)
                self.assertFalse(opts['matches_gate']);self.assertIn(f'f32: {message}', opts['departures']);self.assertIn(f'bf16: {message}', opts['departures'])
        opts = d3g.pass_options({'f32': passes['bf16'], 'bf16': passes['f32']});self.assertEqual(len(opts['departures']), 2)
        self.assertTrue(all('is not the champion at' in x for x in opts['departures']))

    @unittest.skipUnless(N.HAVE_PRICER, 'gpu_engine.device_depthn_pricer is not on this tree')
    def test_the_gates_cli_defaults_are_read_and_a_k1_other_than_9_flagged(self):
        defaults = d3g.gate_defaults();self.assertEqual(defaults, {k: d3g.GATE_PASS[k] for k in d3g.GATE_CLI})   # d3_depthn_gate's own parser
        passes = d3g.build_passes(gate.pricer_module(), 'ex', object(), V.pair());opts = d3g.pass_options(passes)
        self.assertEqual((opts['gate_cli_defaults'], opts['departures']), (defaults, []))
        for k, x in (('k1_own', 3), ('k1_opp', 5), ('capacity', 256), ('block_lanes', 729)):
            with self.subTest(**{k: x}):
                opts = d3g.pass_options(passes, defaults=dict(defaults, **{k: x}))
                self.assertEqual(opts['departures'], [f'the gate\'s --{k.replace("_", "-")} default {x} != {d3g.GATE_PASS[k]}, the value measured here']);self.assertFalse(opts['matches_gate'])

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
