"""Gap 2's validation tool (gpu_engine.champion_leaf_validation) on CPU.

The depth-N fixture's fake engine with a ZERO-SUM V2 layout (token 0's value feature centred, plus an outcome hint one step before the
fixture's decisive terminals, negated on the opponent's view; feature 5 the same value from seat 0's side, for a fixed-side critic), a
readable tiny champion (the real champion.forward with inert layers: its value is tanh of one feature), the fixture's independent
champion-layout leaf (built from the fake engine's features, never from the 29-token tensor or ChampionEvaluator.view) and a V2 stand-in
head. The real DepthNPricer prices; the tool walks, judges and reports:
- the clean champion passes the adapter check (every level exact to 1e-5, one row per root) and all three perspective checks;
- a sign-flipped evaluator FAILs the perspective check (and the adapter check); a mis-sliced one FAILs adapter-vs-raw; a misplaced digest
  FAILs placement; a fixed-side critic FAILs observer-relativity; a globally inverted champion without a V2 head and with too few
  near-terminal parents exits 2 (sign not established);
- the bf16 evaluator under test is gated on its own: check_pair refuses a mismatched pair, and with that guard bypassed a bf16-only
  mis-slice or sign flip still FAILs (its view, its digests, gross divergence from f32, its owner's view); a bf16 that drifts 0.15 WARNs;
- f32 against f32 differs by nothing (two calls); an empty run exits 2 and renders;
- the verdict rules on synthetic numbers (subgroup significance, rollup, seats, near-terminal per seat, precision, pooling); the command
  line parses and refuses; main writes validation.json and validation.md.
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

import jax
import jax.numpy as j
import numpy as np

from pokezero.v5_model import TOKEN_ROLES
from gpu_engine import champion, champion_evaluator as ce, champion_leaf_validation as v, d3_depth2_gate as g2, device_public_teacher as serial, v5_model
from tests import test_gpu_engine_champion_evaluator as CE, test_gpu_engine_depthn_oracle as N

CENTRE, GAIN, HINT = .06, 4., .5
SEAT0 = 5   # the feature holding the value from seat 0's side, whatever the observer (a fixed-side critic reads it)


def features(state, owner, leaf):
    """The fake engine's feature row [n,1,F]: feature 0 a zero-sum value from `owner`'s side, (value - CENTRE) x GAIN plus HINT one step
    before the fixture's decisive terminals (won: modes 1/20 before the node, 21 before the leaf; lost: 9 before the leaf); feature 5 the
    same value from seat 0's side."""
    x = leaf.inputs['numeric_features'][:, :, 0, :];b, shift, turn = state.mode % 100, state.mode // 100, state.turn
    hint = (j.where(((b == 1) | (b == 20)) & (turn == shift), 1., 0.) + j.where((b == 21) & (turn == 1 + shift), 1., 0.)
        - j.where((b == 9) & (turn == 1 + shift), 1., 0.))
    side = j.where(j.asarray(owner) == state.owner, 1., -1.);seat0 = j.where(j.asarray(owner) == 0, 1., -1.)
    x = x.at[..., 0].set(((x[..., 0] - CENTRE) * GAIN + HINT * hint[:, None]) * side[:, None])
    x = j.pad(x, ((0, 0), (0, 0), (0, CE.F - x.shape[-1])))
    return x.at[..., SEAT0].set(x[..., 0] * seat0[:, None])


def zero_sum_layout(make):
    """The fake engine with V2-layout encoder inputs over `features` (as CE.champion_layout: token t = the row x SCALES[t])."""
    def build(**kw):
        base = make(**kw)
        def observe(state, public, owner, mask):
            leaf = base.observe(state, public, owner, mask);row = features(state, owner, leaf);n = row.shape[0]
            return leaf._replace(inputs=dict(numeric_features=row[:, :, None, :] * j.asarray(CE.SCALES)[None, None, :, None],
                categorical_ids=j.zeros((n, 1, CE.V2_TOKENS, CE.CATEGORICAL), j.int32), token_type_ids=j.broadcast_to(j.asarray(TOKEN_ROLES, j.int32), (n, 1, CE.V2_TOKENS)),
                attention_mask=j.ones((n, 1, CE.V2_TOKENS), bool), history_mask=j.ones((n, 1), bool)))
        ex = replace(base, observe=observe, joint=None)
        return replace(ex, joint=serial._make_joint(ex))
    return build


def legacy_observer(base):
    """The fixture's champion-layout leaf (the tool's `observe`), built straight from the fake engine's features: 23 tokens of the row x
    SCALES[:23], categories 0, the v4 roles, every token attended; the leaf's own error flag."""
    def observe(state, public, seat, mask):
        leaf = base.observe(state, public, seat, mask);row = features(state, seat, leaf)[:, 0];n = row.shape[0]
        return leaf._replace(inputs=dict(numeric=row[:, None, :] * j.asarray(CE.SCALES[:23])[None, :, None], categorical=j.zeros((n, 23, CE.CATEGORICAL), j.int32),
            token_types=j.broadcast_to(j.asarray(TOKEN_ROLES[:23], j.int32), (n, 23)), attention=j.ones((n, 23), bool)))
    return observe


def readable_champion(precision='float32', gain=2., channel=0, cls=ce.ChampionEvaluator):
    """The real champion.forward with inert layers (zero norms, so attention and feed-forward add nothing): width 0 carries feature
    `channel` (0: the observer's side; SEAT0: seat 0's side), so the value is tanh(gain x the mean over tokens of it x SCALES[t]); the
    policy reads features 0, 2 and 4 on the action tokens."""
    c = CE.tiny_config();p = {k: np.zeros(s, np.float32) for k, s in champion.shapes(c).items()}
    p['numeric_projection.weight'][0, channel] = 1.;p['numeric_projection.weight'][1, 2] = .25;p['numeric_projection.weight'][1, 4] = .125
    p['value_head.weight'][0, 0] = gain;p['policy_head.weight'][0, :2] = (.5, 1.)
    meta = dict(weights_archive_sha256='b' * 64, checkpoint_sha256=CE.CHECKPOINT, parameter_count=int(sum(x.size for x in p.values())))
    return cls({k: j.asarray(x) for k, x in p.items()}, c, meta, precision)


def pair(**kw):
    """One readable champion at both precisions, sharing one parameter tree (as main builds them)."""
    f32 = readable_champion('float32', **kw);return {'f32': f32, 'bf16': replace(f32, precision='bfloat16')}


def v2_reference(gain=3.):
    """A V2 stand-in head on the V2 layout: tanh(gain x token 0's feature 0), zero logits, valid where finite."""
    def forward(params, inputs):
        x = inputs['numeric_features'][:, 0, 0, 0]
        return v5_model.Output(j.zeros((x.shape[0], 9), j.float32), j.tanh(params['gain'] * x), None, j.isfinite(x))
    return SimpleNamespace(params={'gain': j.asarray(gain, j.float32)}, forward=jax.jit(forward), header=dict(stand_in=True))


class Shrunk(ce.ChampionEvaluator):
    """A bf16-only drift that stays sign-true: the value x 0.85 at bfloat16 (|diff| up to 0.15: above the 0.05 WARN, below the 0.25 FAIL)."""
    def __call__(self, params, inputs):
        out = super().__call__(params, inputs);return out._replace(value=out.value * .85) if self.precision == 'bfloat16' else out


class DigestRolled:
    """A pricer whose leaf digests are rolled one opponent slot (values untouched): a placement the digests no longer certify."""
    def __init__(self, pricer):self.pricer = pricer
    def __getattr__(self, name):return getattr(self.pricer, name)
    def price(self, *args, **kw):
        frame = self.pricer.price(*args, **kw);return replace(frame, leaf_digest=np.roll(np.asarray(frame.leaf_digest), 1, axis=-2))


@unittest.skipUnless(N.HAVE_PRICER, 'gpu_engine.device_depthn_pricer / depthn_backups are not on this tree')
class ValidationTests(N.FixtureN, unittest.TestCase):
    """The fixture's requests on the zero-sum layout (request 7's root inputs shifted: pre code 3), bound to the clean champion at both
    precisions by the tool's bind (g2.bind_evaluator_roots); every validation prices with the real DepthNPricer at depth 2 (full width)
    and depth 3 (prior top-2), the V2 forward patched to raise."""
    make = staticmethod(zero_sum_layout(N.executor_n))
    PRE_REFUSED = None

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        requests = list(cls.requests);r = requests[7]
        requests[7] = replace(r, root_inputs=dict(r.root_inputs, numeric_features=r.root_inputs['numeric_features'].at[..., 5, :].add(1.)))
        cls.v2_requests = tuple(requests);cls.observe = staticmethod(legacy_observer(N.executor_n()))   # a plain function, not a method
        cls.clean = pair()
        cls.bound, cls.dropped, cls.root_values = v.bind([list(cls.v2_requests)], cls.clean);cls.reports = {}
        cls.few = {name: [g[:6] for g in groups] for name, groups in cls.bound.items()}

    @contextlib.contextmanager
    def engine(self):
        with super().engine(), patch.object(v5_model, 'forward', side_effect=CE.no_v2):yield

    def validation(self, evaluators, *, requests=None, key=None, reference=True, wrap=None, **kw):
        if key is not None and key in self.reports:return self.reports[key]
        build = v.build_pricers
        def built(*a, **k):
            pricers = build(*a, **k);return pricers if wrap is None else {x: wrap(p) for x, p in pricers.items()}
        with self.engine(), patch.object(v, 'log', lambda *a, **k: None), patch.object(v, 'build_pricers', built):
            report = v.validate(requests or self.bound, evaluators=evaluators, executor=self.ex, model=object(), materializer=self.tables, recipe=self.recipe,
                observe=self.observe, reference=v2_reference() if reference else None, **dict(dict(capacity=32, depth3_k=2, sample=64, chunk=64), **kw))
        if key is not None:self.reports[key] = report
        return report

    def test_bind_keeps_the_roots_both_evaluators_bound(self):
        self.assertEqual(self.dropped, {'f32': [], 'bf16': []});self.assertEqual([len(g) for g in self.bound['f32']], [len(self.v2_requests)])
        for a, b in zip(self.bound['f32'][0], self.bound['bf16'][0]):
            self.assertEqual(v.address(a), v.address(b));self.assertEqual((a.learner_actor, a.opponent_actor, b.learner_actor, b.opponent_actor), (0, 0, 0, 0))
        self.assertNotEqual([r.root_logits for r in self.bound['f32'][0]], [r.root_logits for r in self.bound['bf16'][0]])   # each its own prior

    def test_the_clean_champion_passes_the_adapter_and_perspective_checks(self):
        report = self.validation(self.clean, key='clean');c = report['checks'];a = c['adapter'];p = c['perspective']
        self.assertEqual(a['status'], 'PASS', a['reasons']);self.assertIn(report['status'], ('PASS', 'WARN'));self.assertEqual(report['exit'], 0)
        for label in ('root', 'depth2.turn1', 'depth2.leaf', 'depth3.turn1', 'depth3.ply2', 'depth3.leaf'):
            with self.subTest(group=label):
                g = a['groups'][label]
                for name in v.NAMES:
                    e = g[name];self.assertGreater(e['compared'], 0);self.assertEqual((e['inputs_unequal'], e['walk_disagrees'], e.get('digest_mismatch', 0)), (0, 0, 0))
                e = g['f32'];self.assertLessEqual(e['value']['max'], v.ADAPTER_VALUE)
                if label in ('root', 'depth2.turn1', 'depth3.turn1', 'depth3.ply2'):self.assertGreater(e['own_logits']['count'], 0);self.assertLessEqual(e['own_logits']['max'], v.ADAPTER_LOGIT)
        o = a['observed'];self.assertEqual((o['f32_value_bound'], o['f32_logit_bound']), (v.ADAPTER_VALUE, v.ADAPTER_LOGIT));self.assertLessEqual(o['f32_value_max'], v.ADAPTER_VALUE)
        self.assertGreater(o['bf16_value_max'], 0.)
        roots = {q for q, w in self.root_rows()};self.assertEqual(a['groups']['root']['walked'], len(roots))   # one row per root (request 7 pre-refused)
        self.assertGreater(a['groups']['root']['f32']['opp_logits']['count'], 0)
        for name in ('v2', 'seats', 'near_terminal'):self.assertEqual(p[name]['status'], 'PASS', (name, p[name]['reasons']))
        groups = p['v2']['by_evaluator']['bf16']['groups'];self.assertGreater(groups['pooled']['pearson'], .9);self.assertGreater(groups['pooled']['count'], 100)
        self.assertGreater(min(groups['seat0']['count'], groups['seat1']['count']), v.MIN_ROWS)
        self.assertLess(p['seats']['pooled']['pearson'], -.9);self.assertEqual(p['seats']['owner_view_f32']['owner_closer'], 1.);self.assertEqual(p['seats']['owner_view_bf16']['owner_closer'], 1.)
        nt = p['near_terminal']['sources'];self.assertGreaterEqual(nt['pooled']['unanimous'], v.MIN_ROWS)
        self.assertEqual(set(nt), {'root', 'depth2.turn1', 'depth3.ply2', 'pooled'});self.assertTrue(all(nt[k]['unanimous'] > 0 for k in ('root', 'depth2.turn1', 'depth3.ply2')))
        self.assertEqual(nt['pooled']['f32']['agree'], 1.);self.assertEqual(nt['pooled']['v2']['agree'], 1.)
        self.assertEqual(nt['root']['unanimous'], 1)   # one root (mode 20) wins at turn 1: ONE parent, both worlds' children pooled
        pr = c['precision'];expected = 'WARN' if pr['max_value_diff']['value'] > v.PRECISION_VALUE_WARN or any(
            d['share'] is not None and d['share'] < v.PRECISION_DECISION_WARN for d in pr['decisions'].values()) else 'PASS'
        self.assertEqual(pr['status'], expected, pr['reasons']);self.assertEqual(pr['sign_disagreements']['pooled']['disagree'], 0)
        for name in ('root', 'depth2.turn1', 'depth2.leaf', 'depth3.ply2', 'depth3.leaf'):self.assertGreater(pr['values'][name]['count'], 0, name)
        self.assertGreater(pr['values']['depth2.leaf']['max'], 0.)   # bf16 really rounds
        self.assertEqual(set(pr['decisions']), {'depth2', 'depth3'});self.assertGreater(pr['decisions']['depth2']['labeled_both'], 0)
        self.assertEqual(set(c['scale']['levels']), set(v.UNIFORM));self.assertIn('depth2.leaf', c['scale']['frame_population'])
        self.assertEqual(report['calls'][0]['rebound_equal'], {'f32': True, 'bf16': True});self.assertEqual(report['calls'][0]['empty_levels'], [])
        self.assertGreater(report['walk']['rows'], 0);self.assertEqual(report['pricers']['depth3.bf16']['k_own'], [2, 2])

    def root_rows(self):
        return [(q, w) for q in range(len(self.v2_requests)) for w in (0, 1) if q != 7 and (q, w) not in self.MISSING and (w == 0 or (q, 0) in self.MISSING)]

    def test_a_sign_flipped_evaluator_fails_the_perspective_check(self):
        """The pricers price with a champion whose value is negated (the raw forward is untouched): the V2 agreement mirrors, the near-terminal
        signs invert and the pricer value sits on the opponent's side of the zero-sum views, each a perspective FAIL; the adapter check
        fails too. The tool's bind ran on the clean champion, so the first-root check (which refuses this evaluator) is not what catches it."""
        report = self.validation(pair(cls=CE.Negated));p = report['checks']['perspective']
        self.assertEqual((report['status'], report['exit'], p['status'], p['v2']['status']), ('FAIL', 1, 'FAIL', 'FAIL'), p['v2']['reasons'])
        self.assertTrue(any('perspective suspect' in r for r in p['v2']['reasons']), p['v2']['reasons'])
        self.assertLess(p['v2']['by_evaluator']['f32']['groups']['pooled']['pearson'], -.9)
        self.assertTrue(p['v2']['by_evaluator']['raw']['status'].startswith('PASS'))   # the raw forward is the champion's own: not flipped
        self.assertEqual(p['near_terminal']['status'], 'FAIL');self.assertEqual(p['seats']['status'], 'FAIL')
        self.assertTrue(any('bf16 pricer value is nearer the opponent' in r for r in p['seats']['reasons']), p['seats']['reasons'])
        self.assertEqual(report['checks']['adapter']['status'], 'FAIL')

    def test_a_mis_sliced_evaluator_fails_adapter_vs_raw(self):
        report = self.validation(pair(cls=CE.Shifted));a = report['checks']['adapter']
        self.assertEqual((a['status'], report['status'], report['exit']), ('FAIL', 'FAIL', 1))
        self.assertTrue(all(g[name]['inputs_unequal'] > 0 for g in a['groups'].values() for name in v.NAMES))
        self.assertGreater(a['groups']['depth2.leaf']['f32']['value']['max'], v.ADAPTER_VALUE)
        self.assertTrue(any('differs from the f32 evaluator\'s view' in r for r in a['reasons']))
        self.assertTrue(any('differs from the bf16 evaluator\'s view' in r for r in a['reasons']))

    def test_check_pair_refuses_two_different_evaluators(self):
        clean = self.clean
        self.assertTrue(v.check_pair(clean));self.assertTrue(v.check_pair(pair()))   # bit-equal parameters in another tree pass
        f32 = clean['f32'];negated = CE.Negated(f32.params, f32.config, f32.meta, 'bfloat16')
        cases = dict(type=negated, precision=replace(f32, precision='float32'), params=replace(f32, precision='bfloat16', params=dict(f32.params, **{
            'value_head.weight': -f32.params['value_head.weight']})), archive=replace(f32, precision='bfloat16', meta=dict(f32.meta, weights_archive_sha256='c' * 64)))
        for name, other in cases.items():
            with self.subTest(differs=name), self.assertRaises(ValueError):v.check_pair({'f32': f32, 'bf16': other})
        with self.assertRaisesRegex(ValueError, 'one evaluator type'):self.validation({'f32': f32, 'bf16': negated})
        self.assertTrue(v.check_pair({'f32': f32, 'bf16': clean['f32']}, tested='float32'))

    def test_the_bf16_evaluator_alone_is_gated(self):
        """With check_pair bypassed, a bf16-only mis-slice or sign flip against a clean f32 still FAILs: the bf16 view (inputs unequal), its
        digests (placement), its gross divergence from f32 (precision: values and confident signs) and its owner's view (seats)."""
        f32 = self.clean['f32']
        for cls in (CE.Shifted, CE.Negated):
            with self.subTest(bf16=cls.__name__), patch.object(v, 'check_pair', return_value=True):
                report = self.validation({'f32': f32, 'bf16': cls(f32.params, f32.config, f32.meta, 'bfloat16')});c = report['checks']
                self.assertEqual((report['status'], report['exit']), ('FAIL', 1));self.assertEqual(c['adapter']['groups']['depth2.leaf']['f32']['inputs_unequal'], 0)
                self.assertEqual(c['precision']['status'], 'FAIL', c['precision']['reasons']);self.assertEqual(c['adapter']['status'], 'FAIL')
                self.assertTrue(any('gross bf16 divergence' in r for r in c['precision']['reasons']))
                if cls is CE.Shifted:
                    self.assertTrue(any('differs from the bf16 evaluator\'s view' in r for r in c['adapter']['reasons']))
                    self.assertTrue(any('bf16 frame\'s (placement)' in r for r in c['adapter']['reasons']))
                else:
                    self.assertGreater(c['precision']['sign_disagreements']['pooled']['share'], .9)
                    self.assertTrue(any('bf16 pricer value is nearer the opponent' in r for r in c['perspective']['seats']['reasons']))

    def test_a_bf16_drift_warns_without_failing(self):
        """A bf16 that moves values by up to 0.15 but keeps every sign: precision and adapter WARN (above 0.05), nothing FAILs, exit 0."""
        report = self.validation(pair(cls=Shrunk));c = report['checks']
        self.assertEqual((report['status'], report['exit']), ('WARN', 0), report['status_reasons'])
        self.assertEqual(c['precision']['status'], 'WARN', c['precision']['reasons']);self.assertEqual(c['adapter']['status'], 'WARN', c['adapter']['reasons'])
        self.assertTrue(v.PRECISION_VALUE_WARN < c['precision']['max_value_diff']['value'] < v.BF16_GROSS_VALUE)
        self.assertEqual(c['precision']['sign_disagreements']['pooled']['disagree'], 0);self.assertEqual(c['perspective']['seats']['status'], 'PASS')

    def test_a_misplaced_digest_fails_placement(self):
        report = self.validation(self.clean, wrap=DigestRolled);a = report['checks']['adapter']
        self.assertEqual((a['status'], report['exit']), ('FAIL', 1));self.assertLessEqual(a['observed']['f32_value_max'], v.ADAPTER_VALUE)   # values still in place
        self.assertTrue(a['reasons'] and all('(placement)' in r for r in a['reasons']), a['reasons'])
        self.assertGreater(a['groups']['depth2.leaf']['f32']['digest_mismatch'], 0);self.assertGreater(a['groups']['depth2.leaf']['bf16']['digest_mismatch'], 0)

    def test_a_fixed_side_critic_fails_observer_relativity(self):
        """A champion that values every view from seat 0's side: its owner's and opponent's views agree (Pearson about +1), so the seats
        check FAILs, and its agreement with the learner-side V2 head inverts at seat 1."""
        report = self.validation(pair(channel=SEAT0));p = report['checks']['perspective']
        self.assertEqual((report['status'], p['seats']['status']), ('FAIL', 'FAIL'));self.assertGreater(p['seats']['pooled']['pearson'], .9)
        self.assertTrue(any('not observer-relative' in r for r in p['seats']['reasons']));self.assertEqual(p['v2']['status'], 'FAIL')
        self.assertTrue(any('seat1' in r for r in p['v2']['reasons']), p['v2']['reasons'])

    def test_an_unverified_sign_exits_2(self):
        """A globally inverted champion with no V2 head and too few near-terminal parents (6 roots): the adapter and seats checks pass (the
        raw forward is inverted too), the V2 check is SKIPPED and the near-terminal check INCONCLUSIVE: sign not established, exit 2."""
        report = self.validation(pair(gain=-2.), requests=self.few, reference=False);p = report['checks']['perspective']
        self.assertEqual((p['v2']['status'], p['near_terminal']['status'], report['checks']['adapter']['status']), ('SKIPPED', 'INCONCLUSIVE', 'PASS'))
        self.assertEqual((report['status'], report['passed'], report['exit']), ('INCONCLUSIVE', False, 2));self.assertIn('sign not established', report['status_reasons'][0])

    def test_f32_against_f32_differs_by_nothing(self):
        f32 = {'f32': self.clean['f32'], 'bf16': readable_champion('float32')}
        report = self.validation(f32, requests={'f32': self.bound['f32'], 'bf16': self.bound['f32']}, capacity=16, tested_precision='float32')
        pr = report['checks']['precision']
        self.assertEqual(pr['status'], 'PASS', pr['reasons']);self.assertEqual([c['roots'] for c in report['calls']], [16, len(self.v2_requests) - 16])   # two calls, merged
        self.assertEqual(report['checks']['adapter']['status'], 'PASS', report['checks']['adapter']['reasons'])
        for part in ('values', 'logits'):
            for k, s in pr[part].items():
                with self.subTest(part=part, quantity=k):self.assertEqual((s.get('max', 0.), s['nonfinite']), (0., 0))
        self.assertGreater(pr['values']['depth3.leaf']['count'], 0);self.assertGreater(pr['logits']['depth3.ply2.own']['count'], 0)
        self.assertTrue(all(x['changed'] == 0 for x in pr['argmax_changes'].values()));self.assertTrue(all(n == 0 for n in pr['status_differs'].values()))
        self.assertTrue(all(e['f32'] == e['bf16'] == e['shared'] for e in pr['enumeration'].values()))
        for family, d in pr['decisions'].items():self.assertGreater(d['labeled_both'], 0, family);self.assertEqual(d['share'], 1., family)

    def run_main(self, loaded, argv=()):
        with tempfile.TemporaryDirectory() as directory, self.engine(), patch.object(g2, 'bank_chunk_requests', return_value=loaded), \
                patch.object(g2, 'load_champion', return_value=self.clean['bf16']), patch.object(v, 'champion_observer', return_value=self.observe), \
                patch.object(v, 'load_reference', return_value=v2_reference()), patch.object(v, 'log', lambda *a, **k: None):
            out = Path(directory) / 'out'
            code = v.main(['--out', str(out), '--d3-root', '/d3', '--chunk', '0', '--capacity', '32', '--sample', '32', '--walk-chunk', '32', '--depth3-k', '2']
                + list(argv) + CE.CHAMPION + CE.HEAD)
            return code, json.loads((out / 'validation.json').read_text()), (out / 'validation.md').read_text()

    def loaded(self, groups):
        return SimpleNamespace(provider=SimpleNamespace(executor=self.ex, materializer=self.tables, recipe=self.recipe), plan=SimpleNamespace(model=object()),
            groups=groups, source=dict(fixture=True))

    def test_main_writes_the_report_and_its_summary(self):
        code, report, summary = self.run_main(self.loaded([list(self.v2_requests)]), ['--fixed-shapes'])
        self.assertEqual(code, 0, report['status']);self.assertIn(report['status'], ('PASS', 'WARN'));self.assertEqual((report['schema'], report['exit']), (v.SCHEMA, 0))
        self.assertEqual(report['header']['thresholds'], v.plain(v.THRESHOLDS));self.assertEqual(report['header']['roots'], len(self.v2_requests))
        self.assertEqual(report['header']['evaluators']['f32']['precision'], 'float32');self.assertEqual(report['header']['evaluators']['bf16']['precision'], 'bfloat16')
        self.assertEqual(report['checks']['adapter']['status'], 'PASS');self.assertEqual(report['walk']['stack_rows'], 64)
        self.assertTrue(all(x['fixed_capacity'] for x in report['pricers'].values()))
        self.assertIn(f'# Champion 9375 leaf validation: {report["status"]} (exit 0)', summary);self.assertIn('| 2 adapter vs raw | PASS |', summary)
        self.assertIn('Observed f32 max |dv|', summary)

    def test_an_empty_run_exits_2_and_renders(self):
        code, report, summary = self.run_main(self.loaded([]))
        self.assertEqual((code, report['status'], report['passed'], report['exit']), (2, 'INCONCLUSIVE', False, 2))
        self.assertEqual(report['status_reasons'], ['no root was validated']);self.assertIn('INCONCLUSIVE (exit 2)', summary)
        empty = v.judge([]);self.assertEqual((empty['status'], empty['passed'], empty['exit']), ('INCONCLUSIVE', False, 2));v.markdown(v.plain(empty))


def synthetic(n=60, *, value=None, peer=None, v2=None, seat=None, valued=True, v2_ok=True, label='depth2.leaf', ply=2, call=0):
    """One walked group of n rows with every field the judges read: the pricers' values equal the raw forward's, the opponent's view its
    negation, the V2 head the same values (each overridable)."""
    rng = np.random.default_rng(1);x = np.asarray(rng.uniform(-.9, .9, n) if value is None else value, np.float32);ones, zeros = np.ones(n, bool), np.zeros((n, 9), np.float32)
    r = dict(call=np.full(n, call), ply=np.full(n, ply), key=np.arange(n), seat=np.arange(n) % 2 if seat is None else np.asarray(seat), walk_ok=ones, peer_ok=ones,
        walk_digest=np.zeros((n, 2), np.uint32), raw_value=x, raw_logits=zeros, raw_peer_value=-x if peer is None else np.asarray(peer, np.float32), raw_peer_logits=zeros,
        v2_value=x if v2 is None else np.asarray(v2, np.float32), v2_ok=np.full(n, v2_ok))
    for name in v.NAMES:
        r.update({f'inputs_equal_{name}': ones, f'same_batch_value_{name}': x, f'same_batch_logits_{name}': zeros, f'found_{name}': ones, f'valued_{name}': np.full(n, valued),
            f'value_{name}': x, **{f'{s}_{k}_{name}': d for s in ('own', 'opp') for k, d in (('logits', zeros), ('legal', np.ones((n, 9), bool)), ('requested', ones))}})
    return {label: r}


def checks(precision='PASS', adapter='PASS', scale='PASS', v2='PASS', seats='PASS', near_terminal='PASS'):
    s = lambda x: dict(status=x, reasons=[])
    return dict(precision=s(precision), adapter=s(adapter), scale=s(scale), perspective=dict(v2=s(v2), seats=s(seats), near_terminal=s(near_terminal)))


class VerdictTests(unittest.TestCase):
    """The verdict rules on synthetic numbers (THRESHOLDS)."""
    def test_sign_verdict_tiers_and_subgroup_significance(self):
        g = lambda n, r, s, confident=0, sc=None: dict(count=n, pearson=r, same_sign=s, confident=confident, same_sign_confident=sc)
        self.assertEqual(v.sign_verdict(dict(pooled=g(500, .7, .85)))[0], 'PASS')
        self.assertEqual(v.sign_verdict(dict(pooled=g(500, .3, .7)))[0], 'WARN')   # weak, not inverted
        self.assertEqual(v.sign_verdict(dict(pooled=g(500, -.6, .2)))[0], 'FAIL')
        self.assertEqual(v.sign_verdict(dict(pooled=g(500, .7, .4)))[0], 'FAIL')   # same-sign below chance
        self.assertEqual(v.sign_verdict(dict(pooled=g(500, .7, .6, 100, .9)))[0], 'PASS')   # confident rows decide the sign
        self.assertEqual(v.sign_verdict(dict(pooled=g(10, .9, .9)))[0], 'INCONCLUSIVE')
        self.assertEqual(v.sign_verdict(dict(pooled=g(500, .6, .85), root=g(10, -1., 0.)))[0], 'PASS')   # a group under MIN_ROWS is not judged
        self.assertEqual(v.sign_verdict(dict(pooled=g(500, .6, .85), seat1=g(30, -.1, .45)))[0], 'PASS')   # a chance dip in a small subgroup
        status, reasons = v.sign_verdict(dict(pooled=g(500, .6, .85), seat1=g(250, -.5, .3)));self.assertEqual(status, 'FAIL');self.assertIn('seat1', reasons[0])
        self.assertEqual(v.sign_verdict(dict(pooled=g(500, .6, .85), seat1=g(30, -.9, .1)))[0], 'FAIL')   # an inversion clears the bound at 30 rows
        self.assertLess(v.fisher_upper(-.9, 30), 0.);self.assertGreater(v.fisher_upper(-.1, 30), 0.);self.assertLess(v.wilson_upper(.1, 30), .5)
        self.assertGreater(v.wilson_upper(.45, 30), .5);self.assertIsNone(v.fisher_upper(None, 30))

    def test_rollup(self):
        self.assertEqual(v.rollup(checks()), ('PASS', []))
        for k in ('precision', 'adapter', 'scale', 'v2', 'seats', 'near_terminal'):
            with self.subTest(failing=k):self.assertEqual(v.rollup(checks(**{k: 'FAIL'}))[0], 'FAIL')
        for kw in (dict(seats='INCONCLUSIVE'), dict(v2='SKIPPED'), dict(near_terminal='INCONCLUSIVE'), dict(scale='WARN'), dict(precision='INCONCLUSIVE')):
            with self.subTest(**kw):self.assertEqual(v.rollup(checks(**kw))[0], 'WARN')
        for kw in (dict(v2='SKIPPED', near_terminal='INCONCLUSIVE'), dict(v2='INCONCLUSIVE', near_terminal='INCONCLUSIVE'), dict(adapter='INCONCLUSIVE')):
            with self.subTest(**kw):self.assertEqual(v.rollup(checks(**kw))[0], 'INCONCLUSIVE')
        self.assertEqual(v.rollup(checks(v2='SKIPPED', near_terminal='FAIL'))[0], 'FAIL');self.assertEqual(v.EXIT, dict(PASS=0, WARN=0, FAIL=1, INCONCLUSIVE=2))

    def test_seats(self):
        self.assertEqual(v.seats_check(synthetic())['status'], 'PASS')
        self.assertEqual(v.seats_check(synthetic(n=20))['status'], 'INCONCLUSIVE')
        x = np.random.default_rng(2).uniform(-.9, .9, 60)
        fixed = v.seats_check(synthetic(value=x, peer=x + .0005));self.assertEqual(fixed['status'], 'FAIL');self.assertIn('not observer-relative', fixed['reasons'][0])
        weak = v.seats_check(synthetic(value=x, peer=-.2 * x + np.random.default_rng(3).normal(0, .3, 60)));self.assertEqual(weak['status'], 'WARN', weak['pooled'])
        r = synthetic(value=x);r['depth2.leaf']['value_bf16'] = -x;flipped = v.seats_check(r)   # the bf16 pricer reads the opponent's side
        self.assertEqual(flipped['status'], 'FAIL');self.assertIn('bf16 pricer value is nearer the opponent', flipped['reasons'][0])
        r = synthetic(value=x);r['depth2.leaf']['value_bf16'] = x + .2 * np.sign(-x)   # within the bf16 margin: still the owner's view
        self.assertEqual(v.seats_check(r)['status'], 'PASS')

    def test_adapter_and_v2_without_evidence(self):
        self.assertEqual(v.adapter_check(synthetic(valued=False))['status'], 'INCONCLUSIVE')
        self.assertEqual(v.adapter_check(synthetic())['status'], 'PASS')
        empty = v.adapter_check(synthetic(), expected=('root', 'depth2.leaf'));self.assertEqual(empty['status'], 'WARN');self.assertIn('root: a priced level', empty['reasons'][0])
        self.assertEqual(v.v2_check(synthetic(v2_ok=False))['status'], 'SKIPPED')
        self.assertEqual(v.v2_check(synthetic(n=200))['status'], 'PASS')
        one = v.v2_check(synthetic(n=200, seat=np.zeros(200, int)));self.assertEqual(one['status'], 'INCONCLUSIVE');self.assertIn('both learner seats', ' '.join(one['reasons']))
        x = np.random.default_rng(4).uniform(-.9, .9, 200);seat = np.arange(200) % 2
        inverted = v.v2_check(synthetic(value=x, v2=np.where(seat == 1, -x, x), seat=seat));self.assertEqual(inverted['status'], 'FAIL')   # seat 1 inverted

    def test_pooling_keeps_each_pricers_rows(self):
        a, b = synthetic(n=40, label='depth2.leaf'), synthetic(n=40, label='depth3.ply2');records = dict(a, **b)
        self.assertEqual(len(v._pooled(records, v.UNIFORM, ('raw_value',), by='state')['raw_value']), 40)   # the same 40 states
        self.assertEqual(len(v._pooled(records, v.UNIFORM, ('value_f32',), by='pricer')['value_f32']), 80)   # two pricers' values
        self.assertEqual(v.v2_check(records)['by_evaluator']['f32']['groups']['pooled']['count'], 80)

    def test_near_terminal_pooled_and_per_seat(self):
        def terminal(values, target, seat=None):
            n = len(target);plus = np.where(np.asarray(target) > 0, 2, 0);minus = np.where(np.asarray(target) < 0, 1, 0)
            return dict(root=dict(call=np.zeros(n, np.int64), seat=np.arange(n) % 2 if seat is None else seat, plus=plus, minus=minus,
                value_f32=np.asarray(values, np.float32), value_bf16=np.asarray(values, np.float32), v2_value=np.full(n, np.nan, np.float32)))
        target = np.where(np.arange(40) % 3 == 0, -1., 1.)
        self.assertEqual(v.near_terminal_check(terminal(.5 * target, target))['status'], 'PASS')
        self.assertEqual(v.near_terminal_check(terminal(-.5 * target, target))['status'], 'FAIL')
        mixed = np.where(np.arange(40) < 30, .5, -.5) * target;self.assertEqual(v.near_terminal_check(terminal(mixed, target))['status'], 'WARN')
        self.assertEqual(v.near_terminal_check(terminal(.5 * target[:10], target[:10]))['status'], 'INCONCLUSIVE')
        report = v.near_terminal_check(terminal(.5 * target, target));self.assertEqual(report['sources']['pooled']['f32']['pair_agree'], 1.)
        mixed = terminal(.5 * target, target);r = mixed['root'];r['plus'][:4], r['minus'][:4] = 1, 3   # four mixed parents: not unanimous, in the pairs
        s = v.near_terminal_check(mixed)['sources']['pooled'];self.assertEqual((s['unanimous'], s['parents_with_decisive_children']), (36, 40))
        rest = sum(2 if t > 0 else 1 for t in target[4:]);mixed_won = sum(1 if t > 0 else 3 for t in target[:4])   # a mixed parent wins its children of its sign
        self.assertEqual(s['f32']['agree'], 1.);self.assertAlmostEqual(s['f32']['pair_agree'], (mixed_won + rest) / (16 + rest))
        t = np.where(np.arange(160) % 3 == 0, -1., 1.);seat = (np.arange(160) >= 120).astype(int)   # 120 seat-0 parents right, 40 seat-1 parents inverted
        report = v.near_terminal_check(terminal(np.where(seat == 1, -.5, .5) * t, t, seat))
        self.assertGreater(report['sources']['pooled']['f32']['agree'], .5);self.assertEqual(report['status'], 'FAIL')
        self.assertTrue(any('seat1' in x for x in report['reasons']), report['reasons'])

    def test_precision_verdicts(self):
        def call(diff=0., flips=0, shared=10, confident=100):
            return dict(call=0, precision=dict(values=dict(leaf=np.full(5, diff)), logits={}, argmax={}, status_differs={}, terminal_differs={},
                enumeration=dict(leaf=[shared, shared, shared]), decisions={}, signs=dict(leaf=[confident, flips])))
        self.assertEqual(v.precision_check([call()])['status'], 'PASS')
        self.assertEqual(v.precision_check([call(.1)])['status'], 'WARN')
        self.assertEqual(v.precision_check([call(.3)])['status'], 'FAIL')
        self.assertEqual(v.precision_check([call(flips=2)])['status'], 'FAIL')   # 2% of the confident values change sign
        self.assertEqual(v.precision_check([call(flips=1, confident=200)])['status'], 'PASS')   # 0.5%: a pathological row or two
        self.assertEqual(v.precision_check([call(shared=0)])['status'], 'INCONCLUSIVE')

    def test_agreement_statistics(self):
        rng = np.random.default_rng(0);y = rng.uniform(-1, 1, 400);x = .5 * y + .1
        a = v.agreement(x, y);self.assertAlmostEqual(a['pearson'], 1., 9);self.assertAlmostEqual(a['spearman'], 1., 9);self.assertAlmostEqual(a['slope'], .5, 9)
        self.assertAlmostEqual(a['intercept'], .1, 9);b = v.agreement(-x, y);self.assertAlmostEqual(b['pearson'], -1., 9);self.assertLess(b['same_sign'], .5)
        np.testing.assert_allclose(v._ranks(np.asarray([3., 1., 3., 2.])), [2.5, 0., 2.5, 1.])
        self.assertEqual(v.agreement([1., np.nan], [1., 2.])['count'], 1)
        s = v.spread(np.asarray([0., 1e-6, -2e-6, np.nan]));self.assertEqual((s['count'], s['nonfinite'], s['max']), (4, 1, 2e-6))

    def test_keys_and_lookup_join_by_action_path(self):
        a = np.asarray([[0, 1, 2, 3, 4, 5], [1, 0, 8, 0, -1, 0], [0, 1, 2, 3, 5, 4]]);b = a[[2, 0]]
        index, found = v.lookup(a, np.concatenate([b, [[3, 1, 0, 0, 0, 0]]]));self.assertEqual(index[:2].tolist(), [2, 0]);self.assertEqual(found.tolist(), [True, True, False])
        self.assertEqual(len(set(v.keys(a).tolist())), 3);self.assertEqual(v.lookup(a[:0], b)[1].tolist(), [False, False])
        with self.assertRaises(ValueError):v.keys(np.zeros(3))


class ParseTests(unittest.TestCase):
    def refused(self, argv, message):
        stderr = io.StringIO()
        with patch.object(sys, 'stderr', stderr), self.assertRaises(SystemExit):v.parse(argv)
        self.assertIn(message, stderr.getvalue())

    def test_the_command_line_parses_and_refuses(self):
        saved = ['--out', '/tmp/o', '--saved-roots', '/bank/attempt', '--index-sha256', '3' * 64]
        args = v.parse(saved + CE.CHAMPION + CE.HEAD + ['--depth3-k', '0', '--fixed-shapes', '--max-roots', '5'])
        self.assertEqual((args.saved_roots, args.champion_archive, args.head_archive, args.depth3_k, args.fixed_shapes, args.max_roots, args.sample),
            ('/bank/attempt', '/shared/x/champion.npz', '/shared/x/actor-99.npz', 0, True, 5, v.SAMPLE))
        chunk = ['--out', '/tmp/o', '--d3-root', '/d3', '--chunk', '0', '--memory-policy'];args = v.parse(chunk + CE.CHAMPION + CE.HEAD)
        self.assertEqual((args.d3_root, args.chunk, args.memory_policy, args.capacity, args.walk_chunk, args.seed), ('/d3', 0, True, 512, v.CHUNK, v.SEED))
        self.refused(chunk + CE.HEAD, 'are required');self.refused(chunk + CE.CHAMPION[:4] + CE.HEAD, 'are required')
        self.refused(chunk + CE.CHAMPION, '--head-archive (the V2 reference');self.refused(chunk + CE.CHAMPION + CE.HEAD[2:], '--head-archive (the V2 reference')
        self.refused(chunk + CE.CHAMPION + CE.HEAD[:4], 'go together');self.refused(chunk + CE.CHAMPION + CE.HEAD[:5] + ['X' * 64], '64-hex')
        self.refused(chunk + CE.CHAMPION + CE.HEAD + ['--champion-precision', 'f32'], 'both precisions')
        self.refused(chunk + CE.CHAMPION[:3] + ['E' * 64] + CE.CHAMPION[4:] + CE.HEAD, '64-hex')
        self.refused(saved + chunk[2:] + CE.CHAMPION + CE.HEAD, 'exactly one');self.refused(saved[:4] + CE.CHAMPION + CE.HEAD, '--index-sha256')
        self.refused(['--out', '/tmp/o', '--d3-root', '/d3'] + CE.CHAMPION + CE.HEAD, '--chunk');self.refused(chunk + CE.CHAMPION + CE.HEAD + ['--depth3-k', '10'], '0..9')
        self.refused(chunk + CE.CHAMPION + CE.HEAD + ['--sample', '0'], 'positive');self.refused(chunk[2:] + CE.CHAMPION + CE.HEAD, '--out')


if __name__ == '__main__':
    unittest.main()
