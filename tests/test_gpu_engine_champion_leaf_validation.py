"""Gap 2's validation tool (gpu_engine.champion_leaf_validation) on CPU.

The depth-N fixture's fake engine with a ZERO-SUM V2 layout (token 0's value feature centred, plus an outcome hint one step before the
fixture's decisive terminals, negated on the opponent's view), a readable tiny champion (the real champion.forward with inert layers:
its value is tanh of that feature), the fixture's independent champion-layout leaf (built from the fake engine's features, never from
the 29-token tensor or ChampionEvaluator.view) and a V2 stand-in head. The real DepthNPricer prices; the tool walks, judges and reports:
- the clean champion passes the adapter check (every level exact to 1e-5) and all three perspective checks;
- a sign-flipped evaluator FAILs the perspective check (and the adapter check); a mis-sliced one FAILs adapter-vs-raw;
- f32 against f32 differs by nothing;
- the verdict rules on synthetic numbers; the command line parses and refuses; main writes validation.json and validation.md.
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


def features(state, owner, leaf):
    """The fake engine's feature row [n,1,F] with feature 0 a zero-sum value from `owner`'s side: (value - CENTRE) x GAIN, plus HINT one
    step before the fixture's decisive terminals (won: modes 1/20 before the node, 21 before the leaf; lost: 9 before the leaf)."""
    x = leaf.inputs['numeric_features'][:, :, 0, :];b, shift, turn = state.mode % 100, state.mode // 100, state.turn
    hint = (j.where(((b == 1) | (b == 20)) & (turn == shift), 1., 0.) + j.where((b == 21) & (turn == 1 + shift), 1., 0.)
        - j.where((b == 9) & (turn == 1 + shift), 1., 0.))
    side = j.where(j.asarray(owner) == state.owner, 1., -1.)
    x = x.at[..., 0].set(((x[..., 0] - CENTRE) * GAIN + HINT * hint[:, None]) * side[:, None])
    return j.pad(x, ((0, 0), (0, 0), (0, CE.F - x.shape[-1])))


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


def readable_champion(precision='float32', gain=2.):
    """The real champion.forward with inert layers (zero norms, so attention and feed-forward add nothing): width 0 carries feature 0, so
    the value is tanh(gain x the mean over tokens of feature 0 x SCALES[t]); the policy reads features 0, 2 and 4 on the action tokens."""
    c = CE.tiny_config();p = {k: np.zeros(s, np.float32) for k, s in champion.shapes(c).items()}
    p['numeric_projection.weight'][0, 0] = 1.;p['numeric_projection.weight'][1, 2] = .25;p['numeric_projection.weight'][1, 4] = .125
    p['value_head.weight'][0, 0] = gain;p['policy_head.weight'][0, :2] = (.5, 1.)
    meta = dict(weights_archive_sha256='b' * 64, checkpoint_sha256=CE.CHECKPOINT, parameter_count=int(sum(x.size for x in p.values())))
    return ce.ChampionEvaluator({k: j.asarray(x) for k, x in p.items()}, c, meta, precision)


def v2_reference(gain=3.):
    """A V2 stand-in head on the V2 layout: tanh(gain x token 0's feature 0), zero logits, valid where finite."""
    def forward(params, inputs):
        x = inputs['numeric_features'][:, 0, 0, 0]
        return v5_model.Output(j.zeros((x.shape[0], 9), j.float32), j.tanh(params['gain'] * x), None, j.isfinite(x))
    return SimpleNamespace(params={'gain': j.asarray(gain, j.float32)}, forward=jax.jit(forward), header=dict(stand_in=True))


@unittest.skipUnless(N.HAVE_PRICER, 'gpu_engine.device_depthn_pricer / depthn_backups are not on this tree')
class ValidationTests(N.FixtureN, unittest.TestCase):
    """The fixture's requests on the zero-sum layout (request 7's root inputs shifted: pre code 3), bound to the clean champion under both
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
        cls.clean = {'f32': readable_champion('float32'), 'bf16': readable_champion('bfloat16')}
        cls.bound, cls.dropped, cls.root_values = v.bind([list(cls.v2_requests)], cls.clean);cls.reports = {}

    @contextlib.contextmanager
    def engine(self):
        with super().engine(), patch.object(v5_model, 'forward', side_effect=CE.no_v2):yield

    def validation(self, evaluators, *, requests=None, key=None, **kw):
        if key is not None and key in self.reports:return self.reports[key]
        with self.engine(), patch.object(v, 'log', lambda *a, **k: None):
            report = v.validate(requests or self.bound, evaluators=evaluators, executor=self.ex, model=object(), materializer=self.tables, recipe=self.recipe,
                observe=self.observe, reference=v2_reference(), **dict(dict(capacity=32, depth3_k=2, sample=64, chunk=64), **kw))
        if key is not None:self.reports[key] = report
        return report

    def test_bind_keeps_the_roots_both_evaluators_bound(self):
        self.assertEqual(self.dropped, {'f32': [], 'bf16': []});self.assertEqual([len(g) for g in self.bound['f32']], [len(self.v2_requests)])
        for a, b in zip(self.bound['f32'][0], self.bound['bf16'][0]):
            self.assertEqual(v.address(a), v.address(b));self.assertEqual((a.learner_actor, a.opponent_actor, b.learner_actor, b.opponent_actor), (0, 0, 0, 0))
        self.assertNotEqual([r.root_logits for r in self.bound['f32'][0]], [r.root_logits for r in self.bound['bf16'][0]])   # each its own prior

    def test_the_clean_champion_passes_the_adapter_and_perspective_checks(self):
        report = self.validation(self.clean, key='clean');c = report['checks'];a = c['adapter'];p = c['perspective']
        self.assertEqual(a['status'], 'PASS', a['reasons'])
        for label in ('root', 'depth2.turn1', 'depth2.leaf', 'depth3.turn1', 'depth3.ply2', 'depth3.leaf'):
            with self.subTest(group=label):
                g = a['groups'][label];e = g['f32']
                self.assertGreater(e['compared'], 0);self.assertEqual((g['inputs_unequal'], e['walk_disagrees'], e.get('digest_mismatch', 0)), (0, 0, 0))
                self.assertLessEqual(e['value']['max'], v.ADAPTER_VALUE);self.assertGreater(g['bf16']['compared'], 0)
                if label in ('root', 'depth2.turn1', 'depth3.turn1', 'depth3.ply2'):self.assertGreater(e['own_logits']['count'], 0);self.assertLessEqual(e['own_logits']['max'], v.ADAPTER_LOGIT)
        self.assertGreater(a['groups']['root']['f32']['opp_logits']['count'], 0)
        for name in ('v2', 'seats', 'near_terminal'):self.assertEqual(p[name]['status'], 'PASS', (name, p[name]['reasons']))
        groups = p['v2']['by_evaluator']['bf16']['groups'];self.assertGreater(groups['pooled']['pearson'], .9);self.assertGreater(groups['pooled']['count'], 100)
        self.assertGreater(min(groups['seat0']['count'], groups['seat1']['count']), v.MIN_ROWS)
        self.assertLess(p['seats']['pooled']['pearson'], -.9);self.assertEqual(p['seats']['owner_view_f32']['owner_closer'], 1.)
        nt = p['near_terminal']['sources'];self.assertGreaterEqual(nt['pooled']['unanimous'], v.MIN_ROWS)
        self.assertEqual(set(nt), {'root', 'depth2.turn1', 'depth3.ply2', 'pooled'});self.assertTrue(all(nt[k]['unanimous'] > 0 for k in ('root', 'depth2.turn1', 'depth3.ply2')))
        self.assertEqual(nt['pooled']['f32']['agree'], 1.);self.assertEqual(nt['pooled']['v2']['agree'], 1.)
        pr = c['precision'];self.assertIn(pr['status'], ('PASS', 'WARN'))
        for name in ('root', 'depth2.turn1', 'depth2.leaf', 'depth3.ply2', 'depth3.leaf'):self.assertGreater(pr['values'][name]['count'], 0, name)
        self.assertGreater(pr['values']['depth2.leaf']['max'], 0.)   # bf16 really rounds
        self.assertEqual(set(pr['decisions']), {'depth2', 'depth3'});self.assertGreater(pr['decisions']['depth2']['labeled_both'], 0)
        self.assertEqual(set(c['scale']['levels']), set(v.UNIFORM));self.assertIn('depth2.leaf', c['scale']['frame_population'])
        self.assertEqual(report['calls'][0]['rebound_equal'], {'f32': True, 'bf16': True})
        self.assertGreater(report['walk']['rows'], 0);self.assertEqual(report['pricers']['depth3.bf16']['k_own'], [2, 2])

    def test_a_sign_flipped_evaluator_fails_the_perspective_check(self):
        """The pricers price with a champion whose value is negated (the raw forward is untouched): the V2 agreement mirrors, the near-terminal
        signs invert and the pricer value sits on the opponent's side of the zero-sum views, each a perspective FAIL; the adapter check
        fails too. The tool's bind ran on the clean champion, so the first-root check (which refuses this evaluator) is not what catches it."""
        negated = {name: CE.Negated(e.params, e.config, e.meta, e.precision) for name, e in self.clean.items()}
        report = self.validation(negated);p = report['checks']['perspective']
        self.assertEqual((report['status'], p['status'], p['v2']['status']), ('FAIL', 'FAIL', 'FAIL'), p['v2']['reasons'])
        self.assertTrue(any('perspective suspect' in r for r in p['v2']['reasons']), p['v2']['reasons'])
        self.assertLess(p['v2']['by_evaluator']['f32']['groups']['pooled']['pearson'], -.9)
        self.assertTrue(p['v2']['by_evaluator']['raw']['status'].startswith('PASS'))   # the raw forward is the champion's own: not flipped
        self.assertEqual(p['near_terminal']['status'], 'FAIL');self.assertEqual(p['seats']['status'], 'FAIL')
        self.assertEqual(report['checks']['adapter']['status'], 'FAIL')

    def test_a_mis_sliced_evaluator_fails_adapter_vs_raw(self):
        shifted = {name: CE.Shifted(e.params, e.config, e.meta, e.precision) for name, e in self.clean.items()}
        report = self.validation(shifted);a = report['checks']['adapter']
        self.assertEqual(a['status'], 'FAIL');self.assertEqual(report['status'], 'FAIL')
        self.assertTrue(all(g['inputs_unequal'] > 0 for g in a['groups'].values()), {k: g['inputs_unequal'] for k, g in a['groups'].items()})
        self.assertGreater(a['groups']['depth2.leaf']['f32']['value']['max'], v.ADAPTER_VALUE)
        self.assertTrue(any('independent champion layout differs' in r for r in a['reasons']))

    def test_f32_against_f32_differs_by_nothing(self):
        f32 = {'f32': self.clean['f32'], 'bf16': readable_champion('float32')}
        report = self.validation(f32, requests={'f32': self.bound['f32'], 'bf16': self.bound['f32']}, capacity=16);pr = report['checks']['precision']
        self.assertEqual(pr['status'], 'PASS', pr['reasons']);self.assertEqual([c['roots'] for c in report['calls']], [16, len(self.v2_requests) - 16])   # two calls, merged
        self.assertEqual(report['checks']['adapter']['status'], 'PASS', report['checks']['adapter']['reasons'])
        for part in ('values', 'logits'):
            for k, s in pr[part].items():
                with self.subTest(part=part, quantity=k):self.assertEqual((s.get('max', 0.), s['nonfinite']), (0., 0))
        self.assertGreater(pr['values']['depth3.leaf']['count'], 0);self.assertGreater(pr['logits']['depth3.ply2.own']['count'], 0)
        self.assertTrue(all(x['changed'] == 0 for x in pr['argmax_changes'].values()));self.assertTrue(all(n == 0 for n in pr['status_differs'].values()))
        self.assertTrue(all(e['f32'] == e['bf16'] == e['shared'] for e in pr['enumeration'].values()))
        for family, d in pr['decisions'].items():self.assertGreater(d['labeled_both'], 0, family);self.assertEqual(d['share'], 1., family)

    def test_main_writes_the_report_and_its_summary(self):
        loaded = SimpleNamespace(provider=SimpleNamespace(executor=self.ex, materializer=self.tables, recipe=self.recipe), plan=SimpleNamespace(model=object()),
            groups=[list(self.v2_requests)], source=dict(fixture=True))
        with tempfile.TemporaryDirectory() as directory, self.engine(), patch.object(g2, 'bank_chunk_requests', return_value=loaded), \
                patch.object(g2, 'load_champion', return_value=self.clean['bf16']), patch.object(v, 'champion_observer', return_value=self.observe), \
                patch.object(v, 'load_reference', return_value=v2_reference()), patch.object(v, 'log', lambda *a, **k: None):
            out = Path(directory) / 'out'
            code = v.main(['--out', str(out), '--d3-root', '/d3', '--chunk', '0', '--capacity', '32', '--sample', '32', '--walk-chunk', '32', '--depth3-k', '2',
                '--fixed-shapes'] + CE.CHAMPION)
            report = json.loads((out / 'validation.json').read_text());summary = (out / 'validation.md').read_text()
        self.assertEqual(code, 0, report['status']);self.assertIn(report['status'], ('PASS', 'WARN'));self.assertEqual(report['schema'], v.SCHEMA)
        self.assertEqual(report['header']['thresholds'], v.plain(v.THRESHOLDS));self.assertEqual(report['header']['roots'], len(self.v2_requests))
        self.assertEqual(report['header']['evaluators']['f32']['precision'], 'float32');self.assertEqual(report['header']['evaluators']['bf16']['precision'], 'bfloat16')
        self.assertEqual(report['checks']['adapter']['status'], 'PASS');self.assertEqual(report['walk']['stack_rows'], 64)
        self.assertTrue(all(x['fixed_capacity'] for x in report['pricers'].values()))
        self.assertIn(f'# Champion 9375 leaf validation: {report["status"]}', summary);self.assertIn('| 2 adapter vs raw | PASS |', summary)


class VerdictTests(unittest.TestCase):
    """The verdict rules on synthetic numbers (THRESHOLDS)."""
    def test_sign_verdict_tiers(self):
        g = lambda n, r, s, confident=0, sc=None: dict(count=n, pearson=r, same_sign=s, confident=confident, same_sign_confident=sc)
        self.assertEqual(v.sign_verdict(dict(pooled=g(500, .7, .85)))[0], 'PASS')
        self.assertEqual(v.sign_verdict(dict(pooled=g(500, .3, .7)))[0], 'WARN')   # weak, not inverted
        self.assertEqual(v.sign_verdict(dict(pooled=g(500, -.6, .2)))[0], 'FAIL')
        self.assertEqual(v.sign_verdict(dict(pooled=g(500, .7, .4)))[0], 'FAIL')   # same-sign below chance
        self.assertEqual(v.sign_verdict(dict(pooled=g(500, .7, .6, 100, .9)))[0], 'PASS')   # confident rows decide the sign
        self.assertEqual(v.sign_verdict(dict(pooled=g(10, .9, .9)))[0], 'INCONCLUSIVE')
        self.assertEqual(v.sign_verdict(dict(pooled=g(500, .6, .85), root=g(10, -1., 0.)))[0], 'PASS')   # a group under MIN_ROWS is not judged
        status, reasons = v.sign_verdict(dict(pooled=g(500, .6, .85), seat1=g(250, -.5, .3)));self.assertEqual(status, 'FAIL');self.assertIn('seat1', reasons[0])

    def test_near_terminal_tiers(self):
        def terminal(values, target, n=40):
            plus = np.where(np.asarray(target) > 0, 2, 0);minus = np.where(np.asarray(target) < 0, 1, 0)
            return dict(root=dict(call=np.zeros(n, np.int64), seat=np.arange(n) % 2, plus=plus, minus=minus, value_f32=np.asarray(values, np.float32),
                value_bf16=np.asarray(values, np.float32), v2_value=np.full(n, np.nan, np.float32)))
        target = np.where(np.arange(40) % 3 == 0, -1., 1.)
        self.assertEqual(v.near_terminal_check(terminal(.5 * target, target))['status'], 'PASS')
        self.assertEqual(v.near_terminal_check(terminal(-.5 * target, target))['status'], 'FAIL')
        mixed = np.where(np.arange(40) < 30, .5, -.5) * target;self.assertEqual(v.near_terminal_check(terminal(mixed, target))['status'], 'WARN')
        self.assertEqual(v.near_terminal_check(terminal(.5 * target[:10], target[:10], n=10))['status'], 'INCONCLUSIVE')
        report = v.near_terminal_check(terminal(.5 * target, target));self.assertEqual(report['sources']['pooled']['f32']['pair_agree'], 1.)
        mixed = terminal(.5 * target, target);r = mixed['root'];r['plus'][:4], r['minus'][:4] = 1, 3   # four mixed parents: not unanimous, in the pairs
        s = v.near_terminal_check(mixed)['sources']['pooled'];self.assertEqual((s['unanimous'], s['parents_with_decisive_children']), (36, 40))
        rest = sum(2 if t > 0 else 1 for t in target[4:]);mixed_won = sum(1 if t > 0 else 3 for t in target[:4])   # a mixed parent wins its children of its sign
        self.assertEqual(s['f32']['agree'], 1.);self.assertAlmostEqual(s['f32']['pair_agree'], (mixed_won + rest) / (16 + rest))

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
        chunk = ['--out', '/tmp/o', '--d3-root', '/d3', '--chunk', '0', '--memory-policy'];args = v.parse(chunk + CE.CHAMPION)
        self.assertEqual((args.d3_root, args.chunk, args.memory_policy, args.head_archive, args.capacity, args.walk_chunk, args.seed), ('/d3', 0, True, None, 512, v.CHUNK, v.SEED))
        self.refused(chunk, 'are required');self.refused(chunk + CE.CHAMPION[:4], 'are required')
        self.refused(chunk + CE.CHAMPION + ['--champion-precision', 'f32'], 'both precisions')
        self.refused(chunk + CE.CHAMPION[:3] + ['E' * 64] + CE.CHAMPION[4:], '64-hex')
        self.refused(saved + chunk[2:] + CE.CHAMPION, 'exactly one');self.refused(saved[:4] + CE.CHAMPION, '--index-sha256')
        self.refused(['--out', '/tmp/o', '--d3-root', '/d3'] + CE.CHAMPION, '--chunk')
        self.refused(chunk + CE.CHAMPION + CE.HEAD[:2], 'go together');self.refused(chunk + CE.CHAMPION + ['--depth3-k', '10'], '0..9')
        self.refused(chunk + CE.CHAMPION + ['--sample', '0'], 'positive');self.refused(chunk[2:] + CE.CHAMPION, '--out')


if __name__ == '__main__':
    unittest.main()
