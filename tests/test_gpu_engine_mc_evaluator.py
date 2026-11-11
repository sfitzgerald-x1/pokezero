"""Model A's evaluator tools: the worst-case CRN-paired significance test, its backfill, the residual head (zero at the start, so the
untrained model is the old head exactly), early stopping and the rungs' train/gate disjointness. Pure numpy except the trainer/rungs
cases, which need torch and use tiny synthetic shards."""
import importlib.util
import json
import math
import tempfile
import unittest
from pathlib import Path

import numpy as np

from gpu_engine import mc_evaluator_features as F
from gpu_engine import mc_evaluator_pairs as P
from gpu_engine import mc_evaluator_train as T

CAPPED = F.CAPPED
HAS_TORCH = importlib.util.find_spec('torch') is not None


def audit_counterexample():
    """Codex's censor-audit case (K=64): A wins four then loses 60; B draws three, loses one, then is capped 60 times."""
    return [[1.] * 4 + [-1.] * 60, [0., 0., 0., -1.] + [CAPPED] * 60]


def ranked(outcomes, c=2.):
    d, se = F.pair_bounds(outcomes);return T.significant_pairs(dict(d_lo=d, se_lo=se, K=len(outcomes[0])), c)


class PairedSignificanceTests(unittest.TestCase):
    def test_the_audit_case_is_ranked_neither_way(self):
        q, _ = F.censored_q(audit_counterexample());self.assertLess(q[0] - q[1], 0)   # the marginal means: A below B
        self.assertEqual(ranked(audit_counterexample()), [])   # A outranks B only by dropping the 60 rollouts B outlived; B over A fails too

    def test_without_caps_the_bound_is_the_plain_paired_test_over_every_rollout(self):
        rng = np.random.default_rng(3);o = rng.choice([-1., 0., 1.], size=(3, 32));o[0] = np.clip(o[0] + 1, -1, 1)
        d, se = F.pair_bounds(o.tolist());diff = o[0] - o[2]
        self.assertAlmostEqual(d[0][2], diff.mean());self.assertAlmostEqual(se[0][2], diff.std(ddof=1) / math.sqrt(32))
        self.assertAlmostEqual(d[2][0], -diff.mean());self.assertIsNone(d[1][1])

    def test_capped_rollouts_count_against_the_favoured_action(self):
        a, b = [1.] * 30 + [CAPPED] * 2, [-1.] * 32
        d, _ = F.pair_bounds([a, b]);self.assertAlmostEqual(d[0][1], (30 * 2 + 2 * 0) / 32);self.assertAlmostEqual(d[1][0], (30 * -2 + 2 * -2) / 32)
        self.assertEqual(ranked([a, b]), [(0, 1, 1., 5.)])

    def test_a_constant_difference_ranks_at_full_weight_in_either_order(self):
        self.assertEqual(ranked([[1.] * 16, [-1.] * 16]), [(0, 1, 1., 5.)]);self.assertEqual(ranked([[-1.] * 16, [1.] * 16]), [(0, 1, -1., 5.)])

    def test_rows_with_few_rollouts_or_under_the_threshold_rank_nothing(self):
        self.assertEqual(ranked([[1.] * 4, [-1.] * 4]), [])
        row = dict(K=16, d_lo=[[None, .1, .5], [-.3, None, None], [-.7, None, None]], se_lo=[[None, .1, .2], [.1, None, None], [.2, None, None]])
        self.assertEqual(T.significant_pairs(row, 2.), [(0, 2, 1., 2.5)])

    def test_a_row_without_the_bound_is_refused(self):
        with self.assertRaises(ValueError):T.significant_pairs(dict(K=64, d_pair=[[None, 1.], [-1., None]], se_pair=[[None, .1], [.1, None]]), 2.)


def bank_row(kind, seed, outcomes, actions, **identity):
    return dict(kind=kind, seed=seed, actions=actions, outcomes=outcomes, **identity)


class BackfillTests(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.TemporaryDirectory();root = Path(self.dir.name)
        self.bank, self.features = root / 'bank', root / 'features';self.bank.mkdir();self.features.mkdir()
        self.rows = [bank_row('foulplay', 5, audit_counterexample(), [0, 3], ordinal=1, ours='p1', target=4),
                     bank_row('self', 9, [[1., 0., -1., 1.], [0., 0., 1., CAPPED]], [1, 2], ours='p2', target=3)]
        (self.bank / 'roots-a-0.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in self.rows))
        (self.features / 'features-0').mkdir()

    def tearDown(self):self.dir.cleanup()

    def index(self, rows):
        (self.features / 'index-0.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in rows))

    def index_row(self, r, **change):
        q, se = F.censored_q(r['outcomes'])
        return dict(dict(kind=r['kind'], ordinal=r.get('ordinal'), seed=r['seed'], ours=r['ours'], target=r['target'], actions=r['actions'],
                         K=len(r['outcomes'][0]), q_mc=q, se_mc=se, se_pair=F.pair_se(r['outcomes']), obs_offset=0), **change)

    def run_backfill(self):
        return P.main(['--features', str(self.features), '--bank', str(self.bank), '--out', str(Path(self.dir.name) / 'out')])

    def test_the_backfill_adds_the_bounds_and_keeps_every_other_field(self):
        before = [self.index_row(r) for r in self.rows];self.index(before);out = Path(self.dir.name) / 'out'
        self.assertEqual(self.run_backfill(), 0)
        after = [json.loads(l) for l in open(out / 'index-0.jsonl')]
        for b, a, r in zip(before, after, self.rows):
            self.assertEqual({k: v for k, v in a.items() if k not in ('d_lo', 'se_lo')}, b)
            self.assertEqual((a['d_lo'], a['se_lo']), F.pair_bounds(r['outcomes']))
        self.assertTrue((out / 'features-0').is_symlink());self.assertEqual((out / 'features-0').resolve(), (self.features / 'features-0').resolve())

    def test_a_row_whose_stored_targets_are_not_the_bank_row_s_is_refused_with_no_output(self):
        flipped = self.index_row(self.rows[1]);flipped['q_mc'] = [-x for x in flipped['q_mc']]   # se_pair is blind to a sign flip
        for bad in (dict(self.index_row(self.rows[1]), se_pair=[[None, .5], [.5, None]]), flipped, dict(self.index_row(self.rows[1]), K=8)):
            self.index([self.index_row(self.rows[0]), bad])
            with self.assertRaises(ValueError):self.run_backfill()
            self.assertFalse((Path(self.dir.name) / 'out').exists())

    def test_a_row_with_no_bank_row_is_refused(self):
        self.index([self.index_row(self.rows[0], target=99)])
        with self.assertRaises(SystemExit):self.run_backfill()


def synthetic_features(directory, games=24, width=8, tokens=3, seed=0, first_seed=1000):
    """A feature directory of two-action roots (a waiting opponent, one chance draw), each child observed, with outcomes and old values."""
    rng = np.random.default_rng(seed);directory = Path(directory);shard = directory / 'features-0';shard.mkdir(parents=True)
    rows, arrays = [], dict(tokens=[], valid=[], old_value=[])
    for g in range(games):
        outcomes = rng.choice([-1., 0., 1.], size=(2, 16)).tolist();d, se = F.pair_bounds(outcomes);q, se_mc = F.censored_q(outcomes)
        offset = len(arrays['old_value'])
        for _ in range(3):   # the root and its two children
            arrays['tokens'].append(rng.normal(size=(tokens, width)).astype(np.float16));arrays['valid'].append(np.ones(tokens, bool))
            arrays['old_value'].append(np.float32(rng.uniform(-.5, .5)))
        rows.append(dict(kind='self', ordinal=None, seed=first_seed + g, ours='p1', target=2, actions=[0, 1], replies=[None], reply_prior=None,
                         K=16, q_mc=q, se_mc=se_mc, se_pair=F.pair_se(outcomes), d_lo=d, se_lo=se,
                         reply_outcomes={str(a): {'None': outcomes[i]} for i, a in enumerate((0, 1))},
                         children=[dict(a=a, b=None, c=0, terminal=None, obs=1 + i) for i, a in enumerate((0, 1))], root_obs=0, draws=1,
                         obs_offset=offset, obs_count=3))
    for name, value in arrays.items():np.save(shard / f'{name}.npy', np.stack(value))
    (directory / 'index-0.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in rows))
    return rows


@unittest.skipUnless(HAS_TORCH, 'torch required')
class ResidualHeadTests(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.TemporaryDirectory();root = Path(self.dir.name)
        self.features, self.gate = root / 'features', root / 'gate'
        synthetic_features(self.features);synthetic_features(self.gate, games=10, seed=1, first_seed=5000)

    def tearDown(self):self.dir.cleanup()

    def test_the_untrained_residual_is_the_old_head_exactly(self):
        import torch
        head, _ = T.make_heads('residual', 8, 51, torch);batch = T.Batch(T.load([self.features])[:5], torch, torch.device('cpu'))
        with torch.no_grad():value, logits = T.predict('residual', head, batch)
        self.assertIsNone(logits);self.assertTrue(torch.equal(value, batch.old))

    def run_train(self, *extra):
        out = Path(self.dir.name) / f'model-{len(list(Path(self.dir.name).iterdir()))}'
        self.assertEqual(T.main(['--features', str(self.features), '--out', str(out), '--device', 'cpu', '--val-fraction', '.5',
                                 '--batch-roots', '8', *extra]), 0)
        return out, json.loads((out / 'metrics.json').read_text())

    def rungs(self, model, features):
        from gpu_engine import mc_evaluator_rungs as R
        out = Path(self.dir.name) / f'rungs-{len(list(Path(self.dir.name).iterdir()))}'
        return R.main(['--features', str(features), '--model', str(model), '--out', str(out), '--device', 'cpu']), out

    def test_early_stopping_saves_the_best_validation_regret_epoch_and_a_still_residual_scores_as_d1old(self):
        import torch
        out, metrics = self.run_train('--head', 'residual', '--lr', '0', '--lr-unc', '0', '--epochs', '3', '--early-stop-patience', '1')
        self.assertGreater(metrics[0]['val']['roots'], 0);self.assertIsNone(metrics[0]['val']['ce'])
        self.assertEqual(len(metrics), 2)   # epoch 1 does not improve on epoch 0 (no step), so patience 1 stops there
        saved = torch.load(out / 'model-a.pt', map_location='cpu');self.assertEqual((saved['head'], saved['epoch']), ('residual', 0))
        self.assertEqual(metrics[1]['val']['regret_new'], metrics[0]['val']['regret_old'])
        code, rungs = self.rungs(out / 'model-a.pt', self.gate);self.assertEqual(code, 0)
        self.assertEqual((rungs / 'rung-D1A.jsonl').read_text(), (rungs / 'rung-D1old.jsonl').read_text())
        self.assertEqual(json.loads((rungs / 'summary.json').read_text())['differ'], 0)

    def test_the_rungs_refuse_roots_from_the_model_s_training_games(self):
        out, _ = self.run_train('--head', 'residual', '--lr', '0', '--epochs', '1')
        with self.assertRaises(SystemExit):self.rungs(out / 'model-a.pt', self.features)

    def test_early_stopping_without_a_validation_regret_is_refused(self):
        with self.assertRaises(SystemExit):
            T.main(['--features', str(self.features), '--out', str(Path(self.dir.name) / 'none'), '--device', 'cpu', '--val-fraction', '0',
                    '--head', 'residual', '--epochs', '2', '--early-stop-patience', '1'])

    def test_a_training_residual_moves_and_the_hl_gauss_head_still_trains(self):
        import torch
        out, metrics = self.run_train('--head', 'residual', '--lr', '1e-2', '--epochs', '2')
        saved = torch.load(out / 'model-a.pt', map_location='cpu');self.assertEqual(saved['epoch'], 2)
        self.assertGreater(float(saved['value_head']['network.4.weight'].abs().sum()), 0)
        out, metrics = self.run_train('--epochs', '1', '--bins', '11')
        self.assertIsNotNone(metrics[1]['val']['ce']);self.assertEqual(torch.load(out / 'model-a.pt', map_location='cpu')['head'], 'hl_gauss')


if __name__ == '__main__':
    unittest.main()
