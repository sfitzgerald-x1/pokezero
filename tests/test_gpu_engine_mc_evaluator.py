"""Model A's evaluator tools: the CRN-paired significance test, the d_pair backfill, and the residual head (zero at the start, so the
untrained model is the old head exactly). Pure numpy except the trainer/rungs cases, which need torch and use tiny synthetic shards."""
import importlib.util
import json
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


class PairedSignificanceTests(unittest.TestCase):
    def test_the_paired_mean_and_se_describe_the_same_rollouts(self):
        d, se = F.pair_stats(audit_counterexample())
        self.assertAlmostEqual(d[0][1], 1.25);self.assertAlmostEqual(d[1][0], -1.25);self.assertAlmostEqual(se[0][1], .25)
        self.assertIsNone(d[0][0]);self.assertEqual(F.pair_se(audit_counterexample()), se)

    def test_selective_censoring_no_longer_reverses_the_ranked_direction(self):
        q, _ = F.censored_q(audit_counterexample());d, se = F.pair_stats(audit_counterexample())
        self.assertLess(q[0] - q[1], 0)   # the marginal gap the old test used: A below B
        self.assertEqual(T.significant_pairs(dict(d_pair=d, se_pair=se), 2.), [(0, 1, 1., 5.)])   # paired z = +5: A above B

    def test_pairs_under_the_threshold_or_without_an_se_are_not_ranked(self):
        row = dict(d_pair=[[None, .1, .5], [-.1, None, None], [-.5, None, None]], se_pair=[[None, .1, .2], [.1, None, None], [.2, None, None]])
        self.assertEqual(T.significant_pairs(row, 2.), [(0, 2, 1., 2.5)])

    def test_a_row_without_the_paired_mean_is_refused(self):
        with self.assertRaises(ValueError):T.significant_pairs(dict(q_mc=[0., 1.], se_pair=[[None, .1], [.1, None]]), 2.)


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
        return dict(dict(kind=r['kind'], ordinal=r.get('ordinal'), seed=r['seed'], ours=r['ours'], target=r['target'], actions=r['actions'],
                         se_pair=F.pair_se(r['outcomes']), obs_offset=0), **change)

    def test_the_backfill_adds_the_paired_mean_and_keeps_every_other_field(self):
        before = [self.index_row(r) for r in self.rows];self.index(before);out = Path(self.dir.name) / 'out'
        self.assertEqual(P.main(['--features', str(self.features), '--bank', str(self.bank), '--out', str(out)]), 0)
        after = [json.loads(l) for l in open(out / 'index-0.jsonl')]
        for b, a, r in zip(before, after, self.rows):
            self.assertEqual({k: v for k, v in a.items() if k != 'd_pair'}, b);self.assertEqual(a['d_pair'], F.pair_stats(r['outcomes'])[0])
        self.assertTrue((out / 'features-0').is_symlink());self.assertEqual((out / 'features-0').resolve(), (self.features / 'features-0').resolve())

    def test_a_row_whose_stored_se_is_not_the_bank_row_s_is_refused_with_no_output(self):
        bad = self.index_row(self.rows[1]);bad['se_pair'] = [[None, .5], [.5, None]];self.index([self.index_row(self.rows[0]), bad])
        out = Path(self.dir.name) / 'out'
        with self.assertRaises(ValueError):P.main(['--features', str(self.features), '--bank', str(self.bank), '--out', str(out)])
        self.assertFalse(out.exists())

    def test_a_row_with_no_bank_row_is_refused(self):
        self.index([self.index_row(self.rows[0], target=99)])
        with self.assertRaises(SystemExit):P.main(['--features', str(self.features), '--bank', str(self.bank), '--out', str(Path(self.dir.name) / 'out')])


def synthetic_features(directory, games=24, width=8, tokens=3, seed=0):
    """A feature directory of two-action roots (a waiting opponent, one chance draw), each child observed, with outcomes and old values."""
    rng = np.random.default_rng(seed);directory = Path(directory);shard = directory / 'features-0';shard.mkdir(parents=True)
    rows, arrays = [], dict(tokens=[], valid=[], old_value=[])
    for g in range(games):
        outcomes = rng.choice([-1., 0., 1.], size=(2, 16)).tolist();d, se = F.pair_stats(outcomes);q, se_mc = F.censored_q(outcomes)
        offset = len(arrays['old_value'])
        for _ in range(3):   # the root and its two children
            arrays['tokens'].append(rng.normal(size=(tokens, width)).astype(np.float16));arrays['valid'].append(np.ones(tokens, bool))
            arrays['old_value'].append(np.float32(rng.uniform(-.5, .5)))
        rows.append(dict(kind='self', ordinal=None, seed=1000 + g, ours='p1', target=2, actions=[0, 1], replies=[None], reply_prior=None,
                         K=16, q_mc=q, se_mc=se_mc, d_pair=d, se_pair=se, reply_outcomes={str(a): {'None': outcomes[i]} for i, a in enumerate((0, 1))},
                         children=[dict(a=a, b=None, c=0, terminal=None, obs=1 + i) for i, a in enumerate((0, 1))], root_obs=0, draws=1,
                         obs_offset=offset, obs_count=3))
    for name, value in arrays.items():np.save(shard / f'{name}.npy', np.stack(value))
    (directory / 'index-0.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in rows))
    return rows


@unittest.skipUnless(HAS_TORCH, 'torch required')
class ResidualHeadTests(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.TemporaryDirectory();self.features = Path(self.dir.name) / 'features';synthetic_features(self.features)

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

    def test_early_stopping_saves_the_best_validation_regret_epoch_and_a_still_residual_scores_as_d1old(self):
        import torch
        from gpu_engine import mc_evaluator_rungs as R
        out, metrics = self.run_train('--head', 'residual', '--lr', '0', '--lr-unc', '0', '--epochs', '3', '--early-stop-patience', '1')
        self.assertGreater(metrics[0]['val']['roots'], 0);self.assertIsNone(metrics[0]['val']['ce'])
        self.assertEqual(len(metrics), 2)   # epoch 1 does not improve on epoch 0 (no step), so patience 1 stops there
        saved = torch.load(out / 'model-a.pt', map_location='cpu');self.assertEqual((saved['head'], saved['epoch']), ('residual', 0))
        self.assertEqual(metrics[1]['val']['regret_new'], metrics[0]['val']['regret_old'])
        rungs = Path(self.dir.name) / 'rungs';self.assertEqual(R.main(['--features', str(self.features), '--model', str(out / 'model-a.pt'),
                                                                     '--out', str(rungs), '--device', 'cpu']), 0)
        a, old = (rungs / 'rung-D1A.jsonl').read_text(), (rungs / 'rung-D1old.jsonl').read_text()
        self.assertEqual(a, old);self.assertEqual(json.loads((rungs / 'summary.json').read_text())['differ'], 0)

    def test_a_training_residual_moves_and_the_hl_gauss_head_still_trains(self):
        import torch
        out, metrics = self.run_train('--head', 'residual', '--lr', '1e-2', '--epochs', '2')
        saved = torch.load(out / 'model-a.pt', map_location='cpu');self.assertEqual(saved['epoch'], 2)
        self.assertGreater(float(saved['value_head']['network.4.weight'].abs().sum()), 0)
        out, metrics = self.run_train('--epochs', '1', '--bins', '11')
        self.assertIsNotNone(metrics[1]['val']['ce']);self.assertEqual(torch.load(out / 'model-a.pt', map_location='cpu')['head'], 'hl_gauss')


if __name__ == '__main__':
    unittest.main()
