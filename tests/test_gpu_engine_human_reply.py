"""gpu_engine.human_reply_train: bands, splits, partial labels, history index, intervals, redaction, and (with torch) the wrapper's
identity at initialisation and the exact band fold."""
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from gpu_engine import fp_reply_train as fpt
from gpu_engine import human_reply_train as h

HAS_TORCH = importlib.util.find_spec('torch') is not None


def rows(n=12, seed=0):
    rng = np.random.default_rng(seed)
    legal = np.ones((n, 9), bool);legal[:, 7:] = False
    return dict(label=rng.integers(0, 7, n), legal=legal, censor=np.zeros(n, np.int64))


class DataTest(unittest.TestCase):
    def test_bands(self):
        self.assertEqual(h.band_of([0, np.nan, 1, 1199.9, 1200, 1499, 1500, 2100]).tolist(), [1, 1, 2, 2, 3, 3, 4, 4])
        self.assertEqual(h.BANDS[0], 'unknown')   # never a data band: dropout and serving only

    def test_replay_split_is_deterministic_and_about_a_tenth(self):
        split = h.replay_split(np.arange(5000))
        self.assertTrue(np.array_equal(split, h.replay_split(np.arange(5000))))
        self.assertTrue(.08 < (split == 'val').mean() < .12)
        self.assertEqual(set(split), {'val', 'train'})

    def test_partial_labels_are_cant_and_fainted_with_a_legal_move(self):
        a = rows(6);a['label'][:] = [-1, -1, -1, -1, 3, -1]
        a['censor'][:] = [h.CENSORS.index(c) for c in ('cant', 'fainted_before_acting', 'no_action', 'recharge', 'none', 'cant')]
        a['legal'][5, :4] = False   # a cant row with no legal move carries no 'some move' label
        self.assertEqual(h.partial_mask(a).tolist(), [True, True, False, False, False, False])
        self.assertEqual(h.keep_rows(a).tolist(), [True, True, False, False, True, False])

    def test_history_index_never_crosses_groups_and_ends_with_the_latest(self):
        group = h.group_ids(np.array([5, 5, 5, 7, 7, 5]), np.zeros(6), np.array([0, 0, 0, 0, 0, 0]))
        self.assertEqual(group.tolist(), [0, 0, 0, 1, 1, 2])   # a key change starts a new group even when a key repeats later
        idx = h.history_index(group, 2)
        self.assertEqual(idx.tolist(), [[-1, -1], [-1, 0], [0, 1], [-1, -1], [-1, 3], [-1, -1]])

    def test_seat_change_is_a_new_group(self):
        self.assertEqual(h.group_ids(np.zeros(4), np.zeros(4), np.array([0, 0, 1, 1])).tolist(), [0, 0, 1, 1])

    def test_partial_nll_is_minus_log_of_the_moves_total(self):
        probs = np.array([[.1, .2, .0, .1, .6, 0, 0, 0, 0], [.5, .1, .1, .1, .2, 0, 0, 0, 0]])
        nll = h.label_nll(probs, np.array([-1, 4]), np.array([True, False]))
        self.assertAlmostEqual(nll[0], -np.log(.4));self.assertAlmostEqual(nll[1], -np.log(.2))

    def test_cluster_interval_brackets_the_mean_and_counts_clusters(self):
        rng = np.random.default_rng(1);v = rng.normal(.3, 1, 4000);c = rng.integers(0, 200, 4000)
        r = h.cluster_ci(v, c)
        self.assertEqual(r['clusters'], 200);self.assertLess(r['interval_90'][0], r['mean']);self.assertLess(r['mean'], r['interval_90'][1])
        self.assertIsNone(h.cluster_ci([], [])['mean'])

    def test_contrast_is_late_minus_early_of_the_paired_gain(self):
        turn = np.array([1, 2, 3, 20, 25, 30] * 50);c = np.repeat(np.arange(50), 6)
        a = np.where(turn >= 18, 1.0, 2.0);b = np.full(len(turn), 2.0)   # a gains 1 nat late, nothing early
        r = h.contrast_ci(a, b, turn, c)
        self.assertAlmostEqual(r['mean'], -1.0);self.assertAlmostEqual(r['interval_90'][0], -1.0);self.assertAlmostEqual(r['interval_90'][1], -1.0)
        self.assertIsNone(h.contrast_ci(a[:3], b[:3], turn[:3], c[:3]))

    def test_redaction_uses_the_rows_own_credit(self):
        num = np.random.default_rng(2).random((3, 23, 132)).astype(np.float32)
        credit = num[:, 0, fpt.OPP_HAZARD_CREDIT].copy();out = h.redact_own_credit(num.copy())
        self.assertTrue(np.array_equal(out[:, 0, fpt.OPP_HAZARD_CREDIT], credit))
        self.assertTrue((out[:, 0, fpt.SELF_HAZARD_CREDIT] == 0).all())
        self.assertTrue((out[:, 1, fpt.LAST_DAMAGE_TAKEN] == 0).all());self.assertTrue((out[:, 7:13, fpt.LAST_DAMAGE_DEALT] == 0).all())
        self.assertTrue(np.array_equal(out[:, 7:13, 0], fpt.public_percent(num[:, 7:13, 0])))

    def test_players_and_the_player_disjoint_subset(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / 'in.jsonl'
            p.write_text('\n'.join(json.dumps(dict(index=i, log=f'|j|x\n|player|p1|{a}||\n|player|p2|{b}|1|\n|turn|1')) for i, a, b in
                                   ((0, 'ua', 'ub'), (1, 'uc', 'ua'), (2, 'ud', 'ue'))) + '\n')
            players = h.players_of(p)
        self.assertEqual(players, {0: ('ua', 'ub'), 1: ('uc', 'ua'), 2: ('ud', 'ue')})
        arrays = dict(replay=np.array([0, 0, 1, 1, 2, 2]), seat=np.array([0, 1, 0, 1, 0, 1]),
                      split=np.array(['train', 'train', 'test', 'test', 'test', 'test']))
        self.assertEqual(h.player_disjoint(arrays, players, np.array([2, 3, 4, 5])).tolist(), [2, 4, 5])   # ua played in train

    def bench_rows(self, n=40):
        rng = np.random.default_rng(5)
        cat = rng.integers(1, 30, (n, 23, 41)).astype(np.int32);num = rng.random((n, 23, 132)).astype(np.float32)
        num[:, 2:7, 0] = np.arange(5) + 10.;num[:, 17:22, 0] = np.arange(5) + 10.   # a marker pairing own slot j with switch token j
        cat[:, 17:22, h.SLOT_ID_COL] = np.arange(808, 813)
        legal = rng.random((n, 9)) < .7;legal[:, 0] = True
        label = np.array([int(rng.choice(np.flatnonzero(l))) for l in legal]);label[::7] = -1
        return dict(categorical_ids=cat, numeric_features=num, token_type_ids=np.tile(np.arange(23) % 16, (n, 1)).astype(np.int16),
                    attention_mask=np.ones((n, 23), bool), legal=legal, label=label)

    def assert_same_choice(self, before, after):
        for i in range(len(before['label'])):
            b, a = int(before['label'][i]), int(after['label'][i])
            if b < 4:self.assertEqual(a, b);continue
            self.assertTrue(after['legal'][i, a])
            self.assertEqual(after['numeric_features'][i, 13 + a, 0], before['numeric_features'][i, 13 + b, 0])   # the same mon
            mask = np.arange(41) != h.SLOT_ID_COL
            self.assertTrue(np.array_equal(after['categorical_ids'][i, 13 + a, mask], before['categorical_ids'][i, 13 + b, mask]))
        self.assertTrue((after['categorical_ids'][:, 17:22, h.SLOT_ID_COL] == np.arange(808, 813)).all())   # slot ids stay positional
        self.assertTrue(np.array_equal(after['numeric_features'][:, 2:7, 0], after['numeric_features'][:, 17:22, 0]))   # own j <-> switch j
        self.assertTrue(np.array_equal(after['legal'][:, :4], before['legal'][:, :4]))
        old = (after['numeric_features'][:, 17:22, 0] - 10).astype(int)   # the marker names each new slot's old slot
        for i in range(len(old)):
            for j in range(5):self.assertEqual(after['legal'][i, 4 + j], before['legal'][i, 4 + old[i, j]])
        self.assertEqual(int(after['legal'].sum()), int(before['legal'].sum()))

    def test_bench_shuffle_keeps_each_choice_and_hides_the_order(self):
        before = self.bench_rows();after = {k: v.copy() for k, v in before.items()}
        perm = h.shuffle_bench(after, 0)
        self.assertGreater((perm != np.arange(5)).any(1).mean(), .9)
        self.assert_same_choice(before, after)

    def test_bench_shuffle_refuses_data_without_positional_slot_ids(self):
        rows = self.bench_rows();rows['categorical_ids'][3, 17, h.SLOT_ID_COL] = 1
        with self.assertRaises(SystemExit):h.shuffle_bench(rows, 0)


    def test_pi_fp_without_a_split_record_is_held_to_its_run_report(self):
        from types import SimpleNamespace
        arrays = dict(seed=np.array([10, 10, 11, 12, 13]));split = dict(train=np.array([0, 1]), test=np.array([2]), late=np.array([3, 4]))
        with tempfile.TemporaryDirectory() as d:
            run = Path(d) / 'run';run.mkdir();(Path(d) / 'fpr').mkdir();seeds = Path(d) / 'seeds.json';seeds.write_text('[11]')
            args = SimpleNamespace(pi_fp=str(run / 'pi-fp.pt'), fpr=str(Path(d) / 'fpr'), test_seeds=str(seeds), late_from=5700)
            with self.assertRaises(SystemExit):h.pi_fp_split_check(args, {}, arrays, split)   # no record, no report
            (run / 'pi-fp.pt').write_bytes(b'unit checkpoint')
            from pokezero.neural_policy import checkpoint_file_sha256
            report = dict(schema='v2s-fp-reply-model.v1', saved=dict(sha256=checkpoint_file_sha256(run / 'pi-fp.pt')),
                          args=dict(data=args.fpr, test_seeds=args.test_seeds, late_from=5700, view='b', redact='peer-view-v2'),
                          rows=dict(train=2, test=1, late=2))
            (run / 'report.json').write_text(json.dumps(report))
            self.assertIn('run report', h.pi_fp_split_check(args, {}, arrays, split))
            (run / 'report.json').write_text(json.dumps(dict(report, rows=dict(train=3, test=1, late=2))))
            with self.assertRaises(SystemExit):h.pi_fp_split_check(args, {}, arrays, split)   # a different split
            (run / 'report.json').write_text(json.dumps(dict(report, saved=dict(sha256='0' * 64))))
            with self.assertRaises(SystemExit):h.pi_fp_split_check(args, {}, arrays, split)   # another checkpoint's report
            marker = dict(split=dict(fit_seeds=[12]))   # a record that fitted a held-out seed
            with self.assertRaises(SystemExit):h.pi_fp_split_check(args, marker, arrays, split)
            self.assertIn('split record', h.pi_fp_split_check(args, dict(split=dict(fit_seeds=[10])), arrays, split))

    def test_model_names_cannot_shadow_baselines(self):
        self.assertEqual(h.parse_models(['a1=/x', 'a2s1=/y']), {'a1': '/x', 'a2s1': '/y'})
        for bad in ('champion=/x', 'pi_fp=/x', 'turns=/x', 'a1_band_unknown=/x', 'A1=/x', 'a1'):
            with self.assertRaises(SystemExit, msg=bad):h.parse_models([bad])
        with self.assertRaises(SystemExit):h.parse_models(['a1=/x', 'a1=/y'])


@unittest.skipUnless(HAS_TORCH, 'torch not installed')
class ModelTest(unittest.TestCase):
    def setUp(self):
        import torch
        from pokezero import neural_policy as neural
        torch.manual_seed(0)
        config = neural.TransformerPolicyConfig(policy_id='unit', window_size=1, categorical_vocab_size=40, token_type_vocab_size=16,
            categorical_feature_count=41, numeric_feature_count=132, token_count=23, embedding_dim=16, transformer_layers=1,
            attention_heads=4, feedforward_dim=32, dropout=0., observation_schema_version='pokezero.observation.v4',
            value_activation='tanh', transition_token_budget=0, category_vocab=tuple('unit%04d' % i for i in range(23)),
            category_oov_buckets=16)   # 23 + 16 buckets + padding = 40
        self.torch = torch;self.base = neural.EntityTokenTransformerPolicy(config).eval()
        rng = np.random.default_rng(3);n = 10
        legal = np.ones((n, 9), bool);legal[:, 6:] = False
        self.arrays = dict(categorical_ids=rng.integers(0, 40, (n, 23, 41)).astype(np.int32), numeric_features=rng.random((n, 23, 132)).astype(np.float32),
                           token_type_ids=np.tile(np.arange(23) % 16, (n, 1)).astype(np.int16), attention_mask=np.ones((n, 23), bool),
                           legal=legal, label=np.array([0, 1, -1, 4, 2, 0, 1, 5, -1, 3]), partial=np.array([0, 0, 1, 0, 0, 0, 0, 0, 1, 0], bool),
                           band=np.array([1, 2, 3, 4, 1, 2, 3, 4, 1, 2]), group=np.array([0] * 6 + [1] * 4), turn=np.array([1, 2, 3, 4, 5, 6, 1, 2, 3, 4]),
                           kind=np.zeros(n, np.int64), forced=np.zeros(n, bool), locked=np.zeros(n, bool))
        self.arrays['attention_mask'][:, 22] = False

    def test_identity_at_initialisation_with_bands_and_history(self):
        torch = self.torch;HumanReply, embed_tokens = h.build(torch)
        model = HumanReply(self.base, bands=True, history=True, events=4).eval()
        store = h.Store(self.arrays, torch, 'cpu', events=4);rows_ = np.arange(10)
        wrapped = h.predict(model, store, rows_, torch, embed_tokens);plain = h.predict(model, store, rows_, torch, embed_tokens, base_only=True)
        self.assertLess(np.abs(wrapped - plain).max(), 1e-6)
        self.assertTrue(np.allclose(wrapped.sum(1), 1.));self.assertTrue((wrapped[:, 6:] == 0).all())

    def test_history_changes_the_output_once_its_projection_is_trained(self):
        torch = self.torch;HumanReply, embed_tokens = h.build(torch)
        model = HumanReply(self.base, bands=False, history=True, events=4).eval()
        with torch.no_grad():model.history.to_all.weight.normal_()
        store = h.Store(self.arrays, torch, 'cpu', events=4)
        p = h.predict(model, store, np.arange(10), torch, embed_tokens);q = h.predict(model, store, np.arange(10), torch, embed_tokens, base_only=True)
        self.assertGreater(np.abs(p[1:6] - q[1:6]).max(), 1e-4)   # rows with earlier decisions in their group

    def test_band_fold_is_exact(self):
        torch = self.torch;HumanReply, embed_tokens = h.build(torch)
        model = HumanReply(self.base, bands=True).eval()
        with torch.no_grad():model.band.normal_(0, .5)
        store = h.Store(self.arrays, torch, 'cpu')
        for band in range(len(h.BANDS)):
            wrapped = h.predict(model, store, np.arange(10), torch, embed_tokens, band=band)
            state = {k[len('base.'):]: v for k, v in model.state_dict().items() if k.startswith('base.')}
            folded = type(self.base)(self.base.config).eval();folded.load_state_dict(h.fold_band(state, model.band.detach(), band))
            plain = h.predict(HumanReply(folded).eval(), store, np.arange(10), torch, embed_tokens, base_only=True)
            self.assertLess(np.abs(wrapped - plain).max(), 1e-6, band)

    def test_loss_terms_full_and_partial(self):
        torch = self.torch
        logits = torch.tensor([[1., 2., 0., 0., 3., 0., 0., 0., 0.]] * 2);legal = torch.ones((2, 9), dtype=torch.bool)
        loss = h.loss_terms(torch, logits, legal, torch.tensor([4, -1]), torch.tensor([False, True]))
        p = torch.softmax(logits[0], -1)
        self.assertAlmostEqual(float(loss[0]), float(-torch.log(p[4])), places=5)
        self.assertAlmostEqual(float(loss[1]), float(-torch.log(p[:4].sum())), places=5)

    def test_gender_blind_model_ignores_the_gender_bits_and_keeps_their_weights_zero(self):
        torch = self.torch;HumanReply, embed_tokens = h.build(torch)
        model = HumanReply(self.base, bands=True, ignore_gender=True)
        flipped = {k: v.copy() for k, v in self.arrays.items()};flipped['numeric_features'][..., list(h.GENDER_COLS)] = 1. - flipped['numeric_features'][..., list(h.GENDER_COLS)]
        a = h.predict(model, h.Store(self.arrays, torch, 'cpu'), np.arange(10), torch, embed_tokens)
        b = h.predict(model, h.Store(flipped, torch, 'cpu'), np.arange(10), torch, embed_tokens)
        self.assertLess(np.abs(a - b).max(), 1e-7)
        model.train();st = h.Store(self.arrays, torch, 'cpu');idx = torch.arange(10)
        h.loss_terms(torch, model(**st.inputs(idx, embed_tokens, model)), st.legal[idx], st.label[idx], st.partial[idx]).mean().backward()
        w = model.base.numeric_projection.weight
        self.assertTrue((w.grad[:, list(h.GENDER_COLS)] == 0).all());self.assertTrue((w[:, list(h.GENDER_COLS)] == 0).all())
        self.assertGreater(float(w.grad.abs().sum()), 0.)

    def test_batch_bench_shuffle_matches_the_numpy_rule(self):
        torch = self.torch;before = DataTest.bench_rows(DataTest())
        inputs = dict(cat=torch.as_tensor(before['categorical_ids']).long(), num=torch.as_tensor(before['numeric_features']),
                      tt=torch.as_tensor(before['token_type_ids']).long(), att=torch.as_tensor(before['attention_mask']))
        out, legal, label = h.shuffle_bench_batch(torch, inputs, torch.as_tensor(before['legal']), torch.as_tensor(before['label']), torch.Generator().manual_seed(1))
        after = dict(categorical_ids=out['cat'].numpy(), numeric_features=out['num'].numpy(), legal=legal.numpy(), label=label.numpy())
        DataTest.assert_same_choice(self, before, after)

    def test_loss_gradient_is_finite_on_a_row_without_legal_moves(self):
        torch = self.torch
        logits = torch.randn(2, 9, requires_grad=True);legal = torch.ones((2, 9), dtype=torch.bool);legal[0, :4] = False   # a forced switch
        h.loss_terms(torch, logits, legal, torch.tensor([5, -1]), torch.tensor([False, True])).sum().backward()
        self.assertTrue(torch.isfinite(logits.grad).all())


if __name__ == '__main__':
    unittest.main()
