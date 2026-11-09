"""Foul Play reply dataset and model helpers on hand-made inputs; no Showdown, checkpoint or device."""
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from gpu_engine import fp_reply_data as d
from gpu_engine import fp_reply_train as t
from gpu_engine import fp_reply_probe as probe

FP_SEAT, OUR_SEAT = 'p2', 'p1'


def legal(*actions):
    return tuple(a in actions for a in range(9))


class FakeEnv:
    """Scripted rounds: each round is {player: (legal tuple, request)}; the battle ends with `winner` after the last round."""

    def __init__(self, rounds, winner):
        self.rounds, self.winner, self.steps = rounds, winner, []

    def reset(self, *, seed):
        self.seed, self.i, self.steps = seed, 0, []

    def _round(self):
        return self.rounds[self.i]

    @property
    def _latest_requests(self):
        return {player: request for player, (_, request) in self._round().items()}

    @property
    def _latest_turn(self):
        return self.i + 1

    def terminal(self):
        return SimpleNamespace(winner=self.winner) if self.i >= len(self.rounds) else None

    def requested_players(self):
        return tuple(self._round())

    def observe(self, player):
        return SimpleNamespace(legal_action_mask=self._round()[player][0], player=player)

    def step(self, actions):
        self.steps.append(dict(actions));self.i += 1


class FakePolicy:
    """Uniform over legal actions; argmax is the lowest legal index."""

    def reset(self):
        pass

    def select_action(self, observation, *, rng):
        mask = observation.legal_action_mask;n = sum(mask)
        probs = tuple(1. / n if m else 0. for m in mask)
        return SimpleNamespace(metadata=dict(policy_distribution=probs), action_index=mask.index(True), value_estimate=0.)


def request(tag, *, force=False):
    body = {'side': {'name': 'Bot', 'id': OUR_SEAT, 'pokemon': [tag]}, 'rqid': 7}
    if force:body['forceSwitch'] = [True]
    return body


def game(**overrides):
    base = dict(seed=11, fp=FP_SEAT, ours=OUR_SEAT, journal={0: dict(round=0, action=2, choice='act 2'), 1: dict(round=1, action=5, choice='act 5')},
                our_actions=[0, 1], our_greedy=[0, 4], winner=OUR_SEAT,
                our_requests=['|request|' + json.dumps(dict(request('a'), side={'name': 'Other', 'id': OUR_SEAT, 'pokemon': ['a']}, rqid=1)),
                              '|request|' + json.dumps(request('b'))])
    base.update(overrides);return base


def two_rounds():
    return [{OUR_SEAT: (legal(0, 1, 4), request('a')), FP_SEAT: (legal(0, 2, 5), {'active': []})},
            {OUR_SEAT: (legal(1, 4), request('b')), FP_SEAT: (legal(2, 5), {'forceSwitch': [True]})}]


def fake_choice_action(env, player, choice):
    """Stands in for decoding a /choose body on the replayed state: 'act N' decodes to N, anything else to None."""
    kind, _, arg = (choice or '').partition(' ')
    return int(arg) if kind == 'act' and arg.isdigit() else None


class ReplayTests(unittest.TestCase):
    def setUp(self):
        self.real_choice_action = d.choice_action;d.choice_action = fake_choice_action

    def tearDown(self):
        d.choice_action = self.real_choice_action

    def test_aligned_game_records_foul_play_rows_and_steps_both_choices(self):
        env = FakeEnv(two_rounds(), OUR_SEAT)
        rows, parity, decided = d.replay(env, FakePolicy(), game(), rng=None)
        self.assertEqual([r['label'] for r in rows], [2, 5])
        self.assertEqual([r['forced'] for r in rows], [False, True])
        self.assertEqual([r['turn'] for r in rows], [1, 2])
        self.assertEqual(env.steps, [{OUR_SEAT: 0, FP_SEAT: 2}, {OUR_SEAT: 1, FP_SEAT: 5}])
        self.assertEqual((parity, decided), (1, 2))   # greedy 0 matches; greedy 4 vs the fake argmax 1 doesn't

    def test_request_compare_ignores_rqid_and_username_but_not_the_team(self):
        bad = game(our_requests=['|request|' + json.dumps(request('z')), '|request|' + json.dumps(request('b'))])
        with self.assertRaisesRegex(d.Mismatch, 'our request differs'):d.replay(FakeEnv(two_rounds(), OUR_SEAT), FakePolicy(), bad, rng=None)

    def test_foul_play_requested_without_entry(self):
        g = game(journal={0: dict(round=0, action=2, choice='act 2')})
        with self.assertRaisesRegex(d.Mismatch, 'without a journal entry'):d.replay(FakeEnv(two_rounds(), OUR_SEAT), FakePolicy(), g, rng=None)

    def test_unused_journal_entry(self):
        g = game(journal={0: dict(round=0, action=2, choice='act 2'), 1: dict(round=1, action=5, choice='act 5'), 4: dict(round=4, action=0, choice='act 0')})
        with self.assertRaisesRegex(d.Mismatch, 'never requested'):d.replay(FakeEnv(two_rounds(), OUR_SEAT), FakePolicy(), g, rng=None)

    def test_illegal_foul_play_action(self):
        g = game(journal={0: dict(round=0, action=3, choice='act 3'), 1: dict(round=1, action=5, choice='act 5')})
        with self.assertRaisesRegex(d.Mismatch, 'illegal'):d.replay(FakeEnv(two_rounds(), OUR_SEAT), FakePolicy(), g, rng=None)

    def test_one_seat_rounds_leave_journal_holes_and_keep_our_actions_in_order(self):
        rounds = [{OUR_SEAT: (legal(0, 1, 4), request('a')), FP_SEAT: (legal(0, 2, 5), {'active': []})},
                  {OUR_SEAT: (legal(1, 4), request('b'))},                               # Foul Play waits: a journal hole
                  {FP_SEAT: (legal(5, 6), {'forceSwitch': [True]})},                     # only Foul Play: none of our actions
                  {OUR_SEAT: (legal(1, 4), request('c')), FP_SEAT: (legal(0, 2), {'active': []})}]
        g = game(journal={0: dict(round=0, action=2, choice='act 2'), 2: dict(round=2, action=6, choice='act 6'),
                          3: dict(round=3, action=0, choice='act 0')},
                 our_actions=[0, 4, 1], our_greedy=[0, 1, 1], our_requests=['|request|' + json.dumps(request(t)) for t in 'abc'])
        env = FakeEnv(rounds, OUR_SEAT);rows, parity, decided = d.replay(env, FakePolicy(), g, rng=None)
        self.assertEqual(env.steps, [{OUR_SEAT: 0, FP_SEAT: 2}, {OUR_SEAT: 4}, {FP_SEAT: 6}, {OUR_SEAT: 1, FP_SEAT: 0}])
        self.assertEqual([(r['round'], r['label'], r['forced']) for r in rows], [(0, 2, False), (2, 6, True), (3, 0, False)])
        self.assertEqual((parity, decided), (3, 3))

    def test_journal_choice_must_decode_to_its_action(self):
        g = game(journal={0: dict(round=0, action=2, choice='act 0'), 1: dict(round=1, action=5, choice='act 5')})
        with self.assertRaisesRegex(d.Mismatch, 'decodes to 0, not its action 2'):d.replay(FakeEnv(two_rounds(), OUR_SEAT), FakePolicy(), g, rng=None)
        g = game(journal={0: dict(round=0, action=2, choice='default'), 1: dict(round=1, action=5, choice='act 5')})
        with self.assertRaisesRegex(d.Mismatch, 'decodes to None'):d.replay(FakeEnv(two_rounds(), OUR_SEAT), FakePolicy(), g, rng=None)

    def test_our_seat_beyond_its_recorded_decisions_or_without_a_capture(self):
        g = game(our_actions=[0], our_greedy=[0], our_requests=game()['our_requests'][:1])
        with self.assertRaisesRegex(d.Mismatch, 'beyond its 1 recorded'):d.replay(FakeEnv(two_rounds(), OUR_SEAT), FakePolicy(), g, rng=None)
        g = game(our_requests=[game()['our_requests'][0], None])
        with self.assertRaisesRegex(d.Mismatch, 'our request differs'):d.replay(FakeEnv(two_rounds(), OUR_SEAT), FakePolicy(), g, rng=None)

    def test_telemetry_count_must_match_the_result(self):
        with self.assertRaisesRegex(d.Mismatch, 'telemetry has 2 decisions, the result 3'):
            d.replay(FakeEnv(two_rounds(), OUR_SEAT), FakePolicy(), game(decisions=3), rng=None)

    def test_winner_and_decision_count(self):
        with self.assertRaisesRegex(d.Mismatch, 'winner'):d.replay(FakeEnv(two_rounds(), FP_SEAT), FakePolicy(), game(), rng=None)
        g = game(our_actions=[0, 1, 1], our_greedy=[0, 4, 4], our_requests=game()['our_requests'] + [None])
        with self.assertRaisesRegex(d.Mismatch, 'made 2 of 3'):d.replay(FakeEnv(two_rounds(), OUR_SEAT), FakePolicy(), g, rng=None)


def write_collection(root, *, seed=2103410001, status='COMPLETE', journal=None):
    for sub in ('results', 'telemetry', 'owning-wire', 'requests'):(root / sub).mkdir(parents=True, exist_ok=True)
    journal = journal if journal is not None else [dict(round=0, seat=FP_SEAT, action=1, choice='move a')]
    (root / 'results' / '3.json').write_text(json.dumps(dict(status=status, cell=dict(seat=0, fp_seed=5),
        benchmark=dict(foulplay_player=FP_SEAT), source_game=dict(seed=seed, opponent_moves=journal, pokezero_decisions=2,
        terminal_witness=dict(winner_player=FP_SEAT)))))
    (root / 'telemetry' / '3.jsonl').write_text('\n'.join(json.dumps(dict(owner=OUR_SEAT, decision_index=i, played_action_index=i,
        greedy=dict(action_index=i))) for i in (1, 0)) + '\n')
    (root / 'owning-wire' / '3.jsonl').write_text('\n'.join(json.dumps(dict(event='ROOT', decision_index=i, own_request_line=f'|request|{{"i":{i}}}'))
                                                           for i in (0, 1)) + '\n')
    (root / 'requests' / '3.json').write_text(json.dumps(dict(base_config=dict(showdown_root='/s'), checkpoint_sha256='c')))
    return root / 'results' / '3.json'


class LoadGameTests(unittest.TestCase):
    def test_load_orders_our_decisions_and_reads_the_journal(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = write_collection(Path(tmp));g, skip = d.load_game(Path(tmp), str(path))
            self.assertIsNone(skip)
            self.assertEqual((g['fp'], g['ours'], g['seed'], g['ordinal']), (FP_SEAT, OUR_SEAT, 2103410001, 3))
            self.assertEqual(g['our_actions'], [0, 1]);self.assertEqual(g['our_requests'], ['|request|{"i":0}', '|request|{"i":1}'])
            self.assertEqual(list(g['journal']), [0]);self.assertEqual(g['winner'], FP_SEAT)

    def test_skips(self):
        cases = (dict(status='CAPPED'), 'not_complete'), (dict(seed=2103400500), 'screen_band'), (dict(journal=[]), 'no_journal'), \
            (dict(journal=[dict(round=0, seat=FP_SEAT, action=1)] * 2), 'journal_shape'), \
            (dict(journal=[dict(round=0, seat=OUR_SEAT, action=1)]), 'journal_shape')
        for kwargs, reason in cases:
            with self.subTest(reason=reason), tempfile.TemporaryDirectory() as tmp:
                self.assertEqual(d.load_game(Path(tmp), str(write_collection(Path(tmp), **kwargs)))[1], reason)


class MetricTests(unittest.TestCase):
    def test_metrics_mask_illegal_actions_and_count_low_label_probabilities(self):
        probs = np.array([[.0, .6, .3, .0, .1, 0, 0, 0, 0], [.95, .0, .0, .0, .045, .005, 0, 0, 0]], np.float32)
        probs[0, 3] = .9   # an illegal action with the largest probability must not count as the top-1
        legality = np.array([legal(1, 2, 4), legal(0, 4, 5)])
        m = d.metrics(probs, np.array([1, 5]), legality)
        self.assertEqual(m['top1'], .5);self.assertEqual(m['label_below_1pct'], .5);self.assertEqual(m['label_below_5pct'], .5)
        self.assertAlmostEqual(m['predicted_switch_mass'], (.1 + .05) / 2, places=6);self.assertEqual(m['actual_switch_share'], .5)
        self.assertAlmostEqual(m['nll'], -(np.log(.6) + np.log(.005)) / 2, places=5)

    def test_calibration_error(self):
        self.assertEqual(d.calibration_error(np.array([.5, .5]), np.array([1., 0.])), 0.)
        self.assertAlmostEqual(d.calibration_error(np.array([.9, .9, .3]), np.array([1., 0., 0.])), (2 / 3) * .4 + (1 / 3) * .3)
        self.assertAlmostEqual(d.calibration_error(np.array([1., .95]), np.array([1., 1.])), .025)   # 1.0 lands in the last bin

    def test_uniform_top1_is_the_expected_accuracy(self):
        legality = np.array([legal(0, 1), legal(0, 1, 2, 3)]);probs = legality / legality.sum(1, keepdims=True)
        m = d.metrics(probs, np.array([0, 0]), legality, uniform=True)
        self.assertAlmostEqual(m['top1'], (1 / 2 + 1 / 4) / 2);self.assertIsNone(m['ece'])

    def test_split_and_band(self):
        self.assertEqual(list(d.split_of(np.array([10, 21, 32]))), ['test', 'val', 'train'])
        self.assertEqual([d.band(x) for x in (1, 4, 19, 20, 0)], ['1-3', '4-9', '10-19', '20+', '0'])


class WindowTests(unittest.TestCase):
    def test_windows_never_cross_games_and_front_pad(self):
        index = t.windows(np.array([7, 7, 7, 9, 9]), 2)
        self.assertEqual(index.tolist(), [[-1, 0], [0, 1], [1, 2], [-1, 3], [3, 4]])
        self.assertEqual(t.windows(np.array([7, 7]), 1).tolist(), [[0], [1]])

    def test_current_only_masks_every_earlier_frame(self):
        index = t.windows(np.array([7, 7, 7]), 3)
        self.assertEqual(t.current_only(index).tolist(), [[-1, -1, 0], [-1, -1, 1], [-1, -1, 2]])


class CurveAndBootstrapTests(unittest.TestCase):
    def arrays(self):
        return dict(turn=np.array([0, 1, 2, 7, 60]), label=np.array([0, 1, 0, 0, 1]), ordinal=np.array([4, 4, 5, 5, 6]),
                    seed=np.array([10, 10, 10, 10, 20]))

    def test_curve_buckets_every_turn_including_zero(self):
        sure = np.tile(np.eye(9)[0] * .9 + .1 / 9, (5, 1))
        curve = t.in_context_curve(dict(m=sure), self.arrays(), np.arange(5))
        self.assertEqual([(c['turns'], c['decisions']) for c in curve], [('0', 1), ('1', 1), ('2', 1), ('6-8', 1), ('50+', 1)])
        self.assertAlmostEqual(curve[1]['m'], -np.log(.1 / 9))

    def test_bootstrap_clusters_by_seed_and_needs_enough_seeds(self):
        p = np.full((5, 9), 1 / 9);q = np.tile(np.eye(9)[0] * .5 + .5 / 9, (5, 1))
        out = t.game_bootstrap(dict(a=p, b=p, c=q), self.arrays(), np.arange(5), (('a', 'b'), ('c', 'a')), draws=200)
        self.assertEqual((out['a - b']['seeds'], out['a - b']['games']), (2, 3))   # games 4 and 5 share seed 10
        self.assertIsNone(out['c - a']['interval_90'])                           # 2 seeds < MIN_CLUSTERS

    def test_bootstrap_interval_brackets_the_mean_with_enough_seeds(self):
        rng = np.random.default_rng(1);n = 60
        arrays = dict(label=rng.integers(0, 9, n), ordinal=np.arange(n), seed=np.repeat(np.arange(n // 2), 2), turn=np.ones(n, int))
        p = np.full((n, 9), 1 / 9);q = rng.dirichlet(np.ones(9), n)
        out = t.game_bootstrap(dict(a=p, b=p, c=q), arrays, np.arange(n), (('a', 'b'), ('c', 'a')), draws=400)
        self.assertEqual(out['a - b']['interval_90'], [0., 0.])
        low, high = out['c - a']['interval_90'];self.assertLess(low, out['c - a']['mean']);self.assertGreater(high, out['c - a']['mean'])


class ProbeHelperTests(unittest.TestCase):
    def test_extras_helpers(self):
        request = {'side': {'pokemon': [{'active': False, 'condition': '10/20'}, {'active': True, 'condition': '30/120 par'}]},
                   'active': [{'moves': [{'id': 'tackle'}, {'id': 'toxic'}]}]}
        self.assertEqual(d.hp_fraction(request), .25);self.assertEqual(d.hp_fraction({'side': {'pokemon': [{'active': True, 'condition': '0 fnt'}]}}), 0.)
        dex = SimpleNamespace(move_info=lambda m: SimpleNamespace(category='Status' if m == 'toxic' else 'Physical'))
        self.assertEqual(d.action_classes(request, legal(0, 1, 5), dex), [0, 1, -1, -1, -1, 2, -1, -1, -1])

    def test_sequences_features_and_masked_softmax(self):
        arrays = dict(label=np.array([0, 4, 1]), label_class=np.array([0, 2, 1]), candidate_class=np.array([[0, 1, -1, -1, 2, -1, -1, -1, -1]] * 3),
                      own_hp=np.array([1., .5, np.nan]), opp_hp=np.array([.2, .3, .4]), forced=np.array([False, True, False]))
        x = probe.step_features(arrays)
        np.testing.assert_allclose(x[1], [0, 0, 1, 1, 1, 1, .5, .3, 1], rtol=1e-6);self.assertEqual(x[2, 6], 0.)
        runs = probe.game_sequences(np.array([5, 5, 7, 7, 7]), np.array([4, 0, 2, 1]))
        self.assertEqual([r.tolist() for r in runs], [[0, 1], [2, 4]])
        p = probe.masked_softmax(np.array([[0., 1., 5.]]), np.array([[True, True, False]]))
        self.assertAlmostEqual(p[0, 2], 0.);self.assertAlmostEqual(p[0, 1], np.e / (1 + np.e))

    def test_cluster_bootstrap_difference_of_subset_means(self):
        values = np.array([1., 3., 2., 4., 0., 0., 5., 1., 2., 2.]);clusters = np.repeat(np.arange(5), 2)
        late = np.tile([True, False], 5);early = ~late
        out = probe.cluster_bootstrap(values, clusters, [late, early], lambda a, b: a - b, draws=300)
        self.assertAlmostEqual(out['mean'], values[late].mean() - values[early].mean())
        self.assertLessEqual(out['interval_90'][0], out['mean']);self.assertGreaterEqual(out['interval_90'][1], out['mean'])
        self.assertIsNone(probe.cluster_bootstrap(values[:4], clusters[:4], [late[:4]], lambda a: a)['interval_90'])

    def test_fit_rows_exclude_test_and_games_pi_fp_trained_on(self):
        # seed % 10: 0 test, 1 val, else train; (seed // 10) % 4 == 0 for 402, 403 and 41 (early stopping when fitting on unseen games)
        seed = np.array([10, 11, 12, 13, 402, 403, 411, 41]);ordinal = np.arange(8);split = d.split_of(seed)
        fit, stop = probe.fit_rows(split, ordinal, seed)
        self.assertEqual(fit.tolist(), [2, 3, 4, 5]);self.assertEqual(stop.tolist(), [1, 6, 7])
        with tempfile.TemporaryDirectory() as tmp:
            np.savez(Path(tmp) / 'shard-0.npz', ordinal=np.array([2, 4, 1, 0]), seed=np.array([12, 402, 11, 10]))
            np.savez(Path(tmp) / 'shard-1.npz', label=np.zeros(0))
            seen = probe.seen_ordinals(tmp)
        self.assertEqual(seen, {2, 4})
        fit, stop = probe.fit_rows(split, ordinal, seed, seen)
        self.assertEqual(fit.tolist(), [1, 3, 6]);self.assertEqual(stop.tolist(), [5, 7])


@unittest.skipUnless(__import__('importlib').util.find_spec('torch'), 'torch not installed')
class RecalibrationTests(unittest.TestCase):
    def test_widen_repeats_the_single_position_row_and_sets_the_window(self):
        import torch
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp) / 'base.pt';row = torch.arange(4.).view(1, 4)
            torch.save(dict(model_config=dict(window_size=1, policy_id='x'), state_dict={'history_position_embedding.weight': row, 'w': torch.ones(2)}), base)
            payload = torch.load(t.widen(base, 3, tmp, torch), weights_only=True)
            self.assertEqual(payload['model_config']['window_size'], 3)
            self.assertTrue(torch.equal(payload['state_dict']['history_position_embedding.weight'], row.repeat(3, 1)))
            self.assertTrue(torch.equal(payload['state_dict']['w'], torch.ones(2)))

    def test_recalibration_recovers_a_flattening_temperature(self):
        import torch
        rng = np.random.default_rng(0);n = 4000;logits = rng.normal(size=(n, 9));legality = np.ones((n, 9), bool)
        true = np.exp(logits) / np.exp(logits).sum(1, keepdims=True)
        labels = np.array([rng.choice(9, p=p) for p in true])
        sharp = np.exp(3 * logits);sharp /= sharp.sum(1, keepdims=True)   # temperature 1/3 too sharp
        fit, apply = t.recalibrate(sharp, labels, legality, torch)
        self.assertAlmostEqual(fit['temperature'], 3., delta=.3);self.assertAlmostEqual(fit['switch_bias'], 0., delta=.15)
        self.assertLess(t.nll(apply(sharp, legality), labels), t.nll(sharp, labels))


if __name__ == '__main__':
    unittest.main()
