"""Foul Play reply dataset and model helpers on hand-made inputs; no Showdown, checkpoint or device."""
import copy
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from gpu_engine import fp_reply_data as d
from gpu_engine import fp_reply_train as t
from gpu_engine import fp_reply_probe as probe
from gpu_engine import fp_reply_reweight as rw

FP_SEAT, OUR_SEAT = 'p2', 'p1'


def legal(*actions):
    return tuple(a in actions for a in range(9))


class FakeEnv:
    """Scripted rounds: each round is {player: (legal tuple, request)}; the battle ends with `winner` after the last round."""

    def __init__(self, rounds, winner, lines=None):
        self.rounds, self.winner, self.steps, self.lines = rounds, winner, [], lines   # lines[i]: protocol lines before round i

    @property
    def protocol_lines(self):
        return tuple(x for chunk in (self.lines or ())[:self.i + 1] for x in chunk)

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
        self.assertEqual([r['fp_opp_hazard_credit'] for r in rows], [0, 0])

    def test_rows_carry_foul_plays_view_of_our_spikes_losses_at_their_decision(self):
        env = FakeEnv(two_rounds(), OUR_SEAT, lines=[SPIKES_LINES[:2], SPIKES_LINES[2:7]])
        rows, _, _ = d.replay(env, FakePolicy(), game(), rng=None)
        self.assertEqual([r['fp_opp_hazard_credit'] for r in rows], [np.float32(.12) / np.float32(6),
                                                                      d.fp_view_opp_hazard_credit(SPIKES_LINES[:7], OUR_SEAT)])
        self.assertAlmostEqual(float(rows[1]['fp_opp_hazard_credit']), .23 / 6, places=7)
        self.assertEqual([r['spikes_unknown_before'] for r in rows], [False, False])

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
        empty = probe.cluster_bootstrap(values, clusters, [late, np.zeros(10, bool)], lambda a, b: a - b, draws=10)
        self.assertIsNone(empty['mean']);self.assertIsNone(empty['interval_90'])

    def test_dataset_collection_must_be_one_collection(self):
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / 'shard-0.json').write_text(json.dumps({'collection': '/c/one'}))
            (Path(tmp) / 'shard-1.json').write_text(json.dumps({'collection': '/c/one'}))
            self.assertEqual(probe.dataset_collection(tmp), '/c/one')
            (Path(tmp) / 'shard-2.json').write_text(json.dumps({'collection': '/c/two'}))
            with self.assertRaises(SystemExit):probe.dataset_collection(tmp)

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



def _rw_request(item='leftovers', locked=False):
    entries = [{'move': n, 'id': i, 'pp': p, 'maxpp': 16, 'target': 'normal', 'disabled': d}
               for n, i, p, d in (('Earthquake', 'earthquake', 14, False), ('Sludge Bomb', 'sludgebomb', 16, False),
                                  ('Crunch', 'crunch', 16, False), ('Flamethrower', 'flamethrower', 16, True))]
    if locked:entries = entries[:1]
    return {'active': [{'moves': entries}], 'side': {'pokemon': [
        {'ident': 'p2: Seviper', 'details': 'Seviper, L88, M', 'active': True, 'moves': ['earthquake', 'sludgebomb', 'crunch', 'flamethrower'],
         'baseAbility': 'shedskin', 'ability': None, 'item': item},
        {'ident': 'p2: Milotic', 'details': 'Milotic, L77, M', 'active': False, 'moves': ['surf'], 'baseAbility': 'marvelscale', 'item': 'leftovers'}]}}


RW_DEX = SimpleNamespace(move_info=lambda m: SimpleNamespace(name={'hiddenpower': 'Hidden Power', 'toxic': 'Toxic'}.get(m, m.title()),
                                                             max_pp={'hiddenpower': 24, 'toxic': 16}.get(m, 8), target='normal'))


class PeerViewRedactionTests(unittest.TestCase):
    def test_public_percent_matches_showdowns_integer_rule_on_float32_fractions(self):
        maxhp = np.concatenate([np.full(m + 1, m) for m in range(1, 801)]);hp = np.concatenate([np.arange(m + 1) for m in range(1, 801)])
        exact = -(-100 * hp // maxhp);exact = np.where((hp < maxhp) & (exact == 100), 99, exact)
        got = t.public_percent((hp / maxhp).astype(np.float32))
        self.assertEqual(got.dtype, np.float32);np.testing.assert_array_equal(np.round(got.astype(np.float64) * 100).astype(int), exact)
        np.testing.assert_allclose(t.public_percent(np.float32([0.939873, 0.94, 0.999, 1., 0.])), [.94, .94, .99, 1., 0.], rtol=1e-6)

    def test_redaction_touches_only_the_three_peer_view_columns(self):
        rng = np.random.default_rng(0);numeric = rng.random((5, 23, 132)).astype(np.float32);before = numeric.copy()
        t.redact_peer_view(numeric)
        self.assertTrue((numeric[:, 1, 39] == 0).all());self.assertTrue((numeric[:, 7:13, 38] == 0).all())
        np.testing.assert_allclose(numeric[:, 7:13, 0], t.public_percent(before[:, 7:13, 0]))
        changed = numeric != before;changed[:, 1, 39] = changed[:, 7:13, 38] = changed[:, 7:13, 0] = False
        self.assertFalse(changed.any());self.assertEqual(before[:, 0, 39].tolist(), numeric[:, 0, 39].tolist())
        with self.assertRaises(SystemExit):t.redact_peer_view(np.zeros((2, 22, 132), np.float32))


# Spikes hits on our seat (p1), shown drops 12, 11, 12, 100: 300/300 -> 263/300 is 100% -> 88%; 298/299 -> 261/299 is 99% (a damaged
# mon never shows 100) -> 88%; 263/300 -> 226/300 is 88% -> 76%; a 20 HP Pichu KO'd is 100%. Their side's hit, a Rest heal and a
# Thunderbolt KO sit between.
SPIKES_LINES = (
    '|switch|p1a: Golem|Golem, L83, M|300/300', '|-damage|p1a: Golem|263/300|[from] Spikes', '|turn|2',
    '|switch|p2a: Zapdos|Zapdos, L74|250/250', '|-damage|p2a: Zapdos|219/250|[from] Spikes',
    '|switch|p1a: Snorlax|Snorlax, L70, M|298/299', '|-damage|p1a: Snorlax|261/299|[from] Spikes',
    '|-heal|p1a: Snorlax|299/299 slp|[from] move: Rest', '|switch|p1a: Golem|Golem, L83, M|263/300',
    '|-damage|p1a: Golem|226/300|[from] Spikes', '|move|p2a: Zapdos|Thunderbolt|p1a: Golem', '|-damage|p1a: Golem|0 fnt',
    '|faint|p1a: Golem', '|switch|p1a: Pichu|Pichu, L100|20/20', '|-damage|p1a: Pichu|0 fnt|[from] Spikes')


class HazardCreditTests(unittest.TestCase):
    def test_shown_percent_is_item_3s_public_percent(self):
        for top in range(1, 801):
            hp = np.arange(top + 1)
            shown = np.array([d.shown_percent(h, top) for h in hp])
            np.testing.assert_array_equal(shown, np.round(t.public_percent((hp / top).astype(np.float32)).astype(np.float64) * 100))
        self.assertEqual((d.shown_percent(263, 300), d.shown_percent(299, 300), d.shown_percent(0, 300)), (88, 99, 0))

    def test_spikes_hits_read_each_hit_against_the_mons_previous_exact_hp(self):
        self.assertEqual(d.spikes_hits(SPIKES_LINES, 'p1'), [(300, 263, 300), (298, 261, 299), (263, 226, 300), (20, 0, 20)])
        self.assertEqual(d.spikes_hits(SPIKES_LINES, 'p2'), [(250, 219, 250)])
        self.assertIsNone(d.spikes_hits(SPIKES_LINES[1:], 'p1'))   # the first hit's previous HP is not on the lines
        self.assertEqual(d.spikes_hits(('|request|{"side": {}}', '|turn|1'), 'p1'), [])

    def test_credit_sums_public_percent_drops_with_twosum_then_divides_by_six(self):
        # drops 12, 11, 12, 100 percent: 1.35 of a mon
        got = d.fp_view_opp_hazard_credit(SPIKES_LINES, 'p1');self.assertEqual(got.dtype, np.float32)
        high = low = np.float32(0)
        for drop in (12, 11, 12, 100):high, low = d.two_sum_add(high, low, np.float32(drop) / np.float32(100))
        self.assertEqual(got, np.float32(np.float32(high + low) / np.float32(6)))
        self.assertAlmostEqual(float(got), 1.35 / 6, places=6)
        exact = (37 / 300 + 37 / 299 + 37 / 300 + 1.) / 6   # the env's own value: our exact fractions
        self.assertGreater(abs(float(got) - exact), 1e-4)
        self.assertEqual(d.fp_view_opp_hazard_credit(SPIKES_LINES[1:], 'p1'), np.float32(0))
        self.assertEqual(d.fp_view_opp_hazard_credit(SPIKES_LINES * 6, 'p1'), np.float32(1))   # clipped at 1, as the device

    def test_twosum_compensates_a_long_float32_sum(self):
        high = low = np.float32(0);naive = np.float32(0)
        for _ in range(997):high, low = d.two_sum_add(high, low, np.float32(.01));naive = np.float32(naive + np.float32(.01))
        self.assertEqual(np.float32(high + low), np.float32(997 * np.float64(np.float32(.01))))
        self.assertNotEqual(naive, np.float32(high + low))
        self.assertEqual(d.two_sum_add(np.float32(.5), np.float32(0), np.float32(0)), (np.float32(.5), np.float32(0)))

    def test_peer_view_v2_zeroes_self_credit_and_writes_the_public_opp_credit(self):
        rng = np.random.default_rng(0);numeric = rng.random((5, 23, 132)).astype(np.float32);before = numeric.copy()
        credit = rng.random(5).astype(np.float32);v1 = t.redact_peer_view(before.copy())
        t.redact_peer_view(numeric, 'peer-view-v2', credit)
        self.assertTrue((numeric[:, 0, 115] == 0).all());np.testing.assert_array_equal(numeric[:, 0, 116], credit)
        changed = numeric != v1;changed[:, 0, 115:117] = False;self.assertFalse(changed.any())
        arrays = dict(numeric_features=before.copy(), fp_opp_hazard_credit=credit)
        np.testing.assert_array_equal(t.apply_redaction(arrays, 'peer-view-v2')['numeric_features'], numeric)
        self.assertIs(t.apply_redaction(dict(numeric_features=before), None)['numeric_features'], before)
        with self.assertRaises(SystemExit):t.redact_peer_view(before.copy(), 'peer-view-v2')
        with self.assertRaises(SystemExit):t.redact_peer_view(before.copy(), 'peer-view-v2', credit[:4])
        with self.assertRaises(SystemExit):t.redact_peer_view(before.copy(), 'peer-view-v3')


@unittest.skipUnless(__import__('importlib').util.find_spec('jax'), 'jax not installed')
class HazardCreditDeviceTests(unittest.TestCase):
    """fp_view_opp_hazard_credit against the device's own pieces: engine_device_trackers._percent, engine_device_damage_history._add
    and engine_device_features' clip(value / 6, 0, 1), the rule the search's peer view (fill_owner_spikes) runs."""

    def test_matches_the_device_rule_bit_for_bit(self):
        import jax.numpy as jnp
        from gpu_engine import engine_device_damage_history as history
        from gpu_engine import engine_device_trackers as trackers
        rng = np.random.default_rng(7)
        for _ in range(300):
            top = int(rng.integers(1, 801));hits = []
            for _ in range(int(rng.integers(1, 40))):
                before = int(rng.integers(0, top + 1));hits.append((before, int(rng.integers(0, before + 1)), top))
            pair = jnp.zeros(2, jnp.float32);high = low = np.float32(0)
            for before, after, maximum in hits:
                drop = max(0, int(trackers._percent(before, maximum)) - int(trackers._percent(after, maximum)))
                self.assertEqual(drop, max(0, d.shown_percent(before, maximum) - d.shown_percent(after, maximum)))
                pair = history._add(pair, jnp.asarray(drop, jnp.float32) / 100)
                high, low = d.two_sum_add(high, low, np.float32(drop) / np.float32(100))
            np.testing.assert_array_equal(np.asarray(pair), np.array([high, low], np.float32))
            device = float(jnp.clip(jnp.nan_to_num(pair.sum()) / 6., 0., 1.))
            lines = [x for before, after, maximum in hits for x in (f'|switch|p1a: Mon|Mon, L80|{before}/{maximum}',
                                                                    f'|-damage|p1a: Mon|{after}/{maximum}|[from] Spikes')]
            self.assertEqual(d.fp_view_opp_hazard_credit(lines, 'p1'), np.float32(device))


class ReweightHelperTests(unittest.TestCase):
    def test_set_keys_match_request_ids_to_variant_names(self):
        row = _rw_request()['side']['pokemon'][0]
        variant = {'moves': ['crunch', 'earthquake', 'flamethrower', 'sludgebomb'], 'ability': 'Shed Skin', 'item': 'Leftovers'}
        self.assertEqual(rw.request_key(row), rw.variant_key(variant))
        self.assertEqual(rw.move_key('Hidden Power Steel 70'), 'hiddenpowersteel');self.assertEqual(rw.species_of(row), 'seviper')
        self.assertEqual(rw.move_key('return102'), 'return');self.assertEqual(rw.request_move_id('return'), 'return102')
        self.assertEqual(rw.request_move_id('hiddenpowergrass'), 'hiddenpowergrass')
        self.assertEqual(rw.fresh_move_entry(RW_DEX, 'return'), {'move': 'Return 102', 'id': 'return', 'pp': 8, 'maxpp': 8,
                                                                 'target': 'normal', 'disabled': False})

    def test_world_request_keeps_shared_slots_and_fills_new_moves_fresh(self):
        request = _rw_request();first = copy.deepcopy(request['side']['pokemon'][0])
        variant = {'moves': ['crunch', 'earthquake', 'hiddenpowergrass', 'sludgebomb'], 'ability': 'Shed Skin', 'item': 'Choice Band'}
        world, labels, replaced_disabled = rw.world_request(request, first, variant, RW_DEX)
        row = world['side']['pokemon'][0];self.assertTrue(replaced_disabled)   # flamethrower's slot is disabled in the true request
        self.assertEqual(row['moves'], ['earthquake', 'sludgebomb', 'crunch', 'hiddenpowergrass'])
        self.assertEqual(world['active'][0]['moves'][:3], request['active'][0]['moves'][:3])
        self.assertEqual(world['active'][0]['moves'][3], {'move': 'Hidden Power Grass 70', 'id': 'hiddenpower', 'pp': 24, 'maxpp': 24,
                                                          'target': 'normal', 'disabled': True})
        self.assertEqual((row['item'], row['baseAbility']), ('choiceband', 'shedskin'));self.assertEqual(labels, [0, 1, 2, None, 4, 5, 6, 7, 8])
        self.assertEqual(request['side']['pokemon'][0]['moves'][3], 'flamethrower')   # input untouched
        powered = _rw_request();powered['side']['pokemon'][0]['moves'][2] = 'return102'
        world, labels, _ = rw.world_request(powered, first, {'moves': ['earthquake', 'frustration', 'return', 'sludgebomb']}, RW_DEX)
        self.assertEqual(world['side']['pokemon'][0]['moves'], ['earthquake', 'sludgebomb', 'return102', 'frustration102'])
        same, labels, replaced_disabled = rw.world_request(request, first, {'moves': first['moves'], 'ability': 'Shed Skin', 'item': 'Leftovers'}, RW_DEX)
        self.assertEqual(same, request);self.assertEqual(labels, list(range(9)));self.assertFalse(replaced_disabled)

    def test_hidden_power_types_are_one_family_the_protocol_cannot_split(self):
        request = _rw_request();request['side']['pokemon'][0]['moves'][1] = 'hiddenpowerfire'
        request['active'][0]['moves'][1] = {'move': 'Hidden Power Fire 70', 'id': 'hiddenpower', 'pp': 20, 'maxpp': 24, 'target': 'normal', 'disabled': False}
        first = copy.deepcopy(request['side']['pokemon'][0])
        variant = {'moves': ['crunch', 'earthquake', 'flamethrower', 'hiddenpowergrass'], 'ability': 'Shed Skin', 'item': 'Leftovers'}
        world, labels, replaced_disabled = rw.world_request(request, first, variant, RW_DEX)
        self.assertEqual(world['side']['pokemon'][0]['moves'], ['earthquake', 'hiddenpowergrass', 'crunch', 'flamethrower'])
        self.assertEqual(world['active'][0]['moves'][1], {'move': 'Hidden Power Grass 70', 'id': 'hiddenpower', 'pp': 20, 'maxpp': 24,
                                                          'target': 'normal', 'disabled': False})   # PP used is public; the type isn't
        self.assertEqual(labels[1], 1);self.assertFalse(replaced_disabled)   # a chosen Hidden Power Fire is a Hidden Power in this world
        _, labels, _ = rw.world_request(request, first, {'moves': ['crunch', 'earthquake', 'flamethrower', 'toxic']}, RW_DEX)
        self.assertIsNone(labels[1])   # a set without Hidden Power can't have used it
        self.assertEqual(rw.family('hiddenpowergrass'), 'hiddenpower');self.assertEqual(rw.family('toxic'), 'toxic')
        fire, grass = rw.variant_key({'moves': ['crunch', 'hiddenpowerfire']}), rw.variant_key({'moves': ['crunch', 'hiddenpowergrass']})
        self.assertNotEqual(fire, grass);self.assertEqual(rw.family_key(fire), rw.family_key(grass))

    def test_generator_stats_dump_attack_without_a_physical_damaging_move(self):
        cats = {'flamethrower': 'Special', 'hiddenpowerfire': 'Special', 'hiddenpowerground': 'Physical', 'earthquake': 'Physical',
                'seismictoss': 'Physical', 'toxic': 'Status'}
        dex = SimpleNamespace(move_info=lambda m: SimpleNamespace(gen3_category=cats[m]) if m in cats else None)
        tables = dict(fixed_damage=frozenset({'seismictoss'}), hp_ivs={'fire': {'atk': 30, 'spa': 30, 'spe': 30}, 'ground': {'spa': 30, 'spd': 30}})
        base = dict(atk=100, **{'def': 80}, spa=90, spd=70, spe=60)
        special = rw.generator_stats(base, 50, ['flamethrower', 'hiddenpowerfire', 'toxic', 'seismictoss'], dex, tables)
        self.assertEqual(special['atk'], (200 + 2) * 50 // 100 + 5)          # EV 0, IV 30 - 28
        self.assertEqual(special['spa'], (180 + 30 + 21) * 50 // 100 + 5)    # Hidden Power Fire's IV
        physical = rw.generator_stats(base, 50, ['flamethrower', 'hiddenpowerground', 'toxic'], dex, tables)
        self.assertEqual(physical['atk'], (200 + 31 + 21) * 50 // 100 + 5);self.assertEqual(physical['spd'], (140 + 30 + 21) * 50 // 100 + 5)
        self.assertEqual(rw.generator_stats(base, 50, ['flamethrower', 'toxic'], dex, tables)['atk'], 200 * 50 // 100 + 5)   # IV 0
        self.assertTrue(rw.attack_dumped(['flamethrower', 'seismictoss'], dex, tables))
        self.assertFalse(rw.attack_dumped(['flamethrower', 'earthquake'], dex, tables))
        self.assertEqual(rw.level_of({'details': 'Seviper, L88, M'}), 88);self.assertEqual(rw.level_of({'details': 'Mew'}), 100)
        request = _rw_request();first = copy.deepcopy(request['side']['pokemon'][0])
        world, _, _ = rw.world_request(request, first, {'moves': first['moves']}, RW_DEX, {'atk': 106})
        self.assertEqual(world['side']['pokemon'][0]['stats'], {'atk': 106})

    def test_world_request_leaves_a_public_item_change_and_a_locked_move(self):
        first = _rw_request()['side']['pokemon'][0]
        variant = {'moves': ['crunch', 'earthquake', 'toxic', 'sludgebomb'], 'ability': 'Shed Skin', 'item': 'Choice Band'}
        world, _, _ = rw.world_request(_rw_request(item=''), first, variant, RW_DEX)
        self.assertEqual(world['side']['pokemon'][0]['item'], '')
        locked = _rw_request(locked=True);world, labels, _ = rw.world_request(locked, first, variant, RW_DEX)
        self.assertEqual(world['active'], locked['active']);self.assertEqual(labels[0], 0)
        self.assertEqual(rw.chosen_move(locked, 0), 'earthquake');self.assertIsNone(rw.chosen_move(locked, 5))

    def test_public_choice_needs_the_chosen_move_or_a_switch_from_foul_plays_side(self):
        lines = ['|move|p2a: Seviper|Hidden Power|p1a: Swellow', '|move|p1a: Swellow|Aerial Ace|p2a: Seviper']
        self.assertTrue(rw.public_choice(lines, 'p2', 3, 'hiddenpowergrass'));self.assertFalse(rw.public_choice(lines, 'p2', 0, 'earthquake'))
        self.assertFalse(rw.public_choice(['|move|p2a: Snorlax|Body Slam|p1a: X|[from]Sleep Talk'], 'p2', 1, 'bodyslam'))
        self.assertFalse(rw.public_choice(['|cant|p2a: Seviper|par'], 'p2', 0, 'earthquake'))
        self.assertTrue(rw.public_choice(['|switch|p2a: Milotic|Milotic, L77, M|273/273'], 'p2', 5, None))

    def test_set_posterior_weights_the_uniform_prior_by_the_likelihood(self):
        a, b = 'A', 'B';keys, prior, post = rw.set_posterior([a, a, b], {a: np.log(.5), b: np.log(.25)}, 1.)
        self.assertEqual(keys, [a, b]);np.testing.assert_allclose(np.exp(prior), [2 / 3, 1 / 3]);np.testing.assert_allclose(np.exp(post), [.8, .2])
        _, _, half = rw.set_posterior([a, b], {a: np.log(.5)}, .5);np.testing.assert_allclose(np.exp(half), [.5 ** .5, 1.] / np.sum([.5 ** .5, 1.]))
        _, prior, post = rw.set_posterior([a, b], {a: -np.inf, b: -np.inf}, 1.);np.testing.assert_allclose(post, prior)
        self.assertAlmostEqual(rw.logsumexp([np.log(.2), np.log(.3)]), np.log(.5));self.assertEqual(rw.logsumexp([-np.inf]), -np.inf)

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

    def test_static_control_recovers_temperature_and_class_bias_on_its_rows(self):
        import torch
        rng = np.random.default_rng(1);n = 4000;logits = rng.normal(size=(n, 9));legality = np.ones((n, 9), bool)
        classes = np.tile(np.array([0, 0, 1, 1, 2, 2, 2, 2, 2]), (n, 1))
        true = np.exp(logits + .5 * (classes == 2));true /= true.sum(1, keepdims=True)
        labels = np.array([rng.choice(9, p=p) for p in true])
        sharp = np.exp(2 * logits);sharp /= sharp.sum(1, keepdims=True)
        rows = np.arange(0, n, 2)
        fit, probs = probe.static_fit(np.log(sharp).astype(np.float32), classes, legality, labels, rows, torch)
        self.assertAlmostEqual(fit['temperature'], 2., delta=.3)
        self.assertAlmostEqual(fit['class_bias'][2] - fit['class_bias'][0], .5, delta=.2)
        self.assertEqual(probs.shape, (n, 9));np.testing.assert_allclose(probs.sum(1), 1., rtol=1e-5)
        self.assertLess(t.nll(probs[1::2], labels[1::2]), t.nll(sharp[1::2], labels[1::2]))


if __name__ == '__main__':
    unittest.main()
