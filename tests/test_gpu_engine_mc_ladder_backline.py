"""mc_ladder_backline: backline rungs and diagnostics on hand-built re-pricer rows (two legal own actions, two kept replies). Pure
Python: no engine, model or device."""
import math
import unittest

from gpu_engine import mc_ladder_backline as M
from gpu_engine import mc_ladder_rungs as L
from gpu_engine import mc_ladder_worlds as W
from tests.test_gpu_engine_mc_ladder_rungs import D1, TRUE, row

SKARM = dict(slot=0, species='skarmory', moves=['spikes', 'toxic', 'drillpeck', 'rest'], base_ability='keeneye', ability='keeneye', item='leftovers')
BLISSEY = dict(slot=1, species='blissey', moves=['softboiled'], base_ability='naturalcure', ability='naturalcure', item='leftovers')
CHANSEY = dict(slot=1, species='chansey', moves=['softboiled'], base_ability='naturalcure', ability='naturalcure', item='leftovers')
LANE0 = [math.log(9.), 0.]   # 0.9 on lane 0, 0.1 on lane 1
LANE1 = [0., math.log(9.)]   # 0.1 on lane 0, 0.9 on lane 1


def make(depth1, lanes, parties, logits=None, cells=None, refused=None, **extra):
    """A re-pricer row whose worlds carry peer lanes, opponent parties, optional per-world reply logits and depth1 cells."""
    raw = row(depth1=depth1, depth2=depth1, refused=refused or [False] * -(-len(depth1) // 2), **extra)
    for w, world in enumerate(raw['worlds']):
        world['peer'] = dict(lanes=lanes[w] + [None] * (9 - len(lanes[w])))
        world['opponent'] = dict(active_species='skarmory', party=parties[w]);world['kernel_seed'] = 100 + w
        if logits is not None:world['reply_peer_logits'] = list(logits[w]) + [0.] * 7
        if cells is not None:world['depth1']['cells'] = cells[w]
    return raw


def cells_at(*forced):
    return dict(opp_forced=[list(x) for x in forced], own_forced=[], terminal=[])


def actions(out):
    return {name: rows[0]['action'] for name, rows in out.items()}


SWITCHY = [{0: (0., 0.), 1: (0., 1.)}, {0: (0., 0.), 1: (0., -1.)}]   # action 1 pays +1 / -1 on reply 1, 0 on reply 0


THREE = [{0: (0., 0., 0.), 1: (.2, .9, -.6)}, {0: (0., 0., 0.), 1: (.1, -.4, .5)}]   # three kept replies


def three_lanes(raw, logits):
    """Kept replies 0, 1 and 2 in every world (lanes 1 and 2: unseen switches), with the given per-world reply logits."""
    for world, lg in zip(raw['worlds'], logits):
        world['depth1']['kept'] = [b < 3 for b in range(9)];world['reply_peer_logits'] = list(lg) + [0.] * 6
    return raw


class BacklineRungTests(unittest.TestCase):
    def test_r3sw_is_r3_and_the_check_counts_it(self):
        raw = make(D1, [['move:tackle', 'switch:blissey']] * 3, [[SKARM, BLISSEY], [SKARM, CHANSEY], [SKARM, BLISSEY]])
        r3 = L.build([raw], 'C')[0]['R3'][0]
        out, summary = M.build([raw], 'foulplay', check_r3=[r3])
        self.assertEqual(actions(out)['R3sw'], r3['action']);self.assertEqual(summary['r3_check'], dict(compared=1, disagreements=0, examples=[]))
        _, summary = M.build([raw], 'foulplay', check_r3=[dict(r3, action=1 - r3['action'])])
        self.assertEqual(summary['r3_check']['disagreements'], 1)

    def test_prior_average_of_identical_worlds_is_r3_with_two_unequal_unseen_lanes(self):
        lanes = [['move:tackle', 'switch:blissey', 'switch:chansey']] * 2;parties = [[SKARM, BLISSEY, dict(CHANSEY, slot=2)]] * 2
        logits = [[0., 2., -1.], [0., 2., -1.]]   # one prior in both worlds, unequal over the two unseen switches
        raw = three_lanes(make([THREE[0]] * 2, lanes, parties, logits=logits), logits)
        got = actions(M.build([raw], 'foulplay')[0]);self.assertEqual(got['R3pa'], got['R3sw'])
        p = W.prices_of(W.depth1_worlds(raw), True)[0];labs = M.labels(raw['worlds'][0], {'skarmory'})
        pa = M.prior_average([p, p], [labs, labs], [.5, .5])[0];own = M.prior(p)
        for b in own:self.assertAlmostEqual(pa[b], own[b], places=12)   # proportional split: the world keeps its own prior

    def test_more_than_one_unseen_lane_lane_and_pool_differ(self):
        lanes = [['move:tackle', 'switch:blissey', 'switch:chansey']] * 2;parties = [[SKARM, BLISSEY, dict(CHANSEY, slot=2)]] * 2
        logits = [[0., 0., 0.], [0., 0., 0.]]
        raw = three_lanes(make(THREE, lanes, parties, logits=logits), logits)
        prices = W.prices_of(W.depth1_worlds(raw), True);labs = [M.labels(w, {'skarmory'}) for w in raw['worlds']]
        legal = tuple(a in (0, 1) for a in range(9));unseen = M.unseen_lanes(prices, labs)
        lane = M.lane_means(prices, unseen, legal, [[0, 1], [0, 1]])
        self.assertAlmostEqual(lane[0, 1, 1], (.9 - .4) / 2);self.assertAlmostEqual(lane[0, 1, 2], (-.6 + .5) / 2)
        pooled = M.switch_rungs(prices, labs, legal, [[0, 1], [0, 1]])['R3swavg_pool']
        self.assertAlmostEqual(pooled[0].joint_values[1][1], (.9 - .6 - .4 + .5) / 4);self.assertAlmostEqual(pooled[1].joint_values[1][2], (.9 - .6 - .4 + .5) / 4)

    def test_switch_averages_keep_unpriced_cells_unpriced(self):
        raw = make(SWITCHY, [['move:tackle', 'switch:blissey'], ['move:tackle', 'switch:chansey']], [[SKARM, BLISSEY], [SKARM, CHANSEY]], logits=[LANE1, LANE0])
        raw['worlds'][0]['depth1']['grid'][1][1] = None   # world 0's action-1 cell against the switch: unpriced
        out, _ = M.build([raw], 'foulplay')
        for name in ('R3sw', 'R3swavg_lane', 'R3swavg_pool', 'R3swavg_seen'):self.assertEqual(out[name][0]['reason'], 'unpriced_cell')

    def test_seen_groups_follow_the_seen_sets(self):
        other = dict(SKARM, moves=['spikes', 'toxic', 'drillpeck', 'whirlwind'])
        raw = make(SWITCHY, [['move:tackle', 'switch:blissey'], ['move:tackle', 'switch:chansey']], [[SKARM, BLISSEY], [other, CHANSEY]],
                   logits=[LANE1, LANE0])
        out, summary = M.build([raw], 'foulplay')
        got = actions(out);self.assertEqual((got['R3swavg_lane'], got['R3swavg_seen']), (0, 1))   # two groups of one: seen = R3
        self.assertEqual(summary['diagnostics']['seen_group_size'], 1.)

    def test_unlabelled_lanes_refuse(self):
        raw = make(SWITCHY, [['move:tackle', 'switch:blissey'], ['move:tackle']], [[SKARM, BLISSEY], [SKARM, CHANSEY]])   # world 1's lane 1
        self.assertEqual({r[0].get('reason') for r in M.build([raw], 'foulplay')[0].values()}, {'lane_unlabelled'})
        raw = make(SWITCHY, [['move:tackle', 'switch:blissey']] * 2, [[SKARM, BLISSEY], [SKARM, CHANSEY]]);raw['worlds'][1]['kept_matches_peer_legal'] = False
        self.assertEqual({r[0].get('reason') for r in M.build([raw], 'foulplay')[0].values()}, {'lane_unlabelled'})

    def test_switch_average_drops_the_prior_payoff_covariance(self):
        # world 0: 0.9 on the unseen switch, which pays +1; world 1: 0.1 on it, paying -1. Averaging: (.9 - .1) / 2 = .4 > 0 plays 1;
        # with the switch value averaged to 0 across worlds, action 1 is worth 0 and the tie goes to action 0.
        raw = make(SWITCHY, [['move:tackle', 'switch:blissey'], ['move:tackle', 'switch:chansey']], [[SKARM, BLISSEY], [SKARM, CHANSEY]],
                   logits=[LANE1, LANE0])
        got = actions(M.build([raw], 'foulplay')[0])
        self.assertEqual((got['R3sw'], got['R3swavg_lane'], got['R3swavg_pool'], got['R3swavg_seen']), (1, 0, 0, 0))   # one seen-set group
        self.assertEqual(got['R3noswu'], 0)   # without the switch reply, action 1 is worth 0 in both worlds
        self.assertEqual(got['R3pa'], 0)      # pi_bar: .5 / .5 in both worlds, so action 1 is worth (.5 - .5) / 2 = 0

    def test_a_seen_switch_is_not_averaged(self):
        raw = make(SWITCHY, [['move:tackle', 'switch:skarmory']] * 2, [[SKARM, BLISSEY], [SKARM, CHANSEY]], logits=[LANE1, LANE0])
        got = actions(M.build([raw], 'foulplay')[0])
        self.assertEqual((got['R3swavg_lane'], got['R3noswu']), (1, 1))   # skarmory is seen: nothing changes

    def test_reveal_average_on_opponent_forced_cells(self):
        lanes = [['move:tackle', 'move:surf']] * 2;parties = [[SKARM, BLISSEY], [SKARM, CHANSEY]]
        raw = make(SWITCHY, lanes, parties, logits=[LANE1, LANE0], cells=[cells_at((1, 1)), cells_at((1, 1))])
        got = actions(M.build([raw], 'foulplay')[0]);self.assertEqual((got['R3sw'], got['R3rv']), (1, 0))
        raw = make(SWITCHY, lanes, parties, logits=[LANE1, LANE0], cells=[cells_at((1, 1)), cells_at()])   # forced in world 0 only
        self.assertEqual(actions(M.build([raw], 'foulplay')[0])['R3rv'], 1)   # world 0's pool is itself
        out, summary = M.build([make(SWITCHY, lanes, parties)], 'foulplay')
        self.assertEqual(out['R3rv'][0]['reason'], 'cells_missing')
        _, summary = M.build([raw], 'foulplay')
        self.assertEqual(summary['diagnostics']['cells'], dict(priced=8, opp_forced=1 / 8, own_forced=0., terminal=0.))

    def test_prior_swap_with_the_true_rows(self):
        lanes = [['move:tackle', 'switch:blissey'], ['move:tackle', 'switch:chansey']]
        sampled = make(SWITCHY, lanes, [[SKARM, BLISSEY], [SKARM, CHANSEY]], logits=[LANE1, LANE0])
        true = make([{0: (0., 0.), 1: (0., .1)}] * 2, [['move:tackle', 'switch:snorlax']] * 2, [[SKARM, dict(BLISSEY, species='snorlax')]] * 2,
                    logits=[LANE0, LANE0], true_worlds=dict(TRUE, worlds=2, built=2))
        out, summary = M.build([sampled], 'foulplay', [true])
        got = actions(out)
        self.assertEqual(got['R3tp'], 0)   # true priors (0.1 on the switch) on the sampled grids: (.1 - .1) / 2 = 0, a tie
        self.assertEqual(got['R1sp'], 1)   # sampled priors on the true grids: (.9 * .1 + .1 * .1) / 2 > 0
        diag = summary['diagnostics']
        self.assertAlmostEqual(diag['unseen_switch_mass'], .5)
        self.assertEqual(diag['backline'], dict(unseen_mons=1, species_share=0., set_share=0., in_no_world=1.))
        self.assertEqual(set(diag['grid_diff']) - {'pairs'}, {'move', 'switch_unseen'})
        self.assertAlmostEqual(diag['grid_diff']['switch_unseen']['mean_abs'], (0. + .9 + 0. + 1.1) / 4)   # action 1: |.1 - 1|, |.1 + 1|
        self.assertEqual(diag['grid_diff']['move']['mean_abs'], 0.);self.assertEqual(diag['grid_diff']['pairs'], 4)
        self.assertAlmostEqual(diag['grid_diff']['switch_unseen']['contribution'], (.1 * .9 + .1 * 1.1) / 4)   # the true switch mass .1
        out, _ = M.build([sampled], 'foulplay', [])
        self.assertEqual(out['R3tp'][0]['reason'], 'true_row_missing')
        shifted = make([{0: (0., 0.), 1: (0., .1)}] * 2, [['move:tackle', 'switch:snorlax']] * 2, [[SKARM, dict(BLISSEY, species='snorlax')]] * 2,
                       logits=[LANE0, LANE0], true_worlds=dict(TRUE, worlds=2, built=2))
        shifted['worlds'][1]['kernel_seed'] = 999
        self.assertEqual(M.build([sampled], 'foulplay', [shifted])[0]['R1sp'][0]['reason'], 'kernel_seed_mismatch')
        other = make([{0: (0., 0.), 1: (0., .1)}] * 2, [['move:tackle', 'switch:snorlax']] * 2, [[SKARM, dict(BLISSEY, species='snorlax')]] * 2,
                     logits=[LANE0, LANE0], true_worlds=dict(TRUE, worlds=2, built=2))
        other['root']['root_logits'] = [0., 1.] + [0.] * 7
        self.assertEqual(M.build([sampled], 'foulplay', [other])[0]['R3tp'][0]['reason'], 'root_prior_differs')
        failed = make([{0: (0., 0.), 1: (0., .1)}] * 2, [['move:tackle', 'switch:snorlax']] * 2, [[SKARM, dict(BLISSEY, species='snorlax')]] * 2,
                      logits=[LANE0, LANE0], true_worlds=dict(TRUE, worlds=2, built=1))
        self.assertEqual(M.build([sampled], 'foulplay', [failed])[0]['R3tp'][0]['reason'], 'true_worlds_failed')

    def test_reweighted_prior_average_and_undecided(self):
        # action 1 pays .3 against the move and -1 against the unseen switch in both worlds: q(1) = .3 pi_bar(move) - pi_bar(switch)
        raw = make([{0: (0., 0.), 1: (.3, -1.)}] * 2, [['move:tackle', 'switch:blissey'], ['move:tackle', 'switch:chansey']],
                   [[SKARM, BLISSEY], [SKARM, CHANSEY]], logits=[LANE1, LANE0])
        heavy = {(7, 3): {0: -50., 1: 0.}}   # only world 1 (0.9 on the move): pi_bar is world 1's prior, q(1) = .27 - .1 > 0
        got = actions(M.build([raw], 'foulplay', weights=heavy)[0])
        self.assertEqual((got['R3pa'], got['R3pa_rw_b1']), (0, 1))   # uniform pi_bar .5 / .5: .15 - .5 < 0
        self.assertEqual(M.build([raw], 'foulplay', weights={})[0]['R3pa_rw_b1'][0]['reason'], 'world_weights_missing')
        bare = make(SWITCHY, [['move:tackle', 'switch:blissey']] * 2, [[SKARM, BLISSEY], [SKARM, CHANSEY]])
        del bare['worlds'][1]['opponent']
        out, _ = M.build([bare], 'foulplay');self.assertEqual({r[0].get('reason') for r in out.values()}, {'world_sets_missing'})


if __name__ == '__main__':
    unittest.main()
