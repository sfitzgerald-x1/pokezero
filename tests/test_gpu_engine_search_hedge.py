"""Search-seat reductions on small hand-made games with known solutions; no engine, model or device."""
import itertools
import math
import random
from types import SimpleNamespace
import unittest

import numpy as np

from pokezero import v5_contract as c, v5_gumbel as g
from gpu_engine import search_hedge as h


def legal(*actions):
    return tuple(a in actions for a in range(9))


def world(rows, *, own=(0, 1), replies=(0, 1), logits=None, requested=True):
    """WorldPrices with rows[i][k] = G[own[i]][replies[k]]; reply logits default to uniform."""
    values = [[None] * 9 for _ in range(9)]
    for i, a in enumerate(own):
        for k, b in enumerate(replies if requested else (0,)):values[a][b] = rows[i][k]
    reply_logits = tuple(logits) if logits is not None else (0.,) * 9
    return g.WorldPrices(requested, legal(*replies) if requested else (False,) * 9, reply_logits, values)


def brute_force(worlds, root_legal, lam=0., steps=400):
    """max over a fine grid on the 2- or 3-action simplex of the hedge objective."""
    gs = h.grids(worlds, root_legal);n = len(gs[0].actions);best = -math.inf
    for point in itertools.product(range(steps + 1), repeat=n - 1):
        if sum(point) > steps:continue
        x = [p / steps for p in point] + [(steps - sum(point)) / steps]
        strategy = [0.] * 9
        for a, p in zip(gs[0].actions, x):strategy[a] = p
        best = max(best, h.hedge_value(gs, strategy, lam=lam))
    return best


class HedgeTests(unittest.TestCase):
    def test_rock_paper_scissors_is_uniform_with_value_zero(self):
        rps = world([[0, -1, 1], [1, 0, -1], [-1, 1, 0]], own=(0, 1, 2), replies=(0, 1, 2))
        choice = h.bayes_hedge((rps, rps), legal(0, 1, 2))
        np.testing.assert_allclose(choice.strategy[:3], (1 / 3,) * 3, atol=1e-9)
        self.assertAlmostEqual(choice.value, 0., places=9)
        self.assertEqual(choice.strategy[3:], (0.,) * 6)

    def test_biased_matching_pennies_known_mixture(self):
        # 1.5x - .5 = .5 - x  ->  x = .4, value .1
        game = world([[1, -.5], [-.5, .5]])
        choice = h.bayes_hedge((game, game), legal(0, 1))
        self.assertAlmostEqual(choice.strategy[0], .4, places=9)
        self.assertAlmostEqual(choice.value, .1, places=9)
        value, x = h.matrix_game([[1, -.5], [-.5, .5]])
        self.assertAlmostEqual(value, .1, places=9);np.testing.assert_allclose(x, (.4, .6), atol=1e-9)

    def test_opponent_knows_its_set_averaging_grids_first_is_wrong(self):
        # Averaged grid: U = (0, 0), D = (-.1, -.1), so solving it once plays U for 0. But each type
        # punishes U with its own reply (U gets -1 in both worlds); the Bayesian solve plays D for -.1.
        w1 = world([[1, -1], [-.1, -.1]]);w2 = world([[-1, 1], [-.1, -.1]])
        choice = h.bayes_hedge((w1, w2), legal(0, 1))
        self.assertEqual(choice.strategy[:2], (0., 1.))
        self.assertAlmostEqual(choice.value, -.1, places=9)
        averaged, _ = h.matrix_game((np.asarray([[1, -1], [-.1, -.1]]) + np.asarray([[-1, 1], [-.1, -.1]])) / 2)
        self.assertAlmostEqual(averaged, 0., places=9)
        # Averaging (best response to the uniform reply prior) is fooled the same way.
        self.assertEqual(h.averaging((w1, w2), legal(0, 1)).strategy[:2], (1., 0.))

    def test_no_strategy_fusion_solving_per_world_then_averaging_overstates(self):
        # U dominates in world 1 and D in world 2: per-world solves give 1 each, but I don't know the world.
        w1 = world([[1, 1], [-1, -1]]);w2 = world([[-1, -1], [1, 1]])
        per_world = [h.matrix_game(np.asarray(rows))[0] for rows in ([[1, 1], [-1, -1]], [[-1, -1], [1, 1]])]
        self.assertEqual(per_world, [1., 1.])
        self.assertAlmostEqual(h.bayes_hedge((w1, w2), legal(0, 1)).value, 0., places=9)

    def test_world_specific_reply_sets(self):
        # World 2's opponent has only reply 0; world 1's has replies 0 and 1.
        w1 = world([[1, -1], [0, .5]]);w2 = world([[.6], [-.4]], replies=(0,))
        choice = h.bayes_hedge((w1, w2), legal(0, 1))
        self.assertAlmostEqual(choice.value, brute_force((w1, w2), legal(0, 1)), places=4)
        self.assertGreaterEqual(choice.value + 1e-9, brute_force((w1, w2), legal(0, 1)))

    def test_waiting_opponent_is_a_pure_argmax(self):
        w1 = world([[.2], [.5], [-.3]], own=(0, 2, 5), requested=False)
        w2 = world([[.4], [.0], [.1]], own=(0, 2, 5), requested=False)
        root = legal(0, 2, 5)
        self.assertEqual(h.bayes_hedge((w1, w2), root).strategy, h.averaging((w1, w2), root).strategy)
        self.assertEqual(h.averaging((w1, w2), root).strategy[0], 1.)   # .3 beats .25 and -.1

    def test_blend_endpoints_and_midpoint(self):
        w1 = world([[1, -1], [-.2, .3]], logits=(2., 0.) + (0.,) * 7)
        w2 = world([[-.5, .8], [.4, -.6]], logits=(0., 1.) + (0.,) * 7)
        root = legal(0, 1)
        self.assertEqual(h.bayes_hedge((w1, w2), root, lam=0.).strategy, h.bayes_hedge((w1, w2), root).strategy)
        # lam = 1 is the reply-prior average: its LP optimum is the averaging argmax (a vertex).
        average = h.averaging((w1, w2), root)
        full = h.bayes_hedge((w1, w2), root, lam=1.)
        self.assertEqual(full.strategy, average.strategy);self.assertAlmostEqual(full.value, average.value, places=9)
        for lam in (0., .5):
            self.assertAlmostEqual(h.bayes_hedge((w1, w2), root, lam=lam).value, brute_force((w1, w2), root, lam), places=3)

    def test_averaging_equals_d_completed_q_exactly(self):
        rng = random.Random(7)
        for _ in range(50):
            root = legal(*sorted(rng.sample(range(9), rng.randint(1, 9))))
            worlds = []
            for _ in range(2):
                replies = tuple(sorted(rng.sample(range(9), rng.randint(1, 9))))
                own = tuple(a for a in range(9) if root[a])
                worlds.append(world([[rng.uniform(-1, 1) for _ in replies] for _ in own], own=own, replies=replies,
                    logits=[rng.gauss(0, 2) for _ in range(9)]))
            logits = tuple(rng.gauss(0, 2) for _ in range(9))
            reduced = g.reduce_completed_q(root_logits=logits, root_legal=root, worlds=worlds)
            ours = h.averaging(worlds, root)
            self.assertEqual(ours.q, reduced.improvement.completed_q)
            q = [v if v is not None else -math.inf for v in reduced.improvement.completed_q]
            self.assertEqual(ours.strategy.index(1.), q.index(max(q)))

    def test_random_games_match_brute_force(self):
        rng = random.Random(11)
        for trial in range(40):
            n = rng.choice((2, 3));root = legal(*range(n))
            worlds = []
            for _ in range(rng.choice((1, 2, 4))):
                k = rng.choice((1, 2, 3));replies = tuple(range(k))
                worlds.append(world([[rng.uniform(-1, 1) for _ in replies] for _ in range(n)], own=tuple(range(n)), replies=replies))
            lam = rng.choice((0., .5))
            choice = h.bayes_hedge(worlds, root, lam=lam)
            self.assertAlmostEqual(sum(choice.strategy), 1., places=12)
            # The LP value is optimal: never below the grid's best and within its resolution.
            reference = brute_force(worlds, root, lam, steps=200 if n == 2 else 60)
            self.assertGreaterEqual(choice.value + 1e-7, reference, msg=trial)
            self.assertLess(choice.value - reference, .05, msg=trial)

    def test_matrix_game_saddle_and_random_minimax(self):
        value, x = h.matrix_game([[3, 1, 4], [1, 0, 2], [2, .5, 3]])
        self.assertEqual(value, 1.);self.assertEqual(list(x), [1., 0., 0.])
        rng = np.random.default_rng(3)
        for _ in range(100):
            m = rng.uniform(-1, 1, size=tuple(rng.integers(1, 6, size=2)))
            value, x = h.matrix_game(m)
            # Duality: the column player's LP value equals the row player's.
            column, _ = h.matrix_game(-m.T)
            self.assertAlmostEqual(value, -column, places=7)
            self.assertAlmostEqual(math.fsum(x), 1., places=12);self.assertTrue(np.all(x >= 0))

    def test_hedge_rule_on_degenerate_node_grids(self):
        self.assertEqual(h.hedge_rule([[.5, -.25, .125]]), -.25)        # learner waiting: worst reply
        self.assertEqual(h.hedge_rule([[.5], [-.25], [.125]]), .5)      # opponent waiting: best own action
        self.assertEqual(h.hedge_rule([[.375]]), .375)
        self.assertAlmostEqual(h.hedge_rule([[1, -.5], [-.5, .5]], (.5, .5), (.5, .5)), .1, places=9)

    def test_unknown_cell_or_unavailable_world_refuses(self):
        w1 = world([[1, -1], [0, None]]);w2 = world([[.1, .2], [.3, .4]])
        for worlds in ((w1, w2), (None, w2), ()):
            self.assertIsNone(h.bayes_hedge(worlds, legal(0, 1)))
            self.assertIsNone(h.averaging(worlds, legal(0, 1)))

    def test_sampling_is_deterministic_and_follows_the_strategy(self):
        strategy = (0., .25, 0., .5, .25, 0., 0., 0., 0.)
        self.assertEqual([h.sample(strategy, u) for u in (0., .2499, .25, .7499, .75, .9999)], [1, 1, 3, 3, 4, 4])
        address = c.DecisionAddress(1773000000, 12, 1)
        self.assertEqual(h.search_uniform(5, address), h.search_uniform(5, address))
        self.assertNotEqual(h.search_uniform(5, address), h.search_uniform(6, address))
        counts = np.bincount([h.sample(strategy, h.search_uniform(5, c.DecisionAddress(1, f, 0))) for f in range(4000)], minlength=9)
        np.testing.assert_allclose(counts / 4000, strategy, atol=.03)


def frame(t1_status, t1_value, t2_status, t2_value, *, pre=(0, 0), peer_requested=(True, True)):
    """A one-root Depth2Frame-shaped namespace holding only the fields the hedge reads."""
    return SimpleNamespace(pre=np.asarray([pre], np.int32), root_peer_requested=np.asarray([peer_requested], bool),
        root_peer_logits=np.zeros((1, 2, 9), np.float32), t1_status=t1_status[None], t1_value=t1_value[None],
        t2_status=t2_status[None], t2_value=t2_value[None])


class PriorRegularizedTests(unittest.TestCase):
    """Amendment 19's rule: argmax of completed-Q + c * log pi over the legal actions."""
    def test_zero_weight_is_averaging_and_a_large_weight_is_greedy(self):
        rng = random.Random(7)
        for _ in range(300):
            own = tuple(sorted(rng.sample(range(9), rng.randint(2, 5))));root = legal(*own)
            worlds = [world([[rng.choice((-1., -.5, 0., .5, 1.)) for _ in range(2)] for _ in own], own=own) for _ in range(4)]
            average = h.averaging(worlds, root);logits = [rng.gauss(0., 2.) for _ in range(9)]
            self.assertEqual(h.prior_regularized(average, logits, root, 0.), average)   # ties included: lowest index
            greedy = max(own, key=lambda a: (logits[a], -a))
            self.assertEqual(h.top_action(h.prior_regularized(average, logits, root, 1e6).strategy), greedy)

    def test_trade_off_between_q_and_the_root_policy(self):
        # Q = (.3, .2); pi over the legal pair = softmax(0, 2): the log-pi gap is exactly 2, so the switch is at c = .05.
        # (An illegal action's logit would only shift log Z for every action alike; it is never playable.)
        root = legal(0, 1);average = h.averaging([world([[.3, .3], [.2, .2]])], root)
        logits = [0., 2., 0., 0., 0., 50., 0., 0., 0.]
        self.assertEqual(h.top_action(h.prior_regularized(average, logits, root, .04).strategy), 0)
        picked = h.prior_regularized(average, logits, root, .06)
        self.assertEqual(h.top_action(picked.strategy), 1);self.assertEqual(picked.value, average.q[1]);self.assertEqual(picked.q, average.q)
        with self.assertRaises(ValueError):h.prior_regularized(average, logits, root, -.1)
        for bad in (float('nan'), float('inf'), True, '0.1'):
            with self.assertRaises(ValueError):h.prior_regularized(average, logits, root, bad)
        # Exactly tied legal logits: even a huge c leaves Q to break the tie (greedy would take the lower index).
        tied = h.averaging([world([[.2, .2], [.3, .3]])], root)
        self.assertEqual(h.top_action(h.prior_regularized(tied, [1.] * 9, root, 1e9).strategy), 1)


class Depth2HedgeTests(unittest.TestCase):
    def setUp(self):
        self.t1_status = np.zeros((2, 9, 9), np.int8);self.t1_value = np.zeros((2, 9, 9), np.float32)
        self.t2_status = np.zeros((2, 9, 9, 9, 9), np.int8);self.t2_value = np.zeros((2, 9, 9, 9, 9), np.float32)
        for w in (0, 1):
            for a1, r1 in itertools.product((0, 1), (0, 1)):
                self.t1_status[w, a1, r1] = 2;self.t2_status[w, a1, r1][np.ix_((0, 1), (0, 1))] = 4
                # Turn-2 grid: biased matching pennies scaled by the cell (value .1 * scale).
                scale = .5 + .25 * a1 - .25 * r1 + .125 * w
                self.t2_value[w, a1, r1][np.ix_((0, 1), (0, 1))] = scale * np.asarray([[1, -.5], [-.5, .5]])

    def test_expanded_cells_take_their_turn2_game_value(self):
        f = frame(self.t1_status, self.t1_value, self.t2_status, self.t2_value)
        worlds = h.depth2_hedged_worlds(f, 0)
        for w in (0, 1):
            self.assertEqual(worlds[w].opponent_legal, legal(0, 1))
            for a1, r1 in itertools.product((0, 1), (0, 1)):
                expected = float(np.float32(.5 + .25 * a1 - .25 * r1 + .125 * w)) * .1
                self.assertAlmostEqual(worlds[w].joint_values[a1][r1], expected, places=6)
        choice = h.bayes_hedge(worlds, legal(0, 1))
        self.assertAlmostEqual(choice.value, brute_force(worlds, legal(0, 1)), places=4)

    def test_terminal_cell_keeps_exact_value_and_error_refuses(self):
        self.t1_status[1, 1, 0] = 1;self.t1_value[1, 1, 0] = -1.;self.t2_status[1, 1, 0] = 0
        worlds = h.depth2_hedged_worlds(frame(self.t1_status, self.t1_value, self.t2_status, self.t2_value), 0)
        self.assertEqual(worlds[1].joint_values[1][0], -1.)
        self.t2_status[0, 0, 1, 1, 0] = 3
        worlds = h.depth2_hedged_worlds(frame(self.t1_status, self.t1_value, self.t2_status, self.t2_value), 0)
        self.assertIsNone(worlds[0].joint_values[0][1])
        self.assertIsNone(h.bayes_hedge(worlds, legal(0, 1)))

    def test_waiting_learner_at_turn2_takes_the_worst_reply(self):
        self.t2_status[0, 1, 1] = 0;self.t2_status[0, 1, 1][np.ix_((0,), (0, 1, 2))] = 4
        self.t2_value[0, 1, 1, 0, :3] = (.5, -.125, .25)
        worlds = h.depth2_hedged_worlds(frame(self.t1_status, self.t1_value, self.t2_status, self.t2_value), 0)
        self.assertEqual(worlds[0].joint_values[1][1], -.125)

    def test_waiting_root_opponent_lists_reply_zero_only(self):
        for w in (0, 1):
            self.t1_status[w, :, 1] = 0;self.t2_status[w, :, 1] = 0
        worlds = h.depth2_hedged_worlds(frame(self.t1_status, self.t1_value, self.t2_status, self.t2_value,
            peer_requested=(False, False)), 0)
        self.assertEqual(worlds[0].opponent_legal, (False,) * 9);self.assertFalse(worlds[0].opponent_requested)
        choice = h.bayes_hedge(worlds, legal(0, 1));average = h.averaging(worlds, legal(0, 1))
        self.assertEqual(choice.strategy, average.strategy)   # a waiting opponent: the hedge is the argmax

    def test_per_root_pool_unit_equals_frame_backup(self):
        f = frame(self.t1_status, self.t1_value, self.t2_status, self.t2_value)
        fields = {k: getattr(f, k)[0] for k in h.DEPTH2_FIELDS}
        self.assertEqual(h.depth2_hedged_root(fields), h.depth2_hedged_worlds(f, 0))
        import concurrent.futures, multiprocessing
        with concurrent.futures.ProcessPoolExecutor(2, mp_context=multiprocessing.get_context('spawn')) as pool:
            self.assertEqual(list(pool.map(h.depth2_hedged_root, [fields, fields])), [h.depth2_hedged_worlds(f, 0)] * 2)

    def test_unavailable_world_and_waiting_seats(self):
        worlds = h.depth2_hedged_worlds(frame(self.t1_status, self.t1_value, self.t2_status, self.t2_value, pre=(0, -1)), 0)
        self.assertIsNone(worlds[1])
        # A waiting opponent at turn 2 enumerates only reply 0: the cell is the learner's best row value.
        self.t2_status[0, 0, 0] = 0;self.t2_status[0, 0, 0][np.ix_((0, 1), (0,))] = 4
        self.t2_value[0, 0, 0, 0, 0] = -.25;self.t2_value[0, 0, 0, 1, 0] = .375
        worlds = h.depth2_hedged_worlds(frame(self.t1_status, self.t1_value, self.t2_status, self.t2_value), 0)
        self.assertEqual(worlds[0].joint_values[0][0], .375)


if __name__ == '__main__':
    unittest.main()
