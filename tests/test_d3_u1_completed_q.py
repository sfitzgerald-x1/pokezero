import random
import unittest

from gpu_engine import d3_u1_completed_q as u1
from pokezero import v5_gumbel as original

SEL = dict(stream=7, frame=4, owner=1)


def full(value):
    return [dict(replicate=i, status='COMPLETE', outcome=value, requests=3, reason=None) for i in range(64)]


def mixed(wins):
    return [dict(replicate=i, status='COMPLETE', outcome=1 if i < wins else -1, requests=3, reason=None) for i in range(64)]


def world(rng, legal, replies):
    reply_legal = tuple(i in replies for i in range(9))
    grid = [[rng.uniform(-1, 1) if legal[a] and reply_legal[r] else None for r in range(9)] for a in range(9)]
    return original.WorldPrices(True, reply_legal, tuple(rng.uniform(-3, 3) for _ in range(9)), grid)


class ReductionTests(unittest.TestCase):
    def test_two_worlds_reproduce_reduce_completed_q_bitwise(self):
        rng = random.Random(3)
        for _ in range(50):
            legal = tuple(rng.random() < .6 for _ in range(9))
            if not any(legal):legal = (True,) + legal[1:]
            logits = tuple(rng.uniform(-4, 4) for _ in range(9))
            worlds = [world(rng, legal, set(rng.sample(range(9), rng.randint(1, 9)))) for _ in range(2)]
            mine = u1.reduce_worlds(logits, legal, worlds)
            target = original.reduce_completed_q(root_logits=logits, root_legal=legal, worlds=worlds).improvement
            self.assertEqual(mine['completed_q'], target.completed_q)
            self.assertEqual(mine['policy_target'], target.policy_target)
            self.assertEqual(mine['root_value'], target.root_value)

    def test_wider_composite_is_the_mean_of_blocks_and_refuses_any_unknown(self):
        rng = random.Random(5);legal = (True, True, False, True) + (False,) * 5;logits = (0.,) * 9
        blocks = [[world(rng, legal, {0, 2, 5}) for _ in range(2)] for _ in range(2)]
        two = [u1.reduce_worlds(logits, legal, b)['completed_q'] for b in blocks]
        four = u1.reduce_worlds(logits, legal, blocks[0] + blocks[1])['completed_q']
        for a in range(9):
            if legal[a]:self.assertAlmostEqual(four[a], (two[0][a] + two[1][a]) / 2, places=12)
            else:self.assertIsNone(four[a])
        broken = [list(map(list, w.joint_values)) for w in blocks[1]];broken[0][1][2] = None
        bad = original.WorldPrices(True, blocks[1][0].opponent_legal, blocks[1][0].opponent_logits, broken[0])
        self.assertIsNone(u1.reduce_worlds(logits, legal, blocks[0] + [bad, blocks[1][1]]))
        with self.assertRaises(ValueError):u1.reduce_worlds(logits, legal, blocks[0][:1])

    def test_block_seeds_keep_the_declared_seed_first(self):
        seeds = u1.world_block_seeds(1779000000, 16)
        self.assertEqual(seeds[0], 1779000000);self.assertEqual(len(set(seeds)), 16)
        self.assertEqual(seeds, u1.world_block_seeds(1779000000, 16));self.assertEqual(seeds[:4], u1.world_block_seeds(1779000000, 4))
        self.assertTrue(all(0 <= s < 2**64 for s in seeds))
        for s in seeds:original.Recipe(s)


class ScoringTests(unittest.TestCase):
    def bank(self):
        """Three root groups: two eligible (gap >= .05), one with a stored-observation hole (terminal child)."""
        groups = [dict(stream=1, frame=0, owner=0), dict(stream=2, frame=4, owner=1), dict(stream=3, frame=0, owner=0)]
        truth = {0: mixed(50), 1: mixed(30), 2: mixed(10)}
        records = [dict(selector=s, action=a, rollouts=truth[a]) for s in groups for a in (0, 1, 2)]
        raw = {(u1.key(s), a): v for s in groups for a, v in zip((0, 1, 2), (.1, .3, -.2))}
        raw[(u1.key(groups[2]), 2)] = None   # no stored head observation: unknown for every predictor
        roots = [dict(selector=s, prediction=0., outcome=1, backup=None) for s in groups]
        return groups, records, raw, roots

    def test_masked_predictor_scores_the_raw_pairs_minus_refusals(self):
        R = u1.rules();groups, records, raw, roots = self.bank()
        available = {k: v is not None for k, v in raw.items()}
        m = R.metrics(roots, u1.child_rows(records, raw, available))
        pairs = m['top_two']['rows'];self.assertEqual(len(pairs), 2);self.assertEqual(m['top_two']['unknown_groups'], 1)
        self.assertEqual([p['actions'] for p in pairs], [[0, 1]] * 2)
        self.assertEqual(u1.pair_outcomes(pairs, raw), [0, 0])
        # Q exists for every legal child, including the hole; group 2 (stream 2) is refused by D.
        q = {(u1.key(s), a): v for s in (groups[0], groups[2]) for a, v in zip((0, 1, 2), (.4, .2, .9))}
        mq = R.metrics(roots, u1.child_rows(records, q, available))
        self.assertEqual(mq['top_two']['eligible_known'], 1);self.assertEqual(mq['top_two']['unknown_groups'], 2)
        self.assertEqual(u1.pair_outcomes(pairs, q), [1, None])
        self.assertEqual(mq['child_known'], 5)   # group 0's three plus group 2's two stored-observation children

    def test_paired_difference_modes(self):
        pairs = [dict(selector=dict(stream=s, frame=0, owner=0), actions=[0, 1]) for s in (1, 1, 2, 3, 4)]
        raw = [0, 1, 0, 1, 0];other = [1, 1, None, 0, 1]
        fallback = u1.paired_difference(pairs, raw, other, mode='fallback', draws=2000)
        self.assertEqual(fallback['pairs'], 5);self.assertAlmostEqual(fallback['difference'], (1 + 0 + 0 - 1 + 1) / 5)
        conditional = u1.paired_difference(pairs, raw, other, mode='conditional', draws=2000)
        self.assertEqual(conditional['pairs'], 4);self.assertAlmostEqual(conditional['difference'], 1 / 4)
        wrong = u1.paired_difference(pairs, raw, other, mode='refused_as_wrong', draws=2000)
        self.assertAlmostEqual(wrong['difference'], 1 / 5)
        for result in (fallback, conditional, wrong):
            for name in ('ci90_game_cluster', 'ci90_pair'):
                low, high = result[name];self.assertLessEqual(low, result['difference']);self.assertGreaterEqual(high, result['difference'])
        self.assertEqual(fallback['games'], 4)
        self.assertEqual(u1.paired_difference(pairs, raw, other, mode='fallback', draws=2000), fallback)
        with self.assertRaises(ValueError):u1.paired_difference(pairs, raw[:-1], other, mode='fallback')

    def test_decision_class(self):
        self.assertEqual(u1.decision_class([(-.01, .03), (.001, .05)]), 'positive')
        self.assertEqual(u1.decision_class([(-.05, .019), (-.04, .0)]), 'negative')
        self.assertEqual(u1.decision_class([(-.05, .019), (-.04, .021)]), 'ambiguous')
        self.assertIsNone(u1.decision_class([]))


if __name__ == '__main__':unittest.main()
