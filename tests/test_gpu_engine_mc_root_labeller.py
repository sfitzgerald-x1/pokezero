"""mc_root_labeller and mc_root_report: sampling, draws, completed-Q, the caching env's cache, censoring and the U1 baseline report.
The labeller's end-to-end parity against the plain env (mc_root_labeller.parity) needs a Showdown runtime and runs pod-side."""
import math
import unittest
from unittest import mock

import numpy as np

from gpu_engine import mc_root_labeller as L
from gpu_engine import mc_root_report as R


class SamplingTests(unittest.TestCase):
    def test_sample_is_the_inverse_cdf_over_the_legal_actions(self):
        probs = np.array([.5, .2, .3] + [0.] * 6);legal = np.array([True, False, True] + [False] * 6)   # renormalized: .625, .375
        self.assertEqual([L.sample(probs, legal, u) for u in (0., .6, .63, .99)], [0, 0, 2, 2])
        self.assertEqual(L.sample(np.full(9, 1 / 9), np.ones(9, bool), .999999), 8)

    def test_draws_are_deterministic_per_root_rollout_ply_and_seat(self):
        self.assertEqual(L.draw(7, 3, 2, 1), L.draw(7, 3, 2, 1))
        self.assertEqual(len({L.draw(7, 3, 2, 1), L.draw(7, 3, 2, 0), L.draw(7, 4, 2, 1), L.draw(8, 3, 2, 1)}), 4)
        self.assertTrue(all(0. <= L.draw(1, k, 0, 0) < 1. for k in range(20)))

    def test_completed_q_weights_replies_by_the_opponent_policy(self):
        pi = np.zeros(9);pi[1], pi[4] = .75, .25
        values = {(0, 1): [1., 0.], (0, 4): [-1.], (2, 1): [0.], (2, 4): [1., 1.]}
        self.assertEqual(L.completed_q(values, pi, [0, 2], [1, 4]), [.75 * .5 + .25 * -1., .75 * 0. + .25 * 1.])
        self.assertEqual(L.completed_q({(0, None): [.2, .4], (3, None): [-.1]}, None, [0, 3], [None]), [math.fsum([.2, .4]) / 2, -.1])


class CachingEnvTests(unittest.TestCase):
    """CachingEnv on the real LocalShowdownEnv class, its battle methods stubbed: a normalized state is reused until the position
    changes (a step, reset, restore or reseed), and a root-PUCT branch observation is never cached."""

    def make(self):
        from pokezero.local_showdown import LocalShowdownEnv
        cls = L.caching_env_class();env = cls.__new__(cls);calls = []
        def state(self_, player, *, root_puct_branch_observation=False):
            calls.append((player, root_puct_branch_observation));return object()
        patches = [mock.patch.object(LocalShowdownEnv, '_state_for_player', state)]
        for name in ('reset', 'restore', 'reseed_simulator_rng', '_submit_step_choices'):
            patches.append(mock.patch.object(LocalShowdownEnv, name, lambda self_, *a, **k:None))
        for p in patches:p.start();self.addCleanup(p.stop)
        return env, calls

    def test_a_state_is_reused_within_a_position_and_recomputed_after_any_change(self):
        env, calls = self.make()
        a = env._state_for_player('p1');self.assertIs(env._state_for_player('p1'), a)   # the step's observation, then the choices
        b = env._state_for_player('p2');self.assertIsNot(a, b);self.assertEqual(len(calls), 2)
        for change in (lambda:env._submit_step_choices(choices={}, payload={}), lambda:env.restore(None), lambda:env.reset(seed=1),
                       lambda:env.reseed_simulator_rng(5)):
            with self.subTest(change=change):
                before = len(calls);change();env._state_for_player('p1');env._state_for_player('p1')
                self.assertEqual(len(calls), before + 1)   # one fresh normalization per position
        n = len(calls);env._state_for_player('p1', root_puct_branch_observation=True);env._state_for_player('p1', root_puct_branch_observation=True)
        self.assertEqual(len(calls), n + 2);self.assertEqual(calls[-1], ('p1', True))


class ReportTests(unittest.TestCase):
    def test_capped_rollouts_are_censored_not_outcomes(self):
        o = [[1., L.CAPPED, 1., -1.], [1., 1., L.CAPPED, -1.]]
        self.assertTrue(np.isnan(R.censored(o)).sum() == 2)
        s = R.pair_stats(np.array(o), 4)
        self.assertEqual(len(s['se_pair']), 1)   # the pair uses the two rollouts neither sibling lost: [1, -1] against [1, -1]
        self.assertEqual(s['separable'], [False])

    def test_the_u1_baseline_pairs_the_value_choice_against_the_policy_choice_by_battle(self):
        def root(seed, policy_best, value_best, means):
            outcomes = [[m] * 4 for m in means]
            q = [0.] * len(means);q[value_best] = 1.;policy = [0.] * 9;policy[policy_best] = 1.
            return dict(seed=seed, actions=list(range(len(means))), outcomes=outcomes, policy=policy, baseline=dict(completed_q=q))
        rows = [root(1, 0, 1, [0., .5]), root(1, 0, 0, [.2, .9]), root(2, 1, 0, [.4, .1])]
        out = R.baseline_report(rows)
        self.assertEqual((out['roots'], out['same_choice']), (3, 1))
        self.assertAlmostEqual(out['value_minus_policy']['mean'], (.5 + 0. + .3) / 3, places=5)
        self.assertEqual(out['value_minus_policy']['clusters'], 2)
        self.assertEqual(out['value_minus_policy_where_different']['roots'], 2)
        self.assertEqual(R.baseline_report([dict(seed=1, actions=[0], outcomes=[[1.]], policy=None, baseline=None)])['roots'], 0)


if __name__ == '__main__':
    unittest.main()
