"""mc_root_labeller and mc_root_report: sampling, draws, completed-Q, the caching env's cache, censoring and the U1 baseline report.
The labeller's end-to-end parity against the plain env (mc_root_labeller.parity) needs a Showdown runtime and runs pod-side."""
import json
import math
import unittest
from types import SimpleNamespace as NS
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
        for name in ('_reset', '_restore_local_snapshot_state', 'reseed_simulator_rng', '_submit_step_choices'):
            patches.append(mock.patch.object(LocalShowdownEnv, name, lambda self_, *a, **k:None))
        for p in patches:p.start();self.addCleanup(p.stop)
        return env, calls

    def test_a_state_is_reused_within_a_position_and_recomputed_after_any_change(self):
        env, calls = self.make()
        a = env._state_for_player('p1');self.assertIs(env._state_for_player('p1'), a)   # the step's observation, then the choices
        b = env._state_for_player('p2');self.assertIsNot(a, b);self.assertEqual(len(calls), 2)
        for change in (lambda:env._submit_step_choices(choices={}, payload={}), lambda:env._restore_local_snapshot_state(None),
                       lambda:env._reset(seed=1), lambda:env.reseed_simulator_rng(5)):
            with self.subTest(change=change):
                before = len(calls);change();env._state_for_player('p1');env._state_for_player('p1')
                self.assertEqual(len(calls), before + 1)   # one fresh normalization per position
        n = len(calls);env._state_for_player('p1', root_puct_branch_observation=True);env._state_for_player('p1', root_puct_branch_observation=True)
        self.assertEqual(len(calls), n + 2);self.assertEqual(calls[-1], ('p1', True))
        for name in ('materialize_public_world', 'snapshot_for_search', 'restore_search_snapshot', 'step_from_search_snapshot',
                     'reset_with_start_override', 'snapshot_actionable_boundary', 'materialize_scenario_state'):
            with self.assertRaises(NotImplementedError):getattr(env, name)()   # paths the labeller never takes are refused


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


class FakeEnv:
    """A tiny two-seat game for the rollout machinery: both seats requested every round, terminal after `length` steps; p1 wins
    when p1's actions plus the reseeded seed sum to an even number. restore refuses while a rollout still owns the env."""
    LEGAL = (True,) * 3 + (False,) * 6

    def __init__(self, length=3, strict=True):
        self.length, self.t, self.hist, self.seed, self.busy, self.log, self.firsts, self.fresh = length, 0, [], None, False, [], [], False
        self.strict = strict   # run_root: a restore while a rollout still owns the env is a sharing bug

    def restore(self, snap):
        assert not (self.strict and self.busy), 'an env restored while a rollout still owns it'
        self.t, self.hist, self.busy, self.fresh = snap['t'], list(snap['hist']), True, True

    def reseed_simulator_rng(self, seed):self.seed = seed;self.log.append(seed)

    def terminal(self):
        if self.t < self.length:return None
        return NS(winner='p1' if (sum(self.hist) + self.seed) % 2 == 0 else 'p2', capped=False)

    def requested_players(self):return () if self.terminal() else ('p1', 'p2')

    def observe(self, p):return NS(legal_action_mask=self.LEGAL, hist=tuple(self.hist), player=p)

    def step(self, actions):
        if self.fresh:self.firsts.append((self.seed, dict(actions)));self.fresh = False
        self.hist.append(actions['p1']);self.t += 1
        if self.terminal() is not None:self.busy = False
        return NS(observations={p: self.observe(p) for p in self.requested_players()})


class FakeModel:
    def evaluate(self, observations):
        probs = np.array([[1 / 3] * 3 + [0.] * 6 for _ in observations]);return probs, np.array([sum(o.hist) / 10 for o in observations])

    def probabilities(self, observations):return self.evaluate(observations)[0]


class RolloutMachineryTests(unittest.TestCase):
    def root(self, K):return dict(ours='p1', opp='p2', actions=[0, 1, 2], root_seed=99, opp_replies=[(2 * k) % 3 for k in range(K)],
                                  opp_policy=np.array([.5, .5] + [0.] * 7))

    def test_siblings_share_their_rollouts_seed_and_root_reply_and_no_env_serves_two_rollouts(self):
        K = 5;envs = [FakeEnv() for _ in range(4)];root = self.root(K);m = FakeModel()
        outcomes, decisions = L.run_root(envs, dict(ours=m, opp=m), root, dict(t=0, hist=[]), K, batched=True)
        self.assertEqual(outcomes.shape, (3, K));self.assertFalse(np.isnan(outcomes).any());self.assertTrue((decisions > 0).all())
        firsts = [f for e in envs for f in e.firsts];self.assertEqual(len(firsts), 3 * K)
        by_seed = {}
        for seed, acts in firsts:by_seed.setdefault(seed, []).append(acts)
        self.assertEqual(len(by_seed), K)   # one simulator seed per k, shared by its three siblings
        for k in range(K):
            seed = int(np.random.default_rng([99, k]).integers(1, 2**31 - 1))
            self.assertEqual(sorted(a['p1'] for a in by_seed[seed]), [0, 1, 2])
            self.assertEqual({a['p2'] for a in by_seed[seed]}, {root['opp_replies'][k]})   # the same root reply b_k
        single, _ = L.run_root([FakeEnv()], dict(ours=m, opp=m), root, dict(t=0, hist=[]), K, batched=False)
        self.assertTrue(np.array_equal(single, outcomes))   # unbatched plays the same rollouts

    def test_the_depth1_baseline_weights_replies_and_draws_from_its_own_seed_bank(self):
        env = FakeEnv(strict=False);m = FakeModel();root = self.root(2)
        out = L.depth1_baseline(env, dict(t=0, hist=[]), root, dict(ours=m, opp=m), draws=2)
        self.assertEqual(out['replies'], [0, 1])
        self.assertEqual([round(q, 9) for q in out['completed_q']], [0., .1, .2])   # V = p1's first action / 10 at the next decision
        bank = {int(np.random.default_rng([99, 7_000_000 + c]).integers(1, 2**31 - 1)) for c in range(2)}
        rollouts = {int(np.random.default_rng([99, k]).integers(1, 2**31 - 1)) for k in range(256)}
        self.assertEqual(set(env.log), bank);self.assertFalse(bank & rollouts)
        waiting = dict(root, opp_policy=None)
        self.assertEqual(L.depth1_baseline(FakeEnv(strict=False), dict(t=0, hist=[]), waiting, dict(ours=m, opp=m), draws=1)['replies'], [None])


class ReplayEnv:
    """Both seats requested every round with request 'req-<round>'; ends after 6 rounds."""

    def __init__(self):self.t = 0

    def reset(self, *, seed):self.t = 0

    def terminal(self):return None if self.t < 6 else NS(winner='p1', capped=False)

    def requested_players(self):return ('p1', 'p2')

    @property
    def _latest_requests(self):return {'p1': f'req-{self.t}', 'p2': f'req-{self.t}'}

    def observe(self, p):return NS(legal_action_mask=(True,) * 3 + (False,) * 6)

    def step(self, actions):self.t += 1


class FakeFpd:
    class Mismatch(Exception):pass

    @staticmethod
    def _request_body(line):return line

    @staticmethod
    def choice_action(env, player, choice):return int(choice.split()[1])


class FoulPlayRootTests(unittest.TestCase):
    def game(self, **over):
        g = dict(seed=1, ours='p1', fp='p2', our_actions=[0, 1, 2, 0, 1], our_requests=[json.dumps(f'req-{t}') for t in range(5)],
                 journal={t: dict(round=t, action=t % 3, choice=f'act {t % 3}') for t in range(5)})
        g.update(over);return g

    def test_the_replay_reaches_the_target_and_refuses_to_leave_the_recording(self):
        self.assertTrue(L.foulplay_root(ReplayEnv(), self.game(), 3, FakeFpd))
        bad = self.game(our_requests=[json.dumps('req-0'), json.dumps('other')] + [json.dumps(f'req-{t}') for t in range(2, 5)])
        with self.assertRaisesRegex(FakeFpd.Mismatch, 'request differs'):L.foulplay_root(ReplayEnv(), bad, 3, FakeFpd)
        journal = {t: dict(round=t, action=t % 3, choice='act 2' if t == 1 else f'act {t % 3}') for t in range(5)}
        with self.assertRaisesRegex(FakeFpd.Mismatch, 'decode'):L.foulplay_root(ReplayEnv(), self.game(journal=journal), 3, FakeFpd)
        with self.assertRaisesRegex(FakeFpd.Mismatch, 'without a journal entry'):
            L.foulplay_root(ReplayEnv(), self.game(journal={0: dict(round=0, action=0, choice='act 0')}), 3, FakeFpd)


if __name__ == '__main__':
    unittest.main()
