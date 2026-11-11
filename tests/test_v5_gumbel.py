"""Tiny literal arithmetic fixtures only; no worlds, actor or engine calls."""
from dataclasses import replace
import math
import random
import unittest

from pokezero import v5_contract as c, v5_gumbel as g

LEGAL = (True, True) + (False,) * 7
LOGITS = (0.,) * 9


def world(*, requested=True, missing=None):
    values = [[None] * 9 for _ in range(9)]
    for a in (0, 1):
        values[a][0] = .8 if a == 0 else -.4
        if requested:values[a][1] = .2 if a == 0 else .4
    if missing is not None:values[missing[0]][missing[1]] = None
    return g.WorldPrices(requested, LEGAL if requested else (False,) * 9,
        (math.log(3),) + (0.,) * 8, values)


class GumbelTests(unittest.TestCase):
    def test_domain_selection_does_not_consume_actor_rng(self):
        recipe = g.Recipe(17)
        before = random.getstate()
        addresses = [c.DecisionAddress(1, frame, owner) for frame in range(256) for owner in (0, 1)]
        self.assertEqual([recipe.selected(a) for a in addresses], [recipe.selected(a) for a in addresses])
        self.assertGreater(sum(map(recipe.selected, addresses)), 70)
        self.assertLess(sum(map(recipe.selected, addresses)), 190)
        address = addresses[0]
        self.assertNotEqual(recipe.world_seed(address, 0), recipe.world_seed(address, 1))
        self.assertNotEqual(recipe.world_seed(address, 0), recipe.world_seed(addresses[1], 0))
        self.assertEqual(set(recipe.pricing_order(address, LOGITS, LEGAL)), {0, 1})
        self.assertEqual(random.getstate(), before)

    def test_all_actions_all_replies_two_worlds_exact_expected_q(self):
        result = g.reduce_completed_q(root_logits=LOGITS, root_legal=LEGAL, worlds=(world(), world()))
        self.assertEqual(result.status, 'LABELED')
        self.assertEqual(result.required_pairs, 8)
        self.assertEqual(result.unknown_pairs, 0)
        target = result.improvement
        target.validate(LEGAL)
        self.assertAlmostEqual(target.completed_q[0], .65)
        self.assertAlmostEqual(target.completed_q[1], -.2)
        self.assertAlmostEqual(target.policy_target[0], 1 / (1 + math.exp(-.85)))
        self.assertTrue(all(q is None for q in target.completed_q[2:]))
        self.assertAlmostEqual(target.root_value, sum(p*q for p, q in zip(target.policy_target[:2], target.completed_q[:2])))

    def test_unknown_positive_mass_refuses_instead_of_renormalizing(self):
        result = g.reduce_completed_q(root_logits=LOGITS, root_legal=LEGAL,
            worlds=(world(missing=(0, 1)), world()))
        self.assertEqual((result.status, result.required_pairs, result.unknown_pairs), ('REFUSED', 8, 1))
        self.assertAlmostEqual(result.unknown_reply_mass, .125)
        self.assertIsNone(result.improvement)

    def test_underflow_reply_is_still_required(self):
        low = replace(world(missing=(0, 1)), opponent_logits=(0., -10000.) + (0.,)*7)
        result = g.reduce_completed_q(root_logits=LOGITS, root_legal=LEGAL, worlds=(low, world()))
        self.assertEqual(result.status, 'REFUSED')
        self.assertEqual(result.unknown_pairs, 1)
        self.assertEqual(result.unknown_reply_mass, 0.)  # count still prevents silently losing this branch

    def test_waiting_opponent_is_one_no_action_branch_not_no_mass(self):
        result = g.reduce_completed_q(root_logits=LOGITS, root_legal=LEGAL,
            worlds=(world(requested=False), world(requested=False)))
        self.assertEqual(result.required_pairs, 4)
        self.assertAlmostEqual(result.improvement.completed_q[0], .8)

    def test_world_or_reply_survivor_only_input_rejected(self):
        with self.assertRaises(ValueError):
            g.reduce_completed_q(root_logits=LOGITS, root_legal=LEGAL, worlds=(world(),))
        with self.assertRaises(ValueError):replace(world(), opponent_requested=False)
        with self.assertRaises(ValueError):replace(world(), joint_values=((0.,)*9,))
        with self.assertRaises(ValueError):replace(world(), opponent_logits=(float('nan'),)*9)
        with self.assertRaises(ValueError):replace(world(), opponent_legal=(1, 1)+(False,)*7)

    def test_outcome_mixing_remains_contract_only(self):
        result = g.reduce_completed_q(root_logits=LOGITS, root_legal=LEGAL, worlds=(world(), world()))
        from test_v5_contract import observation
        obs = observation()
        labels = c.V5Labels(obs.address, terminal_status='COMPLETE', outcome=-1.,
            search_status='LABELED', root_improvement=result.improvement)
        columns = labels.columns(obs)
        self.assertAlmostEqual(columns['search_value_target'], (-1 + result.improvement.root_value)/2)


if __name__ == '__main__':unittest.main()
