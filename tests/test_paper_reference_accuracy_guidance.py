"""Accuracy hints are proposals, never relaxed public-history constraints."""
from fractions import Fraction as F
import itertools
import math
import random
import unittest

from pokezero.mcts_eval.paper_reference import ReferenceRefusal
from pokezero.mcts_eval.paper_reference_factory import PublicRootWorldFactory
from pokezero.mcts_eval.paper_reference_runtime import ShowdownWorkerFactory
from pokezero.mcts_eval.paper_reference_seed_bank import (
    accuracy_hints, chance_hints, guided_bank_choice, matches_hint_bank, matches_hints,
    native_seed_outputs)


def context_trace(numerator=85, ordinal=3):
    return [dict(ordinal=ordinal, chance=dict(numerator=numerator, denominator=100),
        accuracy=dict(source='p1', target='p2', source_ident='p1a: Lanturn',
            target_ident='p2a: Cacturne', move='toxic', numerator=numerator, denominator=100))]


class AccuracyGuidanceTests(unittest.TestCase):
    history = ('|upkeep', '|turn|4')
    move = '|move|p1a: Lanturn|Toxic|p2a: Cacturne'

    def test_observed_miss_and_hit_are_context_bound_and_default_off(self):
        for tag, desired in (('|[miss]', False), ('', True)):
            expected = self.history + (self.move+tag, '|turn|5')
            hints = accuracy_hints(context_trace(), self.history, expected)
            self.assertEqual(hints, [dict(kind='coin', ordinal=3, numerator=85,
                denominator=100, desired=desired)])
            self.assertEqual(chance_hints(context_trace(), self.history, expected), [])
            self.assertEqual(chance_hints(context_trace(), self.history, expected, guide_accuracy=True), hints)

    def test_ambiguous_calls_bad_thresholds_wrong_ident_and_later_turn_are_not_guided(self):
        expected = self.history+(self.move+'|[miss]', '|turn|5')
        for trace in ([*context_trace(), *context_trace(85,4)],
                      context_trace(True), context_trace(float('nan')), context_trace(-1),
                      context_trace(101), context_trace(85,0), context_trace(85,257),
                      [dict(ordinal=3,chance=dict(numerator=85,denominator=100))]):
            self.assertEqual(accuracy_hints(trace,self.history,expected), [])
        trace = context_trace(); trace[0]['accuracy']['source_ident']='p1a: other Lanturn'
        self.assertEqual(accuracy_hints(trace,self.history,expected), [])
        trace = context_trace(); trace[0]['chance']['numerator']=10
        self.assertEqual(accuracy_hints(trace,self.history,expected), [])
        trace = context_trace(); trace[0]['accuracy']['source']='p2'
        self.assertEqual(accuracy_hints(trace,self.history,expected), [])
        with self.assertRaisesRegex(ReferenceRefusal,'boolean opt-in'):
            chance_hints(context_trace(), self.history, expected, guide_accuracy=1)
        for suffix in ((self.move,self.move), ('|turn|5',self.move+'|[miss]'),
                       ('|move|p1a: Lanturn|Toxic||[still]',)):
            self.assertEqual(accuracy_hints(context_trace(),self.history,self.history+suffix), [])
        with self.assertRaisesRegex(ReferenceRefusal,'not an exact public prefix'):
            accuracy_hints(context_trace(), self.history, ('different',))

    def test_fractional_accuracy_scalar_vector_match_native_coin_projection(self):
        seeds = [random.Random(i).getrandbits(64) for i in range(64)]
        for numerator in (0,85/3,85,100):
            for desired in (False,True):
                hints = accuracy_hints(context_trace(numerator), self.history,
                    self.history+(self.move+('' if desired else '|[miss]'),))
                expected = [(native_seed_outputs(seed,3)[-1]*100//(1 << 32) < numerator)==desired
                    for seed in seeds]
                self.assertEqual([matches_hints(seed,hints) for seed in seeds], expected)
                self.assertEqual(matches_hint_bank(seeds,hints), expected)

    def test_wrong_miss_hint_and_zero_mass_keep_every_original_bank_index(self):
        class Pick:
            def __init__(self,i): self.i=i
            def choices(self,indices,weights,k): self.q=weights; return [self.i]
        seeds = [7,7,9,11]
        for compatible in ([False]*4, [True]*4, [True,False,True,False]):
            for i in range(4):
                pick=Pick(i); receipt={}
                seed,weight=guided_bank_choice(seeds,compatible,pick,receipt)
                self.assertEqual(seed,seeds[i])
                self.assertTrue(all(q>0 and math.isfinite(q) for q in pick.q))
                self.assertAlmostEqual(pick.q[i]*weight,1/4)

    def test_two_stage_full_bank_weight_identity_including_wrong_hints(self):
        # Enumerate original banks and every selected index in two stages.
        # Guidance deliberately favors a seed that fails the exact potential.
        original={0:F(1,4),1:F(3,4)}
        recovered=F(0)
        for bank1 in itertools.product(original,repeat=2):
            for bank2 in itertools.product(original,repeat=2):
                bank_probability=F(1)
                for seed in (*bank1,*bank2): bank_probability*=original[seed]
                for i in range(2):
                    for j in range(2):
                        qs=[]
                        for bank,index in ((bank1,i),(bank2,j)):
                            hits=[s==0 for s in bank];mass=sum(hits)
                            qs.append(F(1,2) if not mass else F(1,10)/2+
                                (F(9,10)/mass if hits[index] else 0))
                        q1,q2=qs
                        potential=(bank1[i]==1 and bank2[j]==1)
                        recovered+=bank_probability*q1*q2*(F(1,2)/q1)*(F(1,2)/q2)*potential
        self.assertEqual(recovered,original[1]**2)

    def test_runtime_and_factory_require_explicit_chance_guidance(self):
        self.assertFalse(ShowdownWorkerFactory('c','d','e','s').guide_history_accuracy)
        good=ShowdownWorkerFactory('c','d','e','s',history_particles=32,
            guide_history_chance=True,guide_history_accuracy=True)
        self.assertTrue(good.guide_history_accuracy)
        for bad,chance in ((True,False),(1,True),('yes',True)):
            with self.subTest(bad=bad,chance=chance):
                with self.assertRaisesRegex(ReferenceRefusal,'accuracy guidance'):
                    ShowdownWorkerFactory('c','d','e','s',history_particles=32,
                        guide_history_chance=chance,guide_history_accuracy=bad)
                with self.assertRaisesRegex(ReferenceRefusal,'accuracy guidance'):
                    PublicRootWorldFactory(env=None,state=None,observation=None,evaluator=None,set_source=None,
                        history_particles=32,guide_history_chance=chance,guide_history_accuracy=bad)


if __name__ == '__main__':
    unittest.main()
