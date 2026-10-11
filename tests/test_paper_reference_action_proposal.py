"""Enumerable posterior oracle independent of simulator and champion weights."""
from fractions import Fraction as F
import math
import random
from types import SimpleNamespace
import unittest

from pokezero.mcts_eval.paper_reference import ReferenceRefusal
from pokezero.mcts_eval.paper_reference_action_proposal import (
    defensive_action_proposal, public_guidance_actions)
from pokezero.mcts_eval.paper_reference_particles import bootstrap_population, WeightedAdvance
from pokezero.mcts_eval.paper_reference_runtime import ShowdownWorkerFactory


class ActionProposalTests(unittest.TestCase):
    def model(self):
        # Two actions share the observed public identifier. Chance likelihood
        # differs by hypothesis. Exact posterior B=.112/(.048+.112)=.7.
        return {'A': (F(4,5), (F(1,20), F(3,20), F(4,5)), F(3,10)),
                'B': (F(1,5), (F(3,5), F(1,10), F(3,10)), F(4,5))}

    def exact_target(self):
        values = {h: p*(actions[0]+actions[1])*chance
                  for h, (p, actions, chance) in self.model().items()}
        total = sum(values.values())
        return {h: v/total for h,v in values.items()}

    def test_enumerated_corrected_proposal_equals_exact_posterior_even_wrong_guidance(self):
        self.assertEqual(self.exact_target()['B'], F(7,10))
        for compatible in ((0,1), (0,), (2,), ()):
            recovered = {}
            for h, (p, actions, chance) in self.model().items():
                prior, proposal, _ = defensive_action_proposal((0,1,2), actions, compatible)
                # Enumerate ALL action/chance outcomes, not only our guidance.
                recovered[h] = float(p)*sum(proposal[a]*(prior[a]/proposal[a])*float(chance)
                    for a in (0,1))
                self.assertTrue(all(q > 0 for q in proposal))
            total = sum(recovered.values())
            for h in recovered:
                self.assertAlmostEqual(recovered[h]/total, float(self.exact_target()[h]), places=14)

    def test_missing_importance_weight_would_change_the_posterior(self):
        values = {}
        for h,(p,actions,chance) in self.model().items():
            _,proposal,_ = defensive_action_proposal((0,1,2), actions, (0,1))
            values[h] = float(p)*sum(proposal[a]*float(chance) for a in (0,1))
        self.assertGreater(abs(values['B']/sum(values.values())-.7), .1)

    def test_weighted_particle_engine_matches_oracle_after_two_observations(self):
        for seed in range(3):
            rng, receipt = random.Random(seed), {}
            def initial(i):
                return ('A' if rng.random()<.8 else 'B', 0)
            def advance(p, stage):
                hypothesis,_ = p
                _,actions,chance = self.model()[hypothesis]
                if stage == 0:
                    prior,q,_ = defensive_action_proposal((0,1,2), actions, (0,1))
                    a = rng.choices((0,1,2), weights=q)[0]
                    return WeightedAdvance((hypothesis,1),prior[a]/q[a]) if a in (0,1) else None
                return (hypothesis,2) if rng.random()<float(chance) else None
            population = bootstrap_population(count=12000,stages=2,initial=initial,
                advance=advance,release=lambda p:None,rng=rng,check=lambda:None,receipt=receipt)
            estimate = sum(w for p,w in zip(population,receipt['final_normalized_weights']) if p[0]=='B')
            self.assertAlmostEqual(estimate,.7,delta=.03)
            first=receipt['stages'][0]
            self.assertLess(first['pre_resampling_ess'], first['survivors'])
            self.assertTrue(first['resampled'])

    def test_final_particles_keep_unequal_weights_without_uniformizing(self):
        receipt={}
        particles=bootstrap_population(count=2,stages=1,initial=lambda i:i,
            advance=lambda p,s:WeightedAdvance(p+2, .2 if p==0 else .8),release=lambda p:None,
            rng=random.Random(1),check=lambda:None,receipt=receipt)
        self.assertEqual(particles,[2,3])
        self.assertEqual(receipt['final_normalized_weights'],[.2,.8])
        self.assertAlmostEqual(receipt['stages'][0]['pre_resampling_ess'],1/.68)

    def test_invalid_weight_releases_returned_owner_and_inputs(self):
        for weight in (0.,-1.,float('nan'),float('inf'),True):
            freed=[]
            with self.subTest(weight=weight),self.assertRaises(ReferenceRefusal):
                bootstrap_population(count=1,stages=1,initial=lambda i:1,
                    advance=lambda p,s:WeightedAdvance(2,weight),release=freed.append,
                    rng=random.Random(1),check=lambda:None,receipt={})
            self.assertCountEqual(freed,[1,2])

    def test_zero_mass_guidance_preserves_policy_without_impossibility_claim(self):
        prior,q,row=defensive_action_proposal((0,1),(1.,0.),(1,))
        self.assertEqual(prior,q)
        self.assertEqual(row['guided_fraction'],0.)
        for fraction in (1.,-1.,True,float('nan')):
            with self.subTest(fraction=fraction), self.assertRaises(ReferenceRefusal):
                defensive_action_proposal((0,1),(.5,.5),(0,),guided_fraction=fraction)

    def test_guidance_preserves_all_ambiguous_matches_and_ignores_called_moves(self):
        metadata={'action_candidates':[
            {'action_index':0,'kind':'move','move_id':'surf'},
            {'action_index':1,'kind':'move','move_id':'Surf'},
            {'action_index':2,'kind':'move','move_id':'icebeam'}]}
        obs=SimpleNamespace(metadata=metadata)
        initial=('|turn|4',)
        now=(*initial,'|move|p2a: X|Surf|p1a: Y','|turn|5','|move|p2a: X|Ice Beam|p1a: Y')
        matches,_=public_guidance_actions(obs,(0,1,2),initial,now,'p2')
        self.assertEqual(matches,(0,1))
        called=(*initial,'|move|p2a: X|Surf|p1a: Y|[from] move: Sleep Talk','|turn|5')
        self.assertEqual(public_guidance_actions(obs,(0,1,2),initial,called,'p2')[0],())
        cancelled=(*initial,'|faint|p2a: X','|switch|p2a: Z|Mew|100/100','|turn|5')
        self.assertEqual(public_guidance_actions(obs,(0,1,2),initial,cancelled,'p2')[0],())

    def test_guidance_is_default_off_and_cannot_bypass_missing_particle_opt_in(self):
        self.assertFalse(ShowdownWorkerFactory('c','d','e','s').guide_history_actions)
        for options in ({'guide_history_actions':True},
                        {'history_particles':32,'guide_history_actions':1}):
            with self.assertRaises(ReferenceRefusal):
                ShowdownWorkerFactory('c','d','e','s',**options)
        self.assertTrue(ShowdownWorkerFactory('c','d','e','s',history_particles=32,
            guide_history_actions=True).guide_history_actions)


if __name__=='__main__':
    unittest.main()
