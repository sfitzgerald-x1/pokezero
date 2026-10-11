from fractions import Fraction as F
import itertools
import random
from types import SimpleNamespace
import unittest
from pokezero.mcts_eval.paper_reference import ReferenceRefusal
from pokezero.mcts_eval.paper_reference_anchor_seed import (
    gender_seed_hints,guided_anchor_seed,initial_anchor_weight)
from pokezero.mcts_eval.paper_reference_seed_bank import guided_bank_choice,draw_guided_seed,matches_hints
from pokezero.mcts_eval.paper_reference_runtime import ShowdownWorkerFactory
from pokezero.mcts_eval.paper_reference_particles import HistoryParticle,WeightedAdvance,bootstrap_population

def mon(species,gender=''):return SimpleNamespace(species=species,gender=gender)

class AnchorSeedTests(unittest.TestCase):
    def test_default_off_requires_anchor_constraints_and_particles(self):
        self.assertFalse(ShowdownWorkerFactory('c','d','e','s').guide_anchor_genders)
        for bad in (1,'true',True):
            with self.assertRaises(ReferenceRefusal):
                ShowdownWorkerFactory('c','d','e','s',guide_anchor_genders=bad)
        self.assertTrue(ShowdownWorkerFactory('c','d','e','s',history_particles=32,
            public_anchor_constraints=True,guide_anchor_genders=True).guide_anchor_genders)

    def test_constructor_order_is_only_guidance_no_mon_gender_mutation(self):
        teams={'p1':[mon('Pikachu','M'),mon('Starmie')], 'p2':[mon('Clefable','M'),mon('Shuckle'),mon('Marowak')]}
        source=SimpleNamespace(species_metadata={'starmie':{'gender':'N'}})
        details=[dict(species='shuckle',gender='M'),dict(species='marowak',gender='F')]
        hints=gender_seed_hints(teams,source,details,'p2')
        self.assertEqual([h['ordinal'] for h in hints],[1,2])
        self.assertEqual([h['desired'] for h in hints],[True,False])
        rng=random.Random(3);seed,receipt=guided_anchor_seed(rng,teams,source,details,'p2',lambda:None)
        self.assertTrue(0<=seed<2**32)
        self.assertEqual(teams['p2'][1].gender,'')
        self.assertTrue(all(q>0 for q in receipt['proposal_probabilities']))
        self.assertEqual(initial_anchor_weight({'materialization_seed_proposal':receipt}),receipt['importance_weight'])

    def test_32bit_bank_keeps_original_seed_domain_and_recovered_mass(self):
        hints=[dict(kind='coin',ordinal=1,denominator=2,numerator=1,desired=True)]
        a=random.Random(91);b=random.Random(91);seeds=[a.getrandbits(32) for _ in range(256)]
        expected={};actual={}
        left=guided_bank_choice(seeds,[matches_hints(s,hints) for s in seeds],a,expected)
        right=draw_guided_seed(b,hints,lambda:None,actual,seed_bits=32)
        self.assertEqual(left,right);self.assertEqual(a.getstate(),b.getstate())
        for q in actual['proposal_probabilities']:self.assertGreater(q,0)

    def test_wrong_guidance_rejection_and_once_only_initial_weight_oracle(self):
        # Unequal hidden prior; wrong seed guidance; hard observed predicate
        # depends on BOTH hidden set and seed. Weighted conditional mass must
        # match the original joint, not equal feasible hypotheses.
        recovered=[F(0),F(0)];target=[F(0),F(0)]
        hidden_prior=[F(1,4),F(3,4)]
        for hidden in (0,1):
            target[hidden]=hidden_prior[hidden]*(F(1,2) if hidden==0 else F(1))
            for bank in itertools.product((0,1),repeat=2):
                hits=[seed!=hidden for seed in bank];mass=sum(hits)
                for index,seed in enumerate(bank):
                    q=F(1,2) if not mass else F(1,10)/2+(F(9,10)/mass if hits[index] else 0)
                    w=F(1,2)/q
                    observed=(seed==0 or hidden==1)
                    recovered[hidden]+=hidden_prior[hidden]*F(1,4)*q*w*observed
        self.assertEqual(recovered,target)
        rows={};freed=[]
        def initial(i):return HistoryParticle(object(),i,{},(),initial_importance_weight=(2.,6.)[i])
        def advance(p,stage):
            child=HistoryParticle(object(),p.ancestor,p.anchor_receipt,(*p.steps,stage))
            return WeightedAdvance(child,p.initial_importance_weight)
        rng=SimpleNamespace(choices=lambda candidates,weights,k:[candidates[0],candidates[1]])
        population=bootstrap_population(count=2,stages=2,initial=initial,advance=advance,
            release=freed.append,rng=rng,check=lambda:None,receipt=rows)
        self.assertEqual(rows['stages'][0]['survivor_importance_weights'],[2.,6.])
        self.assertEqual(rows['stages'][1]['survivor_importance_weights'],[1.,1.])
        self.assertEqual(len(population),2)

    def test_no_gender_hints_preserves_original_single_seed_draw(self):
        teams={'p1':[mon('Pikachu','M')],'p2':[mon('Clefable','F')]}
        rng=random.Random(2);other=random.Random(2)
        seed,receipt=guided_anchor_seed(rng,teams,SimpleNamespace(species_metadata={}),
            [dict(species='clefable',gender='F')],'p2',lambda:None)
        self.assertEqual(seed,other.getrandbits(32));self.assertIsNone(receipt)
        self.assertEqual(rng.getstate(),other.getstate())

    def test_unqualified_initial_ratio_refuses(self):
        self.assertEqual(initial_anchor_weight({}),1.)
        for weight in (0.,True,float('nan'),float('inf')):
            with self.assertRaises(ReferenceRefusal):
                initial_anchor_weight({'materialization_seed_proposal':dict(importance_weight=weight)})
