"""Exact chance averaging oracle, posterior tests, censorship and ownership."""
from fractions import Fraction as F
import itertools
import random
import unittest

from pokezero.mcts_eval.paper_reference import ReferenceRefusal,SamplingDeadlineExceeded
from pokezero.mcts_eval.paper_reference_chance_pool import fixed_chance_pool,validate_chance_pool_count
from pokezero.mcts_eval.paper_reference_particles import bootstrap_population
from pokezero.mcts_eval.paper_reference_runtime import ShowdownWorkerFactory


class FixedChancePoolTests(unittest.TestCase):
    def test_enumeration_preserves_original_joint_chance_and_observation_likelihood(self):
        # Enumerate every ordered K=3 pool, and every uniform matching choice.
        # Chance outputs None (reject), HP10, HP20. The original target has
        # unnormalized masses .2 and .1; not a uniform distribution of HP.
        chances={None:F(7,10),10:F(1,5),20:F(1,10)}
        recovered={10:F(0),20:F(0)}
        for pool in itertools.product(chances,repeat=3):
            probability=F(1)
            for value in pool:probability*=chances[value]
            matches=[x for x in pool if x is not None]
            if not matches:continue
            for selected in matches:
                recovered[selected]+=probability*F(len(matches),3)/len(matches)
        self.assertEqual(recovered,{10:F(1,5),20:F(1,10)})

    def test_weighted_population_recovers_hidden_likelihood_not_any_match_probability(self):
        for seed in range(3):
            rng=random.Random(seed);receipt={}
            def initial(i):return 'A' if rng.random()<.8 else 'B'
            def advance(h,stage):
                chance=.1 if h=='A' else .8
                return fixed_chance_pool(count=4,
                    trial=lambda ordinal:h if rng.random()<chance else None,
                    release=lambda p:None,rng=rng,check=lambda:None,receipt={})
            particles=bootstrap_population(count=16000,stages=1,initial=initial,advance=advance,
                release=lambda p:None,rng=rng,check=lambda:None,receipt=receipt)
            weighted_B=sum(w for h,w in zip(particles,receipt['final_normalized_weights']) if h=='B')
            # Original posterior B=.16/(.08+.16)=2/3. Treating every
            # any-match pool equally would use 1-(1-p)^4 and gives ~.42.
            self.assertAlmostEqual(weighted_B,2/3,delta=.025)
            equal_B=sum(h=='B' for h in particles)/len(particles)
            self.assertLess(equal_B,.48)

    def test_completed_fixed_pool_checks_all_trials_even_after_a_match(self):
        calls=[];freed=[];receipt={}
        values=[object(),None,object(),None]
        def trial(i):calls.append(i);return values[i]
        result=fixed_chance_pool(count=4,trial=trial,release=freed.append,
            rng=random.Random(1),check=lambda:None,receipt=receipt)
        self.assertEqual(calls,[0,1,2,3]);self.assertEqual(result.weight,.5)
        self.assertTrue(receipt['pool_complete']);self.assertFalse(receipt['partial_pool_used'])
        self.assertEqual(len(freed),1);self.assertNotIn(result.particle,freed)

    def test_partial_deadline_never_returns_a_first_match_or_partial_estimate(self):
        owner=object();freed=[];receipt={};calls=0
        def check():
            nonlocal calls
            calls+=1
            if calls==4:raise SamplingDeadlineExceeded('after two results')
        with self.assertRaises(SamplingDeadlineExceeded) as failure:
            fixed_chance_pool(count=4,trial=lambda i:owner if i==0 else None,
                release=freed.append,rng=random.Random(1),check=check,receipt=receipt)
        self.assertEqual(freed,[owner]);self.assertFalse(receipt['pool_complete'])
        self.assertIsNone(receipt['observation_likelihood_estimate'])
        self.assertEqual(failure.exception.sampling_diagnostic['chance_pool']['completed'],2)

    def test_native_failure_is_not_zero_potential_and_releases_previous_match(self):
        owner=object();freed=[]
        def trial(i):
            if i==1:raise RuntimeError('bridge failed')
            return owner
        with self.assertRaisesRegex(RuntimeError,'bridge failed'):
            fixed_chance_pool(count=4,trial=trial,release=freed.append,
                rng=random.Random(1),check=lambda:None,receipt={})
        self.assertEqual(freed,[owner])

    def test_zero_matches_do_not_retry_and_one_trial_preserves_rng_stream(self):
        calls=[]
        result=fixed_chance_pool(count=4,trial=lambda i:calls.append(i),
            release=lambda p:None,rng=random.Random(1),check=lambda:None,receipt={})
        self.assertIsNone(result);self.assertEqual(calls,[0,1,2,3])
        rng=random.Random(4);before=rng.getstate();owner=object()
        result=fixed_chance_pool(count=1,trial=lambda i:owner,release=lambda p:None,
            rng=rng,check=lambda:None,receipt={})
        self.assertEqual(rng.getstate(),before);self.assertEqual(result.weight,1.)

    def test_opt_in_bounded_integer_requires_particles(self):
        self.assertEqual(ShowdownWorkerFactory('c','d','e','s').history_chance_pool,1)
        for bad in (True,False,0,9,4.,'4'):
            with self.subTest(bad=bad),self.assertRaises(ReferenceRefusal):
                validate_chance_pool_count(bad,particles=32)
        with self.assertRaises(ReferenceRefusal):validate_chance_pool_count(4,particles=0)
        self.assertEqual(validate_chance_pool_count(4,particles=32),4)


if __name__=='__main__':unittest.main()
