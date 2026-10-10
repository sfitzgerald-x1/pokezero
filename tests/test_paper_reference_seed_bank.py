from fractions import Fraction as F
import itertools
import json
from pathlib import Path
import random
import subprocess
from types import SimpleNamespace
import unittest

from _showdown_root import requires_showdown, showdown_root
from pokezero.mcts_eval.paper_reference import ReferenceRefusal, SamplingDeadlineExceeded
from pokezero.mcts_eval.paper_reference_seed_bank import (
    native_seed_outputs, guided_bank_choice, draw_guided_seed, chance_hints,
    matches_hints, validate_native_trace, critical_alignment_hints)
from pokezero.mcts_eval.paper_reference_runtime import ShowdownWorkerFactory
from pokezero.mcts_eval.paper_reference_chance_pool import fixed_chance_pool
from pokezero.mcts_eval.paper_reference_particles import WeightedAdvance


class SeedBankOracleTests(unittest.TestCase):
    def test_enumerate_all_banks_wrong_hints_and_duplicate_seeds_preserves_original_mass(self):
        # The guidance deliberately favors a wrong future outcome. Exact target
        # mass is recovered for every hidden hypothesis, not only feasible ones.
        original = {0:F(1,4), 1:F(3,4)}
        for hidden in (0,1):
            recovered = F(0)
            for bank in itertools.product(original, repeat=3):
                probability = F(1)
                for seed in bank: probability *= original[seed]
                compatible = [seed != hidden for seed in bank]
                mass = sum(compatible)
                for index, seed in enumerate(bank):
                    q = F(1,3) if not mass else F(1,10)/3 + (F(9,10)/mass if compatible[index] else 0)
                    weight = F(1,3)/q
                    recovered += probability*q*weight*(seed==hidden)
            self.assertEqual(recovered, original[hidden])

    def test_implementation_weights_recover_every_index_and_duplicates_count(self):
        class Pick:
            def __init__(self,i):self.i=i
            def choices(self, indices, weights, k):self.q=weights;return [self.i]
        for hits in ([False]*3, [True]*3, [True, False, True]):
            recovery = [0.,0.,0.]
            for i in range(3):
                rng=Pick(i);receipt={}
                seed,weight=guided_bank_choice([7,7,9],hits,rng,receipt)
                self.assertEqual(seed,[7,7,9][i])
                self.assertTrue(all(p>0 for p in rng.q))
                recovery[i]=rng.q[i]*weight
            for p in recovery:self.assertAlmostEqual(p,1/3)

    def test_weighted_fixed_native_pool_selects_by_weights_not_uniform(self):
        owners=[object(),object()];freed=[];receipt={}
        rng=SimpleNamespace(choices=lambda indices,weights,k:[0])
        result=fixed_chance_pool(count=4,trial=lambda i:
            WeightedAdvance(owners[i],(2.,6.)[i]) if i<2 else None,
            release=freed.append,rng=rng,check=lambda:None,receipt=receipt)
        self.assertIs(result.particle,owners[0]);self.assertEqual(result.weight,2.)
        self.assertEqual(freed,[owners[1]])
        self.assertEqual(receipt['survivor_importance_weights'],[2.,6.])

    def test_invalid_chance_weight_refuses_and_releases_owner(self):
        for weight in (True,0.,float('nan'),float('inf')):
            owner=object();freed=[]
            with self.subTest(weight=weight),self.assertRaises(ReferenceRefusal):
                fixed_chance_pool(count=1,trial=lambda i:WeightedAdvance(owner,weight),
                    release=freed.append,rng=random.Random(1),check=lambda:None,receipt={})
            self.assertEqual(freed,[owner])

    def test_deadline_partial_bank_is_never_sampled(self):
        receipt={}; calls=0
        def check():
            nonlocal calls
            calls+=1
            if calls==2: raise SamplingDeadlineExceeded('partial bank')
        with self.assertRaises(SamplingDeadlineExceeded):
            draw_guided_seed(random.Random(1),[],check,receipt)
        self.assertFalse(receipt['bank_complete'])
        self.assertNotIn('selected_seed',receipt)

    def test_trace_hints_are_heuristic_and_trace_drift_is_not_accepted(self):
        history=('|turn|4',)
        expected=history+('|move|p1a: Lugia|Psychic|p2a: Shuckle','|-crit|p2a: Shuckle',
            '|-damage|p2a: Shuckle|149/198 slp','|turn|5')
        d=dict(source='p1',target='p2',move='psychic',target_hp=198,target_max_hp=198)
        trace=[dict(ordinal=3,chance=dict(numerator=1,denominator=16),damage=d),
            dict(ordinal=4,chance=None,range=[16,None],damage=dict(d,base_damage=27,critical=False))]
        hints=chance_hints(trace,history,expected)
        self.assertEqual(hints[0],dict(kind='coin',ordinal=3,numerator=1,denominator=16,desired=True))
        self.assertEqual(len(hints),1, 'critical damage must not be extrapolated by doubling')
        self.assertEqual(critical_alignment_hints(trace,history,expected),hints)
        trace[1]['range']=[16]
        self.assertEqual(chance_hints(trace,history,expected),hints)
        trace[1]['damage'].update(base_damage=53,critical=True)
        hints=chance_hints(trace,history,expected)
        self.assertEqual(hints[1]['support'],[6,7])
        self.assertEqual(critical_alignment_hints(trace,history,expected),[])
        with self.assertRaises(ReferenceRefusal):validate_native_trace([dict(ordinal=1,value=-1)],0)
        self.assertTrue(matches_hints(0,[]))

    def test_default_off_and_cannot_combine_with_expensive_pool_allocation(self):
        self.assertFalse(ShowdownWorkerFactory('c','d','e','s').guide_history_chance)
        for value,particles,batch in ((1,32,False),(True,0,False),(True,32,True)):
            with self.assertRaises(ReferenceRefusal):
                ShowdownWorkerFactory('c','d','e','s',history_particles=particles,
                    batch_history_chance=batch,guide_history_chance=value)


@requires_showdown()
class NativeRngParityTests(unittest.TestCase):
    def test_original_sha_seed_mapping_and_32_outputs_match_pinned_native_rng(self):
        from pokezero.local_showdown import showdown_seed_from_int
        rng=random.Random(13); seeds=[0,1,2**64-1,*[rng.getrandbits(64) for _ in range(16)]]
        module=str(Path(showdown_root())/'dist/sim/prng.js')
        code='const {PRNG}=require('+json.dumps(module)+');console.log(JSON.stringify('+\
             json.dumps([showdown_seed_from_int(s) for s in seeds])+\
             '.map(seed=>{const p=new PRNG(seed);return Array.from({length:32},()=>p.rng.next())})));'
        completed=subprocess.run(['node','-e',code],check=True,text=True,capture_output=True)
        self.assertEqual(json.loads(completed.stdout),[native_seed_outputs(s,32) for s in seeds])


if __name__=='__main__':unittest.main()
