import random
import unittest
from pokezero.mcts_eval.paper_reference import SamplingDeadlineExceeded
from pokezero.mcts_eval.paper_reference_seed_bank import (
    matches_hints,matches_hint_bank,draw_guided_seed,guided_bank_choice)
from pokezero.mcts_eval.paper_reference_stage_memo import StageIdentityMemo

class VectorSeedTests(unittest.TestCase):
    def test_full_bank_scalar_equivalence_edges_duplicates_and_wrong_hints(self):
        rng=random.Random(129)
        seeds=[0,2**64-1,7,7,*[rng.getrandbits(64) for _ in range(252)]]
        for hints in ([],[dict(kind='coin',ordinal=2,denominator=16,numerator=1,desired=False)],
            [dict(kind='damage',ordinal=3,support=[8,9,10]),
             dict(kind='coin',ordinal=6,denominator=16,numerator=1,desired=True)],
            [dict(kind='integer_range',ordinal=256,lower=-3,upper=7,support=[-1,0,6])]):
            self.assertEqual(matches_hint_bank(seeds,hints),[matches_hints(s,hints) for s in seeds])

    def test_full_draw_same_proposal_weights_selected_seed_and_next_rng(self):
        hints=[dict(kind='damage',ordinal=3,support=[8,9,10]),
               dict(kind='coin',ordinal=6,denominator=16,numerator=1,desired=False)]
        for count in (1,16,256,1024):
            for seed in (0,93,20261008):
                a=random.Random(seed);b=random.Random(seed)
                seeds=[a.getrandbits(64) for _ in range(count)]
                expected={};actual={}
                left=guided_bank_choice(seeds,[matches_hints(s,hints) for s in seeds],a,expected)
                right=draw_guided_seed(b,hints,lambda:None,actual,bank_size=count)
                self.assertEqual(left,right)
                self.assertEqual(a.getstate(),b.getstate())
                for key in expected:self.assertEqual(expected[key],actual[key])

    def test_deadline_during_vector_projection_never_samples_partial_bank(self):
        receipt={};calls=0
        def check():
            nonlocal calls
            calls+=1
            if calls==20:raise SamplingDeadlineExceeded('projection timeout')
        with self.assertRaises(SamplingDeadlineExceeded):
            draw_guided_seed(random.Random(3),
                [dict(kind='damage',ordinal=9,support=[4])],check,receipt)
        self.assertFalse(receipt['bank_complete'])
        self.assertNotIn('selected_seed',receipt)

class StageMemoTests(unittest.TestCase):
    def test_only_same_immutable_resampled_owner_and_stage_reuse(self):
        receipt={};memo=StageIdentityMemo(receipt);a=object();b=object();built=[]
        def build():
            built.append(len(built))
            return (tuple(range(3)),tuple([.1,.2,.7]))
        one=memo.get(a,0,build)
        self.assertIs(memo.get(a,0,build),one)
        self.assertEqual(len(built),1)
        memo.get(b,0,build);memo.get(a,1,build)
        self.assertEqual(len(built),3)
        self.assertEqual(len(memo.entries),1)
        self.assertEqual(receipt['hits'],1)
        self.assertEqual(receipt['misses'],3)

    def test_failure_is_not_cached_or_converted(self):
        memo=StageIdentityMemo({});particle=object()
        def fail():raise RuntimeError('model failure')
        with self.assertRaisesRegex(RuntimeError,'model failure'):
            memo.get(particle,0,fail)
        self.assertEqual(memo.entries,{})
