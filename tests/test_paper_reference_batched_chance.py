import random
from types import SimpleNamespace
import unittest

from pokezero.mcts_eval.paper_reference import ReferenceRefusal, SamplingDeadlineExceeded
from pokezero.mcts_eval.paper_reference_batched_chance import batched_chance_pool, public_chance_pool_size
from pokezero.mcts_eval.paper_reference_runtime import ShowdownWorkerFactory


class BatchedLikelihoodTests(unittest.TestCase):
    def test_public_allocation_is_fixed_before_draws_and_ignores_later_rounds(self):
        base = ('|turn|7',)
        self.assertEqual(public_chance_pool_size(base, base+('|move|x', '|-crit|y', '|turn|8'), 4), 32)
        self.assertEqual(public_chance_pool_size(base, base+('|move|x', '|turn|8', '|-crit|y'), 4), 4)
        with self.assertRaises(ReferenceRefusal):
            public_chance_pool_size(base, ('|turn|6',), 4)

    def run_pool(self, *, matches, consumed=4, deadline=False, final=True):
        rng = random.Random(12)
        expected = random.Random(12)
        seeds = [expected.getrandbits(64) for _ in range(4)]
        calls = []; replayed = []; receipt = {}
        def batch(snapshot, actions, **kw):
            calls.append(kw)
            return dict(consumed=consumed, deadline_reached=deadline, matching_indices=matches)
        factory = SimpleNamespace(env=SimpleNamespace(conditioning_batch_from_search_snapshot=batch),
            sampling_deadline_at=1000, check_sampling_deadline=lambda: None)
        child = object()
        result = batched_chance_pool(factory=factory, snapshot='retained', actions={'p1': 2},
            expected=('public',), count=4, trial=lambda i,s: (replayed.append((i,s)) or (child if final else None)),
            rng=rng, receipt=receipt)
        return result, receipt, seeds, calls, replayed

    def test_all_trials_count_and_only_original_selected_seed_is_replayed(self):
        result, receipt, seeds, calls, replayed = self.run_pool(matches=[0, 3])
        self.assertEqual(result.weight, .5)
        self.assertEqual(calls[0]['chance_seeds'], seeds)
        self.assertTrue(calls[0]['fixed_pool'])
        self.assertEqual(len(replayed), 1)
        ordinal, seed = replayed[0]
        self.assertIn(ordinal, [0, 3]); self.assertEqual(seed, seeds[ordinal])
        self.assertTrue(receipt['pool_complete'])
        self.assertFalse(receipt['partial_pool_used'])

    def test_final_authoritative_rejection_is_zero_not_retry(self):
        result, receipt, seeds, calls, replayed = self.run_pool(matches=[0, 3], final=False)
        self.assertIsNone(result); self.assertEqual(len(replayed), 1)
        self.assertEqual(receipt['selected_final_potential'], 0)

    def test_prefix_weight_then_final_indicator_preserves_valid_mass(self):
        # Two of four seeds match the public prefix; only one also matches the
        # exact actor root. Uniform prefix choice then root-indicator has mass
        # (2/4)*(1/2)=1/4, NOT 1/2 and not retry-until-valid.
        class ExactChoice:
            def __init__(self, index): self.index=index
            def getrandbits(self, bits): return 42
            def randrange(self, size): return self.index
        weights=[]
        for choice in (0, 1):
            factory=SimpleNamespace(env=SimpleNamespace(conditioning_batch_from_search_snapshot=lambda *a,**k:
                dict(consumed=4,deadline_reached=False,matching_indices=[0,3])),
                sampling_deadline_at=1,check_sampling_deadline=lambda:None)
            value=batched_chance_pool(factory=factory,snapshot=None,actions={},expected=(),count=4,
                trial=lambda i,s:object() if i==0 else None,rng=ExactChoice(choice),receipt={})
            weights.append(value.weight if value is not None else 0)
        self.assertEqual(sum(weights)/2, .25)

    def test_zero_matches_never_replay_or_retry(self):
        result, receipt, seeds, calls, replayed = self.run_pool(matches=[])
        self.assertIsNone(result); self.assertEqual(replayed, [])
        self.assertEqual(receipt['observation_likelihood_estimate'], 0)

    def test_partial_or_invalid_pool_is_not_a_likelihood(self):
        for kw in (dict(matches=[0], consumed=2), dict(matches=[0], deadline=True),
                   dict(matches=[0,0]), dict(matches=[4])):
            with self.subTest(kw=kw), self.assertRaises(ReferenceRefusal):
                self.run_pool(**kw)

    def test_native_and_deadline_failures_propagate(self):
        for exception in (RuntimeError('native failure'), SamplingDeadlineExceeded('deadline')):
            def batch(*args, **kw): raise exception
            factory = SimpleNamespace(env=SimpleNamespace(conditioning_batch_from_search_snapshot=batch),
                sampling_deadline_at=1, check_sampling_deadline=lambda:None)
            with self.assertRaises(type(exception)):
                batched_chance_pool(factory=factory, snapshot=None, actions={}, expected=(), count=4,
                    trial=lambda *args:self.fail('must not replay'), rng=random.Random(1), receipt={})

    def test_default_off_and_requires_particles(self):
        self.assertFalse(ShowdownWorkerFactory('c','d','e','s').batch_history_chance)
        for value, particles in ((1,32), ('yes',32), (True,0)):
            with self.assertRaises(ReferenceRefusal):
                ShowdownWorkerFactory('c','d','e','s',history_particles=particles,batch_history_chance=value)


if __name__ == '__main__': unittest.main()
