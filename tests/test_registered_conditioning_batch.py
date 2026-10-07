"""Qualification and confirmation bind the same explicit spawn-safe batch size."""
import pickle
from types import SimpleNamespace as NS
import unittest
from unittest import mock

from pokezero.mcts_eval.paper_reference import ReferenceRefusal
from pokezero.mcts_eval.paper_reference_runtime import (
    ShowdownWorkerFactory, validated_conditioning_batch_size,
)
from wider_search_comparison import registered_conditioning_batch_size, register


class RegisteredConditioningBatchTests(unittest.TestCase):
    def test_legacy_registration_and_factory_preserve_unbatched_default(self):
        self.assertEqual(registered_conditioning_batch_size({}),1)
        self.assertEqual(registered_conditioning_batch_size(dict(reference=dict(staged_substitute_conditioning=True))),1)
        factory=ShowdownWorkerFactory('checkpoint','digest','showdown','source')
        self.assertEqual(factory.conditioning_batch_size,1)
        self.assertFalse(factory.staged_substitute_conditioning)

    def test_explicit_size8_is_pickle_bound_to_spawned_runtime(self):
        factory=ShowdownWorkerFactory('checkpoint','digest','showdown','source',
            staged_substitute_conditioning=True,conditioning_batch_size=8)
        restored=pickle.loads(pickle.dumps(factory))
        with mock.patch('pokezero.mcts_eval.paper_reference_runtime._ShowdownRuntime') as runtime:
            restored(7)
        runtime.assert_called_once_with(restored,7)
        self.assertEqual(restored.conditioning_batch_size,8)
        self.assertEqual(registered_conditioning_batch_size(dict(reference=dict(
            staged_substitute_conditioning=True,conditioning_batch_size=8))),8)

    def test_all_allowed_sizes_and_invalid_types_fail_before_spawn_or_registration(self):
        for size in range(1,17):
            self.assertEqual(validated_conditioning_batch_size(size,staged=True),size)
        for size in (True,False,0,17,-1,8.,'8',None):
            with self.assertRaises(ReferenceRefusal):
                ShowdownWorkerFactory('checkpoint','digest','showdown','source',
                    staged_substitute_conditioning=True,conditioning_batch_size=size)
            with self.assertRaises(ReferenceRefusal):
                registered_conditioning_batch_size(dict(reference=dict(
                    staged_substitute_conditioning=True,conditioning_batch_size=size)))
            # Validation must precede native import, source checks or file writes.
            with self.assertRaises(ReferenceRefusal):
                register(NS(staged_substitute_conditioning=True,conditioning_batch_size=size))

    def test_batching_without_staged_gate_and_nonboolean_gate_fail_closed(self):
        for staged in (False,None,1,'true'):
            with self.assertRaises(ReferenceRefusal):
                validated_conditioning_batch_size(8,staged=staged)


if __name__=='__main__':
    unittest.main()
