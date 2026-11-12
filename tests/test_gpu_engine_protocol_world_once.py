"""phase_d_protocol_world._once: one root's world-independent scans computed once for its worlds, with the same values (a fresh
copy per world unless shared) and the same exception re-raised at the same call site."""
import unittest

from gpu_engine import phase_d_protocol_world as w


def inputs():
    return w.ProtocolInputs(None, ('|start',), None, {}, None)


class OnceTests(unittest.TestCase):
    def test_computed_once_per_root_with_a_fresh_copy_per_world_unless_shared(self):
        x, calls, root = inputs(), [], object()
        def compute():
            calls.append(1);return {'a': [1, 2]}
        first = w._once(x, root, 'facts', compute);second = w._once(x, root, 'facts', compute)
        self.assertEqual(first, second);self.assertIsNot(first, second);self.assertEqual(len(calls), 1)
        first['a'].append(3);self.assertEqual(w._once(x, root, 'facts', compute), {'a': [1, 2]})
        shared = w._once(x, root, 'revealed', lambda: ((1, 2),), shared=True)
        self.assertIs(shared, w._once(x, root, 'revealed', lambda: None, shared=True))
        w._once(x, object(), 'facts', compute);self.assertEqual(len(calls), 2)   # another root computes its own

    def test_an_error_is_raised_again_for_every_world_without_recomputing(self):
        x, calls, root = inputs(), [], object()
        def compute():
            calls.append(1);raise ValueError('PUBLIC_REPLACEMENT_PHASE')
        for _ in range(3):
            with self.assertRaisesRegex(ValueError, 'PUBLIC_REPLACEMENT_PHASE'):w._once(x, root, 'facts', compute)
        self.assertEqual(len(calls), 1)

    def test_the_memo_is_not_part_of_the_inputs_identity(self):
        a, b = inputs(), inputs();w._once(a, object(), 'prefix', lambda: 'x')
        self.assertEqual(a, b);self.assertNotIn('memo', repr(a))


if __name__ == '__main__':
    unittest.main()
