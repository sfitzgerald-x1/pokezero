"""Finite-law accounting tests, not a declaration of transition-rule coverage."""
from fractions import Fraction
import unittest
from gpu_engine.finite_law import (
    Axis, COORDINATES, CardinalityRefusal, Observation, Plan, compare_laws, partition, pushforward,
)


def plan(changes=None):
    changes = changes or {}
    return Plan(tuple(Axis(name, changes[name], source_uses=("authored rule",)) if name in changes
                      else Axis(name, (), inactive_reason="authored function does not read this coordinate")
                      for name, _ in COORDINATES))


class FiniteLawTests(unittest.TestCase):
    def test_full_physical_inventory(self):
        self.assertEqual(len(COORDINATES), 33)
        self.assertEqual(len(dict(COORDINATES)), 33)

    def test_shared_draw_refines_one_axis_not_independent_coins(self):
        p = plan({'draws.0.2': (Observation('less_than', 25), Observation('less_than', 50))})
        law = pushforward(p, lambda c: (dict(c.values)['draws.0.2'] < 25,
                                       dict(c.values)['draws.0.2'] < 50), cap=3)
        self.assertEqual(law, {(True, True): Fraction(1, 4), (False, True): Fraction(1, 4),
                               (False, False): Fraction(1, 2)})

    def test_all_values_exhausted_with_exact_masses(self):
        cells = partition(100, (Observation('less_than', Fraction(19, 2)),))
        self.assertEqual(cells[0].mass, Fraction(1, 10))
        self.assertEqual(sorted(x for cell in cells for x in cell.members), list(range(100)))
        self.assertEqual(sum(c.mass for c in cells), 1)

    def test_modulo_joint_consumers_keep_their_correlation(self):
        cells = partition(60, (Observation('modulo', 2), Observation('modulo', 3)))
        self.assertEqual(len(cells), 6)
        self.assertTrue(all(c.mass == Fraction(1, 6) for c in cells))

    def test_identity_never_merges_distinct_damage_inputs(self):
        self.assertEqual(len(partition(16, (Observation('identity'),))), 16)

    def test_missing_reordered_or_duplicate_coordinate_refuses(self):
        axes = plan().axes
        for bad in (axes[:-1], (axes[1], axes[0], *axes[2:]), (*axes[:-1], axes[0])):
            with self.assertRaises(ValueError): Plan(tuple(bad))

    def test_inactive_coordinate_requires_explicit_domain_reason(self):
        with self.assertRaises(ValueError): Axis('draws.0.0', ())
        with self.assertRaises(ValueError): Axis('draws.0.0', (Observation('identity'),))
        for reason in (['mutable reason'],{'mutable':'reason'},1,True):
            with self.assertRaises(ValueError):Axis('draws.0.0',(),inactive_reason=reason)

    def test_cardinality_refuses_before_any_case_or_callback(self):
        p = plan({'draws.0.5': (Observation('identity'),), 'draws.1.5': (Observation('identity'),)})
        self.assertEqual(p.cardinality, 256)
        with self.assertRaises(CardinalityRefusal) as caught: p.bounded_cases(255)
        self.assertEqual(caught.exception.required, 256)
        called = []
        with self.assertRaises(CardinalityRefusal): pushforward(p, lambda c: called.append(c), cap=255)
        self.assertEqual(called, [])

    def test_execution_refusal_is_not_dropped_or_normalized(self):
        p = plan({'initial_tie': (Observation('identity'),)})
        def run(c):
            if dict(c.values)['initial_tie']: raise RuntimeError('unsupported future state')
            return 'one'
        with self.assertRaisesRegex(RuntimeError, 'unsupported'): pushforward(p, run, cap=2)

    def test_product_mass_and_output_collapsing(self):
        p = plan({'draws.0.4': (Observation('less_than', 1),), 'draws.1.4': (Observation('less_than', 1),)})
        law = pushforward(p, lambda c: sum(dict(c.values)[f'draws.{s}.4'] == 0 for s in range(2)), cap=4)
        self.assertEqual(law, {2: Fraction(1, 256), 1: Fraction(30, 256), 0: Fraction(225, 256)})

    def test_native_mass_is_reported_without_renormalization(self):
        result = compare_laws({'a': Fraction(1)}, {'a': 1.0000001}, mass_tolerance=1e-6)
        self.assertGreater(result['native_mass_residual'], 0)
        self.assertGreater(result['half_l1'], 0)
        self.assertFalse(result['exact_only'] | result['native_only'])
        with self.assertRaises(ValueError): compare_laws({'a': Fraction(1)}, {'a': .99}, mass_tolerance=1e-6)

    def test_support_differences_are_not_hidden_by_equal_total_mass(self):
        result = compare_laws({'a': Fraction(1)}, {'b': 1.0}, mass_tolerance=1e-6)
        self.assertEqual(result['exact_only'], {'a'})
        self.assertEqual(result['native_only'], {'b'})
        self.assertEqual(result['half_l1'], 1)

    def test_invalid_partition_and_mass_parameters_refuse(self):
        for args in [('less_than', .5), ('modulo', 0), ('identity', 1), ('random', 0)]:
            with self.assertRaises(ValueError): Observation(*args)
        with self.assertRaises(ValueError): plan().bounded_cases(True)
        for native in ({'a': float('nan')}, {'a': 0.0}, {'a': -1.0}):
            with self.assertRaises(ValueError): compare_laws({'a': Fraction(1)}, native, mass_tolerance=1e-6)


if __name__ == '__main__': unittest.main()
