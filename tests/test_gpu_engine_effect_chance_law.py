"""§6: the effect stream in the finite chance law (pure Python; no JAX import, no transition).

The plan-level consumer tests (Bonemerang, Yawn, the Wrap and Perish pairs) live with the other consumer guards in
test_gpu_engine_chance_consumers; this module holds the coordinate inventory, the case-to-ChanceInputs mapping every
enumerated caller shares, the suspended-root law and the enumerated callers' source bindings.
"""
import ast
from fractions import Fraction
from pathlib import Path
import re
import unittest

from gpu_engine import conditional_law
from gpu_engine.finite_law import (COORDINATES, DRAW_BOUNDS, EFFECT_BOUNDS, EFFECT_COORDINATES, Axis, Observation, Plan,
    chance_fields)

ENGINE = Path(conditional_law.__file__).parent


def _assigned(path, name):
    tree = ast.parse(path.read_text())
    return next(ast.literal_eval(node.value) for node in tree.body
                if isinstance(node, ast.Assign) and any(getattr(t, 'id', None) == name for t in node.targets))


def _case(**changes):
    values = {name: 0 for name, _ in COORDINATES}
    values.update(changes)
    return values


class EffectCoordinateTests(unittest.TestCase):
    def test_eight_effect_axes_are_appended_with_the_engine_bounds(self):
        self.assertEqual(EFFECT_BOUNDS, (16, 16, 4, 2))
        self.assertEqual(EFFECT_BOUNDS, _assigned(ENGINE/'transition.py', 'EFFECT_BOUNDS'))   # transition's own constant
        self.assertEqual(len(COORDINATES), 41)
        self.assertEqual(len(dict(COORDINATES)), 41)
        # The 33 original coordinates keep their positions; the stream follows initial_tie.
        self.assertEqual(COORDINATES[32], ('initial_tie', 2))
        self.assertEqual(COORDINATES[:24], tuple((f'draws.{s}.{c}', b) for s in range(2) for c, b in enumerate(DRAW_BOUNDS)))
        self.assertEqual(COORDINATES[33:], tuple((f'effect.{s}.{c}', b) for s in range(2) for c, b in enumerate(EFFECT_BOUNDS)))
        self.assertEqual(EFFECT_COORDINATES, tuple(name for name, _ in COORDINATES[33:]))

    def test_an_inactive_effect_axis_is_one_point_and_an_active_one_partitions_its_bound(self):
        axes = tuple(Axis(name, (Observation('identity'),), ('authored',)) if name == 'effect.1.2'
                     else Axis(name, (), inactive_reason='authored: no consumer') for name, _ in COORDINATES)
        plan = Plan(axes)
        self.assertEqual(plan.cardinality, 4)
        cases = list(plan.bounded_cases(4))
        self.assertEqual([dict(c.values)['effect.1.2'] for c in cases], [0, 1, 2, 3])
        self.assertEqual({c.mass for c in cases}, {Fraction(1, 4)})
        self.assertTrue(all(dict(c.values)['effect.0.0'] == 0 for c in cases))


class ChanceFieldsTests(unittest.TestCase):
    def test_every_field_comes_from_the_case_effect_included(self):
        fields = chance_fields(_case(**{'effect.0.1': 7, 'effect.1.3': 1, 'draws.1.5': 15, 'initial_tie': 1}))
        self.assertEqual(set(fields), {'draws', 'phazing', 'secondary', 'sleep_talk', 'entry_ties', 'initial_tie', 'effect'})
        self.assertEqual(fields['effect'], [[0, 7, 0, 0], [0, 0, 0, 1]])
        self.assertEqual(fields['draws'][1][5], 15)
        self.assertEqual(fields['initial_tie'], 1)

    def test_a_case_missing_a_declared_effect_key_refuses(self):
        for name in EFFECT_COORDINATES:
            values = _case()
            del values[name]
            with self.subTest(missing=name), self.assertRaisesRegex(ValueError, 'CASE_COORDINATE_MISSING: ' + re.escape(name)):
                chance_fields(values)
        legacy = {name: 0 for name, _ in COORDINATES[:33]}   # a pre-§6 six-field case
        with self.assertRaisesRegex(ValueError, 'CASE_COORDINATE_MISSING'):
            chance_fields(legacy)

    def test_enumerated_callers_build_chance_inputs_through_chance_fields(self):
        # The three plan-case executors; none may build a six-field ChanceInputs (effect=None samples the stream).
        for module, owner in (('fidelity_panel.py', '__call__'), ('phase_d_search.py', 'process_many'),
                              ('ordinary_law.py', 'compare_plan')):
            tree = ast.parse((ENGINE/module).read_text())
            function = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == owner)
            calls = [n for n in ast.walk(function) if isinstance(n, ast.Call)]
            names = {getattr(c.func, 'id', getattr(c.func, 'attr', None)) for c in calls}
            with self.subTest(module=module):
                self.assertIn('chance_fields', names)
                for call in calls:
                    if getattr(call.func, 'id', getattr(call.func, 'attr', None)) == 'ChanceInputs':
                        # Every field by name, from ChanceInputs._fields (so effect is always among them).
                        self.assertEqual(call.args, [])
                        self.assertEqual([k.arg for k in call.keywords], [None])
                        self.assertIn('_fields', ast.dump(call.keywords[0].value))


class SuspendedLawTests(unittest.TestCase):
    def test_a_suspended_root_reads_the_saved_effect_block(self):
        law = conditional_law._law()
        self.assertEqual(law.cardinality, 4)                       # the two fresh entry ties only
        for axis in law.axes:
            if axis.coordinate in EFFECT_COORDINATES:
                with self.subTest(axis=axis.coordinate):
                    self.assertEqual(axis.observations, ())
                    self.assertIn('saved effect block', axis.inactive_reason)
        # The engine line the inactivity cites: a resumed step replaces the supplied stream by the saved block.
        source = (ENGINE/'transition.py').read_text()
        self.assertIn('effect_draws=jnp.where(resuming[:,None,None],state.effect_draws,effect_draws)', source)
        self.assertIn('work = work._replace(effect_draws=jnp.where(defer_residual[:,None,None],effect_draws,0))', source)


if __name__ == '__main__':
    unittest.main()
