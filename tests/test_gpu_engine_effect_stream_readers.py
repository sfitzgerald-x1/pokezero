"""Every read of transition's effect stream is a consumer the exact-chance plans observe; pure Python, no JAX import.

finite_law's effect.{s}.{c} axes are enumerated only where chance_consumers, phase_d_chance and ordinary_law gate them, and
an inactive axis carries a fixed 0. So the proof is lossless only while every place transition.py reads an effect value is
one of the consumers those plans observe; a new, unobserved reader would misprice an exact root without any error.
transition.py's SOURCE_PINS refuse any edit until the pins are refreshed, but a refresh is mechanical
(scripts/refresh_source_pins.py). This test makes it insufficient: a new or moved reader changes the snapshot below, and its
author has to re-review the plans' effect gating (chance_consumers.plan_for_state, phase_d_chance.plan_for_state,
ordinary_law.plan_for_state) before updating it.
"""
import ast
import collections
from pathlib import Path
import unittest

TRANSITION = Path(__file__).resolve().parents[1] / 'gpu_engine' / 'transition.py'
STREAM_NAMES = frozenset({'effect_draws', '_effect'})
STREAM_ATTRS = frozenset({'effect_draws', 'effect'})

# (enclosing function, the expression that uses the stream) -> count. Each entry's role:
REVIEWED = {
    # the step's effect block: the caller's ChanceInputs.effect, sampled when None, the saved block when resuming
    ('step_with_chance', 'effect_draws = chance_inputs.effect'): 1,
    ('step_with_chance', 'effect_draws is None'): 1,
    ('step_with_chance', 'jnp.where(resuming[:, None, None], state.effect_draws, effect_draws)'): 2,
    # effect.s.0 / effect.s.1: Bonemerang's second hit (crit threshold, damage roll 85+d), read only under _second
    ('step_with_chance', 'effect_draws[rows, side]'): 1,
    ('step_with_chance', '_effect[:, 0]'): 2,
    ('step_with_chance', '_effect[:, 1]'): 2,
    # effect.0.3: the Wrap pair tie
    ('step_with_chance', 'effect_draws[:, 0, 3, None]'): 1,
    # effect.s.2: Yawn's sleep duration 2+d
    ('step_with_chance', 'effect_draws[rows, side, 2]'): 1,
    # effect.1.3: the Perish pair tie
    ('step_with_chance', 'effect_draws[:, 1, 3]'): 1,
    # the save of the block into state.effect_draws when the residual is deferred (a suspended turn)
    ('step_with_chance', 'jnp.where(defer_residual[:, None, None], effect_draws, 0)'): 1,
}


def readers(source):
    """Counter of (enclosing function, using expression) over every load of the effect stream in `source`."""
    tree = ast.parse(source)
    parents = {}
    for node in ast.walk(tree):
        for child in ast.iter_child_nodes(node):
            parents[child] = node

    def enclosing(node):
        while node in parents:
            node = parents[node]
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                return node.name
        return '<module>'

    def using(node):
        parent = parents.get(node)
        if isinstance(parent, (ast.keyword, ast.Starred)):
            parent = parents[parent]
        return ast.unparse(parent)

    found = collections.Counter()
    for node in ast.walk(tree):
        name = isinstance(node, ast.Name) and node.id in STREAM_NAMES
        attr = isinstance(node, ast.Attribute) and node.attr in STREAM_ATTRS
        if (name or attr) and isinstance(node.ctx, ast.Load):
            found[(enclosing(node), using(node))] += 1
    return found


class EffectStreamReaderTests(unittest.TestCase):
    def test_every_effect_read_in_transition_is_a_reviewed_consumer(self):
        found = readers(TRANSITION.read_text())
        new = {k: v for k, v in found.items() if REVIEWED.get(k) != v}
        gone = {k: v for k, v in REVIEWED.items() if found.get(k) != v}
        self.assertEqual((new, gone), ({}, {}),
                         'transition.py reads the effect stream somewhere the exact-chance plans have not reviewed. Re-check '
                         "the effect gating in chance_consumers, phase_d_chance and ordinary_law's plan_for_state, then "
                         'update REVIEWED.')

    def test_the_scan_sees_a_new_reader_an_alias_and_a_pass_through(self):
        base = TRANSITION.read_text()
        for extra in ('\ndef _new(effect_draws):\n    return effect_draws[:, 0, 2]\n',
                      '\ndef _alias(effect_draws):\n    e = effect_draws\n    return e\n',
                      '\ndef _through(effect_draws):\n    return helper(x=effect_draws)\n'):
            with self.subTest(extra=extra.split('(')[0].strip()):
                self.assertNotEqual(readers(base + extra), readers(base))


if __name__ == '__main__':
    unittest.main()
